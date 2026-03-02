import sys
import time

import requests

BASE_URL = "https://api.spacetraders.io/v2"
_RETRYABLE = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


class SpaceTradersClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        # Simple in-memory cache: {path: (timestamp, data)}
        self._cache = {}
        self._cache_ttl = 2.0  # Seconds to trust a cached GET response

    def _request(self, method: str, path: str, retries: int = 3, **kwargs) -> dict:
        # 1. READ CACHE (GET only)
        if method == "GET":
            cached = self._cache.get(path)
            if cached:
                ts, data = cached
                if time.time() - ts < self._cache_ttl:
                    # Debug: Uncomment to see cache hits
                    # print(f"[CACHE] HIT {path}", file=sys.stderr)
                    return data
                else:
                    # Expired
                    del self._cache[path]

        # 2. PERFORM REQUEST
        if method == "POST" and "json" not in kwargs:
            kwargs["json"] = {}

        print(f"[API] {method} {path}", file=sys.stderr)
        if "json" in kwargs and kwargs["json"]:
            print(f"      Payload: {kwargs['json']}", file=sys.stderr)

        # 3. INVALIDATION (Write operations)
        # If we change state (POST/PATCH), we must assume related GETs are stale.
        if method in ["POST", "PATCH", "DELETE"]:
            # Logic: If I modify '/my/ships/WHATER-1/refuel',
            # I must invalidate '/my/ships/WHATER-1' and '/my/ships' and '/my/agent'

            # Always invalidate agent (credits change on almost everything)
            self._cache.pop("/my/agent", None)

            # If acting on a ship, invalidate that ship and the ship list
            if "/my/ships" in path:
                self._cache.pop("/my/ships", None) # Invalidate full list

                # Try to extract ship ID to invalidate specific entry
                # Path usually looks like: /my/ships/WHATER-1/navigate
                parts = path.split("/")
                # parts = ['', 'my', 'ships', 'WHATER-1', 'navigate']
                if len(parts) >= 4:
                    ship_path = f"/{parts[1]}/{parts[2]}/{parts[3]}" # /my/ships/WHATER-1
                    self._cache.pop(ship_path, None)
                    # Also invalidate cargo/cooldown specific sub-paths
                    self._cache.pop(f"{ship_path}/cargo", None)
                    self._cache.pop(f"{ship_path}/cooldown", None)
                    self._cache.pop(f"{ship_path}/nav", None)

        last_err = None
        for attempt in range(retries):
            try:
                resp = self.session.request(
                    method, f"{BASE_URL}{path}", timeout=30, **kwargs
                )
            except _RETRYABLE as e:
                last_err = str(e)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s
                continue
            except requests.exceptions.RequestException as e:
                return {"error": str(e)}

            # Rate limited — wait and retry
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2))
                if attempt < retries - 1:
                    time.sleep(retry_after)
                    continue
                return {"error": "Rate limited, all retries exhausted"}

            # 204 No Content (e.g. cooldown endpoint when no cooldown active)
            if resp.status_code == 204 or not resp.content:
                return {}

            body = resp.json()
            if "error" in body:
                return {"error": body["error"].get("message", str(body["error"]))}

            data = body.get("data", body)

            # 4. WRITE CACHE (GET only, Success only)
            if method == "GET":
                self._cache[path] = (time.time(), data)

            return data

        return {"error": f"Connection failed after {retries} attempts: {last_err}"}

    # --- Agent ---

    def get_agent(self) -> dict:
        return self._request("GET", "/my/agent")

    # --- Contracts ---

    def list_contracts(self) -> dict:
        return self._request("GET", "/my/contracts")

    def accept_contract(self, contract_id: str) -> dict:
        return self._request("POST", f"/my/contracts/{contract_id}/accept")

    def deliver_contract(self, contract_id: str, ship_symbol: str, trade_symbol: str, units: int) -> dict:
        return self._request("POST", f"/my/contracts/{contract_id}/deliver", json={
            "shipSymbol": ship_symbol,
            "tradeSymbol": trade_symbol,
            "units": units,
        })

    def fulfill_contract(self, contract_id: str) -> dict:
        return self._request("POST", f"/my/contracts/{contract_id}/fulfill")

    # --- Systems / Waypoints ---

    def list_waypoints(self, system: str, **params) -> dict:
        """List waypoints with automatic pagination to get ALL results."""
        all_data = []
        page = 1
        limit = 20  # SpaceTraders default

        while True:
            params_with_page = {**params, "page": page, "limit": limit}
            #print(f"[API] list_waypoints {params_with_page}", file=sys.stderr)
            resp = self.session.request("GET", f"{BASE_URL}/systems/{system}/waypoints", params=params_with_page)
            #print(f"[API] list_waypoints {resp.content}", file=sys.stderr)

            if resp.status_code == 204 or not resp.content:
                break

            body = resp.json()
            if "error" in body:
                return {"error": body["error"].get("message", str(body["error"]))}

            data = body.get("data", [])
            if not data:
                break

            all_data.extend(data)

            # Check if there are more pages
            meta = body.get("meta", {})
            total = meta.get("total", 0)
            if len(all_data) >= total:
                break

            page += 1

        return all_data

    def get_shipyard(self, system: str, waypoint: str) -> dict:
        return self._request("GET", f"/systems/{system}/waypoints/{waypoint}/shipyard")

    def get_market(self, system: str, waypoint: str) -> dict:
        return self._request("GET", f"/systems/{system}/waypoints/{waypoint}/market")

    # --- Ships ---

    def list_ships(self) -> dict:
        return self._request("GET", "/my/ships")

    def get_ship(self, ship: str) -> dict:
        return self._request("GET", f"/my/ships/{ship}")

    def purchase_ship(self, ship_type: str, waypoint: str) -> dict:
        return self._request("POST", "/my/ships", json={
            "shipType": ship_type,
            "waypointSymbol": waypoint,
        })

    def get_cargo(self, ship: str) -> dict:
        return self._request("GET", f"/my/ships/{ship}/cargo")

    def orbit(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/orbit")

    def dock(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/dock")

    def navigate(self, ship: str, waypoint: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/navigate", json={
            "waypointSymbol": waypoint,
        })

    def refuel(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/refuel")

    def extract(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/extract")

    def sell_cargo(self, ship: str, symbol: str, units: int) -> dict:
        return self._request("POST", f"/my/ships/{ship}/sell", json={
            "symbol": symbol,
            "units": units,
        })

    def buy_cargo(self, ship: str, symbol: str, units: int) -> dict:
        return self._request("POST", f"/my/ships/{ship}/purchase", json={
            "symbol": symbol,
            "units": units,
        })

    def jettison(self, ship: str, symbol: str, units: int) -> dict:
        return self._request("POST", f"/my/ships/{ship}/jettison", json={
            "symbol": symbol,
            "units": units,
        })

    def transfer_cargo(self, from_ship: str, to_ship: str, symbol: str, units: int) -> dict:
        return self._request("POST", f"/my/ships/{from_ship}/transfer", json={
            "shipSymbol": to_ship,
            "tradeSymbol": symbol,
            "units": units,
        })

    # --- Advanced Ship Operations ---

    def survey(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/survey")

    def extract_with_survey(self, ship: str, survey: dict) -> dict:
        return self._request("POST", f"/my/ships/{ship}/extract/survey", json={
            "survey": survey,
        })

    def scan_waypoints(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/scan/waypoints")

    def scan_ships(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/scan/ships")

    def scan_systems(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/scan/systems")

    def jump(self, ship: str, system: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/jump", json={
            "systemSymbol": system,
        })

    def warp(self, ship: str, waypoint: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/warp", json={
            "waypointSymbol": waypoint,
        })

    def negotiate_contract(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/negotiate/contract")

    def chart(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/chart")

    def refine(self, ship: str, produce: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/refine", json={
            "produce": produce,
        })

    def siphon(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/siphon")

    def get_cooldown(self, ship: str) -> dict:
        return self._request("GET", f"/my/ships/{ship}/cooldown")

    def set_flight_mode(self, ship: str, mode: str) -> dict:
        return self._request("PATCH", f"/my/ships/{ship}/nav", json={
            "flightMode": mode,
        })

    # --- Jump Gate ---

    def get_jump_gate(self, system: str, waypoint: str) -> dict:
        return self._request("GET", f"/systems/{system}/waypoints/{waypoint}/jump-gate")

    # --- Construction ---

    def get_construction(self, system: str, waypoint: str) -> dict:
        return self._request("GET", f"/systems/{system}/waypoints/{waypoint}/construction")

    def supply_construction(self, system: str, waypoint: str, ship: str, symbol: str, units: int) -> dict:
        return self._request("POST", f"/systems/{system}/waypoints/{waypoint}/construction/supply", json={
            "shipSymbol": ship,
            "tradeSymbol": symbol,
            "units": units,
        })
