import logging
import sys
import threading
import time

import requests

BASE_URL = "https://api.spacetraders.io/v2"
log = logging.getLogger("api_client")
_RETRYABLE = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


class RateLimiter:
    """
    Thread-safe Token Bucket Rate Limiter.
    Ensures we don't exceed max_rate requests per second.
    """

    def __init__(self, max_rate: float = 2.0, time_period: float = 1.0):
        self._max_tokens = max_rate
        self._tokens = max_rate
        self._lock = threading.Lock()
        self._last_update = time.monotonic()

        # Calculate how many seconds per token (e.g., 0.5s for 2 req/s)
        self._refill_rate = max_rate / time_period

    def acquire(self):
        """Blocks until a token is available."""
        with self._lock:
            while True:
                now = time.monotonic()
                # Refill tokens based on time passed
                elapsed = now - self._last_update
                new_tokens = elapsed * self._refill_rate

                if new_tokens > 0:
                    self._tokens = min(self._max_tokens, self._tokens + new_tokens)
                    self._last_update = now

                if self._tokens >= 1:
                    self._tokens -= 1
                    return

                # Calculate wait time for next token
                # We need 1.0 token, we have self._tokens. We need (1 - self._tokens) more.
                needed = 1.0 - self._tokens
                wait_time = needed / self._refill_rate
                time.sleep(wait_time)


class SpaceTradersClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        )
        self._cache = {}
        self._cache_ttl = 3.0

        # Strict 2 requests per second limiter
        self._limiter = RateLimiter(max_rate=2.0, time_period=1.0)

    def _paginate_request(self, endpoint: str, **params) -> dict:
        """Helper: Paginate through GET results until all data is retrieved.

        Args:
            endpoint: Full URL path (e.g., "/my/ships" or "/systems/{system}/waypoints")
            **params: Query parameters to include

        Returns:
            List of all data from all pages, or dict with error if failed
        """
        all_data = []
        page = 1
        limit = 20  # SpaceTraders default

        while True:
            params_with_page = {**params, "page": page, "limit": limit}
            self._limiter.acquire()

            try:
                resp = self.session.request(
                    "GET", f"{BASE_URL}{endpoint}", params=params_with_page, timeout=30
                )
            except _RETRYABLE:
                # On connection error, return what we have so far
                return (
                    all_data
                    if all_data
                    else {"error": "Connection failed during pagination"}
                )

            if resp.status_code == 204 or not resp.content:
                break

            body = resp.json()
            if "error" in body:
                err_obj = body["error"]
                err_code = err_obj.get("code", "UNKNOWN")
                err_msg = err_obj.get("message", str(err_obj))
                err_data = err_obj.get("data", {})

                detailed_err = f"API Error {err_code}: {err_msg}"
                if err_data:
                    detailed_err += f" | Details: {err_data}"

                print(f"\n" + "=" * 50, file=sys.stderr)
                print(f"🚨 PAGINATION API ERROR", file=sys.stderr)
                print(f"Request: GET {endpoint}", file=sys.stderr)
                print(f"Query Params: {params_with_page}", file=sys.stderr)
                print(f"Response: {detailed_err}", file=sys.stderr)
                print("=" * 50 + "\n", file=sys.stderr)

                return {"error": detailed_err}

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

    def _request(self, method: str, path: str, retries: int = 3, **kwargs) -> dict:
        # 1. CACHE CHECK (GET only)
        if method == "GET":
            cached = self._cache.get(path)
            if cached:
                ts, data = cached
                if time.time() - ts < self._cache_ttl:
                    return data
                else:
                    del self._cache[path]

        if method == "POST" and "json" not in kwargs:
            kwargs["json"] = {}

        # 2. INVALIDATION LOGIC
        if method in ["POST", "PATCH", "DELETE"]:
            self._cache.pop("/my/agent", None)
            if "/my/ships" in path:
                self._cache.pop("/my/ships", None)
                parts = path.split("/")
                if len(parts) >= 4:
                    ship_path = f"/{parts[1]}/{parts[2]}/{parts[3]}"
                    self._cache.pop(ship_path, None)
                    self._cache.pop(f"{ship_path}/cargo", None)
                    self._cache.pop(f"{ship_path}/cooldown", None)
                    self._cache.pop(f"{ship_path}/nav", None)

        log.info(f"{method} {path}")

        last_err = None
        for attempt in range(retries):
            # --- BLOCK HERE IF RATE LIMITED ---
            self._limiter.acquire()

            try:
                resp = self.session.request(
                    method, f"{BASE_URL}{path}", timeout=30, **kwargs
                )
            except _RETRYABLE as e:
                last_err = str(e)
                if attempt < retries - 1:
                    time.sleep(1 + attempt)
                continue
            except requests.exceptions.RequestException as e:
                if 'resp' in locals():
                    log.error(f"API error: {resp.text}")
                log.error(f"API error: {str(e)}")
                return {"error": str(e)}

            # Handle 429 specifically provided by server
            if resp.status_code == 429:
                # If we hit this, our local limiter was slightly off or server is strict
                retry_after = float(resp.headers.get("Retry-After", 2))
                print(
                    f"[429] Rate limit hit. Server requested wait: {retry_after}s",
                    file=sys.stderr,
                )
                time.sleep(retry_after)
                continue

            if resp.status_code == 204 or not resp.content:
                return {}

            body = resp.json()
            if "error" in body:
                err_obj = body["error"]
                err_code = err_obj.get("code", "UNKNOWN")
                err_msg = err_obj.get("message", str(err_obj))
                err_data = err_obj.get("data", {})

                # Build a detailed error string for the UI / LLM
                detailed_err = f"API Error {err_code}: {err_msg}"
                if err_data:
                    detailed_err += f" | Details: {err_data}"

                # Print a massive debug block to the server console
                print(f"\n" + "=" * 50, file=sys.stderr)
                print(f"🚨 API ERROR DETECTED", file=sys.stderr)
                print(f"Request: {method} {path}", file=sys.stderr)
                if "json" in kwargs and kwargs["json"]:
                    print(f"Payload: {kwargs['json']}", file=sys.stderr)
                elif "params" in kwargs and kwargs["params"]:
                    print(f"Query Params: {kwargs['params']}", file=sys.stderr)
                print(f"Response: {detailed_err}", file=sys.stderr)
                print("=" * 50 + "\n", file=sys.stderr)

                return {"error": detailed_err}

            data = body.get("data", body)

            # Write to cache
            if method == "GET":
                self._cache[path] = (time.time(), data)

            return data

        return {"error": f"Connection failed after {retries} attempts: {last_err}"}

    # --- Agent ---

    def get_agent(self) -> dict:
        return self._request("GET", "/my/agent")

    # --- Contracts ---

    def get_contract(self, contract_id: str) -> dict:
        return self._request("GET", f"/my/contracts/{contract_id}")

    def list_contracts(self) -> list | dict:
        """Fetch only the most recent contracts (last 2 pages) to save API calls and avoid infinite pagination bloat."""
        import math

        limit = 20
        self._limiter.acquire()

        try:
            # 1. Fetch page 1 just to get the 'meta.total' count
            resp = self.session.request(
                "GET",
                f"{BASE_URL}/my/contracts",
                params={"page": 1, "limit": limit},
                timeout=30,
            )
            if resp.status_code == 204 or not resp.content:
                return []

            body = resp.json()
            if "error" in body:
                err_obj = body["error"]
                err_code = err_obj.get("code", "UNKNOWN")
                err_msg = err_obj.get("message", str(err_obj))
                err_data = err_obj.get("data", {})

                detailed_err = f"API Error {err_code}: {err_msg}"
                if err_data:
                    detailed_err += f" | Details: {err_data}"

                return {"error": detailed_err}

            meta = body.get("meta", {})
            total = meta.get("total", 0)
            data = body.get("data", [])

            # If we have 20 or fewer contracts, we already have all of them.
            if total <= limit:
                return data

            # 2. If we have > 20, the NEWEST contracts are on the LAST pages.
            # Calculate total pages and grab up to the last 2 pages (max 40 recent contracts)
            total_pages = math.ceil(total / limit)
            start_page = max(2, total_pages - 1)

            recent_contracts = []
            for page in range(start_page, total_pages + 1):
                self._limiter.acquire()
                page_resp = self.session.request(
                    "GET",
                    f"{BASE_URL}/my/contracts",
                    params={"page": page, "limit": limit},
                    timeout=30,
                )
                if page_resp.status_code == 200:
                    recent_contracts.extend(page_resp.json().get("data", []))

            return recent_contracts

        except Exception as e:
            return {"error": str(e)}

    def accept_contract(self, contract_id: str) -> dict:
        return self._request("POST", f"/my/contracts/{contract_id}/accept")

    def deliver_contract(
        self, contract_id: str, ship_symbol: str, trade_symbol: str, units: int
    ) -> dict:
        return self._request(
            "POST",
            f"/my/contracts/{contract_id}/deliver",
            json={
                "shipSymbol": ship_symbol,
                "tradeSymbol": trade_symbol,
                "units": units,
            },
        )

    def fulfill_contract(self, contract_id: str) -> dict:
        return self._request("POST", f"/my/contracts/{contract_id}/fulfill")

    # --- Systems / Waypoints ---

    def get_system(self, system: str) -> dict:
        return self._request("GET", f"/systems/{system}")

    def list_waypoints(self, system: str, **params) -> list:
        """List waypoints with automatic pagination to get ALL results."""
        return self._paginate_request(f"/systems/{system}/waypoints", **params)

    def get_waypoint(self, system: str, waypoint: str) -> dict:
        """Fetch a specific waypoint to check traits and chart status."""
        return self._request("GET", f"/systems/{system}/waypoints/{waypoint}")

    def get_shipyard(self, system: str, waypoint: str) -> dict:
        return self._request("GET", f"/systems/{system}/waypoints/{waypoint}/shipyard")

    def get_market(self, system: str, waypoint: str) -> dict:
        return self._request("GET", f"/systems/{system}/waypoints/{waypoint}/market")

    # --- Ships ---

    def list_ships(self) -> list:
        """List all ships with automatic pagination."""
        return self._paginate_request("/my/ships")

    def get_ship(self, ship: str) -> dict:
        return self._request("GET", f"/my/ships/{ship}")

    def purchase_ship(self, ship_type: str, waypoint: str) -> dict:
        return self._request(
            "POST",
            "/my/ships",
            json={
                "shipType": ship_type,
                "waypointSymbol": waypoint,
            },
        )

    def get_cargo(self, ship: str) -> dict:
        return self._request("GET", f"/my/ships/{ship}/cargo")

    def orbit(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/orbit")

    def dock(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/dock")

    def navigate(self, ship: str, waypoint: str) -> dict:
        return self._request(
            "POST",
            f"/my/ships/{ship}/navigate",
            json={
                "waypointSymbol": waypoint,
            },
        )

    def refuel(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/refuel")

    def extract(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/extract")

    def sell_cargo(self, ship: str, symbol: str, units: int) -> dict:
        return self._request(
            "POST",
            f"/my/ships/{ship}/sell",
            json={
                "symbol": symbol,
                "units": units,
            },
        )

    def buy_cargo(self, ship: str, symbol: str, units: int) -> dict:
        return self._request(
            "POST",
            f"/my/ships/{ship}/purchase",
            json={
                "symbol": symbol,
                "units": units,
            },
        )

    def jettison(self, ship: str, symbol: str, units: int) -> dict:
        return self._request(
            "POST",
            f"/my/ships/{ship}/jettison",
            json={
                "symbol": symbol,
                "units": units,
            },
        )

    def transfer_cargo(
        self, from_ship: str, to_ship: str, symbol: str, units: int
    ) -> dict:
        return self._request(
            "POST",
            f"/my/ships/{from_ship}/transfer",
            json={
                "shipSymbol": to_ship,
                "tradeSymbol": symbol,
                "units": units,
            },
        )

    # --- Advanced Ship Operations ---

    def survey(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/survey")

    def extract_with_survey(self, ship: str, survey: dict) -> dict:
        return self._request(
            "POST",
            f"/my/ships/{ship}/extract/survey",
            json={
                "survey": survey,
            },
        )

    def scan_waypoints(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/scan/waypoints")

    def scan_ships(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/scan/ships")

    def scan_systems(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/scan/systems")

    def jump(self, ship: str, waypoint: str) -> dict:
        return self._request(
            "POST",
            f"/my/ships/{ship}/jump",
            json={
                "waypointSymbol": waypoint,
            },
        )

    def warp(self, ship: str, waypoint: str) -> dict:
        return self._request(
            "POST",
            f"/my/ships/{ship}/warp",
            json={
                "waypointSymbol": waypoint,
            },
        )

    def negotiate_contract(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/negotiate/contract")

    def chart(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/chart")

    def refine(self, ship: str, produce: str) -> dict:
        return self._request(
            "POST",
            f"/my/ships/{ship}/refine",
            json={
                "produce": produce,
            },
        )

    def siphon(self, ship: str) -> dict:
        return self._request("POST", f"/my/ships/{ship}/siphon")

    def get_cooldown(self, ship: str) -> dict:
        return self._request("GET", f"/my/ships/{ship}/cooldown")

    def set_flight_mode(self, ship: str, mode: str) -> dict:
        return self._request(
            "PATCH",
            f"/my/ships/{ship}/nav",
            json={
                "flightMode": mode,
            },
        )

    # --- Jump Gate ---

    def get_jump_gate(self, system: str, waypoint: str) -> dict:
        return self._request("GET", f"/systems/{system}/waypoints/{waypoint}/jump-gate")

    # --- Construction ---

    def get_construction(self, system: str, waypoint: str) -> dict:
        return self._request(
            "GET", f"/systems/{system}/waypoints/{waypoint}/construction"
        )

    def supply_construction(
        self, system: str, waypoint: str, ship: str, symbol: str, units: int
    ) -> dict:
        return self._request(
            "POST",
            f"/systems/{system}/waypoints/{waypoint}/construction/supply",
            json={
                "shipSymbol": ship,
                "tradeSymbol": symbol,
                "units": units,
            },
        )
