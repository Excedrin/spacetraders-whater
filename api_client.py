import requests

BASE_URL = "https://api.spacetraders.io/v2"


class SpaceTradersClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    def _request(self, method: str, path: str, **kwargs) -> dict:
        if method == "POST" and "json" not in kwargs:
            kwargs["json"] = {}
        resp = self.session.request(method, f"{BASE_URL}{path}", **kwargs)
        body = resp.json()
        if "error" in body:
            return {"error": body["error"].get("message", str(body["error"]))}
        return body.get("data", body)

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
        return self._request("GET", f"/systems/{system}/waypoints", params=params)

    def get_shipyard(self, system: str, waypoint: str) -> dict:
        return self._request("GET", f"/systems/{system}/waypoints/{waypoint}/shipyard")

    def get_market(self, system: str, waypoint: str) -> dict:
        return self._request("GET", f"/systems/{system}/waypoints/{waypoint}/market")

    # --- Ships ---

    def list_ships(self) -> dict:
        return self._request("GET", "/my/ships")

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

    def jettison(self, ship: str, symbol: str, units: int) -> dict:
        return self._request("POST", f"/my/ships/{ship}/jettison", json={
            "symbol": symbol,
            "units": units,
        })
