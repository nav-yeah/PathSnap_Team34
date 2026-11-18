"""Thin HTTP client used by the UI (Laptop 2). Converts transport failures into
one exception type carrying a message an evaluator can act on."""
import requests


class ApiError(RuntimeError):
    pass


class PathSnapClient:
    def __init__(self, base_url, timeout=(5, 120)):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout          # (connect, read) seconds

    def _call(self, method, path, **kwargs):
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
        except requests.exceptions.ConnectTimeout:
            raise ApiError(f"Timed out connecting to {self.base_url}. Check the IP address, "
                           "that both laptops are on the same network, and the firewall on Laptop 1.")
        except requests.exceptions.ConnectionError:
            raise ApiError(f"Cannot reach the API at {self.base_url}. Is uvicorn running on "
                           "Laptop 1 with --host 0.0.0.0?")
        except requests.exceptions.ReadTimeout:
            raise ApiError("The API accepted the request but did not answer in time.")
        except requests.exceptions.RequestException as error:
            raise ApiError(f"Request failed: {error}")
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"API returned HTTP {response.status_code}: {detail}")
        try:
            return response.json()
        except ValueError:
            raise ApiError("API returned a non-JSON response.")

    def health(self):
        return self._call("GET", "/health")

    def trips(self):
        return self._call("GET", "/trips")

    def predict(self, trip_id):
        return self._call("POST", "/predict", json={"trip_id": trip_id})
