import requests
import re
from datetime import datetime, timedelta

class UsdaFoodDataService:

    def __init__(self):
        self.api_key = "SgYYLonfWdLOU229LXZ87qYaooIVj8uwoLhL1Pan"
        self.base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "grocery-ml"})
        self.cache = {}
        self.cache_ttl = timedelta(days=7)
#
    def _normalize_text(self, value):
        if value is None:
            return ""
        value = value.lower()
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        value = re.sub(r"\s+", " ", value)
        return value.strip()
#
    def _is_cache_valid(self, cache_entry):
        return datetime.now() - cache_entry["time"] < self.cache_ttl
#
    def search_food(self, item_name):
        item_norm = self._normalize_text(item_name)

        if item_norm in self.cache:
            if self._is_cache_valid(self.cache[item_norm]):
                return self.cache[item_norm]["data"]

        params = {
            "query": item_name,
            "pageSize": 10,
            "api_key": self.api_key
        }

        response = self.session.get(self.base_url, params=params, timeout=10)
        foods = response.json().get("foods", [])

        self.cache[item_norm] = {
            "time": datetime.now(),
            "data": foods
        }

        return foods
#
