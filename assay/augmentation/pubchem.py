"""PubChem REST API client with JSON file caching."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

import requests

_DATA_DIR = Path(__file__).resolve().parent / "data"
_CACHE_FILE = _DATA_DIR / "pubchem_cache.json"
_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_REQUEST_INTERVAL = 0.25


class PubChemClient:
    """PubChem REST API client with persistent JSON cache."""

    def __init__(self, cache_file: str | Path | None = None):
        self.cache_file = Path(cache_file) if cache_file else _CACHE_FILE
        self._cache: dict = self._load_cache()
        self._last_request_time: float = 0.0

    def _load_cache(self) -> dict:
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < _REQUEST_INTERVAL:
            time.sleep(_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str) -> Optional[dict]:
        self._rate_limit()
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            return None
        except (requests.RequestException, json.JSONDecodeError):
            return None

    def get_properties(self, name: str) -> Optional[dict]:
        """Get compound properties (IUPACName, MolecularFormula) by name."""
        cache_key = f"props:{name.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        url = f"{_BASE_URL}/compound/name/{requests.utils.quote(name)}/property/IUPACName,MolecularFormula/JSON"
        data = self._get(url)

        result = None
        if data and "PropertyTable" in data:
            props = data["PropertyTable"].get("Properties", [])
            if props:
                result = {
                    "CID": props[0].get("CID"),
                    "IUPACName": props[0].get("IUPACName"),
                    "MolecularFormula": props[0].get("MolecularFormula"),
                }

        self._cache[cache_key] = result
        self._save_cache()
        return result

    def get_synonyms(self, name: str, max_synonyms: int = 20) -> Optional[list[str]]:
        cache_key = f"syns:{name.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        props = self.get_properties(name)
        if not props or not props.get("CID"):
            self._cache[cache_key] = None
            self._save_cache()
            return None

        cid = props["CID"]
        url = f"{_BASE_URL}/compound/cid/{cid}/synonyms/JSON"
        data = self._get(url)

        result = None
        if data and "InformationList" in data:
            info = data["InformationList"].get("Information", [])
            if info:
                syns = info[0].get("Synonym", [])
                result = syns[:max_synonyms]

        self._cache[cache_key] = result
        self._save_cache()
        return result

    def get_common_name(self, name: str) -> Optional[str]:
        cache_key = f"common:{name.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        synonyms = self.get_synonyms(name)
        result = None
        if synonyms:
            registry_num = re.compile(r'^\d{2,7}-\d{2}-\d$')
            for syn in synonyms:
                if not registry_num.match(syn) and len(syn) > 1:
                    result = syn
                    break

        self._cache[cache_key] = result
        self._save_cache()
        return result

    def get_formula(self, name: str) -> Optional[str]:
        props = self.get_properties(name)
        return props.get("MolecularFormula") if props else None

    def get_iupac_name(self, name: str) -> Optional[str]:
        props = self.get_properties(name)
        return props.get("IUPACName") if props else None

    @property
    def cache_size(self) -> int:
        return len(self._cache)
