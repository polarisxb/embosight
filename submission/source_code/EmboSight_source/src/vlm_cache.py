"""VLMCache: episode 级 in-memory cache, 按 (image_hash, prompt_hash) 去重。"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class VLMCache:
    def __init__(self, max_size: int = 100):
        self._max = max_size
        self._store: OrderedDict[str, str] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    @staticmethod
    def _key(image_path: str, prompt: str) -> str:
        with open(image_path, "rb") as f:
            img_hash = hashlib.sha256(f.read()).hexdigest()
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return f"{img_hash}::{prompt_hash}"
    
    def get(self, image_path: str, prompt: str) -> Optional[str]:
        try:
            key = self._key(image_path, prompt)
        except FileNotFoundError:
            self._misses += 1
            return None
        if key in self._store:
            self._hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self._misses += 1
        return None
    
    def put(self, image_path: str, prompt: str, response: str) -> None:
        try:
            key = self._key(image_path, prompt)
        except FileNotFoundError:
            return
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = response
            return
        self._store[key] = response
        if len(self._store) > self._max:
            self._store.popitem(last=False)
    
    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0
    
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "size": len(self._store),
        }
