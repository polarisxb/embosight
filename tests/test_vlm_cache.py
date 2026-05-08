"""VLMCache 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestVLMCache:
    @pytest.fixture
    def tmp_image(self, tmp_path):
        """生成一张极小的测试图。"""
        from PIL import Image
        p = tmp_path / "img.png"
        Image.new("RGB", (4, 4), (255, 0, 0)).save(p)
        return str(p)
    
    def test_miss_then_hit(self, tmp_image):
        from src.vlm_cache import VLMCache
        cache = VLMCache(max_size=10)
        assert cache.get(tmp_image, "prompt A") is None
        cache.put(tmp_image, "prompt A", "response 1")
        assert cache.get(tmp_image, "prompt A") == "response 1"
    
    def test_different_prompt_no_collision(self, tmp_image):
        from src.vlm_cache import VLMCache
        cache = VLMCache()
        cache.put(tmp_image, "prompt A", "response A")
        assert cache.get(tmp_image, "prompt B") is None
    
    def test_lru_eviction(self, tmp_path):
        """超过 max_size 时, 最久未用的被剔除。"""
        from PIL import Image
        from src.vlm_cache import VLMCache
        cache = VLMCache(max_size=2)
        paths = []
        for i in range(3):
            p = tmp_path / f"img{i}.png"
            Image.new("RGB", (4, 4), (i * 50, 0, 0)).save(p)
            paths.append(str(p))
            cache.put(str(p), "p", f"r{i}")
        # img0 应已被剔除
        assert cache.get(paths[0], "p") is None
        assert cache.get(paths[1], "p") == "r1"
        assert cache.get(paths[2], "p") == "r2"
    
    def test_clear_empties(self, tmp_image):
        from src.vlm_cache import VLMCache
        cache = VLMCache()
        cache.put(tmp_image, "p", "r")
        cache.clear()
        assert cache.get(tmp_image, "p") is None
    
    def test_stats(self, tmp_image):
        from src.vlm_cache import VLMCache
        cache = VLMCache()
        cache.get(tmp_image, "p")              # miss
        cache.put(tmp_image, "p", "r")
        cache.get(tmp_image, "p")              # hit
        cache.get(tmp_image, "q")              # miss
        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 2
        assert s["hit_rate"] == pytest.approx(1 / 3)
