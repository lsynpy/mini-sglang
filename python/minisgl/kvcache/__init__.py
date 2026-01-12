from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry, init_logger

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseCacheManager,
    BaseKVCache,
    KVCacheLayout,
    SizeInfo,
)

logger = init_logger(__name__)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BaseCacheManager: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache(
    model_config: ModelConfig,
    num_pages: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_layout: KVCacheLayout = KVCacheLayout.LayerFirst,
) -> BaseKVCache:
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    shape_desc = (
        f"shape: [2, {model_config.num_layers}(#layers), {num_pages}(#pages)"
        f", {model_config.num_kv_heads}(#heads), {model_config.head_dim}(#head_dim)]"
        if cache_layout == KVCacheLayout.LayerFirst
        else f"shape: [2, {num_pages}(#pages), {model_config.num_layers}(#layers)"
        f", {model_config.num_kv_heads}(#heads), {model_config.head_dim}(#head_dim)]"
    )
    logger.info(f"Creating {MHAKVCache.__name__} KV cache with {cache_layout}, {shape_desc}")
    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        kv_layout=cache_layout,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache_manager(device: torch.device):
    from .naive_manager import NaiveCacheManager

    return NaiveCacheManager(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache_manager(device: torch.device):
    from .radix_manager import RadixCacheManager

    return RadixCacheManager(device=device)


def create_cache_manager(device: torch.device, type: str) -> BaseCacheManager:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache",
    "create_cache_manager",
    "BaseKVCache",
    "KVCacheLayout",
    "BaseCacheHandle",
    "BaseCacheManager",
    "SizeInfo",
    "SUPPORTED_CACHE_MANAGER",
]
