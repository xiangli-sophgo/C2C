from enum import Enum


class MemoryType(Enum):
    """内存类型枚举"""
    GMEM = "GMEM"    # 全局内存
    L2M = "L2M"      # L2缓存内存
    LMEM = "LMEM"    # 本地内存