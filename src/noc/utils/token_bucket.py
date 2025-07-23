"""
令牌桶(Token Bucket)算法实现，用于NoC带宽限制。

提供简单且高效的速率限制机制，支持分数令牌，
用于限制IP接口的数据传输速率。
"""

from typing import Optional


class TokenBucket:
    """
    令牌桶算法实现，用于速率限制。
    
    支持分数令牌，能够精确控制数据传输速率。
    主要用于限制不同类型IP（DDR、L2M、DMA等）的带宽使用。
    
    Attributes:
        rate: 每周期添加的令牌数（浮点数）
        bucket_size: 桶的最大容量（浮点数）
        tokens: 当前令牌数量（浮点数）
        last_cycle: 上次填充令牌的周期
    """
    
    def __init__(self, rate: float = 1.0, bucket_size: float = 10.0):
        """
        初始化令牌桶。
        
        Args:
            rate: 每周期添加的令牌数，通常计算为 bandwidth_limit / flit_size
            bucket_size: 桶的最大容量，限制突发传输的最大量
        """
        self.rate = float(rate)
        self.bucket_size = float(bucket_size)
        self.tokens = self.bucket_size  # 初始时桶是满的
        self.last_cycle = 0
        
    def consume(self, num: float = 1.0) -> bool:
        """
        尝试消耗指定数量的令牌。
        
        Args:
            num: 需要消耗的令牌数量
            
        Returns:
            如果有足够的令牌返回True，否则返回False
        """
        # 添加小的误差容限，避免浮点数精度问题
        if self.tokens + 1e-8 >= num:
            self.tokens -= num
            return True
        return False
        
    def refill(self, cycle: int) -> None:
        """
        根据经过的周期数重新填充令牌。
        
        Args:
            cycle: 当前周期数
        """
        # 计算经过的周期数
        dt = cycle - self.last_cycle
        if dt <= 0:
            return
            
        # 添加分数令牌
        added = dt * self.rate
        
        # 更新上次填充时间
        self.last_cycle = cycle
        
        # 令牌数不能超过桶的容量
        self.tokens = min(self.tokens + added, self.bucket_size)
        
    def get_tokens(self) -> float:
        """
        获取当前可用的令牌数量。
        
        Returns:
            当前令牌数量
        """
        return self.tokens
        
    def get_rate(self) -> float:
        """
        获取令牌生成速率。
        
        Returns:
            每周期生成的令牌数
        """
        return self.rate
        
    def get_bucket_size(self) -> float:
        """
        获取桶的最大容量。
        
        Returns:
            桶的最大容量
        """
        return self.bucket_size
        
    def reset(self) -> None:
        """重置令牌桶到初始状态。"""
        self.tokens = self.bucket_size
        self.last_cycle = 0
        
    def is_empty(self) -> bool:
        """
        检查令牌桶是否为空。
        
        Returns:
            如果令牌数接近0返回True
        """
        return self.tokens < 1e-8
        
    def is_full(self) -> bool:
        """
        检查令牌桶是否已满。
        
        Returns:
            如果令牌数接近最大容量返回True
        """
        return abs(self.tokens - self.bucket_size) < 1e-8
        
    def __str__(self) -> str:
        """返回令牌桶状态的字符串表示。"""
        return f"TokenBucket(rate={self.rate}, size={self.bucket_size}, tokens={self.tokens:.2f})"
        
    def __repr__(self) -> str:
        """返回令牌桶的详细表示。"""
        return f"TokenBucket(rate={self.rate}, bucket_size={self.bucket_size}, tokens={self.tokens}, last_cycle={self.last_cycle})"