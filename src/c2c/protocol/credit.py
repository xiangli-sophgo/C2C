from typing import Dict, Any, Optional
from dataclasses import dataclass
try:
    from config.constants import MAX_CREDIT_CAPACITY
except ImportError:
    # 如果找不到config模块，使用默认值
    MAX_CREDIT_CAPACITY = 2
import time


@dataclass
class AddressInfo:
    """地址信息结构"""
    address: int
    shape: tuple
    mem_type: str
    data_type: str = "float32"


@dataclass
class CreditWithAddressInfo:
    """携带地址信息的Credit"""
    credit_count: int
    dst_address_info: AddressInfo
    transaction_id: str
    timestamp: float
    expires_at: float


class CreditManager:
    """Credit机制管理 - 支持地址信息传递"""

    def __init__(self, node_id: str):
        self._node_id = node_id
        self._credits: Dict[str, int] = {}
        self._max_capacity = MAX_CREDIT_CAPACITY
        self._address_credits: Dict[str, CreditWithAddressInfo] = {}  # src_chip -> credit_with_address
        self._pending_address_info: Dict[str, AddressInfo] = {}  # transaction_id -> address_info

    def initialize_credits(self, destinations: list[str]):
        """为指定目的地初始化Credit"""
        for dest in destinations:
            self._credits[dest] = self._max_capacity

    def get_available_credits(self, destination: str) -> int:
        """获取指定目的地的可用Credit数量"""
        return self._credits.get(destination, 0)

    def request_credit(self, destination: str) -> bool:
        """申请Credit。如果可用，则减少并返回True，否则返回False"""
        if self._credits.get(destination, 0) > 0:
            self._credits[destination] -= 1
            print(f"节点 {self._node_id}：为 {destination} 授予信用。剩余：{self._credits[destination]}")
            return True
        print(f"节点 {self._node_id}：没有可用于 {destination} 的信用。")
        return False

    def grant_credit(self, destination: str, amount: int = 1):
        """授予Credit。增加指定目的地的Credit数量，不超过最大容量"""
        current_credits = self._credits.get(destination, 0)
        self._credits[destination] = min(self._max_capacity, current_credits + amount)
        print(f"节点 {self._node_id}：向 {destination} 授予信用。新总数：{self._credits[destination]}")

    def send_credit_with_address(self, target_chip: str, dst_info: AddressInfo, 
                               transaction_id: str, timeout_seconds: float = 30.0) -> bool:
        """发送Credit同时携带目标地址信息"""
        credit_info = CreditWithAddressInfo(
            credit_count=1,
            dst_address_info=dst_info,
            transaction_id=transaction_id,
            timestamp=time.time(),
            expires_at=time.time() + timeout_seconds
        )
        
        # 在实际系统中，这里会通过硬件总线发送Credit+地址信息
        print(f"节点 {self._node_id}：向 {target_chip} 发送Credit+地址信息")
        print(f"  目标地址: 0x{dst_info.address:08x}, 形状: {dst_info.shape}")
        print(f"  内存类型: {dst_info.mem_type}, 事务ID: {transaction_id}")
        
        return True

    def receive_credit_with_address(self, src_chip: str, credit_info: CreditWithAddressInfo):
        """接收来自其他芯片的Credit+地址信息"""
        self._address_credits[src_chip] = credit_info
        print(f"节点 {self._node_id}：收到来自 {src_chip} 的Credit+地址信息")
        print(f"  目标地址: 0x{credit_info.dst_address_info.address:08x}")
        print(f"  形状: {credit_info.dst_address_info.shape}")

    def check_and_consume_credit(self, src_chip: str) -> Optional[AddressInfo]:
        """检查并消费Credit，返回目标地址信息"""
        if src_chip not in self._address_credits:
            print(f"节点 {self._node_id}：没有来自 {src_chip} 的Credit+地址信息")
            return None
        
        credit_info = self._address_credits[src_chip]
        
        # 检查是否过期
        if time.time() > credit_info.expires_at:
            print(f"节点 {self._node_id}：来自 {src_chip} 的Credit已过期")
            del self._address_credits[src_chip]
            return None
        
        # 消费Credit
        dst_address_info = credit_info.dst_address_info
        del self._address_credits[src_chip]
        
        print(f"节点 {self._node_id}：消费来自 {src_chip} 的Credit")
        print(f"  获取目标地址信息: 0x{dst_address_info.address:08x}, 形状: {dst_address_info.shape}")
        
        return dst_address_info

    def get_credit_status(self) -> Dict[str, Any]:
        """获取所有目的地的Credit状态"""
        return {
            "basic_credits": self._credits,
            "address_credits": {
                chip_id: {
                    "transaction_id": info.transaction_id,
                    "address": f"0x{info.dst_address_info.address:08x}",
                    "shape": info.dst_address_info.shape,
                    "mem_type": info.dst_address_info.mem_type,
                    "expires_at": info.expires_at
                }
                for chip_id, info in self._address_credits.items()
            }
        }

    def cleanup_expired_credits(self):
        """清理过期的Credit信息"""
        current_time = time.time()
        expired_chips = [
            chip_id for chip_id, credit_info in self._address_credits.items()
            if current_time > credit_info.expires_at
        ]
        
        for chip_id in expired_chips:
            print(f"节点 {self._node_id}：清理来自 {chip_id} 的过期Credit")
            del self._address_credits[chip_id]

    def has_credit_with_address(self, src_chip: str) -> bool:
        """检查是否有来自指定芯片的Credit+地址信息"""
        if src_chip not in self._address_credits:
            return False
        
        # 检查是否过期
        if time.time() > self._address_credits[src_chip].expires_at:
            del self._address_credits[src_chip]
            return False
        
        return True
