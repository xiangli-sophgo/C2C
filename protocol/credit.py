from typing import Dict, Any
from utils.constants import MAX_CREDIT_CAPACITY

class CreditManager:
    """Credit机制管理"""
    def __init__(self, node_id: str):
        self._node_id = node_id
        self._credits: Dict[str, int] = {}
        self._max_capacity = MAX_CREDIT_CAPACITY

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
            print(f"Node {self._node_id}: Credit granted for {destination}. Remaining: {self._credits[destination]}")
            return True
        print(f"Node {self._node_id}: No credit available for {destination}.")
        return False

    def grant_credit(self, destination: str, amount: int = 1):
        """授予Credit。增加指定目的地的Credit数量，不超过最大容量"""
        current_credits = self._credits.get(destination, 0)
        self._credits[destination] = min(self._max_capacity, current_credits + amount)
        print(f"Node {self._node_id}: Credit granted to {destination}. New total: {self._credits[destination]}")

    def get_credit_status(self) -> Dict[str, int]:
        """获取所有目的地的Credit状态"""
        return self._credits

    # TODO: Add timeout handling mechanism
