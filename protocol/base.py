from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

class ProtocolState(Enum):
    """协议状态枚举"""
    IDLE = "idle"
    WAITING = "waiting"
    TRANSMITTING = "transmitting"
    DONE = "done"
    ERROR = "error"

class BaseProtocol(ABC):
    """协议抽象基类"""
    def __init__(self, protocol_id: str):
        self._protocol_id = protocol_id
        self._state: ProtocolState = ProtocolState.IDLE

    @property
    def protocol_id(self) -> str:
        return self._protocol_id

    @property
    def state(self) -> ProtocolState:
        return self._state

    def set_state(self, new_state: ProtocolState):
        """设置协议状态"""
        self._state = new_state

    @abstractmethod
    def process_message(self, message: Any) -> Any:
        """处理消息接口"""
        pass

    @abstractmethod
    def send_message(self, message: Any):
        """发送消息接口"""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}(id='{self.protocol_id}', state='{self.state.value}')>"
