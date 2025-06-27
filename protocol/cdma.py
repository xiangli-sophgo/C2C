from typing import Any, Dict
from dataclasses import dataclass
from protocol.base import BaseProtocol, ProtocolState


@dataclass
class CDMAMessage:
    """CDMA消息格式"""

    source_id: str
    destination_id: str
    message_type: str  # e.g., "send", "receive", "ack"
    tensor_shape: Any = None  # e.g., (H, W, C)
    data_type: str = None  # e.g., "float32", "int8"
    payload: Any = None  # Actual data payload
    reduce_op: str = None  # e.g., "sum", "mean"
    sequence_number: int = 0
    transaction_id: str = None


class CDMAProtocol(BaseProtocol):
    """CDMA协议实现"""

    def __init__(self, protocol_id: str, node_id: str):
        super().__init__(protocol_id)
        self._node_id = node_id
        self._transaction_counter = 0
        self._pending_transactions: Dict[str, CDMAMessage] = {}

    def _generate_transaction_id(self) -> str:
        self._transaction_counter += 1
        return f"txn_{self._node_id}_{self._transaction_counter}"

    def process_message(self, message: CDMAMessage) -> Any:
        """处理CDMA消息"""
        print(f"节点 {self._node_id} 处理CDMA消息: {message.message_type} 从 {message.source_id} 到 {message.destination_id}")
        if message.destination_id != self._node_id:
            print(f"警告：消息不属于该节点。期望 {self._node_id}，实际得到 {message.destination_id}")
            return None

        if message.message_type == "send":
            print(f"收到来自 {message.source_id} 的SEND请求。数据：{message.payload}")
            # 模拟数据处理
            self.set_state(ProtocolState.TRANSMITTING)
            # 确认发送
            ack_message = CDMAMessage(source_id=self._node_id, destination_id=message.source_id, message_type="ack", transaction_id=message.transaction_id)
            self.set_state(ProtocolState.DONE)
            return ack_message
        elif message.message_type == "receive":
            print(f"收到来自 {message.source_id} 的RECEIVE请求。准备数据。")
            self.set_state(ProtocolState.WAITING)
            # 模拟数据检索和发送回复
            response_payload = f"来自 {self._node_id} 的数据，事务ID {message.transaction_id}"
            response_message = CDMAMessage(source_id=self._node_id, destination_id=message.source_id, message_type="data_response", payload=response_payload, transaction_id=message.transaction_id)
            self.set_state(ProtocolState.TRANSMITTING)
            return response_message
        elif message.message_type == "ack":
            print(f"收到来自 {message.source_id} 的事务 {message.transaction_id} 的ACK")
            if message.transaction_id in self._pending_transactions:
                del self._pending_transactions[message.transaction_id]
                self.set_state(ProtocolState.DONE)
            else:
                print(f"警告：未知事务 {message.transaction_id} 的ACK")
            return None
        elif message.message_type == "data_response":
            print(f"收到来自 {message.source_id} 的事务 {message.transaction_id} 的DATA_RESPONSE。数据：{message.payload}")
            if message.transaction_id in self._pending_transactions:
                del self._pending_transactions[message.transaction_id]
                self.set_state(ProtocolState.DONE)
            else:
                print(f"警告：未知事务 {message.transaction_id} 的DATA_RESPONSE")
            return None
        else:
            print(f"未知的CDMA消息类型：{message.message_type}")
            self.set_state(ProtocolState.ERROR)
            return None

    def send_message(self, message: CDMAMessage):
        """发送CDMA消息 (模拟)"""
        if not message.transaction_id:
            message.transaction_id = self._generate_transaction_id()
        self._pending_transactions[message.transaction_id] = message
        self.set_state(ProtocolState.TRANSMITTING)
        print(f"节点 {self._node_id} 发送CDMA消息：{message.message_type} 到 {message.destination_id} （事务ID：{message.transaction_id}）")
        # 在实际系统中，这将涉及将消息传递给网络层
