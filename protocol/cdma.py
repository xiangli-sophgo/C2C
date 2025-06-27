from typing import Any, Dict
from dataclasses import dataclass
from protocol.base import BaseProtocol, ProtocolState

@dataclass
class CDMAMessage:
    """CDMA消息格式"""
    source_id: str
    destination_id: str
    message_type: str # e.g., "send", "receive", "ack"
    tensor_shape: Any = None # e.g., (H, W, C)
    data_type: str = None # e.g., "float32", "int8"
    payload: Any = None # Actual data payload
    reduce_op: str = None # e.g., "sum", "mean"
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
        print(f"Node {self._node_id} processing CDMA message: {message.message_type} from {message.source_id} to {message.destination_id}")
        if message.destination_id != self._node_id:
            print(f"Warning: Message not for this node. Expected {self._node_id}, got {message.destination_id}")
            return None

        if message.message_type == "send":
            print(f"Received SEND request from {message.source_id}. Data: {message.payload}")
            # Simulate data processing
            self.set_state(ProtocolState.TRANSMITTING)
            # Acknowledge the send
            ack_message = CDMAMessage(
                source_id=self._node_id,
                destination_id=message.source_id,
                message_type="ack",
                transaction_id=message.transaction_id
            )
            self.set_state(ProtocolState.DONE)
            return ack_message
        elif message.message_type == "receive":
            print(f"Received RECEIVE request from {message.source_id}. Preparing data.")
            self.set_state(ProtocolState.WAITING)
            # Simulate data retrieval and sending back
            response_payload = f"Data from {self._node_id} for {message.transaction_id}"
            response_message = CDMAMessage(
                source_id=self._node_id,
                destination_id=message.source_id,
                message_type="data_response",
                payload=response_payload,
                transaction_id=message.transaction_id
            )
            self.set_state(ProtocolState.TRANSMITTING)
            return response_message
        elif message.message_type == "ack":
            print(f"Received ACK for transaction {message.transaction_id} from {message.source_id}")
            if message.transaction_id in self._pending_transactions:
                del self._pending_transactions[message.transaction_id]
                self.set_state(ProtocolState.DONE)
            else:
                print(f"Warning: ACK for unknown transaction {message.transaction_id}")
            return None
        elif message.message_type == "data_response":
            print(f"Received DATA_RESPONSE for transaction {message.transaction_id} from {message.source_id}. Data: {message.payload}")
            if message.transaction_id in self._pending_transactions:
                del self._pending_transactions[message.transaction_id]
                self.set_state(ProtocolState.DONE)
            else:
                print(f"Warning: DATA_RESPONSE for unknown transaction {message.transaction_id}")
            return None
        else:
            print(f"Unknown CDMA message type: {message.message_type}")
            self.set_state(ProtocolState.ERROR)
            return None

    def send_message(self, message: CDMAMessage):
        """发送CDMA消息 (模拟)"""
        if not message.transaction_id:
            message.transaction_id = self._generate_transaction_id()
        self._pending_transactions[message.transaction_id] = message
        self.set_state(ProtocolState.TRANSMITTING)
        print(f"Node {self._node_id} sending CDMA message: {message.message_type} to {message.destination_id} (Txn ID: {message.transaction_id})")
        # In a real system, this would involve passing the message to the network layer

