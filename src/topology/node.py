from typing import List, Dict, Any
from src.topology.base import BaseNode


class ChipNode(BaseNode):
    """芯片节点 - 代表SG2260E芯片"""

    def __init__(self, chip_id: str, board_id: str, cdma_engines: int = 10, memory_types: List[str] = None, properties: Dict[str, Any] = None):
        super().__init__(node_id=chip_id, node_type="chip", properties=properties)
        self.add_property("board_id", board_id)
        self.add_property("cdma_engines", cdma_engines)
        self.add_property("memory_types", memory_types if memory_types is not None else [])

    @property
    def chip_id(self) -> str:
        return self.node_id

    @property
    def board_id(self) -> str:
        return self.get_property("board_id")

    @property
    def cdma_engines(self) -> int:
        return self.get_property("cdma_engines")

    @property
    def memory_types(self) -> List[str]:
        return self.get_property("memory_types")

    def send_message(self, destination_node: BaseNode, message: Any):
        """模拟芯片节点发送消息"""
        print(f"ChipNode {self.node_id} sending message to {destination_node.node_id}: {message}")
        # Placeholder for actual message sending logic

    def receive_message(self, sender_node: BaseNode, message: Any):
        """模拟芯片节点接收消息"""
        print(f"ChipNode {self.node_id} receiving message from {sender_node.node_id}: {message}")
        # Placeholder for actual message receiving logic


class SwitchNode(BaseNode):
    """PCIe Switch节点"""

    def __init__(self, switch_id: str, port_count: int, bandwidth: float, properties: Dict[str, Any] = None):
        super().__init__(node_id=switch_id, node_type="switch", properties=properties)
        self.add_property("port_count", port_count)
        self.add_property("bandwidth", bandwidth)

    @property
    def switch_id(self) -> str:
        return self.node_id

    @property
    def port_count(self) -> int:
        return self.get_property("port_count")

    @property
    def bandwidth(self) -> float:
        return self.get_property("bandwidth")

    def send_message(self, destination_node: BaseNode, message: Any):
        """模拟Switch节点发送消息"""
        print(f"SwitchNode {self.node_id} forwarding message to {destination_node.node_id}: {message}")
        # Placeholder for actual message forwarding logic

    def receive_message(self, sender_node: BaseNode, message: Any):
        """模拟Switch节点接收消息"""
        print(f"SwitchNode {self.node_id} receiving message from {sender_node.node_id}: {message}")
        # Placeholder for actual message receiving logic


class HostNode(BaseNode):
    """Host PC节点"""

    def __init__(self, host_id: str, pcie_lanes: int, properties: Dict[str, Any] = None):
        super().__init__(node_id=host_id, node_type="host", properties=properties)
        self.add_property("pcie_lanes", pcie_lanes)

    @property
    def host_id(self) -> str:
        return self.node_id

    @property
    def pcie_lanes(self) -> int:
        return self.get_property("pcie_lanes")

    def send_message(self, destination_node: BaseNode, message: Any):
        """模拟Host节点发送消息"""
        print(f"HostNode {self.node_id} sending message to {destination_node.node_id}: {message}")
        # Placeholder for actual message sending logic

    def receive_message(self, sender_node: BaseNode, message: Any):
        """模拟Host节点接收消息"""
        print(f"HostNode {self.node_id} receiving message from {sender_node.node_id}: {message}")
        # Placeholder for actual message receiving logic
