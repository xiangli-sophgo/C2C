from typing import Dict, Any
from src.c2c.topology.base import BaseLink, BaseNode
import config.constants as constants


class PCIeLink(BaseLink):
    """PCIe链路"""

    def __init__(self, link_id: str, endpoint_a: BaseNode, endpoint_b: BaseNode, pcie_type: str, properties: Dict[str, Any] = None):
        super().__init__(link_id, endpoint_a, endpoint_b, "PCIe", properties)
        self.add_property("pcie_type", pcie_type)  # e.g., "x4", "x8"
        self.add_property("rc_ep_config", self.get_property("rc_ep_config", "unknown"))  # RC/EP配置

    @property
    def bandwidth(self) -> float:
        """根据PCIe类型返回带宽"""
        pcie_type = self.get_property("pcie_type")
        if pcie_type == "x8":
            return constants.DEFAULT_PCIE_X8_BANDWIDTH
        elif pcie_type == "x4":
            return constants.DEFAULT_PCIE_X4_BANDWIDTH
        else:
            return 0.0  # Unknown PCIe type

    @property
    def latency(self) -> float:
        """PCIe链路延迟"""
        return constants.PCIE_BASE_LATENCY  # + transmission delay (simplified for now)

    @property
    def status(self) -> str:
        """PCIe链路状态"""
        return "active"


class C2CDirectLink(BaseLink):
    """C2C直连链路"""

    def __init__(self, link_id: str, endpoint_a: BaseNode, endpoint_b: BaseNode, properties: Dict[str, Any] = None):
        super().__init__(link_id, endpoint_a, endpoint_b, "C2C_Direct", properties)

    @property
    def bandwidth(self) -> float:
        """C2C直连链路带宽"""
        return constants.DEFAULT_C2C_BANDWIDTH

    @property
    def latency(self) -> float:
        """C2C直连链路延迟"""
        return constants.C2C_BASE_LATENCY

    @property
    def status(self) -> str:
        """C2C直连链路状态"""
        return "active"
