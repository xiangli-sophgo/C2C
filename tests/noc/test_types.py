"""
测试NoC类型定义和枚举。
"""

import unittest
from dataclasses import asdict

from src.noc.utils.types import (
    TopologyType,
    RoutingStrategy,
    FlowControlType,
    BufferType,
    TrafficPattern,
    NodeType,
    LinkType,
    Priority,
    SimulationState,
    EventType,
    NoCMetrics,
    NoCConfiguration,
    LinkMetrics,
    NodeMetrics,
    TrafficFlow,
    SimulationEvent,
    QoSRequirement,
    FaultModel,
)


class TestEnums(unittest.TestCase):
    """测试枚举类型。"""

    def test_topology_type_enum(self):
        """测试拓扑类型枚举。"""
        self.assertEqual(TopologyType.CROSSRING.value, "crossring")
        self.assertEqual(TopologyType.MESH.value, "mesh")
        self.assertEqual(TopologyType.RING.value, "ring")
        self.assertEqual(TopologyType.TORUS.value, "torus")

    def test_routing_strategy_enum(self):
        """测试路由策略枚举。"""
        self.assertEqual(RoutingStrategy.SHORTEST.value, "shortest")
        self.assertEqual(RoutingStrategy.LOAD_BALANCED.value, "load_balanced")
        self.assertEqual(RoutingStrategy.ADAPTIVE.value, "adaptive")

    def test_flow_control_enum(self):
        """测试Tag类型枚举。"""
        self.assertEqual(FlowControlType.WORMHOLE.value, "wormhole")
        self.assertEqual(FlowControlType.STORE_AND_FORWARD.value, "store_and_forward")

    def test_priority_enum(self):
        """测试优先级枚举。"""
        self.assertEqual(Priority.LOW.value, 0)
        self.assertEqual(Priority.MEDIUM.value, 1)
        self.assertEqual(Priority.HIGH.value, 2)
        self.assertEqual(Priority.CRITICAL.value, 3)


class TestDataClasses(unittest.TestCase):
    """测试数据类。"""

    def test_noc_metrics_creation(self):
        """测试NoC性能指标创建。"""
        metrics = NoCMetrics()
        self.assertEqual(metrics.average_latency, 0.0)
        self.assertEqual(metrics.throughput, 0.0)
        self.assertEqual(metrics.average_hop_count, 0.0)
        self.assertEqual(metrics.network_diameter, 0)
        self.assertIsInstance(metrics.custom_metrics, dict)

    def test_noc_metrics_update(self):
        """测试NoC性能指标更新。"""
        metrics = NoCMetrics()
        metrics.average_latency = 10.5
        metrics.throughput = 100.0
        metrics.custom_metrics["test_metric"] = 42

        self.assertEqual(metrics.average_latency, 10.5)
        self.assertEqual(metrics.throughput, 100.0)
        self.assertEqual(metrics.custom_metrics["test_metric"], 42)

    def test_noc_configuration_creation(self):
        """测试NoC配置创建。"""
        config = NoCConfiguration()
        self.assertEqual(config.topology_type, TopologyType.MESH)
        self.assertEqual(config.num_nodes, 16)
        self.assertEqual(config.routing_strategy, RoutingStrategy.SHORTEST)
        self.assertEqual(config.flit_size, 64)
        self.assertEqual(config.packet_size, 512)

    def test_noc_configuration_custom_params(self):
        """测试NoC配置自定义参数。"""
        config = NoCConfiguration()
        config.custom_params["test_param"] = "test_value"
        self.assertEqual(config.custom_params["test_param"], "test_value")

    def test_link_metrics_creation(self):
        """测试链路指标创建。"""
        link_metrics = LinkMetrics()
        self.assertEqual(link_metrics.utilization, 0.0)
        self.assertEqual(link_metrics.bandwidth, 0.0)
        self.assertEqual(link_metrics.total_flits, 0)

    def test_node_metrics_creation(self):
        """测试节点指标创建。"""
        node_metrics = NodeMetrics()
        self.assertEqual(node_metrics.input_buffer_occupancy, 0.0)
        self.assertEqual(node_metrics.packets_processed, 0)
        self.assertIsInstance(node_metrics.routing_decisions, dict)

    def test_traffic_flow_creation(self):
        """测试流量流创建。"""
        flow = TrafficFlow(source=0, destination=5)
        self.assertEqual(flow.source, 0)
        self.assertEqual(flow.destination, 5)
        self.assertEqual(flow.priority, Priority.MEDIUM)
        self.assertEqual(flow.packet_size, 512)

    def test_simulation_event_creation(self):
        """测试仿真事件创建。"""
        event = SimulationEvent(timestamp=100.0, event_type=EventType.PACKET_INJECTION, node_id=1)
        self.assertEqual(event.timestamp, 100.0)
        self.assertEqual(event.event_type, EventType.PACKET_INJECTION)
        self.assertEqual(event.node_id, 1)

    def test_qos_requirement_creation(self):
        """测试QoS需求创建。"""
        qos = QoSRequirement(max_latency=50.0, min_bandwidth=10.0, priority=Priority.HIGH)
        self.assertEqual(qos.max_latency, 50.0)
        self.assertEqual(qos.min_bandwidth, 10.0)
        self.assertEqual(qos.priority, Priority.HIGH)

    def test_fault_model_creation(self):
        """测试故障模型创建。"""
        fault = FaultModel(fault_type="transient", affected_component="link", fault_rate=0.01)
        self.assertEqual(fault.fault_type, "transient")
        self.assertEqual(fault.affected_component, "link")
        self.assertEqual(fault.fault_rate, 0.01)


class TestConstants(unittest.TestCase):
    """测试常量定义。"""

    def test_default_values(self):
        """测试默认值常量。"""
        from src.noc.utils.types import DEFAULT_FLIT_SIZE, DEFAULT_PACKET_SIZE, DEFAULT_BUFFER_SIZE, DEFAULT_LINK_BANDWIDTH, DEFAULT_CLOCK_FREQUENCY, MAX_NODES, MAX_DIMENSIONS

        self.assertEqual(DEFAULT_FLIT_SIZE, 64)
        self.assertEqual(DEFAULT_PACKET_SIZE, 512)
        self.assertEqual(DEFAULT_BUFFER_SIZE, 8)
        self.assertEqual(DEFAULT_LINK_BANDWIDTH, 1.0)
        self.assertEqual(DEFAULT_CLOCK_FREQUENCY, 1.0)
        self.assertEqual(MAX_NODES, 10000)
        self.assertEqual(MAX_DIMENSIONS, 10)


if __name__ == "__main__":
    unittest.main()
