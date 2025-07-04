"""
测试NoC拓扑工厂类。
"""

import unittest
from unittest.mock import Mock, patch

from src.noc.utils.factory import (
    NoCTopologyFactory,
    TopologyRegistryError,
    TopologyCreationError,
    TopologyValidator,
    MeshTopologyValidator,
    RingTopologyValidator,
    CrossRingTopologyValidator,
    create_mesh_topology,
    create_ring_topology,
    create_crossring_topology,
    auto_select_topology,
)
from src.noc.base.topology import BaseNoCTopology
from src.noc.base.config import BaseNoCConfig, CrossRingCompatibleConfig
from src.noc.utils.types import TopologyType, RoutingStrategy


class MockTopology(BaseNoCTopology):
    """用于测试的模拟拓扑类。"""

    def build_topology(self):
        """实现抽象方法。"""
        self._adjacency_matrix = [[0 for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]

    def get_neighbors(self, node_id):
        """实现抽象方法。"""
        return []

    def calculate_shortest_path(self, src, dst):
        """实现抽象方法。"""
        return [src, dst] if src != dst else [src]

    def get_node_position(self, node_id):
        """实现抽象方法。"""
        return (node_id % 4, node_id // 4)

    def validate_topology(self):
        """实现抽象方法。"""
        return True, None


class MockConfig(BaseNoCConfig):
    """用于测试的模拟配置类。"""

    def validate_config(self):
        """实现抽象方法。"""
        return True, None

    def get_topology_params(self):
        """实现抽象方法。"""
        return {"test_param": "test_value"}


class TestTopologyValidators(unittest.TestCase):
    """测试拓扑验证器。"""

    def test_mesh_topology_validator(self):
        """测试Mesh拓扑验证器。"""
        validator = MeshTopologyValidator()

        # 有效的Mesh配置（16节点 = 4x4）
        config = MockConfig(TopologyType.MESH)
        config.num_nodes = 16
        valid, message = validator.validate(config)
        self.assertTrue(valid)

        # 无效的Mesh配置（15节点不是完全平方数）
        config.num_nodes = 15
        valid, message = validator.validate(config)
        self.assertFalse(valid)
        self.assertIn("不是完全平方数", message)

        # 无效节点数
        config.num_nodes = 0
        valid, message = validator.validate(config)
        self.assertFalse(valid)
        self.assertIn("节点数必须为正数", message)

    def test_ring_topology_validator(self):
        """测试Ring拓扑验证器。"""
        validator = RingTopologyValidator()

        # 有效的Ring配置
        config = MockConfig(TopologyType.RING)
        config.num_nodes = 8
        valid, message = validator.validate(config)
        self.assertTrue(valid)

        # 无效的Ring配置（节点数太少）
        config.num_nodes = 2
        valid, message = validator.validate(config)
        self.assertFalse(valid)
        self.assertIn("至少需要3个节点", message)

    def test_crossring_topology_validator(self):
        """测试CrossRing拓扑验证器。"""
        validator = CrossRingTopologyValidator()

        # 有效的CrossRing配置
        config = CrossRingCompatibleConfig()
        valid, message = validator.validate(config)
        self.assertTrue(valid)

        # 无效的配置类型
        config = MockConfig(TopologyType.CROSSRING)
        valid, message = validator.validate(config)
        self.assertFalse(valid)
        self.assertIn("需要CrossRingCompatibleConfig配置", message)


class TestNoCTopologyFactory(unittest.TestCase):
    """测试NoC拓扑工厂类。"""

    def setUp(self):
        """设置测试环境。"""
        # 清理注册表
        NoCTopologyFactory._topology_registry.clear()
        NoCTopologyFactory._config_creators.clear()
        NoCTopologyFactory._validators.clear()

        # 注册测试拓扑
        NoCTopologyFactory.register_topology(TopologyType.MESH, MockTopology)

    def tearDown(self):
        """清理测试环境。"""
        # 恢复默认注册表状态
        NoCTopologyFactory._topology_registry.clear()
        NoCTopologyFactory._config_creators.clear()
        NoCTopologyFactory._validators.clear()

    def test_register_topology(self):
        """测试拓扑注册。"""
        # 注册新拓扑
        NoCTopologyFactory.register_topology(TopologyType.RING, MockTopology)

        # 验证注册成功
        self.assertIn(TopologyType.RING, NoCTopologyFactory._topology_registry)
        self.assertEqual(NoCTopologyFactory._topology_registry[TopologyType.RING], MockTopology)

        # 测试注册无效类型
        with self.assertRaises(TopologyRegistryError):
            NoCTopologyFactory.register_topology(TopologyType.TORUS, str)  # str不继承自BaseNoCTopology

    def test_unregister_topology(self):
        """测试拓扑注销。"""
        # 注册后注销
        NoCTopologyFactory.register_topology(TopologyType.RING, MockTopology)
        self.assertIn(TopologyType.RING, NoCTopologyFactory._topology_registry)

        NoCTopologyFactory.unregister_topology(TopologyType.RING)
        self.assertNotIn(TopologyType.RING, NoCTopologyFactory._topology_registry)

    def test_create_topology(self):
        """测试拓扑创建。"""
        config = MockConfig(TopologyType.MESH)

        # 成功创建拓扑
        topology = NoCTopologyFactory.create_topology(config, validate=False)
        self.assertIsInstance(topology, MockTopology)
        self.assertEqual(topology.config, config)

        # 不支持的拓扑类型
        config.topology_type = TopologyType.TORUS
        with self.assertRaises(TopologyCreationError):
            NoCTopologyFactory.create_topology(config)

    def test_create_config(self):
        """测试配置创建。"""
        # 创建Mesh配置
        config = NoCTopologyFactory.create_config(TopologyType.MESH, num_nodes=25)
        self.assertEqual(config.topology_type, TopologyType.MESH)
        self.assertEqual(config.num_nodes, 25)

        # 创建CrossRing配置
        config = NoCTopologyFactory.create_config(TopologyType.CROSSRING, NUM_NODE=40)
        self.assertIsInstance(config, CrossRingCompatibleConfig)

    def test_create_topology_from_type(self):
        """测试从类型创建拓扑。"""
        topology = NoCTopologyFactory.create_topology_from_type(TopologyType.MESH, num_nodes=16, validate=False)
        self.assertIsInstance(topology, MockTopology)
        self.assertEqual(topology.num_nodes, 16)

    def test_validate_config(self):
        """测试配置验证。"""
        config = MockConfig(TopologyType.MESH)

        # 基础验证通过
        valid, message = NoCTopologyFactory.validate_config(config)
        self.assertTrue(valid)

        # 基础验证失败
        config.num_nodes = -1
        valid, message = NoCTopologyFactory.validate_config(config)
        self.assertFalse(valid)
        self.assertIn("节点数必须为正数", message)

    def test_get_supported_topologies(self):
        """测试获取支持的拓扑类型。"""
        supported = NoCTopologyFactory.get_supported_topologies()
        self.assertIn(TopologyType.MESH, supported)
        self.assertEqual(len(supported), 1)  # 只注册了MESH

    def test_is_topology_supported(self):
        """测试检查拓扑支持。"""
        self.assertTrue(NoCTopologyFactory.is_topology_supported(TopologyType.MESH))
        self.assertFalse(NoCTopologyFactory.is_topology_supported(TopologyType.RING))

    def test_get_topology_info(self):
        """测试获取拓扑信息。"""
        # 支持的拓扑
        info = NoCTopologyFactory.get_topology_info(TopologyType.MESH)
        self.assertTrue(info["supported"])
        self.assertEqual(info["class_name"], "MockTopology")

        # 不支持的拓扑
        info = NoCTopologyFactory.get_topology_info(TopologyType.RING)
        self.assertFalse(info["supported"])

    def test_list_topologies(self):
        """测试列出拓扑类型。"""
        topologies = NoCTopologyFactory.list_topologies()
        self.assertIn("mesh", topologies)
        self.assertTrue(topologies["mesh"]["supported"])

    def test_clone_topology(self):
        """测试拓扑克隆。"""
        original_config = MockConfig(TopologyType.MESH)
        original_topology = NoCTopologyFactory.create_topology(original_config, validate=False)

        # 克隆拓扑
        cloned_topology = NoCTopologyFactory.clone_topology(original_topology, {"num_nodes": 25})

        self.assertIsInstance(cloned_topology, MockTopology)
        self.assertEqual(cloned_topology.num_nodes, 25)
        self.assertNotEqual(cloned_topology.config, original_topology.config)

    def test_optimize_topology(self):
        """测试拓扑优化。"""
        config = MockConfig(TopologyType.MESH)

        # 延迟优化
        optimized = NoCTopologyFactory.optimize_topology(config, "latency")
        self.assertEqual(optimized.routing_strategy, "shortest")

        # 吞吐量优化
        optimized = NoCTopologyFactory.optimize_topology(config, "throughput")
        self.assertEqual(optimized.routing_strategy, "load_balanced")

        # 功耗优化
        optimized = NoCTopologyFactory.optimize_topology(config, "power")
        self.assertTrue(optimized.enable_power_management)

    def test_compare_topologies(self):
        """测试拓扑比较。"""
        config1 = MockConfig(TopologyType.MESH)
        config1.num_nodes = 16

        config2 = MockConfig(TopologyType.MESH)
        config2.num_nodes = 25

        comparison = NoCTopologyFactory.compare_topologies([config1, config2])

        self.assertEqual(len(comparison["configs"]), 2)
        self.assertEqual(comparison["metrics"]["node_count"], [16, 25])

    def test_get_factory_stats(self):
        """测试工厂统计。"""
        stats = NoCTopologyFactory.get_factory_stats()

        self.assertIn("registered_topologies", stats)
        self.assertIn("supported_types", stats)
        self.assertEqual(stats["registered_topologies"], 1)  # 只注册了MESH


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数。"""

    @patch("src.noc.utils.factory.NoCTopologyFactory.create_topology_from_type")
    def test_create_mesh_topology(self, mock_create):
        """测试创建Mesh拓扑便捷函数。"""
        mock_topology = Mock()
        mock_create.return_value = mock_topology

        result = create_mesh_topology(4, 4, buffer_depth=16)

        mock_create.assert_called_once_with(TopologyType.MESH, num_nodes=16, buffer_depth=16)
        self.assertEqual(result, mock_topology)

    @patch("src.noc.utils.factory.NoCTopologyFactory.create_topology_from_type")
    def test_create_ring_topology(self, mock_create):
        """测试创建Ring拓扑便捷函数。"""
        mock_topology = Mock()
        mock_create.return_value = mock_topology

        result = create_ring_topology(8, routing_strategy="adaptive")

        mock_create.assert_called_once_with(TopologyType.RING, num_nodes=8, routing_strategy="adaptive")
        self.assertEqual(result, mock_topology)

    @patch("src.noc.utils.factory.NoCTopologyFactory.create_topology_from_type")
    def test_create_crossring_topology(self, mock_create):
        """测试创建CrossRing拓扑便捷函数。"""
        mock_topology = Mock()
        mock_create.return_value = mock_topology

        result = create_crossring_topology(40, 4, 32)

        mock_create.assert_called_once_with(TopologyType.CROSSRING, NUM_NODE=40, NUM_COL=4, NUM_IP=32)
        self.assertEqual(result, mock_topology)

    def test_auto_select_topology(self):
        """测试自动选择拓扑。"""
        # 延迟敏感的小型网络
        requirements = {"num_nodes": 16, "latency_critical": True}
        selected = auto_select_topology(requirements)
        self.assertEqual(selected, TopologyType.MESH)

        # 吞吐量敏感
        requirements = {"throughput_critical": True}
        selected = auto_select_topology(requirements)
        self.assertEqual(selected, TopologyType.CROSSRING)

        # 容错需求
        requirements = {"fault_tolerance": True}
        selected = auto_select_topology(requirements)
        self.assertEqual(selected, TopologyType.TORUS)

        # 小型Ring网络
        requirements = {"num_nodes": 6}
        selected = auto_select_topology(requirements)
        self.assertEqual(selected, TopologyType.RING)

        # 默认Mesh
        requirements = {"num_nodes": 32}
        selected = auto_select_topology(requirements)
        self.assertEqual(selected, TopologyType.MESH)


if __name__ == "__main__":
    unittest.main()
