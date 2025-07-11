"""
测试NoC配置类。
"""

import unittest
import tempfile
import os
import json

from src.noc.base.config import BaseNoCConfig, create_config_from_type
from src.noc.utils.types import TopologyType, RoutingStrategy


class MockNoCConfig(BaseNoCConfig):
    """用于测试的模拟配置类。"""

    def validate_config(self):
        """实现抽象方法。"""
        if self.num_nodes <= 0:
            return False, "节点数必须为正数"
        return True, None

    def get_topology_params(self):
        """实现抽象方法。"""
        return {"test_param": "test_value", "num_nodes": self.num_nodes}


class TestBaseNoCConfig(unittest.TestCase):
    """测试基础NoC配置类。"""

    def setUp(self):
        """设置测试环境。"""
        self.config = MockNoCConfig()

    def test_default_initialization(self):
        """测试默认初始化。"""
        self.assertEqual(self.config.topology_type, TopologyType.MESH)
        self.assertEqual(self.config.num_nodes, 16)
        self.assertEqual(self.config.routing_strategy, RoutingStrategy.SHORTEST)
        self.assertEqual(self.config.flit_size, 128)
        self.assertEqual(self.config.buffer_depth, 8)

    def test_set_parameter(self):
        """测试设置参数。"""
        # 设置现有属性
        result = self.config.set_parameter("num_nodes", 32)
        self.assertTrue(result)
        self.assertEqual(self.config.num_nodes, 32)

        # 设置自定义参数
        result = self.config.set_parameter("custom_param", "custom_value")
        self.assertTrue(result)
        self.assertEqual(self.config.get_parameter("custom_param"), "custom_value")

    def test_get_parameter(self):
        """测试获取参数。"""
        # 获取现有属性
        value = self.config.get_parameter("num_nodes")
        self.assertEqual(value, 16)

        # 获取不存在的参数，使用默认值
        value = self.config.get_parameter("nonexistent", "default")
        self.assertEqual(value, "default")

        # 获取自定义参数
        self.config.set_parameter("custom_param", 42)
        value = self.config.get_parameter("custom_param")
        self.assertEqual(value, 42)

    def test_to_dict(self):
        """测试转换为字典。"""
        config_dict = self.config.to_dict()

        # 检查基本属性
        self.assertIn("topology_type", config_dict)
        self.assertIn("num_nodes", config_dict)
        self.assertIn("routing_strategy", config_dict)

        # 检查枚举值转换
        self.assertEqual(config_dict["topology_type"], "mesh")
        self.assertEqual(config_dict["routing_strategy"], "shortest")

        # 检查拓扑特定参数
        self.assertIn("test_param", config_dict)
        self.assertEqual(config_dict["test_param"], "test_value")

    def test_from_dict(self):
        """测试从字典加载。"""
        test_dict = {"num_nodes": 64, "flit_size": 128, "custom_param": "test_value"}

        self.config.from_dict(test_dict)

        self.assertEqual(self.config.num_nodes, 64)
        self.assertEqual(self.config.flit_size, 128)
        self.assertEqual(self.config.get_parameter("custom_param"), "test_value")

    def test_copy(self):
        """测试配置复制。"""
        self.config.set_parameter("test_param", "test_value")
        self.config.num_nodes = 32

        copied_config = self.config.copy()

        self.assertEqual(copied_config.num_nodes, 32)
        self.assertEqual(copied_config.get_parameter("test_param"), "test_value")

        # 验证是深拷贝
        copied_config.num_nodes = 64
        self.assertEqual(self.config.num_nodes, 32)

    def test_update(self):
        """测试配置更新。"""
        other_config = MockNoCConfig()
        other_config.num_nodes = 32
        other_config.set_parameter("new_param", "new_value")

        self.config.update(other_config)

        self.assertEqual(self.config.num_nodes, 32)
        self.assertEqual(self.config.get_parameter("new_param"), "new_value")

    def test_save_and_load_file(self):
        """测试保存和加载文件。"""
        self.config.num_nodes = 32
        self.config.set_parameter("test_param", "test_value")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # 保存到文件
            self.config.save_to_file(temp_path)

            # 验证文件存在且内容正确
            self.assertTrue(os.path.exists(temp_path))

            with open(temp_path, "r") as f:
                saved_data = json.load(f)

            self.assertEqual(saved_data["num_nodes"], 32)
            self.assertEqual(saved_data["test_param"], "test_value")

            # 从文件加载
            loaded_config = MockNoCConfig.load_from_file(temp_path)
            self.assertEqual(loaded_config.num_nodes, 32)
            self.assertEqual(loaded_config.get_parameter("test_param"), "test_value")

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_validate_basic_params(self):
        """测试基础参数验证。"""
        # 有效配置
        valid, message = self.config.validate_basic_params()
        self.assertTrue(valid)
        self.assertIsNone(message)

        # 无效节点数
        self.config.num_nodes = -1
        valid, message = self.config.validate_basic_params()
        self.assertFalse(valid)
        self.assertIn("节点数必须为正数", message)

        # 无效flit大小
        self.config.num_nodes = 16  # 恢复有效值
        self.config.flit_size = -1
        valid, message = self.config.validate_basic_params()
        self.assertFalse(valid)
        self.assertIn("Flit大小必须为正数", message)

    def test_get_performance_bounds(self):
        """测试性能界限计算。"""
        bounds = self.config.get_performance_bounds()

        self.assertIn("max_injection_rate", bounds)
        self.assertIn("min_latency", bounds)
        self.assertIn("total_buffer_capacity", bounds)

        # 验证计算结果合理
        self.assertGreater(bounds["max_injection_rate"], 0)
        self.assertGreater(bounds["min_latency"], 0)
        self.assertGreater(bounds["total_buffer_capacity"], 0)


class TestConfigFactory(unittest.TestCase):
    """测试配置工厂函数。"""

    def test_create_config_from_type(self):
        """测试从类型创建配置。"""
        # 创建Mesh配置
        # mesh_config = create_config_from_type(TopologyType.MESH, num_nodes=25)
        # self.assertEqual(mesh_config.topology_type, TopologyType.MESH)
        # self.assertEqual(mesh_config.num_nodes, 25)

        # 创建CrossRing配置
        crossring_config = create_config_from_type(TopologyType.CROSSRING)
        self.assertEqual(crossring_config.topology_type, TopologyType.CROSSRING)


if __name__ == "__main__":
    unittest.main()
