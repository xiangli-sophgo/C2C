"""
CrossRing配置系统单元测试。

测试CrossRingConfig和CrossRingConfigFactory的功能。
"""

import unittest
import tempfile
import json
import os
from typing import Dict, Any

# 导入待测试的模块
# import sys
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.noc.crossring.config import CrossRingConfig, IPConfiguration, FIFOConfiguration
from src.noc.crossring.config_factory import CrossRingConfigFactory
from src.noc.utils.types import TopologyType


class TestCrossRingConfig(unittest.TestCase):
    """CrossRingConfig类测试"""

    def setUp(self):
        """测试前设置"""
        self.config = CrossRingConfig(num_col=2, num_row=4, config_name="test")

    def test_initialization(self):
        """测试配置初始化"""
        self.assertEqual(self.config.num_col, 2)
        self.assertEqual(self.config.num_row, 4)
        self.assertEqual(self.config.num_nodes, 8)
        self.assertEqual(self.config.config_name, "test")
        self.assertEqual(self.config.topology_type, TopologyType.CROSSRING)
        self.assertIsInstance(self.config.ip_config, IPConfiguration)
        self.assertIsInstance(self.config.fifo_config, FIFOConfiguration)

    def test_topology_validation(self):
        """测试拓扑参数验证"""
        # 正常配置应该通过验证
        is_valid, error_msg = self.config.validate_config()
        self.assertTrue(is_valid, f"配置验证失败: {error_msg}")

        # 测试节点数不匹配的情况
        self.config.num_nodes = 10  # 不等于 num_col * num_row
        is_valid, error_msg = self.config.validate_config()
        self.assertFalse(is_valid)
        self.assertIn("节点数必须等于行数×列数", error_msg)

    def test_fifo_depth_validation(self):
        """测试FIFO深度验证"""
        # 无效的FIFO深度
        self.config.fifo_config.rb_in_depth = 0
        is_valid, error_msg = self.config.validate_config()
        self.assertFalse(is_valid)
        self.assertIn("RB输入FIFO深度必须为正数", error_msg)

    def test_etag_constraints(self):
        """测试ETag约束验证"""
        # 违反ETag约束
        self.config.tag_config.tl_etag_t1_ue_max = 2
        self.config.tag_config.tl_etag_t2_ue_max = 3  # T2 > T1
        is_valid, error_msg = self.config.validate_config()
        self.assertFalse(is_valid)
        self.assertIn("TL ETag T1必须大于T2", error_msg)

    def test_buffer_size_consistency(self):
        """测试缓冲区大小一致性"""
        # 修改 Tracker 配置但不更新缓冲区大小
        self.config.tracker_config.rn_r_tracker_ostd = 32
        # rn_rdb_size 仍然是旧值，应该验证失败
        is_valid, error_msg = self.config.validate_config()
        self.assertFalse(is_valid)
        self.assertIn("RN_RDB_SIZE必须等于", error_msg)

    def test_ip_position_validation(self):
        """测试IP位置验证"""
        # 添加超出范围的IP位置
        self.config.ddr_send_position_list.append(self.config.num_nodes + 1)
        is_valid, error_msg = self.config.validate_config()
        self.assertFalse(is_valid)
        self.assertIn("IP位置", error_msg)
        self.assertIn("超出节点范围", error_msg)

    def test_topology_size_update(self):
        """测试拓扑大小更新"""
        self.config.update_topology_size(4, 4)
        self.assertEqual(self.config.num_col, 4)
        self.assertEqual(self.config.num_row, 4)
        self.assertEqual(self.config.num_nodes, 16)

        # IP位置应该重新生成
        for pos in self.config.ddr_send_position_list:
            self.assertLess(pos, self.config.num_nodes)

    def test_config_updates(self):
        """测试配置更新方法"""
        # 更新基础配置
        self.config.update_basic_config(slice_per_link=16)
        self.assertEqual(self.config.basic_config.slice_per_link, 16)

        # 更新IP配置
        self.config.update_ip_config(gdma_count=8, ddr_bw_limit=100.0)
        self.assertEqual(self.config.ip_config.gdma_count, 8)
        self.assertEqual(self.config.ip_config.ddr_bw_limit, 100.0)

        # 更新FIFO配置
        self.config.update_fifo_config(rb_in_depth=16)
        self.assertEqual(self.config.fifo_config.rb_in_depth, 16)

        # 更新tag配置
        self.config.update_basic_config(etag_bothside_upgrade=0)
        self.assertEqual(self.config.tag_config.etag_bothside_upgrade, 0)

    def test_preset_configurations(self):
        """测试预设配置"""
        presets = ["2260E", "2262"]

        for preset in presets:
            self.config.set_preset_configuration(preset)
            is_valid, error_msg = self.config.validate_config()
            self.assertTrue(is_valid, f"预设配置 '{preset}' 验证失败: {error_msg}")

    def test_serialization(self):
        """测试序列化和反序列化"""
        # 转换为字典
        config_dict = self.config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn("config_name", config_dict)
        self.assertIn("num_col", config_dict)

        # 从字典创建新配置
        new_config = CrossRingConfig()
        new_config.from_dict(config_dict)
        self.assertEqual(new_config.config_name, self.config.config_name)
        self.assertEqual(new_config.num_col, self.config.num_col)

    def test_json_serialization(self):
        """测试JSON序列化"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # 保存到文件
            self.config.save_to_file(temp_path)
            self.assertTrue(os.path.exists(temp_path))

            # 从文件加载
            loaded_config = CrossRingConfig.load_from_file(temp_path)
            self.assertEqual(loaded_config.config_name, self.config.config_name)
            self.assertEqual(loaded_config.num_col, self.config.num_col)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestCrossRingConfigFactory(unittest.TestCase):
    """CrossRingConfigFactory类测试"""

    def test_create_default(self):
        """测试创建默认配置"""
        config = CrossRingConfigFactory.create_default()
        self.assertIsInstance(config, CrossRingConfig)
        self.assertEqual(config.config_name, "default")

        # 验证配置有效性
        is_valid, error_msg = config.validate_config()
        self.assertTrue(is_valid, f"默认配置验证失败: {error_msg}")

    def test_create_presets(self):
        """测试创建预设配置"""
        presets = {
            "2260E": CrossRingConfigFactory.create_2260E,
            "2262": CrossRingConfigFactory.create_2262,
        }

        for preset_name, create_func in presets.items():
            with self.subTest(preset=preset_name):
                config = create_func()
                self.assertIsInstance(config, CrossRingConfig)

                # 验证配置有效性
                is_valid, error_msg = config.validate_config()
                self.assertTrue(is_valid, f"预设配置 '{preset_name}' 验证失败: {error_msg}")

    def test_create_custom(self):
        """测试创建自定义配置"""
        config = CrossRingConfigFactory.create_custom(num_col=3, num_row=3, config_name="custom_test", burst=8)

        self.assertEqual(config.num_col, 3)
        self.assertEqual(config.num_row, 3)
        self.assertEqual(config.config_name, "custom_test")

    def test_json_file_operations(self):
        """测试JSON文件操作"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            config_data = {"num_col": 3, "num_row": 3, "config_name": "file_test", "ip_config": {"gdma_count": 6, "ddr_count": 6}}
            json.dump(config_data, f)

        try:
            # 从文件加载
            config = CrossRingConfigFactory.from_json_file(temp_path)
            self.assertEqual(config.num_col, 3)
            self.assertEqual(config.num_row, 3)
            self.assertEqual(config.config_name, "file_test")

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_file_not_found(self):
        """测试文件不存在的情况"""
        with self.assertRaises(FileNotFoundError):
            CrossRingConfigFactory.from_json_file("nonexistent_file.json")

    def test_list_available_presets(self):
        """测试列出可用预设"""
        presets = CrossRingConfigFactory.list_available_presets()
        self.assertIsInstance(presets, dict)
        self.assertIn("default", presets)
        self.assertIn("2260E", presets)
        self.assertIn("2262", presets)

    def test_get_preset_config(self):
        """测试获取预设配置"""

        config_2260e = CrossRingConfigFactory.get_preset_config("2260E")
        self.assertIsInstance(config_2260e, CrossRingConfig)
        self.assertEqual(config_2260e.config_name, "2260E")

        config_2262 = CrossRingConfigFactory.get_preset_config("2262")
        self.assertIsInstance(config_2262, CrossRingConfig)
        self.assertEqual(config_2262.config_name, "2262")

        # 测试无效预设名称
        with self.assertRaises(ValueError):
            CrossRingConfigFactory.get_preset_config("invalid_preset")

    def test_validate_all_presets(self):
        """测试验证所有预设配置"""
        results = CrossRingConfigFactory.validate_all_presets()
        self.assertIsInstance(results, dict)

        # 所有预设配置都应该有效
        for preset_name, is_valid in results.items():
            self.assertTrue(is_valid, f"预设配置 '{preset_name}' 验证失败")
        self.assertIn("2260E", results)
        self.assertIn("2262", results)

    def test_save_preset_configs(self):
        """测试保存预设配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            CrossRingConfigFactory.save_preset_configs(temp_dir)

            # 检查是否生成了文件
            files = os.listdir(temp_dir)
            json_files = [f for f in files if f.endswith(".json")]
            self.assertGreater(len(json_files), 0)

            # 检查文件内容
            for json_file in json_files:
                filepath = os.path.join(temp_dir, json_file)
                with open(filepath, "r") as f:
                    config_data = json.load(f)
                self.assertIsInstance(config_data, dict)
                self.assertIn("num_col", config_data)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_factory_and_validation_integration(self):
        """测试工厂创建和验证的集成"""
        # 创建各种配置并验证
        factory_methods = [
            CrossRingConfigFactory.create_default,
            CrossRingConfigFactory.create_2260E,
            CrossRingConfigFactory.create_2262,
        ]

        for method in factory_methods:
            with self.subTest(method=method.__name__):
                config = method()
                is_valid, error_msg = config.validate_config()
                self.assertTrue(is_valid, f"{method.__name__} 创建的配置验证失败: {error_msg}")

    def test_serialization_roundtrip(self):
        """测试序列化往返"""
        original_config = CrossRingConfigFactory.create_default()

        # 转换为字典再恢复
        config_dict = original_config.to_dict()
        restored_config = CrossRingConfig()
        restored_config.from_dict(config_dict)

        # 验证关键参数一致
        self.assertEqual(original_config.num_col, restored_config.num_col)
        self.assertEqual(original_config.num_row, restored_config.num_row)
        # 处理ip_config可能是字典的情况
        if hasattr(restored_config.ip_config, "gdma_count"):
            self.assertEqual(original_config.ip_config.gdma_count, restored_config.ip_config.gdma_count)
        else:
            # ip_config被序列化为字典，从字典中获取值
            self.assertEqual(original_config.ip_config.gdma_count, restored_config.ip_config["gdma_count"])

        # 两个配置都应该有效
        orig_valid, _ = original_config.validate_config()
        rest_valid, _ = restored_config.validate_config()
        self.assertTrue(orig_valid)
        self.assertTrue(rest_valid)


if __name__ == "__main__":
    # 设置测试环境
    unittest.main(verbosity=2)
