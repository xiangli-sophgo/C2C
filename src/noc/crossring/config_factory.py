"""
CrossRing配置工厂类。

本模块提供工厂方法来创建各种预定义的CrossRing配置，
包括小规模、中规模、大规模的配置模板。
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

from .config import CrossRingConfig, IPConfiguration, FIFOConfiguration, TagConfiguration


class CrossRingConfigFactory:
    """
    CrossRing配置工厂类。

    提供便捷的方法来创建各种预定义的CrossRing配置，
    支持从JSON文件加载和创建标准配置模板。
    """

    @staticmethod
    def create_default() -> CrossRingConfig:
        """
        创建默认CrossRing配置。

        Returns:
            默认的CrossRing配置实例
        """
        return CrossRingConfig(num_col=2, num_row=4, config_name="default")

    @staticmethod
    def create_2260E() -> CrossRingConfig:
        """
        创建2260E专用CrossRing配置。
        """
        config = CrossRingConfig()
        config.set_preset_configuration("2260E")
        config.config_name = "2260E"
        return config

    @staticmethod
    def create_2262() -> CrossRingConfig:
        """
        创建2262专用CrossRing配置。
        """
        config = CrossRingConfig()
        config.set_preset_configuration("2262")
        config.config_name = "2262"
        return config

    @staticmethod
    def create_custom(num_col: int, num_row: int, config_name: str = "custom", **kwargs) -> CrossRingConfig:
        """
        创建自定义CrossRing配置。

        Args:
            num_col: 列数
            num_row: 行数
            config_name: 配置名称
            **kwargs: 其他配置参数

        Returns:
            自定义CrossRing配置实例
        """
        config = CrossRingConfig(num_col=num_col, num_row=num_row, config_name=config_name)

        # 应用自定义参数
        for key, value in kwargs.items():
            config.set_parameter(key, value)

        return config

    @staticmethod
    def from_json_file(filepath: str) -> CrossRingConfig:
        """
        从JSON文件加载CrossRing配置。

        Args:
            filepath: JSON配置文件路径

        Returns:
            从文件加载的CrossRing配置实例

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON格式错误
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"配置文件不存在: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # 提取基本拓扑参数
        num_col = config_data.get("num_col", 2)
        num_row = config_data.get("num_row", 4)
        config_name = config_data.get("config_name", "loaded")

        # 创建基础配置
        config = CrossRingConfig(num_col=num_col, num_row=num_row, config_name=config_name)

        # 加载其他参数
        config.from_dict(config_data)

        return config

    @staticmethod
    def list_available_presets() -> Dict[str, str]:
        """
        列出可用的预设配置。
        """
        return {
            "default": "默认配置",
            "2260E": "2260E专用配置",
            "2262": "2262专用配置",
        }

    @staticmethod
    def get_preset_config(preset_name: str) -> CrossRingConfig:
        """
        根据预设名称获取配置。
        """
        preset_methods = {
            "default": CrossRingConfigFactory.create_default,
            "2260E": CrossRingConfigFactory.create_2260E,
            "2262": CrossRingConfigFactory.create_2262,
        }
        if preset_name not in preset_methods:
            available = ", ".join(preset_methods.keys())
            raise ValueError(f"不支持的预设配置 '{preset_name}'。可用配置: {available}")
        return preset_methods[preset_name]()

    @staticmethod
    def save_preset_configs(output_dir: str) -> None:
        """
        保存所有预设配置到指定目录。

        Args:
            output_dir: 输出目录路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        presets = CrossRingConfigFactory.list_available_presets()

        for preset_name in presets:
            config = CrossRingConfigFactory.get_preset_config(preset_name)
            filepath = output_path / f"crossring_{preset_name}.json"
            config.save_to_file(str(filepath))
            print(f"保存预设配置 '{preset_name}' 到 {filepath}")

    @staticmethod
    def validate_all_presets() -> Dict[str, bool]:
        """
        验证所有预设配置。

        Returns:
            各预设配置的验证结果
        """
        results = {}
        presets = CrossRingConfigFactory.list_available_presets()

        for preset_name in presets:
            try:
                config = CrossRingConfigFactory.get_preset_config(preset_name)
                is_valid, error_msg = config.validate_config()
                results[preset_name] = is_valid
                if not is_valid:
                    print(f"预设配置 '{preset_name}' 验证失败: {error_msg}")
            except Exception as e:
                results[preset_name] = False
                print(f"预设配置 '{preset_name}' 创建失败: {e}")

        return results
