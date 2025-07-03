#!/usr/bin/env python3
"""
CrossRing配置系统使用示例。

本示例展示如何使用新的CrossRing配置系统来：
1. 创建和配置CrossRing拓扑
2. 使用预设配置模板
3. 自定义配置参数
4. 验证配置有效性
5. 保存和加载配置
6. 与原CrossRing配置格式的兼容性
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# 添加src路径以便导入模块
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.noc.crossring.config import CrossRingConfig
from src.noc.crossring.config_factory import CrossRingConfigFactory


def basic_usage_example():
    """基本使用示例。"""
    print("=" * 60)
    print("基本使用示例")
    print("=" * 60)

    # 1. 创建基本配置
    print("1. 创建基本CrossRing配置...")
    config = CrossRingConfig(num_col=2, num_row=4, config_name="basic_example")
    print(f"   创建的配置: {config}")

    # 2. 检查配置参数
    print("\n2. 基本参数:")
    print(f"   拓扑大小: {config.num_col} × {config.num_row} = {config.num_nodes} 节点")
    print(f"   IP数量: {config.num_ip}")
    print(f"   GDMA数量: {config.ip_config.gdma_count}")
    print(f"   RB输入FIFO深度: {config.fifo_config.rb_in_depth}")

    # 3. 验证配置
    print("\n3. 验证配置...")
    is_valid, error_msg = config.validate_config()
    if is_valid:
        print("   ✓ 配置验证通过")
    else:
        print(f"   ✗ 配置验证失败: {error_msg}")

    # 4. 获取拓扑参数
    print("\n4. 拓扑参数:")
    topo_params = config.get_topology_params()
    for key, value in topo_params.items():
        if isinstance(value, (list, dict)) and len(str(value)) > 50:
            print(f"   {key}: <{type(value).__name__} with {len(value)} items>")
        else:
            print(f"   {key}: {value}")


def preset_configurations_example():
    """预设配置示例。"""
    print("\n" + "=" * 60)
    print("预设配置示例")
    print("=" * 60)

    # 1. 列出可用预设
    print("1. 可用预设配置:")
    presets = CrossRingConfigFactory.list_available_presets()
    for name, description in presets.items():
        print(f"   {name}: {description}")

    # 2. 创建不同规模的配置
    print("\n2. 创建不同规模的配置:")
    configs = {
        "2260E": CrossRingConfigFactory.create_2260E(),
        "2262": CrossRingConfigFactory.create_2262(),
    }

    for name, config in configs.items():
        print(f"\n   {name.upper()} 配置:")
        print(f"     拓扑: {config.num_col} × {config.num_row}")
        print(f"     节点数: {config.num_nodes}")
        print(f"     IP数量: {config.num_ip}")
        print(f"     GDMA数量: {config.ip_config.gdma_count}")
        print(f"     RB FIFO深度: {config.fifo_config.rb_in_depth}")

        # 验证配置
        is_valid, _ = config.validate_config()
        print(f"     验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")


def custom_configuration_example():
    """自定义配置示例。"""
    print("\n" + "=" * 60)
    print("自定义配置示例")
    print("=" * 60)

    # 1. 创建自定义配置
    print("1. 创建自定义配置...")
    config = CrossRingConfigFactory.create_custom(num_col=3, num_row=3, config_name="custom_3x3", burst=8)
    print(f"   基础配置: {config}")

    # 2. 修改基础配置
    print("\n2. 修改基础配置...")
    config.update_basic_config(network_frequency=2)
    print(f"   网络频率: {config.basic_config.network_frequency} GHz")

    # 3. 修改IP配置
    print("\n2. 修改IP配置...")
    config.update_ip_config(gdma_count=6, sdma_count=6, ddr_count=6, gdma_bw_limit=12.0, ddr_bw_limit=120.0)
    print(f"   GDMA数量: {config.ip_config.gdma_count}")
    print(f"   DDR带宽限制: {config.ip_config.ddr_bw_limit} GB/s")

    # 4. 修改FIFO配置
    print("\n3. 修改FIFO配置...")
    config.update_fifo_config(rb_in_depth=16, eq_in_depth=16)
    print(f"   RB输入FIFO深度: {config.fifo_config.rb_in_depth}")
    print(f"   EQ输入FIFO深度: {config.fifo_config.eq_in_depth}")

    # 5. 修改Tag配置
    print("\n4. 修改Tag配置...")
    config.update_tag_config(itag_trigger_th_h=70, itag_trigger_th_v=70)
    print(f"   ITag触发阈值: {config.tag_config.itag_trigger_th_h}")

    # 5. 验证自定义配置
    print("\n5. 验证自定义配置...")
    is_valid, error_msg = config.validate_config()
    if is_valid:
        print("   ✓ 自定义配置验证通过")
    else:
        print(f"   ✗ 自定义配置验证失败: {error_msg}")


def file_operations_example():
    """文件操作示例。"""
    print("\n" + "=" * 60)
    print("文件操作示例")
    print("=" * 60)

    # 1. 保存配置到文件
    print("1. 保存配置到文件...")
    config = CrossRingConfigFactory.create_default()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        config.save_to_file(temp_path)
        print(f"   配置已保存到: {temp_path}")

        # 检查文件大小
        file_size = os.path.getsize(temp_path)
        print(f"   文件大小: {file_size} 字节")

        # 2. 从文件加载配置
        print("\n2. 从文件加载配置...")
        loaded_config = CrossRingConfigFactory.from_json_file(temp_path)
        print(f"   加载的配置: {loaded_config}")
        print(f"   配置名: {loaded_config.config_name}")
        print(f"   拓扑: {loaded_config.num_col} × {loaded_config.num_row}")

        # 3. 验证加载的配置
        print("\n3. 验证加载的配置...")
        is_valid, error_msg = loaded_config.validate_config()
        if is_valid:
            print("   ✓ 加载的配置验证通过")
        else:
            print(f"   ✗ 加载的配置验证失败: {error_msg}")

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def validation_examples():
    """配置验证示例。"""
    print("\n" + "=" * 60)
    print("配置验证示例")
    print("=" * 60)

    # 1. 有效配置
    print("1. 有效配置验证:")
    valid_config = CrossRingConfigFactory.create_default()
    is_valid, error_msg = valid_config.validate_config()
    print(f"   结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    if error_msg:
        print(f"   错误: {error_msg}")

    # 2. 无效配置示例
    print("\n2. 无效配置示例:")

    # 2.1 节点数不匹配
    print("\n   2.1 节点数不匹配:")
    invalid_config1 = CrossRingConfig(num_col=2, num_row=4)
    invalid_config1.num_nodes = 10  # 应该是8
    is_valid, error_msg = invalid_config1.validate_config()
    print(f"   结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    print(f"   错误: {error_msg}")

    # 2.2 ETag约束违反
    print("\n   2.2 ETag约束违反:")
    invalid_config2 = CrossRingConfig(num_col=2, num_row=4)
    invalid_config2.tag_config.tl_etag_t1_ue_max = 2
    invalid_config2.tag_config.tl_etag_t2_ue_max = 5  # T2 > T1
    is_valid, error_msg = invalid_config2.validate_config()
    print(f"   结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    print(f"   错误: {error_msg}")

    # 2.3 FIFO深度无效
    print("\n   2.3 FIFO深度无效:")
    invalid_config3 = CrossRingConfig(num_col=2, num_row=4)
    invalid_config3.fifo_config.rb_in_depth = 0
    is_valid, error_msg = invalid_config3.validate_config()
    print(f"   结果: {'✓ 通过' if is_valid else '✗ 失败'}")
    print(f"   错误: {error_msg}")


def main():
    """主函数，运行所有示例。"""
    print("CrossRing配置系统使用示例")
    print("=" * 60)

    try:
        # 运行各种示例
        basic_usage_example()
        preset_configurations_example()
        custom_configuration_example()
        file_operations_example()
        validation_examples()

        print("\n" + "=" * 60)
        print("所有示例运行完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n运行示例时发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
