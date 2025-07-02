#!/usr/bin/env python3
"""
NoC抽象层使用示例。

本文件展示了如何使用NoC抽象层创建和配置不同类型的拓扑结构，
以及如何进行基本的性能分析和路由计算。
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.noc import (
    NoCTopologyFactory, TopologyType, RoutingStrategy,
    create_crossring_topology, create_mesh_topology, create_ring_topology,
    CrossRingCompatibleConfig, BaseNoCConfig
)


def example_1_basic_topology_creation():
    """示例1：基本拓扑创建。"""
    print("=" * 60)
    print("示例1：基本拓扑创建")
    print("=" * 60)
    
    try:
        # 创建一个简单的8节点Ring拓扑
        print("创建8节点Ring拓扑...")
        ring_topology = create_ring_topology(num_nodes=8)
        print(f"创建成功: {ring_topology}")
        
        # 获取拓扑基本信息
        info = ring_topology.get_topology_info()
        print("拓扑信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def example_2_crossring_configuration():
    """示例2：CrossRing配置。"""
    print("=" * 60)
    print("示例2：CrossRing配置")
    print("=" * 60)
    
    try:
        # 创建CrossRing兼容配置
        print("创建CrossRing配置...")
        config = CrossRingCompatibleConfig()
        
        # 显示默认配置
        print("默认配置参数:")
        print(f"  节点数: {config.NUM_NODE}")
        print(f"  列数: {config.NUM_COL}")
        print(f"  IP数量: {config.NUM_IP}")
        print(f"  缓冲区深度: {config.buffer_depth}")
        
        # 验证配置
        is_valid, error = config.validate_config()
        print(f"配置验证: {'通过' if is_valid else '失败'}")
        if error:
            print(f"错误信息: {error}")
        
        # 更新拓扑配置
        print("\n更新为5x4拓扑...")
        config.update_topology("5x4")
        print(f"  新节点数: {config.NUM_NODE}")
        print(f"  新列数: {config.NUM_COL}")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def example_3_factory_usage():
    """示例3：工厂模式使用。"""
    print("=" * 60)
    print("示例3：工厂模式使用")
    print("=" * 60)
    
    try:
        # 列出支持的拓扑类型
        supported = NoCTopologyFactory.get_supported_topologies()
        print("支持的拓扑类型:")
        for topo_type in supported:
            print(f"  - {topo_type.value}")
        
        # 使用工厂创建不同类型的配置
        print("\n创建不同类型的配置:")
        
        # Mesh配置
        mesh_config = NoCTopologyFactory.create_config(
            TopologyType.MESH,
            num_nodes=16,
            routing_strategy=RoutingStrategy.SHORTEST
        )
        print(f"Mesh配置: {mesh_config.num_nodes}节点, 路由策略: {mesh_config.routing_strategy.value}")
        
        # Ring配置
        ring_config = NoCTopologyFactory.create_config(
            TopologyType.RING,
            num_nodes=8,
            routing_strategy=RoutingStrategy.LOAD_BALANCED
        )
        print(f"Ring配置: {ring_config.num_nodes}节点, 路由策略: {ring_config.routing_strategy.value}")
        
        # CrossRing配置
        crossring_config = NoCTopologyFactory.create_config(
            TopologyType.CROSSRING,
            NUM_NODE=20,
            NUM_COL=2
        )
        print(f"CrossRing配置: {crossring_config.NUM_NODE}节点, {crossring_config.NUM_COL}列")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def example_4_configuration_comparison():
    """示例4：配置比较。"""
    print("=" * 60)
    print("示例4：配置比较")
    print("=" * 60)
    
    try:
        # 创建多个配置用于比较
        configs = []
        
        # Mesh配置
        mesh_config = BaseNoCConfig(TopologyType.MESH)
        mesh_config.num_nodes = 16
        mesh_config.routing_strategy = RoutingStrategy.SHORTEST
        # 添加抽象方法的实现
        mesh_config.validate_config = lambda: (True, None)
        mesh_config.get_topology_params = lambda: {"dimensions": (4, 4)}
        configs.append(mesh_config)
        
        # Ring配置
        ring_config = BaseNoCConfig(TopologyType.RING)
        ring_config.num_nodes = 8
        ring_config.routing_strategy = RoutingStrategy.LOAD_BALANCED
        # 添加抽象方法的实现
        ring_config.validate_config = lambda: (True, None)
        ring_config.get_topology_params = lambda: {"diameter": 4}
        configs.append(ring_config)
        
        # 比较配置
        comparison = NoCTopologyFactory.compare_topologies(configs)
        
        print("配置比较结果:")
        print("配置信息:")
        for i, config_info in enumerate(comparison["configs"]):
            print(f"  配置{i+1}: {config_info['topology_type']}, {config_info['num_nodes']}节点")
        
        print("性能指标:")
        print(f"  节点数: {comparison['metrics']['node_count']}")
        print(f"  估算直径: {comparison['metrics']['estimated_diameter']}")
        print(f"  估算延迟: {comparison['metrics']['estimated_latency']}")
        print(f"  估算吞吐量: {comparison['metrics']['estimated_throughput']}")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def example_5_configuration_optimization():
    """示例5：配置优化。"""
    print("=" * 60)
    print("示例5：配置优化")
    print("=" * 60)
    
    try:
        # 创建基础配置
        base_config = BaseNoCConfig(TopologyType.MESH)
        base_config.num_nodes = 16
        base_config.buffer_depth = 8
        base_config.virtual_channels = 2
        # 添加抽象方法的实现
        base_config.validate_config = lambda: (True, None)
        base_config.get_topology_params = lambda: {}
        
        print("原始配置:")
        print(f"  路由策略: {base_config.routing_strategy.value}")
        print(f"  缓冲区深度: {base_config.buffer_depth}")
        print(f"  虚拟通道数: {base_config.virtual_channels}")
        
        # 为不同目标优化
        optimization_targets = ["latency", "throughput", "power"]
        
        for target in optimization_targets:
            optimized = NoCTopologyFactory.optimize_topology(base_config, target)
            print(f"\n为{target}优化后的配置:")
            print(f"  路由策略: {optimized.routing_strategy}")
            print(f"  缓冲区深度: {optimized.buffer_depth}")
            if hasattr(optimized, 'virtual_channels'):
                print(f"  虚拟通道数: {optimized.virtual_channels}")
            if hasattr(optimized, 'enable_power_management'):
                print(f"  功耗管理: {optimized.enable_power_management}")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def example_6_config_serialization():
    """示例6：配置序列化。"""
    print("=" * 60)
    print("示例6：配置序列化")
    print("=" * 60)
    
    try:
        # 创建配置
        config = CrossRingCompatibleConfig()
        config.NUM_NODE = 40
        config.NUM_COL = 4
        config.set_parameter("自定义参数", "测试值")
        
        # 转换为字典
        config_dict = config.to_dict()
        print("配置转字典（部分内容）:")
        important_keys = ["NUM_NODE", "NUM_COL", "NUM_IP", "topology_type"]
        for key in important_keys:
            if key in config_dict:
                value = config_dict[key]
                if hasattr(value, 'value'):  # 处理枚举
                    value = value.value
                print(f"  {key}: {value}")
        
        # 从字典恢复配置
        new_config = CrossRingCompatibleConfig()
        new_config.from_dict(config_dict)
        
        print("\n从字典恢复的配置:")
        print(f"  NUM_NODE: {new_config.NUM_NODE}")
        print(f"  NUM_COL: {new_config.NUM_COL}")
        print(f"  自定义参数: {new_config.get_parameter('自定义参数')}")
        
        # 保存和加载文件（可选，需要文件系统权限）
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_file = f.name
            
            config.save_to_file(temp_file)
            print(f"\n配置已保存到: {temp_file}")
            
            loaded_config = CrossRingCompatibleConfig.load_from_file(temp_file)
            print(f"从文件加载的配置 NUM_NODE: {loaded_config.NUM_NODE}")
            
            # 清理临时文件
            os.unlink(temp_file)
            
        except Exception as file_error:
            print(f"文件操作跳过: {file_error}")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def example_7_factory_statistics():
    """示例7：工厂统计信息。"""
    print("=" * 60)
    print("示例7：工厂统计信息")
    print("=" * 60)
    
    try:
        # 获取工厂统计信息
        stats = NoCTopologyFactory.get_factory_stats()
        
        print("工厂统计信息:")
        print(f"  注册的拓扑数量: {stats['registered_topologies']}")
        print(f"  验证器数量: {stats['validators_count']}")
        print(f"  配置创建器数量: {stats['config_creators_count']}")
        print(f"  默认配置数量: {stats['default_configs_count']}")
        
        print("支持的拓扑类型:")
        for topo_type in stats['supported_types']:
            print(f"  - {topo_type}")
        
        # 获取详细的拓扑信息
        print("\n详细拓扑信息:")
        all_topologies = NoCTopologyFactory.list_topologies()
        for topo_name, topo_info in all_topologies.items():
            if topo_info.get("supported", False):
                print(f"  {topo_name}:")
                print(f"    类名: {topo_info.get('class_name', 'N/A')}")
                print(f"    有验证器: {topo_info.get('has_validator', False)}")
                print(f"    有配置创建器: {topo_info.get('has_config_creator', False)}")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def main():
    """主函数 - 运行所有示例。"""
    print("NoC抽象层使用示例")
    print("=" * 60)
    
    # 运行所有示例
    examples = [
        example_1_basic_topology_creation,
        example_2_crossring_configuration,
        example_3_factory_usage,
        example_4_configuration_comparison,
        example_5_configuration_optimization,
        example_6_config_serialization,
        example_7_factory_statistics,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"示例{i}执行出错: {e}")
            print()
    
    print("所有示例执行完成!")


if __name__ == "__main__":
    main()