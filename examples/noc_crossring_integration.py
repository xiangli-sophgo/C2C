#!/usr/bin/env python3
"""
NoC抽象层与CrossRing集成示例。

展示如何将NoC抽象层与现有的CrossRing实现集成，
以及如何在保持兼容性的同时使用新的抽象接口。
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.noc import (
    CrossRingCompatibleConfig, NoCTopologyFactory, TopologyType,
    RoutingStrategy, create_crossring_topology
)


def example_crossring_basic_setup():
    """CrossRing基础设置示例。"""
    print("=" * 60)
    print("CrossRing基础设置示例")
    print("=" * 60)
    
    try:
        # 创建CrossRing兼容配置
        config = CrossRingCompatibleConfig()
        
        print("默认CrossRing配置:")
        print(f"  拓扑类型: {config.topology_type.value}")
        print(f"  节点总数: {config.NUM_NODE}")
        print(f"  列数: {config.NUM_COL}")
        print(f"  行数: {config.NUM_ROW}")
        print(f"  IP数量: {config.NUM_IP}")
        print(f"  网络频率: {config.NETWORK_FREQUENCY}")
        
        # 显示缓冲区配置
        print("\n缓冲区配置:")
        print(f"  RB输入FIFO深度: {config.RB_IN_FIFO_DEPTH}")
        print(f"  RB输出FIFO深度: {config.RB_OUT_FIFO_DEPTH}")
        print(f"  IQ输出FIFO深度: {config.IQ_OUT_FIFO_DEPTH}")
        print(f"  EQ输入FIFO深度: {config.EQ_IN_FIFO_DEPTH}")
        
        # 显示跟踪器配置
        print("\n跟踪器配置:")
        print(f"  RN读跟踪器OSTD: {config.RN_R_TRACKER_OSTD}")
        print(f"  RN写跟踪器OSTD: {config.RN_W_TRACKER_OSTD}")
        print(f"  SN DDR读跟踪器OSTD: {config.SN_DDR_R_TRACKER_OSTD}")
        print(f"  SN DDR写跟踪器OSTD: {config.SN_DDR_W_TRACKER_OSTD}")
        
        # 显示延迟配置
        print("\n延迟配置:")
        print(f"  DDR读延迟: {config.DDR_R_LATENCY}")
        print(f"  DDR写延迟: {config.DDR_W_LATENCY}")
        print(f"  L2M读延迟: {config.L2M_R_LATENCY}")
        print(f"  L2M写延迟: {config.L2M_W_LATENCY}")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def example_crossring_channel_configuration():
    """CrossRing通道配置示例。"""
    print("=" * 60)
    print("CrossRing通道配置示例")
    print("=" * 60)
    
    try:
        config = CrossRingCompatibleConfig()
        
        # 显示通道规格
        print("通道规格:")
        for channel_type, count in config.CHANNEL_SPEC.items():
            print(f"  {channel_type}: {count}个通道")
        
        # 显示通道名称列表
        print(f"\n通道名称列表 (共{len(config.CH_NAME_LIST)}个):")
        for i, ch_name in enumerate(config.CH_NAME_LIST):
            print(f"  [{i}] {ch_name}")
        
        # 显示IP位置配置
        print(f"\nIP位置配置:")
        print(f"  GDMA发送位置: {config.GDMA_SEND_POSITION_LIST[:5]}{'...' if len(config.GDMA_SEND_POSITION_LIST) > 5 else ''}")
        print(f"  SDMA发送位置: {config.SDMA_SEND_POSITION_LIST[:5]}{'...' if len(config.SDMA_SEND_POSITION_LIST) > 5 else ''}")
        print(f"  DDR发送位置: {config.DDR_SEND_POSITION_LIST[:5]}{'...' if len(config.DDR_SEND_POSITION_LIST) > 5 else ''}")
        print(f"  L2M发送位置: {config.L2M_SEND_POSITION_LIST[:5]}{'...' if len(config.L2M_SEND_POSITION_LIST) > 5 else ''}")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def example_crossring_topology_variants():
    """CrossRing拓扑变体示例。"""
    print("=" * 60)
    print("CrossRing拓扑变体示例")
    print("=" * 60)
    
    try:
        topologies = ["5x2", "5x4", "default"]
        
        for topo_type in topologies:
            print(f"{topo_type}拓扑配置:")
            
            config = CrossRingCompatibleConfig()
            if topo_type != "default":
                config.update_topology(topo_type)
            
            print(f"  节点总数: {config.NUM_NODE}")
            print(f"  列数: {config.NUM_COL}")
            print(f"  行数: {config.NUM_ROW}")
            print(f"  IP数量: {config.NUM_IP}")
            print(f"  GDMA位置数量: {len(config.GDMA_SEND_POSITION_LIST)}")
            print(f"  DDR位置数量: {len(config.DDR_SEND_POSITION_LIST)}")
            
            # 验证配置
            is_valid, error = config.validate_config()
            print(f"  配置验证: {'✓ 通过' if is_valid else '✗ 失败'}")
            if error:
                print(f"  错误: {error}")
            
            print()
        
    except Exception as e:
        print(f"错误: {e}")


def example_crossring_config_conversion():
    """CrossRing配置转换示例。"""
    print("=" * 60)
    print("CrossRing配置转换示例")
    print("=" * 60)
    
    try:
        # 创建CrossRing配置
        noc_config = CrossRingCompatibleConfig()
        noc_config.NUM_NODE = 40
        noc_config.NUM_COL = 4
        noc_config.NUM_IP = 32
        noc_config.update_topology("5x4")
        
        # 转换为CrossRing格式的配置字典
        crossring_dict = noc_config.get_crossring_config_dict()
        
        print("转换为CrossRing格式的配置:")
        important_params = [
            'NUM_NODE', 'NUM_COL', 'NUM_IP', 'NUM_RN', 'NUM_SN',
            'FLIT_SIZE', 'SLICE_PER_LINK', 'BURST', 'NETWORK_FREQUENCY'
        ]
        
        for param in important_params:
            if param in crossring_dict:
                print(f"  {param}: {crossring_dict[param]}")
        
        # 显示如何获取拓扑特定参数
        print("\n拓扑特定参数:")
        topo_params = noc_config.get_topology_params()
        for key, value in topo_params.items():
            if isinstance(value, list) and len(value) > 5:
                print(f"  {key}: [{', '.join(map(str, value[:3]))}, ...] (共{len(value)}个)")
            else:
                print(f"  {key}: {value}")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def example_crossring_performance_configuration():
    """CrossRing性能配置示例。"""
    print("=" * 60)
    print("CrossRing性能配置示例")
    print("=" * 60)
    
    try:
        config = CrossRingCompatibleConfig()
        
        # 显示带宽限制配置
        print("带宽限制配置:")
        bandwidth_params = ['GDMA_BW_LIMIT', 'SDMA_BW_LIMIT', 'CDMA_BW_LIMIT', 
                           'DDR_BW_LIMIT', 'L2M_BW_LIMIT']
        for param in bandwidth_params:
            value = getattr(config, param)
            print(f"  {param}: {value}")
        
        # 显示ETag配置
        print("\nETag配置:")
        etag_params = ['TL_Etag_T1_UE_MAX', 'TL_Etag_T2_UE_MAX', 'TR_Etag_T2_UE_MAX',
                      'TU_Etag_T1_UE_MAX', 'TU_Etag_T2_UE_MAX', 'TD_Etag_T2_UE_MAX']
        for param in etag_params:
            value = getattr(config, param)
            print(f"  {param}: {value}")
        
        # 显示ITag配置
        print("\nITag配置:")
        itag_params = ['ITag_TRIGGER_Th_H', 'ITag_TRIGGER_Th_V', 
                      'ITag_MAX_NUM_H', 'ITag_MAX_NUM_V']
        for param in itag_params:
            value = getattr(config, param)
            print(f"  {param}: {value}")
        
        # 显示读写间隔配置
        print("\n读写间隔配置:")
        gap_params = ['GDMA_RW_GAP', 'SDMA_RW_GAP']
        for param in gap_params:
            value = getattr(config, param)
            print(f"  {param}: {value}")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def example_crossring_validation_and_optimization():
    """CrossRing验证和优化示例。"""
    print("=" * 60)
    print("CrossRing验证和优化示例")
    print("=" * 60)
    
    try:
        # 创建基础配置
        config = CrossRingCompatibleConfig()
        
        # 基础验证
        print("基础配置验证:")
        is_valid, error = config.validate_config()
        print(f"  验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
        if error:
            print(f"  错误信息: {error}")
        
        # 使用工厂验证器进行验证
        print("\n工厂验证器验证:")
        factory_valid, factory_error = NoCTopologyFactory.validate_config(config)
        print(f"  验证结果: {'✓ 通过' if factory_valid else '✗ 失败'}")
        if factory_error:
            print(f"  错误信息: {factory_error}")
        
        # 获取性能边界
        print("\n性能边界计算:")
        bounds = config.get_performance_bounds()
        for key, value in bounds.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # 为不同工作负载优化
        print("\n工作负载优化:")
        workloads = [
            {"high_throughput": True},
            {"low_latency": True},
            {"power_sensitive": True}
        ]
        
        for i, workload in enumerate(workloads):
            print(f"  工作负载{i+1}: {list(workload.keys())[0]}")
            optimized_config = config.copy()
            optimized_config.optimize_for_workload(workload)
            
            print(f"    缓冲区深度: {optimized_config.buffer_depth}")
            if hasattr(optimized_config, 'virtual_channels'):
                print(f"    虚拟通道: {optimized_config.virtual_channels}")
            if hasattr(optimized_config, 'enable_power_management'):
                print(f"    功耗管理: {optimized_config.enable_power_management}")
        
        print("\n")
        
    except Exception as e:
        print(f"错误: {e}")


def example_crossring_integration_workflow():
    """CrossRing集成工作流示例。"""
    print("=" * 60)
    print("CrossRing集成工作流示例")
    print("=" * 60)
    
    try:
        print("步骤1: 创建CrossRing配置")
        config = CrossRingCompatibleConfig()
        config.NUM_NODE = 20
        config.NUM_COL = 2
        print(f"  创建了{config.NUM_NODE}节点，{config.NUM_COL}列的CrossRing配置")
        
        print("\n步骤2: 配置验证")
        is_valid, error = config.validate_config()
        if not is_valid:
            print(f"  配置验证失败: {error}")
            return
        print("  配置验证通过")
        
        print("\n步骤3: 使用工厂创建拓扑（模拟）")
        # 注意：这里只是展示工厂接口，实际的拓扑创建需要具体的实现类
        try:
            topology = NoCTopologyFactory.create_topology(config, validate=False)
            print(f"  成功创建拓扑: {topology}")
        except Exception as topo_error:
            print(f"  拓扑创建跳过（需要具体实现类）: {topo_error}")
        
        print("\n步骤4: 配置序列化")
        config_dict = config.to_dict()
        print(f"  配置转字典包含{len(config_dict)}个参数")
        
        # 转换为CrossRing兼容格式
        crossring_format = config.get_crossring_config_dict()
        print(f"  CrossRing格式包含{len(crossring_format)}个参数")
        
        print("\n步骤5: 配置优化建议")
        suggestions = []
        if config.buffer_depth < 8:
            suggestions.append("考虑增加缓冲区深度以提高吞吐量")
        if config.NUM_IP > config.NUM_NODE * 0.8:
            suggestions.append("IP密度较高，注意拥塞控制")
        
        if suggestions:
            print("  优化建议:")
            for suggestion in suggestions:
                print(f"    - {suggestion}")
        else:
            print("  当前配置已经较为优化")
        
        print("\n工作流完成!")
        
    except Exception as e:
        print(f"错误: {e}")


def main():
    """主函数 - 运行所有CrossRing集成示例。"""
    print("NoC抽象层与CrossRing集成示例")
    print("=" * 60)
    
    examples = [
        example_crossring_basic_setup,
        example_crossring_channel_configuration,
        example_crossring_topology_variants,
        example_crossring_config_conversion,
        example_crossring_performance_configuration,
        example_crossring_validation_and_optimization,
        example_crossring_integration_workflow,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"示例{i}执行出错: {e}")
            print()
    
    print("所有CrossRing集成示例执行完成!")


if __name__ == "__main__":
    main()