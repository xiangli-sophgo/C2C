#!/usr/bin/env python3
"""
CrossRing XY和YX路由对比测试

测试验证CrossRing网络中XY和YX路由策略的实现正确性。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.noc.crossring.config import CrossRingConfig, RoutingStrategy
from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.ring_directions import RingDirectionMapper, RingDirection


def test_routing_strategy_configuration():
    """测试路由策略配置功能"""
    print("=== 测试路由策略配置 ===")
    
    # 测试默认XY路由
    config_xy = CrossRingConfig(num_row=3, num_col=3, config_name="test_xy")
    print(f"默认路由策略: {config_xy.get_routing_strategy()}")
    assert config_xy.routing_strategy == RoutingStrategy.XY
    
    # 测试设置YX路由
    config_yx = CrossRingConfig(num_row=3, num_col=3, config_name="test_yx")
    config_yx.set_routing_strategy("YX")
    print(f"设置后路由策略: {config_yx.get_routing_strategy()}")
    assert config_yx.routing_strategy == RoutingStrategy.YX
    
    # 测试无效路由策略
    try:
        config_yx.set_routing_strategy("INVALID")
        assert False, "应该抛出ValueError"
    except ValueError:
        print("无效路由策略正确被拒绝")
    
    print("✓ 路由策略配置测试通过\n")


def test_routing_decision_logic():
    """测试路由决策逻辑"""
    print("=== 测试路由决策逻辑 ===")
    
    # 创建两个配置：XY和YX
    config_xy = CrossRingConfig(num_row=3, num_col=3, config_name="test_xy")
    config_yx = CrossRingConfig(num_row=3, num_col=3, config_name="test_yx")
    config_yx.set_routing_strategy("YX")
    
    model_xy = CrossRingModel(config_xy)
    model_yx = CrossRingModel(config_yx)
    
    # 测试路由方向决策
    source = 0  # (0,0)
    destination = 8  # (2,2)
    
    print(f"从节点{source}到节点{destination}:")
    print(f"源坐标: {model_xy._get_node_coordinates(source)}")
    print(f"目标坐标: {model_xy._get_node_coordinates(destination)}")
    
    # XY路由：应该先水平后垂直
    directions_xy = model_xy._determine_ring_directions_four_way(source, destination)
    print(f"XY路由方向: {[d.value for d in directions_xy]}")
    
    # YX路由：应该先垂直后水平
    directions_yx = model_yx._determine_ring_directions_four_way(source, destination)
    print(f"YX路由方向: {[d.value for d in directions_yx]}")
    
    # 验证方向顺序不同
    if len(directions_xy) == 2 and len(directions_yx) == 2:
        # 对于对角线路径，应该方向相反
        assert directions_xy[0] != directions_yx[0], "XY和YX路由的第一个方向应该不同"
        assert directions_xy[1] != directions_yx[1], "XY和YX路由的第二个方向应该不同"
        print("✓ 路由方向顺序验证通过")
    
    print("✓ 路由决策逻辑测试通过\n")


def test_single_dimension_routing():
    """测试单维度路由（验证基本功能）"""
    print("=== 测试单维度路由 ===")
    
    config = CrossRingConfig(num_row=3, num_col=3, config_name="test_single")
    model = CrossRingModel(config)
    
    # 测试只需要水平移动的情况
    source = 0  # (0,0)
    destination = 2  # (2,0)
    
    directions = model._determine_ring_directions_four_way(source, destination)
    print(f"水平移动 {source} -> {destination}: {[d.value for d in directions]}")
    assert len(directions) == 1, "单维度移动应该只有一个方向"
    assert directions[0].value in ["TL", "TR"], "水平移动应该是TL或TR"
    
    # 测试只需要垂直移动的情况
    source = 0  # (0,0)
    destination = 6  # (0,2)
    
    directions = model._determine_ring_directions_four_way(source, destination)
    print(f"垂直移动 {source} -> {destination}: {[d.value for d in directions]}")
    assert len(directions) == 1, "单维度移动应该只有一个方向"
    assert directions[0].value in ["TU", "TD"], "垂直移动应该是TU或TD"
    
    print("✓ 单维度路由测试通过\n")


def test_routing_strategy_in_model():
    """测试模型中的路由策略应用"""
    print("=== 测试模型中的路由策略应用 ===")
    
    # 创建3x3网格用于测试
    config_xy = CrossRingConfig(num_row=3, num_col=3, config_name="model_test_xy")
    config_yx = CrossRingConfig(num_row=3, num_col=3, config_name="model_test_yx")
    config_yx.set_routing_strategy("YX")
    
    model_xy = CrossRingModel(config_xy)
    model_yx = CrossRingModel(config_yx)
    
    print(f"XY模型路由策略: {model_xy.config.routing_strategy}")
    print(f"YX模型路由策略: {model_yx.config.routing_strategy}")
    
    # 验证模型能正确访问路由策略
    assert model_xy.config.routing_strategy == RoutingStrategy.XY
    assert model_yx.config.routing_strategy == RoutingStrategy.YX
    
    print("✓ 模型路由策略应用测试通过\n")


def print_routing_comparison():
    """打印XY和YX路由的详细对比"""
    print("=== XY vs YX 路由对比分析 ===")
    
    config_xy = CrossRingConfig(num_row=3, num_col=3, config_name="compare_xy")
    config_yx = CrossRingConfig(num_row=3, num_col=3, config_name="compare_yx")
    config_yx.set_routing_strategy("YX")
    
    model_xy = CrossRingModel(config_xy)
    model_yx = CrossRingModel(config_yx)
    
    # 测试多个路径
    test_paths = [
        (0, 8),  # (0,0) -> (2,2) 对角线
        (0, 2),  # (0,0) -> (2,0) 纯水平  
        (0, 6),  # (0,0) -> (0,2) 纯垂直
        (1, 7),  # (1,0) -> (1,2) 纯垂直
        (3, 5),  # (0,1) -> (2,1) 纯水平
        (1, 5),  # (1,0) -> (2,1) 对角线
    ]
    
    print(f"{'Source':<8} {'Dest':<8} {'XY Route':<20} {'YX Route':<20}")
    print("-" * 60)
    
    for source, dest in test_paths:
        src_coord = model_xy._get_node_coordinates(source)
        dst_coord = model_xy._get_node_coordinates(dest)
        
        directions_xy = model_xy._determine_ring_directions_four_way(source, dest)
        directions_yx = model_yx._determine_ring_directions_four_way(source, dest)
        
        xy_route = " -> ".join([d.value for d in directions_xy]) if directions_xy else "Local"
        yx_route = " -> ".join([d.value for d in directions_yx]) if directions_yx else "Local"
        
        print(f"{source}({src_coord[0]},{src_coord[1]})      {dest}({dst_coord[0]},{dst_coord[1]})      {xy_route:<20} {yx_route:<20}")
    
    print("\n说明:")
    print("- TL: Top-Left (逆时针水平)")
    print("- TR: Top-Right (顺时针水平)")
    print("- TU: Top-Up (向上垂直)")
    print("- TD: Top-Down (向下垂直)")
    print()


def main():
    """主测试函数"""
    print("CrossRing XY/YX 路由测试")
    print("=" * 50)
    
    try:
        test_routing_strategy_configuration()
        test_routing_decision_logic()
        test_single_dimension_routing()
        test_routing_strategy_in_model()
        print_routing_comparison()
        
        print("🎉 所有测试通过！XY和YX路由实现正确。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())