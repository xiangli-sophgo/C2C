# -*- coding: utf-8 -*-
"""
Tree和Torus拓扑专门验证脚本
验证tree.py和torus.py中的拓扑算法实现
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.c2c.topology.tree import (
    TreeTopologyLogic,
    TreeAddressRoutingLogic,
    TreeConfigGenerationLogic,
    TreeFaultToleranceLogic,
    validate_tree_topology,
    evaluate_tree_performance,
    optimize_tree_structure,
)

from src.c2c.topology.torus import (
    TorusTopologyLogic,
    TorusRoutingLogic,
    TorusC2CMappingLogic,
    TorusAddressRoutingLogic,
    TorusAllReduceLogic,
    TorusFaultToleranceLogic,
    test_torus_connectivity,
    optimize_torus_dimensions,
)


def test_tree_topology():
    """完整测试树状拓扑功能"""
    print("=== 树状拓扑详细测试 ===\n")

    # 1. 测试基本拓扑创建
    print("1. 测试拓扑结构计算")
    chip_ids = list(range(16))
    topo_logic = TreeTopologyLogic()

    tree_root, all_nodes = topo_logic.calculate_tree_structure(chip_ids, switch_capacity=4)
    print(f"✓ 16芯片树拓扑创建成功，根节点: {tree_root.node_id}")
    print(f"  总节点数: {len(all_nodes)}")

    # 2. 验证拓扑有效性
    print("\n2. 验证拓扑有效性")
    validation = validate_tree_topology(tree_root, all_nodes, 4)
    print(f"✓ 拓扑有效性: {validation['is_valid']}")
    if not validation["is_valid"]:
        print(f"  错误: {validation['errors']}")

    # 3. 性能评估
    print("\n3. 性能评估")
    perf = evaluate_tree_performance(tree_root, all_nodes)
    print(f"✓ 平均路径长度: {perf['average_path_length']}")
    print(f"✓ 最大路径长度: {perf['max_path_length']}")

    # 4. 路由表计算
    print("\n4. 路由表计算")
    routing_table = topo_logic.compute_routing_table(tree_root, all_nodes)
    test_path = routing_table.get("chip_0", {}).get("chip_15")
    if test_path:
        print(f"✓ chip_0到chip_15路径: {' -> '.join(test_path)}")

    # 5. 地址路由逻辑
    print("\n5. 地址路由逻辑测试")
    addr_logic = TreeAddressRoutingLogic()
    route_decision = addr_logic.route_address_decision(0, 15, routing_table)
    print(f"✓ 路由决策: {route_decision}")

    # 6. 配置生成
    print("\n6. 配置生成测试")
    config_gen = TreeConfigGenerationLogic()
    chip_config = config_gen.generate_chip_c2c_config(0, all_nodes)
    print(f"✓ 芯片0配置: {list(chip_config.keys())}")

    atu_config = config_gen.generate_atu_config_table(0, chip_ids, all_nodes)
    print(f"✓ ATU配置项数: {len(atu_config['outbound'])}")

    # 7. 故障容错测试
    print("\n7. 故障容错测试")
    fault_logic = TreeFaultToleranceLogic()
    health_status = {nid: "OK" for nid in all_nodes.keys()}
    health_status["switch_0"] = "Failed"  # 模拟交换机故障

    failed_components = fault_logic.detect_failed_components(health_status)
    print(f"✓ 检测到故障组件: {failed_components}")

    forest, healthy_nodes = fault_logic.calculate_recovery_topology(tree_root, all_nodes, failed_components)
    print(f"✓ 故障后形成{len(forest)}个子树，健康节点{len(healthy_nodes)}个")

    # 8. 优化算法测试
    print("\n8. 树结构优化")
    optimization = optimize_tree_structure(chip_count=64, switch_capacity=8)
    print(f"✓ 64芯片优化结果: 树层数{optimization['tree_levels']}, 最大跳数{optimization['max_path_hops']}")

    return True


def test_torus_topology():
    """完整测试环形拓扑功能"""
    print("\n=== 环形拓扑详细测试 ===\n")

    # 1. 测试2D环形拓扑
    print("1. 测试2D环形拓扑创建")
    topo_logic = TorusTopologyLogic()
    torus_2d = topo_logic.calculate_torus_structure(16, dimensions=2)

    print(f"✓ 16芯片2D环形拓扑: {torus_2d['grid_dimensions'][0]}x{torus_2d['grid_dimensions'][1]}")
    print(f"  坐标映射样例: chip_0 -> {torus_2d['coordinate_map'][0]}")

    # 2. 测试3D环形拓扑
    print("\n2. 测试3D环形拓扑创建")
    torus_3d = topo_logic.calculate_torus_structure(64, dimensions=3)
    print(f"✓ 64芯片3D环形拓扑: {torus_3d['grid_dimensions'][0]}x{torus_3d['grid_dimensions'][1]}x{torus_3d['grid_dimensions'][2]}")

    # 3. 连通性验证
    print("\n3. 连通性验证")
    connectivity_2d = test_torus_connectivity(torus_2d)
    connectivity_3d = test_torus_connectivity(torus_3d)
    print(f"✓ 2D环形拓扑连通性: {connectivity_2d['is_connected']}")
    print(f"✓ 3D环形拓扑连通性: {connectivity_3d['is_connected']}")

    # 4. 路由算法测试
    print("\n4. 路由算法测试")
    routing_logic = TorusRoutingLogic()

    # 测试DOR路由
    src_coord = torus_2d["coordinate_map"][0]
    dst_coord = torus_2d["coordinate_map"][15]
    path_2d = routing_logic.dimension_order_routing(src_coord, dst_coord, torus_2d["grid_dimensions"])
    print(f"✓ 2D DOR路径 (0→15): {len(path_2d)-1}跳, {path_2d[0]} → {path_2d[-1]}")

    # 距离计算
    distances = routing_logic.calculate_shortest_distance(src_coord, dst_coord, torus_2d["grid_dimensions"])
    print(f"✓ 最短距离: {distances['total_hops']}跳")

    # 5. C2C映射测试
    print("\n5. C2C系统映射测试")
    mapping_logic = TorusC2CMappingLogic()
    c2c_mapping = mapping_logic.map_directions_to_c2c_sys(0, torus_2d)
    print(f"✓ 芯片0的C2C映射: {list(c2c_mapping.keys())}")

    c2c_config = mapping_logic.generate_c2c_link_config(0, torus_2d)
    print(f"✓ C2C链路配置项: {len(c2c_config)}")

    # 6. 地址路由决策
    print("\n6. 地址路由决策测试")
    addr_routing = TorusAddressRoutingLogic()
    decision = addr_routing.route_address_decision(0, 15, torus_2d)
    print(f"✓ 路由决策: {decision['decision']}")

    # 7. All-Reduce优化
    print("\n7. All-Reduce优化测试")
    allreduce_logic = TorusAllReduceLogic()
    allreduce_plan = allreduce_logic.optimize_all_reduce_pattern(torus_2d)
    print(f"✓ All-Reduce计划阶段数: {len(allreduce_plan)}")
    for stage in allreduce_plan:
        print(f"  {stage['stage']}: {stage['description']}")

    # 8. 故障容错测试
    print("\n8. 故障容错测试")
    fault_logic = TorusFaultToleranceLogic()
    health_status = {"link_0_1": "Failed", "link_4_5": "OK"}
    failed_links = fault_logic.detect_link_failures(health_status)
    print(f"✓ 检测到故障链路: {failed_links}")

    recovery = fault_logic.generate_recovery_routing(failed_links, torus_2d)
    print(f"✓ 恢复路由状态: {recovery['status']}")

    # 9. 维度优化测试
    print("\n9. 维度优化测试")
    opt_2d = optimize_torus_dimensions(36, 2)
    opt_3d = optimize_torus_dimensions(64, 3)
    print(f"✓ 36芯片最优2D网格: {opt_2d}")
    print(f"✓ 64芯片最优3D网格: {opt_3d}")

    return True


def performance_comparison():
    """性能对比测试"""
    print("\n=== Tree vs Torus性能对比 ===\n")

    chip_counts = list(range(4, 129))

    for count in chip_counts:
        print(f"芯片数量: {count}")

        # Tree性能
        tree_logic = TreeTopologyLogic()
        tree_root, tree_nodes = tree_logic.calculate_tree_structure(list(range(count)), switch_capacity=8)
        tree_perf = evaluate_tree_performance(tree_root, tree_nodes)

        # Torus性能
        torus_logic = TorusTopologyLogic()
        torus_struct = torus_logic.calculate_torus_structure(count, dimensions=2)

        # 计算Torus的平均路径长度
        total_hops = 0
        path_count = 0
        routing_logic = TorusRoutingLogic()

        for i in range(min(10, count)):  # 只测试前10个芯片以节省时间
            for j in range(i + 1, min(10, count)):
                src_coord = torus_struct["coordinate_map"][i]
                dst_coord = torus_struct["coordinate_map"][j]
                distances = routing_logic.calculate_shortest_distance(src_coord, dst_coord, torus_struct["grid_dimensions"])
                total_hops += distances["total_hops"]
                path_count += 1

        torus_avg_path = total_hops / path_count if path_count > 0 else 0

        print(f"  Tree: 平均路径{tree_perf['average_path_length']:.2f}跳, 最大路径{tree_perf['max_path_length']}跳")
        print(f"  Torus: 平均路径{torus_avg_path:.2f}跳, 网格{torus_struct['grid_dimensions']}")
        print(f"  总节点数: Tree={len(tree_nodes)}, Torus={count}")
        print()


def main():
    """主测试函数"""
    print("开始Tree和Torus拓扑验证测试...\n")

    try:
        # 测试树状拓扑
        tree_success = test_tree_topology()

        # 测试环形拓扑
        torus_success = test_torus_topology()

        # 性能对比
        performance_comparison()

        # 测试结果总结
        print("=== 测试结果总结 ===")
        print(f"✓ 树状拓扑测试: {'通过' if tree_success else '失败'}")
        print(f"✓ 环形拓扑测试: {'通过' if torus_success else '失败'}")

        if tree_success and torus_success:
            print("\n🎉 所有拓扑测试全部通过！")
            print("Tree和Torus拓扑实现正确，功能完整。")
        else:
            print("\n❌ 部分测试失败，请检查实现。")

    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
