# -*- coding: utf-8 -*-
"""
C2C拓扑修复和验证脚本
修复所有英文内容为中文，并验证tree和torus拓扑实现
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from src.topology.tree import TreeTopologyLogic, validate_tree_topology, evaluate_tree_performance
from src.topology.torus import TorusTopologyLogic, test_torus_connectivity
from src.topology.node import ChipNode, SwitchNode
from src.topology.link import C2CDirectLink
from src.topology.builder import TopologyBuilder
from src.protocol.cdma import CDMAProtocol, CDMAMessage


def validate_tree_topology_demo():
    """验证树状拓扑实现"""
    print("\n=== 树状拓扑验证 ===")

    # 测试不同芯片数量的树拓扑
    test_cases = [4, 8, 16, 32, 64]

    for chip_count in test_cases:
        print(f"\n--- 测试 {chip_count} 芯片树拓扑 ---")

        try:
            # 创建树拓扑
            topo_logic = TreeTopologyLogic()
            tree_root, all_nodes = topo_logic.calculate_tree_structure(list(range(chip_count)), switch_capacity=8)

            if not tree_root:
                print(f"错误：无法为 {chip_count} 芯片创建树拓扑")
                continue

            print(f"成功创建树拓扑，根节点：{tree_root.node_id}")
            print(f"总节点数：{len(all_nodes)}")

            # 验证拓扑有效性
            validation_result = validate_tree_topology(tree_root, all_nodes, 8)
            if validation_result["is_valid"]:
                print("✓ 拓扑结构有效")
            else:
                print(f"✗ 拓扑结构无效：{validation_result['errors']}")

            # 评估性能
            perf_result = evaluate_tree_performance(tree_root, all_nodes)
            print(f"平均路径长度：{perf_result['average_path_length']}")
            print(f"最大路径长度：{perf_result['max_path_length']}")

            # 测试路由表生成
            routing_table = topo_logic.compute_routing_table(tree_root, all_nodes)
            if len(routing_table) > 0:
                print("✓ 路由表生成成功")
            else:
                print("✗ 路由表生成失败")

        except Exception as e:
            print(f"✗ 树拓扑测试失败：{str(e)}")


def validate_torus_topology_demo():
    """验证环形拓扑实现"""
    print("\n=== 环形拓扑验证 ===")

    # 测试不同芯片数量和维度的环形拓扑
    test_cases = [
        (16, 2),  # 4x4 2D Torus
        (64, 2),  # 8x8 2D Torus
        (27, 3),  # 3x3x3 3D Torus
        (125, 3),  # 5x5x5 3D Torus
    ]

    for chip_count, dimensions in test_cases:
        print(f"\n--- 测试 {chip_count} 芯片 {dimensions}D 环形拓扑 ---")

        try:
            # 创建环形拓扑
            topo_logic = TorusTopologyLogic()
            torus_structure = topo_logic.calculate_torus_structure(chip_count, dimensions)

            print(f"网格尺寸：{torus_structure['grid_dimensions']}")
            print(f"实际芯片数：{torus_structure['chip_count']}")

            # 验证连通性
            connectivity_result = test_torus_connectivity(torus_structure)
            if connectivity_result["is_connected"]:
                print("✓ 环形拓扑连通性验证通过")
            else:
                print(f"✗ 环形拓扑连通性验证失败：{connectivity_result.get('error', '未知错误')}")

            # 测试路由算法
            from src.topology.torus import TorusRoutingLogic

            routing_logic = TorusRoutingLogic()

            # 测试几个路由路径
            test_pairs = [(0, chip_count // 2), (0, chip_count - 1)]
            for src_id, dst_id in test_pairs:
                if dst_id < chip_count:
                    src_coord = torus_structure["coordinate_map"][src_id]
                    dst_coord = torus_structure["coordinate_map"][dst_id]

                    path = routing_logic.dimension_order_routing(src_coord, dst_coord, torus_structure["grid_dimensions"])

                    if path and len(path) > 1:
                        print(f"✓ 芯片 {src_id} 到 {dst_id} 的路由路径长度：{len(path)-1}")
                    else:
                        print(f"✗ 芯片 {src_id} 到 {dst_id} 的路由计算失败")

        except Exception as e:
            print(f"✗ 环形拓扑测试失败：{str(e)}")


def create_tree_topology_with_framework():
    """使用框架创建树状拓扑"""
    print("\n=== 使用框架创建树状拓扑 ===")

    try:
        # 创建8个芯片节点
        chips = []
        for i in range(8):
            chip = ChipNode(chip_id=f"chip_{i}", board_id=f"board_{i//4}", memory_types=["DDR", "HBM"])  # 每4个芯片一个板卡
            chips.append(chip)

        # 创建2个交换机
        switch1 = SwitchNode(switch_id="switch_0", port_count=8, bandwidth=128.0)
        switch2 = SwitchNode(switch_id="switch_1", port_count=8, bandwidth=128.0)
        root_switch = SwitchNode(switch_id="root_switch", port_count=4, bandwidth=256.0)

        # 构建拓扑
        builder = TopologyBuilder("tree_demo")

        # 添加所有节点
        for chip in chips:
            builder.add_node(chip)
        builder.add_node(switch1)
        builder.add_node(switch2)
        builder.add_node(root_switch)

        # 连接芯片到交换机
        for i in range(4):
            builder.add_link(C2CDirectLink(link_id=f"link_chip{i}_switch0", endpoint_a=chips[i], endpoint_b=switch1))

        for i in range(4, 8):
            builder.add_link(C2CDirectLink(link_id=f"link_chip{i}_switch1", endpoint_a=chips[i], endpoint_b=switch2))

        # 连接交换机到根交换机
        builder.add_link(C2CDirectLink(link_id="link_switch0_root", endpoint_a=switch1, endpoint_b=root_switch))

        builder.add_link(C2CDirectLink(link_id="link_switch1_root", endpoint_a=switch2, endpoint_b=root_switch))

        # 构建拓扑图
        topology = builder.build()
        print(f"✓ 成功创建树状拓扑：{topology}")
        print(f"拓扑统计：{topology.get_topology_statistics()}")

        # 测试路径查找
        path = topology.find_path("chip_0", "chip_7")
        if path:
            print(f"✓ 芯片0到芯片7的路径：{' -> '.join(path)}")
        else:
            print("✗ 未找到芯片0到芯片7的路径")

        return topology

    except Exception as e:
        print(f"✗ 树状拓扑创建失败：{str(e)}")
        return None


def create_torus_topology_with_framework():
    """使用框架创建环形拓扑"""
    print("\n=== 使用框架创建环形拓扑 ===")

    try:
        # 创建16个芯片组成4x4网格
        chips = []
        chip_coords = {}  # 芯片坐标映射

        grid_size = 4
        chip_id = 0

        for x in range(grid_size):
            for y in range(grid_size):
                chip = ChipNode(chip_id=f"chip_{chip_id}", board_id=f"board_{x}_{y}", memory_types=["DDR"])
                chips.append(chip)
                chip_coords[chip_id] = (x, y)
                chip_id += 1

        # 构建拓扑
        builder = TopologyBuilder("torus_demo")

        # 添加所有芯片
        for chip in chips:
            builder.add_node(chip)

        # 创建环形连接
        for chip_id, (x, y) in chip_coords.items():
            # 计算邻居
            neighbors = [
                ((x + 1) % grid_size, y),  # 右邻居
                (x, (y + 1) % grid_size),  # 下邻居
            ]

            for nx, ny in neighbors:
                # 找到邻居芯片ID
                neighbor_id = nx * grid_size + ny

                if neighbor_id > chip_id:  # 避免重复连接
                    link_id = f"link_{chip_id}_{neighbor_id}"
                    builder.add_link(C2CDirectLink(link_id=link_id, endpoint_a=chips[chip_id], endpoint_b=chips[neighbor_id]))

        # 构建拓扑图
        topology = builder.build()
        print(f"✓ 成功创建环形拓扑：{topology}")
        print(f"拓扑统计：{topology.get_topology_statistics()}")

        # 测试路径查找
        path = topology.find_path("chip_0", "chip_15")
        if path:
            print(f"✓ 芯片0到芯片15的路径：{' -> '.join(path)}")
        else:
            print("✗ 未找到芯片0到芯片15的路径")

        return topology

    except Exception as e:
        print(f"✗ 环形拓扑创建失败：{str(e)}")
        return None


def test_cdma_communication(topology):
    """测试CDMA通信"""
    print("\n=== 测试CDMA通信 ===")

    try:
        # 创建CDMA协议实例
        cdma_0 = CDMAProtocol("cdma_0", "chip_0")
        cdma_7 = CDMAProtocol("cdma_7", "chip_7")

        # 创建测试消息
        test_message = CDMAMessage(source_id="chip_0", destination_id="chip_7", message_type="send", payload="测试数据从芯片0发送", tensor_shape=(128, 256), data_type="float32")

        print(f"发送消息：{test_message.message_type}")

        # 模拟发送
        cdma_0.send_message(test_message)

        # 模拟接收处理
        ack_message = cdma_7.process_message(test_message)

        if ack_message:
            print("✓ CDMA通信测试成功")
            # 处理ACK
            cdma_0.process_message(ack_message)
        else:
            print("✗ CDMA通信测试失败")

    except Exception as e:
        print(f"✗ CDMA通信测试异常：{str(e)}")


def run_comprehensive_validation():
    """运行综合验证"""
    print("开始C2C拓扑综合验证...")

    # 1. 验证树状拓扑算法
    validate_tree_topology_demo()

    # 2. 验证环形拓扑算法
    validate_torus_topology_demo()

    # 3. 使用框架创建树状拓扑
    tree_topology = create_tree_topology_with_framework()

    # 4. 使用框架创建环形拓扑
    torus_topology = create_torus_topology_with_framework()

    # 5. 测试通信协议
    if tree_topology:
        test_cdma_communication(tree_topology)

    print("\n=== 验证完成 ===")
    print("所有拓扑验证测试已完成")


if __name__ == "__main__":
    run_comprehensive_validation()
