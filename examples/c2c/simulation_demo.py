#!/usr/bin/env python3
"""
C2C仿真框架演示脚本
展示如何使用新的仿真功能与现有拓扑和协议组件集成
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.c2c.topology.builder import TopologyBuilder
from src.c2c.topology.node import ChipNode
from src.c2c.topology.link import C2CDirectLink
from src.simulation.engine import C2CSimulationEngine
from src.simulation.fake_chip import FakeChip
from src.simulation.events import EventFactory


def create_simple_topology():
    """创建简单的两芯片拓扑"""
    print("创建简单的两芯片拓扑...")

    # 使用现有的TopologyBuilder
    builder = TopologyBuilder("c2c_simulation_test")

    # 创建芯片节点
    chip0 = ChipNode("chip_0", "board_A", cdma_engines=10)
    chip1 = ChipNode("chip_1", "board_A", cdma_engines=10)

    # 添加节点到拓扑
    builder.add_node(chip0)
    builder.add_node(chip1)

    # 创建C2C直连链路
    c2c_link = C2CDirectLink("link_0_1", chip0, chip1)
    builder.add_link(c2c_link)

    print(f"拓扑创建完成：包含 {len(builder._nodes)} 个节点，{len(builder._links)} 个链路")

    return builder


def run_basic_simulation():
    """运行基础仿真测试"""
    print("\n" + "=" * 60)
    print("开始基础仿真测试")
    print("=" * 60)

    # 创建拓扑
    topology_builder = create_simple_topology()

    # 创建仿真引擎
    print("\n创建C2C仿真引擎...")
    simulator = C2CSimulationEngine(topology_builder)

    # 添加测试事件
    print("\n添加仿真事件...")

    # 单次CDMA发送事件
    simulator.add_cdma_send_event(timestamp_ns=1000, source_chip_id="chip_0", target_chip_id="chip_1", data_size=1024)

    # 反向发送事件
    simulator.add_cdma_send_event(timestamp_ns=5000, source_chip_id="chip_1", target_chip_id="chip_0", data_size=2048)

    # 添加周期性流量
    simulator.add_periodic_traffic(source_chip_id="chip_0", target_chip_id="chip_1", period_ns=10000, data_size=512, start_time_ns=20000, end_time_ns=100000)  # 10微秒周期

    # 运行仿真
    print(f"\n事件队列中有 {len(simulator.event_queue)} 个事件")
    print("开始运行仿真...")

    # 仿真1毫秒（1,000,000纳秒）
    simulation_time_ns = 1_000_000
    stats = simulator.run_simulation(simulation_time_ns)

    # 显示结果
    print("\n仿真结果：")
    stats.print_summary()

    # 获取详细结果
    results = simulator.get_simulation_results()

    print(f"\n详细结果：")
    print(f"拓扑信息：{results['topology_info']}")

    # 显示各芯片统计
    print(f"\n各芯片统计：")
    for chip_id, chip_stats in results["chips"].items():
        print(f"  {chip_id}:")
        print(f"    发送消息: {chip_stats['sent_messages']}")
        print(f"    接收消息: {chip_stats['received_messages']}")
        print(f"    发送字节: {chip_stats['total_bytes_sent']}")
        print(f"    接收字节: {chip_stats['total_bytes_received']}")

    return simulator, stats


def run_complex_simulation():
    """运行复杂的多芯片仿真"""
    print("\n" + "=" * 60)
    print("开始复杂多芯片仿真测试")
    print("=" * 60)

    # 创建4芯片环形拓扑
    builder = TopologyBuilder("4chip_ring_topology")

    # 创建4个芯片
    chips = []
    for i in range(4):
        chip = ChipNode(f"chip_{i}", f"board_{i//2}", cdma_engines=10)
        chips.append(chip)
        builder.add_node(chip)

    # 创建环形连接: 0->1->2->3->0
    for i in range(4):
        next_i = (i + 1) % 4
        link = C2CDirectLink(f"link_{i}_{next_i}", chips[i], chips[next_i])
        builder.add_link(link)

    print(f"创建4芯片环形拓扑，包含 {len(builder._nodes)} 个节点，{len(builder._links)} 个链路")

    # 创建仿真引擎
    simulator = C2CSimulationEngine(builder)

    # 添加复杂的流量模式
    print("\n添加复杂流量模式...")

    # 每个芯片都向下一个芯片发送数据
    for i in range(4):
        next_i = (i + 1) % 4
        simulator.add_periodic_traffic(
            source_chip_id=f"chip_{i}", target_chip_id=f"chip_{next_i}", period_ns=25000, data_size=1024, start_time_ns=i * 5000, end_time_ns=500000  # 25微秒周期  # 错开启动时间  # 500微秒
        )

    # 添加跨芯片通信（chip_0 <-> chip_2）
    simulator.add_periodic_traffic(source_chip_id="chip_0", target_chip_id="chip_2", period_ns=50000, data_size=2048, start_time_ns=10000, end_time_ns=400000)  # 50微秒周期

    # 运行更长时间的仿真
    simulation_time_ns = 1_000_000  # 1毫秒
    print(f"运行仿真，时长: {simulation_time_ns/1e6:.1f} ms")

    stats = simulator.run_simulation(simulation_time_ns)

    # 显示结果
    print("\n复杂仿真结果：")
    stats.print_summary()

    # 导出统计数据
    stats.export_to_json("complex_simulation_stats.json")

    return simulator, stats


def test_fake_chip_functionality():
    """测试FakeChip独立功能"""
    print("\n" + "=" * 60)
    print("测试FakeChip独立功能")
    print("=" * 60)

    # 创建两个FakeChip实例
    chip_a = FakeChip("test_chip_a", "test_board", cdma_engines=5)
    chip_b = FakeChip("test_chip_b", "test_board", cdma_engines=5)

    # 连接C2C端口
    chip_a.connect_c2c_port(0, chip_b)
    chip_b.connect_c2c_port(0, chip_a)

    # 创建测试事件
    test_events = [
        EventFactory.create_cdma_send_event(1000, "test_chip_a", "test_chip_b", 1024),
        EventFactory.create_cdma_receive_event(2000, "test_chip_a", "test_chip_b"),
        EventFactory.create_cdma_send_event(3000, "test_chip_b", "test_chip_a", 2048),
    ]

    # 处理事件
    print("\n处理测试事件...")
    for event in test_events:
        if event.source_chip_id == "test_chip_a":
            chip_a.process_simulation_event(event)
        elif event.source_chip_id == "test_chip_b":
            chip_b.process_simulation_event(event)

    # 显示统计信息
    print(f"\n芯片A统计: {chip_a}")
    print(f"芯片B统计: {chip_b}")

    print(f"\n芯片A详细统计:")
    stats_a = chip_a.get_statistics()
    for key, value in stats_a.items():
        print(f"  {key}: {value}")


def main():
    """主函数"""
    print("C2C仿真框架集成测试")
    print("测试与现有拓扑和协议组件的集成")

    try:
        # 测试1：基础双芯片仿真
        simulator1, stats1 = run_basic_simulation()

        # 测试2：复杂多芯片仿真
        simulator2, stats2 = run_complex_simulation()

        # 测试3：FakeChip独立功能测试
        test_fake_chip_functionality()

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("仿真框架与现有组件集成成功")
        print("=" * 60)

        # 比较性能
        print(f"\n性能对比:")
        metrics1 = stats1.get_performance_metrics()
        metrics2 = stats2.get_performance_metrics()

        print(f"基础仿真:")
        for metric, value in metrics1.items():
            print(f"  {metric}: {value:.2f}")

        print(f"复杂仿真:")
        for metric, value in metrics2.items():
            print(f"  {metric}: {value:.2f}")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
