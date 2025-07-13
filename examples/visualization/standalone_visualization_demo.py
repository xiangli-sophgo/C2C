#!/usr/bin/env python3
"""
独立可视化演示

不依赖复杂模型的简单可视化演示，展示：
1. Link可视化器基本功能
2. CrossRing节点可视化器基本功能
3. 两者的简单集成

Usage:
    python standalone_visualization_demo.py
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from pathlib import Path
import matplotlib

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.noc.visualization.link_visualizer import BaseLinkVisualizer, SlotData, LinkStats, SlotState
from src.noc.visualization.crossring_node_visualizer import CrossRingNodeVisualizer, FlitProxy, CrossPointData


def demo_link_only():
    """仅演示Link可视化器"""
    print("🔗 Link可视化器独立演示")
    print("-" * 40)

    # 创建Link可视化器
    fig, ax = plt.subplots(figsize=(12, 6))
    visualizer = BaseLinkVisualizer(ax=ax, link_id="演示链路", num_slots=6)

    print("💡 演示内容:")
    print("- 随机生成slot占用状态")
    print("- 不同颜色表示不同优先级")
    print("- 实时更新统计信息")
    print("- 按Ctrl+C结束演示")

    try:
        for cycle in range(50):
            # 生成随机slot数据
            channels = ["req", "rsp", "data"]
            slots_data = {}

            for channel in channels:
                slot_list = []
                for i in range(6):
                    if random.random() < 0.4:  # 40%概率被占用
                        slot = SlotData(
                            slot_id=i,
                            cycle=cycle,
                            state=SlotState.OCCUPIED,
                            flit_id=f"F{i}",
                            packet_id=f"P{random.randint(1,4)}",
                            priority=random.choice(["T0", "T1", "T2"]),
                            valid=True,
                            itag=random.random() < 0.1,
                            etag=random.random() < 0.05,
                        )
                    else:
                        slot = SlotData(slot_id=i, cycle=cycle, state=SlotState.EMPTY)
                    slot_list.append(slot)
                slots_data[channel] = slot_list

            # 更新显示
            visualizer.update_slots(slots_data)

            # 更新统计信息
            stats = LinkStats(
                bandwidth_utilization=0.3 + 0.2 * np.sin(cycle * 0.1),
                average_latency=12 + 3 * np.sin(cycle * 0.08),
                congestion_level=0.1 + 0.15 * np.sin(cycle * 0.12),
                itag_triggers=random.randint(0, 3),
                etag_upgrades=random.randint(0, 2),
                total_flits=cycle * 5 + random.randint(0, 5),
            )
            visualizer.update_statistics(stats)

            plt.pause(0.5)

            if cycle % 10 == 0:
                print(f"周期 {cycle}: 带宽 {stats.bandwidth_utilization:.1%}, " f"延迟 {stats.average_latency:.1f}")

    except KeyboardInterrupt:
        print("\n演示结束")

    plt.show()


def demo_node_only():
    """仅演示Node可视化器"""
    print("🎯 CrossRing节点可视化器独立演示")
    print("-" * 40)

    from types import SimpleNamespace

    # 创建简单配置
    config = SimpleNamespace(
        NUM_COL=2, NUM_ROW=2, IQ_OUT_FIFO_DEPTH=6, EQ_IN_FIFO_DEPTH=6, RB_IN_FIFO_DEPTH=4, RB_OUT_FIFO_DEPTH=4, IQ_CH_FIFO_DEPTH=3, EQ_CH_FIFO_DEPTH=3, CH_NAME_LIST=["gdma", "ddr"]
    )

    # 创建节点可视化器
    fig, ax = plt.subplots(figsize=(10, 8))
    visualizer = CrossRingNodeVisualizer(config, ax=ax, node_id=0)

    print("💡 演示内容:")
    print("- 注入队列(Inject Queue)动态变化")
    print("- 提取队列(Eject Queue)动态变化")
    print("- Ring Bridge状态变化")
    print("- CrossPoint仲裁状态")
    print("- 按Ctrl+C结束演示")

    try:
        for cycle in range(30):
            # 生成节点数据
            node_data = {"inject_queues": {}, "eject_queues": {}, "ring_bridge": {}, "crosspoints": {}}

            # 注入队列数据
            for lane in ["gdma", "ddr", "TL", "TR"]:
                flits = []
                for i in range(random.randint(0, 4)):
                    flit = FlitProxy(
                        packet_id=f"P{random.randint(1,3)}",
                        flit_id=f"F{i}",
                        etag_priority=random.choice(["T0", "T1", "T2"]),
                        itag_h=random.random() < 0.1,
                        itag_v=random.random() < 0.1,
                    )
                    flits.append(flit)
                node_data["inject_queues"][lane] = flits

            # 提取队列数据
            for lane in ["gdma", "ddr", "TU", "TD"]:
                flits = []
                for i in range(random.randint(0, 3)):
                    flit = FlitProxy(packet_id=f"P{random.randint(1,3)}", flit_id=f"F{i}", etag_priority=random.choice(["T0", "T1", "T2"]))
                    flits.append(flit)
                node_data["eject_queues"][lane] = flits

            # Ring Bridge数据
            for lane in ["TL_in", "TR_out"]:
                flits = []
                if random.random() < 0.3:
                    flit = FlitProxy(packet_id=f"P{random.randint(1,3)}", flit_id="F0", etag_priority=random.choice(["T0", "T1", "T2"]))
                    flits.append(flit)
                node_data["ring_bridge"][lane] = flits

            # CrossPoint数据
            node_data["crosspoints"] = {
                "horizontal": CrossPointData(
                    cp_id="h_cp",
                    direction="horizontal",
                    arbitration_state=random.choice(["idle", "active", "blocked"]),
                    active_connections=[("TL", "TR")] if random.random() < 0.3 else [],
                ),
                "vertical": CrossPointData(
                    cp_id="v_cp", direction="vertical", arbitration_state=random.choice(["idle", "active"]), active_connections=[("TU", "TD")] if random.random() < 0.2 else []
                ),
            }

            # 更新显示
            visualizer.update_node_state(node_data)
            plt.pause(0.8)

            if cycle % 5 == 0:
                iq_total = sum(len(flits) for flits in node_data["inject_queues"].values())
                eq_total = sum(len(flits) for flits in node_data["eject_queues"].values())
                print(f"周期 {cycle}: 注入队列 {iq_total} flits, 提取队列 {eq_total} flits")

    except KeyboardInterrupt:
        print("\n演示结束")

    plt.show()


def demo_combined():
    """联合演示Link和Node可视化器"""
    print("🔄 Link + Node联合演示")
    print("-" * 40)

    from types import SimpleNamespace

    # 创建配置
    config = SimpleNamespace(
        NUM_COL=2, NUM_ROW=2, IQ_OUT_FIFO_DEPTH=6, EQ_IN_FIFO_DEPTH=6, RB_IN_FIFO_DEPTH=4, RB_OUT_FIFO_DEPTH=4, IQ_CH_FIFO_DEPTH=3, EQ_CH_FIFO_DEPTH=3, CH_NAME_LIST=["gdma", "ddr"]
    )

    # 创建分布图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("CrossRing可视化系统联合演示", fontsize=16, fontweight="bold")

    # 创建可视化器
    link_vis = BaseLinkVisualizer(ax=ax1, link_id="链路0", num_slots=6)
    node_vis1 = CrossRingNodeVisualizer(config, ax=ax2, node_id=0)
    node_vis2 = CrossRingNodeVisualizer(config, ax=ax3, node_id=1)

    # 性能监控图
    ax4.set_title("性能监控")
    ax4.set_xlabel("周期")
    ax4.set_ylabel("指标值")
    (bandwidth_line,) = ax4.plot([], [], "b-", label="带宽利用率")
    (latency_line,) = ax4.plot([], [], "r-", label="延迟")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 30)
    ax4.set_ylim(0, 1)

    # 存储性能数据
    perf_data = {"cycles": [], "bandwidth": [], "latency": []}

    print("💡 联合演示内容:")
    print("- 左上: Link状态可视化")
    print("- 右上: 节点0内部结构")
    print("- 左下: 节点1内部结构")
    print("- 右下: 性能监控图表")
    print("- 按Ctrl+C结束演示")

    try:
        for cycle in range(30):
            # 更新Link
            channels = ["req", "rsp", "data"]
            slots_data = {}
            for channel in channels:
                slot_list = []
                for i in range(6):
                    if random.random() < 0.35:
                        slot = SlotData(
                            slot_id=i,
                            cycle=cycle,
                            state=SlotState.OCCUPIED,
                            flit_id=f"F{i}",
                            packet_id=f"P{random.randint(1,3)}",
                            priority=random.choice(["T0", "T1", "T2"]),
                            valid=True,
                        )
                    else:
                        slot = SlotData(slot_id=i, cycle=cycle, state=SlotState.EMPTY)
                    slot_list.append(slot)
                slots_data[channel] = slot_list

            link_vis.update_slots(slots_data)

            # 更新统计信息
            bandwidth = 0.4 + 0.3 * np.sin(cycle * 0.2)
            latency = 0.3 + 0.2 * np.sin(cycle * 0.15)

            stats = LinkStats(bandwidth_utilization=bandwidth, average_latency=latency * 50, congestion_level=0.1, total_flits=cycle * 4)  # 转换为实际延迟值
            link_vis.update_statistics(stats)

            # 更新Node数据
            for node_vis, node_id in [(node_vis1, 0), (node_vis2, 1)]:
                node_data = {"inject_queues": {}, "eject_queues": {}, "ring_bridge": {}, "crosspoints": {}}

                # 生成节点数据
                for lane in ["gdma", "ddr", "TL", "TR"]:
                    flits = []
                    for i in range(random.randint(0, 3)):
                        flit = FlitProxy(packet_id=f"P{random.randint(1,3)}", flit_id=f"F{i}", etag_priority=random.choice(["T0", "T1", "T2"]))
                        flits.append(flit)
                    node_data["inject_queues"][lane] = flits

                for lane in ["gdma", "ddr"]:
                    flits = []
                    if random.random() < 0.4:
                        flit = FlitProxy(packet_id=f"P{random.randint(1,3)}", flit_id="F0")
                        flits.append(flit)
                    node_data["eject_queues"][lane] = flits

                node_data["crosspoints"] = {
                    "horizontal": CrossPointData("h_cp", "horizontal", arbitration_state=random.choice(["idle", "active"])),
                    "vertical": CrossPointData("v_cp", "vertical", arbitration_state=random.choice(["idle", "active"])),
                }

                node_vis.update_node_state(node_data)

            # 更新性能图表
            perf_data["cycles"].append(cycle)
            perf_data["bandwidth"].append(bandwidth)
            perf_data["latency"].append(latency)

            bandwidth_line.set_data(perf_data["cycles"], perf_data["bandwidth"])
            latency_line.set_data(perf_data["cycles"], perf_data["latency"])

            if len(perf_data["cycles"]) > 1:
                ax4.set_xlim(max(0, cycle - 20), cycle + 2)

            plt.pause(0.6)

            if cycle % 5 == 0:
                print(f"周期 {cycle}: 带宽 {bandwidth:.1%}, 延迟 {latency:.2f}")

    except KeyboardInterrupt:
        print("\n演示结束")

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("🎪 CrossRing可视化系统独立演示")
    print("=" * 50)
    print("基于旧版本重构的新可视化架构演示")
    print()

    demos = {"1": ("Link可视化器", demo_link_only), "2": ("Node可视化器", demo_node_only), "3": ("联合演示", demo_combined)}

    print("请选择演示类型:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")
    print()

    choice = input("输入选择 (1-3, 默认3): ").strip() or "3"

    if choice not in demos:
        print("无效选择，使用默认联合演示")
        choice = "3"

    name, demo_func = demos[choice]
    print(f"\n🚀 启动 {name}...")

    try:
        demo_func()
        print(f"\n✅ {name} 演示完成")
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
