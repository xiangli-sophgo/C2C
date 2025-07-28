#!/usr/bin/env python3
"""
CrossRing Link State Visualizer 演示

基于原版Link_State_Visualizer重新实现的完整可视化系统演示。
展示左侧网络拓扑 + 右侧节点详细视图的完整布局。

Usage:
    python crossring_link_state_demo.py
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from pathlib import Path
import threading
from types import SimpleNamespace
import matplotlib

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.noc.visualization.link_state_visualizer import LinkStateVisualizer, _FlitProxy


def create_demo_crossring_model():
    """创建演示用的CrossRing模型"""
    # 创建模型结构
    model = SimpleNamespace()
    model.nodes = {}
    model.links = {}

    # 创建4个节点 (2x2网格)
    for node_id in range(4):
        node = SimpleNamespace()

        # 注入方向FIFOs
        node.inject_input_fifos = {"TL": create_demo_fifo(), "TR": create_demo_fifo(), "TU": create_demo_fifo(), "TD": create_demo_fifo()}

        # 提取输入FIFOs
        node.eject_input_fifos = {"TL": create_demo_fifo(), "TR": create_demo_fifo(), "TU": create_demo_fifo(), "TD": create_demo_fifo()}

        # 通道缓冲区
        node.channel_buffer = {"gdma": create_demo_fifo(), "ddr": create_demo_fifo()}

        # IP提取通道缓冲区
        node.ip_eject_channel_buffers = {"gdma": create_demo_fifo(), "ddr": create_demo_fifo()}

        # IP接口
        node.ip_interfaces = {"gdma": create_demo_ip_interface(), "ddr": create_demo_ip_interface()}

        # Ring Bridge
        node.ring_bridge = SimpleNamespace()
        node.ring_bridge.ring_bridge_input = {"TL": create_demo_fifo(), "TR": create_demo_fifo(), "TU": create_demo_fifo(), "TD": create_demo_fifo()}
        node.ring_bridge.ring_bridge_output = {"TL": create_demo_fifo(), "TR": create_demo_fifo(), "TU": create_demo_fifo(), "TD": create_demo_fifo()}

        # CrossPoint
        node.horizontal_cp = SimpleNamespace()
        node.horizontal_cp.arbitration_state = "idle"
        node.horizontal_cp.active_connections = []
        node.horizontal_cp.priority_state = "normal"

        node.vertical_cp = SimpleNamespace()
        node.vertical_cp.arbitration_state = "idle"
        node.vertical_cp.active_connections = []
        node.vertical_cp.priority_state = "normal"

        model.nodes[node_id] = node

    # 创建链路
    link_configs = [("h_0_1", 0, 1), ("h_2_3", 2, 3), ("v_0_2", 0, 2), ("v_1_3", 1, 3)]  # 水平链路  # 垂直链路

    for link_id, src, dest in link_configs:
        link = SimpleNamespace()
        link.slices = []

        # 创建8个slice
        for i in range(8):
            slice_obj = SimpleNamespace()
            slice_obj.slot = create_demo_slot() if random.random() < 0.3 else None
            link.slices.append(slice_obj)

        model.links[link_id] = link

    return model


def create_demo_fifo():
    """创建演示FIFO"""
    fifo = SimpleNamespace()
    fifo.queue = []

    # 随机添加一些flit
    for i in range(random.randint(0, 3)):
        flit = _FlitProxy(pid=random.randint(1, 4), fid=f"F{i}", etag=random.choice(["T0", "T1", "T2"]), ih=random.random() < 0.1, iv=random.random() < 0.1)
        fifo.queue.append(flit)

    return fifo


def create_demo_ip_interface():
    """创建演示IP接口"""
    ip = SimpleNamespace()

    # L2H FIFOs
    ip.l2h_fifos = {"req": create_demo_fifo(), "rsp": create_demo_fifo(), "data": create_demo_fifo()}

    # H2L FIFOs
    ip.h2l_fifos = {"req": create_demo_fifo(), "rsp": create_demo_fifo(), "data": create_demo_fifo()}

    return ip


def create_demo_slot():
    """创建演示slot"""
    slot = SimpleNamespace()
    slot.valid = True
    slot.packet_id = random.randint(1, 4)
    slot.flit_id = f"F{random.randint(0, 7)}"
    slot.etag_priority = random.choice(["T0", "T1", "T2"])
    slot.itag_h = random.random() < 0.1
    slot.itag_v = random.random() < 0.1
    return slot


def demo_static():
    """静态演示 - 展示基本布局和功能"""
    print("🏗️  静态演示 - 基本布局")
    print("-" * 40)

    # 创建配置
    config = SimpleNamespace(NUM_ROW=2, NUM_COL=2, IQ_OUT_FIFO_DEPTH=8, EQ_IN_FIFO_DEPTH=8, RB_IN_FIFO_DEPTH=4, RB_OUT_FIFO_DEPTH=4, IQ_CH_FIFO_DEPTH=4, EQ_CH_FIFO_DEPTH=4, SLICE_PER_LINK=8)

    # 创建演示模型
    demo_model = create_demo_crossring_model()

    # 创建可视化器
    visualizer = LinkStateVisualizer(config, demo_model)

    print("💡 静态演示内容:")
    print("- 左侧: 2x2 CrossRing网络拓扑")
    print("- 右侧: 选中节点的详细视图")
    print("- 底部: 控制按钮 (REQ/RSP/DATA, Clear HL, Show Tags)")
    print("- 点击节点可切换详细视图")
    print("- 点击关闭窗口结束演示")

    # 初始更新
    visualizer.update(demo_model)

    # 显示
    visualizer.show()


def demo_animated():
    """动态演示 - 展示数据流动"""
    print("🎬 动态演示 - 数据流动")
    print("-" * 40)

    # 创建配置
    config = SimpleNamespace(NUM_ROW=2, NUM_COL=2, IQ_OUT_FIFO_DEPTH=8, EQ_IN_FIFO_DEPTH=8, RB_IN_FIFO_DEPTH=4, RB_OUT_FIFO_DEPTH=4, IQ_CH_FIFO_DEPTH=4, EQ_CH_FIFO_DEPTH=4, SLICE_PER_LINK=8)

    # 创建可视化器
    demo_model = create_demo_crossring_model()
    visualizer = LinkStateVisualizer(config, demo_model)

    print("💡 动态演示内容:")
    print("- 实时更新节点FIFO状态")
    print("- 实时更新链路slot占用")
    print("- 模拟flit在网络中流动")
    print("- 按Ctrl+C结束演示")

    try:
        for cycle in range(100):
            # 更新演示数据
            update_demo_model(demo_model)

            # 更新可视化
            visualizer.update(demo_model, cycle=cycle)

            # 暂停
            plt.pause(0.8)

            if cycle % 10 == 0:
                node_count = sum(1 for node in demo_model.nodes.values() for fifo in node.inject_input_fifos.values() if fifo.queue)
                link_count = sum(1 for link in demo_model.links.values() for slice_obj in link.slices if slice_obj.slot and slice_obj.slot.valid)
                print(f"周期 {cycle}: 节点队列 {node_count}, 链路活跃 {link_count}")

    except KeyboardInterrupt:
        print("\n演示结束")

    # 保持窗口打开
    plt.ioff()
    plt.show()


def update_demo_model(model):
    """更新演示模型的数据"""
    # 随机更新节点FIFO数据
    for node in model.nodes.values():
        # 更新inject_input_fifos
        for direction, fifo in node.inject_input_fifos.items():
            # 随机添加或移除flit
            if random.random() < 0.2 and len(fifo.queue) < 3:
                flit = _FlitProxy(pid=random.randint(1, 4), fid=f"F{random.randint(0, 7)}", etag=random.choice(["T0", "T1", "T2"]), ih=random.random() < 0.1, iv=random.random() < 0.1)
                fifo.queue.append(flit)
            elif random.random() < 0.3 and fifo.queue:
                fifo.queue.pop(0)

        # 更新eject_input_fifos
        for direction, fifo in node.eject_input_fifos.items():
            if random.random() < 0.15 and len(fifo.queue) < 2:
                flit = _FlitProxy(pid=random.randint(1, 4), fid=f"F{random.randint(0, 7)}", etag=random.choice(["T0", "T1", "T2"]), ih=random.random() < 0.1, iv=random.random() < 0.1)
                fifo.queue.append(flit)
            elif random.random() < 0.25 and fifo.queue:
                fifo.queue.pop(0)

        # 更新channel_buffer
        for channel, buffer in node.channel_buffer.items():
            if random.random() < 0.1 and len(buffer.queue) < 2:
                flit = _FlitProxy(pid=random.randint(1, 4), fid=f"F{random.randint(0, 7)}", etag=random.choice(["T0", "T1", "T2"]), ih=random.random() < 0.1, iv=random.random() < 0.1)
                buffer.queue.append(flit)
            elif random.random() < 0.2 and buffer.queue:
                buffer.queue.pop(0)

    # 随机更新链路slot数据
    for link in model.links.values():
        for slice_obj in link.slices:
            if random.random() < 0.1:
                if slice_obj.slot is None:
                    slice_obj.slot = create_demo_slot()
                else:
                    slice_obj.slot = None


def main():
    """主函数"""
    print("🎪 CrossRing Link State Visualizer 演示")
    print("=" * 50)
    print("基于原版Link_State_Visualizer完整重新实现")
    print()

    demos = {"1": ("静态演示", demo_static), "2": ("动态演示", demo_animated)}

    choice = "2"

    if choice not in demos:
        print("无效选择，使用默认静态演示")
        choice = "1"

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
