#!/usr/bin/env python3
"""
CrossRing可视化演示

展示新的可视化系统功能，包括：
1. 独立的Link可视化器演示
2. 独立的CrossRing节点可视化器演示  
3. 完整的实时可视化系统演示
4. 与真实CrossRing模型的集成演示

使用方法:
    python crossring_visualization_demo.py [demo_type]
    
demo_type选项:
    - link: Link可视化器演示
    - node: 节点可视化器演示  
    - realtime: 实时可视化演示
    - integration: 与CrossRing模型集成演示
    - all: 运行所有演示（默认）
"""

import sys
import logging
from pathlib import Path
import time
import threading

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.noc.visualization import BaseLinkVisualizer, CrossRingNodeVisualizer, RealtimeVisualizer
from src.noc.visualization.link_visualizer import create_demo_slot_data, SlotData, LinkStats, SlotState
from src.noc.visualization.crossring_node_visualizer import create_demo_node_data, FlitProxy
from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig

import matplotlib.pyplot as plt
import numpy as np
import random


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("VisualizationDemo")
    return logger


def demo_link_visualizer():
    """演示Link可视化器"""
    print("\n" + "="*60)
    print("🔗 Link可视化器演示")
    print("="*60)
    print("展示通用Link状态可视化功能，适用于所有拓扑类型")
    
    # 创建Link可视化器
    visualizer = BaseLinkVisualizer(link_id="demo_link", num_slots=8)
    
    # 模拟动态数据更新
    def update_demo_data():
        for cycle in range(20):
            # 生成随机slot数据
            slots_data = create_demo_slot_data(8)
            
            # 随机调整一些参数
            for channel_data in slots_data.values():
                for slot in channel_data:
                    if slot.state == SlotState.OCCUPIED:
                        # 随机分配优先级
                        slot.priority = random.choice(['T0', 'T1', 'T2'])
                        # 随机设置Tag
                        slot.itag = random.random() < 0.15
                        slot.etag = random.random() < 0.08
            
            visualizer.update_slots(slots_data)
            
            # 更新统计信息
            stats = LinkStats(
                bandwidth_utilization=0.3 + 0.4 * np.sin(cycle * 0.3),
                average_latency=10 + 5 * np.sin(cycle * 0.2),
                congestion_level=0.1 + 0.2 * np.sin(cycle * 0.4),
                itag_triggers=random.randint(0, 5),
                etag_upgrades=random.randint(0, 3),
                total_flits=cycle * 8 + random.randint(0, 10)
            )
            visualizer.update_statistics(stats)
            
            # 渲染拥塞热力图
            congestion_data = {
                'req': 0.2 + 0.3 * np.sin(cycle * 0.1),
                'rsp': 0.1 + 0.2 * np.sin(cycle * 0.15), 
                'data': 0.3 + 0.4 * np.sin(cycle * 0.12)
            }
            visualizer.render_congestion_heatmap(congestion_data)
            
            plt.pause(0.5)  # 暂停0.5秒
            
            if cycle % 5 == 0:
                print(f"  周期 {cycle}: 带宽利用率 {stats.bandwidth_utilization:.2%}, "
                     f"延迟 {stats.average_latency:.1f}, I-Tag触发 {stats.itag_triggers}")
    
    print("\n💡 演示说明:")
    print("- 不同颜色的slot表示不同优先级 (T0=红色, T1=橙色, T2=蓝色)")
    print("- 粗边框表示E-Tag高优先级flit")
    print("- 黄色边框表示I-Tag预留slot") 
    print("- 右侧显示实时统计信息和拥塞热力图")
    print("- 点击slot可查看详细flit信息")
    
    try:
        update_demo_data()
    except KeyboardInterrupt:
        print("\n演示结束")
    
    plt.show()


def demo_node_visualizer():
    """演示CrossRing节点可视化器"""
    print("\n" + "="*60)
    print("🎯 CrossRing节点可视化器演示")
    print("="*60)
    print("展示CrossRing特定的节点内部结构可视化")
    
    from types import SimpleNamespace
    
    # 创建配置
    config = SimpleNamespace(
        NUM_COL=3, NUM_ROW=2,
        IQ_OUT_FIFO_DEPTH=8, EQ_IN_FIFO_DEPTH=8,
        RB_IN_FIFO_DEPTH=4, RB_OUT_FIFO_DEPTH=4,
        IQ_CH_FIFO_DEPTH=4, EQ_CH_FIFO_DEPTH=4,
        CH_NAME_LIST=['gdma', 'ddr', 'l2m']
    )
    
    # 创建节点可视化器
    visualizer = CrossRingNodeVisualizer(config, node_id=0)
    
    print("\n💡 演示说明:")
    print("- Inject Queue: 显示各通道的注入队列状态")
    print("- Eject Queue: 显示各通道的提取队列状态")
    print("- Ring Bridge: 显示环形桥接FIFO状态")
    print("- CrossPoint: 显示水平/垂直CrossPoint状态")
    print("- 不同颜色表示不同的packet_id")
    print("- 点击flit查看详细信息")
    
    # 模拟动态更新
    def update_node_demo():
        for cycle in range(15):
            # 生成演示数据
            node_data = create_demo_node_data()
            
            # 随机调整数据
            for queue_type in ['inject_queues', 'eject_queues', 'ring_bridge']:
                for lane, flits in node_data[queue_type].items():
                    # 随机移除一些flit模拟传输
                    if random.random() < 0.3 and flits:
                        flits.pop(0)
                    
                    # 随机添加新flit
                    if random.random() < 0.4:
                        new_flit = FlitProxy(
                            packet_id=f"P{random.randint(1, 6)}",
                            flit_id=f"F{random.randint(0, 3)}",
                            etag_priority=random.choice(['T0', 'T1', 'T2']),
                            itag_h=random.random() < 0.1,
                            itag_v=random.random() < 0.1
                        )
                        flits.append(new_flit)
            
            # 更新CrossPoint状态
            for cp_data in node_data['crosspoints'].values():
                cp_data.arbitration_state = random.choice(['idle', 'active', 'blocked'])
                if cp_data.arbitration_state == 'active':
                    cp_data.active_connections = [('input', 'output')]
                else:
                    cp_data.active_connections = []
            
            visualizer.update_node_state(node_data)
            plt.pause(0.8)
            
            if cycle % 3 == 0:
                iq_count = sum(len(flits) for flits in node_data['inject_queues'].values())
                eq_count = sum(len(flits) for flits in node_data['eject_queues'].values())
                print(f"  周期 {cycle}: 注入队列 {iq_count} flits, 提取队列 {eq_count} flits")
    
    try:
        update_node_demo()
    except KeyboardInterrupt:
        print("\n演示结束")
    
    plt.show()


def demo_realtime_visualizer():
    """演示实时可视化器"""
    print("\n" + "="*60)
    print("⚡ 实时可视化器演示")
    print("="*60)
    print("展示完整的实时可视化系统，集成Link和Node可视化")
    
    from types import SimpleNamespace
    
    # 创建配置
    config = SimpleNamespace(
        num_nodes=4, num_row=2, num_col=2,
        IQ_OUT_FIFO_DEPTH=8, EQ_IN_FIFO_DEPTH=8,
        RB_IN_FIFO_DEPTH=4, RB_OUT_FIFO_DEPTH=4,
        IQ_CH_FIFO_DEPTH=4, EQ_CH_FIFO_DEPTH=4,
        CH_NAME_LIST=['gdma', 'ddr', 'l2m']
    )
    
    # 创建实时可视化器
    visualizer = RealtimeVisualizer(config, update_interval=0.5)
    
    print("\n💡 演示说明:")
    print("- 左上方: 4个节点的内部结构可视化")
    print("- 右上方: 主要链路状态和性能监控图表")
    print("- 下方: 播放控制面板 (播放/暂停/单步/重置)")
    print("- 速度滑块: 调整播放速度")
    print("- 包追踪: 开启后可点击flit进行高亮追踪")
    print("\n🎮 控制说明:")
    print("- 点击'播放'开始自动演示")
    print("- 点击'单步'手动推进一帧")
    print("- 拖动速度滑块调整播放速度")
    print("- 勾选'包追踪'后点击任意flit进行高亮")
    
    # 创建模拟模型进行演示
    class MockModel:
        def __init__(self):
            self.cycle = 0
            self.nodes = {i: self._create_mock_node(i) for i in range(4)}
            self.links = {"link_0": self._create_mock_link()}
        
        def _create_mock_node(self, node_id):
            node = SimpleNamespace()
            node.node_id = node_id
            node.inject_direction_fifos = {}
            node.ip_eject_channel_buffers = {}
            node.horizontal_crosspoint = SimpleNamespace(state='idle')
            node.vertical_crosspoint = SimpleNamespace(state='idle')
            return node
        
        def _create_mock_link(self):
            link = SimpleNamespace()
            link.ring_slices = []
            return link
        
        def step(self):
            self.cycle += 1
            # 模拟一些状态变化
            time.sleep(0.1)  # 模拟计算时间
        
        def get_statistics(self):
            return {
                'bandwidth_utilization': 0.3 + 0.3 * np.sin(self.cycle * 0.2),
                'average_latency': 15 + 5 * np.sin(self.cycle * 0.15),
                'congestion_level': 0.1 + 0.2 * np.sin(self.cycle * 0.3),
                'total_flits': self.cycle * 12
            }
    
    # 设置模拟模型
    mock_model = MockModel()
    visualizer.set_model(mock_model)
    
    print("\n🚀 启动实时可视化...")
    visualizer.start_visualization()


def demo_integration_with_crossring():
    """演示与真实CrossRing模型的集成"""
    print("\n" + "="*60)
    print("🔄 CrossRing模型集成演示")
    print("="*60)
    print("展示可视化系统与真实CrossRing模型的集成")
    
    try:
        # 创建CrossRing配置
        config = CrossRingConfig(num_row=2, num_col=2, config_name="visualization_demo")
        config.num_nodes = 4
        
        # 配置IP接口
        all_nodes = list(range(4))
        config.gdma_send_position_list = all_nodes
        config.ddr_send_position_list = all_nodes
        config.l2m_send_position_list = all_nodes
        
        print("✅ CrossRing配置创建成功")
        
        # 创建traffic文件
        traffic_file = Path(__file__).parent.parent.parent / "traffic_data" / "crossring_traffic.txt"
        if not traffic_file.exists():
            print("❌ Traffic文件不存在，请确保运行过simple_crossring_demo")
            return
        
        # 创建CrossRing模型
        model = CrossRingModel(config, traffic_file_path=str(traffic_file))
        print("✅ CrossRing模型创建成功")
        
        # 注入流量
        injected = model.inject_from_traffic_file(traffic_file_path=str(traffic_file), cycle_accurate=True)
        print(f"✅ 成功注入 {injected} 个请求")
        
        # 创建可视化器
        visualizer = RealtimeVisualizer(config, model, update_interval=0.2)
        
        print("\n💡 集成演示说明:")
        print("- 可视化器已连接到真实的CrossRing模型")
        print("- 显示真实的流量注入和传输过程")
        print("- 所有FIFO、CrossPoint状态都是实时的")
        print("- 性能指标来自真实的仿真数据")
        print("\n🎮 操作提示:")
        print("- 点击'播放'观察真实流量传输")
        print("- 开启'包追踪'可跟踪特定包的路径")
        print("- 观察性能监控图表的实时变化")
        
        print("\n🚀 启动集成可视化...")
        visualizer.start_visualization()
        
    except Exception as e:
        print(f"❌ 集成演示失败: {e}")
        import traceback
        traceback.print_exc()


def run_all_demos():
    """运行所有演示"""
    print("🎪 CrossRing可视化系统完整演示")
    print("=" * 60)
    print("将依次运行所有演示，请按任意键继续...")
    
    demos = [
        ("Link可视化器", demo_link_visualizer),
        ("节点可视化器", demo_node_visualizer), 
        ("实时可视化器", demo_realtime_visualizer),
        ("模型集成", demo_integration_with_crossring)
    ]
    
    for name, demo_func in demos:
        print(f"\n🎬 即将运行: {name}")
        input("按Enter键开始...")
        try:
            demo_func()
        except Exception as e:
            print(f"❌ {name}演示出错: {e}")
        
        print(f"\n✅ {name}演示结束")
        input("按Enter键继续下一个演示...")
    
    print("\n🎉 所有演示完成！")


def main():
    """主函数"""
    logger = setup_logging()
    
    # 解析命令行参数
    demo_type = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    print("🚀 CrossRing可视化系统演示")
    print("=" * 60)
    print("基于旧版本Link_State_Visualizer重构的新可视化架构")
    print("支持模块化、可扩展的NoC可视化")
    print()
    
    # 演示映射
    demo_map = {
        "link": demo_link_visualizer,
        "node": demo_node_visualizer,
        "realtime": demo_realtime_visualizer,
        "integration": demo_integration_with_crossring,
        "all": run_all_demos
    }
    
    if demo_type not in demo_map:
        print(f"❌ 未知的演示类型: {demo_type}")
        print(f"可用选项: {', '.join(demo_map.keys())}")
        return 1
    
    try:
        logger.info(f"开始运行演示: {demo_type}")
        demo_map[demo_type]()
        logger.info("演示运行完成")
        return 0
    except KeyboardInterrupt:
        print("\n\n⏸️ 演示被用户中断")
        return 0
    except Exception as e:
        logger.error(f"演示运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())