#!/usr/bin/env python3
"""
简单测试CrossRing可视化器的基本功能
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# 添加src路径
sys.path.insert(0, str(Path.cwd()))

def test_visualizer_creation():
    """测试可视化器创建"""
    print("🧪 测试可视化器创建...")
    
    try:
        from src.noc.visualization.crossring_link_state_visualizer import CrossRingLinkStateVisualizer, _FlitProxy
        print("✅ 导入成功")
        
        # 创建配置
        config = SimpleNamespace(
            NUM_ROW=2, NUM_COL=2,
            IQ_OUT_FIFO_DEPTH=8, EQ_IN_FIFO_DEPTH=8,
            RB_IN_FIFO_DEPTH=4, RB_OUT_FIFO_DEPTH=4,
            IQ_CH_FIFO_DEPTH=4, EQ_CH_FIFO_DEPTH=4,
            SLICE_PER_LINK=8
        )
        
        # 创建演示网络
        network = SimpleNamespace()
        network.nodes = {}
        network.links = {}
        
        # 创建可视化器
        visualizer = CrossRingLinkStateVisualizer(config, network)
        print("✅ 可视化器创建成功")
        
        # 测试基本方法
        selected_node = visualizer.get_selected_node()
        print(f"✅ 当前选中节点: {selected_node}")
        
        # 创建FlitProxy测试
        flit = _FlitProxy(pid=1, fid="F0", etag="T1", ih=False, iv=True)
        print(f"✅ FlitProxy创建成功: {flit}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_piece_visualizer():
    """测试PieceVisualizer"""
    print("\n🧪 测试PieceVisualizer...")
    
    try:
        from src.noc.visualization.crossring_link_state_visualizer import CrossRingLinkStateVisualizer
        import matplotlib.pyplot as plt
        
        # 创建配置
        config = SimpleNamespace(
            NUM_ROW=2, NUM_COL=2,
            IQ_OUT_FIFO_DEPTH=8, EQ_IN_FIFO_DEPTH=8,
            RB_IN_FIFO_DEPTH=4, RB_OUT_FIFO_DEPTH=4,
            IQ_CH_FIFO_DEPTH=4, EQ_CH_FIFO_DEPTH=4,
            SLICE_PER_LINK=8
        )
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 创建PieceVisualizer
        piece_vis = CrossRingLinkStateVisualizer.PieceVisualizer(config, ax)
        print("✅ PieceVisualizer创建成功")
        
        # 测试数据适配器方法
        dummy_network = SimpleNamespace()
        dummy_network.nodes = {}
        
        inject_data = piece_vis._get_inject_queues_data(dummy_network, 0)
        print(f"✅ inject_queues_data: {inject_data}")
        
        eject_data = piece_vis._get_eject_queues_data(dummy_network, 0)
        print(f"✅ eject_queues_data: {eject_data}")
        
        rb_data = piece_vis._get_ring_bridge_data(dummy_network, 0)
        print(f"✅ ring_bridge_data: {rb_data}")
        
        plt.close(fig)  # 关闭图形避免显示
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_structures():
    """测试数据结构兼容性"""
    print("\n🧪 测试数据结构兼容性...")
    
    try:
        # 模拟CrossRing节点结构
        node = SimpleNamespace()
        
        # inject_direction_fifos
        node.inject_direction_fifos = {}
        for direction in ['TL', 'TR', 'TU', 'TD']:
            fifo = SimpleNamespace()
            fifo.queue = []
            # 添加一个测试flit
            from src.noc.visualization.crossring_link_state_visualizer import _FlitProxy
            test_flit = _FlitProxy(pid=1, fid="F0", etag="T1", ih=False, iv=False)
            fifo.queue.append(test_flit)
            node.inject_direction_fifos[direction] = fifo
        
        print("✅ inject_direction_fifos结构创建成功")
        print(f"   TL队列长度: {len(node.inject_direction_fifos['TL'].queue)}")
        
        # channel_buffer
        node.channel_buffer = {}
        for channel in ['gdma', 'ddr']:
            buffer = SimpleNamespace()
            buffer.queue = []
            node.channel_buffer[channel] = buffer
        
        print("✅ channel_buffer结构创建成功")
        
        # ring_bridge
        node.ring_bridge = SimpleNamespace()
        node.ring_bridge.ring_bridge_input = {}
        node.ring_bridge.ring_bridge_output = {}
        
        print("✅ ring_bridge结构创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🎪 CrossRing Link State Visualizer 测试")
    print("=" * 50)
    
    tests = [
        test_visualizer_creation,
        test_piece_visualizer, 
        test_data_structures
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! CrossRing可视化器基本功能正常")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步调试")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)