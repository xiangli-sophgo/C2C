#!/usr/bin/env python3
"""
简化的CrossRing NoC演示
=====================

最简单的CrossRing仿真演示，只需几行代码：
1. 创建CrossRing模型
2. 从traffic文件注入流量
3. 运行仿真
4. 显示结果

Usage:
    python simple_crossring_demo.py [rows] [cols] [max_cycles]
"""

import sys
import logging
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig


def create_config(rows=2, cols=3, config_name="simple_demo"):
    """创建CrossRing配置"""
    config = CrossRingConfig(num_row=rows, num_col=cols, config_name=config_name)

    # 确保num_nodes正确设置
    config.num_nodes = rows * cols

    # 为所有节点配置IP接口
    all_nodes = list(range(rows * cols))
    config.gdma_send_position_list = all_nodes
    config.ddr_send_position_list = all_nodes
    config.l2m_send_position_list = all_nodes

    return config


def run_crossring_simulation(rows=3, cols=3, max_cycles=1000):
    """运行CrossRing仿真 - 简化版本"""

    print(f"📡 CrossRing仿真: {rows}×{cols} 网格, {max_cycles}周期")

    # 1. 读取traffic文件
    traffic_file = Path(__file__).parent.parent.parent / "traffic_data" / "test1.txt"
    if not traffic_file.exists():
        print(f"❌ Traffic文件不存在: {traffic_file}")
        return False

    # 2. 创建模型并设置traffic
    config = create_config(rows, cols)
    model = CrossRingModel(config, str(traffic_file))

    # 设置TrafficScheduler
    model.setup_traffic_scheduler([[traffic_file.name]], str(traffic_file.parent))
    
    # 检查TrafficScheduler状态
    traffic_status = model.get_traffic_status()
    print(f"📊 Traffic状态: {traffic_status}")

    # 3. 运行仿真
    results = model.run_simulation(max_cycles)

    # 3. 分析结果
    if results:
        # 使用model内置的结果分析功能
        analysis = model.analyze_simulation_results(results)
        
        completed = len(model.request_tracker.completed_requests) if hasattr(model, "request_tracker") else 0
        print(f"✅ 仿真完成: 处理了 {completed} 个请求")
        
        # 显示关键指标
        if "带宽指标" in analysis and "总体带宽" in analysis["带宽指标"]:
            bw = analysis["带宽指标"]["总体带宽"]
            print(f"  平均带宽: {bw.get('非加权带宽_GB/s', 'N/A')} GB/s")
        
        if "延迟指标" in analysis and "总体延迟" in analysis["延迟指标"]:
            lat = analysis["延迟指标"]["总体延迟"]
            print(f"  平均延迟: {lat.get('平均延迟_ns', 'N/A')} ns")
            
        return True
    else:
        print("❌ 仿真失败")
        return False


def main():
    """主函数"""
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    print("🚀 CrossRing仿真演示 - 简化版本")
    return 0 if run_crossring_simulation(rows, cols, max_cycles) else 1


if __name__ == "__main__":
    sys.exit(main())
