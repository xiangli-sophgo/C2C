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


def setup_logging(level=logging.INFO):
    """设置简单的日志配置"""
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")
    return logging.getLogger(__name__)


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


def run_crossring_simulation(rows=2, cols=3, max_cycles=1000, max_requests=10):
    """运行CrossRing仿真 - 极简版本"""
    logger = setup_logging()
    
    print(f"📡 CrossRing仿真开始: {rows}×{cols} 网格, 最大{max_cycles}周期")
    
    try:
        # 1. 创建配置和模型
        traffic_file = Path(__file__).parent.parent.parent / "traffic_data" / "crossring_traffic.txt"
        if not traffic_file.exists():
            print(f"❌ Traffic文件不存在: {traffic_file}")
            return False
            
        config = create_config(rows, cols)
        model = CrossRingModel(config, traffic_file_path=str(traffic_file))
        
        # 2. 注入流量并运行仿真
        injected = model.inject_from_traffic_file(
            traffic_file_path=str(traffic_file),
            max_requests=max_requests,
            cycle_accurate=True  # 使用周期精确模式
        )
        
        if injected == 0:
            print("❌ 没有成功注入任何请求")
            return False
            
        print(f"✅ 成功注入 {injected} 个请求")
        
        # 3. 运行仿真
        results = model.run_simulation(
            max_cycles=max_cycles,
            warmup_cycles=0,
            stats_start_cycle=0
        )
        
        if not results:
            print("❌ 仿真失败")
            return False
            
        # 4. 分析并显示结果
        analysis = model.analyze_simulation_results(results)
        
        print("\n" + "=" * 50)
        print("📊 仿真结果")
        print("=" * 50)
        print(f"总周期: {results.get('total_cycles', 0)}")
        print(f"总请求: {results.get('total_requests', 0)}")
        print(f"完成请求: {results.get('completed_requests', 0)}")
        print(f"平均延迟: {analysis.get('avg_latency', 0):.1f} 周期")
        print(f"最大延迟: {analysis.get('max_latency', 0)} 周期")
        print(f"网络利用率: {analysis.get('network_utilization', 0):.1f}%")
        
        if analysis.get('completion_rate', 0) < 100:
            print(f"⚠️  完成率: {analysis.get('completion_rate', 0):.1f}%")
        else:
            print("✅ 所有请求完成")
            
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ 仿真异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'model' in locals():
            model.cleanup()


def main():
    """主函数"""
    # 解析命令行参数
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    max_requests = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    print("=" * 60)
    print("🚀 CrossRing NoC 仿真演示")
    print("=" * 60)
    print("只需几行代码即可完成完整的NoC仿真！")
    print()
    
    # 核心代码示例
    print("💡 核心代码示例:")
    print("```python")
    print("config = create_config(rows, cols)")
    print("model = CrossRingModel(config, traffic_file_path)")
    print("model.inject_from_traffic_file(traffic_file_path)")
    print("results = model.run_simulation(max_cycles)")
    print("analysis = model.analyze_simulation_results(results)")
    print("```")
    print()
    
    # 运行仿真
    success = run_crossring_simulation(rows, cols, max_cycles, max_requests)
    
    if success:
        print("\n✅ 演示成功完成！")
        print("\n📋 演示功能:")
        print("- ✅ 参数化配置 (支持命令行参数)")
        print("- ✅ 优化的IP接口创建")
        print("- ✅ 周期精确的流量注入")
        print("- ✅ 完整的仿真执行")
        print("- ✅ 自动化结果分析")
        print(f"\n🎯 使用方法: python {Path(__file__).name} [行数] [列数] [最大周期] [最大请求]")
        print(f"📝 当前参数: {rows}×{cols} 网格, {max_cycles}周期, {max_requests}请求")
        return 0
    else:
        print("\n❌ 演示失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
