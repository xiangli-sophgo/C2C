#!/usr/bin/env python3
"""
简单的可视化测试脚本，用于验证可视化组件是否正常工作
"""

import sys
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig

def test_visualization():
    """测试可视化功能"""
    
    # 创建简单配置
    config = CrossRingConfig(num_row=2, num_col=2, config_name="viz_test")
    config.basic_config.NETWORK_FREQUENCY = 2
    config.validate_config()
    
    # 创建模型
    model = CrossRingModel(config)
    
    # 设置traffic
    traffic_file_path = str(Path(__file__).parent / "traffic_data")
    traffic_chains = [["test1.txt"]]
    
    model.setup_traffic_scheduler(traffic_file_path=traffic_file_path, traffic_chains=traffic_chains)
    
    # 配置可视化
    print("🔧 配置可视化...")
    model.setup_visualization(enable=1, update_interval=1, start_cycle=5)
    
    # 运行短时间仿真
    print("▶️  开始仿真...")
    try:
        model.run_simulation(max_time_ns=30.0, progress_interval_ns=100.0, results_analysis=0, verbose=1)
    except KeyboardInterrupt:
        print("\n⚠️  用户中断仿真")
    except Exception as e:
        print(f"❌ 仿真错误: {e}")
    finally:
        # 关闭可视化
        model.close_visualization()
        print("✅ 测试完成")

if __name__ == "__main__":
    test_visualization()