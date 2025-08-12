#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

from src.noc.crossring.model import CrossRingModel
from src.noc.configs.crossring.config import create_3x3_config, create_5x2_config, create_5x4_config


def main():
    """运行CrossRing仿真 - 使用新的简化接口"""

    # 1. 设置traffic文件
    traffic_file_path = str(Path(__file__).parent.parent / "traffic_data")
    traffic_chains = [
        [
            "LLama2_AllReduce.txt",
            # "test1.txt",
            # "R_5x2.txt",
        ]
    ]

    # 2. 创建模型
    # config = create_3x3_config()
    # config = create_5x2_config()
    config = create_5x4_config()
    model = CrossRingModel(config)
    np.random.seed(811)

    save_dir = None
    save_dir = f"../output/noc/CrossRing/{config.NUM_COL}x{config.NUM_ROW}/"

    # 3. 配置各种选项
    model.setup_traffic_scheduler(traffic_file_path=traffic_file_path, traffic_chains=traffic_chains)
    # model.setup_debug(trace_packets=[11], update_interval=0.0)
    # model.setup_visualization(enable=True, update_interval=0.8, start_cycle=120)

    model.setup_result_analysis(
        # 图片生成控制
        flow_distribution_fig=0,
        bandwidth_analysis_fig=0,
        latency_analysis_fig=0,
        save_figures=0,
        # CSV文件导出控制
        export_request_csv=1,
        export_fifo_csv=1,
        export_link_csv=1,
        # 通用设置
        save_dir=save_dir,
    )

    # 4. 运行仿真 - 减小仿真时间进行调试
    print("▶️  开始仿真")
    model.run_simulation(max_time_ns=5000.0, progress_interval_ns=1000.0, results_analysis=True, verbose=1)


if __name__ == "__main__":
    main()
    # 保持程序运行，让matplotlib图表持续显示
    try:
        import matplotlib.pyplot as plt

        if plt.get_fignums():  # 如果有打开的图形
            print("图表已显示，按Ctrl+C退出程序")
            plt.show(block=True)  # 阻塞显示，直到用户关闭所有图形窗口
    except KeyboardInterrupt:
        print("\n程序已退出")
    except Exception as e:
        print(f"显示图表时出错: {e}")
        pass
