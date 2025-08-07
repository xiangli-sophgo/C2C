#!/usr/bin/env python3
import sys
from pathlib import Path

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig


def create_3x3_config():
    config = CrossRingConfig(num_row=3, num_col=3)
    config.basic_config.NETWORK_FREQUENCY = 2

    # config.ip_config.DDR_BW_LIMIT = 64
    # config.ip_config.GDMA_BW_LIMIT = 20
    config.ip_config.IP_H2L_H_FIFO_DEPTH = 4
    config.ip_config.IP_H2L_L_FIFO_DEPTH = 4

    config.tracker_config.RN_R_TRACKER_OSTD = 128
    config.tracker_config.RN_W_TRACKER_OSTD = 32
    config.tracker_config.SN_DDR_R_TRACKER_OSTD = 32
    config.tracker_config.SN_DDR_W_TRACKER_OSTD = 16
    config.tracker_config.SN_L2M_R_TRACKER_OSTD = 64
    config.tracker_config.SN_L2M_W_TRACKER_OSTD = 64
    config.tracker_config.SN_TRACKER_RELEASE_LATENCY = 40

    config.latency_config.DDR_R_LATENCY = 155
    config.latency_config.DDR_W_LATENCY = 16
    config.latency_config.L2M_R_LATENCY = 12
    config.latency_config.L2M_W_LATENCY = 16

    config.fifo_config.IQ_CH_DEPTH = 8
    config.fifo_config.EQ_CH_DEPTH = 8
    config.fifo_config.IQ_OUT_FIFO_DEPTH = 8
    config.fifo_config.RB_OUT_FIFO_DEPTH = 8
    config.fifo_config.RB_IN_FIFO_DEPTH = 16
    config.fifo_config.EQ_IN_FIFO_DEPTH = 16

    config.tag_config.TL_ETAG_T2_UE_MAX = 8
    config.tag_config.TL_ETAG_T1_UE_MAX = 12
    config.tag_config.TR_ETAG_T2_UE_MAX = 12
    config.tag_config.TU_ETAG_T2_UE_MAX = 8
    config.tag_config.TU_ETAG_T1_UE_MAX = 12
    config.tag_config.TD_ETAG_T2_UE_MAX = 12
    config.tag_config.ETAG_BOTHSIDE_UPGRADE = 0
    config.validate_config()

    return config


def create_5x2_config():
    config = CrossRingConfig(num_row=5, num_col=2)
    config.basic_config.NETWORK_FREQUENCY = 2

    # config.ip_config.DDR_BW_LIMIT = 64
    # config.ip_config.GDMA_BW_LIMIT = 20

    config.tracker_config.RN_R_TRACKER_OSTD = 128
    config.tracker_config.RN_W_TRACKER_OSTD = 32
    config.tracker_config.SN_DDR_R_TRACKER_OSTD = 32
    config.tracker_config.SN_DDR_W_TRACKER_OSTD = 16
    config.tracker_config.SN_L2M_R_TRACKER_OSTD = 64
    config.tracker_config.SN_L2M_W_TRACKER_OSTD = 64
    config.tracker_config.SN_TRACKER_RELEASE_LATENCY = 40

    config.latency_config.DDR_R_LATENCY = 100
    config.latency_config.DDR_W_LATENCY = 40
    config.latency_config.L2M_R_LATENCY = 12
    config.latency_config.L2M_W_LATENCY = 16

    config.fifo_config.IQ_CH_DEPTH = 10
    config.fifo_config.EQ_CH_DEPTH = 10
    config.fifo_config.IQ_OUT_FIFO_DEPTH = 8
    config.fifo_config.RB_OUT_FIFO_DEPTH = 8
    config.fifo_config.RB_IN_FIFO_DEPTH = 16
    config.fifo_config.EQ_IN_FIFO_DEPTH = 16

    config.tag_config.TL_ETAG_T2_UE_MAX = 8
    config.tag_config.TL_ETAG_T1_UE_MAX = 15
    config.tag_config.TR_ETAG_T2_UE_MAX = 12
    config.tag_config.TU_ETAG_T2_UE_MAX = 8
    config.tag_config.TU_ETAG_T1_UE_MAX = 15
    config.tag_config.TD_ETAG_T2_UE_MAX = 12
    config.validate_config()

    return config


def create_5x4_config():
    config = CrossRingConfig(num_row=5, num_col=4)
    config.basic_config.NETWORK_FREQUENCY = 2

    config.tracker_config.RN_R_TRACKER_OSTD = 64
    config.tracker_config.RN_W_TRACKER_OSTD = 64
    config.tracker_config.SN_DDR_R_TRACKER_OSTD = 64
    config.tracker_config.SN_DDR_W_TRACKER_OSTD = 64
    config.tracker_config.SN_L2M_R_TRACKER_OSTD = 64
    config.tracker_config.SN_L2M_W_TRACKER_OSTD = 64
    config.tracker_config.SN_TRACKER_RELEASE_LATENCY = 40

    config.latency_config.DDR_R_LATENCY = 40
    config.latency_config.DDR_W_LATENCY = 0
    config.latency_config.L2M_R_LATENCY = 12
    config.latency_config.L2M_W_LATENCY = 16

    config.fifo_config.IQ_CH_DEPTH = 4
    config.fifo_config.EQ_CH_DEPTH = 4
    config.fifo_config.IQ_OUT_FIFO_DEPTH = 8
    config.fifo_config.RB_OUT_FIFO_DEPTH = 8
    config.fifo_config.RB_IN_FIFO_DEPTH = 16
    config.fifo_config.EQ_IN_FIFO_DEPTH = 16

    config.tag_config.TL_ETAG_T2_UE_MAX = 8
    config.tag_config.TL_ETAG_T1_UE_MAX = 15
    config.tag_config.TR_ETAG_T2_UE_MAX = 12
    config.tag_config.TU_ETAG_T2_UE_MAX = 8
    config.tag_config.TU_ETAG_T1_UE_MAX = 15
    config.tag_config.TD_ETAG_T2_UE_MAX = 12
    config.tag_config.ITAG_TRIGGER_TH_H = 80
    config.tag_config.ITAG_TRIGGER_TH_V = 80
    config.tag_config.ITAG_MAX_NUM_H = 1
    config.tag_config.ITAG_MAX_NUM_V = 1
    config.validate_config()

    return config


def main():
    """运行CrossRing仿真 - 使用新的简化接口"""

    # 1. 设置traffic文件
    traffic_file_path = str(Path(__file__).parent.parent / "traffic_data")
    traffic_chains = [
        [
            # "LLama2_AllReduce.txt",
            "test1.txt",
            # "R_5x2.txt",
        ]
    ]

    # 2. 创建模型
    config = create_3x3_config()
    # config = create_5x2_config()
    # config = create_5x4_config()

    model = CrossRingModel(config)

    save_dir = None
    save_dir = f"../output/noc/CrossRing/{config.NUM_COL}x{config.NUM_ROW}/"

    # 3. 配置各种选项
    model.setup_traffic_scheduler(traffic_file_path=traffic_file_path, traffic_chains=traffic_chains)
    model.setup_debug(trace_packets=[2], update_interval=0.0)

    model.setup_visualization(enable=True, update_interval=0.3, start_cycle=50, gpu_accelerated=True)

    model.setup_result_analysis(
        # 图片生成控制
        flow_distribution_fig=0,
        bandwidth_analysis_fig=0,
        latency_analysis_fig=0,
        save_figures=0,
        # CSV文件导出控制
        export_request_csv=1,
        export_fifo_csv=1,
        export_link_csv=0,
        # 通用设置
        save_dir=save_dir,
    )

    # 4. 运行仿真 - 减小仿真时间进行调试
    print("▶️  开始仿真")
    model.run_simulation(max_time_ns=12000.0, progress_interval_ns=1000.0, results_analysis=True, verbose=1)


if __name__ == "__main__":
    main()

    # 保持程序运行，让可视化图表持续显示
    try:
        import matplotlib.pyplot as plt

        if plt.get_fignums():  # 如果有打开的图形窗口
            print("📊 可视化图表已显示，按Ctrl+C退出程序")
            print("💡 提示：如果启用了GPU加速，可视化会更流畅")
            plt.show(block=True)  # 阻塞显示，直到用户关闭所有图形窗口
    except KeyboardInterrupt:
        print("\n🛑 程序已退出")
    except Exception as e:
        print(f"❌ 显示图表时出错: {e}")
        pass
