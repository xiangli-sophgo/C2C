#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig


def create_config(rows=2, cols=3, config_name="demo"):
    """创建CrossRing配置"""
    config = CrossRingConfig(num_row=rows, num_col=cols, config_name=config_name)
    config.basic_config.NETWORK_FREQUENCY = 2

    # config.ip_config.DDR_BW_LIMIT = 20
    # config.ip_config.GDMA_BW_LIMIT = 20

    config.tracker_config.RN_R_TRACKER_OSTD = 128
    config.tracker_config.RN_W_TRACKER_OSTD = 32
    config.tracker_config.SN_DDR_R_TRACKER_OSTD = 32
    config.tracker_config.SN_DDR_W_TRACKER_OSTD = 16
    config.tracker_config.SN_L2M_R_TRACKER_OSTD = 64
    config.tracker_config.SN_L2M_W_TRACKER_OSTD = 64
    config.tracker_config.SN_TRACKER_RELEASE_LATENCY = 40

    config.latency_config.DDR_R_LATENCY = 155
    config.latency_config.DDR_W_LATENCY = 0
    config.latency_config.L2M_R_LATENCY = 12
    config.latency_config.L2M_W_LATENCY = 16

    config.fifo_config.IQ_CH_DEPTH = 8
    config.fifo_config.EQ_CH_DEPTH = 8
    config.fifo_config.IQ_OUT_FIFO_DEPTH = 8
    config.fifo_config.RB_OUT_FIFO_DEPTH = 8  # 增加Ring Bridge输出FIFO深度
    config.fifo_config.RB_IN_FIFO_DEPTH = 16
    config.fifo_config.EQ_IN_FIFO_DEPTH = 16

    config.tag_config.TL_ETAG_T2_UE_MAX = 8
    config.tag_config.TL_ETAG_T1_UE_MAX = 12
    config.tag_config.TR_ETAG_T2_UE_MAX = 12
    config.tag_config.TU_ETAG_T2_UE_MAX = 8
    config.tag_config.TU_ETAG_T1_UE_MAX = 12
    config.tag_config.TD_ETAG_T2_UE_MAX = 12
    config.validate_config()

    return config


def main():
    """运行CrossRing仿真 - 使用新的简化接口"""

    # 1. 设置traffic文件
    traffic_file_path = str(Path(__file__).parent.parent.parent / "traffic_data")
    traffic_chains = [
        [
            # "LLama2_AllReduce.txt",
            "test1.txt",
        ]
    ]

    # 2. 创建模型（不传递traffic_file_path，IP接口将在setup_traffic_scheduler中动态创建）
    rows = 3
    cols = 3
    config = create_config(rows, cols)
    model = CrossRingModel(config)

    save_dir = None
    # save_dir = f"../../output/noc/CrossRing/{rows}x{cols}/"

    # 3. 配置各种选项
    model.setup_traffic_scheduler(traffic_file_path=traffic_file_path, traffic_chains=traffic_chains)  # Traffic文件设置，节点的IP会根据数据流连接。
    # model.setup_debug(trace_packets=[1, 2], sleep_time=0.0)
    model.setup_result_analysis(flow_distribution=1, bandwidth_analysis=1, latency_analysis=1, save_figures=0, save_dir=save_dir)  # 可视化设置

    # 4. 配置实时可视化（可选）
    model.setup_visualization(enable=1, update_interval=1, start_cycle=0)

    # 5. 运行仿真 - 延长时间以观察flit流动
    print("▶️  开始仿真")
    model.run_simulation(max_time_ns=2000.0, progress_interval_ns=1000.0, results_analysis=0, verbose=1)


if __name__ == "__main__":
    sys.exit(main())
