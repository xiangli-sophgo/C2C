#!/usr/bin/env python3
import sys
import logging
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
    config.tracker_config.RN_W_TRACKER_OSTD = 64
    config.tracker_config.SN_DDR_R_TRACKER_OSTD = 2  # 减少资源以触发negative响应
    config.tracker_config.SN_DDR_W_TRACKER_OSTD = 2  # 减少资源以触发negative响应
    config.tracker_config.SN_L2M_R_TRACKER_OSTD = 2
    config.tracker_config.SN_L2M_W_TRACKER_OSTD = 2

    config.latency_config.DDR_R_LATENCY = 0
    config.latency_config.DDR_R_LATENCY_VAR = 0
    config.latency_config.DDR_W_LATENCY = 0
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

    return config


def main():
    """运行CrossRing仿真 - 使用新的简化接口"""

    # 1. 设置traffic文件
    traffic_file_path = r"../../traffic_data"
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

    # 3. 配置各种选项
    model.setup_traffic_scheduler(traffic_file_path=traffic_file_path, traffic_chains=traffic_chains)  # Traffic文件设置，节点的IP会根据数据流连接。
    model.setup_debug(level=2, trace_packets=["1", "2", "3"], sleep_time=0.0)  # debug设置，跟踪多个写请求
    model.setup_result_analysis(flow_distribution=True, bandwidth_analysis=True, save_figures=1, save_dir="../../output/noc/CrossRing/")  # 可视化设置

    # 4. 运行仿真 - 减小仿真时间进行调试
    print("▶️  开始仿真")
    model.run_simulation(max_time_ns=400.0, progress_interval_ns=100.0, results_analysis=True, verbose=1)


if __name__ == "__main__":
    sys.exit(main())
