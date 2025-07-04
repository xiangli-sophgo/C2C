import numpy as np
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
from src.utils.components import *
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from functools import lru_cache
from src.core.result_processor import *
import time, sys
import pandas as pd
import matplotlib

if sys.platform == "darwin":  # macOS 的系统标识是 'darwin'
    matplotlib.use("macosx")  # 仅在 macOS 上使用该后端


@dataclass
class RequestInfo:
    """请求信息数据结构"""

    packet_id: int
    start_time: int  # ns
    end_time: int  # ns (整体网络结束时间)
    rn_end_time: int  # ns (RN端口结束时间)
    sn_end_time: int  # ns (SN端口结束时间)
    req_type: str  # 'read' or 'write'
    source_node: int
    dest_node: int
    source_type: str
    dest_type: str
    burst_length: int
    total_bytes: int
    cmd_latency: int
    data_latency: int
    transaction_latency: int


@dataclass
class WorkingInterval:
    """工作区间数据结构"""

    start_time: int
    end_time: int
    duration: int
    flit_count: int
    total_bytes: int
    request_count: int

    @property
    def bandwidth_bytes_per_ns(self) -> float:
        """区间内平均带宽 (bytes/ns)"""
        return self.total_bytes / self.duration if self.duration > 0 else 0.0


@dataclass
class BandwidthMetrics:
    """带宽指标数据结构"""

    unweighted_bandwidth: float  # GB/s
    weighted_bandwidth: float  # GB/s
    working_intervals: List[WorkingInterval]
    total_working_time: int  # ns
    network_start_time: int  # ns
    network_end_time: int  # ns
    total_bytes: int
    total_requests: int


@dataclass
class PortBandwidthMetrics:
    """端口带宽指标"""

    port_id: str
    read_metrics: BandwidthMetrics
    write_metrics: BandwidthMetrics
    mixed_metrics: BandwidthMetrics  # 新增：混合读写指标


class BandwidthAnalyzer:
    """
    带宽分析器 - 统一的带宽统计分析类

    功能：
    1. 统计工作区间（去除空闲时间段）
    2. 计算非加权和加权带宽
    3. 分别统计读写操作
    4. 统计混合读写操作（不区分读写类型）
    5. 网络整体和RN端口带宽统计
    6. 生成统一报告
    """

    def _calculate_latency_stats(self):
        """计算并返回延迟统计数据字典，过滤掉无穷大"""
        import math

        stats = {
            "cmd": {"read": {"sum": 0, "max": 0, "count": 0}, "write": {"sum": 0, "max": 0, "count": 0}, "mixed": {"sum": 0, "max": 0, "count": 0}},
            "data": {"read": {"sum": 0, "max": 0, "count": 0}, "write": {"sum": 0, "max": 0, "count": 0}, "mixed": {"sum": 0, "max": 0, "count": 0}},
            "trans": {"read": {"sum": 0, "max": 0, "count": 0}, "write": {"sum": 0, "max": 0, "count": 0}, "mixed": {"sum": 0, "max": 0, "count": 0}},
        }
        for r in self.requests:
            # CMD
            if math.isfinite(r.cmd_latency):
                group = stats["cmd"][r.req_type]
                group["sum"] += r.cmd_latency
                group["count"] += 1
                group["max"] = max(group["max"], r.cmd_latency)
                mixed = stats["cmd"]["mixed"]
                mixed["sum"] += r.cmd_latency
                mixed["count"] += 1
                mixed["max"] = max(mixed["max"], r.cmd_latency)
            # Data
            if math.isfinite(r.data_latency):
                group = stats["data"][r.req_type]
                group["sum"] += r.data_latency
                group["count"] += 1
                group["max"] = max(group["max"], r.data_latency)
                mixed = stats["data"]["mixed"]
                mixed["sum"] += r.data_latency
                mixed["count"] += 1
                mixed["max"] = max(mixed["max"], r.data_latency)
            # Transaction
            if math.isfinite(r.transaction_latency):
                group = stats["trans"][r.req_type]
                group["sum"] += r.transaction_latency
                group["count"] += 1
                group["max"] = max(group["max"], r.transaction_latency)
                mixed = stats["trans"]["mixed"]
                mixed["sum"] += r.transaction_latency
                mixed["count"] += 1
                mixed["max"] = max(mixed["max"], r.transaction_latency)
        return stats

    def __init__(
        self,
        config,
        min_gap_threshold: int = 200,
        plot_rn_bw_fig: bool = False,
        plot_flow_graph: bool = False,
    ):
        """
        初始化带宽分析器

        Args:
            config: 网络配置对象
            min_gap_threshold: 工作区间合并阈值(ns)，小于此值的间隔被视为同一工作区间
        """
        self.config = config
        self.min_gap_threshold = min_gap_threshold
        self.network_frequency = config.NETWORK_FREQUENCY  # GHz
        self.plot_rn_bw_fig = plot_rn_bw_fig
        self.plot_flow_graph = plot_flow_graph
        self.finish_cycle = -np.inf

        # 数据存储
        self.requests: List[RequestInfo] = []
        self.rn_positions = set()
        self.sn_positions = set()
        self.rn_bandwidth_time_series = defaultdict(lambda: {"time": [], "start_times": [], "bytes": []})
        self.ip_bandwidth_data = None
        self.read_ip_intervals = defaultdict(list)
        self.write_ip_intervals = defaultdict(list)

        # 初始化节点位置
        self._initialize_node_positions()

    def _initialize_node_positions(self):
        """初始化RN和SN节点位置"""
        if hasattr(self.config, "GDMA_SEND_POSITION_LIST"):
            self.rn_positions.update(self.config.GDMA_SEND_POSITION_LIST)
        if hasattr(self.config, "SDMA_SEND_POSITION_LIST"):
            self.rn_positions.update(self.config.SDMA_SEND_POSITION_LIST)
        if hasattr(self.config, "CDMA_SEND_POSITION_LIST"):
            self.rn_positions.update(self.config.CDMA_SEND_POSITION_LIST)
        if hasattr(self.config, "DDR_SEND_POSITION_LIST"):
            self.sn_positions.update(pos - self.config.NUM_COL for pos in self.config.DDR_SEND_POSITION_LIST)
            self.sn_positions.update(self.config.DDR_SEND_POSITION_LIST)
        if hasattr(self.config, "L2M_SEND_POSITION_LIST"):
            self.sn_positions.update(pos - self.config.NUM_COL for pos in self.config.L2M_SEND_POSITION_LIST)
            self.sn_positions.update(self.config.L2M_SEND_POSITION_LIST)

    def collect_requests_data(self, sim_model) -> None:
        """从sim_model收集请求数据"""
        self.requests.clear()
        self.sim_model = sim_model

        for packet_id, flits in sim_model.data_network.arrive_flits.items():
            if not flits or len(flits) != flits[0].burst_length:
                continue

            representative_flit: Flit = flits[-1]

            # 计算不同角度的结束时间
            network_end_time = representative_flit.data_received_complete_cycle // self.network_frequency

            if representative_flit.req_type == "read":
                # 读请求：RN在收到数据时结束，SN在发出数据时结束
                rn_end_time = representative_flit.data_received_complete_cycle // self.network_frequency  # RN收到数据
                sn_end_time = representative_flit.data_entry_noc_from_cake1_cycle // self.network_frequency  # SN发出数据

                # 读请求：flit的source是SN(DDR/L2M)，destination是RN(SDMA/GDMA/CDMA)
                actual_source_node = representative_flit.destination + (self.config.NUM_COL if not self.sim_model.topo_type_stat.startswith("Ring") else 0)  # 实际发起请求的节点
                actual_dest_node = representative_flit.source - (self.config.NUM_COL if not self.sim_model.topo_type_stat.startswith("Ring") else 0)  # 实际目标节点
                actual_source_type = representative_flit.original_source_type  # 实际发起请求的类型
                actual_dest_type = representative_flit.original_destination_type  # 实际目标类型
            else:  # write
                # 写请求：RN在发出数据时结束，SN在收到数据时结束
                rn_end_time = representative_flit.data_entry_noc_from_cake0_cycle // self.network_frequency  # RN发出数据
                sn_end_time = representative_flit.data_received_complete_cycle // self.network_frequency  # SN收到数据

                # 写请求：flit的source是RN(SDMA/GDMA/CDMA)，destination是SN(DDR/L2M)
                actual_source_node = representative_flit.source  # 实际发起请求的节点
                actual_dest_node = representative_flit.destination  # 实际目标节点
                actual_source_type = representative_flit.original_source_type  # 实际发起请求的类型
                actual_dest_type = representative_flit.original_destination_type  # 实际目标类型

            request_info = RequestInfo(
                packet_id=packet_id,
                start_time=representative_flit.cmd_entry_cake0_cycle // self.network_frequency,
                end_time=network_end_time,  # 整体网络结束时间
                rn_end_time=rn_end_time,
                sn_end_time=sn_end_time,
                req_type=representative_flit.req_type,
                source_node=actual_source_node,  # 使用修正后的源节点
                dest_node=actual_dest_node,  # 使用修正后的目标节点
                source_type=actual_source_type,  # 使用修正后的源类型
                dest_type=actual_dest_type,  # 使用修正后的目标类型
                burst_length=representative_flit.burst_length,
                total_bytes=representative_flit.burst_length * 128,
                cmd_latency=representative_flit.cmd_latency // self.network_frequency,
                data_latency=representative_flit.data_latency // self.network_frequency,
                transaction_latency=representative_flit.transaction_latency // self.network_frequency,
            )

            # 收集RN带宽时间序列数据
            # 使用修正后的类型信息
            port_key = f"{actual_source_type[:-2].upper()} {representative_flit.req_type} {actual_dest_type[:3].upper()}"

            if representative_flit.req_type == "read":
                completion_time = rn_end_time
            else:  # write
                completion_time = rn_end_time

            self.rn_bandwidth_time_series[port_key]["time"].append(completion_time)
            self.rn_bandwidth_time_series[port_key]["start_times"].append(request_info.start_time)
            self.rn_bandwidth_time_series[port_key]["bytes"].append(representative_flit.burst_length * 128)

            # 更新finish_cycle
            if representative_flit.data_received_complete_cycle != float("inf"):
                self.finish_cycle = max(self.finish_cycle, representative_flit.data_received_complete_cycle)

            self.requests.append(request_info)

        # 按开始时间排序
        self.requests.sort(key=lambda x: x.start_time)

    def calculate_ip_bandwidth_data(self):
        """计算IP带宽数据矩阵"""
        rows = self.config.NUM_ROW
        cols = self.config.NUM_COL
        if getattr(self.sim_model, "topo_type_stat", None).startswith("Ring"):
            rows = self.config.RING_NUM_NODE // 2
            cols = 2
        elif getattr(self.sim_model, "topo_type_stat", None) != "4x5":
            rows -= 1

        # 初始化数据结构
        self.ip_bandwidth_data = {
            "read": {
                "sdma": np.zeros((rows, cols)),
                "gdma": np.zeros((rows, cols)),
                "cdma": np.zeros((rows, cols)),
                "ddr": np.zeros((rows, cols)),
                "l2m": np.zeros((rows, cols)),
            },
            "write": {
                "sdma": np.zeros((rows, cols)),
                "gdma": np.zeros((rows, cols)),
                "cdma": np.zeros((rows, cols)),
                "ddr": np.zeros((rows, cols)),
                "l2m": np.zeros((rows, cols)),
            },
            "total": {
                "sdma": np.zeros((rows, cols)),
                "gdma": np.zeros((rows, cols)),
                "cdma": np.zeros((rows, cols)),
                "ddr": np.zeros((rows, cols)),
                "l2m": np.zeros((rows, cols)),
            },
        }

        # 处理RN端口（按照source_node分组，避免重复计算）
        rn_requests_by_source = defaultdict(list)

        for req in self.requests:
            if req.source_node in self.rn_positions:
                rn_requests_by_source[req.source_node].append(req)

        # 计算每个RN源节点的带宽
        for source_node, node_requests in rn_requests_by_source.items():
            # 按source_type分组
            by_type = defaultdict(list)
            for req in node_requests:
                # 提取source_type的前缀（去掉_ip后缀）
                source_type = req.source_type.lower()[:-2]
                by_type[source_type].append(req)

            # 计算物理位置
            if getattr(self.sim_model, "topo_type_stat", None).startswith("Ring"):
                if source_node < rows:
                    physical_col = 0
                    physical_row = source_node
                else:
                    physical_col = 1
                    physical_row = self.config.RING_NUM_NODE - 1 - source_node
            else:
                physical_col = source_node % cols
                physical_row = source_node // cols // 2

            # 为每种类型计算带宽
            for source_type, type_requests in by_type.items():
                # 分别计算读写带宽
                read_requests = [req for req in type_requests if req.req_type == "read"]
                write_requests = [req for req in type_requests if req.req_type == "write"]

                # 计算读带宽
                if read_requests:
                    read_metrics = self.calculate_bandwidth_metrics(read_requests, "read")
                    self.ip_bandwidth_data["read"][source_type][physical_row, physical_col] = read_metrics.weighted_bandwidth

                # 计算写带宽
                if write_requests:
                    write_metrics = self.calculate_bandwidth_metrics(write_requests, "write")
                    self.ip_bandwidth_data["write"][source_type][physical_row, physical_col] = write_metrics.weighted_bandwidth

                # 计算总带宽
                if type_requests:
                    total_metrics = self.calculate_bandwidth_metrics(type_requests, operation_type=None)
                    self.ip_bandwidth_data["total"][source_type][physical_row, physical_col] = total_metrics.weighted_bandwidth

        # 处理SN端口
        sn_requests_by_dest = defaultdict(list)

        for req in self.requests:
            if req.dest_node in self.sn_positions:
                sn_requests_by_dest[req.dest_node].append(req)

        # 计算每个SN目标节点的带宽
        for dest_node, node_requests in sn_requests_by_dest.items():
            # 按dest_type分组
            by_type = defaultdict(list)
            for req in node_requests:
                # 提取dest_type的前缀（去掉_ip后缀）
                dest_type = req.dest_type.lower()[:-2]
                by_type[dest_type].append(req)

            # 计算物理位置
            if getattr(self.sim_model, "topo_type_stat", None).startswith("Ring"):
                if dest_node < rows:
                    physical_col = 0
                    physical_row = dest_node
                else:
                    physical_col = 1
                    physical_row = self.config.RING_NUM_NODE - 1 - dest_node
            else:
                physical_col = dest_node % cols
                physical_row = dest_node // cols // 2

            # 为每种类型计算带宽
            for dest_type, type_requests in by_type.items():
                # 分别计算读写带宽
                read_requests = [req for req in type_requests if req.req_type == "read"]
                write_requests = [req for req in type_requests if req.req_type == "write"]

                # 计算读带宽
                if read_requests:
                    read_metrics = self.calculate_bandwidth_metrics(read_requests, "read")
                    self.ip_bandwidth_data["read"][dest_type][physical_row, physical_col] = read_metrics.weighted_bandwidth

                # 计算写带宽
                if write_requests:
                    write_metrics = self.calculate_bandwidth_metrics(write_requests, "write")
                    self.ip_bandwidth_data["write"][dest_type][physical_row, physical_col] = write_metrics.weighted_bandwidth

                # 计算总带宽
                if type_requests:
                    total_metrics = self.calculate_bandwidth_metrics(type_requests, operation_type=None)
                    self.ip_bandwidth_data["total"][dest_type][physical_row, physical_col] = total_metrics.weighted_bandwidth

    def plot_rn_bandwidth_curves_work_interval(self) -> float:
        """
        绘制RN端带宽曲线图，使用累积和计算带宽，区分工作区间（gap大于阈值时分段）

        Returns:
            总带宽 (GB/s)
        """
        if self.plot_rn_bw_fig:
            fig = plt.figure(figsize=(12, 8))

        total_bw = 0

        for port_key, data_dict in self.rn_bandwidth_time_series.items():
            if not data_dict["time"]:
                continue

            # 排序时间戳并去除nan值，同时处理start_times
            raw_end = np.array(data_dict["time"])
            raw_start = np.array(data_dict["start_times"])
            # Prepare consistent color for this port_key's segments
            segment_color = None
            first_segment = True
            # 去除nan
            mask = ~np.isnan(raw_end)
            end_clean = raw_end[mask]
            start_clean = raw_start[mask]
            # 同步排序
            sort_idx = np.argsort(end_clean)
            times = end_clean[sort_idx]
            start_times = start_clean[sort_idx]

            if len(times) == 0:
                continue

            # 分割工作区间：当相邻时间差大于阈值时，认为是新区间
            gap_indices = np.where(np.diff(times) > self.min_gap_threshold)[0]
            # 定义区间的起止索引
            segment_bounds = np.concatenate(([0], gap_indices + 1, [len(times)]))

            last_bw = 0
            # 对每个工作区间分别绘制，从0开始累积
            for i in range(len(segment_bounds) - 1):
                start_idx = segment_bounds[i]
                end_idx = segment_bounds[i + 1]
                # 获取绝对时间和相对时间
                abs_end = times[start_idx:end_idx]
                # 使用该段第一个请求的start_time作为起点
                segment_start = start_times[start_idx]
                rel_times = abs_end - segment_start
                if len(rel_times) == 0:
                    continue
                cum_counts = np.arange(1, len(rel_times) + 1)
                # 防止除以0
                rel_nonzero = rel_times.copy()
                rel_nonzero[rel_nonzero == 0] = 1e-9
                bandwidth = (cum_counts * 128 * self.config.BURST) / rel_nonzero
                # 绘制曲线片段，使用绝对时间轴
                if self.plot_rn_bw_fig:
                    if first_segment:
                        (line,) = plt.plot(abs_end / 1000, bandwidth, drawstyle="default", label=port_key)
                        segment_color = line.get_color()
                        first_segment = False
                    else:
                        plt.plot(abs_end / 1000, bandwidth, drawstyle="default", color=segment_color)
                    plt.text(abs_end[-1] / 1000, bandwidth[-1], f"{bandwidth[-1]:.2f}", va="center", color=segment_color, fontsize=12)
                last_bw = bandwidth[-1]
                # 打印每个区间的最终带宽值（可选）
                # if hasattr(self, "sim_model") and self.sim_model and hasattr(self.sim_model, "verbose") and self.sim_model.verbose:
                #     print(f"{port_key} seg{i} Final Bandwidth: {bandwidth[-1]:.2f} GB/s")
            total_bw += last_bw

        if hasattr(self, "sim_model") and self.sim_model and hasattr(self.sim_model, "verbose") and self.sim_model.verbose:
            print(f"Total Bandwidth: {total_bw:.2f} GB/s")
            print("=" * 60)

        if self.plot_rn_bw_fig:
            plt.xlabel("Time (us)")
            plt.ylabel("Bandwidth (GB/s)")
            plt.title("RN Bandwidth")
            plt.legend()
            plt.grid(True)
            # 自动保存RN带宽曲线到结果文件夹
            if self.plot_rn_bw_fig and hasattr(self, "sim_model") and getattr(self.sim_model, "results_fig_save_path", None):
                rn_save_path = os.path.join(self.sim_model.results_fig_save_path, f"rn_bandwidth_{self.config.TOPO_TYPE}_{self.sim_model.file_name}_{time.time_ns()}.png")
                fig.savefig(rn_save_path, bbox_inches="tight")
            else:
                plt.show()

        return total_bw

    def plot_rn_bandwidth_curves(self) -> float:
        """
        绘制RN端带宽曲线图，使用累积和计算带宽

        Returns:
            总带宽 (GB/s)
        """
        if self.plot_rn_bw_fig:
            fig = plt.figure(figsize=(12, 8))

        total_bw = 0

        for port_key, data_dict in self.rn_bandwidth_time_series.items():
            if not data_dict["time"]:
                continue

            # 排序时间戳并去除nan值
            raw_times = np.array(data_dict["time"])
            clean_times = raw_times[~np.isnan(raw_times)]
            times = np.sort(clean_times)

            if len(times) == 0:
                continue

            # 计算累积带宽
            cum_counts = np.arange(1, len(times) + 1)
            bandwidth = (cum_counts * 128 * self.config.BURST) / times  # bytes/ns转换为GB/s

            # 只显示前100%的时间段
            t = np.percentile(times, 100)
            mask = times <= t

            if self.plot_rn_bw_fig:
                (line,) = plt.plot(times[mask] / 1000, bandwidth[mask], drawstyle="default", label=f"{port_key}")
                plt.text(times[mask][-1] / 1000, bandwidth[mask][-1], f"{bandwidth[mask][-1]:.2f}", va="center", color=line.get_color(), fontsize=12)

            # 打印最终带宽值
            if hasattr(self, "sim_model") and self.sim_model and hasattr(self.sim_model, "verbose") and self.sim_model.verbose:
                print(f"{port_key} Final Bandwidth: {bandwidth[mask][-1]:.2f} GB/s")

            total_bw += bandwidth[mask][-1]

        if hasattr(self, "sim_model") and self.sim_model and hasattr(self.sim_model, "verbose") and self.sim_model.verbose:
            print(f"Total Bandwidth: {total_bw:.2f} GB/s")
            print("=" * 60)

        if self.plot_rn_bw_fig:
            plt.xlabel("Time (us)")
            plt.ylabel("Bandwidth (GB/s)")
            plt.title("RN Bandwidth")
            plt.legend()
            plt.grid(True)
            # 自动保存RN带宽曲线到结果文件夹
            if self.plot_rn_bw_fig and hasattr(self, "sim_model") and getattr(self.sim_model, "results_fig_save_path", None):
                rn_save_path = os.path.join(self.sim_model.results_fig_save_path, f"rn_bandwidth_{self.config.TOPO_TYPE}_{self.sim_model.file_name}.png")
                fig.savefig(rn_save_path, bbox_inches="tight")
            else:
                plt.show()

        return total_bw

    def calculate_working_intervals(self, requests: List[RequestInfo]) -> List[WorkingInterval]:
        """
        计算工作区间，去除空闲时间段

        Args:
            requests: 请求列表

        Returns:
            工作区间列表
        """
        if not requests:
            return []

        # 构建时间轴事件
        events = []
        for req in requests:
            events.append((req.start_time, "start", req))
            events.append((req.end_time, "end", req))
        events.sort(key=lambda x: (x[0], x[1]))  # 按时间排序，相同时间时'end'在'start'前面

        # 识别连续工作时段
        active_requests = set()
        raw_intervals = []
        current_start = None

        for time_point, event_type, req in events:
            # 检查时间点是否有效（非nan且非None）
            if time_point is not None and not (isinstance(time_point, float) and np.isnan(time_point)):

                if event_type == "start":
                    if not active_requests:  # 开始新的工作区间
                        current_start = time_point
                    active_requests.add(req.packet_id)
                else:  # 'end'
                    active_requests.discard(req.packet_id)
                    if not active_requests and current_start is not None:
                        # 工作区间结束
                        raw_intervals.append((current_start, time_point))
                        current_start = None

        # 处理最后未结束的区间
        if active_requests and current_start is not None:
            last_end = max(req.end_time for req in requests)
            raw_intervals.append((current_start, last_end))

        # 合并相近区间（间隔小于阈值）
        merged_intervals = self._merge_close_intervals(raw_intervals)

        # 构建WorkingInterval对象
        working_intervals = []
        for start, end in merged_intervals:
            # 找到该区间内的所有请求
            interval_requests = [req for req in requests if req.start_time < end and req.end_time > start]

            if not interval_requests:
                continue

            # 计算区间统计
            total_bytes = sum(req.total_bytes for req in interval_requests)
            flit_count = sum(req.burst_length for req in interval_requests)

            interval = WorkingInterval(start_time=start, end_time=end, duration=end - start, flit_count=flit_count, total_bytes=total_bytes, request_count=len(interval_requests))
            working_intervals.append(interval)

        return working_intervals

    def _merge_close_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """合并相近的时间区间"""
        if not intervals:
            return []

        # 按开始时间排序
        sorted_intervals = sorted(intervals)
        merged = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]

            # 如果间隙小于阈值，则合并
            if current_start - last_end <= self.min_gap_threshold:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    def calculate_bandwidth_metrics(self, requests: List[RequestInfo], operation_type: str = None, endpoint_type: str = "network") -> BandwidthMetrics:
        """
        计算指定操作类型的带宽指标

        Args:
            requests: 所有请求列表
            operation_type: 'read'、'write' 或 None（表示混合读写）
            endpoint_type: 'network'(整体网络)、'rn'(RN端口)、'sn'(SN端口)
        """
        # 筛选请求并创建临时请求列表（使用正确的结束时间）
        filtered_requests = []

        for req in requests:
            if operation_type is not None and req.req_type != operation_type:
                continue

            # 根据endpoint_type选择正确的结束时间
            if endpoint_type == "rn":
                end_time = req.rn_end_time
            elif endpoint_type == "sn":
                end_time = req.sn_end_time
            else:  # network
                end_time = req.end_time

            # 创建临时请求对象，使用正确的结束时间
            temp_req = RequestInfo(
                packet_id=req.packet_id,
                start_time=req.start_time,
                end_time=end_time,
                rn_end_time=req.rn_end_time,
                sn_end_time=req.sn_end_time,
                req_type=req.req_type,
                source_node=req.source_node,
                dest_node=req.dest_node,
                source_type=req.source_type,
                dest_type=req.dest_type,
                burst_length=req.burst_length,
                total_bytes=req.total_bytes,
                cmd_latency=req.cmd_latency,
                data_latency=req.data_latency,
                transaction_latency=req.transaction_latency,
            )
            filtered_requests.append(temp_req)

        if not filtered_requests:
            return BandwidthMetrics(
                unweighted_bandwidth=0.0,
                weighted_bandwidth=0.0,
                working_intervals=[],
                total_working_time=0,
                network_start_time=0,
                network_end_time=0,
                total_bytes=0,
                total_requests=0,
            )

        # 计算工作区间
        working_intervals = self.calculate_working_intervals(filtered_requests)

        # 网络工作时间窗口
        network_start = min(req.start_time for req in filtered_requests)
        network_end = max(req.end_time for req in filtered_requests)
        total_network_time = network_end - network_start

        # 总工作时间和总字节数
        total_working_time = sum(interval.duration for interval in working_intervals)
        total_bytes = sum(req.total_bytes for req in filtered_requests)

        # 计算非加权带宽：总数据量 / 网络总时间 / IP数量
        unweighted_bandwidth = (total_bytes / total_network_time) if total_network_time > 0 else 0.0

        # 计算加权带宽：各区间带宽按flit数量加权平均
        if working_intervals:
            total_weighted_bandwidth = 0.0
            total_weight = 0

            for interval in working_intervals:
                weight = interval.flit_count  # 权重是工作时间段的flit数量
                bandwidth = interval.bandwidth_bytes_per_ns  # bytes
                total_weighted_bandwidth += bandwidth * weight
                total_weight += weight

            weighted_bandwidth = (total_weighted_bandwidth / total_weight) if total_weight > 0 else 0.0
        else:
            weighted_bandwidth = 0.0

        return BandwidthMetrics(
            unweighted_bandwidth=unweighted_bandwidth,
            weighted_bandwidth=weighted_bandwidth,
            working_intervals=working_intervals,
            total_working_time=total_working_time,
            network_start_time=network_start,
            network_end_time=network_end,
            total_bytes=total_bytes,
            total_requests=len(filtered_requests),
        )

    def calculate_network_overall_bandwidth(self) -> Dict[str, BandwidthMetrics]:
        """
        计算网络整体带宽（读、写和混合）

        Returns:
            {'read': BandwidthMetrics, 'write': BandwidthMetrics, 'mixed': BandwidthMetrics}
        """
        results = {}

        # 分别计算读写带宽
        for operation in ["read", "write"]:
            results[operation] = self.calculate_bandwidth_metrics(self.requests, operation)

        # 计算混合读写带宽（不区分请求类型）
        results["mixed"] = self.calculate_bandwidth_metrics(self.requests, operation_type=None)

        return results

    def calculate_rn_port_bandwidth(self) -> Dict[str, PortBandwidthMetrics]:
        """计算每个RN端口的带宽"""
        port_metrics = {}
        rn_requests_by_port = defaultdict(list)

        for req in self.requests:
            if req.source_node in self.rn_positions:
                port_id = f"{req.source_type}_{req.source_node}"
                rn_requests_by_port[port_id].append(req)

        for port_id, port_requests in rn_requests_by_port.items():
            # 使用RN端点类型计算带宽
            read_metrics = self.calculate_bandwidth_metrics(port_requests, "read", "rn")
            write_metrics = self.calculate_bandwidth_metrics(port_requests, "write", "rn")
            mixed_metrics = self.calculate_bandwidth_metrics(port_requests, None, "rn")

            port_metrics[port_id] = PortBandwidthMetrics(port_id=port_id, read_metrics=read_metrics, write_metrics=write_metrics, mixed_metrics=mixed_metrics)

        return port_metrics

    def calculate_sn_port_bandwidth(self) -> Dict[str, PortBandwidthMetrics]:
        """计算每个SN端口的带宽"""
        port_metrics = {}
        sn_requests_by_port = defaultdict(list)

        for req in self.requests:
            if req.dest_node in self.sn_positions:
                port_id = f"{req.dest_type}_{req.dest_node}"
                sn_requests_by_port[port_id].append(req)

        for port_id, port_requests in sn_requests_by_port.items():
            # 使用SN端点类型计算带宽
            read_metrics = self.calculate_bandwidth_metrics(port_requests, "read", "sn")
            write_metrics = self.calculate_bandwidth_metrics(port_requests, "write", "sn")
            mixed_metrics = self.calculate_bandwidth_metrics(port_requests, None, "sn")

            port_metrics[port_id] = PortBandwidthMetrics(port_id=port_id, read_metrics=read_metrics, write_metrics=write_metrics, mixed_metrics=mixed_metrics)

        return port_metrics

    def precalculate_ip_bandwidth_data(self):
        """预计算IP带宽数据，供绘图使用"""
        if self.ip_bandwidth_data is not None:
            return  # 已经计算过了

        # 使用现有的calculate_ip_bandwidth_data逻辑
        # 但基于已有的端口带宽计算结果
        self.calculate_ip_bandwidth_data()

    def _calculate_port_bandwidth_averages(self, all_ports: Dict[str, 'PortBandwidthMetrics']) -> Dict[str, float]:
        """
        计算每种端口类型的平均带宽

        Args:
            all_ports: 所有端口的带宽指标字典

        Returns:
            一个包含每种端口类型平均带宽的字典
        """
        port_bw_groups = defaultdict(list)
        for port_id, metrics in all_ports.items():
            port_type = port_id.split('_')[0]  # 提取端口类型 (gdma, sdma, cdma, ddr, l2m)
            port_bw_groups[port_type].append(metrics.mixed_metrics.weighted_bandwidth)

        avg_port_metrics = {}
        for port_type, bw_list in port_bw_groups.items():
            if bw_list:
                # 使用 f-string 格式化键名
                avg_port_metrics[f"avg_{port_type}_bw"] = sum(bw_list) / len(bw_list)
        
        return avg_port_metrics

    def analyze_all_bandwidth(self) -> Dict:
        """
        执行完整的带宽分析

        Returns:
            包含所有分析结果的字典
        """
        if not self.requests:
            raise ValueError("没有请求数据，请先调用 collect_requests_data()")

        # 网络整体带宽分析
        network_overall = self.calculate_network_overall_bandwidth()

        # RN端口带宽分析
        rn_port_metrics = self.calculate_rn_port_bandwidth()

        # SN端口带宽分析
        sn_port_metrics = self.calculate_sn_port_bandwidth()

        # 计算并合并端口平均带宽
        all_ports = {**rn_port_metrics, **sn_port_metrics}
        port_averages = self._calculate_port_bandwidth_averages(all_ports)

        # 汇总统计
        total_read_requests = len([r for r in self.requests if r.req_type == "read"])
        total_write_requests = len([r for r in self.requests if r.req_type == "write"])
        total_read_flits = sum(req.burst_length for req in self.requests if req.req_type == "read")
        total_write_flits = sum(req.burst_length for req in self.requests if req.req_type == "write")

        # 获取Circuit统计数据（如果存在）
        circuit_stats = {}
        if hasattr(self, "sim_model") and self.sim_model:
            circuit_stats = {
                "req_circuits_h": getattr(self.sim_model, "req_cir_h_num_stat", 0),
                "req_circuits_v": getattr(self.sim_model, "req_cir_v_num_stat", 0),
                "rsp_circuits_h": getattr(self.sim_model, "rsp_cir_h_num_stat", 0),
                "rsp_circuits_v": getattr(self.sim_model, "rsp_cir_v_num_stat", 0),
                "data_circuits_h": getattr(self.sim_model, "data_cir_h_num_stat", 0),
                "data_circuits_v": getattr(self.sim_model, "data_cir_v_num_stat", 0),
                "req_wait_cycles_h": getattr(self.sim_model, "req_wait_cycle_h_num_stat", 0),
                "req_wait_cycles_v": getattr(self.sim_model, "req_wait_cycle_v_num_stat", 0),
                "rsp_wait_cycles_h": getattr(self.sim_model, "rsp_wait_cycle_h_num_stat", 0),
                "rsp_wait_cycles_v": getattr(self.sim_model, "rsp_wait_cycle_v_num_stat", 0),
                "data_wait_cycles_h": getattr(self.sim_model, "data_wait_cycle_h_num_stat", 0),
                "data_wait_cycles_v": getattr(self.sim_model, "data_wait_cycle_v_num_stat", 0),
                "read_retry_num": getattr(self.sim_model, "read_retry_num_stat", 0),
                "write_retry_num": getattr(self.sim_model, "write_retry_num_stat", 0),
                "EQ_ETag_T1_num": getattr(self.sim_model, "EQ_ETag_T1_num_stat", 0),
                "EQ_ETag_T0_num": getattr(self.sim_model, "EQ_ETag_T0_num_stat", 0),
                "RB_ETag_T1_num": getattr(self.sim_model, "RB_ETag_T1_num_stat", 0),
                "RB_ETag_T0_num": getattr(self.sim_model, "RB_ETag_T0_num_stat", 0),
                "ITag_h_num": getattr(self.sim_model, "ITag_h_num_stat", 0),
                "ITag_v_num": getattr(self.sim_model, "ITag_v_num_stat", 0),
            }

        results = {
            "network_overall": network_overall,
            "rn_ports": rn_port_metrics,
            "sn_ports": sn_port_metrics,
            "port_averages": port_averages,
            "summary": {
                "total_requests": len(self.requests),
                "read_requests": total_read_requests,
                "write_requests": total_write_requests,
                "total_read_flits": total_read_flits,
                "total_write_flits": total_write_flits,
                "analysis_config": {"min_gap_threshold_ns": self.min_gap_threshold, "network_frequency_ghz": self.network_frequency},
                "circuit_stats": circuit_stats,
            },
        }

        # 控制台输出重要数据
        if hasattr(self, "sim_model") and self.sim_model and hasattr(self.sim_model, "verbose") and self.sim_model.verbose:
            self._print_summary_to_console(results)

        # 绘制RN带宽曲线
        total_bandwidth = self.plot_rn_bandwidth_curves_work_interval()
        results["summary"]["Total_sum_BW"] = total_bandwidth
        results["Total_sum_BW"] = total_bandwidth

        if self.plot_flow_graph:
            if self.sim_model.results_fig_save_path:
                # 保存流量图到结果文件夹
                flow_fname = f"flow_graph_{self.config.TOPO_TYPE}_{self.sim_model.file_name}_{time.time_ns()}.png"
                flow_save_path = os.path.join(self.sim_model.results_fig_save_path, flow_fname)
            else:
                flow_save_path = None
            if self.sim_model.topo_type_stat.startswith("Ring"):
                self.draw_ring_flow_graph(self.sim_model.data_network, save_path=flow_save_path)
            else:
                self.draw_flow_graph(self.sim_model.data_network, mode="total", save_path=flow_save_path, show_cdma=self.sim_model.flow_fig_show_CDMA)
        return results

    def draw_flow_graph(self, network: Network, mode="total", node_size=2000, save_path=None, show_cdma=True):
        """
        绘制合并的网络流图和IP

        :param network: 网络对象
        :param mode: 显示模式，可以是'read', 'write'或'total'
        :param node_size: 节点大小
        :param save_path: 图片保存路径
        :param show_cdma: 是否展示CDMA，True显示SDMA/GDMA/CDMA三分区，False显示SDMA/GDMA两分区
        """
        # 确保IP带宽数据已计算
        self.precalculate_ip_bandwidth_data()

        # 准备网络流数据
        G = nx.DiGraph()

        # 处理不同模式的网络流数据
        if mode == "read":
            links = network.links_flow_stat.get("read", {})
        elif mode == "write":
            links = network.links_flow_stat.get("write", {})
        else:  # total模式，需要合并读和写的数据
            read_links = network.links_flow_stat.get("read", {})
            write_links = network.links_flow_stat.get("write", {})

            # 合并读和写的流量
            all_keys = set(read_links.keys()) | set(write_links.keys())
            links = {}
            for key in all_keys:
                read_val = read_links.get(key, 0)
                write_val = write_links.get(key, 0)
                links[key] = read_val + write_val

        link_values = []
        for (i, j), value in links.items():
            link_value = value * 128 / (self.finish_cycle // self.config.NETWORK_FREQUENCY) if value else 0
            link_values.append(link_value)
            formatted_label = f"{link_value:.1f}"
            G.add_edge(i, j, label=formatted_label)

        # 链路颜色映射范围：动态最大值的60%阈值起红
        link_mapping_max = max(link_values) if link_values else 0.0
        link_mapping_min = max(0.6 * link_mapping_max, 100)

        # IP颜色映射范围：动态最大值的60%阈值起红
        # 从 ip_bandwidth_data 提取所有带宽值，计算当前最大
        all_ip_vals = []
        ip_services = ["sdma", "gdma", "cdma", "ddr", "l2m"]
        if show_cdma and "cdma" in self.ip_bandwidth_data[mode]:
            ip_services.append("cdma")

        for svc in ip_services:
            all_ip_vals.extend(self.ip_bandwidth_data[mode][svc].flatten())
        ip_mapping_max = max(all_ip_vals) if all_ip_vals else 0.0
        ip_mapping_min = max(0.6 * ip_mapping_max, 80)

        # 计算节点位置
        pos = {}
        for node in G.nodes():
            x = node % self.config.NUM_COL
            y = node // self.config.NUM_COL
            if y % 2 == 1:  # 奇数行左移
                x -= 0.25
                y -= 0.6
            pos[node] = (x * 3, -y * 1.5)

        # 动态计算字体大小，添加最大值上限以避免过大
        node_count = len(G.nodes())
        base_font = 9
        dynamic_font = max(4, base_font * (65 / node_count) ** 0.5)
        max_font = 14  # 最大字号上限，可以根据需要调整
        dynamic_font = min(dynamic_font, max_font)

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))  # 增大画布以容纳更多内容
        ax.set_aspect("equal")

        # 调整方形节点大小
        square_size = np.sqrt(node_size) / 100

        # 绘制网络流图
        nx.draw_networkx_nodes(G, pos, node_size=square_size, node_shape="s", ax=ax)

        # 绘制方形节点并添加IP信息
        for node, (x, y) in pos.items():
            # 绘制主节点方框
            rect = Rectangle(
                (x - square_size / 2, y - square_size / 2),
                width=square_size,
                height=square_size,
                color="lightblue",
                ec="black",
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(x, y, str(node), ha="center", va="center", fontsize=dynamic_font)

            # 在节点左侧添加IP信息
            physical_row = node // self.config.NUM_COL
            physical_col = node % self.config.NUM_COL

            if physical_row % 2 == 0:
                # IP信息框位置和大小
                ip_width = square_size * 3.2
                ip_height = square_size * 2.6
                ip_x = x - square_size - ip_width / 2.5
                ip_y = y + 0.26

                # 绘制IP信息框外框
                ip_rect = Rectangle(
                    (ip_x - ip_width / 2, ip_y - ip_height / 2),
                    width=ip_width,
                    height=ip_height,
                    color="white",
                    ec="black",
                    linewidth=1,
                    zorder=2,
                )
                ax.add_patch(ip_rect)

                # 绘制分割线：垂直线分割左右两部分
                ax.plot(
                    [ip_x, ip_x],
                    [ip_y - ip_height / 2, ip_y + ip_height / 2],
                    color="black",
                    linewidth=1,
                    zorder=3,
                )

                # 绘制水平线分割右侧上下两部分
                ax.plot(
                    [ip_x, ip_x + ip_width / 2],
                    [ip_y, ip_y],
                    color="black",
                    linewidth=1,
                    zorder=3,
                )

                # 左侧DMA区域分割
                left_width = ip_width / 2

                if show_cdma:
                    # 三分区：SDMA、GDMA、CDMA
                    dma_height = ip_height / 3
                    # 绘制左侧的两条水平分割线
                    for i in range(1, 3):
                        line_y = ip_y - ip_height / 2 + i * dma_height
                        ax.plot(
                            [ip_x - ip_width / 2, ip_x],
                            [line_y, line_y],
                            color="black",
                            linewidth=0.8,
                            zorder=3,
                        )
                else:
                    # 两分区：SDMA、GDMA
                    dma_height = ip_height / 2
                    # 绘制左侧的一条水平分割线
                    line_y = ip_y
                    ax.plot(
                        [ip_x - ip_width / 2, ip_x],
                        [line_y, line_y],
                        color="black",
                        linewidth=0.8,
                        zorder=3,
                    )

                # 为左侧DMA区域填充颜色
                dma_color = "honeydew"  # DMA区域统一颜色
                right_color = "aliceblue"  # 右侧DDR/L2M区域颜色

                if show_cdma:
                    # SDMA区域（最上方）
                    sdma_rect = Rectangle(
                        (ip_x - ip_width / 2, ip_y + dma_height / 2),
                        width=left_width,
                        height=dma_height,
                        color=dma_color,
                        ec="none",
                        zorder=2,
                    )
                    ax.add_patch(sdma_rect)

                    # GDMA区域（中间）
                    gdma_rect = Rectangle(
                        (ip_x - ip_width / 2, ip_y - dma_height / 2),
                        width=left_width,
                        height=dma_height,
                        color=dma_color,
                        ec="none",
                        zorder=2,
                    )
                    ax.add_patch(gdma_rect)

                    # CDMA区域（最下方）
                    cdma_rect = Rectangle(
                        (ip_x - ip_width / 2, ip_y - ip_height / 2),
                        width=left_width,
                        height=dma_height,
                        color=dma_color,
                        ec="none",
                        zorder=2,
                    )
                    ax.add_patch(cdma_rect)
                else:
                    # SDMA区域（上半部分）
                    sdma_rect = Rectangle(
                        (ip_x - ip_width / 2, ip_y),
                        width=left_width,
                        height=dma_height,
                        color=dma_color,
                        ec="none",
                        zorder=2,
                    )
                    ax.add_patch(sdma_rect)

                    # GDMA区域（下半部分）
                    gdma_rect = Rectangle(
                        (ip_x - ip_width / 2, ip_y - ip_height / 2),
                        width=left_width,
                        height=dma_height,
                        color=dma_color,
                        ec="none",
                        zorder=2,
                    )
                    ax.add_patch(gdma_rect)

                # 右侧DDR/L2M区域
                right_rect = Rectangle(
                    (ip_x, ip_y - ip_height / 2),
                    width=ip_width / 2,
                    height=ip_height,
                    color=right_color,
                    ec="none",
                    zorder=2,
                )
                ax.add_patch(right_rect)

                # 获取IP带宽数据
                if mode == "read":
                    sdma_value = self.ip_bandwidth_data["read"]["sdma"][physical_row // 2, physical_col]
                    gdma_value = self.ip_bandwidth_data["read"]["gdma"][physical_row // 2, physical_col]
                    if show_cdma and "cdma" in self.ip_bandwidth_data["read"]:
                        cdma_value = self.ip_bandwidth_data["read"]["cdma"][physical_row // 2, physical_col]
                    ddr_value = self.ip_bandwidth_data["read"]["ddr"][physical_row // 2, physical_col]
                    l2m_value = self.ip_bandwidth_data["read"]["l2m"][physical_row // 2, physical_col]
                elif mode == "write":
                    sdma_value = self.ip_bandwidth_data["write"]["sdma"][physical_row // 2, physical_col]
                    gdma_value = self.ip_bandwidth_data["write"]["gdma"][physical_row // 2, physical_col]
                    if show_cdma and "cdma" in self.ip_bandwidth_data["write"]:
                        cdma_value = self.ip_bandwidth_data["write"]["cdma"][physical_row // 2, physical_col]
                    ddr_value = self.ip_bandwidth_data["write"]["ddr"][physical_row // 2, physical_col]
                    l2m_value = self.ip_bandwidth_data["write"]["l2m"][physical_row // 2, physical_col]
                else:  # total
                    sdma_value = self.ip_bandwidth_data["total"]["sdma"][physical_row // 2, physical_col]
                    gdma_value = self.ip_bandwidth_data["total"]["gdma"][physical_row // 2, physical_col]
                    if show_cdma and "cdma" in self.ip_bandwidth_data["total"]:
                        cdma_value = self.ip_bandwidth_data["total"]["cdma"][physical_row // 2, physical_col]
                    ddr_value = self.ip_bandwidth_data["total"]["ddr"][physical_row // 2, physical_col]
                    l2m_value = self.ip_bandwidth_data["total"]["l2m"][physical_row // 2, physical_col]

                # 定义颜色计算函数
                def get_intensity_color(value):
                    if value <= ip_mapping_min:
                        intensity = 0.0
                    else:
                        intensity = (value - ip_mapping_min) / (ip_mapping_max - ip_mapping_min)
                    intensity = min(max(intensity, 0.0), 1.0)
                    return (intensity, 0, 0)

                if show_cdma:
                    # 三分区布局
                    # SDMA在最上方区域
                    sdma_color_val = get_intensity_color(sdma_value)
                    ax.text(
                        ip_x - ip_width / 4,
                        ip_y + dma_height,
                        f"S:{sdma_value:.1f}",
                        fontweight="bold",
                        ha="center",
                        va="center",
                        fontsize=dynamic_font * 0.5,
                        color=sdma_color_val,
                    )

                    # GDMA在中间区域
                    gdma_color_val = get_intensity_color(gdma_value)
                    ax.text(
                        ip_x - ip_width / 4,
                        ip_y,
                        f"G:{gdma_value:.1f}",
                        fontweight="bold",
                        ha="center",
                        va="center",
                        fontsize=dynamic_font * 0.5,
                        color=gdma_color_val,
                    )

                    # CDMA在最下方区域
                    if "cdma" in self.ip_bandwidth_data[mode]:
                        cdma_color_val = get_intensity_color(cdma_value)
                        ax.text(
                            ip_x - ip_width / 4,
                            ip_y - dma_height,
                            f"C:{cdma_value:.1f}",
                            fontweight="bold",
                            ha="center",
                            va="center",
                            fontsize=dynamic_font * 0.5,
                            color=cdma_color_val,
                        )
                    else:
                        # 如果没有CDMA数据，显示空值
                        ax.text(
                            ip_x - ip_width / 4,
                            ip_y - dma_height,
                            "C:0.0",
                            fontweight="bold",
                            ha="center",
                            va="center",
                            fontsize=dynamic_font * 0.5,
                            color=(0, 0, 0),
                        )
                else:
                    # 两分区布局
                    # SDMA在上半区域
                    sdma_color_val = get_intensity_color(sdma_value)
                    ax.text(
                        ip_x - ip_width / 4,
                        ip_y + dma_height / 2,
                        f"S:{sdma_value:.1f}",
                        fontweight="bold",
                        ha="center",
                        va="center",
                        fontsize=dynamic_font * 0.6,
                        color=sdma_color_val,
                    )

                    # GDMA在下半区域
                    gdma_color_val = get_intensity_color(gdma_value)
                    ax.text(
                        ip_x - ip_width / 4,
                        ip_y - dma_height / 2,
                        f"G:{gdma_value:.1f}",
                        fontweight="bold",
                        ha="center",
                        va="center",
                        fontsize=dynamic_font * 0.6,
                        color=gdma_color_val,
                    )

                # L2M在右上区域
                l2m_color_val = get_intensity_color(l2m_value)
                ax.text(
                    ip_x + ip_width / 4,
                    ip_y + ip_height / 4,
                    f"L:{l2m_value:.1f}",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    fontsize=dynamic_font * 0.5,
                    color=l2m_color_val,
                )

                # DDR在右下区域
                ddr_color_val = get_intensity_color(ddr_value)
                ax.text(
                    ip_x + ip_width / 4,
                    ip_y - ip_height / 4,
                    f"D:{ddr_value:.1f}",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    fontsize=dynamic_font * 0.5,
                    color=ddr_color_val,
                )

        # 绘制边和边标签（保持原有逻辑）
        edge_value_threshold = np.percentile(link_values, 90)

        for i, j, data in G.edges(data=True):
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            # 根据流量值映射颜色
            value = float(data["label"])
            # 链路60%阈值起红
            if value <= link_mapping_min:
                intensity = 0.0
            else:
                intensity = (value - link_mapping_min) / (link_mapping_max - link_mapping_min)
            # clamp to [0,1]
            intensity = min(max(intensity, 0.0), 1.0)
            color = (intensity, 0, 0)

            if i != j:  # 普通边
                dx, dy = x2 - x1, y2 - y1
                dist = np.hypot(dx, dy)
                if dist > 0:
                    dx, dy = dx / dist, dy / dist
                    perp_dx, perp_dy = -dy * 0.1, dx * 0.1

                    has_reverse = G.has_edge(j, i)
                    if has_reverse:
                        start_x = x1 + dx * square_size / 2 + perp_dx
                        start_y = y1 + dy * square_size / 2 + perp_dy
                        end_x = x2 - dx * square_size / 2 + perp_dx
                        end_y = y2 - dy * square_size / 2 + perp_dy
                    else:
                        start_x = x1 + dx * square_size / 2
                        start_y = y1 + dy * square_size / 2
                        end_x = x2 - dx * square_size / 2
                        end_y = y2 - dy * square_size / 2

                    arrow = FancyArrowPatch(
                        (start_x, start_y),
                        (end_x, end_y),
                        arrowstyle="-|>",
                        mutation_scale=dynamic_font * 0.8,
                        color=color,
                        zorder=1,
                        linewidth=1,
                    )
                    ax.add_patch(arrow)

        # 绘制边标签（保持原有逻辑）
        edge_labels = nx.get_edge_attributes(G, "label")
        for edge, label in edge_labels.items():
            i, j = edge
            if float(label) == 0.0:
                continue
            # 根据带宽值映射标签颜色
            value = float(label)
            # 链路60%阈值起红
            if value <= link_mapping_min:
                intensity = 0.0
            else:
                intensity = (value - link_mapping_min) / (link_mapping_max - link_mapping_min)
            # clamp to [0,1]
            intensity = min(max(intensity, 0.0), 1.0)
            color = (intensity, 0, 0)

            if i == j:
                # 计算标签位置
                original_row = i // self.config.NUM_COL
                original_col = i % self.config.NUM_COL
                x, y = pos[i]

                offset = 0.17  # 标签偏移量
                if original_row == 0:
                    label_pos = (x, y + square_size / 2 + offset)
                    angle = 0
                elif original_row == self.config.NUM_ROW - 2:
                    label_pos = (x, y - square_size / 2 - offset)
                    angle = 0
                elif original_col == 0:
                    label_pos = (x - square_size / 2 - offset, y)
                    angle = -90
                elif original_col == self.config.NUM_COL - 1:
                    label_pos = (x + square_size / 2 + offset, y)
                    angle = 90
                else:
                    label_pos = (x, y + square_size / 2 + offset)
                    angle = 0

                ax.text(
                    *label_pos,
                    str(label),
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    fontsize=dynamic_font * 0.9,
                    rotation=angle,
                )

            if i != j:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                angle = np.degrees(np.arctan2(dy, dx))

                has_reverse = G.has_edge(j, i)
                is_horizontal = abs(dx) > abs(dy)

                if has_reverse:
                    if is_horizontal:
                        perp_dx, perp_dy = -dy * 0.1 + 0.2, dx * 0.1
                    else:
                        perp_dx, perp_dy = -dy * 0.18, dx * 0.18 - 0.3
                    label_x = mid_x + perp_dx
                    label_y = mid_y + perp_dy
                else:
                    if is_horizontal:
                        label_x = mid_x + dx * 0.1
                        label_y = mid_y + dy * 0.1
                    else:
                        label_x = mid_x + (-dy * 0.1 if dx > 0 else dy * 0.1)
                        label_y = mid_y - 0.1

                ax.text(
                    label_x,
                    label_y,
                    str(label),
                    ha="center",
                    va="center",
                    fontsize=dynamic_font,
                    fontweight="bold",
                    color=color,
                )

        plt.axis("off")
        title = f"{network.name} - {mode.capitalize()} Bandwidth"

        if self.config.SPARE_CORE_ROW != -1:
            title += f"\nRow: {self.config.SPARE_CORE_ROW}, Failed cores: {self.config.FAIL_CORE_POS}, Spare cores: {self.config.spare_core_pos}"
        plt.title(title, fontsize=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(
                os.path.join(save_path),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def _save_analysis_config(self, output_path: str):
        """保存分析配置用于重新绘图"""
        config_file = os.path.join(output_path, "analysis_config.json")

        config_data = {
            "min_gap_threshold": self.min_gap_threshold,
            "network_frequency": self.network_frequency,
            "burst": getattr(self.config, "BURST", 4),
            "topo_type": getattr(self.config, "TOPO_TYPE", "unknown"),
            "num_ip": getattr(self.config, "NUM_IP", 1),
            "num_col": getattr(self.config, "NUM_COL", 1),
            "num_row": getattr(self.config, "NUM_ROW", 1),
            "spare_core_row": getattr(self.config, "SPARE_CORE_ROW", -1),
            "fail_core_pos": getattr(self.config, "FAIL_CORE_POS", []),
            "spare_core_pos": getattr(self.config, "spare_core_pos", []),
            "rn_positions": list(self.rn_positions),
            "sn_positions": list(self.sn_positions),
        }

        # 保存网络链路流量数据（如果存在）
        if hasattr(self, "sim_model") and hasattr(self.sim_model, "data_network"):
            network = self.sim_model.data_network
            if hasattr(network, "links_flow_stat"):
                config_data["links_flow_stat"] = {}
                for mode, links in network.links_flow_stat.items():
                    # 转换为可序列化格式，键需要转换为字符串
                    config_data["links_flow_stat"][mode] = {f"{i},{j}": value for (i, j), value in links.items()}

            # 保存finish_cycle用于计算链路带宽
            config_data["finish_cycle"] = float(self.finish_cycle)
            config_data["network_name"] = getattr(network, "name", "Network")

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        if hasattr(self, "sim_model") and self.sim_model and hasattr(self.sim_model, "verbose") and self.sim_model.verbose:
            print(f"分析配置已保存: {config_file}")

    def load_requests_from_csv(self, csv_folder: str, config_dict: Dict = None):
        """
        从CSV文件重新加载请求数据

        Args:
            csv_folder: 包含read_requests.csv和write_requests.csv的文件夹
            config_dict: 配置字典，如果为None则尝试从保存的配置加载
        """
        self.requests.clear()
        self.rn_bandwidth_time_series.clear()

        # 加载配置
        if config_dict is None:
            config_file = os.path.join(csv_folder, "analysis_config.json")
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            else:
                print("警告: 未找到配置文件，使用默认配置")
                config_data = {
                    # "min_gap_threshold": 50,
                    "network_frequency": 1.0,
                    "burst": 4,
                    "topo_type": "unknown",
                    "num_ip": 1,
                    "num_col": 4,
                    "num_row": 5,
                    "rn_positions": [],
                    "sn_positions": [],
                }
        else:
            config_data = config_dict

        # 设置分析器属性
        # self.min_gap_threshold = config_data.get("min_gap_threshold", 50)
        self.network_frequency = config_data.get("network_frequency", 1.0)
        self.finish_cycle = config_data.get("finish_cycle", 1000000)

        # 创建临时config对象
        class TempConfig:
            def __init__(self, config_dict):
                self.BURST = config_dict.get("burst", 4)
                self.TOPO_TYPE = config_dict.get("topo_type", "unknown")
                self.NUM_IP = config_dict.get("num_ip", 1)
                self.NUM_COL = config_dict.get("num_col", 4)
                self.NUM_ROW = config_dict.get("num_row", 5)
                self.SPARE_CORE_ROW = config_dict.get("spare_core_row", -1)
                self.FAIL_CORE_POS = config_dict.get("fail_core_pos", [])
                self.spare_core_pos = config_dict.get("spare_core_pos", [])
                self.NETWORK_FREQUENCY = config_dict.get("network_frequency", 1.0)

        self.config = TempConfig(config_data)

        # 恢复节点位置
        self.rn_positions = set(config_data.get("rn_positions", []))
        self.sn_positions = set(config_data.get("sn_positions", []))

        # 创建临时网络对象用于流图绘制
        if "links_flow_stat" in config_data:

            class TempNetwork:
                def __init__(self, config_data):
                    self.name = config_data.get("network_name", "Network")
                    self.links_flow_stat = {}

                    # 恢复链路流量数据
                    for mode, links_str in config_data["links_flow_stat"].items():
                        self.links_flow_stat[mode] = {}
                        for key_str, value in links_str.items():
                            # 将字符串键转换回元组
                            i, j = map(int, key_str.split(","))
                            self.links_flow_stat[mode][(i, j)] = value

            self.temp_network = TempNetwork(config_data)
        else:
            self.temp_network = None

        # 读取CSV文件（保持原有逻辑）
        read_csv = os.path.join(csv_folder, "read_requests.csv")
        write_csv = os.path.join(csv_folder, "write_requests.csv")

        # 处理读请求
        if os.path.exists(read_csv):
            df_read = pd.read_csv(read_csv)
            for _, row in df_read.iterrows():
                request_info = RequestInfo(
                    packet_id=int(row["packet_id"]),
                    start_time=int(row["start_time_ns"]),
                    end_time=int(row["end_time_ns"]),
                    rn_end_time=int(row["end_time_ns"]),
                    sn_end_time=int(row["end_time_ns"]),
                    req_type="read",
                    source_node=int(row["source_node"]),
                    dest_node=int(row["dest_node"]),
                    source_type=str(row["source_type"]),
                    dest_type=str(row["dest_type"]),
                    burst_length=int(row["burst_length"]),
                    total_bytes=int(row["burst_length"]) * 128,
                    cmd_latency=int(row["cmd_latency_ns"]),
                    data_latency=int(row["data_latency_ns"]),
                    transaction_latency=int(row["transaction_latency_ns"]),
                )
                self.requests.append(request_info)

                # 更新节点位置信息（如果配置中没有）
                if not self.rn_positions and request_info.source_type.endswith("_ip"):
                    self.rn_positions.add(request_info.source_node)
                if not self.sn_positions and request_info.dest_type.endswith("_ip"):
                    self.sn_positions.add(request_info.dest_node)

        # 处理写请求
        if os.path.exists(write_csv):
            df_write = pd.read_csv(write_csv)
            for _, row in df_write.iterrows():
                request_info = RequestInfo(
                    packet_id=int(row["packet_id"]),
                    start_time=int(row["start_time_ns"]),
                    end_time=int(row["end_time_ns"]),
                    rn_end_time=int(row["end_time_ns"]),
                    sn_end_time=int(row["end_time_ns"]),
                    req_type="write",
                    source_node=int(row["source_node"]),
                    dest_node=int(row["dest_node"]),
                    source_type=str(row["source_type"]),
                    dest_type=str(row["dest_type"]),
                    burst_length=int(row["burst_length"]),
                    total_bytes=int(row["burst_length"]) * 128,
                    cmd_latency=int(row["cmd_latency_ns"]),
                    data_latency=int(row["data_latency_ns"]),
                    transaction_latency=int(row["transaction_latency_ns"]),
                )
                self.requests.append(request_info)

                # 更新节点位置信息（如果配置中没有）
                if not self.rn_positions and request_info.source_type.endswith("_ip"):
                    self.rn_positions.add(request_info.source_node)
                if not self.sn_positions and request_info.dest_type.endswith("_ip"):
                    self.sn_positions.add(request_info.dest_node)

        # 按开始时间排序
        self.requests.sort(key=lambda x: x.start_time)

        # 重建RN带宽时间序列数据
        self._rebuild_rn_bandwidth_time_series()

        print(f"从CSV加载了 {len(self.requests)} 个请求 (读: {len([r for r in self.requests if r.req_type == 'read'])}, " f"写: {len([r for r in self.requests if r.req_type == 'write'])})")

    def _rebuild_rn_bandwidth_time_series(self):
        """重建RN带宽时间序列数据"""
        for req in self.requests:
            if req.source_node in self.rn_positions:
                # 使用与原始collect_requests_data相同的逻辑
                port_key = f"{req.source_type[:-2].upper()} {req.req_type} {req.dest_type[:3].upper()}"

                if req.req_type == "read":
                    completion_time = req.rn_end_time
                else:  # write
                    completion_time = req.rn_end_time

                self.rn_bandwidth_time_series[port_key]["time"].append(completion_time)
                self.rn_bandwidth_time_series[port_key]["start_times"].append(req.start_time)
                self.rn_bandwidth_time_series[port_key]["bytes"].append(req.burst_length * 128)

    def generate_unified_report(self, results: Dict, output_path: str) -> None:
        """
        生成统一的带宽分析报告

        Args:
            results: analyze_all_bandwidth()的返回结果
            output_path: 输出目录路径
        """
        os.makedirs(output_path, exist_ok=True)

        report_file = os.path.join(output_path, "bandwidth_analysis_report.txt")

        with open(report_file, "w", encoding="utf-8") as f:
            self._write_report_header(f, results)
            self._write_network_overall_section(f, results["network_overall"])
            self._generate_ports_csv(results["rn_ports"], output_path)

        # 生成详细请求记录的CSV文件
        self._generate_detailed_request_csv(output_path)

        # 保存分析配置
        self._save_analysis_config(output_path)

        if self.sim_model.verbose:
            print(f"带宽分析报告： {report_file}")
            print(f"具体端口的统计CSV： {output_path}ports_bandwidth.csv")

    @staticmethod
    def reanalyze_and_plot_from_csv(csv_folder: str, output_path: str = None, plot_rn_bw: bool = True, plot_flow: bool = False, show_cdma: bool = False, min_gap_threshold=50) -> Dict:
        """
        从CSV文件重新分析并绘图

        Args:
            csv_folder: 包含CSV文件的文件夹
            output_path: 图片保存路径，如果为None则保存到csv_folder
            plot_rn_bw: 是否绘制RN带宽曲线
            plot_flow: 是否绘制流图

        Returns:
            分析结果字典
        """
        # 创建临时分析器实例
        analyzer = BandwidthAnalyzer.__new__(BandwidthAnalyzer)

        # 初始化基本属性
        analyzer.requests = []
        analyzer.rn_positions = set()
        analyzer.sn_positions = set()
        analyzer.rn_bandwidth_time_series = defaultdict(lambda: {"time": [], "start_times": [], "bytes": []})
        analyzer.plot_rn_bw_fig = plot_rn_bw
        analyzer.plot_flow_graph = plot_flow
        analyzer.finish_cycle = -np.inf
        analyzer.ip_bandwidth_data = None
        analyzer.min_gap_threshold = min_gap_threshold

        # 从CSV加载数据
        analyzer.load_requests_from_csv(csv_folder)

        # 执行分析
        if output_path is None:
            output_path = csv_folder

        # 创建一个临时的sim_model用于保存路径
        class TempBaseModel:
            def __init__(self, save_path, config):
                self.results_fig_save_path = save_path
                self.verbose = True
                self.file_name = "replot"
                self.topo_type_stat = getattr(config, "TOPO_TYPE", "unknown")
                self.flow_fig_show_CDMA = show_cdma  # 默认显示CDMA

        analyzer.sim_model = TempBaseModel(output_path, analyzer.config)

        if plot_rn_bw:
            # 绘制RN带宽曲线
            total_bandwidth = analyzer.plot_rn_bandwidth_curves_work_interval()
            print(f"重新分析得到的总带宽: {total_bandwidth:.2f} GB/s")

        # 绘制流图（如果有网络数据且用户要求）
        if plot_flow and hasattr(analyzer, "temp_network") and analyzer.temp_network:
            # 预计算IP带宽数据
            analyzer.calculate_ip_bandwidth_data()

            # 生成流图
            flow_save_path = os.path.join(output_path, f"flow_graph_{analyzer.config.TOPO_TYPE}_replot.png")
            analyzer.draw_flow_graph(analyzer.temp_network, mode="total", save_path=flow_save_path, show_cdma=show_cdma)
            print(f"网络带宽图已保存: {flow_save_path}")
        elif plot_flow:
            print("警告: 没有找到网络链路数据，无法绘制流图")

        # 计算整体带宽指标
        network_overall = analyzer.calculate_network_overall_bandwidth()
        rn_port_metrics = analyzer.calculate_rn_port_bandwidth()

        results = {
            "network_overall": network_overall,
            "rn_ports": rn_port_metrics,
            "summary": {
                "total_requests": len(analyzer.requests),
                "read_requests": len([r for r in analyzer.requests if r.req_type == "read"]),
                "write_requests": len([r for r in analyzer.requests if r.req_type == "write"]),
                "reanalyzed_from_csv": True,
            },
        }

        if plot_rn_bw:
            results["Total_sum_BW"] = total_bandwidth

        return results

    def _print_summary_to_console(self, results):
        """输出重要数据到控制台"""
        print("\n" + "=" * 60)
        print("网络带宽分析结果摘要")
        print("=" * 60)

        # 网络整体带宽
        read_metrics = results["network_overall"]["read"]
        write_metrics = results["network_overall"]["write"]
        mixed_metrics = results["network_overall"]["mixed"]

        print(f"网络整体带宽:")
        print(f"  读带宽  - 非加权: {read_metrics.unweighted_bandwidth:.3f} GB/s, 加权: {read_metrics.weighted_bandwidth:.3f} GB/s")
        print(f"  写带宽  - 非加权: {write_metrics.unweighted_bandwidth:.3f} GB/s, 加权: {write_metrics.weighted_bandwidth:.3f} GB/s")
        print(f"  混合带宽 - 非加权: {mixed_metrics.unweighted_bandwidth:.3f} GB/s, 加权: {mixed_metrics.weighted_bandwidth:.3f} GB/s")
        print(
            f"  总带宽  - 非加权: {read_metrics.unweighted_bandwidth + write_metrics.unweighted_bandwidth:.3f} GB/s, 加权: {read_metrics.weighted_bandwidth + write_metrics.weighted_bandwidth:.3f} GB/s"
        )
        print(f"  读带宽  - 平均非加权: {read_metrics.unweighted_bandwidth / self.config.NUM_IP:.3f} GB/s, 平均加权: {read_metrics.weighted_bandwidth / self.config.NUM_IP:.3f} GB/s")
        print(f"  写带宽  - 平均非加权: {write_metrics.unweighted_bandwidth / self.config.NUM_IP:.3f} GB/s, 平均加权: {write_metrics.weighted_bandwidth / self.config.NUM_IP:.3f} GB/s")
        print(f"  混合带宽 - 平均非加权: {mixed_metrics.unweighted_bandwidth / self.config.NUM_IP:.3f} GB/s, 平均加权: {mixed_metrics.weighted_bandwidth / self.config.NUM_IP:.3f} GB/s")
        print(
            f"  总带宽  - 平均非加权: {(read_metrics.unweighted_bandwidth + write_metrics.unweighted_bandwidth) / self.config.NUM_IP:.3f} GB/s, 平均加权: {(read_metrics.weighted_bandwidth + write_metrics.weighted_bandwidth) / self.config.NUM_IP:.3f} GB/s"
        )

        # 请求统计
        summary = results["summary"]
        print(f"\n请求统计:")
        print(f"  总请求数: {summary['total_requests']} (读: {summary['read_requests']}, 写: {summary['write_requests']})")
        print(f"  总flit数: {summary['total_read_flits'] + summary['total_write_flits']} (读: {summary['total_read_flits']}, 写: {summary['total_write_flits']})")

        # Circuit统计
        circuit_stats = summary.get("circuit_stats", {})
        if circuit_stats:
            print(f"\n绕环与Tag统计:")
            print(f"  Circuits req  - h: {circuit_stats.get('req_circuits_h', 0)}, v: {circuit_stats.get('req_circuits_v', 0)}")
            print(f"  Circuits rsp  - h: {circuit_stats.get('rsp_circuits_h', 0)}, v: {circuit_stats.get('rsp_circuits_v', 0)}")
            print(f"  Circuits data - h: {circuit_stats.get('data_circuits_h', 0)}, v: {circuit_stats.get('data_circuits_v', 0)}")
            print(f"  Wait cycle req  - h: {circuit_stats.get('req_wait_cycles_h', 0)}, v: {circuit_stats.get('req_wait_cycles_v', 0)}")
            print(f"  Wait cycle rsp  - h: {circuit_stats.get('rsp_wait_cycles_h', 0)}, v: {circuit_stats.get('rsp_wait_cycles_v', 0)}")
            print(f"  Wait cycle data - h: {circuit_stats.get('data_wait_cycles_h', 0)}, v: {circuit_stats.get('data_wait_cycles_v', 0)}")
            print(f"  RB ETag - T1: {circuit_stats.get('RB_ETag_T1_num', 0)}, T0: {circuit_stats.get('RB_ETag_T0_num', 0)}")
            print(f"  EQ ETag - T1: {circuit_stats.get('EQ_ETag_T1_num', 0)}, T0: {circuit_stats.get('EQ_ETag_T0_num', 0)}")
            print(f"  ITag - h: {circuit_stats.get('ITag_h_num', 0)}, v: {circuit_stats.get('ITag_v_num', 0)}")
            print(f"  Retry - read: {circuit_stats.get('read_retry_num', 0)}, write: {circuit_stats.get('write_retry_num', 0)}")

        # 工作区间统计
        print(f"\n工作区间统计:")
        print(f"  读操作工作区间: {len(read_metrics.working_intervals)}")
        print(f"  写操作工作区间: {len(write_metrics.working_intervals)}")
        print(f"  混合操作工作区间: {len(mixed_metrics.working_intervals)}")

        print("=" * 60)

        # 延迟统计 (cmd, data, transaction) - 用新helper
        stats = self._calculate_latency_stats()

        def _avg(cat, op):
            s = stats[cat][op]
            return s["sum"] / s["count"] if s["count"] else 0.0

        print("\n延迟统计 (单位: cycle)")
        for key, label in [("cmd", "CMD"), ("data", "Data"), ("trans", "Trans")]:
            print(
                f"  {label} 延迟  - "
                f"读: avg {_avg(key,'read'):.2f}, max {stats[key]['read']['max']}；"
                f"写: avg {_avg(key,'write'):.2f}, max {stats[key]['write']['max']}；"
                f"混合: avg {_avg(key,'mixed'):.2f}, max {stats[key]['mixed']['max']}"
            )

    def _write_report_header(self, f, results):
        """写入报告头部"""
        summary = results["summary"]
        f.write("=" * 80 + "\n")
        f.write("网络带宽分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("分析概览:\n")
        f.write(f"  总请求数: {summary['total_requests']}\n")
        f.write(f"  读请求数: {summary['read_requests']}\n")
        f.write(f"  写请求数: {summary['write_requests']}\n")
        f.write(f"  读取总flit数: {summary['total_read_flits']}\n")
        f.write(f"  写入总flit数: {summary['total_write_flits']}\n")
        f.write(f"  工作区间阈值: {summary['analysis_config']['min_gap_threshold_ns']} ns\n")

        # 添加Circuit统计信息
        circuit_stats = summary.get("circuit_stats", {})
        if circuit_stats:
            f.write("绕环与Tag统计:\n")
            f.write(f"  Circuits req  - h: {circuit_stats.get('req_circuits_h', 0)}, v: {circuit_stats.get('req_circuits_v', 0)}\n")
            f.write(f"  Circuits rsp  - h: {circuit_stats.get('rsp_circuits_h', 0)}, v: {circuit_stats.get('rsp_circuits_v', 0)}\n")
            f.write(f"  Circuits data - h: {circuit_stats.get('data_circuits_h', 0)}, v: {circuit_stats.get('data_circuits_v', 0)}\n")
            f.write(f"  Wait cycle req  - h: {circuit_stats.get('req_wait_cycles_h', 0)}, v: {circuit_stats.get('req_wait_cycles_v', 0)}\n")
            f.write(f"  Wait cycle rsp  - h: {circuit_stats.get('rsp_wait_cycles_h', 0)}, v: {circuit_stats.get('rsp_wait_cycles_v', 0)}\n")
            f.write(f"  Wait cycle data - h: {circuit_stats.get('data_wait_cycles_h', 0)}, v: {circuit_stats.get('data_wait_cycles_v', 0)}\n")
            f.write(f"  RB ETag - T1: {circuit_stats.get('RB_ETag_T1_num', 0)}, T0: {circuit_stats.get('RB_ETag_T0_num', 0)}\n")
            f.write(f"  EQ ETag - T1: {circuit_stats.get('EQ_ETag_T1_num', 0)}, T0: {circuit_stats.get('EQ_ETag_T0_num', 0)}\n")
            f.write(f"  ITag - h: {circuit_stats.get('ITag_h_num', 0)}, v: {circuit_stats.get('ITag_v_num', 0)}\n")
            f.write(f"  Retry - read: {circuit_stats.get('read_retry_num', 0)}, write: {circuit_stats.get('write_retry_num', 0)}\n")

        # 延迟统计 (cmd, data, transaction)
        stats = self._calculate_latency_stats()

        def _avg(cat, op):
            s = stats[cat][op]
            return s["sum"] / s["count"] if s["count"] else 0.0

        f.write("延迟统计 (cycle)：\n")
        for cat, label in [("cmd", "CMD"), ("data", "Data"), ("trans", "Trans")]:
            rl = stats[cat]
            f.write(
                f"  {label} 延迟 - "
                f"读 avg {_avg(cat,'read'):.2f}, max {rl['read']['max']}; "
                f"写 avg {_avg(cat,'write'):.2f}, max {rl['write']['max']}; "
                f"混合 avg {_avg(cat,'mixed'):.2f}, max {rl['mixed']['max']}\n"
            )

        f.write("\n")

    def _write_network_overall_section(self, f, network_overall):
        """写入网络整体带宽统计部分"""
        f.write("=" * 50 + "\n")
        f.write("网络带宽统计\n")
        f.write("=" * 50 + "\n\n")

        for operation in ["read", "write", "mixed"]:
            metrics = network_overall[operation]
            f.write(f"{operation.upper()} 操作带宽:\n")
            f.write(f"  非加权带宽: {metrics.unweighted_bandwidth:.3f} GB/s, 平均：{metrics.unweighted_bandwidth/self.config.NUM_IP:.3f}\n")
            f.write(f"  加权带宽:   {metrics.weighted_bandwidth:.3f} GB/s, 平均：{metrics.weighted_bandwidth/self.config.NUM_IP:.3f}\n\n")
            f.write(f"  网络工作时间: {metrics.network_start_time} - {metrics.network_end_time} ns (总计 {metrics.network_end_time - metrics.network_start_time} ns)\n")
            f.write(f"  实际工作时间: {metrics.total_working_time} ns\n")
            f.write(f"  工作区间数: {len(metrics.working_intervals)}\n")
            f.write(f"  请求总数: {metrics.total_requests}\n")
            f.write(f"  flit总数: {sum(interval.flit_count for interval in metrics.working_intervals)}\n")

            if metrics.working_intervals:
                f.write(f"  工作区间详情:\n")
                for i, interval in enumerate(metrics.working_intervals):
                    f.write(
                        f"    区间{i+1}: {interval.start_time}-{interval.end_time}ns, "
                        f"持续{interval.duration}ns, {interval.flit_count}个flit, "
                        f"平均带宽{interval.bandwidth_bytes_per_ns / self.config.NUM_IP:.3f}GB/s\n"
                    )
            f.write("\n")

    def _generate_ports_csv(self, rn_ports: Dict[str, PortBandwidthMetrics], output_path: str):
        """
        生成所有端口（RN + SN）详细统计的CSV文件

        ```
        Args:
            rn_ports: RN端口带宽统计字典，key 格式为 "{type}_{node_id}"
            output_path: 输出目录路径
        """
        # 1. 计算 SN 端口带宽并合并
        sn_ports = self.calculate_sn_port_bandwidth()  # :contentReference[oaicite:0]{index=0}
        all_ports = {**rn_ports, **sn_ports}

        # 2. 若无任何端口数据则跳过
        if not all_ports:
            if hasattr(self, "sim_model") and self.sim_model and getattr(self.sim_model, "verbose", False):
                print("没有端口数据，跳过 CSV 生成")
            return

        # 3. 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)

        # 4. CSV 文件头（沿用原定义） :contentReference[oaicite:1]{index=1}
        csv_header = [
            "port_id",
            "coordinate",
            "read_unweighted_bandwidth_gbps",
            "read_weighted_bandwidth_gbps",
            "write_unweighted_bandwidth_gbps",
            "write_weighted_bandwidth_gbps",
            "mixed_unweighted_bandwidth_gbps",
            "mixed_weighted_bandwidth_gbps",
            "read_requests_count",
            "write_requests_count",
            "total_requests_count",
            "read_flits_count",
            "write_flits_count",
            "total_flits_count",
            "read_working_intervals_count",
            "write_working_intervals_count",
            "mixed_working_intervals_count",
            "read_total_working_time_ns",
            "write_total_working_time_ns",
            "mixed_total_working_time_ns",
            "read_network_start_time_ns",
            "read_network_end_time_ns",
            "write_network_start_time_ns",
            "write_network_end_time_ns",
            "mixed_network_start_time_ns",
            "mixed_network_end_time_ns",
        ]

        csv_file = os.path.join(output_path, "ports_bandwidth.csv")
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            # 写入头部
            f.write(",".join(csv_header) + "\n")

            # 5. 排序：先按端口类型字符串，再按节点编号大小
            sorted_ports = sorted(all_ports.items(), key=lambda x: (x[0].split("_")[0], int(x[0].rsplit("_", 1)[1])))

            for port_id, metrics in sorted_ports:
                # 提取节点索引
                idx = int(port_id.rsplit("_", 1)[1])

                # 6. 计算 coordinate
                if getattr(self.sim_model, "topo_type_stat", "").startswith("Ring"):
                    # Ring 拓扑：直接使用节点编号
                    coordinate = str(idx)
                else:
                    # CrossRing 拓扑：从左下角开始算 x/y
                    cols = self.config.NUM_COL
                    rows = self.config.NUM_ROW
                    # 原编号从左上角行优先递增
                    row_from_top = idx // cols
                    col = idx % cols
                    row = rows - 1 - row_from_top
                    coordinate = f"x{col}_y{row}"
                # —— 以上逻辑参考原有实现 :contentReference[oaicite:2]{index=2}

                # 7. 统计 flit 数量
                read_flits = sum(iv.flit_count for iv in metrics.read_metrics.working_intervals) if metrics.read_metrics.working_intervals else 0
                write_flits = sum(iv.flit_count for iv in metrics.write_metrics.working_intervals) if metrics.write_metrics.working_intervals else 0
                mixed_flits = sum(iv.flit_count for iv in metrics.mixed_metrics.working_intervals) if metrics.mixed_metrics.working_intervals else 0

                # 8. 组装 CSV 行
                row_data = [
                    port_id,
                    coordinate,
                    metrics.read_metrics.unweighted_bandwidth,
                    metrics.read_metrics.weighted_bandwidth,
                    metrics.write_metrics.unweighted_bandwidth,
                    metrics.write_metrics.weighted_bandwidth,
                    metrics.mixed_metrics.unweighted_bandwidth,
                    metrics.mixed_metrics.weighted_bandwidth,
                    metrics.read_metrics.total_requests,
                    metrics.write_metrics.total_requests,
                    metrics.mixed_metrics.total_requests,
                    read_flits,
                    write_flits,
                    mixed_flits,
                    len(metrics.read_metrics.working_intervals),
                    len(metrics.write_metrics.working_intervals),
                    len(metrics.mixed_metrics.working_intervals),
                    metrics.read_metrics.total_working_time,
                    metrics.write_metrics.total_working_time,
                    metrics.mixed_metrics.total_working_time,
                    metrics.read_metrics.network_start_time,
                    metrics.read_metrics.network_end_time,
                    metrics.write_metrics.network_start_time,
                    metrics.write_metrics.network_end_time,
                    metrics.mixed_metrics.network_start_time,
                    metrics.mixed_metrics.network_end_time,
                ]
                f.write(",".join(map(str, row_data)) + "\n")
            
            # 添加端口带宽平均值统计
            self._write_port_bandwidth_averages(f, all_ports)

    def _write_port_bandwidth_averages(self, f, all_ports: Dict[str, 'PortBandwidthMetrics']):
        """
        写入端口带宽平均值统计行到CSV文件末尾
        
        Args:
            f: CSV文件句柄
            all_ports: 所有端口的带宽指标字典
        """
        # 按端口类型分组
        port_groups = defaultdict(list)
        for port_id, metrics in all_ports.items():
            port_type = port_id.split('_')[0]  # 提取端口类型 (gdma, sdma, cdma, ddr, l2m)
            port_groups[port_type].append(metrics)
        
        # 写入分隔行
        f.write("\n# Port Bandwidth Averages by Type\n")
        
        # 为每种端口类型计算并写入平均值
        for port_type, metrics_list in sorted(port_groups.items()):
            if not metrics_list:
                continue
                
            # 计算各类带宽的平均值
            read_unweighted_avg = sum(m.read_metrics.unweighted_bandwidth for m in metrics_list) / len(metrics_list)
            read_weighted_avg = sum(m.read_metrics.weighted_bandwidth for m in metrics_list) / len(metrics_list)
            write_unweighted_avg = sum(m.write_metrics.unweighted_bandwidth for m in metrics_list) / len(metrics_list)
            write_weighted_avg = sum(m.write_metrics.weighted_bandwidth for m in metrics_list) / len(metrics_list)
            mixed_unweighted_avg = sum(m.mixed_metrics.unweighted_bandwidth for m in metrics_list) / len(metrics_list)
            mixed_weighted_avg = sum(m.mixed_metrics.weighted_bandwidth for m in metrics_list) / len(metrics_list)
            
            # 计算总请求数和flits平均值
            read_requests_avg = sum(m.read_metrics.total_requests for m in metrics_list) / len(metrics_list)
            write_requests_avg = sum(m.write_metrics.total_requests for m in metrics_list) / len(metrics_list)
            total_requests_avg = sum(m.mixed_metrics.total_requests for m in metrics_list) / len(metrics_list)
            
            # 计算flits平均值
            read_flits_avg = sum(sum(iv.flit_count for iv in m.read_metrics.working_intervals) if m.read_metrics.working_intervals else 0 for m in metrics_list) / len(metrics_list)
            write_flits_avg = sum(sum(iv.flit_count for iv in m.write_metrics.working_intervals) if m.write_metrics.working_intervals else 0 for m in metrics_list) / len(metrics_list)
            mixed_flits_avg = sum(sum(iv.flit_count for iv in m.mixed_metrics.working_intervals) if m.mixed_metrics.working_intervals else 0 for m in metrics_list) / len(metrics_list)
            
            # 计算工作区间平均值
            read_intervals_avg = sum(len(m.read_metrics.working_intervals) for m in metrics_list) / len(metrics_list)
            write_intervals_avg = sum(len(m.write_metrics.working_intervals) for m in metrics_list) / len(metrics_list)
            mixed_intervals_avg = sum(len(m.mixed_metrics.working_intervals) for m in metrics_list) / len(metrics_list)
            
            # 计算工作时间平均值
            read_working_time_avg = sum(m.read_metrics.total_working_time for m in metrics_list) / len(metrics_list)
            write_working_time_avg = sum(m.write_metrics.total_working_time for m in metrics_list) / len(metrics_list)
            mixed_working_time_avg = sum(m.mixed_metrics.total_working_time for m in metrics_list) / len(metrics_list)
            
            # 计算网络时间平均值
            read_start_time_avg = sum(m.read_metrics.network_start_time for m in metrics_list if m.read_metrics.network_start_time > 0) / max(1, len([m for m in metrics_list if m.read_metrics.network_start_time > 0]))
            read_end_time_avg = sum(m.read_metrics.network_end_time for m in metrics_list if m.read_metrics.network_end_time > 0) / max(1, len([m for m in metrics_list if m.read_metrics.network_end_time > 0]))
            write_start_time_avg = sum(m.write_metrics.network_start_time for m in metrics_list if m.write_metrics.network_start_time > 0) / max(1, len([m for m in metrics_list if m.write_metrics.network_start_time > 0]))
            write_end_time_avg = sum(m.write_metrics.network_end_time for m in metrics_list if m.write_metrics.network_end_time > 0) / max(1, len([m for m in metrics_list if m.write_metrics.network_end_time > 0]))
            mixed_start_time_avg = sum(m.mixed_metrics.network_start_time for m in metrics_list if m.mixed_metrics.network_start_time > 0) / max(1, len([m for m in metrics_list if m.mixed_metrics.network_start_time > 0]))
            mixed_end_time_avg = sum(m.mixed_metrics.network_end_time for m in metrics_list if m.mixed_metrics.network_end_time > 0) / max(1, len([m for m in metrics_list if m.mixed_metrics.network_end_time > 0]))
            
            # 组装平均值行数据
            avg_row_data = [
                f"{port_type}_AVG",  # port_id
                f"AVG_of_{len(metrics_list)}_ports",  # coordinate
                f"{read_unweighted_avg:.6f}",
                f"{read_weighted_avg:.6f}",
                f"{write_unweighted_avg:.6f}",
                f"{write_weighted_avg:.6f}",
                f"{mixed_unweighted_avg:.6f}",
                f"{mixed_weighted_avg:.6f}",
                f"{read_requests_avg:.2f}",
                f"{write_requests_avg:.2f}",
                f"{total_requests_avg:.2f}",
                f"{read_flits_avg:.2f}",
                f"{write_flits_avg:.2f}",
                f"{mixed_flits_avg:.2f}",
                f"{read_intervals_avg:.2f}",
                f"{write_intervals_avg:.2f}",
                f"{mixed_intervals_avg:.2f}",
                f"{read_working_time_avg:.2f}",
                f"{write_working_time_avg:.2f}",
                f"{mixed_working_time_avg:.2f}",
                f"{read_start_time_avg:.2f}",
                f"{read_end_time_avg:.2f}",
                f"{write_start_time_avg:.2f}",
                f"{write_end_time_avg:.2f}",
                f"{mixed_start_time_avg:.2f}",
                f"{mixed_end_time_avg:.2f}",
            ]
            f.write(",".join(avg_row_data) + "\n")

    def _generate_detailed_request_csv(self, output_path: str):
        """
        生成详细请求记录的CSV文件，读写分开

        Args:
            output_path: 输出目录路径
        """
        # 分离读写请求
        read_requests = [req for req in self.requests if req.req_type == "read"]
        write_requests = [req for req in self.requests if req.req_type == "write"]

        # CSV文件头部
        csv_header = [
            "packet_id",
            "start_time_ns",
            "end_time_ns",
            "source_node",
            "source_type",
            "dest_node",
            "dest_type",
            "burst_length",
            "cmd_latency_ns",
            "data_latency_ns",
            "transaction_latency_ns",
        ]

        # 生成读请求CSV
        read_csv_file = os.path.join(output_path, "read_requests.csv")
        with open(read_csv_file, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(csv_header) + "\n")
            for req in read_requests:
                row = [
                    req.packet_id,
                    req.start_time,
                    req.end_time,
                    req.source_node,
                    req.source_type,
                    req.dest_node,
                    req.dest_type,
                    req.burst_length,
                    req.cmd_latency,
                    req.data_latency,
                    req.transaction_latency,
                ]
                f.write(",".join(map(str, row)) + "\n")

        # 生成写请求CSV
        write_csv_file = os.path.join(output_path, "write_requests.csv")
        with open(write_csv_file, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(csv_header) + "\n")
            for req in write_requests:
                row = [
                    req.packet_id,
                    req.start_time,
                    req.end_time,
                    req.source_node,
                    req.source_type,
                    req.dest_node,
                    req.dest_type,
                    req.burst_length,
                    req.cmd_latency,
                    req.data_latency,
                    req.transaction_latency,
                ]
                f.write(",".join(map(str, row)) + "\n")

        # 输出统计信息
        if hasattr(self, "sim_model") and self.sim_model and hasattr(self.sim_model, "verbose") and self.sim_model.verbose:
            print(f"详细请求记录统计:")
            print(f"  读请求CSV, {len(read_requests)} 条记录:  {read_csv_file}")
            print(f"  写请求CSV, {len(write_requests)} 条记录:  {write_csv_file}")

    def _generate_json_report(self, results: Dict, json_file: str):
        """生成JSON格式的详细数据报告"""
        # 转换为可序列化的格式
        serializable_results = {}

        # 网络整体数据 - 包含混合带宽
        serializable_results["network_overall"] = {}
        for op_type, metrics in results["network_overall"].items():
            serializable_results["network_overall"][op_type] = {
                "unweighted_bandwidth_gbps": metrics.unweighted_bandwidth,
                "weighted_bandwidth_gbps": metrics.weighted_bandwidth,
                "total_working_time_ns": metrics.total_working_time,
                "network_start_time_ns": metrics.network_start_time,
                "network_end_time_ns": metrics.network_end_time,
                "total_bytes": metrics.total_bytes,
                "total_requests": metrics.total_requests,
                "working_intervals": [
                    {
                        "start_time_ns": interval.start_time,
                        "end_time_ns": interval.end_time,
                        "duration_ns": interval.duration,
                        "flit_count": interval.flit_count,
                        "total_bytes": interval.total_bytes,
                        "request_count": interval.request_count,
                        "bandwidth_bytes_per_ns": interval.bandwidth_bytes_per_ns,
                    }
                    for interval in metrics.working_intervals
                ],
            }

        # RN端口数据 - 包含混合带宽
        serializable_results["rn_ports"] = {}
        for port_id, port_metrics in results["rn_ports"].items():
            serializable_results["rn_ports"][port_id] = {
                "read": {
                    "unweighted_bandwidth_gbps": port_metrics.read_metrics.unweighted_bandwidth,
                    "weighted_bandwidth_gbps": port_metrics.read_metrics.weighted_bandwidth,
                    "total_requests": port_metrics.read_metrics.total_requests,
                    "total_bytes": port_metrics.read_metrics.total_bytes,
                },
                "write": {
                    "unweighted_bandwidth_gbps": port_metrics.write_metrics.unweighted_bandwidth,
                    "weighted_bandwidth_gbps": port_metrics.write_metrics.weighted_bandwidth,
                    "total_requests": port_metrics.write_metrics.total_requests,
                    "total_bytes": port_metrics.write_metrics.total_bytes,
                },
                "mixed": {
                    "unweighted_bandwidth_gbps": port_metrics.mixed_metrics.unweighted_bandwidth,
                    "weighted_bandwidth_gbps": port_metrics.mixed_metrics.weighted_bandwidth,
                    "total_requests": port_metrics.mixed_metrics.total_requests,
                    "total_bytes": port_metrics.mixed_metrics.total_bytes,
                },
            }

        # 汇总数据
        serializable_results["summary"] = results["summary"]

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    def _get_merged_ip_bandwidth_from_existing_data(self, node_id):
        """
        从现有的 self.ip_bandwidth_data 中获取指定Ring节点的合并IP带宽
        这是使用现有带宽计算机制的正确方法

        Args:
            node_id: Ring节点ID

        Returns:
            dict: {ip_type: total_bandwidth} 合并后的IP带宽字典
        """
        merged_bandwidth = {}

        if hasattr(self.config, "NUM_COL") and self.config.NUM_COL > 0:
            # 方案1: 环形节点 → 两列网格映射
            half_nodes = self.config.RING_NUM_NODE // 2
            if node_id < half_nodes:
                # 左边列: 0,1,2,3,4 → (行=节点号, 列=0)
                r, c = node_id, 0
            else:
                # 右边列: 9,8,7,6,5 → (行=总节点数-1-节点号, 列=1)
                r, c = (self.config.RING_NUM_NODE - 1 - node_id), 1
        else:
            # 方案2: 默认映射
            r, c = 0, 0

        # 确保索引不超出矩阵范围
        if "sdma" in self.ip_bandwidth_data["total"]:
            max_r, max_c = self.ip_bandwidth_data["total"]["sdma"].shape
            r = min(r, max_r - 1)
            c = min(c, max_c - 1)
            # print(f"节点 {node_id} 映射到网格坐标 ({r}, {c})")

        # 从已计算的 'total' 数据中提取各IP类型的带宽
        for svc in ["sdma", "gdma", "cdma", "ddr", "l2m"]:
            if svc in self.ip_bandwidth_data["total"]:
                mat = self.ip_bandwidth_data["total"][svc]
                if mat.shape[0] > r and mat.shape[1] > c:
                    bw = mat[r, c]
                    if bw > 0:
                        merged_bandwidth[svc] = bw
                        # print(f"  {svc.upper()}: {bw:.2f} GB/s")

        # 如果Ring节点有多个同类型的IP实例，需要合并它们的带宽
        if hasattr(self, "sim_model") and hasattr(self.sim_model, "ip_modules"):
            # 从Ring拓扑的ip_modules中获取该节点的实际IP连接
            node_ip_types = []

            for (ip_type, ip_pos), ip_interface in self.sim_model.ip_modules.items():
                if ip_pos == node_id:
                    node_ip_types.append(ip_type)

            # 按前缀分组并合并带宽
            ip_groups = {}
            for ip_type in node_ip_types:
                if isinstance(ip_type, str):
                    prefix = ip_type.split("_")[0].lower()
                    if prefix not in ip_groups:
                        ip_groups[prefix] = []
                    ip_groups[prefix].append(ip_type)

        return merged_bandwidth

    def draw_ring_flow_graph(self, ring_network, save_path=None):
        """
        绘制 Ring 拓扑流图 - n×2 矩形布局
        使用现有的带宽计算机制，从 self.ip_bandwidth_data 中获取已计算的带宽数据
        顺时针箭头绘制在外环，逆时针箭头绘制在内环
        """
        # 确保 IP 带宽数据已计算
        self.precalculate_ip_bandwidth_data()

        # 准备画布
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.set_aspect("equal")
        ax.axis("off")

        # 获取节点数
        num_nodes = len(ring_network.ring_nodes)

        # 合并所有模式下的链路流量
        links_stat = {}
        if hasattr(ring_network, "links_flow_stat"):
            for mode_links in ring_network.links_flow_stat.values():
                for edge, val in mode_links.items():
                    links_stat[edge] = links_stat.get(edge, 0) + val
        else:
            print("警告: ring_network 没有 links_flow_stat 属性")

        max_flow = max(links_stat.values()) if links_stat else 1.0

        # 计算节点位置 - n×2 矩形布局
        pos = {}
        cols = 2
        rows = (num_nodes + 1) // cols
        x_spacing, y_spacing = 5.0, 2.5
        node_w, node_h = 2.0, 1.5

        # 左列从上到下
        for i in range(rows):
            if i < num_nodes:
                pos[i] = (0, (rows - 1 - i) * y_spacing)
        # 右列从下到上
        for i in range(rows, num_nodes):
            right_index = i - rows
            pos[i] = (x_spacing, right_index * y_spacing)

        # 箭头绘制辅助函数
        def get_arrow_props(flow_val):
            if flow_val > 0:
                intensity = min(max(flow_val / max_flow, 0.2), 1.0)
                width = max(2.0, 4 * intensity)
                alpha = 0.8
            else:
                intensity = 0.1
                width = 1.5
                alpha = 0.4
            return width, intensity, alpha

        def get_edge_connection_points(x1, y1, x2, y2, w, h):
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) > abs(dy):  # 水平连接
                if dx > 0:  # 向右
                    return x1 + w / 2, y1, x2 - w / 2, y2
                else:  # 向左
                    return x1 - w / 2, y1, x2 + w / 2, y2
            else:  # 垂直连接
                if dy > 0:  # 向上
                    return x1, y1 + h / 2, x2, y2 - h / 2
                else:  # 向下
                    return x1, y1 - h / 2, x2, y2 + h / 2

        offset_size = 0.28
        for i in range(num_nodes):
            next_node = (i + 1) % num_nodes
            cw_flow = links_stat.get((i, next_node), 0)
            ccw_flow = links_stat.get((next_node, i), 0)
            if i not in pos or next_node not in pos:
                continue
            x1, y1 = pos[i]
            x2, y2 = pos[next_node]

            # 获取连接点
            sx, sy, ex, ey = get_edge_connection_points(x1, y1, x2, y2, node_w, node_h)
            # 计算法向量：外侧和内侧偏移
            dxl, dyl = ex - sx, ey - sy
            length = np.hypot(dxl, dyl)
            if length > 0:
                perp_out_x, perp_out_y = dyl / length, -dxl / length
                perp_in_x, perp_in_y = -dyl / length, dxl / length
            else:
                perp_out_x = perp_out_y = perp_in_x = perp_in_y = 0

            # 顺时针箭头（外环）
            if cw_flow > 0 or ccw_flow == 0:
                width, intensity, alpha = get_arrow_props(cw_flow)
                color = (intensity, 0, 0) if cw_flow > 0 else (0.6, 0.6, 0.6)
                off_x, off_y = perp_out_x * offset_size, perp_out_y * offset_size
                fsx, fsy = sx + off_x, sy + off_y
                fex, fey = ex + off_x, ey + off_y
                arrow = FancyArrowPatch((fsx, fsy), (fex, fey), arrowstyle="-|>", linewidth=width, color=color, alpha=alpha, mutation_scale=20, zorder=5)
                ax.add_patch(arrow)
                if cw_flow > 0:
                    mid_x, mid_y = (fsx + fex) / 2, (fsy + fey) / 2
                    bw = (cw_flow * 128) / (self.finish_cycle // self.config.NETWORK_FREQUENCY) if getattr(self, "finish_cycle", 0) > 0 else 0
                    ax.text(mid_x + off_x * 1.8, mid_y + off_y * 1.8, f"{bw:.1f}", fontsize=12, ha="center", va="center", fontweight="bold", zorder=6)

            # 逆时针箭头（内环）
            if ccw_flow > 0:
                width, intensity, alpha = get_arrow_props(ccw_flow)
                color = (0, 0, intensity)
                # 反向连接点
                rsx, rsy, rex, rey = get_edge_connection_points(x2, y2, x1, y1, node_w, node_h)
                off_x, off_y = perp_in_x * offset_size, perp_in_y * offset_size
                fsx2, fsy2 = rsx + off_x, rsy + off_y
                fex2, fey2 = rex + off_x, rey + off_y
                arrow2 = FancyArrowPatch((fsx2, fsy2), (fex2, fey2), arrowstyle="-|>", linewidth=width, color=color, alpha=alpha, mutation_scale=20, zorder=5)
                ax.add_patch(arrow2)
                mid_x2, mid_y2 = (fsx2 + fex2) / 2, (fsy2 + fey2) / 2
                bw2 = (ccw_flow * 128) / (self.finish_cycle // self.config.NETWORK_FREQUENCY) if getattr(self, "finish_cycle", 0) > 0 else 0
                ax.text(mid_x2 + off_x * 1.8, mid_y2 + off_y * 1.8, f"{bw2:.1f}", fontsize=12, ha="center", va="center", fontweight="bold", zorder=6)

        # 绘制节点、框和 IP 带宽信息
        for nid in ring_network.ring_nodes:
            if nid not in pos:
                continue
            x, y = pos[nid]
            rect = Rectangle((x - node_w / 2, y - node_h / 2), node_w, node_h, edgecolor="black", facecolor="lightcyan", linewidth=2.0, zorder=4)
            ax.add_patch(rect)
            merged_data = self._get_merged_ip_bandwidth_from_existing_data(nid)
            lines = [f"{k.upper()}:{v:.1f}" for k, v in merged_data.items() if v > 0.1]
            if lines:
                max_lines = 5
                disp = lines[:max_lines]
                lh = (node_h - 0.4) / max(len(disp), 1)
                fs = max(8, min(11, int(16 - len(disp))))
                for j, txt in enumerate(disp):
                    ty = y + node_h / 2 - 0.2 - (j + 0.5) * lh
                    ax.text(x, ty, txt, ha="center", va="center", fontsize=fs, fontweight="bold", zorder=7)
                if len(lines) > max_lines:
                    ax.text(x, y - node_h / 2 + 0.15, f"...+{len(lines)-max_lines}", ha="center", va="center", fontsize=8, style="italic", zorder=7)
            else:
                ax.text(x, y, str(nid), ha="center", va="center", fontsize=16, fontweight="bold", zorder=7)

        plt.title("Ring Topology", fontsize=18, fontweight="bold", pad=20)
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            m = 2.0
            ax.set_xlim(min(xs) - m, max(xs) + m)
            ax.set_ylim(min(ys) - m, max(ys) + m)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Ring 矩形布局流图已保存: {save_path}")
        else:
            plt.show()
        # return fig


# 便捷使用函数
def analyze_bandwidth(sim_model, config, output_path: str = "./bandwidth_analysis", min_gap_threshold: int = 50, plot_rn_bw_fig: bool = False, plot_flow_graph: bool = False) -> Dict:
    """
    便捷的带宽分析函数

    Args:
        sim_model: BaseModel实例
        config: 配置对象
        output_path: 输出路径
        min_gap_threshold: 工作区间合并阈值(ns)

    Returns:
        分析结果字典
    """
    # 创建分析器
    analyzer = BandwidthAnalyzer(config, min_gap_threshold, plot_rn_bw_fig)

    # 收集数据
    analyzer.collect_requests_data(sim_model)

    # 执行分析
    results = analyzer.analyze_all_bandwidth()

    # 生成报告
    analyzer.generate_unified_report(results, output_path)

    return results


def replot_from_result_folder(csv_folder: str, plot_rn_bw: bool = True, plot_flow: bool = True, output_filename: str = None, show_cdma: bool = False, min_gap_threshold=50) -> Dict:
    """
    便捷函数：从CSV文件夹重新分析并绘制所有图表

    Args:
        csv_folder: 包含CSV文件的文件夹路径
        plot_rn_bw: 是否绘制RN带宽曲线
        plot_flow: 是否绘制流图
        output_filename: 输出图片文件名前缀

    Returns:
        分析结果字典
    """
    output_path = csv_folder

    results = BandwidthAnalyzer.reanalyze_and_plot_from_csv(csv_folder, output_path, plot_rn_bw=plot_rn_bw, plot_flow=plot_flow, show_cdma=show_cdma, min_gap_threshold=min_gap_threshold)

    return results


def batch_replot_all_from_csv_folders(parent_folder: str, pattern: str = "*bandwidth_analysis*", plot_rn_bw: bool = True, plot_flow: bool = True, min_gap_threshold=50):
    """批量从CSV重新绘制所有图表"""
    import glob

    folders = glob.glob(os.path.join(parent_folder, pattern))

    results_summary = []
    for folder in folders:
        if os.path.isdir(folder):
            try:
                start_time = time.time()
                results = replot_from_result_folder(folder, plot_rn_bw, plot_flow, min_gap_threshold=min_gap_threshold)
                end_time = time.time()

                folder_name = os.path.basename(folder)
                processing_time = end_time - start_time
                total_bw = results.get("Total_sum_BW", 0.0)

                results_summary.append({"folder": folder_name, "bandwidth": total_bw, "time": processing_time, "status": "success"})

                plots_generated = []
                if plot_rn_bw:
                    plots_generated.append("RN带宽")
                if plot_flow:
                    plots_generated.append("流图")

                print(f"✓ {folder_name}: {total_bw:.2f} GB/s, " f"已生成: {', '.join(plots_generated)} (耗时: {processing_time:.2f}s)")

            except Exception as e:
                folder_name = os.path.basename(folder)
                results_summary.append({"folder": folder_name, "bandwidth": 0.0, "time": 0.0, "status": f"error: {str(e)}"})
                print(f"✗ {folder_name}: 错误 - {e}")

    # 输出汇总
    print(f"\n批量处理完成，共处理 {len(results_summary)} 个文件夹")
    total_time = sum(r["time"] for r in results_summary if r["status"] == "success")
    success_count = len([r for r in results_summary if r["status"] == "success"])
    print(f"成功: {success_count}, 总耗时: {total_time:.2f}s")

    return results_summary


# 使用示例
def main():
    # 1. 从CSV重新分析和绘图
    total_bw = replot_from_result_folder(
        r"../../Result/CrossRing/TMB/5x4/p_MLP_MoE_All2All_Dispatch",
        show_cdma=True,
        min_gap_threshold=1000,
    )

    # # 2. 批量处理多个结果文件夹
    # summary = batch_replot_from_csv_folders("./all_results")

    # # # 3. 更详细的控制
    # results = BandwidthAnalyzer.reanalyze_and_plot_from_csv(
    #     r"../../Result/CrossRing/REQ_RSP/5x4/Add",
    #     plot_rn_bw=True,
    # )


if __name__ == "__main__":
    main()
