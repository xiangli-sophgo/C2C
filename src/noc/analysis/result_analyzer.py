"""
NoC结果分析器
通用的NoC性能分析工具，包含带宽、延迟、流量分析等功能
支持多种NoC拓扑（CrossRing、Mesh等）
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
import networkx as nx
import logging
import time
import os
import json
from dataclasses import dataclass
from enum import Enum


class RequestType(Enum):
    """请求类型"""

    READ = "read"
    WRITE = "write"
    ALL = "all"


@dataclass
class RequestInfo:
    """请求信息数据结构（与老版本兼容）"""

    packet_id: str
    start_time: int  # ns
    end_time: int  # ns
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
    """工作区间数据结构（与老版本兼容）"""

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


class ResultAnalyzer:
    """通用NoC结果分析器"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        plt.rcParams["font.sans-serif"] = [
            "SimHei",  # 黑体
            "Microsoft YaHei",  # 微软雅黑
            "WenQuanYi Zen Hei",  # 文泉驿正黑
            "Noto Sans CJK SC",  # 思源黑体
            "PingFang SC",  # 苹方
            "Heiti SC",  # 黑体-简
            "Arial Unicode MS",  # 一种包含多种字符的字体
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        
        # 用于存储带宽时间序列数据（仿照旧版本）
        self.bandwidth_time_series = defaultdict(lambda: {"time": [], "start_times": [], "bytes": []})

    def convert_tracker_to_request_info(self, request_tracker, config) -> List[RequestInfo]:
        """转换RequestTracker数据为RequestInfo格式（使用正确的延迟计算）"""
        requests = []

        # 获取配置参数
        network_frequency = getattr(config.basic_config, "NETWORK_FREQUENCY", 2.0) if hasattr(config, "basic_config") else 2.0
        # cycle时间：1000 / frequency (ns per cycle) - 例如2GHz = 0.5ns per cycle
        cycle_time_ns = 1000.0 / (network_frequency * 1000)  # frequency是GHz，转换为ns

        for req_id, lifecycle in request_tracker.completed_requests.items():
            # 使用实际的时间戳
            # 时间转换：cycle -> ns
            start_time = int(lifecycle.created_cycle * cycle_time_ns)
            end_time = int(lifecycle.completed_cycle * cycle_time_ns)
            rn_end_time = end_time  # 简化处理
            sn_end_time = end_time  # 简化处理

            # 提取source_type和dest_type
            source_type = "unknown"
            dest_type = "unknown"
            
            # 直接从所有flits中收集时间戳
            cmd_entry_cake0_cycle = np.inf
            cmd_entry_noc_from_cake0_cycle = np.inf
            cmd_entry_noc_from_cake1_cycle = np.inf
            cmd_received_by_cake0_cycle = np.inf
            cmd_received_by_cake1_cycle = np.inf
            data_entry_noc_from_cake0_cycle = np.inf
            data_entry_noc_from_cake1_cycle = np.inf
            data_received_complete_cycle = np.inf
            
            # 从request flits中收集时间戳和IP类型
            for flit in lifecycle.request_flits:
                # 提取IP类型信息
                if hasattr(flit, 'source_type') and source_type == "unknown":
                    source_type = flit.source_type
                if hasattr(flit, 'destination_type') and dest_type == "unknown":
                    dest_type = flit.destination_type
                
                # 收集时间戳
                if hasattr(flit, 'cmd_entry_cake0_cycle') and flit.cmd_entry_cake0_cycle < np.inf:
                    cmd_entry_cake0_cycle = min(cmd_entry_cake0_cycle, flit.cmd_entry_cake0_cycle)
                if hasattr(flit, 'cmd_entry_noc_from_cake0_cycle') and flit.cmd_entry_noc_from_cake0_cycle < np.inf:
                    cmd_entry_noc_from_cake0_cycle = min(cmd_entry_noc_from_cake0_cycle, flit.cmd_entry_noc_from_cake0_cycle)
                if hasattr(flit, 'cmd_entry_noc_from_cake1_cycle') and flit.cmd_entry_noc_from_cake1_cycle < np.inf:
                    cmd_entry_noc_from_cake1_cycle = min(cmd_entry_noc_from_cake1_cycle, flit.cmd_entry_noc_from_cake1_cycle)
                if hasattr(flit, 'cmd_received_by_cake0_cycle') and flit.cmd_received_by_cake0_cycle < np.inf:
                    cmd_received_by_cake0_cycle = min(cmd_received_by_cake0_cycle, flit.cmd_received_by_cake0_cycle)
                if hasattr(flit, 'cmd_received_by_cake1_cycle') and flit.cmd_received_by_cake1_cycle < np.inf:
                    cmd_received_by_cake1_cycle = min(cmd_received_by_cake1_cycle, flit.cmd_received_by_cake1_cycle)
            
            # 从response flits中收集时间戳
            for flit in lifecycle.response_flits:
                if hasattr(flit, 'cmd_received_by_cake0_cycle') and flit.cmd_received_by_cake0_cycle < np.inf:
                    cmd_received_by_cake0_cycle = min(cmd_received_by_cake0_cycle, flit.cmd_received_by_cake0_cycle)
                if hasattr(flit, 'cmd_received_by_cake1_cycle') and flit.cmd_received_by_cake1_cycle < np.inf:
                    cmd_received_by_cake1_cycle = min(cmd_received_by_cake1_cycle, flit.cmd_received_by_cake1_cycle)
            
            # 从data flits中收集时间戳
            for flit in lifecycle.data_flits:
                if hasattr(flit, 'data_entry_noc_from_cake0_cycle') and flit.data_entry_noc_from_cake0_cycle < np.inf:
                    data_entry_noc_from_cake0_cycle = min(data_entry_noc_from_cake0_cycle, flit.data_entry_noc_from_cake0_cycle)
                if hasattr(flit, 'data_entry_noc_from_cake1_cycle') and flit.data_entry_noc_from_cake1_cycle < np.inf:
                    data_entry_noc_from_cake1_cycle = min(data_entry_noc_from_cake1_cycle, flit.data_entry_noc_from_cake1_cycle)
                if hasattr(flit, 'data_received_complete_cycle') and flit.data_received_complete_cycle < np.inf:
                    data_received_complete_cycle = min(data_received_complete_cycle, flit.data_received_complete_cycle)
            
            # 按照BaseFlit的calculate_latencies方法计算延迟
            cmd_latency = np.inf
            data_latency = np.inf
            transaction_latency = np.inf
            
            # 调试：打印收集到的时间戳
            if req_id == list(request_tracker.completed_requests.keys())[0]:  # 只打印第一个请求
                self.logger.debug(f"请求 {req_id} 时间戳:")
                self.logger.debug(f"  cmd_entry_cake0_cycle: {cmd_entry_cake0_cycle}")
                self.logger.debug(f"  cmd_entry_noc_from_cake0_cycle: {cmd_entry_noc_from_cake0_cycle}")
                self.logger.debug(f"  cmd_received_by_cake1_cycle: {cmd_received_by_cake1_cycle}")
                self.logger.debug(f"  data_entry_noc_from_cake0_cycle: {data_entry_noc_from_cake0_cycle}")
                self.logger.debug(f"  data_entry_noc_from_cake1_cycle: {data_entry_noc_from_cake1_cycle}")
                self.logger.debug(f"  data_received_complete_cycle: {data_received_complete_cycle}")
                self.logger.debug(f"  lifecycle中的flit数量: req={len(lifecycle.request_flits)}, rsp={len(lifecycle.response_flits)}, data={len(lifecycle.data_flits)}")
            
            # 命令延迟：cmd_received_by_cake1_cycle - cmd_entry_noc_from_cake0_cycle
            if cmd_entry_noc_from_cake0_cycle < np.inf and cmd_received_by_cake1_cycle < np.inf:
                cmd_latency = cmd_received_by_cake1_cycle - cmd_entry_noc_from_cake0_cycle
            
            # 数据延迟：根据读写类型不同
            if lifecycle.op_type == "read":
                # 读操作：data_received_complete_cycle - data_entry_noc_from_cake1_cycle
                if data_entry_noc_from_cake1_cycle < np.inf and data_received_complete_cycle < np.inf:
                    data_latency = data_received_complete_cycle - data_entry_noc_from_cake1_cycle
            elif lifecycle.op_type == "write":
                # 写操作：data_received_complete_cycle - data_entry_noc_from_cake0_cycle
                if data_entry_noc_from_cake0_cycle < np.inf and data_received_complete_cycle < np.inf:
                    data_latency = data_received_complete_cycle - data_entry_noc_from_cake0_cycle
            
            # 事务延迟：data_received_complete_cycle - cmd_entry_cake0_cycle
            if cmd_entry_cake0_cycle < np.inf and data_received_complete_cycle < np.inf:
                transaction_latency = data_received_complete_cycle - cmd_entry_cake0_cycle
            
            # 将cycle延迟转换为ns
            cmd_latency_ns = int(cmd_latency * cycle_time_ns) if cmd_latency < np.inf else 0
            data_latency_ns = int(data_latency * cycle_time_ns) if data_latency < np.inf else 0
            transaction_latency_ns = int(transaction_latency * cycle_time_ns) if transaction_latency < np.inf else 0

            # 计算字节数
            burst_length = lifecycle.burst_size
            total_bytes = burst_length * 128  # 128字节/flit

            request_info = RequestInfo(
                packet_id=str(req_id),
                start_time=start_time,
                end_time=end_time,
                rn_end_time=rn_end_time,
                sn_end_time=sn_end_time,
                req_type=lifecycle.op_type,
                source_node=lifecycle.source,
                dest_node=lifecycle.destination,
                source_type=source_type,
                dest_type=dest_type,
                burst_length=burst_length,
                total_bytes=total_bytes,
                cmd_latency=cmd_latency_ns,
                data_latency=data_latency_ns,
                transaction_latency=transaction_latency_ns,
            )
            requests.append(request_info)

        return requests

    def calculate_working_intervals(self, requests: List[RequestInfo], min_gap_threshold: int = 200) -> List[WorkingInterval]:
        """计算工作区间（按照老版本逻辑）"""
        if not requests:
            return []

        # 创建时间点事件列表
        events = []
        for req in requests:
            events.append((req.start_time, "start", req.packet_id))
            events.append((req.end_time, "end", req.packet_id))

        # 按时间排序
        events.sort()

        # 识别工作区间
        raw_intervals = []
        active_requests = set()
        current_start = None

        for time_point, event_type, packet_id in events:
            if event_type == "start":
                if not active_requests:
                    current_start = time_point
                active_requests.add(packet_id)
            else:  # 'end'
                active_requests.discard(packet_id)
                if not active_requests and current_start is not None:
                    # 工作区间结束
                    raw_intervals.append((current_start, time_point))
                    current_start = None

        # 处理最后未结束的区间
        if active_requests and current_start is not None:
            last_end = max(req.end_time for req in requests)
            raw_intervals.append((current_start, last_end))

        # 合并相近区间
        merged_intervals = self._merge_close_intervals(raw_intervals, min_gap_threshold)

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

    def _merge_close_intervals(self, intervals: List[tuple], min_gap_threshold: int) -> List[tuple]:
        """合并相近的时间区间（按照老版本逻辑）"""
        if not intervals:
            return []

        # 按开始时间排序
        sorted_intervals = sorted(intervals)
        merged = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]

            # 如果间隙小于阈值，则合并
            if current_start - last_end <= min_gap_threshold:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    def calculate_bandwidth_metrics(self, requests: List[RequestInfo], operation_type: str = None, min_gap_threshold: int = 200) -> Dict[str, Any]:
        """计算带宽指标（按照老版本完整逻辑）"""
        if not requests:
            return {}

        # 筛选请求
        filtered_requests = []
        for req in requests:
            if operation_type is not None and req.req_type != operation_type:
                continue
            filtered_requests.append(req)

        if not filtered_requests:
            return {}

        # 计算工作区间
        working_intervals = self.calculate_working_intervals(filtered_requests, min_gap_threshold)

        # 网络工作时间窗口
        network_start = min(req.start_time for req in filtered_requests)
        network_end = max(req.end_time for req in filtered_requests)
        total_network_time = network_end - network_start

        # 总工作时间和总字节数
        total_working_time = sum(interval.duration for interval in working_intervals)
        total_bytes = sum(req.total_bytes for req in filtered_requests)

        # 计算非加权带宽：总数据量 / 网络总时间
        unweighted_bandwidth = (total_bytes / total_network_time) if total_network_time > 0 else 0.0

        # 计算加权带宽：各区间带宽按flit数量加权平均
        if working_intervals:
            total_weighted_bandwidth = 0.0
            total_weight = 0

            for interval in working_intervals:
                weight = interval.flit_count  # 权重是工作时间段的flit数量
                bandwidth = interval.bandwidth_bytes_per_ns  # bytes/ns
                total_weighted_bandwidth += bandwidth * weight
                total_weight += weight

            weighted_bandwidth = (total_weighted_bandwidth / total_weight) if total_weight > 0 else 0.0
        else:
            weighted_bandwidth = 0.0


        return {
            "非加权带宽_GB/s": f"{unweighted_bandwidth:.2f}",
            "加权带宽_GB/s": f"{weighted_bandwidth:.2f}",
            "总传输字节数": total_bytes,
            "总请求数": len(filtered_requests),
            "工作区间数量": len(working_intervals),
            "总工作时间_ns": total_working_time,
            "网络时间_ns": total_network_time,
        }

    def analyze_bandwidth(self, requests: List[RequestInfo]) -> Dict[str, Any]:
        """分析带宽指标（按照老版本逻辑）"""
        if not requests:
            return {}

        # 总体带宽分析
        overall_metrics = self.calculate_bandwidth_metrics(requests, operation_type=None)

        # 读操作带宽分析
        read_metrics = self.calculate_bandwidth_metrics(requests, operation_type="read")

        # 写操作带宽分析
        write_metrics = self.calculate_bandwidth_metrics(requests, operation_type="write")

        return {"总体带宽": overall_metrics, "读操作带宽": read_metrics, "写操作带宽": write_metrics}

    def analyze_latency(self, metrics) -> Dict[str, Any]:
        """分析延迟指标"""
        if not metrics:
            return {}

        latencies = [m.transaction_latency for m in metrics]
        read_latencies = [m.transaction_latency for m in metrics if m.req_type == "read"]
        write_latencies = [m.transaction_latency for m in metrics if m.req_type == "write"]

        result = {
            "总体延迟": {
                "平均延迟_ns": f"{np.mean(latencies):.2f}",
                "最小延迟_ns": f"{np.min(latencies):.2f}",
                "最大延迟_ns": f"{np.max(latencies):.2f}",
                "P95延迟_ns": f"{np.percentile(latencies, 95):.2f}",
            }
        }

        if read_latencies:
            result["读操作延迟"] = {
                "平均延迟_ns": f"{np.mean(read_latencies):.2f}",
                "最小延迟_ns": f"{np.min(read_latencies):.2f}",
                "最大延迟_ns": f"{np.max(read_latencies):.2f}",
            }

        if write_latencies:
            result["写操作延迟"] = {
                "平均延迟_ns": f"{np.mean(write_latencies):.2f}",
                "最小延迟_ns": f"{np.min(write_latencies):.2f}",
                "最大延迟_ns": f"{np.max(write_latencies):.2f}",
            }

        return result

    def analyze_port_bandwidth(self, metrics) -> Dict[str, Any]:
        """分析端口级别带宽（按IP类型分组）"""
        ip_analysis = defaultdict(lambda: {"read": [], "write": []})

        for metric in metrics:
            # 按源IP类型分组（谁发起的请求）
            source_ip_type = metric.source_type  # 'gdma' 或 'ddr'
            ip_analysis[source_ip_type.upper()][metric.req_type].append(metric)

        ip_summary = {}
        for ip_type, data in ip_analysis.items():
            read_reqs = data["read"]
            write_reqs = data["write"]

            # 计算该IP类型的读写带宽
            read_bw = 0
            if read_reqs:
                read_total_bytes = sum(r.total_bytes for r in read_reqs)
                read_total_time = max(r.end_time for r in read_reqs) - min(r.start_time for r in read_reqs)
                read_bw = read_total_bytes / read_total_time if read_total_time > 0 else 0

            write_bw = 0
            if write_reqs:
                write_total_bytes = sum(r.total_bytes for r in write_reqs)
                write_total_time = max(r.end_time for r in write_reqs) - min(r.start_time for r in write_reqs)
                write_bw = write_total_bytes / write_total_time if write_total_time > 0 else 0

            ip_summary[ip_type] = {
                "读带宽_GB/s": f"{read_bw:.2f}",
                "写带宽_GB/s": f"{write_bw:.2f}",
                "总带宽_GB/s": f"{read_bw + write_bw:.2f}",
                "读请求数": len(read_reqs),
                "写请求数": len(write_reqs),
                "总请求数": len(read_reqs) + len(write_reqs),
            }

        return ip_summary

    def analyze_tag_data(self, model) -> Dict[str, Any]:
        """分析Tag机制数据（I-Tag和E-Tag）"""
        tag_analysis = {
            "I-Tag统计": {"触发次数": 0, "平均等待周期": 0},
            "E-Tag统计": {"升级次数": 0, "T0使用次数": 0, "T1使用次数": 0, "T2使用次数": 0},
            "绕环统计": {"总绕环次数": 0, "平均绕环距离": 0},
        }

        # 从NoC节点中收集Tag统计数据
        total_itag_triggers = 0
        total_etag_upgrades = 0
        total_ring_hops = 0

        try:
            for node in model.nodes.values():
                # 收集I-Tag数据
                if hasattr(node, "horizontal_crosspoint") and hasattr(node.horizontal_crosspoint, "tag_manager"):
                    tag_mgr = node.horizontal_crosspoint.tag_manager
                    if hasattr(tag_mgr, "itag_trigger_count"):
                        total_itag_triggers += getattr(tag_mgr, "itag_trigger_count", 0)

                # 收集E-Tag数据
                if hasattr(node, "vertical_crosspoint") and hasattr(node.vertical_crosspoint, "tag_manager"):
                    tag_mgr = node.vertical_crosspoint.tag_manager
                    if hasattr(tag_mgr, "etag_upgrade_count"):
                        total_etag_upgrades += getattr(tag_mgr, "etag_upgrade_count", 0)

        except Exception as e:
            self.logger.warning(f"收集Tag数据时出错: {e}")

        tag_analysis["I-Tag统计"]["触发次数"] = total_itag_triggers
        tag_analysis["E-Tag统计"]["升级次数"] = total_etag_upgrades
        tag_analysis["绕环统计"]["总绕环次数"] = total_ring_hops

        return tag_analysis

    def _collect_bandwidth_time_series_data(self, metrics):
        """收集带宽时间序列数据（仿照老版本逻辑）"""
        # 清空之前的数据
        self.bandwidth_time_series.clear()
        
        # 按端口类型分组请求
        for req in metrics:
            # 生成端口键名（类似老版本的格式）
            if hasattr(req, 'source_type') and hasattr(req, 'dest_type'):
                if req.req_type == "read":
                    port_key = f"{req.source_type} read {req.dest_type}"
                else:
                    port_key = f"{req.source_type} write {req.dest_type}"
            else:
                # 如果没有端口类型信息，使用读写类型
                port_key = f"{req.req_type}"
            
            # 添加到时间序列数据
            self.bandwidth_time_series[port_key]["time"].append(req.end_time)
            self.bandwidth_time_series[port_key]["start_times"].append(req.start_time)
            self.bandwidth_time_series[port_key]["bytes"].append(req.total_bytes)

    def plot_bandwidth_curves(self, metrics, save_dir: str = "output") -> str:
        """生成带宽时间曲线图（使用累积带宽算法，仿照老版本）"""
        if not metrics:
            return ""

        try:
            # 按端口类型分组数据（仿照老版本的rn_bandwidth_time_series）
            port_time_series = defaultdict(lambda: {"time": [], "start_times": [], "bytes": []})
            
            for metric in metrics:
                # 构造端口标识：格式为 "SOURCE_TYPE REQUEST_TYPE DEST_TYPE"，例如 "GDMA READ DDR"
                port_key = f"{metric.source_type.upper()} {metric.req_type.upper()} {metric.dest_type.upper()}"
                
                port_time_series[port_key]["time"].append(metric.end_time)
                port_time_series[port_key]["start_times"].append(metric.start_time)
                port_time_series[port_key]["bytes"].append(metric.total_bytes)

            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 绘制累积带宽曲线
            total_final_bw = 0
            
            for port_key, data_dict in port_time_series.items():
                if not data_dict["time"]:
                    continue

                # 排序时间戳并去除nan值（仿照老版本逻辑）
                raw_end = np.array(data_dict["time"])
                raw_start = np.array(data_dict["start_times"])
                raw_bytes = np.array(data_dict["bytes"])
                
                # 去除nan值和无效数据
                mask = ~np.isnan(raw_end) & (raw_end > 0)
                end_clean = raw_end[mask]
                start_clean = raw_start[mask]
                bytes_clean = raw_bytes[mask]
                
                if len(end_clean) == 0:
                    continue

                # 同步排序
                sort_idx = np.argsort(end_clean)
                times = end_clean[sort_idx]
                start_times = start_clean[sort_idx]
                bytes_data = bytes_clean[sort_idx]

                # 仿照老版本：使用第一个请求的开始时间作为基准
                if len(start_times) > 0:
                    base_start = start_times[0]
                    rel_times = times - base_start
                    
                    # 防止除以0
                    rel_times[rel_times <= 0] = 1e-9
                    
                    # 计算累积请求数和累积带宽
                    cum_counts = np.arange(1, len(rel_times) + 1)
                    
                    # 仿照老版本：bandwidth = (cum_counts * 128 * BURST) / rel_times
                    # 这里使用实际字节数：累积字节数 / 时间
                    cum_bytes = np.cumsum(bytes_data)
                    bandwidth_bytes_per_ns = cum_bytes / rel_times  # bytes/ns
                    
                    # 转换为GB/s: bytes/ns * 1e9 / (1024^3)
                    bandwidth_gbps = bandwidth_bytes_per_ns * 1e9 / (1024**3)

                    # 绘制曲线（使用绝对时间轴）
                    time_us = times / 1000  # 转换为微秒
                    line, = ax.plot(time_us, bandwidth_gbps, drawstyle="default", label=port_key, linewidth=2)
                    
                    # 在曲线末尾添加数值标注
                    if len(bandwidth_gbps) > 0:
                        final_bw = bandwidth_gbps[-1]
                        ax.text(time_us[-1], final_bw, f"{final_bw:.2f}", 
                               va="center", color=line.get_color(), fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                        total_final_bw += final_bw

            # 设置图表属性
            ax.set_xlabel("时间 (μs)", fontsize=12)
            ax.set_ylabel("带宽 (GB/s)", fontsize=12)
            ax.set_title("CrossRing NoC 累积带宽时间曲线", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

            # 添加总带宽信息
            if total_final_bw > 0:
                ax.text(0.02, 0.98, f"总带宽: {total_final_bw:.2f} GB/s", 
                       transform=ax.transAxes, fontsize=12, va="top", ha="left",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

            # 保存图表
            timestamp = int(time.time())
            save_path = f"{save_dir}/crossring_bandwidth_curve_{timestamp}.png"
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=100)
            plt.close(fig)

            self.logger.info(f"累积带宽曲线图已保存到: {save_path}")
            self.logger.info(f"总带宽: {total_final_bw:.2f} GB/s")
            return save_path

        except Exception as e:
            self.logger.error(f"生成带宽曲线图失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return ""

    def save_detailed_requests_csv(self, metrics, save_dir: str = "output") -> Dict[str, str]:
        """保存详细请求CSV文件（仿照老版本格式）
        
        Returns:
            包含保存文件路径的字典: {"read_requests_csv": path, "write_requests_csv": path}
        """
        if not metrics:
            return {}

        try:
            import csv
            os.makedirs(save_dir, exist_ok=True)
            
            # CSV文件头（与老版本完全一致）
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
            
            # 分离读写请求
            read_requests = [req for req in metrics if req.req_type == "read"]
            write_requests = [req for req in metrics if req.req_type == "write"]
            
            saved_files = {}
            
            # 保存读请求CSV
            if read_requests:
                timestamp = int(time.time())
                read_csv_path = f"{save_dir}/read_requests_{timestamp}.csv"
                
                with open(read_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_header)
                    
                    for req in read_requests:
                        row = [
                            req.packet_id,
                            req.start_time,
                            req.end_time,
                            req.source_node,
                            getattr(req, 'source_type', 'unknown'),
                            req.dest_node,
                            getattr(req, 'dest_type', 'unknown'),
                            req.burst_length,
                            req.cmd_latency,
                            req.data_latency,
                            req.transaction_latency,
                        ]
                        writer.writerow(row)
                
                saved_files["read_requests_csv"] = read_csv_path
                self.logger.info(f"读请求CSV已保存: {read_csv_path} ({len(read_requests)} 条记录)")
            
            # 保存写请求CSV
            if write_requests:
                timestamp = int(time.time())
                write_csv_path = f"{save_dir}/write_requests_{timestamp}.csv"
                
                with open(write_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_header)
                    
                    for req in write_requests:
                        row = [
                            req.packet_id,
                            req.start_time,
                            req.end_time,
                            req.source_node,
                            getattr(req, 'source_type', 'unknown'),
                            req.dest_node,
                            getattr(req, 'dest_type', 'unknown'),
                            req.burst_length,
                            req.cmd_latency,
                            req.data_latency,
                            req.transaction_latency,
                        ]
                        writer.writerow(row)
                
                saved_files["write_requests_csv"] = write_csv_path
                self.logger.info(f"写请求CSV已保存: {write_csv_path} ({len(write_requests)} 条记录)")
            
            return saved_files

        except Exception as e:
            self.logger.error(f"保存详细请求CSV失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return {}

    def save_ports_bandwidth_csv(self, metrics, save_dir: str = "output", config=None) -> str:
        """保存端口带宽CSV文件（仿照老版本格式）
        
        Returns:
            保存的CSV文件路径
        """
        if not metrics:
            return ""

        try:
            import csv
            os.makedirs(save_dir, exist_ok=True)
            
            # CSV文件头（与老版本完全一致）
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
            
            # 按端口分组统计 - 使用具体IP名称和坐标
            port_stats = {}
            
            # 从config获取网格尺寸
            num_cols = getattr(config, 'NUM_COL', 3) if config else 3  # 默认3列
            
            for req in metrics:
                # 统计所有涉及的端口：RN端口（读请求源）和SN端口（写请求目标）
                ports_to_update = []
                
                # 对于每个请求，都要统计RN和SN两个端口
                # RN端口：请求发起者（读/写请求的源）
                source_port_id = req.source_type  # 如 "gdma_0"
                source_node_id = req.source_node
                source_row = source_node_id // num_cols
                source_col = source_node_id % num_cols
                source_coordinate = f"x_{source_col}_y_{source_row}"
                ports_to_update.append((source_port_id, source_node_id, source_coordinate))
                
                # SN端口：请求接收者（读/写请求的目标）
                dest_port_id = req.dest_type    # 如 "ddr_0"  
                dest_node_id = req.dest_node
                dest_row = dest_node_id // num_cols
                dest_col = dest_node_id % num_cols  
                dest_coordinate = f"x_{dest_col}_y_{dest_row}"
                ports_to_update.append((dest_port_id, dest_node_id, dest_coordinate))
                
                # 更新所有相关端口的统计
                for port_id, node_id, coordinate in ports_to_update:
                    if port_id not in port_stats:
                        port_stats[port_id] = {
                            "coordinate": coordinate,
                            "node_id": node_id,
                            "read_requests": [],
                            "write_requests": [],
                            "all_requests": []
                        }
                    
                    port_stats[port_id]["all_requests"].append(req)
                    if req.req_type == "read":
                        port_stats[port_id]["read_requests"].append(req)
                    else:
                        port_stats[port_id]["write_requests"].append(req)
            
            # 生成CSV文件
            timestamp = int(time.time())
            csv_path = f"{save_dir}/ports_bandwidth_{timestamp}.csv"
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_header)
                
                for port_id, stats in port_stats.items():
                    # 计算各种带宽指标
                    read_reqs = stats["read_requests"]
                    write_reqs = stats["write_requests"]
                    all_reqs = stats["all_requests"]
                    
                    # 带宽计算 - 使用工作区间计算，与calculate_bandwidth_metrics完全一致
                    def calc_bandwidth_metrics(requests):
                        if not requests:
                            return {
                                "unweighted_bw": 0.0,
                                "weighted_bw": 0.0,
                                "start_time": 0,
                                "end_time": 0,
                                "total_time": 0,
                                "working_intervals": 0,
                                "flits_count": 0
                            }
                        
                        # 计算工作区间（与calculate_bandwidth_metrics相同的逻辑）
                        working_intervals = self.calculate_working_intervals(requests, min_gap_threshold=200)
                        
                        # 网络工作时间窗口
                        network_start = min(r.start_time for r in requests)
                        network_end = max(r.end_time for r in requests)
                        total_network_time = network_end - network_start
                        
                        # 总工作时间和总字节数
                        total_working_time = sum(interval.duration for interval in working_intervals)
                        total_bytes = sum(r.total_bytes for r in requests)
                        
                        # 计算非加权带宽：总数据量 / 网络总时间
                        unweighted_bandwidth = (total_bytes / total_network_time) if total_network_time > 0 else 0.0
                        
                        # 计算加权带宽：各区间带宽按flit数量加权平均
                        if working_intervals:
                            total_weighted_bandwidth = 0.0
                            total_weight = 0
                            
                            for interval in working_intervals:
                                weight = interval.flit_count  # 权重是工作时间段的flit数量
                                bandwidth = interval.bandwidth_bytes_per_ns  # bytes/ns
                                total_weighted_bandwidth += bandwidth * weight
                                total_weight += weight
                            
                            weighted_bandwidth = (total_weighted_bandwidth / total_weight) if total_weight > 0 else 0.0
                        else:
                            weighted_bandwidth = 0.0
                        
                        return {
                            "unweighted_bw": unweighted_bandwidth,
                            "weighted_bw": weighted_bandwidth,
                            "start_time": network_start,
                            "end_time": network_end,
                            "total_time": total_network_time,
                            "working_intervals": len(working_intervals),
                            "flits_count": sum(r.burst_length for r in requests)
                        }
                    
                    read_metrics = calc_bandwidth_metrics(read_reqs)
                    write_metrics = calc_bandwidth_metrics(write_reqs)
                    mixed_metrics = calc_bandwidth_metrics(all_reqs)
                    
                    row_data = [
                        port_id,
                        stats["coordinate"],
                        read_metrics["unweighted_bw"],
                        read_metrics["weighted_bw"],
                        write_metrics["unweighted_bw"],
                        write_metrics["weighted_bw"],
                        mixed_metrics["unweighted_bw"],
                        mixed_metrics["weighted_bw"],
                        len(read_reqs),
                        len(write_reqs),
                        len(all_reqs),
                        read_metrics["flits_count"],
                        write_metrics["flits_count"],
                        mixed_metrics["flits_count"],
                        read_metrics["working_intervals"],
                        write_metrics["working_intervals"],
                        mixed_metrics["working_intervals"],
                        read_metrics["total_time"],
                        write_metrics["total_time"],
                        mixed_metrics["total_time"],
                        read_metrics["start_time"],
                        read_metrics["end_time"],
                        write_metrics["start_time"],
                        write_metrics["end_time"],
                        mixed_metrics["start_time"],
                        mixed_metrics["end_time"],
                    ]
                    writer.writerow(row_data)
                
                # 计算IP类型汇总统计
                ip_type_aggregates = {}
                for port_id, stats in port_stats.items():
                    # 提取IP类型（去掉数字后缀）
                    ip_type = port_id.split('_')[0]  # "gdma_0" -> "gdma"
                    
                    if ip_type not in ip_type_aggregates:
                        ip_type_aggregates[ip_type] = {
                            "ports": [],
                            "total_read_requests": 0,
                            "total_write_requests": 0,
                            "total_requests": 0,
                            "total_read_flits": 0,
                            "total_write_flits": 0,
                            "total_flits": 0,
                            "read_bandwidth_sum": 0,
                            "write_bandwidth_sum": 0,
                            "mixed_bandwidth_sum": 0,
                        }
                    
                    # 计算该端口的指标
                    read_reqs = stats["read_requests"]
                    write_reqs = stats["write_requests"]
                    all_reqs = stats["all_requests"]
                    
                    read_metrics = calc_bandwidth_metrics(read_reqs)
                    write_metrics = calc_bandwidth_metrics(write_reqs)
                    mixed_metrics = calc_bandwidth_metrics(all_reqs)
                    
                    # 累加到IP类型统计
                    agg = ip_type_aggregates[ip_type]
                    agg["ports"].append(port_id)
                    agg["total_read_requests"] += len(read_reqs)
                    agg["total_write_requests"] += len(write_reqs)
                    agg["total_requests"] += len(all_reqs)
                    agg["total_read_flits"] += read_metrics["flits_count"]
                    agg["total_write_flits"] += write_metrics["flits_count"]
                    agg["total_flits"] += mixed_metrics["flits_count"]
                    agg["read_bandwidth_sum"] += read_metrics["unweighted_bw"]
                    agg["write_bandwidth_sum"] += write_metrics["unweighted_bw"]
                    agg["mixed_bandwidth_sum"] += mixed_metrics["unweighted_bw"]
                
                # 添加IP类型汇总行
                writer.writerow([])  # 空行分隔
                writer.writerow(["=== IP类型汇总统计 ==="])
                
                for ip_type, agg in sorted(ip_type_aggregates.items()):
                    port_count = len(agg["ports"])
                    avg_read_bw = agg["read_bandwidth_sum"] / port_count if port_count > 0 else 0
                    avg_write_bw = agg["write_bandwidth_sum"] / port_count if port_count > 0 else 0
                    avg_mixed_bw = agg["mixed_bandwidth_sum"] / port_count if port_count > 0 else 0
                    
                    summary_row = [
                        f"{ip_type}_AVG",  # port_id格式：gdma_AVG, ddr_AVG
                        f"avg_of_{port_count}_ports",  # coordinate显示端口数
                        avg_read_bw,    # 平均读带宽
                        avg_read_bw,    # 平均读带宽（加权，简化为相同）
                        avg_write_bw,   # 平均写带宽
                        avg_write_bw,   # 平均写带宽（加权，简化为相同）
                        avg_mixed_bw,   # 平均混合带宽
                        avg_mixed_bw,   # 平均混合带宽（加权，简化为相同）
                        agg["total_read_requests"],   # 总读请求数
                        agg["total_write_requests"],  # 总写请求数
                        agg["total_requests"],        # 总请求数
                        agg["total_read_flits"],      # 总读flit数
                        agg["total_write_flits"],     # 总写flit数
                        agg["total_flits"],           # 总flit数
                        port_count,  # 工作区间数用端口数表示
                        port_count,
                        port_count,
                        0, 0, 0,  # 时间相关字段为0（汇总数据）
                        0, 0, 0, 0, 0, 0
                    ]
                    writer.writerow(summary_row)
            
            self.logger.info(f"端口带宽CSV已保存: {csv_path} ({len(port_stats)} 个端口)")
            return csv_path

        except Exception as e:
            self.logger.error(f"保存端口带宽CSV失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return ""

    def plot_latency_distribution(self, metrics, save_dir: str = "output") -> str:
        """生成延迟分布图"""
        if not metrics:
            return ""

        try:
            # 三种延迟类型数据
            cmd_latencies = [m.cmd_latency for m in metrics]
            data_latencies = [m.data_latency for m in metrics]
            transaction_latencies = [m.transaction_latency for m in metrics]
            
            # 按读写操作分类
            read_cmd = [m.cmd_latency for m in metrics if m.req_type == "read"]
            read_data = [m.data_latency for m in metrics if m.req_type == "read"]
            read_transaction = [m.transaction_latency for m in metrics if m.req_type == "read"]
            
            write_cmd = [m.cmd_latency for m in metrics if m.req_type == "write"]
            write_data = [m.data_latency for m in metrics if m.req_type == "write"]
            write_transaction = [m.transaction_latency for m in metrics if m.req_type == "write"]

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # 1. 三种延迟类型对比直方图
            ax1.hist(cmd_latencies, bins=20, alpha=0.7, label="CMD延迟", color="blue")
            ax1.hist(data_latencies, bins=20, alpha=0.7, label="DATA延迟", color="green")
            ax1.hist(transaction_latencies, bins=20, alpha=0.7, label="TRANSACTION延迟", color="red")
            ax1.set_xlabel("延迟 (ns)")
            ax1.set_ylabel("频次")
            ax1.set_title("三种延迟类型分布直方图")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. 延迟类型箱线图
            latency_data = [cmd_latencies, data_latencies, transaction_latencies]
            latency_labels = ["CMD延迟", "DATA延迟", "TRANSACTION延迟"]
            ax2.boxplot(latency_data, labels=latency_labels)
            ax2.set_ylabel("延迟 (ns)")
            ax2.set_title("延迟类型箱线图")
            ax2.grid(True, alpha=0.3)

            # 3. 读操作延迟对比
            if read_cmd or read_data or read_transaction:
                read_data_list = []
                read_labels = []
                if read_cmd:
                    read_data_list.append(read_cmd)
                    read_labels.append("读CMD")
                if read_data:
                    read_data_list.append(read_data)
                    read_labels.append("读DATA")
                if read_transaction:
                    read_data_list.append(read_transaction)
                    read_labels.append("读TRANSACTION")
                    
                if read_data_list:
                    ax3.boxplot(read_data_list, labels=read_labels)
                    ax3.set_ylabel("延迟 (ns)")
                    ax3.set_title("读操作延迟对比")
                    ax3.grid(True, alpha=0.3)

            # 4. 写操作延迟对比
            if write_cmd or write_data or write_transaction:
                write_data_list = []
                write_labels = []
                if write_cmd:
                    write_data_list.append(write_cmd)
                    write_labels.append("写CMD")
                if write_data:
                    write_data_list.append(write_data)
                    write_labels.append("写DATA")
                if write_transaction:
                    write_data_list.append(write_transaction)
                    write_labels.append("写TRANSACTION")
                    
                if write_data_list:
                    ax4.boxplot(write_data_list, labels=write_labels)
                    ax4.set_ylabel("延迟 (ns)")
                    ax4.set_title("写操作延迟对比")
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, "无写操作数据", ha='center', va='center', transform=ax4.transAxes, fontsize=14)
                    ax4.set_title("写操作延迟对比")

            plt.tight_layout()

            # 保存图表
            timestamp = int(time.time())
            save_path = f"{save_dir}/crossring_latency_distribution_{timestamp}.png"
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close(fig)

            self.logger.info(f"延迟分布图已保存到: {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"生成延迟分布图失败: {e}")
            return ""

    def plot_port_bandwidth_comparison(self, ip_analysis: Dict[str, Any], save_dir: str = "output") -> str:
        """生成IP类型带宽对比图"""
        if not ip_analysis:
            return ""

        try:
            ip_types = list(ip_analysis.keys())

            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(ip_types))
            width = 0.35

            # 提取读写带宽数据
            read_bw = [float(ip_analysis[ip_type]["读带宽_GB/s"]) for ip_type in ip_types]
            write_bw = [float(ip_analysis[ip_type]["写带宽_GB/s"]) for ip_type in ip_types]

            bars1 = ax.bar(x - width / 2, read_bw, width, label="读带宽", color="green", alpha=0.7)
            bars2 = ax.bar(x + width / 2, write_bw, width, label="写带宽", color="red", alpha=0.7)

            ax.set_xlabel("IP类型", fontsize=12)
            ax.set_ylabel("带宽 (GB/s)", fontsize=12)
            ax.set_title("各IP类型带宽对比", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(ip_types)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=10)

            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=10)

            # 添加请求数量信息
            for i, ip_type in enumerate(ip_types):
                total_requests = ip_analysis[ip_type]["总请求数"]
                read_requests = ip_analysis[ip_type]["读请求数"]
                write_requests = ip_analysis[ip_type]["写请求数"]

                # 在X轴标签下方添加请求数信息
                ax.text(i, -max(max(read_bw), max(write_bw)) * 0.1, f"总请求: {total_requests}\n(读:{read_requests}, 写:{write_requests})", ha="center", va="top", fontsize=8, alpha=0.7)

            plt.tight_layout()

            # 保存图表
            timestamp = int(time.time())
            save_path = f"{save_dir}/crossring_ip_bandwidth_{timestamp}.png"
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            self.logger.info(f"IP带宽对比图已保存到: {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"生成IP带宽对比图失败: {e}")
            return ""

    def save_results(self, analysis: Dict[str, Any], save_dir: str = "output") -> str:
        """保存分析结果到JSON文件"""
        try:
            timestamp = int(time.time())
            results_file = f"{save_dir}/crossring_analysis_{timestamp}.json"

            os.makedirs(save_dir, exist_ok=True)
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            self.logger.info(f"分析结果已保存到: {results_file}")
            return results_file
        except Exception as e:
            self.logger.error(f"保存分析结果失败: {e}")
            return ""

    def plot_traffic_distribution(self, model, metrics, save_dir: str = "output", mode: str = "total") -> str:
        """
        绘制流量分布图，显示节点IP带宽和链路带宽

        Args:
            model: CrossRingModel实例
            metrics: 请求度量数据
            save_dir: 保存目录
            mode: 显示模式 ("read", "write", "total")

        Returns:
            保存的文件路径
        """
        if not metrics:
            return ""

        try:
            # 获取节点数量和拓扑信息
            num_nodes = len(model.nodes)
            num_rows = model.config.NUM_ROW
            num_cols = model.config.NUM_COL

            # 收集节点级IP带宽数据和链路带宽数据
            # 支持的IP类型：GDMA, SDMA, CDMA, DDR, L2M等，用小写存储
            node_ip_bandwidth = defaultdict(lambda: defaultdict(float))
            link_bandwidth = defaultdict(float)

            # 首先计算整体时间窗口
            if not metrics:
                return ""
                
            overall_start_time = min(metric.start_time for metric in metrics)
            overall_end_time = max(metric.end_time for metric in metrics)
            overall_time_window = overall_end_time - overall_start_time if overall_end_time > overall_start_time else 1

            # 按IP类型分组收集字节数
            ip_type_bytes = defaultdict(int)
            node_ip_bytes = defaultdict(lambda: defaultdict(int))
            link_bytes = defaultdict(int)
            
            # 分析每个请求的字节数贡献
            for metric in metrics:
                source_ip_type = metric.source_type.lower()  # gdma/ddr
                dest_ip_type = metric.dest_type.lower()  # gdma/ddr
                
                # 累计字节数（不是带宽）
                # 源节点：发送字节数
                node_ip_bytes[metric.source_node][source_ip_type] += metric.total_bytes
                # 目标节点：接收字节数  
                node_ip_bytes[metric.dest_node][dest_ip_type] += metric.total_bytes

                # 计算链路字节数（只处理跨节点通信）
                if metric.source_node != metric.dest_node:
                    src_row = metric.source_node // num_cols
                    src_col = metric.source_node % num_cols
                    dst_row = metric.dest_node // num_cols
                    dst_col = metric.dest_node % num_cols

                    # 水平路由
                    if src_row == dst_row:
                        step = 1 if dst_col > src_col else -1
                        for col in range(src_col, dst_col, step):
                            curr_node = src_row * num_cols + col
                            next_node = src_row * num_cols + col + step
                            if mode == "total" or mode == metric.req_type:
                                link_bytes[(curr_node, next_node)] += metric.total_bytes

                    # 垂直路由
                    elif src_col == dst_col:
                        step = 1 if dst_row > src_row else -1
                        for row in range(src_row, dst_row, step):
                            curr_node = row * num_cols + src_col
                            next_node = (row + step) * num_cols + src_col
                            if mode == "total" or mode == metric.req_type:
                                link_bytes[(curr_node, next_node)] += metric.total_bytes
            
            # 计算最终带宽：使用工作区间方法计算加权带宽
            # 按节点和IP类型分组请求，计算各自的工作区间带宽
            for node_id, ip_data in node_ip_bytes.items():
                for ip_type, total_bytes in ip_data.items():
                    # 找到该节点该IP类型的所有请求
                    node_ip_requests = []
                    for metric in metrics:
                        if ((metric.source_node == node_id and metric.source_type.lower() == ip_type) or
                            (metric.dest_node == node_id and metric.dest_type.lower() == ip_type)):
                            node_ip_requests.append(metric)
                    
                    if node_ip_requests:
                        # 使用工作区间计算该节点该IP的加权带宽
                        working_intervals = self.calculate_working_intervals(node_ip_requests, min_gap_threshold=200)
                        
                        # 计算加权带宽
                        if working_intervals:
                            total_weighted_bandwidth = 0.0
                            total_weight = 0
                            
                            for interval in working_intervals:
                                weight = interval.flit_count
                                bandwidth = interval.bandwidth_bytes_per_ns  # bytes/ns
                                total_weighted_bandwidth += bandwidth * weight
                                total_weight += weight
                            
                            weighted_bandwidth = (total_weighted_bandwidth / total_weight) if total_weight > 0 else 0.0
                            bandwidth_gbps = weighted_bandwidth * 1e9 / (1024**3)
                        else:
                            bandwidth_gbps = 0.0
                    else:
                        bandwidth_gbps = 0.0
                        
                    node_ip_bandwidth[node_id][ip_type] = bandwidth_gbps
            
            # 计算链路带宽：使用通过该链路的请求计算工作区间带宽
            link_bandwidth = {}
            for link_key, total_bytes in link_bytes.items():
                # 找到通过该链路的所有请求
                link_requests = []
                curr_node, next_node = link_key
                
                for metric in metrics:
                    if metric.source_node != metric.dest_node:
                        # 检查该请求是否通过这条链路
                        src_row = metric.source_node // num_cols
                        src_col = metric.source_node % num_cols
                        dst_row = metric.dest_node // num_cols
                        dst_col = metric.dest_node % num_cols
                        
                        passes_through_link = False
                        
                        # 水平路由检查
                        if src_row == dst_row:
                            step = 1 if dst_col > src_col else -1
                            for col in range(src_col, dst_col, step):
                                check_curr = src_row * num_cols + col
                                check_next = src_row * num_cols + col + step
                                if (check_curr, check_next) == link_key:
                                    passes_through_link = True
                                    break
                        
                        # 垂直路由检查
                        elif src_col == dst_col:
                            step = 1 if dst_row > src_row else -1
                            for row in range(src_row, dst_row, step):
                                check_curr = row * num_cols + src_col
                                check_next = (row + step) * num_cols + src_col
                                if (check_curr, check_next) == link_key:
                                    passes_through_link = True
                                    break
                        
                        if passes_through_link:
                            link_requests.append(metric)
                
                if link_requests:
                    # 使用工作区间计算链路加权带宽
                    working_intervals = self.calculate_working_intervals(link_requests, min_gap_threshold=200)
                    
                    if working_intervals:
                        total_weighted_bandwidth = 0.0
                        total_weight = 0
                        
                        for interval in working_intervals:
                            weight = interval.flit_count
                            bandwidth = interval.bandwidth_bytes_per_ns  # bytes/ns
                            total_weighted_bandwidth += bandwidth * weight
                            total_weight += weight
                        
                        weighted_bandwidth = (total_weighted_bandwidth / total_weight) if total_weight > 0 else 0.0
                        bandwidth_gbps = weighted_bandwidth * 1e9 / (1024**3)
                    else:
                        bandwidth_gbps = 0.0
                else:
                    bandwidth_gbps = 0.0
                    
                link_bandwidth[link_key] = bandwidth_gbps

            # 计算总IP类型带宽（用于汇总显示）
            # 动态计算所有IP类型的总带宽
            ip_type_totals = defaultdict(float)
            for node_data in node_ip_bandwidth.values():
                for ip_type, bandwidth in node_data.items():
                    ip_type_totals[ip_type] += bandwidth

            # 计算节点位置（网格对齐，不交错）
            pos = {}
            for node_id in range(num_nodes):
                row = node_id // num_cols
                col = node_id % num_cols
                pos[node_id] = (col * 3, -row * 2)  # 规整网格，不偏移

            # 创建图形
            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_aspect("equal")

            # 动态计算字体大小
            base_font = 10
            dynamic_font = min(14, max(6, base_font * (65 / num_nodes) ** 0.5))

            # 节点大小
            node_size = 4000  # 增大节点
            square_size = np.sqrt(node_size) / 60  # 调整节点大小，更大的方框

            # 添加所有可能的链路（包括没有流量的）
            all_links = set()
            for node_id in range(num_nodes):
                row = node_id // num_cols
                col = node_id % num_cols

                # 水平连接
                if col < num_cols - 1:
                    all_links.add((node_id, node_id + 1))
                    all_links.add((node_id + 1, node_id))

                # 垂直连接
                if row < num_rows - 1:
                    all_links.add((node_id, node_id + num_cols))
                    all_links.add((node_id + num_cols, node_id))

            # 绘制所有链路（包括无流量的），使用双向箭头显示
            max_link_bw = max(link_bandwidth.values()) if link_bandwidth else 1.0
            
            # 为了避免双向箭头重叠，需要为每个方向计算偏移
            for src, dst in all_links:
                x1, y1 = pos[src]
                x2, y2 = pos[dst]

                bandwidth = link_bandwidth.get((src, dst), 0.0)

                # 计算链路颜色和线宽
                if bandwidth > 0:
                    intensity = min(1.0, bandwidth / max_link_bw)
                    color = (intensity, 0, 0)
                    linewidth = 1 + intensity * 2  # 减小线宽
                    alpha = 0.9
                else:
                    color = (0.7, 0.7, 0.7)  # 灰色表示无流量
                    linewidth = 0.8  # 减小无流量链路线宽
                    alpha = 0.5

                # 计算基本方向向量
                dx, dy = x2 - x1, y2 - y1
                dist = np.hypot(dx, dy)
                if dist > 0:
                    dx, dy = dx / dist, dy / dist
                    
                    # 计算垂直偏移向量（用于分离双向箭头）
                    perp_dx, perp_dy = -dy, dx  # 垂直方向
                    offset = 0.08  # 减小偏移距离，让双向箭头更近
                    
                    # 为该方向的箭头添加偏移
                    offset_x1 = x1 + perp_dx * offset
                    offset_y1 = y1 + perp_dy * offset
                    offset_x2 = x2 + perp_dx * offset
                    offset_y2 = y2 + perp_dy * offset
                    
                    # 计算从偏移后节点边缘的起止点
                    start_x = offset_x1 + dx * square_size / 2
                    start_y = offset_y1 + dy * square_size / 2
                    end_x = offset_x2 - dx * square_size / 2
                    end_y = offset_y2 - dy * square_size / 2

                    # 绘制带箭头的连接线
                    arrow = FancyArrowPatch(
                        (start_x, start_y), (end_x, end_y),
                        arrowstyle='-|>',
                        mutation_scale=dynamic_font * 1.2,  # 增大箭头大小
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=1
                    )
                    ax.add_patch(arrow)

                    # 如果有带宽，在箭头旁边添加标签
                    if bandwidth > 0:
                        # 判断链路方向并决定标签位置（更靠近链路，字体更大）
                        if abs(dx) > abs(dy):  # 水平链路
                            # 水平链路标签放在上下
                            label_offset_x = 0
                            label_offset_y = 0.2 if src < dst else -0.2  # 减小距离
                        else:  # 垂直链路
                            # 垂直链路标签放在左右
                            label_offset_x = 0.2 if src < dst else -0.2  # 减小距离
                            label_offset_y = 0
                        
                        mid_x = (start_x + end_x) / 2 + label_offset_x
                        mid_y = (start_y + end_y) / 2 + label_offset_y
                        
                        ax.text(
                            mid_x,
                            mid_y,
                            f"{bandwidth:.1f}",
                            ha="center",
                            va="center",
                            fontsize=dynamic_font * 0.7,  # 增大字体
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8),
                            color=color,
                            fontweight="bold",
                        )

            # 绘制节点和IP信息
            for node_id in range(num_nodes):
                x, y = pos[node_id]

                # 绘制主节点方框
                rect = Rectangle((x - square_size / 2, y - square_size / 2), width=square_size, height=square_size, color="lightblue", ec="black", linewidth=2, zorder=2)
                ax.add_patch(rect)

                # 节点编号和IP带宽信息写在方框内
                # 获取该节点的实际IP带宽（动态支持所有IP类型）
                node_ip_data = node_ip_bandwidth[node_id]
                
                # IP类型首字母映射
                def get_ip_abbreviation(ip_type):
                    """获取IP类型的首字母缩写"""
                    return ip_type.upper()[0] if ip_type else ""
                
                # 找出该节点有带宽的IP类型
                active_ips = [(ip_type, bw) for ip_type, bw in node_ip_data.items() if bw > 0]
                
                if len(active_ips) == 0:
                    ip_text = ""  # 无流量时不显示任何文字
                elif len(active_ips) == 1:
                    ip_type, bw = active_ips[0]
                    ip_abbrev = get_ip_abbreviation(ip_type)
                    ip_text = f"{ip_abbrev}: {bw:.1f}"
                else:
                    # 多个IP类型，每个IP类型分行显示
                    ip_lines = []
                    for ip_type, bw in active_ips:
                        ip_abbrev = get_ip_abbreviation(ip_type)
                        ip_lines.append(f"{ip_abbrev}: {bw:.1f}")
                    ip_text = "\n".join(ip_lines)

                # 在节点方框内显示信息
                if ip_text:
                    node_text = f"{node_id}\n{ip_text}"
                else:
                    node_text = f"{node_id}"
                ax.text(x, y, node_text, ha="center", va="center", fontsize=dynamic_font * 0.8, fontweight="bold")  # 增大字体

            # 添加总结信息框（动态显示所有IP类型）
            summary_lines = ["总带宽统计:"]
            for ip_type, total_bw in sorted(ip_type_totals.items()):
                ip_display = ip_type.upper()  # 显示大写
                summary_lines.append(f"{ip_display}: {total_bw:.2f} GB/s")
            
            summary_text = "\n".join(summary_lines)
            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))

            # 设置标题和布局
            title = f"CrossRing NoC 流量分布图 ({mode.upper()}模式)"
            ax.set_title(title, fontsize=16, fontweight="bold")

            # 设置坐标轴范围
            all_x = [x for x, y in pos.values()]
            all_y = [y for x, y in pos.values()]
            margin = 1.5
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
            ax.axis("off")

            # 添加图例
            legend_elements = [
                mpatches.Patch(color="lightblue", label="节点(含IP信息)"),
                mpatches.Patch(color="red", label="高带宽链路"),
                mpatches.Patch(color="gray", label="无流量链路"),
                mpatches.Patch(color="lightgray", label="带宽统计"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

            # 保存图表
            timestamp = int(time.time())
            save_path = f"{save_dir}/crossring_traffic_distribution_{timestamp}.png"
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            self.logger.info(f"流量分布图已保存到: {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"生成流量分布图失败: {e}")
            import traceback

            traceback.print_exc()
            return ""

    def analyze_noc_results(self, request_tracker, config, model, results: Dict[str, Any], enable_visualization: bool = True, save_results: bool = True, save_dir: str = "output") -> Dict[str, Any]:
        """
        NoC仿真结果完整分析

        Args:
            request_tracker: RequestTracker实例
            config: NoC配置
            model: NoC模型实例
            results: 仿真基础结果
            enable_visualization: 是否生成图表
            save_results: 是否保存结果文件

        Returns:
            详细的分析结果（中文格式）
        """
        analysis = {}

        # 基础指标
        sim_info = results.get("simulation_info", {})
        total_requests = len(request_tracker.completed_requests) + len(request_tracker.active_requests)
        completed_requests = len(request_tracker.completed_requests)
        active_requests = len(request_tracker.active_requests)

        analysis["基础指标"] = {
            "总周期数": sim_info.get("total_cycles", 0),
            "有效周期数": sim_info.get("effective_cycles", 0),
            "总请求数": total_requests,
            "已完成请求数": completed_requests,
            "活跃请求数": active_requests,
            "完成率": f"{(completed_requests / total_requests * 100):.2f}%" if total_requests > 0 else "0.00%",
        }

        # 转换数据格式
        metrics = self.convert_tracker_to_request_info(request_tracker, config)
        
        if not metrics:
            self.logger.warning("没有找到已完成的请求数据")
            return analysis

        # 带宽分析
        analysis["带宽指标"] = self.analyze_bandwidth(metrics)

        # 延迟分析
        analysis["延迟指标"] = self.analyze_latency(metrics)

        # 端口带宽分析
        analysis["端口带宽分析"] = self.analyze_port_bandwidth(metrics)

        # Tag和绕环数据分析
        analysis["Tag和绕环分析"] = self.analyze_tag_data(model)

        # 生成图表
        if enable_visualization:
            chart_paths = []

            # 带宽曲线图
            bw_path = self.plot_bandwidth_curves(metrics, save_dir=save_dir)
            if bw_path:
                chart_paths.append(bw_path)

            # 延迟分布图
            lat_path = self.plot_latency_distribution(metrics, save_dir=save_dir)
            if lat_path:
                chart_paths.append(lat_path)

            # 端口带宽对比图
            port_path = self.plot_port_bandwidth_comparison(analysis["端口带宽分析"], save_dir=save_dir)
            if port_path:
                chart_paths.append(port_path)

            # 流量分布图
            traffic_path = self.plot_traffic_distribution(model, metrics, save_dir=save_dir, mode="total")
            if traffic_path:
                chart_paths.append(traffic_path)

            analysis["可视化文件"] = {"生成的图表": chart_paths, "图表数量": len(chart_paths)}

        # 保存结果文件
        if save_results:
            # 保存分析结果JSON
            results_file = self.save_results(analysis, save_dir=save_dir)
            
            # 保存详细请求CSV文件
            csv_files = self.save_detailed_requests_csv(metrics, save_dir=save_dir)
            
            # 保存端口带宽CSV文件
            ports_csv = self.save_ports_bandwidth_csv(metrics, save_dir=save_dir, config=config)
            
            output_files = {}
            if results_file:
                output_files["分析结果文件"] = results_file
            
            # 添加CSV文件信息
            if csv_files:
                if "read_requests_csv" in csv_files:
                    output_files["读请求CSV"] = csv_files["read_requests_csv"]
                if "write_requests_csv" in csv_files:
                    output_files["写请求CSV"] = csv_files["write_requests_csv"]
            
            if ports_csv:
                output_files["端口带宽CSV"] = ports_csv
                
            if output_files:
                analysis["输出文件"] = {
                    **output_files,
                    "保存时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                }

        return analysis
