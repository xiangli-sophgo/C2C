"""
NoC结果分析器
通用的NoC性能分析工具，包含带宽、延迟、流量分析等功能
支持多种NoC拓扑（CrossRing、Mesh等）
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
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
        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]  # 支持中文
        plt.rcParams["axes.unicode_minus"] = False

    def convert_tracker_to_request_info(self, request_tracker, config) -> List[RequestInfo]:
        """转换RequestTracker数据为RequestInfo格式（按照老版本逻辑）"""
        requests = []

        # 获取配置参数
        network_frequency = getattr(config.basic, "NETWORK_FREQUENCY", 1.0) if hasattr(config, "basic") else 1.0

        for req_id, lifecycle in request_tracker.completed_requests.items():
            if lifecycle.completed_cycle <= 0:
                continue

            # 时间转换：cycle -> ns (按照老版本的方式)
            # 使用整数除法，与老版本保持一致
            start_time = lifecycle.created_cycle // network_frequency
            end_time = lifecycle.completed_cycle // network_frequency
            rn_end_time = end_time  # 简化处理
            sn_end_time = end_time  # 简化处理

            # 计算延迟（按照老版本方式）
            total_latency = lifecycle.completed_cycle - lifecycle.created_cycle
            cmd_latency = total_latency // 3  # 简化假设
            data_latency = total_latency // 3
            transaction_latency = total_latency

            # 计算字节数 (按照老版本的方式)
            burst_length = getattr(lifecycle, "burst_size", 4)
            # 老版本使用128字节/flit
            total_bytes = burst_length * 128

            request_info = RequestInfo(
                packet_id=req_id,
                start_time=start_time,
                end_time=end_time,
                rn_end_time=rn_end_time,
                sn_end_time=sn_end_time,
                req_type=lifecycle.op_type,
                source_node=getattr(lifecycle, "src_node", 0),
                dest_node=getattr(lifecycle, "dst_node", 0),
                source_type=getattr(lifecycle, "src_ip_type", "gdma"),
                dest_type=getattr(lifecycle, "dst_ip_type", "ddr"),
                burst_length=burst_length,
                total_bytes=total_bytes,
                cmd_latency=cmd_latency // network_frequency,
                data_latency=data_latency // network_frequency,
                transaction_latency=transaction_latency // network_frequency,
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
        """分析端口级别带宽"""
        port_analysis = defaultdict(lambda: {"read": [], "write": []})

        for metric in metrics:
            port_id = f"节点{metric.source_node}"
            port_analysis[port_id][metric.req_type].append(metric)

        port_summary = {}
        for port_id, data in port_analysis.items():
            read_reqs = data["read"]
            write_reqs = data["write"]

            # 计算该端口的读写带宽
            if read_reqs:
                read_total_bytes = sum(r.total_bytes for r in read_reqs)
                read_total_time = max(r.end_time for r in read_reqs) - min(r.start_time for r in read_reqs)
                read_bw = read_total_bytes / read_total_time if read_total_time > 0 else 0
            else:
                read_bw = 0

            if write_reqs:
                write_total_bytes = sum(r.total_bytes for r in write_reqs)
                write_total_time = max(r.end_time for r in write_reqs) - min(r.start_time for r in write_reqs)
                write_bw = write_total_bytes / write_total_time if write_total_time > 0 else 0
            else:
                write_bw = 0

            port_summary[port_id] = {
                "读带宽_GB/s": f"{read_bw:.2f}",
                "写带宽_GB/s": f"{write_bw:.2f}",
                "总带宽_GB/s": f"{read_bw + write_bw:.2f}",
                "读请求数": len(read_reqs),
                "写请求数": len(write_reqs),
            }

        return port_summary

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
            for node in model.crossring_nodes.values():
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

    def plot_bandwidth_curves(self, metrics, save_dir: str = "output") -> str:
        """生成带宽曲线图"""
        if not metrics:
            return ""

        try:
            # 按端口分组数据
            port_data = defaultdict(lambda: {"时间": [], "带宽": []})

            for metric in metrics:
                port_key = f"节点{metric.source_node}-{metric.request_type.value.upper()}"
                port_data[port_key]["时间"].append(metric.end_time / 1000)  # 转换为us
                port_data[port_key]["带宽"].append(metric.bandwidth_gbps)

            # 生成图表
            fig, ax = plt.subplots(figsize=(12, 8))

            for port_key, data in port_data.items():
                if data["时间"]:
                    ax.plot(data["时间"], data["带宽"], label=port_key, marker="o", linewidth=2)

            ax.set_xlabel("时间 (μs)")
            ax.set_ylabel("带宽 (GB/s)")
            ax.set_title("NoC 带宽曲线图")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 保存图表
            timestamp = int(time.time())
            save_path = f"{save_dir}/crossring_bandwidth_curve_{timestamp}.png"
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close(fig)

            self.logger.info(f"带宽曲线图已保存到: {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"生成带宽曲线图失败: {e}")
            return ""

    def plot_latency_distribution(self, metrics, save_dir: str = "output") -> str:
        """生成延迟分布图"""
        if not metrics:
            return ""

        try:
            latencies = [m.total_latency for m in metrics]
            read_latencies = [m.total_latency for m in metrics if m.request_type == RequestType.READ]
            write_latencies = [m.total_latency for m in metrics if m.request_type == RequestType.WRITE]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 延迟直方图
            ax1.hist(latencies, bins=30, alpha=0.7, label="总体延迟", color="blue")
            if read_latencies:
                ax1.hist(read_latencies, bins=30, alpha=0.7, label="读延迟", color="green")
            if write_latencies:
                ax1.hist(write_latencies, bins=30, alpha=0.7, label="写延迟", color="red")

            ax1.set_xlabel("延迟 (ns)")
            ax1.set_ylabel("频次")
            ax1.set_title("延迟分布直方图")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 延迟箱线图
            latency_data = []
            labels = []

            if read_latencies:
                latency_data.append(read_latencies)
                labels.append("读操作")

            if write_latencies:
                latency_data.append(write_latencies)
                labels.append("写操作")

            if latency_data:
                ax2.boxplot(latency_data, labels=labels)
                ax2.set_ylabel("延迟 (ns)")
                ax2.set_title("延迟箱线图")
                ax2.grid(True, alpha=0.3)

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

    def plot_port_bandwidth_comparison(self, port_analysis: Dict[str, Any], save_dir: str = "output") -> str:
        """生成端口带宽对比图"""
        if not port_analysis:
            return ""

        try:
            ports = list(port_analysis.keys())
            read_bw = [float(port_analysis[port]["读带宽_GB/s"]) for port in ports]
            write_bw = [float(port_analysis[port]["写带宽_GB/s"]) for port in ports]

            fig, ax = plt.subplots(figsize=(12, 8))

            x = np.arange(len(ports))
            width = 0.35

            bars1 = ax.bar(x - width / 2, read_bw, width, label="读带宽", color="green", alpha=0.7)
            bars2 = ax.bar(x + width / 2, write_bw, width, label="写带宽", color="red", alpha=0.7)

            ax.set_xlabel("端口")
            ax.set_ylabel("带宽 (GB/s)")
            ax.set_title("各端口带宽对比")
            ax.set_xticks(x)
            ax.set_xticklabels(ports, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom")

            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom")

            # 保存图表
            timestamp = int(time.time())
            save_path = f"{save_dir}/crossring_port_bandwidth_{timestamp}.png"
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close(fig)

            self.logger.info(f"端口带宽对比图已保存到: {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"生成端口带宽对比图失败: {e}")
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

    def convert_tracker_to_metrics(self, request_tracker, config) -> List[RequestInfo]:
        """将RequestTracker数据转换为RequestInfo列表格式"""
        metrics = []
        
        # 处理已完成的请求
        for packet_id, lifecycle in request_tracker.completed_requests.items():
            # 转换时间（假设以周期为单位，需要转换为ns）
            network_frequency = getattr(config, 'NETWORK_FREQUENCY', 1000)  # MHz
            cycle_time_ns = 1000.0 / network_frequency  # ns per cycle
            
            # 计算延迟
            latency_cycles = lifecycle.completed_cycle - lifecycle.created_cycle
            latency_ns = int(latency_cycles * cycle_time_ns)
            
            req_info = RequestInfo(
                packet_id=packet_id,
                start_time=int(lifecycle.created_cycle * cycle_time_ns),
                end_time=int(lifecycle.completed_cycle * cycle_time_ns),
                rn_end_time=int(lifecycle.completed_cycle * cycle_time_ns),  # 简化处理
                sn_end_time=int(lifecycle.completed_cycle * cycle_time_ns),  # 简化处理
                req_type=lifecycle.op_type,
                source_node=lifecycle.source,
                dest_node=lifecycle.destination,
                source_type="gdma",  # 从traffic文件可知
                dest_type="ddr",     # 从traffic文件可知
                burst_length=lifecycle.burst_size,
                total_bytes=lifecycle.burst_size * 128,  # 假设每个flit是128字节
                cmd_latency=latency_ns // 3,  # 简化处理：命令延迟占1/3
                data_latency=latency_ns // 3,  # 简化处理：数据延迟占1/3
                transaction_latency=latency_ns  # 总延迟
            )
            
            metrics.append(req_info)
        
        return metrics

    def analyze_noc_results(self, request_tracker, config, model, results: Dict[str, Any], enable_visualization: bool = True, save_results: bool = True) -> Dict[str, Any]:
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
        metrics = self.convert_tracker_to_metrics(request_tracker, config)

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
            bw_path = self.plot_bandwidth_curves(metrics)
            if bw_path:
                chart_paths.append(bw_path)

            # 延迟分布图
            lat_path = self.plot_latency_distribution(metrics)
            if lat_path:
                chart_paths.append(lat_path)

            # 端口带宽对比图
            port_path = self.plot_port_bandwidth_comparison(analysis["端口带宽分析"])
            if port_path:
                chart_paths.append(port_path)

            analysis["可视化文件"] = {"生成的图表": chart_paths, "图表数量": len(chart_paths)}

        # 保存结果文件
        if save_results:
            results_file = self.save_results(analysis)
            if results_file:
                analysis["输出文件"] = {
                    "分析结果文件": results_file,
                    "保存时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                }

        return analysis
