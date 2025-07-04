import numpy as np
from collections import deque, defaultdict

from src.utils.optimal_placement import create_adjacency_matrix, find_shortest_paths
from config.config import CrossRingConfig
from src.utils.components import Flit, Network, Node, TokenBucket, IPInterface

from src.core.Link_State_Visualizer import NetworkLinkVisualizer
import matplotlib.pyplot as plt

import os
import sys, time
import inspect, logging
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np

from functools import lru_cache
from src.core.result_processor import *
from src.core.traffic_scheduler import TrafficScheduler
import threading


class PerformanceMonitor:
    """Performance monitoring utility for tracking simulation metrics"""

    def __init__(self):
        self.method_times = {}
        self.call_counts = {}
        self.cache_hits = {}
        self.cache_misses = {}

    def time_method(self, method_name):
        """Decorator to time method execution"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()

                execution_time = end_time - start_time
                if method_name not in self.method_times:
                    self.method_times[method_name] = []
                    self.call_counts[method_name] = 0

                self.method_times[method_name].append(execution_time)
                self.call_counts[method_name] += 1

                return result

            return wrapper

        return decorator

    def record_cache_hit(self, cache_name):
        """Record a cache hit"""
        if cache_name not in self.cache_hits:
            self.cache_hits[cache_name] = 0
        self.cache_hits[cache_name] += 1

    def record_cache_miss(self, cache_name):
        """Record a cache miss"""
        if cache_name not in self.cache_misses:
            self.cache_misses[cache_name] = 0
        self.cache_misses[cache_name] += 1

    def get_summary(self):
        """Get performance summary statistics"""
        summary = {}

        # Method timing statistics
        for method_name, times in self.method_times.items():
            if times:
                summary[f"{method_name}_avg_time"] = sum(times) / len(times)
                summary[f"{method_name}_total_time"] = sum(times)
                summary[f"{method_name}_call_count"] = self.call_counts[method_name]
                summary[f"{method_name}_max_time"] = max(times)
                summary[f"{method_name}_min_time"] = min(times)

        # Cache statistics
        for cache_name in set(list(self.cache_hits.keys()) + list(self.cache_misses.keys())):
            hits = self.cache_hits.get(cache_name, 0)
            misses = self.cache_misses.get(cache_name, 0)
            total = hits + misses

            summary[f"{cache_name}_cache_hits"] = hits
            summary[f"{cache_name}_cache_misses"] = misses
            summary[f"{cache_name}_cache_hit_rate"] = hits / total if total > 0 else 0

        return summary


@lru_cache(maxsize=None)
def _parse_traffic_file(abs_path: str, net_freq: int):
    """
    解析 traffic 文件并缓存结果。
    返回 (lines, (read_req, write_req, read_flit, write_flit))
    """
    lines = []
    read_req = write_req = read_flit = write_flit = 0
    with open(abs_path, "r") as f:
        for raw in f:
            t, src, src_t, dst, dst_t, op, burst = raw.strip().split(",")
            burst = int(burst)
            tup = (int(t) * net_freq, int(src), src_t, int(dst), dst_t, op, burst)
            lines.append(tup)
            if op == "R":
                read_req += 1
                read_flit += burst
            else:
                write_req += 1
                write_flit += burst
    return lines, (read_req, write_req, read_flit, write_flit)


class BaseModel:
    def __init__(
        self,
        model_type,
        config: CrossRingConfig,
        topo_type,
        traffic_file_path,
        result_save_path: str,
        traffic_config,  # 可以是 "single_file.txt" 或者 [["file1.txt", "file2.txt"], ["file3.txt"]]
        results_fig_save_path: str = "",
        plot_flow_fig=False,
        flow_fig_show_CDMA=False,
        plot_RN_BW_fig=False,
        plot_link_state=False,
        plot_start_time=-1,
        print_trace=False,
        show_trace_id=0,
        show_node_id=3,
        verbose=0,
    ):
        self.model_type_stat = model_type
        self.config = config
        self.topo_type_stat = topo_type
        self.traffic_file_path = traffic_file_path
        self.result_save_path = None

        # 初始化TrafficScheduler
        self.traffic_scheduler = TrafficScheduler(config, traffic_file_path)
        self.traffic_scheduler.set_verbose(verbose > 0)

        # 处理traffic配置
        if isinstance(traffic_config, str):
            # 单个文件，向后兼容
            self.file_name = traffic_config
            self.traffic_scheduler.setup_single_chain([traffic_config])
        elif isinstance(traffic_config, list):
            # 多traffic链配置
            self.traffic_scheduler.setup_parallel_chains(traffic_config)
            self.file_name = self.traffic_scheduler.get_save_filename() + ".txt"
        else:
            raise ValueError("traffic_config must be a string (single file) or list of lists (multiple chains)")

        self.result_save_path_original = result_save_path
        self.plot_flow_fig = plot_flow_fig
        self.flow_fig_show_CDMA = flow_fig_show_CDMA
        self.plot_RN_BW_fig = plot_RN_BW_fig
        self.plot_link_state = plot_link_state
        self.plot_start_time = plot_start_time
        self.print_trace = print_trace
        self._done_flags = {
            "req": False,
            "rsp": False,
            "flit": False,
        }
        self.show_trace_id = show_trace_id
        self.show_node_id = show_node_id
        self.verbose = verbose
        if self.verbose:
            print(f"\nModel Type: {model_type}, Topology: {self.topo_type_stat}, file_name: {self.file_name[:-4]}")
        self.results_fig_save_path = None
        if result_save_path:
            self.result_save_path = self.result_save_path_original + str(topo_type) + "/" + self.file_name[:-4] + "/"
            os.makedirs(self.result_save_path, exist_ok=True)
        if results_fig_save_path:
            self.results_fig_save_path = results_fig_save_path
            os.makedirs(self.results_fig_save_path, exist_ok=True)

    def initial(self):
        self.topo_type_stat = self.config.TOPO_TYPE
        self.config.update_config(self.topo_type_stat)
        self.adjacency_matrix = create_adjacency_matrix("CrossRing", self.config.NUM_NODE, self.config.NUM_COL)
        self.req_network = Network(self.config, self.adjacency_matrix, name="Request Network")
        self.rsp_network = Network(self.config, self.adjacency_matrix, name="Response Network")
        self.data_network = Network(self.config, self.adjacency_matrix, name="Data Network")
        self.result_processor = BandwidthAnalyzer(self.config, min_gap_threshold=200, plot_rn_bw_fig=self.plot_RN_BW_fig, plot_flow_graph=self.plot_flow_fig)
        if self.plot_link_state:
            self.link_state_vis = NetworkLinkVisualizer(self.data_network)
        if self.config.ETag_BOTHSIDE_UPGRADE:
            self.req_network.ETag_BOTHSIDE_UPGRADE = self.rsp_network.ETag_BOTHSIDE_UPGRADE = self.data_network.ETag_BOTHSIDE_UPGRADE = True
        self.rn_positions = set(self.config.GDMA_SEND_POSITION_LIST + self.config.SDMA_SEND_POSITION_LIST + self.config.CDMA_SEND_POSITION_LIST)
        self.sn_positions = set(self.config.DDR_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST)
        self.flit_positions = set(
            self.config.GDMA_SEND_POSITION_LIST + self.config.SDMA_SEND_POSITION_LIST + self.config.CDMA_SEND_POSITION_LIST + self.config.DDR_SEND_POSITION_LIST + self.config.L2M_SEND_POSITION_LIST
        )

        # 缓存位置列表以避免重复转换
        self.rn_positions_list = list(self.rn_positions)
        self.sn_positions_list = list(self.sn_positions)
        self.flit_positions_list = list(self.flit_positions)

        # 缓存网络类型到IP类型的映射
        self.network_ip_types = {
            "req": [ip_type for ip_type in self.config.CH_NAME_LIST if ip_type.startswith("gdma") or ip_type.startswith("sdma") or ip_type.startswith("cdma")],
            "rsp": [ip_type for ip_type in self.config.CH_NAME_LIST if ip_type.startswith("ddr") or ip_type.startswith("l2m")],
            "data": self.config.CH_NAME_LIST,  # data网络不筛选
        }

        # Pre-calculate frequently used lists to avoid repeated conversions
        self.rn_positions_list = list(self.rn_positions)
        self.sn_positions_list = list(self.sn_positions)
        self.flit_positions_list = list(self.flit_positions)
        self.routes = find_shortest_paths(self.adjacency_matrix)
        self.node = Node(self.config)
        self.ip_modules = {}
        for ip_pos in self.flit_positions:
            for ip_type in self.config.CH_NAME_LIST:
                self.ip_modules[(ip_type, ip_pos)] = IPInterface(
                    ip_type,
                    ip_pos,
                    self.config,
                    self.req_network,
                    self.rsp_network,
                    self.data_network,
                    self.node,
                    self.routes,
                )
        self.flits = []
        self.throughput_time = []
        self.data_count = 0
        self.req_count = 0
        self.flit_id_count = 0
        self.send_flits_num = 0
        self.send_reqs_num = 0
        self.trans_flits_num = 0
        self.end_time = np.inf
        self.print_interval = 5000
        self._last_printed_cycle = -1
        self.flit_num, self.req_num, self.rsp_num = 0, 0, 0
        self.new_write_req = []
        self.IQ_directions = ["TR", "TL", "TU", "TD", "EQ"]
        self.IQ_direction_conditions = {
            "TR": lambda flit: flit.path[1] - flit.path[0] == 1 and flit.source - flit.destination != self.config.NUM_COL,
            "TL": lambda flit: flit.path[1] - flit.path[0] == -1 and flit.source - flit.destination != self.config.NUM_COL,
            "TU": lambda flit: (
                len(flit.path) >= 3
                and flit.path[2] - flit.path[1] == -self.config.NUM_COL * 2
                and flit.path[1] - flit.path[0] == -self.config.NUM_COL
                and flit.source - flit.destination != self.config.NUM_COL
            ),
            "TD": lambda flit: (
                len(flit.path) >= 3
                and flit.path[2] - flit.path[1] == self.config.NUM_COL * 2
                and flit.path[1] - flit.path[0] == -self.config.NUM_COL
                and flit.source - flit.destination != self.config.NUM_COL
            ),
            "EQ": lambda flit: flit.source - flit.destination == self.config.NUM_COL,
        }
        # 如果只有1列，禁用横向和垂直环注入，仅保留EQ方向
        if self.config.NUM_COL <= 1:
            self.IQ_directions = ["EQ", "TU", "TD"]
            self.IQ_direction_conditions = {
                "TU": lambda flit: (
                    len(flit.path) >= 3
                    and flit.path[2] - flit.path[1] == -self.config.NUM_COL * 2
                    and flit.path[1] - flit.path[0] == -self.config.NUM_COL
                    and flit.source - flit.destination != self.config.NUM_COL
                ),
                "TD": lambda flit: (
                    len(flit.path) >= 3
                    and flit.path[2] - flit.path[1] == self.config.NUM_COL * 2
                    and flit.path[1] - flit.path[0] == -self.config.NUM_COL
                    and flit.source - flit.destination != self.config.NUM_COL
                ),
                "EQ": lambda flit: flit.source - flit.destination == self.config.NUM_COL,
            }
        self.read_ip_intervals = defaultdict(list)  # 存储每个IP的读请求时间区间
        self.write_ip_intervals = defaultdict(list)  # 存储每个IP的写请求时间区间

        self.type_to_positions = {
            "req": self.sn_positions,
            "rsp": self.rn_positions,
            "data": self.flit_positions,
        }

        self.dma_rw_counts = self.config._make_channels(
            ("gdma", "sdma", "cdma"),
            {ip: {"read": 0, "write": 0} for ip in set(self.config.GDMA_SEND_POSITION_LIST + self.config.SDMA_SEND_POSITION_LIST + self.config.CDMA_SEND_POSITION_LIST)},
        )

        self.rn_bandwidth = {
            "SDMA read DDR": {"time": [], "bandwidth": []},
            "SDMA read L2M": {"time": [], "bandwidth": []},
            "GDMA read DDR": {"time": [], "bandwidth": []},
            "GDMA read L2M": {"time": [], "bandwidth": []},
            "SDMA write DDR": {"time": [], "bandwidth": []},
            "SDMA write L2M": {"time": [], "bandwidth": []},
            "GDMA write DDR": {"time": [], "bandwidth": []},
            "GDMA write L2M": {"time": [], "bandwidth": []},
            "total": {"time": [], "bandwidth": []},
        }

        # statistical data
        self.send_read_flits_num_stat = 0
        self.send_write_flits_num_stat = 0
        self.file_name_stat = self.file_name[:-4]
        self.R_finish_time_stat, self.W_finish_time_stat = 0, 0
        self.Total_finish_time_stat = 0
        self.R_tail_latency_stat, self.W_tail_latency_stat = 0, 0
        self.req_cir_h_num_stat, self.req_cir_v_num_stat = 0, 0
        self.rsp_cir_h_num_stat, self.rsp_cir_v_num_stat = 0, 0
        self.data_cir_h_num_stat, self.data_cir_v_num_stat = 0, 0
        self.req_wait_cycle_h_num_stat, self.req_wait_cycle_v_num_stat = 0, 0
        self.rsp_wait_cycle_h_num_stat, self.rsp_wait_cycle_v_num_stat = 0, 0
        self.data_wait_cycle_h_num_stat, self.data_wait_cycle_v_num_stat = 0, 0
        self.read_retry_num_stat, self.write_retry_num_stat = 0, 0
        self.EQ_ETag_T1_num_stat, self.EQ_ETag_T0_num_stat = 0, 0
        self.RB_ETag_T1_num_stat, self.RB_ETag_T0_num_stat = 0, 0
        self.ITag_h_num_stat, self.ITag_v_num_stat = 0, 0
        self.Total_sum_BW_stat = 0

        # Mixed (total) bandwidth/latency stats initialization
        self.mixed_unweighted_bw_stat = 0
        self.mixed_weighted_bw_stat = 0
        self.cmd_mixed_avg_latency_stat = 0
        self.cmd_mixed_max_latency_stat = 0
        self.data_mixed_avg_latency_stat = 0
        self.data_mixed_max_latency_stat = 0
        self.trans_mixed_avg_latency_stat = 0
        self.trans_mixed_max_latency_stat = 0
        # Overall average bandwidth stats (unweighted and weighted)
        self.total_unweighted_bw_stat = 0
        self.total_weighted_bw_stat = 0

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.start_time = time.time()

    def run(self):
        """Main simulation loop."""
        simulation_start = time.perf_counter()
        self.load_request_stream()
        flits, reqs, rsps = [], [], []
        self.cycle = 0
        tail_time = 0

        while True:
            self.cycle += 1
            self.cycle_mod = self.cycle % self.config.NETWORK_FREQUENCY

            # Tag moves

            self.release_completed_sn_tracker()

            self.process_new_request()

            self.tag_move_all_networks()

            self.ip_inject_to_network()

            self._inject_queue_arbitration(self.req_network, self.rn_positions_list, "req")
            reqs = self.move_flits_in_network(self.req_network, reqs, "req")

            self._inject_queue_arbitration(self.rsp_network, self.sn_positions_list, "rsp")
            rsps = self.move_flits_in_network(self.rsp_network, rsps, "rsp")

            self._inject_queue_arbitration(self.data_network, self.flit_positions_list, "data")
            flits = self.move_flits_in_network(self.data_network, flits, "data")

            self.network_to_ip_eject()

            self.move_pre_to_queues_all()

            self.debug_func()

            # Evaluate throughput time
            self.update_throughput_metrics(flits)

            if self.cycle / self.config.NETWORK_FREQUENCY % self.print_interval == 0:
                self.log_summary()

            # 检查traffic完成情况并推进链
            completed_traffics = self.traffic_scheduler.check_and_advance_chains(self.cycle)
            if completed_traffics and self.verbose:
                print(f"Completed traffics: {completed_traffics}")

            if (self.traffic_scheduler.is_all_completed() and self.trans_flits_num == 0 and not self.new_write_req) or self.cycle > self.end_time * self.config.NETWORK_FREQUENCY:
                if tail_time == 0:
                    if self.verbose:
                        print("Finish!")
                    break
                else:
                    tail_time -= 1

        # Performance evaluation
        self.print_data_statistic()
        self.log_summary()
        self.syn_IP_stat()

        # 更新结束时间统计
        self.update_finish_time_stats()

        # 结果统计
        self.process_comprehensive_results()

        # Record simulation performance
        simulation_end = time.perf_counter()
        simulation_time = simulation_end - simulation_start
        self.performance_monitor.method_times["total_simulation"] = [simulation_time]
        self.performance_monitor.call_counts["total_simulation"] = 1

        if self.verbose:
            print(f"Simulation completed in {simulation_time:.2f} seconds")
            print(f"Processed {self.cycle} cycles")
            print(f"Performance: {self.cycle / simulation_time:.0f} cycles/second")

    def update_finish_time_stats(self):
        """从traffic_scheduler和result_processor获取结束时间并更新统计"""
        read_end_times = []
        write_end_times = []
        all_end_times = []

        # 从traffic_scheduler获取结束时间统计
        try:
            finish_stats = self.traffic_scheduler.get_finish_time_stats()
            if finish_stats["R_finish_time"] > 0:
                read_end_times.append(finish_stats["R_finish_time"])
                all_end_times.append(finish_stats["R_finish_time"])
            if finish_stats["W_finish_time"] > 0:
                write_end_times.append(finish_stats["W_finish_time"])
                all_end_times.append(finish_stats["W_finish_time"])
            if finish_stats["Total_finish_time"] > 0:
                all_end_times.append(finish_stats["Total_finish_time"])
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not get finish time stats from traffic_scheduler: {e}")

        # 从result_processor获取请求的结束时间
        try:
            if hasattr(self, "result_processor") and hasattr(self.result_processor, "requests"):
                for req_info in self.result_processor.requests:
                    end_time_ns = req_info.end_time // self.config.NETWORK_FREQUENCY
                    all_end_times.append(end_time_ns)
                    if req_info.req_type == "read":
                        read_end_times.append(end_time_ns)
                    elif req_info.req_type == "write":
                        write_end_times.append(end_time_ns)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not get finish time stats from result_processor: {e}")

        # 更新统计数据，使用当前cycle作为备选
        current_time_ns = self.cycle // self.config.NETWORK_FREQUENCY

        if read_end_times:
            self.R_finish_time_stat = max(read_end_times)
        else:
            self.R_finish_time_stat = current_time_ns

        if write_end_times:
            self.W_finish_time_stat = max(write_end_times)
        else:
            self.W_finish_time_stat = current_time_ns

        if all_end_times:
            self.Total_finish_time_stat = max(all_end_times)
        else:
            self.Total_finish_time_stat = current_time_ns

        if self.verbose:
            print(f"Updated finish times - Read: {self.R_finish_time_stat}ns, Write: {self.W_finish_time_stat}ns, Total: {self.Total_finish_time_stat}ns")

    def update_traffic_completion_stats(self, flit):
        """在flit完成时更新TrafficScheduler的统计"""
        if hasattr(flit, "traffic_id") and getattr(flit, "is_arrive", False):
            self.traffic_scheduler.update_traffic_stats(flit.traffic_id, "received_flit")

    def syn_IP_stat(self):
        for ip_pos in self.flit_positions_list:
            for ip_type in self.config.CH_NAME_LIST:
                ip_interface: IPInterface = self.ip_modules[(ip_type, ip_pos)]
                if self.model_type_stat == "REQ_RSP":
                    self.read_retry_num_stat += ip_interface.read_retry_num_stat
                    self.write_retry_num_stat += ip_interface.write_retry_num_stat
                self.req_cir_h_num_stat += ip_interface.req_cir_h_num
                self.req_cir_v_num_stat += ip_interface.req_cir_v_num
                self.rsp_cir_h_num_stat += ip_interface.rsp_cir_h_num
                self.rsp_cir_v_num_stat += ip_interface.rsp_cir_v_num
                self.data_cir_h_num_stat += ip_interface.data_cir_h_num
                self.data_cir_v_num_stat += ip_interface.data_cir_v_num
                self.req_wait_cycle_h_num_stat += ip_interface.req_wait_cycles_h
                self.req_wait_cycle_v_num_stat += ip_interface.req_wait_cycles_v
                self.rsp_wait_cycle_h_num_stat += ip_interface.rsp_wait_cycles_h
                self.rsp_wait_cycle_v_num_stat += ip_interface.rsp_wait_cycles_v
                self.data_wait_cycle_h_num_stat += ip_interface.data_wait_cycles_h
                self.data_wait_cycle_v_num_stat += ip_interface.data_wait_cycles_v

    def debug_func(self):
        if self.print_trace:
            self.flit_trace(self.show_trace_id)
        if self.plot_link_state:
            while self.link_state_vis.paused and not self.link_state_vis.should_stop:
                plt.pause(0.05)
            if self.link_state_vis.should_stop:
                return
            if self.cycle / self.config.NETWORK_FREQUENCY < self.plot_start_time:
                return

            self.link_state_vis.update([self.req_network, self.rsp_network, self.data_network], self.cycle)

    def ip_inject_to_network(self):
        for ip_pos in self.flit_positions_list:
            for ip_type in self.config.CH_NAME_LIST:
                ip_interface: IPInterface = self.ip_modules[(ip_type, ip_pos)]
                ip_interface.inject_step(self.cycle)

    def network_to_ip_eject(self):
        """从网络到IP的eject步骤，并更新received_flit统计"""
        for ip_pos in self.flit_positions_list:
            for ip_type in self.config.CH_NAME_LIST:
                ip_interface: IPInterface = self.ip_modules[(ip_type, ip_pos)]
                # 执行eject，获取已到达目的IP的flit列表
                ejected_flits = ip_interface.eject_step(self.cycle)
                # 更新TrafficScheduler中的received_flit统计
                if ejected_flits:
                    for flit in ejected_flits:
                        self.update_traffic_completion_stats(flit)

    def release_completed_sn_tracker(self):
        """Check if any trackers can be released based on the current cycle."""
        for release_time in sorted(self.node.sn_tracker_release_time.keys()):
            if release_time > self.cycle:
                continue
            tracker_list = self.node.sn_tracker_release_time.pop(release_time)
            for sn_type, ip_pos, req in tracker_list:
                # 检查 tracker 是否还在列表中（避免重复释放）
                if req in self.node.sn_tracker[sn_type][ip_pos]:
                    ip_interface: IPInterface = self.ip_modules[(sn_type, ip_pos)]
                    ip_interface.release_completed_sn_tracker(req)

    def _move_pre_to_queues(self, network: Network, in_pos):
        """Move all items from pre-injection queues to injection queues for a given network."""
        # ===  注入队列 *_pre → *_FIFO ===
        ip_pos = in_pos - self.config.NUM_COL  # 本列对应的 IP 位置

        # IQ_pre → IQ_OUT
        for direction in self.IQ_directions:
            queue_pre = network.inject_queues_pre[direction]
            queue = network.inject_queues[direction]
            if queue_pre[in_pos] and len(queue[in_pos]) < self.config.RB_OUT_FIFO_DEPTH:
                flit = queue_pre[in_pos]
                flit.departure_inject_cycle = self.cycle
                flit.flit_position = f"IQ_{direction}"
                queue[in_pos].append(flit)
                queue_pre[in_pos] = None

        # RB_IN_PRE → RB_IN
        for direction in ["TL", "TR"]:
            queue_pre = network.ring_bridge_pre[direction]
            queue = network.ring_bridge[direction]
            key = (in_pos, ip_pos)
            if queue_pre[key] and len(queue[key]) < self.config.RB_IN_FIFO_DEPTH:
                flit = queue_pre[key]
                flit.flit_position = f"RB_{direction}"
                queue[key].append(flit)
                queue_pre[key] = None

        # RB_OUT_PRE → RB_OUT
        for fifo_pos in ("EQ", "TU", "TD"):
            queue_pre = network.ring_bridge_pre[fifo_pos]
            queue = network.ring_bridge[fifo_pos]
            key = (in_pos, ip_pos)
            if queue_pre[key] and len(queue[key]) < self.config.RB_OUT_FIFO_DEPTH:
                flit = queue_pre[key]
                flit.is_arrive = fifo_pos == "EQ"
                flit.flit_position = f"RB_{fifo_pos}"
                queue[key].append(flit)
                queue_pre[key] = None

        # EQ_IN_PRE → EQ_IN
        for fifo_pos in ("TU", "TD"):
            queue_pre = network.eject_queues_in_pre[fifo_pos]
            queue = network.eject_queues[fifo_pos]
            if queue_pre[ip_pos] and len(queue[ip_pos]) < self.config.EQ_IN_FIFO_DEPTH:
                flit = queue_pre[ip_pos]
                flit.is_arrive = fifo_pos == "EQ"
                flit.flit_position = f"EQ_{fifo_pos}"
                queue[ip_pos].append(flit)
                queue_pre[ip_pos] = None

        # EQ_channel_buffer_pre → EQ_channel_buffer
        for ip_type in network.EQ_channel_buffer_pre.keys():
            queue_pre = network.EQ_channel_buffer_pre[ip_type]
            queue = network.EQ_channel_buffer[ip_type]
            if queue_pre[ip_pos] and len(queue[ip_pos]) < self.config.EQ_CH_FIFO_DEPTH:
                flit = queue_pre[ip_pos]
                flit.flit_position = "EQ_CH"
                queue[ip_pos].append(flit)
                queue_pre[ip_pos] = None

    def print_data_statistic(self):
        if self.verbose:
            print(f"Data statistic: Read: {self.read_req, self.read_flit}, " f"Write: {self.write_req, self.write_flit}, " f"Total: {self.read_req + self.write_req, self.read_flit + self.write_flit}")

    def log_summary(self):
        if self.verbose:
            print(
                f"T: {self.cycle // self.config.NETWORK_FREQUENCY}, Req_cnt: {self.req_count} In_Req: {self.req_num}, Rsp: {self.rsp_num},"
                f" R_fn: {self.send_read_flits_num_stat}, W_fn: {self.send_write_flits_num_stat}, "
                f"Trans_fn: {self.trans_flits_num}, Recv_fn: {self.data_network.recv_flits_num}"
            )

    def move_flits_in_network(self, network, flits, flit_type):
        """Process injection queues and move flits."""
        flits = self._flit_move(network, flits, flit_type)
        return flits

    def _try_inject_to_direction(self, req: Flit, ip_type, ip_pos, direction, counts):
        """检查tracker空间并尝试注入到指定direction的pre缓冲"""
        # 直接注入到指定direction的pre缓冲
        queue_pre = self.req_network.inject_queues_pre[direction]
        queue_pre[ip_pos] = req

        # 从channel buffer移除
        self.req_network.IQ_channel_buffer[ip_type][ip_pos].popleft()

        # 更新计数和状态
        if req.req_attr == "new":  # 只有新请求才更新计数器和tracker
            if req.req_type == "read":
                counts["read"] += 1

            elif req.req_type == "write":
                counts["write"] += 1

        # req.cmd_entry_noc_from_cake0_cycle = self.cycle
        return True

    # ------------------------------------------------------------------
    # IQ仲裁：把 channel‑buffer 的 flit/req/rsp 放到 inject_queues_pre
    # ------------------------------------------------------------------
    def _inject_queue_arbitration(self, network, ip_positions, network_type):
        """
        Parameters
        ----------
        network : Network
            要操作的网络实例 (req / rsp / data)
        ip_positions : Iterable[int]
            需要遍历的 IP 物理位置集合
        network_type : str
            "req" | "rsp" | "data"
        """
        # Use cached pre-filtered IP types to avoid repeated list comprehensions
        if not hasattr(self, "_cached_ip_types"):
            self._cached_ip_types = {
                "req": [ip_type for ip_type in self.config.CH_NAME_LIST if ip_type.startswith(("sdma", "gdma", "cdma"))],
                "rsp": [ip_type for ip_type in self.config.CH_NAME_LIST if ip_type.startswith(("ddr", "l2m"))],
                "data": self.config.CH_NAME_LIST,
            }

        valid_ip_types = self._cached_ip_types[network_type]

        for ip_pos in ip_positions:
            for direction in self.IQ_directions:
                rr_queue = network.round_robin["IQ"][direction][ip_pos - self.config.NUM_COL]
                queue_pre = network.inject_queues_pre[direction]
                if queue_pre[ip_pos]:
                    continue  # pre 槽占用
                queue = network.inject_queues[direction]
                if len(queue[ip_pos]) >= self.config.IQ_OUT_FIFO_DEPTH:
                    continue  # FIFO 满

                # Use optimized IP type list instead of full rr_queue iteration
                processed = False
                for ip_type in valid_ip_types:
                    if ip_type not in rr_queue:
                        continue

                    if not network.IQ_channel_buffer[ip_type][ip_pos]:
                        continue  # channel‑buffer 空

                    flit = network.IQ_channel_buffer[ip_type][ip_pos][0]
                    if not self.IQ_direction_conditions[direction](flit):
                        continue  # 方向不匹配

                    # —— 网络‑特定前置检查 / 统计 ——
                    if network_type == "req":
                        max_gap = self.config.GDMA_RW_GAP if ip_type.startswith("gdma") else self.config.SDMA_RW_GAP
                        counts = self.dma_rw_counts[ip_type][ip_pos]
                        rd, wr = counts["read"], counts["write"]
                        if flit.req_type == "read" and abs(rd + 1 - wr) >= max_gap:
                            continue
                        if flit.req_type == "write" and abs(wr + 1 - rd) >= max_gap:
                            continue
                        # 使用现有函数做资源检查 + 注入
                        if not self._try_inject_to_direction(flit, ip_type, ip_pos, direction, counts):
                            continue
                        # _try_inject_to_direction 已经做了 popleft & pre‑缓冲写入，故直接 break
                        rr_queue.remove(ip_type)
                        rr_queue.append(ip_type)
                        processed = True
                        break

                    else:
                        # —— rsp / data 网络：直接移动到 pre‑缓冲 ——
                        network.IQ_channel_buffer[ip_type][ip_pos].popleft()
                        queue_pre[ip_pos] = flit

                        if network_type == "rsp":
                            flit.rsp_entry_network_cycle = self.cycle
                        elif network_type == "data":
                            req = self.req_network.send_flits[flit.packet_id][0]
                            flit.sync_latency_record(req)
                            flit.data_entry_network_cycle = self.cycle
                            self.send_flits_num += 1
                            self.trans_flits_num += 1
                            if hasattr(flit, "traffic_id"):
                                self.traffic_scheduler.update_traffic_stats(flit.traffic_id, "sent_flit")

                        rr_queue.remove(ip_type)
                        rr_queue.append(ip_type)
                        processed = True
                        break

                if processed:
                    continue

    def move_pre_to_queues_all(self):
        #  所有 IPInterface 的 *_pre → FIFO
        for ip_pos in self.flit_positions_list:
            for ip_type in self.config.CH_NAME_LIST:
                self.ip_modules[(ip_type, ip_pos)].move_pre_to_fifo()

        # 所有网络的 *_pre → FIFO
        for in_pos in self.flit_positions_list:
            self._move_pre_to_queues(self.req_network, in_pos)
            self._move_pre_to_queues(self.rsp_network, in_pos)
            self._move_pre_to_queues(self.data_network, in_pos)

    def update_throughput_metrics(self, flits):
        """Update throughput metrics based on flit counts."""
        self.trans_flits_num = len(flits)

    def load_request_stream(self):
        """修改：使用TrafficScheduler来管理请求流"""
        # 启动初始的traffic
        self.traffic_scheduler.start_initial_traffics()

        # 从TrafficScheduler获取统计信息
        total_stats = self._get_total_traffic_stats()
        self.read_req = total_stats["read_req"]
        self.write_req = total_stats["write_req"]
        self.read_flit = total_stats["read_flit"]
        self.write_flit = total_stats["write_flit"]

        # 统计输出保持原行为
        self.print_data_statistic()

    def _get_total_traffic_stats(self):
        """从所有链中统计总的请求和flit数量"""
        total_read_req = total_write_req = total_read_flit = total_write_flit = 0

        for chain in self.traffic_scheduler.parallel_chains:
            for traffic_file in chain.traffic_files:
                abs_path = os.path.join(self.traffic_file_path, traffic_file)
                with open(abs_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) >= 7:
                            op, burst = parts[5], int(parts[6])
                            if op == "R":

                                total_read_req += 1
                                total_read_flit += burst
                            else:
                                total_write_req += 1
                                total_write_flit += burst

        return {"read_req": total_read_req, "write_req": total_write_req, "read_flit": total_read_flit, "write_flit": total_write_flit}

    def process_new_request(self):
        """修改：使用TrafficScheduler获取多个准备就绪的请求"""
        # 从TrafficScheduler获取所有准备就绪的请求
        ready_requests = self.traffic_scheduler.get_ready_requests(self.cycle)

        if not ready_requests:
            return  # 没有待处理的请求

        # 处理所有准备就绪的请求
        for req_data in ready_requests:
            self._process_single_request(req_data)

    def _process_single_request(self, req_data):
        """处理单个请求"""
        # 解析请求数据（注意最后一个元素是traffic_id）
        source = self.node_map(req_data[1])
        destination = self.node_map(req_data[3], False)
        path = self.routes[source][destination]
        traffic_id = req_data[7]  # 最后一个元素是traffic_id

        # 创建flit对象 (使用对象池)
        req = Flit.create_flit(source, destination, path)
        req.source_original = req_data[1]
        req.destination_original = req_data[3]
        req.flit_type = "req"
        req.departure_cycle = req_data[0]
        req.burst_length = req_data[6]
        req.source_type = f"{req_data[2]}_0" if "_" not in req_data[2] else req_data[2]
        req.destination_type = f"{req_data[4]}_0" if "_" not in req_data[4] else req_data[4]
        req.original_source_type = f"{req_data[2]}_0" if "_" not in req_data[2] else req_data[2]
        req.original_destination_type = f"{req_data[4]}_0" if "_" not in req_data[4] else req_data[4]
        req.traffic_id = traffic_id  # 添加traffic_id标记

        req.packet_id = Node.get_next_packet_id()
        req.req_type = "read" if req_data[5] == "R" else "write"
        req.req_attr = "new"
        req.cmd_entry_cake0_cycle = self.cycle

        try:
            # 通过IPInterface处理请求
            ip_pos = req.source
            ip_type = req.source_type

            ip_interface: IPInterface = self.ip_modules[(ip_type, ip_pos)]
            if ip_interface is None:
                raise ValueError(f"IP module setup error for ({ip_type}, {ip_pos})!")

            # 检查IP接口是否能接受新请求（可选的流控机制）
            if hasattr(ip_interface, "can_accept_request") and not ip_interface.can_accept_request():
                if self.traffic_scheduler.verbose:
                    print(f"Warning: IP interface ({ip_type}, {ip_pos}) is busy, request may be delayed")

            ip_interface.enqueue(req, "req")

            # 更新统计信息
            if req.req_type == "read":
                self.R_tail_latency_stat = req_data[0]
            elif req.req_type == "write":
                self.W_tail_latency_stat = req_data[0]

            # 更新请求计数
            self.req_count += 1

        except Exception as e:
            logging.exception(f"Error processing request {req_data} in chain {traffic_id}")

    def error_log(self, flit, target_id, flit_id):
        if flit and flit.packet_id == target_id and flit.flit_id == flit_id:
            print(inspect.currentframe().f_back.f_code.co_name, self.cycle, flit)

    def flit_trace(self, packet_id):
        """打印指定 packet_id 或 packet_id 列表的调试信息"""
        if self.plot_link_state and self.link_state_vis.should_stop:
            return
        # 统一处理 packet_id（兼容单个值或列表）
        packet_ids = [packet_id] if isinstance(packet_id, (int, str)) else packet_id

        for pid in packet_ids:
            self._debug_print(self.req_network, "req", pid)
            self._debug_print(self.rsp_network, "rsp", pid)
            self._debug_print(self.data_network, "flit", pid)

    def _debug_print(self, net, net_type, packet_id):
        # 取出所有 flit
        flits = net.send_flits.get(packet_id)
        if not flits:
            return

        first_flit: Flit = flits[0] if len(flits) < 2 else flits[-2]
        if first_flit.is_finish or not first_flit.start_inject:
            return

        # 如果已经标记完成，直接跳过
        if self._done_flags.get(net_type, False):
            return

        # 对于单 flit 的 negative rsp，到达后不打印也不更新状态
        if net_type == "rsp":
            last_flit = flits[-1]
            if last_flit.rsp_type == "negative" and len(flits) == 1 and last_flit.is_finish:
                return

        # —— 到这里，说明需要打印调试信息 ——
        if self.cycle != self._last_printed_cycle:
            print(f"Cycle {self.cycle}:")  # 醒目标记当前 cycle
            self._last_printed_cycle = self.cycle  # 更新记录
        print(
            self.req_network.send_flits.get(packet_id),
            self.rsp_network.send_flits.get(packet_id),
            self.data_network.send_flits.get(packet_id),
        )

        # —— 更新完成标记 ——
        last_flit = flits[-1]
        if net_type == "rsp":
            # 只有最后一个 datasend 到达时才算完成
            if last_flit.rsp_type == "datasend" and last_flit.is_finish:
                self._done_flags[net_type] = True
        else:
            # 其他网络类型，只要最后一个 flit 到达就算完成
            if last_flit.is_finish:
                self._done_flags[net_type] = True

        time.sleep(0.3)

    def _flit_move(self, network: Network, flits, flit_type):
        # link 上的flit移动
        link_flits, ring_bridge_EQ_flits = [], []
        for flit in flits:
            if flit.flit_position == "Link":
                link_flits.append(flit)
            if flit.current_link[0] - flit.current_link[1] == self.config.NUM_COL and flit.current_link[1] == flit.destination:
                ring_bridge_EQ_flits.append(flit)
        for flit in link_flits:
            network.plan_move(flit, self.cycle)
        for flit in link_flits:
            if network.execute_moves(flit, self.cycle):
                flits.remove(flit)

        self.Ring_Bridge_arbitration(network)

        # eject arbitration
        self.Eject_Queue_arbitration(network, flit_type)

        for flit in ring_bridge_EQ_flits:
            if flit.is_arrive:
                flits.remove(flit)

        for direction, inject_queues in network.inject_queues.items():
            num, IQ_inject_flits = self.process_inject_queues(network, inject_queues, direction)
            if num == 0 and not IQ_inject_flits:
                continue
            if flit_type == "req":
                self.req_num += num
            elif flit_type == "rsp":
                self.rsp_num += num
            elif flit_type == "data":
                self.flit_num += num
            for flit in IQ_inject_flits:
                if flit not in flits:
                    flits.append(flit)

        RB_inject_flits = self.RB_inject_vertical(network)
        for flit in RB_inject_flits:
            if flit not in flits:
                flits.append(flit)

        network.update_excess_ITag()
        network.update_cross_point()
        return flits

    def Ring_Bridge_arbitration(self, network: Network):
        for col in range(1, self.config.NUM_ROW, 2):
            for row in range(self.config.NUM_COL):
                pos = col * self.config.NUM_COL + row
                next_pos = pos - self.config.NUM_COL

                # 获取各方向的flit
                station_flits = [network.ring_bridge[fifo_name][(pos, next_pos)][0] if network.ring_bridge[fifo_name][(pos, next_pos)] else None for fifo_name in ["TL", "TR"]] + [
                    network.inject_queues[fifo_name][pos][0] if pos in network.inject_queues[fifo_name] and network.inject_queues[fifo_name][pos] else None for fifo_name in ["TU", "TD"]
                ]

                # 处理EQ操作
                if any(station_flits) and len(network.ring_bridge["EQ"][(pos, next_pos)]) < self.config.RB_OUT_FIFO_DEPTH:
                    network.ring_bridge_pre["EQ"][(pos, next_pos)] = self._ring_bridge_arbitrate(network, station_flits, pos, next_pos, "EQ", lambda d, n: d == n)
                # 处理TU操作
                if any(station_flits) and len(network.ring_bridge["TU"][(pos, next_pos)]) < self.config.RB_OUT_FIFO_DEPTH:
                    network.ring_bridge_pre["TU"][(pos, next_pos)] = self._ring_bridge_arbitrate(network, station_flits, pos, next_pos, "TU", lambda d, n: d < n)

                # 处理TD操作
                if any(station_flits) and len(network.ring_bridge["TD"][(pos, next_pos)]) < self.config.RB_OUT_FIFO_DEPTH:
                    network.ring_bridge_pre["TD"][(pos, next_pos)] = self._ring_bridge_arbitrate(network, station_flits, pos, next_pos, "TD", lambda d, n: d > n)

    def RB_inject_vertical(self, network: Network):
        RB_inject_flits = []
        for ip_pos in self.flit_positions:
            next_pos = ip_pos - self.config.NUM_COL
            up_node, down_node = (
                next_pos - self.config.NUM_COL * 2,
                next_pos + self.config.NUM_COL * 2,
            )
            if up_node < 0:
                up_node = next_pos
            if down_node >= self.config.NUM_NODE:
                down_node = next_pos

            # 处理TU方向
            TU_inject_flit = self._process_ring_bridge_inject(network, "TU", ip_pos, next_pos, down_node, up_node)
            if TU_inject_flit:
                RB_inject_flits.append(TU_inject_flit)

            # 处理TD方向
            TD_inject_flit = self._process_ring_bridge_inject(network, "TD", ip_pos, next_pos, up_node, down_node)
            if TD_inject_flit:
                RB_inject_flits.append(TD_inject_flit)

        return RB_inject_flits

    def _ring_bridge_arbitrate(self, network: Network, station_flits, pos, next_pos, direction, cmp_func):
        """
        通用的ring_bridge仲裁函数
        cmp_func: 一个函数，接受 (flit.destination, next_pos)，返回True/False
        """
        RB_flit = None
        rr_index = network.round_robin["RB"][direction][next_pos]
        for i in rr_index:
            if station_flits[i] and cmp_func(station_flits[i].destination, next_pos):
                RB_flit = station_flits[i]
                station_flits[i] = None
                self._update_ring_bridge(network, pos, next_pos, direction, i)
                return RB_flit
        return RB_flit

    def _update_ring_bridge(self, network: Network, pos, next_pos, direction, index):
        """更新transfer stations"""
        if index == 0:
            flit = network.ring_bridge["TL"][(pos, next_pos)].popleft()
            if flit.used_entry_level == "T0":
                network.RB_UE_Counters["TL"][(pos, next_pos)]["T0"] -= 1
            elif flit.used_entry_level == "T1":
                network.RB_UE_Counters["TL"][(pos, next_pos)]["T1"] -= 1
            elif flit.used_entry_level == "T2":
                network.RB_UE_Counters["TL"][(pos, next_pos)]["T2"] -= 1
        elif index == 1:
            flit = network.ring_bridge["TR"][(pos, next_pos)].popleft()
            if flit.used_entry_level == "T1":
                network.RB_UE_Counters["TR"][(pos, next_pos)]["T1"] -= 1
            elif flit.used_entry_level == "T2":
                network.RB_UE_Counters["TR"][(pos, next_pos)]["T2"] -= 1
        elif index == 2:
            flit = network.inject_queues["TU"][pos].popleft()
        elif index == 3:
            flit = network.inject_queues["TD"][pos].popleft()

        if flit.ETag_priority == "T1":
            self.RB_ETag_T1_num_stat += 1
        elif flit.ETag_priority == "T0":
            self.RB_ETag_T0_num_stat += 1

        flit.ETag_priority = "T2"
        # flit.used_entry_level = None
        network.round_robin["RB"][direction][next_pos].remove(index)
        network.round_robin["RB"][direction][next_pos].append(index)

    def Eject_Queue_arbitration(self, network: Network, flit_type):
        """处理eject的仲裁逻辑,根据flit类型处理不同的eject队列"""

        # 1. 映射flit_type到对应的positions
        in_pos_position = self.type_to_positions.get(flit_type)
        if in_pos_position is None:
            return  # 不合法的flit_type

        # 2. 统一处理eject_queues和ring_bridge
        for in_pos in in_pos_position:
            ip_pos = in_pos - self.config.NUM_COL
            # 构造eject_flits
            eject_flits = (
                [network.eject_queues[fifo_pos][ip_pos][0] if network.eject_queues[fifo_pos][ip_pos] else None for fifo_pos in ["TU", "TD"]]
                + [network.inject_queues[fifo_pos][in_pos][0] if network.inject_queues[fifo_pos][in_pos] else None for fifo_pos in ["EQ"]]
                + [network.ring_bridge["EQ"][(in_pos, ip_pos)][0] if network.ring_bridge["EQ"][(in_pos, ip_pos)] else None]
            )
            if not any(eject_flits):
                continue
            self._move_to_eject_queues_pre(network, eject_flits, ip_pos)

    def _process_ring_bridge_inject(self, network: Network, dir_key, pos, next_pos, curr_node, opposite_node):
        direction = dir_key  # "TU" or "TD"
        link = (next_pos, opposite_node)

        # Early return if ring bridge is not active for this direction and position
        if not network.ring_bridge[dir_key][(pos, next_pos)]:
            return None
        flit = network.ring_bridge[dir_key][(pos, next_pos)][0]
        # self.error_log(flit, 6212, 1)
        # Case 1: No flit in the link
        if not network.links[link][0]:
            # Handle empty link cases
            if network.links_tag[link][0] is None:
                if self._update_flit_state(network, dir_key, pos, next_pos, opposite_node, direction):
                    return flit
                return self._handle_wait_cycles(network, dir_key, pos, next_pos, direction, link)

            # Case 2: Has ITag reservation
            if network.links_tag[link][0] == [next_pos, direction]:
                # 使用预约并更新双计数器
                network.remain_tag[direction][next_pos] += 1
                network.tagged_counter[direction][next_pos] -= 1  # 新增：更新tagged计数器
                network.links_tag[link][0] = None

                if self._update_flit_state(network, dir_key, pos, next_pos, opposite_node, direction):
                    return flit
                return self._handle_wait_cycles(network, dir_key, pos, next_pos, direction, link)

        return self._handle_wait_cycles(network, dir_key, pos, next_pos, direction, link)

    def _update_flit_state(self, network: Network, ts_key, ip_pos, next_pos, target_node, direction):
        """更新Flit状态并处理ITag需求变化"""
        if network.links[(next_pos, target_node)][0] is not None:
            return False

        flit: Flit = network.ring_bridge[ts_key][(ip_pos, next_pos)].popleft()

        # 检查这个Flit是否曾经申请过ITag → 减少需求计数
        if flit.wait_cycle_v >= self.config.ITag_TRIGGER_Th_V:
            network.itag_req_counter[direction][ip_pos] -= 1

            # 检查多余预约（内联逻辑）
            excess = network.tagged_counter[direction][ip_pos] - network.itag_req_counter[direction][ip_pos]
            if excess > 0:
                network.excess_ITag_to_remove[direction][ip_pos] += excess

        # 更新Flit位置
        flit.current_position = next_pos
        flit.path_index += 1
        flit.is_new_on_network = False
        flit.itag_v = False
        flit.current_link = (next_pos, target_node)
        flit.current_seat_index = 0
        flit.flit_position = "Link"
        network.execute_moves(flit, self.cycle)
        return True

    def _handle_wait_cycles(self, network: Network, ts_key, pos, next_pos, direction, link):
        """处理等待周期和ITag标记逻辑"""
        if not network.ring_bridge[ts_key][(pos, next_pos)]:
            return None

        first_flit = network.ring_bridge[ts_key][(pos, next_pos)][0]
        first_flit.wait_cycle_v += 1
        # 检查第一个Flit刚达到阈值 → 增加需求计数
        if first_flit.wait_cycle_v == self.config.ITag_TRIGGER_Th_V:
            network.itag_req_counter[direction][pos] += 1

        # 检查是否需要标记ITag（内联所有检查逻辑）
        if (
            first_flit.wait_cycle_v >= self.config.ITag_TRIGGER_Th_V
            and not first_flit.itag_v
            and network.links_tag[link][0] is None
            and network.tagged_counter[direction][pos] < self.config.ITag_MAX_NUM_V
            and network.itag_req_counter[direction][pos] > 0
            and network.remain_tag[direction][pos] > 0
        ):

            # 创建ITag标记（内联逻辑）
            network.remain_tag[direction][pos] -= 1
            network.tagged_counter[direction][pos] += 1
            network.links_tag[link][0] = [pos, direction]
            first_flit.itag_v = True
            self.ITag_v_num_stat += 1

        # 更新所有Flit的等待时间并检查新的需求
        for i, flit in enumerate(network.ring_bridge[ts_key][(pos, next_pos)]):
            if i == 0:
                continue
            flit.wait_cycle_v += 1
            # 检查其他Flit是否刚达到阈值
            if i > 0 and flit.wait_cycle_v == self.config.ITag_TRIGGER_Th_V:
                network.itag_req_counter[direction][pos] += 1

        return None

    def tag_move_all_networks(self):
        self._tag_move(self.req_network)
        self._tag_move(self.rsp_network)
        self._tag_move(self.data_network)

    def _tag_move(self, network: Network):
        # 第一部分：纵向环处理
        for col_start in range(self.config.NUM_COL):
            interval = self.config.NUM_COL * 2  # 8
            col_end = col_start + interval * (self.config.NUM_ROW // 2 - 1)  # col_start + 32

            # 保存起始位置的tag
            last_position = network.links_tag[(col_start, col_start)][0]

            # 前向传递：从起点到终点
            network.links_tag[(col_start, col_start)][0] = network.links_tag[(col_start + interval, col_start)][-1]

            for i in range(1, self.config.NUM_ROW // 2):  # range(1, 5) = [1,2,3,4]
                current_node = col_start + i * interval
                next_node = col_start + (i - 1) * interval

                for j in range(self.config.SLICE_PER_LINK - 1, -1, -1):
                    if j == 0 and current_node == col_end:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                    elif j == 0:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node + interval, current_node)][-1]
                    else:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            # 终点自环处理
            network.links_tag[(col_end, col_end)][-1] = network.links_tag[(col_end, col_end)][0]
            network.links_tag[(col_end, col_end)][0] = network.links_tag[(col_end - interval, col_end)][-1]

            # 回程传递：从终点回到起点
            # 修复：确保处理所有回程连接
            for i in range(1, self.config.NUM_ROW // 2):  # range(1, 5) = [1,2,3,4]
                current_node = col_end - i * interval
                next_node = col_end - (i - 1) * interval

                for j in range(self.config.SLICE_PER_LINK - 1, -1, -1):
                    if j == 0 and current_node == col_start:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                    elif j == 0:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node - interval, current_node)][-1]
                    else:
                        network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            # 起点自环处理
            network.links_tag[(col_start, col_start)][-1] = last_position

        # 第二部分：横向环处理（保持原逻辑）
        # Skip horizontal tag movement if only one column or links_tag missing
        if self.config.NUM_COL <= 1:
            return
        for row_start in range(self.config.NUM_COL, self.config.NUM_NODE, self.config.NUM_COL * 2):
            row_end = row_start + self.config.NUM_COL - 1
            # Existence check for links_tag entry
            if (row_start, row_start) not in network.links_tag:
                continue
            last_position = network.links_tag[(row_start, row_start)][0]
            if (row_start + 1, row_start) in network.links_tag:
                network.links_tag[(row_start, row_start)][0] = network.links_tag[(row_start + 1, row_start)][-1]
            else:
                network.links_tag[(row_start, row_start)][0] = last_position

            for i in range(1, self.config.NUM_COL):
                current_node, next_node = row_start + i, row_start + i - 1
                for j in range(self.config.SLICE_PER_LINK - 1, -1, -1):
                    if j == 0 and current_node == row_end:
                        if (current_node, current_node) in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                    elif j == 0:
                        if (current_node + 1, current_node) in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node + 1, current_node)][-1]
                    else:
                        if (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            if (row_end, row_end) in network.links_tag:
                network.links_tag[(row_end, row_end)][-1] = network.links_tag[(row_end, row_end)][0]
                if (row_end - 1, row_end) in network.links_tag:
                    network.links_tag[(row_end, row_end)][0] = network.links_tag[(row_end - 1, row_end)][-1]
                else:
                    network.links_tag[(row_end, row_end)][0] = last_position

            for i in range(1, self.config.NUM_COL):
                current_node, next_node = row_end - i, row_end - i + 1
                for j in range(self.config.SLICE_PER_LINK - 1, -1, -1):
                    if j == 0 and current_node == row_start:
                        if (current_node, current_node) in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, current_node)][-1]
                    elif j == 0:
                        if (current_node - 1, current_node) in network.links_tag and (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node - 1, current_node)][-1]
                    else:
                        if (current_node, next_node) in network.links_tag:
                            network.links_tag[(current_node, next_node)][j] = network.links_tag[(current_node, next_node)][j - 1]

            if (row_start, row_start) in network.links_tag:
                network.links_tag[(row_start, row_start)][-1] = last_position

    def _move_to_eject_queues_pre(self, network: Network, eject_flits, ip_pos):
        for ip_type in network.EQ_channel_buffer.keys():
            rr_queue = network.round_robin["EQ"][ip_type][ip_pos]
            for i in rr_queue:
                if eject_flits[i] is None:
                    continue
                if eject_flits[i].destination_type == ip_type and len(network.EQ_channel_buffer[ip_type][ip_pos]) < network.config.EQ_CH_FIFO_DEPTH:
                    in_pos = ip_pos + self.config.NUM_COL
                    network.EQ_channel_buffer_pre[ip_type][ip_pos] = eject_flits[i]
                    eject_flits[i].is_arrive = True
                    eject_flits[i].arrival_eject_cycle = self.cycle
                    eject_flits[i] = None
                    if i == 0:
                        flit = network.eject_queues["TU"][ip_pos].popleft()
                        if flit.used_entry_level == "T0":
                            network.EQ_UE_Counters["TU"][ip_pos]["T0"] -= 1
                        elif flit.used_entry_level == "T1":
                            network.EQ_UE_Counters["TU"][ip_pos]["T1"] -= 1
                        elif flit.used_entry_level == "T2":
                            network.EQ_UE_Counters["TU"][ip_pos]["T2"] -= 1
                    elif i == 1:
                        flit = network.eject_queues["TD"][ip_pos].popleft()
                        if flit.used_entry_level == "T1":
                            network.EQ_UE_Counters["TD"][ip_pos]["T1"] -= 1
                        elif flit.used_entry_level == "T2":
                            network.EQ_UE_Counters["TD"][ip_pos]["T2"] -= 1
                    elif i == 2:
                        flit = network.inject_queues["EQ"][in_pos].popleft()
                    elif i == 3:
                        flit = network.ring_bridge["EQ"][(in_pos, ip_pos)].popleft()

                    if flit.ETag_priority == "T1":
                        self.EQ_ETag_T1_num_stat += 1
                    elif flit.ETag_priority == "T0":
                        self.EQ_ETag_T0_num_stat += 1
                    flit.ETag_priority = "T2"

                    rr_queue.remove(i)
                    rr_queue.append(i)
                    return eject_flits
        return eject_flits

    def process_inject_queues(self, network: Network, inject_queues, direction):
        flit_num = 0
        flits = []
        for ip_pos, queue in inject_queues.items():
            if queue and queue[0]:
                flit: Flit = queue.popleft()

                # 检查是否需要生成Buffer_Reach_Th信号
                if flit.wait_cycle_h == self.config.ITag_TRIGGER_Th_H and direction != "EQ":
                    network.itag_req_counter[direction][ip_pos] += 1
                if flit.inject(network):
                    network.inject_num += 1
                    flit_num += 1
                    flit.departure_network_cycle = self.cycle
                    network.plan_move(flit, self.cycle)
                    network.execute_moves(flit, self.cycle)
                    flits.append(flit)

                    # 生成Reduce_ITag_Req信号
                    if flit.itag_h and direction not in ["EQ", "TU", "TD"]:
                        network.itag_req_counter[direction][ip_pos] -= 1
                        flit.itag_h = False

                    if direction in ["EQ", "TU", "TD"]:
                        queue.appendleft(flit)
                        flit.itag_h = False
                else:
                    queue.appendleft(flit)
                    # 更新FIFO中所有Flit的等待时间
                    if direction in ["TR", "TL"]:
                        for f in queue:
                            f.wait_cycle_h += 1
                            # 检查新达到阈值的Flit
                            if f.wait_cycle_h == self.config.ITag_TRIGGER_Th_H:
                                flit.itag_h = True
                                if direction != "EQ":
                                    network.itag_req_counter[direction][ip_pos] += 1
                if flit.itag_h:
                    self.ITag_h_num_stat += 1

        return flit_num, flits

    def process_comprehensive_results(self):
        """处理综合统计结果"""
        if not self.result_save_path:
            return

        self.result_processor.collect_requests_data(self)
        results = self.result_processor.analyze_all_bandwidth()
        self.result_processor.generate_unified_report(results, self.result_save_path)
        self.Total_sum_BW_stat = results["Total_sum_BW"]

        # 额外带宽统计
        read_metrics = results["network_overall"]["read"]
        write_metrics = results["network_overall"]["write"]
        # 非加权 / 加权 带宽
        self.read_unweighted_bw_stat = read_metrics.unweighted_bandwidth
        self.read_weighted_bw_stat = read_metrics.weighted_bandwidth
        self.write_unweighted_bw_stat = write_metrics.unweighted_bandwidth
        self.write_weighted_bw_stat = write_metrics.weighted_bandwidth

        # 延迟统计
        latency_stats = self.result_processor._calculate_latency_stats()
        # CMD 延迟
        self.cmd_read_avg_latency_stat = (latency_stats["cmd"]["read"]["sum"] / latency_stats["cmd"]["read"]["count"]) if latency_stats["cmd"]["read"]["count"] else 0.0
        self.cmd_read_max_latency_stat = latency_stats["cmd"]["read"]["max"]
        self.cmd_write_avg_latency_stat = (latency_stats["cmd"]["write"]["sum"] / latency_stats["cmd"]["write"]["count"]) if latency_stats["cmd"]["write"]["count"] else 0.0
        self.cmd_write_max_latency_stat = latency_stats["cmd"]["write"]["max"]
        # Data 延迟
        self.data_read_avg_latency_stat = (latency_stats["data"]["read"]["sum"] / latency_stats["data"]["read"]["count"]) if latency_stats["data"]["read"]["count"] else 0.0
        self.data_read_max_latency_stat = latency_stats["data"]["read"]["max"]
        self.data_write_avg_latency_stat = (latency_stats["data"]["write"]["sum"] / latency_stats["data"]["write"]["count"]) if latency_stats["data"]["write"]["count"] else 0.0
        self.data_write_max_latency_stat = latency_stats["data"]["write"]["max"]
        # Transaction 延迟
        self.trans_read_avg_latency_stat = (latency_stats["trans"]["read"]["sum"] / latency_stats["trans"]["read"]["count"]) if latency_stats["trans"]["read"]["count"] else 0.0
        self.trans_read_max_latency_stat = latency_stats["trans"]["read"]["max"]
        self.trans_write_avg_latency_stat = (latency_stats["trans"]["write"]["sum"] / latency_stats["trans"]["write"]["count"]) if latency_stats["trans"]["write"]["count"] else 0.0
        self.trans_write_max_latency_stat = latency_stats["trans"]["write"]["max"]

        # Mixed 带宽统计
        mixed_metrics = results["network_overall"]["mixed"]
        self.mixed_unweighted_bw_stat = mixed_metrics.unweighted_bandwidth
        self.mixed_weighted_bw_stat = mixed_metrics.weighted_bandwidth
        # Total average bandwidth stats (unweighted and weighted)
        self.mixed_avg_unweighted_bw_stat = mixed_metrics.unweighted_bandwidth / self.config.NUM_IP
        self.mixed_avg_weighted_bw_stat = mixed_metrics.weighted_bandwidth / self.config.NUM_IP

        # Mixed 延迟统计
        # CMD 混合
        self.cmd_mixed_avg_latency_stat = (latency_stats["cmd"]["mixed"]["sum"] / latency_stats["cmd"]["mixed"]["count"]) if latency_stats["cmd"]["mixed"]["count"] else 0.0
        self.cmd_mixed_max_latency_stat = latency_stats["cmd"]["mixed"]["max"]
        # Data 混合
        self.data_mixed_avg_latency_stat = (latency_stats["data"]["mixed"]["sum"] / latency_stats["data"]["mixed"]["count"]) if latency_stats["data"]["mixed"]["count"] else 0.0
        self.data_mixed_max_latency_stat = latency_stats["data"]["mixed"]["max"]
        # Trans 混合
        self.trans_mixed_avg_latency_stat = (latency_stats["trans"]["mixed"]["sum"] / latency_stats["trans"]["mixed"]["count"]) if latency_stats["trans"]["mixed"]["count"] else 0.0
        self.trans_mixed_max_latency_stat = latency_stats["trans"]["mixed"]["max"]

    def calculate_ip_bandwidth(self, intervals):
        """计算给定区间的加权带宽"""
        total_count = 0
        total_interval_time = 0
        # finish_time = self.cycle // self.config.network_frequency
        for start, end, count in intervals:
            if start >= end:
                continue  # 跳过无效区间
            interval_time = end - start
            # bandwidth = (count * 128) / duration  # 计算该区间的带宽（不除以IP总数）
            # weighted_sum += bandwidth * count  # 加权求和
            total_count += count
            total_interval_time += interval_time

        # return weighted_sum / total_count if total_count > 0 else 0.0
        return total_count * 128 / total_interval_time if total_interval_time > 0 else 0.0

    @lru_cache(maxsize=1024)
    def node_map(self, node, is_source=True):
        if is_source:
            # if self.topo_type_stat in ["5x4"]:
            #     # return self.config.GDMA_SEND_POSITION_LIST[node] if node < 16 else self.config.SDMA_SEND_POSITION_LIST[node % 16]
            #     return self.config.GDMA_SEND_POSITION_LIST[node] if node < 16 else self.config.CDMA_SEND_POSITION_LIST[node - 16]
            # elif self.topo_type_stat == "6x5":
            #     return node % self.config.NUM_COL + self.config.NUM_COL + node // self.config.NUM_COL * 2 * self.config.NUM_COL
            return node % self.config.NUM_COL + self.config.NUM_COL + node // self.config.NUM_COL * 2 * self.config.NUM_COL
        else:
            # if self.topo_type_stat in ["5x4"]:
            #     return self.config.DDR_SEND_POSITION_LIST[node] - self.config.NUM_COL if node < 16 else self.config.L2M_SEND_POSITION_LIST[node % 16] - self.config.NUM_COL
            # elif self.topo_type_stat == "6x5":
            #     return node % self.config.NUM_COL + node // self.config.NUM_COL * 2 * self.config.NUM_COL
            return node % self.config.NUM_COL + node // self.config.NUM_COL * 2 * self.config.NUM_COL

    def get_results(self):
        """
        Extract simulation statistics and configuration variables.

        Returns:
            dict: A combined dictionary of configuration variables and statistics.
        """
        # Get all variables from the sim instance
        self.config.finish_del()

        sim_vars = vars(self)

        # Extract statistics (ending with "_stat")
        results = {key.rsplit("_stat", 1)[0]: value for key, value in sim_vars.items() if key.endswith("_stat")}

        # Add configuration variables
        config_var = {key: value for key, value in vars(self.config).items()}
        results = {**config_var, **results}

        # Clear flit and packet IDs (assuming these are class methods)
        Flit.clear_flit_id()
        Node.clear_packet_id()

        # Add performance statistics
        perf_stats = self.performance_monitor.get_summary()
        results.update(perf_stats)

        # Add object pool statistics
        flit_pool_stats = Flit.get_pool_stats()
        results["flit_pool_size"] = flit_pool_stats["pool_size"]
        results["flit_pool_created"] = flit_pool_stats["created_count"]
        results["flit_pool_reuse_rate"] = (flit_pool_stats["created_count"] - flit_pool_stats["pool_size"]) / max(flit_pool_stats["created_count"], 1)

        # Add result processor analysis for port bandwidth data
        try:
            if hasattr(self, "result_processor") and self.result_processor:
                # Collect request data and analyze bandwidth
                self.result_processor.collect_requests_data(self)
                bandwidth_analysis = self.result_processor.analyze_all_bandwidth()

                # Include port averages in results
                if "port_averages" in bandwidth_analysis:
                    results["port_averages"] = bandwidth_analysis["port_averages"]

                # Include other useful bandwidth metrics
                if "Total_sum_BW" in bandwidth_analysis:
                    results["Total_sum_BW"] = bandwidth_analysis["Total_sum_BW"]

        except Exception as e:
            if hasattr(self, "verbose") and self.verbose:
                print(f"Warning: Could not get port bandwidth analysis: {e}")
            # Set empty port_averages to avoid errors in downstream code
            results["port_averages"] = {}

        return results

    def get_performance_stats(self):
        """Get performance optimization statistics"""
        stats = {
            "simulation_cycles": self.cycle,
            "total_flits_processed": self.trans_flits_num,
            "flit_pool_stats": Flit.get_pool_stats(),
            "cache_hit_info": {
                "node_map_cache_size": getattr(self.node_map, "cache_info", lambda: {"hits": 0, "misses": 0})(),
            },
        }

        # Add I/O performance stats if available
        if hasattr(self.traffic_scheduler, "get_io_stats"):
            stats["io_performance"] = self.traffic_scheduler.get_io_stats()

        return stats
