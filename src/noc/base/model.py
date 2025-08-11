"""
通用NoC模型基类。

提供所有NoC拓扑共用的模型功能，包括仿真循环控制、
IP接口管理、性能统计等。各拓扑可以继承并扩展特有功能。
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Type, Tuple
from abc import ABC, abstractmethod
import os
import time
from collections import defaultdict

from .flit import BaseFlit, FlitPool
from .ip_interface import BaseIPInterface
from src.noc.utils.types import NodeId
from src.noc.debug import RequestTracker, RequestState, FlitType

# 为了避免循环导入，使用TYPE_CHECKING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .topology import BaseNoCTopology


class BaseNoCModel(ABC):
    """
    NoC基础模型类。

    提供所有NoC拓扑共用的功能：
    1. 仿真循环控制
    2. IP接口管理
    3. 性能统计收集
    4. 调试和监控功能
    """

    def __init__(self, config: Any, model_name: str = "BaseNoCModel", traffic_file_path: str = None):
        """
        初始化NoC基础模型

        Args:
            config: 配置对象
            model_name: 模型名称
            traffic_file_path: 可选的traffic文件路径，用于优化IP接口创建
        """
        self.config = config
        self.model_name = model_name
        self.cycle = 0
        self.traffic_file_path = traffic_file_path

        # 拓扑实例（通过组合使用）
        self.topology = None

        # IP接口管理
        self.ip_interfaces: Dict[str, BaseIPInterface] = {}
        self._ip_registry: Dict[str, BaseIPInterface] = {}

        # Traffic调度器（可选）
        self.traffic_scheduler = None

        # Flit对象池
        self.flit_pools: Dict[Type[BaseFlit], FlitPool] = {}

        # 性能统计已移除，改为按需计算

        # 仿真状态
        self.is_running = False
        self.is_finished = False
        self.user_interrupted = False  # 用户中断标志
        self.start_time = 0.0
        self.end_time = 0.0

        # 事件队列（可选）
        self.event_queue = []

        # 日志配置 - 默认设置为CRITICAL级别，只有在明确调用setup_debug时才显示信息

        # 调试配置
        self.debug_config = {
            "trace_flits": False,
            "trace_channels": [],
            "log_interval": 1000,
            "detailed_stats": False,
            "sleep_time": 0.0,  # debug模式下每个周期的休眠时间（秒）
        }

        # 调试模式标志
        self.debug_enabled = False
        self.trace_packets = set()

        # 请求追踪器 - 包含完整的flit追踪功能
        self.request_tracker = RequestTracker(network_frequency=getattr(config, "NETWORK_FREQUENCY", 1))

        # packet_id生成器 - 使用简单数字确保唯一性
        self.next_packet_id = 1
        self.packet_id_map = {}  # {packet_id: {source, destination, req_type, burst_length}}

        # 只在明确启用调试模式时才显示这些信息

    # ========== 抽象方法（拓扑特定实现） ==========

    @abstractmethod
    def _setup_topology_network(self) -> None:
        """设置拓扑网络（拓扑特定）"""
        pass

    # ========== 四阶段执行抽象方法 ==========

    @abstractmethod
    def _step_link_compute_phase(self) -> None:
        """Link层计算阶段：计算slice移动规划，不实际移动flit"""
        pass

    @abstractmethod
    def _step_link_update_phase(self) -> None:
        """Link层更新阶段：执行slice移动，腾空slot[0]位置"""
        pass

    @abstractmethod
    def _step_node_compute_phase(self) -> None:
        """Node层计算阶段：计算注入/弹出/转发决策，不实际传输flit"""
        pass

    @abstractmethod
    def _step_node_update_phase(self) -> None:
        """Node层更新阶段：执行flit传输，包括注入到腾空的slot[0]"""
        pass

    @abstractmethod
    def _create_topology_instance(self, config) -> "BaseNoCTopology":
        """创建拓扑实例（子类实现具体拓扑类型）"""
        pass

    def get_topology_info(self) -> Dict[str, Any]:
        """获取拓扑信息（通过拓扑实例）"""
        if hasattr(self, "topology") and self.topology:
            return self.topology.get_topology_summary()
        return {"type": "unknown", "nodes": 0, "status": "topology_not_initialized"}

    def calculate_path(self, source: NodeId, destination: NodeId) -> List[NodeId]:
        """计算路径（通过拓扑实例）"""
        if hasattr(self, "topology") and self.topology:
            path_result = self.topology.calculate_route(source, destination)
            return path_result.node_path if hasattr(path_result, "node_path") else []
        raise NotImplementedError("拓扑实例未初始化，无法计算路径")

    def _get_all_fifos_for_statistics(self) -> Dict[str, Any]:
        """获取所有FIFO用于统计收集（子类可重写）"""
        # 默认返回空字典，子类可以重写此方法
        return {}

    def _register_all_fifos_for_statistics(self) -> None:
        """注册所有FIFO到统计收集器（子类可重写）"""
        # 基类提供默认实现，子类可以重写此方法
        fifos = self._get_all_fifos_for_statistics()

    # ========== 通用方法 ==========

    def initialize_model(self) -> None:
        """初始化模型"""
        try:

            # 创建拓扑实例
            self.topology = self._create_topology_instance(self.config)

            # 设置拓扑网络
            self._setup_topology_network()

            # IP接口创建延后到setup_traffic_scheduler中进行

            # 初始化Flit对象池
            self._setup_flit_pools()

        except Exception as e:
            import traceback

            traceback.print_exc()
            raise

    def _setup_ip_interfaces(self) -> None:
        """设置IP接口（支持基于traffic文件的优化创建）"""
        # 如果提供了traffic文件路径，使用优化模式
        if self.traffic_file_path:
            self._setup_optimized_ip_interfaces()
        else:
            self._setup_all_ip_interfaces()

    def _setup_optimized_ip_interfaces(self) -> None:
        """基于traffic文件分析，只创建需要的IP接口"""
        from src.noc.utils.traffic_scheduler import TrafficFileReader

        try:
            # 分析traffic文件获取需要的IP接口
            traffic_reader = TrafficFileReader(
                filename=self.traffic_file_path.split("/")[-1],
                traffic_file_path="/".join(self.traffic_file_path.split("/")[:-1]),
                config=self.config,
                time_offset=0,
                traffic_id="analysis",
            )

            ip_info = traffic_reader.get_required_ip_interfaces()
            required_ips = ip_info["required_ips"]

            # 调用子类实现的创建方法
            self._create_specific_ip_interfaces(required_ips)

        except Exception as e:
            import traceback

            traceback.print_exc()
            self._setup_all_ip_interfaces()

    def _setup_all_ip_interfaces(self) -> None:
        """创建所有IP接口（传统模式）- 由子类实现"""
        # 默认实现为空，由子类重写
        pass

    def _create_specific_ip_interfaces(self, required_ips: List[Tuple[int, str]]) -> None:
        """创建特定的IP接口 - 由子类实现"""
        # 默认实现为空，由子类重写
        pass

    def _setup_flit_pools(self) -> None:
        """设置Flit对象池"""
        # 默认使用BaseFlit
        self.flit_pools[BaseFlit] = FlitPool(BaseFlit)

    def register_ip_interface(self, ip_interface: BaseIPInterface) -> None:
        """
        注册IP接口

        Args:
            ip_interface: IP接口实例
        """
        # 验证IP接口的属性
        if not hasattr(ip_interface, "ip_type") or not ip_interface.ip_type:
            return

        if not hasattr(ip_interface, "node_id") or ip_interface.node_id is None:
            return

        key = f"{ip_interface.ip_type}_{ip_interface.node_id}"
        self._ip_registry[key] = ip_interface

    def step(self) -> None:
        """执行一个仿真周期（使用两阶段执行模型）"""
        self.cycle += 1

        # 阶段0：时钟同步阶段 - 确保所有组件使用统一的时钟值
        self._sync_global_clock()

        # 阶段0.1：TrafficScheduler处理请求注入（如果有配置）
        if getattr(self, "traffic_scheduler", None):
            ready_requests = self.traffic_scheduler.get_ready_requests(self.cycle)
            if ready_requests:
                injected = self._inject_traffic_requests(ready_requests)
                if injected > 0:
                    print(f"🎯 周期{self.cycle}: 从traffic文件注入了{injected}个请求")

        # 步骤1：IP接口处理（请求生成和处理）
        for ip_interface in self.ip_interfaces.values():
            ip_interface.step_compute_phase(self.cycle)

        for ip_interface in self.ip_interfaces.values():
            ip_interface.step_update_phase(self.cycle)

        # 步骤2：Node层处理（节点内注入和仲裁）
        self._step_node_compute_phase()

        self._step_node_update_phase()

        # 步骤3：Link层传输（环路slice移动）
        self._step_link_compute_phase()

        self._step_link_update_phase()

        # 调试功能
        if self.debug_enabled:
            self.debug_func()

        # 定期输出调试信息
        if self.cycle % self.debug_config["log_interval"] == 0:
            self._log_periodic_status()

        # Debug休眠已移至具体模型的_print_debug_info中，只有在打印信息时才执行

    def _sync_global_clock(self) -> None:
        """时钟同步阶段：确保所有组件使用统一的时钟值"""
        # 同步所有IP接口的时钟
        for ip_interface in self.ip_interfaces.values():
            if hasattr(ip_interface, "current_cycle"):  # 保留hasattr，因为这是接口兼容性检查
                ip_interface.current_cycle = self.cycle

    def run_simulation(
        self, max_time_ns: float = 5000.0, stats_start_time_ns: float = 0.0, progress_interval_ns: float = 1000.0, results_analysis: bool = False, verbose: bool = True
    ) -> Dict[str, Any]:
        """
        运行完整仿真

        Args:
            max_time_ns: 最大仿真时间（纳秒）
            stats_start_time_ns: 统计开始时间（纳秒）
            progress_interval_ns: 进度显示间隔（纳秒）
            results_analysis: 是否在仿真结束后执行结果分析
            verbose: 是否打印详细的模型信息和中间结果

        Returns:
            仿真结果字典
        """
        # 获取网络频率进行ns到cycle的转换
        network_freq = 1.0  # 默认1GHz
        if hasattr(self.config, "basic_config") and hasattr(self.config.basic_config, "NETWORK_FREQUENCY"):
            network_freq = self.config.basic_config.NETWORK_FREQUENCY
        elif hasattr(self.config, "NETWORK_FREQUENCY"):
            network_freq = self.config.NETWORK_FREQUENCY
        elif hasattr(self.config, "clock_frequency"):
            network_freq = self.config.clock_frequency

        # ns转换为cycle：cycle = time_ns * frequency_GHz
        max_cycles = int(max_time_ns * network_freq)
        stats_start_cycle = int(stats_start_time_ns * network_freq)
        progress_interval = int(progress_interval_ns * network_freq)

        cycle_time_ns = 1.0 / network_freq  # 1个周期的纳秒数

        # 如果启用详细模式，打印traffic统计信息
        if verbose and hasattr(self, "traffic_scheduler") and self.traffic_scheduler:
            self._print_traffic_statistics()

        self.is_running = True
        self.start_time = time.time()

        try:
            for cycle in range(1, max_cycles + 1):
                self.step()

                # 启用统计收集
                if cycle == stats_start_cycle:
                    self._reset_statistics()

                # 检查仿真结束条件（总是检查）
                if self._should_stop_simulation():
                    break

                # 定期输出进度
                if cycle % progress_interval == 0 and cycle > 0:
                    if verbose:
                        self._print_simulation_progress(cycle, max_cycles, progress_interval)
                    else:
                        active_requests = self.get_total_active_requests()
                        completed_requests = 0
                        if hasattr(self, "request_tracker") and self.request_tracker:
                            completed_requests = len(self.request_tracker.completed_requests)

                        # 计算时间（ns）
                        current_time_ns = cycle * cycle_time_ns

        except KeyboardInterrupt:
            print("🛑 用户中断仿真，正在进行结果分析...")
            self.user_interrupted = True
            # 不重新抛出异常，继续执行结果分析
        except Exception as e:
            raise

        finally:
            self.is_running = False
            self.is_finished = True
            self.end_time = time.time()

        # 生成仿真结果
        results = self._generate_simulation_results(stats_start_cycle)

        # 如果启用详细模式，打印最终统计信息
        if verbose:
            self._print_final_statistics()

        # 结果分析（如果启用）
        if results_analysis and hasattr(self, "analyze_simulation_results"):
            try:
                analysis_results = self.analyze_simulation_results(results, enable_visualization=True, save_results=True, verbose=verbose)
                results["analysis"] = analysis_results
            except Exception as e:
                print(f"结果分析过程中出错: {e}")

        return results

    def _should_stop_simulation(self) -> bool:
        """检查是否应该停止仿真"""
        # 如果用户中断，立即停止
        if self.user_interrupted:
            return True

        # 直接从TrafficScheduler获取总请求数（已经计算过的）
        total_requests = 0
        if hasattr(self, "traffic_scheduler") and self.traffic_scheduler:
            total_requests = self.traffic_scheduler.get_total_requests()

        completed_requests = 0
        if hasattr(self, "request_tracker") and self.request_tracker:
            completed_requests = len(self.request_tracker.completed_requests)

        # 直接比较：如果所有请求都完成了就停止
        return completed_requests >= total_requests and total_requests > 0

    def _reset_statistics(self) -> None:
        """重置统计计数器"""
        # 统计已移至按需计算，这里只重置IP接口统计
        for ip in self._ip_registry.values():
            ip.reset_stats()

        # 重置IP接口统计
        for ip in self._ip_registry.values():
            ip.stats = {
                "requests_sent": {"read": 0, "write": 0},
                "responses_received": {"ack": 0, "nack": 0},
                "data_transferred": {"sent": 0, "received": 0},
                "retries": {"read": 0, "write": 0},
                "latencies": {"injection": [], "network": [], "total": []},
                "throughput": {"requests_per_cycle": 0.0, "data_per_cycle": 0.0},
            }

    def _generate_simulation_results(self, stats_start_cycle: int) -> Dict[str, Any]:
        """生成仿真结果"""
        simulation_time = self.end_time - self.start_time

        # 汇总IP接口详细统计
        ip_detailed_stats = {}
        for key, ip in self._ip_registry.items():
            ip_detailed_stats[key] = ip.get_status()

        # 汇总Flit池统计
        pool_stats = {}
        for flit_type, pool in self.flit_pools.items():
            pool_stats[flit_type.__name__] = pool.get_stats()

        # 计算全局统计（替代原来的global_stats）
        total_requests = 0
        total_responses = 0
        total_data_flits = 0
        total_retries = 0
        all_latencies = []

        for ip in self._ip_registry.values():
            total_requests += sum(ip.stats["requests_sent"].values())
            total_responses += sum(ip.stats["responses_received"].values())
            total_data_flits += sum(ip.stats["data_transferred"].values())
            total_retries += sum(ip.stats["retries"].values())
            all_latencies.extend(ip.stats["latencies"]["total"])

        # 计算平均延迟和吞吐量
        average_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
        throughput = total_requests / self.cycle if self.cycle > 0 else 0.0

        results = {
            "simulation_info": {
                "model_name": self.model_name,
                "total_cycles": self.cycle,
                "simulation_time": simulation_time,
                "cycles_per_second": self.cycle / simulation_time if simulation_time > 0 else 0,
                "config": self._get_config_summary(),
                "topology": self._get_topology_info() if hasattr(self, "_get_topology_info") else {},
            },
            "global_stats": {
                "total_cycles": self.cycle,
                "total_requests": total_requests,
                "total_responses": total_responses,
                "total_data_flits": total_data_flits,
                "total_retries": total_retries,
                "average_latency": average_latency,
                "throughput": throughput,
            },
            "ip_interface_stats": ip_detailed_stats,
            "memory_stats": {
                "flit_pools": pool_stats,
            },
            "performance_metrics": self._calculate_performance_metrics_direct(total_requests, total_retries, total_data_flits, all_latencies),
        }

        return results

    def _get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        # 子类可重写以提供更详细的配置信息
        return {
            "model_type": self.__class__.__name__,
            "ip_interface_count": len(self.ip_interfaces),
        }

    def _calculate_performance_metrics_direct(self, total_requests: int, total_retries: int, total_data_flits: int, all_latencies: list) -> Dict[str, Any]:
        """直接计算性能指标（不依赖global_stats）"""
        metrics = {}

        # 计算延迟分布
        if all_latencies:
            all_latencies_sorted = sorted(all_latencies)
            n = len(all_latencies_sorted)
            metrics["latency_percentiles"] = {
                "p50": all_latencies_sorted[int(n * 0.5)],
                "p90": all_latencies_sorted[int(n * 0.9)],
                "p95": all_latencies_sorted[int(n * 0.95)],
                "p99": all_latencies_sorted[int(n * 0.99)],
                "min": min(all_latencies_sorted),
                "max": max(all_latencies_sorted),
            }

        # 计算重试率
        if total_requests > 0:
            metrics["retry_rate"] = total_retries / total_requests

        # 计算网络效率
        if self.cycle > 0:
            metrics["network_efficiency"] = {
                "requests_per_cycle": total_requests / self.cycle,
                "data_flits_per_cycle": total_data_flits / self.cycle,
            }

        return metrics

    def _log_periodic_status(self) -> None:
        """定期状态日志"""
        active_requests = self.get_total_active_requests()

    def get_total_active_requests(self) -> int:
        """获取总活跃请求数"""
        total = 0
        for ip in self._ip_registry.values():
            total += len(ip.active_requests)
        return total

    def _print_simulation_progress(self, cycle: int, max_cycles: int, progress_interval: int) -> None:
        """打印仿真进度统计信息（详细模式）"""
        # 计算时间（ns） - 从配置获取网络频率
        network_freq = 1.0  # 默认1GHz
        if hasattr(self.config, "basic_config") and hasattr(self.config.basic_config, "NETWORK_FREQUENCY"):
            network_freq = self.config.basic_config.NETWORK_FREQUENCY
        elif hasattr(self.config, "NETWORK_FREQUENCY"):
            network_freq = self.config.NETWORK_FREQUENCY

        cycle_time_ns = 1.0 / network_freq  # ns/cycle
        current_time_ns = cycle * cycle_time_ns

        # 获取基本统计信息
        active_requests = 0
        completed_requests = 0
        injected_requests = 0
        response_count = 0
        received_flits = 0

        if hasattr(self, "request_tracker") and self.request_tracker:
            active_requests = len(self.request_tracker.active_requests)
            completed_requests = len(self.request_tracker.completed_requests)
            injected_requests = active_requests + completed_requests

            # 统计响应flit数量
            for req_info in self.request_tracker.active_requests.values():
                response_count += len(req_info.response_flits)
            for req_info in self.request_tracker.completed_requests.values():
                response_count += len(req_info.response_flits)

        # 获取traffic统计信息 - 按请求类型分类统计
        read_finish_count = 0
        write_finish_count = 0
        trans_finish_count = completed_requests

        if hasattr(self, "request_tracker") and self.request_tracker:
            for req_info in self.request_tracker.completed_requests.values():
                if hasattr(req_info, "op_type"):
                    if req_info.op_type == "read":
                        read_finish_count += 1
                    elif req_info.op_type == "write":
                        write_finish_count += 1

        # 计算传输的数据flit数量 - 基于已完成请求的burst_length
        total_data_flits = 0
        if hasattr(self, "request_tracker") and self.request_tracker:
            for req_id, req_info in self.request_tracker.completed_requests.items():
                if hasattr(req_info, "burst_size"):
                    total_data_flits += req_info.burst_size
        received_flits = total_data_flits

        # 打印进度信息（中文格式）
        print(
            f"时间: {int(current_time_ns)}ns, 总请求: {injected_requests}, 活跃请求: {active_requests}, "
            f"读完成: {read_finish_count}, 写完成: {write_finish_count}, 传输完成: {trans_finish_count}, "
            f"传输响应: {response_count}, 传输数据: {received_flits}"
        )

    def _print_traffic_statistics(self) -> None:
        """打印traffic统计信息（在仿真开始时）"""
        if not hasattr(self, "traffic_scheduler") or not self.traffic_scheduler:
            return

        # 统计所有traffic文件的请求和flit数量
        total_read_req = 0
        total_write_req = 0
        total_read_flit = 0
        total_write_flit = 0

        for chain in self.traffic_scheduler.parallel_chains:
            for traffic_file in chain.traffic_files:
                try:
                    # 快速扫描traffic文件获取统计信息
                    abs_path = os.path.join(self.traffic_scheduler.traffic_file_path, traffic_file)
                    with open(abs_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#"):
                                continue

                            # 支持逗号和空格分隔符
                            if "," in line:
                                parts = line.split(",")
                            else:
                                parts = line.split()

                            if len(parts) >= 7:
                                try:
                                    op = parts[5]
                                    burst = int(parts[6])

                                    if op.upper() in ["R", "READ"]:
                                        total_read_req += 1
                                        total_read_flit += burst
                                    else:
                                        total_write_req += 1
                                        total_write_flit += burst
                                except (ValueError, IndexError):
                                    continue
                except Exception as e:
                    continue

        total_req = total_read_req + total_write_req
        total_flit = total_read_flit + total_write_flit

        print(f"数据统计: 读: ({total_read_req}, {total_read_flit}), " f"写: ({total_write_req}, {total_write_flit}), " f"总计: ({total_req}, {total_flit})")

    def _print_final_statistics(self) -> None:
        """打印最终统计信息"""
        print("仿真完成!")

        # 计算仿真用时统计
        simulation_time = self.end_time - self.start_time
        cycles_per_second = self.cycle / simulation_time if simulation_time > 0 else 0

        print(f"仿真用时: {simulation_time:.2f} 秒")
        print(f"处理周期数: {self.cycle} 个周期")
        print(f"仿真性能: {cycles_per_second:.0f} 周期/秒")

        # 计算时间（ns） - 从配置获取网络频率
        network_freq = 1.0  # 默认1GHz
        if hasattr(self.config, "basic_config") and hasattr(self.config.basic_config, "NETWORK_FREQUENCY"):
            network_freq = self.config.basic_config.NETWORK_FREQUENCY
        elif hasattr(self.config, "NETWORK_FREQUENCY"):
            network_freq = self.config.NETWORK_FREQUENCY

        cycle_time_ns = 1.0 / network_freq  # ns/cycle
        current_time_ns = self.cycle * cycle_time_ns

        # 获取最终统计信息
        active_requests = 0
        completed_requests = 0
        injected_requests = 0
        response_count = 0
        received_flits = 0

        if hasattr(self, "request_tracker") and self.request_tracker:
            active_requests = len(self.request_tracker.active_requests)
            completed_requests = len(self.request_tracker.completed_requests)
            injected_requests = active_requests + completed_requests

            # 统计响应flit数量
            for req_info in self.request_tracker.active_requests.values():
                response_count += len(req_info.response_flits)
            for req_info in self.request_tracker.completed_requests.values():
                response_count += len(req_info.response_flits)

        # 按请求类型分类统计完成数量
        read_finish_count = 0
        write_finish_count = 0
        if hasattr(self, "request_tracker") and self.request_tracker:
            for req_info in self.request_tracker.completed_requests.values():
                if hasattr(req_info, "op_type"):
                    if req_info.op_type == "read":
                        read_finish_count += 1
                    elif req_info.op_type == "write":
                        write_finish_count += 1

        # 计算传输的数据flit数量 - 基于已完成请求的burst_length
        total_data_flits = 0
        if hasattr(self, "request_tracker") and self.request_tracker:
            for req_id, req_info in self.request_tracker.completed_requests.items():
                if hasattr(req_info, "burst_size"):
                    total_data_flits += req_info.burst_size
        received_flits = total_data_flits

        print(
            f"时间: {int(current_time_ns)}ns, 总请求: {injected_requests}, 活跃请求: {active_requests}, "
            f"读完成: {read_finish_count}, 写完成: {write_finish_count}, 传输完成: {completed_requests}, "
            f"传输响应: {response_count}, 传输数据: {received_flits}"
        )

    def inject_request(
        self,
        source: NodeId,
        destination: NodeId,
        req_type: str,
        count: int = 1,
        burst_length: int = 4,
        ip_type: str = None,
        source_type: str = None,
        destination_type: str = None,
        **kwargs,
    ) -> List[str]:
        """
        注入请求

        Args:
            source: 源节点
            destination: 目标节点
            req_type: 请求类型
            count: 请求数量
            burst_length: 突发长度
            ip_type: IP类型（可选）
            source_type: 源IP类型（从traffic文件获取）
            destination_type: 目标IP类型（从traffic文件获取）
            **kwargs: 其他参数

        Returns:
            生成的packet_id列表
        """
        packet_ids = []

        # 找到合适的IP接口
        ip_interface = self._find_ip_interface_for_request(source, req_type, ip_type)

        for i in range(count):
            # 生成简单的数字packet_id
            packet_id = self.next_packet_id
            self.next_packet_id += 1

            # 保存packet_id映射信息
            self.packet_id_map[packet_id] = {
                "source": source,
                "destination": destination,
                "req_type": req_type,
                "burst_length": burst_length,
                "cycle": self.cycle,
                "source_type": source_type,
                "destination_type": destination_type,
            }

            success = ip_interface.inject_request(
                source=source,
                destination=destination,
                req_type=req_type,
                burst_length=burst_length,
                packet_id=packet_id,
                source_type=source_type,
                destination_type=destination_type,
                **kwargs,
            )

            if success:
                packet_ids.append(packet_id)

        return packet_ids

    def get_packet_info(self, packet_id) -> Optional[Dict[str, Any]]:
        """获取packet_id的详细信息"""
        return self.packet_id_map.get(packet_id)

    def print_packet_id_map(self) -> None:
        """打印packet_id映射表"""
        if not self.packet_id_map:
            print("📦 尚未生成任何packet")
            return

        print(f"\n📦 生成的Packet列表 (共{len(self.packet_id_map)}个):")
        print("=" * 60)
        for packet_id, info in self.packet_id_map.items():
            src_type = info["source_type"] if info["source_type"] else "??"
            dst_type = info["destination_type"] if info["destination_type"] else "??"
            print(f"  {packet_id}: {info['source']}:{src_type} -> {info['destination']}:{dst_type} " f"({info['req_type']}, burst={info['burst_length']})")
        print("=" * 60)

    def _find_ip_interface(self, node_id: NodeId, req_type: str = None, ip_type: str = None) -> Optional[BaseIPInterface]:
        """
        通用IP接口查找方法 (base版本)

        Args:
            node_id: 节点ID
            req_type: 请求类型 (可选)
            ip_type: IP类型 (可选)

        Returns:
            找到的IP接口，未找到返回None
        """
        if ip_type:
            # 精确匹配指定IP类型
            matching_ips = [ip for ip in self._ip_registry.values() if ip.node_id == node_id and getattr(ip, "ip_type", "").startswith(ip_type)]
            if not matching_ips:
                # 调试：显示当前注册的所有IP
                return None
        else:
            # 获取该节点的所有IP接口
            matching_ips = [ip for ip in self._ip_registry.values() if ip.node_id == node_id]
            if not matching_ips:
                return None

        return matching_ips[0]

    def _find_ip_interface_for_request(self, node_id: NodeId, req_type: str, ip_type: str = None) -> Optional[BaseIPInterface]:
        """为请求查找合适的IP接口"""
        return self._find_ip_interface(node_id, req_type, ip_type)

    # ========== TrafficScheduler集成方法 ==========

    def setup_traffic_scheduler(self, traffic_chains: List[List[str]], traffic_file_path: str = None) -> None:
        """
        设置TrafficScheduler

        Args:
            traffic_chains: traffic链配置，每个链包含文件名列表
            traffic_file_path: traffic文件路径，默认使用初始化时的路径
        """
        from src.noc.utils.traffic_scheduler import TrafficScheduler

        file_path = traffic_file_path or self.traffic_file_path or "traffic_data"
        self.traffic_scheduler = TrafficScheduler(self.config, file_path)
        self.traffic_scheduler.setup_parallel_chains(traffic_chains)
        self.traffic_scheduler.start_initial_traffics()

    def _inject_traffic_requests(self, ready_requests: List[Tuple]) -> int:
        """
        注入TrafficScheduler提供的请求

        Args:
            ready_requests: 准备就绪的请求列表

        Returns:
            成功注入的请求数量
        """
        injected_count = 0

        for req in ready_requests:
            try:
                cycle, src, src_type, dst, dst_type, op, burst, traffic_id = req
                op_type = "read" if op.upper() == "R" else "write"

                packet_ids = self.inject_request(
                    source=src,
                    destination=dst,
                    req_type=op_type,
                    count=1,
                    burst_length=burst,
                    ip_type=src_type,
                    source_type=src_type,
                    destination_type=dst_type,
                    inject_cycle=cycle,  # 传递正确的注入时间
                )

                if packet_ids:
                    injected_count += 1
                    # 更新TrafficScheduler统计
                    if self.traffic_scheduler:
                        self.traffic_scheduler.update_traffic_stats(traffic_id, "injected_req")

            except (ValueError, IndexError) as e:
                continue

        return injected_count

    def get_traffic_status(self) -> Dict[str, Any]:
        """获取traffic调度器状态"""
        if not self.traffic_scheduler:
            return {"status": "未配置TrafficScheduler"}

        return {
            "active_traffics": self.traffic_scheduler.get_active_traffic_count(),
            "chain_status": self.traffic_scheduler.get_chain_status(),
            "has_pending": self.traffic_scheduler.has_pending_requests(),
            "is_completed": self.traffic_scheduler.is_all_completed(),
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        return {
            "model_name": self.model_name,
            "model_type": self.__class__.__name__,
            "current_cycle": self.cycle,
            "total_ip_interfaces": len(self.ip_interfaces),
            "active_requests": self.get_total_active_requests(),
            "simulation_status": {
                "is_running": self.is_running,
                "is_finished": self.is_finished,
            },
            "topology_info": self._get_topology_info() if hasattr(self, "_get_topology_info") else {},
            "performance": {
                "note": "性能统计已移至仿真结束时计算",
            },
        }

    def print_debug_status(self) -> None:
        """打印调试状态"""
        print(f"\n=== {self.model_name} 调试状态 (周期 {self.cycle}) ===")
        print(f"活跃请求总数: {self.get_total_active_requests()}")
        print(f"当前周期: {self.cycle}")

        if self.debug_config["detailed_stats"]:
            print("\nIP接口详细状态:")
            for key, ip in self._ip_registry.items():
                status = ip.get_status()
                print(f"  {key}: 活跃={status['active_requests']}, 完成={status['completed_requests']}")

    def enable_debug_tracing(self, trace_flits: bool = True, trace_channels: List[str] = None, detailed_stats: bool = True) -> None:
        """启用调试跟踪"""
        self.debug_config["trace_flits"] = trace_flits
        self.debug_config["trace_channels"] = trace_channels or ["req", "rsp", "data"]
        self.debug_config["detailed_stats"] = detailed_stats

    def setup_debug(self, level: int = 1, trace_packets: List[str] = None, sleep_time: float = 0.0) -> None:
        """启用调试模式

        Args:
            level: 调试级别 (1-3)
            trace_packets: 要追踪的特定包ID列表
            sleep_time: 每步的睡眠时间(秒)
        """

        self.debug_enabled = True
        self.debug_config["sleep_time"] = sleep_time

        if trace_packets:
            if isinstance(trace_packets, (list, tuple, set)):
                self.trace_packets.update(trace_packets)
            else:
                self.trace_packets.add(trace_packets)

        # 启用请求跟踪器的调试功能
        if hasattr(self.request_tracker, "enable_debug"):
            self.request_tracker.enable_debug(level, trace_packets)

    def track_packet(self, packet_id: str) -> None:
        """添加要追踪的包"""
        self.trace_packets.add(packet_id)
        if hasattr(self.request_tracker, "track_packet"):
            self.request_tracker.track_packet(packet_id)

    def disable_debug(self) -> None:
        """禁用调试模式"""
        self.debug_enabled = False
        self.trace_packets.clear()
        self.debug_config["sleep_time"] = 0.0

    def add_debug_packet(self, packet_id) -> None:
        """添加要跟踪的packet_id"""
        self.trace_packets.add(packet_id)

    def remove_debug_packet(self, packet_id) -> None:
        """移除跟踪的packet_id"""
        self.trace_packets.discard(packet_id)

    def _should_debug_packet(self, packet_id) -> bool:
        """检查是否应该调试此packet_id"""
        if not self.debug_enabled:
            return False
        # 空集合表示跟踪所有
        if not self.trace_packets:
            return True
        # 支持整数和字符串形式的packet_id比较
        return packet_id in self.trace_packets or str(packet_id) in self.trace_packets

    def print_debug_report(self) -> None:
        """打印调试报告"""
        if not self.debug_enabled:
            print("调试模式未启用")
            return

        print(f"\n=== {self.model_name} 调试报告 ===")
        print(f"当前周期: {self.cycle}")
        print(f"活跃请求: {self.get_total_active_requests()}")

        # 打印请求追踪器报告
        if hasattr(self.request_tracker, "print_final_report"):
            self.request_tracker.print_final_report()

        # 打印统计信息
        if self.debug_config["detailed_stats"]:
            print(f"\n当前周期: {self.cycle}, 活跃请求: {self.get_total_active_requests()}")

    def validate_traffic_correctness(self) -> Dict[str, Any]:
        """验证流量的正确性"""
        if not hasattr(self.request_tracker, "get_statistics"):
            return {"error": "请求追踪器不支持统计"}

        stats = self.request_tracker.get_statistics()

        validation_result = {
            "total_requests": stats.get("total_requests", 0),
            "completed_requests": stats.get("completed_requests", 0),
            "failed_requests": stats.get("failed_requests", 0),
            "completion_rate": stats.get("completed_requests", 0) / max(1, stats.get("total_requests", 1)) * 100,
            "response_errors": stats.get("response_errors", 0),
            "data_errors": stats.get("data_errors", 0),
            "avg_latency": stats.get("avg_latency", 0.0),
            "max_latency": stats.get("max_latency", 0),
            "is_correct": stats.get("response_errors", 0) == 0 and stats.get("data_errors", 0) == 0,
        }

        return validation_result

    # ========== 调试相关方法 ==========

    def debug_func(self) -> None:
        """主调试函数，每个周期调用（可被子类重写）"""
        pass

    def cleanup(self) -> None:
        """清理资源"""

        # 清理IP接口
        for ip in self.ip_interfaces.values():
            # 可以添加IP接口特定的清理逻辑
            pass

        # 清理Flit对象池
        self.flit_pools.clear()

        # 统计信息已移除

    def __repr__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}({self.model_name}, " f"cycle={self.cycle}, " f"active_requests={self.get_total_active_requests()})"
