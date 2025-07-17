"""
通用NoC模型基类。

提供所有NoC拓扑共用的模型功能，包括仿真循环控制、
IP接口管理、性能统计等。各拓扑可以继承并扩展特有功能。
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Type, Tuple
from abc import ABC, abstractmethod
import logging
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

        # 性能统计
        self.global_stats = {
            "total_cycles": 0,
            "total_requests": 0,
            "total_responses": 0,
            "total_data_flits": 0,
            "total_retries": 0,
            "peak_active_requests": 0,
            "current_active_requests": 0,
            "average_latency": 0.0,
            "throughput": 0.0,
            "network_utilization": 0.0,
        }

        # 仿真状态
        self.is_running = False
        self.is_finished = False
        self.start_time = 0.0
        self.end_time = 0.0

        # 事件队列（可选）
        self.event_queue = []

        # 日志配置
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{model_name}")

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

        self.logger.info(f"NoC模型初始化: {model_name}")

    # ========== 抽象方法（拓扑特定实现） ==========

    @abstractmethod
    def _setup_topology_network(self) -> None:
        """设置拓扑网络（拓扑特定）"""
        pass

    @abstractmethod
    def _step_topology_network_compute(self) -> None:
        """拓扑网络计算阶段（拓扑特定）"""
        pass

    @abstractmethod
    def _step_topology_network_update(self) -> None:
        """拓扑网络更新阶段（拓扑特定）"""
        pass

    @abstractmethod
    def _create_topology_instance(self, config) -> 'BaseNoCTopology':
        """创建拓扑实例（子类实现具体拓扑类型）"""
        pass

    def get_topology_info(self) -> Dict[str, Any]:
        """获取拓扑信息（通过拓扑实例）"""
        if hasattr(self, 'topology') and self.topology:
            return self.topology.get_topology_summary()
        return {"type": "unknown", "nodes": 0, "status": "topology_not_initialized"}

    def calculate_path(self, source: NodeId, destination: NodeId) -> List[NodeId]:
        """计算路径（通过拓扑实例）"""
        if hasattr(self, 'topology') and self.topology:
            path_result = self.topology.calculate_route(source, destination)
            return path_result.node_path if hasattr(path_result, 'node_path') else []
        raise NotImplementedError("拓扑实例未初始化，无法计算路径")

    def _get_all_fifos_for_statistics(self) -> Dict[str, Any]:
        """获取所有FIFO用于统计收集（子类可重写）"""
        # 默认返回空字典，子类可以重写此方法
        return {}

    def _register_all_fifos_for_statistics(self) -> None:
        """注册所有FIFO到统计收集器（子类可重写）"""
        # 基类提供默认实现，子类可以重写此方法
        fifos = self._get_all_fifos_for_statistics()
        self.logger.info(f"注册了 {len(fifos)} 个FIFO到统计收集器")

    # ========== 通用方法 ==========

    def initialize_model(self) -> None:
        """初始化模型"""
        try:
            self.logger.info("开始初始化NoC模型...")

            # 创建拓扑实例
            self.logger.info("创建拓扑实例...")
            self.topology = self._create_topology_instance(self.config)
            self.logger.info(f"拓扑实例创建成功: {type(self.topology).__name__}")

            # 设置拓扑网络
            self.logger.info("调用_setup_topology_network...")
            self._setup_topology_network()
            self.logger.info("_setup_topology_network完成")

            # 设置IP接口
            self.logger.info("调用_setup_ip_interfaces...")
            self._setup_ip_interfaces()
            self.logger.info("_setup_ip_interfaces完成")

            # 初始化Flit对象池
            self.logger.info("调用_setup_flit_pools...")
            self._setup_flit_pools()
            self.logger.info("_setup_flit_pools完成")

            self.logger.info(f"NoC模型初始化完成: {len(self.ip_interfaces)}个IP接口")
        except Exception as e:
            self.logger.error(f"NoC模型初始化失败: {e}")
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

        self.logger.info(f"开始优化IP接口创建，分析traffic文件: {self.traffic_file_path}")

        try:
            # 分析traffic文件获取需要的IP接口
            traffic_reader = TrafficFileReader(
                filename=self.traffic_file_path.split("/")[-1], traffic_file_path="/".join(self.traffic_file_path.split("/")[:-1]), config=self.config, time_offset=0, traffic_id="analysis"
            )

            ip_info = traffic_reader.get_required_ip_interfaces()
            required_ips = ip_info["required_ips"]

            self.logger.info(f"Traffic文件分析完成: 需要 {len(required_ips)} 个IP接口，涉及 {len(ip_info['used_nodes'])} 个节点")
            self.logger.info(f"Required IPs: {required_ips}")

            # 调用子类实现的创建方法
            self._create_specific_ip_interfaces(required_ips)

        except Exception as e:
            self.logger.warning(f"Traffic文件分析失败: {e}，回退到全量创建模式")
            import traceback

            traceback.print_exc()
            self._setup_all_ip_interfaces()

    def _setup_all_ip_interfaces(self) -> None:
        """创建所有IP接口（传统模式）- 由子类实现"""
        # 默认实现为空，由子类重写
        self.logger.debug("使用默认的IP接口创建（需要子类实现）")

    def _create_specific_ip_interfaces(self, required_ips: List[Tuple[int, str]]) -> None:
        """创建特定的IP接口 - 由子类实现"""
        # 默认实现为空，由子类重写
        self.logger.debug("创建特定IP接口（需要子类实现）")

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
            self.logger.warning(f"IP接口缺少ip_type属性: {ip_interface}")
            return

        if not hasattr(ip_interface, "node_id") or ip_interface.node_id is None:
            self.logger.warning(f"IP接口缺少node_id属性: {ip_interface}")
            return

        key = f"{ip_interface.ip_type}_{ip_interface.node_id}"
        self._ip_registry[key] = ip_interface
        self.logger.debug(f"注册IP接口: {key}")

    def step(self) -> None:
        """执行一个仿真周期（使用两阶段执行模型）"""
        self.cycle += 1

        # 阶段0：时钟同步阶段 - 确保所有组件使用统一的时钟值
        self._sync_global_clock()

        # 阶段0.1：TrafficScheduler处理请求注入（如果有配置）
        if hasattr(self, "traffic_scheduler") and self.traffic_scheduler:
            ready_requests = self.traffic_scheduler.get_ready_requests(self.cycle)
            if ready_requests:
                injected = self._inject_traffic_requests(ready_requests)
                if injected > 0:
                    print(f"🎯 周期{self.cycle}: 从traffic文件注入了{injected}个请求")

        # 阶段1：组合逻辑阶段 - 所有组件计算传输决策
        self._step_compute_phase()

        # 阶段2：时序逻辑阶段 - 所有组件执行传输和状态更新
        self._step_update_phase()

        # 更新全局统计
        self._update_global_statistics()

        # 调试功能
        if self.debug_enabled:
            self.debug_func()

        # 定期输出调试信息
        if self.cycle % self.debug_config["log_interval"] == 0:
            self._log_periodic_status()

        # Debug模式下的休眠功能
        if self.debug_enabled and self.debug_config["sleep_time"] > 0:
            time.sleep(self.debug_config["sleep_time"])

    def _step_compute_phase(self) -> None:
        """阶段1：组合逻辑阶段 - 所有组件计算传输决策，不修改状态"""
        # 1. 所有IP接口计算阶段
        for ip_interface in self.ip_interfaces.values():
            ip_interface.step_compute_phase(self.cycle)

        # 2. 拓扑网络组件计算阶段
        self._step_topology_network_compute()

    def _step_update_phase(self) -> None:
        """阶段2：时序逻辑阶段 - 所有组件执行传输和状态更新"""
        # 1. 所有IP接口更新阶段
        for ip_interface in self.ip_interfaces.values():
            ip_interface.step_update_phase(self.cycle)

        # 2. 拓扑网络组件更新阶段
        self._step_topology_network_update()

    def _sync_global_clock(self) -> None:
        """时钟同步阶段：确保所有组件使用统一的时钟值"""
        # 同步所有IP接口的时钟
        for ip_interface in self.ip_interfaces.values():
            if hasattr(ip_interface, "current_cycle"):
                ip_interface.current_cycle = self.cycle

    def run_simulation(self, max_cycles: int = 10000, warmup_cycles: int = 1000, stats_start_cycle: int = 1000, convergence_check: bool = True) -> Dict[str, Any]:
        """
        运行完整仿真

        Args:
            max_cycles: 最大仿真周期
            warmup_cycles: 热身周期
            stats_start_cycle: 统计开始周期
            convergence_check: 是否检查收敛

        Returns:
            仿真结果字典
        """
        self.logger.info(f"开始NoC仿真: max_cycles={max_cycles}")

        self.is_running = True
        self.start_time = time.time()
        stats_enabled = False

        try:
            for cycle in range(1, max_cycles + 1):
                self.step()

                # 启用统计收集
                if cycle == stats_start_cycle:
                    stats_enabled = True
                    self._reset_statistics()
                    self.logger.info(f"周期 {cycle}: 开始收集统计数据")

                # 检查仿真结束条件
                if convergence_check and self._should_stop_simulation():
                    self.logger.info(f"周期 {cycle}: 检测到仿真收敛")
                    break

                # 定期输出进度
                if cycle % 5000 == 0:
                    self.logger.info(f"仿真进度: {cycle}/{max_cycles} 周期")

        except KeyboardInterrupt:
            self.logger.warning("仿真被用户中断")

        except Exception as e:
            self.logger.error(f"仿真过程中发生错误: {e}")
            raise

        finally:
            self.is_running = False
            self.is_finished = True
            self.end_time = time.time()

        # 生成仿真结果
        results = self._generate_simulation_results(stats_start_cycle)
        self.logger.info(f"NoC仿真完成: 总周期={self.cycle}")

        return results

    def _should_stop_simulation(self) -> bool:
        """检查是否应该停止仿真"""
        # 简单的收敛判断：没有活跃请求
        active_requests = self.get_total_active_requests()

        if not hasattr(self, "_idle_cycles"):
            self._idle_cycles = 0

        if active_requests == 0:
            self._idle_cycles += 1
        else:
            self._idle_cycles = 0

        return self._idle_cycles >= 1000

    def _reset_statistics(self) -> None:
        """重置统计计数器"""
        self.global_stats = {
            "total_cycles": 0,
            "total_requests": 0,
            "total_responses": 0,
            "total_data_flits": 0,
            "total_retries": 0,
            "peak_active_requests": 0,
            "current_active_requests": 0,
            "average_latency": 0.0,
            "throughput": 0.0,
            "network_utilization": 0.0,
        }

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

    def _update_global_statistics(self) -> None:
        """更新全局统计"""
        self.global_stats["total_cycles"] = self.cycle

        # 汇总IP接口统计
        total_requests = 0
        total_responses = 0
        total_data = 0
        total_retries = 0
        all_latencies = []

        for ip in self._ip_registry.values():
            total_requests += sum(ip.stats["requests_sent"].values())
            total_responses += sum(ip.stats["responses_received"].values())
            total_data += sum(ip.stats["data_transferred"].values())
            total_retries += sum(ip.stats["retries"].values())
            all_latencies.extend(ip.stats["latencies"]["total"])

        self.global_stats["total_requests"] = total_requests
        self.global_stats["total_responses"] = total_responses
        self.global_stats["total_data_flits"] = total_data
        self.global_stats["total_retries"] = total_retries

        # 计算平均延迟
        if all_latencies:
            self.global_stats["average_latency"] = sum(all_latencies) / len(all_latencies)

        # 计算吞吐量
        if self.cycle > 0:
            self.global_stats["throughput"] = total_requests / self.cycle

        # 更新当前活跃请求数
        current_active = self.get_total_active_requests()
        self.global_stats["current_active_requests"] = current_active
        if current_active > self.global_stats["peak_active_requests"]:
            self.global_stats["peak_active_requests"] = current_active

    def _generate_simulation_results(self, stats_start_cycle: int) -> Dict[str, Any]:
        """生成仿真结果"""
        effective_cycles = self.cycle - stats_start_cycle
        simulation_time = self.end_time - self.start_time

        # 汇总IP接口详细统计
        ip_detailed_stats = {}
        for key, ip in self._ip_registry.items():
            ip_detailed_stats[key] = ip.get_status()

        # 汇总Flit池统计
        pool_stats = {}
        for flit_type, pool in self.flit_pools.items():
            pool_stats[flit_type.__name__] = pool.get_stats()

        results = {
            "simulation_info": {
                "model_name": self.model_name,
                "total_cycles": self.cycle,
                "effective_cycles": effective_cycles,
                "simulation_time": simulation_time,
                "cycles_per_second": self.cycle / simulation_time if simulation_time > 0 else 0,
                "config": self._get_config_summary(),
                "topology": self._get_topology_info(),
            },
            "global_stats": self.global_stats.copy(),
            "ip_interface_stats": ip_detailed_stats,
            "memory_stats": {
                "flit_pools": pool_stats,
            },
            "performance_metrics": self._calculate_performance_metrics(),
        }

        return results

    def _get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        # 子类可重写以提供更详细的配置信息
        return {
            "model_type": self.__class__.__name__,
            "ip_interface_count": len(self.ip_interfaces),
        }

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """计算性能指标"""
        metrics = {}

        # 计算延迟分布
        all_latencies = []
        for ip in self._ip_registry.values():
            all_latencies.extend(ip.stats["latencies"]["total"])

        if all_latencies:
            all_latencies.sort()
            n = len(all_latencies)
            metrics["latency_percentiles"] = {
                "p50": all_latencies[int(n * 0.5)],
                "p90": all_latencies[int(n * 0.9)],
                "p95": all_latencies[int(n * 0.95)],
                "p99": all_latencies[int(n * 0.99)],
                "min": min(all_latencies),
                "max": max(all_latencies),
            }

        # 计算重试率
        total_requests = self.global_stats["total_requests"]
        total_retries = self.global_stats["total_retries"]
        if total_requests > 0:
            metrics["retry_rate"] = total_retries / total_requests

        # 计算网络效率
        if self.cycle > 0:
            metrics["network_efficiency"] = {
                "requests_per_cycle": total_requests / self.cycle,
                "data_flits_per_cycle": self.global_stats["total_data_flits"] / self.cycle,
            }

        return metrics

    def _log_periodic_status(self) -> None:
        """定期状态日志"""
        active_requests = self.get_total_active_requests()
        self.logger.debug(f"周期 {self.cycle}: " f"活跃请求={active_requests}, " f"总吞吐={self.global_stats['throughput']:.2f}, " f"平均延迟={self.global_stats['average_latency']:.2f}")

    def get_total_active_requests(self) -> int:
        """获取总活跃请求数"""
        total = 0
        for ip in self._ip_registry.values():
            total += len(ip.active_requests)
        return total

    def inject_request(
        self, source: NodeId, destination: NodeId, req_type: str, count: int = 1, burst_length: int = 4, ip_type: str = None, source_type: str = None, destination_type: str = None, **kwargs
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

        if not ip_interface:
            if ip_type:
                self.logger.warning(f"源节点 {source} 没有找到对应的IP接口 (类型: {ip_type})")
            else:
                self.logger.warning(f"源节点 {source} 没有找到对应的IP接口")
            return packet_ids

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
                source=source, destination=destination, req_type=req_type, burst_length=burst_length, packet_id=packet_id, source_type=source_type, destination_type=destination_type, **kwargs
            )

            if success:
                packet_ids.append(packet_id)
            else:
                self.logger.warning(f"测试请求注入失败: {packet_id}")

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
            matching_ips = [ip for ip in self._ip_registry.values() 
                          if ip.node_id == node_id and getattr(ip, "ip_type", "").startswith(ip_type)]
            if not matching_ips:
                self.logger.error(f"未找到指定IP类型: node_id={node_id}, ip_type={ip_type}")
                return None
        else:
            # 获取该节点的所有IP接口
            matching_ips = [ip for ip in self._ip_registry.values() if ip.node_id == node_id]
            if not matching_ips:
                self.logger.error(f"节点{node_id}没有任何IP接口")
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
        
        self.logger.info(f"TrafficScheduler已设置: {len(traffic_chains)}条链")

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
                    destination_type=dst_type
                )
                
                if packet_ids:
                    injected_count += 1
                    # 更新TrafficScheduler统计
                    if self.traffic_scheduler:
                        self.traffic_scheduler.update_traffic_stats(traffic_id, "injected_req")
                        
            except (ValueError, IndexError) as e:
                self.logger.warning(f"处理traffic请求失败: {e}")
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
            "is_completed": self.traffic_scheduler.is_all_completed()
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
            "topology_info": self._get_topology_info(),
            "performance": {
                "throughput": self.global_stats["throughput"],
                "average_latency": self.global_stats["average_latency"],
                "retry_rate": (self.global_stats["total_retries"] / max(1, self.global_stats["total_requests"])),
            },
        }

    def print_debug_status(self) -> None:
        """打印调试状态"""
        print(f"\n=== {self.model_name} 调试状态 (周期 {self.cycle}) ===")
        print(f"活跃请求总数: {self.get_total_active_requests()}")
        print(f"全局统计: {self.global_stats}")

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

        self.logger.info(f"启用调试跟踪: flits={trace_flits}, channels={trace_channels}")

    def enable_debug(self, level: int = 1, trace_packets: List[str] = None, sleep_time: float = 0.0) -> None:
        """启用调试模式

        Args:
            level: 调试级别 (1-3)
            trace_packets: 要追踪的特定包ID列表
            sleep_time: 每步的睡眠时间(秒)
        """
        self.debug_enabled = True
        self.debug_config["sleep_time"] = sleep_time

        if trace_packets:
            if isinstance(trace_packets, (list, tuple)):
                self.trace_packets.update(trace_packets)
            else:
                self.trace_packets.add(trace_packets)

        # 启用请求跟踪器的调试功能
        if hasattr(self.request_tracker, "enable_debug"):
            self.request_tracker.enable_debug(level, trace_packets)

        self.logger.info(f"调试模式已启用，级别: {level}")
        if trace_packets:
            self.logger.info(f"追踪包: {trace_packets}")
        if sleep_time > 0:
            self.logger.info(f"调试睡眠时间: {sleep_time}s")

    def track_packet(self, packet_id: str) -> None:
        """添加要追踪的包"""
        self.trace_packets.add(packet_id)
        if hasattr(self.request_tracker, "track_packet"):
            self.request_tracker.track_packet(packet_id)
        self.logger.debug(f"开始追踪包: {packet_id}")

    def disable_debug(self) -> None:
        """禁用调试模式"""
        self.debug_enabled = False
        self.trace_packets.clear()
        self.debug_config["sleep_time"] = 0.0
        self.logger.info("调试模式已禁用")

    def add_debug_packet(self, packet_id) -> None:
        """添加要跟踪的packet_id"""
        self.trace_packets.add(packet_id)
        self.logger.info(f"添加调试跟踪: {packet_id}")

    def remove_debug_packet(self, packet_id) -> None:
        """移除跟踪的packet_id"""
        self.trace_packets.discard(packet_id)
        self.logger.info(f"移除调试跟踪: {packet_id}")

    def _should_debug_packet(self, packet_id) -> bool:
        """检查是否应该调试此packet_id"""
        if not self.debug_enabled:
            return False
        # 空集合表示跟踪所有
        if not self.trace_packets:
            return True
        return packet_id in self.trace_packets

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
            print(f"\n全局统计: {self.global_stats}")

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
        if not self.debug_enabled:
            return

        # 默认实现：打印基本状态
        if self.cycle % 100 == 0:  # 每100周期打印一次
            active_requests = self.get_total_active_requests()
            self.logger.debug(f"周期 {self.cycle}: 活跃请求={active_requests}")

        # 追踪特定包
        if self.trace_packets:
            for packet_id in self.trace_packets:
                lifecycle = self.request_tracker.get_request_status(packet_id)
                if lifecycle and lifecycle.current_state != RequestState.COMPLETED:
                    self.logger.debug(f"追踪包 {packet_id}: 状态={lifecycle.current_state.value}")

    def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("开始清理NoC模型资源")

        # 清理IP接口
        for ip in self.ip_interfaces.values():
            # 可以添加IP接口特定的清理逻辑
            pass

        # 清理Flit对象池
        self.flit_pools.clear()

        # 清理统计信息
        self.global_stats.clear()

        self.logger.info("NoC模型资源清理完成")



    def __repr__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}({self.model_name}, " f"cycle={self.cycle}, " f"active_requests={self.get_total_active_requests()})"
