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


class BaseNoCModel(ABC):
    """
    NoC基础模型类。

    提供所有NoC拓扑共用的功能：
    1. 仿真循环控制
    2. IP接口管理
    3. 性能统计收集
    4. 调试和监控功能
    """

    def __init__(self, config: Any, model_name: str = "BaseNoCModel"):
        """
        初始化NoC基础模型

        Args:
            config: 配置对象
            model_name: 模型名称
        """
        self.config = config
        self.model_name = model_name
        self.cycle = 0

        # IP接口管理
        self.ip_interfaces: Dict[str, BaseIPInterface] = {}
        self._ip_registry: Dict[str, BaseIPInterface] = {}

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
        }
        
        # 请求追踪器 - 包含完整的flit追踪功能
        self.request_tracker = RequestTracker(network_frequency=getattr(config, 'network_frequency', 1))

        self.logger.info(f"NoC模型初始化: {model_name}")

    # ========== 抽象方法（拓扑特定实现） ==========

    @abstractmethod
    def _setup_topology_network(self) -> None:
        """设置拓扑网络（拓扑特定）"""
        pass

    @abstractmethod
    def _step_topology_network(self) -> None:
        """拓扑网络步进（拓扑特定）"""
        pass

    @abstractmethod
    def _get_topology_info(self) -> Dict[str, Any]:
        """获取拓扑信息（拓扑特定）"""
        pass

    @abstractmethod
    def _calculate_path(self, source: NodeId, destination: NodeId) -> List[NodeId]:
        """计算路径（拓扑特定）"""
        pass

    # ========== 通用方法 ==========

    def initialize_model(self) -> None:
        """初始化模型"""
        self.logger.info("开始初始化NoC模型...")

        # 设置拓扑网络
        self._setup_topology_network()

        # 设置IP接口
        self._setup_ip_interfaces()

        # 初始化Flit对象池
        self._setup_flit_pools()

        self.logger.info(f"NoC模型初始化完成: {len(self.ip_interfaces)}个IP接口")

    def _setup_ip_interfaces(self) -> None:
        """设置IP接口（通用部分）"""
        # 由子类实现具体的IP接口创建
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
        key = f"{ip_interface.ip_type}_{ip_interface.node_id}"
        self._ip_registry[key] = ip_interface
        self.logger.debug(f"注册IP接口: {key}")

    def step(self) -> None:
        """执行一个仿真周期"""
        self.cycle += 1

        # 处理所有IP接口
        for ip_interface in self.ip_interfaces.values():
            ip_interface.step(self.cycle)

        # 处理拓扑网络
        self._step_topology_network()

        # 更新全局统计
        self._update_global_statistics()

        # 调试输出
        if self.cycle % self.debug_config["log_interval"] == 0:
            self._log_periodic_status()

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

    def inject_test_traffic(self, source: NodeId, destination: NodeId, req_type: str, count: int = 1, burst_length: int = 4, **kwargs) -> List[str]:
        """
        注入测试流量

        Args:
            source: 源节点
            destination: 目标节点
            req_type: 请求类型
            count: 请求数量
            burst_length: 突发长度
            **kwargs: 其他参数

        Returns:
            生成的packet_id列表
        """
        packet_ids = []

        # 找到源节点对应的IP接口
        source_ips = [ip for ip in self._ip_registry.values() if ip.node_id == source]

        if not source_ips:
            self.logger.warning(f"源节点 {source} 没有找到对应的IP接口")
            return packet_ids

        # 使用第一个找到的IP接口
        ip_interface = source_ips[0]

        for i in range(count):
            packet_id = f"test_{source}_{destination}_{req_type}_{self.cycle}_{i}"
            success = ip_interface.enqueue_request(source=source, destination=destination, req_type=req_type, burst_length=burst_length, packet_id=packet_id, **kwargs)

            if success:
                packet_ids.append(packet_id)
            else:
                self.logger.warning(f"测试请求注入失败: {packet_id}")

        return packet_ids

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
        
    # ========== 请求和Flit追踪相关方法 ==========
    
    def start_request_tracking(self, packet_id: str, source: int, destination: int, 
                              op_type: str, burst_size: int) -> None:
        """开始追踪一个新请求"""
        self.request_tracker.start_request(packet_id, source, destination, op_type, burst_size, self.cycle)
        
        if self.debug_config["trace_flits"]:
            self.logger.debug(f"开始追踪请求: {packet_id}")
    
    def track_request_flit(self, packet_id: str, flit, node_id: int = None) -> None:
        """追踪请求flit对象"""
        # 添加flit到RequestLifecycle中
        if packet_id in self.request_tracker.active_requests:
            self.request_tracker.active_requests[packet_id].request_flits.append(flit)
        
        # 追踪flit位置
        if node_id is not None:
            self.request_tracker.track_flit_position(packet_id, FlitType.REQUEST, node_id, self.cycle, flit)
        
        if self.debug_config["trace_flits"]:
            self.logger.debug(f"追踪请求flit: {packet_id} @ 周期{self.cycle}")
    
    def track_response_flit(self, packet_id: str, flit, node_id: int = None) -> None:
        """追踪响应flit对象"""
        # 添加flit到RequestLifecycle中
        if packet_id in self.request_tracker.active_requests:
            self.request_tracker.active_requests[packet_id].response_flits.append(flit)
        elif packet_id in self.request_tracker.completed_requests:
            self.request_tracker.completed_requests[packet_id].response_flits.append(flit)
        
        # 追踪flit位置
        if node_id is not None:
            self.request_tracker.track_flit_position(packet_id, FlitType.RESPONSE, node_id, self.cycle, flit)
        
        if self.debug_config["trace_flits"]:
            self.logger.debug(f"追踪响应flit: {packet_id} @ 周期{self.cycle}")
    
    def track_data_flit(self, packet_id: str, flit, node_id: int = None) -> None:
        """追踪数据flit对象"""
        # 添加flit到RequestLifecycle中
        if packet_id in self.request_tracker.active_requests:
            self.request_tracker.active_requests[packet_id].data_flits.append(flit)
        elif packet_id in self.request_tracker.completed_requests:
            self.request_tracker.completed_requests[packet_id].data_flits.append(flit)
        
        # 追踪flit位置
        if node_id is not None:
            self.request_tracker.track_flit_position(packet_id, FlitType.DATA, node_id, self.cycle, flit)
        
        if self.debug_config["trace_flits"]:
            self.logger.debug(f"追踪数据flit: {packet_id} @ 周期{self.cycle}")
    
    def update_request_state(self, packet_id: str, new_state: RequestState, **kwargs) -> None:
        """更新请求状态"""
        self.request_tracker.update_request_state(packet_id, new_state, self.cycle, **kwargs)
        
        if self.debug_config["trace_flits"]:
            self.logger.debug(f"更新请求状态: {packet_id} -> {new_state.value}")
    
    def print_packet_flit_status(self, packet_id: str) -> None:
        """打印指定包的详细状态，包括flit信息"""
        lifecycle = self.request_tracker.get_request_status(packet_id)
        if not lifecycle:
            print(f"  包 {packet_id} 未找到")
            return
            
        print(f"  包 {packet_id} 的详细状态:")
        print(f"    状态: {lifecycle.current_state.value}")
        print(f"    源: {lifecycle.source} -> 目标: {lifecycle.destination}")
        print(f"    操作: {lifecycle.op_type}, 突发长度: {lifecycle.burst_size}")
        
        # 显示flit信息（利用flit的__repr__方法）
        if lifecycle.request_flits:
            print(f"    请求flits ({len(lifecycle.request_flits)}):")
            for i, flit in enumerate(lifecycle.request_flits):
                print(f"      [{i}] {flit}")
        
        if lifecycle.response_flits:
            print(f"    响应flits ({len(lifecycle.response_flits)}):")
            for i, flit in enumerate(lifecycle.response_flits):
                print(f"      [{i}] {flit}")
        
        if lifecycle.data_flits:
            print(f"    数据flits ({len(lifecycle.data_flits)}):")
            for i, flit in enumerate(lifecycle.data_flits):
                print(f"      [{i}] {flit}")
        
        # 显示路径信息
        if lifecycle.request_path:
            print(f"    请求路径: {lifecycle.request_path[-3:]}...")  # 显示最后3个位置
        if lifecycle.data_path:
            print(f"    数据路径: {lifecycle.data_path[-3:]}...")
    
    def get_packet_flits(self, packet_id: str) -> Dict[str, List[Any]]:
        """获取指定包的所有flit"""
        lifecycle = self.request_tracker.get_request_status(packet_id)
        if lifecycle:
            return {
                'request_flits': lifecycle.request_flits,
                'response_flits': lifecycle.response_flits,
                'data_flits': lifecycle.data_flits
            }
        return {'request_flits': [], 'response_flits': [], 'data_flits': []}
    
    def get_all_tracked_packets(self) -> List[str]:
        """获取所有被追踪的packet_id"""
        active_ids = list(self.request_tracker.active_requests.keys())
        completed_ids = list(self.request_tracker.completed_requests.keys())
        return active_ids + completed_ids
    
    def get_request_tracker_statistics(self) -> Dict[str, Any]:
        """获取请求追踪器统计信息"""
        return self.request_tracker.get_statistics()
    
    def print_request_tracker_report(self) -> None:
        """打印请求追踪器完整报告"""
        self.request_tracker.print_final_report()
    
    def clear_request_tracker(self) -> None:
        """清空请求追踪器"""
        self.request_tracker.reset()
        self.logger.info("请求追踪器已清空")

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
