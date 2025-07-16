"""
通用NoC IP接口基类。

提供所有NoC拓扑共用的IP接口功能，包括时钟域转换、
基础资源管理、STI协议处理等。各拓扑可以继承并扩展特有功能。
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Deque, Type
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import logging

from .flit import BaseFlit
from src.noc.utils.types import NodeId


class FIFOStatistics:
    """FIFO统计信息收集器"""
    
    def __init__(self):
        # 利用率统计
        self.current_depth = 0
        self.peak_depth = 0
        self.depth_sum = 0
        self.sample_count = 0
        self.empty_cycles = 0
        self.full_cycles = 0
        
        # 吞吐量统计
        self.total_writes_attempted = 0
        self.total_writes_successful = 0
        self.total_reads_attempted = 0
        self.total_reads_successful = 0
        
        # 流控统计
        self.write_stalls = 0
        self.read_stalls = 0
        self.overflow_attempts = 0
        self.underflow_attempts = 0
        
        # 延迟统计
        self.flit_timestamps = {}  # {flit_id: enter_time}
        self.residence_times = []
        
        # 行为模式统计
        self.priority_writes = 0
        self.total_simulation_cycles = 0
        self.active_cycles = 0  # 有数据传输的周期
        
    def update_depth_stats(self, current_depth: int, max_capacity: int, cycle: int):
        """更新深度相关统计"""
        self.current_depth = current_depth
        self.peak_depth = max(self.peak_depth, current_depth)
        self.depth_sum += current_depth
        self.sample_count += 1
        self.total_simulation_cycles = cycle
        
        if current_depth == 0:
            self.empty_cycles += 1
        elif current_depth == max_capacity:
            self.full_cycles += 1
            
    def record_write_attempt(self, successful: bool, is_priority: bool = False):
        """记录写入尝试"""
        self.total_writes_attempted += 1
        if successful:
            self.total_writes_successful += 1
            self.active_cycles += 1
            if is_priority:
                self.priority_writes += 1
        else:
            self.write_stalls += 1
            
    def record_read_attempt(self, successful: bool):
        """记录读取尝试"""
        self.total_reads_attempted += 1
        if successful:
            self.total_reads_successful += 1
            self.active_cycles += 1
        else:
            self.read_stalls += 1
            
    def record_flit_enter(self, flit_id: str, cycle: int):
        """记录flit进入时间"""
        self.flit_timestamps[flit_id] = cycle
        
    def record_flit_exit(self, flit_id: str, cycle: int):
        """记录flit离开时间并计算停留时间"""
        if flit_id in self.flit_timestamps:
            residence_time = cycle - self.flit_timestamps[flit_id]
            self.residence_times.append(residence_time)
            del self.flit_timestamps[flit_id]
            
    def record_overflow_attempt(self):
        """记录溢出尝试"""
        self.overflow_attempts += 1
        
    def record_underflow_attempt(self):
        """记录下溢尝试"""
        self.underflow_attempts += 1
        
    def get_statistics(self) -> dict:
        """获取统计数据字典"""
        avg_depth = self.depth_sum / max(1, self.sample_count)
        utilization = (avg_depth / max(1, self.current_depth)) if self.current_depth > 0 else 0
        
        write_efficiency = self.total_writes_successful / max(1, self.total_writes_attempted)
        read_efficiency = self.total_reads_successful / max(1, self.total_reads_attempted)
        
        avg_residence = sum(self.residence_times) / max(1, len(self.residence_times)) if self.residence_times else 0
        min_residence = min(self.residence_times) if self.residence_times else 0
        max_residence = max(self.residence_times) if self.residence_times else 0
        
        active_percentage = self.active_cycles / max(1, self.total_simulation_cycles)
        
        return {
            "当前深度": self.current_depth,
            "峰值深度": self.peak_depth,
            "平均深度": round(avg_depth, 2),
            "利用率百分比": round(utilization * 100, 2),
            "空队列周期数": self.empty_cycles,
            "满队列周期数": self.full_cycles,
            "总写入尝试": self.total_writes_attempted,
            "成功写入次数": self.total_writes_successful,
            "总读取尝试": self.total_reads_attempted,
            "成功读取次数": self.total_reads_successful,
            "写入效率": round(write_efficiency * 100, 2),
            "读取效率": round(read_efficiency * 100, 2),
            "写入阻塞次数": self.write_stalls,
            "读取阻塞次数": self.read_stalls,
            "溢出尝试次数": self.overflow_attempts,
            "下溢尝试次数": self.underflow_attempts,
            "平均停留时间": round(avg_residence, 2),
            "最小停留时间": min_residence,
            "最大停留时间": max_residence,
            "高优先级写入": self.priority_writes,
            "总仿真周期": self.total_simulation_cycles,
            "活跃周期百分比": round(active_percentage * 100, 2)
        }


class PipelinedFIFO:
    """
    流水线FIFO，实现输出寄存器模型以确保正确的时序行为。

    每个FIFO都有输出寄存器，数据只能在时钟边沿更新，避免单周期多跳问题。
    """

    def __init__(self, name: str, depth: int):
        self.name = name
        self.max_depth = depth
        self.internal_queue = deque(maxlen=depth)
        self.output_register = None
        self.output_valid = False  # 输出有效信号
        self.read_this_cycle = False  # 本周期是否被读取
        self.next_output_valid = False  # 下周期输出是否有效
        
        # 统计信息收集器
        self.stats = FIFOStatistics()
        self.current_cycle = 0

    def step_compute_phase(self, cycle: int = None):
        """组合逻辑：计算流控信号"""
        if cycle is not None:
            self.current_cycle = cycle
            
        # 更新深度统计
        current_depth = len(self.internal_queue) + (1 if self.output_valid else 0)
        self.stats.update_depth_stats(current_depth, self.max_depth, self.current_cycle)
        
        # 计算下周期的输出有效性
        if self.internal_queue and not self.output_valid:
            self.next_output_valid = True
        elif self.read_this_cycle and self.internal_queue:
            self.next_output_valid = True  # 本周期被读，下周期继续输出
        else:
            self.next_output_valid = False

    def step_update_phase(self):
        """时序逻辑：更新寄存器 - 优化版本，减少1拍延迟"""
        # 首先检查是否需要首次输出（优先处理新写入的数据）
        if not self.output_valid and self.internal_queue:
            # 首次输出：立即从队列取出数据到输出寄存器
            self.output_register = self.internal_queue.popleft()
            self.output_valid = True
        # 然后处理读取后的更新
        elif self.read_this_cycle:
            if self.internal_queue:
                # 读取后继续输出：从队列补充输出寄存器
                self.output_register = self.internal_queue.popleft()
                self.output_valid = True
            else:
                # 队列已空：清空输出寄存器
                self.output_register = None
                self.output_valid = False

        # 重置读取标志
        self.read_this_cycle = False

    def peek_output(self):
        """查看输出（组合逻辑）"""
        return self.output_register if self.output_valid else None

    def read_output(self):
        """读取输出（只能每周期调用一次）"""
        if self.output_valid and not self.read_this_cycle:
            self.read_this_cycle = True
            self.stats.record_read_attempt(successful=True)
            
            # 记录flit退出时间
            if hasattr(self.output_register, 'packet_id'):
                self.stats.record_flit_exit(str(self.output_register.packet_id), self.current_cycle)
            elif hasattr(self.output_register, '__hash__'):
                self.stats.record_flit_exit(str(hash(self.output_register)), self.current_cycle)
                
            return self.output_register
        else:
            # 尝试读取但失败
            if not self.output_valid:
                self.stats.record_read_attempt(successful=False)
                self.stats.record_underflow_attempt()
        return None

    def can_accept_input(self):
        """检查是否能接受输入（组合逻辑）"""
        return len(self.internal_queue) < self.internal_queue.maxlen

    def write_input(self, data) -> bool:
        """写入新数据"""
        if self.can_accept_input():
            self.internal_queue.append(data)
            self.stats.record_write_attempt(successful=True)
            
            # 记录flit进入时间
            if hasattr(data, 'packet_id'):
                self.stats.record_flit_enter(str(data.packet_id), self.current_cycle)
            elif hasattr(data, '__hash__'):
                self.stats.record_flit_enter(str(hash(data)), self.current_cycle)
                
            return True
        else:
            self.stats.record_write_attempt(successful=False)
            self.stats.record_overflow_attempt()
            return False

    def ready_signal(self):
        """Ready信号：能否接受新数据"""
        return self.can_accept_input()

    def valid_signal(self):
        """Valid信号：输出是否有效"""
        return self.output_valid

    def __len__(self):
        """兼容性：返回内部队列长度"""
        return len(self.internal_queue) + (1 if self.output_valid else 0)

    def append(self, item):
        """兼容性接口：向队列添加元素"""
        return self.write_input(item)

    def popleft(self):
        """兼容性接口：从队列弹出元素"""
        return self.read_output()

    def appendleft(self, item):
        """兼容性接口：向队列头部添加元素（高优先级）"""
        return self.priority_write(item)

    def priority_write(self, data) -> bool:
        """高优先级写入（用于重试等场景）"""
        if self.can_accept_input():
            # 将数据插入到队列头部
            self.internal_queue.appendleft(data)
            self.stats.record_write_attempt(successful=True, is_priority=True)
            
            # 记录flit进入时间
            if hasattr(data, 'packet_id'):
                self.stats.record_flit_enter(str(data.packet_id), self.current_cycle)
            elif hasattr(data, '__hash__'):
                self.stats.record_flit_enter(str(hash(data)), self.current_cycle)
                
            return True
        else:
            self.stats.record_write_attempt(successful=False, is_priority=True)
            self.stats.record_overflow_attempt()
            return False
            
    def get_statistics(self) -> dict:
        """获取FIFO统计信息"""
        stats_dict = self.stats.get_statistics()
        stats_dict["FIFO名称"] = self.name
        stats_dict["最大容量"] = self.max_depth
        return stats_dict


class FlowControlledTransfer:
    """实现标准的Valid/Ready握手协议"""

    @staticmethod
    def can_transfer(source_fifo: PipelinedFIFO, dest_fifo: PipelinedFIFO, additional_check=None) -> bool:
        """检查是否可以传输，遵循Valid/Ready协议"""
        # 组合逻辑：检查传输条件
        source_valid = source_fifo.valid_signal()
        dest_ready = dest_fifo.ready_signal()

        # 额外的传输条件检查（如资源、时序等）
        extra_ok = additional_check() if additional_check else True

        # 只有在valid && ready && extra_ok时才能传输
        return source_valid and dest_ready and extra_ok

    @staticmethod
    def try_transfer(source_fifo: PipelinedFIFO, dest_fifo: PipelinedFIFO, additional_check=None) -> bool:
        """尝试传输，遵循Valid/Ready协议"""
        if FlowControlledTransfer.can_transfer(source_fifo, dest_fifo, additional_check):
            data = source_fifo.read_output()
            if data is not None:
                return dest_fifo.write_input(data)
        return False


class BaseResourceManager:
    """基础资源管理器"""

    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity
        self.available = capacity
        self.allocated = {}  # {resource_id: allocation_info}
        self.waiting_queue = deque()
        self.stats = {
            "total_allocations": 0,
            "total_releases": 0,
            "peak_usage": 0,
            "wait_time_total": 0,
        }

    def can_allocate(self, amount: int = 1) -> bool:
        """检查是否可以分配资源"""
        return self.available >= amount

    def allocate(self, resource_id: str, amount: int = 1, **kwargs) -> bool:
        """分配资源"""
        if not self.can_allocate(amount):
            return False

        self.available -= amount
        self.allocated[resource_id] = {"amount": amount, "allocated_time": kwargs.get("time", 0), "info": kwargs}

        self.stats["total_allocations"] += 1
        usage = self.capacity - self.available
        if usage > self.stats["peak_usage"]:
            self.stats["peak_usage"] = usage

        return True

    def release(self, resource_id: str) -> bool:
        """释放资源"""
        if resource_id not in self.allocated:
            return False

        amount = self.allocated[resource_id]["amount"]
        self.available += amount
        del self.allocated[resource_id]

        self.stats["total_releases"] += 1
        return True

    def get_usage(self) -> Dict[str, Any]:
        """获取使用情况"""
        return {
            "capacity": self.capacity,
            "available": self.available,
            "allocated": self.capacity - self.available,
            "utilization": (self.capacity - self.available) / self.capacity,
            "active_allocations": len(self.allocated),
            "waiting_requests": len(self.waiting_queue),
            "stats": self.stats.copy(),
        }


class BaseIPInterface(ABC):
    """
    NoC基础IP接口类。

    提供所有NoC拓扑共用的IP接口功能：
    1. 时钟域转换
    2. 基础资源管理
    3. STI三通道协议处理
    4. 统计收集
    """

    def __init__(self, ip_type: str, node_id: NodeId, config: Any, model: Any, flit_class: Type[BaseFlit]):
        """
        初始化基础IP接口

        Args:
            ip_type: IP类型
            node_id: 节点ID
            config: 配置对象
            model: 主模型实例
            flit_class: 使用的Flit类型
        """
        self.ip_type = ip_type
        self.node_id = node_id
        self.config = config
        self.model = model
        self.flit_class = flit_class
        self.current_cycle = 0

        # ========== 时钟域转换 ==========
        if hasattr(config, "basic_config"):
            self.clock_ratio = getattr(config.basic_config, "NETWORK_FREQUENCY", 2)
        else:
            self.clock_ratio = getattr(config, "NETWORK_FREQUENCY", 2)
        self._setup_clock_domain_fifos()

        # ========== STI三通道FIFO ==========
        self.inject_fifos = {"req": PipelinedFIFO("inject_req", depth=16), "rsp": PipelinedFIFO("inject_rsp", depth=16), "data": PipelinedFIFO("inject_data", depth=16)}
        
        # ========== IP内部处理FIFO ==========
        # IP内部处理FIFO，位于H2L FIFO和最终IP处理之间
        ip_proc_depth = getattr(config, "IP_PROCESSING_FIFO_DEPTH", 4) if hasattr(config, "IP_PROCESSING_FIFO_DEPTH") else 4
        self.ip_processing_fifos = {
            "req": PipelinedFIFO("ip_proc_req", depth=ip_proc_depth), 
            "rsp": PipelinedFIFO("ip_proc_rsp", depth=ip_proc_depth), 
            "data": PipelinedFIFO("ip_proc_data", depth=ip_proc_depth)
        }

        # ========== 统计信息 ==========
        self.stats = {
            "requests_sent": {"read": 0, "write": 0},
            "responses_received": {"ack": 0, "nack": 0},
            "data_transferred": {"sent": 0, "received": 0},
            "retries": {"read": 0, "write": 0},
            "latencies": {"injection": [], "network": [], "total": []},
            "throughput": {"requests_per_cycle": 0.0, "data_per_cycle": 0.0},
        }

        # ========== 等待队列和跟踪 ==========
        self.active_requests = {}  # {packet_id: request_info}
        self.completed_requests = {}  # {packet_id: completion_info}
        self.pending_requests = deque()  # 无限大的待注入队列

        # 日志
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{ip_type}_{node_id}")

        # 注册到模型
        if hasattr(model, "register_ip_interface"):
            model.register_ip_interface(self)

    def _setup_clock_domain_fifos(self) -> None:
        """设置时钟域转换FIFO"""
        # 支持CrossRing的组合配置
        if hasattr(self.config, "basic_config"):
            l2h_depth = getattr(self.config.basic_config, "IP_L2H_FIFO_DEPTH", 4)
            h2l_depth = getattr(self.config.basic_config, "IP_H2L_FIFO_DEPTH", 4)
        else:
            l2h_depth = getattr(self.config, "IP_L2H_FIFO_DEPTH", 4)
            h2l_depth = getattr(self.config, "IP_H2L_FIFO_DEPTH", 4)

        self.l2h_fifos = {"req": PipelinedFIFO("l2h_req", depth=l2h_depth), "rsp": PipelinedFIFO("l2h_rsp", depth=l2h_depth), "data": PipelinedFIFO("l2h_data", depth=l2h_depth)}

        self.h2l_fifos = {"req": PipelinedFIFO("h2l_req", depth=h2l_depth), "rsp": PipelinedFIFO("h2l_rsp", depth=h2l_depth), "data": PipelinedFIFO("h2l_data", depth=h2l_depth)}

        # 传输状态跟踪（替代pre_buffer）
        self._transfer_states = {
            "inject_to_l2h": {"req": False, "rsp": False, "data": False},
            "l2h_to_network": {"req": False, "rsp": False, "data": False},
            "network_to_h2l": {"req": False, "rsp": False, "data": False},
            "h2l_to_ip_processing": {"req": False, "rsp": False, "data": False},
            "ip_processing_to_completion": {"req": False, "rsp": False, "data": False},
        }


    def step(self, cycle: int) -> None:
        """
        执行一个周期，使用三阶段执行模型确保正确的时序行为

        数据流路径：
        1. pending_requests → L2H FIFO (时钟域转换)
        2. L2H FIFO → 拓扑网络

        Args:
            cycle: 当前仿真周期
        """
        self.current_cycle = cycle

        # 阶段1：组合逻辑（计算所有流控信号和传输可能性）
        self._compute_phase(cycle)

        # 阶段2：时钟域同步和资源处理
        self._sync_phase(cycle)

        # 阶段3：时序逻辑（更新所有寄存器和状态）
        self._update_phase(cycle)

        # 更新统计
        self._update_statistics()

    def _compute_phase(self, cycle: int) -> None:
        """阶段1：组合逻辑，计算所有流控信号和传输可能性"""
        # 所有FIFO计算流控信号
        for fifo_dict in [self.inject_fifos, self.l2h_fifos, self.h2l_fifos, self.ip_processing_fifos]:
            for fifo in fifo_dict.values():
                fifo.step_compute_phase(cycle)

        # 计算传输可能性
        self._compute_transfer_possibilities(cycle)

    def _sync_phase(self, cycle: int) -> None:
        """阶段2：时钟域同步和资源处理"""
        # 处理延迟释放的资源
        self._process_delayed_resource_release()

        # 时钟域特定的处理
        if cycle % self.clock_ratio == 0:
            self._compute_1ghz_transfers()
        self._compute_2ghz_transfers()

    def _update_phase(self, cycle: int) -> None:
        """阶段3：时序逻辑，更新所有寄存器和执行传输"""
        # 执行传输
        if cycle % self.clock_ratio == 0:
            self._execute_1ghz_transfers()
        self._execute_2ghz_transfers()

        # 更新所有FIFO的寄存器
        for fifo_dict in [self.inject_fifos, self.l2h_fifos, self.h2l_fifos, self.ip_processing_fifos]:
            for fifo in fifo_dict.values():
                fifo.step_update_phase()

    def _compute_transfer_possibilities(self, cycle: int) -> None:
        """计算所有可能的传输"""
        for channel in ["req", "rsp", "data"]:
            # inject → l2h 传输可能性
            self._transfer_states["inject_to_l2h"][channel] = FlowControlledTransfer.can_transfer(
                source_fifo=self.inject_fifos[channel], dest_fifo=self.l2h_fifos[channel], additional_check=lambda ch=channel: self._can_inject_to_l2h(ch)
            )

            # l2h → network 传输可能性
            self._transfer_states["l2h_to_network"][channel] = self.l2h_fifos[channel].valid_signal() and self._network_can_accept(channel)

            # network → h2l 传输可能性
            self._transfer_states["network_to_h2l"][channel] = self._network_has_data(channel) and self.h2l_fifos[channel].ready_signal()

            # h2l → ip_processing 传输可能性
            self._transfer_states["h2l_to_ip_processing"][channel] = FlowControlledTransfer.can_transfer(
                source_fifo=self.h2l_fifos[channel], dest_fifo=self.ip_processing_fifos[channel]
            )
            
            # ip_processing → completion 传输可能性
            self._transfer_states["ip_processing_to_completion"][channel] = self.ip_processing_fifos[channel].valid_signal()

    def _compute_1ghz_transfers(self) -> None:
        """计算1GHz域的传输"""
        # 在1GHz边沿执行的传输逻辑
        pass

    def _compute_2ghz_transfers(self) -> None:
        """计算2GHz域的传输"""
        # 每个2GHz周期执行的传输逻辑
        pass

    def _execute_1ghz_transfers(self) -> None:
        """执行1GHz域的传输"""
        for channel in ["req", "rsp", "data"]:
            # inject → l2h 传输
            if self._transfer_states["inject_to_l2h"][channel]:
                FlowControlledTransfer.try_transfer(
                    source_fifo=self.inject_fifos[channel], dest_fifo=self.l2h_fifos[channel], additional_check=lambda ch=channel: self._check_and_reserve_resources_for_channel(ch)
                )

            # h2l → ip_processing 传输
            if self._transfer_states["h2l_to_ip_processing"][channel]:
                FlowControlledTransfer.try_transfer(
                    source_fifo=self.h2l_fifos[channel], dest_fifo=self.ip_processing_fifos[channel]
                )
                
            # ip_processing → completion 传输
            if self._transfer_states["ip_processing_to_completion"][channel]:
                self._execute_ip_processing_to_completion(channel)

    def _execute_2ghz_transfers(self) -> None:
        """执行2GHz域的传输"""
        for channel in ["req", "rsp", "data"]:
            # l2h → network 传输
            if self._transfer_states["l2h_to_network"][channel]:
                self._execute_l2h_to_network(channel)

            # network → h2l 传输
            if self._transfer_states["network_to_h2l"][channel]:
                self._execute_network_to_h2l(channel)

    def _can_inject_to_l2h(self, channel: str) -> bool:
        """检查是否可以从inject FIFO传输到l2h FIFO"""
        if channel == "req":
            # 对于请求，需要检查资源
            flit = self.inject_fifos[channel].peek_output()
            if flit and hasattr(flit, "req_attr") and flit.req_attr == "new":
                return self._check_and_reserve_resources(flit)
            return True
        elif channel == "data":
            # 对于数据，检查发送时间
            flit = self.inject_fifos[channel].peek_output()
            if flit and hasattr(flit, "departure_cycle"):
                return flit.departure_cycle <= self.current_cycle
            return True
        else:
            # 响应直接允许
            return True

    def _check_and_reserve_resources_for_channel(self, channel: str) -> bool:
        """为特定通道检查并预占资源"""
        if channel == "req":
            flit = self.inject_fifos[channel].peek_output()
            return flit and self._check_and_reserve_resources(flit)
        return True

    @abstractmethod
    def _check_and_reserve_resources(self, flit: BaseFlit) -> bool:
        """检查并预占资源（拓扑特定）"""
        pass

    def _execute_l2h_to_network(self, channel: str) -> None:
        """执行l2h FIFO到网络的传输"""
        flit = self.l2h_fifos[channel].read_output()
        if flit:
            flit.flit_position = "network"
            # 调用拓扑特定的网络注入方法
            self._inject_to_topology_network(flit, channel)

    def _execute_network_to_h2l(self, channel: str) -> None:
        """执行网络到h2l FIFO的传输"""
        flit = self._eject_from_topology_network(channel)
        if flit:
            flit.flit_position = "h2l_fifo"
            self.h2l_fifos[channel].write_input(flit)

    def _execute_ip_processing_to_completion(self, channel: str) -> None:
        """执行IP内部处理FIFO到IP完成的传输"""
        flit = self.ip_processing_fifos[channel].read_output()
        if flit:
            # 根据通道类型处理
            if channel == "req":
                self._handle_received_request(flit)
            elif channel == "rsp":
                self._handle_received_response(flit)
            elif channel == "data":
                self._handle_received_data(flit)

    def _network_can_accept(self, channel: str) -> bool:
        """检查网络是否可以接受数据"""
        # 默认实现，子类可以重写
        return True

    def _network_has_data(self, channel: str) -> bool:
        """检查网络是否有数据要弹出"""
        # 默认实现，子类可以重写
        return self._eject_from_topology_network(channel) is not None

    @abstractmethod
    def _inject_to_topology_network(self, flit: BaseFlit, channel: str) -> None:
        """注入到拓扑网络（拓扑特定）"""
        pass

    @abstractmethod
    def _eject_from_topology_network(self, channel: str) -> Optional[BaseFlit]:
        """从拓扑网络弹出（拓扑特定）"""
        pass

    def _handle_received_request(self, req: BaseFlit) -> None:
        """处理收到的请求（通用STI协议处理）"""
        # 根据请求类型和属性处理
        if req.req_type == "read":
            if req.req_attr == "new":
                if self._can_handle_new_read_request(req):
                    self._process_read_request(req)
                else:
                    self._send_negative_response(req, "resource_unavailable")
            else:  # retry
                self._process_read_request(req)

        elif req.req_type == "write":
            if req.req_attr == "new":
                if self._can_handle_new_write_request(req):
                    self._process_write_request(req)
                else:
                    self._send_negative_response(req, "resource_unavailable")
            else:  # retry
                self._process_write_request(req)

    def _handle_received_response(self, rsp: BaseFlit) -> None:
        """处理收到的响应（通用STI协议处理）"""
        # 查找对应的请求
        req_info = self.active_requests.get(rsp.packet_id)
        if not req_info:
            self.logger.warning(f"收到未知响应: {rsp.packet_id}")
            return

        req = req_info["flit"]
        req.sync_latency_record(rsp)

        # 更新统计
        self.stats["responses_received"][rsp.rsp_type] += 1

        # 根据响应类型处理
        if rsp.rsp_type == "nack":
            self._handle_negative_response(rsp, req)
        elif rsp.rsp_type == "ack":
            self._handle_positive_response(rsp, req)
        elif rsp.rsp_type == "data_ready":
            self._handle_data_ready_response(rsp, req)

    def _handle_received_data(self, data: BaseFlit) -> None:
        """处理收到的数据（通用STI协议处理）"""
        # 查找对应的请求
        req_info = self.active_requests.get(data.packet_id)
        if not req_info:
            self.logger.warning(f"收到未知数据: {data.packet_id}")
            return

        req = req_info["flit"]
        req.sync_latency_record(data)

        # 收集数据
        if data.packet_id not in self.completed_requests:
            self.completed_requests[data.packet_id] = {"req": req, "data_flits": [], "completion_time": None}

        self.completed_requests[data.packet_id]["data_flits"].append(data)

        # 检查是否收集完整
        expected_flits = req.burst_length
        received_flits = len(self.completed_requests[data.packet_id]["data_flits"])

        if received_flits >= expected_flits:
            self._complete_data_transfer(data.packet_id)

    # ========== 抽象方法（拓扑特定实现） ==========

    @abstractmethod
    def _can_handle_new_read_request(self, req: BaseFlit) -> bool:
        """检查是否可以处理新读请求"""
        pass

    @abstractmethod
    def _can_handle_new_write_request(self, req: BaseFlit) -> bool:
        """检查是否可以处理新写请求"""
        pass

    @abstractmethod
    def _process_read_request(self, req: BaseFlit) -> None:
        """处理读请求"""
        pass

    @abstractmethod
    def _process_write_request(self, req: BaseFlit) -> None:
        """处理写请求"""
        pass

    def _send_negative_response(self, req: BaseFlit, reason: str) -> None:
        """发送负响应"""
        rsp = req.create_response("nack", retry_reason=reason)
        self.inject_fifos["rsp"].append(rsp)

    def _handle_negative_response(self, rsp: BaseFlit, req: BaseFlit) -> None:
        """处理负响应（准备重试）"""
        if req.can_retry():
            req.prepare_for_retry(rsp.retry_reason)
            self.inject_fifos["req"].appendleft(req)  # 重新入队
            self.stats["retries"][req.req_type] += 1
        else:
            # 达到最大重试次数，标记失败
            self._mark_request_failed(req, "max_retries_exceeded")

    def _handle_positive_response(self, rsp: BaseFlit, req: BaseFlit) -> None:
        """处理正响应"""
        req.mark_retry_success()
        # 具体处理由子类实现

    def _handle_data_ready_response(self, rsp: BaseFlit, req: BaseFlit) -> None:
        """处理数据就绪响应"""
        # 通常用于写请求，表示可以发送数据
        if req.req_type == "write":
            self._send_write_data(req)

    def _send_write_data(self, req: BaseFlit) -> None:
        """发送写数据"""
        for i in range(req.burst_length):
            data_flit = req.create_data_flit(flit_id=i)
            data_flit.departure_cycle = self.current_cycle + i
            self.inject_fifos["data"].append(data_flit)

    def _complete_data_transfer(self, packet_id: str) -> None:
        """完成数据传输"""
        completion_info = self.completed_requests[packet_id]
        completion_info["completion_time"] = self.current_cycle

        # 更新统计
        req = completion_info["req"]
        data_flits = completion_info["data_flits"]

        if data_flits:
            total_latency = self.current_cycle - req.creation_time
            network_latency = self.current_cycle - req.injection_time

            self.stats["latencies"]["total"].append(total_latency)
            self.stats["latencies"]["network"].append(network_latency)
            self.stats["data_transferred"]["received"] += len(data_flits)

        # 从活跃请求中移除
        if packet_id in self.active_requests:
            del self.active_requests[packet_id]

        self.logger.debug(f"完成数据传输: {packet_id}")

    def _mark_request_failed(self, req: BaseFlit, reason: str) -> None:
        """标记请求失败"""
        self.logger.warning(f"请求失败: {req.packet_id}, 原因: {reason}")

        # 从活跃请求中移除
        if req.packet_id in self.active_requests:
            del self.active_requests[req.packet_id]

    def _process_delayed_resource_release(self) -> None:
        """处理延迟释放的资源（子类可重写）"""
        pass

    def _update_statistics(self) -> None:
        """更新统计信息"""
        if self.current_cycle > 0:
            total_requests = sum(self.stats["requests_sent"].values())
            total_data = self.stats["data_transferred"]["sent"] + self.stats["data_transferred"]["received"]

            self.stats["throughput"]["requests_per_cycle"] = total_requests / self.current_cycle
            self.stats["throughput"]["data_per_cycle"] = total_data / self.current_cycle

    def get_status(self) -> Dict[str, Any]:
        """获取IP接口状态"""
        return {
            "ip_type": self.ip_type,
            "node_id": self.node_id,
            "current_cycle": self.current_cycle,
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.completed_requests),
            "stats": self.stats.copy(),
            "fifo_status": {
                channel: {
                    "inject": len(self.inject_fifos[channel]),
                    "inject_valid": self.inject_fifos[channel].valid_signal(),
                    "inject_ready": self.inject_fifos[channel].ready_signal(),
                    "l2h": len(self.l2h_fifos[channel]),
                    "l2h_valid": self.l2h_fifos[channel].valid_signal(),
                    "l2h_ready": self.l2h_fifos[channel].ready_signal(),
                    "h2l": len(self.h2l_fifos[channel]),
                    "h2l_valid": self.h2l_fifos[channel].valid_signal(),
                    "h2l_ready": self.h2l_fifos[channel].ready_signal(),
                    "ip_processing": len(self.ip_processing_fifos[channel]),
                    "ip_processing_valid": self.ip_processing_fifos[channel].valid_signal(),
                    "ip_processing_ready": self.ip_processing_fifos[channel].ready_signal(),
                }
                for channel in ["req", "rsp", "data"]
            },
        }
