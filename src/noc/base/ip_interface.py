"""
通用NoC IP接口基类。

提供所有NoC拓扑共用的IP接口基础功能，包括：
- PipelinedFIFO实现
- 两阶段执行框架
- 基础统计收集
各拓扑可以继承并扩展特有功能。
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




class BaseIPInterface(ABC):
    """
    NoC基础IP接口类。

    提供所有NoC拓扑共用的IP接口基础功能：
    1. 两阶段执行框架
    2. 时钟域转换FIFO
    3. 基础统计收集
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

        # ========== 统计信息 ==========
        self.stats = {
            "requests_sent": {"read": 0, "write": 0},
            "responses_received": {"ack": 0, "nack": 0},
            "data_transferred": {"sent": 0, "received": 0},
            "retries": {"read": 0, "write": 0},
            "latencies": {"injection": [], "network": [], "total": []},
            "throughput": {"requests_per_cycle": 0.0, "data_per_cycle": 0.0},
        }

        # ========== 请求跟踪 ==========
        self.active_requests = {}  # {packet_id: request_info}
        self.completed_requests = {}  # {packet_id: completion_info}

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


    def step(self, cycle: int) -> None:
        """
        执行一个周期，使用两阶段执行模型

        Args:
            cycle: 当前仿真周期
        """
        self.current_cycle = cycle

        # 阶段1：计算阶段（组合逻辑）
        self.step_compute_phase(cycle)

        # 阶段2：更新阶段（时序逻辑）
        self.step_update_phase(cycle)

        # 更新统计
        self._update_statistics()

    def step_compute_phase(self, cycle: int) -> None:
        """
        计算阶段：计算传输决策但不执行
        子类应该重写此方法实现自己的计算逻辑
        """
        # 更新所有FIFO的计算阶段
        for channel in ["req", "rsp", "data"]:
            if hasattr(self, 'l2h_fifos'):
                self.l2h_fifos[channel].step_compute_phase(cycle)
            if hasattr(self, 'h2l_fifos'):
                self.h2l_fifos[channel].step_compute_phase(cycle)

    def step_update_phase(self, cycle: int) -> None:
        """
        更新阶段：执行传输决策
        子类应该重写此方法实现自己的更新逻辑
        """
        # 更新所有FIFO的时序状态
        for channel in ["req", "rsp", "data"]:
            if hasattr(self, 'l2h_fifos'):
                self.l2h_fifos[channel].step_update_phase()
            if hasattr(self, 'h2l_fifos'):
                self.h2l_fifos[channel].step_update_phase()

    # ========== 抽象方法（拓扑特定实现） ==========

    @abstractmethod
    def _inject_to_topology_network(self, flit: BaseFlit, channel: str) -> None:
        """注入到拓扑网络（拓扑特定）"""
        pass

    @abstractmethod
    def _eject_from_topology_network(self, channel: str) -> Optional[BaseFlit]:
        """从拓扑网络弹出（拓扑特定）"""
        pass

    @abstractmethod
    def _check_and_reserve_resources(self, flit: BaseFlit) -> bool:
        """检查并预占资源（拓扑特定）"""
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
                    "l2h": len(self.l2h_fifos[channel]),
                    "l2h_valid": self.l2h_fifos[channel].valid_signal(),
                    "l2h_ready": self.l2h_fifos[channel].ready_signal(),
                    "h2l": len(self.h2l_fifos[channel]),
                    "h2l_valid": self.h2l_fifos[channel].valid_signal(),
                    "h2l_ready": self.h2l_fifos[channel].ready_signal(),
                }
                for channel in ["req", "rsp", "data"]
            },
        }
