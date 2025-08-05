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

from .flit import BaseFlit
from src.noc.utils.types import NodeId
from src.noc.utils.token_bucket import TokenBucket


class FIFOStatistics:
    """简化的FIFO统计信息收集器 - 基于累加数据"""
    
    def __init__(self):
        # 核心累加字段
        self.depth_sum = 0          # 深度累加和（用于计算平均深度）
        self.peak_depth = 0         # 峰值深度
        self.sample_count = 0       # 采样次数
        self.write_stalls = 0       # 写入阻塞累加次数
        
    def update_depth_stats(self, current_depth: int, max_capacity: int):
        """更新深度相关统计 - 仅累加数据"""
        self.peak_depth = max(self.peak_depth, current_depth)
        self.depth_sum += current_depth
        self.sample_count += 1
            
    def record_write_stall(self):
        """记录写入阻塞 - 仅累加计数"""
        self.write_stalls += 1
        
    def get_raw_statistics(self) -> dict:
        """获取原始累加数据，供最终统计时计算"""
        return {
            "depth_sum": self.depth_sum,
            "peak_depth": self.peak_depth,
            "sample_count": self.sample_count,
            "write_stalls": self.write_stalls
        }
        
    def calculate_final_statistics(self, fifo_capacity: int) -> dict:
        """计算最终统计指标"""
        if self.sample_count == 0:
            return {
                "峰值深度": 0,
                "平均深度": 0.0,
                "利用率百分比": 0.0,
                "写入阻塞次数": self.write_stalls,
                "采样次数": 0
            }
            
        avg_depth = self.depth_sum / self.sample_count
        utilization = avg_depth / max(1, fifo_capacity)
        
        return {
            "峰值深度": self.peak_depth,
            "平均深度": round(avg_depth, 2),
            "利用率百分比": round(utilization * 100, 2),
            "写入阻塞次数": self.write_stalls,
            "采样次数": self.sample_count
        }


class PipelinedFIFO:
    """
    流水线FIFO，实现输出寄存器模型以确保正确的时序行为。

    每个FIFO都有输出寄存器，数据只能在时钟边沿更新，避免单周期多跳问题。
    """

    def __init__(self, name: str, depth: int):
        self.name = name
        self.max_depth = depth
        # 确保总容量等于配置深度：internal_queue + output_register = depth
        # 至少保证internal_queue深度为1，以避免退化情况
        internal_depth = max(1, depth - 1)
        self.internal_queue = deque(maxlen=internal_depth)
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
            
        # 更新深度统计 - 每周期直接统计（累加统计无需采样）
        current_depth = len(self.internal_queue) + (1 if self.output_valid else 0)
        self.stats.update_depth_stats(current_depth, self.max_depth)
        
        # 优化：简化下周期输出有效性计算
        self.next_output_valid = (self.internal_queue and 
                                  (not self.output_valid or self.read_this_cycle))

    def step_update_phase(self):
        """时序逻辑：更新寄存器 - 修复同周期读写竞争"""
        # 处理读取后的更新（优先处理）
        if self.read_this_cycle:
            if self.internal_queue:
                # 读取后继续输出：从队列补充输出寄存器
                self.output_register = self.internal_queue.popleft()
                self.output_valid = True
            else:
                # 队列已空：清空输出寄存器
                self.output_register = None
                self.output_valid = False
        # 然后检查是否需要首次输出（处理新写入的数据）
        elif not self.output_valid and self.internal_queue:
            # 首次输出：立即从队列取出数据到输出寄存器
            self.output_register = self.internal_queue.popleft()
            self.output_valid = True

        # 重置读取标志
        self.read_this_cycle = False

    def peek_output(self):
        """查看输出（组合逻辑）"""
        return self.output_register if self.output_valid else None

    def read_output(self):
        """读取输出（只能每周期调用一次）"""
        if self.output_valid and not self.read_this_cycle:
            self.read_this_cycle = True
            return self.output_register
        else:
            # 读取失败，但不记录统计（简化后移除读取统计）
            return None

    def can_accept_input(self):
        """检查是否能接受输入（组合逻辑）"""
        return len(self.internal_queue) < self.internal_queue.maxlen

    def write_input(self, data) -> bool:
        """写入新数据"""
        if self.can_accept_input():
            self.internal_queue.append(data)
            return True
        else:
            # 写入失败，记录阻塞统计
            self.stats.record_write_stall()
            return False

    def ready_signal(self):
        """优化Ready信号：直接计算，避免函数调用开销"""
        return len(self.internal_queue) < self.internal_queue.maxlen

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
            return True
        else:
            # 写入失败，记录阻塞统计
            self.stats.record_write_stall()
            return False
    
    def get_all_flits(self) -> List[Any]:
        """
        获取FIFO中所有flit的列表（用于I-Tag检查等场景）
        
        注意：此方法仅返回数据的副本，不会修改FIFO状态
        
        Returns:
            包含所有flit的列表，按照FIFO顺序排列
        """
        all_flits = []
        
        # 首先添加internal_queue中的所有元素
        all_flits.extend(list(self.internal_queue))
        
        # 然后添加output_register中的元素（如果有效）
        if self.output_valid and self.output_register:
            all_flits.append(self.output_register)
            
        return all_flits
            
    def get_statistics(self) -> dict:
        """获取FIFO统计信息"""
        stats_dict = self.stats.calculate_final_statistics(self.max_depth)
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
        
        # ========== 带宽限制 ==========
        self.token_bucket = None  # 将由子类根据IP类型初始化

        # 日志

        # 注册到模型
        if hasattr(model, "register_ip_interface"):
            model.register_ip_interface(self)

    def _setup_clock_domain_fifos(self) -> None:
        """设置时钟域转换FIFO - 双级H2L结构"""
        l2h_depth = self.config.ip_config.IP_L2H_FIFO_DEPTH
        h2l_h_depth = self.config.ip_config.IP_H2L_H_FIFO_DEPTH
        h2l_l_depth = self.config.ip_config.IP_H2L_L_FIFO_DEPTH
        
        # 获取统计采样间隔配置
        sample_interval = getattr(self.config.basic_config, 'FIFO_STATS_SAMPLE_INTERVAL', 100)

        # L2H FIFO (保持不变)
        self.l2h_fifos = {
            "req": PipelinedFIFO("l2h_req", depth=l2h_depth),
            "rsp": PipelinedFIFO("l2h_rsp", depth=l2h_depth),
            "data": PipelinedFIFO("l2h_data", depth=l2h_depth)
        }

        # H2L 双级FIFO
        self.h2l_h_fifos = {  # 网络域高级FIFO
            "req": PipelinedFIFO("h2l_h_req", depth=h2l_h_depth),
            "rsp": PipelinedFIFO("h2l_h_rsp", depth=h2l_h_depth),
            "data": PipelinedFIFO("h2l_h_data", depth=h2l_h_depth)
        }
        
        self.h2l_l_fifos = {  # IP域低级FIFO
            "req": PipelinedFIFO("h2l_l_req", depth=h2l_l_depth),
            "rsp": PipelinedFIFO("h2l_l_rsp", depth=h2l_l_depth),
            "data": PipelinedFIFO("h2l_l_data", depth=h2l_l_depth)
        }
        
        # 为所有FIFO设置统计采样间隔
        for fifo_dict in [self.l2h_fifos, self.h2l_h_fifos, self.h2l_l_fifos]:
            for fifo in fifo_dict.values():
                fifo._stats_sample_interval = sample_interval
        
    def _setup_token_bucket(self, rate: float, bucket_size: float) -> None:
        """
        设置令牌桶用于带宽限制。
        
        Args:
            rate: 每周期生成的令牌数（通常为 bandwidth_limit / flit_size）
            bucket_size: 桶的最大容量
        """
        self.token_bucket = TokenBucket(rate=rate, bucket_size=bucket_size)


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
