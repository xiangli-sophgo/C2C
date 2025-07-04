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
        self.clock_ratio = getattr(config, "network_frequency", 2)
        self._setup_clock_domain_fifos()

        # ========== STI三通道FIFO ==========
        self.inject_fifos = {"req": deque(), "rsp": deque(), "data": deque()}

        # ========== 资源管理器 ==========
        self._setup_resource_managers()

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

        # 日志
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{ip_type}_{node_id}")

        # 注册到模型
        if hasattr(model, "register_ip_interface"):
            model.register_ip_interface(self)

    def _setup_clock_domain_fifos(self) -> None:
        """设置时钟域转换FIFO"""
        l2h_depth = getattr(self.config, "IP_L2H_FIFO_DEPTH", 4)
        h2l_depth = getattr(self.config, "IP_H2L_FIFO_DEPTH", 4)

        self.l2h_fifos = {"req": deque(maxlen=l2h_depth), "rsp": deque(maxlen=h2l_depth), "data": deque(maxlen=l2h_depth)}

        self.h2l_fifos = {"req": deque(maxlen=h2l_depth), "rsp": deque(maxlen=h2l_depth), "data": deque(maxlen=h2l_depth)}

        # 预缓冲区
        self.l2h_pre_buffers = {"req": None, "rsp": None, "data": None}
        self.h2l_pre_buffers = {"req": None, "rsp": None, "data": None}

    @abstractmethod
    def _setup_resource_managers(self) -> None:
        """设置资源管理器（拓扑特定）"""
        pass

    @abstractmethod
    def _get_destination_ip_type(self, destination: NodeId) -> str:
        """获取目标节点的IP类型（拓扑特定）"""
        pass

    def step(self, cycle: int) -> None:
        """
        执行一个周期

        Args:
            cycle: 当前仿真周期
        """
        self.current_cycle = cycle

        # 处理延迟释放的资源
        self._process_delayed_resource_release()

        # 时钟域处理
        if cycle % self.clock_ratio == 0:
            self._step_low_frequency()  # 1GHz域

        self._step_high_frequency()  # 2GHz域

        # 预缓冲区移动
        self._move_pre_to_fifo()

        # 更新统计
        self._update_statistics()

    def _step_low_frequency(self) -> None:
        """低频域处理（如1GHz）"""
        # IP生成新请求（由外部调用enqueue_request）
        # H2L FIFO → IP完成处理
        self._h2l_to_ip_completion()

    def _step_high_frequency(self) -> None:
        """高频域处理（如2GHz）"""
        # inject FIFO → L2H FIFO
        for channel in ["req", "rsp", "data"]:
            self._inject_to_l2h_pre(channel)

        # L2H FIFO → 网络
        for channel in ["req", "rsp", "data"]:
            self._l2h_to_network(channel)

        # 网络 → H2L FIFO
        for channel in ["req", "rsp", "data"]:
            self._network_to_h2l(channel)

    def _move_pre_to_fifo(self) -> None:
        """预缓冲区到正式FIFO的移动"""
        for channel in ["req", "rsp", "data"]:
            # L2H预缓冲区 → L2H FIFO
            if self.l2h_pre_buffers[channel] is not None and len(self.l2h_fifos[channel]) < self.l2h_fifos[channel].maxlen:
                self.l2h_fifos[channel].append(self.l2h_pre_buffers[channel])
                self.l2h_pre_buffers[channel] = None

            # H2L预缓冲区 → H2L FIFO
            if self.h2l_pre_buffers[channel] is not None and len(self.h2l_fifos[channel]) < self.h2l_fifos[channel].maxlen:
                self.h2l_fifos[channel].append(self.h2l_pre_buffers[channel])
                self.h2l_pre_buffers[channel] = None

    def enqueue_request(self, source: NodeId, destination: NodeId, req_type: str, burst_length: int = 4, packet_id: str = None, **kwargs) -> bool:
        """
        将新请求加入inject FIFO

        Args:
            source: 源节点
            destination: 目标节点
            req_type: 请求类型
            burst_length: 突发长度
            packet_id: 包ID
            **kwargs: 其他参数

        Returns:
            是否成功入队
        """
        if packet_id is None:
            packet_id = f"{self.ip_type}_{self.node_id}_{self.current_cycle}_{len(self.inject_fifos['req'])}"

        # 创建请求flit
        flit = self.flit_class(
            source=source, destination=destination, req_type=req_type, burst_length=burst_length, packet_id=packet_id, channel="req", flit_type="req", creation_time=self.current_cycle, **kwargs
        )

        # 设置IP类型信息
        flit.source_ip_type = self.ip_type
        flit.dest_ip_type = self._get_destination_ip_type(destination)

        # 加入inject FIFO
        self.inject_fifos["req"].append(flit)

        # 跟踪活跃请求
        self.active_requests[packet_id] = {"flit": flit, "created_time": self.current_cycle, "status": "created"}

        self.stats["requests_sent"][req_type] += 1
        return True

    def _inject_to_l2h_pre(self, channel: str) -> None:
        """inject FIFO → l2h_pre_buffer"""
        if not self.inject_fifos[channel] or len(self.l2h_fifos[channel]) >= self.l2h_fifos[channel].maxlen or self.l2h_pre_buffers[channel] is not None:
            return

        flit = self.inject_fifos[channel][0]

        # 根据通道类型进行不同处理
        if channel == "req":
            if not self._check_and_reserve_resources(flit):
                return  # 资源不足
        elif channel == "data":
            # 检查发送时间
            if hasattr(flit, "departure_cycle") and flit.departure_cycle > self.current_cycle:
                return

        # 移动到预缓冲区
        flit.flit_position = "l2h_fifo"
        self.l2h_pre_buffers[channel] = self.inject_fifos[channel].popleft()

    @abstractmethod
    def _check_and_reserve_resources(self, flit: BaseFlit) -> bool:
        """检查并预占资源（拓扑特定）"""
        pass

    def _l2h_to_network(self, channel: str) -> None:
        """l2h_fifo → 网络注入"""
        if not self.l2h_fifos[channel]:
            return

        flit = self.l2h_fifos[channel].popleft()
        flit.set_injection_time(self.current_cycle)
        flit.flit_position = "network"

        # 调用拓扑特定的网络注入方法
        self._inject_to_topology_network(flit, channel)

    @abstractmethod
    def _inject_to_topology_network(self, flit: BaseFlit, channel: str) -> None:
        """注入到拓扑网络（拓扑特定）"""
        pass

    def _network_to_h2l(self, channel: str) -> None:
        """网络 → h2l_pre_buffer"""
        if self.h2l_pre_buffers[channel] is not None:
            return

        # 调用拓扑特定的网络弹出方法
        flit = self._eject_from_topology_network(channel)
        if flit:
            flit.flit_position = "h2l_fifo"
            self.h2l_pre_buffers[channel] = flit

    @abstractmethod
    def _eject_from_topology_network(self, channel: str) -> Optional[BaseFlit]:
        """从拓扑网络弹出（拓扑特定）"""
        pass

    def _h2l_to_ip_completion(self) -> None:
        """h2l_fifo → IP处理完成"""
        for channel in ["req", "rsp", "data"]:
            if not self.h2l_fifos[channel]:
                continue

            flit = self.h2l_fifos[channel].popleft()
            flit.set_ejection_time(self.current_cycle)

            # 根据通道类型处理
            if channel == "req":
                self._handle_received_request(flit)
            elif channel == "rsp":
                self._handle_received_response(flit)
            elif channel == "data":
                self._handle_received_data(flit)

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
                    "l2h": len(self.l2h_fifos[channel]),
                    "h2l": len(self.h2l_fifos[channel]),
                    "l2h_pre": self.l2h_pre_buffers[channel] is not None,
                    "h2l_pre": self.h2l_pre_buffers[channel] is not None,
                }
                for channel in ["req", "rsp", "data"]
            },
        }
