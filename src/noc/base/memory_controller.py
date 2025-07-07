"""
NoC内存控制器实现。

本模块实现了NoC网络中的内存控制器节点，提供：
- 内存访问调度和管理
- 内存层次结构支持
- 带宽和延迟控制
- 内存一致性维护
- 错误检测和纠正
"""

from typing import Dict, List, Optional, Any, Tuple, Deque
from collections import deque, defaultdict
from enum import Enum
import logging
import random

from .node import BaseNoCNode, NodeState, BufferStatus
from .flit import BaseFlit
from ..crossring.flit import CrossRingFlit
from src.noc.utils.types import NodeId, Position, Priority


class MemoryType(Enum):
    """内存类型"""

    DDR4 = "ddr4"
    DDR5 = "ddr5"
    HBM = "hbm"
    MRAM = "mram"
    SRAM = "sram"


class MemoryOperation(Enum):
    """内存操作类型"""

    READ = "read"
    WRITE = "write"
    READX = "readx"  # 独占读
    WRITEBACK = "writeback"
    EVICT = "evict"


class RequestState(Enum):
    """请求状态"""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class SchedulingPolicy(Enum):
    """调度策略"""

    FIFO = "fifo"
    FCFS = "fcfs"
    FRFCFS = "frfcfs"  # First Ready - First Come First Serve
    OPEN_PAGE = "open_page"
    CLOSE_PAGE = "close_page"
    PRIORITY_BASED = "priority_based"


class MemoryController(BaseNoCNode):
    """
    内存控制器节点实现。

    MC负责管理对内存系统的访问，提供：
    1. 内存请求调度
    2. 地址映射和译码
    3. 内存时序控制
    4. 带宽管理
    5. 错误处理
    """

    def __init__(self, node_id: NodeId, position: Position, memory_type: MemoryType = MemoryType.DDR4, **kwargs):
        """
        初始化内存控制器。

        Args:
            node_id: 节点ID
            position: 节点位置
            memory_type: 内存类型
            **kwargs: 其他配置参数
        """
        from src.noc.utils.types import NodeType

        super().__init__(node_id, position, NodeType.MEMORY_CONTROLLER)
        self.current_cycle = 0

        self.memory_type = memory_type

        # 内存配置
        self.memory_capacity = kwargs.get("memory_capacity", 16 * 1024 * 1024 * 1024)  # 16GB
        self.memory_channels = kwargs.get("channels", 4)
        self.channel_width = kwargs.get("channel_width", 64)  # bits
        self.memory_frequency = kwargs.get("memory_freq", 3200)  # MHz

        # 内存时序参数
        self.timing_params = self._init_timing_params(memory_type, kwargs)

        # 调度策略
        self.scheduling_policy = SchedulingPolicy(kwargs.get("scheduling", "frfcfs"))

        # 地址映射
        self.address_mapping = kwargs.get("address_mapping", "row_bank_col_channel")
        self.page_size = kwargs.get("page_size", 4096)  # bytes
        self.row_buffer_size = kwargs.get("row_buffer_size", 8192)  # bytes

        # 请求队列
        self.request_queue = deque(maxlen=64)
        self.read_queue = deque(maxlen=32)
        self.write_queue = deque(maxlen=32)

        # 内存银行状态
        self.num_banks = kwargs.get("num_banks", 16)
        self.bank_states = {}
        self._init_bank_states()

        # 行缓冲区管理
        self.row_buffers = {}  # {bank_id: {row_id: data, open_time}}
        self.open_pages = {}  # {bank_id: row_id}

        # 调度器状态
        self.scheduler_state = {"current_bank": 0, "last_command_time": {}, "command_queue": deque(maxlen=16), "ready_commands": deque(maxlen=8)}

        # 性能统计
        self.mc_stats = {
            "requests_received": {"read": 0, "write": 0},
            "requests_completed": {"read": 0, "write": 0},
            "average_latency": {"read": 0.0, "write": 0.0},
            "bandwidth_utilization": 0.0,
            "row_buffer_hits": 0,
            "row_buffer_misses": 0,
            "page_hit_rate": 0.0,
            "bank_conflicts": 0,
            "command_counts": defaultdict(int),
            "cycles_busy": 0,
            "cycles_idle": 0,
        }

        # 延迟跟踪
        self.latency_tracker = {"read_latencies": [], "write_latencies": [], "queue_delays": []}

        # 内存一致性
        self.coherence_enabled = kwargs.get("coherence", False)
        if self.coherence_enabled:
            self.coherence_directory = {}

        # 错误检测和纠正
        self.ecc_enabled = kwargs.get("ecc", True)
        self.error_stats = {"correctable_errors": 0, "uncorrectable_errors": 0, "error_rate": 0.0}

        # 预取器
        self.prefetcher_enabled = kwargs.get("prefetcher", False)
        if self.prefetcher_enabled:
            self.prefetcher_state = {"stride_table": {}, "prefetch_queue": deque(maxlen=8), "prefetch_hits": 0, "prefetch_misses": 0}

        # 初始化组件
        self._initialize_mc_components()

        # 设置日志
        self.logger = logging.getLogger(f"MemoryController_{node_id}")

    def _init_timing_params(self, memory_type: MemoryType, kwargs: Dict[str, Any]) -> Dict[str, int]:
        """初始化内存时序参数"""
        if memory_type == MemoryType.DDR4:
            return {
                "tRCD": kwargs.get("tRCD", 14),  # RAS to CAS delay
                "tRP": kwargs.get("tRP", 14),  # Row precharge time
                "tRAS": kwargs.get("tRAS", 35),  # Row active time
                "tCAS": kwargs.get("tCAS", 14),  # CAS latency
                "tWR": kwargs.get("tWR", 15),  # Write recovery time
                "tRFC": kwargs.get("tRFC", 350),  # Refresh cycle time
                "tREFI": kwargs.get("tREFI", 7800),  # Refresh interval
                "tBurst": kwargs.get("tBurst", 4),  # Burst length
            }
        elif memory_type == MemoryType.DDR5:
            return {
                "tRCD": kwargs.get("tRCD", 16),
                "tRP": kwargs.get("tRP", 16),
                "tRAS": kwargs.get("tRAS", 39),
                "tCAS": kwargs.get("tCAS", 16),
                "tWR": kwargs.get("tWR", 18),
                "tRFC": kwargs.get("tRFC", 400),
                "tREFI": kwargs.get("tREFI", 3900),
                "tBurst": kwargs.get("tBurst", 8),
            }
        elif memory_type == MemoryType.HBM:
            return {
                "tRCD": kwargs.get("tRCD", 12),
                "tRP": kwargs.get("tRP", 12),
                "tRAS": kwargs.get("tRAS", 28),
                "tCAS": kwargs.get("tCAS", 12),
                "tWR": kwargs.get("tWR", 10),
                "tRFC": kwargs.get("tRFC", 180),
                "tREFI": kwargs.get("tREFI", 3900),
                "tBurst": kwargs.get("tBurst", 4),
            }
        else:
            # 默认参数
            return {"tRCD": 14, "tRP": 14, "tRAS": 35, "tCAS": 14, "tWR": 15, "tRFC": 350, "tREFI": 7800, "tBurst": 4}

    def _init_bank_states(self) -> None:
        """初始化银行状态"""
        for bank_id in range(self.num_banks):
            self.bank_states[bank_id] = {"state": "idle", "open_row": None, "last_access_time": 0, "precharge_time": 0, "activation_time": 0, "pending_commands": deque()}  # idle, active, precharging

    def _initialize_mc_components(self) -> None:
        """初始化内存控制器组件"""
        # 初始化调度器
        for bank_id in range(self.num_banks):
            self.scheduler_state["last_command_time"][bank_id] = 0

        # 初始化行缓冲区
        for bank_id in range(self.num_banks):
            self.row_buffers[bank_id] = {}
            self.open_pages[bank_id] = None

    def process_flit(self, flit: BaseFlit, input_port: str) -> bool:
        """
        处理接收到的内存请求flit。

        Args:
            flit: 要处理的flit对象
            input_port: 输入端口名称

        Returns:
            是否成功处理
        """
        try:
            # 检查是否为内存请求
            if not hasattr(flit, "request_type"):
                self.logger.warning(f"接收到非内存请求flit: {flit}")
                return False

            # 解析内存请求
            request = self._parse_memory_request(flit)
            if not request:
                return False

            # 添加到请求队列
            if len(self.request_queue) >= self.request_queue.maxlen:
                self.logger.warning("内存请求队列已满")
                return False

            self.request_queue.append(request)

            # 更新统计
            req_type = request["operation"].value
            self.mc_stats["requests_received"][req_type] += 1

            return True

        except Exception as e:
            self.logger.error(f"处理内存请求时发生错误: {e}")
            return False

    def _parse_memory_request(self, flit: BaseFlit) -> Optional[Dict[str, Any]]:
        """解析内存请求"""
        try:
            operation = MemoryOperation(flit.request_type)
            address = getattr(flit, "memory_address", 0)

            # 地址解码
            bank_id, row_id, col_id = self._decode_address(address)

            request = {
                "flit": flit,
                "operation": operation,
                "address": address,
                "bank_id": bank_id,
                "row_id": row_id,
                "col_id": col_id,
                "size": getattr(flit, "data_size", 64),
                "priority": getattr(flit, "priority", Priority.MEDIUM),
                "arrival_time": self.current_cycle,
                "state": RequestState.PENDING,
                "scheduled_time": None,
                "completion_time": None,
            }

            return request

        except Exception as e:
            self.logger.error(f"解析内存请求失败: {e}")
            return None

    def _decode_address(self, address: int) -> Tuple[int, int, int]:
        """地址解码"""
        # 简化的地址映射: row_bank_col_channel
        col_bits = 6  # 64 bytes per cache line
        bank_bits = 4  # 16 banks

        col_id = (address >> col_bits) & ((1 << col_bits) - 1)
        bank_id = (address >> (col_bits + bank_bits)) & ((1 << bank_bits) - 1)
        row_id = address >> (col_bits + bank_bits)

        return bank_id % self.num_banks, row_id, col_id

    def route_flit(self, flit: BaseFlit) -> Optional[str]:
        """
        为flit进行路由决策。

        Args:
            flit: 要路由的flit对象

        Returns:
            输出端口名称
        """
        # 内存控制器通常发送响应回源节点
        if hasattr(flit, "source"):
            return "network"
        return None

    def can_accept_flit(self, input_port: str, priority: Priority = Priority.MEDIUM) -> bool:
        """
        检查是否可以接收新的flit。

        Args:
            input_port: 输入端口名称
            priority: flit优先级

        Returns:
            是否可以接收
        """
        # 检查请求队列是否有空间
        if len(self.request_queue) >= self.request_queue.maxlen:
            return False

        # 高优先级请求总是接受
        if priority in [Priority.HIGH, Priority.CRITICAL]:
            return True

        # 检查银行冲突
        return True

    def step_mc(self, cycle: int) -> None:
        """
        执行内存控制器的一个周期操作。

        Args:
            cycle: 当前周期
        """
        self.current_cycle = cycle

        # 请求调度
        self._schedule_requests()

        # 执行内存命令
        self._execute_memory_commands()

        # 处理行缓冲区
        self._manage_row_buffers()

        # 处理预取
        if self.prefetcher_enabled:
            self._process_prefetcher()

        # 刷新处理
        self._handle_refresh()

        # 更新统计
        self._update_mc_statistics()

    def _schedule_requests(self) -> None:
        """调度内存请求"""
        if not self.request_queue:
            return

        # 根据调度策略选择请求
        if self.scheduling_policy == SchedulingPolicy.FIFO:
            self._schedule_fifo()
        elif self.scheduling_policy == SchedulingPolicy.FRFCFS:
            self._schedule_frfcfs()
        elif self.scheduling_policy == SchedulingPolicy.PRIORITY_BASED:
            self._schedule_priority_based()
        else:
            self._schedule_fifo()  # 默认FIFO

    def _schedule_fifo(self) -> None:
        """FIFO调度"""
        if self.request_queue:
            request = self.request_queue.popleft()
            self._process_memory_request(request)

    def _schedule_frfcfs(self) -> None:
        """First Ready - First Come First Serve调度"""
        # 查找可以立即执行的请求
        ready_requests = []

        for i, request in enumerate(self.request_queue):
            if self._is_request_ready(request):
                ready_requests.append((i, request))

        if ready_requests:
            # 选择最早到达的ready请求
            _, request = min(ready_requests, key=lambda x: x[1]["arrival_time"])
            self.request_queue.remove(request)
            self._process_memory_request(request)
        elif self.request_queue:
            # 如果没有ready请求，处理最老的请求
            request = self.request_queue.popleft()
            self._process_memory_request(request)

    def _schedule_priority_based(self) -> None:
        """基于优先级的调度"""
        if not self.request_queue:
            return

        # 按优先级排序
        priority_order = [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]

        for priority in priority_order:
            for request in list(self.request_queue):
                if request["priority"] == priority:
                    self.request_queue.remove(request)
                    self._process_memory_request(request)
                    return

    def _is_request_ready(self, request: Dict[str, Any]) -> bool:
        """检查请求是否ready（可以立即执行）"""
        bank_id = request["bank_id"]
        row_id = request["row_id"]
        bank_state = self.bank_states[bank_id]

        # 检查银行状态
        if bank_state["state"] != "idle":
            return False

        # 检查行缓冲区命中
        if bank_state["open_row"] == row_id:
            return True

        # 检查时序约束
        current_time = self.current_cycle
        last_command_time = self.scheduler_state["last_command_time"][bank_id]

        # 简化的时序检查
        if current_time - last_command_time >= self.timing_params["tRCD"]:
            return True

        return False

    def _process_memory_request(self, request: Dict[str, Any]) -> None:
        """处理内存请求"""
        bank_id = request["bank_id"]
        row_id = request["row_id"]
        operation = request["operation"]

        # 更新请求状态
        request["state"] = RequestState.SCHEDULED
        request["scheduled_time"] = self.current_cycle

        # 检查行缓冲区状态
        bank_state = self.bank_states[bank_id]

        if bank_state["open_row"] == row_id:
            # 行缓冲区命中
            self._execute_row_hit(request)
            self.mc_stats["row_buffer_hits"] += 1
        else:
            # 行缓冲区未命中
            self._execute_row_miss(request)
            self.mc_stats["row_buffer_misses"] += 1

    def _execute_row_hit(self, request: Dict[str, Any]) -> None:
        """执行行缓冲区命中的请求"""
        # 可以直接访问数据
        latency = self.timing_params["tCAS"] + self.timing_params["tBurst"]
        self._complete_request(request, latency)

    def _execute_row_miss(self, request: Dict[str, Any]) -> None:
        """执行行缓冲区未命中的请求"""
        bank_id = request["bank_id"]
        row_id = request["row_id"]
        bank_state = self.bank_states[bank_id]

        # 需要先关闭当前行，再打开新行
        total_latency = 0

        if bank_state["open_row"] is not None:
            # 预充电当前行
            total_latency += self.timing_params["tRP"]

        # 激活新行
        total_latency += self.timing_params["tRCD"]

        # 执行命令
        total_latency += self.timing_params["tCAS"] + self.timing_params["tBurst"]

        # 更新银行状态
        bank_state["open_row"] = row_id
        bank_state["activation_time"] = self.current_cycle

        self._complete_request(request, total_latency)

    def _complete_request(self, request: Dict[str, Any], latency: int) -> None:
        """完成内存请求"""
        completion_time = self.current_cycle + latency
        request["completion_time"] = completion_time
        request["state"] = RequestState.COMPLETED

        # 创建响应flit
        response_flit = self._create_response_flit(request)

        # 发送响应
        if len(self.output_buffers["local"]) < self.output_buffer_size:
            self.output_buffers["local"].append(response_flit)

        # 更新统计
        operation = request["operation"].value
        self.mc_stats["requests_completed"][operation] += 1

        # 记录延迟
        total_latency = completion_time - request["arrival_time"]
        self.latency_tracker[f"{operation}_latencies"].append(total_latency)

        # 更新平均延迟
        latencies = self.latency_tracker[f"{operation}_latencies"]
        self.mc_stats["average_latency"][operation] = sum(latencies) / len(latencies)

    def _create_response_flit(self, request: Dict[str, Any]) -> BaseFlit:
        """创建响应flit"""
        original_flit = request["flit"]

        response_flit = CrossRingFlit(source=self.node_id, destination=original_flit.source, packet_id=original_flit.packet_id + "_response", creation_time=self.current_cycle)

        response_flit.response_type = "memory"
        response_flit.original_request = request["operation"].value
        response_flit.memory_address = request["address"]
        response_flit.latency = request["completion_time"] - request["arrival_time"]

        # 模拟数据
        if request["operation"] == MemoryOperation.READ:
            response_flit.data_size = request["size"]
            response_flit.data = f"data_from_addr_{request['address']}"

        # ECC检查
        if self.ecc_enabled:
            self._apply_ecc_check(response_flit)

        return response_flit

    def _apply_ecc_check(self, flit: BaseFlit) -> None:
        """应用ECC检查"""
        # 模拟ECC错误
        error_probability = 1e-6  # 很低的错误率

        if random.random() < error_probability:
            if random.random() < 0.9:  # 90%是可纠正错误
                self.error_stats["correctable_errors"] += 1
                flit.ecc_status = "corrected"
            else:  # 10%是不可纠正错误
                self.error_stats["uncorrectable_errors"] += 1
                flit.ecc_status = "uncorrectable_error"
        else:
            flit.ecc_status = "no_error"

    def _execute_memory_commands(self) -> None:
        """执行内存命令"""
        # 处理ready命令队列
        if self.scheduler_state["ready_commands"]:
            command = self.scheduler_state["ready_commands"].popleft()
            self._execute_command(command)

    def _execute_command(self, command: Dict[str, Any]) -> None:
        """执行单个内存命令"""
        cmd_type = command["type"]
        bank_id = command["bank_id"]

        # 更新银行状态
        bank_state = self.bank_states[bank_id]
        bank_state["last_access_time"] = self.current_cycle

        # 更新调度器状态
        self.scheduler_state["last_command_time"][bank_id] = self.current_cycle

        # 更新统计
        self.mc_stats["command_counts"][cmd_type] += 1

    def _manage_row_buffers(self) -> None:
        """管理行缓冲区"""
        current_time = self.current_cycle

        for bank_id, bank_state in self.bank_states.items():
            if bank_state["open_row"] is not None:
                activation_time = bank_state["activation_time"]

                # 检查是否需要预充电（基于策略）
                if self._should_precharge_row(bank_id, current_time - activation_time):
                    self._precharge_row(bank_id)

    def _should_precharge_row(self, bank_id: int, open_time: int) -> bool:
        """判断是否应该预充电行"""
        # 简单策略：如果行开放时间超过阈值且没有pending请求
        threshold = 100  # cycles

        if open_time > threshold:
            # 检查是否有pending请求访问此行
            for request in self.request_queue:
                if request["bank_id"] == bank_id:
                    return False
            return True

        return False

    def _precharge_row(self, bank_id: int) -> None:
        """预充电行"""
        bank_state = self.bank_states[bank_id]
        bank_state["open_row"] = None
        bank_state["state"] = "precharging"
        bank_state["precharge_time"] = self.current_cycle

    def _process_prefetcher(self) -> None:
        """处理预取器"""
        # 简单的顺序预取器
        if self.prefetcher_state["prefetch_queue"]:
            prefetch_addr = self.prefetcher_state["prefetch_queue"].popleft()

            # 创建预取请求
            prefetch_request = {"operation": MemoryOperation.READ, "address": prefetch_addr, "priority": Priority.LOW, "is_prefetch": True}

            # 如果队列有空间，添加预取请求
            if len(self.request_queue) < self.request_queue.maxlen - 2:
                # 保留一些空间给正常请求
                self.request_queue.append(prefetch_request)

    def _handle_refresh(self) -> None:
        """处理内存刷新"""
        # 简化的刷新处理
        refresh_interval = self.timing_params["tREFI"]

        if self.current_cycle % refresh_interval == 0:
            # 执行刷新操作
            for bank_id in range(self.num_banks):
                bank_state = self.bank_states[bank_id]
                if bank_state["state"] == "idle":
                    # 刷新银行
                    bank_state["state"] = "refreshing"
                    # 刷新需要 tRFC 时间

    def _update_mc_statistics(self) -> None:
        """更新内存控制器统计"""
        # 计算带宽利用率
        total_requests = sum(self.mc_stats["requests_completed"].values())
        if self.current_cycle > 0:
            # 理论峰值带宽
            peak_bandwidth = self.memory_channels * self.channel_width / 8  # bytes per cycle
            actual_bandwidth = total_requests * 64 / self.current_cycle  # 假设64字节请求
            self.mc_stats["bandwidth_utilization"] = actual_bandwidth / peak_bandwidth

        # 计算页命中率
        total_hits = self.mc_stats["row_buffer_hits"]
        total_misses = self.mc_stats["row_buffer_misses"]
        if total_hits + total_misses > 0:
            self.mc_stats["page_hit_rate"] = total_hits / (total_hits + total_misses)

        # 更新错误率
        total_errors = self.error_stats["correctable_errors"] + self.error_stats["uncorrectable_errors"]
        if total_requests > 0:
            self.error_stats["error_rate"] = total_errors / total_requests

    def get_mc_status(self) -> Dict[str, Any]:
        """
        获取内存控制器状态。

        Returns:
            内存控制器状态字典
        """
        status = self.get_performance_stats()
        status.update(
            {
                "memory_type": self.memory_type.value,
                "memory_capacity": self.memory_capacity,
                "memory_channels": self.memory_channels,
                "scheduling_policy": self.scheduling_policy.value,
                "mc_stats": self.mc_stats.copy(),
                "bank_states": {str(k): v for k, v in self.bank_states.items()},
                "request_queue_length": len(self.request_queue),
                "open_pages": {str(k): v for k, v in self.open_pages.items()},
                "timing_params": self.timing_params.copy(),
                "error_stats": self.error_stats.copy(),
            }
        )

        if self.prefetcher_enabled:
            status["prefetcher_stats"] = {
                "prefetch_hits": self.prefetcher_state.get("prefetch_hits", 0),
                "prefetch_misses": self.prefetcher_state.get("prefetch_misses", 0),
                "prefetch_queue_length": len(self.prefetcher_state["prefetch_queue"]),
            }

        return status

    def set_scheduling_policy(self, policy: SchedulingPolicy) -> None:
        """
        设置调度策略。

        Args:
            policy: 新的调度策略
        """
        self.scheduling_policy = policy
        self.logger.info(f"调度策略已更改为: {policy.value}")

    def configure_timing(self, timing_params: Dict[str, int]) -> None:
        """
        配置内存时序参数。

        Args:
            timing_params: 时序参数字典
        """
        self.timing_params.update(timing_params)

    def add_memory_mapping(self, address_range: Tuple[int, int], bank_mapping: Dict[str, Any]) -> None:
        """
        添加内存地址映射。

        Args:
            address_range: 地址范围
            bank_mapping: 银行映射配置
        """
        # 实现内存映射配置
        pass

    def get_bank_utilization(self) -> Dict[int, float]:
        """
        获取银行利用率。

        Returns:
            银行利用率字典
        """
        utilization = {}

        for bank_id, bank_state in self.bank_states.items():
            if bank_state["last_access_time"] > 0:
                active_time = self.current_cycle - bank_state["last_access_time"]
                utilization[bank_id] = min(active_time / self.current_cycle, 1.0)
            else:
                utilization[bank_id] = 0.0

        return utilization
