"""
NoC处理元素实现。

本模块实现了NoC网络中的处理元素(PE)，提供：
- 计算任务执行和调度
- 内存访问请求生成
- 工作负载建模
- 性能监控
- 多线程支持
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import deque, defaultdict
from enum import Enum
import logging
import random

from .node import BaseNoCNode, NodeState, BufferStatus
from .flit import BaseFlit
from src.noc.utils.types import NodeId, Position, Priority, TrafficPattern
from src.noc.crossring.flit import CrossRingFlit


class TaskType(Enum):
    """任务类型"""

    COMPUTE = "compute"  # 计算任务
    MEMORY_READ = "memory_read"  # 内存读
    MEMORY_WRITE = "memory_write"  # 内存写
    SYNCHRONIZATION = "sync"  # 同步任务
    COMMUNICATION = "comm"  # 通信任务


class TaskState(Enum):
    """任务状态"""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_MEMORY = "waiting_memory"
    WAITING_SYNC = "waiting_sync"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkloadType(Enum):
    """工作负载类型"""

    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    MIXED = "mixed"
    SYNTHETIC = "synthetic"
    TRACE_BASED = "trace_based"


class ProcessingElement(BaseNoCNode):
    """
    处理元素节点实现。

    PE是NoC网络中的计算节点，提供：
    1. 任务执行和调度
    2. 内存访问管理
    3. 工作负载生成
    4. 性能监控
    5. 并发处理支持
    """

    def __init__(self, node_id: NodeId, position: Position, pe_type: str = "generic", **kwargs):
        """
        初始化处理元素。

        Args:
            node_id: 节点ID
            position: 节点位置
            pe_type: PE类型
            **kwargs: 其他配置参数
        """
        from src.noc.utils.types import NodeType

        super().__init__(node_id, position, NodeType.PROCESSING_ELEMENT)
        self.current_cycle = 0

        self.pe_type = pe_type

        # 计算资源配置
        self.num_cores = kwargs.get("num_cores", 1)
        self.clock_frequency = kwargs.get("clock_frequency", 1.0)  # GHz
        self.instruction_per_cycle = kwargs.get("ipc", 1.0)

        # 内存层次配置
        self.l1_cache_size = kwargs.get("l1_cache_size", 32)  # KB
        self.l2_cache_size = kwargs.get("l2_cache_size", 256)  # KB
        self.cache_line_size = kwargs.get("cache_line_size", 64)  # bytes

        # 任务队列和调度
        self.task_queue = deque(maxlen=64)
        self.running_tasks: Dict[int, Dict[str, Any]] = {}  # {core_id: task_info}
        self.completed_tasks: List[Dict[str, Any]] = []

        # 工作负载配置
        self.workload_type = WorkloadType(kwargs.get("workload_type", "synthetic"))
        self.workload_params = kwargs.get("workload_params", {})

        # 内存访问模式
        self.memory_access_pattern = kwargs.get("memory_pattern", "random")
        self.locality_factor = kwargs.get("locality_factor", 0.8)

        # 流量生成配置
        self.traffic_pattern = TrafficPattern(kwargs.get("traffic_pattern", "uniform_random"))
        self.injection_rate = kwargs.get("injection_rate", 0.1)  # packets per cycle
        self.packet_size_dist = kwargs.get("packet_size_dist", [64, 128, 256, 512])

        # 同步和通信
        self.sync_points: Dict[str, Dict[str, Any]] = {}
        self.communication_partners: List[NodeId] = kwargs.get("partners", [])

        # 性能监控
        self.pe_stats = {
            "instructions_executed": 0,
            "cycles_busy": 0,
            "cycles_idle": 0,
            "cache_hits": {"l1": 0, "l2": 0},
            "cache_misses": {"l1": 0, "l2": 0},
            "memory_requests": {"read": 0, "write": 0},
            "tasks_completed": 0,
            "tasks_failed": 0,
            "ipc_actual": 0.0,
            "cache_hit_rate": {"l1": 0.0, "l2": 0.0},
            "memory_bandwidth_used": 0.0,
        }

        # 缓存模拟
        self.cache_simulation = kwargs.get("cache_simulation", False)
        if self.cache_simulation:
            self._init_cache_simulation()

        # 初始化PE组件
        self._initialize_pe_components()

        # 设置日志
        self.logger = logging.getLogger(f"ProcessingElement_{node_id}")

    def _initialize_pe_components(self) -> None:
        """初始化处理元素组件"""
        # 初始化核心状态
        self.core_states = {}
        for core_id in range(self.num_cores):
            self.core_states[core_id] = {"state": "idle", "current_task": None, "remaining_cycles": 0, "last_instruction_time": 0}

        # 初始化工作负载生成器
        self._init_workload_generator()

        # 初始化流量生成器
        self._init_traffic_generator()

    def _init_cache_simulation(self) -> None:
        """初始化缓存模拟"""
        # 简单的缓存模拟器
        self.cache_simulator = {
            "l1": {"size": self.l1_cache_size * 1024, "associativity": 4, "cache_lines": {}, "access_count": 0, "hit_count": 0},  # 转换为bytes
            "l2": {"size": self.l2_cache_size * 1024, "associativity": 8, "cache_lines": {}, "access_count": 0, "hit_count": 0},
        }

    def _init_workload_generator(self) -> None:
        """初始化工作负载生成器"""
        self.workload_generator = {
            "next_task_time": 0,
            "task_inter_arrival_time": 100,  # cycles
            "task_size_dist": [10, 50, 100, 500, 1000],  # cycles
            "memory_intensity": 0.3,  # fraction of instructions that access memory
            "compute_intensity": 0.7,
        }

        # 根据工作负载类型调整参数
        if self.workload_type == WorkloadType.CPU_INTENSIVE:
            self.workload_generator["memory_intensity"] = 0.1
            self.workload_generator["compute_intensity"] = 0.9
        elif self.workload_type == WorkloadType.MEMORY_INTENSIVE:
            self.workload_generator["memory_intensity"] = 0.7
            self.workload_generator["compute_intensity"] = 0.3

    def _init_traffic_generator(self) -> None:
        """初始化流量生成器"""
        self.traffic_generator = {"next_injection_time": 0, "destination_list": list(range(16)), "packet_id_counter": 0, "burst_mode": False, "burst_remaining": 0}  # 假设16个节点

        # 移除自己的节点ID
        if self.node_id in self.traffic_generator["destination_list"]:
            self.traffic_generator["destination_list"].remove(self.node_id)

    def process_flit(self, flit: BaseFlit, input_port: str) -> bool:
        """
        处理接收到的flit。

        Args:
            flit: 要处理的flit对象
            input_port: 输入端口名称

        Returns:
            是否成功处理
        """
        try:
            # 处理内存响应
            if hasattr(flit, "response_type") and flit.response_type == "memory":
                return self._handle_memory_response(flit)

            # 处理同步消息
            elif hasattr(flit, "message_type") and flit.message_type == "sync":
                return self._handle_sync_message(flit)

            # 处理通信数据
            elif hasattr(flit, "message_type") and flit.message_type == "data":
                return self._handle_communication_data(flit)

            else:
                self.logger.warning(f"未知的flit类型: {flit}")
                return False

        except Exception as e:
            self.logger.error(f"处理flit时发生错误: {e}")
            return False

    def _handle_memory_response(self, flit: BaseFlit) -> bool:
        """处理内存响应"""
        # 查找等待此响应的任务
        for core_id, task_info in self.running_tasks.items():
            if task_info and task_info.get("waiting_for_memory") and task_info.get("memory_request_id") == flit.packet_id:

                # 恢复任务执行
                task_info["waiting_for_memory"] = False
                task_info["state"] = TaskState.RUNNING
                self.core_states[core_id]["state"] = "busy"

                # 更新统计
                if hasattr(flit, "cache_level"):
                    cache_level = flit.cache_level
                    self.pe_stats["cache_hits"][cache_level] += 1
                else:
                    self.pe_stats["cache_misses"]["l2"] += 1

                return True

        return False

    def _handle_sync_message(self, flit: BaseFlit) -> bool:
        """处理同步消息"""
        if hasattr(flit, "sync_id"):
            sync_id = flit.sync_id

            if sync_id in self.sync_points:
                self.sync_points[sync_id]["received_count"] += 1

                # 检查是否所有参与者都已到达
                expected = self.sync_points[sync_id]["expected_count"]
                received = self.sync_points[sync_id]["received_count"]

                if received >= expected:
                    # 释放等待同步的任务
                    self._release_sync_waiting_tasks(sync_id)

        return True

    def _handle_communication_data(self, flit: BaseFlit) -> bool:
        """处理通信数据"""
        # 将数据放入接收缓冲区
        if len(self.input_buffers["local"]) < self.input_buffer_size:
            self.input_buffers["local"].append(flit)
            return True
        return False

    def route_flit(self, flit: BaseFlit) -> Optional[str]:
        """
        为flit进行路由决策。

        Args:
            flit: 要路由的flit对象

        Returns:
            输出端口名称
        """
        # PE通常只连接到一个路由器
        if flit.destination != self.node_id:
            return "network"  # 发送到网络
        else:
            return "local"  # 本地处理

    def can_accept_flit(self, input_port: str, priority: Priority = Priority.MEDIUM) -> bool:
        """
        检查是否可以接收新的flit。

        Args:
            input_port: 输入端口名称
            priority: flit优先级

        Returns:
            是否可以接收
        """
        # 检查输入缓冲区空间
        if input_port == "network":
            return len(self.input_buffers["local"]) < self.input_buffer_size
        return False

    def step_pe(self, cycle: int) -> None:
        """
        执行处理元素的一个周期操作。

        Args:
            cycle: 当前周期
        """
        self.current_cycle = cycle

        # 执行任务调度
        self._schedule_tasks()

        # 执行核心处理
        self._execute_cores()

        # 生成工作负载
        self._generate_workload()

        # 生成流量
        self._generate_traffic()

        # 处理同步
        self._process_synchronization()

        # 更新统计
        self._update_pe_statistics()

    def _schedule_tasks(self) -> None:
        """任务调度"""
        # 简单的FIFO调度策略
        for core_id in range(self.num_cores):
            core_state = self.core_states[core_id]

            # 如果核心空闲且有任务等待
            if core_state["state"] == "idle" and self.task_queue:
                task = self.task_queue.popleft()
                self._assign_task_to_core(task, core_id)

    def _assign_task_to_core(self, task: Dict[str, Any], core_id: int) -> None:
        """将任务分配给核心"""
        self.running_tasks[core_id] = task
        self.core_states[core_id]["state"] = "busy"
        self.core_states[core_id]["current_task"] = task["task_id"]
        self.core_states[core_id]["remaining_cycles"] = task["cycles_needed"]

        task["state"] = TaskState.RUNNING
        task["start_time"] = self.current_cycle
        task["assigned_core"] = core_id

    def _execute_cores(self) -> None:
        """执行核心处理"""
        for core_id in range(self.num_cores):
            core_state = self.core_states[core_id]

            if core_state["state"] == "busy":
                task = self.running_tasks.get(core_id)
                if task:
                    self._execute_task_on_core(task, core_id)

    def _execute_task_on_core(self, task: Dict[str, Any], core_id: int) -> None:
        """在核心上执行任务"""
        # 检查是否需要内存访问
        if self._needs_memory_access(task):
            if self._issue_memory_request(task, core_id):
                # 内存请求已发出，任务等待
                task["state"] = TaskState.WAITING_MEMORY
                task["waiting_for_memory"] = True
                self.core_states[core_id]["state"] = "waiting"
                return

        # 执行计算
        cycles_to_execute = min(int(self.instruction_per_cycle), self.core_states[core_id]["remaining_cycles"])

        self.core_states[core_id]["remaining_cycles"] -= cycles_to_execute
        self.pe_stats["instructions_executed"] += cycles_to_execute
        self.pe_stats["cycles_busy"] += 1

        # 检查任务是否完成
        if self.core_states[core_id]["remaining_cycles"] <= 0:
            self._complete_task(task, core_id)

    def _needs_memory_access(self, task: Dict[str, Any]) -> bool:
        """检查任务是否需要内存访问"""
        if task["type"] in [TaskType.MEMORY_READ, TaskType.MEMORY_WRITE]:
            return True

        # 根据内存强度随机决定
        memory_intensity = self.workload_generator["memory_intensity"]
        return random.random() < memory_intensity

    def _issue_memory_request(self, task: Dict[str, Any], core_id: int) -> bool:
        """发出内存请求"""
        # 生成内存地址
        if task["type"] == TaskType.MEMORY_READ:
            req_type = "read"
        elif task["type"] == TaskType.MEMORY_WRITE:
            req_type = "write"
        else:
            req_type = "read"  # 默认读请求

        memory_address = self._generate_memory_address(task)

        # 检查缓存
        if self.cache_simulation and self._check_cache(memory_address):
            # 缓存命中，无需发送网络请求
            return False

        # 创建内存请求flit
        request_id = f"mem_{self.node_id}_{core_id}_{self.current_cycle}"
        memory_flit = self._create_memory_request_flit(req_type, memory_address, request_id)

        # 发送到网络
        if len(self.output_buffers["local"]) < self.output_buffer_size:
            self.output_buffers["local"].append(memory_flit)
            task["memory_request_id"] = request_id
            self.pe_stats["memory_requests"][req_type] += 1
            return True

        return False

    def _generate_memory_address(self, task: Dict[str, Any]) -> int:
        """生成内存地址"""
        if self.memory_access_pattern == "sequential":
            # 顺序访问
            base_addr = task.get("base_address", 0)
            offset = task.get("current_offset", 0)
            task["current_offset"] = offset + self.cache_line_size
            return base_addr + offset

        elif self.memory_access_pattern == "random":
            # 随机访问
            return random.randint(0, 1024 * 1024 * 1024)  # 1GB地址空间

        elif self.memory_access_pattern == "locality":
            # 局部性访问
            if random.random() < self.locality_factor:
                # 局部访问
                last_addr = task.get("last_address", 0)
                return last_addr + random.randint(-64, 64) * self.cache_line_size
            else:
                # 随机访问
                return random.randint(0, 1024 * 1024 * 1024)

        return 0

    def _check_cache(self, address: int) -> bool:
        """检查缓存"""
        # 简单的缓存模拟
        cache_line = address // self.cache_line_size

        # 检查L1缓存
        l1_cache = self.cache_simulator["l1"]
        l1_cache["access_count"] += 1

        if cache_line in l1_cache["cache_lines"]:
            l1_cache["hit_count"] += 1
            self.pe_stats["cache_hits"]["l1"] += 1
            return True

        # 检查L2缓存
        l2_cache = self.cache_simulator["l2"]
        l2_cache["access_count"] += 1

        if cache_line in l2_cache["cache_lines"]:
            l2_cache["hit_count"] += 1
            self.pe_stats["cache_hits"]["l2"] += 1

            # 将数据提升到L1
            l1_cache["cache_lines"][cache_line] = self.current_cycle
            return True

        # 缓存未命中
        self.pe_stats["cache_misses"]["l2"] += 1
        return False

    def _create_memory_request_flit(self, req_type: str, memory_controller: int, request_id: str) -> CrossRingFlit:
        flit = CrossRingFlit(source=self.node_id, destination=memory_controller, packet_id=request_id, creation_time=self.current_cycle)
        flit.request_type = req_type
        return flit

    def _complete_task(self, task: Dict[str, Any], core_id: int) -> None:
        """完成任务"""
        task["state"] = TaskState.COMPLETED
        task["completion_time"] = self.current_cycle
        task["execution_time"] = task["completion_time"] - task["start_time"]

        # 移动到完成列表
        self.completed_tasks.append(task)
        del self.running_tasks[core_id]

        # 释放核心
        self.core_states[core_id]["state"] = "idle"
        self.core_states[core_id]["current_task"] = None
        self.core_states[core_id]["remaining_cycles"] = 0

        # 更新统计
        self.pe_stats["tasks_completed"] += 1

    def _generate_workload(self) -> None:
        """生成工作负载"""
        if self.current_cycle >= self.workload_generator["next_task_time"]:
            # 生成新任务
            task = self._create_synthetic_task()

            if len(self.task_queue) < self.task_queue.maxlen:
                self.task_queue.append(task)

            # 计算下一个任务的到达时间
            inter_arrival = self.workload_generator["task_inter_arrival_time"]
            next_time = self.current_cycle + inter_arrival
            self.workload_generator["next_task_time"] = next_time

    def _create_synthetic_task(self) -> Dict[str, Any]:
        """创建合成任务"""
        task_id = f"task_{self.node_id}_{len(self.completed_tasks) + len(self.running_tasks)}"

        # 随机选择任务类型
        task_types = [TaskType.COMPUTE, TaskType.MEMORY_READ, TaskType.MEMORY_WRITE]
        task_type = random.choice(task_types)

        # 随机选择任务大小
        task_size = random.choice(self.workload_generator["task_size_dist"])

        task = {
            "task_id": task_id,
            "type": task_type,
            "cycles_needed": task_size,
            "state": TaskState.PENDING,
            "creation_time": self.current_cycle,
            "base_address": random.randint(0, 1024 * 1024) * 64,
            "current_offset": 0,
            "last_address": 0,
        }

        return task

    def _generate_traffic(self) -> None:
        """生成网络流量"""
        if self.current_cycle >= self.traffic_generator["next_injection_time"] and random.random() < self.injection_rate:

            # 生成数据包
            packet = self._create_synthetic_packet()

            if len(self.output_buffers["local"]) < self.output_buffer_size:
                self.output_buffers["local"].append(packet)

                # 计算下一次注入时间
                next_injection = self.current_cycle + int(1.0 / self.injection_rate)
                self.traffic_generator["next_injection_time"] = next_injection

    def _create_synthetic_packet(self) -> CrossRingFlit:
        destination = random.choice(self.traffic_generator["destination_list"])
        packet_id = f"pkt_{self.node_id}_{self.current_cycle}_{self.traffic_generator['packet_id_counter']}"
        self.traffic_generator["packet_id_counter"] += 1
        flit = CrossRingFlit(source=self.node_id, destination=destination, packet_id=packet_id, creation_time=self.current_cycle)
        return flit

    def _process_synchronization(self) -> None:
        """处理同步"""
        # 检查是否有任务等待同步
        for core_id, task in self.running_tasks.items():
            if task and task.get("state") == TaskState.WAITING_SYNC and task.get("sync_id") in self.sync_points:

                sync_id = task["sync_id"]
                sync_point = self.sync_points[sync_id]

                if sync_point["status"] == "released":
                    # 同步点已释放，恢复任务执行
                    task["state"] = TaskState.RUNNING
                    self.core_states[core_id]["state"] = "busy"

    def _release_sync_waiting_tasks(self, sync_id: str) -> None:
        """释放等待同步的任务"""
        self.sync_points[sync_id]["status"] = "released"
        self.sync_points[sync_id]["release_time"] = self.current_cycle

    def _update_pe_statistics(self) -> None:
        """更新PE统计"""
        # 计算IPC
        if self.current_cycle > 0:
            total_cycles = self.current_cycle * self.num_cores
            self.pe_stats["ipc_actual"] = self.pe_stats["instructions_executed"] / total_cycles

        # 计算缓存命中率
        if self.cache_simulation:
            for level in ["l1", "l2"]:
                cache = self.cache_simulator[level]
                if cache["access_count"] > 0:
                    hit_rate = cache["hit_count"] / cache["access_count"]
                    self.pe_stats["cache_hit_rate"][level] = hit_rate

        # 计算空闲周期
        busy_cores = sum(1 for state in self.core_states.values() if state["state"] == "busy")
        if busy_cores == 0:
            self.pe_stats["cycles_idle"] += 1

    def create_sync_point(self, sync_id: str, expected_participants: int) -> None:
        """
        创建同步点。

        Args:
            sync_id: 同步点ID
            expected_participants: 期望的参与者数量
        """
        self.sync_points[sync_id] = {"expected_count": expected_participants, "received_count": 0, "status": "waiting", "creation_time": self.current_cycle}

    def add_task(self, task_type: TaskType, cycles_needed: int, **kwargs) -> str:
        """
        添加任务到队列。

        Args:
            task_type: 任务类型
            cycles_needed: 需要的周期数
            **kwargs: 其他任务参数

        Returns:
            任务ID
        """
        task_id = f"task_{self.node_id}_{len(self.completed_tasks) + len(self.running_tasks)}"

        task = {"task_id": task_id, "type": task_type, "cycles_needed": cycles_needed, "state": TaskState.PENDING, "creation_time": self.current_cycle, **kwargs}

        if len(self.task_queue) < self.task_queue.maxlen:
            self.task_queue.append(task)
            return task_id
        else:
            raise Exception("任务队列已满")

    def get_pe_status(self) -> Dict[str, Any]:
        """
        获取PE状态。

        Returns:
            PE状态字典
        """
        status = self.get_performance_stats()
        status.update(
            {
                "pe_type": self.pe_type,
                "num_cores": self.num_cores,
                "clock_frequency": self.clock_frequency,
                "workload_type": self.workload_type.value,
                "pe_stats": self.pe_stats.copy(),
                "core_states": {str(k): v for k, v in self.core_states.items()},
                "task_queue_length": len(self.task_queue),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "sync_points": len(self.sync_points),
            }
        )

        if self.cache_simulation:
            status["cache_status"] = {
                level: {
                    "size": cache["size"],
                    "access_count": cache["access_count"],
                    "hit_count": cache["hit_count"],
                    "hit_rate": cache["hit_count"] / cache["access_count"] if cache["access_count"] > 0 else 0.0,
                }
                for level, cache in self.cache_simulator.items()
            }

        return status

    def set_workload_pattern(self, workload_type: WorkloadType, params: Dict[str, Any]) -> None:
        """
        设置工作负载模式。

        Args:
            workload_type: 工作负载类型
            params: 工作负载参数
        """
        self.workload_type = workload_type
        self.workload_params.update(params)

        # 更新工作负载生成器参数
        if "task_inter_arrival_time" in params:
            self.workload_generator["task_inter_arrival_time"] = params["task_inter_arrival_time"]
        if "memory_intensity" in params:
            self.workload_generator["memory_intensity"] = params["memory_intensity"]

    def set_traffic_pattern(self, pattern: TrafficPattern, injection_rate: float) -> None:
        """
        设置流量模式。

        Args:
            pattern: 流量模式
            injection_rate: 注入率
        """
        self.traffic_pattern = pattern
        self.injection_rate = injection_rate
