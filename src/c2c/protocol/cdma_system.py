"""
CDMA系统模块
集成了消息同步、包格式、流控、性能监控和错误处理的完整CDMA系统
"""

from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
import time
import threading
from enum import Enum
import logging

# 导入现有组件
from .memory_types import MemoryType
from .transaction_manager import TransactionManager, TransactionInfo
from .credit import CreditManager, AddressInfo as CreditAddressInfo

# 导入新组件
from .message_sync import MessageSyncManager, SyncMessage, SyncMessageType, SyncState
from .packet_format import CDMAPacket, PacketFactory, PacketSerializer, PacketType, AddressInfo as PacketAddressInfo, DataType
from .flow_control import FlowController, FlowState
from .performance_monitor import PerformanceMonitor
from .error_handler import ErrorHandler, ErrorRecord, ErrorType, ErrorSeverity

from src.c2c.utils.exceptions import CDMAError, AddressError, ShapeCompatibilityError
import queue


class CDMASystemState(Enum):
    """CDMA系统状态"""

    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    CONGESTED = "congested"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class AddressInfo:
    """地址信息结构"""

    address: int
    shape: Tuple[int, ...]
    mem_type: MemoryType
    data_type: str = "float32"

    def size_bytes(self) -> int:
        """计算数据大小（字节）"""
        element_count = 1
        for dim in self.shape:
            element_count *= dim

        type_sizes = {"float32": 4, "float16": 2, "int32": 4, "int16": 2, "int8": 1, "uint8": 1}
        return element_count * type_sizes.get(self.data_type, 4)


@dataclass
class CreditWithAddress:
    """携带地址信息的Credit"""

    credit_count: int
    dst_address_info: AddressInfo
    transaction_id: str
    timestamp: float


@dataclass
class DMATransaction:
    """DMA传输事务"""

    transaction_id: str
    src_chip_id: str
    dst_chip_id: str
    src_address_info: AddressInfo
    dst_address_info: AddressInfo
    status: str = "pending"  # pending, transferring, completed, failed
    created_time: float = 0.0
    completed_time: float = 0.0


@dataclass
class MemoryRegion:
    """内存区域描述"""

    start_addr: int
    size: int
    mem_type: MemoryType
    alignment: int
    bandwidth_gbps: float  # 带宽 GB/s


@dataclass
class DMATransferRequest:
    """DMA传输请求"""

    request_id: str
    src_addr: int
    dst_addr: int
    size: int
    src_chip_id: str
    dst_chip_id: str
    src_mem_type: MemoryType
    dst_mem_type: MemoryType
    priority: int = 0
    created_time: float = 0.0


@dataclass
class DMATransferResult:
    """DMA传输结果"""

    request_id: str
    success: bool
    start_time: float
    end_time: float
    bytes_transferred: int
    error_message: Optional[str] = None
    bandwidth_achieved: float = 0.0


@dataclass
class CDMAMessage:
    """简化的CDMA消息格式（为兼容性保留）"""

    source_id: str
    destination_id: str
    message_type: str  # e.g., "send", "receive", "ack"
    tensor_shape: Any = None  # e.g., (H, W, C)
    data_type: str = None  # e.g., "float32", "int8"
    payload: Any = None  # Actual data payload
    reduce_op: str = None  # e.g., "sum", "mean"
    sequence_number: int = 0
    transaction_id: str = None


@dataclass
class CDMAOperationResult:
    """CDMA操作结果"""

    success: bool
    transaction_id: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    bytes_transferred: int = 0

    # 性能指标
    latency_ms: float = 0.0
    throughput_mbps: float = 0.0
    error_count: int = 0
    retry_count: int = 0

    # 系统状态
    system_state: Optional[str] = None
    flow_state: Optional[str] = None


class CDMASystem:
    """CDMA系统 - 完整的协议栈实现"""

    def __init__(self, chip_id: str, config: Dict[str, Any] = None):
        self._chip_id = chip_id
        self._state = CDMASystemState.INITIALIZING
        self._config = config or {}

        # 初始化核心组件
        self._cdma_engine = self._init_cdma_engine()
        self._dma_controller = self._init_dma_controller()
        self._transaction_manager = TransactionManager()
        self._credit_manager = CreditManager(chip_id)

        # 初始化新组件
        self._message_sync = MessageSyncManager(chip_id)
        self._flow_controller = FlowController(chip_id)
        self._performance_monitor = PerformanceMonitor(chip_id)
        self._error_handler = ErrorHandler(chip_id)

        # 系统级连接
        self._connected_systems: Dict[str, "CDMASystem"] = {}

        # 包序列号管理
        self._packet_sequence_counter = 0
        self._packet_sequence_lock = threading.Lock()

        # 回调函数
        self._error_callbacks: List[Callable[[ErrorRecord], None]] = []
        self._performance_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # 同步机制
        self._lock = threading.RLock()

        # 日志记录器
        self._logger = logging.getLogger(f"CDMASystem-{chip_id}")

        # 完成初始化
        self._setup_components()
        self._state = CDMASystemState.READY

        self._logger.info(f"CDMA系统初始化完成: {chip_id}")

    def _init_cdma_engine(self):
        """初始化CDMA引擎"""
        return {
            "credit_with_address": {},  # src_chip -> credit_info
            "pending_receives": {},  # transaction_id -> dst_address_info
            "active_transactions": {},
            "transaction_counter": 0,
            "memory_simulator": {},  # 模拟内存空间
        }

    def _init_dma_controller(self):
        """初始化DMA控制器"""
        memory_regions = {
            MemoryType.GMEM: MemoryRegion(start_addr=0x00000000, size=16 * 1024 * 1024 * 1024, mem_type=MemoryType.GMEM, alignment=64, bandwidth_gbps=200.0),
            MemoryType.L2M: MemoryRegion(start_addr=0x40000000, size=128 * 1024 * 1024, mem_type=MemoryType.L2M, alignment=32, bandwidth_gbps=800.0),
            MemoryType.LMEM: MemoryRegion(start_addr=0x80000000, size=512 * 1024 * 1024, mem_type=MemoryType.LMEM, alignment=16, bandwidth_gbps=1000.0),
        }

        return {
            "memory_regions": memory_regions,
            "memory_simulator": {},
            "transfer_queue": queue.PriorityQueue(),
            "active_transfers": {},
            "transfer_history": {},
            "is_running": False,
            "worker_thread": None,
            "transfer_counter": 0,
            "total_bytes_transferred": 0,
            "total_transfers": 0,
            "total_transfer_time": 0.0,
        }

    def _setup_components(self):
        """设置组件间的连接和回调"""
        # 设置消息同步回调
        self._message_sync.register_message_handler(SyncMessageType.ACK, self._handle_sync_ack)
        self._message_sync.register_message_handler(SyncMessageType.COMPLETE, self._handle_sync_complete)

        # 设置错误处理回调
        self._error_handler.register_error_callback(ErrorType.TRANSMISSION_ERROR, self._handle_transmission_error)
        self._error_handler.register_error_callback(ErrorType.TIMEOUT_ERROR, self._handle_timeout_error)

        # 设置流控回调
        self._flow_controller.set_congestion_callback(self._handle_congestion_state_change)
        self._flow_controller.set_buffer_full_callback(self._handle_buffer_full)

        # 启动DMA控制器
        self._start_dma_controller()

    def _start_dma_controller(self):
        """启动DMA控制器"""
        if not self._dma_controller["is_running"]:
            self._dma_controller["is_running"] = True
            self._dma_controller["worker_thread"] = threading.Thread(target=self._dma_worker_loop, daemon=True)
            self._dma_controller["worker_thread"].start()
            self._logger.info(f"芯片 {self._chip_id}：DMA控制器已启动")

    def _stop_dma_controller(self):
        """停止DMA控制器"""
        if self._dma_controller["is_running"]:
            self._dma_controller["is_running"] = False
            if self._dma_controller["worker_thread"]:
                self._dma_controller["worker_thread"].join(timeout=1.0)
            self._logger.info(f"芯片 {self._chip_id}：DMA控制器已停止")

    def _dma_worker_loop(self):
        """DMA工作线程主循环"""
        while self._dma_controller["is_running"]:
            try:
                # 获取传输请求（1秒超时）
                _, _, request = self._dma_controller["transfer_queue"].get(timeout=1.0)

                # 执行传输
                result = self._perform_dma_transfer(request)

                # 记录结果
                self._dma_controller["transfer_history"][request.request_id] = result

                # 更新统计信息
                if result.success:
                    self._dma_controller["total_bytes_transferred"] += result.bytes_transferred
                    self._dma_controller["total_transfers"] += 1
                    self._dma_controller["total_transfer_time"] += result.end_time - result.start_time

                # 从活跃传输中移除
                if request.request_id in self._dma_controller["active_transfers"]:
                    del self._dma_controller["active_transfers"][request.request_id]

                self._dma_controller["transfer_queue"].task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self._logger.error(f"DMA工作线程错误: {str(e)}")

    def _perform_dma_transfer(self, request: DMATransferRequest) -> DMATransferResult:
        """执行实际的DMA传输"""
        start_time = time.time()
        self._dma_controller["active_transfers"][request.request_id] = request

        try:
            self._logger.info(f"开始执行DMA传输 {request.request_id}")

            # 模拟从源地址读取数据
            src_data = self._read_dma_memory(request.src_addr, request.size)

            # 计算传输时间（基于带宽）
            src_bandwidth = self._dma_controller["memory_regions"][request.src_mem_type].bandwidth_gbps
            dst_bandwidth = self._dma_controller["memory_regions"][request.dst_mem_type].bandwidth_gbps
            effective_bandwidth = min(src_bandwidth, dst_bandwidth)

            # 如果是跨芯片传输，考虑C2C链路带宽
            if request.src_chip_id != request.dst_chip_id:
                c2c_bandwidth = 25.0  # 假设C2C带宽为25GB/s
                effective_bandwidth = min(effective_bandwidth, c2c_bandwidth)

            transfer_time = request.size / (effective_bandwidth * 1024 * 1024 * 1024)

            # 模拟传输延迟
            time.sleep(min(transfer_time, 0.1))  # 最多延迟100ms

            # 模拟写入目标地址
            self._write_dma_memory(request.dst_addr, src_data)

            end_time = time.time()
            actual_bandwidth = request.size / (end_time - start_time) / (1024 * 1024 * 1024)

            self._logger.info(f"DMA传输完成 {request.request_id}, 带宽: {actual_bandwidth:.2f} GB/s")

            return DMATransferResult(request_id=request.request_id, success=True, start_time=start_time, end_time=end_time, bytes_transferred=request.size, bandwidth_achieved=actual_bandwidth)

        except Exception as e:
            end_time = time.time()
            error_msg = f"DMA传输失败: {str(e)}"
            self._logger.error(error_msg)

            return DMATransferResult(request_id=request.request_id, success=False, start_time=start_time, end_time=end_time, bytes_transferred=0, error_message=error_msg)

    def _read_dma_memory(self, address: int, size: int) -> bytes:
        """从模拟内存读取数据"""
        memory_sim = self._dma_controller["memory_simulator"]
        if address in memory_sim:
            data = memory_sim[address]
            if len(data) >= size:
                return data[:size]

        # 生成模拟数据
        data = bytes([(address + i) % 256 for i in range(size)])
        memory_sim[address] = data
        return data

    def _write_dma_memory(self, address: int, data: bytes):
        """向模拟内存写入数据"""
        self._dma_controller["memory_simulator"][address] = data

    def _validate_dma_address(self, address: int, size: int, mem_type: MemoryType) -> bool:
        """验证DMA地址合法性"""
        if address < 0 or size <= 0 or size > 1024 * 1024 * 1024:
            return False
        if address > 0xFFFFFFFF:  # 32位地址空间
            return False
        return True

    def execute_dma_transfer(self, src_addr: int, dst_addr: int, data_size: int, src_chip_id: str, dst_chip_id: str, src_mem_type: MemoryType, dst_mem_type: MemoryType, priority: int = 0) -> str:
        """执行DMA传输"""
        # 验证地址
        if not self._validate_dma_address(src_addr, data_size, src_mem_type):
            raise CDMAError(f"源地址验证失败: 0x{src_addr:08x}")
        if not self._validate_dma_address(dst_addr, data_size, dst_mem_type):
            raise CDMAError(f"目标地址验证失败: 0x{dst_addr:08x}")

        request_id = f"dma_{self._chip_id}_{self._dma_controller['transfer_counter']}_{int(time.time() * 1000000)}"
        self._dma_controller["transfer_counter"] += 1

        request = DMATransferRequest(
            request_id=request_id,
            src_addr=src_addr,
            dst_addr=dst_addr,
            size=data_size,
            src_chip_id=src_chip_id,
            dst_chip_id=dst_chip_id,
            src_mem_type=src_mem_type,
            dst_mem_type=dst_mem_type,
            priority=priority,
            created_time=time.time(),
        )

        # 加入传输队列
        self._dma_controller["transfer_queue"].put((priority, time.time(), request))
        self._logger.info(f"DMA传输请求已排队: {request_id}")

        return request_id

    def connect_to_chip(self, other_chip_id: str, other_system: "CDMASystem"):
        """连接到其他芯片的CDMA系统"""
        with self._lock:
            self._connected_systems[other_chip_id] = other_system
            self._logger.info(f"连接到芯片: {other_chip_id}")

    def _get_next_packet_sequence(self) -> int:
        """获取下一个包序列号"""
        with self._packet_sequence_lock:
            self._packet_sequence_counter += 1
            return self._packet_sequence_counter

    def cdma_receive(
        self, dst_addr: int, dst_shape: Tuple[int, ...], dst_mem_type: MemoryType, src_chip_id: str, data_type: str = "float32", reduce_op: str = "none", timeout: float = 30.0
    ) -> CDMAOperationResult:
        """
        执行CDMA接收操作

        Args:
            dst_addr: 目标地址
            dst_shape: 目标tensor形状
            dst_mem_type: 目标内存类型
            src_chip_id: 源芯片ID
            data_type: 数据类型
            reduce_op: All Reduce操作
            timeout: 超时时间（秒）

        Returns:
            CDMAOperationResult: 操作结果
        """
        operation_start_time = self._performance_monitor.record_operation_start("cdma_receive")

        try:
            with self._lock:
                # 检查系统状态
                if self._state not in [CDMASystemState.READY, CDMASystemState.ACTIVE]:
                    return self._create_error_result(f"系统状态不正确: {self._state.value}", operation_start_time)

                # 检查连接
                if src_chip_id not in self._connected_systems:
                    return self._create_error_result(f"未连接到源芯片: {src_chip_id}", operation_start_time)

                # 生成事务ID
                transaction_id = f"cdma_recv_{self._chip_id}_{src_chip_id}_{int(time.time() * 1000000)}"

                # 创建配对事务
                transaction = self._transaction_manager.create_paired_transaction(send_chip_id=src_chip_id, recv_chip_id=self._chip_id, transaction_id=transaction_id, timeout_seconds=timeout)

                # 注册接收操作
                success = self._transaction_manager.register_receive_operation(
                    transaction_id=transaction_id, dst_addr=dst_addr, dst_shape=dst_shape, dst_mem_type=dst_mem_type.value, data_type=data_type
                )

                if not success:
                    return self._create_error_result("注册接收操作失败", operation_start_time, transaction_id)

                # 创建同步会话
                sync_session_id = self._message_sync.create_sync_session(src_chip_id, timeout)

                # 创建Credit包发送地址信息
                packet_addr_info = PacketAddressInfo(base_address=dst_addr, shape=dst_shape, data_type=DataType[data_type.upper()], memory_type=dst_mem_type.value)

                # 创建控制包
                control_packet = PacketFactory.create_control_packet(
                    source_id=self._chip_id,
                    dest_id=src_chip_id,
                    sequence_number=self._get_next_packet_sequence(),
                    control_info={
                        "operation": "cdma_receive",
                        "transaction_id": transaction_id,
                        "dst_address_info": packet_addr_info.to_dict(),
                        "reduce_operation": reduce_op,
                        "sync_session_id": sync_session_id,
                    },
                    transaction_id=transaction_id,
                )

                # 序列化并发送包
                packet_data = PacketSerializer.serialize_packet(control_packet)

                # 通过连接发送到源芯片
                src_system = self._connected_systems[src_chip_id]
                receive_success = src_system._receive_packet(packet_data, self._chip_id)

                if not receive_success:
                    return self._create_error_result("发送控制包失败", operation_start_time, transaction_id)

                # 发送同步消息
                self._message_sync.send_sync_message(session_id=sync_session_id, message_type=SyncMessageType.RX_SEND, payload={"transaction_id": transaction_id}, transaction_id=transaction_id)

                # 记录性能
                execution_time = time.time() - operation_start_time
                self._performance_monitor.record_operation_end("cdma_receive", operation_start_time, 0, True)

                self._logger.info(f"CDMA_receive操作完成: {transaction_id}")

                return CDMAOperationResult(
                    success=True,
                    transaction_id=transaction_id,
                    execution_time=execution_time,
                    system_state=self._state.value,
                    flow_state=str(self._flow_controller.get_flow_metrics().buffer_utilization),
                )

        except Exception as e:
            error_msg = f"CDMA_receive操作失败: {str(e)}"
            self._logger.error(error_msg)

            # 记录错误
            self._performance_monitor.record_operation_end("cdma_receive", operation_start_time, 0, False)

            return self._create_error_result(error_msg, operation_start_time)

    def cdma_send(
        self, src_addr: int, src_shape: Tuple[int, ...], dst_chip_id: str, src_mem_type: MemoryType = MemoryType.GMEM, data_type: str = "float32", reduce_op: str = "none"
    ) -> CDMAOperationResult:
        """
        执行CDMA发送操作

        Args:
            src_addr: 源地址
            src_shape: 源tensor形状
            dst_chip_id: 目标芯片ID
            src_mem_type: 源内存类型
            data_type: 数据类型
            reduce_op: All Reduce操作

        Returns:
            CDMAOperationResult: 操作结果
        """
        operation_start_time = self._performance_monitor.record_operation_start("cdma_send")

        try:
            with self._lock:
                # 检查系统状态
                if self._state not in [CDMASystemState.READY, CDMASystemState.ACTIVE]:
                    return self._create_error_result(f"系统状态不正确: {self._state.value}", operation_start_time)

                # 检查连接
                if dst_chip_id not in self._connected_systems:
                    return self._create_error_result(f"未连接到目标芯片: {dst_chip_id}", operation_start_time)

                # 检查流控状态
                data_size = self._calculate_data_size(src_shape, data_type)
                if not self._flow_controller.can_send_packet(data_size):
                    return self._create_error_result("流控限制，暂时无法发送", operation_start_time)

                # 检查Credit（这里需要等待receive操作的Credit信息）
                # 实际实现中，这里会等待来自目标芯片的Credit+地址信息

                # 创建数据包
                packet_sequence = self._get_next_packet_sequence()

                # 模拟数据读取
                data = self._read_tensor_data(src_addr, src_shape, data_type)

                # 应用All Reduce操作（如果指定）
                if reduce_op != "none":
                    data = self._apply_reduce_operation(data, reduce_op, data_type)

                # 创建地址信息
                src_addr_info = PacketAddressInfo(base_address=src_addr, shape=src_shape, data_type=DataType[data_type.upper()], memory_type=src_mem_type.value)

                # 这里需要从Credit信息中获取目标地址信息
                # 暂时创建一个模拟的目标地址信息
                dst_addr_info = PacketAddressInfo(base_address=0x1000, shape=src_shape, data_type=DataType[data_type.upper()], memory_type="GMEM")  # 模拟地址

                # 创建数据包
                data_packet = PacketFactory.create_data_packet(
                    source_id=self._chip_id, dest_id=dst_chip_id, sequence_number=packet_sequence, src_address_info=src_addr_info, dst_address_info=dst_addr_info, data=data, reduce_operation=reduce_op
                )

                # 序列化包
                packet_data = PacketSerializer.serialize_packet(data_packet)

                # 记录包发送（用于错误检测）
                self._error_handler.process_packet_sent(packet_id=f"data_{packet_sequence}", sequence_number=packet_sequence, destination=dst_chip_id, data=packet_data)

                # 记录流控
                self._flow_controller.packet_sent(packet_sequence, len(packet_data))

                # 发送到目标芯片
                dst_system = self._connected_systems[dst_chip_id]
                send_success = dst_system._receive_packet(packet_data, self._chip_id)

                if not send_success:
                    return self._create_error_result("发送数据包失败", operation_start_time)

                # 记录性能
                execution_time = time.time() - operation_start_time
                self._performance_monitor.record_operation_end("cdma_send", operation_start_time, len(data), True)

                # 计算吞吐量
                throughput_mbps = (len(data) / (1024 * 1024)) / max(execution_time, 0.001)

                self._logger.info(f"CDMA_send操作完成: 传输 {len(data)} 字节")

                return CDMAOperationResult(
                    success=True,
                    execution_time=execution_time,
                    bytes_transferred=len(data),
                    latency_ms=execution_time * 1000,
                    throughput_mbps=throughput_mbps,
                    system_state=self._state.value,
                    flow_state=str(self._flow_controller.get_flow_metrics().current_bandwidth),
                )

        except Exception as e:
            error_msg = f"CDMA_send操作失败: {str(e)}"
            self._logger.error(error_msg)

            # 记录错误
            self._performance_monitor.record_operation_end("cdma_send", operation_start_time, 0, False)

            return self._create_error_result(error_msg, operation_start_time)

    def _receive_packet(self, packet_data: bytes, sender_chip_id: str) -> bool:
        """
        接收来自其他芯片的包

        Args:
            packet_data: 包数据
            sender_chip_id: 发送方芯片ID

        Returns:
            bool: 接收是否成功
        """
        try:
            # 反序列化包
            packet = PacketSerializer.deserialize_packet(packet_data)

            # 记录包接收（用于错误检测）
            # 提取载荷数据用于校验
            payload_data = packet_data[packet.header.header_size : packet.header.total_size]
            errors = self._error_handler.process_packet_received(
                packet_id=f"{packet.header.packet_type.value}_{packet.header.sequence_number}",
                sequence_number=packet.header.sequence_number,
                source=sender_chip_id,
                data=payload_data,  # 只传递载荷数据
                metadata={"expected_size": packet.header.payload_size, "checksum": packet.header.payload_checksum, "checksum_algorithm": "crc32"},
            )

            # 更新流控
            self._flow_controller.receive_packet(packet_data, sender_chip_id, packet.header.sequence_number)

            # 根据包类型处理
            if packet.header.packet_type == PacketType.CONTROL:
                return self._handle_control_packet(packet, sender_chip_id)
            elif packet.header.packet_type == PacketType.DATA:
                return self._handle_data_packet(packet, sender_chip_id)
            elif packet.header.packet_type == PacketType.SYNC:
                return self._handle_sync_packet(packet, sender_chip_id)
            elif packet.header.packet_type == PacketType.ACK:
                return self._handle_ack_packet(packet, sender_chip_id)
            else:
                self._logger.warning(f"未知包类型: {packet.header.packet_type}")
                return False

        except Exception as e:
            self._logger.error(f"包接收处理失败: {e}")
            return False

    def _handle_control_packet(self, packet: CDMAPacket, sender_chip_id: str) -> bool:
        """处理控制包"""
        try:
            control_info = packet.payload.control_info
            operation = control_info.get("operation")

            if operation == "cdma_receive":
                # 处理接收方发送的地址信息
                transaction_id = control_info.get("transaction_id")
                dst_address_info = control_info.get("dst_address_info")

                # 这里可以存储地址信息，供后续send操作使用
                self._logger.info(f"收到CDMA_receive控制包: {transaction_id}")

                return True
            else:
                self._logger.warning(f"未知控制操作: {operation}")
                return False

        except Exception as e:
            self._logger.error(f"控制包处理失败: {e}")
            return False

    def _handle_data_packet(self, packet: CDMAPacket, sender_chip_id: str) -> bool:
        """处理数据包"""
        try:
            # 获取目标地址信息
            dst_addr_info = packet.payload.dst_address_info

            if dst_addr_info:
                # 检查数据包数据有效性
                if packet.payload.data is None:
                    self._logger.error("数据包数据为空")
                    return False

                # 写入数据到目标地址
                success = self._write_tensor_data(dst_addr_info.base_address, dst_addr_info.shape, packet.payload.data, dst_addr_info.data_type.value)

                if success:
                    self._logger.info(f"数据包写入成功: {len(packet.payload.data)} 字节")

                    # 发送确认
                    ack_packet = PacketFactory.create_ack_packet(
                        source_id=self._chip_id,
                        dest_id=sender_chip_id,
                        sequence_number=self._get_next_packet_sequence(),
                        ack_info={"original_sequence": packet.header.sequence_number, "bytes_received": len(packet.payload.data), "status": "success"},
                        transaction_id=packet.header.transaction_id,
                    )

                    # 发送确认包
                    ack_data = PacketSerializer.serialize_packet(ack_packet)
                    if sender_chip_id in self._connected_systems:
                        self._connected_systems[sender_chip_id]._receive_packet(ack_data, self._chip_id)

                    return True
                else:
                    self._logger.error("数据包写入失败")
                    return False
            else:
                self._logger.error("数据包缺少目标地址信息")
                return False

        except Exception as e:
            self._logger.error(f"数据包处理失败: {e}")
            return False

    def _handle_sync_packet(self, packet: CDMAPacket, sender_chip_id: str) -> bool:
        """处理同步包"""
        try:
            # 将同步包转换为同步消息
            sync_message = SyncMessage(
                message_id=f"sync_{packet.header.sequence_number}",
                sender_chip_id=sender_chip_id,
                receiver_chip_id=self._chip_id,
                message_type=SyncMessageType.ACK,  # 这里需要根据实际情况确定
                timestamp=time.time(),
                transaction_id=packet.header.transaction_id,
                payload=packet.payload.control_info,
            )

            # 交给消息同步管理器处理
            return self._message_sync.receive_sync_message(sync_message)

        except Exception as e:
            self._logger.error(f"同步包处理失败: {e}")
            return False

    def _handle_ack_packet(self, packet: CDMAPacket, sender_chip_id: str) -> bool:
        """处理ACK包"""
        try:
            ack_info = packet.payload.control_info
            original_sequence = ack_info.get("original_sequence")
            bytes_received = ack_info.get("bytes_received")
            status = ack_info.get("status")
            transaction_id = packet.header.transaction_id

            self._logger.info(f"收到来自 {sender_chip_id} 的ACK包: 原始序列号={original_sequence}, 状态={status}, 事务ID={transaction_id}")

            # 通知重传管理器，该包已成功接收
            self._error_handler.mark_retransmission_successful(original_sequence)

            # 更新事务状态（如果需要）
            if transaction_id:
                self._transaction_manager.complete_transaction(transaction_id, success=True)

            return True
        except Exception as e:
            self._logger.error(f"ACK包处理失败: {e}")
            return False

    def _calculate_data_size(self, shape: Tuple[int, ...], data_type: str) -> int:
        """计算数据大小"""
        element_count = 1
        for dim in shape:
            element_count *= dim

        type_sizes = {"float32": 4, "float16": 2, "int32": 4, "int16": 2, "int8": 1, "uint8": 1}

        return element_count * type_sizes.get(data_type, 4)

    def _read_tensor_data(self, addr: int, shape: Tuple[int, ...], data_type: str) -> bytes:
        """读取tensor数据（模拟）"""
        data_size = self._calculate_data_size(shape, data_type)
        # 生成可预测的模拟数据，确保每次读取相同地址和大小的数据时，内容一致
        # 使用地址和大小作为种子，生成一个简单的重复模式
        seed = (addr + data_size) % 256
        data = bytes([(seed + i) % 256 for i in range(data_size)])
        return data

    def _write_tensor_data(self, addr: int, shape: Tuple[int, ...], data: bytes, data_type: str) -> bool:
        """写入tensor数据（模拟）"""
        try:
            # 检查数据有效性
            if data is None:
                self._logger.error("数据为空，无法写入")
                return False

            if not isinstance(data, bytes):
                self._logger.error(f"数据类型错误: 期望bytes, 实际{type(data)}")
                return False

            # 模拟数据写入
            expected_size = self._calculate_data_size(shape, data_type)
            if len(data) != expected_size:
                self._logger.error(f"数据大小不匹配: 期望{expected_size}, 实际{len(data)}")
                return False

            # 这里是实际的内存写入操作
            self._dma_controller["memory_simulator"][addr] = data
            self._logger.debug(f"写入数据到地址 0x{addr:08x}: {len(data)} 字节")
            return True

        except Exception as e:
            self._logger.error(f"数据写入失败: {e}")
            return False

    def _apply_reduce_operation(self, data: bytes, reduce_op: str, data_type: str) -> bytes:
        """应用All Reduce操作（模拟）"""
        if reduce_op == "none":
            return data

        # 这里应该实现实际的reduce操作
        # 暂时返回原数据
        self._logger.debug(f"应用Reduce操作: {reduce_op}")
        return data

    def _create_error_result(self, error_msg: str, start_time: float, transaction_id: str = None) -> CDMAOperationResult:
        """创建错误结果"""
        execution_time = time.time() - start_time
        return CDMAOperationResult(success=False, transaction_id=transaction_id, error_message=error_msg, execution_time=execution_time, system_state=self._state.value, error_count=1)

    # 回调处理方法
    def _handle_sync_ack(self, message: SyncMessage):
        """处理同步确认"""
        self._logger.debug(f"收到同步确认: {message.message_id}")

    def _handle_sync_complete(self, message: SyncMessage):
        """处理同步完成"""
        self._logger.info(f"同步完成: {message.message_id}")

    def _handle_transmission_error(self, error: ErrorRecord):
        """处理传输错误"""
        self._logger.warning(f"传输错误: {error.message}")

        # 可以在这里实现特定的错误处理逻辑
        if error.severity == ErrorSeverity.CRITICAL:
            self._state = CDMASystemState.ERROR

    def _handle_timeout_error(self, error: ErrorRecord):
        """处理超时错误"""
        self._logger.warning(f"超时错误: {error.message}")

    def _handle_congestion_state_change(self, new_state: FlowState):
        """处理拥塞状态变化"""
        if new_state == FlowState.CONGESTED:
            self._state = CDMASystemState.CONGESTED
            self._logger.warning("系统进入拥塞状态")
        elif new_state == FlowState.NORMAL and self._state == CDMASystemState.CONGESTED:
            self._state = CDMASystemState.READY
            self._logger.info("系统退出拥塞状态")

    def _handle_buffer_full(self):
        """处理缓冲区满"""
        self._logger.warning("接收缓冲区已满")

    # 公共接口方法
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """获取综合系统状态"""
        with self._lock:
            return {
                "chip_id": self._chip_id,
                "system_state": self._state.value,
                "performance_stats": self._performance_monitor.get_summary_stats(),
                "flow_stats": self._flow_controller.get_comprehensive_status(),
                "error_stats": self._error_handler.get_error_statistics(),
                "sync_stats": self._message_sync.get_sync_statistics(),
                "transaction_stats": self._transaction_manager.get_statistics(),
                "connected_chips": list(self._connected_systems.keys()),
            }

    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        return self._performance_monitor.generate_performance_report()

    def register_error_callback(self, callback: Callable[[ErrorRecord], None]):
        """注册错误回调"""
        self._error_callbacks.append(callback)
        self._error_handler.register_error_callback(ErrorType.TRANSMISSION_ERROR, callback)

    def register_performance_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """注册性能回调"""
        self._performance_callbacks.append(callback)
        self._performance_monitor.register_report_callback(callback)

    def cleanup(self):
        """清理系统资源"""
        with self._lock:
            self._logger.info(f"开始清理CDMA系统: {self._chip_id}")

            # 清理各个组件
            self._message_sync.cleanup_expired_sessions()
            self._error_handler.process_periodic_checks()
            self._performance_monitor.cleanup_old_data()

            # 清理过期事务
            self._transaction_manager.cleanup_expired_transactions()
            self._transaction_manager.clear_completed_transactions()

            # 清理Credit
            self._credit_manager.cleanup_expired_credits()

            self._logger.info("系统清理完成")

    def shutdown(self):
        """关闭系统"""
        with self._lock:
            self._logger.info(f"开始关闭CDMA系统: {self._chip_id}")

            # 停止各个组件
            self._stop_dma_controller()
            self._message_sync.shutdown()
            self._performance_monitor.shutdown()
            self._error_handler.shutdown()
            self._flow_controller.shutdown()

            # 断开所有连接
            self._connected_systems.clear()

            self._state = CDMASystemState.IDLE
            self._logger.info("系统已关闭")

    @property
    def chip_id(self) -> str:
        return self._chip_id

    @property
    def state(self) -> CDMASystemState:
        return self._state

    # 为兼容性提供简化的协议接口
    def send_message(self, message: CDMAMessage) -> CDMAOperationResult:
        """发送CDMA消息（简化接口，兼容旧代码）"""
        if message.message_type == "send":
            return self.cdma_send(src_addr=0x0, src_shape=message.tensor_shape, dst_chip_id=message.destination_id, data_type=message.data_type, reduce_op=message.reduce_op or "none")  # 模拟地址
        elif message.message_type == "receive":
            return self.cdma_receive(
                dst_addr=0x0,  # 模拟地址
                dst_shape=message.tensor_shape,
                dst_mem_type=MemoryType.GMEM,
                src_chip_id=message.source_id,
                data_type=message.data_type,
                reduce_op=message.reduce_op or "none",
            )
        else:
            return CDMAOperationResult(success=False, error_message=f"不支持的消息类型: {message.message_type}")

    def process_message(self, message: CDMAMessage) -> Any:
        """处理CDMA消息（简化接口，兼容旧代码）"""
        self._logger.info(f"节点 {self._chip_id} 收到CDMA消息: {message.message_type} 从 {message.source_id} 到 {message.destination_id}")

        if message.destination_id != self._chip_id:
            self._logger.warning(f"消息不属于该节点。期望 {self._chip_id}，实际得到 {message.destination_id}")
            return None

        if message.message_type == "ack":
            self._logger.info(f"收到来自 {message.source_id} 的事务 {message.transaction_id} 的ACK")
            return None
        elif message.message_type == "data_response":
            self._logger.info(f"收到来自 {message.source_id} 的事务 {message.transaction_id} 的DATA_RESPONSE。数据：{message.payload}")
            return None
        else:
            self._logger.warning(f"未知的CDMA消息类型：{message.message_type}")
            return None


# 为兼容性提供别名
CDMAProtocol = CDMASystem
