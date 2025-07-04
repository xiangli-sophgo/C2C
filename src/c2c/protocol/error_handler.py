"""
错误处理模块
实现传输错误检测和恢复，包括重传机制和数据完整性验证
"""

from typing import Dict, Any, Optional, List, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import deque, defaultdict
import hashlib
import zlib
import logging
import random

from src.c2c.utils.exceptions import CDMAError


class ErrorType(Enum):
    """错误类型枚举"""

    TRANSMISSION_ERROR = "transmission_error"  # 传输错误
    TIMEOUT_ERROR = "timeout_error"  # 超时错误
    CHECKSUM_ERROR = "checksum_error"  # 校验和错误
    SEQUENCE_ERROR = "sequence_error"  # 序列错误
    BUFFER_OVERFLOW = "buffer_overflow"  # 缓冲区溢出
    PROTOCOL_ERROR = "protocol_error"  # 协议错误
    MEMORY_ERROR = "memory_error"  # 内存错误
    NETWORK_ERROR = "network_error"  # 网络错误


class ErrorSeverity(Enum):
    """错误严重程度枚举"""

    LOW = "low"  # 低 - 可以继续运行
    MEDIUM = "medium"  # 中 - 需要重试
    HIGH = "high"  # 高 - 需要恢复措施
    CRITICAL = "critical"  # 严重 - 需要停止操作


class RecoveryAction(Enum):
    """恢复动作枚举"""

    RETRY = "retry"  # 重试
    RETRANSMIT = "retransmit"  # 重传
    RESET_CONNECTION = "reset_connection"  # 重置连接
    FALLBACK = "fallback"  # 降级处理
    IGNORE = "ignore"  # 忽略
    ABORT = "abort"  # 中止


@dataclass
class ErrorRecord:
    """错误记录"""

    error_id: str
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: float
    source_id: str
    context: Dict[str, Any] = field(default_factory=dict)

    # 相关信息
    transaction_id: Optional[str] = None
    sequence_number: Optional[int] = None
    packet_id: Optional[str] = None

    # 处理信息
    recovery_action: Optional[RecoveryAction] = None
    recovery_attempts: int = 0
    resolved: bool = False
    resolved_timestamp: Optional[float] = None

    def age(self) -> float:
        """获取错误年龄（秒）"""
        return time.time() - self.timestamp


@dataclass
class RetransmissionRequest:
    """重传请求"""

    request_id: str
    sequence_number: int
    original_timestamp: float
    retry_count: int
    max_retries: int
    backoff_factor: float
    next_retry_time: float

    # 数据信息
    data: bytes
    destination: str
    packet_id: str

    def should_retry(self) -> bool:
        """是否应该重试"""
        return self.retry_count < self.max_retries and time.time() >= self.next_retry_time

    def calculate_next_retry_time(self):
        """计算下次重试时间（指数退避）"""
        delay = min(0.1 * (self.backoff_factor**self.retry_count), 30.0)  # 最大30秒
        self.next_retry_time = time.time() + delay


class ErrorDetector:
    """错误检测器"""

    def __init__(self, source_id: str):
        self._source_id = source_id

        # 检测阈值
        self._timeout_threshold = 5.0  # 超时阈值（秒）
        self._error_rate_threshold = 0.05  # 错误率阈值（5%）
        self._consecutive_errors_threshold = 3  # 连续错误阈值

        # 检测状态
        self._packet_history: deque = deque(maxlen=1000)  # 包历史
        self._sequence_tracker: Dict[str, Set[int]] = defaultdict(set)  # 序列跟踪
        self._timeout_tracker: Dict[str, float] = {}  # 超时跟踪

        # 统计信息
        self._total_packets = 0
        self._error_packets = 0
        self._timeout_packets = 0
        self._checksum_errors = 0
        self._sequence_errors = 0

        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"ErrorDetector-{source_id}")

    def check_packet_timeout(self, packet_id: str, timestamp: float) -> bool:
        """
        检查包超时

        Args:
            packet_id: 包ID
            timestamp: 时间戳

        Returns:
            bool: 是否超时
        """
        with self._lock:
            current_time = time.time()

            if packet_id in self._timeout_tracker:
                send_time = self._timeout_tracker[packet_id]
                if current_time - send_time > self._timeout_threshold:
                    self._timeout_packets += 1
                    del self._timeout_tracker[packet_id]
                    return True

            return False

    def register_packet_sent(self, packet_id: str, sequence_number: int, destination: str):
        """注册包发送"""
        with self._lock:
            self._timeout_tracker[packet_id] = time.time()
            self._sequence_tracker[destination].add(sequence_number)
            self._total_packets += 1

    def register_packet_received(self, packet_id: str, sequence_number: int, source: str, checksum_valid: bool) -> List[ErrorRecord]:
        """
        注册包接收并检测错误

        Args:
            packet_id: 包ID
            sequence_number: 序列号
            source: 源ID
            checksum_valid: 校验和是否有效

        Returns:
            List[ErrorRecord]: 检测到的错误列表
        """
        with self._lock:
            errors = []
            current_time = time.time()

            # 移除超时跟踪
            if packet_id in self._timeout_tracker:
                del self._timeout_tracker[packet_id]

            # 检查校验和错误
            if not checksum_valid:
                self._checksum_errors += 1
                errors.append(
                    ErrorRecord(
                        error_id=f"checksum_{packet_id}",
                        error_type=ErrorType.CHECKSUM_ERROR,
                        severity=ErrorSeverity.MEDIUM,
                        message=f"包校验和验证失败: {packet_id}",
                        timestamp=current_time,
                        source_id=self._source_id,
                        packet_id=packet_id,
                        sequence_number=sequence_number,
                        context={"source": source},
                    )
                )

            # 检查序列错误
            expected_sequences = self._sequence_tracker[source]
            if sequence_number in expected_sequences:
                # 重复包
                errors.append(
                    ErrorRecord(
                        error_id=f"duplicate_{packet_id}",
                        error_type=ErrorType.SEQUENCE_ERROR,
                        severity=ErrorSeverity.LOW,
                        message=f"重复包序列号: {sequence_number}",
                        timestamp=current_time,
                        source_id=self._source_id,
                        packet_id=packet_id,
                        sequence_number=sequence_number,
                        context={"source": source, "type": "duplicate"},
                    )
                )
            else:
                # 检查序列号间隙
                if expected_sequences and sequence_number > 0:
                    max_seq = max(expected_sequences)
                    if sequence_number > max_seq + 1:
                        # 可能的丢包
                        for missing_seq in range(max_seq + 1, sequence_number):
                            errors.append(
                                ErrorRecord(
                                    error_id=f"missing_{source}_{missing_seq}",
                                    error_type=ErrorType.SEQUENCE_ERROR,
                                    severity=ErrorSeverity.MEDIUM,
                                    message=f"缺失包序列号: {missing_seq}",
                                    timestamp=current_time,
                                    source_id=self._source_id,
                                    sequence_number=missing_seq,
                                    context={"source": source, "type": "missing"},
                                )
                            )

                expected_sequences.add(sequence_number)

            # 记录到历史
            self._packet_history.append({"packet_id": packet_id, "sequence_number": sequence_number, "source": source, "timestamp": current_time, "errors": len(errors) > 0})

            if errors:
                self._error_packets += 1

            return errors

    def check_periodic_errors(self) -> List[ErrorRecord]:
        """检查周期性错误"""
        with self._lock:
            errors = []
            current_time = time.time()

            # 检查超时的包
            timeout_packets = []
            for packet_id, send_time in self._timeout_tracker.items():
                if current_time - send_time > self._timeout_threshold:
                    timeout_packets.append(packet_id)

            for packet_id in timeout_packets:
                del self._timeout_tracker[packet_id]
                self._timeout_packets += 1

                errors.append(
                    ErrorRecord(
                        error_id=f"timeout_{packet_id}",
                        error_type=ErrorType.TIMEOUT_ERROR,
                        severity=ErrorSeverity.MEDIUM,
                        message=f"包传输超时: {packet_id}",
                        timestamp=current_time,
                        source_id=self._source_id,
                        packet_id=packet_id,
                        context={"timeout_threshold": self._timeout_threshold},
                    )
                )

            # 检查错误率
            if len(self._packet_history) >= 100:  # 需要足够的样本
                recent_packets = list(self._packet_history)[-100:]
                error_count = sum(1 for p in recent_packets if p["errors"])
                error_rate = error_count / len(recent_packets)

                if error_rate > self._error_rate_threshold:
                    errors.append(
                        ErrorRecord(
                            error_id=f"high_error_rate_{int(current_time)}",
                            error_type=ErrorType.TRANSMISSION_ERROR,
                            severity=ErrorSeverity.HIGH,
                            message=f"高错误率检测: {error_rate:.2%}",
                            timestamp=current_time,
                            source_id=self._source_id,
                            context={"error_rate": error_rate, "threshold": self._error_rate_threshold},
                        )
                    )

            return errors

    def get_detection_stats(self) -> Dict[str, Any]:
        """获取检测统计信息"""
        with self._lock:
            error_rate = self._error_packets / max(1, self._total_packets)
            timeout_rate = self._timeout_packets / max(1, self._total_packets)

            return {
                "source_id": self._source_id,
                "total_packets": self._total_packets,
                "error_packets": self._error_packets,
                "timeout_packets": self._timeout_packets,
                "checksum_errors": self._checksum_errors,
                "sequence_errors": self._sequence_errors,
                "error_rate": error_rate,
                "timeout_rate": timeout_rate,
                "pending_timeouts": len(self._timeout_tracker),
            }


class RetransmissionManager:
    """重传管理器"""

    def __init__(self, source_id: str):
        self._source_id = source_id

        # 重传配置
        self._max_retries = 3
        self._initial_backoff = 0.1  # 100ms
        self._backoff_factor = 2.0
        self._max_backoff = 30.0  # 30秒

        # 重传队列
        self._retransmission_queue: Dict[str, RetransmissionRequest] = {}
        self._sequence_to_request: Dict[int, str] = {}  # 序列号到请求ID的映射

        # 统计信息
        self._total_retransmissions = 0
        self._successful_retransmissions = 0
        self._failed_retransmissions = 0

        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"RetransmissionManager-{source_id}")

    def schedule_retransmission(self, sequence_number: int, data: bytes, destination: str, packet_id: str) -> str:
        """
        调度重传

        Args:
            sequence_number: 序列号
            data: 数据
            destination: 目标
            packet_id: 包ID

        Returns:
            str: 重传请求ID
        """
        with self._lock:
            request_id = f"retx_{self._source_id}_{sequence_number}_{int(time.time() * 1000)}"

            request = RetransmissionRequest(
                request_id=request_id,
                sequence_number=sequence_number,
                original_timestamp=time.time(),
                retry_count=0,
                max_retries=self._max_retries,
                backoff_factor=self._backoff_factor,
                next_retry_time=time.time() + self._initial_backoff,
                data=data,
                destination=destination,
                packet_id=packet_id,
            )

            self._retransmission_queue[request_id] = request
            self._sequence_to_request[sequence_number] = request_id

            self._logger.debug(f"调度重传: seq={sequence_number}, dest={destination}")
            return request_id

    def get_ready_retransmissions(self) -> List[RetransmissionRequest]:
        """获取准备重传的请求"""
        with self._lock:
            ready_requests = []
            current_time = time.time()

            for request in self._retransmission_queue.values():
                if request.should_retry():
                    ready_requests.append(request)

            return ready_requests

    def mark_retransmission_sent(self, request_id: str) -> bool:
        """
        标记重传已发送

        Args:
            request_id: 请求ID

        Returns:
            bool: 标记是否成功
        """
        with self._lock:
            if request_id not in self._retransmission_queue:
                return False

            request = self._retransmission_queue[request_id]
            request.retry_count += 1
            request.calculate_next_retry_time()

            self._total_retransmissions += 1

            # 如果达到最大重试次数，标记为失败
            if request.retry_count >= request.max_retries:
                self._failed_retransmissions += 1
                self._remove_retransmission_request(request_id)
                self._logger.warning(f"重传失败，达到最大重试次数: seq={request.sequence_number}")

            return True

    def mark_retransmission_successful(self, sequence_number: int) -> bool:
        """
        标记重传成功

        Args:
            sequence_number: 序列号

        Returns:
            bool: 标记是否成功
        """
        with self._lock:
            if sequence_number not in self._sequence_to_request:
                return False

            request_id = self._sequence_to_request[sequence_number]

            if request_id in self._retransmission_queue:
                self._successful_retransmissions += 1
                self._remove_retransmission_request(request_id)
                self._logger.debug(f"重传成功: seq={sequence_number}")
                return True

            return False

    def _remove_retransmission_request(self, request_id: str):
        """移除重传请求"""
        if request_id in self._retransmission_queue:
            request = self._retransmission_queue[request_id]

            # 从序列号映射中移除
            if request.sequence_number in self._sequence_to_request:
                del self._sequence_to_request[request.sequence_number]

            # 从队列中移除
            del self._retransmission_queue[request_id]

    def cleanup_expired_requests(self):
        """清理过期的重传请求"""
        with self._lock:
            current_time = time.time()
            expired_requests = []

            for request_id, request in self._retransmission_queue.items():
                # 如果请求超过最大生命周期（例如5分钟），则清理
                if current_time - request.original_timestamp > 300:
                    expired_requests.append(request_id)

            for request_id in expired_requests:
                self._remove_retransmission_request(request_id)
                self._logger.debug(f"清理过期重传请求: {request_id}")

    def get_retransmission_stats(self) -> Dict[str, Any]:
        """获取重传统计信息"""
        with self._lock:
            success_rate = 0.0
            if self._total_retransmissions > 0:
                success_rate = self._successful_retransmissions / self._total_retransmissions

            return {
                "source_id": self._source_id,
                "pending_requests": len(self._retransmission_queue),
                "total_retransmissions": self._total_retransmissions,
                "successful_retransmissions": self._successful_retransmissions,
                "failed_retransmissions": self._failed_retransmissions,
                "success_rate": success_rate,
            }


class IntegrityChecker:
    """完整性校验器"""

    def __init__(self, source_id: str):
        self._source_id = source_id

        # 校验配置
        self._use_crc32 = True
        self._use_md5 = False
        self._use_sha256 = False

        # 统计信息
        self._total_checks = 0
        self._passed_checks = 0
        self._failed_checks = 0

        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"IntegrityChecker-{source_id}")

    def calculate_checksum(self, data: bytes, algorithm: str = "crc32") -> str:
        """
        计算校验和

        Args:
            data: 数据
            algorithm: 算法类型

        Returns:
            str: 校验和
        """
        try:
            if algorithm == "crc32":
                return zlib.crc32(data) & 0xFFFFFFFF
            elif algorithm == "md5":
                return hashlib.md5(data).hexdigest()
            elif algorithm == "sha256":
                return hashlib.sha256(data).hexdigest()
            else:
                raise ValueError(f"不支持的校验算法: {algorithm}")
        except Exception as e:
            self._logger.error(f"校验和计算失败: {e}")
            return ""

    def verify_checksum(self, data: bytes, expected_checksum, algorithm: str = "crc32") -> bool:
        """
        验证校验和

        Args:
            data: 数据
            expected_checksum: 期望的校验和 (支持int或str类型)
            algorithm: 算法类型

        Returns:
            bool: 校验是否通过
        """
        with self._lock:
            self._total_checks += 1

            try:
                calculated_checksum = self.calculate_checksum(data, algorithm)

                # 确保类型一致性
                if algorithm == "crc32":
                    # CRC32应该统一为整数类型进行比较
                    if isinstance(expected_checksum, str):
                        try:
                            expected_checksum = int(expected_checksum)
                        except ValueError:
                            self._failed_checks += 1
                            self._logger.error(f"无效的校验和格式: {expected_checksum}")
                            return False

                if calculated_checksum == expected_checksum:
                    self._passed_checks += 1
                    return True
                else:
                    self._failed_checks += 1
                    self._logger.warning(f"校验和不匹配: 期望={expected_checksum}, 实际={calculated_checksum}")
                    return False

            except Exception as e:
                self._failed_checks += 1
                self._logger.error(f"校验和验证失败: {e}")
                return False

    def verify_data_integrity(self, data: bytes, metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证数据完整性

        Args:
            data: 数据
            metadata: 元数据（包含校验信息）

        Returns:
            Tuple[bool, List[str]]: (是否通过, 错误消息列表)
        """
        errors = []

        # 检查数据长度
        if "expected_size" in metadata:
            expected_size = metadata["expected_size"]
            if len(data) != expected_size:
                errors.append(f"数据长度不匹配: 期望={expected_size}, 实际={len(data)}")

        # 检查校验和
        if "checksum" in metadata and "checksum_algorithm" in metadata:
            checksum = metadata["checksum"]
            algorithm = metadata["checksum_algorithm"]

            if not self.verify_checksum(data, checksum, algorithm):
                errors.append(f"校验和验证失败: {algorithm}")

        # 检查数据格式
        if "data_format" in metadata:
            format_type = metadata["data_format"]
            if not self._validate_data_format(data, format_type):
                errors.append(f"数据格式验证失败: {format_type}")

        return len(errors) == 0, errors

    def _validate_data_format(self, data: bytes, format_type: str) -> bool:
        """验证数据格式"""
        try:
            if format_type == "json":
                import json

                json.loads(data.decode("utf-8"))
                return True
            elif format_type == "binary":
                # 简单的二进制格式检查
                return len(data) > 0
            else:
                # 其他格式默认通过
                return True
        except Exception:
            return False

    def get_integrity_stats(self) -> Dict[str, Any]:
        """获取完整性统计信息"""
        with self._lock:
            pass_rate = 0.0
            if self._total_checks > 0:
                pass_rate = self._passed_checks / self._total_checks

            return {"source_id": self._source_id, "total_checks": self._total_checks, "passed_checks": self._passed_checks, "failed_checks": self._failed_checks, "pass_rate": pass_rate}


class ErrorHandler:
    """错误处理器主类"""

    def __init__(self, chip_id: str):
        self._chip_id = chip_id

        # 核心组件
        self._error_detector = ErrorDetector(chip_id)
        self._retransmission_manager = RetransmissionManager(chip_id)
        self._integrity_checker = IntegrityChecker(chip_id)

        # 错误记录
        self._error_history: deque = deque(maxlen=10000)
        self._error_callbacks: Dict[ErrorType, List[Callable[[ErrorRecord], None]]] = defaultdict(list)

        # 恢复策略配置
        self._recovery_strategies: Dict[ErrorType, RecoveryAction] = {
            ErrorType.TRANSMISSION_ERROR: RecoveryAction.RETRANSMIT,
            ErrorType.TIMEOUT_ERROR: RecoveryAction.RETRANSMIT,
            ErrorType.CHECKSUM_ERROR: RecoveryAction.RETRANSMIT,
            ErrorType.SEQUENCE_ERROR: RecoveryAction.RETRY,
            ErrorType.BUFFER_OVERFLOW: RecoveryAction.FALLBACK,
            ErrorType.PROTOCOL_ERROR: RecoveryAction.RESET_CONNECTION,
            ErrorType.MEMORY_ERROR: RecoveryAction.ABORT,
            ErrorType.NETWORK_ERROR: RecoveryAction.RETRY,
        }

        # 错误限制
        self._max_error_rate = 0.1  # 最大错误率10%
        self._max_consecutive_errors = 5  # 最大连续错误数
        self._consecutive_error_count = 0

        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"ErrorHandler-{chip_id}")

        self._logger.info(f"错误处理器初始化完成: {chip_id}")

    def register_error_callback(self, error_type: ErrorType, callback: Callable[[ErrorRecord], None]):
        """注册错误回调"""
        self._error_callbacks[error_type].append(callback)

    def handle_error(self, error_record: ErrorRecord) -> RecoveryAction:
        """
        处理错误

        Args:
            error_record: 错误记录

        Returns:
            RecoveryAction: 恢复动作
        """
        with self._lock:
            # 添加到历史记录
            self._error_history.append(error_record)

            # 确定恢复策略
            recovery_action = self._recovery_strategies.get(error_record.error_type, RecoveryAction.IGNORE)
            error_record.recovery_action = recovery_action

            # 更新连续错误计数
            if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self._consecutive_error_count += 1
            else:
                self._consecutive_error_count = 0

            # 检查是否需要紧急措施
            if self._consecutive_error_count >= self._max_consecutive_errors:
                self._logger.error(f"连续严重错误达到阈值，执行紧急恢复: {self._consecutive_error_count}")
                recovery_action = RecoveryAction.ABORT
                error_record.recovery_action = recovery_action
                self._consecutive_error_count = 0

            # 执行恢复动作
            success = self._execute_recovery_action(error_record, recovery_action)

            if success:
                error_record.resolved = True
                error_record.resolved_timestamp = time.time()

            # 调用错误回调
            for callback in self._error_callbacks[error_record.error_type]:
                try:
                    callback(error_record)
                except Exception as e:
                    self._logger.error(f"错误回调执行失败: {e}")

            self._logger.info(f"处理错误: {error_record.error_type.value} -> {recovery_action.value}")
            return recovery_action

    def _execute_recovery_action(self, error_record: ErrorRecord, action: RecoveryAction) -> bool:
        """执行恢复动作"""
        try:
            if action == RecoveryAction.RETRANSMIT:
                return self._handle_retransmission(error_record)
            elif action == RecoveryAction.RETRY:
                return self._handle_retry(error_record)
            elif action == RecoveryAction.RESET_CONNECTION:
                return self._handle_connection_reset(error_record)
            elif action == RecoveryAction.FALLBACK:
                return self._handle_fallback(error_record)
            elif action == RecoveryAction.IGNORE:
                return True
            elif action == RecoveryAction.ABORT:
                return self._handle_abort(error_record)
            else:
                return False
        except Exception as e:
            self._logger.error(f"恢复动作执行失败: {action.value}, 错误: {e}")
            return False

    def _handle_retransmission(self, error_record: ErrorRecord) -> bool:
        """处理重传"""
        if error_record.sequence_number is not None and "data" in error_record.context and "destination" in error_record.context:

            self._retransmission_manager.schedule_retransmission(
                sequence_number=error_record.sequence_number, data=error_record.context["data"], destination=error_record.context["destination"], packet_id=error_record.packet_id or ""
            )
            return True
        return False

    def _handle_retry(self, error_record: ErrorRecord) -> bool:
        """处理重试"""
        # 简单的重试标记，具体实现由上层决定
        error_record.recovery_attempts += 1
        return error_record.recovery_attempts <= 3

    def _handle_connection_reset(self, error_record: ErrorRecord) -> bool:
        """处理连接重置"""
        # 这里可以触发连接重置逻辑
        self._logger.warning(f"触发连接重置: {error_record.error_id}")
        return True

    def _handle_fallback(self, error_record: ErrorRecord) -> bool:
        """处理降级"""
        # 这里可以实现降级逻辑
        self._logger.info(f"触发降级处理: {error_record.error_id}")
        return True

    def _handle_abort(self, error_record: ErrorRecord) -> bool:
        """处理中止"""
        self._logger.error(f"触发操作中止: {error_record.error_id}")
        return False

    def process_packet_sent(self, packet_id: str, sequence_number: int, destination: str, data: bytes):
        """处理包发送"""
        self._error_detector.register_packet_sent(packet_id, sequence_number, destination)

    def process_packet_received(self, packet_id: str, sequence_number: int, source: str, data: bytes, metadata: Dict[str, Any] = None) -> List[ErrorRecord]:
        """
        处理包接收

        Args:
            packet_id: 包ID
            sequence_number: 序列号
            source: 源ID
            data: 数据
            metadata: 元数据

        Returns:
            List[ErrorRecord]: 检测到的错误列表
        """
        # 完整性检查
        integrity_valid = True
        integrity_errors = []

        if metadata:
            integrity_valid, integrity_errors = self._integrity_checker.verify_data_integrity(data, metadata)

        # 错误检测
        errors = self._error_detector.register_packet_received(packet_id, sequence_number, source, integrity_valid)

        # 添加完整性错误
        if not integrity_valid:
            for error_msg in integrity_errors:
                error_record = ErrorRecord(
                    error_id=f"integrity_{packet_id}",
                    error_type=ErrorType.CHECKSUM_ERROR,
                    severity=ErrorSeverity.MEDIUM,
                    message=error_msg,
                    timestamp=time.time(),
                    source_id=self._chip_id,
                    packet_id=packet_id,
                    sequence_number=sequence_number,
                    context={"source": source},
                )
                errors.append(error_record)

        # 处理检测到的错误
        for error in errors:
            self.handle_error(error)

        return errors

    def process_periodic_checks(self) -> List[ErrorRecord]:
        """处理周期性检查"""
        errors = self._error_detector.check_periodic_errors()

        for error in errors:
            self.handle_error(error)

        # 清理过期的重传请求
        self._retransmission_manager.cleanup_expired_requests()

        return errors

    def get_ready_retransmissions(self) -> List[RetransmissionRequest]:
        """获取准备重传的请求"""
        return self._retransmission_manager.get_ready_retransmissions()

    def mark_retransmission_sent(self, request_id: str) -> bool:
        """标记重传已发送"""
        return self._retransmission_manager.mark_retransmission_sent(request_id)

    def mark_retransmission_successful(self, sequence_number: int) -> bool:
        """标记重传成功"""
        return self._retransmission_manager.mark_retransmission_successful(sequence_number)

    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        with self._lock:
            # 按错误类型统计
            error_type_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            resolved_count = 0

            for error in self._error_history:
                error_type_counts[error.error_type.value] += 1
                severity_counts[error.severity.value] += 1
                if error.resolved:
                    resolved_count += 1

            return {
                "chip_id": self._chip_id,
                "total_errors": len(self._error_history),
                "resolved_errors": resolved_count,
                "consecutive_errors": self._consecutive_error_count,
                "error_type_counts": dict(error_type_counts),
                "severity_counts": dict(severity_counts),
                "detection_stats": self._error_detector.get_detection_stats(),
                "retransmission_stats": self._retransmission_manager.get_retransmission_stats(),
                "integrity_stats": self._integrity_checker.get_integrity_stats(),
            }

    def get_recent_errors(self, hours: float = 1.0) -> List[Dict[str, Any]]:
        """获取最近的错误"""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - (hours * 3600)

            recent_errors = [
                {
                    "error_id": error.error_id,
                    "error_type": error.error_type.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "timestamp": error.timestamp,
                    "age_seconds": error.age(),
                    "resolved": error.resolved,
                    "recovery_action": error.recovery_action.value if error.recovery_action else None,
                    "recovery_attempts": error.recovery_attempts,
                }
                for error in self._error_history
                if error.timestamp >= cutoff_time
            ]

            return recent_errors

    def reset_statistics(self):
        """重置统计信息"""
        with self._lock:
            self._error_history.clear()
            self._consecutive_error_count = 0

            # 重新创建组件以清空统计
            self._error_detector = ErrorDetector(self._chip_id)
            self._retransmission_manager = RetransmissionManager(self._chip_id)
            self._integrity_checker = IntegrityChecker(self._chip_id)

            self._logger.info("错误处理统计信息已重置")

    def shutdown(self):
        """关闭错误处理器"""
        with self._lock:
            self._logger.info(f"错误处理器已关闭: {self._chip_id}")

    @property
    def chip_id(self) -> str:
        return self._chip_id
