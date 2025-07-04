from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import defaultdict
from src.c2c.utils.exceptions import CDMAError


class TransactionState(Enum):
    """事务状态枚举"""

    CREATED = "created"  # 事务已创建
    RECEIVE_POSTED = "receive_posted"  # 接收方已发布CDMA_receive
    SEND_EXECUTED = "send_executed"  # 发送方已执行CDMA_send
    TRANSFERRING = "transferring"  # 数据传输中
    COMPLETED = "completed"  # 传输完成
    FAILED = "failed"  # 传输失败
    TIMEOUT = "timeout"  # 事务超时


@dataclass
class TransactionInfo:
    """事务信息"""

    transaction_id: str
    send_chip_id: str
    recv_chip_id: str
    state: TransactionState
    created_time: float

    # 地址信息
    src_addr: Optional[int] = None
    dst_addr: Optional[int] = None
    src_shape: Optional[Tuple[int, ...]] = None
    dst_shape: Optional[Tuple[int, ...]] = None
    src_mem_type: Optional[str] = None
    dst_mem_type: Optional[str] = None
    data_type: Optional[str] = None

    # 时间戳
    receive_posted_time: Optional[float] = None
    send_executed_time: Optional[float] = None
    transfer_start_time: Optional[float] = None
    completed_time: Optional[float] = None

    # 结果信息
    success: bool = False
    error_message: Optional[str] = None
    bytes_transferred: int = 0

    # 超时设置
    timeout_seconds: float = 30.0


@dataclass
class PairedOperation:
    """配对操作记录"""

    send_chip_id: str
    recv_chip_id: str
    pending_receives: List[str] = field(default_factory=list)  # 待匹配的receive操作
    pending_sends: List[str] = field(default_factory=list)  # 待匹配的send操作


class TransactionManager:
    """事务管理器 - 负责管理配对的send/receive操作"""

    def __init__(self):
        self._transactions: Dict[str, TransactionInfo] = {}
        self._paired_operations: Dict[Tuple[str, str], PairedOperation] = {}  # (send_chip, recv_chip) -> operations
        self._chip_transactions: Dict[str, List[str]] = defaultdict(list)  # chip_id -> transaction_ids
        self._lock = threading.RLock()

        # 统计信息
        self._total_transactions = 0
        self._successful_transactions = 0
        self._failed_transactions = 0
        self._timeout_transactions = 0

    def create_paired_transaction(self, send_chip_id: str, recv_chip_id: str, transaction_id: str, timeout_seconds: float = 30.0) -> TransactionInfo:
        """
        创建配对传输事务

        Args:
            send_chip_id: 发送方芯片ID
            recv_chip_id: 接收方芯片ID
            transaction_id: 事务ID
            timeout_seconds: 超时时间（秒）

        Returns:
            TransactionInfo: 事务信息
        """
        with self._lock:
            if transaction_id in self._transactions:
                raise CDMAError(f"事务ID {transaction_id} 已存在")

            transaction = TransactionInfo(
                transaction_id=transaction_id, send_chip_id=send_chip_id, recv_chip_id=recv_chip_id, state=TransactionState.CREATED, created_time=time.time(), timeout_seconds=timeout_seconds
            )

            self._transactions[transaction_id] = transaction
            self._chip_transactions[send_chip_id].append(transaction_id)
            self._chip_transactions[recv_chip_id].append(transaction_id)
            self._total_transactions += 1

            # 初始化配对操作记录
            pair_key = (send_chip_id, recv_chip_id)
            if pair_key not in self._paired_operations:
                self._paired_operations[pair_key] = PairedOperation(send_chip_id, recv_chip_id)

            print(f"事务管理器：创建配对事务 {transaction_id}")
            print(f"  发送方: {send_chip_id}, 接收方: {recv_chip_id}")
            print(f"  超时时间: {timeout_seconds} 秒")

            return transaction

    def register_receive_operation(self, transaction_id: str, dst_addr: int, dst_shape: Tuple[int, ...], dst_mem_type: str, data_type: str = "float32") -> bool:
        """
        注册CDMA_receive操作

        Args:
            transaction_id: 事务ID
            dst_addr: 目标地址
            dst_shape: 目标tensor形状
            dst_mem_type: 目标内存类型
            data_type: 数据类型

        Returns:
            bool: 注册是否成功
        """
        with self._lock:
            if transaction_id not in self._transactions:
                print(f"事务管理器：事务 {transaction_id} 不存在")
                return False

            transaction = self._transactions[transaction_id]

            if transaction.state != TransactionState.CREATED:
                print(f"事务管理器：事务 {transaction_id} 状态不正确，当前状态: {transaction.state.value}")
                return False

            # 更新事务信息
            transaction.dst_addr = dst_addr
            transaction.dst_shape = dst_shape
            transaction.dst_mem_type = dst_mem_type
            transaction.data_type = data_type
            transaction.state = TransactionState.RECEIVE_POSTED
            transaction.receive_posted_time = time.time()

            # 添加到配对操作中
            pair_key = (transaction.send_chip_id, transaction.recv_chip_id)
            self._paired_operations[pair_key].pending_receives.append(transaction_id)

            print(f"事务管理器：注册CDMA_receive操作 {transaction_id}")
            print(f"  目标地址: 0x{dst_addr:08x}, 形状: {dst_shape}")
            print(f"  内存类型: {dst_mem_type}, 数据类型: {data_type}")

            return True

    def register_send_operation(self, transaction_id: str, src_addr: int, src_shape: Tuple[int, ...], src_mem_type: str, data_type: str = "float32") -> Optional[TransactionInfo]:
        """
        注册CDMA_send操作并检查是否可以配对

        Args:
            transaction_id: 事务ID
            src_addr: 源地址
            src_shape: 源tensor形状
            src_mem_type: 源内存类型
            data_type: 数据类型

        Returns:
            TransactionInfo: 如果配对成功返回事务信息，否则返回None
        """
        with self._lock:
            if transaction_id not in self._transactions:
                print(f"事务管理器：事务 {transaction_id} 不存在")
                return None

            transaction = self._transactions[transaction_id]

            if transaction.state != TransactionState.RECEIVE_POSTED:
                print(f"事务管理器：事务 {transaction_id} 尚未执行CDMA_receive")
                return None

            # 验证形状和数据类型兼容性
            if not self._validate_compatibility(transaction, src_shape, data_type):
                transaction.state = TransactionState.FAILED
                transaction.error_message = f"形状或数据类型不兼容: src_shape={src_shape}, dst_shape={transaction.dst_shape}"
                print(f"事务管理器：{transaction.error_message}")
                return None

            # 更新事务信息
            transaction.src_addr = src_addr
            transaction.src_shape = src_shape
            transaction.src_mem_type = src_mem_type
            transaction.state = TransactionState.SEND_EXECUTED
            transaction.send_executed_time = time.time()

            # 从配对操作中移除
            pair_key = (transaction.send_chip_id, transaction.recv_chip_id)
            if transaction_id in self._paired_operations[pair_key].pending_receives:
                self._paired_operations[pair_key].pending_receives.remove(transaction_id)

            print(f"事务管理器：注册CDMA_send操作 {transaction_id}")
            print(f"  源地址: 0x{src_addr:08x}, 形状: {src_shape}")
            print(f"  内存类型: {src_mem_type}, 数据类型: {data_type}")
            print(f"  配对成功，准备执行DMA传输")

            return transaction

    def start_transfer(self, transaction_id: str) -> bool:
        """
        开始DMA传输

        Args:
            transaction_id: 事务ID

        Returns:
            bool: 是否成功开始传输
        """
        with self._lock:
            if transaction_id not in self._transactions:
                return False

            transaction = self._transactions[transaction_id]

            if transaction.state != TransactionState.SEND_EXECUTED:
                return False

            transaction.state = TransactionState.TRANSFERRING
            transaction.transfer_start_time = time.time()

            print(f"事务管理器：开始DMA传输 {transaction_id}")
            return True

    def complete_transaction(self, transaction_id: str, success: bool, bytes_transferred: int = 0, error_message: str = None) -> bool:
        """
        完成事务

        Args:
            transaction_id: 事务ID
            success: 是否成功
            bytes_transferred: 传输字节数
            error_message: 错误消息

        Returns:
            bool: 是否成功完成
        """
        with self._lock:
            if transaction_id not in self._transactions:
                return False

            transaction = self._transactions[transaction_id]
            transaction.completed_time = time.time()
            transaction.success = success
            transaction.bytes_transferred = bytes_transferred
            transaction.error_message = error_message

            if success:
                transaction.state = TransactionState.COMPLETED
                self._successful_transactions += 1
                print(f"事务管理器：事务 {transaction_id} 完成成功")
                print(f"  传输字节数: {bytes_transferred}")
            else:
                transaction.state = TransactionState.FAILED
                self._failed_transactions += 1
                print(f"事务管理器：事务 {transaction_id} 完成失败")
                if error_message:
                    print(f"  错误信息: {error_message}")

            return True

    def sync_transaction_completion(self, transaction_id: str) -> bool:
        """
        同步事务完成状态

        Args:
            transaction_id: 事务ID

        Returns:
            bool: 同步是否成功
        """
        with self._lock:
            if transaction_id not in self._transactions:
                return False

            transaction = self._transactions[transaction_id]

            if transaction.state in [TransactionState.COMPLETED, TransactionState.FAILED]:
                print(f"事务管理器：同步事务完成状态 {transaction_id}")
                print(f"  最终状态: {transaction.state.value}")
                if transaction.success:
                    print(f"  传输时间: {(transaction.completed_time - transaction.transfer_start_time) * 1000:.2f} ms")
                return True

            return False

    def cleanup_expired_transactions(self):
        """清理过期事务"""
        with self._lock:
            current_time = time.time()
            expired_transactions = []

            for transaction_id, transaction in self._transactions.items():
                if (current_time - transaction.created_time) > transaction.timeout_seconds:
                    if transaction.state not in [TransactionState.COMPLETED, TransactionState.FAILED]:
                        expired_transactions.append(transaction_id)

            for transaction_id in expired_transactions:
                transaction = self._transactions[transaction_id]
                transaction.state = TransactionState.TIMEOUT
                transaction.completed_time = current_time
                transaction.error_message = "事务超时"
                self._timeout_transactions += 1

                print(f"事务管理器：事务 {transaction_id} 超时")

                # 从配对操作中清理
                pair_key = (transaction.send_chip_id, transaction.recv_chip_id)
                if pair_key in self._paired_operations:
                    paired_op = self._paired_operations[pair_key]
                    if transaction_id in paired_op.pending_receives:
                        paired_op.pending_receives.remove(transaction_id)
                    if transaction_id in paired_op.pending_sends:
                        paired_op.pending_sends.remove(transaction_id)

    def _validate_compatibility(self, transaction: TransactionInfo, src_shape: Tuple[int, ...], data_type: str) -> bool:
        """验证源和目标的兼容性"""
        # 检查形状是否匹配
        if src_shape != transaction.dst_shape:
            return False

        # 检查数据类型是否匹配
        if data_type != transaction.data_type:
            return False

        return True

    def get_transaction_info(self, transaction_id: str) -> Optional[TransactionInfo]:
        """获取事务信息"""
        with self._lock:
            return self._transactions.get(transaction_id)

    def get_chip_transactions(self, chip_id: str) -> List[TransactionInfo]:
        """获取芯片相关的所有事务"""
        with self._lock:
            transaction_ids = self._chip_transactions.get(chip_id, [])
            return [self._transactions[tid] for tid in transaction_ids if tid in self._transactions]

    def get_paired_operations_status(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """获取配对操作状态"""
        with self._lock:
            status = {}
            for pair_key, paired_op in self._paired_operations.items():
                status[pair_key] = {
                    "pending_receives": len(paired_op.pending_receives),
                    "pending_sends": len(paired_op.pending_sends),
                    "receive_transaction_ids": paired_op.pending_receives.copy(),
                    "send_transaction_ids": paired_op.pending_sends.copy(),
                }
            return status

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            active_transactions = len(
                [t for t in self._transactions.values() if t.state in [TransactionState.CREATED, TransactionState.RECEIVE_POSTED, TransactionState.SEND_EXECUTED, TransactionState.TRANSFERRING]]
            )

            return {
                "total_transactions": self._total_transactions,
                "successful_transactions": self._successful_transactions,
                "failed_transactions": self._failed_transactions,
                "timeout_transactions": self._timeout_transactions,
                "active_transactions": active_transactions,
                "success_rate": self._successful_transactions / max(1, self._total_transactions) * 100,
            }

    def clear_completed_transactions(self):
        """清理已完成的事务"""
        with self._lock:
            completed_transaction_ids = [tid for tid, transaction in self._transactions.items() if transaction.state in [TransactionState.COMPLETED, TransactionState.FAILED, TransactionState.TIMEOUT]]

            for tid in completed_transaction_ids:
                transaction = self._transactions[tid]

                # 从芯片事务列表中移除
                if tid in self._chip_transactions[transaction.send_chip_id]:
                    self._chip_transactions[transaction.send_chip_id].remove(tid)
                if tid in self._chip_transactions[transaction.recv_chip_id]:
                    self._chip_transactions[transaction.recv_chip_id].remove(tid)

                # 从事务字典中移除
                del self._transactions[tid]

            print(f"事务管理器：清理了 {len(completed_transaction_ids)} 个已完成的事务")
