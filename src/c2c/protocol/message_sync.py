"""
消息同步机制模块
实现完整的MSG sync状态机，支持Engine间的同步机制
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import defaultdict, deque
import logging

from src.c2c.utils.exceptions import CDMAError


class SyncState(Enum):
    """同步状态枚举"""

    IDLE = "idle"  # 空闲状态，没有同步操作
    WAITING = "waiting"  # 等待状态，已发送同步消息等待响应
    SYNCED = "synced"  # 同步完成状态
    TIMEOUT = "timeout"  # 超时状态
    ERROR = "error"  # 错误状态


class SyncMessageType(Enum):
    """同步消息类型"""

    TX_SEND = "tx_send"  # 发送方发起同步
    TX_WAIT = "tx_wait"  # 发送方等待确认
    RX_SEND = "rx_send"  # 接收方发起同步
    RX_WAIT = "rx_wait"  # 接收方等待确认
    ACK = "ack"  # 同步确认
    NACK = "nack"  # 同步拒绝
    COMPLETE = "complete"  # 传输完成通知


@dataclass
class SyncMessage:
    """同步消息定义"""

    message_id: str
    sender_chip_id: str
    receiver_chip_id: str
    message_type: SyncMessageType
    timestamp: float
    transaction_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    sequence_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "message_id": self.message_id,
            "sender_chip_id": self.sender_chip_id,
            "receiver_chip_id": self.receiver_chip_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "transaction_id": self.transaction_id,
            "payload": self.payload,
            "sequence_number": self.sequence_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncMessage":
        """从字典创建同步消息"""
        return cls(
            message_id=data["message_id"],
            sender_chip_id=data["sender_chip_id"],
            receiver_chip_id=data["receiver_chip_id"],
            message_type=SyncMessageType(data["message_type"]),
            timestamp=data["timestamp"],
            transaction_id=data.get("transaction_id"),
            payload=data.get("payload", {}),
            sequence_number=data.get("sequence_number", 0),
        )


@dataclass
class SyncSession:
    """同步会话"""

    session_id: str
    sender_chip_id: str
    receiver_chip_id: str
    state: SyncState
    created_time: float
    last_activity_time: float
    timeout_seconds: float = 5.0

    # 消息历史
    sent_messages: List[SyncMessage] = field(default_factory=list)
    received_messages: List[SyncMessage] = field(default_factory=list)

    # 重试机制
    retry_count: int = 0
    max_retries: int = 3

    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return (time.time() - self.last_activity_time) > self.timeout_seconds

    def update_activity(self):
        """更新活动时间"""
        self.last_activity_time = time.time()


class MessageSyncManager:
    """消息同步管理器"""

    def __init__(self, chip_id: str):
        self._chip_id = chip_id
        self._lock = threading.RLock()

        # 同步会话管理
        self._sync_sessions: Dict[str, SyncSession] = {}
        self._chip_sessions: Dict[str, List[str]] = defaultdict(list)  # chip_id -> session_ids

        # 消息队列
        self._outbound_queue: deque = deque()
        self._inbound_queue: deque = deque()

        # 回调函数
        self._message_handlers: Dict[SyncMessageType, Callable] = {}

        # 统计信息
        self._total_messages_sent = 0
        self._total_messages_received = 0
        self._successful_syncs = 0
        self._failed_syncs = 0
        self._timeout_syncs = 0

        # 序列号生成器
        self._sequence_counter = 0

        # 日志记录器
        self._logger = logging.getLogger(f"MessageSync-{chip_id}")

        self._logger.info(f"消息同步管理器初始化完成: {chip_id}")

    def register_message_handler(self, message_type: SyncMessageType, handler: Callable[[SyncMessage], None]):
        """注册消息处理器"""
        with self._lock:
            self._message_handlers[message_type] = handler
            self._logger.debug(f"注册消息处理器: {message_type.value}")

    def create_sync_session(self, target_chip_id: str, timeout_seconds: float = 5.0) -> str:
        """
        创建同步会话

        Args:
            target_chip_id: 目标芯片ID
            timeout_seconds: 超时时间（秒）

        Returns:
            str: 会话ID
        """
        with self._lock:
            session_id = f"sync_{self._chip_id}_{target_chip_id}_{int(time.time() * 1000000)}"

            session = SyncSession(
                session_id=session_id,
                sender_chip_id=self._chip_id,
                receiver_chip_id=target_chip_id,
                state=SyncState.IDLE,
                created_time=time.time(),
                last_activity_time=time.time(),
                timeout_seconds=timeout_seconds,
            )

            self._sync_sessions[session_id] = session
            self._chip_sessions[target_chip_id].append(session_id)

            self._logger.info(f"创建同步会话: {session_id} -> {target_chip_id}")
            return session_id

    def send_sync_message(self, session_id: str, message_type: SyncMessageType, payload: Dict[str, Any] = None, transaction_id: str = None) -> bool:
        """
        发送同步消息

        Args:
            session_id: 会话ID
            message_type: 消息类型
            payload: 消息载荷
            transaction_id: 事务ID

        Returns:
            bool: 发送是否成功
        """
        with self._lock:
            if session_id not in self._sync_sessions:
                self._logger.error(f"会话不存在: {session_id}")
                return False

            session = self._sync_sessions[session_id]

            if session.is_expired():
                self._logger.warning(f"会话已过期: {session_id}")
                self._set_session_state(session_id, SyncState.TIMEOUT)
                return False

            # 生成消息ID和序列号
            self._sequence_counter += 1
            message_id = f"msg_{self._chip_id}_{self._sequence_counter}_{int(time.time() * 1000000)}"

            # 创建同步消息
            sync_message = SyncMessage(
                message_id=message_id,
                sender_chip_id=self._chip_id,
                receiver_chip_id=session.receiver_chip_id,
                message_type=message_type,
                timestamp=time.time(),
                transaction_id=transaction_id,
                payload=payload or {},
                sequence_number=self._sequence_counter,
            )

            # 加入发送队列
            self._outbound_queue.append(sync_message)
            session.sent_messages.append(sync_message)
            session.update_activity()

            # 更新会话状态
            if message_type in [SyncMessageType.TX_SEND, SyncMessageType.RX_SEND]:
                self._set_session_state(session_id, SyncState.WAITING)

            self._total_messages_sent += 1

            self._logger.debug(f"发送同步消息: {message_id} ({message_type.value}) -> {session.receiver_chip_id}")
            return True

    def receive_sync_message(self, message: SyncMessage) -> bool:
        """
        接收同步消息

        Args:
            message: 同步消息

        Returns:
            bool: 处理是否成功
        """
        with self._lock:
            # 加入接收队列
            self._inbound_queue.append(message)
            self._total_messages_received += 1

            # 查找或创建对应的会话
            session_id = self._find_or_create_session_for_message(message)
            if not session_id:
                self._logger.error(f"无法找到或创建会话用于消息: {message.message_id}")
                return False

            session = self._sync_sessions[session_id]
            session.received_messages.append(message)
            session.update_activity()

            # 处理消息
            success = self._process_received_message(session_id, message)

            # 调用注册的处理器
            if message.message_type in self._message_handlers:
                try:
                    self._message_handlers[message.message_type](message)
                except Exception as e:
                    self._logger.error(f"消息处理器执行失败: {e}")

            self._logger.debug(f"接收同步消息: {message.message_id} ({message.message_type.value}) <- {message.sender_chip_id}")
            return success

    def _find_or_create_session_for_message(self, message: SyncMessage) -> Optional[str]:
        """为接收到的消息找到或创建会话"""
        # 首先尝试在现有会话中查找
        for session_id, session in self._sync_sessions.items():
            if session.sender_chip_id == message.receiver_chip_id and session.receiver_chip_id == message.sender_chip_id:
                return session_id

        # 如果没有找到，创建新会话（作为接收方）
        session_id = f"sync_{message.receiver_chip_id}_{message.sender_chip_id}_{int(time.time() * 1000000)}"

        session = SyncSession(
            session_id=session_id, sender_chip_id=message.receiver_chip_id, receiver_chip_id=message.sender_chip_id, state=SyncState.IDLE, created_time=time.time(), last_activity_time=time.time()
        )

        self._sync_sessions[session_id] = session
        self._chip_sessions[message.sender_chip_id].append(session_id)

        return session_id

    def _process_received_message(self, session_id: str, message: SyncMessage) -> bool:
        """处理接收到的消息"""
        session = self._sync_sessions[session_id]

        # 根据消息类型处理
        if message.message_type == SyncMessageType.TX_SEND:
            # 接收到发送方的同步请求，回复ACK
            self._send_ack_message(session_id, message)
            return True

        elif message.message_type == SyncMessageType.RX_SEND:
            # 接收到接收方的同步请求，回复ACK
            self._send_ack_message(session_id, message)
            return True

        elif message.message_type == SyncMessageType.ACK:
            # 接收到确认消息，更新会话状态
            self._set_session_state(session_id, SyncState.SYNCED)
            self._successful_syncs += 1
            return True

        elif message.message_type == SyncMessageType.NACK:
            # 接收到拒绝消息，设置错误状态
            self._set_session_state(session_id, SyncState.ERROR)
            self._failed_syncs += 1
            return True

        elif message.message_type == SyncMessageType.COMPLETE:
            # 接收到完成通知，关闭会话
            self._set_session_state(session_id, SyncState.SYNCED)
            return True

        return False

    def _send_ack_message(self, session_id: str, original_message: SyncMessage):
        """发送ACK消息"""
        ack_payload = {"original_message_id": original_message.message_id, "ack_timestamp": time.time()}

        self.send_sync_message(session_id=session_id, message_type=SyncMessageType.ACK, payload=ack_payload, transaction_id=original_message.transaction_id)

    def _set_session_state(self, session_id: str, new_state: SyncState):
        """设置会话状态"""
        if session_id in self._sync_sessions:
            old_state = self._sync_sessions[session_id].state
            self._sync_sessions[session_id].state = new_state
            self._sync_sessions[session_id].update_activity()

            self._logger.debug(f"会话状态变更: {session_id} {old_state.value} -> {new_state.value}")

    def wait_for_sync(self, session_id: str, timeout_seconds: float = 5.0) -> bool:
        """
        等待同步完成

        Args:
            session_id: 会话ID
            timeout_seconds: 超时时间

        Returns:
            bool: 同步是否成功
        """
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            with self._lock:
                if session_id not in self._sync_sessions:
                    return False

                session = self._sync_sessions[session_id]

                if session.state == SyncState.SYNCED:
                    return True
                elif session.state in [SyncState.ERROR, SyncState.TIMEOUT]:
                    return False

            time.sleep(0.001)  # 1ms轮询间隔

        # 超时处理
        with self._lock:
            if session_id in self._sync_sessions:
                self._set_session_state(session_id, SyncState.TIMEOUT)
                self._timeout_syncs += 1

        return False

    def get_outbound_messages(self) -> List[SyncMessage]:
        """获取待发送的消息"""
        with self._lock:
            messages = list(self._outbound_queue)
            self._outbound_queue.clear()
            return messages

    def cleanup_expired_sessions(self):
        """清理过期的会话"""
        with self._lock:
            expired_sessions = []
            current_time = time.time()

            for session_id, session in self._sync_sessions.items():
                if session.is_expired():
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                session = self._sync_sessions[session_id]

                if session.state not in [SyncState.SYNCED, SyncState.ERROR, SyncState.TIMEOUT]:
                    self._set_session_state(session_id, SyncState.TIMEOUT)
                    self._timeout_syncs += 1

                # 从芯片会话列表中移除
                if session.receiver_chip_id in self._chip_sessions:
                    if session_id in self._chip_sessions[session.receiver_chip_id]:
                        self._chip_sessions[session.receiver_chip_id].remove(session_id)

                del self._sync_sessions[session_id]

                self._logger.debug(f"清理过期会话: {session_id}")

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话状态"""
        with self._lock:
            if session_id not in self._sync_sessions:
                return None

            session = self._sync_sessions[session_id]
            return {
                "session_id": session.session_id,
                "sender_chip_id": session.sender_chip_id,
                "receiver_chip_id": session.receiver_chip_id,
                "state": session.state.value,
                "created_time": session.created_time,
                "last_activity_time": session.last_activity_time,
                "is_expired": session.is_expired(),
                "sent_messages_count": len(session.sent_messages),
                "received_messages_count": len(session.received_messages),
                "retry_count": session.retry_count,
            }

    def get_sync_statistics(self) -> Dict[str, Any]:
        """获取同步统计信息"""
        with self._lock:
            active_sessions = len([s for s in self._sync_sessions.values() if s.state not in [SyncState.SYNCED, SyncState.ERROR, SyncState.TIMEOUT]])

            return {
                "chip_id": self._chip_id,
                "total_sessions": len(self._sync_sessions),
                "active_sessions": active_sessions,
                "total_messages_sent": self._total_messages_sent,
                "total_messages_received": self._total_messages_received,
                "successful_syncs": self._successful_syncs,
                "failed_syncs": self._failed_syncs,
                "timeout_syncs": self._timeout_syncs,
                "success_rate": self._successful_syncs / max(1, self._successful_syncs + self._failed_syncs + self._timeout_syncs) * 100,
            }

    def shutdown(self):
        """关闭同步管理器"""
        with self._lock:
            self._logger.info(f"关闭消息同步管理器: {self._chip_id}")

            # 清理所有会话
            for session_id in list(self._sync_sessions.keys()):
                self._set_session_state(session_id, SyncState.ERROR)

            # 清空队列
            self._outbound_queue.clear()
            self._inbound_queue.clear()

            # 清空会话
            self._sync_sessions.clear()
            self._chip_sessions.clear()

    @property
    def chip_id(self) -> str:
        return self._chip_id
