"""
C2C 协议模块 - 芯片间通信协议实现
包含CDMA协议、地址转换、路由和流控
"""

# 导入基础协议类
from .base import BaseProtocol, ProtocolState

# 导入CDMA系统
from .cdma_system import CDMASystem, CDMASystemState, CDMAMessage, CDMAOperationResult

# 导入地址转换
from .address import AddressTranslator, AddressFormat

# 导入信用管理
from .credit import CreditManager

# 导入路由
from .router import Router

# 导入包格式
from .packet_format import CDMAPacket, PacketFactory, PacketSerializer, PacketType, DataType

# 导入流控
from .flow_control import FlowController, FlowState

# 导入性能监控
from .performance_monitor import PerformanceMonitor

# 导入错误处理
from .error_handler import ErrorHandler, ErrorRecord, ErrorType, ErrorSeverity

# 导入消息同步
from .message_sync import MessageSyncManager, SyncMessage, SyncState, SyncMessageType

# 导入事务管理
from .transaction_manager import TransactionManager

# 导入内存类型
from .memory_types import MemoryType

__version__ = "1.0.0"

__all__ = [
    # 基础协议
    "BaseProtocol",
    "ProtocolState",
    # CDMA系统
    "CDMASystem",
    "CDMASystemState",
    "CDMAMessage",
    "CDMAOperationResult",
    # 地址转换
    "AddressTranslator",
    "AddressFormat",
    # 信用管理
    "CreditManager",
    # 路由
    "Router",
    # 包格式
    "CDMAPacket",
    "PacketFactory",
    "PacketSerializer",
    "PacketType",
    "DataType",
    # 流控
    "FlowController",
    "FlowState",
    # 性能监控
    "PerformanceMonitor",
    # 错误处理
    "ErrorHandler",
    "ErrorRecord",
    "ErrorType",
    "ErrorSeverity",
    # 消息同步
    "MessageSyncManager",
    "SyncMessage",
    "SyncState",
    "SyncMessageType",
    # 事务管理
    "TransactionManager",
    # 内存类型
    "MemoryType",
]
