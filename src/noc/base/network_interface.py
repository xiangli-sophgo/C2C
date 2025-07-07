"""
NoC网络接口实现。

本模块实现了连接IP和NoC网络的网络接口(NI)，提供：
- 协议转换（IP协议 ↔ NoC协议）
- 时钟域转换
- 流量整形和QoS支持
- 缓存管理
- 错误处理和重传机制
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import deque, defaultdict
from enum import Enum
import logging

from .node import BaseNoCNode, NodeState, BufferStatus
from .flit import BaseFlit
from src.noc.utils.types import NodeId, Position, Priority, TrafficPattern


class ProtocolType(Enum):
    """协议类型"""

    MEMORY = "memory"  # 内存协议
    CACHE = "cache"  # 缓存协议
    DMA = "dma"  # DMA协议
    CUSTOM = "custom"  # 自定义协议


class QoSClass(Enum):
    """QoS服务类别"""

    BEST_EFFORT = "best_effort"
    GUARANTEED = "guaranteed"
    REAL_TIME = "real_time"
    BULK = "bulk"


class TransactionState(Enum):
    """事务状态"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class NetworkInterface(BaseNoCNode):
    """
    网络接口节点实现。

    NI是IP和NoC网络之间的桥梁，提供：
    1. 协议转换和适配
    2. 时钟域转换
    3. 流量整形和QoS
    4. 缓存管理
    5. 错误处理和重传
    """

    def __init__(self, node_id: NodeId, position: Position, protocol_type: ProtocolType = ProtocolType.MEMORY, **kwargs):
        """
        初始化网络接口。

        Args:
            node_id: 节点ID
            position: 节点位置
            protocol_type: 协议类型
            **kwargs: 其他配置参数
        """
        from src.noc.utils.types import NodeType

        super().__init__(node_id, position, NodeType.NETWORK_INTERFACE)
        self.current_cycle = 0

        self.protocol_type = protocol_type

        # 时钟域配置
        self.ip_clock_freq = kwargs.get("ip_clock_freq", 1.0)  # GHz
        self.network_clock_freq = kwargs.get("network_clock_freq", 2.0)  # GHz
        self.clock_ratio = int(self.network_clock_freq / self.ip_clock_freq)

        # 协议转换配置
        self.packet_size = kwargs.get("packet_size", 64)  # bytes
        self.max_payload_size = kwargs.get("max_payload_size", 256)  # bytes
        self.header_size = kwargs.get("header_size", 8)  # bytes

        # QoS配置
        self.qos_enabled = kwargs.get("qos_enabled", False)
        self.qos_classes = {
            QoSClass.REAL_TIME: {"priority": Priority.CRITICAL, "weight": 0.4},
            QoSClass.GUARANTEED: {"priority": Priority.HIGH, "weight": 0.3},
            QoSClass.BEST_EFFORT: {"priority": Priority.MEDIUM, "weight": 0.2},
            QoSClass.BULK: {"priority": Priority.LOW, "weight": 0.1},
        }

        # 缓存管理
        self.ip_to_noc_buffer = deque(maxlen=32)
        self.noc_to_ip_buffer = deque(maxlen=32)
        self.clock_crossing_buffer = deque(maxlen=16)

        # 事务管理
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        self.transaction_timeout = kwargs.get("transaction_timeout", 1000)  # cycles

        # 流量整形
        self.traffic_shaper = {
            "rate_limit": kwargs.get("rate_limit", 1000),  # MB/s
            "burst_size": kwargs.get("burst_size", 64),  # packets
            "token_bucket": kwargs.get("burst_size", 64),  # 令牌桶
            "last_update": 0,
        }

        # 重传机制
        self.retry_config = {"max_retries": kwargs.get("max_retries", 3), "retry_delay": kwargs.get("retry_delay", 10), "backoff_multiplier": kwargs.get("backoff_multiplier", 2.0)}  # cycles

        # 统计信息
        self.ni_stats = {
            "ip_to_noc_packets": 0,
            "noc_to_ip_packets": 0,
            "protocol_conversions": 0,
            "clock_domain_crossings": 0,
            "qos_violations": 0,
            "retransmissions": 0,
            "buffer_overflows": 0,
            "latency_stats": {"ip_to_noc": [], "noc_to_ip": [], "protocol_conversion": []},
        }

        # 协议处理器
        self.protocol_handlers = {
            ProtocolType.MEMORY: self._handle_memory_protocol,
            ProtocolType.CACHE: self._handle_cache_protocol,
            ProtocolType.DMA: self._handle_dma_protocol,
            ProtocolType.CUSTOM: self._handle_custom_protocol,
        }

        # 初始化组件
        self._initialize_ni_components()

        # 设置日志
        self.logger = logging.getLogger(f"NetworkInterface_{node_id}")

    def _initialize_ni_components(self) -> None:
        """初始化网络接口组件"""
        # 初始化QoS队列
        if self.qos_enabled:
            self.qos_queues = {qos_class: deque(maxlen=16) for qos_class in QoSClass}
        else:
            self.qos_queues = {QoSClass.BEST_EFFORT: deque(maxlen=64)}

        # 初始化时钟域转换
        self._init_clock_domain_conversion()

        # 初始化流量整形器
        self._init_traffic_shaper()

    def _init_clock_domain_conversion(self) -> None:
        """初始化时钟域转换"""
        # 时钟域转换缓存
        self.clock_buffers = {"ip_to_noc": deque(maxlen=self.clock_ratio * 4), "noc_to_ip": deque(maxlen=self.clock_ratio * 4)}

        # 时钟域同步信号
        self.clock_sync = {"ip_cycle": 0, "noc_cycle": 0, "sync_ready": False}

    def _init_traffic_shaper(self) -> None:
        """初始化流量整形器"""
        # 令牌桶算法初始化
        self.traffic_shaper["token_bucket"] = self.traffic_shaper["burst_size"]
        self.traffic_shaper["last_update"] = 0

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
            # 记录接收时间
            flit.ni_receive_time = self.current_cycle

            # 根据来源处理flit
            if input_port == "ip":
                return self._process_ip_flit(flit)
            elif input_port == "network":
                return self._process_network_flit(flit)
            else:
                self.logger.error(f"未知的输入端口: {input_port}")
                return False

        except Exception as e:
            self.logger.error(f"处理flit时发生错误: {e}")
            return False

    def _process_ip_flit(self, flit: BaseFlit) -> bool:
        """处理来自IP的flit"""
        # 协议转换
        if not self._convert_ip_to_noc_protocol(flit):
            return False

        # 应用QoS策略
        if not self._apply_qos_policy(flit):
            return False

        # 流量整形
        if not self._apply_traffic_shaping(flit):
            return False

        # 放入NoC缓存
        if len(self.ip_to_noc_buffer) >= self.ip_to_noc_buffer.maxlen:
            self.ni_stats["buffer_overflows"] += 1
            return False

        self.ip_to_noc_buffer.append(flit)
        self.ni_stats["ip_to_noc_packets"] += 1

        return True

    def _process_network_flit(self, flit: BaseFlit) -> bool:
        """处理来自网络的flit"""
        # 协议转换
        if not self._convert_noc_to_ip_protocol(flit):
            return False

        # 更新事务状态
        self._update_transaction_state(flit)

        # 放入IP缓存
        if len(self.noc_to_ip_buffer) >= self.noc_to_ip_buffer.maxlen:
            self.ni_stats["buffer_overflows"] += 1
            return False

        self.noc_to_ip_buffer.append(flit)
        self.ni_stats["noc_to_ip_packets"] += 1

        return True

    def _convert_ip_to_noc_protocol(self, flit: BaseFlit) -> bool:
        """IP协议到NoC协议转换"""
        start_time = self.current_cycle

        # 根据协议类型选择处理器
        handler = self.protocol_handlers.get(self.protocol_type)
        if not handler:
            self.logger.error(f"不支持的协议类型: {self.protocol_type}")
            return False

        # 执行协议转换
        success = handler(flit, "ip_to_noc")

        if success:
            # 更新统计
            self.ni_stats["protocol_conversions"] += 1
            conversion_time = self.current_cycle - start_time
            self.ni_stats["latency_stats"]["protocol_conversion"].append(conversion_time)

            # 设置NoC协议字段
            flit.protocol_version = "noc_v1.0"
            flit.ni_conversion_time = conversion_time

        return success

    def _convert_noc_to_ip_protocol(self, flit: BaseFlit) -> bool:
        """NoC协议到IP协议转换"""
        start_time = self.current_cycle

        # 根据协议类型选择处理器
        handler = self.protocol_handlers.get(self.protocol_type)
        if not handler:
            self.logger.error(f"不支持的协议类型: {self.protocol_type}")
            return False

        # 执行协议转换
        success = handler(flit, "noc_to_ip")

        if success:
            # 更新统计
            self.ni_stats["protocol_conversions"] += 1
            conversion_time = self.current_cycle - start_time
            self.ni_stats["latency_stats"]["protocol_conversion"].append(conversion_time)

            # 设置IP协议字段
            flit.ip_protocol_version = "ip_v1.0"
            flit.ni_conversion_time = conversion_time

        return success

    def _apply_qos_policy(self, flit: BaseFlit) -> bool:
        """应用QoS策略"""
        if not self.qos_enabled:
            flit.qos_class = QoSClass.BEST_EFFORT
            return True

        # 根据flit属性确定QoS类别
        qos_class = self._determine_qos_class(flit)
        flit.qos_class = qos_class

        # 检查QoS约束
        if not self._check_qos_constraints(flit):
            self.ni_stats["qos_violations"] += 1
            return False

        # 设置优先级
        qos_config = self.qos_classes[qos_class]
        flit.priority = qos_config["priority"]

        return True

    def _determine_qos_class(self, flit: BaseFlit) -> QoSClass:
        """确定flit的QoS类别"""
        # 根据flit的属性确定QoS类别
        if hasattr(flit, "deadline") and flit.deadline is not None:
            return QoSClass.REAL_TIME
        elif hasattr(flit, "guaranteed_bandwidth") and flit.guaranteed_bandwidth:
            return QoSClass.GUARANTEED
        elif hasattr(flit, "bulk_transfer") and flit.bulk_transfer:
            return QoSClass.BULK
        else:
            return QoSClass.BEST_EFFORT

    def _check_qos_constraints(self, flit: BaseFlit) -> bool:
        """检查QoS约束"""
        qos_class = flit.qos_class

        if qos_class == QoSClass.REAL_TIME:
            # 检查截止时间
            if hasattr(flit, "deadline"):
                remaining_time = flit.deadline - self.current_cycle
                if remaining_time <= 0:
                    return False

        elif qos_class == QoSClass.GUARANTEED:
            # 检查带宽保证
            if hasattr(flit, "guaranteed_bandwidth"):
                # 这里需要实现带宽分配算法
                pass

        return True

    def _apply_traffic_shaping(self, flit: BaseFlit) -> bool:
        """应用流量整形"""
        # 更新令牌桶
        self._update_token_bucket()

        # 检查是否有足够的令牌
        if self.traffic_shaper["token_bucket"] <= 0:
            # 没有令牌，需要等待
            return False

        # 消耗令牌
        self.traffic_shaper["token_bucket"] -= 1

        return True

    def _update_token_bucket(self) -> None:
        """更新令牌桶"""
        current_time = self.current_cycle
        last_update = self.traffic_shaper["last_update"]

        if current_time > last_update:
            # 计算需要添加的令牌数
            time_diff = current_time - last_update
            rate_limit = self.traffic_shaper["rate_limit"]

            # 简化的令牌添加逻辑
            tokens_to_add = min(time_diff, self.traffic_shaper["burst_size"])

            self.traffic_shaper["token_bucket"] = min(self.traffic_shaper["token_bucket"] + tokens_to_add, self.traffic_shaper["burst_size"])

            self.traffic_shaper["last_update"] = current_time

    def _handle_memory_protocol(self, flit: BaseFlit, direction: str) -> bool:
        """处理内存协议"""
        if direction == "ip_to_noc":
            # IP内存请求到NoC格式
            if hasattr(flit, "memory_address"):
                flit.noc_address = self._translate_memory_address(flit.memory_address)
            if hasattr(flit, "memory_op"):
                flit.noc_op_type = self._translate_memory_op(flit.memory_op)
        else:
            # NoC格式到IP内存响应
            if hasattr(flit, "noc_address"):
                flit.memory_address = self._translate_noc_address(flit.noc_address)
            if hasattr(flit, "noc_op_type"):
                flit.memory_op = self._translate_noc_op(flit.noc_op_type)

        return True

    def _handle_cache_protocol(self, flit: BaseFlit, direction: str) -> bool:
        """处理缓存协议"""
        if direction == "ip_to_noc":
            # 缓存一致性协议转换
            if hasattr(flit, "cache_state"):
                flit.noc_cache_state = self._translate_cache_state(flit.cache_state)
        else:
            # NoC缓存协议到IP
            if hasattr(flit, "noc_cache_state"):
                flit.cache_state = self._translate_noc_cache_state(flit.noc_cache_state)

        return True

    def _handle_dma_protocol(self, flit: BaseFlit, direction: str) -> bool:
        """处理DMA协议"""
        if direction == "ip_to_noc":
            # DMA传输请求转换
            if hasattr(flit, "dma_length"):
                flit.noc_burst_length = flit.dma_length
            if hasattr(flit, "dma_stride"):
                flit.noc_stride = flit.dma_stride
        else:
            # NoC DMA响应到IP
            if hasattr(flit, "noc_burst_length"):
                flit.dma_length = flit.noc_burst_length

        return True

    def _handle_custom_protocol(self, flit: BaseFlit, direction: str) -> bool:
        """处理自定义协议"""
        # 用户可以在这里实现自定义协议转换
        return True

    def _translate_memory_address(self, address: int) -> int:
        """转换内存地址"""
        # 简单的地址转换，实际实现可能更复杂
        return address

    def _translate_memory_op(self, op: str) -> str:
        """转换内存操作"""
        op_map = {"READ": "noc_read", "WRITE": "noc_write", "READX": "noc_readx"}
        return op_map.get(op, "noc_unknown")

    def _translate_noc_address(self, address: int) -> int:
        """转换NoC地址"""
        return address

    def _translate_noc_op(self, op: str) -> str:
        """转换NoC操作"""
        op_map = {"noc_read": "READ", "noc_write": "WRITE", "noc_readx": "READX"}
        return op_map.get(op, "UNKNOWN")

    def _translate_cache_state(self, state: str) -> str:
        """转换缓存状态"""
        return state

    def _translate_noc_cache_state(self, state: str) -> str:
        """转换NoC缓存状态"""
        return state

    def _update_transaction_state(self, flit: BaseFlit) -> None:
        """更新事务状态"""
        if hasattr(flit, "transaction_id"):
            trans_id = flit.transaction_id
            if trans_id in self.active_transactions:
                self.active_transactions[trans_id]["state"] = TransactionState.COMPLETED
                self.active_transactions[trans_id]["completion_time"] = self.current_cycle

    def route_flit(self, flit: BaseFlit) -> Optional[str]:
        """
        为flit进行路由决策。

        Args:
            flit: 要路由的flit对象

        Returns:
            输出端口名称
        """
        # 网络接口通常只有两个端口：IP侧和网络侧
        if flit.source == self.node_id:
            return "network"  # 发送到网络
        else:
            return "ip"  # 发送到IP

    def can_accept_flit(self, input_port: str, priority: Priority = Priority.MEDIUM) -> bool:
        """
        检查是否可以接收新的flit。

        Args:
            input_port: 输入端口名称
            priority: flit优先级

        Returns:
            是否可以接收
        """
        # 检查相应缓存是否有空间
        if input_port == "ip":
            return len(self.ip_to_noc_buffer) < self.ip_to_noc_buffer.maxlen
        elif input_port == "network":
            return len(self.noc_to_ip_buffer) < self.noc_to_ip_buffer.maxlen
        else:
            return False

    def step_ni(self, cycle: int) -> None:
        """
        执行网络接口的一个周期操作。

        Args:
            cycle: 当前周期
        """
        self.current_cycle = cycle

        # 处理时钟域转换
        self._process_clock_domain_conversion()

        # 处理QoS队列
        if self.qos_enabled:
            self._process_qos_queues()

        # 处理重传
        self._process_retransmissions()

        # 更新事务超时
        self._check_transaction_timeouts()

        # 更新统计
        self._update_ni_statistics()

    def _process_clock_domain_conversion(self) -> None:
        """处理时钟域转换"""
        # 更新时钟计数
        self.clock_sync["noc_cycle"] += 1

        if self.clock_sync["noc_cycle"] % self.clock_ratio == 0:
            self.clock_sync["ip_cycle"] += 1
            self.clock_sync["sync_ready"] = True

            # 在IP时钟边沿处理时钟域转换
            self._transfer_clock_domain_data()
        else:
            self.clock_sync["sync_ready"] = False

    def _transfer_clock_domain_data(self) -> None:
        """在时钟域之间传输数据"""
        # IP到NoC的数据传输
        if self.ip_to_noc_buffer:
            flit = self.ip_to_noc_buffer.popleft()
            self.clock_buffers["ip_to_noc"].append(flit)
            self.ni_stats["clock_domain_crossings"] += 1

        # NoC到IP的数据传输
        if self.noc_to_ip_buffer:
            flit = self.noc_to_ip_buffer.popleft()
            self.clock_buffers["noc_to_ip"].append(flit)
            self.ni_stats["clock_domain_crossings"] += 1

    def _process_qos_queues(self) -> None:
        """处理QoS队列"""
        # 按优先级处理队列
        qos_order = [QoSClass.REAL_TIME, QoSClass.GUARANTEED, QoSClass.BEST_EFFORT, QoSClass.BULK]

        for qos_class in qos_order:
            if qos_class in self.qos_queues and self.qos_queues[qos_class]:
                flit = self.qos_queues[qos_class].popleft()
                # 处理高优先级flit
                self._process_high_priority_flit(flit)
                break  # 每周期只处理一个flit

    def _process_high_priority_flit(self, flit: BaseFlit) -> None:
        """处理高优先级flit"""
        # 高优先级flit可以抢占资源
        if flit.priority in [Priority.HIGH, Priority.CRITICAL]:
            # 直接放入输出缓存
            self.ip_to_noc_buffer.append(flit)

    def _process_retransmissions(self) -> None:
        """处理重传"""
        current_time = self.current_cycle

        for trans_id, trans_info in list(self.active_transactions.items()):
            if trans_info["state"] == TransactionState.FAILED:
                # 检查是否需要重传
                if trans_info["retry_count"] < self.retry_config["max_retries"]:
                    retry_time = trans_info["last_retry_time"] + self.retry_config["retry_delay"]

                    if current_time >= retry_time:
                        self._retry_transaction(trans_id)

    def _retry_transaction(self, trans_id: str) -> None:
        """重传事务"""
        trans_info = self.active_transactions[trans_id]
        trans_info["retry_count"] += 1
        trans_info["last_retry_time"] = self.current_cycle
        trans_info["state"] = TransactionState.RETRYING

        # 增加退避时间
        backoff = self.retry_config["backoff_multiplier"] ** trans_info["retry_count"]
        trans_info["retry_delay"] = int(self.retry_config["retry_delay"] * backoff)

        self.ni_stats["retransmissions"] += 1

    def _check_transaction_timeouts(self) -> None:
        """检查事务超时"""
        current_time = self.current_cycle

        for trans_id, trans_info in list(self.active_transactions.items()):
            start_time = trans_info["start_time"]
            if current_time - start_time > self.transaction_timeout:
                # 事务超时
                trans_info["state"] = TransactionState.FAILED
                self.logger.warning(f"事务超时: {trans_id}")

    def _update_ni_statistics(self) -> None:
        """更新网络接口统计"""
        # 计算平均延迟
        for metric in self.ni_stats["latency_stats"]:
            if self.ni_stats["latency_stats"][metric]:
                avg_latency = sum(self.ni_stats["latency_stats"][metric]) / len(self.ni_stats["latency_stats"][metric])
                self.ni_stats[f"avg_{metric}_latency"] = avg_latency

    def get_ni_status(self) -> Dict[str, Any]:
        """
        获取网络接口状态。

        Returns:
            网络接口状态字典
        """
        status = self.get_performance_stats()
        status.update(
            {
                "protocol_type": self.protocol_type.value,
                "clock_ratio": self.clock_ratio,
                "qos_enabled": self.qos_enabled,
                "ni_stats": self.ni_stats.copy(),
                "active_transactions": len(self.active_transactions),
                "buffer_occupancy": {"ip_to_noc": len(self.ip_to_noc_buffer), "noc_to_ip": len(self.noc_to_ip_buffer)},
                "traffic_shaper": self.traffic_shaper.copy(),
                "clock_sync": self.clock_sync.copy(),
            }
        )

        return status

    def create_transaction(self, trans_id: str, flit: BaseFlit) -> None:
        """
        创建新事务。

        Args:
            trans_id: 事务ID
            flit: 相关的flit
        """
        self.active_transactions[trans_id] = {
            "flit": flit,
            "state": TransactionState.PENDING,
            "start_time": self.current_cycle,
            "retry_count": 0,
            "last_retry_time": 0,
            "retry_delay": self.retry_config["retry_delay"],
        }

    def complete_transaction(self, trans_id: str) -> None:
        """
        完成事务。

        Args:
            trans_id: 事务ID
        """
        if trans_id in self.active_transactions:
            self.active_transactions[trans_id]["state"] = TransactionState.COMPLETED
            self.active_transactions[trans_id]["completion_time"] = self.current_cycle

    def set_qos_policy(self, qos_class: QoSClass, priority: Priority, weight: float) -> None:
        """
        设置QoS策略。

        Args:
            qos_class: QoS类别
            priority: 优先级
            weight: 权重
        """
        self.qos_classes[qos_class] = {"priority": priority, "weight": weight}

    def configure_traffic_shaper(self, rate_limit: int, burst_size: int) -> None:
        """
        配置流量整形器。

        Args:
            rate_limit: 速率限制
            burst_size: 突发大小
        """
        self.traffic_shaper["rate_limit"] = rate_limit
        self.traffic_shaper["burst_size"] = burst_size
        self.traffic_shaper["token_bucket"] = burst_size
