"""
性能监控模块
实现实时性能指标收集，包括延迟、带宽、吞吐量和错误率统计
"""

from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import deque, defaultdict
import statistics
import logging
import json
import os

from src.utils.exceptions import CDMAError


class MetricType(Enum):
    """指标类型枚举"""

    LATENCY = "latency"
    BANDWIDTH = "bandwidth"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_DEPTH = "queue_depth"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"


class AlertLevel(Enum):
    """告警级别枚举"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricSample:
    """指标样本"""

    timestamp: float
    value: float
    metric_type: MetricType
    source_id: str
    tags: Dict[str, str] = field(default_factory=dict)

    def age(self) -> float:
        """获取样本年龄（秒）"""
        return time.time() - self.timestamp


@dataclass
class PerformanceAlert:
    """性能告警"""

    alert_id: str
    level: AlertLevel
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    timestamp: float
    source_id: str
    acknowledged: bool = False


@dataclass
class LatencyStats:
    """延迟统计信息"""

    min_latency: float = float("inf")
    max_latency: float = 0.0
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    sample_count: int = 0

    def update_from_samples(self, samples: List[float]):
        """从样本更新统计信息"""
        if not samples:
            return

        self.sample_count = len(samples)
        self.min_latency = min(samples)
        self.max_latency = max(samples)
        self.avg_latency = statistics.mean(samples)

        # 计算百分位数
        sorted_samples = sorted(samples)
        self.p50_latency = statistics.median(sorted_samples)

        if len(sorted_samples) >= 20:  # 需要足够的样本计算百分位数
            p95_idx = int(0.95 * len(sorted_samples))
            p99_idx = int(0.99 * len(sorted_samples))
            self.p95_latency = sorted_samples[p95_idx]
            self.p99_latency = sorted_samples[p99_idx]
        else:
            self.p95_latency = self.max_latency
            self.p99_latency = self.max_latency


class LatencyTracker:
    """延迟 Tracker"""

    def __init__(self, source_id: str, max_samples: int = 10000):
        self._source_id = source_id
        self._max_samples = max_samples

        # 延迟样本存储
        self._latency_samples: deque = deque(maxlen=max_samples)
        self._end_to_end_samples: deque = deque(maxlen=max_samples)

        # 分组延迟跟踪
        self._component_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # 实时统计
        self._current_stats = LatencyStats()
        self._e2e_stats = LatencyStats()

        # 告警阈值
        self._latency_warning_threshold = 0.010  # 10ms
        self._latency_error_threshold = 0.050  # 50ms
        self._latency_critical_threshold = 0.100  # 100ms

        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"LatencyTracker-{source_id}")

    def record_latency(self, latency: float, component: str = "total"):
        """
        记录延迟样本

        Args:
            latency: 延迟值（秒）
            component: 组件名称
        """
        with self._lock:
            timestamp = time.time()

            if component == "total":
                self._latency_samples.append(latency)

                # 更新实时统计
                self._update_latency_stats()
            else:
                self._component_latencies[component].append(latency)

            # 检查告警
            self._check_latency_alerts(latency, component)

    def record_end_to_end_latency(self, start_timestamp: float, end_timestamp: float):
        """
        记录端到端延迟

        Args:
            start_timestamp: 开始时间戳
            end_timestamp: 结束时间戳
        """
        with self._lock:
            latency = end_timestamp - start_timestamp
            self._end_to_end_samples.append(latency)

            # 更新端到端统计
            self._update_e2e_stats()

    def _update_latency_stats(self):
        """更新延迟统计"""
        if not self._latency_samples:
            return

        # 获取最近的样本（最多1000个）
        recent_samples = list(self._latency_samples)[-1000:]
        self._current_stats.update_from_samples(recent_samples)

    def _update_e2e_stats(self):
        """更新端到端延迟统计"""
        if not self._end_to_end_samples:
            return

        recent_samples = list(self._end_to_end_samples)[-1000:]
        self._e2e_stats.update_from_samples(recent_samples)

    def _check_latency_alerts(self, latency: float, component: str):
        """检查延迟告警"""
        # 这里可以触发告警回调，暂时只记录日志
        if latency > self._latency_critical_threshold:
            self._logger.error(f"严重延迟告警: {component}={latency*1000:.2f}ms")
        elif latency > self._latency_error_threshold:
            self._logger.warning(f"延迟错误告警: {component}={latency*1000:.2f}ms")
        elif latency > self._latency_warning_threshold:
            self._logger.info(f"延迟警告: {component}={latency*1000:.2f}ms")

    def get_latency_stats(self) -> Dict[str, Any]:
        """获取延迟统计信息"""
        with self._lock:
            result = {
                "source_id": self._source_id,
                "total_events": self._current_stats.sample_count + self._e2e_stats.sample_count,
                "total_latency": {
                    "min_ms": self._current_stats.min_latency * 1000,
                    "max_ms": self._current_stats.max_latency * 1000,
                    "avg_ms": self._current_stats.avg_latency * 1000,
                    "p50_ms": self._current_stats.p50_latency * 1000,
                    "p95_ms": self._current_stats.p95_latency * 1000,
                    "p99_ms": self._current_stats.p99_latency * 1000,
                    "sample_count": self._current_stats.sample_count,
                },
                "end_to_end_latency": {
                    "min_ms": self._e2e_stats.min_latency * 1000,
                    "max_ms": self._e2e_stats.max_latency * 1000,
                    "avg_ms": self._e2e_stats.avg_latency * 1000,
                    "p50_ms": self._e2e_stats.p50_latency * 1000,
                    "p95_ms": self._e2e_stats.p95_latency * 1000,
                    "p99_ms": self._e2e_stats.p99_latency * 1000,
                    "sample_count": self._e2e_stats.sample_count,
                },
                "component_latencies": {},
            }

            # 添加组件延迟统计
            for component, samples in self._component_latencies.items():
                if samples:
                    recent_samples = list(samples)[-100:]  # 最近100个样本
                    stats = LatencyStats()
                    stats.update_from_samples(recent_samples)

                    result["component_latencies"][component] = {
                        "min_ms": stats.min_latency * 1000,
                        "max_ms": stats.max_latency * 1000,
                        "avg_ms": stats.avg_latency * 1000,
                        "p95_ms": stats.p95_latency * 1000,
                        "sample_count": stats.sample_count,
                    }

            return result


class ThroughputCounter:
    """吞吐量计数器"""

    def __init__(self, source_id: str):
        self._source_id = source_id

        # 事件计数
        self._event_timestamps: deque = deque(maxlen=10000)
        self._byte_counts: deque = deque(maxlen=10000)

        # 分类计数
        self._categorized_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # 实时统计
        self._current_tps = 0.0  # 每秒事务数
        self._current_bps = 0.0  # 每秒字节数
        self._total_events = 0
        self._total_bytes = 0

        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"ThroughputCounter-{source_id}")

    def record_event(self, byte_count: int = 0, category: str = "default"):
        """
        记录事件

        Args:
            byte_count: 字节数
            category: 事件类别
        """
        with self._lock:
            timestamp = time.time()

            self._event_timestamps.append(timestamp)
            self._byte_counts.append(byte_count)
            self._categorized_events[category].append(timestamp)

            self._total_events += 1
            self._total_bytes += byte_count

            # 更新实时统计
            self._update_throughput_stats()

    def _update_throughput_stats(self):
        """更新吞吐量统计"""
        current_time = time.time()

        # 计算最近1秒的事件数
        recent_events = [ts for ts in self._event_timestamps if current_time - ts <= 1.0]
        self._current_tps = len(recent_events)

        # 计算最近1秒的字节数
        recent_bytes = [self._byte_counts[i] for i, ts in enumerate(self._event_timestamps) if current_time - ts <= 1.0]
        self._current_bps = sum(recent_bytes)

    def get_throughput_stats(self) -> Dict[str, Any]:
        """获取吞吐量统计信息"""
        with self._lock:
            current_time = time.time()

            # 计算不同时间窗口的统计
            windows = [1, 5, 10, 60]  # 1秒、5秒、10秒、1分钟
            window_stats = {}

            for window in windows:
                recent_events = [ts for ts in self._event_timestamps if current_time - ts <= window]
                recent_bytes = [self._byte_counts[i] for i, ts in enumerate(self._event_timestamps) if current_time - ts <= window]

                window_stats[f"{window}s"] = {
                    "tps": len(recent_events) / window,
                    "bps": sum(recent_bytes) / window,
                    "mbps": sum(recent_bytes) / (window * 1024 * 1024),
                    "event_count": len(recent_events),
                }

            # 分类统计
            category_stats = {}
            for category, timestamps in self._categorized_events.items():
                recent_category_events = [ts for ts in timestamps if current_time - ts <= 60]
                category_stats[category] = {"tps_1min": len(recent_category_events) / 60, "total_events": len(timestamps)}

            return {
                "source_id": self._source_id,
                "current_tps": self._current_tps,
                "current_bps": self._current_bps,
                "current_mbps": self._current_bps / (1024 * 1024),
                "total_events": self._total_events,
                "total_bytes": self._total_bytes,
                "total_mb": self._total_bytes / (1024 * 1024),
                "window_stats": window_stats,
                "category_stats": category_stats,
            }


class MetricsCollector:
    """指标收集器"""

    def __init__(self, source_id: str):
        self._source_id = source_id

        # 指标存储
        self._metrics: Dict[MetricType, deque] = {metric_type: deque(maxlen=10000) for metric_type in MetricType}

        # 告警配置
        self._alert_thresholds: Dict[MetricType, Dict[AlertLevel, float]] = {
            MetricType.LATENCY: {AlertLevel.WARNING: 0.010, AlertLevel.ERROR: 0.050, AlertLevel.CRITICAL: 0.100},  # 10ms  # 50ms  # 100ms
            MetricType.ERROR_RATE: {AlertLevel.WARNING: 0.01, AlertLevel.ERROR: 0.05, AlertLevel.CRITICAL: 0.10},  # 1%  # 5%  # 10%
            MetricType.CPU_USAGE: {AlertLevel.WARNING: 0.70, AlertLevel.ERROR: 0.85, AlertLevel.CRITICAL: 0.95},  # 70%  # 85%  # 95%
        }

        # 告警历史
        self._alerts: deque = deque(maxlen=1000)
        self._alert_callbacks: List[Callable[[PerformanceAlert], None]] = []

        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"MetricsCollector-{source_id}")

    def add_metric_sample(self, metric_type: MetricType, value: float, tags: Dict[str, str] = None):
        """
        添加指标样本

        Args:
            metric_type: 指标类型
            value: 指标值
            tags: 标签
        """
        with self._lock:
            sample = MetricSample(timestamp=time.time(), value=value, metric_type=metric_type, source_id=self._source_id, tags=tags or {})

            self._metrics[metric_type].append(sample)

            # 检查告警
            self._check_metric_alerts(metric_type, value)

    def _check_metric_alerts(self, metric_type: MetricType, value: float):
        """检查指标告警"""
        if metric_type not in self._alert_thresholds:
            return

        thresholds = self._alert_thresholds[metric_type]

        # 检查严重级别（从高到低）
        for level in [AlertLevel.CRITICAL, AlertLevel.ERROR, AlertLevel.WARNING]:
            if level in thresholds and value >= thresholds[level]:
                self._trigger_alert(metric_type, level, value, thresholds[level])
                break

    def _trigger_alert(self, metric_type: MetricType, level: AlertLevel, value: float, threshold: float):
        """触发告警"""
        alert_id = f"alert_{self._source_id}_{metric_type.value}_{int(time.time() * 1000)}"

        alert = PerformanceAlert(
            alert_id=alert_id,
            level=level,
            metric_type=metric_type,
            message=f"{metric_type.value} 超出阈值: {value:.4f} >= {threshold:.4f}",
            value=value,
            threshold=threshold,
            timestamp=time.time(),
            source_id=self._source_id,
        )

        self._alerts.append(alert)

        # 调用告警回调
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self._logger.error(f"告警回调执行失败: {e}")

        # 记录日志
        if level == AlertLevel.CRITICAL:
            self._logger.error(f"严重告警: {alert.message}")
        elif level == AlertLevel.ERROR:
            self._logger.warning(f"错误告警: {alert.message}")
        else:
            self._logger.info(f"警告: {alert.message}")

    def register_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """注册告警回调"""
        self._alert_callbacks.append(callback)

    def get_metric_summary(self, metric_type: MetricType, window_seconds: float = 60.0) -> Dict[str, Any]:
        """
        获取指标摘要

        Args:
            metric_type: 指标类型
            window_seconds: 时间窗口（秒）

        Returns:
            Dict: 指标摘要
        """
        with self._lock:
            current_time = time.time()

            # 获取时间窗口内的样本
            recent_samples = [sample for sample in self._metrics[metric_type] if current_time - sample.timestamp <= window_seconds]

            if not recent_samples:
                return {"metric_type": metric_type.value, "sample_count": 0, "window_seconds": window_seconds}

            values = [sample.value for sample in recent_samples]

            return {
                "metric_type": metric_type.value,
                "sample_count": len(values),
                "window_seconds": window_seconds,
                "min_value": min(values),
                "max_value": max(values),
                "avg_value": statistics.mean(values),
                "median_value": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "latest_value": values[-1],
                "oldest_value": values[0],
            }

    def get_all_alerts(self, acknowledged_only: bool = False) -> List[PerformanceAlert]:
        """获取所有告警"""
        with self._lock:
            if acknowledged_only:
                return [alert for alert in self._alerts if alert.acknowledged]
            else:
                return list(self._alerts)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return True
            return False


class PerformanceMonitor:
    """性能监控主类"""

    def __init__(self, chip_id: str):
        self._chip_id = chip_id

        # 核心组件
        self._latency_tracker = LatencyTracker(chip_id)
        self._throughput_counter = ThroughputCounter(chip_id)
        self._metrics_collector = MetricsCollector(chip_id)

        # 监控状态
        self._monitoring_enabled = True
        self._start_time = time.time()

        # 定期任务
        self._cleanup_interval = 300.0  # 5分钟清理一次过期数据
        self._last_cleanup_time = time.time()

        # 性能报告
        self._report_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"PerformanceMonitor-{chip_id}")

        # 注册默认告警处理
        self._metrics_collector.register_alert_callback(self._default_alert_handler)

        self._logger.info(f"性能监控器初始化完成: {chip_id}")

    def enable_monitoring(self):
        """启用监控"""
        with self._lock:
            self._monitoring_enabled = True
            self._logger.info("性能监控已启用")

    def disable_monitoring(self):
        """禁用监控"""
        with self._lock:
            self._monitoring_enabled = False
            self._logger.info("性能监控已禁用")

    def record_operation_start(self, operation_id: str) -> float:
        """
        记录操作开始

        Args:
            operation_id: 操作ID

        Returns:
            float: 开始时间戳
        """
        if not self._monitoring_enabled:
            return time.time()

        start_time = time.time()
        # 可以在这里存储操作上下文
        return start_time

    def record_operation_end(self, operation_id: str, start_time: float, byte_count: int = 0, success: bool = True):
        """
        记录操作结束

        Args:
            operation_id: 操作ID
            start_time: 开始时间戳
            byte_count: 处理的字节数
            success: 操作是否成功
        """
        if not self._monitoring_enabled:
            return

        end_time = time.time()
        latency = end_time - start_time

        with self._lock:
            # 记录延迟
            self._latency_tracker.record_latency(latency)
            self._latency_tracker.record_end_to_end_latency(start_time, end_time)

            # 记录吞吐量
            self._throughput_counter.record_event(byte_count, operation_id)

            # 记录指标
            self._metrics_collector.add_metric_sample(MetricType.LATENCY, latency)
            if byte_count > 0:
                throughput = byte_count / max(latency, 0.001)  # 避免除零
                self._metrics_collector.add_metric_sample(MetricType.THROUGHPUT, throughput)

            # 记录错误率
            if not success:
                self._metrics_collector.add_metric_sample(MetricType.ERROR_RATE, 1.0)
            else:
                self._metrics_collector.add_metric_sample(MetricType.ERROR_RATE, 0.0)

    def record_custom_metric(self, metric_type: MetricType, value: float, tags: Dict[str, str] = None):
        """记录自定义指标"""
        if not self._monitoring_enabled:
            return

        self._metrics_collector.add_metric_sample(metric_type, value, tags)

    def _default_alert_handler(self, alert: PerformanceAlert):
        """默认告警处理器"""
        self._logger.warning(f"性能告警: {alert.level.value} - {alert.message}")

    def register_report_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """注册性能报告回调"""
        self._report_callbacks.append(callback)

    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        with self._lock:
            current_time = time.time()
            uptime = current_time - self._start_time

            report = {
                "chip_id": self._chip_id,
                "report_timestamp": current_time,
                "uptime_seconds": uptime,
                "monitoring_enabled": self._monitoring_enabled,
                "latency_stats": self._latency_tracker.get_latency_stats(),
                "throughput_stats": self._throughput_counter.get_throughput_stats(),
                "metric_summaries": {},
                "recent_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "level": alert.level.value,
                        "metric_type": alert.metric_type.value,
                        "message": alert.message,
                        "value": alert.value,
                        "threshold": alert.threshold,
                        "timestamp": alert.timestamp,
                        "acknowledged": alert.acknowledged,
                    }
                    for alert in self._metrics_collector.get_all_alerts()
                    if current_time - alert.timestamp <= 3600  # 最近1小时的告警
                ],
            }

            # 添加各种指标的摘要
            for metric_type in MetricType:
                report["metric_summaries"][metric_type.value] = self._metrics_collector.get_metric_summary(metric_type)

            return report

    def save_performance_report(self, file_path: str):
        """保存性能报告到文件"""
        try:
            report = self.generate_performance_report()

            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self._logger.info(f"性能报告已保存: {file_path}")

        except Exception as e:
            self._logger.error(f"保存性能报告失败: {e}")

    def cleanup_old_data(self):
        """清理过期数据"""
        with self._lock:
            current_time = time.time()

            # 只在指定间隔后执行清理
            if current_time - self._last_cleanup_time < self._cleanup_interval:
                return

            self._last_cleanup_time = current_time

            # 这里可以添加更多清理逻辑
            # 例如清理过期的指标样本等

            self._logger.debug("执行数据清理")

    def get_summary_stats(self) -> Dict[str, Any]:
        """获取摘要统计信息"""
        with self._lock:
            return {
                "chip_id": self._chip_id,
                "monitoring_enabled": self._monitoring_enabled,
                "uptime_seconds": time.time() - self._start_time,
                "latency_summary": self._latency_tracker.get_latency_stats(),
                "throughput_summary": self._throughput_counter.get_throughput_stats(),
                "alert_count": len(self._metrics_collector.get_all_alerts()),
                "unacknowledged_alerts": len([a for a in self._metrics_collector.get_all_alerts() if not a.acknowledged]),
            }

    def reset_statistics(self):
        """重置所有统计信息"""
        with self._lock:
            # 重新创建组件以清空数据
            self._latency_tracker = LatencyTracker(self._chip_id)
            self._throughput_counter = ThroughputCounter(self._chip_id)
            self._metrics_collector = MetricsCollector(self._chip_id)

            # 重新注册告警处理
            self._metrics_collector.register_alert_callback(self._default_alert_handler)

            self._start_time = time.time()
            self._logger.info("性能统计信息已重置")

    def shutdown(self):
        """关闭性能监控器"""
        with self._lock:
            self._monitoring_enabled = False
            self._logger.info(f"性能监控器已关闭: {self._chip_id}")

    @property
    def chip_id(self) -> str:
        return self._chip_id
