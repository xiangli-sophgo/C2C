from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import threading
import queue
from utils.exceptions import CDMAError
from .memory_types import MemoryType


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


class DMAController:
    """DMA控制器 - 负责模拟内存数据搬运"""

    def __init__(self, chip_id: str):
        self._chip_id = chip_id
        self._memory_regions = self._init_memory_regions()
        self._memory_simulator: Dict[int, bytes] = {}  # 模拟内存空间
        self._transfer_queue = queue.PriorityQueue()
        self._active_transfers: Dict[str, DMATransferRequest] = {}
        self._transfer_history: Dict[str, DMATransferResult] = {}
        self._is_running = False
        self._worker_thread = None
        self._transfer_counter = 0

        # 性能统计
        self._total_bytes_transferred = 0
        self._total_transfers = 0
        self._total_transfer_time = 0.0

    def _init_memory_regions(self) -> Dict[MemoryType, MemoryRegion]:
        """初始化内存区域配置"""
        return {
            MemoryType.GMEM: MemoryRegion(start_addr=0x00000000, size=16 * 1024 * 1024 * 1024, mem_type=MemoryType.GMEM, alignment=64, bandwidth_gbps=200.0),  # 16GB  # 64字节对齐  # 200GB/s
            MemoryType.L2M: MemoryRegion(start_addr=0x40000000, size=128 * 1024 * 1024, mem_type=MemoryType.L2M, alignment=32, bandwidth_gbps=800.0),  # 128MB  # 32字节对齐  # 800GB/s
            MemoryType.LMEM: MemoryRegion(start_addr=0x80000000, size=512 * 1024 * 1024, mem_type=MemoryType.LMEM, alignment=16, bandwidth_gbps=1000.0),  # 512MB  # 16字节对齐  # 1TB/s
        }

    def start(self):
        """启动DMA控制器"""
        if not self._is_running:
            self._is_running = True
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
            print(f"芯片 {self._chip_id}：DMA控制器已启动")

    def stop(self):
        """停止DMA控制器"""
        if self._is_running:
            self._is_running = False
            if self._worker_thread:
                self._worker_thread.join(timeout=1.0)
            print(f"芯片 {self._chip_id}：DMA控制器已停止")

    def execute_dma_transfer(self, src_addr: int, dst_addr: int, data_size: int, src_chip_id: str, dst_chip_id: str, src_mem_type: MemoryType, dst_mem_type: MemoryType, priority: int = 0) -> str:
        """
        执行DMA传输

        Args:
            src_addr: 源地址
            dst_addr: 目标地址
            data_size: 数据大小（字节）
            src_chip_id: 源芯片ID
            dst_chip_id: 目标芯片ID
            src_mem_type: 源内存类型
            dst_mem_type: 目标内存类型
            priority: 优先级（数值越小优先级越高）

        Returns:
            str: 传输请求ID
        """
        # 验证地址和大小
        if not self._validate_address(src_addr, data_size, src_mem_type):
            raise CDMAError(f"源地址验证失败: 0x{src_addr:08x}, 大小: {data_size}, 类型: {src_mem_type.value}")

        if not self._validate_address(dst_addr, data_size, dst_mem_type):
            raise CDMAError(f"目标地址验证失败: 0x{dst_addr:08x}, 大小: {data_size}, 类型: {dst_mem_type.value}")

        request_id = f"dma_{self._chip_id}_{self._transfer_counter}_{int(time.time() * 1000000)}"
        self._transfer_counter += 1

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
        self._transfer_queue.put((priority, time.time(), request))
        print(f"芯片 {self._chip_id}：DMA传输请求已排队")
        print(f"  请求ID: {request_id}")
        print(f"  源地址: 0x{src_addr:08x} ({src_mem_type.value})")
        print(f"  目标地址: 0x{dst_addr:08x} ({dst_mem_type.value})")
        print(f"  数据大小: {data_size} 字节")

        return request_id

    def _worker_loop(self):
        """DMA工作线程主循环"""
        while self._is_running:
            try:
                # 获取传输请求（1秒超时）
                _, _, request = self._transfer_queue.get(timeout=1.0)

                # 执行传输
                result = self._perform_transfer(request)

                # 记录结果
                self._transfer_history[request.request_id] = result

                # 更新统计信息
                if result.success:
                    self._total_bytes_transferred += result.bytes_transferred
                    self._total_transfers += 1
                    self._total_transfer_time += result.end_time - result.start_time

                # 从活跃传输中移除
                if request.request_id in self._active_transfers:
                    del self._active_transfers[request.request_id]

                self._transfer_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"芯片 {self._chip_id}：DMA工作线程错误: {str(e)}")

    def _perform_transfer(self, request: DMATransferRequest) -> DMATransferResult:
        """执行实际的DMA传输"""
        start_time = time.time()
        self._active_transfers[request.request_id] = request

        try:
            print(f"芯片 {self._chip_id}：开始执行DMA传输 {request.request_id}")

            # 模拟从源地址读取数据
            src_data = self._read_memory(request.src_addr, request.size)

            # 计算传输时间（基于带宽）
            src_bandwidth = self._memory_regions[request.src_mem_type].bandwidth_gbps
            dst_bandwidth = self._memory_regions[request.dst_mem_type].bandwidth_gbps
            effective_bandwidth = min(src_bandwidth, dst_bandwidth)

            # 如果是跨芯片传输，考虑C2C链路带宽
            if request.src_chip_id != request.dst_chip_id:
                c2c_bandwidth = 25.0  # 假设C2C带宽为25GB/s
                effective_bandwidth = min(effective_bandwidth, c2c_bandwidth)

            transfer_time = request.size / (effective_bandwidth * 1024 * 1024 * 1024)

            # 模拟传输延迟
            time.sleep(min(transfer_time, 0.1))  # 最多延迟100ms

            # 模拟写入目标地址
            self._write_memory(request.dst_addr, src_data)

            end_time = time.time()
            actual_bandwidth = request.size / (end_time - start_time) / (1024 * 1024 * 1024)

            print(f"芯片 {self._chip_id}：DMA传输完成 {request.request_id}")
            print(f"  传输时间: {(end_time - start_time) * 1000:.2f} ms")
            print(f"  实际带宽: {actual_bandwidth:.2f} GB/s")

            return DMATransferResult(request_id=request.request_id, success=True, start_time=start_time, end_time=end_time, bytes_transferred=request.size, bandwidth_achieved=actual_bandwidth)

        except Exception as e:
            end_time = time.time()
            error_msg = f"DMA传输失败: {str(e)}"
            print(f"芯片 {self._chip_id}：{error_msg}")

            return DMATransferResult(request_id=request.request_id, success=False, start_time=start_time, end_time=end_time, bytes_transferred=0, error_message=error_msg)

    def _validate_address(self, address: int, size: int, mem_type: MemoryType) -> bool:
        """验证地址合法性（测试模式 - 非常宽松）"""
        # 为了测试，基本上只做最基本的检查

        # 检查地址是否为正数
        if address < 0:
            print(f"DMA控制器：地址为负数 0x{address:08x}")
            return False

        # 检查大小是否合理（不超过1GB）
        if size <= 0 or size > 1024 * 1024 * 1024:
            print(f"DMA控制器：数据大小不合理 {size}")
            return False

        # 非常宽松的地址范围检查
        if address > 0xFFFFFFFF:  # 32位地址空间
            print(f"DMA控制器：地址超出32位范围 0x{address:08x}")
            return False

        return True

    def validate_address_compatibility(self, src_addr: int, src_size: int, src_mem_type: MemoryType, dst_addr: int, dst_size: int, dst_mem_type: MemoryType) -> bool:
        """验证源和目标地址兼容性"""
        # 检查大小是否匹配
        if src_size != dst_size:
            return False

        # 验证源地址
        if not self._validate_address(src_addr, src_size, src_mem_type):
            return False

        # 验证目标地址
        if not self._validate_address(dst_addr, dst_size, dst_mem_type):
            return False

        return True

    def _read_memory(self, address: int, size: int) -> bytes:
        """从模拟内存读取数据"""
        if address in self._memory_simulator:
            data = self._memory_simulator[address]
            if len(data) >= size:
                return data[:size]

        # 生成模拟数据
        data = bytes([(address + i) % 256 for i in range(size)])
        self._memory_simulator[address] = data
        return data

    def _write_memory(self, address: int, data: bytes):
        """向模拟内存写入数据"""
        self._memory_simulator[address] = data

    def get_transfer_status(self, request_id: str) -> Optional[str]:
        """获取传输状态"""
        if request_id in self._active_transfers:
            return "transferring"
        elif request_id in self._transfer_history:
            result = self._transfer_history[request_id]
            return "completed" if result.success else "failed"
        else:
            return None

    def get_transfer_result(self, request_id: str) -> Optional[DMATransferResult]:
        """获取传输结果"""
        return self._transfer_history.get(request_id)

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        avg_bandwidth = 0.0
        if self._total_transfer_time > 0:
            avg_bandwidth = self._total_bytes_transferred / self._total_transfer_time / (1024 * 1024 * 1024)

        return {
            "total_transfers": self._total_transfers,
            "total_bytes_transferred": self._total_bytes_transferred,
            "total_transfer_time": self._total_transfer_time,
            "average_bandwidth_gbps": avg_bandwidth,
            "active_transfers": len(self._active_transfers),
            "queued_transfers": self._transfer_queue.qsize(),
        }

    def clear_history(self):
        """清理传输历史记录"""
        self._transfer_history.clear()
        self._total_bytes_transferred = 0
        self._total_transfers = 0
        self._total_transfer_time = 0.0
        print(f"芯片 {self._chip_id}：DMA传输历史已清理")

    @property
    def chip_id(self) -> str:
        return self._chip_id

    def __del__(self):
        """析构函数"""
        self.stop()
