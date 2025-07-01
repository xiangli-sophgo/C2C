from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import time
import threading
from enum import Enum

from .memory_types import MemoryType
from .cdma_engine import CDMAEngine, AddressInfo
from .dma_controller import DMAController
from .transaction_manager import TransactionManager, TransactionInfo
from .credit import CreditManager, AddressInfo as CreditAddressInfo
from utils.exceptions import CDMAError, AddressError, ShapeCompatibilityError


class CDMASystemState(Enum):
    """CDMA系统状态"""
    IDLE = "idle"
    READY = "ready"
    ACTIVE = "active"
    ERROR = "error"


@dataclass
class CDMAOperationResult:
    """CDMA操作结果"""
    success: bool
    transaction_id: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    bytes_transferred: int = 0


class CDMASystem:
    """CDMA系统 - 整合所有CDMA组件的主要接口"""
    
    def __init__(self, chip_id: str):
        self._chip_id = chip_id
        self._state = CDMASystemState.IDLE
        
        # 核心组件
        self._cdma_engine = CDMAEngine(chip_id)
        self._dma_controller = DMAController(chip_id)
        self._transaction_manager = TransactionManager()
        self._credit_manager = CreditManager(chip_id)
        
        # 系统级连接（模拟芯片间通信）
        self._connected_systems: Dict[str, 'CDMASystem'] = {}
        
        # 同步机制
        self._lock = threading.RLock()
        
        # 启动DMA控制器
        self._dma_controller.start()
        self._state = CDMASystemState.READY
        
        print(f"CDMA系统 {chip_id} 初始化完成")
    
    def connect_to_chip(self, other_chip_id: str, other_system: 'CDMASystem'):
        """连接到其他芯片的CDMA系统"""
        with self._lock:
            self._connected_systems[other_chip_id] = other_system
            print(f"CDMA系统 {self._chip_id}: 连接到芯片 {other_chip_id}")
    
    def cdma_receive(self, dst_addr: int, dst_shape: Tuple[int, ...], 
                    dst_mem_type: MemoryType, src_chip_id: str,
                    data_type: str = "float32", timeout: float = 30.0) -> CDMAOperationResult:
        """
        执行CDMA接收操作
        
        Args:
            dst_addr: 目标地址
            dst_shape: 目标tensor形状
            dst_mem_type: 目标内存类型
            src_chip_id: 源芯片ID
            data_type: 数据类型
            timeout: 超时时间（秒）
            
        Returns:
            CDMAOperationResult: 操作结果
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # 检查系统状态
                if self._state != CDMASystemState.READY:
                    return CDMAOperationResult(
                        success=False,
                        error_message=f"CDMA系统状态不正确: {self._state.value}"
                    )
                
                # 检查是否连接到源芯片
                if src_chip_id not in self._connected_systems:
                    return CDMAOperationResult(
                        success=False,
                        error_message=f"未连接到源芯片: {src_chip_id}"
                    )
                
                # 生成事务ID
                transaction_id = f"cdma_{self._chip_id}_{src_chip_id}_{int(time.time() * 1000000)}"
                
                # 创建配对事务
                transaction = self._transaction_manager.create_paired_transaction(
                    send_chip_id=src_chip_id,
                    recv_chip_id=self._chip_id,
                    transaction_id=transaction_id,
                    timeout_seconds=timeout
                )
                
                # 注册接收操作
                success = self._transaction_manager.register_receive_operation(
                    transaction_id=transaction_id,
                    dst_addr=dst_addr,
                    dst_shape=dst_shape,
                    dst_mem_type=dst_mem_type.value,
                    data_type=data_type
                )
                
                if not success:
                    return CDMAOperationResult(
                        success=False,
                        transaction_id=transaction_id,
                        error_message="注册接收操作失败"
                    )
                
                # 向源芯片发送Credit+地址信息
                dst_address_info = CreditAddressInfo(
                    address=dst_addr,
                    shape=dst_shape,
                    mem_type=dst_mem_type.value,
                    data_type=data_type
                )
                
                # 通过连接发送Credit信息到源芯片
                src_system = self._connected_systems[src_chip_id]
                src_system._receive_credit_from_chip(self._chip_id, dst_address_info, transaction_id)
                
                execution_time = time.time() - start_time
                
                print(f"CDMA系统 {self._chip_id}: CDMA_receive操作完成")
                print(f"  事务ID: {transaction_id}")
                print(f"  执行时间: {execution_time * 1000:.2f} ms")
                
                return CDMAOperationResult(
                    success=True,
                    transaction_id=transaction_id,
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"CDMA_receive操作失败: {str(e)}"
            print(f"CDMA系统 {self._chip_id}: {error_msg}")
            
            return CDMAOperationResult(
                success=False,
                error_message=error_msg,
                execution_time=execution_time
            )
    
    def cdma_send(self, src_addr: int, src_shape: Tuple[int, ...], 
                 dst_chip_id: str, src_mem_type: MemoryType = MemoryType.GMEM,
                 data_type: str = "float32") -> CDMAOperationResult:
        """
        执行CDMA发送操作
        
        Args:
            src_addr: 源地址
            src_shape: 源tensor形状
            dst_chip_id: 目标芯片ID
            src_mem_type: 源内存类型
            data_type: 数据类型
            
        Returns:
            CDMAOperationResult: 操作结果
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # 检查系统状态
                if self._state != CDMASystemState.READY:
                    return CDMAOperationResult(
                        success=False,
                        error_message=f"CDMA系统状态不正确: {self._state.value}"
                    )
                
                # 检查是否连接到目标芯片
                if dst_chip_id not in self._connected_systems:
                    return CDMAOperationResult(
                        success=False,
                        error_message=f"未连接到目标芯片: {dst_chip_id}"
                    )
                
                # 检查是否有Credit+地址信息
                if not self._credit_manager.has_credit_with_address(dst_chip_id):
                    return CDMAOperationResult(
                        success=False,
                        error_message=f"没有来自 {dst_chip_id} 的Credit，请先执行CDMA_receive"
                    )
                
                # 获取目标地址信息
                dst_address_info = self._credit_manager.check_and_consume_credit(dst_chip_id)
                if not dst_address_info:
                    return CDMAOperationResult(
                        success=False,
                        error_message="无法获取目标地址信息"
                    )
                
                # 验证形状兼容性
                if src_shape != dst_address_info.shape:
                    raise ShapeCompatibilityError(src_shape, dst_address_info.shape)
                
                # 验证数据类型兼容性
                if data_type != dst_address_info.data_type:
                    return CDMAOperationResult(
                        success=False,
                        error_message=f"数据类型不匹配: src={data_type}, dst={dst_address_info.data_type}"
                    )
                
                # 计算数据大小
                element_count = 1
                for dim in src_shape:
                    element_count *= dim
                
                type_sizes = {
                    "float32": 4, "float16": 2, "int32": 4, 
                    "int16": 2, "int8": 1, "uint8": 1
                }
                data_size = element_count * type_sizes.get(data_type, 4)
                
                # 转换内存类型
                dst_mem_type = MemoryType(dst_address_info.mem_type)
                
                # 执行DMA传输
                dma_request_id = self._dma_controller.execute_dma_transfer(
                    src_addr=src_addr,
                    dst_addr=dst_address_info.address,
                    data_size=data_size,
                    src_chip_id=self._chip_id,
                    dst_chip_id=dst_chip_id,
                    src_mem_type=src_mem_type,
                    dst_mem_type=dst_mem_type
                )
                
                # 等待传输完成
                max_wait_time = 10.0  # 最多等待10秒
                wait_start = time.time()
                
                while time.time() - wait_start < max_wait_time:
                    status = self._dma_controller.get_transfer_status(dma_request_id)
                    if status == "completed":
                        result = self._dma_controller.get_transfer_result(dma_request_id)
                        execution_time = time.time() - start_time
                        
                        if result and result.success:
                            print(f"CDMA系统 {self._chip_id}: CDMA_send操作完成")
                            print(f"  传输字节数: {result.bytes_transferred}")
                            print(f"  实际带宽: {result.bandwidth_achieved:.2f} GB/s")
                            print(f"  总执行时间: {execution_time * 1000:.2f} ms")
                            
                            return CDMAOperationResult(
                                success=True,
                                transaction_id=dma_request_id,
                                execution_time=execution_time,
                                bytes_transferred=result.bytes_transferred
                            )
                        else:
                            return CDMAOperationResult(
                                success=False,
                                transaction_id=dma_request_id,
                                error_message=result.error_message if result else "DMA传输失败",
                                execution_time=execution_time
                            )
                    elif status == "failed":
                        result = self._dma_controller.get_transfer_result(dma_request_id)
                        return CDMAOperationResult(
                            success=False,
                            transaction_id=dma_request_id,
                            error_message=result.error_message if result else "DMA传输失败",
                            execution_time=time.time() - start_time
                        )
                    
                    time.sleep(0.001)  # 等待1ms
                
                # 超时
                return CDMAOperationResult(
                    success=False,
                    transaction_id=dma_request_id,
                    error_message="DMA传输超时",
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"CDMA_send操作失败: {str(e)}"
            print(f"CDMA系统 {self._chip_id}: {error_msg}")
            
            return CDMAOperationResult(
                success=False,
                error_message=error_msg,
                execution_time=execution_time
            )
    
    def cdma_sys_send_msg(self, target_chip_id: str, message: str = "sync") -> CDMAOperationResult:
        """
        发送同步消息
        
        Args:
            target_chip_id: 目标芯片ID
            message: 同步消息内容
            
        Returns:
            CDMAOperationResult: 操作结果
        """
        start_time = time.time()
        
        try:
            with self._lock:
                if target_chip_id not in self._connected_systems:
                    return CDMAOperationResult(
                        success=False,
                        error_message=f"未连接到目标芯片: {target_chip_id}"
                    )
                
                # 发送同步消息到目标芯片
                target_system = self._connected_systems[target_chip_id]
                target_system._receive_sync_message(self._chip_id, message)
                
                execution_time = time.time() - start_time
                
                print(f"CDMA系统 {self._chip_id}: 向 {target_chip_id} 发送同步消息: {message}")
                
                return CDMAOperationResult(
                    success=True,
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"发送同步消息失败: {str(e)}"
            print(f"CDMA系统 {self._chip_id}: {error_msg}")
            
            return CDMAOperationResult(
                success=False,
                error_message=error_msg,
                execution_time=execution_time
            )
    
    def _receive_credit_from_chip(self, src_chip_id: str, dst_address_info: CreditAddressInfo, 
                                transaction_id: str):
        """接收来自其他芯片的Credit+地址信息（内部方法）"""
        from .credit import CreditWithAddressInfo
        self._credit_manager.receive_credit_with_address(
            src_chip_id, 
            CreditWithAddressInfo(
                credit_count=1,
                dst_address_info=dst_address_info,
                transaction_id=transaction_id,
                timestamp=time.time(),
                expires_at=time.time() + 30.0
            )
        )
    
    def _receive_sync_message(self, src_chip_id: str, message: str):
        """接收来自其他芯片的同步消息（内部方法）"""
        print(f"CDMA系统 {self._chip_id}: 收到来自 {src_chip_id} 的同步消息: {message}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self._lock:
            return {
                "chip_id": self._chip_id,
                "state": self._state.value,
                "connected_chips": list(self._connected_systems.keys()),
                "credit_status": self._credit_manager.get_credit_status(),
                "dma_performance": self._dma_controller.get_performance_stats(),
                "transaction_stats": self._transaction_manager.get_statistics()
            }
    
    def cleanup(self):
        """清理系统资源"""
        with self._lock:
            print(f"CDMA系统 {self._chip_id}: 开始清理...")
            
            # 清理过期的Credit
            self._credit_manager.cleanup_expired_credits()
            
            # 清理过期的事务
            self._transaction_manager.cleanup_expired_transactions()
            
            # 清理已完成的事务
            self._transaction_manager.clear_completed_transactions()
            
            # 清理DMA历史
            self._dma_controller.clear_history()
            
            print(f"CDMA系统 {self._chip_id}: 清理完成")
    
    def shutdown(self):
        """关闭系统"""
        with self._lock:
            print(f"CDMA系统 {self._chip_id}: 开始关闭...")
            
            # 停止DMA控制器
            self._dma_controller.stop()
            
            # 清理资源
            self.cleanup()
            
            # 断开所有连接
            self._connected_systems.clear()
            
            self._state = CDMASystemState.IDLE
            
            print(f"CDMA系统 {self._chip_id}: 已关闭")
    
    @property
    def chip_id(self) -> str:
        return self._chip_id
    
    @property
    def state(self) -> CDMASystemState:
        return self._state