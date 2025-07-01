from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import copy
from utils.exceptions import CDMAError
from .memory_types import MemoryType


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
        
        type_sizes = {
            "float32": 4, "float16": 2, "int32": 4, 
            "int16": 2, "int8": 1, "uint8": 1
        }
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


class CDMAEngine:
    """CDMA硬件引擎模拟器"""
    
    def __init__(self, chip_id: str):
        self._chip_id = chip_id
        self._credit_with_address: Dict[str, CreditWithAddress] = {}  # src_chip -> credit_info
        self._pending_receives: Dict[str, AddressInfo] = {}  # transaction_id -> dst_address_info
        self._active_transactions: Dict[str, DMATransaction] = {}
        self._transaction_counter = 0
        self._memory_simulator: Dict[int, bytes] = {}  # 模拟内存空间
        
    def _generate_transaction_id(self) -> str:
        """生成事务ID"""
        self._transaction_counter += 1
        return f"cdma_{self._chip_id}_{self._transaction_counter}_{int(time.time() * 1000000)}"
    
    def cdma_receive(self, dst_addr: int, dst_shape: Tuple[int, ...], 
                    dst_mem_type: MemoryType, src_chip_id: str,
                    data_type: str = "float32") -> str:
        """
        执行CDMA接收操作
        
        Args:
            dst_addr: 目标地址
            dst_shape: 目标tensor形状
            dst_mem_type: 目标内存类型
            src_chip_id: 源芯片ID
            data_type: 数据类型
            
        Returns:
            transaction_id: 事务ID
        """
        transaction_id = self._generate_transaction_id()
        
        # 创建目标地址信息
        dst_address_info = AddressInfo(
            address=dst_addr,
            shape=dst_shape,
            mem_type=dst_mem_type,
            data_type=data_type
        )
        
        # 存储接收请求
        self._pending_receives[transaction_id] = dst_address_info
        
        # 发送Credit + 地址信息给源芯片
        credit_info = CreditWithAddress(
            credit_count=1,
            dst_address_info=dst_address_info,
            transaction_id=transaction_id,
            timestamp=time.time()
        )
        
        # 模拟发送Credit到源芯片（实际系统中这里会通过硬件总线发送）
        self._send_credit_to_chip(src_chip_id, credit_info)
        
        print(f"芯片 {self._chip_id}: 执行CDMA_receive，向 {src_chip_id} 发送Credit+地址信息")
        print(f"  目标地址: 0x{dst_addr:08x}, 形状: {dst_shape}, 内存类型: {dst_mem_type.value}")
        print(f"  事务ID: {transaction_id}")
        
        return transaction_id
    
    def cdma_send(self, src_addr: int, src_shape: Tuple[int, ...], 
                 dst_chip_id: str, data_type: str = "float32") -> Optional[str]:
        """
        执行CDMA发送操作
        
        Args:
            src_addr: 源地址
            src_shape: 源tensor形状
            dst_chip_id: 目标芯片ID
            data_type: 数据类型
            
        Returns:
            transaction_id: 成功时返回事务ID，失败时返回None
        """
        # 检查是否有来自目标芯片的Credit
        if dst_chip_id not in self._credit_with_address:
            print(f"芯片 {self._chip_id}: 没有来自 {dst_chip_id} 的Credit，等待CDMA_receive")
            return None
        
        credit_info = self._credit_with_address[dst_chip_id]
        
        # 验证形状兼容性
        src_address_info = AddressInfo(
            address=src_addr,
            shape=src_shape,
            mem_type=MemoryType.GMEM,  # 假设源数据在GMEM
            data_type=data_type
        )
        
        if not self._validate_address_compatibility(src_address_info, credit_info.dst_address_info):
            raise CDMAError(f"地址兼容性检查失败: src_shape={src_shape}, dst_shape={credit_info.dst_address_info.shape}")
        
        # 消费Credit
        transaction_id = credit_info.transaction_id
        dst_address_info = credit_info.dst_address_info
        del self._credit_with_address[dst_chip_id]
        
        # 创建DMA事务
        dma_transaction = DMATransaction(
            transaction_id=transaction_id,
            src_chip_id=self._chip_id,
            dst_chip_id=dst_chip_id,
            src_address_info=src_address_info,
            dst_address_info=dst_address_info,
            status="transferring",
            created_time=time.time()
        )
        
        self._active_transactions[transaction_id] = dma_transaction
        
        # 执行DMA传输
        success = self._execute_dma_transfer(src_address_info, dst_address_info, dst_chip_id)
        
        if success:
            dma_transaction.status = "completed"
            dma_transaction.completed_time = time.time()
            print(f"芯片 {self._chip_id}: CDMA_send完成，数据传输到 {dst_chip_id}")
            print(f"  源地址: 0x{src_addr:08x} -> 目标地址: 0x{dst_address_info.address:08x}")
            print(f"  数据大小: {src_address_info.size_bytes()} bytes")
        else:
            dma_transaction.status = "failed"
            print(f"芯片 {self._chip_id}: CDMA_send失败")
        
        return transaction_id
    
    def cdma_sys_send_msg(self, target_chip_id: str, message: str = "sync") -> bool:
        """
        发送同步消息
        
        Args:
            target_chip_id: 目标芯片ID
            message: 同步消息内容
            
        Returns:
            bool: 发送是否成功
        """
        print(f"芯片 {self._chip_id}: 向 {target_chip_id} 发送同步消息: {message}")
        # 在实际系统中，这里会通过硬件总线发送同步消息
        return True
    
    def _send_credit_to_chip(self, target_chip_id: str, credit_info: CreditWithAddress):
        """发送Credit信息到目标芯片（模拟）"""
        # 在实际系统中，这里会通过硬件总线发送Credit信息
        # 这里我们模拟目标芯片接收Credit信息
        pass
    
    def receive_credit_from_chip(self, src_chip_id: str, credit_info: CreditWithAddress):
        """从其他芯片接收Credit信息"""
        self._credit_with_address[src_chip_id] = credit_info
        print(f"芯片 {self._chip_id}: 收到来自 {src_chip_id} 的Credit+地址信息")
    
    def _validate_address_compatibility(self, src_info: AddressInfo, dst_info: AddressInfo) -> bool:
        """验证源和目标地址兼容性"""
        # 检查形状是否匹配
        if src_info.shape != dst_info.shape:
            return False
        
        # 检查数据类型是否匹配
        if src_info.data_type != dst_info.data_type:
            return False
        
        # 简化地址对齐检查（用于测试）
        if src_info.address % 8 != 0 or dst_info.address % 8 != 0:
            return False
        
        return True
    
    def _execute_dma_transfer(self, src_info: AddressInfo, dst_info: AddressInfo, dst_chip_id: str) -> bool:
        """执行DMA数据传输（模拟）"""
        try:
            # 模拟从源地址读取数据
            data_size = src_info.size_bytes()
            
            # 检查源地址是否有数据（在实际系统中这是硬件操作）
            if src_info.address in self._memory_simulator:
                src_data = self._memory_simulator[src_info.address]
            else:
                # 模拟数据生成
                src_data = bytes(range(data_size % 256)) * (data_size // 256 + 1)
                src_data = src_data[:data_size]
                self._memory_simulator[src_info.address] = src_data
            
            # 模拟DMA传输延迟
            transfer_time = data_size / (10 * 1024 * 1024 * 1024)  # 假设10GB/s带宽
            time.sleep(min(transfer_time, 0.001))  # 最多延迟1ms
            
            # 在实际系统中，这里会通过C2C链路将数据传输到目标芯片
            print(f"DMA传输: {data_size} bytes from 0x{src_info.address:08x} to chip_{dst_chip_id}:0x{dst_info.address:08x}")
            
            return True
            
        except Exception as e:
            print(f"DMA传输失败: {str(e)}")
            return False
    
    def get_transaction_status(self, transaction_id: str) -> Optional[str]:
        """获取事务状态"""
        if transaction_id in self._active_transactions:
            return self._active_transactions[transaction_id].status
        return None
    
    def get_credit_status(self) -> Dict[str, Any]:
        """获取Credit状态"""
        return {
            "available_credits": {chip_id: info.credit_count for chip_id, info in self._credit_with_address.items()},
            "pending_receives": len(self._pending_receives),
            "active_transactions": len(self._active_transactions)
        }
    
    def cleanup_completed_transactions(self):
        """清理已完成的事务"""
        completed_transactions = [tid for tid, trans in self._active_transactions.items() 
                                if trans.status in ["completed", "failed"]]
        
        for tid in completed_transactions:
            del self._active_transactions[tid]
    
    @property
    def chip_id(self) -> str:
        return self._chip_id