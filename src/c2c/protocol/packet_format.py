"""
包格式定义模块
建立标准化的CDMA包格式，包含包头、包体结构和校验机制
"""

from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass, field
from enum import Enum
import struct
import zlib
import time
import json
from io import BytesIO

from src.c2c.utils.exceptions import CDMAError

# 定义固定长度的字符串字段长度
SOURCE_ID_LEN = 32
DEST_ID_LEN = 32
PACKET_TYPE_LEN = 16
COMPRESSION_TYPE_LEN = 16
TRANSACTION_ID_LEN = 32


class PacketType(Enum):
    """包类型枚举"""

    DATA = "data"  # 数据包
    CONTROL = "control"  # 控制包
    SYNC = "sync"  # 同步包
    CREDIT = "credit"  # Credit包
    ACK = "ack"  # 确认包
    NACK = "nack"  # 否认包
    HEARTBEAT = "heartbeat"  # 心跳包
    ERROR = "error"  # 错误包


class DataType(Enum):
    """数据类型枚举"""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT8 = "uint8"


class CompressionType(Enum):
    """压缩类型枚举"""

    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    SNAPPY = "snappy"


@dataclass
class PacketHeader:
    """包头结构"""

    # 基本标识信息
    source_id: str  # 源芯片ID
    dest_id: str  # 目标芯片ID
    packet_type: PacketType  # 包类型
    sequence_number: int  # 序列号

    # 包大小信息
    header_size: int = 168  # 包头大小（字节）
    payload_size: int = 0  # 载荷大小（字节）
    total_size: int = 0  # 总大小（字节）

    # 时间戳信息
    timestamp: float = 0.0  # 时间戳

    # 校验信息
    header_checksum: int = 0  # 包头校验和
    payload_checksum: int = 0  # 载荷校验和

    # 协议版本和标志
    version: int = 1  # 协议版本
    flags: int = 0  # 标志位

    # 可选字段
    transaction_id: Optional[str] = None  # 事务ID
    compression: CompressionType = CompressionType.NONE  # 压缩类型

    def __post_init__(self):
        """后处理，计算总大小"""
        self.total_size = self.header_size + self.payload_size
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def calculate_header_checksum(self) -> int:
        """计算包头校验和"""
        # 将包头转换为字节序列（排除校验和字段）
        header_data = self._pack_header_for_checksum()
        return zlib.crc32(header_data) & 0xFFFFFFFF

    def _pack_header_for_checksum(self) -> bytes:
        """打包包头用于校验和计算"""
        # 简化的格式，排除校验和字段
        data = []
        data.extend(self.source_id.encode("utf-8")[:SOURCE_ID_LEN].ljust(SOURCE_ID_LEN, b"\x00"))
        data.extend(self.dest_id.encode("utf-8")[:DEST_ID_LEN].ljust(DEST_ID_LEN, b"\x00"))
        data.extend(self.packet_type.value.encode("utf-8")[:PACKET_TYPE_LEN].ljust(PACKET_TYPE_LEN, b"\x00"))
        data.extend(struct.pack("!I", self.sequence_number))
        data.extend(struct.pack("!I", self.header_size))
        data.extend(struct.pack("!I", self.payload_size))
        data.extend(struct.pack("!I", self.total_size))
        data.extend(struct.pack("!d", self.timestamp))
        data.extend(struct.pack("!I", self.version))
        data.extend(struct.pack("!I", self.flags))
        data.extend(self.compression.value.encode("utf-8")[:COMPRESSION_TYPE_LEN].ljust(COMPRESSION_TYPE_LEN, b"\x00"))
        data.extend((self.transaction_id or "").encode("utf-8")[:TRANSACTION_ID_LEN].ljust(TRANSACTION_ID_LEN, b"\x00"))

        return bytes(data)

    def validate(self) -> bool:
        """验证包头完整性"""
        calculated_checksum = self.calculate_header_checksum()
        return calculated_checksum == self.header_checksum

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "source_id": self.source_id,
            "dest_id": self.dest_id,
            "packet_type": self.packet_type.value,
            "sequence_number": self.sequence_number,
            "header_size": self.header_size,
            "payload_size": self.payload_size,
            "total_size": self.total_size,
            "timestamp": self.timestamp,
            "header_checksum": self.header_checksum,
            "payload_checksum": self.payload_checksum,
            "version": self.version,
            "flags": self.flags,
            "transaction_id": self.transaction_id,
            "compression": self.compression.value,
        }


@dataclass
class AddressInfo:
    """地址信息结构"""

    base_address: int  # 基地址
    shape: Tuple[int, ...]  # 数据形状
    stride: Tuple[int, ...] = field(default_factory=tuple)  # 步长
    data_type: DataType = DataType.FLOAT32  # 数据类型
    memory_type: str = "GMEM"  # 内存类型

    def element_count(self) -> int:
        """计算元素总数"""
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def size_bytes(self) -> int:
        """计算数据大小（字节）"""
        type_sizes = {DataType.FLOAT32: 4, DataType.FLOAT16: 2, DataType.INT32: 4, DataType.INT16: 2, DataType.INT8: 1, DataType.UINT8: 1}
        return self.element_count() * type_sizes.get(self.data_type, 4)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "base_address": self.base_address,
            "shape": self.shape,
            "stride": self.stride,
            "data_type": self.data_type.value,
            "memory_type": self.memory_type,
            "element_count": self.element_count(),
            "size_bytes": self.size_bytes(),
        }


@dataclass
class PacketPayload:
    """包体结构"""

    # 控制信息
    control_info: Dict[str, Any] = field(default_factory=dict)

    # 地址信息
    src_address_info: Optional[AddressInfo] = None
    dst_address_info: Optional[AddressInfo] = None

    # 实际数据
    data: Optional[bytes] = None
    data_size: int = 0

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    # All Reduce参数
    reduce_operation: str = "none"  # sum, mean, max, min, none
    reduce_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """后处理，计算数据大小"""
        if self.data:
            self.data_size = len(self.data)

    def calculate_checksum(self) -> int:
        """计算载荷校验和"""
        # 将载荷转换为字节序列
        payload_bytes = self._serialize_for_checksum()
        return zlib.crc32(payload_bytes) & 0xFFFFFFFF

    def _serialize_for_checksum(self) -> bytes:
        """序列化载荷用于校验和计算"""
        # 创建一个包含所有载荷数据的字节序列
        buffer = BytesIO()

        # 控制信息 - 使用确定性序列化
        control_json = json.dumps(self.control_info, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        buffer.write(len(control_json).to_bytes(4, "big"))
        buffer.write(control_json)

        # 地址信息
        if self.src_address_info:
            src_json = json.dumps(self.src_address_info.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            buffer.write(len(src_json).to_bytes(4, "big"))
            buffer.write(src_json)
        else:
            buffer.write((0).to_bytes(4, "big"))

        if self.dst_address_info:
            dst_json = json.dumps(self.dst_address_info.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            buffer.write(len(dst_json).to_bytes(4, "big"))
            buffer.write(dst_json)
        else:
            buffer.write((0).to_bytes(4, "big"))

        # 实际数据
        if self.data:
            buffer.write(len(self.data).to_bytes(4, "big"))
            buffer.write(self.data)
        else:
            buffer.write((0).to_bytes(4, "big"))

        # 元数据
        metadata_json = json.dumps(self.metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        buffer.write(len(metadata_json).to_bytes(4, "big"))
        buffer.write(metadata_json)

        # Reduce参数
        reduce_json = json.dumps({"operation": self.reduce_operation, "params": self.reduce_params}, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        buffer.write(len(reduce_json).to_bytes(4, "big"))
        buffer.write(reduce_json)

        return buffer.getvalue()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "control_info": self.control_info,
            "src_address_info": self.src_address_info.to_dict() if self.src_address_info else None,
            "dst_address_info": self.dst_address_info.to_dict() if self.dst_address_info else None,
            "data_size": self.data_size,
            "metadata": self.metadata,
            "reduce_operation": self.reduce_operation,
            "reduce_params": self.reduce_params,
        }


class CDMAPacket:
    """CDMA包的完整定义"""

    def __init__(self, header: PacketHeader, payload: PacketPayload):
        self.header = header
        self.payload = payload

        # 更新包头中的载荷大小
        self._update_header_payload_size()

        # 计算载荷校验和
        self.header.payload_checksum = self.payload.calculate_checksum()
        # 计算包头校验和 (在payload_checksum确定后)
        self.header.header_checksum = self.header.calculate_header_checksum()

        # 缓存序列化后的载荷数据，确保校验和一致性
        self._serialized_payload = None

    def _update_header_payload_size(self):
        """更新包头中的载荷大小信息"""
        payload_bytes = self.payload._serialize_for_checksum()
        self.header.payload_size = len(payload_bytes)
        self.header.total_size = self.header.header_size + self.header.payload_size

    def validate(self) -> bool:
        """验证包的完整性"""
        # 验证包头
        if not self.header.validate():
            return False

        # 验证载荷校验和
        # 如果有缓存的序列化数据，使用缓存数据计算校验和
        if hasattr(self, "_serialized_payload") and self._serialized_payload is not None:
            calculated_payload_checksum = zlib.crc32(self._serialized_payload) & 0xFFFFFFFF
        else:
            calculated_payload_checksum = self.payload.calculate_checksum()

        if calculated_payload_checksum != self.header.payload_checksum:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {"header": self.header.to_dict(), "payload": self.payload.to_dict(), "is_valid": self.validate()}

    def __str__(self) -> str:
        """字符串表示"""
        return f"CDMAPacket(type={self.header.packet_type.value}, " f"src={self.header.source_id}, dst={self.header.dest_id}, " f"seq={self.header.sequence_number}, size={self.header.total_size})"


class PacketSerializer:
    """包的序列化工具"""

    @staticmethod
    def serialize_packet(packet: CDMAPacket) -> bytes:
        """序列化CDMA包为字节流"""
        try:
            # 序列化载荷 - 先序列化载荷以确保一致性
            payload_bytes = PacketSerializer._serialize_payload(packet.payload)

            # 缓存序列化后的载荷数据
            packet._serialized_payload = payload_bytes

            # 应用压缩（如果指定）
            if packet.header.compression != CompressionType.NONE:
                payload_bytes = PacketSerializer._compress_data(payload_bytes, packet.header.compression)
                # 更新压缩后的大小
                packet.header.payload_size = len(payload_bytes)
                packet.header.total_size = packet.header.header_size + packet.header.payload_size
                # 重新计算包头校验和（因为大小字段改变了）
                packet.header.header_checksum = packet.header.calculate_header_checksum()

            # 序列化包头
            header_bytes = PacketSerializer._serialize_header(packet.header)

            return header_bytes + payload_bytes

        except Exception as e:
            raise CDMAError(f"包序列化失败: {e}")

    @staticmethod
    def deserialize_packet(data: bytes) -> CDMAPacket:
        """从字节流反序列化CDMA包"""
        try:
            if len(data) < 168:  # 最小包头大小
                raise CDMAError("数据长度不足，无法解析包头")

            # 反序列化包头
            header = PacketSerializer._deserialize_header(data[:168])

            # 检查数据长度
            if len(data) < header.total_size:
                raise CDMAError(f"数据长度不足: 期望{header.total_size}, 实际{len(data)}")

            # 提取载荷数据
            payload_data = data[header.header_size : header.total_size]

            # 应用解压缩（如果需要）
            if header.compression != CompressionType.NONE:
                payload_data = PacketSerializer._decompress_data(payload_data, header.compression)

            # 反序列化载荷
            payload = PacketSerializer._deserialize_payload(payload_data)

            # 创建包对象，但不重新计算校验和
            packet = CDMAPacket.__new__(CDMAPacket)
            packet.header = header
            packet.payload = payload
            packet._serialized_payload = payload_data  # 保存原始载荷数据

            # 验证包完整性 - 使用原始数据进行校验
            calculated_payload_checksum = zlib.crc32(payload_data) & 0xFFFFFFFF
            if calculated_payload_checksum != header.payload_checksum:
                raise CDMAError(f"载荷校验和验证失败: 期望={header.payload_checksum}, 实际={calculated_payload_checksum}")

            return packet

        except Exception as e:
            raise CDMAError(f"包反序列化失败: {e}")

    @staticmethod
    def _serialize_header(header: PacketHeader) -> bytes:
        """序列化包头"""
        # 使用固定128字节包头格式
        data = []
        data.extend(header.source_id.encode("utf-8")[:SOURCE_ID_LEN].ljust(SOURCE_ID_LEN, b"\x00"))
        data.extend(header.dest_id.encode("utf-8")[:DEST_ID_LEN].ljust(DEST_ID_LEN, b"\x00"))
        data.extend(header.packet_type.value.encode("utf-8")[:PACKET_TYPE_LEN].ljust(PACKET_TYPE_LEN, b"\x00"))
        data.extend(struct.pack("!I", header.sequence_number))
        data.extend(struct.pack("!I", header.header_size))
        data.extend(struct.pack("!I", header.payload_size))
        data.extend(struct.pack("!I", header.total_size))
        data.extend(struct.pack("!d", header.timestamp))
        data.extend(struct.pack("!I", header.version))
        data.extend(struct.pack("!I", header.flags))
        data.extend(header.compression.value.encode("utf-8")[:COMPRESSION_TYPE_LEN].ljust(COMPRESSION_TYPE_LEN, b"\x00"))
        data.extend(struct.pack("!I", header.header_checksum))
        data.extend(struct.pack("!I", header.payload_checksum))
        data.extend((header.transaction_id or "").encode("utf-8")[:TRANSACTION_ID_LEN].ljust(TRANSACTION_ID_LEN, b"\x00"))

        # 返回实际包头数据
        result = bytes(data)
        return result

    @staticmethod
    def _deserialize_header(data: bytes) -> PacketHeader:
        """反序列化包头"""
        try:
            if len(data) < 168:  # 实际包头大小
                raise CDMAError("包头数据长度不足")

            offset = 0

            # 解析固定字段
            source_id = data[offset : offset + SOURCE_ID_LEN].rstrip(b"\x00").decode("utf-8")
            offset += SOURCE_ID_LEN

            dest_id = data[offset : offset + DEST_ID_LEN].rstrip(b"\x00").decode("utf-8")
            offset += DEST_ID_LEN

            packet_type_str = data[offset : offset + PACKET_TYPE_LEN].rstrip(b"\x00").decode("utf-8")
            offset += PACKET_TYPE_LEN

            sequence_number = struct.unpack("!I", data[offset : offset + 4])[0]
            offset += 4

            header_size = struct.unpack("!I", data[offset : offset + 4])[0]
            offset += 4

            payload_size = struct.unpack("!I", data[offset : offset + 4])[0]
            offset += 4

            total_size = struct.unpack("!I", data[offset : offset + 4])[0]
            offset += 4

            timestamp = struct.unpack("!d", data[offset : offset + 8])[0]
            offset += 8

            version = struct.unpack("!I", data[offset : offset + 4])[0]
            offset += 4

            flags = struct.unpack("!I", data[offset : offset + 4])[0]
            offset += 4

            compression_str = data[offset : offset + COMPRESSION_TYPE_LEN].rstrip(b"\x00").decode("utf-8")
            offset += COMPRESSION_TYPE_LEN

            header_checksum = struct.unpack("!I", data[offset : offset + 4])[0]
            offset += 4

            payload_checksum = struct.unpack("!I", data[offset : offset + 4])[0]
            offset += 4

            transaction_id = data[offset : offset + TRANSACTION_ID_LEN].rstrip(b"\x00").decode("utf-8") or None

            # 转换枚举类型
            packet_type = PacketType.DATA
            for pt in PacketType:
                if pt.value == packet_type_str:
                    packet_type = pt
                    break

            compression = CompressionType.NONE
            for ct in CompressionType:
                if ct.value == compression_str:
                    compression = ct
                    break

            header = PacketHeader(
                source_id=source_id,
                dest_id=dest_id,
                packet_type=packet_type,
                sequence_number=sequence_number,
                header_size=header_size,
                payload_size=payload_size,
                total_size=total_size,
                timestamp=timestamp,
                version=version,
                flags=flags,
                compression=compression,
                header_checksum=header_checksum,
                payload_checksum=payload_checksum,
                transaction_id=transaction_id,
            )

            return header

        except Exception as e:
            raise CDMAError(f"包头反序列化失败: {e}")

    @staticmethod
    def _serialize_payload(payload: PacketPayload) -> bytes:
        """序列化载荷"""
        return payload._serialize_for_checksum()

    @staticmethod
    def _deserialize_payload(data: bytes) -> PacketPayload:
        """反序列化载荷"""
        try:
            buffer = BytesIO(data)

            # 读取控制信息
            control_size = int.from_bytes(buffer.read(4), "big")
            control_bytes = buffer.read(control_size).decode("utf-8")
            control_info = json.loads(control_bytes)  # 使用安全的json.loads替代eval

            # 读取源地址信息
            src_size = int.from_bytes(buffer.read(4), "big")
            src_address_info = None
            if src_size > 0:
                src_bytes = buffer.read(src_size).decode("utf-8")
                src_dict = json.loads(src_bytes)  # 使用安全的json.loads替代eval
                src_address_info = AddressInfo(
                    base_address=src_dict["base_address"],
                    shape=tuple(src_dict["shape"]),
                    stride=tuple(src_dict["stride"]),
                    data_type=DataType(src_dict["data_type"]),
                    memory_type=src_dict["memory_type"],
                )

            # 读取目标地址信息
            dst_size = int.from_bytes(buffer.read(4), "big")
            dst_address_info = None
            if dst_size > 0:
                dst_bytes = buffer.read(dst_size).decode("utf-8")
                dst_dict = json.loads(dst_bytes)  # 使用安全的json.loads替代eval
                dst_address_info = AddressInfo(
                    base_address=dst_dict["base_address"],
                    shape=tuple(dst_dict["shape"]),
                    stride=tuple(dst_dict["stride"]),
                    data_type=DataType(dst_dict["data_type"]),
                    memory_type=dst_dict["memory_type"],
                )

            # 读取实际数据
            data_size = int.from_bytes(buffer.read(4), "big")
            actual_data = None
            if data_size > 0:
                actual_data = buffer.read(data_size)

            # 读取元数据
            metadata_size = int.from_bytes(buffer.read(4), "big")
            metadata_bytes = buffer.read(metadata_size).decode("utf-8")
            metadata = json.loads(metadata_bytes)  # 使用安全的json.loads替代eval

            # 读取Reduce参数
            reduce_size = int.from_bytes(buffer.read(4), "big")
            reduce_bytes = buffer.read(reduce_size).decode("utf-8")
            reduce_dict = json.loads(reduce_bytes)  # 使用安全的json.loads替代eval

            payload = PacketPayload(
                control_info=control_info,
                src_address_info=src_address_info,
                dst_address_info=dst_address_info,
                data=actual_data,
                data_size=data_size,
                metadata=metadata,
                reduce_operation=reduce_dict["operation"],
                reduce_params=reduce_dict["params"],
            )

            return payload

        except Exception as e:
            raise CDMAError(f"载荷反序列化失败: {e}")

    @staticmethod
    def _compress_data(data: bytes, compression: CompressionType) -> bytes:
        """压缩数据"""
        if compression == CompressionType.ZLIB:
            return zlib.compress(data)
        elif compression == CompressionType.LZ4:
            # 这里需要导入lz4库，暂时使用zlib替代
            return zlib.compress(data)
        elif compression == CompressionType.SNAPPY:
            # 这里需要导入python-snappy库，暂时使用zlib替代
            return zlib.compress(data)
        else:
            return data

    @staticmethod
    def _decompress_data(data: bytes, compression: CompressionType) -> bytes:
        """解压缩数据"""
        if compression == CompressionType.ZLIB:
            return zlib.decompress(data)
        elif compression == CompressionType.LZ4:
            # 这里需要导入lz4库，暂时使用zlib替代
            return zlib.decompress(data)
        elif compression == CompressionType.SNAPPY:
            # 这里需要导入python-snappy库，暂时使用zlib替代
            return zlib.decompress(data)
        else:
            return data


class PacketFactory:
    """包工厂类，用于创建不同类型的包"""

    @staticmethod
    def create_data_packet(
        source_id: str, dest_id: str, sequence_number: int, src_address_info: AddressInfo, dst_address_info: AddressInfo, data: bytes, transaction_id: str = None, reduce_operation: str = "none"
    ) -> CDMAPacket:
        """创建数据包"""
        header = PacketHeader(source_id=source_id, dest_id=dest_id, packet_type=PacketType.DATA, sequence_number=sequence_number, transaction_id=transaction_id)

        payload = PacketPayload(src_address_info=src_address_info, dst_address_info=dst_address_info, data=data, reduce_operation=reduce_operation)

        return CDMAPacket(header, payload)

    @staticmethod
    def create_control_packet(source_id: str, dest_id: str, sequence_number: int, control_info: Dict[str, Any], transaction_id: str = None) -> CDMAPacket:
        """创建控制包"""
        header = PacketHeader(source_id=source_id, dest_id=dest_id, packet_type=PacketType.CONTROL, sequence_number=sequence_number, transaction_id=transaction_id)

        payload = PacketPayload(control_info=control_info)

        return CDMAPacket(header, payload)

    @staticmethod
    def create_sync_packet(source_id: str, dest_id: str, sequence_number: int, sync_info: Dict[str, Any], transaction_id: str = None) -> CDMAPacket:
        """创建同步包"""
        header = PacketHeader(source_id=source_id, dest_id=dest_id, packet_type=PacketType.SYNC, sequence_number=sequence_number, transaction_id=transaction_id)

        payload = PacketPayload(control_info=sync_info)

        return CDMAPacket(header, payload)

    @staticmethod
    def create_ack_packet(source_id: str, dest_id: str, sequence_number: int, ack_info: Dict[str, Any], transaction_id: str = None) -> CDMAPacket:
        """创建确认包"""
        header = PacketHeader(source_id=source_id, dest_id=dest_id, packet_type=PacketType.ACK, sequence_number=sequence_number, transaction_id=transaction_id)

        payload = PacketPayload(control_info=ack_info)

        return CDMAPacket(header, payload)
