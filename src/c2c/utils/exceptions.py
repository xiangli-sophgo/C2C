class C2CModelingError(Exception):
    """Base exception for C2C Modeling errors."""
    pass

class NodeNotFoundError(C2CModelingError):
    """Exception raised when a specified node is not found."""
    def __init__(self, node_id: str, message="Node not found"):
        self.node_id = node_id
        super().__init__(f"{message}: {node_id}")

class LinkNotFoundError(C2CModelingError):
    """Exception raised when a specified link is not found."""
    def __init__(self, link_id: str, message="Link not found"):
        self.link_id = link_id
        super().__init__(f"{message}: {link_id}")

class InvalidTopologyError(C2CModelingError):
    """Exception raised for invalid topology configurations."""
    pass

class ProtocolError(C2CModelingError):
    """Base exception for protocol-related errors."""
    pass

class CreditError(ProtocolError):
    """Exception raised for credit management errors."""
    pass

class CDMAError(ProtocolError):
    """Exception raised for CDMA protocol errors."""
    pass

class AddressError(CDMAError):
    """Exception raised for address-related errors."""
    def __init__(self, address: int, message="Address error"):
        self.address = address
        super().__init__(f"{message}: 0x{address:08x}")

class ShapeCompatibilityError(CDMAError):
    """Exception raised for tensor shape compatibility errors."""
    def __init__(self, src_shape, dst_shape, message="Shape compatibility error"):
        self.src_shape = src_shape
        self.dst_shape = dst_shape
        super().__init__(f"{message}: src_shape={src_shape}, dst_shape={dst_shape}")

class MemoryTypeError(CDMAError):
    """Exception raised for memory type errors."""
    def __init__(self, mem_type: str, message="Memory type error"):
        self.mem_type = mem_type
        super().__init__(f"{message}: {mem_type}")

class TransactionError(CDMAError):
    """Exception raised for transaction management errors."""
    def __init__(self, transaction_id: str, message="Transaction error"):
        self.transaction_id = transaction_id
        super().__init__(f"{message}: {transaction_id}")

class DMATransferError(CDMAError):
    """Exception raised for DMA transfer errors."""
    pass
