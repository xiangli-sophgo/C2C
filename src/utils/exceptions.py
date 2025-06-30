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
