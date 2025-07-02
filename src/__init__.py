"""
C2C Framework - 芯片间通信建模框架
含拓扑建模、协议实现和仿真引擎
"""

from . import topology
from . import protocol
from . import utils
from . import visualization
from . import simulation

from .topology.builder import TopologyBuilder
from .topology.node import ChipNode, SwitchNode
from .topology.link import C2CDirectLink, PCIeLink

from .protocol.cdma_system import CDMASystem
from .protocol.address import AddressTranslator

from .simulation.engine import C2CSimulationEngine
from .simulation.fake_chip import FakeChip
from .simulation.events import SimulationEvent, EventFactory

__version__ = "1.0.0"

__all__ = [
    'TopologyBuilder',
    'ChipNode', 
    'SwitchNode',
    'C2CDirectLink',
    'PCIeLink',
    
    'CDMASystem',
    'AddressTranslator',
    
    'C2CSimulationEngine',
    'FakeChip',
    'SimulationEvent',
    'EventFactory',
    
    'topology',
    'protocol', 
    'utils',
    'visualization',
    'simulation'
]