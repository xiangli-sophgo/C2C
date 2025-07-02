# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

C2C is a topology modeling framework chip-to-chip communication. It provides an object-oriented architecture for modeling various interconnect topologies (cascade, switch-based) and communication protocols (CDMA). The framework is designed to simulate chip-to-chip communication in multi-chip systems.

## Architecture

The codebase follows a layered architecture with clear separation of concerns:

### Core Components

1. **Topology Layer** (`topology/`):
   - `base.py`: Abstract base classes (BaseNode, BaseLink, BaseTopology) defining the fundamental interfaces
   - `node.py`: Concrete node implementations (ChipNode, SwitchNode, HostNode)
   - `link.py`: Link implementations (PCIeLink, C2CDirectLink)
   - `graph.py`: NetworkX-based topology graph representation with pathfinding and visualization
   - `builder.py`: Builder pattern for constructing topologies

2. **Protocol Layer** (`protocol/`):
   - `base.py`: Protocol state management and abstract interfaces
   - `cdma.py`: CDMA protocol implementation with message types and state transitions
   - `credit.py`: Credit-based flow control mechanism
   - `address.py`: Address space translation between different formats
   - `router.py`: Message routing with topology-aware path calculation

3. **Support Modules**:
   - `utils/`: Constants and custom exceptions
   - `config/`: Configuration loading utilities
   - `examples/`: Demonstration scripts

### Key Design Patterns

- **Abstract Base Classes**: Extensive use of ABC for defining interfaces
- **Builder Pattern**: TopologyBuilder for constructing complex topologies
- **Strategy Pattern**: Different node types and link types with common interfaces
- **State Machine**: Protocol state management in BaseProtocol

## Common Development Tasks

### Running the Basic Demo
```bash
# Install dependencies first
pip install networkx matplotlib pyyaml streamlit plotly seaborn pandas numpy

# Run the basic demonstration
python3 examples/basic_demo.py
```

### Using Visualization Tools
```bash
# Run static visualization demo
python3 examples/visualization_demo.py

# Launch interactive web interface
python3 run_webapp.py
# or
streamlit run visualization/interactive.py

# Run topology validation tests
python3 examples/tree_torus_validation.py
```

### Testing Topologies
The framework supports creating various topology types:
- Cascade topologies: Direct chip-to-chip connections
- Switch-based topologies: PCIe switches connecting multiple chips
- Mixed topologies: Combination of direct and switch connections

### Key Classes to Understand

- **TopologyGraph**: Central class for topology representation using NetworkX
- **TopologyBuilder**: Main interface for constructing topologies
- **CDMAProtocol**: Core communication protocol implementation
- **CreditManager**: Flow control mechanism
- **Router**: Message routing with path calculation

### Visualization Classes

- **TopologyVisualizer**: Main visualization engine for all topology types
- **TreeLayout/TorusLayout**: Specialized layout algorithms for different topologies
- **PerformanceComparator**: Tool for comparing topology performance metrics
- **Interactive Web App**: Streamlit-based GUI for real-time topology exploration

## Development Guidelines

### Module Dependencies
- Core dependencies: `networkx`, `matplotlib`, `pyyaml`
- Standard library: `abc`, `enum`, `typing`, `dataclasses`

### Code Style
- Follows PEP 8 conventions
- Uses type hints throughout
- Comprehensive docstrings for all public methods
- Abstract base classes for extensibility

### Testing
Run the demo script to verify core functionality:
```bash
python examples/basic_demo.py
```

The demo validates:
- Node and link creation
- Topology construction
- Path finding algorithms
- CDMA protocol state transitions
- Credit management
- Address translation
- Message routing

## Architecture Notes

- The framework is designed for extensibility - new node types, link types, and protocols can be added by inheriting from base classes
- NetworkX is used for graph operations and pathfinding algorithms
- State management is centralized in protocol classes
- The builder pattern ensures consistent topology construction
- Address translation supports multiple address formats for different communication contexts