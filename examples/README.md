# NoC Traffic Simulation Examples

This directory contains examples demonstrating how to run NoC (Network on Chip) simulations using traffic files with the C2C framework.

## Available Examples

### 1. basic_traffic_example.py
**Purpose**: Simple traffic processing example using minimal dependencies
- Uses a simple configuration class
- Demonstrates basic traffic file parsing and processing
- Shows traffic scheduler usage with cycle-by-cycle simulation
- Minimal NoC model dependencies

**Usage**:
```bash
python examples/basic_traffic_example.py
```

### 2. simple_traffic_example.py
**Purpose**: CrossRing NoC simulation with real configuration
- Uses actual CrossRingConfig class
- Demonstrates integration with the NoC configuration system
- Shows basic simulation workflow
- Compatible with existing traffic file format

**Usage**:
```bash
python examples/simple_traffic_example.py
```

### 3. crossring_noc_traffic_example.py
**Purpose**: Complete CrossRing NoC traffic simulation
- Full-featured example with comprehensive statistics
- Detailed progress reporting and configuration summary
- Shows all features of the traffic scheduler
- Demonstrates proper CrossRing configuration usage

**Usage**:
```bash
python examples/crossring_noc_traffic_example.py
```

## Traffic File Format

All examples use traffic files with the following CSV format:
```
time,src,src_type,dst,dst_type,op,burst
0,0,gdma_0,1,ddr_0,R,4
16,0,gdma_0,1,ddr_0,R,4
...
```

Where:
- `time`: Request time in nanoseconds
- `src`: Source node ID
- `src_type`: Source component type (e.g., gdma_0, sdma_1)
- `dst`: Destination node ID  
- `dst_type`: Destination component type (e.g., ddr_0, l2m_1)
- `op`: Operation type ('R' for read, 'W' for write)
- `burst`: Number of flits in the burst

## Example Traffic File

The examples use `src_old/example/test_data.txt` which contains sample traffic patterns for demonstration.

## Key Features Demonstrated

### Traffic Scheduler Usage
- Setting up parallel or serial traffic chains
- Reading and parsing traffic files
- Cycle-by-cycle request injection
- Traffic completion tracking
- Statistics collection

### Configuration Management
- CrossRing topology configuration
- Network frequency settings
- Resource allocation (trackers, buffers)
- IP interface configuration

### Simulation Flow
- Traffic initialization
- Cycle-based simulation loop
- Request processing and injection
- Progress monitoring
- Results analysis

## Output Examples

All examples provide detailed output including:
- Configuration summary
- Traffic loading information
- Real-time simulation progress
- Final statistics (requests, latency, completion times)
- Resource utilization summary

## Extending the Examples

To create your own traffic simulation:

1. **Prepare traffic file**: Create a CSV file following the format above
2. **Configure topology**: Set up CrossRingConfig with desired parameters
3. **Setup scheduler**: Initialize TrafficScheduler with your traffic files
4. **Run simulation**: Implement the cycle-based simulation loop
5. **Collect results**: Extract statistics and analyze performance

## Dependencies

- Python 3.9+
- NumPy
- SciPy
- Custom C2C NoC framework modules

## Notes

- The examples are compatible with the existing C2C codebase structure
- Traffic files from `src_old/example` are used for backward compatibility
- All examples handle the frequency conversion between the old and new configuration systems
- The simulation is cycle-accurate and supports detailed performance analysis