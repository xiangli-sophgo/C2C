#!/usr/bin/env python3
"""
Complete CrossRing NoC Traffic Simulation Example

This example demonstrates a complete CrossRing NoC simulation using traffic files,
similar to src_old/example but using the new NoC architecture and traffic scheduler.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.noc.crossring.config import CrossRingConfig
from src.noc.utils.traffic_scheduler import TrafficScheduler


def main():
    """Main simulation function"""
    print("CrossRing NoC Traffic Simulation Example")
    print("=" * 50)
    
    # Create CrossRing configuration
    config = CrossRingConfig(num_row=4, num_col=4, config_name="4x4_example")
    
    # Add compatibility attribute for TrafficScheduler
    config.NETWORK_FREQUENCY = int(config.basic_config.network_frequency * 1000000000)  # Convert to Hz
    
    print(f"Configuration: {config}")
    print(f"Topology: {config.num_row} x {config.num_col} = {config.num_nodes} nodes")
    print(f"Network frequency: {config.NETWORK_FREQUENCY} Hz")
    
    # Use existing traffic file from src_old/example
    traffic_file_path = os.path.join(os.path.dirname(__file__), "..", "src_old", "example")
    traffic_files = ["test_data.txt"]
    
    # Check if traffic file exists
    test_file = os.path.join(traffic_file_path, traffic_files[0])
    if not os.path.exists(test_file):
        print(f"Error: Traffic file not found: {test_file}")
        return
    
    print(f"Using traffic file: {test_file}")
    
    # Setup traffic scheduler
    traffic_scheduler = TrafficScheduler(config, traffic_file_path)
    traffic_scheduler.set_verbose(True)
    
    # Setup single chain with the traffic file
    traffic_scheduler.setup_single_chain(traffic_files)
    
    # Start initial traffic
    print("\n=== Initializing Traffic Scheduler ===")
    traffic_scheduler.start_initial_traffics()
    
    print(f"Traffic scheduler initialized")
    print(f"Active traffics: {traffic_scheduler.get_active_traffic_count()}")
    
    # Display chain status
    chain_status = traffic_scheduler.get_chain_status()
    for chain_id, status in chain_status.items():
        print(f"Chain {chain_id}: {status['total_traffics']} traffic files")
        print(f"  Current file: {status['current_file']}")
        print(f"  Pending requests: {status['pending_requests']}")
    
    # Simulation parameters
    max_cycles = 5000
    current_cycle = 0
    total_requests = 0
    injected_requests = 0
    
    print(f"\n=== Running Simulation ===")
    print(f"Max cycles: {max_cycles}")
    start_time = time.time()
    
    # Main simulation loop
    while current_cycle < max_cycles:
        # Get ready requests for current cycle
        ready_requests = traffic_scheduler.get_ready_requests(current_cycle)
        
        # Process ready requests
        for request in ready_requests:
            # Parse request: (time, src, src_type, dst, dst_type, op, burst, traffic_id)
            req_time, src, src_type, dst, dst_type, op, burst, traffic_id = request
            
            # Display detailed request info
            time_ns = req_time // config.NETWORK_FREQUENCY
            print(f"Cycle {current_cycle} ({time_ns}ns): {src}({src_type}) -> {dst}({dst_type}), {op}, burst={burst}")
            injected_requests += 1
            
            # Simulate packet processing and completion
            # In a real simulation, this would be handled by the NoC model
            for _ in range(burst):
                traffic_scheduler.update_traffic_stats(traffic_id, "received_flit")
        
        total_requests += len(ready_requests)
        
        # Check and advance traffic chains
        completed_traffics = traffic_scheduler.check_and_advance_chains(current_cycle)
        
        if completed_traffics:
            print(f"Completed traffics at cycle {current_cycle}: {completed_traffics}")
        
        # Progress reporting
        if current_cycle % 500 == 0 and current_cycle > 0:
            active_count = traffic_scheduler.get_active_traffic_count()
            print(f"Progress - Cycle {current_cycle}: Requests={total_requests}, Injected={injected_requests}, Active={active_count}")
        
        # Check if all traffic is completed
        if traffic_scheduler.is_all_completed():
            print(f"All traffic completed at cycle {current_cycle}")
            break
        
        current_cycle += 1
    
    end_time = time.time()
    simulation_time = end_time - start_time
    
    # Print comprehensive statistics
    print(f"\n=== Simulation Results ===")
    print(f"Simulation completed at cycle {current_cycle}")
    print(f"Total requests processed: {total_requests}")
    print(f"Injected requests: {injected_requests}")
    print(f"Simulation time: {simulation_time:.3f} seconds")
    print(f"Simulation rate: {current_cycle/simulation_time:.0f} cycles/second")
    
    # Get traffic completion statistics
    finish_stats = traffic_scheduler.get_finish_time_stats()
    print(f"\n=== Traffic Completion Stats ===")
    print(f"Read finish time: {finish_stats.get('R_finish_time', 0)} ns")
    print(f"Write finish time: {finish_stats.get('W_finish_time', 0)} ns")
    print(f"Total finish time: {finish_stats.get('Total_finish_time', 0)} ns")
    
    # Final chain status
    print(f"\n=== Final Chain Status ===")
    final_chain_status = traffic_scheduler.get_chain_status()
    for chain_id, status in final_chain_status.items():
        print(f"{chain_id}:")
        print(f"  Progress: {status['current_index']}/{status['total_traffics']}")
        print(f"  Time offset: {status['time_offset']} ns")
        print(f"  Completed: {status['is_completed']}")
        print(f"  Pending requests: {status['pending_requests']}")
        print(f"  Current traffic: {status['current_traffic_id']}")
    
    # Configuration summary
    print(f"\n=== Configuration Summary ===")
    print(f"Topology: {config.num_row}x{config.num_col} CrossRing")
    print(f"Network frequency: {config.NETWORK_FREQUENCY} Hz")
    print(f"Basic burst size: {config.basic_config.burst}")
    print(f"Ring buffer depth: {config.ring_buffer_depth}")
    print(f"Slice per link: {config.slice_per_link}")
    
    print(f"\n=== Network Resource Summary ===")
    print(f"RN read trackers: {config.tracker_config.rn_r_tracker_ostd}")
    print(f"RN write trackers: {config.tracker_config.rn_w_tracker_ostd}")
    print(f"SN DDR read trackers: {config.tracker_config.sn_ddr_r_tracker_ostd}")
    print(f"SN DDR write trackers: {config.tracker_config.sn_ddr_w_tracker_ostd}")
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()