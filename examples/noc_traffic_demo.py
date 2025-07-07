#!/usr/bin/env python3
"""
NoC CrossRing Traffic Simulation Demo

This example demonstrates how to run a CrossRing NoC simulation using traffic files,
similar to the src_old/example but with the new NoC structure.
"""

import sys
import os
import time
from typing import List, Dict, Any


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig
from src.noc.utils.traffic_scheduler import TrafficScheduler


def create_sample_traffic_file(filename: str):
    """Create a sample traffic file compatible with the existing format"""
    # Copy the format from src_old/example/test_data.txt
    traffic_data = [
        "0,0,gdma_0,1,ddr_0,R,4",
        "16,0,gdma_0,1,ddr_0,R,4",
        "32,0,gdma_0,1,ddr_0,R,4",
        "48,0,gdma_0,1,ddr_0,R,4",
        "64,0,gdma_0,1,ddr_0,R,4",
        "80,1,gdma_1,2,ddr_1,W,4",
        "96,1,gdma_1,2,ddr_1,W,4",
        "112,2,gdma_2,3,ddr_2,R,4",
        "128,2,gdma_2,3,ddr_2,R,4",
        "144,3,gdma_3,0,ddr_3,W,4",
        "160,3,gdma_3,0,ddr_3,W,4",
        "176,0,gdma_0,2,ddr_0,R,4",
        "192,1,gdma_1,3,ddr_1,W,4",
        "208,2,gdma_2,0,ddr_2,R,4",
        "224,3,gdma_3,1,ddr_3,W,4",
        "240,0,gdma_0,3,ddr_0,R,4",
    ]

    with open(filename, "w") as f:
        for line in traffic_data:
            f.write(line + "\n")

    print(f"Created sample traffic file: {filename}")


def run_crossring_simulation():
    """Run CrossRing simulation with traffic files"""
    print("Starting CrossRing NoC Traffic Simulation")
    print("=" * 50)

    # Create configuration
    config = CrossRingConfig()

    # Create traffic directory and sample file
    traffic_dir = "traffic_data"
    if not os.path.exists(traffic_dir):
        os.makedirs(traffic_dir)

    sample_file = os.path.join(traffic_dir, "crossring_traffic.txt")
    create_sample_traffic_file(sample_file)

    # Setup traffic scheduler
    traffic_scheduler = TrafficScheduler(config, traffic_dir)
    traffic_scheduler.set_verbose(True)

    # Setup single chain with the traffic file
    traffic_files = ["crossring_traffic.txt"]
    traffic_scheduler.setup_single_chain(traffic_files)

    # Start initial traffic
    traffic_scheduler.start_initial_traffics()

    print(f"\nTraffic scheduler initialized")
    print(f"Active traffics: {traffic_scheduler.get_active_traffic_count()}")

    # Create CrossRing model
    model = CrossRingModel(config, topology_size=(4, 4))
    model.initialize_network()

    print(f"CrossRing model initialized")
    print(f"Network nodes: {model.get_node_count()}")

    # Simulation parameters
    max_cycles = 2000
    current_cycle = 0
    total_packets = 0
    total_flits = 0

    print(f"\nRunning simulation for {max_cycles} cycles...")
    start_time = time.time()

    # Main simulation loop
    while current_cycle < max_cycles:
        # Get ready requests for current cycle
        ready_requests = traffic_scheduler.get_ready_requests(current_cycle)

        # Process ready requests
        for request in ready_requests:
            # Parse request: (time, src, src_type, dst, dst_type, op, burst, traffic_id)
            req_time, src, src_type, dst, dst_type, op, burst, traffic_id = request

            print(f"Cycle {current_cycle}: Injecting packet from {src} to {dst}, op={op}, burst={burst}")

            # Simulate packet injection
            total_packets += 1
            total_flits += burst

            # Update traffic statistics
            traffic_scheduler.update_traffic_stats(traffic_id, "injected_req")

            # Simulate packet completion (simplified)
            for _ in range(burst):
                traffic_scheduler.update_traffic_stats(traffic_id, "received_flit")

        # Advance network simulation (simplified)
        model.advance_cycle()

        # Check and advance traffic chains
        completed_traffics = traffic_scheduler.check_and_advance_chains(current_cycle)

        # Progress reporting
        if current_cycle % 100 == 0 and current_cycle > 0:
            active_count = traffic_scheduler.get_active_traffic_count()
            print(f"Cycle {current_cycle}: Total packets={total_packets}, Active traffics={active_count}")

        # Check if all traffic is completed
        if traffic_scheduler.is_all_completed():
            print(f"All traffic completed at cycle {current_cycle}")
            break

        current_cycle += 1

    end_time = time.time()
    simulation_time = end_time - start_time

    # Print final statistics
    print(f"\n=== Simulation Results ===")
    print(f"Simulation completed at cycle {current_cycle}")
    print(f"Total packets injected: {total_packets}")
    print(f"Total flits: {total_flits}")
    print(f"Simulation time: {simulation_time:.3f} seconds")

    # Get traffic completion statistics
    finish_stats = traffic_scheduler.get_finish_time_stats()
    print(f"Read finish time: {finish_stats.get('R_finish_time', 0)} ns")
    print(f"Write finish time: {finish_stats.get('W_finish_time', 0)} ns")
    print(f"Total finish time: {finish_stats.get('Total_finish_time', 0)} ns")

    # Chain status
    print(f"\n=== Chain Status ===")
    chain_status = traffic_scheduler.get_chain_status()
    for chain_id, status in chain_status.items():
        print(f"{chain_id}: {status['current_index']}/{status['total_traffics']} completed")
        print(f"  Time offset: {status['time_offset']} ns")
        print(f"  Is completed: {status['is_completed']}")


def main():
    """Main function"""
    try:
        run_crossring_simulation()
        print("\nSimulation completed successfully!")
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
