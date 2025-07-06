#!/usr/bin/env python3
"""
Basic Traffic Example for NoC Simulation

This example shows how to parse and process traffic files for NoC simulation,
using the traffic_scheduler directly without complex model dependencies.
"""

import sys
import os

# Add src_old to path to access traffic_scheduler
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_old'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.noc.utils.traffic_scheduler import TrafficScheduler


class SimpleConfig:
    """Simple configuration class for traffic simulation"""

    def __init__(self):
        self.NETWORK_FREQUENCY = 1000000000  # 1 GHz


def main():
    """Main simulation function"""
    print("Basic NoC Traffic Example")
    print("=" * 30)

    # Create simple configuration
    config = SimpleConfig()
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

    # Setup single chain
    traffic_scheduler.setup_single_chain(traffic_files)

    # Start initial traffic
    traffic_scheduler.start_initial_traffics()

    print(f"\nTraffic scheduler initialized")
    print(f"Active traffics: {traffic_scheduler.get_active_traffic_count()}")

    # Simulation parameters
    max_cycles = 2000
    current_cycle = 0
    total_requests = 0
    injected_requests = 0

    print(f"\nRunning simulation for {max_cycles} cycles...")

    # Main simulation loop
    while current_cycle < max_cycles:
        # Get ready requests for current cycle
        ready_requests = traffic_scheduler.get_ready_requests(current_cycle)

        # Process ready requests
        for request in ready_requests:
            # Parse request: (time, src, src_type, dst, dst_type, op, burst, traffic_id)
            req_time, src, src_type, dst, dst_type, op, burst, traffic_id = request

            print(f"Cycle {current_cycle}: Injecting packet from {src}({src_type}) to {dst}({dst_type}), op={op}, burst={burst}")
            injected_requests += 1

            # Simulate packet processing latency (simplified)
            # In real simulation, this would be handled by the NoC model
            latency_cycles = 5  # Assume 5 cycles latency
            completion_cycle = current_cycle + latency_cycles

            # Update statistics for packet completion
            for _ in range(burst):
                traffic_scheduler.update_traffic_stats(traffic_id, "received_flit")

        total_requests += len(ready_requests)

        # Check and advance traffic chains
        completed_traffics = traffic_scheduler.check_and_advance_chains(current_cycle)

        if completed_traffics:
            print(f"Completed traffics: {completed_traffics}")

        # Progress reporting
        if current_cycle % 100 == 0 and (ready_requests or current_cycle == 0):
            active_count = traffic_scheduler.get_active_traffic_count()
            print(f"Cycle {current_cycle}: Requests={total_requests}, Injected={injected_requests}, Active={active_count}")

        # Check if all traffic is completed
        if traffic_scheduler.is_all_completed():
            print(f"All traffic completed at cycle {current_cycle}")
            break

        current_cycle += 1

    # Print final statistics
    print(f"\n=== Final Results ===")
    print(f"Simulation completed at cycle {current_cycle}")
    print(f"Total requests processed: {total_requests}")
    print(f"Injected requests: {injected_requests}")

    # Get traffic completion statistics
    finish_stats = traffic_scheduler.get_finish_time_stats()
    print(f"Read finish time: {finish_stats.get('R_finish_time', 0)} ns")
    print(f"Write finish time: {finish_stats.get('W_finish_time', 0)} ns")
    print(f"Total finish time: {finish_stats.get('Total_finish_time', 0)} ns")

    # Chain status
    print(f"\n=== Chain Status ===")
    chain_status = traffic_scheduler.get_chain_status()
    for chain_id, status in chain_status.items():
        print(f"{chain_id}:")
        print(f"  Progress: {status['current_index']}/{status['total_traffics']}")
        print(f"  Time offset: {status['time_offset']} ns")
        print(f"  Completed: {status['is_completed']}")
        print(f"  Pending requests: {status['pending_requests']}")
        print(f"  Current traffic: {status['current_traffic_id']}")


if __name__ == "__main__":
    main()
