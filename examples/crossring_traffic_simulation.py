#!/usr/bin/env python3
"""
CrossRing NoC Traffic Simulation Example

This example demonstrates how to run a CrossRing NoC simulation using traffic files.
It reads traffic patterns from files and simulates the network behavior.
"""

import sys
import os
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.noc.crossring.model import CrossRingModel
from src.noc.crossring.config import CrossRingConfig
from src.noc.utils.traffic_scheduler import TrafficScheduler


class CrossRingTrafficSimulation:
    """CrossRing NoC traffic simulation with file-based traffic patterns"""

    def __init__(self, config: CrossRingConfig):
        self.config = config
        self.model = None
        self.traffic_scheduler = None
        self.stats = {"total_packets": 0, "total_flits": 0, "simulation_time": 0, "start_time": 0, "end_time": 0}

    def setup_model(self, topology_size: tuple = (4, 4)):
        """Setup the CrossRing model with specified topology"""
        # Update config topology if needed
        self.config.num_row, self.config.num_col = topology_size
        # Create the CrossRing model
        self.model = CrossRingModel(config=self.config)

        # Initialize the network
        self.model.initialize_network()

        print(f"CrossRing model initialized with topology {topology_size}")
        print(f"Total nodes: {self.model.get_node_count()}")
        network_freq = getattr(self.config, 'NETWORK_FREQUENCY', 
                             getattr(self.config, 'basic_config', None) and 
                             getattr(self.config.basic_config, 'network_frequency', 1.0) * 1000000000)
        print(f"Network frequency: {network_freq} Hz")

    def setup_traffic_scheduler(self, traffic_file_path: str, traffic_files: List[str]):
        """Setup traffic scheduler with traffic files"""
        self.traffic_scheduler = TrafficScheduler(config=self.config, traffic_file_path=traffic_file_path)

        # Setup traffic chains (can be parallel or serial)
        if isinstance(traffic_files[0], list):
            # Multiple parallel chains
            self.traffic_scheduler.setup_parallel_chains(traffic_files)
        else:
            # Single chain
            self.traffic_scheduler.setup_single_chain(traffic_files)

        # Enable verbose output
        self.traffic_scheduler.set_verbose(True)

        print(f"Traffic scheduler setup complete")
        print(f"Traffic files: {traffic_files}")

    def run_simulation(self, max_cycles: int = 10000):
        """Run the simulation for specified number of cycles"""
        if not self.model or not self.traffic_scheduler:
            raise ValueError("Model and traffic scheduler must be setup before running simulation")

        print(f"\nStarting simulation for {max_cycles} cycles...")
        self.stats["start_time"] = time.time()

        # Start initial traffic
        self.traffic_scheduler.start_initial_traffics()

        cycle = 0
        packets_injected = 0

        while cycle < max_cycles:
            # Get ready requests for current cycle
            ready_requests = self.traffic_scheduler.get_ready_requests(cycle)

            # Inject packets into the network
            for request in ready_requests:
                # Parse request: (time, src, src_type, dst, dst_type, op, burst, traffic_id)
                req_time, src, src_type, dst, dst_type, op, burst, traffic_id = request

                # Create and inject packet
                packet_injected = self.model.inject_packet(src_node=src, dst_node=dst, op_type=op, burst_size=burst, cycle=cycle)

                if packet_injected:
                    packets_injected += 1
                    self.stats["total_packets"] += 1
                    self.stats["total_flits"] += burst

                    # Update traffic statistics
                    self.traffic_scheduler.update_traffic_stats(traffic_id, "injected_req")

            # Advance network simulation one cycle
            self.model.advance_cycle()

            # Check for completed packets and update statistics
            completed_packets = self.model.get_completed_packets()
            for packet in completed_packets:
                traffic_id = packet.get("traffic_id", "unknown")
                # Update received flit count
                for _ in range(packet.get("flit_count", 1)):
                    self.traffic_scheduler.update_traffic_stats(traffic_id, "received_flit")

            # Check and advance traffic chains
            completed_traffics = self.traffic_scheduler.check_and_advance_chains(cycle)

            # Progress reporting
            if cycle % 1000 == 0:
                active_count = self.traffic_scheduler.get_active_traffic_count()
                print(f"Cycle {cycle}: Packets injected={packets_injected}, Active traffics={active_count}")

            # Check if all traffic is completed
            if self.traffic_scheduler.is_all_completed():
                print(f"All traffic completed at cycle {cycle}")
                break

            cycle += 1

        self.stats["end_time"] = time.time()
        self.stats["simulation_time"] = self.stats["end_time"] - self.stats["start_time"]

        print(f"\nSimulation completed at cycle {cycle}")
        self.print_simulation_stats()

    def print_simulation_stats(self):
        """Print simulation statistics"""
        print("\n=== Simulation Statistics ===")
        print(f"Total packets injected: {self.stats['total_packets']}")
        print(f"Total flits: {self.stats['total_flits']}")
        print(f"Simulation time: {self.stats['simulation_time']:.2f} seconds")

        # Get network statistics
        network_stats = self.model.get_network_statistics()
        print(f"Network utilization: {network_stats.get('utilization', 0):.2f}%")
        print(f"Average latency: {network_stats.get('avg_latency', 0):.2f} cycles")

        # Get traffic completion statistics
        finish_stats = self.traffic_scheduler.get_finish_time_stats()
        print(f"Read finish time: {finish_stats.get('R_finish_time', 0)} ns")
        print(f"Write finish time: {finish_stats.get('W_finish_time', 0)} ns")
        print(f"Total finish time: {finish_stats.get('Total_finish_time', 0)} ns")

        # Chain status
        print("\n=== Chain Status ===")
        chain_status = self.traffic_scheduler.get_chain_status()
        for chain_id, status in chain_status.items():
            print(f"{chain_id}: {status['current_index']}/{status['total_traffics']} completed")


def create_sample_traffic_file(filename: str, num_requests: int = 100):
    """Create a sample traffic file for testing"""
    with open(filename, "w") as f:
        for i in range(num_requests):
            # Format: time,src,src_type,dst,dst_type,op,burst
            time_ns = i * 16  # 16ns intervals
            src = i % 4  # Source node 0-3
            dst = (i + 1) % 4  # Destination node
            op = "R" if i % 2 == 0 else "W"  # Alternate read/write
            burst = 4  # 4 flits per burst

            f.write(f"{time_ns},{src},gdma_{src},{dst},ddr_{dst},{op},{burst}\n")

    print(f"Created sample traffic file: {filename}")


def main():
    """Main function to demonstrate CrossRing traffic simulation"""
    print("CrossRing NoC Traffic Simulation Example")
    print("=" * 50)

    # Create configuration
    config = CrossRingConfig()
    # Add network frequency attribute to config for compatibility
    config.NETWORK_FREQUENCY = 1000000000  # 1 GHz

    # Create simulation instance
    sim = CrossRingTrafficSimulation(config)

    # Setup model with 4x4 topology
    sim.setup_model(topology_size=(4, 4))

    # Create sample traffic data if not exists
    traffic_dir = "traffic_data"
    if not os.path.exists(traffic_dir):
        os.makedirs(traffic_dir)

    sample_traffic_file = os.path.join(traffic_dir, "sample_traffic.txt")
    if not os.path.exists(sample_traffic_file):
        create_sample_traffic_file(sample_traffic_file, 50)

    # Setup traffic scheduler
    traffic_files = ["sample_traffic.txt"]
    sim.setup_traffic_scheduler(traffic_dir, traffic_files)

    # Run simulation
    sim.run_simulation(max_cycles=5000)

    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()
