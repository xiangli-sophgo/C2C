import unittest
import sys
import os

from src.topology.builder import TopologyBuilder
from src.topology.node import ChipNode
from src.topology.link import C2CDirectLink
from src.protocol.cdma_system import CDMASystem, MemoryType


class TestTopologyProtocolIntegration(unittest.TestCase):
    """Integration tests for topology and CDMA protocol"""

    def setUp(self):
        # Build a simple 3-chip line topology
        builder = TopologyBuilder("test_topo")
        self.node_a = ChipNode("chip_A", "board_0")
        self.node_b = ChipNode("chip_B", "board_0")
        self.node_c = ChipNode("chip_C", "board_0")
        builder.add_node(self.node_a)
        builder.add_node(self.node_b)
        builder.add_node(self.node_c)
        builder.add_link(C2CDirectLink("link_ab", self.node_a, self.node_b))
        builder.add_link(C2CDirectLink("link_bc", self.node_b, self.node_c))
        self.topology = builder.build()

        # Create CDMA systems and establish connections
        self.sys_a = CDMASystem("chip_A")
        self.sys_b = CDMASystem("chip_B")
        self.sys_c = CDMASystem("chip_C")
        self.sys_a.connect_to_chip("chip_B", self.sys_b)
        self.sys_b.connect_to_chip("chip_A", self.sys_a)
        self.sys_b.connect_to_chip("chip_C", self.sys_c)
        self.sys_c.connect_to_chip("chip_B", self.sys_b)

    def tearDown(self):
        for sys_inst in (self.sys_a, self.sys_b, self.sys_c):
            try:
                sys_inst.shutdown()
            except Exception as e:
                print(f"cleanup warning: {e}")

    def test_topology_statistics(self):
        """Verify topology creation statistics"""
        stats = self.topology.get_topology_statistics()
        self.assertEqual(stats["num_nodes"], 3)
        self.assertEqual(stats["num_links"], 2)
        self.assertTrue(stats["is_connected"])

    def test_path_and_transfer(self):
        """Test path finding and a basic CDMA send/receive"""
        path = self.topology.find_path("chip_A", "chip_C")
        self.assertEqual(path, ["chip_A", "chip_B", "chip_C"], "Path should traverse chip_B")

        recv_res = self.sys_c.cdma_receive(
            dst_addr=0x1000,
            dst_shape=(64,),
            dst_mem_type=MemoryType.GMEM,
            src_chip_id="chip_A",
            data_type="float32",
        )
        self.assertTrue(recv_res.success, f"Receive failed: {recv_res.error_message}")

        send_res = self.sys_a.cdma_send(
            src_addr=0x2000,
            src_shape=(64,),
            dst_chip_id="chip_C",
            src_mem_type=MemoryType.GMEM,
            data_type="float32",
        )
        self.assertTrue(send_res.success, f"Send failed: {send_res.error_message}")


if __name__ == "__main__":
    unittest.main()
