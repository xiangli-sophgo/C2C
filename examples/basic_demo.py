from topology.node import ChipNode, SwitchNode, HostNode
from topology.link import PCIeLink, C2CDirectLink
from topology.builder import TopologyBuilder
from protocol.cdma import CDMAProtocol, CDMAMessage
from protocol.credit import CreditManager
from protocol.address import AddressTranslator, AddressFormat
from protocol.router import Router

def demo_cascade_topology():
    """演示4芯片级联拓扑的创建和通信"""
    print("\n--- Demonstrating 4-Chip Cascade Topology ---")

    # 1. 创建4个芯片节点
    chip0 = ChipNode(chip_id="chip_0", board_id="board_A", memory_types=["DDR", "HBM"])
    chip1 = ChipNode(chip_id="chip_1", board_id="board_A", memory_types=["DDR"])
    chip2 = ChipNode(chip_id="chip_2", board_id="board_B", memory_types=["HBM"])
    chip3 = ChipNode(chip_id="chip_3", board_id="board_B", memory_types=["DDR", "HBM"])

    # Create a host node and a switch node for more complex scenarios
    host = HostNode(host_id="host_0", pcie_lanes=16)
    switch = SwitchNode(switch_id="switch_0", port_count=24, bandwidth=128.0)

    # Initialize Topology Builder
    builder = TopologyBuilder("4_chip_cascade_demo")
    builder.add_node(chip0)
    builder.add_node(chip1)
    builder.add_node(chip2)
    builder.add_node(chip3)
    builder.add_node(host)
    builder.add_node(switch)

    # 2. 创建PCIe链路连接
    # Host to Switch
    builder.add_link(PCIeLink(link_id="link_host_switch", endpoint_a=host, endpoint_b=switch, pcie_type="x8"))
    # Switch to Chip0
    builder.add_link(PCIeLink(link_id="link_switch_chip0", endpoint_a=switch, endpoint_b=chip0, pcie_type="x8"))

    # 3. 构建级联拓扑 (C2C Direct Links)
    builder.add_link(C2CDirectLink(link_id="link_0_1", endpoint_a=chip0, endpoint_b=chip1))
    builder.add_link(C2CDirectLink(link_id="link_1_2", endpoint_a=chip1, endpoint_b=chip2))
    builder.add_link(C2CDirectLink(link_id="link_2_3", endpoint_a=chip2, endpoint_b=chip3))

    topology = builder.build()
    print(f"Created Topology: {topology}")
    print("Topology Statistics:", topology.get_topology_statistics())

    # 4. 计算chip_0到chip_3的路径
    path_0_3 = topology.find_path("chip_0", "chip_3")
    print(f"Path from chip_0 to chip_3: {path_0_3}")

    # Test path from host to chip3
    path_host_3 = topology.find_path("host_0", "chip_3")
    print(f"Path from host_0 to chip_3: {path_host_3}")

    # 5. 模拟CDMA Send/Receive通信
    print("\n--- Simulating CDMA Communication ---")
    cdma_protocol_chip0 = CDMAProtocol(protocol_id="cdma_0", node_id="chip_0")
    cdma_protocol_chip3 = CDMAProtocol(protocol_id="cdma_3", node_id="chip_3")

    # Simulate a send operation from chip_0 to chip_3
    send_message = CDMAMessage(
        source_id="chip_0",
        destination_id="chip_3",
        message_type="send",
        payload="Hello from chip_0!",
        tensor_shape=(10, 20),
        data_type="float32"
    )

    print("\nChip_0 initiates SEND to Chip_3")
    cdma_protocol_chip0.send_message(send_message)

    # Simulate message routing and processing at chip_3
    print("\nChip_3 receives and processes SEND message")
    ack_message = cdma_protocol_chip3.process_message(send_message)
    if ack_message:
        print("Chip_3 sends ACK back to Chip_0")
        cdma_protocol_chip0.process_message(ack_message) # Chip_0 receives ACK

    # Simulate a receive operation from chip_3 to chip_0
    receive_message = CDMAMessage(
        source_id="chip_3",
        destination_id="chip_0",
        message_type="receive"
    )

    print("\nChip_3 initiates RECEIVE from Chip_0")
    cdma_protocol_chip3.send_message(receive_message)

    print("\nChip_0 receives and processes RECEIVE message")
    data_response = cdma_protocol_chip0.process_message(receive_message)
    if data_response:
        print("Chip_0 sends DATA_RESPONSE back to Chip_3")
        cdma_protocol_chip3.process_message(data_response) # Chip_3 receives DATA_RESPONSE

    # 6. Credit Management Demo
    print("\n--- Demonstrating Credit Management ---")
    credit_manager_chip0 = CreditManager(node_id="chip_0")
    credit_manager_chip0.initialize_credits(destinations=["chip_1", "chip_3"])

    print(f"Initial credits for chip_0: {credit_manager_chip0.get_credit_status()}")

    # Request credits
    print("Requesting credit for chip_1...")
    if credit_manager_chip0.request_credit("chip_1"):
        print("Credit for chip_1 granted.")
    else:
        print("Credit for chip_1 denied.")

    print("Requesting credit for chip_1 again...")
    if credit_manager_chip0.request_credit("chip_1"):
        print("Credit for chip_1 granted.")
    else:
        print("Credit for chip_1 denied.")

    print("Requesting credit for chip_1 a third time (should be denied)...")
    if credit_manager_chip0.request_credit("chip_1"):
        print("Credit for chip_1 granted.")
    else:
        print("Credit for chip_1 denied.")

    print(f"Credits after requests: {credit_manager_chip0.get_credit_status()}")

    # Grant credits
    print("Granting 1 credit to chip_1...")
    credit_manager_chip0.grant_credit("chip_1", 1)
    print(f"Credits after granting: {credit_manager_chip0.get_credit_status()}")

    # 7. Address Translation Demo
    print("\n--- Demonstrating Address Translation ---")
    addr_translator = AddressTranslator()
    cdma_addr = "CDMA_ADDR_0x12345678"
    pcie_addr = addr_translator.translate(cdma_addr, AddressFormat.CDMA_FMT, AddressFormat.PCIE_FMT)
    print(f"CDMA Address: {cdma_addr} -> PCIe Address: {pcie_addr}")

    pcie_addr_back = addr_translator.translate(pcie_addr, AddressFormat.PCIE_FMT, AddressFormat.CDMA_FMT)
    print(f"PCIe Address: {pcie_addr} -> CDMA Address: {pcie_addr_back}")

    # 8. Router Demo
    print("\n--- Demonstrating Router ---")
    router_chip0 = Router(router_id="router_chip0", topology_graph=topology)
    router_chip0.calculate_and_set_routes(source_node_id="chip_0")
    print(f"Chip_0's routing table: {router_chip0._routing_table}")

    # Simulate routing a message from chip_0 to chip_3
    msg_to_route = CDMAMessage(source_id="chip_0", destination_id="chip_3", message_type="data")
    next_hop = router_chip0.route_message(msg_to_route, chip0)
    if next_hop:
        print(f"Message from chip_0 to chip_3, next hop is: {next_hop.node_id}")

    # Try routing a message to an unknown destination
    msg_unknown_dest = CDMAMessage(source_id="chip_0", destination_id="unknown_chip", message_type="data")
    next_hop_unknown = router_chip0.route_message(msg_unknown_dest, chip0)
    if not next_hop_unknown:
        print("Correctly handled routing to unknown destination (no path found).")

    # Optional: Draw the topology (requires matplotlib)
    # topology.draw_topology()

if __name__ == "__main__":
    demo_cascade_topology()
