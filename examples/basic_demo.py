from src.topology.node import ChipNode, SwitchNode, HostNode
from src.topology.link import PCIeLink, C2CDirectLink
from src.topology.builder import TopologyBuilder
from src.protocol.cdma import CDMAProtocol, CDMAMessage
from src.protocol.credit import CreditManager
from src.protocol.address import AddressTranslator, AddressFormat
from src.protocol.router import Router
from src.protocol.cdma_system import CDMASystem, MemoryType


def demo_cascade_topology():
    """演示4芯片级联拓扑的创建和通信"""
    print("\n--- 演示4芯片级联拓扑 ---")

    # 1. 创建4个芯片节点
    chip0 = ChipNode(chip_id="chip_0", board_id="board_A", memory_types=["DDR", "HBM"])
    chip1 = ChipNode(chip_id="chip_1", board_id="board_A", memory_types=["DDR"])
    chip2 = ChipNode(chip_id="chip_2", board_id="board_B", memory_types=["HBM"])
    chip3 = ChipNode(chip_id="chip_3", board_id="board_B", memory_types=["DDR", "HBM"])

    # 创建主机节点和交换机节点用于更复杂的场景
    host = HostNode(host_id="host_0", pcie_lanes=16)
    switch = SwitchNode(switch_id="switch_0", port_count=24, bandwidth=128.0)

    # 初始化拓扑构建器
    builder = TopologyBuilder("4_chip_cascade_demo")
    builder.add_node(chip0)
    builder.add_node(chip1)
    builder.add_node(chip2)
    builder.add_node(chip3)
    builder.add_node(host)
    builder.add_node(switch)

    # 2. 创建PCIe链路连接
    # 主机到交换机
    builder.add_link(PCIeLink(link_id="link_host_switch", endpoint_a=host, endpoint_b=switch, pcie_type="x8"))
    # 交换机到芯片0
    builder.add_link(PCIeLink(link_id="link_switch_chip0", endpoint_a=switch, endpoint_b=chip0, pcie_type="x8"))

    # 3. 构建级联拓扑 (C2C Direct Links)
    builder.add_link(C2CDirectLink(link_id="link_0_1", endpoint_a=chip0, endpoint_b=chip1))
    builder.add_link(C2CDirectLink(link_id="link_1_2", endpoint_a=chip1, endpoint_b=chip2))
    builder.add_link(C2CDirectLink(link_id="link_2_3", endpoint_a=chip2, endpoint_b=chip3))

    topology = builder.build()
    print(f"创建的拓扑: {topology}")
    print("拓扑统计信息:", topology.get_topology_statistics())

    # 4. 计算chip_0到chip_3的路径
    path_0_3 = topology.find_path("chip_0", "chip_3")
    print(f"从 chip_0 到 chip_3 的路径: {path_0_3}")

    # 测试从主机到芯片3的路径
    path_host_3 = topology.find_path("host_0", "chip_3")
    print(f"从 host_0 到 chip_3 的路径: {path_host_3}")

    # 5. 模拟CDMA Send/Receive通信
    print("\n--- 模拟 CDMA 通信 ---")
    cdma_protocol_chip0 = CDMAProtocol(protocol_id="cdma_0", node_id="chip_0")
    cdma_protocol_chip3 = CDMAProtocol(protocol_id="cdma_3", node_id="chip_3")

    # 模拟从芯片0到芯片3的发送操作
    send_message = CDMAMessage(
        source_id="chip_0",
        destination_id="chip_3",
        message_type="send",
        payload="来自芯片0的问候!",
        tensor_shape=(10, 20),
        data_type="float32",
    )

    print("\nChip_0 发起 SEND 到 Chip_3")
    cdma_protocol_chip0.send_message(send_message)

    # 模拟在芯片3上的消息路由和处理
    print("\nChip_3 接收并处理 SEND 消息")
    ack_message = cdma_protocol_chip3.process_message(send_message)
    if ack_message:
        print("Chip_3 发送 ACK 回 Chip_0")
        cdma_protocol_chip0.process_message(ack_message)  # 芯片0接收ACK

    # 模拟从芯片3到芯片0的接收操作
    receive_message = CDMAMessage(source_id="chip_3", destination_id="chip_0", message_type="receive")

    print("\nChip_3 发起 RECEIVE 从 Chip_0")
    cdma_protocol_chip3.send_message(receive_message)

    print("\nChip_0 接收并处理 RECEIVE 消息")
    data_response = cdma_protocol_chip0.process_message(receive_message)
    if data_response:
        print("Chip_0 发送 DATA_RESPONSE 回 Chip_3")
        cdma_protocol_chip3.process_message(data_response)  # Chip_3 receives DATA_RESPONSE

    # 6. Credit Management Demo
    print("\n--- 演示 Credit 管理 ---")
    credit_manager_chip0 = CreditManager(node_id="chip_0")
    credit_manager_chip0.initialize_credits(destinations=["chip_1", "chip_3"])

    print(f"chip_0 的初始 Credit: {credit_manager_chip0.get_credit_status()}")

    # 请求信用
    print("请求 chip_1 的 Credit...")
    if credit_manager_chip0.request_credit("chip_1"):
        print("chip_1 的 Credit 已授予。")
    else:
        print("chip_1 的 Credit 被拒绝。")

    print("再次请求 chip_1 的 Credit...")
    if credit_manager_chip0.request_credit("chip_1"):
        print("chip_1 的 Credit 已授予。")
    else:
        print("chip_1 的 Credit 被拒绝。")

    print("第三次请求 chip_1 的 Credit (应该被拒绝)... ")
    if credit_manager_chip0.request_credit("chip_1"):
        print("chip_1 的 Credit 已授予。")
    else:
        print("chip_1 的 Credit 被拒绝。")

    print(f"请求后的 Credit: {credit_manager_chip0.get_credit_status()}")

    # 授予信用
    print("授予 chip_1 1 个 Credit...")
    credit_manager_chip0.grant_credit("chip_1", 1)
    print(f"授予后的 Credit: {credit_manager_chip0.get_credit_status()}")

    # 7. Address Translation Demo
    print("\n--- 演示地址转换 ---")
    addr_translator = AddressTranslator()
    cdma_addr = "CDMA_ADDR_0x12345678"
    pcie_addr = addr_translator.translate(cdma_addr, AddressFormat.CDMA_FMT, AddressFormat.PCIE_FMT)
    print(f"CDMA 地址: {cdma_addr} -> PCIe 地址: {pcie_addr}")

    pcie_addr_back = addr_translator.translate(pcie_addr, AddressFormat.PCIE_FMT, AddressFormat.CDMA_FMT)
    print(f"PCIe 地址: {pcie_addr} -> CDMA 地址: {pcie_addr_back}")

    # 8. Router Demo
    print("\n--- 演示路由器 ---")
    router_chip0 = Router(router_id="router_chip0", topology_graph=topology)
    router_chip0.calculate_and_set_routes(source_node_id="chip_0")
    print(f"chip_0 的路由表: {router_chip0._routing_table}")

    # 模拟从芯片0到芯片3的消息路由
    msg_to_route = CDMAMessage(source_id="chip_0", destination_id="chip_3", message_type="data")
    next_hop = router_chip0.route_message(msg_to_route, chip0)
    if next_hop:
        print(f"从 chip_0 到 chip_3 的消息，下一跳是: {next_hop.node_id}")

    # 尝试路由消息到未知目的地
    msg_unknown_dest = CDMAMessage(source_id="chip_0", destination_id="unknown_chip", message_type="data")
    next_hop_unknown = router_chip0.route_message(msg_unknown_dest, chip0)
    if not next_hop_unknown:
        print("正确处理了路由到未知目的地 (未找到路径)。")

    print("\n--- 演示新的CDMA Send/Receive配对协议 ---")
    demo_cdma_paired_protocol()

    # 可选：绘制拓扑图（需要matplotlib）
    # topology.draw_topology()


def demo_cdma_paired_protocol():
    """演示基于SG2260E的CDMA配对指令协议"""
    print("\n=== CDMA配对指令协议演示 ===")
    
    # 创建两个芯片的CDMA系统
    chip_A_system = CDMASystem("chip_A")
    chip_B_system = CDMASystem("chip_B")
    
    # 建立芯片间连接
    chip_A_system.connect_to_chip("chip_B", chip_B_system)
    chip_B_system.connect_to_chip("chip_A", chip_A_system)
    
    print("✓ 创建并连接了两个CDMA系统")
    
    # 1. 演示基本配对传输
    print("\n1. 基本配对传输演示")
    print("-" * 30)
    
    # chip_A执行CDMA_receive
    print("步骤1: chip_A执行CDMA_receive...")
    receive_result = chip_A_system.cdma_receive(
        dst_addr=0x10000000,
        dst_shape=(1024, 512),
        dst_mem_type=MemoryType.GMEM,
        src_chip_id="chip_B",
        data_type="float32"
    )
    
    if receive_result.success:
        print(f"✓ CDMA_receive完成，事务ID: {receive_result.transaction_id}")
    else:
        print(f"✗ CDMA_receive失败: {receive_result.error_message}")
        return
    
    # 等待Credit传递
    import time
    time.sleep(0.1)
    
    # chip_B执行CDMA_send
    print("步骤2: chip_B执行CDMA_send...")
    send_result = chip_B_system.cdma_send(
        src_addr=0x20000000,
        src_shape=(1024, 512),
        dst_chip_id="chip_A",
        src_mem_type=MemoryType.GMEM,
        data_type="float32"
    )
    
    if send_result.success:
        print(f"✓ CDMA_send完成，传输了 {send_result.bytes_transferred} 字节")
        print(f"✓ 执行时间: {send_result.execution_time * 1000:.2f} ms")
    else:
        print(f"✗ CDMA_send失败: {send_result.error_message}")
    
    # 2. 演示同步消息
    print("\n2. 同步消息演示")
    print("-" * 30)
    
    sync_result = chip_A_system.cdma_sys_send_msg("chip_B", "传输完成同步")
    if sync_result.success:
        print("✓ 同步消息发送成功")
    else:
        print(f"✗ 同步消息发送失败: {sync_result.error_message}")
    
    # 3. 演示错误处理
    print("\n3. 错误处理演示")
    print("-" * 30)
    
    # 尝试没有Credit的发送
    print("测试Credit不足错误...")
    invalid_send = chip_B_system.cdma_send(
        src_addr=0x30000000,
        src_shape=(256,),
        dst_chip_id="chip_A",
        src_mem_type=MemoryType.GMEM,
        data_type="float32"
    )
    
    if not invalid_send.success:
        print(f"✓ 正确检测到错误: {invalid_send.error_message}")
    
    # 4. 演示不同内存类型
    print("\n4. 不同内存类型演示")
    print("-" * 30)
    
    # L2M到LMEM的传输
    print("L2M -> LMEM 传输...")
    l2m_receive = chip_A_system.cdma_receive(
        dst_addr=0x80001000,
        dst_shape=(128, 128),
        dst_mem_type=MemoryType.LMEM,
        src_chip_id="chip_B",
        data_type="float16"
    )
    
    if l2m_receive.success:
        time.sleep(0.1)
        l2m_send = chip_B_system.cdma_send(
            src_addr=0x40001000,
            src_shape=(128, 128),
            dst_chip_id="chip_A",
            src_mem_type=MemoryType.L2M,
            data_type="float16"
        )
        
        if l2m_send.success:
            print(f"✓ L2M->LMEM传输完成，{l2m_send.bytes_transferred} 字节")
        else:
            print(f"✗ L2M->LMEM传输失败: {l2m_send.error_message}")
    
    # 5. 显示系统状态
    print("\n5. 系统状态信息")
    print("-" * 30)
    
    print("\nchip_A 状态:")
    chip_A_status = chip_A_system.get_system_status()
    print(f"  系统状态: {chip_A_status['state']}")
    print(f"  DMA传输次数: {chip_A_status['dma_performance']['total_transfers']}")
    print(f"  总传输字节: {chip_A_status['dma_performance']['total_bytes_transferred']}")
    
    print("\nchip_B 状态:")
    chip_B_status = chip_B_system.get_system_status()
    print(f"  系统状态: {chip_B_status['state']}")
    print(f"  DMA传输次数: {chip_B_status['dma_performance']['total_transfers']}")
    print(f"  总传输字节: {chip_B_status['dma_performance']['total_bytes_transferred']}")
    
    # 清理系统
    chip_A_system.cleanup()
    chip_B_system.cleanup()
    
    print("\n✓ CDMA配对协议演示完成！")


if __name__ == "__main__":
    demo_cascade_topology()
