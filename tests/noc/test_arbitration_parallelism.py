"""
测试CrossRing仲裁逻辑的并行性

验证修改后的仲裁逻辑是否支持：
1. InjectQueue: 不同方向的并行传输
2. EjectQueue: 不同IP的并行传输  
3. RingBridge: 不同输出方向的并行传输
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.noc.crossring.config import CrossRingConfig, FIFOConfiguration, BasicConfiguration, TagConfiguration
from src.noc.crossring.components.inject_queue import InjectQueue
from src.noc.crossring.components.eject_queue import EjectQueue
from src.noc.crossring.components.ring_bridge import RingBridge
from src.noc.crossring.flit import CrossRingFlit
from src.noc.base.ip_interface import PipelinedFIFO


def create_test_config():
    """创建测试配置"""
    config = CrossRingConfig(num_col=3, num_row=3)
    config.ROUTING_STRATEGY = "XY"
    
    # 修改FIFO深度
    config.fifo_config.IQ_CH_DEPTH = 4
    config.fifo_config.IQ_OUT_FIFO_DEPTH = 4
    config.fifo_config.EQ_IN_FIFO_DEPTH = 4
    config.fifo_config.EQ_CH_DEPTH = 4
    config.fifo_config.RB_IN_FIFO_DEPTH = 4
    config.fifo_config.RB_OUT_FIFO_DEPTH = 4
    
    return config


def test_inject_queue_parallelism():
    """测试InjectQueue的并行传输能力"""
    print("\n=== 测试InjectQueue并行传输 ===")
    
    config = create_test_config()
    iq = InjectQueue(node_id=4, coordinates=(1, 1), config=config)
    
    # 连接多个IP
    iq.connect_ip("IP0")
    iq.connect_ip("IP1")
    
    # 创建去往不同方向的flit
    flit1 = CrossRingFlit(
        packet_id=1,
        flit_type="HEAD",
        source=4,
        destination=5,  # 向右(TR)
        channel="req"
    )
    flit1.dest_coordinates = (2, 1)
    
    flit2 = CrossRingFlit(
        packet_id=2,
        flit_type="HEAD", 
        source=4,
        destination=3,  # 向左(TL)
        channel="req"
    )
    flit2.dest_coordinates = (0, 1)
    
    # IP0发送flit1, IP1发送flit2
    iq.add_to_inject_queue(flit1, "req", "IP0")
    iq.add_to_inject_queue(flit2, "req", "IP1")
    
    # 更新FIFO状态
    iq.step_compute_phase(0)
    iq.step_update_phase()
    
    # 执行仲裁
    iq.compute_arbitration(1)
    
    # 检查传输计划
    print(f"传输计划数量: {len(iq._inject_transfer_plan)}")
    for ip_id, channel, flit, direction in iq._inject_transfer_plan:
        print(f"  IP{ip_id[-1]} -> {direction}: flit {flit.packet_id}")
    
    # 执行传输
    iq.execute_arbitration(1)
    
    # 验证两个flit都被传输到对应方向
    tr_fifo = iq.inject_input_fifos["req"]["TR"]
    tl_fifo = iq.inject_input_fifos["req"]["TL"]
    
    print(f"\nTR FIFO: {len(tr_fifo)} flits")
    print(f"TL FIFO: {len(tl_fifo)} flits")
    
    assert len(iq._inject_transfer_plan) == 2, "应该有2个并行传输"
    print("✅ InjectQueue并行传输测试通过")


def test_eject_queue_parallelism():
    """测试EjectQueue的并行传输能力"""
    print("\n=== 测试EjectQueue并行传输 ===")
    
    config = create_test_config()
    eq = EjectQueue(node_id=4, coordinates=(1, 1), config=config)
    
    # 连接多个IP
    eq.connect_ip("IP0")
    eq.connect_ip("IP1")
    
    # 创建inject_input_fifos和ring_bridge模拟输入
    inject_input_fifos = {
        "req": {"EQ": PipelinedFIFO("test_eq", 4)},
        "rsp": {"EQ": PipelinedFIFO("test_eq", 4)},
        "data": {"EQ": PipelinedFIFO("test_eq", 4)}
    }
    
    # 创建eject_input_fifos
    for channel in ["req", "rsp", "data"]:
        for direction in ["TU", "TD"]:
            if direction not in eq.eject_input_fifos[channel]:
                eq.eject_input_fifos[channel][direction] = PipelinedFIFO(f"test_{direction}", 4)
    
    # 创建去往不同IP的flit
    flit1 = CrossRingFlit(
        packet_id=1,
        flit_type="HEAD",
        source=0,
        destination=4,
        channel="req"
    )
    flit1.destination_type = "IP0"
    
    flit2 = CrossRingFlit(
        packet_id=2,
        flit_type="HEAD",
        source=1,
        destination=4,
        channel="req"
    )
    flit2.destination_type = "IP1"
    
    # 从不同源添加flit
    eq.eject_input_fifos["req"]["TU"].write_input(flit1)
    eq.eject_input_fifos["req"]["TD"].write_input(flit2)
    
    # 更新FIFO状态
    eq.step_compute_phase(0)
    eq.step_update_phase()
    
    # 执行仲裁
    eq.compute_arbitration(1, inject_input_fifos, None)
    
    # 检查传输计划
    print(f"传输计划数量: {len(eq._eject_transfer_plan)}")
    for source, channel, flit, target_ip in eq._eject_transfer_plan:
        print(f"  {source} -> {target_ip}: flit {flit.packet_id}")
    
    # 执行传输
    eq.execute_arbitration(1, inject_input_fifos, None)
    
    # 更新FIFO状态
    eq.step_compute_phase(1)
    eq.step_update_phase()
    
    # 验证两个IP都收到了flit
    ip0_buffer = eq.ip_eject_channel_buffers["IP0"]["req"]
    ip1_buffer = eq.ip_eject_channel_buffers["IP1"]["req"]
    
    print(f"\nIP0 buffer: {len(ip0_buffer)} flits")
    print(f"IP1 buffer: {len(ip1_buffer)} flits")
    
    assert len(eq._eject_transfer_plan) == 2, "应该有2个并行传输"
    print("✅ EjectQueue并行传输测试通过")


def test_ring_bridge_parallelism():
    """测试RingBridge的并行传输能力"""
    print("\n=== 测试RingBridge并行传输 ===")
    
    config = create_test_config()
    rb = RingBridge(node_id=4, coordinates=(1, 1), config=config)
    
    # 创建inject_input_fifos模拟IQ输入
    inject_input_fifos = {
        "req": {
            "TU": PipelinedFIFO("test_tu", 4),
            "TD": PipelinedFIFO("test_td", 4),
            "TR": PipelinedFIFO("test_tr", 4),
            "TL": PipelinedFIFO("test_tl", 4),
            "EQ": PipelinedFIFO("test_eq", 4)
        },
        "rsp": {
            "TU": PipelinedFIFO("test_tu", 4),
            "TD": PipelinedFIFO("test_td", 4),
            "TR": PipelinedFIFO("test_tr", 4),
            "TL": PipelinedFIFO("test_tl", 4),
            "EQ": PipelinedFIFO("test_eq", 4)
        },
        "data": {
            "TU": PipelinedFIFO("test_tu", 4),
            "TD": PipelinedFIFO("test_td", 4),
            "TR": PipelinedFIFO("test_tr", 4),
            "TL": PipelinedFIFO("test_tl", 4),
            "EQ": PipelinedFIFO("test_eq", 4)
        }
    }
    
    # 创建去往不同输出方向的flit
    # flit1: 本地弹出(EQ)
    flit1 = CrossRingFlit(
        packet_id=1,
        flit_type="HEAD",
        source=0,
        destination=4,  # 本地目标
        channel="req"
    )
    
    # flit2: 继续传输(TD)
    flit2 = CrossRingFlit(
        packet_id=2,
        flit_type="HEAD",
        source=4,
        destination=7,  # 向下
        channel="req"
    )
    flit2.dest_coordinates = (1, 2)
    
    # 添加到IQ FIFO
    inject_input_fifos["req"]["TU"].write_input(flit1)
    inject_input_fifos["req"]["TD"].write_input(flit2)
    
    # 更新FIFO状态
    for fifo in inject_input_fifos["req"].values():
        fifo.step_compute_phase(0)
        fifo.step_update_phase()
    
    rb.step_compute_phase(0)
    rb.step_update_phase()
    
    # 执行仲裁
    rb.compute_arbitration(1, inject_input_fifos)
    
    # 检查传输计划
    decisions = rb.ring_bridge_arbitration_decisions["req"]
    print(f"传输决策数量: {len(decisions)}")
    for decision in decisions:
        if decision["flit"]:
            print(f"  {decision['input_source']} -> {decision['output_direction']}: flit {decision['flit'].packet_id}")
    
    # 执行传输
    rb.execute_arbitration(1, inject_input_fifos)
    
    # 更新FIFO状态
    rb.step_compute_phase(1)
    rb.step_update_phase()
    
    # 验证输出
    eq_fifo = rb.ring_bridge_output_fifos["req"]["EQ"]
    td_fifo = rb.ring_bridge_output_fifos["req"]["TD"]
    
    print(f"\nEQ output FIFO: {len(eq_fifo)} flits")
    print(f"TD output FIFO: {len(td_fifo)} flits")
    
    assert len(decisions) == 2, "应该有2个并行传输"
    print("✅ RingBridge并行传输测试通过")


def test_conflict_scenarios():
    """测试冲突场景：多个flit去往同一资源"""
    print("\n=== 测试冲突场景 ===")
    
    config = create_test_config()
    iq = InjectQueue(node_id=4, coordinates=(1, 1), config=config)
    
    # 连接多个IP
    iq.connect_ip("IP0")
    iq.connect_ip("IP1")
    iq.connect_ip("IP2")
    
    # 创建3个都去往TR方向的flit
    flits = []
    for i in range(3):
        flit = CrossRingFlit(
            packet_id=i,
            flit_type="HEAD",
            source=4,
            destination=5,  # 都向右(TR)
            channel="req"
        )
        flit.dest_coordinates = (2, 1)
        flits.append(flit)
        iq.add_to_inject_queue(flit, "req", f"IP{i}")
    
    # 更新FIFO状态
    iq.step_compute_phase(0)
    iq.step_update_phase()
    
    # 执行仲裁
    iq.compute_arbitration(1)
    
    # 检查传输计划
    print(f"传输计划数量: {len(iq._inject_transfer_plan)}")
    for ip_id, channel, flit, direction in iq._inject_transfer_plan:
        print(f"  IP{ip_id[-1]} -> {direction}: flit {flit.packet_id}")
    
    # 应该只有1个传输（因为都去TR方向）
    assert len(iq._inject_transfer_plan) == 1, "同方向冲突时应该只有1个传输"
    print("✅ 冲突场景测试通过：同方向只允许一个传输")


if __name__ == "__main__":
    print("开始测试CrossRing仲裁并行性...")
    
    try:
        test_inject_queue_parallelism()
        test_eject_queue_parallelism()
        test_ring_bridge_parallelism()
        test_conflict_scenarios()
        
        print("\n🎉 所有测试通过！仲裁逻辑正确支持并行传输")
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        raise