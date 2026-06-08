import sys
import os
import time
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'proto_generated'))

from buffer.lockfree_ring import LockFreeRingBuffer
from simulator.pressure_simulator import PressureSimulator
from dashboard.app import run_dashboard


def main():
    print("=" * 60)
    print("  涡扇发动机喘振监控大屏 - Aero Engine Surge Monitor")
    print("=" * 60)
    print()
    print("  架构说明:")
    print("  - 压力传感器模拟器: 100kHz 采样率, Morlet CWT 频域分析")
    print("  - 无锁环形缓冲区: 内存级数据流转, 秒级切片")
    print("  - Dash/Plotly 大屏: 实时波形 + 频谱热力图 + 频带能量")
    print()
    print("  启动中...")
    print()

    run_dashboard(host="0.0.0.0", port=8051, with_simulator=True)


if __name__ == "__main__":
    main()
