import grpc
from concurrent import futures
import time
import threading
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'proto_generated'))

import surge_monitor_pb2
import surge_monitor_pb2_grpc
from buffer.lockfree_ring import LockFreeRingBuffer


class PressureSensorServicer(surge_monitor_pb2_grpc.PressureSensorServiceServicer):
    def __init__(self, ring_buffer: LockFreeRingBuffer):
        self._buffer = ring_buffer
        self._lock = threading.Lock()
        self._total_received = 0
        self._healthy = True

    def StreamPressureData(self, request_iterator, context):
        for batch in request_iterator:
            samples = np.array(batch.samples, dtype=np.float32)
            if len(samples) == 0:
                continue

            self._buffer.sample_rate = batch.sample_rate_hz
            n = len(samples)
            dt = 1.0 / batch.sample_rate_hz
            timestamps = np.arange(n, dtype=np.float64) * dt + batch.timestamp_base

            self._buffer.write(samples, timestamps)

            with self._lock:
                self._total_received += n

        return surge_monitor_pb2.StreamAck(
            accepted=True,
            received_count=self._total_received,
            message="OK",
        )

    def GetStatus(self, request, context):
        return surge_monitor_pb2.SystemStatus(
            healthy=self._healthy,
            buffer_utilization_pct=self._buffer.utilization,
            latest_timestamp=self._buffer.latest_timestamp,
            total_samples_received=self._total_received,
        )


def serve(ring_buffer: LockFreeRingBuffer, port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    surge_monitor_pb2_grpc.add_PressureSensorServiceServicer_to_server(
        PressureSensorServicer(ring_buffer), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"gRPC server started on port {port}")
    return server
