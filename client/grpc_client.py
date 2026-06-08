import grpc
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'proto_generated'))

import surge_monitor_pb2
import surge_monitor_pb2_grpc
from simulator.pressure_simulator import PressureSimulator


def run_grpc_client(server_address="localhost:50051", duration=60.0, sample_rate=100000):
    simulator = PressureSimulator(sample_rate=sample_rate)

    channel = grpc.insecure_channel(server_address)
    stub = surge_monitor_pb2_grpc.PressureSensorServiceServicerStub(channel)

    print(f"Connecting to gRPC server at {server_address}...")

    def batch_generator():
        elapsed = 0.0
        batch_duration = 0.01
        while elapsed < duration:
            data, ts = simulator.generate_batch(batch_duration)
            batch = surge_monitor_pb2.PressureDataBatch(
                sensor_id=1,
                timestamp_base=ts[0],
                sample_rate_hz=sample_rate,
                samples=data.tolist(),
            )
            yield batch
            elapsed += batch_duration
            time.sleep(batch_duration * 0.8)

    try:
        response = stub.StreamPressureData(batch_generator())
        print(f"Server response: accepted={response.accepted}, count={response.received_count}")
    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()} - {e.details()}")
    finally:
        channel.close()


if __name__ == "__main__":
    run_grpc_client()
