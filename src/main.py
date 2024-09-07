import tensorflow as tf
import getpass

def print_gpu_status():
    gpus = tf.config.list_physical_devices('GPU')  # List all physical GPUs

    if not gpus:
        print("No GPUs found.")
    else:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}, Type: {gpu.device_type}")
            details = tf.config.experimental.get_device_details(gpu)
            print(f"  Name: {details.get('device_name', 'Unknown')}")
            print(f"  Memory: {details.get('memory_limit_bytes', 'Unknown')} bytes")
            print(f"  Compute Capability: {details.get('compute_capability', 'Unknown')}")

if __name__ == "__main__":
    print(f"Hello, {getpass.getuser()}!")
    print("Checking GPU status...")
    print_gpu_status()
