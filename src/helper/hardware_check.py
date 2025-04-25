import psutil    # psutil
import GPUtil    # gputil
from cpuinfo import get_cpu_info    # py-cpuinfo



def get_hardware() -> str:
    """
    Gets the current detected hardware and ai support.
    """
    harware_info = ""
    harware_info += f"\n{'-'*32} \nYour Hardware:\n"

    # General
    harware_info += f"\n    ---> General <---"
    harware_info += f"\nOperatingsystem: {platform.system()}"
    harware_info += f"\nVersion: {platform.version()}"
    harware_info += f"\nArchitecture: {platform.architecture()}"
    harware_info += f"\nProcessor: {platform.processor()}"

    # GPU-Information
    harware_info += f"\n\n    ---> GPU <---"
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        harware_info += f"\nGPU Name: {gpu.name}"
        harware_info += f"\nVRAM Total: {int(gpu.memoryTotal)} MB"
        harware_info += f"\nVRAM Used: {int(gpu.memoryUsed)} MB"
        harware_info += f"\nUtilization: {round(gpu.load * 100, 1)} %"
    try:
        import torch
        gpus = [torch.cuda.get_device_name(device_nr) for device_nr in range(torch.cuda.device_count())]
        torch_support = False
        if torch.cuda.is_available():
            torch_support = True 
            gpu_str = f"({','.join(gpus)})"
        gpu_addition = f" {gpu_str}" if torch_support else ""
        harware_info += f"\nPyTorch Support: {torch_support}{gpu_addition}"
    except Exception:
        harware_info += f"\nPyTorch Support: False -> not installed"
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        tf_support = False
        if len(gpus) > 0:
            tf_support = True 
            gpu_str = f"({','.join(gpus)})"
        gpu_addition = f" {gpu_str}" if tf_support else ""
        harware_info += f"\nTensorFlow Support: {tf_support}{gpu_addition}" 
    except Exception:
        harware_info += f"\nTensorFlow Support: False -> not installed"

    # CPU-Information
    harware_info += f"\n\n    ---> CPU <---"
    cpu_info = get_cpu_info()
    harware_info += f"\nCPU-Name: {cpu_info['brand_raw']}"
    harware_info += f"\nCPU Kernels: {psutil.cpu_count(logical=False)}"
    harware_info += f"\nLogical CPU-Kernels: {psutil.cpu_count(logical=True)}"
    harware_info += f"\nCPU-Frequence: {int(psutil.cpu_freq().max)} MHz"
    harware_info += f"\nCPU-Utilization: {round(psutil.cpu_percent(interval=1), 1)} %"
    
    # RAM-Information
    harware_info += f"\n\n    ---> RAM <---"
    ram = psutil.virtual_memory()
    harware_info += f"\nRAM Total: {ram.total // (1024**3)} GB"
    harware_info += f"\nRAM Available: {ram.available // (1024**3)} GB"
    harware_info += f"\nRAM-Utilization: {round(ram.percent, 1)} %"

    harware_info += f"\n\n{'-'*32}"

    return harware_info





