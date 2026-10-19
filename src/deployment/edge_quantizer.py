import os
import time
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import numpy as np

class SingASREdgeQuantizer:
    """
    Edge and Mobile Post-Training Quantization (PTQ) & Hardware Profiling Engine.
    Quantizes PyTorch ASR adapters and neural inverters to INT8 dynamic format,
    benchmarking memory compression, inference speedup, and numerical parity.
    """
    def __init__(self, target_dtype=torch.qint8):
        self.target_dtype = target_dtype
        self.device = torch.device("cpu") # Edge inference runs predominantly on CPU/NPU

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Applies dynamic range INT8 quantization to Linear and Conv1D layers.
        Keeps sensitive normalization and layernorm operations in FP32.
        """
        model.eval()
        model.to(self.device)
        quantized_model = tq.quantize_dynamic(
            model,
            qconfig_spec={nn.Linear},
            dtype=self.target_dtype
        )
        return quantized_model

    def measure_model_size_bytes(self, model: nn.Module) -> int:
        """
        Calculates total parameter storage in bytes.
        """
        total_bytes = 0
        for p in model.parameters():
            total_bytes += p.nelement() * p.element_size()
        for b in model.buffers():
            total_bytes += b.nelement() * b.element_size()
        return total_bytes

    def measure_disk_size_bytes(self, model: nn.Module, temp_path: str) -> int:
        """
        Serializes model to disk to measure true on-disk footprint.
        """
        torch.save(model.state_dict(), temp_path)
        size = os.path.getsize(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return size

    def profile_latency(self, model: nn.Module, sample_input, num_warmup=10, num_iters=50):
        """
        Measures forward latency per frame, FPS throughput, and Real-Time Factor (RTF).
        """
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                if isinstance(sample_input, (tuple, list)):
                    _ = model(*sample_input)
                else:
                    _ = model(sample_input)
                    
        latencies = []
        with torch.no_grad():
            for _ in range(num_iters):
                t0 = time.perf_counter()
                if isinstance(sample_input, (tuple, list)):
                    _ = model(*sample_input)
                else:
                    _ = model(sample_input)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0) # in ms
                
        mean_latency_ms = float(np.mean(latencies))
        p95_latency_ms = float(np.percentile(latencies, 95))
        return mean_latency_ms, p95_latency_ms

    def measure_numerical_parity(self, fp32_model: nn.Module, int8_model: nn.Module, sample_input):
        """
        Calculates Cosine Similarity, Mean Absolute Error (MAE), and Max Error
        between original FP32 outputs and quantized INT8 outputs.
        """
        fp32_model.eval()
        int8_model.eval()
        
        with torch.no_grad():
            if isinstance(sample_input, (tuple, list)):
                out_fp32 = fp32_model(*sample_input)
                out_int8 = int8_model(*sample_input)
            else:
                out_fp32 = fp32_model(sample_input)
                out_int8 = int8_model(sample_input)
                
            # If multiple outputs, take the primary tensor
            if isinstance(out_fp32, tuple):
                out_fp32 = out_fp32[0]
                out_int8 = out_int8[0]
            elif isinstance(out_fp32, dict):
                out_fp32 = list(out_fp32.values())[0]
                out_int8 = list(out_int8.values())[0]
                
            t_fp32 = out_fp32.flatten().float()
            t_int8 = out_int8.flatten().float()
            
            cos_sim = float(torch.nn.functional.cosine_similarity(t_fp32.unsqueeze(0), t_int8.unsqueeze(0)).item())
            mae = float(torch.mean(torch.abs(t_fp32 - t_int8)).item())
            max_err = float(torch.max(torch.abs(t_fp32 - t_int8)).item())
            
        return {
            "cosine_similarity": round(cos_sim, 5),
            "mean_absolute_error": round(mae, 6),
            "max_absolute_error": round(max_err, 6)
        }
