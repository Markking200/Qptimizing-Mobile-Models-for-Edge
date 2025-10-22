import numpy as np
import torch
import time

#for later testing

@torch.no_grad()
def measure_latency(model, input_shape=(1, 3, 224, 224), repetitions=100):
    torch.backends.quantized.engine = 'qnnpack'
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape, device=device)

    # GPU warm-up
    for _ in range(10):
        _ = model(dummy_input)

    # Measure inference time
    timings = []
    for _ in range(repetitions):
        starter = time.perf_counter()
        _ = model(dummy_input)
        ender = time.perf_counter()
        timings.append((ender - starter) * 1000) # Convert to milliseconds

    latency = np.mean(timings)
    std = np.std(timings)
    return latency, std