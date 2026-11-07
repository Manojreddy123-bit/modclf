import sys, time
import numpy as np
import psutil
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

it = Interpreter(model_path=sys.argv[1], num_threads=2)
it.allocate_tensors()
i = it.get_input_details()[0]['index']
x = np.random.randn(1, 128, 2).astype(np.float32)

for _ in range(50):                      # warm-up
    it.set_tensor(i, x)
    it.invoke()

psutil.cpu_percent(None)
ts = []
for _ in range(1000):
    t = time.perf_counter()
    it.set_tensor(i, x)
    it.invoke()
    ts.append((time.perf_counter() - t) * 1000)

ts = np.array(ts)
print('mean ms', ts.mean().round(3), 'median', np.median(ts).round(3), 'p95', np.percentile(ts, 95).round(3))
print('system CPU % during run', psutil.cpu_percent(None))
