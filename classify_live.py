import json
import time
import numpy as np
import paho.mqtt.client as mqtt
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter
from signals import generate, channel, to_iq

MODEL = 'model_int8.tflite'
LABELS = json.load(open('labels.json'))          # same order as training
BROKER, PORT, TOPIC = 'localhost', 1883, 'rf/detections'
SNRS = list(range(-10, 21, 2))
rng = np.random.default_rng()

it = Interpreter(model_path=MODEL, num_threads=2)
it.allocate_tensors()
inp = it.get_input_details()[0]['index']
out = it.get_output_details()[0]['index']

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id='modclf')
# client.username_pw_set('user', 'password')     # only if your broker requires it
client.connect(BROKER, PORT, 60)
client.loop_start()
print('publishing to', TOPIC, '- press Ctrl+C to stop')

while True:
    truth = LABELS[int(rng.integers(len(LABELS)))]
    snr = int(rng.choice(SNRS))
    frame = to_iq(channel(generate(truth, rng=rng), snr, rng))[None]   # shape (1, 128, 2)
    t0 = time.perf_counter()
    it.set_tensor(inp, frame)
    it.invoke()
    probs = it.get_tensor(out)[0]
    ms = (time.perf_counter() - t0) * 1000
    k = int(probs.argmax())
    msg = {
        'modulation': LABELS[k],
        'confidence': round(float(probs[k]), 3),
        'truth': truth,
        'snr_db': snr,
        'latency_ms': round(ms, 3),
        'ts': time.time(),
    }
    client.publish(TOPIC, json.dumps(msg))
    print(truth, '->', LABELS[k], msg['confidence'], 'snr', snr)
    time.sleep(1)
