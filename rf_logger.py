import json
import os
import sqlite3
import paho.mqtt.client as mqtt

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rf.db')
db = sqlite3.connect(DB, check_same_thread=False)
db.execute('CREATE TABLE IF NOT EXISTS detections(ts REAL, modulation TEXT, truth TEXT, confidence REAL, snr_db INTEGER, latency_ms REAL)')
db.commit()

def on_connect(c, userdata, flags, reason_code, properties):
    c.subscribe('rf/detections')
    print('connected, logging to', DB)

def on_message(c, userdata, m):
    d = json.loads(m.payload)
    db.execute('INSERT INTO detections VALUES (?,?,?,?,?,?)',
               (d['ts'], d['modulation'], d['truth'], d['confidence'], d['snr_db'], d['latency_ms']))
    db.commit()

c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id='rf-logger')
c.on_connect = on_connect
c.on_message = on_message
c.connect('localhost', 1883, 60)
c.loop_forever()
