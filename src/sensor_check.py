import serial, time
s = serial.Serial('/dev/ttyUSB0', 921600, timeout=0.5)
s.setDTR(False); s.setRTS(True); time.sleep(0.2); s.setRTS(False)   # reset the ESP32
end, data, errors = time.time() + 6, 0, 0
while time.time() < end:
    l = s.readline().decode(errors='ignore').strip()
    if not l: continue
    if 'I2C read error' in l: errors += 1
    elif l.startswith('#'): print(l)
    elif l.count(',') == 7: data += 1
print(f'--> data lines: {data}, I2C error reports: {errors}')
