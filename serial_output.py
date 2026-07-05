import serial
import serial.tools.list_ports
import time


def find_esp32_port():
    """
    Auto-detect the ESP32/Arduino serial port.
    Looks for common USB-serial chip descriptions.
    """
    ports = serial.tools.list_ports.comports()
    for port in ports:
        desc = port.description.lower()
        if any(keyword in desc for keyword in [
            'cp210', 'ch340', 'ch341', 'ftdi', 'arduino', 'esp32', 'usb serial'
        ]):
            print(f"Found ESP32/Arduino on {port.device} ({port.description})")
            return port.device
    return None


class SerialOutput:
    """
    Sends roll/pitch/yaw to ESP32 over serial.
    Fails gracefully if no device is connected.
    """
    def __init__(self, port=None, baud=115200):
        self.ser = None
        self.enabled = False

        try:
            # Auto-detect if no port specified
            if port is None:
                port = find_esp32_port()

            if port is None:
                print("Serial: No ESP32/Arduino found — running without serial output")
                return

            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2.0)  # wait for ESP32 to reset after connection
            self.enabled = True
            print(f"Serial: Connected on {port} at {baud} baud")

        except Exception as e:
            print(f"Serial: Failed to connect — {e}")
            print("Serial: Running without serial output")

    def send(self, roll: float, pitch: float, yaw: float, status: str = "OK"):
        if not self.enabled or self.ser is None:
            return

        try:
            #msg = f"C,{roll:+.2f},{pitch:+.2f},{yaw:+.2f}\n"
            msg = f"C,{int(roll)},{int(pitch)},{int(yaw)}\n"
            self.ser.write(msg.encode())
        except Exception as e:
            print(f"Serial: Send failed — {e}")
            self.enabled = False

    def send_no_marker(self):
        pass
    def receive(self):
        if not self.enabled:
            return None

        try:
            if self.ser.in_waiting:
                line = self.ser.readline().decode().strip()

                if line.startswith("I,"):
                    data = line.split(",")

                    return (
                        float(data[1]),
                        float(data[2]),
                        float(data[3])
                    )

        except:
            pass

        return None

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial: Connection closed")
