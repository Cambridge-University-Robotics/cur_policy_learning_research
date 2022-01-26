import serial


class Object:
    """
    Usage:
    o=Object()
    o.connect()
    while True:
        print(o.get_state())
    """
    def __init__(self):
        self.ser = None

    def __del__(self):
        self.disconnect()

    def connect(self, port: str = None) -> None:
        port = port or '/dev/ttyACM0'
        self.ser = serial.Serial(port=port, timeout=0.1)

    def disconnect(self) -> None:
        self.ser.close()

    def get_state(self) -> dict:
        self.ser.read_all()
        self.ser.readline()
        while True:
            try:
                u = self.ser.readline().decode('utf-8')
                if len(u) > 0:
                    return {'height': int(u)}
            except Exception as e:
                print(e)
                continue
