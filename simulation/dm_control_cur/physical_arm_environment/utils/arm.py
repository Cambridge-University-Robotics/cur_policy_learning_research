import abc
from abc import abstractmethod
from typing import Optional

import serial
from serial.tools.list_ports import comports
import re


class AbstractFireflyArm(abc.ABC):

    @abstractmethod
    def connect(self, port: str = None):
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def calibrate(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_state(self) -> Optional[dict]:
        pass

    @abstractmethod
    def set_state(self, x: int, y: int, z: int, pitch: int, roll: int) -> None:
        pass


class PortException(Exception):
    def __init__(self, message=None):
        self.message = message


class FireflyArm(AbstractFireflyArm):
    """
    Usage:
    fa = FireflyArm()
    fa.connect(port='/dev/ttyUSB0')
    fa.calibrate()  # arm needs to be calibrated each startup
    fa.set_state()
    fa.disconnect()

    There is also a terminal() function for command line manipulation.
    """

    def __init__(self):
        self.ser = None
        reg_nums = r'[^0-9]*([0-9]*[.][0-9]*)[^0-9]*([0-9]*[.][0-9]*)[^0-9]*([0-9]*[.][0-9]*)[^0-9]*([0-9]*[.][' \
                   r'0-9]*)[^0-9]*([0-9]*[.][0-9]*).* '
        self.pattern_nums = re.compile(reg_nums)
        reg_setstate = r'SETSTATE ([0-9]*)[^0-9]*([0-9]*)[^0-9]*([0-9]*)[^0-9]*([0-9]*)[^0-9]*([0-9]*)[^0-9]*'
        self.pattern_setstate = re.compile(reg_setstate)

    def __del__(self):
        self.disconnect()

    def connect(self, port: str = None) -> None:
        if port is None:
            ports = comports()
            if len(ports) > 0:  # take the first device's value
                port = ports[0].device
            else:
                raise PortException("Port not found")

        self.ser = serial.Serial(
            port=port,
            timeout=30,
            baudrate=19200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_TWO,
            bytesize=serial.EIGHTBITS
        )  # will raise SerialException if fail

    def disconnect(self) -> None:
        self.ser.close()

    def _write(self, msg: str) -> None:
        msg = bytes(msg + '\r\n', 'utf-8')
        self.ser.write(msg)

    def _read(self) -> str:
        reply = ''
        while True:
            if reply.endswith('OK') or reply.endswith('ABORTED'):
                break
            else:
                reply += self.ser.read().decode('latin_1')
        return reply

    def command(self, msg: str) -> str:
        self._write(msg)
        return self._read()

    def command_print(self, cmd: str) -> None:
        print(self.command(cmd))

    def commands_print(self, *cmds: str) -> None:
        for cmd in cmds:
            self.command_print(cmd)

    def reset(self) -> None:
        self.commands_print(
            'TELL SHOULDER 2800 MOVETO',
            'TELL ELBOW 9000 MOVETO',
            'TELL L-HAND -1240 MOVETO',
            'TELL WRIST 10510 MOVETO',
        )

    def calibrate(self) -> None:
        self.commands_print(
            'ROBOFORTH',
            'START',
            'DE-ENERGISE',
        )
        input('Press a key when in home position')
        self.commands_print(
            'ENERGISE',
            'CALIBRATE',
            'JOINT',
        )

    def get_state(self) -> Optional[dict]:  # returns a dict of state key-value pairs
        self.command('CARTESIAN')
        replies = self.command('COMPUTE CARTWHERE')
        replies = replies.splitlines()[3:-1]  # prev is thrown away
        curr, prev = replies
        a = self.pattern_nums.match(curr)
        if a is not None:
            keys = ('x', 'y', 'z', 'pitch', 'roll')
            return dict(zip(keys, a.group(*range(1, 6))))
        else:
            return None

    def set_state(self, x: int, y: int, z: int, pitch: int, roll: int) -> None:  # move arm by tuple
        self.command_print('CARTESIAN')
        x, y, z = map(lambda x: x * 10, (x, y, z))  # times 10 for units
        self.command_print(f'{x} {y} {z} MOVE')
        pitch = int(pitch / 360 * 18000)  # 360 degrees is 18000
        roll = int(roll / 360 * 18000)
        self.commands_print(
            'JOINT',
            f'TELL L-HAND {pitch} MOVE',
            f'TELL WRIST {roll} MOVE',
        )

    def terminal(self) -> None:  # command line interface
        while True:
            command = input('Write a command or DONE')
            if command == 'DONE':
                break
            elif command == 'DEFAULT':
                self.reset()
            elif command == 'GETSTATE':
                print(self.get_state())
            elif command == 'r':
                self.command_print('JOINT')
                self.command_print('TELL WRIST 18000 MOVE')
            elif command.startswith('SETSTATE'):
                matches = self.pattern_setstate.match(command)
                self.set_state(*map(int, matches.group(*range(1, 6))))
            else:
                self.command_print(command)
