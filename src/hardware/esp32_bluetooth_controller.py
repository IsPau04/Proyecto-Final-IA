import time
from typing import List, Optional

import serial
from serial import SerialException


class ESP32BluetoothController:
    def __init__(self, port: str = "COM4", baudrate: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
        except SerialException as exc:
            raise ConnectionError(
                f"No se pudo abrir el puerto {port}. "
                "Verifica que la ESP32 este emparejada por Bluetooth clasico, "
                "que el puerto COM sea correcto y que no este ocupado."
            ) from exc
        time.sleep(2)

    def send_command(self, command: str) -> None:
        command = command.strip().upper()

        if not command:
            return

        if command == "RUIDO_FONDO":
            print("ESP32: comando ignorado porque es RUIDO_FONDO.")
            return

        try:
            self.ser.write((command + "\n").encode("utf-8"))
        except SerialException as exc:
            raise ConnectionError(
                f"No se pudo enviar el comando por {self.port}. "
                "Verifica que el puerto COM siga disponible."
            ) from exc
        print(f"Enviado a ESP32: {command}")

    def read_responses(self) -> List[str]:
        responses = []
        time.sleep(0.15)

        try:
            while self.ser.in_waiting:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    responses.append(line)
        except SerialException as exc:
            raise ConnectionError(
                f"No se pudieron leer respuestas desde {self.port}."
            ) from exc

        return responses

    def read_line(self) -> Optional[str]:
        """Lee una linea desde la ESP32 si hay datos disponibles."""
        try:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
        except SerialException as exc:
            raise ConnectionError(
                f"No se pudieron leer datos desde {self.port}."
            ) from exc

        return line or None

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
