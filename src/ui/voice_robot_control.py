"""Interfaz para ejecutar inferencia de voz y escuchar botones de ESP32."""

import contextlib
import importlib
import io
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from src.hardware.esp32_bluetooth_controller import ESP32BluetoothController


DEFAULT_PORT = "COM4"
BACKGROUND_LABEL = "RUIDO_FONDO"
SIMPLE_DURATION = 2.0
SIMPLE_THRESHOLD = 0.60
SEQUENCE_DURATION = 10.0
SEQUENCE_THRESHOLD = 0.78
SEQUENCE_ENERGY_THRESHOLD = 0.003
SEQUENCE_MIN_SEGMENT_DURATION = 0.25
SEQUENCE_MIN_SILENCE_DURATION = 0.45
SEQUENCE_SEGMENT_PADDING = 0.70
VALID_COMMANDS = {"ABRE", "CIERRA", "SUBE", "BAJA", "POSICION_INICIAL"}
predict_live_once = None
predict_sequence_live = None


class VoiceRobotControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Control de brazo robótico por voz")
        self.root.geometry("980x680")
        self.root.minsize(880, 620)

        self.controller = None
        self.resources = None
        self.busy = False
        self.connecting = False
        self.listening = False
        self.closing = False
        self.listener_thread = None
        self.ui_queue = queue.Queue()
        self.serial_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.status_var = tk.StringVar(value="Desconectado")
        self.port_var = tk.StringVar(value=DEFAULT_PORT)
        self.connection_var = tk.StringVar(value="Puerto actual: COM4")

        self.configure_style()
        self.build_layout()
        self.refresh_controls()
        self.root.after(50, self.process_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_style(self):
        self.root.configure(bg="#eef2f6")
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", font=("Segoe UI", 10))
        self.style.configure("Root.TFrame", background="#eef2f6")
        self.style.configure("Panel.TFrame", background="#ffffff", relief="flat")
        self.style.configure("Header.TLabel", background="#eef2f6", foreground="#172033")
        self.style.configure(
            "Title.TLabel",
            background="#eef2f6",
            foreground="#172033",
            font=("Segoe UI", 22, "bold"),
        )
        self.style.configure(
            "Subtitle.TLabel",
            background="#eef2f6",
            foreground="#536174",
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "Status.TLabel",
            background="#0f6b57",
            foreground="#ffffff",
            font=("Segoe UI", 20, "bold"),
            anchor="center",
            padding=(18, 28),
        )
        self.style.configure("PanelTitle.TLabel", background="#ffffff", foreground="#172033")
        self.style.configure("PanelText.TLabel", background="#ffffff", foreground="#3f4c5f")
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 10))
        self.style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 10))
        self.style.configure("Action.TButton", padding=(12, 10))
        self.style.map(
            "Primary.TButton",
            background=[("active", "#0b5f4f"), ("!disabled", "#0f6b57")],
            foreground=[("!disabled", "#ffffff")],
        )
        self.style.map(
            "Danger.TButton",
            background=[("active", "#8f2d2d"), ("!disabled", "#a13a3a")],
            foreground=[("!disabled", "#ffffff")],
        )

    def build_layout(self):
        main = ttk.Frame(self.root, style="Root.TFrame", padding=18)
        main.pack(fill="both", expand=True)

        header = ttk.Frame(main, style="Root.TFrame")
        header.pack(fill="x", pady=(0, 14))
        ttk.Label(
            header,
            text="Control de brazo robótico por voz",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Interfaz local por Bluetooth clásico para la ESP32",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        content = ttk.Frame(main, style="Root.TFrame")
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=0, minsize=350)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left = ttk.Frame(content, style="Root.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))

        right = ttk.Frame(content, style="Root.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self.build_status_panel(left)
        self.build_connection_panel(left)
        self.build_button_panel(left)
        self.build_instruction_panel(left)
        self.build_log_panel(right)

    def build_status_panel(self, parent):
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        panel.pack(fill="x", pady=(0, 12))
        ttk.Label(panel, text="Estado del sistema", style="PanelTitle.TLabel").pack(anchor="w")
        self.status_label = ttk.Label(panel, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(fill="x", pady=(10, 0))

    def build_connection_panel(self, parent):
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        panel.pack(fill="x", pady=(0, 12))
        ttk.Label(panel, text="Conexión ESP32", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(panel, textvariable=self.connection_var, style="PanelText.TLabel").pack(
            anchor="w",
            pady=(8, 4),
        )

        row = ttk.Frame(panel, style="Panel.TFrame")
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Puerto:", style="PanelText.TLabel").pack(side="left")
        self.port_combo = ttk.Combobox(
            row,
            textvariable=self.port_var,
            values=self.get_default_ports(),
            width=14,
        )
        self.port_combo.pack(side="left", padx=(8, 0), fill="x", expand=True)

    def build_button_panel(self, parent):
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        panel.pack(fill="x", pady=(0, 12))
        ttk.Label(panel, text="Controles", style="PanelTitle.TLabel").pack(anchor="w")

        grid = ttk.Frame(panel, style="Panel.TFrame")
        grid.pack(fill="x", pady=(10, 0))
        for column in range(2):
            grid.columnconfigure(column, weight=1, uniform="buttons")

        self.connect_button = ttk.Button(
            grid,
            text="Conectar ESP32",
            style="Primary.TButton",
            command=self.connect_esp32,
        )
        self.connect_button.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=4)

        self.disconnect_button = ttk.Button(
            grid,
            text="Desconectar ESP32",
            style="Danger.TButton",
            command=self.disconnect_esp32,
        )
        self.disconnect_button.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=4)

        self.listen_button = ttk.Button(
            grid,
            text="Escuchar botones físicos",
            style="Action.TButton",
            command=self.start_listening,
        )
        self.listen_button.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=4)

        self.stop_listen_button = ttk.Button(
            grid,
            text="Detener escucha",
            style="Action.TButton",
            command=self.stop_listening,
        )
        self.stop_listen_button.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=4)

        self.simple_button = ttk.Button(
            grid,
            text="Grabar comando simple",
            style="Action.TButton",
            command=lambda: self.start_simple(source="interfaz"),
        )
        self.simple_button.grid(row=2, column=0, sticky="ew", padx=(0, 6), pady=4)

        self.sequence_button = ttk.Button(
            grid,
            text="Grabar secuencia",
            style="Action.TButton",
            command=lambda: self.start_sequence(source="interfaz"),
        )
        self.sequence_button.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=4)

    def build_instruction_panel(self, parent):
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        panel.pack(fill="x")
        ttk.Label(panel, text="Instrucciones", style="PanelTitle.TLabel").pack(anchor="w")
        instructions = (
            "1. Conecte la ESP32.\n"
            "2. Presione 'Escuchar botones físicos'.\n"
            "3. Use el botón físico simple o secuencia.\n"
            "4. Espere la predicción y ejecución del servo."
        )
        ttk.Label(
            panel,
            text=instructions,
            style="PanelText.TLabel",
            justify="left",
            wraplength=310,
        ).pack(anchor="w", pady=(8, 0))

    def build_log_panel(self, parent):
        ttk.Label(parent, text="Log técnico", style="Header.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 8),
        )
        self.output = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            height=24,
            bg="#101722",
            fg="#dce7f3",
            insertbackground="#dce7f3",
            relief="flat",
            borderwidth=0,
            font=("Consolas", 10),
        )
        self.output.grid(row=1, column=0, sticky="nsew")
        self.output.insert(tk.END, "Listo. Puerto actual: COM4\n")
        self.output.configure(state=tk.DISABLED)

    def get_default_ports(self):
        ports = [DEFAULT_PORT, "COM3", "COM5", "COM6", "COM7", "COM8"]
        try:
            from serial.tools import list_ports

            detected = [port.device for port in list_ports.comports()]
            for port in detected:
                if port not in ports:
                    ports.append(port)
        except Exception:
            pass
        return ports

    def log(self, text):
        if threading.current_thread() is threading.main_thread():
            self._append_output(text)
            return None
        self.ui_queue.put(("log", text))

    def _append_output(self, text):
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, text)
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def set_status(self, text):
        if threading.current_thread() is threading.main_thread():
            self._set_status(text)
            return None
        self.ui_queue.put(("status", text))

    def _set_status(self, text):
        self.status_var.set(text)

    def process_ui_queue(self):
        try:
            while True:
                message = self.ui_queue.get_nowait()
                message_type = message[0]
                if message_type == "log":
                    self._append_output(message[1])
                elif message_type == "status":
                    self._set_status(message[1])
                elif message_type == "refresh_controls":
                    self.refresh_controls()
                elif message_type == "show_error":
                    _message_type, title, text = message
                    messagebox.showerror(title, text)
                elif message_type == "stop_listening":
                    self.stop_listening()
                elif message_type == "connect_success":
                    _message_type, controller, port = message
                    self.handle_connect_success(controller, port)
                elif message_type == "connect_error":
                    _message_type, port, text = message
                    self.handle_connect_error(port, text)
        except queue.Empty:
            pass

        if not self.closing:
            self.root.after(50, self.process_ui_queue)

    def refresh_controls(self):
        connected = self.controller is not None
        busy = self.is_busy()

        self.connect_button.config(
            state=tk.DISABLED if connected or self.connecting else tk.NORMAL
        )
        self.disconnect_button.config(state=tk.NORMAL if connected else tk.DISABLED)
        self.listen_button.config(
            state=tk.NORMAL if connected and not self.listening else tk.DISABLED
        )
        self.stop_listen_button.config(state=tk.NORMAL if self.listening else tk.DISABLED)
        record_state = tk.NORMAL if connected and not busy else tk.DISABLED
        self.simple_button.config(state=record_state)
        self.sequence_button.config(state=record_state)
        self.port_combo.config(state=tk.DISABLED if connected or self.connecting else tk.NORMAL)

    def set_busy(self, busy):
        with self.state_lock:
            self.busy = busy
        self.ui_queue.put(("refresh_controls",))

    def is_busy(self):
        with self.state_lock:
            return self.busy

    def get_selected_port(self):
        return self.port_var.get().strip() or DEFAULT_PORT

    def connect_esp32(self):
        if self.controller is not None:
            self.log("ESP32 ya conectada. Se reutiliza la conexión abierta.\n")
            self.set_status("ESP32 conectada")
            return True

        if self.connecting:
            self.log("Conexión ESP32 en progreso.\n")
            return False

        port = self.get_selected_port()
        self.connecting = True
        self.connection_var.set(f"Puerto actual: {port}")
        self.set_status("Conectando ESP32...")
        self.log(f"Conectando ESP32 en {port}.\n")
        self.refresh_controls()

        thread = threading.Thread(
            target=self.connect_worker,
            args=(port,),
            daemon=True,
        )
        thread.start()
        return False

    def connect_worker(self, port):
        try:
            controller = ESP32BluetoothController(port=port)
            try:
                controller.ser.timeout = 0.05
            except Exception:
                pass
        except ConnectionError as exc:
            message = (
                f"No se pudo conectar con ESP32 en {port}.\n"
                "El puerto puede no existir, no estar emparejado o estar ocupado.\n"
                f"Detalle: {exc}"
            )
            self.ui_queue.put(("connect_error", port, message))
            return

        self.ui_queue.put(("connect_success", controller, port))

    def handle_connect_success(self, controller, port):
        self.connecting = False
        if self.closing:
            controller.close()
            return
        if self.controller is not None:
            controller.close()
            self.log("Ya existía una conexión ESP32; se conserva la conexión abierta.\n")
            self.refresh_controls()
            return
        self.controller = controller
        self.set_status("ESP32 conectada")
        self.log(f"ESP32 conectada en {port}.\n")
        self.refresh_controls()

    def handle_connect_error(self, port, message):
        self.connecting = False
        self.controller = None
        self.set_status("Desconectado")
        self.log(message + "\n")
        messagebox.showerror("Error ESP32", message)
        self.refresh_controls()

    def disconnect_esp32(self):
        self.stop_listening()
        if self.connecting:
            self.log("Hay una conexión en progreso; espere a que termine.\n")
            return
        if self.controller is None:
            self.log("ESP32 ya estaba desconectada.\n")
            self.set_status("Desconectado")
            self.refresh_controls()
            return None

        with self.serial_lock:
            self.controller.close()
            self.controller = None
        self.set_status("Desconectado")
        self.log("ESP32 desconectada.\n")
        self.refresh_controls()

    def start_listening(self):
        if self.listening:
            return
        if not self.connect_esp32():
            return

        self.listening = True
        self.listener_thread = threading.Thread(target=self.listen_worker, daemon=True)
        self.listener_thread.start()
        self.log("Escuchando botones físicos de ESP32.\n")
        self.refresh_controls()

    def stop_listening(self):
        if not self.listening:
            self.refresh_controls()
            return
        self.listening = False
        self.log("Escucha de botones físicos detenida.\n")
        self.refresh_controls()
        if self.controller is not None:
            self.set_status("ESP32 conectada")
        else:
            self.set_status("Desconectado")

    def listen_worker(self):
        while self.listening and self.controller is not None:
            try:
                with self.serial_lock:
                    controller = self.controller
                    if controller is None:
                        return
                    line = self.read_line_nonblocking(controller)
            except ConnectionError as exc:
                self.log(f"Error ESP32: {exc}\n")
                self.ui_queue.put(("stop_listening",))
                return

            if not line:
                time.sleep(0.02)
                continue

            self.handle_esp32_line(line)

    def read_line_nonblocking(self, controller):
        try:
            if not controller.ser.in_waiting:
                return None
        except Exception as exc:
            raise ConnectionError(f"No se pudo consultar el puerto {controller.port}.") from exc

        return controller.read_line()

    def handle_esp32_line(self, line):
        self.log(f"ESP32: {line}\n")

        if line == "BTN:SIMPLE":
            self.log("Evento recibido: BTN:SIMPLE\n")
            self.start_simple(source="botón físico")
        elif line == "BTN:SEQUENCE":
            self.log("Evento recibido: BTN:SEQUENCE\n")
            self.start_sequence(source="botón físico")
        elif line == "BTN:HOME":
            self.log("Evento recibido: BTN:HOME. Enviando POSICION_INICIAL.\n")
            self.start_send_command("POSICION_INICIAL")
        elif line == "BTN:STOP":
            self.log("Evento recibido: BTN:STOP. No se envía comando de voz.\n")
        elif line in {"OK", "DONE"}:
            self.log(f"Respuesta ESP32: {line}\n")

    def ensure_resources(self):
        live_module = self.get_predict_live_once()
        if self.resources is None:
            self.log("Cargando modelo de IA...\n")
            self.resources = live_module.load_inference_resources()
            self.log("Modelo de IA cargado.\n")
        return self.resources

    def get_predict_live_once(self):
        global predict_live_once
        if predict_live_once is None:
            predict_live_once = importlib.import_module("src.inference.predict_live_once")
        return predict_live_once

    def get_predict_sequence_live(self):
        global predict_sequence_live
        if predict_sequence_live is None:
            predict_sequence_live = importlib.import_module(
                "src.inference.predict_sequence_live"
            )
        return predict_sequence_live

    def start_simple(self, source):
        if self.is_busy():
            self.log("Sistema ocupado: se ignora la solicitud de comando simple.\n")
            return
        if self.controller is None and not self.connect_esp32():
            return

        thread = threading.Thread(
            target=self.simple_worker,
            args=(source,),
            daemon=True,
        )
        thread.start()

    def start_sequence(self, source):
        if self.is_busy():
            self.log("Sistema ocupado: se ignora la solicitud de secuencia.\n")
            return
        if self.controller is None and not self.connect_esp32():
            return

        thread = threading.Thread(
            target=self.sequence_worker,
            args=(source,),
            daemon=True,
        )
        thread.start()

    def simple_worker(self, source):
        self.set_busy(True)
        self.set_status("Escuchando comando simple...")
        self.log(f"\nGrabando comando simple desde {source}.\n")
        worker_start = time.perf_counter()
        record_end = None
        try:
            live_module = self.get_predict_live_once()
            resources = self.ensure_resources()
            model, labels, mean, std, input_shape = resources
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                record_start = time.perf_counter()
                audio = live_module.record_audio(SIMPLE_DURATION)
                record_end = time.perf_counter()

                self.set_status("Procesando audio...")
                inference_start = time.perf_counter()
                X = live_module.prepare_audio_for_model(audio, input_shape, mean, std)
                probabilities = model.predict(X, verbose=0)[0]
                label, confidence = live_module.print_prediction(
                    labels,
                    probabilities,
                    SIMPLE_THRESHOLD,
                )
                inference_end = time.perf_counter()
            self.log(buffer.getvalue())
            self.log(f"Tiempo de grabación: {record_end - record_start:.3f} s\n")
            self.log(
                "Tiempo de inferencia/preprocesamiento: "
                f"{inference_end - inference_start:.3f} s\n"
            )
            self.log(f"Clase predicha: {label}\n")
            self.log(f"Confianza: {confidence:.4f}\n")

            if label == BACKGROUND_LABEL:
                self.set_status("ESP32 conectada")
                self.log("RUIDO_FONDO ignorado, no se envía comando.\n")
            elif confidence < SIMPLE_THRESHOLD:
                self.set_status("ESP32 conectada")
                self.log("Predicción bajo el umbral, no se envía comando.\n")
            else:
                self.set_status(f"Ejecutando: {label}")
                sent_at = self.send_command_with_responses(label)
                if sent_at is not None and record_end is not None:
                    self.log(
                        "Tiempo desde fin de grabación hasta comando enviado: "
                        f"{sent_at - record_end:.3f} s\n"
                    )
        except Exception as exc:
            self.log(f"Error en comando simple: {exc}\n")
            self.ui_queue.put(("show_error", "Error", str(exc)))
        finally:
            self.log(
                f"Tiempo total comando simple: {time.perf_counter() - worker_start:.3f} s\n"
            )
            self.set_busy(False)
            if self.controller is not None:
                self.set_status("ESP32 conectada")
            else:
                self.set_status("Desconectado")

    def sequence_worker(self, source):
        self.set_busy(True)
        self.set_status("Escuchando secuencia...")
        self.log(f"\nGrabando secuencia desde {source}.\n")
        worker_start = time.perf_counter()
        record_end = None
        try:
            sequence_module = self.get_predict_sequence_live()
            resources = self.ensure_resources()
            model, labels, mean, std, input_shape = resources
            args = sequence_module.build_default_args(
                duration=SEQUENCE_DURATION,
                segmentation_mode="vad",
                threshold=SEQUENCE_THRESHOLD,
                energy_threshold=SEQUENCE_ENERGY_THRESHOLD,
                min_segment_duration=SEQUENCE_MIN_SEGMENT_DURATION,
                min_silence_duration=SEQUENCE_MIN_SILENCE_DURATION,
                segment_padding=SEQUENCE_SEGMENT_PADDING,
            )
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                record_start = time.perf_counter()
                audio = sequence_module.record_audio(args.duration, device=args.device)
                record_end = time.perf_counter()

                self.set_status("Procesando audio...")
                inference_start = time.perf_counter()
                if args.segmentation_mode == "windows":
                    rows, detected_sequence, confirmed_groups = (
                        sequence_module.run_sequence_inference(
                            model,
                            labels,
                            mean,
                            std,
                            input_shape,
                            audio,
                            args,
                        )
                    )
                    sequence_module.print_window_table(rows)
                    sequence_module.print_confirmed_groups(confirmed_groups)
                else:
                    rows, detected_sequence = sequence_module.run_vad_sequence_inference(
                        model,
                        labels,
                        mean,
                        std,
                        input_shape,
                        audio,
                        args,
                    )
                    sequence_module.print_segment_table(rows)
                sequence_module.print_sequence(detected_sequence)
                inference_end = time.perf_counter()
            self.log(buffer.getvalue())
            self.log(f"Tiempo de grabación: {record_end - record_start:.3f} s\n")
            self.log(
                "Tiempo de inferencia/preprocesamiento: "
                f"{inference_end - inference_start:.3f} s\n"
            )

            sequence = [
                command
                for command in detected_sequence
                if command != BACKGROUND_LABEL
            ]
            if not sequence:
                self.set_status("ESP32 conectada")
                self.log("No hay comandos válidos para enviar a ESP32.\n")
                return

            sequence_text = " → ".join(sequence)
            self.set_status(f"Secuencia detectada: {sequence_text}")
            self.log("Secuencia detectada: " + " -> ".join(sequence) + "\n")
            for index, command in enumerate(sequence):
                self.set_status(f"Ejecutando: {command}")
                sent_at = self.send_command_with_responses(command)
                if index == 0 and sent_at is not None and record_end is not None:
                    self.log(
                        "Tiempo desde fin de grabación hasta primer comando enviado: "
                        f"{sent_at - record_end:.3f} s\n"
                    )
                if index < len(sequence) - 1:
                    time.sleep(sequence_module.DEFAULT_COMMAND_DELAY)
        except Exception as exc:
            self.log(f"Error en secuencia: {exc}\n")
            self.ui_queue.put(("show_error", "Error", str(exc)))
        finally:
            self.log(f"Tiempo total secuencia: {time.perf_counter() - worker_start:.3f} s\n")
            self.set_busy(False)
            if self.controller is not None:
                self.set_status("ESP32 conectada")
            else:
                self.set_status("Desconectado")

    def start_send_command(self, command):
        if self.is_busy():
            self.log(f"Sistema ocupado: no se envía {command}.\n")
            return
        if self.controller is None and not self.connect_esp32():
            return

        thread = threading.Thread(
            target=self.send_command_worker,
            args=(command,),
            daemon=True,
        )
        thread.start()

    def send_command_worker(self, command):
        self.set_busy(True)
        self.set_status(f"Ejecutando: {command}")
        try:
            self.send_command_with_responses(command)
        except Exception as exc:
            self.log(f"Error enviando {command}: {exc}\n")
        finally:
            self.set_busy(False)
            if self.controller is not None:
                self.set_status("ESP32 conectada")
            else:
                self.set_status("Desconectado")

    def send_command_with_responses(self, command):
        if command == BACKGROUND_LABEL:
            self.log("RUIDO_FONDO ignorado, no se envía comando.\n")
            return None

        if command not in VALID_COMMANDS:
            self.log(f"Comando inválido ignorado: {command}\n")
            return None

        if self.controller is None:
            self.log("ESP32 no conectada, no se envía comando.\n")
            return None

        send_start = time.perf_counter()
        sent_at = None
        with self.serial_lock:
            self.controller.send_command(command)
            sent_at = time.perf_counter()
            self.log(f"Comando enviado a ESP32: {command}\n")
            for response in self.controller.read_responses():
                if response in {"OK", "DONE"}:
                    self.log(f"Respuesta ESP32: {response}\n")
                else:
                    self.log(f"Respuesta ESP32: {response}\n")
        self.log(f"Tiempo de envío Bluetooth: {time.perf_counter() - send_start:.3f} s\n")
        return sent_at

    def on_close(self):
        self.closing = True
        self.listening = False
        if self.controller is not None:
            with self.serial_lock:
                self.controller.close()
                self.controller = None
        self.root.destroy()


def main():
    root = tk.Tk()
    VoiceRobotControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
