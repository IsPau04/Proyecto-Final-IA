# Control de robot por voz con ESP32 Bluetooth clasico

El modelo de reconocimiento de voz predice estas clases:

- `SUBE`
- `BAJA`
- `ABRE`
- `CIERRA`
- `POSICION_INICIAL`
- `RUIDO_FONDO`

`RUIDO_FONDO` nunca se envia a la ESP32.

## Uso desde terminal

Comando simple sin ESP32:

```powershell
python -m src.inference.predict_live_once --duration 2 --threshold 0.60
```

Comando simple enviando a ESP32 por Bluetooth clasico:

```powershell
python -m src.inference.predict_live_once --duration 2 --threshold 0.60 --send-esp32 --esp32-port COM5
```

Secuencia sin ESP32:

```powershell
python -m src.inference.predict_sequence_live --duration 10 --segmentation-mode vad --threshold 0.78 --energy-threshold 0.003 --min-segment-duration 0.25 --min-silence-duration 0.45 --segment-padding 0.70
```

Secuencia enviando a ESP32:

```powershell
python -m src.inference.predict_sequence_live --duration 10 --segmentation-mode vad --threshold 0.78 --energy-threshold 0.003 --min-segment-duration 0.25 --min-silence-duration 0.45 --segment-padding 0.70 --send-esp32 --esp32-port COM5
```

## Interfaz grafica

Ejecuta:

```powershell
python -m src.ui.voice_robot_control
```

Botones disponibles:

- `Conectar ESP32`: abre el puerto Bluetooth clasico, por defecto `COM5`.
- `Escuchar botones fisicos`: mantiene el puerto abierto para recibir eventos enviados por la ESP32.
- `Grabar comando simple`: graba 2 segundos, predice una clase y envia el comando valido usando el mismo controlador Bluetooth abierto.
- `Grabar secuencia`: graba 10 segundos, segmenta por VAD y envia cada comando valido en orden.

La consola visual de la interfaz muestra conexion, eventos recibidos, clase predicha, comandos enviados y respuestas de la ESP32 como `OK` o `DONE`.

## Botones fisicos de ESP32

La ESP32 debe enviar estos eventos por Bluetooth clasico:

- `BTN:SIMPLE`: Python graba un comando simple y envia la clase predicha si supera el umbral.
- `BTN:SEQUENCE`: Python graba una frase larga, segmenta por VAD y envia cada comando valido.

La interfaz tambien reconoce:

- `BTN:HOME`: envia `POSICION_INICIAL`.
- `BTN:STOP`: se muestra en consola visual, sin enviar comando de voz.

Python envia a la ESP32 estos comandos:

- `ABRE`
- `CIERRA`
- `SUBE`
- `BAJA`
- `POSICION_INICIAL`

No se usa Wi-Fi ni BLE.
