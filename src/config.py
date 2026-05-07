"""Configuracion compartida del proyecto de reconocimiento de voz."""

# Usamos 16 kHz porque es suficiente para comandos de voz cortos y reduce el
# tamano de los archivos sin perder informacion importante para MFCC o Mel.
SAMPLE_RATE = 16000

# Las etiquetas internas no tienen espacios ni tildes para que sean estables al
# usarlas como nombres de carpetas, clases del modelo, claves de diccionarios y
# valores en archivos CSV/JSON. La frase hablada si puede tener espacios o
# tildes; por ejemplo, POSICION_INICIAL se pronuncia como "posicion inicial".
SIMPLE_COMMANDS = [
    "SUBE",
    "BAJA",
    "ABRE",
    "CIERRA",
    "POSICION_INICIAL",
    "RUIDO_FONDO",
]

# Mapea cada etiqueta interna a la frase que se espera escuchar al grabar datos.
# Esto permite mantener etiquetas portables sin perder la forma natural hablada.
SIMPLE_COMMAND_PHRASES = {
    "SUBE": "sube",
    "BAJA": "baja",
    "ABRE": "abre",
    "CIERRA": "cierra",
    "POSICION_INICIAL": "posicion inicial",
    "RUIDO_FONDO": "ruido fondo",
}

# Comandos compuestos para el modulo secuencial futuro. La clave identifica la
# etiqueta compuesta y el valor conserva el orden exacto de acciones simples.
SEQUENCE_COMMANDS = {
    "ABRE_Y_CIERRA": ["ABRE", "CIERRA"],
    "BAJA_Y_ABRE": ["BAJA", "ABRE"],
    "BAJA_Y_CIERRA": ["BAJA", "CIERRA"],
    "SUBE_Y_ABRE": ["SUBE", "ABRE"],
    "SUBE_Y_CIERRA": ["SUBE", "CIERRA"],
    "ABRE_Y_POSICION_INICIAL": ["ABRE", "POSICION_INICIAL"],
    "BAJA_Y_ABRE_Y_CIERRA": ["BAJA", "ABRE", "CIERRA"],
    "SUBE_Y_ABRE_Y_CIERRA": ["SUBE", "ABRE", "CIERRA"],
}

# Frases naturales esperadas para grabar el dataset secuencial. Las claves se
# mantienen alineadas con SEQUENCE_COMMANDS para validar etiquetas compuestas.
SEQUENCE_COMMAND_PHRASES = {
    "ABRE_Y_CIERRA": "abre y cierra",
    "BAJA_Y_ABRE": "baja y abre",
    "BAJA_Y_CIERRA": "baja y cierra",
    "SUBE_Y_ABRE": "sube y abre",
    "SUBE_Y_CIERRA": "sube y cierra",
    "ABRE_Y_POSICION_INICIAL": "abre y posicion inicial",
    "BAJA_Y_ABRE_Y_CIERRA": "baja y abre y cierra",
    "SUBE_Y_ABRE_Y_CIERRA": "sube y abre y cierra",
}
