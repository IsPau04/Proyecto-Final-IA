"""Configuracion compartida del proyecto de reconocimiento de voz."""

# Usamos 16 kHz porque es suficiente para comandos de voz cortos y reduce el
# tamano de los archivos sin perder informacion importante para MFCC o Mel.
SAMPLE_RATE = 16000

# El modelo se entrena solo con comandos base individuales. Las frases largas o
# combinaciones de acciones se resolveran despues con un parser de predicciones,
# no con clases ni carpetas compuestas dentro del dataset de entrenamiento.
#
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
