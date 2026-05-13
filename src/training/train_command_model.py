"""Entrena una CNN base para reconocer comandos de voz individuales."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "commands_dataset.npz"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
MODEL_PATH = MODELS_DIR / "command_cnn.keras"
PREPROCESS_PARAMS_PATH = MODELS_DIR / "preprocess_params.npz"
LABELS_PATH = MODELS_DIR / "labels.json"
TRAINING_METRICS_PATH = DOCS_DIR / "training_metrics.json"
CLASSIFICATION_REPORT_PATH = DOCS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = DOCS_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_NORMALIZED_PATH = DOCS_DIR / "confusion_matrix_normalized.png"
TRAINING_CURVES_PATH = DOCS_DIR / "training_curves.png"
RANDOM_STATE = 42


def parse_args():
    """Define los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrena una CNN 2D desde cero para comandos de voz.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Ruta del dataset .npz. Default: {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Numero maximo de epocas de entrenamiento.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamano de batch para entrenamiento.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=MODEL_PATH,
        help=f"Ruta del modelo entrenado. Default: {MODEL_PATH}",
    )
    parser.add_argument(
        "--architecture",
        choices=["baseline", "improved_light"],
        default="baseline",
        help="Arquitectura del modelo a entrenar. Default: baseline.",
    )
    parser.add_argument(
        "--use-class-weight",
        action="store_true",
        help="Activa pesos de clase balanceados durante el entrenamiento.",
    )
    return parser.parse_args()


def resolve_project_path(path):
    """Convierte rutas relativas a rutas dentro del proyecto."""
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_dataset(dataset_path):
    """Carga X, y, labels y file_paths desde el dataset NPZ."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

    with np.load(dataset_path, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        labels = data["labels"].astype(str)
        file_paths = data["file_paths"].astype(str)

    if X.ndim != 4:
        raise ValueError(f"X debe tener 4 dimensiones, pero tiene forma {X.shape}.")

    if len(X) != len(y) or len(y) != len(file_paths):
        raise ValueError("X, y y file_paths deben tener la misma cantidad de muestras.")

    return X, y, labels, file_paths


def split_dataset(X, y, file_paths):
    """Divide el dataset en train, validation y test con estratificacion."""
    X_train, X_temp, y_train, y_temp, paths_train, paths_temp = train_test_split(
        X,
        y,
        file_paths,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test, paths_val, paths_test = train_test_split(
        X_temp,
        y_temp,
        paths_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        paths_train,
        paths_val,
        paths_test,
    )


def normalize_splits(X_train, X_val, X_test):
    """Normaliza usando media y desviacion estandar calculadas solo sobre train."""
    mean = np.mean(X_train, dtype=np.float64).astype(np.float32)
    std = np.std(X_train, dtype=np.float64).astype(np.float32)

    if std == 0:
        raise ValueError("La desviacion estandar de train es 0; no se puede normalizar.")

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_val_norm, X_test_norm, mean, std


def compile_model(model):
    """Compila un modelo con la configuracion comun de entrenamiento."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_baseline_cnn(input_shape, num_classes):
    """Crea la CNN 2D baseline original entrenable desde cero."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.30),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.40),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return compile_model(model)


def build_improved_light_cnn(input_shape, num_classes):
    """Crea una CNN 2D mejorada moderada entrenable desde cero."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.15),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.20),
            layers.Conv2D(96, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.GlobalAveragePooling2D(),
            layers.Dense(96, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return compile_model(model)


def build_model(input_shape, num_classes, architecture):
    """Construye el modelo segun la arquitectura solicitada."""
    builders = {
        "baseline": build_baseline_cnn,
        "improved_light": build_improved_light_cnn,
    }
    return builders[architecture](input_shape, num_classes)


def build_class_weights(y_train):
    """Calcula pesos de clase balanceados para compensar desbalance del dataset."""
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return {int(class_id): float(weight) for class_id, weight in zip(classes, weights)}


def save_labels(labels):
    """Guarda las etiquetas del modelo en JSON."""
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("w", encoding="utf-8") as labels_file:
        json.dump(labels.tolist(), labels_file, ensure_ascii=False, indent=2)


def plot_training_curves(history):
    """Guarda las curvas de accuracy y loss del entrenamiento."""
    epochs = np.arange(1, len(history.history.get("loss", [])) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history.history.get("accuracy", []), label="accuracy")
    axes[0].plot(epochs, history.history.get("val_accuracy", []), label="val_accuracy")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.history.get("loss", []), label="loss")
    axes[1].plot(epochs, history.history.get("val_loss", []), label="val_loss")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    TRAINING_CURVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(TRAINING_CURVES_PATH, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(matrix, labels, output_path, title, value_format="d"):
    """Guarda la matriz de confusion como imagen PNG."""
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)

    tick_marks = np.arange(len(labels))
    axis.set_xticks(tick_marks)
    axis.set_yticks(tick_marks)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("Prediccion")
    axis.set_ylabel("Etiqueta real")
    axis.set_title(title)

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value > threshold else "black"
            if value_format == "d":
                label = str(int(value))
            else:
                label = format(value, value_format)
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=color,
            )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def normalize_confusion_matrix(matrix):
    """Normaliza la matriz de confusion por fila."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix.astype(np.float64),
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )


def save_training_metrics(history, test_loss, test_accuracy, final_accuracy, split_sizes):
    """Guarda metricas principales de entrenamiento y evaluacion."""
    best_val_accuracy = max(history.history.get("val_accuracy", [0.0]))
    best_val_loss = min(history.history.get("val_loss", [0.0]))
    metrics = {
        "train_samples": split_sizes["train"],
        "validation_samples": split_sizes["validation"],
        "test_samples": split_sizes["test"],
        "epochs_ran": len(history.history.get("loss", [])),
        "best_validation_accuracy": float(best_val_accuracy),
        "best_validation_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "final_accuracy": float(final_accuracy),
    }

    TRAINING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)

    return metrics


def main():
    args = parse_args()
    dataset_path = resolve_project_path(args.dataset)
    model_output_path = resolve_project_path(args.model_output)

    X, y, labels, file_paths = load_dataset(dataset_path)
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _paths_train,
        _paths_val,
        _paths_test,
    ) = split_dataset(X, y, file_paths)

    X_train, X_val, X_test, mean, std = normalize_splits(X_train, X_val, X_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREPROCESS_PARAMS_PATH, mean=mean, std=std)
    save_labels(labels)

    model = build_model(
        input_shape=X_train.shape[1:],
        num_classes=len(labels),
        architecture=args.architecture,
    )
    fit_kwargs = {}
    if args.use_class_weight:
        fit_kwargs["class_weight"] = build_class_weights(y_train)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_output_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"Dataset: {dataset_path}")
    print(f"Arquitectura: {args.architecture}")
    print(f"Class weight: {'activado' if args.use_class_weight else 'desactivado'}")
    print(f"Forma de entrada: {X_train.shape[1:]}")
    print(f"Clases: {list(labels)}")
    print(
        "Division de datos: "
        f"train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}"
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
        **fit_kwargs,
    )

    best_model = keras.models.load_model(model_output_path)
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    y_probabilities = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probabilities, axis=1)
    final_accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=labels,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=np.arange(len(labels)))
    normalized_matrix = normalize_confusion_matrix(matrix)

    with CLASSIFICATION_REPORT_PATH.open("w", encoding="utf-8") as report_file:
        report_file.write(report)
        report_file.write("\n")

    plot_training_curves(history)
    plot_confusion_matrix(
        matrix=matrix,
        labels=labels,
        output_path=CONFUSION_MATRIX_PATH,
        title="Matriz de confusion",
        value_format="d",
    )
    plot_confusion_matrix(
        matrix=normalized_matrix,
        labels=labels,
        output_path=CONFUSION_MATRIX_NORMALIZED_PATH,
        title="Matriz de confusion normalizada",
        value_format=".2f",
    )
    metrics = save_training_metrics(
        history=history,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        final_accuracy=final_accuracy,
        split_sizes={
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
    )

    print("\nMetricas principales")
    print(f"  Accuracy test (Keras): {metrics['test_accuracy']:.4f}")
    print(f"  Accuracy final: {metrics['final_accuracy']:.4f}")
    print(f"  Mejor val_accuracy: {metrics['best_validation_accuracy']:.4f}")
    print(f"  Mejor val_loss: {metrics['best_validation_loss']:.4f}")
    print(f"Modelo guardado en: {model_output_path}")
    print(f"Parametros de normalizacion guardados en: {PREPROCESS_PARAMS_PATH}")
    print(f"Labels guardados en: {LABELS_PATH}")
    print(f"Metricas guardadas en: {TRAINING_METRICS_PATH}")
    print(f"Reporte guardado en: {CLASSIFICATION_REPORT_PATH}")
    print(f"Matriz de confusion guardada en: {CONFUSION_MATRIX_PATH}")
    print(f"Matriz de confusion normalizada guardada en: {CONFUSION_MATRIX_NORMALIZED_PATH}")
    print(f"Curvas de entrenamiento guardadas en: {TRAINING_CURVES_PATH}")


if __name__ == "__main__":
    main()
