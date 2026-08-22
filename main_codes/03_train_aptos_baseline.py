# 03_train_aptos_baseline.py - Stage 2: Train APTOS baseline (50 epochs, frozen backbone)

from pathlib import Path
import os
import random

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

APTOS_TRAIN_CHECKED = REPORTS_DIR / "aptos_train_checked.csv"
APTOS_VALID_CHECKED = REPORTS_DIR / "aptos_validation_checked.csv"
APTOS_TEST_CHECKED = REPORTS_DIR / "aptos_test_checked.csv"

MODEL_PATH = MODELS_DIR / "aptos_baseline_efficientnetb0.keras"
BEST_MODEL_PATH = MODELS_DIR / "aptos_baseline_efficientnetb0_best.keras"

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 5
EPOCHS = 50
AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = True

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]


def load_checked_dataframe(csv_path: Path, split_name: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run 01_check_datasets.py first.")

    df = pd.read_csv(csv_path)
    required = ["id_code", "image_path", "diagnosis", "image_exists"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    if int((df["image_exists"] == False).sum()) > 0:
        raise FileNotFoundError(f"Missing APTOS images in {split_name}. Re-run dataset checking.")

    df["id_code"] = df["id_code"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    df["diagnosis"] = df["diagnosis"].astype(int)
    return df


def decode_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    label = tf.one_hot(label, depth=NUM_CLASSES)
    return image, label


def make_dataset(df: pd.DataFrame, training: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (df["image_path"].values, df["diagnosis"].values.astype(np.int32))
    )

    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(decode_image, num_parallel_calls=AUTOTUNE, deterministic=True)

    if training:
        augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal", seed=SEED),
                layers.RandomRotation(0.05, seed=SEED),
                layers.RandomZoom(0.10, seed=SEED),
                layers.RandomContrast(0.10, seed=SEED),
            ],
            name="aptos_baseline_augmentation",
        )
        ds = ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE).with_options(DATASET_OPTIONS)
    return ds

# Compute balanced class weights to handle APTOS dataset imbalance.

def compute_weights(train_df: pd.DataFrame) -> dict:
    y_train = train_df["diagnosis"].values.astype(int)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2, 3, 4]),
        y=y_train,
    )
    return {i: float(weights[i]) for i in range(NUM_CLASSES)}


def build_model() -> tf.keras.Model:
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        name="efficientnetb0",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="aptos_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.35, seed=SEED)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25, seed=SEED)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="aptos_output")(x)

    model = models.Model(inputs, outputs, name="aptos_baseline_efficientnetb0")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def plot_history(history, metric: str, output_path: Path, title: str):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history[metric], label=f"Training {metric}")

    val_metric = f"val_{metric}"
    if val_metric in history.history:
        plt.plot(history.history[val_metric], label=f"Validation {metric}")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_confusion_matrix(y_true, y_pred, output_path: Path, title: str, labels: list[str]):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# Collapse 5-class DR to binary referable (0-1=Non-referable, 2-4=Referable).

def to_referable_dr(y):
    y = np.asarray(y)
    return np.where(np.isin(y, [2, 3, 4]), 1, 0)


def save_training_settings(actual_epochs_ran: int, best_epoch: int):
    settings = {
        "stage": "Stage 2",
        "model": "APTOS Baseline EfficientNetB0",
        "dataset": "APTOS 2019 Blindness Detection Dataset",
        "task": "Five-class diabetic-retinopathy grading",
        "maximum_epochs_set": EPOCHS,
        "actual_epochs_ran": actual_epochs_ran,
        "best_epoch_selected": best_epoch,
        "early_stopping_used": "No",
        "fixed_random_seed_used": "Yes",
        "random_seed_value": SEED,
        "tensorflow_determinism_enabled": "Yes",
        "model_checkpoint_used": "Yes",
        "reduce_lr_on_plateau_used": "Yes",
        "class_weighting_used": "Yes",
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
        "optimizer": "Adam",
        "initial_learning_rate": 1e-4,
    }
    pd.DataFrame([settings]).to_csv(REPORTS_DIR / "stage2_aptos_baseline_training_settings.csv", index=False)


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset, test_df: pd.DataFrame):
    y_true = test_df["diagnosis"].values.astype(int)
    y_prob = model.predict(test_ds)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(REPORTS_DIR / "aptos_baseline_classification_report.csv")

    save_confusion_matrix(
        y_true,
        y_pred,
        FIGURES_DIR / "aptos_baseline_confusion_matrix.png",
        "APTOS Baseline Five-Class Confusion Matrix",
        CLASS_NAMES,
    )

    y_true_binary = to_referable_dr(y_true)
    y_pred_binary = to_referable_dr(y_pred)

    save_confusion_matrix(
        y_true_binary,
        y_pred_binary,
        FIGURES_DIR / "aptos_baseline_referable_dr_confusion_matrix.png",
        "APTOS Baseline Binary Referable DR Confusion Matrix",
        ["Non-referable", "Referable"],
    )

    summary = {
        "stage": "Stage 2",
        "model": "APTOS Baseline EfficientNetB0",
        "dataset": "APTOS 2019 Blindness Detection Dataset",
        "task": "5-class DR grading",
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    binary_summary = {
        "stage": "Stage 2",
        "model": "APTOS Baseline EfficientNetB0",
        "dataset": "APTOS 2019 Blindness Detection Dataset",
        "task": "binary referable DR",
        "accuracy": accuracy_score(y_true_binary, y_pred_binary),
        "macro_precision": precision_score(y_true_binary, y_pred_binary, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true_binary, y_pred_binary, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true_binary, y_pred_binary, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true_binary, y_pred_binary, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true_binary, y_pred_binary, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true_binary, y_pred_binary, average="weighted", zero_division=0),
    }

    pd.DataFrame([summary, binary_summary]).to_csv(REPORTS_DIR / "aptos_baseline_summary_metrics.csv", index=False)

    pred_df = test_df.copy()
    pred_df["predicted_label"] = y_pred

    for i, name in enumerate(CLASS_NAMES):
        safe_name = name.replace(" ", "_").replace("-", "_")
        pred_df[f"probability_{i}_{safe_name}"] = y_prob[:, i]

    pred_df.to_csv(REPORTS_DIR / "aptos_baseline_test_predictions.csv", index=False)

    print("\nAPTOS baseline test performance")
    print("-" * 80)
    print(pd.DataFrame([summary, binary_summary]).to_string(index=False))


def main():

    # Load APTOS data, train 50 epochs with frozen backbone, save baseline model.

    train_df = load_checked_dataframe(APTOS_TRAIN_CHECKED, "train")
    valid_df = load_checked_dataframe(APTOS_VALID_CHECKED, "validation")
    test_df = load_checked_dataframe(APTOS_TEST_CHECKED, "test")

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(valid_df)}")
    print(f"Test rows: {len(test_df)}")

    class_weights = compute_weights(train_df)
    print(f"\nAPTOS class weights: {class_weights}")

    train_ds = make_dataset(train_df, training=True)
    valid_ds = make_dataset(valid_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print(f"\nTraining APTOS baseline model for exactly {EPOCHS} epochs...")

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    actual_epochs_ran = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

    print(f"\nActual epochs completed: {actual_epochs_ran}")
    print(f"Best validation loss selected from epoch: {best_epoch}")

    if BEST_MODEL_PATH.exists():
        model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)

    model.save(MODEL_PATH)
    save_training_settings(actual_epochs_ran, best_epoch)

    plot_history(history, "accuracy", FIGURES_DIR / "aptos_baseline_training_accuracy.png", "APTOS Baseline Training and Validation Accuracy")
    plot_history(history, "loss", FIGURES_DIR / "aptos_baseline_training_loss.png", "APTOS Baseline Training and Validation Loss")
    plot_history(history, "precision", FIGURES_DIR / "aptos_baseline_training_precision.png", "APTOS Baseline Training and Validation Precision")
    plot_history(history, "recall", FIGURES_DIR / "aptos_baseline_training_recall.png", "APTOS Baseline Training and Validation Recall")

    evaluate_model(model, test_ds, test_df)

    print("\nStage 2 completed successfully.")


if __name__ == "__main__":
    main()