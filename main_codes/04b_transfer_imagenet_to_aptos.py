# 04b_transfer_imagenet_to_aptos.py - Stage 4: Train with ImageNet weights (no glaucoma pretraining) to compare against Script 04

import os
import random
from pathlib import Path

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

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

APTOS_TRAIN_CSV = REPORTS_DIR / "aptos_train_checked.csv"
APTOS_VALID_CSV = REPORTS_DIR / "aptos_validation_checked.csv"
APTOS_TEST_CSV = REPORTS_DIR / "aptos_test_checked.csv"

FINAL_MODEL_PATH = MODELS_DIR / "imagenet_transfer_efficientnetb0.keras"
BEST_MODEL_PATH = MODELS_DIR / "imagenet_transfer_efficientnetb0_best.keras"

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 5

FROZEN_EPOCHS = 25
FINE_TUNE_EPOCHS = 25
TOTAL_EPOCHS = FROZEN_EPOCHS + FINE_TUNE_EPOCHS

FINE_TUNE_LAST_N_LAYERS = 25

PHASE1_LR = 1e-4
PHASE2_LR = 1e-5

AUTOTUNE = tf.data.AUTOTUNE

CLASS_NAMES = [
    "No_DR",
    "Mild_DR",
    "Moderate_DR",
    "Severe_DR",
    "Proliferative_DR",
]

BINARY_CLASS_NAMES = [
    "Non_referable_DR",
    "Referable_DR",
]

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = True



def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_checked_csv(path: Path) -> pd.DataFrame:
    require_file(path)

    df = pd.read_csv(path)

    required_columns = ["image_path", "diagnosis", "image_exists"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")

    df = df[df["image_exists"] == True].copy()
    df["image_path"] = df["image_path"].astype(str)
    df["diagnosis"] = df["diagnosis"].astype(int)

    if "id_code" not in df.columns:
        df["id_code"] = df["image_path"].apply(lambda value: Path(value).stem)

    if df.empty:
        raise ValueError(f"No usable APTOS image rows found in: {path}")

    return df


def decode_multiclass_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)

    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=NUM_CLASSES)

    return image, label


def make_dataset(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
    image_paths = df["image_path"].values
    labels = df["diagnosis"].values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        decode_multiclass_image,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(DATASET_OPTIONS)

    return ds


def get_class_weights(train_df: pd.DataFrame) -> dict:
    y_train = train_df["diagnosis"].values.astype(int)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_train,
    )

    return {class_index: float(weights[class_index]) for class_index in range(NUM_CLASSES)}


def binary_labels_from_five_class(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    return np.where(labels >= 2, 1, 0)


# Build model: ImageNet backbone + trainable head (same architecture as Script 04, but NO glaucoma pretraining)

def build_imagenet_transfer_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="imagenet_transfer_input")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.05, seed=SEED),
            tf.keras.layers.RandomZoom(0.05, seed=SEED),
            tf.keras.layers.RandomContrast(0.05, seed=SEED),
        ],
        name="imagenet_transfer_augmentation",
    )

    x = augmentation(inputs)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        name="efficientnetb0",
    )

    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.35, seed=SEED, name="dropout_imagenet_transfer_1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_imagenet_transfer_1")(x)
    x = tf.keras.layers.Dropout(0.25, seed=SEED, name="dropout_imagenet_transfer_2")(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="imagenet_transfer_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="imagenet_transfer_efficientnetb0",
    )

    return model, backbone


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def freeze_backbone(backbone: tf.keras.Model) -> None:
    backbone.trainable = False

# Unfreeze last 25 layers of ImageNet backbone for APTOS fine-tuning

def fine_tune_backbone(backbone: tf.keras.Model, last_n_layers: int) -> int:
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    for layer in backbone.layers[-last_n_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    trainable_layer_count = sum(1 for layer in backbone.layers if layer.trainable)
    return trainable_layer_count


def make_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def merge_histories(history_phase1, history_phase2) -> pd.DataFrame:
    rows = []

    for epoch_index in range(len(history_phase1.history["loss"])):
        row = {
            "epoch": epoch_index + 1,
            "phase": "aptos_head_training_on_frozen_imagenet_backbone",
        }
        for key, values in history_phase1.history.items():
            row[key] = values[epoch_index]
        rows.append(row)

    for epoch_index in range(len(history_phase2.history["loss"])):
        row = {
            "epoch": FROZEN_EPOCHS + epoch_index + 1,
            "phase": "aptos_fine_tuning_on_imagenet_backbone",
        }
        for key, values in history_phase2.history.items():
            row[key] = values[epoch_index]
        rows.append(row)

    return pd.DataFrame(rows)



def predict_dataset(model: tf.keras.Model, dataset: tf.data.Dataset):
    probabilities = model.predict(dataset, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    true_labels = []

    for _, batch_labels in dataset:
        true_labels.extend(np.argmax(batch_labels.numpy(), axis=1).tolist())

    true_labels = np.array(true_labels).astype(int)

    return true_labels, predictions, probabilities


def calculate_multiclass_metrics(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def calculate_binary_metrics(y_true, y_pred) -> dict:
    y_true_binary = binary_labels_from_five_class(y_true)
    y_pred_binary = binary_labels_from_five_class(y_pred)

    accuracy = accuracy_score(y_true_binary, y_pred_binary)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1],
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def save_prediction_file(test_df, y_true, y_pred, probabilities):
    output_df = test_df.copy()

    output_df["model_key"] = "imagenet_transfer"
    output_df["model_name"] = "APTOS transfer with ImageNet EfficientNetB0 backbone"

    output_df["true_label"] = y_true
    output_df["pred_label"] = y_pred
    output_df["true_class_name"] = [CLASS_NAMES[label] for label in y_true]
    output_df["pred_class_name"] = [CLASS_NAMES[label] for label in y_pred]
    output_df["confidence"] = np.max(probabilities, axis=1)
    output_df["correct"] = output_df["true_label"] == output_df["pred_label"]

    for class_index in range(NUM_CLASSES):
        output_df[f"prob_{class_index}"] = probabilities[:, class_index]

    output_df["true_binary"] = binary_labels_from_five_class(y_true)
    output_df["pred_binary"] = binary_labels_from_five_class(y_pred)
    output_df["true_binary_name"] = [BINARY_CLASS_NAMES[label] for label in output_df["true_binary"]]
    output_df["pred_binary_name"] = [BINARY_CLASS_NAMES[label] for label in output_df["pred_binary"]]
    output_df["binary_correct"] = output_df["true_binary"] == output_df["pred_binary"]

    output_df.to_csv(REPORTS_DIR / "imagenet_transfer_test_predictions.csv", index=False)

    misclassified_df = output_df[output_df["correct"] == False].copy()
    misclassified_df = misclassified_df.sort_values(
        by=["true_label", "confidence"],
        ascending=[True, False],
    )
    misclassified_df.to_csv(REPORTS_DIR / "imagenet_transfer_misclassified_images.csv", index=False)


def save_metrics_and_reports(y_true, y_pred):
    five_class_metrics = calculate_multiclass_metrics(y_true, y_pred)
    binary_metrics = calculate_binary_metrics(y_true, y_pred)

    summary_df = pd.DataFrame(
        [
            {
                "stage": "Stage 4b",
                "model": "APTOS transfer with ImageNet EfficientNetB0 backbone",
                "dataset": "APTOS 2019 Blindness Detection Dataset",
                "task": "5-class diabetic retinopathy grading",
                **five_class_metrics,
            },
            {
                "stage": "Stage 4b",
                "model": "APTOS transfer with ImageNet EfficientNetB0 backbone",
                "dataset": "APTOS 2019 Blindness Detection Dataset",
                "task": "Binary referable diabetic retinopathy",
                **binary_metrics,
            },
        ]
    )

    summary_df.to_csv(REPORTS_DIR / "imagenet_transfer_summary_metrics.csv", index=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "class_name"})
    report_df.to_csv(REPORTS_DIR / "imagenet_transfer_per_class_metrics.csv", index=False)

    y_true_binary = binary_labels_from_five_class(y_true)
    y_pred_binary = binary_labels_from_five_class(y_pred)

    binary_report = classification_report(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1],
        target_names=BINARY_CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    binary_report_df = pd.DataFrame(binary_report).transpose().reset_index()
    binary_report_df = binary_report_df.rename(columns={"index": "class_name"})
    binary_report_df.to_csv(REPORTS_DIR / "imagenet_transfer_binary_per_class_metrics.csv", index=False)

    print("\nImageNet-transfer test performance")
    print("-" * 80)
    print(summary_df.to_string(index=False))


def save_confusion_matrices(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    )
    cm_df.to_csv(REPORTS_DIR / "imagenet_transfer_confusion_matrix.csv")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    )
    cm_norm_df.to_csv(REPORTS_DIR / "imagenet_transfer_normalized_confusion_matrix.csv")

    plot_confusion_matrix(
        cm,
        CLASS_NAMES,
        FIGURES_DIR / "imagenet_transfer_confusion_matrix.png",
        "ImageNet transfer five-class confusion matrix",
        normalized=False,
    )

    plot_confusion_matrix(
        cm_norm,
        CLASS_NAMES,
        FIGURES_DIR / "imagenet_transfer_normalized_confusion_matrix.png",
        "ImageNet transfer normalized five-class confusion matrix",
        normalized=True,
    )


def save_binary_confusion_matrices(y_true, y_pred):
    y_true_binary = binary_labels_from_five_class(y_true)
    y_pred_binary = binary_labels_from_five_class(y_pred)

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in BINARY_CLASS_NAMES],
        columns=[f"pred_{name}" for name in BINARY_CLASS_NAMES],
    )
    cm_df.to_csv(REPORTS_DIR / "imagenet_transfer_binary_confusion_matrix.csv")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=[f"true_{name}" for name in BINARY_CLASS_NAMES],
        columns=[f"pred_{name}" for name in BINARY_CLASS_NAMES],
    )
    cm_norm_df.to_csv(REPORTS_DIR / "imagenet_transfer_binary_normalized_confusion_matrix.csv")

    plot_confusion_matrix(
        cm,
        BINARY_CLASS_NAMES,
        FIGURES_DIR / "imagenet_transfer_referable_dr_confusion_matrix.png",
        "ImageNet transfer binary referable DR confusion matrix",
        normalized=False,
    )

    plot_confusion_matrix(
        cm_norm,
        BINARY_CLASS_NAMES,
        FIGURES_DIR / "imagenet_transfer_referable_dr_normalized_confusion_matrix.png",
        "ImageNet transfer normalized binary referable DR confusion matrix",
        normalized=True,
    )


def plot_confusion_matrix(cm, class_names, output_path: Path, title: str, normalized: bool):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = f"{cm[i, j]:.2f}" if normalized else str(int(cm[i, j]))
            plt.text(j, i, value, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def class_error_analysis(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    total_errors = int(np.sum(cm) - np.trace(cm))

    rows = []

    for true_class in range(NUM_CLASSES):
        support = int(np.sum(cm[true_class, :]))
        correct = int(cm[true_class, true_class])
        errors = support - correct
        error_rate = errors / support if support > 0 else 0

        wrong_predictions = []

        for pred_class in range(NUM_CLASSES):
            if pred_class == true_class:
                continue

            count = int(cm[true_class, pred_class])

            if count > 0:
                wrong_predictions.append(
                    {
                        "pred_class": pred_class,
                        "pred_class_name": CLASS_NAMES[pred_class],
                        "count": count,
                    }
                )

        wrong_predictions = sorted(
            wrong_predictions,
            key=lambda item: item["count"],
            reverse=True,
        )

        if wrong_predictions:
            most_common_wrong_prediction = wrong_predictions[0]["pred_class_name"]
            most_common_wrong_prediction_count = wrong_predictions[0]["count"]
            common_error_patterns = "; ".join(
                [
                    f"{CLASS_NAMES[true_class]} predicted as {item['pred_class_name']}: {item['count']}"
                    for item in wrong_predictions
                ]
            )
        else:
            most_common_wrong_prediction = "None"
            most_common_wrong_prediction_count = 0
            common_error_patterns = "No errors"

        rows.append(
            {
                "model_key": "imagenet_transfer",
                "true_class": true_class,
                "true_class_name": CLASS_NAMES[true_class],
                "support": support,
                "correct_predictions": correct,
                "error_count": errors,
                "error_rate": error_rate,
                "error_share_of_all_errors": errors / total_errors if total_errors > 0 else 0,
                "most_common_wrong_prediction": most_common_wrong_prediction,
                "most_common_wrong_prediction_count": most_common_wrong_prediction_count,
                "common_error_patterns": common_error_patterns,
            }
        )

    error_df = pd.DataFrame(rows)
    error_df = error_df.sort_values(
        by=["error_count", "error_rate"],
        ascending=[False, False],
    )
    error_df.to_csv(REPORTS_DIR / "imagenet_transfer_error_analysis_by_class.csv", index=False)


def confusion_pair_analysis(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    rows = []

    for true_class in range(NUM_CLASSES):
        for pred_class in range(NUM_CLASSES):
            if true_class == pred_class:
                continue

            count = int(cm[true_class, pred_class])

            if count > 0:
                rows.append(
                    {
                        "model_key": "imagenet_transfer",
                        "true_class": true_class,
                        "true_class_name": CLASS_NAMES[true_class],
                        "pred_class": pred_class,
                        "pred_class_name": CLASS_NAMES[pred_class],
                        "error_count": count,
                        "error_pattern": f"{CLASS_NAMES[true_class]} predicted as {CLASS_NAMES[pred_class]}",
                    }
                )

    pair_df = pd.DataFrame(rows)

    if not pair_df.empty:
        pair_df = pair_df.sort_values(by="error_count", ascending=False)

    pair_df.to_csv(REPORTS_DIR / "imagenet_transfer_confusion_pairs.csv", index=False)




def plot_training_curve(history_df: pd.DataFrame, metric: str, filename: str, ylabel: str):
    plt.figure(figsize=(8, 5))

    if metric in history_df.columns:
        plt.plot(history_df["epoch"], history_df[metric], label=f"Training {metric}")

    val_metric = f"val_{metric}"

    if val_metric in history_df.columns:
        plt.plot(history_df["epoch"], history_df[val_metric], label=f"Validation {metric}")

    plt.axvline(FROZEN_EPOCHS, linestyle="--", linewidth=1, label="Fine-tuning starts")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"ImageNet transfer {ylabel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def save_training_settings(history_df: pd.DataFrame, trainable_layer_count: int):
    best_row = history_df.loc[history_df["val_loss"].idxmin()]

    settings_df = pd.DataFrame(
        [
            {
                "stage": "Stage 4b",
                "script": "04b_transfer_imagenet_to_aptos.py",
                "model": "APTOS transfer with ImageNet EfficientNetB0 backbone",
                "total_epochs_requested": TOTAL_EPOCHS,
                "frozen_epochs": FROZEN_EPOCHS,
                "fine_tune_epochs": FINE_TUNE_EPOCHS,
                "actual_epochs_completed": len(history_df),
                "initial_backbone_weights": "ImageNet",
                "phase1_backbone_frozen": True,
                "phase2_backbone_fine_tuned_on_aptos": True,
                "fine_tuned_last_n_layers": FINE_TUNE_LAST_N_LAYERS,
                "trainable_backbone_layers_in_phase2": trainable_layer_count,
                "batch_normalization_layers_frozen": True,
                "early_stopping_used": False,
                "best_validation_loss_epoch": int(best_row["epoch"]),
                "best_validation_loss": float(best_row["val_loss"]),
                "final_model_path": str(FINAL_MODEL_PATH),
                "best_model_path": str(BEST_MODEL_PATH),
            }
        ]
    )

    settings_df.to_csv(REPORTS_DIR / "stage4b_imagenet_transfer_training_settings.csv", index=False)

# Load ImageNet backbone, train 2 phases (frozen head then fine-tune), save control model for comparison

def main():
    print("\nStarting Script 4b: ImageNet-transfer APTOS model")

    train_df = load_checked_csv(APTOS_TRAIN_CSV)
    valid_df = load_checked_csv(APTOS_VALID_CSV)
    test_df = load_checked_csv(APTOS_TEST_CSV)

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(valid_df)}")
    print(f"Test rows: {len(test_df)}")

    class_weights = get_class_weights(train_df)
    print("\nAPTOS class weights:")
    print(class_weights)

    train_ds = make_dataset(train_df, training=True)
    valid_ds = make_dataset(valid_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    model, backbone = build_imagenet_transfer_model()
    model.summary()

    print("\nPhase 1: training APTOS head on frozen ImageNet EfficientNetB0 backbone...")
    freeze_backbone(backbone)
    compile_model(model, learning_rate=PHASE1_LR)

    history_phase1 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=FROZEN_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(),
        verbose=1,
    )

    print("\nPhase 2: fine-tuning later ImageNet EfficientNetB0 layers on APTOS...")
    trainable_layer_count = fine_tune_backbone(
        backbone=backbone,
        last_n_layers=FINE_TUNE_LAST_N_LAYERS,
    )

    print(f"Trainable EfficientNetB0 backbone layers in Phase 2: {trainable_layer_count}")

    compile_model(model, learning_rate=PHASE2_LR)

    history_phase2 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=FROZEN_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(),
        verbose=1,
    )

    history_df = merge_histories(history_phase1, history_phase2)
    history_df.to_csv(REPORTS_DIR / "stage4b_imagenet_transfer_training_history.csv", index=False)

    print(f"\nActual epochs completed: {len(history_df)}")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError("Best ImageNet-transfer model checkpoint was not saved.")

    selected_model = tf.keras.models.load_model(BEST_MODEL_PATH)
    selected_model.save(FINAL_MODEL_PATH)

    y_true, y_pred, probabilities = predict_dataset(selected_model, test_ds)

    save_prediction_file(test_df, y_true, y_pred, probabilities)
    save_metrics_and_reports(y_true, y_pred)
    save_confusion_matrices(y_true, y_pred)
    save_binary_confusion_matrices(y_true, y_pred)
    class_error_analysis(y_true, y_pred)
    confusion_pair_analysis(y_true, y_pred)
    save_training_settings(history_df, trainable_layer_count)

    plot_training_curve(
        history_df,
        "accuracy",
        "imagenet_transfer_training_accuracy.png",
        "Accuracy",
    )

    plot_training_curve(
        history_df,
        "loss",
        "imagenet_transfer_training_loss.png",
        "Loss",
    )

    plot_training_curve(
        history_df,
        "precision",
        "imagenet_transfer_training_precision.png",
        "Precision",
    )

    plot_training_curve(
        history_df,
        "recall",
        "imagenet_transfer_training_recall.png",
        "Recall",
    )

    print("\nScript 4b completed successfully.")
    print("ImageNet-transfer model saved.")
    print(f"Final model path: {FINAL_MODEL_PATH}")
    print("ImageNet-transfer summary and per-class metrics saved in outputs/reports.")


if __name__ == "__main__":
    main()