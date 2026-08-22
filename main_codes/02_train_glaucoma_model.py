# 02_train_glaucoma_model.py - Stage 1: Train glaucoma model with 2-phase training (25 frozen + 25 fine-tune)
# This produces a genuinely glaucoma-adapted retinal backbone for Script 4.

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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
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

TRAIN_CSV = REPORTS_DIR / "glaucoma_train_checked.csv"
VALID_CSV = REPORTS_DIR / "glaucoma_validation_checked.csv"
TEST_CSV = REPORTS_DIR / "glaucoma_test_checked.csv"

FINAL_MODEL_PATH = MODELS_DIR / "glaucoma_efficientnetb0.keras"
FINAL_MODEL_H5_PATH = MODELS_DIR / "glaucoma_efficientnetb0.h5"

PHASE1_BEST_MODEL_PATH = MODELS_DIR / "glaucoma_efficientnetb0_head_best.keras"
PHASE2_BEST_MODEL_PATH = MODELS_DIR / "glaucoma_efficientnetb0_backbone_best.keras"

IMG_SIZE = 224
BATCH_SIZE = 16

PHASE1_EPOCHS = 25
PHASE2_EPOCHS = 25
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

FINE_TUNE_LAST_N_LAYERS = 25

PHASE1_LR = 1e-4
PHASE2_LR = 1e-5

AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = True


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_checked_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)

    required_columns = ["image_path", "label_binary", "image_exists"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {missing_columns}")

    df = df[df["image_exists"] == True].copy()
    df["image_path"] = df["image_path"].astype(str)
    df["label_binary"] = df["label_binary"].astype(int)

    if df.empty:
        raise ValueError(f"No usable image rows found in: {path}")

    return df


def decode_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def make_dataset(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
    image_paths = df["image_path"].values
    labels = df["label_binary"].values.astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        decode_image,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(DATASET_OPTIONS)

    return ds

# Build glaucoma model: EfficientNetB0 backbone + new head with sigmoid output for binary classification (NRG vs RG).

def build_glaucoma_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="glaucoma_input")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.05, seed=SEED),
            tf.keras.layers.RandomZoom(0.05, seed=SEED),
            tf.keras.layers.RandomContrast(0.05, seed=SEED),
        ],
        name="glaucoma_augmentation",
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
    x = tf.keras.layers.Dropout(0.30, seed=SEED, name="dropout_head_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_head_1")(x)
    x = tf.keras.layers.Dropout(0.20, seed=SEED, name="dropout_head_2")(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="glaucoma_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="glaucoma_efficientnetb0",
    )

    return model, backbone


def compile_binary_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

# Freeze all backbone layers (used in Phase 1)

def freeze_backbone(backbone: tf.keras.Model) -> None:
    backbone.trainable = False

# Unfreeze last 25 backbone layers for fine-tuning, keep BatchNorm frozen

def fine_tune_backbone(backbone: tf.keras.Model, last_n_layers: int) -> int:
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    for layer in backbone.layers[-last_n_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    trainable_layer_count = sum(1 for layer in backbone.layers if layer.trainable)
    return trainable_layer_count


def make_callbacks(checkpoint_path: Path, monitor: str = "val_auc", mode: str = "max"):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode=mode,
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

# Combine Phase 1 and Phase 2 training histories into one DataFrame.

def merge_histories(history_phase1, history_phase2) -> pd.DataFrame:
    rows = []

    for epoch_index in range(len(history_phase1.history["loss"])):
        row = {"epoch": epoch_index + 1, "phase": "head_training"}
        for key, values in history_phase1.history.items():
            row[key] = values[epoch_index]
        rows.append(row)

    for epoch_index in range(len(history_phase2.history["loss"])):
        row = {
            "epoch": PHASE1_EPOCHS + epoch_index + 1,
            "phase": "glaucoma_backbone_fine_tuning",
        }
        for key, values in history_phase2.history.items():
            row[key] = values[epoch_index]
        rows.append(row)

    return pd.DataFrame(rows)


def predict_dataset(model: tf.keras.Model, dataset: tf.data.Dataset):
    probabilities = model.predict(dataset, verbose=1).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)

    true_labels = []
    for _, batch_labels in dataset:
        true_labels.extend(batch_labels.numpy().astype(int).tolist())

    true_labels = np.array(true_labels).astype(int)

    return true_labels, predictions, probabilities


def plot_training_curve(history_df: pd.DataFrame, metric: str, filename: str, ylabel: str):
    plt.figure(figsize=(8, 5))

    if metric in history_df.columns:
        plt.plot(history_df["epoch"], history_df[metric], label=f"Training {metric}")

    val_metric = f"val_{metric}"
    if val_metric in history_df.columns:
        plt.plot(history_df["epoch"], history_df[val_metric], label=f"Validation {metric}")

    plt.axvline(PHASE1_EPOCHS, linestyle="--", linewidth=1, label="Backbone fine-tuning starts")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Glaucoma model {ylabel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, filename: str):
    class_names = ["NRG", "RG"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Glaucoma confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names)
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def save_predictions(df_test: pd.DataFrame, y_true, y_pred, probabilities):
    output_df = df_test.copy()
    output_df["true_label"] = y_true
    output_df["pred_label"] = y_pred
    output_df["probability_RG"] = probabilities
    output_df["confidence"] = np.where(y_pred == 1, probabilities, 1 - probabilities)
    output_df["correct"] = output_df["true_label"] == output_df["pred_label"]

    output_df.to_csv(REPORTS_DIR / "glaucoma_test_predictions.csv", index=False)


def save_metrics(y_true, y_pred, probabilities):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, probabilities)

    summary = pd.DataFrame(
        [
            {
                "stage": "Stage 1",
                "model": "Glaucoma EfficientNetB0 with glaucoma-fine-tuned backbone",
                "dataset": "EyePACS-AIROGS-Light-V2 Glaucoma Dataset",
                "task": "NRG vs RG glaucoma classification",
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc": auc,
            }
        ]
    )

    summary.to_csv(REPORTS_DIR / "glaucoma_summary_metrics.csv", index=False)
    summary.to_csv(REPORTS_DIR / "stage1_glaucoma_clean_comparison_table.csv", index=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["NRG", "RG"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORTS_DIR / "glaucoma_classification_report.csv")

    print("\nGlaucoma test performance")
    print("-" * 80)
    print(summary.to_string(index=False))


def save_training_settings(history_df: pd.DataFrame, trainable_layer_count: int):
    best_phase2_row = history_df[history_df["phase"] == "glaucoma_backbone_fine_tuning"].copy()

    if "val_auc" in best_phase2_row.columns and not best_phase2_row.empty:
        best_epoch = int(best_phase2_row.loc[best_phase2_row["val_auc"].idxmax(), "epoch"])
        best_val_auc = float(best_phase2_row["val_auc"].max())
    else:
        best_epoch = None
        best_val_auc = None

    settings = pd.DataFrame(
        [
            {
                "stage": "Stage 1",
                "script": "02_train_glaucoma_model.py",
                "total_epochs_requested": TOTAL_EPOCHS,
                "phase1_head_training_epochs": PHASE1_EPOCHS,
                "phase2_backbone_fine_tuning_epochs": PHASE2_EPOCHS,
                "actual_epochs_completed": len(history_df),
                "backbone_name": "EfficientNetB0",
                "initial_backbone_weights": "ImageNet",
                "backbone_frozen_in_phase1": True,
                "backbone_fine_tuned_in_phase2": True,
                "fine_tuned_last_n_backbone_layers": FINE_TUNE_LAST_N_LAYERS,
                "trainable_backbone_layers_in_phase2": trainable_layer_count,
                "batch_normalization_layers_frozen": True,
                "early_stopping_used": False,
                "best_phase2_validation_auc_epoch": best_epoch,
                "best_phase2_validation_auc": best_val_auc,
                "final_model_path": str(FINAL_MODEL_PATH),
                "phase2_best_model_path": str(PHASE2_BEST_MODEL_PATH),
                "transfer_claim_supported": "Yes - later EfficientNetB0 backbone layers were fine-tuned on glaucoma images before transfer.",
            }
        ]
    )

    settings.to_csv(REPORTS_DIR / "stage1_glaucoma_training_settings.csv", index=False)


def main():

    # Load glaucoma data, train 2 phases (frozen then fine-tune), save model for transfer.

    train_df = load_checked_csv(TRAIN_CSV)
    valid_df = load_checked_csv(VALID_CSV)
    test_df = load_checked_csv(TEST_CSV)

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(valid_df)}")
    print(f"Test rows: {len(test_df)}")

    train_ds = make_dataset(train_df, training=True)
    valid_ds = make_dataset(valid_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    model, backbone = build_glaucoma_model()
    model.summary()

    print("\nPhase 1: training glaucoma classification head only...")
    freeze_backbone(backbone)
    compile_binary_model(model, learning_rate=PHASE1_LR)

    history_phase1 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=make_callbacks(PHASE1_BEST_MODEL_PATH),
        verbose=1,
    )

    print("\nPhase 2: fine-tuning later EfficientNetB0 backbone layers on glaucoma...")
    trainable_layer_count = fine_tune_backbone(
        backbone=backbone,
        last_n_layers=FINE_TUNE_LAST_N_LAYERS,
    )

    print(f"Trainable EfficientNetB0 backbone layers in Phase 2: {trainable_layer_count}")

    compile_binary_model(model, learning_rate=PHASE2_LR)

    history_phase2 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=PHASE1_EPOCHS,
        callbacks=make_callbacks(PHASE2_BEST_MODEL_PATH),
        verbose=1,
    )

    history_df = merge_histories(history_phase1, history_phase2)
    history_df.to_csv(REPORTS_DIR / "stage1_glaucoma_training_history.csv", index=False)

    print(f"\nActual epochs completed: {len(history_df)}")

    if not PHASE2_BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The Phase 2 backbone fine-tuned checkpoint was not saved. "
            "Script 4 requires this checkpoint."
        )

    selected_model = tf.keras.models.load_model(PHASE2_BEST_MODEL_PATH)
    selected_model.save(FINAL_MODEL_PATH)

    try:
        selected_model.save(FINAL_MODEL_H5_PATH)
    except Exception as error:
        print(f"Could not save H5 copy. Keras model was saved correctly. Error: {error}")

    y_true, y_pred, probabilities = predict_dataset(selected_model, test_ds)

    save_predictions(test_df, y_true, y_pred, probabilities)
    save_metrics(y_true, y_pred, probabilities)
    save_training_settings(history_df, trainable_layer_count)

    plot_training_curve(history_df, "accuracy", "glaucoma_training_accuracy.png", "Accuracy")
    plot_training_curve(history_df, "loss", "glaucoma_training_loss.png", "Loss")
    plot_training_curve(history_df, "auc", "glaucoma_training_auc.png", "AUC")
    plot_confusion_matrix(y_true, y_pred, "glaucoma_confusion_matrix.png")

    print("\nStage 1 completed successfully.")
    print("The saved model now contains a glaucoma-fine-tuned EfficientNetB0 backbone.")
    print(f"Final model saved at: {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()