# 04_transfer_glaucoma_to_aptos.py - Stage 3: CORE EXPERIMENT - Transfer glaucoma features to DR grading.

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

GLAUCOMA_MODEL_PATH = MODELS_DIR / "glaucoma_efficientnetb0.keras"
STAGE1_SETTINGS_PATH = REPORTS_DIR / "stage1_glaucoma_training_settings.csv"

FINAL_TRANSFER_MODEL_PATH = MODELS_DIR / "glaucoma_to_aptos_transfer_efficientnetb0.keras"
BEST_TRANSFER_MODEL_PATH = MODELS_DIR / "glaucoma_to_aptos_transfer_efficientnetb0_best.keras"

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 5

FROZEN_EPOCHS = 25
FINE_TUNE_EPOCHS = 25
TOTAL_EPOCHS = FROZEN_EPOCHS + FINE_TUNE_EPOCHS

TRANSFER_FINE_TUNE_LAST_N_LAYERS = 25

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

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = True


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

# Verify Script 02 actually fine-tuned backbone on glaucoma (prevents using non-fine-tuned backbone by mistake)

def verify_glaucoma_backbone_training() -> None:
    require_file(STAGE1_SETTINGS_PATH)

    settings = pd.read_csv(STAGE1_SETTINGS_PATH)

    required_column = "backbone_fine_tuned_in_phase2"
    if required_column not in settings.columns:
        raise RuntimeError(
            "Script 4 cannot verify glaucoma backbone fine-tuning. "
            "Please rerun the updated Script 2 first."
        )

    value = str(settings.loc[0, required_column]).strip().lower()

    if value not in ["true", "1", "yes"]:
        raise RuntimeError(
            "The loaded Stage 1 settings show that the backbone was not fine-tuned on glaucoma. "
            "Please rerun the updated Script 2 first."
        )

    print("\nVerification passed:")
    print("Script 2 fine-tuned later EfficientNetB0 backbone layers on glaucoma images.")
    print("Script 4 will now transfer this glaucoma-fine-tuned backbone to APTOS.")


def load_checked_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)

    required_columns = ["image_path", "diagnosis", "image_exists"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {missing_columns}")

    df = df[df["image_exists"] == True].copy()
    df["image_path"] = df["image_path"].astype(str)
    df["diagnosis"] = df["diagnosis"].astype(int)

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

    class_weights = {i: float(weights[i]) for i in range(NUM_CLASSES)}
    return class_weights


def count_trainable_layers(model: tf.keras.Model) -> int:
    return sum(1 for layer in model.layers if layer.trainable)

# Load the glaucoma-fine-tuned EfficientNetB0 backbone from Script 02

def load_glaucoma_fine_tuned_backbone() -> tf.keras.Model:
    require_file(GLAUCOMA_MODEL_PATH)

    glaucoma_model = tf.keras.models.load_model(GLAUCOMA_MODEL_PATH)

    try:
        feature_extractor = glaucoma_model.get_layer("efficientnetb0")
    except ValueError as error:
        raise ValueError(
            "Could not find the EfficientNetB0 layer named 'efficientnetb0' "
            "inside the glaucoma model."
        ) from error

    print("\nLoaded glaucoma model:")
    print(GLAUCOMA_MODEL_PATH)

    print("\nLoaded feature extractor:")
    print(feature_extractor.name)

    print(
        "Trainable layers inside loaded glaucoma backbone before transfer freezing:",
        count_trainable_layers(feature_extractor),
    )

    return feature_extractor

# Build transfer model: glaucoma-fine-tuned backbone + new head (dropout + dense layers) for 5-class DR classification.

def build_transfer_model(feature_extractor: tf.keras.Model) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="aptos_transfer_input")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.05, seed=SEED),
            tf.keras.layers.RandomZoom(0.05, seed=SEED),
            tf.keras.layers.RandomContrast(0.05, seed=SEED),
        ],
        name="aptos_transfer_augmentation",
    )

    x = augmentation(inputs)

    feature_extractor.trainable = False
    x = feature_extractor(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.35, seed=SEED, name="dropout_transfer_1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_transfer_1")(x)
    x = tf.keras.layers.Dropout(0.25, seed=SEED, name="dropout_transfer_2")(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="aptos_transfer_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="glaucoma_to_aptos_transfer_efficientnetb0",
    )

    return model


def compile_multiclass_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

# Unfreeze last 25 layers of glaucoma-fine-tuned backbone for APTOS fine-tuning.

def fine_tune_transfer_backbone(feature_extractor: tf.keras.Model, last_n_layers: int) -> int:
    feature_extractor.trainable = True

    for layer in feature_extractor.layers:
        layer.trainable = False

    for layer in feature_extractor.layers[-last_n_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    trainable_layer_count = count_trainable_layers(feature_extractor)
    return trainable_layer_count


def make_callbacks():
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_TRANSFER_MODEL_PATH),
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
            "phase": "aptos_head_training_on_frozen_glaucoma_backbone",
        }
        for key, values in history_phase1.history.items():
            row[key] = values[epoch_index]
        rows.append(row)

    for epoch_index in range(len(history_phase2.history["loss"])):
        row = {
            "epoch": FROZEN_EPOCHS + epoch_index + 1,
            "phase": "aptos_fine_tuning_on_glaucoma_backbone",
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


def binary_labels_from_five_class(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    return np.where(labels >= 2, 1, 0)


def calculate_metrics(y_true, y_pred) -> dict:
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


def save_predictions(test_df, y_true, y_pred, probabilities):
    output_df = test_df.copy()

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
    output_df["true_binary_name"] = np.where(
        output_df["true_binary"] == 1,
        "Referable_DR",
        "Non_referable_DR",
    )
    output_df["pred_binary_name"] = np.where(
        output_df["pred_binary"] == 1,
        "Referable_DR",
        "Non_referable_DR",
    )
    output_df["binary_correct"] = output_df["true_binary"] == output_df["pred_binary"]

    output_df.to_csv(REPORTS_DIR / "glaucoma_transfer_test_predictions.csv", index=False)


def save_reports_and_metrics(y_true, y_pred):
    five_class_metrics = calculate_metrics(y_true, y_pred)
    binary_metrics = calculate_binary_metrics(y_true, y_pred)

    summary = pd.DataFrame(
        [
            {
                "stage": "Stage 3",
                "model": "Glaucoma-fine-tuned EfficientNetB0 backbone transferred to APTOS",
                "dataset": "APTOS 2019 Blindness Detection Dataset",
                "task": "5-class diabetic retinopathy grading",
                **five_class_metrics,
            },
            {
                "stage": "Stage 3",
                "model": "Glaucoma-fine-tuned EfficientNetB0 backbone transferred to APTOS",
                "dataset": "APTOS 2019 Blindness Detection Dataset",
                "task": "Binary referable diabetic retinopathy",
                **binary_metrics,
            },
        ]
    )

    summary.to_csv(REPORTS_DIR / "glaucoma_transfer_summary_metrics.csv", index=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORTS_DIR / "glaucoma_transfer_classification_report.csv"
    )

    print("\nStage 3 transfer test performance")
    print("-" * 80)
    print(summary.to_string(index=False))


def plot_training_curve(history_df: pd.DataFrame, metric: str, filename: str, ylabel: str):
    plt.figure(figsize=(8, 5))

    if metric in history_df.columns:
        plt.plot(history_df["epoch"], history_df[metric], label=f"Training {metric}")

    val_metric = f"val_{metric}"
    if val_metric in history_df.columns:
        plt.plot(history_df["epoch"], history_df[val_metric], label=f"Validation {metric}")

    plt.axvline(FROZEN_EPOCHS, linestyle="--", linewidth=1, label="APTOS fine-tuning starts")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"Glaucoma-to-APTOS transfer {ylabel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, filename: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))

    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(ticks=np.arange(NUM_CLASSES), labels=CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(NUM_CLASSES), labels=CLASS_NAMES)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def plot_binary_confusion_matrix(y_true, y_pred, filename: str, title: str):
    y_true_binary = binary_labels_from_five_class(y_true)
    y_pred_binary = binary_labels_from_five_class(y_pred)

    labels = ["Non_referable_DR", "Referable_DR"]
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(ticks=np.arange(2), labels=labels, rotation=20, ha="right")
    plt.yticks(ticks=np.arange(2), labels=labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def save_training_settings(history_df: pd.DataFrame, trainable_layer_count: int):
    best_row = history_df.loc[history_df["val_loss"].idxmin()]

    settings = pd.DataFrame(
        [
            {
                "stage": "Stage 3",
                "script": "04_transfer_glaucoma_to_aptos.py",
                "total_epochs_requested": TOTAL_EPOCHS,
                "frozen_epochs": FROZEN_EPOCHS,
                "fine_tune_epochs": FINE_TUNE_EPOCHS,
                "actual_epochs_completed": len(history_df),
                "source_model": str(GLAUCOMA_MODEL_PATH),
                "source_backbone": "EfficientNetB0",
                "source_backbone_status": "Glaucoma fine-tuned in Script 2",
                "transfer_phase1_backbone_frozen": True,
                "transfer_phase2_backbone_fine_tuned_on_aptos": True,
                "transfer_fine_tuned_last_n_layers": TRANSFER_FINE_TUNE_LAST_N_LAYERS,
                "trainable_backbone_layers_in_transfer_phase2": trainable_layer_count,
                "batch_normalization_layers_frozen": True,
                "early_stopping_used": False,
                "best_validation_loss_epoch": int(best_row["epoch"]),
                "best_validation_loss": float(best_row["val_loss"]),
                "final_model_path": str(FINAL_TRANSFER_MODEL_PATH),
                "best_model_path": str(BEST_TRANSFER_MODEL_PATH),
            }
        ]
    )

    settings.to_csv(REPORTS_DIR / "stage3_transfer_training_settings.csv", index=False)


def main():

    # Verify glaucoma backbone, 2-phase transfer (25 frozen + 25 fine-tune), save transfer model.

    verify_glaucoma_backbone_training()

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

    feature_extractor = load_glaucoma_fine_tuned_backbone()
    model = build_transfer_model(feature_extractor)

    model.summary()

    print("\nPhase 1: training APTOS head on frozen glaucoma-fine-tuned backbone...")
    feature_extractor.trainable = False
    compile_multiclass_model(model, learning_rate=PHASE1_LR)

    history_phase1 = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=FROZEN_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(),
        verbose=1,
    )

    print("\nPhase 2: fine-tuning later glaucoma-fine-tuned backbone layers on APTOS...")
    trainable_layer_count = fine_tune_transfer_backbone(
        feature_extractor,
        TRANSFER_FINE_TUNE_LAST_N_LAYERS,
    )

    print(f"Trainable EfficientNetB0 backbone layers in Stage 3 Phase 2: {trainable_layer_count}")

    compile_multiclass_model(model, learning_rate=PHASE2_LR)

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
    history_df.to_csv(REPORTS_DIR / "stage3_transfer_training_history.csv", index=False)

    print(f"\nActual epochs completed: {len(history_df)}")

    if not BEST_TRANSFER_MODEL_PATH.exists():
        raise FileNotFoundError("Best Stage 3 transfer model was not saved.")

    selected_model = tf.keras.models.load_model(BEST_TRANSFER_MODEL_PATH)
    selected_model.save(FINAL_TRANSFER_MODEL_PATH)

    y_true, y_pred, probabilities = predict_dataset(selected_model, test_ds)

    save_predictions(test_df, y_true, y_pred, probabilities)
    save_reports_and_metrics(y_true, y_pred)
    save_training_settings(history_df, trainable_layer_count)

    plot_training_curve(
        history_df,
        "accuracy",
        "glaucoma_transfer_training_accuracy.png",
        "Accuracy",
    )
    plot_training_curve(
        history_df,
        "loss",
        "glaucoma_transfer_training_loss.png",
        "Loss",
    )
    plot_training_curve(
        history_df,
        "precision",
        "glaucoma_transfer_training_precision.png",
        "Precision",
    )
    plot_training_curve(
        history_df,
        "recall",
        "glaucoma_transfer_training_recall.png",
        "Recall",
    )

    plot_confusion_matrix(
        y_true,
        y_pred,
        "glaucoma_transfer_confusion_matrix.png",
        "Glaucoma-to-APTOS transfer five-class confusion matrix",
    )

    plot_binary_confusion_matrix(
        y_true,
        y_pred,
        "glaucoma_transfer_referable_dr_confusion_matrix.png",
        "Glaucoma-to-APTOS transfer binary referable DR confusion matrix",
    )

    print("\nStage 3 completed successfully.")
    print("The transfer model used the glaucoma-fine-tuned EfficientNetB0 backbone from Script 2.")
    print(f"Final transfer model saved at: {FINAL_TRANSFER_MODEL_PATH}")


if __name__ == "__main__":
    main()