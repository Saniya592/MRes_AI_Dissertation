# 05_evaluate_models.py - Stage 5: Evaluate all 3 models and create comparison tables for dissertation

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

APTOS_TEST_CSV = REPORTS_DIR / "aptos_test_checked.csv"

MODEL_CONFIGS = {
    "aptos_baseline": {
        "model_name": "APTOS baseline EfficientNetB0",
        "model_path": MODELS_DIR / "aptos_baseline_efficientnetb0.keras",
        "required": True,
    },
    "imagenet_transfer": {
        "model_name": "APTOS transfer with ImageNet EfficientNetB0 backbone",
        "model_path": MODELS_DIR / "imagenet_transfer_efficientnetb0.keras",
        "required": True,
    },
    "glaucoma_transfer": {
        "model_name": "APTOS transfer with glaucoma-trained EfficientNetB0 backbone",
        "model_path": MODELS_DIR / "glaucoma_to_aptos_transfer_efficientnetb0.keras",
        "required": True,
    },
}

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 5
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


def make_dataset(df: pd.DataFrame) -> tf.data.Dataset:
    image_paths = df["image_path"].values
    labels = df["diagnosis"].values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    ds = ds.map(
        decode_multiclass_image,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(DATASET_OPTIONS)

    return ds




def load_model(path: Path) -> tf.keras.Model:
    require_file(path)
    print(f"\nLoading model: {path}")
    return tf.keras.models.load_model(path)


def predict_model(model: tf.keras.Model, dataset: tf.data.Dataset):
    probabilities = model.predict(dataset, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    true_labels = []

    for _, batch_labels in dataset:
        true_labels.extend(np.argmax(batch_labels.numpy(), axis=1).tolist())

    true_labels = np.array(true_labels).astype(int)

    return true_labels, predictions, probabilities

# Convert 5-class DR grades to binary (0-1=Non-referable, 2-4=Referable) for screening evaluation

def binary_labels_from_five_class(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(int)
    return np.where(labels >= 2, 1, 0)

# Calculate metrics for 5-class classification: accuracy, macro-F1 (equal weight per class), weighted-F1 (weighted by class size).

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



def save_prediction_file(
    model_key: str,
    model_name: str,
    test_df: pd.DataFrame,
    y_true,
    y_pred,
    probabilities,
) -> pd.DataFrame:
    output_df = test_df.copy()

    output_df["model_key"] = model_key
    output_df["model_name"] = model_name

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

    output_path = REPORTS_DIR / f"{model_key}_test_predictions.csv"
    output_df.to_csv(output_path, index=False)

    misclassified_df = output_df[output_df["correct"] == False].copy()
    misclassified_df = misclassified_df.sort_values(
        by=["true_label", "confidence"],
        ascending=[True, False],
    )

    misclassified_path = REPORTS_DIR / f"{model_key}_misclassified_images.csv"
    misclassified_df.to_csv(misclassified_path, index=False)

    return output_df




def save_classification_report(model_key: str, y_true, y_pred) -> pd.DataFrame:
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

    output_path = REPORTS_DIR / f"{model_key}_per_class_metrics.csv"
    report_df.to_csv(output_path, index=False)

    return report_df


def save_binary_classification_report(model_key: str, y_true, y_pred) -> pd.DataFrame:
    y_true_binary = binary_labels_from_five_class(y_true)
    y_pred_binary = binary_labels_from_five_class(y_pred)

    report = classification_report(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1],
        target_names=BINARY_CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "class_name"})

    output_path = REPORTS_DIR / f"{model_key}_binary_per_class_metrics.csv"
    report_df.to_csv(output_path, index=False)

    return report_df



def make_confusion_dataframe(cm: np.ndarray, class_names: list) -> pd.DataFrame:
    return pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    )

# Save confusion matrices (raw counts and normalized percentages) for 5-class DR grading to identify misclassification patterns.

def save_confusion_matrices(model_key: str, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))

    cm_df = make_confusion_dataframe(cm, CLASS_NAMES)
    cm_df.to_csv(REPORTS_DIR / f"{model_key}_confusion_matrix.csv")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    cm_norm_df = make_confusion_dataframe(cm_norm, CLASS_NAMES)
    cm_norm_df.to_csv(REPORTS_DIR / f"{model_key}_normalized_confusion_matrix.csv")

    plot_confusion_matrix(
        cm,
        CLASS_NAMES,
        FIGURES_DIR / f"{model_key}_confusion_matrix.png",
        f"{model_key} five-class confusion matrix",
        normalized=False,
    )

    plot_confusion_matrix(
        cm_norm,
        CLASS_NAMES,
        FIGURES_DIR / f"{model_key}_normalized_confusion_matrix.png",
        f"{model_key} normalized five-class confusion matrix",
        normalized=True,
    )

    return cm, cm_norm


def save_binary_confusion_matrices(model_key: str, y_true, y_pred):
    y_true_binary = binary_labels_from_five_class(y_true)
    y_pred_binary = binary_labels_from_five_class(y_pred)

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    cm_df = make_confusion_dataframe(cm, BINARY_CLASS_NAMES)
    cm_df.to_csv(REPORTS_DIR / f"{model_key}_binary_confusion_matrix.csv")

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    cm_norm_df = make_confusion_dataframe(cm_norm, BINARY_CLASS_NAMES)
    cm_norm_df.to_csv(REPORTS_DIR / f"{model_key}_binary_normalized_confusion_matrix.csv")

    plot_confusion_matrix(
        cm,
        BINARY_CLASS_NAMES,
        FIGURES_DIR / f"{model_key}_referable_dr_confusion_matrix.png",
        f"{model_key} binary referable DR confusion matrix",
        normalized=False,
    )

    plot_confusion_matrix(
        cm_norm,
        BINARY_CLASS_NAMES,
        FIGURES_DIR / f"{model_key}_referable_dr_normalized_confusion_matrix.png",
        f"{model_key} normalized binary referable DR confusion matrix",
        normalized=True,
    )

    return cm, cm_norm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: Path,
    title: str,
    normalized: bool,
):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(title.replace("_", " ").title())
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


# Per-class error analysis For each DR grade, find support, correct predictions, error count, and most common misclassification.

def class_error_analysis(model_key: str, y_true, y_pred) -> pd.DataFrame:
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
                "model_key": model_key,
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

    analysis_df = pd.DataFrame(rows)
    analysis_df = analysis_df.sort_values(
        by=["error_count", "error_rate"],
        ascending=[False, False],
    )

    analysis_df.to_csv(REPORTS_DIR / f"{model_key}_error_analysis_by_class.csv", index=False)

    return analysis_df


def confusion_pair_analysis(model_key: str, y_true, y_pred) -> pd.DataFrame:
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
                        "model_key": model_key,
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

    pair_df.to_csv(REPORTS_DIR / f"{model_key}_confusion_pairs.csv", index=False)

    return pair_df




def make_summary_rows(model_key: str, model_name: str, y_true, y_pred):
    five_class_metrics = calculate_multiclass_metrics(y_true, y_pred)
    binary_metrics = calculate_binary_metrics(y_true, y_pred)

    return [
        {
            "model_key": model_key,
            "model_name": model_name,
            "task": "5-class diabetic retinopathy grading",
            **five_class_metrics,
        },
        {
            "model_key": model_key,
            "model_name": model_name,
            "task": "Binary referable diabetic retinopathy",
            **binary_metrics,
        },
    ]


def save_individual_summary(model_key: str, model_name: str, y_true, y_pred):
    rows = make_summary_rows(model_key, model_name, y_true, y_pred)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(REPORTS_DIR / f"{model_key}_summary_metrics.csv", index=False)
    return summary_df



def plot_all_model_metric_comparison(summary_df: pd.DataFrame):
    plot_df = summary_df[
        summary_df["task"] == "5-class diabetic retinopathy grading"
    ].copy()

    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(10, 5))

    for index, (_, row) in enumerate(plot_df.iterrows()):
        values = [row[metric] for metric in metrics]
        offset = (index - 1) * width
        plt.bar(x + offset, values, width, label=row["model_key"])

    plt.xticks(x, ["Accuracy", "Macro F1", "Weighted F1"])
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("APTOS baseline vs ImageNet transfer vs glaucoma transfer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "all_model_metric_comparison.png", dpi=300)
    plt.close()


def plot_stage2_stage3_metric_comparison(summary_df: pd.DataFrame):
    selected = summary_df[
        (summary_df["task"] == "5-class diabetic retinopathy grading")
        & (summary_df["model_key"].isin(["aptos_baseline", "glaucoma_transfer"]))
    ].copy()

    if selected["model_key"].nunique() < 2:
        return

    metrics = ["accuracy", "macro_f1", "weighted_f1"]

    baseline_values = selected[selected["model_key"] == "aptos_baseline"][metrics].iloc[0].values
    transfer_values = selected[selected["model_key"] == "glaucoma_transfer"][metrics].iloc[0].values

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, baseline_values, width, label="APTOS baseline")
    plt.bar(x + width / 2, transfer_values, width, label="Glaucoma transfer")

    plt.xticks(x, ["Accuracy", "Macro F1", "Weighted F1"])
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Stage 2 baseline vs glaucoma-trained transfer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage2_stage3_metric_comparison.png", dpi=300)
    plt.close()


def plot_imagenet_vs_glaucoma_metric_comparison(summary_df: pd.DataFrame):
    selected = summary_df[
        (summary_df["task"] == "5-class diabetic retinopathy grading")
        & (summary_df["model_key"].isin(["imagenet_transfer", "glaucoma_transfer"]))
    ].copy()

    if selected["model_key"].nunique() < 2:
        return

    metrics = ["accuracy", "macro_f1", "weighted_f1"]

    imagenet_values = selected[selected["model_key"] == "imagenet_transfer"][metrics].iloc[0].values
    glaucoma_values = selected[selected["model_key"] == "glaucoma_transfer"][metrics].iloc[0].values

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, imagenet_values, width, label="ImageNet transfer")
    plt.bar(x + width / 2, glaucoma_values, width, label="Glaucoma-trained transfer")

    plt.xticks(x, ["Accuracy", "Macro F1", "Weighted F1"])
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("ImageNet transfer vs glaucoma-trained transfer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "imagenet_vs_glaucoma_transfer_metric_comparison.png", dpi=300)
    plt.close()


def plot_per_class_f1_comparison():
    imagenet_path = REPORTS_DIR / "imagenet_transfer_per_class_metrics.csv"
    glaucoma_path = REPORTS_DIR / "glaucoma_transfer_per_class_metrics.csv"

    if not imagenet_path.exists() or not glaucoma_path.exists():
        return

    imagenet_df = pd.read_csv(imagenet_path)
    glaucoma_df = pd.read_csv(glaucoma_path)

    imagenet_df = imagenet_df[imagenet_df["class_name"].isin(CLASS_NAMES)].copy()
    glaucoma_df = glaucoma_df[glaucoma_df["class_name"].isin(CLASS_NAMES)].copy()

    merged = pd.merge(
        imagenet_df[["class_name", "f1-score"]],
        glaucoma_df[["class_name", "f1-score"]],
        on="class_name",
        suffixes=("_imagenet", "_glaucoma"),
    )

    merged["class_name"] = pd.Categorical(
        merged["class_name"],
        categories=CLASS_NAMES,
        ordered=True,
    )
    merged = merged.sort_values("class_name")

    x = np.arange(len(merged))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, merged["f1-score_imagenet"], width, label="ImageNet transfer")
    plt.bar(x + width / 2, merged["f1-score_glaucoma"], width, label="Glaucoma-trained transfer")

    plt.xticks(x, merged["class_name"], rotation=35, ha="right")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.title("Per-class F1-score: ImageNet transfer vs glaucoma-trained transfer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "imagenet_vs_glaucoma_transfer_per_class_f1.png", dpi=300)
    plt.close()


def plot_error_count_comparison():
    imagenet_path = REPORTS_DIR / "imagenet_transfer_error_analysis_by_class.csv"
    glaucoma_path = REPORTS_DIR / "glaucoma_transfer_error_analysis_by_class.csv"

    if not imagenet_path.exists() or not glaucoma_path.exists():
        return

    imagenet_df = pd.read_csv(imagenet_path)
    glaucoma_df = pd.read_csv(glaucoma_path)

    merged = pd.merge(
        imagenet_df[["true_class_name", "error_count"]],
        glaucoma_df[["true_class_name", "error_count"]],
        on="true_class_name",
        suffixes=("_imagenet", "_glaucoma"),
    )

    merged["true_class_name"] = pd.Categorical(
        merged["true_class_name"],
        categories=CLASS_NAMES,
        ordered=True,
    )
    merged = merged.sort_values("true_class_name")

    x = np.arange(len(merged))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, merged["error_count_imagenet"], width, label="ImageNet transfer")
    plt.bar(x + width / 2, merged["error_count_glaucoma"], width, label="Glaucoma-trained transfer")

    plt.xticks(x, merged["true_class_name"], rotation=35, ha="right")
    plt.ylabel("Error count")
    plt.title("Per-class errors: ImageNet transfer vs glaucoma-trained transfer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "imagenet_vs_glaucoma_transfer_error_count.png", dpi=300)
    plt.close()


# Create side-by-side comparison table: ImageNet transfer vs Glaucoma transfer.

def save_transfer_backbone_comparison(summary_df: pd.DataFrame):
    selected = summary_df[
        summary_df["model_key"].isin(["imagenet_transfer", "glaucoma_transfer"])
    ].copy()

    selected.to_csv(REPORTS_DIR / "transfer_backbone_comparison_summary.csv", index=False)

    rows = []

    for task in selected["task"].unique():
        task_df = selected[selected["task"] == task].copy()

        if task_df["model_key"].nunique() < 2:
            continue

        imagenet_row = task_df[task_df["model_key"] == "imagenet_transfer"].iloc[0]
        glaucoma_row = task_df[task_df["model_key"] == "glaucoma_transfer"].iloc[0]

        for metric in ["accuracy", "macro_f1", "weighted_f1"]:
            imagenet_value = float(imagenet_row[metric])
            glaucoma_value = float(glaucoma_row[metric])
            difference = glaucoma_value - imagenet_value

            if glaucoma_value > imagenet_value:
                better_model = "Glaucoma-trained transfer"
            elif glaucoma_value < imagenet_value:
                better_model = "ImageNet transfer"
            else:
                better_model = "Tie"

            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "imagenet_transfer": imagenet_value,
                    "glaucoma_transfer": glaucoma_value,
                    "difference_glaucoma_minus_imagenet": difference,
                    "better_model": better_model,
                }
            )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(REPORTS_DIR / "imagenet_vs_glaucoma_transfer_side_by_side.csv", index=False)

    return comparison_df

# This summary helps interpret results and guides dissertation writing

def save_interpretive_summary(summary_df: pd.DataFrame, transfer_comparison_df: pd.DataFrame):
    text_lines = []

    text_lines.append("Script 5 evaluation summary")
    text_lines.append("=" * 80)
    text_lines.append("")
    text_lines.append("Models evaluated:")
    text_lines.append("1. APTOS baseline EfficientNetB0")
    text_lines.append("2. APTOS transfer with ImageNet EfficientNetB0 backbone")
    text_lines.append("3. APTOS transfer with glaucoma-trained EfficientNetB0 backbone")
    text_lines.append("")
    text_lines.append("Main outputs created:")
    text_lines.append("- imagenet_transfer_summary_metrics.csv")
    text_lines.append("- imagenet_transfer_per_class_metrics.csv")
    text_lines.append("- glaucoma_transfer_summary_metrics.csv")
    text_lines.append("- glaucoma_transfer_per_class_metrics.csv")
    text_lines.append("- model_comparison_summary.csv")
    text_lines.append("- transfer_backbone_comparison_summary.csv")
    text_lines.append("- imagenet_vs_glaucoma_transfer_side_by_side.csv")
    text_lines.append("")

    if not transfer_comparison_df.empty:
        text_lines.append("ImageNet-transfer vs glaucoma-trained-transfer comparison:")
        for _, row in transfer_comparison_df.iterrows():
            text_lines.append(
                f"{row['task']} | {row['metric']}: "
                f"ImageNet={row['imagenet_transfer']:.4f}, "
                f"Glaucoma={row['glaucoma_transfer']:.4f}, "
                f"Difference={row['difference_glaucoma_minus_imagenet']:.4f}, "
                f"Better={row['better_model']}"
            )
    else:
        text_lines.append("ImageNet-transfer vs glaucoma-trained-transfer comparison was not available.")

    text_lines.append("")
    text_lines.append("Report interpretation:")
    text_lines.append(
        "If the glaucoma-trained transfer model is better, the report should state that "
        "glaucoma backbone fine-tuning improved transfer performance. If the ImageNet-transfer "
        "model is better, the report should state that glaucoma backbone fine-tuning did not "
        "improve APTOS performance under this setup."
    )
    text_lines.append("")
    text_lines.append(
        "Five-class grading is expected to be harder than binary referable DR classification "
        "because adjacent DR grades can be visually similar and minority classes have fewer examples."
    )

    output_path = REPORTS_DIR / "script5_report_ready_interpretive_summary.txt"
    output_path.write_text("\n".join(text_lines), encoding="utf-8")



def evaluate_one_model(model_key: str, model_config: dict, test_df: pd.DataFrame):
    model_path = model_config["model_path"]
    model_name = model_config["model_name"]

    model = load_model(model_path)
    test_ds = make_dataset(test_df)

    y_true, y_pred, probabilities = predict_model(model, test_ds)

    predictions_df = save_prediction_file(
        model_key=model_key,
        model_name=model_name,
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
    )

    save_individual_summary(model_key, model_name, y_true, y_pred)
    save_classification_report(model_key, y_true, y_pred)
    save_binary_classification_report(model_key, y_true, y_pred)

    save_confusion_matrices(model_key, y_true, y_pred)
    save_binary_confusion_matrices(model_key, y_true, y_pred)

    error_df = class_error_analysis(model_key, y_true, y_pred)
    pair_df = confusion_pair_analysis(model_key, y_true, y_pred)

    summary_rows = make_summary_rows(model_key, model_name, y_true, y_pred)

    return {
        "model_key": model_key,
        "model_name": model_name,
        "summary_rows": summary_rows,
        "predictions_df": predictions_df,
        "error_df": error_df,
        "pair_df": pair_df,
    }


# Load all 3 models (Baseline, ImageNet, Glaucoma), evaluate on APTOS test set, generate comparison tables and figures

def main():
    print("\nStarting Script 5: full model evaluation and comparison")

    test_df = load_checked_csv(APTOS_TEST_CSV)

    print("\nAPTOS test distribution:")
    print(test_df["diagnosis"].value_counts().sort_index())

    all_results = []
    summary_rows = []
    error_frames = []
    pair_frames = []
    prediction_frames = []

    for model_key, model_config in MODEL_CONFIGS.items():
        model_path = model_config["model_path"]

        if not model_path.exists():
            message = f"Model file not found for {model_key}: {model_path}"

            if model_config.get("required", True):
                raise FileNotFoundError(message)

            print(message)
            continue

        result = evaluate_one_model(
            model_key=model_key,
            model_config=model_config,
            test_df=test_df,
        )

        all_results.append(result)
        summary_rows.extend(result["summary_rows"])
        error_frames.append(result["error_df"])
        pair_frames.append(result["pair_df"])
        prediction_frames.append(result["predictions_df"])

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(REPORTS_DIR / "model_comparison_summary.csv", index=False)
    summary_df.to_csv(REPORTS_DIR / "stage2_stage3_side_by_side_comparison.csv", index=False)

    if error_frames:
        combined_error_df = pd.concat(error_frames, ignore_index=True)
        combined_error_df.to_csv(REPORTS_DIR / "combined_error_analysis_by_class.csv", index=False)

    if pair_frames:
        combined_pair_df = pd.concat(pair_frames, ignore_index=True)
        combined_pair_df.to_csv(REPORTS_DIR / "combined_confusion_pairs.csv", index=False)

    if prediction_frames:
        combined_predictions_df = pd.concat(prediction_frames, ignore_index=True)
        combined_predictions_df.to_csv(REPORTS_DIR / "combined_model_predictions.csv", index=False)

    transfer_comparison_df = save_transfer_backbone_comparison(summary_df)

    plot_all_model_metric_comparison(summary_df)
    plot_stage2_stage3_metric_comparison(summary_df)
    plot_imagenet_vs_glaucoma_metric_comparison(summary_df)
    plot_per_class_f1_comparison()
    plot_error_count_comparison()

    save_interpretive_summary(summary_df, transfer_comparison_df)

    print("\nFinal model comparison")
    print("-" * 80)
    print(summary_df.to_string(index=False))

    print("\nScript 5 completed successfully.")
    print("Required ImageNet-transfer files created:")
    print(REPORTS_DIR / "imagenet_transfer_summary_metrics.csv")
    print(REPORTS_DIR / "imagenet_transfer_per_class_metrics.csv")
    print("\nNow run Script 7 again.")


if __name__ == "__main__":
    main()