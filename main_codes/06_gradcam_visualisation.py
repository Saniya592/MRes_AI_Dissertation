# 06_gradcam_visualisation.py - Stage 6: Grad-CAM visualisations for qualitative error analysis.

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
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
GRADCAM_DIR = PROJECT_ROOT / "outputs" / "gradcam"

BASELINE_GRADCAM_DIR = GRADCAM_DIR / "misclassified_cases" / "aptos_baseline"
TRANSFER_GRADCAM_DIR = GRADCAM_DIR / "misclassified_cases" / "glaucoma_to_aptos_transfer"
COMPARISON_GRADCAM_DIR = GRADCAM_DIR / "misclassified_case_comparisons"

BASELINE_GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
TRANSFER_GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_MODEL_PATH = MODELS_DIR / "aptos_baseline_efficientnetb0.keras"
TRANSFER_MODEL_PATH = MODELS_DIR / "glaucoma_to_aptos_transfer_efficientnetb0.keras"

BASELINE_PREDICTIONS_CSV = REPORTS_DIR / "aptos_baseline_test_predictions.csv"
TRANSFER_PREDICTIONS_CSV = REPORTS_DIR / "glaucoma_transfer_test_predictions.csv"

IMG_SIZE = 224
MAX_CASES_PER_GROUP = 8
MAX_TOTAL_CASES = 32

CLASS_NAMES = [
    "No_DR",
    "Mild_DR",
    "Moderate_DR",
    "Severe_DR",
    "Proliferative_DR",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_model(path: Path) -> tf.keras.Model:
    require_file(path)
    print(f"\nLoading model: {path}")
    return tf.keras.models.load_model(path)


def load_predictions() -> pd.DataFrame:
    require_file(BASELINE_PREDICTIONS_CSV)
    require_file(TRANSFER_PREDICTIONS_CSV)

    baseline = pd.read_csv(BASELINE_PREDICTIONS_CSV)
    transfer = pd.read_csv(TRANSFER_PREDICTIONS_CSV)

    required_columns = [
        "id_code",
        "image_path",
        "true_label",
        "true_class_name",
        "pred_label",
        "pred_class_name",
        "confidence",
        "correct",
    ]

    for file_name, df in [
        ("aptos_baseline_test_predictions.csv", baseline),
        ("glaucoma_transfer_test_predictions.csv", transfer),
    ]:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{file_name} is missing columns: {missing_columns}")

    baseline_small = baseline[
        [
            "id_code",
            "image_path",
            "true_label",
            "true_class_name",
            "pred_label",
            "pred_class_name",
            "confidence",
            "correct",
        ]
    ].copy()

    transfer_small = transfer[
        [
            "id_code",
            "image_path",
            "true_label",
            "true_class_name",
            "pred_label",
            "pred_class_name",
            "confidence",
            "correct",
        ]
    ].copy()

    baseline_small = baseline_small.rename(
        columns={
            "pred_label": "baseline_pred_label",
            "pred_class_name": "baseline_pred_class_name",
            "confidence": "baseline_confidence",
            "correct": "baseline_correct",
        }
    )

    transfer_small = transfer_small.rename(
        columns={
            "pred_label": "transfer_pred_label",
            "pred_class_name": "transfer_pred_class_name",
            "confidence": "transfer_confidence",
            "correct": "transfer_correct",
        }
    )

    merged = pd.merge(
        baseline_small,
        transfer_small[
            [
                "id_code",
                "transfer_pred_label",
                "transfer_pred_class_name",
                "transfer_confidence",
                "transfer_correct",
            ]
        ],
        on="id_code",
        how="inner",
    )

    merged["baseline_correct"] = merged["baseline_correct"].astype(bool)
    merged["transfer_correct"] = merged["transfer_correct"].astype(bool)

    return merged

# Select 32 cases from 4 error groups: baseline wrong/transfer right, transfer wrong/baseline right, both wrong, high-confidence transfer errors.

def select_cases_for_gradcam(merged: pd.DataFrame) -> pd.DataFrame:
    selected_groups = []

    baseline_wrong_transfer_right = merged[
        (merged["baseline_correct"] == False)
        & (merged["transfer_correct"] == True)
    ].copy()

    baseline_wrong_transfer_right["case_group"] = "baseline_wrong_transfer_right"
    baseline_wrong_transfer_right = baseline_wrong_transfer_right.sort_values(
        by="baseline_confidence",
        ascending=False,
    ).head(MAX_CASES_PER_GROUP)

    selected_groups.append(baseline_wrong_transfer_right)

    transfer_wrong_baseline_right = merged[
        (merged["baseline_correct"] == True)
        & (merged["transfer_correct"] == False)
    ].copy()

    transfer_wrong_baseline_right["case_group"] = "transfer_wrong_baseline_right"
    transfer_wrong_baseline_right = transfer_wrong_baseline_right.sort_values(
        by="transfer_confidence",
        ascending=False,
    ).head(MAX_CASES_PER_GROUP)

    selected_groups.append(transfer_wrong_baseline_right)

    both_wrong = merged[
        (merged["baseline_correct"] == False)
        & (merged["transfer_correct"] == False)
    ].copy()

    both_wrong["case_group"] = "both_models_wrong"
    both_wrong = both_wrong.sort_values(
        by=["transfer_confidence", "baseline_confidence"],
        ascending=False,
    ).head(MAX_CASES_PER_GROUP)

    selected_groups.append(both_wrong)

    transfer_wrong_high_confidence = merged[
        merged["transfer_correct"] == False
    ].copy()

    transfer_wrong_high_confidence["case_group"] = "transfer_wrong_high_confidence"
    transfer_wrong_high_confidence = transfer_wrong_high_confidence.sort_values(
        by="transfer_confidence",
        ascending=False,
    ).head(MAX_CASES_PER_GROUP)

    selected_groups.append(transfer_wrong_high_confidence)

    selected = pd.concat(selected_groups, ignore_index=True)

    selected = selected.drop_duplicates(subset=["id_code"], keep="first")
    selected = selected.head(MAX_TOTAL_CASES).copy()

    selected["case_number"] = np.arange(1, len(selected) + 1)

    selected.to_csv(
        REPORTS_DIR / "gradcam_misclassified_case_selection.csv",
        index=False,
    )

    return selected


def load_image_for_model(image_path: str):
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)

    original_uint8 = tf.clip_by_value(image, 0, 255).numpy().astype("uint8")
    image_batch = tf.expand_dims(image, axis=0)

    return image_batch, original_uint8


def find_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    try:
        return model.get_layer("efficientnetb0")
    except ValueError as error:
        raise ValueError(
            "Could not find EfficientNetB0 layer named 'efficientnetb0' in the model."
        ) from error


def find_target_layer(feature_extractor: tf.keras.Model) -> tf.keras.layers.Layer:
    preferred_layer_names = [
        "top_activation",
        "block7a_project_conv",
        "block7a_expand_activation",
    ]

    for layer_name in preferred_layer_names:
        try:
            layer = feature_extractor.get_layer(layer_name)
            print(f"Using nested Grad-CAM layer: {layer.name}")
            return layer
        except ValueError:
            continue

    for layer in reversed(feature_extractor.layers):
        try:
            output_shape = layer.output.shape
            if len(output_shape) == 4:
                print(f"Using fallback Grad-CAM layer: {layer.name}")
                return layer
        except Exception:
            continue

    raise ValueError("Could not find a suitable 4D convolutional layer for Grad-CAM.")


def call_layer(layer, x):
    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def apply_layers_before_feature_extractor(
    model: tf.keras.Model,
    feature_extractor: tf.keras.Model,
    image_batch,
):
    feature_index = model.layers.index(feature_extractor)

    x = image_batch

    for layer in model.layers[1:feature_index]:
        x = call_layer(layer, x)

    return x, feature_index


def apply_classifier_after_feature_extractor(
    model: tf.keras.Model,
    feature_index: int,
    feature_output,
):
    x = feature_output

    for layer in model.layers[feature_index + 1:]:
        x = call_layer(layer, x)

    return x

# Grad-CAM heatmap: Shows which retinal regions influenced the model's prediction for a specific class

def make_gradcam_heatmap(
    model: tf.keras.Model,
    image_batch,
    class_index: int,
) -> np.ndarray:
    feature_extractor = find_feature_extractor(model)
    target_layer = find_target_layer(feature_extractor)

    image_for_feature_extractor, feature_index = apply_layers_before_feature_extractor(
        model,
        feature_extractor,
        image_batch,
    )

    conv_model = tf.keras.Model(
        inputs=feature_extractor.input,
        outputs=[target_layer.output, feature_extractor.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, feature_output = conv_model(
            image_for_feature_extractor,
            training=False,
        )

        predictions = apply_classifier_after_feature_extractor(
            model,
            feature_index,
            feature_output,
        )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError("Gradients are None. Grad-CAM could not be generated.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    max_value = tf.reduce_max(heatmap)

    if max_value == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    heatmap = heatmap / max_value
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], [IMG_SIZE, IMG_SIZE])
    heatmap = tf.squeeze(heatmap).numpy()

    return heatmap


def make_overlay(original_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.40):
    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap)[:, :, :3]
    colored_heatmap = np.uint8(255 * colored_heatmap)

    overlay = np.uint8(
        np.clip(
            (1 - alpha) * original_uint8 + alpha * colored_heatmap,
            0,
            255,
        )
    )

    return overlay


def save_single_gradcam_image(
    original_uint8: np.ndarray,
    overlay: np.ndarray,
    output_path: Path,
    title: str,
):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original_uint8)
    plt.title("Original image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM overlay")
    plt.axis("off")

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# Save 3-panel comparison: Original image | Baseline Grad-CAM | Glaucoma transfer Grad-CAM (side-by-side)

def save_comparison_gradcam_image(
    original_uint8: np.ndarray,
    baseline_overlay: np.ndarray,
    transfer_overlay: np.ndarray,
    output_path: Path,
    title: str,
):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_uint8)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(baseline_overlay)
    plt.title("APTOS baseline")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(transfer_overlay)
    plt.title("Glaucoma transfer")
    plt.axis("off")

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_gradcam_for_model(
    model: tf.keras.Model,
    image_path: str,
    predicted_class: int,
):
    image_batch, original_uint8 = load_image_for_model(image_path)
    heatmap = make_gradcam_heatmap(model, image_batch, predicted_class)
    overlay = make_overlay(original_uint8, heatmap)

    return original_uint8, heatmap, overlay


def safe_filename(text: str) -> str:
    text = str(text)
    replacements = {
        " ": "_",
        "/": "_",
        "\\": "_",
        ":": "_",
        ";": "_",
        ",": "_",
        "(": "",
        ")": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def generate_gradcams(selected_cases: pd.DataFrame):
    baseline_model = load_model(BASELINE_MODEL_PATH)
    transfer_model = load_model(TRANSFER_MODEL_PATH)

    appendix_rows = []

    for _, row in selected_cases.iterrows():
        case_number = int(row["case_number"])
        id_code = str(row["id_code"])
        image_path = str(row["image_path"])

        true_label = int(row["true_label"])
        baseline_pred_label = int(row["baseline_pred_label"])
        transfer_pred_label = int(row["transfer_pred_label"])

        true_class_name = str(row["true_class_name"])
        baseline_pred_name = str(row["baseline_pred_class_name"])
        transfer_pred_name = str(row["transfer_pred_class_name"])
        case_group = str(row["case_group"])

        base_name = safe_filename(
            f"{case_number:02d}_{id_code}_{case_group}_true_{true_label}_{true_class_name}"
        )

        print(f"\nGenerating Grad-CAM for case {case_number}: {id_code}")
        print(f"True class: {true_class_name}")
        print(f"Baseline prediction: {baseline_pred_name}")
        print(f"Transfer prediction: {transfer_pred_name}")

        original_uint8, _, baseline_overlay = create_gradcam_for_model(
            baseline_model,
            image_path,
            baseline_pred_label,
        )

        _, _, transfer_overlay = create_gradcam_for_model(
            transfer_model,
            image_path,
            transfer_pred_label,
        )

        baseline_output_path = BASELINE_GRADCAM_DIR / (
            f"{base_name}_baseline_pred_{baseline_pred_label}_{safe_filename(baseline_pred_name)}.png"
        )

        transfer_output_path = TRANSFER_GRADCAM_DIR / (
            f"{base_name}_transfer_pred_{transfer_pred_label}_{safe_filename(transfer_pred_name)}.png"
        )

        comparison_output_path = COMPARISON_GRADCAM_DIR / (
            f"{base_name}_baseline_vs_transfer.png"
        )

        baseline_title = (
            f"Baseline | True: {true_class_name} | Pred: {baseline_pred_name}"
        )

        transfer_title = (
            f"Transfer | True: {true_class_name} | Pred: {transfer_pred_name}"
        )

        comparison_title = (
            f"{case_group} | True: {true_class_name} | "
            f"Baseline: {baseline_pred_name} | Transfer: {transfer_pred_name}"
        )

        save_single_gradcam_image(
            original_uint8,
            baseline_overlay,
            baseline_output_path,
            baseline_title,
        )

        save_single_gradcam_image(
            original_uint8,
            transfer_overlay,
            transfer_output_path,
            transfer_title,
        )

        save_comparison_gradcam_image(
            original_uint8,
            baseline_overlay,
            transfer_overlay,
            comparison_output_path,
            comparison_title,
        )

        if case_group == "baseline_wrong_transfer_right":
            explanation = (
                "This case is useful because the baseline model was wrong while the "
                "glaucoma-transfer model was right."
            )
        elif case_group == "transfer_wrong_baseline_right":
            explanation = (
                "This case is useful because the glaucoma-transfer model was wrong while "
                "the baseline model was right."
            )
        elif case_group == "both_models_wrong":
            explanation = (
                "This case is useful because both models failed, showing a difficult or "
                "borderline image."
            )
        else:
            explanation = (
                "This case is useful because it is a high-confidence transfer-model error."
            )

        appendix_rows.append(
            {
                "case_number": case_number,
                "id_code": id_code,
                "case_group": case_group,
                "true_class": true_class_name,
                "baseline_prediction": baseline_pred_name,
                "transfer_prediction": transfer_pred_name,
                "baseline_gradcam_file": str(baseline_output_path),
                "transfer_gradcam_file": str(transfer_output_path),
                "comparison_gradcam_file": str(comparison_output_path),
                "appendix_explanation": explanation,
            }
        )

    appendix_df = pd.DataFrame(appendix_rows)
    appendix_df.to_csv(REPORTS_DIR / "gradcam_appendix_image_list.csv", index=False)

    return appendix_df


def create_gradcam_summary(selected_cases: pd.DataFrame, appendix_df: pd.DataFrame):
    summary_rows = []

    for case_group, group_df in selected_cases.groupby("case_group"):
        summary_rows.append(
            {
                "case_group": case_group,
                "number_of_cases": len(group_df),
                "reason_for_selection": {
                    "baseline_wrong_transfer_right": "Baseline failed but transfer succeeded.",
                    "transfer_wrong_baseline_right": "Transfer failed but baseline succeeded.",
                    "both_models_wrong": "Both models failed.",
                    "transfer_wrong_high_confidence": "Transfer made a high-confidence wrong prediction.",
                }.get(case_group, "Selected error group."),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REPORTS_DIR / "gradcam_error_focus_summary.csv", index=False)


def main():

    # Load predictions from both models, select error cases, generate Grad-CAM images, create appendix list for dissertation.

    merged_predictions = load_predictions()
    selected_cases = select_cases_for_gradcam(merged_predictions)

    print("\nSelected Grad-CAM cases:")
    print(
        selected_cases[
            [
                "case_number",
                "id_code",
                "case_group",
                "true_class_name",
                "baseline_pred_class_name",
                "transfer_pred_class_name",
            ]
        ].to_string(index=False)
    )

    appendix_df = generate_gradcams(selected_cases)
    create_gradcam_summary(selected_cases, appendix_df)

    print("\nGrad-CAM generation completed successfully.")
    print(f"Baseline Grad-CAM folder: {BASELINE_GRADCAM_DIR}")
    print(f"Transfer Grad-CAM folder: {TRANSFER_GRADCAM_DIR}")
    print(f"Comparison Grad-CAM folder: {COMPARISON_GRADCAM_DIR}")
    print(f"Appendix image list saved at: {REPORTS_DIR / 'gradcam_appendix_image_list.csv'}")


if __name__ == "__main__":
    main()