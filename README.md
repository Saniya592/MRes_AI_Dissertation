# Cross-Disease Retinal Feature Transfer for diabetic retinopathy Prediction Using Glaucoma Fundus Images.

1. Stage 1 trains an EfficientNetB0 source-domain model on the EyePACS-AIROGS-Light-V2 Glaucoma Dataset.
2. Stage 2 trains a direct baseline model on the APTOS 2019 Blindness Detection Dataset.
3. Stage 3 transfers the glaucoma-trained EfficientNetB0 feature extractor to the APTOS 2019 diabetic-retinopathy task.
4. stage 4 Train with ImageNet weights on APTOS

## Requirements fulfilled

- Exactly 50 epochs for all final model training.
- EarlyStopping removed.
- ModelCheckpoint retained.
- ReduceLROnPlateau retained.
- Fixed random seed used.
- Python, NumPy, TensorFlow, dataset shuffling, augmentation layers, and dropout layers seeded.
- TensorFlow deterministic settings enabled where possible.
- Class weights applied for APTOS 2019 imbalance.
- Stage 2 and Stage 3 compared on the same APTOS test set.
- Five-class diabetic-retinopathy grading and binary referable DR classification reported.
- Grad-CAM explainability included.

## Required data structure

Datasets are not included because they are large public image datasets. The dataset is extracted from kaggel.


```bash
pip install -r requirements.txt
```

## Run order

```bash
python src/01_check_datasets.py
python src/02_train_glaucoma_model.py
python src/03_train_aptos_baseline.py
python src/04_transfer_glaucoma_to_aptos.py
python src/04b_transfer_imagenet_to_aptos.py
python src/05_evaluate_models.py
python src/06_gradcam_visualisation.py
```

Or:

```bash
python run_all.py
```

## Main outputs

Reports:

```text
outputs/reports/glaucoma_summary_metrics.csv
outputs/reports/aptos_baseline_summary_metrics.csv
outputs/reports/imagenet_transfer_summary_metrics.csv
outputs/reports/glaucoma_transfer_summary_metrics.csv
outputs/reports/model_comparison_summary.csv
outputs/reports/stage2_stage3_side_by_side_comparison.csv
outputs/reports/imagenet_vs_glaucoma_transfer_side_by_side.csv
outputs/reports/glaucoma_transfer_classification_report.csv
outputs/reports/aptos_baseline_classification_report.csv
outputs/reports/imagenet_transfer_per_class_metrics.csv
```

Figures:

```text
outputs/figures/glaucoma_training_auc.png
outputs/figures/aptos_baseline_confusion_matrix.png
outputs/figures/aptos_baseline_referable_dr_confusion_matrix.png
outputs/figures/glaucoma_transfer_confusion_matrix.png
outputs/figures/glaucoma_transfer_referable_dr_confusion_matrix.png
outputs/figures/imagenet_transfer_confusion_matrix.png
outputs/figures/stage2_stage3_metric_comparison.png
outputs/figures/imagenet_vs_glaucoma_transfer_metric_comparison.png
outputs/figures/all_model_metric_comparison.png
```

Grad-CAM:

```text
outputs/gradcam/misclassified_cases/aptos_baseline/
outputs/gradcam/misclassified_cases/glaucoma_to_aptos_transfer/
outputs/gradcam/misclassified_case_comparisons/
```
