# 01_check_datasets.py - Validate glaucoma and APTOS datasets, save checked CSVs for all scripts.

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
GLAUCOMA_DIR = DATA_DIR / "glaucoma"
APTOS_DIR = DATA_DIR / "aptos"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


GLAUCOMA_SPLITS = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}

GLAUCOMA_CLASSES = {
    "NRG": 0,
    "RG": 1,
}

APTOS_SPLITS = {
    "train": {
        "csv": "train_1.csv",
        "image_folder": "train_images",
        "output": "aptos_train_checked.csv",
    },
    "validation": {
        "csv": "valid.csv",
        "image_folder": "val_images",
        "output": "aptos_validation_checked.csv",
    },
    "test": {
        "csv": "test.csv",
        "image_folder": "test_images",
        "output": "aptos_test_checked.csv",
    },
}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def require_folder(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Expected a folder but found something else: {path}")


def require_file(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Expected a file but found something else: {path}")


def collect_glaucoma_split(split_name: str, split_folder_name: str) -> pd.DataFrame:
    split_dir = GLAUCOMA_DIR / split_folder_name

    require_folder(
        split_dir,
        f"Missing glaucoma {split_name} folder",
    )

    rows = []

    for class_name, label_binary in GLAUCOMA_CLASSES.items():
        class_dir = split_dir / class_name

        require_folder(
            class_dir,
            f"Missing glaucoma class folder for {split_name}/{class_name}",
        )

        # Only accept actual image files, NOT folders (prevents ...\NRG or ...\RG from being treated as images)

        image_files = sorted(
            [
                path
                for path in class_dir.rglob("*")
                if is_image_file(path)
            ]
        )

        for image_path in image_files:
            rows.append(
                {
                    "dataset": "EyePACS-AIROGS-Light-V2 Glaucoma Dataset",
                    "split": split_name,
                    "id": image_path.stem,
                    "file_name": image_path.name,
                    "label": class_name,
                    "label_binary": label_binary,
                    "folder": split_folder_name,
                    "source_dataset": "EyePACS-AIROGS-Light-V2",
                    "image_path": str(image_path),
                    "image_exists": image_path.exists() and image_path.is_file(),
                    "image_extension": image_path.suffix.lower(),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No valid glaucoma image files were found in: {split_dir}. "
            f"Expected image files inside NRG and RG folders."
        )

    invalid_paths = []

    for image_path in df["image_path"]:
        path = Path(image_path)

        if not is_image_file(path):
            invalid_paths.append(image_path)

    if invalid_paths:
        invalid_report = pd.DataFrame({"invalid_image_path": invalid_paths})
        invalid_report.to_csv(
            REPORTS_DIR / f"invalid_glaucoma_{split_name}_paths.csv",
            index=False,
        )

        raise ValueError(
            f"Invalid glaucoma paths found in {split_name}. "
            f"Some paths are not real image files. "
            f"See: outputs/reports/invalid_glaucoma_{split_name}_paths.csv"
        )

    return df


def check_glaucoma_dataset() -> list:
    print("\nChecking EyePACS-AIROGS-Light-V2 glaucoma dataset...")

    require_folder(
        GLAUCOMA_DIR,
        "Missing glaucoma dataset folder",
    )

    summary_rows = []

    for split_name, split_folder_name in GLAUCOMA_SPLITS.items():
        df = collect_glaucoma_split(split_name, split_folder_name)

        output_path = REPORTS_DIR / f"glaucoma_{split_name}_checked.csv"
        df.to_csv(output_path, index=False)

        class_counts = df["label"].value_counts().to_dict()

        summary_rows.append(
            {
                "dataset": "EyePACS-AIROGS-Light-V2 Glaucoma Dataset",
                "split": split_name,
                "rows": len(df),
                "missing_images": int((df["image_exists"] == False).sum()),
                "class_NRG": int(class_counts.get("NRG", 0)),
                "class_RG": int(class_counts.get("RG", 0)),
                "class_0": None,
                "class_1": None,
                "class_2": None,
                "class_3": None,
                "class_4": None,
            }
        )

    return summary_rows


def normalise_aptos_columns(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    df = df.copy()

    if "id_code" not in df.columns:
        possible_id_columns = [
            "id",
            "image",
            "image_id",
            "filename",
            "file_name",
            "name",
        ]

        found_id_column = None

        for col in possible_id_columns:
            if col in df.columns:
                found_id_column = col
                break

        if found_id_column is None:
            raise ValueError(
                f"{csv_path.name} must contain an id_code column. "
                f"Available columns are: {list(df.columns)}"
            )

        df = df.rename(columns={found_id_column: "id_code"})

    if "diagnosis" not in df.columns:
        possible_label_columns = [
            "label",
            "class",
            "target",
            "dr",
            "grade",
        ]

        found_label_column = None

        for col in possible_label_columns:
            if col in df.columns:
                found_label_column = col
                break

        if found_label_column is None:
            raise ValueError(
                f"{csv_path.name} must contain a diagnosis column. "
                f"Available columns are: {list(df.columns)}"
            )

        df = df.rename(columns={found_label_column: "diagnosis"})

    df["id_code"] = df["id_code"].astype(str)
    df["diagnosis"] = df["diagnosis"].astype(int)

    return df


def find_aptos_image(image_folder: Path, id_code: str) -> Path | None:
    id_code = str(id_code)

    for extension in IMAGE_EXTENSIONS:
        candidate = image_folder / f"{id_code}{extension}"

        if candidate.exists() and candidate.is_file():
            return candidate

    recursive_matches = sorted(
        [
            path
            for path in image_folder.rglob("*")
            if is_image_file(path) and path.stem == id_code
        ]
    )

    if recursive_matches:
        return recursive_matches[0]

    return None


def check_aptos_split(split_name: str, split_info: dict) -> pd.DataFrame:
    csv_path = APTOS_DIR / split_info["csv"]
    image_folder = APTOS_DIR / split_info["image_folder"]

    require_file(
        csv_path,
        f"Missing APTOS {split_name} CSV file",
    )

    require_folder(
        image_folder,
        f"Missing APTOS {split_name} image folder",
    )

    df = pd.read_csv(csv_path)
    df = normalise_aptos_columns(df, csv_path)

    rows = []

    for _, row in df.iterrows():
        id_code = str(row["id_code"])
        diagnosis = int(row["diagnosis"])

        image_path = find_aptos_image(image_folder, id_code)

        if image_path is None:
            image_path_string = str(image_folder / f"{id_code}.png")
            image_exists = False
            image_extension = None
            file_name = None
        else:
            image_path_string = str(image_path)
            image_exists = image_path.exists() and image_path.is_file()
            image_extension = image_path.suffix.lower()
            file_name = image_path.name

        rows.append(
            {
                "dataset": "APTOS 2019 Blindness Detection Dataset",
                "split": split_name,
                "id_code": id_code,
                "file_name": file_name,
                "diagnosis": diagnosis,
                "image_path": image_path_string,
                "image_exists": image_exists,
                "image_extension": image_extension,
            }
        )

    checked_df = pd.DataFrame(rows)

    # APTOS labels: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative.

    invalid_labels = sorted(
        checked_df.loc[
            ~checked_df["diagnosis"].isin([0, 1, 2, 3, 4]),
            "diagnosis",
        ].unique()
    )

    if invalid_labels:
        raise ValueError(
            f"Invalid APTOS labels found in {csv_path.name}: {invalid_labels}. "
            f"Expected labels are 0, 1, 2, 3, 4."
        )

    invalid_paths = []

    for _, row in checked_df.iterrows():
        path = Path(row["image_path"])

        if row["image_exists"] and not is_image_file(path):
            invalid_paths.append(row["image_path"])

    if invalid_paths:
        invalid_report = pd.DataFrame({"invalid_image_path": invalid_paths})
        invalid_report.to_csv(
            REPORTS_DIR / f"invalid_aptos_{split_name}_paths.csv",
            index=False,
        )

        raise ValueError(
            f"Invalid APTOS paths found in {split_name}. "
            f"Some paths are not real image files. "
            f"See: outputs/reports/invalid_aptos_{split_name}_paths.csv"
        )

    return checked_df


def check_aptos_dataset() -> list:
    print("\nChecking APTOS 2019 Blindness Detection dataset...")

    require_folder(
        APTOS_DIR,
        "Missing APTOS dataset folder",
    )

    summary_rows = []

    for split_name, split_info in APTOS_SPLITS.items():
        df = check_aptos_split(split_name, split_info)

        output_path = REPORTS_DIR / split_info["output"]
        df.to_csv(output_path, index=False)

        class_counts = df["diagnosis"].value_counts().sort_index().to_dict()

        summary_rows.append(
            {
                "dataset": "APTOS 2019 Blindness Detection Dataset",
                "split": split_name,
                "rows": len(df),
                "missing_images": int((df["image_exists"] == False).sum()),
                "class_NRG": None,
                "class_RG": None,
                "class_0": int(class_counts.get(0, 0)),
                "class_1": int(class_counts.get(1, 0)),
                "class_2": int(class_counts.get(2, 0)),
                "class_3": int(class_counts.get(3, 0)),
                "class_4": int(class_counts.get(4, 0)),
            }
        )

    return summary_rows


def save_missing_image_reports() -> None:
    checked_files = [
        REPORTS_DIR / "glaucoma_train_checked.csv",
        REPORTS_DIR / "glaucoma_validation_checked.csv",
        REPORTS_DIR / "glaucoma_test_checked.csv",
        REPORTS_DIR / "aptos_train_checked.csv",
        REPORTS_DIR / "aptos_validation_checked.csv",
        REPORTS_DIR / "aptos_test_checked.csv",
    ]

    all_missing_rows = []

    for checked_file in checked_files:
        if not checked_file.exists():
            continue

        df = pd.read_csv(checked_file)

        if "image_exists" not in df.columns:
            continue

        missing_df = df[df["image_exists"] == False].copy()

        if not missing_df.empty:
            missing_df["checked_file"] = checked_file.name
            all_missing_rows.append(missing_df)

    if all_missing_rows:
        missing_all = pd.concat(all_missing_rows, ignore_index=True)
        missing_all.to_csv(
            REPORTS_DIR / "missing_images_all.csv",
            index=False,
        )


def validate_checked_outputs() -> None:
    required_outputs = [
        REPORTS_DIR / "glaucoma_train_checked.csv",
        REPORTS_DIR / "glaucoma_validation_checked.csv",
        REPORTS_DIR / "glaucoma_test_checked.csv",
        REPORTS_DIR / "aptos_train_checked.csv",
        REPORTS_DIR / "aptos_validation_checked.csv",
        REPORTS_DIR / "aptos_test_checked.csv",
    ]

    for output_path in required_outputs:
        if not output_path.exists():
            raise FileNotFoundError(f"Required checked output was not created: {output_path}")

        df = pd.read_csv(output_path)

        if df.empty:
            raise ValueError(f"Checked output is empty: {output_path}")

        if "image_path" not in df.columns:
            raise ValueError(f"Checked output has no image_path column: {output_path}")

        invalid_paths = []

        for image_path in df["image_path"].astype(str):
            path = Path(image_path)

            if path.exists():
                if not is_image_file(path):
                    invalid_paths.append(image_path)

        if invalid_paths:
            invalid_report = pd.DataFrame({"invalid_image_path": invalid_paths})
            invalid_report.to_csv(
                REPORTS_DIR / f"invalid_paths_in_{output_path.stem}.csv",
                index=False,
            )

            raise ValueError(
                f"{output_path.name} contains paths that are not image files. "
                f"See outputs/reports/invalid_paths_in_{output_path.stem}.csv"
            )


def main():

    # Check both datasets and save validated CSV files for all subsequent scripts.

    summary_rows = []

    summary_rows.extend(check_glaucoma_dataset())
    summary_rows.extend(check_aptos_dataset())

    summary_df = pd.DataFrame(summary_rows)

    summary_path = REPORTS_DIR / "dataset_check_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    save_missing_image_reports()
    validate_checked_outputs()

    print("\nDataset check summary")
    print("-" * 80)
    print(summary_df.to_string(index=False))

    missing_total = int(summary_df["missing_images"].sum())

    if missing_total > 0:
        print("\nWarning: Some images are missing.")
        print(f"Missing image report saved in: {REPORTS_DIR / 'missing_images_all.csv'}")
    else:
        print("\nDataset checking completed successfully.")

    print(f"Checked CSV files saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()