import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "src/01_check_datasets.py",
    "src/02_train_glaucoma_model.py",
    "src/03_train_aptos_baseline.py",
    "src/04_transfer_glaucoma_to_aptos.py",
    "src/04b_transfer_imagenet_to_aptos.py",
    "src/05_evaluate_models.py",
    "src/06_gradcam_visualisation.py",
]


def run_script(script_path: str):
    full_path = PROJECT_ROOT / script_path

    if not full_path.exists():
        raise FileNotFoundError(f"Script not found: {full_path}")

    print("\n" + "=" * 80)
    print(f"Running: {script_path}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script_path} failed with return code {result.returncode}")


def main():
    for script in SCRIPTS:
        run_script(script)

    print("\nFull project pipeline completed successfully.")


if __name__ == "__main__":
    main()