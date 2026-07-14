"""
run_pipeline.py

Runs the ENTIRE dataset pipeline in one command:

1. Deletes old outputs (dataset/raw, dataset/processed, output/)
2. fetch_repositories.py
3. fetch_commits.py
4. fetch_issues.py
5. fetch_prs.py
6. fetch_files.py
7. build_dataset.py

Usage:
    python run_pipeline.py

Place this file in the SAME folder as your other scripts
(config.py, fetch_repositories.py, fetch_commits.py, fetch_issues.py,
fetch_prs.py, fetch_files.py, build_dataset.py, github_client.py).
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    DATASET_DIR,
)

# ==========================================================
# Pipeline Steps (in order)
# ==========================================================

PIPELINE_STEPS = [
    "fetch_repositories.py",
    "fetch_commits.py",
    "fetch_issues.py",
    "fetch_prs.py",
    "fetch_files.py",
    "build_dataset.py",
]

PROJECT_ROOT = Path(__file__).resolve().parent


# ==========================================================
# Step 1: Clean old outputs
# ==========================================================

def clean_previous_run():

    print("=" * 70)
    print("STEP 0 : Cleaning previous run")
    print("=" * 70)

    folders_to_clean = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
    ]

    for folder in folders_to_clean:

        if folder.exists():
            print(f"Deleting -> {folder}")
            shutil.rmtree(folder)
        else:
            print(f"Skip (not found) -> {folder}")

    # Recreate empty structure so downstream scripts don't error
    for folder in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        RAW_DATA_DIR / "commits",
        RAW_DATA_DIR / "issues",
        RAW_DATA_DIR / "prs",
        RAW_DATA_DIR / "files",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    print("\n✅ Old outputs cleaned. Starting fresh run.\n")


# ==========================================================
# Step 2: Run each script as a subprocess
# ==========================================================

def run_script(script_name):

    script_path = PROJECT_ROOT / script_name

    if not script_path.exists():
        print(f"\n❌ ERROR: {script_name} not found at {script_path}")
        sys.exit(1)

    print("=" * 70)
    print(f"RUNNING : {script_name}")
    print("=" * 70)

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )

    elapsed = round(time.time() - start, 2)

    if result.returncode != 0:

        print(
            f"\n❌ {script_name} FAILED "
            f"(exit code {result.returncode}). Stopping pipeline."
        )
        sys.exit(1)

    print(f"\n✅ {script_name} completed in {elapsed}s\n")


# ==========================================================
# Main
# ==========================================================

def main():

    pipeline_start = time.time()

    print("\n" + "#" * 70)
    print("#  FULL DATASET PIPELINE - START")
    print("#" * 70 + "\n")

    clean_previous_run()

    for step in PIPELINE_STEPS:
        run_script(step)

    total_elapsed = round(time.time() - pipeline_start, 2)

    print("#" * 70)
    print(f"#  PIPELINE COMPLETE in {total_elapsed}s")
    print(f"#  Final dataset  : {PROCESSED_DATA_DIR / 'final_dataset.csv'}")
    print(f"#  ML-ready data  : {PROCESSED_DATA_DIR / 'ml_ready_dataset.csv'}")
    print(f"#  Train/Val/Test : {PROCESSED_DATA_DIR}")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()