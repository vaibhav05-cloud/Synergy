"""
config.py

Central configuration file for the entire project.
Edit values here instead of changing them across multiple files.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError(
        "GitHub token not found!\n"
        "Create a .env file in the ML folder and add:\n"
        "GITHUB_TOKEN=your_github_personal_access_token"
    )

# ==========================================================
# Project Structure
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

COMMITS_DIR = RAW_DATA_DIR / "commits"
ISSUES_DIR = RAW_DATA_DIR / "issues"
PRS_DIR = RAW_DATA_DIR / "prs"
FILES_DIR = RAW_DATA_DIR / "files"

OUTPUT_DIR = PROJECT_ROOT / "output"

# ==========================================================
# Create directories automatically
# ==========================================================

for folder in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    COMMITS_DIR,
    ISSUES_DIR,
    PRS_DIR,
    FILES_DIR,
    OUTPUT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================================
# File Paths
# ==========================================================

REPOSITORIES_FILE = RAW_DATA_DIR / "repositories.json"

FINAL_DATASET = PROCESSED_DATA_DIR / "final_dataset.csv"
TRAIN_DATASET = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_DATASET = PROCESSED_DATA_DIR / "validation.csv"
TEST_DATASET = PROCESSED_DATA_DIR / "test.csv"

# ==========================================================
# GitHub API Configuration
# ==========================================================

GITHUB_API_URL = "https://api.github.com"

# Maximum items per API request
PER_PAGE = 100

# Request timeout (seconds)
REQUEST_TIMEOUT = 30

# Number of retries for failed requests
MAX_RETRIES = 3

# Delay between retries (seconds)
RETRY_DELAY = 2

# ==========================================================
# Dataset Collection Limits (Hackathon Optimized)
# ==========================================================

# Total repositories
MAX_REPOSITORIES = 800

# Per repository limits
MAX_COMMITS_PER_REPO = 30
MAX_ISSUES_PER_REPO = 20
MAX_PRS_PER_REPO = 20
MAX_FILES_PER_REPO = 100

# ==========================================================
# Supported Programming Languages
# ==========================================================

SUPPORTED_LANGUAGES = [
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "C",
    "C++",
    "C#",
    "Go",
    "Rust",
    "PHP",
    "Ruby",
    "Kotlin",
    "Swift",
]

# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = "INFO"

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_SEED = 42