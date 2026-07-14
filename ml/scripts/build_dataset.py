"""
build_dataset.py

Build ML-ready dataset from GitHub repository metadata.

Pipeline

Repositories
Commits
Issues
PRs
Files

↓

Feature Engineering

↓

Final Dataset

↓

Train / Validation / Test Split
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FINAL_DATASET,
    TRAIN_DATASET,
    VALIDATION_DATASET,
    TEST_DATASET,
    RANDOM_SEED,
)

# ==========================================================
# Logger
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("DatasetBuilder")

# ==========================================================
# Paths
# ==========================================================

REPOSITORIES_FILE = RAW_DATA_DIR / "repositories.json"

COMMITS_DIR = RAW_DATA_DIR / "commits"

ISSUES_DIR = RAW_DATA_DIR / "issues"

PRS_DIR = RAW_DATA_DIR / "prs"

FILES_DIR = RAW_DATA_DIR / "files"

# ==========================================================
# Helpers
# ==========================================================


def load_json(path: Path):

    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_folder(folder: Path):

    data = {}

    if not folder.exists():
        return data

    for file in folder.glob("*.json"):

        data[file.stem] = load_json(file)

    return data


def save_dataframe(df, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )


# ==========================================================
# Utility Functions
# ==========================================================


def safe_ratio(a, b):

    if b == 0:
        return 0

    return round(a / b, 4)


def safe_mean(values):

    values = [
        x
        for x in values
        if isinstance(x, (int, float))
    ]

    if len(values) == 0:
        return 0

    return round(float(np.mean(values)), 4)


def parse_datetime(value):

    if not value:
        return None

    try:

        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

    except Exception:

        return None


def repository_age(created_at):

    created = parse_datetime(created_at)

    if created is None:
        return 0

    return (
        datetime.now(timezone.utc) - created
    ).days


def last_updated(updated_at):

    updated = parse_datetime(updated_at)

    if updated is None:
        return 0

    return (
        datetime.now(timezone.utc) - updated
    ).days


# ==========================================================
# Dataset Loader
# ==========================================================


def load_processed_data():

    logger.info("Loading repositories...")

    repositories = load_json(
        REPOSITORIES_FILE
    )

    logger.info("Loading commits...")

    commits = load_folder(
        COMMITS_DIR
    )

    logger.info("Loading issues...")

    issues = load_folder(
        ISSUES_DIR
    )

    logger.info("Loading pull requests...")

    prs = load_folder(
        PRS_DIR
    )

    logger.info("Loading repository files...")

    files = load_folder(
        FILES_DIR
    )

    logger.info(
        f"{len(repositories)} repositories loaded."
    )

    return (
        repositories,
        commits,
        issues,
        prs,
        files,
    )


# ==========================================================
# Repository Initialization
# ==========================================================


def repository_key(repo):

    return (
        f"{repo['owner']}_{repo['name']}"
    )


def initialize_dataset(repositories):

    logger.info(
        "Initializing dataset..."
    )

    dataset = {}

    for repo in repositories:

        key = repository_key(repo)

        stars = repo.get("stars", 0)

        forks = repo.get("forks", 0)

        watchers = repo.get("watchers", 0)

        popularity = (
            stars * 0.60
            +
            forks * 0.25
            +
            watchers * 0.15
        )

        dataset[key] = {

            # Repository Identity

            "repo_key": key,

            "owner": repo.get("owner"),

            "repository": repo.get("name"),

            "language": repo.get(
                "language",
                "Unknown",
            ),

            # Metadata

            "stars": stars,

            "forks": forks,

            "watchers": watchers,

            "size": repo.get(
                "size",
                0,
            ),

            "open_issues": repo.get(
                "open_issues",
                0,
            ),

            "repository_age_days":
                repository_age(
                    repo.get("created_at")
                ),

            "last_update_days":
                last_updated(
                    repo.get("updated_at")
                ),

            "has_license":
                int(
                    repo.get("license")
                    is not None
                ),

            "has_description":
                int(
                    bool(
                        repo.get(
                            "description"
                        )
                    )
                ),

            "popularity_score":
                round(
                    popularity,
                    2,
                ),

            # Commit Features

            "total_commits": 0,

            "bug_fix_ratio": 0,

            "feature_commit_ratio": 0,

            "refactor_commit_ratio": 0,

            "documentation_commit_ratio": 0,

            "average_commit_length": 0,

            "average_commit_words": 0,

            "author_diversity": 0,

            "commit_frequency": 0,

            "days_since_last_commit": 0,

            # Issue Features

            "total_issues": 0,

            "closed_issue_ratio": 0,

            "bug_issue_ratio": 0,

            "average_issue_comments": 0,

            "stale_issue_ratio": 0,

            # PR Features

            "total_prs": 0,

            "merged_pr_ratio": 0,

            "average_changed_files": 0,

            "average_code_churn": 0,

            # File Features

            "total_files": 0,

            "source_code_ratio": 0,

            "documentation_ratio": 0,

            "config_ratio": 0,

            "test_ratio": 0,

            "binary_ratio": 0,

            "has_ci_cd": 0,

            "has_dockerfile": 0,

            "has_contributing_guide": 0,

            "dependency_file_count": 0,

            "max_directory_depth": 0,

            # Final Scores

            "repository_score": 0,

            "commit_activity_score": 0,

            "issue_activity_score": 0,

            "pr_activity_score": 0,

            "file_complexity_score": 0,

            "repository_quality_index": 0,

            "deployment_ready": 0,

        }

    logger.info(
        f"{len(dataset)} repositories initialized."
    )

    return dataset

# ==========================================================
# Commit Feature Engineering
# ==========================================================

BUG_KEYWORDS = {
    "fix",
    "bug",
    "error",
    "issue",
    "hotfix",
    "patch",
    "resolve",
}

FEATURE_KEYWORDS = {
    "add",
    "feature",
    "implement",
    "create",
    "support",
}

REFACTOR_KEYWORDS = {
    "refactor",
    "cleanup",
    "optimize",
    "improve",
}

DOC_KEYWORDS = {
    "doc",
    "docs",
    "readme",
    "documentation",
}


def aggregate_commits(dataset, commits):

    logger.info("Aggregating commits...")

    for repo_key, repo_commits in commits.items():

        if repo_key not in dataset:
            continue

        repository = dataset[repo_key]

        total = len(repo_commits)

        if total == 0:
            continue

        authors = set()

        message_lengths = []

        word_counts = []

        bug = 0
        feature = 0
        refactor = 0
        docs = 0

        weekend = 0
        night = 0

        commit_dates = []

        for commit in repo_commits:

            author = commit.get("author")

            if author:
                authors.add(author)

            message = (
                commit.get("message") or ""
            )

            message = message.strip()

            message_lengths.append(
                len(message)
            )

            words = message.split()

            word_counts.append(
                len(words)
            )

            lower = message.lower()

            if any(
                keyword in lower
                for keyword in BUG_KEYWORDS
            ):
                bug += 1

            if any(
                keyword in lower
                for keyword in FEATURE_KEYWORDS
            ):
                feature += 1

            if any(
                keyword in lower
                for keyword in REFACTOR_KEYWORDS
            ):
                refactor += 1

            if any(
                keyword in lower
                for keyword in DOC_KEYWORDS
            ):
                docs += 1

            dt = parse_datetime(
                commit.get("date")
            )

            if dt:

                commit_dates.append(dt)

                if dt.weekday() >= 5:
                    weekend += 1

                if (
                    dt.hour >= 22
                    or
                    dt.hour <= 5
                ):
                    night += 1

        repository["total_commits"] = total

        repository["author_diversity"] = len(
            authors
        )

        repository[
            "average_commit_length"
        ] = safe_mean(
            message_lengths
        )

        repository[
            "average_commit_words"
        ] = safe_mean(
            word_counts
        )

        repository[
            "bug_fix_ratio"
        ] = safe_ratio(
            bug,
            total,
        )

        repository[
            "feature_commit_ratio"
        ] = safe_ratio(
            feature,
            total,
        )

        repository[
            "refactor_commit_ratio"
        ] = safe_ratio(
            refactor,
            total,
        )

        repository[
            "documentation_commit_ratio"
        ] = safe_ratio(
            docs,
            total,
        )

        repository[
            "weekend_commit_ratio"
        ] = safe_ratio(
            weekend,
            total,
        )

        repository[
            "night_commit_ratio"
        ] = safe_ratio(
            night,
            total,
        )

        # commits per day since repo creation -> maturity/activity signal
        repository[
            "commit_frequency"
        ] = safe_ratio(
            total,
            repository.get(
                "repository_age_days", 0
            ),
        )

        # recency of last observed commit (in this sample)
        if commit_dates:

            latest_commit = max(commit_dates)

            repository[
                "days_since_last_commit"
            ] = (
                datetime.now(timezone.utc)
                - latest_commit
            ).days

        # -------------------------------
        # Commit Activity Score
        # -------------------------------

        score = (

            repository[
                "bug_fix_ratio"
            ] * 20

            +

            repository[
                "feature_commit_ratio"
            ] * 20

            +

            repository[
                "refactor_commit_ratio"
            ] * 15

            +

            repository[
                "documentation_commit_ratio"
            ] * 10

            +

            min(
                repository[
                    "author_diversity"
                ] * 3,
                15,
            )

            +

            min(
                repository[
                    "average_commit_words"
                ],
                20,
            )

        )

        repository[
            "commit_activity_score"
        ] = round(
            score,
            2,
        )

    logger.info(
        "Commit aggregation complete."
    )

    return dataset

# ==========================================================
# Issue Feature Engineering
# ==========================================================

def aggregate_issues(dataset, issues):

    logger.info("Aggregating issues...")

    for repo_key, repo_issues in issues.items():

        if repo_key not in dataset:
            continue

        repository = dataset[repo_key]

        total = len(repo_issues)

        if total == 0:
            continue

        closed = 0
        bug = 0
        stale = 0
        open_count = 0

        comments = []

        resolution_days = []

        for issue in repo_issues:

            if (
                issue.get("state", "")
                .lower()
                == "closed"
            ):
                closed += 1

            else:

                open_count += 1

                created_open = parse_datetime(
                    issue.get("created_at")
                )

                if (
                    created_open
                    and
                    (
                        datetime.now(timezone.utc)
                        - created_open
                    ).days > 90
                ):
                    stale += 1

            title = (
                issue.get("title")
                or ""
            ).lower()

            body = (
                issue.get("body")
                or ""
            ).lower()

            text = title + " " + body

            if any(
                word in text
                for word in BUG_KEYWORDS
            ):
                bug += 1

            comments.append(
                issue.get(
                    "comments",
                    0,
                )
            )

            created = parse_datetime(
                issue.get(
                    "created_at"
                )
            )

            closed_at = parse_datetime(
                issue.get(
                    "closed_at"
                )
            )

            if (
                created
                and
                closed_at
            ):

                resolution_days.append(

                    (
                        closed_at
                        -
                        created
                    ).days

                )

        repository[
            "total_issues"
        ] = total

        repository[
            "closed_issue_ratio"
        ] = safe_ratio(
            closed,
            total,
        )

        repository[
            "bug_issue_ratio"
        ] = safe_ratio(
            bug,
            total,
        )

        repository[
            "average_issue_comments"
        ] = safe_mean(
            comments
        )

        repository[
            "average_issue_resolution_days"
        ] = safe_mean(
            resolution_days
        )

        repository[
            "stale_issue_ratio"
        ] = safe_ratio(
            stale,
            open_count,
        )

        score = (

            repository[
                "closed_issue_ratio"
            ] * 40

            +

            (
                1
                -
                repository[
                    "bug_issue_ratio"
                ]
            ) * 30

            +

            min(
                repository[
                    "average_issue_comments"
                ],
                20,
            )

            +

            max(

                0,

                10

                -

                repository[
                    "average_issue_resolution_days"
                ],

            )

        )

        repository[
            "issue_activity_score"
        ] = round(
            score,
            2,
        )

    logger.info(
        "Issue aggregation complete."
    )

    return dataset


# ==========================================================
# Pull Request Feature Engineering
# ==========================================================


def aggregate_prs(dataset, prs):

    logger.info("Aggregating pull requests...")

    for repo_key, repo_prs in prs.items():

        if repo_key not in dataset:
            continue

        repository = dataset[repo_key]

        total = len(repo_prs)

        if total == 0:
            continue

        merged = 0

        changed_files = []

        code_churn = []

        review_comments = []

        merge_days = []

        for pr in repo_prs:

            if pr.get(
                "merged"
            ):
                merged += 1

            changed = (
                pr.get(
                    "changed_files"
                )
                or 0
            )

            changed_files.append(
                changed
            )

            additions = (
                pr.get(
                    "additions"
                )
                or 0
            )

            deletions = (
                pr.get(
                    "deletions"
                )
                or 0
            )

            code_churn.append(
                additions +
                deletions
            )

            review_comments.append(

                pr.get(
                    "review_comments",
                    0,
                )

            )

            created = parse_datetime(
                pr.get(
                    "created_at"
                )
            )

            merged_at = parse_datetime(
                pr.get(
                    "merged_at"
                )
            )

            if (
                created
                and
                merged_at
            ):

                merge_days.append(

                    (
                        merged_at
                        -
                        created
                    ).days

                )

        repository[
            "total_prs"
        ] = total

        repository[
            "merged_pr_ratio"
        ] = safe_ratio(
            merged,
            total,
        )

        repository[
            "average_changed_files"
        ] = safe_mean(
            changed_files
        )

        repository[
            "average_code_churn"
        ] = safe_mean(
            code_churn
        )

        repository[
            "average_review_comments"
        ] = safe_mean(
            review_comments
        )

        repository[
            "average_merge_days"
        ] = safe_mean(
            merge_days
        )

        score = (

            repository[
                "merged_pr_ratio"
            ] * 45

            +

            min(

                repository[
                    "average_review_comments"
                ],

                20,

            )

            +

            min(

                repository[
                    "average_changed_files"
                ],

                15,

            )

            +

            max(

                0,

                20

                -

                repository[
                    "average_merge_days"
                ],

            )

        )

        repository[
            "pr_activity_score"
        ] = round(
            score,
            2,
        )

    logger.info(
        "Pull Request aggregation complete."
    )

    return dataset

# ==========================================================
# File Feature Engineering
# ==========================================================

def aggregate_files(dataset, files):

    logger.info("Aggregating repository files...")

    for repo_key, repo_files in files.items():

        if repo_key not in dataset:
            continue

        repository = dataset[repo_key]

        total = len(repo_files)

        if total == 0:
            continue

        source = 0
        docs = 0
        configs = 0
        tests = 0
        binaries = 0

        has_ci_cd = 0
        has_dockerfile = 0
        has_contributing = 0
        dependency_files = 0

        directory_depths = []
        file_sizes = []

        languages = set()

        CI_CD_MARKERS = (
            ".github/workflows/",
            ".gitlab-ci.yml",
            ".circleci/",
            ".travis.yml",
            "azure-pipelines.yml",
            "jenkinsfile",
        )

        DEPENDENCY_FILES = {
            "package.json",
            "requirements.txt",
            "pipfile",
            "pyproject.toml",
            "pom.xml",
            "build.gradle",
            "go.mod",
            "cargo.toml",
            "gemfile",
            "composer.json",
        }

        for file in repo_files:

            path_lower = file.get(
                "path", ""
            ).lower()

            filename_lower = file.get(
                "filename", ""
            ).lower()

            category = file.get(
                "category",
                "Other"
            )

            if category == "Source Code":
                source += 1

            if file.get(
                "is_documentation"
            ):
                docs += 1

            if file.get(
                "is_config_file"
            ):
                configs += 1

            if file.get(
                "is_test_file"
            ):
                tests += 1

            if file.get(
                "is_binary"
            ):
                binaries += 1

            if any(
                marker in path_lower
                for marker in CI_CD_MARKERS
            ):
                has_ci_cd = 1

            if filename_lower in (
                "dockerfile",
                "docker-compose.yml",
            ):
                has_dockerfile = 1

            if filename_lower == "contributing.md":
                has_contributing = 1

            if filename_lower in DEPENDENCY_FILES:
                dependency_files += 1

            directory_depths.append(

                file.get(
                    "directory_depth",
                    0,
                )

            )

            file_sizes.append(

                file.get(
                    "size",
                    0,
                )

            )

            language = file.get(
                "language",
                "Unknown",
            )

            if language != "Unknown":

                languages.add(
                    language
                )

        repository[
            "total_files"
        ] = total

        repository[
            "source_code_ratio"
        ] = safe_ratio(
            source,
            total,
        )

        repository[
            "documentation_ratio"
        ] = safe_ratio(
            docs,
            total,
        )

        repository[
            "config_ratio"
        ] = safe_ratio(
            configs,
            total,
        )

        repository[
            "test_ratio"
        ] = safe_ratio(
            tests,
            total,
        )

        repository[
            "binary_ratio"
        ] = safe_ratio(
            binaries,
            total,
        )

        repository[
            "language_diversity"
        ] = len(
            languages
        )

        repository[
            "average_directory_depth"
        ] = safe_mean(
            directory_depths
        )

        repository[
            "average_file_size"
        ] = safe_mean(
            file_sizes
        )

        repository[
            "has_ci_cd"
        ] = has_ci_cd

        repository[
            "has_dockerfile"
        ] = has_dockerfile

        repository[
            "has_contributing_guide"
        ] = has_contributing

        repository[
            "dependency_file_count"
        ] = dependency_files

        repository[
            "max_directory_depth"
        ] = (
            max(directory_depths)
            if directory_depths
            else 0
        )

        # ----------------------------------------
        # File Complexity Score
        # ----------------------------------------

        score = (

            repository[
                "source_code_ratio"
            ] * 35

            +

            repository[
                "documentation_ratio"
            ] * 15

            +

            repository[
                "config_ratio"
            ] * 10

            +

            repository[
                "test_ratio"
            ] * 20

            +

            min(

                repository[
                    "language_diversity"
                ] * 4,

                12,

            )

            +

            (

                1
                -
                repository[
                    "binary_ratio"
                ]

            ) * 8

        )

        repository[
            "file_complexity_score"
        ] = round(
            score,
            2,
        )

    logger.info(
        "File aggregation complete."
    )

    return dataset

# ==========================================================
# Repository Quality Index (RQI)
# ==========================================================

def normalize(values):

    if len(values) == 0:
        return []

    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return [50.0 for _ in values]

    return [

        round(

            ((value - minimum) /
             (maximum - minimum)) * 100,

            2,

        )

        for value in values

    ]


# ==========================================================
# Compute Repository Quality Index
# ==========================================================

def compute_repository_quality(dataset):

    logger.info(
        "Computing Repository Quality Index..."
    )

    repositories = list(
        dataset.values()
    )

    popularity = normalize([
        repo["popularity_score"]
        for repo in repositories
    ])

    commits = normalize([
        repo["commit_activity_score"]
        for repo in repositories
    ])

    issues = normalize([
        repo["issue_activity_score"]
        for repo in repositories
    ])

    prs = normalize([
        repo["pr_activity_score"]
        for repo in repositories
    ])

    files = normalize([
        repo["file_complexity_score"]
        for repo in repositories
    ])

    for i, repo in enumerate(repositories):

        score = (

            popularity[i] * 0.30

            +

            commits[i] * 0.25

            +

            issues[i] * 0.15

            +

            prs[i] * 0.15

            +

            files[i] * 0.15

        )

        repo[
            "repository_score"
        ] = round(
            score,
            2,
        )

        repo[
            "repository_quality_index"
        ] = round(
            score,
            2,
        )

    logger.info(
        "RQI computed successfully."
    )

    return dataset


# ==========================================================
# Deployment Ready Label
# ==========================================================

def generate_labels(dataset):

    logger.info(
        "Generating Deployment Labels..."
    )

    repositories = list(dataset.values())

    scores = [
        repo["repository_quality_index"]
        for repo in repositories
    ]

    # Quantile-based thresholds instead of fixed cutoffs.
    # Fixed cutoffs (>=80 / >=60) caused severe class imbalance
    # (489 vs 5 in the earlier run). Quantiles guarantee a
    # roughly balanced, usable label distribution.
    q33 = float(np.percentile(scores, 33))
    q66 = float(np.percentile(scores, 66))

    for repo in repositories:

        score = repo[
            "repository_quality_index"
        ]

        if score >= q66:

            label = "Production Ready"

            target = 1

        elif score >= q33:

            label = "Needs Improvements"

            target = 1

        else:

            label = "Not Ready"

            target = 0

        repo[
            "deployment_label"
        ] = label

        repo[
            "deployment_ready"
        ] = target

    logger.info(
        f"Thresholds -> q33: {q33:.2f}, q66: {q66:.2f}"
    )

    logger.info(
        "Labels generated."
    )

    return dataset


# ==========================================================
# Final DataFrame
# ==========================================================

def build_dataframe(dataset):

    logger.info(
        "Building final dataframe..."
    )

    rows = list(
        dataset.values()
    )

    df = pd.DataFrame(
        rows
    )

    numeric = df.select_dtypes(
        include=np.number
    ).columns

    categorical = df.select_dtypes(
        include="object"
    ).columns

    df[numeric] = df[
        numeric
    ].fillna(0)

    df[categorical] = df[
        categorical
    ].fillna("Unknown")

    df = df.drop_duplicates(
        subset=["repo_key"]
    )

    df = df.sort_values(

        by="repository_quality_index",

        ascending=False,

    ).reset_index(
        drop=True
    )

    priority = [

        "repo_key",

        "owner",

        "repository",

        "language",

        "repository_quality_index",

        "deployment_label",

        "deployment_ready",

        "repository_score",

        "commit_activity_score",

        "issue_activity_score",

        "pr_activity_score",

        "file_complexity_score",

        "stars",

        "forks",

        "watchers",

        "total_commits",

        "total_issues",

        "total_prs",

        "total_files",

    ]

    remaining = [

        column

        for column in df.columns

        if column not in priority

    ]

    df = df[
        priority +
        remaining
    ]

    save_dataframe(
        df,
        FINAL_DATASET,
    )

    logger.info(
        f"Dataset saved : {FINAL_DATASET}"
    )

    logger.info(
        f"Shape : {df.shape}"
    )

    return df

# ==========================================================
# ML-Ready Export (No Leakage)
# ==========================================================

# These columns were used to CONSTRUCT repository_quality_index
# (and therefore deployment_label / deployment_ready). Training a
# model with these as input features causes target leakage -- the
# model just re-learns the weighted-sum formula instead of learning
# real patterns from raw GitHub signals. Drop them before training.
LEAKY_COLUMNS = [
    "repository_score",
    "commit_activity_score",
    "issue_activity_score",
    "pr_activity_score",
    "file_complexity_score",
    "popularity_score",
    "repository_quality_index",
]

# These columns are always 0 for every repository. GitHub's PR
# *list* endpoint (/pulls) does not return changed_files, additions,
# deletions, or review_comments -- those only come from fetching each
# PR individually, which fetch_prs.py does not do. Since they carry
# zero variance/signal, they're dropped rather than fed to the model.
DEAD_COLUMNS = [
    "average_changed_files",
    "average_code_churn",
    "average_review_comments",
]

MLREADY_DATASET = PROCESSED_DATA_DIR / "ml_ready_dataset.csv"


def build_ml_ready_dataset(df):

    logger.info(
        "Building leakage-free ML-ready dataset..."
    )

    ml_df = df.drop(
        columns=LEAKY_COLUMNS + DEAD_COLUMNS,
        errors="ignore",
    )

    save_dataframe(
        ml_df,
        MLREADY_DATASET,
    )

    logger.info(
        f"ML-ready dataset saved : {MLREADY_DATASET}"
    )

    logger.info(
        f"Shape : {ml_df.shape}"
    )

    return ml_df


# ==========================================================
# Train / Validation / Test Split
# ==========================================================

def split_dataset(df):

    logger.info(
        "Splitting dataset..."
    )

    if len(df) < 10:

        logger.warning(
            "Dataset too small. Skipping split."
        )

        save_dataframe(
            df,
            TRAIN_DATASET,
        )

        return

    train_df, temp_df = train_test_split(

        df,

        test_size=0.20,

        random_state=RANDOM_SEED,

        shuffle=True,

    )

    validation_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        random_state=RANDOM_SEED,

        shuffle=True,

    )

    save_dataframe(
        train_df,
        TRAIN_DATASET,
    )

    save_dataframe(
        validation_df,
        VALIDATION_DATASET,
    )

    save_dataframe(
        test_df,
        TEST_DATASET,
    )

    logger.info(
        f"Train : {len(train_df)}"
    )

    logger.info(
        f"Validation : {len(validation_df)}"
    )

    logger.info(
        f"Test : {len(test_df)}"
    )


# ==========================================================
# Dataset Summary
# ==========================================================

def generate_summary(df):

    summary = {

        "repositories":
            int(len(df)),

        "commits":
            int(df["total_commits"].sum()),

        "issues":
            int(df["total_issues"].sum()),

        "pull_requests":
            int(df["total_prs"].sum()),

        "files":
            int(df["total_files"].sum()),

        "average_rqi":
            round(

                float(

                    df[
                        "repository_quality_index"
                    ].mean()

                ),

                2,

            ),

        "deployment_ready":

            int(

                df[
                    "deployment_ready"
                ].sum()

            ),

        "generated_at":

            datetime.now().isoformat(),

    }

    summary_path = (
        PROCESSED_DATA_DIR /
        "dataset_summary.json"
    )

    with open(

        summary_path,

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            summary,

            f,

            indent=4,

            ensure_ascii=False,

        )

    logger.info(
        f"Summary saved : {summary_path}"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("GitHub Dataset Builder Started")
    logger.info("=" * 60)

    (

        repositories,

        commits,

        issues,

        prs,

        files,

    ) = load_processed_data()

    dataset = initialize_dataset(
        repositories
    )

    dataset = aggregate_commits(
        dataset,
        commits,
    )

    dataset = aggregate_issues(
        dataset,
        issues,
    )

    dataset = aggregate_prs(
        dataset,
        prs,
    )

    dataset = aggregate_files(
        dataset,
        files,
    )

    dataset = compute_repository_quality(
        dataset,
    )

    dataset = generate_labels(
        dataset,
    )

    df = build_dataframe(
        dataset,
    )

    ml_df = build_ml_ready_dataset(
        df,
    )

    # Split the leakage-free version -- this is what you should
    # actually train your model on (train.csv / validation.csv / test.csv)
    split_dataset(
        ml_df,
    )

    generate_summary(
        df,
    )

    logger.info("=" * 60)
    logger.info("Dataset Build Completed Successfully")
    logger.info("=" * 60)


if __name__ == "__main__":

    main()