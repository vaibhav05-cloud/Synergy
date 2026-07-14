"""
preprocess.py

Production preprocessing pipeline
for GitHub Repository Quality Prediction.
"""

from __future__ import annotations

import json
import logging
import re

from pathlib import Path

from datetime import (
    datetime,
    timezone,
)

from config import (

    REPOSITORIES_FILE,

    COMMITS_DIR,

    ISSUES_DIR,

    PRS_DIR,

    FILES_DIR,

    PROCESSED_DATA_DIR,

)

# =====================================================
# Logger
# =====================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

)

logger = logging.getLogger(
    "Preprocess"
)

# =====================================================
# Create Output Directories
# =====================================================

(PROCESSED_DATA_DIR / "commits").mkdir(
    parents=True,
    exist_ok=True,
)

(PROCESSED_DATA_DIR / "issues").mkdir(
    parents=True,
    exist_ok=True,
)

(PROCESSED_DATA_DIR / "prs").mkdir(
    parents=True,
    exist_ok=True,
)

(PROCESSED_DATA_DIR / "files").mkdir(
    parents=True,
    exist_ok=True,
)

# =====================================================
# Constants
# =====================================================

BUG_KEYWORDS = {

    "bug",

    "fix",

    "fixed",

    "issue",

    "error",

    "crash",

    "failure",

    "fault",

    "patch",

    "hotfix",

    "resolve",

    "resolved",

}

STOPWORDS = {

    "the",

    "is",

    "are",

    "was",

    "were",

    "to",

    "of",

    "a",

    "an",

    "in",

    "for",

    "and",

    "or",

    "on",

    "at",

    "with",

    "from",

    "by",

    "be",

    "this",

    "that",

    "it",

    "as",

    "into",

}

# =====================================================
# JSON Helpers
# =====================================================

def load_json(path: Path):

    if not path.exists():

        return []

    with open(

        path,

        "r",

        encoding="utf-8",

    ) as f:

        return json.load(f)


def save_json(data, path: Path):

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    with open(

        path,

        "w",

        encoding="utf-8",

    ) as f:

        json.dump(

            data,

            f,

            indent=4,

            ensure_ascii=False,

        )

# =====================================================
# Text Helpers
# =====================================================

def clean_text(text):

    if text is None:

        return ""

    text = str(text).lower()

    text = re.sub(

        r"http\\S+",

        "",

        text,

    )

    text = re.sub(

        r"[^a-z0-9 ]",

        " ",

        text,

    )

    text = re.sub(

        r"\\s+",

        " ",

        text,

    )

    return text.strip()


def tokenize(text):

    return [

        word

        for word in clean_text(text).split()

        if word not in STOPWORDS

    ]


def word_count(text):

    return len(

        tokenize(text)

    )


def contains_bug_keyword(text):

    words = set(

        tokenize(text)

    )

    return len(

        words & BUG_KEYWORDS

    ) > 0

# =====================================================
# Date Helpers
# =====================================================

def parse_datetime(value):

    if not value:

        return None

    try:

        return datetime.strptime(

            value,

            "%Y-%m-%dT%H:%M:%SZ",

        ).replace(

            tzinfo=timezone.utc,

        )

    except Exception:

        return None


def repository_age(created_at):

    dt = parse_datetime(

        created_at

    )

    if dt is None:

        return 0

    return max(

        (

            datetime.now(

                timezone.utc

            )

            -

            dt

        ).days,

        0,

    )


def last_updated(updated_at):

    dt = parse_datetime(

        updated_at

    )

    if dt is None:

        return 0

    return max(

        (

            datetime.now(

                timezone.utc

            )

            -

            dt

        ).days,

        0,

    )

# =====================================================
# Utility Helpers
# =====================================================

def safe_int(value):

    try:

        return int(value)

    except Exception:

        return 0


def unique(items, key):

    seen = set()

    result = []

    for item in items:

        value = item.get(key)

        if value in seen:

            continue

        seen.add(value)

        result.append(item)

    return result


logger.info(
    "Preprocess Pipeline Loaded."
)

# =====================================================
# Repository Processing
# =====================================================

def normalize_repository(repo):

    return {

        "id":
            safe_int(
                repo.get("id")
            ),

        "owner":
            clean_text(
                repo.get("owner")
            ),

        "name":
            clean_text(
                repo.get("name")
            ),

        "full_name":
            clean_text(
                repo.get("full_name")
            ),

        "language":
            repo.get(
                "language"
            ) or "Unknown",

        "description":
            clean_text(
                repo.get(
                    "description"
                )
            ),

        "stars":
            safe_int(
                repo.get(
                    "stars"
                )
            ),

        "forks":
            safe_int(
                repo.get(
                    "forks"
                )
            ),

        "watchers":
            safe_int(
                repo.get(
                    "watchers"
                )
            ),

        "open_issues":
            safe_int(
                repo.get(
                    "open_issues"
                )
            ),

        "size":
            safe_int(
                repo.get(
                    "size"
                )
            ),

        "license":
            repo.get(
                "license"
            ) or "Unknown",

        "default_branch":
            repo.get(
                "default_branch"
            ) or "main",

        "created_at":
            repo.get(
                "created_at"
            ),

        "updated_at":
            repo.get(
                "updated_at"
            ),

        "html_url":
            repo.get(
                "html_url"
            ),

    }


# -----------------------------------------------------


def repository_features(repo):

    repo[
        "repository_age_days"
    ] = repository_age(
        repo[
            "created_at"
        ]
    )

    repo[
        "last_update_days"
    ] = last_updated(
        repo[
            "updated_at"
        ]
    )

    repo[
        "description_length"
    ] = len(
        repo[
            "description"
        ]
    )

    repo[
        "description_words"
    ] = word_count(
        repo[
            "description"
        ]
    )

    repo[
        "has_description"
    ] = int(

        bool(

            repo[
                "description"
            ]

        )

    )

    repo[
        "has_license"
    ] = int(

        repo[
            "license"
        ] != "Unknown"

    )

    repo[
        "is_popular"
    ] = int(

        repo[
            "stars"
        ] >= 1000

    )

    repo[
        "is_active"
    ] = int(

        repo[
            "last_update_days"
        ] <= 90

    )

    repo[
        "is_large_repository"
    ] = int(

        repo[
            "size"
        ] >= 50000

    )

    repo[
        "popularity_score"
    ] = round(

        (

            repo[
                "stars"
            ] * 0.60

        )

        +

        (

            repo[
                "forks"
            ] * 0.25

        )

        +

        (

            repo[
                "watchers"
            ] * 0.15

        ),

        2,

    )

    return repo


# -----------------------------------------------------


def preprocess_repositories():

    logger.info(
        "Processing repositories..."
    )

    repositories = load_json(
        REPOSITORIES_FILE
    )

    repositories = unique(

        repositories,

        "id",

    )

    processed = []

    for repo in repositories:

        repo = normalize_repository(
            repo
        )

        repo = repository_features(
            repo
        )

        processed.append(
            repo
        )

    save_json(

        processed,

        PROCESSED_DATA_DIR
        /
        "repositories.json",

    )

    logger.info(

        f"{len(processed)} repositories processed."

    )

    return processed

# =====================================================
# Commit Processing
# =====================================================

def normalize_commit(commit):

    message = clean_text(
        commit.get("message")
    )

    return {

        "sha":
            commit.get("sha"),

        "author":
            clean_text(
                commit.get("author")
            ),

        "email":
            clean_text(
                commit.get("email")
            ),

        "date":
            commit.get("date"),

        "datetime":
            parse_datetime(
                commit.get("date")
            ),

        "message":
            message,

        "url":
            commit.get("url"),
    }


# -----------------------------------------------------


def commit_features(commit):

    dt = commit["datetime"]

    commit[
        "message_length"
    ] = len(
        commit["message"]
    )

    commit[
        "word_count"
    ] = word_count(
        commit["message"]
    )

    commit[
        "contains_bug_keyword"
    ] = contains_bug_keyword(
        commit["message"]
    )

    if dt:

        commit["hour"] = dt.hour

        commit["weekday"] = dt.weekday()

        commit["is_weekend"] = (
            dt.weekday() >= 5
        )

        commit["is_night"] = (
            dt.hour >= 22
            or
            dt.hour <= 5
        )

    else:

        commit["hour"] = -1

        commit["weekday"] = -1

        commit["is_weekend"] = False

        commit["is_night"] = False

    length = commit[
        "message_length"
    ]

    if length < 20:

        commit[
            "commit_size"
        ] = "small"

    elif length < 60:

        commit[
            "commit_size"
        ] = "medium"

    else:

        commit[
            "commit_size"
        ] = "large"

    commit[
        "is_empty_message"
    ] = (
        length == 0
    )

    return commit


# -----------------------------------------------------


def preprocess_commits():

    logger.info(
        "Processing commits..."
    )

    output_dir = (
        PROCESSED_DATA_DIR /
        "commits"
    )

    total = 0

    for file in COMMITS_DIR.glob(
        "*.json"
    ):

        commits = load_json(
            file
        )

        commits = unique(
            commits,
            "sha",
        )

        commits = sorted(

            commits,

            key=lambda x:
            x.get("date", ""),

        )

        processed = []

        for commit in commits:

            commit = normalize_commit(
                commit
            )

            commit = commit_features(
                commit
            )

            # datetime object JSON me save nahi hota

            commit.pop(
                "datetime",
                None,
            )

            processed.append(
                commit
            )

        save_json(

            processed,

            output_dir /
            file.name,

        )

        logger.info(

            f"{file.name} : "

            f"{len(processed)} commits"

        )

        total += len(
            processed
        )

    logger.info(

        f"Processed {total} commits."

    )
    # =====================================================
# Issue Processing
# =====================================================

def normalize_issue(issue):

    return {

        "id":
            safe_int(
                issue.get("id")
            ),

        "number":
            safe_int(
                issue.get("number")
            ),

        "title":
            clean_text(
                issue.get("title")
            ),

        "state":
            (
                issue.get("state")
                or "open"
            ).lower(),

        "created_at":
            issue.get("created_at"),

        "updated_at":
            issue.get("updated_at"),

        "closed_at":
            issue.get("closed_at"),

        "author":
            clean_text(
                issue.get("author")
            ),

        "comments":
            safe_int(
                issue.get("comments")
            ),

        "labels":
            issue.get(
                "labels",
                []
            ),

        "assignees":
            issue.get(
                "assignees",
                []
            ),

        "locked":
            bool(
                issue.get("locked")
            ),

        "body":
            clean_text(
                issue.get("body")
            ),

        "url":
            issue.get("url"),

    }


# -----------------------------------------------------


def issue_features(issue):

    issue[
        "title_length"
    ] = len(
        issue["title"]
    )

    issue[
        "body_length"
    ] = len(
        issue["body"]
    )

    issue[
        "title_words"
    ] = word_count(
        issue["title"]
    )

    issue[
        "body_words"
    ] = word_count(
        issue["body"]
    )

    issue[
        "contains_bug_keyword"
    ] = (

        contains_bug_keyword(
            issue["title"]
        )

        or

        contains_bug_keyword(
            issue["body"]
        )

    )

    created = parse_datetime(
        issue[
            "created_at"
        ]
    )

    closed = parse_datetime(
        issue[
            "closed_at"
        ]
    )

    if created and closed:

        issue[
            "resolution_days"
        ] = (

            closed -
            created

        ).days

    else:

        issue[
            "resolution_days"
        ] = -1

    issue[
        "is_closed"
    ] = (

        issue[
            "state"
        ] == "closed"

    )

    issue[
        "label_count"
    ] = len(

        issue[
            "labels"
        ]

    )

    issue[
        "assignee_count"
    ] = len(

        issue[
            "assignees"
        ]

    )

    return issue


# -----------------------------------------------------


def preprocess_issues():

    logger.info(
        "Processing issues..."
    )

    output_dir = (
        PROCESSED_DATA_DIR /
        "issues"
    )

    total = 0

    for file in ISSUES_DIR.glob(
        "*.json"
    ):

        issues = load_json(
            file
        )

        issues = unique(
            issues,
            "id",
        )

        processed = []

        for issue in issues:

            issue = normalize_issue(
                issue
            )

            issue = issue_features(
                issue
            )

            processed.append(
                issue
            )

        save_json(

            processed,

            output_dir /
            file.name,

        )

        logger.info(

            f"{file.name} : "

            f"{len(processed)} issues"

        )

        total += len(
            processed
        )

    logger.info(

        f"Processed {total} issues."

    )
    
    # =====================================================
# Pull Request Processing
# =====================================================

def normalize_pr(pr):

    return {

        "id": safe_int(pr.get("id")),

        "number": safe_int(pr.get("number")),

        "title": clean_text(pr.get("title")),

        "state": (pr.get("state") or "open").lower(),

        "draft": bool(pr.get("draft")),

        "merged": bool(pr.get("merged")),

        "created_at": pr.get("created_at"),

        "updated_at": pr.get("updated_at"),

        "closed_at": pr.get("closed_at"),

        "merged_at": pr.get("merged_at"),

        "author": clean_text(pr.get("author")),

        "base_branch": pr.get("base_branch"),

        "head_branch": pr.get("head_branch"),

        "commits": safe_int(pr.get("commits")),

        "changed_files": safe_int(pr.get("changed_files")),

        "additions": safe_int(pr.get("additions")),

        "deletions": safe_int(pr.get("deletions")),

        "comments": safe_int(pr.get("comments")),

        "review_comments": safe_int(pr.get("review_comments")),

        "labels": pr.get("labels", []),

        "url": pr.get("url"),

    }


# -----------------------------------------------------


def pr_features(pr):

    pr["title_length"] = len(pr["title"])

    pr["title_words"] = word_count(pr["title"])

    pr["contains_bug_keyword"] = contains_bug_keyword(
        pr["title"]
    )

    created = parse_datetime(
        pr["created_at"]
    )

    merged = parse_datetime(
        pr["merged_at"]
    )

    if created and merged:

        pr["merge_days"] = (

            merged -
            created

        ).days

    else:

        pr["merge_days"] = -1

    pr["code_churn"] = (

        pr["additions"]

        +

        pr["deletions"]

    )

    return pr


# -----------------------------------------------------


def preprocess_prs():

    logger.info(
        "Processing pull requests..."
    )

    output_dir = (
        PROCESSED_DATA_DIR /
        "prs"
    )

    total = 0

    for file in PRS_DIR.glob("*.json"):

        prs = load_json(file)

        prs = unique(
            prs,
            "id"
        )

        processed = []

        for pr in prs:

            pr = normalize_pr(pr)

            pr = pr_features(pr)

            processed.append(pr)

        save_json(

            processed,

            output_dir /
            file.name,

        )

        logger.info(

            f"{file.name} : "

            f"{len(processed)} PRs"

        )

        total += len(processed)

    logger.info(

        f"Processed {total} PRs."

    )


# =====================================================
# Repository Files Processing
# =====================================================

def normalize_file(file):

    return {

        "path": file.get("path"),

        "filename": file.get("filename"),

        "parent_directory": file.get("parent_directory"),

        "extension": file.get("extension"),

        "language": file.get("language"),

        "category": file.get("category"),

        "directory_depth": safe_int(

            file.get(
                "directory_depth"
            )

        ),

        "size": safe_int(
            file.get("size")
        ),

        "sha": file.get("sha"),

        "git_url": file.get("git_url"),

        "is_test_file": bool(
            file.get(
                "is_test_file"
            )
        ),

        "is_documentation": bool(
            file.get(
                "is_documentation"
            )
        ),

        "is_config_file": bool(
            file.get(
                "is_config_file"
            )
        ),

        "is_hidden": bool(
            file.get(
                "is_hidden"
            )
        ),

        "is_binary": bool(
            file.get(
                "is_binary"
            )
        ),

    }


# -----------------------------------------------------


def file_features(file):

    file["filename_length"] = len(
        file["filename"]
    )

    file["path_depth"] = len(

        file["path"].split("/")

    )

    file["is_source_code"] = (

        file["category"]

        ==

        "Source Code"

    )

    return file


# -----------------------------------------------------


def preprocess_files():

    logger.info(
        "Processing repository files..."
    )

    output_dir = (
        PROCESSED_DATA_DIR /
        "files"
    )

    total = 0

    for file in FILES_DIR.glob("*.json"):

        files = load_json(file)

        processed = []

        for item in files:

            item = normalize_file(
                item
            )

            item = file_features(
                item
            )

            processed.append(
                item
            )

        save_json(

            processed,

            output_dir /
            file.name,

        )

        logger.info(

            f"{file.name} : "

            f"{len(processed)} files"

        )

        total += len(
            processed
        )

    logger.info(

        f"Processed {total} files."

    )
    
    # =====================================================
# Complete Pipeline
# =====================================================

def preprocess_all():

    logger.info("=" * 60)
    logger.info("GitHub Dataset Preprocessing Started")
    logger.info("=" * 60)

    preprocess_repositories()

    preprocess_commits()

    preprocess_issues()

    preprocess_prs()

    preprocess_files()

    logger.info("=" * 60)
    logger.info("Preprocessing Completed Successfully")
    logger.info("=" * 60)


# =====================================================
# Dataset Statistics
# =====================================================

def dataset_statistics():

    logger.info("Generating dataset statistics...")

    stats = {}

    stats["repositories"] = len(
        load_json(
            PROCESSED_DATA_DIR /
            "repositories.json"
        )
    )

    stats["commit_files"] = len(

        list(

            (
                PROCESSED_DATA_DIR /
                "commits"
            ).glob("*.json")

        )

    )

    stats["issue_files"] = len(

        list(

            (
                PROCESSED_DATA_DIR /
                "issues"
            ).glob("*.json")

        )

    )

    stats["pr_files"] = len(

        list(

            (
                PROCESSED_DATA_DIR /
                "prs"
            ).glob("*.json")

        )

    )

    stats["file_metadata_files"] = len(

        list(

            (
                PROCESSED_DATA_DIR /
                "files"
            ).glob("*.json")

        )

    )

    logger.info("")

    logger.info("=" * 60)

    logger.info("DATASET SUMMARY")

    logger.info("=" * 60)

    for key, value in stats.items():

        logger.info(
            f"{key:25} : {value}"
        )

    logger.info("=" * 60)


# =====================================================
# Main
# =====================================================

def main():

    preprocess_all()

    dataset_statistics()


if __name__ == "__main__":

    main()