"""Build a leakage-safe deployment-risk dataset from GitHub Actions metadata.

The GitHub API exposes workflow and commit facts, but not private operational
facts such as incidents and on-call coverage. This script keeps those two worlds
separate: API data is cached unchanged, while operational signals are generated
causally and documented in the final data card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "github_api"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CONFIG_PATH = ROOT / "config" / "repositories.json"

API_URL = "https://api.github.com"
RISK_SEED = 20260714
LOW_MEDIUM_QUANTILE = 0.45
MEDIUM_HIGH_QUANTILE = 0.80
RISK_LEVELS = ["Low", "Medium", "High"]
SPLIT_LEVELS = ["train", "validation", "test"]

DEPLOYMENT_TERMS = re.compile(
    r"deploy|release|publish|production|staging|ship|delivery|promote|rollout",
    re.IGNORECASE,
)
HOTFIX_TERMS = re.compile(r"hotfix|urgent|critical fix|sev[ -]?[0-9]", re.IGNORECASE)
REVERT_TERMS = re.compile(r"\brevert\b", re.IGNORECASE)
BREAKING_TERMS = re.compile(r"breaking|!:", re.IGNORECASE)

FAILURE_CONCLUSIONS = {"failure", "timed_out", "action_required", "startup_failure"}
EXCLUDED_EVENTS = {"pull_request", "pull_request_target", "issue_comment", "merge_group"}

AREA_RULES: list[tuple[str, re.Pattern[str], int]] = [
    ("identity_security", re.compile(r"auth|oauth|oidc|sso|security|permission|rbac|credential|secret"), 5),
    ("database", re.compile(r"migration|migrate|schema|database|\bdb/|sql/"), 5),
    ("infrastructure", re.compile(r"terraform|\.tf$|helm|kubernetes|k8s|docker|infra/|cloudformation|pulumi"), 4),
    ("payments", re.compile(r"payment|billing|invoice|checkout|subscription"), 5),
    ("api_backend", re.compile(r"api/|server/|backend/|controller|service/|gateway"), 4),
    ("observability", re.compile(r"monitor|metric|tracing|telemetry|observability|alert"), 3),
    ("frontend", re.compile(r"frontend|client/|web/|ui/|\.tsx?$|\.vue$|\.svelte$"), 2),
    ("documentation", re.compile(r"(^|/)docs?/|\.md$|\.rst$|changelog|readme"), 1),
]

MODEL_FEATURES = [
    "repo_domain",
    "candidate_type",
    "primary_service",
    "branch_type",
    "workflow_kind",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_friday",
    "is_off_hours",
    "is_deployment_intent",
    "run_attempt",
    "commit_age_hours",
    "commit_parent_count",
    "additions",
    "deletions",
    "code_churn",
    "changed_files",
    "file_extension_entropy",
    "service_area_count",
    "service_criticality",
    "is_database_migration",
    "is_infrastructure_change",
    "is_dependency_change",
    "is_config_change",
    "is_security_change",
    "is_test_change",
    "test_to_source_file_ratio",
    "is_docs_only",
    "has_lockfile_change",
    "is_hotfix",
    "is_revert",
    "is_breaking_change",
    "deployments_7d",
    "deployments_30d",
    "recent_change_velocity_7d",
    "past_failed_runs_7d",
    "past_failure_rate_30d",
    "past_incidents_14d",
    "past_rollbacks_30d",
    "time_since_last_deployment_hours",
    "actor_prior_deployments",
    "actor_prior_failure_rate",
    "service_deployments_30d",
    "service_incident_rate_30d",
    "on_call_available",
    "expected_on_call_response_minutes",
    "on_call_load_24h",
]

FORBIDDEN_MODEL_COLUMNS = {
    "workflow_conclusion_observed",
    "workflow_failed_observed",
    "completed_at",
    "run_duration_seconds",
    "post_deploy_incident_7d",
    "rollback_within_24h",
    "risk_score",
    "calibrated_risk_index",
    "risk_probability",
    "latent_reliability_shock",
    "current_conclusion",
}

FEATURE_DICTIONARY = {
    "repo_domain": ("categorical", "Configured public-project domain; reduces repository-name memorization."),
    "candidate_type": ("categorical", "Whether the run has explicit deploy wording or is a mainline delivery proxy."),
    "primary_service": ("categorical", "Highest-risk service area inferred from changed file paths."),
    "branch_type": ("categorical", "Main, release, hotfix, feature, or other branch category."),
    "workflow_kind": ("categorical", "Workflow-name category available before the deployment starts."),
    "hour": ("numeric", "UTC hour when the candidate deployment was created."),
    "day_of_week": ("numeric", "UTC weekday, Monday=0."),
    "is_weekend": ("binary", "Candidate occurs on Saturday or Sunday."),
    "is_friday": ("binary", "Candidate occurs on Friday."),
    "is_off_hours": ("binary", "Candidate is outside 07:00-19:59 UTC."),
    "is_deployment_intent": ("binary", "Workflow name contains deployment-oriented language."),
    "run_attempt": ("numeric", "GitHub Actions rerun attempt number."),
    "commit_age_hours": ("numeric", "Hours from commit author time to workflow creation."),
    "commit_parent_count": ("numeric", "Number of commit parents; merge commits can carry broader risk."),
    "additions": ("numeric", "Lines added in the candidate commit."),
    "deletions": ("numeric", "Lines deleted in the candidate commit."),
    "code_churn": ("numeric", "Total added plus deleted lines."),
    "changed_files": ("numeric", "Number of files changed in the candidate commit."),
    "file_extension_entropy": ("numeric", "Diversity of changed file types; broader changes have higher entropy."),
    "service_area_count": ("numeric", "Number of distinct service areas touched."),
    "service_criticality": ("numeric", "1-5 criticality based on affected path categories."),
    "is_database_migration": ("binary", "Commit contains schema, migration, or database paths."),
    "is_infrastructure_change": ("binary", "Commit contains infrastructure or deployment configuration paths."),
    "is_dependency_change": ("binary", "Commit changes package manifests or dependency declarations."),
    "is_config_change": ("binary", "Commit changes application or deployment configuration."),
    "is_security_change": ("binary", "Commit touches auth, identity, permissions, or secrets paths."),
    "is_test_change": ("binary", "Commit includes a test file change."),
    "test_to_source_file_ratio": ("numeric", "Changed test files divided by changed source files."),
    "is_docs_only": ("binary", "All touched files are documentation-like files."),
    "has_lockfile_change": ("binary", "Commit changes a dependency lock file."),
    "is_hotfix": ("binary", "Branch or commit message signals a hotfix."),
    "is_revert": ("binary", "Commit message signals a revert."),
    "is_breaking_change": ("binary", "Commit message signals a breaking change."),
    "deployments_7d": ("numeric", "Earlier candidate deployments in the repository during the prior 7 days."),
    "deployments_30d": ("numeric", "Earlier candidate deployments in the repository during the prior 30 days."),
    "recent_change_velocity_7d": ("numeric", "Prior seven-day repository code churn."),
    "past_failed_runs_7d": ("numeric", "Earlier observed GitHub workflow failures in the prior 7 days."),
    "past_failure_rate_30d": ("numeric", "Earlier 30-day observed workflow failure rate."),
    "past_incidents_14d": ("numeric", "Earlier simulated post-deployment incidents in the prior 14 days."),
    "past_rollbacks_30d": ("numeric", "Earlier simulated rollbacks in the prior 30 days."),
    "time_since_last_deployment_hours": ("numeric", "Elapsed hours since the prior repository candidate deployment."),
    "actor_prior_deployments": ("numeric", "Earlier candidate deployments created by the same public GitHub actor."),
    "actor_prior_failure_rate": ("numeric", "Earlier observed workflow failure rate for that actor in this repository."),
    "service_deployments_30d": ("numeric", "Earlier 30-day deployments affecting the same inferred service area."),
    "service_incident_rate_30d": ("numeric", "Earlier simulated incident rate for the inferred service area."),
    "on_call_available": ("binary", "Synthetic, pre-scheduled on-call availability at candidate time."),
    "expected_on_call_response_minutes": ("numeric", "Synthetic expected response time derived before deployment."),
    "on_call_load_24h": ("numeric", "Earlier candidate deployments in the prior 24 hours."),
}


class GitHubApiError(RuntimeError):
    """Raised when GitHub API collection cannot continue safely."""


class GitHubClient:
    def __init__(self, token: str) -> None:
        self.token = token
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "DeploySense-AI-Dataset-Builder",
            }
        )

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{API_URL}{path}"
        last_error = "unknown API error"
        for attempt in range(1, 5):
            try:
                response = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                last_error = str(exc)
                time.sleep(min(2**attempt, 12))
                continue

            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                return None

            last_error = f"HTTP {response.status_code}: {response.text[:240]}"
            if response.status_code not in {403, 429, 500, 502, 503, 504}:
                break

            retry_after = response.headers.get("Retry-After")
            reset_at = response.headers.get("X-RateLimit-Reset")
            if retry_after:
                wait_seconds = min(int(retry_after), 30)
            elif response.headers.get("X-RateLimit-Remaining") == "0" and reset_at:
                wait_seconds = max(1, min(int(float(reset_at) - time.time()) + 1, 30))
            else:
                wait_seconds = min(2**attempt, 12)
            time.sleep(wait_seconds)

        raise GitHubApiError(f"GitHub request failed for {path}. {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the DeploySense dataset.")
    parser.add_argument("--runs-per-repo", type=int, default=300, help="Candidate deployments to keep per repository.")
    parser.add_argument(
        "--max-pages-per-repo",
        type=int,
        default=15,
        help="GitHub Actions pages to scan per repository; higher values find more deployment-like runs.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent commit-detail requests.")
    parser.add_argument("--refresh", action="store_true", help="Refresh workflow and commit caches from GitHub.")
    parser.add_argument("--rebuild-from-cache", action="store_true", help="Build outputs from existing raw caches without API access.")
    parser.add_argument("--seed", type=int, default=RISK_SEED, help="Seed for reproducible operational simulation.")
    parser.add_argument(
        "--split-strategy",
        choices=["stratified_temporal", "temporal"],
        default="stratified_temporal",
        help="Use stratified_temporal for ML-ready class coverage, or temporal for pure chronological holdout.",
    )
    return parser.parse_args()


def ensure_directories() -> None:
    for directory in (DATA_RAW, DATA_PROCESSED, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)


def load_repositories() -> list[dict[str, str]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    repositories = payload.get("repositories", [])
    if not repositories:
        raise ValueError(f"No repositories configured in {CONFIG_PATH}")
    for item in repositories:
        if not item.get("repo") or not item.get("domain"):
            raise ValueError("Each configured repository needs both 'repo' and 'domain'.")
    return repositories


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def parse_file_paths(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(path) for path in value]
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
            return [str(path) for path in decoded] if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def safe_timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return parsed if not pd.isna(parsed) else pd.NaT


def collect_repository_metadata(client: GitHubClient, repositories: list[dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in repositories:
        repo = item["repo"]
        payload = client.get_json(f"/repos/{repo}")
        if payload is None:
            print(f"[warn] Repository unavailable, skipped: {repo}")
            continue
        rows.append(
            {
                "repo": repo,
                "repo_domain": item["domain"],
                "default_branch": payload.get("default_branch") or "main",
                "repo_language": payload.get("language") or "unknown",
                "repo_archived": int(bool(payload.get("archived"))),
                "repo_size_kb": payload.get("size") or 0,
            }
        )
    if not rows:
        raise GitHubApiError("No configured public repositories could be collected.")
    return pd.DataFrame(rows)


def run_is_candidate(run: dict[str, Any], default_branch: str) -> tuple[bool, str]:
    if run.get("status") != "completed" or run.get("event") in EXCLUDED_EVENTS or not run.get("head_sha"):
        return False, ""
    workflow_name = str(run.get("name") or "")
    branch = str(run.get("head_branch") or "")
    explicit_intent = bool(DEPLOYMENT_TERMS.search(workflow_name))
    mainline = branch in {default_branch, "main", "master"}
    release_like = bool(re.search(r"release|hotfix|production|staging", branch, re.IGNORECASE))
    release_event = run.get("event") == "release"
    if explicit_intent or release_event:
        return True, "explicit_deployment_intent"
    if mainline or release_like:
        return True, "mainline_delivery_proxy"
    return False, ""


def collect_workflow_runs(
    client: GitHubClient, repository_metadata: pd.DataFrame, runs_per_repo: int, max_pages_per_repo: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pages = max(1, max_pages_per_repo)
    for metadata in repository_metadata.to_dict("records"):
        repo = metadata["repo"]
        repo_rows: list[dict[str, Any]] = []
        for page in range(1, pages + 1):
            payload = client.get_json(
                f"/repos/{repo}/actions/runs", params={"per_page": 100, "page": page, "exclude_pull_requests": "true"}
            )
            for run in (payload or {}).get("workflow_runs", []):
                include, candidate_type = run_is_candidate(run, metadata["default_branch"])
                if not include:
                    continue
                repo_rows.append(
                    {
                        "deployment_id": f"{repo}:{run['id']}",
                        "repo": repo,
                        "repo_domain": metadata["repo_domain"],
                        "default_branch": metadata["default_branch"],
                        "repo_language": metadata["repo_language"],
                        "repo_size_kb": metadata["repo_size_kb"],
                        "run_id": run["id"],
                        "workflow_id": run.get("workflow_id"),
                        "workflow_name": run.get("name") or "unknown",
                        "event": run.get("event") or "unknown",
                        "head_branch": run.get("head_branch") or "unknown",
                        "head_sha": run.get("head_sha"),
                        "actor_login": (run.get("actor") or {}).get("login") or "unknown",
                        "created_at": run.get("created_at"),
                        "updated_at": run.get("updated_at"),
                        "completed_at": run.get("updated_at"),
                        "workflow_conclusion_observed": run.get("conclusion") or "unknown",
                        "workflow_failed_observed": int((run.get("conclusion") or "") in FAILURE_CONCLUSIONS),
                        "run_attempt": run.get("run_attempt") or 1,
                        "run_number": run.get("run_number") or 0,
                        "candidate_type": candidate_type,
                    }
                )
            if len(repo_rows) >= runs_per_repo or not (payload or {}).get("workflow_runs"):
                break
        rows.extend(repo_rows[:runs_per_repo])
        print(f"[collect] {repo}: {len(repo_rows[:runs_per_repo])} candidate deployments")

    runs = pd.DataFrame(rows)
    if runs.empty:
        raise GitHubApiError("No candidate workflow runs were returned. Try increasing --runs-per-repo.")
    return runs.drop_duplicates(subset=["deployment_id"]).sort_values("created_at").reset_index(drop=True)


def fetch_commit_details(client: GitHubClient, repo: str, sha: str) -> dict[str, Any]:
    payload = client.get_json(f"/repos/{repo}/commits/{sha}")
    if payload is None:
        return {"repo": repo, "head_sha": sha, "commit_details_available": 0}
    files = payload.get("files") or []
    commit = payload.get("commit") or {}
    author = commit.get("author") or {}
    stats = payload.get("stats") or {}
    return {
        "repo": repo,
        "head_sha": sha,
        "commit_details_available": 1,
        "commit_authored_at": author.get("date"),
        "commit_message": commit.get("message") or "",
        "commit_parent_count": len(payload.get("parents") or []),
        "additions": stats.get("additions") or 0,
        "deletions": stats.get("deletions") or 0,
        "changed_files": stats.get("total") and len(files) or len(files),
        "file_paths_json": json.dumps([file.get("filename", "") for file in files]),
        "files_truncated": int(len(files) >= 300),
    }


def load_or_collect_commit_details(
    client: GitHubClient | None, runs: pd.DataFrame, refresh: bool, workers: int
) -> pd.DataFrame:
    cache_path = DATA_RAW / "commit_details.csv"
    existing = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    existing_keys = set()
    if not existing.empty:
        existing_keys = set(zip(existing["repo"].astype(str), existing["head_sha"].astype(str)))

    needed = sorted(set(zip(runs["repo"].astype(str), runs["head_sha"].astype(str))))
    missing = needed if refresh else [key for key in needed if key not in existing_keys]
    if missing and client is None:
        raise FileNotFoundError("Commit cache is incomplete. Run without --rebuild-from-cache first.")

    fetched: list[dict[str, Any]] = []
    if missing:
        print(f"[collect] Fetching {len(missing)} unique commit records with {workers} workers...")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(fetch_commit_details, client, repo, sha): (repo, sha) for repo, sha in missing}
            for index, future in enumerate(as_completed(futures), start=1):
                repo, sha = futures[future]
                try:
                    fetched.append(future.result())
                except Exception as exc:  # Preserve the workflow row with a documented missing commit payload.
                    print(f"[warn] Commit details unavailable for {repo}@{sha[:10]}: {exc}")
                    fetched.append({"repo": repo, "head_sha": sha, "commit_details_available": 0})
                if index % 50 == 0 or index == len(missing):
                    print(f"[collect] Commit details: {index}/{len(missing)}")

    fetched_frame = pd.DataFrame(fetched)
    if refresh:
        cache = fetched_frame
    elif existing.empty:
        cache = fetched_frame
    elif fetched_frame.empty:
        cache = existing
    else:
        cache = pd.concat([existing, fetched_frame], ignore_index=True)
        cache = cache.drop_duplicates(subset=["repo", "head_sha"], keep="last")
    cache.to_csv(cache_path, index=False)
    return cache


def branch_type(branch: str, default_branch: str) -> str:
    value = branch.lower()
    if branch in {default_branch, "main", "master"}:
        return "main"
    if "hotfix" in value:
        return "hotfix"
    if "release" in value or value.startswith("v"):
        return "release"
    if any(token in value for token in ("feature", "feat/", "bugfix", "fix/")):
        return "feature"
    return "other"


def workflow_kind(name: str) -> str:
    lower = name.lower()
    if DEPLOYMENT_TERMS.search(lower):
        return "deployment"
    if any(word in lower for word in ("test", "ci", "check", "verify")):
        return "verification"
    if any(word in lower for word in ("build", "compile", "package")):
        return "build"
    if any(word in lower for word in ("security", "scan", "lint")):
        return "quality"
    return "other"


def classify_paths(paths: list[str]) -> dict[str, Any]:
    normalized = [path.lower() for path in paths if path]
    joined = "\n".join(normalized)
    matched_areas: list[tuple[str, int]] = []
    for area, pattern, criticality in AREA_RULES:
        if pattern.search(joined):
            matched_areas.append((area, criticality))

    if not matched_areas:
        matched_areas = [("shared_platform", 3)]
    area_priority = {area: criticality for area, criticality in matched_areas}
    primary_service = sorted(area_priority, key=lambda area: (-area_priority[area], area))[0]

    extensions = []
    for path in normalized:
        filename = Path(path).name
        suffix = Path(filename).suffix.lower() or "[no_extension]"
        extensions.append(suffix)
    counts = Counter(extensions)
    total_extensions = sum(counts.values())
    entropy = -sum((count / total_extensions) * math.log2(count / total_extensions) for count in counts.values()) if total_extensions else 0.0

    test_files = sum(1 for path in normalized if re.search(r"(^|/)(test|tests|spec|__tests__)(/|$)|(_test|\.spec|\.test)\.", path))
    docs_files = sum(1 for path in normalized if re.search(r"(^|/)docs?/|\.md$|\.rst$|changelog|readme", path))
    source_files = max(0, len(normalized) - test_files - docs_files)
    is_docs_only = int(bool(normalized) and docs_files == len(normalized))
    return {
        "primary_service": primary_service,
        "service_area_count": len(area_priority),
        "service_criticality": max(area_priority.values()),
        "file_extension_entropy": round(entropy, 4),
        "is_database_migration": int(bool(re.search(r"migration|migrate|schema|database|\bdb/|sql/", joined))),
        "is_infrastructure_change": int(bool(re.search(r"terraform|\.tf$|helm|kubernetes|k8s|docker|infra/|cloudformation|pulumi", joined))),
        "is_dependency_change": int(bool(re.search(r"package\.json|pyproject\.toml|requirements.*\.txt|pom\.xml|build\.gradle|go\.mod|cargo\.toml", joined))),
        "is_config_change": int(bool(re.search(r"\.ya?ml$|\.json$|\.toml$|\.ini$|\.cfg$|config/|settings", joined))),
        "is_security_change": int(bool(re.search(r"auth|oauth|oidc|sso|security|permission|rbac|credential|secret", joined))),
        "is_test_change": int(test_files > 0),
        "test_to_source_file_ratio": round(min(test_files / max(1, source_files), 3.0), 4),
        "is_docs_only": is_docs_only,
        "has_lockfile_change": int(bool(re.search(r"package-lock\.json|yarn\.lock|pnpm-lock|poetry\.lock|cargo\.lock|go\.sum", joined))),
    }


def add_static_features(runs: pd.DataFrame, commits: pd.DataFrame) -> pd.DataFrame:
    commit_columns = [
        "repo", "head_sha", "commit_details_available", "commit_authored_at", "commit_message", "commit_parent_count",
        "additions", "deletions", "changed_files", "file_paths_json", "files_truncated",
    ]
    for column in commit_columns:
        if column not in commits.columns:
            commits[column] = np.nan
    data = runs.merge(commits[commit_columns], on=["repo", "head_sha"], how="left")
    data["deployment_timestamp"] = pd.to_datetime(data["created_at"], utc=True, errors="coerce")
    data["commit_timestamp"] = pd.to_datetime(data["commit_authored_at"], utc=True, errors="coerce")
    data["commit_age_hours"] = ((data["deployment_timestamp"] - data["commit_timestamp"]).dt.total_seconds() / 3600).clip(lower=0, upper=24 * 30)
    data["commit_age_hours"] = data["commit_age_hours"].fillna(24.0)

    numeric_defaults = {"additions": 0, "deletions": 0, "changed_files": 0, "commit_parent_count": 1, "run_attempt": 1}
    for column, default in numeric_defaults.items():
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(default).clip(lower=0)
    data["code_churn"] = data["additions"] + data["deletions"]
    data["hour"] = data["deployment_timestamp"].dt.hour.fillna(12).astype(int)
    data["day_of_week"] = data["deployment_timestamp"].dt.dayofweek.fillna(0).astype(int)
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    data["is_friday"] = (data["day_of_week"] == 4).astype(int)
    data["is_off_hours"] = ((data["hour"] < 7) | (data["hour"] >= 20)).astype(int)
    data["branch_type"] = [branch_type(branch, default) for branch, default in zip(data["head_branch"], data["default_branch"])]
    data["workflow_kind"] = data["workflow_name"].map(workflow_kind)
    data["is_deployment_intent"] = data["candidate_type"].eq("explicit_deployment_intent").astype(int)
    messages = data["commit_message"].fillna("").astype(str)
    data["is_hotfix"] = (messages.str.contains(HOTFIX_TERMS) | data["branch_type"].eq("hotfix")).astype(int)
    data["is_revert"] = messages.str.contains(REVERT_TERMS).astype(int)
    data["is_breaking_change"] = messages.str.contains(BREAKING_TERMS).astype(int)

    path_features = pd.DataFrame([classify_paths(parse_file_paths(value)) for value in data["file_paths_json"]])
    data = pd.concat([data.reset_index(drop=True), path_features], axis=1)
    # Keep the latest retry of an identical workflow on an identical commit. This avoids
    # near-duplicate reruns leaking across the later temporal split.
    data = data.sort_values(["deployment_timestamp", "run_attempt", "deployment_id"])
    data = data.drop_duplicates(subset=["repo", "head_sha", "workflow_name"], keep="last")
    return data.sort_values(["deployment_timestamp", "deployment_id"]).reset_index(drop=True)


def keep_within(events: deque[dict[str, Any]], now: pd.Timestamp, days: int) -> list[dict[str, Any]]:
    cutoff = now - pd.Timedelta(days=days)
    while events and events[0]["timestamp"] < cutoff:
        events.popleft()
    return list(events)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def percentile_rank(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series([0.5] * len(values), index=values.index)
    return values.rank(method="average", pct=True)


def add_causal_history_and_labels(data: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add previous-only history features, then simulate only unavailable operational outcomes."""
    rng = np.random.default_rng(seed)
    repo_history: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    service_history: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    actor_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"deployments": 0.0, "failures": 0.0})
    last_deployment: dict[str, pd.Timestamp] = {}
    enriched: list[dict[str, Any]] = []

    for row in data.to_dict("records"):
        now = row["deployment_timestamp"]
        repo = row["repo"]
        service_key = (repo, row["primary_service"])
        history = keep_within(repo_history[repo], now, 30)
        service_events = keep_within(service_history[service_key], now, 30)

        history_7d = [event for event in history if event["timestamp"] >= now - pd.Timedelta(days=7)]
        history_14d = [event for event in history if event["timestamp"] >= now - pd.Timedelta(days=14)]
        history_24h = [event for event in history if event["timestamp"] >= now - pd.Timedelta(hours=24)]
        service_30d = service_events
        actor = actor_stats[(repo, row["actor_login"])]

        row["deployments_7d"] = len(history_7d)
        row["deployments_30d"] = len(history)
        row["recent_change_velocity_7d"] = sum(event["code_churn"] for event in history_7d)
        row["past_failed_runs_7d"] = sum(event["workflow_failed_observed"] for event in history_7d)
        row["past_failure_rate_30d"] = sum(event["workflow_failed_observed"] for event in history) / max(1, len(history))
        row["past_incidents_14d"] = sum(event["post_deploy_incident_7d"] for event in history_14d)
        row["past_rollbacks_30d"] = sum(event["rollback_within_24h"] for event in history)
        row["time_since_last_deployment_hours"] = (
            min((now - last_deployment[repo]).total_seconds() / 3600, 24 * 30) if repo in last_deployment else 24 * 30
        )
        row["actor_prior_deployments"] = actor["deployments"]
        row["actor_prior_failure_rate"] = actor["failures"] / max(1.0, actor["deployments"])
        row["service_deployments_30d"] = len(service_30d)
        row["service_incident_rate_30d"] = sum(event["post_deploy_incident_7d"] for event in service_30d) / max(1, len(service_30d))
        row["on_call_load_24h"] = len(history_24h)

        availability_probability = 0.98 - 0.18 * row["is_off_hours"] - 0.08 * row["is_weekend"] - 0.025 * min(row["on_call_load_24h"], 6)
        if row["branch_type"] == "hotfix":
            availability_probability -= 0.05
        row["on_call_available"] = int(rng.random() < max(0.45, availability_probability))
        response_base = 18 + 3.0 * min(row["on_call_load_24h"], 8) + 12 * row["is_off_hours"] + 7 * row["is_weekend"]
        row["expected_on_call_response_minutes"] = round(response_base if row["on_call_available"] else response_base + 65 + rng.uniform(0, 35), 2)

        churn_risk = min(math.log1p(row["code_churn"]) / 8.5, 1.4)
        breadth_risk = min(row["changed_files"] / 30.0, 1.3)
        velocity_risk = min(math.log1p(row["recent_change_velocity_7d"]) / 10.0, 1.2)
        history_risk = min(row["past_failed_runs_7d"] / 4.0, 1.5) + min(row["past_incidents_14d"] / 3.0, 1.5)
        experience_risk = 1.0 if row["actor_prior_deployments"] < 3 else min(row["actor_prior_failure_rate"] * 2.0, 0.8)
        recovery_risk = 0.45 if row["time_since_last_deployment_hours"] < 1 else 0.0
        low_test_risk = 0.28 if not row["is_docs_only"] and row["is_test_change"] == 0 else 0.0

        logit = (
            -2.05
            + 0.65 * churn_risk
            + 0.45 * breadth_risk
            + 0.18 * max(row["service_criticality"] - 2, 0)
            + 0.75 * row["is_database_migration"]
            + 0.50 * row["is_infrastructure_change"]
            + 0.28 * row["is_dependency_change"]
            + 0.20 * row["is_config_change"]
            + 0.62 * row["is_security_change"]
            + 0.35 * row["is_hotfix"]
            + 0.20 * row["is_breaking_change"]
            + 0.22 * row["is_off_hours"]
            + 0.12 * row["is_friday"]
            + 0.32 * velocity_risk
            + 0.38 * history_risk
            + 0.26 * min(row["past_rollbacks_30d"] / 3.0, 1.0)
            + 0.28 * experience_risk
            + 0.33 * (1 - row["on_call_available"])
            + 0.16 * min(row["expected_on_call_response_minutes"] / 90.0, 1.5)
            + recovery_risk
            + low_test_risk
            - 1.0 * row["is_docs_only"]
            - 0.12 * min(row["test_to_source_file_ratio"], 1.0)
        )
        # This shock is deliberately not exposed as a feature: it represents unknown live-system conditions.
        latent_shock = float(rng.normal(0, 0.28))
        probability = sigmoid(logit)
        row["risk_probability"] = round(probability, 6)
        row["latent_reliability_shock"] = round(latent_shock, 6)
        row["risk_score"] = round(100 * sigmoid(logit + latent_shock), 2)
        incident_probability = sigmoid(logit + latent_shock + 0.15)
        row["post_deploy_incident_7d"] = int(rng.random() < incident_probability)
        rollback_probability = sigmoid(logit + latent_shock - 1.35) if row["post_deploy_incident_7d"] else 0.0
        row["rollback_within_24h"] = int(rng.random() < rollback_probability)

        historical_event = {
            "timestamp": now,
            "code_churn": float(row["code_churn"]),
            "workflow_failed_observed": int(row["workflow_failed_observed"]),
            "post_deploy_incident_7d": int(row["post_deploy_incident_7d"]),
            "rollback_within_24h": int(row["rollback_within_24h"]),
        }
        repo_history[repo].append(historical_event)
        service_history[service_key].append(historical_event)
        last_deployment[repo] = now
        actor["deployments"] += 1
        actor["failures"] += int(row["workflow_failed_observed"])
        enriched.append(row)

    result = pd.DataFrame(enriched)
    global_percentile = percentile_rank(result["risk_score"])
    repo_counts = result.groupby("repo")["risk_score"].transform("count")
    domain_counts = result.groupby("repo_domain")["risk_score"].transform("count")
    repo_percentile = result.groupby("repo")["risk_score"].transform(percentile_rank).where(repo_counts >= 20, global_percentile)
    domain_percentile = result.groupby("repo_domain")["risk_score"].transform(percentile_rank).where(
        domain_counts >= 20, global_percentile
    )
    result["calibrated_risk_index"] = (100 * (0.55 * global_percentile + 0.30 * repo_percentile + 0.15 * domain_percentile)).round(2)
    lower_cutoff = float(result["calibrated_risk_index"].quantile(LOW_MEDIUM_QUANTILE))
    upper_cutoff = float(result["calibrated_risk_index"].quantile(MEDIUM_HIGH_QUANTILE))
    result["risk_level"] = np.select(
        [result["calibrated_risk_index"] < lower_cutoff, result["calibrated_risk_index"] < upper_cutoff],
        ["Low", "Medium"],
        default="High",
    )
    return result, {"low_medium_cutoff": round(lower_cutoff, 2), "medium_high_cutoff": round(upper_cutoff, 2)}


def split_sizes(total_rows: int) -> tuple[int, int, int]:
    if total_rows <= 0:
        return 0, 0, 0
    if total_rows < 3:
        return total_rows, 0, 0
    validation_rows = max(1, int(round(total_rows * 0.15)))
    test_rows = max(1, int(round(total_rows * 0.15)))
    train_rows = total_rows - validation_rows - test_rows
    while train_rows < 1:
        if validation_rows >= test_rows and validation_rows > 1:
            validation_rows -= 1
        elif test_rows > 1:
            test_rows -= 1
        train_rows = total_rows - validation_rows - test_rows
    return train_rows, validation_rows, test_rows


def add_temporal_splits(data: pd.DataFrame, strategy: str = "stratified_temporal") -> pd.DataFrame:
    ordered = data.sort_values(["deployment_timestamp", "deployment_id"]).reset_index(drop=True).copy()
    if strategy == "temporal":
        train_end = int(len(ordered) * 0.70)
        validation_end = int(len(ordered) * 0.85)
        ordered["split"] = "test"
        ordered.loc[: max(train_end - 1, 0), "split"] = "train"
        ordered.loc[train_end: max(validation_end - 1, train_end), "split"] = "validation"
        return ordered

    if strategy != "stratified_temporal":
        raise ValueError(f"Unsupported split strategy: {strategy}")

    ordered["split"] = "train"
    for risk_level in RISK_LEVELS:
        indices = ordered.index[ordered["risk_level"].eq(risk_level)].tolist()
        train_rows, validation_rows, _ = split_sizes(len(indices))
        validation_start = train_rows
        test_start = train_rows + validation_rows
        ordered.loc[indices[validation_start:test_start], "split"] = "validation"
        ordered.loc[indices[test_start:], "split"] = "test"
    return ordered


def validate_dataset(master: pd.DataFrame, model_data: pd.DataFrame, split_strategy: str) -> dict[str, Any]:
    if master["deployment_id"].duplicated().any():
        raise ValueError("Duplicate deployment_id values detected.")
    if master.duplicated(subset=["repo", "head_sha", "workflow_name"]).any():
        raise ValueError("Near-duplicate workflow retries remain in the dataset.")
    if not master["deployment_timestamp"].is_monotonic_increasing:
        raise ValueError("Master dataset is not chronologically ordered.")
    forbidden = FORBIDDEN_MODEL_COLUMNS.intersection(model_data.columns)
    if forbidden:
        raise ValueError(f"Leakage columns found in model dataset: {sorted(forbidden)}")
    missing_features = sorted(set(MODEL_FEATURES) - set(model_data.columns))
    if missing_features:
        raise ValueError(f"Model features missing from output: {missing_features}")
    if model_data[MODEL_FEATURES].isna().any().any():
        bad_columns = model_data[MODEL_FEATURES].columns[model_data[MODEL_FEATURES].isna().any()].tolist()
        raise ValueError(f"Null model features remain: {bad_columns}")

    split_dates = master.groupby("split")["deployment_timestamp"].agg(["min", "max", "count"]).reindex(SPLIT_LEVELS)
    split_class_counts = pd.crosstab(master["split"], master["risk_level"]).reindex(
        index=SPLIT_LEVELS, columns=RISK_LEVELS, fill_value=0
    )
    missing_split_classes = [
        f"{split}:{risk_level}"
        for split in SPLIT_LEVELS
        for risk_level in RISK_LEVELS
        if int(split_class_counts.loc[split, risk_level]) == 0
    ]
    if missing_split_classes:
        raise ValueError(f"Every split must contain every risk class. Missing: {missing_split_classes}")

    if split_strategy == "temporal":
        if split_dates.loc["train", "max"] > split_dates.loc["validation", "min"]:
            raise ValueError("Train/validation temporal ordering is invalid.")
        if split_dates.loc["validation", "max"] > split_dates.loc["test", "min"]:
            raise ValueError("Validation/test temporal ordering is invalid.")
    else:
        for risk_level in RISK_LEVELS:
            per_class_dates = master.loc[master["risk_level"].eq(risk_level)].groupby("split")["deployment_timestamp"].agg(["min", "max"])
            if per_class_dates.loc["train", "max"] > per_class_dates.loc["validation", "min"]:
                raise ValueError(f"{risk_level} train/validation temporal ordering is invalid.")
            if per_class_dates.loc["validation", "max"] > per_class_dates.loc["test", "min"]:
                raise ValueError(f"{risk_level} validation/test temporal ordering is invalid.")

    class_counts = master["risk_level"].value_counts().reindex(RISK_LEVELS, fill_value=0).to_dict()
    if min(class_counts.values()) < max(10, len(master) * 0.08):
        raise ValueError(f"Risk labels are too imbalanced: {class_counts}")

    dominant_repo_by_class: dict[str, dict[str, Any]] = {}
    dataset_warnings: list[str] = []
    for risk_level in RISK_LEVELS:
        counts = master.loc[master["risk_level"].eq(risk_level), "repo"].value_counts()
        top_repo = str(counts.index[0])
        top_rows = int(counts.iloc[0])
        top_share = round(top_rows / max(1, int(class_counts[risk_level])), 4)
        dominant_repo_by_class[risk_level] = {"repo": top_repo, "rows": top_rows, "share": top_share}
        if top_share > 0.70:
            dataset_warnings.append(f"{risk_level} labels are still concentrated in {top_repo} ({top_share:.0%}).")

    return {
        "row_count": int(len(master)),
        "repo_count": int(master["repo"].nunique()),
        "split_strategy": split_strategy,
        "time_range": {"start": master["deployment_timestamp"].min().isoformat(), "end": master["deployment_timestamp"].max().isoformat()},
        "risk_class_counts": {key: int(value) for key, value in class_counts.items()},
        "risk_class_by_split": {
            split: {risk_level: int(split_class_counts.loc[split, risk_level]) for risk_level in RISK_LEVELS}
            for split in SPLIT_LEVELS
        },
        "dominant_repo_by_class": dominant_repo_by_class,
        "dataset_warnings": dataset_warnings,
        "split_summary": {
            split: {
                "rows": int(values["count"]),
                "start": values["min"].isoformat(),
                "end": values["max"].isoformat(),
            }
            for split, values in split_dates.to_dict("index").items()
        },
        "commit_details_coverage": round(float(master["commit_details_available"].fillna(0).mean()), 4),
        "observed_workflow_failure_rate": round(float(master["workflow_failed_observed"].mean()), 4),
        "simulated_incident_rate": round(float(master["post_deploy_incident_7d"].mean()), 4),
        "simulated_rollback_rate": round(float(master["rollback_within_24h"].mean()), 4),
        "null_model_feature_count": 0,
    }


def write_feature_dictionary() -> None:
    rows = []
    for feature in MODEL_FEATURES:
        feature_type, description = FEATURE_DICTIONARY[feature]
        rows.append({"feature": feature, "type": feature_type, "available_before_deployment": True, "description": description})
    pd.DataFrame(rows).to_csv(REPORTS / "feature_dictionary.csv", index=False)


def write_data_card(master: pd.DataFrame, thresholds: dict[str, float], split_strategy: str) -> None:
    card = f"""# DeploySense Dataset Card

## Purpose
This dataset supports a pre-deployment classifier that assigns **Low**, **Medium**, or **High** risk to a candidate release. One row is one completed GitHub Actions workflow run that qualifies as an explicit deployment workflow or a mainline delivery proxy.

## Sources
- Public GitHub REST API workflow-run metadata.
- Public GitHub REST API commit metadata and changed file paths.
- Causally generated operational context for facts GitHub does not publish: on-call availability, post-deployment incidents, and rollbacks.

## Leakage policy
The model dataset excludes the current run conclusion, completion timestamp, duration, synthetic outcome columns, risk score, calibrated risk index, and latent reliability shock. Historical failure and incident features contain only records strictly earlier than the candidate timestamp.

## Target policy
`risk_level` is derived from a calibrated pre-deployment risk index with thresholds calculated for this collection: Low < {thresholds['low_medium_cutoff']}, Medium < {thresholds['medium_high_cutoff']}, High otherwise. The index blends global, repository-relative, and domain-relative risk so the target is less dominated by one unusually busy repository.

## Dataset snapshot
- Rows: {len(master)}
- Public repositories: {master['repo'].nunique()}
- Time range: {master['deployment_timestamp'].min().isoformat()} to {master['deployment_timestamp'].max().isoformat()}
- Split strategy: {split_strategy}; 70% train, 15% validation, 15% test within each risk class when stratified.

## Appropriate use
Use `data/processed/model_features.csv` for model training. Use `deployment_risk_master.csv` only for auditing and explanation work; it contains post-deployment outcomes and must not be passed directly to a model.

## Limitations
GitHub Actions workflow runs are a public proxy for deployments, not an organization's private production-deployment record. Operational fields are explicitly synthetic and intended for a hackathon simulation, not for real production approval without organizational incident and paging data.
"""
    (REPORTS / "dataset_card.md").write_text(card, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.runs_per_repo < 25:
        raise ValueError("Use at least 25 runs per repository for meaningful historical features.")
    ensure_directories()
    repositories = load_repositories()
    runs_cache = DATA_RAW / "workflow_runs.csv"

    if args.rebuild_from_cache:
        if not runs_cache.exists():
            raise FileNotFoundError("workflow_runs.csv does not exist. Run a normal collection first.")
        runs = pd.read_csv(runs_cache)
        commits = load_or_collect_commit_details(None, runs, refresh=False, workers=args.workers)
    else:
        load_dotenv(ROOT / ".env")
        token = os.getenv("GITHUB_TOKEN", "").strip()
        if not token:
            raise EnvironmentError(
                "GITHUB_TOKEN is missing. Add it to .env in the project root using .env.example as the format."
            )
        client = GitHubClient(token)
        if runs_cache.exists() and not args.refresh:
            runs = pd.read_csv(runs_cache)
            print(f"[cache] Reusing {len(runs)} workflow-run rows. Use --refresh to recollect GitHub data.")
        else:
            metadata = collect_repository_metadata(client, repositories)
            runs = collect_workflow_runs(client, metadata, args.runs_per_repo, args.max_pages_per_repo)
            runs.to_csv(runs_cache, index=False)
        commits = load_or_collect_commit_details(client, runs, refresh=args.refresh, workers=args.workers)

    static_data = add_static_features(runs, commits)
    master, thresholds = add_causal_history_and_labels(static_data, args.seed)
    master = add_temporal_splits(master, args.split_strategy)

    categorical_features = [feature for feature in MODEL_FEATURES if FEATURE_DICTIONARY[feature][0] == "categorical"]
    numeric_features = [feature for feature in MODEL_FEATURES if feature not in categorical_features]
    for column in categorical_features:
        master[column] = master[column].fillna("unknown").astype(str)
    for column in numeric_features:
        master[column] = pd.to_numeric(master[column], errors="coerce").fillna(0.0)

    model_data = master[["deployment_id", "deployment_timestamp", "split", "risk_level", *MODEL_FEATURES]].copy()
    quality = validate_dataset(master, model_data, args.split_strategy)
    quality["label_thresholds"] = thresholds
    quality["model_feature_count"] = len(MODEL_FEATURES)
    quality["collection_seed"] = args.seed

    master.to_csv(DATA_PROCESSED / "deployment_risk_master.csv", index=False)
    model_data.to_csv(DATA_PROCESSED / "model_features.csv", index=False)
    for split in ("train", "validation", "test"):
        training_frame = model_data.loc[model_data["split"] == split, ["risk_level", *MODEL_FEATURES]]
        training_frame.to_csv(DATA_PROCESSED / f"{split}.csv", index=False)
    write_feature_dictionary()
    write_data_card(master, thresholds, args.split_strategy)
    (REPORTS / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (REPORTS / "labeling_policy.json").write_text(
        json.dumps(
            {
                "target": "risk_level",
                "class_order": ["Low", "Medium", "High"],
                "thresholds": thresholds,
                "method": "Causal synthetic risk simulation calibrated with global, repository-relative, and domain-relative percentiles.",
                "split_strategy": args.split_strategy,
                "forbidden_model_columns": sorted(FORBIDDEN_MODEL_COLUMNS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (REPORTS / "model_training_schema.json").write_text(
        json.dumps(
            {
                "target": "risk_level",
                "feature_columns": MODEL_FEATURES,
                "categorical_feature_columns": categorical_features,
                "metadata_columns_in_model_features_csv": ["deployment_id", "deployment_timestamp", "split"],
                "split_files": "train.csv, validation.csv, and test.csv contain only the target and feature columns.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDataset build complete.")
    print(f"  Master audit dataset: {DATA_PROCESSED / 'deployment_risk_master.csv'}")
    print(f"  Leakage-free model data: {DATA_PROCESSED / 'model_features.csv'}")
    print(f"  Quality report: {REPORTS / 'data_quality_report.json'}")
    print(f"  Risk-class counts: {quality['risk_class_counts']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\n[error] {error}", file=sys.stderr)
        raise SystemExit(1)