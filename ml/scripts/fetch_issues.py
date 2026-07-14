"""
fetch_issues.py

Fetch GitHub Issues (excluding Pull Requests) for all repositories.

Output:
dataset/raw/issues/<owner_repo>.json
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from github_client import GitHubClient
from config import (
    REPOSITORIES_FILE,
    ISSUES_DIR,
    PER_PAGE,
    MAX_ISSUES_PER_REPO,
)

client = GitHubClient()


def load_repositories():

    with open(REPOSITORIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_issues(owner, repo):

    issues = []
    page = 1

    while len(issues) < MAX_ISSUES_PER_REPO:

        params = {
            "state": "all",
            "per_page": PER_PAGE,
            "page": page,
        }

        try:
            data = client.get(
                f"/repos/{owner}/{repo}/issues",
                params=params,
            )

        except Exception as e:
            print(f"Error fetching issues for {owner}/{repo}: {e}")
            break

        if not data or not isinstance(data, list):
            break

        for issue in data:

            # Skip pull requests
            if "pull_request" in issue:
                continue

            issue_info = {

                "id": issue.get("id"),

                "number": issue.get("number"),

                "title": issue.get("title"),

                "state": issue.get("state"),

                "created_at": issue.get("created_at"),

                "updated_at": issue.get("updated_at"),

                "closed_at": issue.get("closed_at"),

                "author": (
                    issue.get("user", {})
                    .get("login")
                ),

                "comments": issue.get("comments"),

                "labels": [
                    label.get("name")
                    for label in issue.get("labels", [])
                ],

                "assignees": [
                    user.get("login")
                    for user in issue.get("assignees", [])
                ],

                "locked": issue.get("locked"),

                "body": issue.get("body"),

                "url": issue.get("html_url"),
            }

            issues.append(issue_info)

            if len(issues) >= MAX_ISSUES_PER_REPO:
                break

        page += 1

    return issues


def save_issues(owner, repo, issues):

    filename = f"{owner}_{repo}.json"

    output_path = ISSUES_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            issues,
            f,
            indent=4,
            ensure_ascii=False,
        )


def process_repository(index, total, repository):

    owner = repository["owner"]
    repo = repository["name"]

    print(f"[{index}/{total}] Fetching Issues : {owner}/{repo}")

    try:

        issues = fetch_issues(owner, repo)

        save_issues(owner, repo, issues)

        print(
            f"✓ [{index}/{total}] {owner}/{repo} -> {len(issues)} issues"
        )

    except Exception as e:

        print(
            f"✗ [{index}/{total}] {owner}/{repo} -> {e}"
        )


def main():

    repositories = load_repositories()

    total = len(repositories)

    print(f"\nFound {total} repositories\n")

    workers = 12

    with ThreadPoolExecutor(max_workers=workers) as executor:

        futures = []

        for index, repository in enumerate(repositories, start=1):

            futures.append(
                executor.submit(
                    process_repository,
                    index,
                    total,
                    repository,
                )
            )

        for future in as_completed(futures):
            future.result()

    print("\n✅ All issues fetched successfully!")


if __name__ == "__main__":
    main()