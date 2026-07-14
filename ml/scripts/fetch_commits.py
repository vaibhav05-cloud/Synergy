"""
fetch_commits.py

Fetch commit history for all repositories present in
dataset/raw/repositories.json

Output:
dataset/raw/commits/<owner_repo>.json
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from github_client import GitHubClient
from config import (
    REPOSITORIES_FILE,
    COMMITS_DIR,
    PER_PAGE,
    MAX_COMMITS_PER_REPO,
)

client = GitHubClient()


def load_repositories():

    if not REPOSITORIES_FILE.exists():
        raise FileNotFoundError(
            f"{REPOSITORIES_FILE} not found.\n"
            "Run fetch_repositories.py first."
        )

    with open(REPOSITORIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_commits(owner, repo):

    commits = []
    page = 1

    while len(commits) < MAX_COMMITS_PER_REPO:

        params = {
            "per_page": PER_PAGE,
            "page": page,
        }

        try:
            data = client.get(
                f"/repos/{owner}/{repo}/commits",
                params=params,
            )

        except Exception as e:
            print(f"Error fetching {owner}/{repo}: {e}")
            break

        if not data or not isinstance(data, list):
            break

        for commit in data:

            commit_info = {

                "sha": commit.get("sha"),

                "author": (
                    commit.get("commit", {})
                    .get("author", {})
                    .get("name")
                ),

                "email": (
                    commit.get("commit", {})
                    .get("author", {})
                    .get("email")
                ),

                "date": (
                    commit.get("commit", {})
                    .get("author", {})
                    .get("date")
                ),

                "message": (
                    commit.get("commit", {})
                    .get("message")
                ),

                "url": commit.get("html_url"),
            }

            commits.append(commit_info)

            if len(commits) >= MAX_COMMITS_PER_REPO:
                break

        page += 1

    return commits


def save_commits(owner, repo, commits):

    filename = f"{owner}_{repo}.json"

    output_path = COMMITS_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            commits,
            f,
            indent=4,
            ensure_ascii=False,
        )


def process_repository(index, total, repository):

    owner = repository["owner"]
    repo = repository["name"]

    print(f"[{index}/{total}] Fetching {owner}/{repo}")

    try:

        commits = fetch_commits(owner, repo)

        save_commits(owner, repo, commits)

        print(
            f"✓ [{index}/{total}] {owner}/{repo} -> {len(commits)} commits"
        )

    except Exception as e:

        print(
            f"✗ [{index}/{total}] {owner}/{repo} -> {e}"
        )


def main():

    repositories = load_repositories()

    total = len(repositories)

    print(f"\nFound {total} repositories\n")

    # Change between 8-16 depending on your internet
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

    print("\n✅ All repositories processed successfully!")


if __name__ == "__main__":
    main()