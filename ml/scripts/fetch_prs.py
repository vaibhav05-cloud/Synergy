"""
fetch_prs.py

Fetch Pull Requests for all repositories.

Output:
dataset/raw/prs/<owner_repo>.json
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from github_client import GitHubClient
from config import (
    REPOSITORIES_FILE,
    PRS_DIR,
    PER_PAGE,
    MAX_PRS_PER_REPO,
)

client = GitHubClient()


def load_repositories():

    with open(REPOSITORIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_pull_requests(owner, repo):

    prs = []
    page = 1

    while len(prs) < MAX_PRS_PER_REPO:

        params = {
            "state": "all",
            "per_page": PER_PAGE,
            "page": page,
        }

        try:

            data = client.get(
                f"/repos/{owner}/{repo}/pulls",
                params=params,
            )

        except Exception as e:

            print(f"Error fetching PRs for {owner}/{repo}: {e}")
            break

        if not data or not isinstance(data, list):
            break

        for pr in data:

            pr_info = {

                "id": pr.get("id"),

                "number": pr.get("number"),

                "title": pr.get("title"),

                "state": pr.get("state"),

                "draft": pr.get("draft"),

                "merged": pr.get("merged_at") is not None,

                "created_at": pr.get("created_at"),

                "updated_at": pr.get("updated_at"),

                "closed_at": pr.get("closed_at"),

                "merged_at": pr.get("merged_at"),

                "author": (
                    pr.get("user", {})
                    .get("login")
                ),

                "base_branch": (
                    pr.get("base", {})
                    .get("ref")
                ),

                "head_branch": (
                    pr.get("head", {})
                    .get("ref")
                ),

                "commits": pr.get("commits"),

                "changed_files": pr.get("changed_files"),

                "additions": pr.get("additions"),

                "deletions": pr.get("deletions"),

                "comments": pr.get("comments"),

                "review_comments": pr.get("review_comments"),

                "labels": [
                    label.get("name")
                    for label in pr.get("labels", [])
                ],

                "url": pr.get("html_url"),
            }

            prs.append(pr_info)

            if len(prs) >= MAX_PRS_PER_REPO:
                break

        page += 1

    return prs


def save_prs(owner, repo, prs):

    filename = f"{owner}_{repo}.json"

    output_path = PRS_DIR / filename

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            prs,
            f,
            indent=4,
            ensure_ascii=False,
        )


def process_repository(index, total, repository):

    owner = repository["owner"]
    repo = repository["name"]

    print(f"[{index}/{total}] Fetching PRs : {owner}/{repo}")

    try:

        prs = fetch_pull_requests(owner, repo)

        save_prs(owner, repo, prs)

        print(
            f"✓ [{index}/{total}] {owner}/{repo} -> {len(prs)} PRs"
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

    print("\n✅ All Pull Requests fetched successfully!")


if __name__ == "__main__":
    main()