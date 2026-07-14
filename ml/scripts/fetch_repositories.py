"""
fetch_repositories.py

Fetch GitHub repositories across multiple star tiers and save them to:
dataset/raw/repositories.json

Star tiers are used deliberately (instead of a single "stars:>500"
filter) so the dataset contains a genuine mix of highly-maintained,
mid-tier, and smaller/newer repositories. Without this, every repo
in the dataset is "already famous", which biases the model toward
predicting popularity instead of actual deployment readiness.
"""

import json

from github_client import GitHubClient
from config import (
    REPOSITORIES_FILE,
    MAX_REPOSITORIES,
    PER_PAGE,
    SUPPORTED_LANGUAGES,
)

client = GitHubClient()

# ==========================================================
# Star Tiers (for diversity / reduced selection bias)
# ==========================================================
# Each tier is a separate GitHub search qualifier. Repos are
# collected roughly evenly across tiers per language, so the
# final dataset isn't dominated by only the most-famous repos.

STAR_TIERS = [
    "stars:>2000",        # flagship / highly established projects
    "stars:300..2000",    # solid, actively maintained mid-tier
    "stars:20..300",      # smaller / emerging / less mature projects
]


def fetch_repositories():

    repositories = []
    seen = set()

    print("=" * 70)
    print("Fetching GitHub Repositories (multi-tier)")
    print("=" * 70)

    per_language_limit = max(
        MAX_REPOSITORIES // len(SUPPORTED_LANGUAGES),
        1,
    )

    per_tier_limit = max(
        per_language_limit // len(STAR_TIERS),
        1,
    )

    for language in SUPPORTED_LANGUAGES:

        print(f"\nLanguage : {language}")

        for tier in STAR_TIERS:

            print(f"  Tier : {tier}")

            collected = 0
            page = 1

            while collected < per_tier_limit:

                params = {
                    "q": (
                        f"language:{language} "
                        f"{tier} "
                        "archived:false "
                        "fork:false "
                        "pushed:>2024-01-01"
                    ),
                    "sort": "stars",
                    "order": "desc",
                    "per_page": PER_PAGE,
                    "page": page,
                }

                data = client.get(
                    "/search/repositories",
                    params=params,
                )

                if not data or "items" not in data:
                    break

                items = data["items"]

                if len(items) == 0:
                    break

                for repo in items:

                    full_name = repo["full_name"]

                    if full_name in seen:
                        continue

                    if repo.get("archived"):
                        continue

                    if repo.get("fork"):
                        continue

                    seen.add(full_name)

                    repositories.append({

                        "id": repo["id"],

                        "name": repo["name"],

                        "full_name": full_name,

                        "owner": repo["owner"]["login"],

                        "language": repo["language"],

                        "description": repo["description"],

                        "stars": repo["stargazers_count"],

                        "forks": repo["forks_count"],

                        "watchers": repo["watchers_count"],

                        "open_issues": repo["open_issues_count"],

                        "default_branch": repo["default_branch"],

                        "created_at": repo["created_at"],

                        "updated_at": repo["updated_at"],

                        "size": repo["size"],

                        "license": (
                            repo["license"]["name"]
                            if repo["license"]
                            else None
                        ),

                        "html_url": repo["html_url"],

                        "star_tier": tier,

                    })

                    collected += 1

                    print(
                        f"    [{len(repositories)}/{MAX_REPOSITORIES}] "
                        f"{full_name}"
                    )

                    if len(repositories) >= MAX_REPOSITORIES:
                        return repositories

                    if collected >= per_tier_limit:
                        break

                page += 1

    return repositories


def save_repositories(repositories):

    with open(
        REPOSITORIES_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            repositories,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print(f"Saved {len(repositories)} repositories")
    print(REPOSITORIES_FILE)
    print("=" * 70)


def main():

    repositories = fetch_repositories()

    save_repositories(repositories)


if __name__ == "__main__":
    main()