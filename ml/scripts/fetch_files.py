"""
fetch_files.py

Fetch repository file tree with rich metadata.

Output:
dataset/raw/files/<owner_repo>.json
"""

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from github_client import GitHubClient
from config import (
    REPOSITORIES_FILE,
    FILES_DIR,
    MAX_FILES_PER_REPO,
)

client = GitHubClient()

# -------------------------------------------------------
# Extension -> Language Mapping
# -------------------------------------------------------

LANGUAGE_MAP = {
    ".py": "Python",
    ".java": "Java",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".c": "C",
    ".cs": "C#",
    ".go": "Go",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".jsx": "React",
    ".tsx": "React",
    ".php": "PHP",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sql": "SQL",
    ".sh": "Shell",
    ".ipynb": "Jupyter Notebook",
}

CONFIG_FILES = {
    "package.json","package-lock.json","requirements.txt",
    "Dockerfile","docker-compose.yml","pom.xml",
    "build.gradle","gradle.properties",".gitignore",
    ".gitattributes",".editorconfig","Makefile",
    "CMakeLists.txt","setup.py","pyproject.toml",
    "Cargo.toml","Cargo.lock",
}

DOC_FILES = {
    "README.md",
    "README",
    "LICENSE",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
}

BINARY_EXTENSIONS = {
    ".png",".jpg",".jpeg",".gif",".bmp",".ico",".svg",
    ".pdf",".zip",".rar",".7z",".tar",".gz",
    ".exe",".dll",".so",".class",".jar",
    ".mp3",".mp4",".avi",".mov",".wav",
    ".ttf",".otf",".woff",".woff2",
}


def load_repositories():

    with open(REPOSITORIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_category(path, extension):

    lower = path.lower()

    if extension in BINARY_EXTENSIONS:
        return "Binary"

    if "test" in lower or "spec" in lower:
        return "Test"

    if "docs/" in lower:
        return "Documentation"

    if extension in {
        ".py",".java",".cpp",".c",".js",".ts",
        ".go",".php",".rb",".rs",".kt",".swift",".cs"
    }:
        return "Source Code"

    if extension in {
        ".json",".yaml",".yml",".xml",".toml",
        ".ini",".cfg"
    }:
        return "Configuration"

    if extension in {".md",".txt"}:
        return "Documentation"

    return "Other"


def fetch_repository_files(owner, repo, branch):

    data = client.get(
        f"/repos/{owner}/{repo}/git/trees/{branch}",
        params={"recursive": 1},
    )

    if not data:
        return []

    tree = data.get("tree", [])

    result = []

    for item in tree:

        if item["type"] != "blob":
            continue

        file = Path(item["path"])

        extension = file.suffix.lower()

        filename = file.name

        result.append({

            "path": item["path"],

            "filename": filename,

            "parent_directory": str(file.parent),

            "extension": extension,

            "language": LANGUAGE_MAP.get(extension, "Unknown"),

            "category": get_category(
                item["path"],
                extension,
            ),

            "directory_depth": len(file.parts)-1,

            "size": item.get("size"),

            "sha": item.get("sha"),

            "git_url": item.get("url"),

            "is_test_file":
                "test" in item["path"].lower()
                or "spec" in item["path"].lower(),

            "is_documentation":
                filename in DOC_FILES
                or "docs/" in item["path"].lower(),

            "is_config_file":
                filename in CONFIG_FILES,

            "is_hidden":
                filename.startswith("."),

            "is_binary":
                extension in BINARY_EXTENSIONS,

        })

        if len(result) >= MAX_FILES_PER_REPO:
            break

    return result


def save_files(owner, repo, files):

    output = FILES_DIR / f"{owner}_{repo}.json"

    with open(output, "w", encoding="utf-8") as f:

        json.dump(
            files,
            f,
            indent=4,
            ensure_ascii=False,
        )


def process_repository(index, total, repository):

    owner = repository["owner"]
    repo = repository["name"]
    branch = repository["default_branch"]

    print(f"[{index}/{total}] {owner}/{repo}")

    try:

        files = fetch_repository_files(
            owner,
            repo,
            branch,
        )

        save_files(
            owner,
            repo,
            files,
        )

        print(
            f"✓ [{index}/{total}] {owner}/{repo} -> {len(files)} files"
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

    print("\n✅ All repository files fetched successfully!")


if __name__ == "__main__":
    main()