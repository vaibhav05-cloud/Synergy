"""
github_client.py

Reusable GitHub API client for the dataset collection pipeline.
"""

import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    GITHUB_TOKEN,
    GITHUB_API_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
)


class GitHubClient:
    """Reusable GitHub REST API client."""

    def __init__(self):

        self.session = requests.Session()

        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20,
            pool_maxsize=20,
        )

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update({
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Dataset-Collector"
        })

    # -----------------------------------------------------

    def _request(self, method, endpoint, params=None):

        url = f"{GITHUB_API_URL}{endpoint}"

        for attempt in range(MAX_RETRIES):

            try:

                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.status_code == 200:
                    return response.json()

                # Rate limit
                if response.status_code in [403, 429]:

                    remaining = response.headers.get(
                        "X-RateLimit-Remaining", "1"
                    )

                    if remaining == "0":

                        reset = int(
                            response.headers.get(
                                "X-RateLimit-Reset",
                                time.time() + 60
                            )
                        )

                        wait_time = max(
                            reset - int(time.time()) + 2,
                            2
                        )

                        print(
                            f"[Rate Limit] Waiting {wait_time} sec..."
                        )

                        time.sleep(wait_time)
                        continue

                if response.status_code >= 500:

                    print(
                        f"[Retry {attempt+1}/{MAX_RETRIES}] "
                        f"Server Error {response.status_code}"
                    )

                    time.sleep(RETRY_DELAY)

                    continue

                print(
                    f"[ERROR {response.status_code}] "
                    f"{endpoint}"
                )

                return None

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ):

                print(
                    f"[Retry {attempt+1}/{MAX_RETRIES}] "
                    "Connection issue..."
                )

                time.sleep(RETRY_DELAY)

            except Exception as e:

                print(e)

                time.sleep(RETRY_DELAY)

        return None

    # -----------------------------------------------------

    def get(self, endpoint, params=None):

        return self._request(
            "GET",
            endpoint,
            params=params,
        )

    # -----------------------------------------------------

    def get_paginated(self, endpoint, params=None):

        if params is None:
            params = {}

        page = 1

        items = []

        while True:

            params["page"] = page

            data = self.get(
                endpoint,
                params=params,
            )

            if not data:

                break

            if not isinstance(data, list):

                break

            items.extend(data)

            if len(data) < params.get("per_page", 100):

                break

            page += 1

        return items