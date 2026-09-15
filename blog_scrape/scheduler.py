"""
Scheduler for blog_scraper.py.

Runs the scraper once immediately on container startup, then repeats
daily at SCRAPE_TIME (UTC). Calls blog_scraper.py as a subprocess so its
existing argparse CLI and logic stay untouched.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
import schedule
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCRAPE_TIME = os.environ.get("SCRAPE_TIME", "06:00")  # HH:MM, 24h, UTC
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "data/blog_posts.csv")
PAGES = os.environ.get("PAGES", "10")

# Path to the existing scraper script, relative to /app (the container WORKDIR)
SCRAPER_SCRIPT = os.environ.get("SCRAPER_SCRIPT", "blog_scrape/blog_scraper.py")


def run_scraper():
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"\n{'=' * 60}")
    print(f"[{timestamp}] Starting scrape run")
    print("=" * 60, flush=True)

    try:
        subprocess.run(
            [
                sys.executable,
                SCRAPER_SCRIPT,
                "--output", OUTPUT_PATH,
                "--pages", str(PAGES),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Scraper exited with error code {e.returncode}", file=sys.stderr)


if __name__ == "__main__":
    # Run once immediately so you get feedback right after `docker compose up`
    run_scraper()

    schedule.every().day.at(SCRAPE_TIME).do(run_scraper)

    print(f"\nScheduler active. Next runs daily at {SCRAPE_TIME} UTC.", flush=True)

    while True:
        schedule.run_pending()
        time.sleep(60)