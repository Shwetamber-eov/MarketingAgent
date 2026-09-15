#!/usr/bin/env python3
"""
Blog scraper for embarkingonvoyage.com/blog

Crawls every paginated listing page (page/2/, page/3/, ... following the
"next" link automatically, so it keeps working even if the total page
count changes) and extracts each post card's title, excerpt, URL, author,
date, and read time.

Compares discovered posts against an existing archive CSV (--input,
typically old_blog_posts.csv) and writes ONLY the newly-found posts to
the output CSV (--output, typically blog_posts.csv). This output file is
meant to be a hand-off buffer for a downstream ingestion step, which is
responsible for merging it into the archive and clearing it afterward.

Usage:
    python blog_scraper.py                  # scrape all pages, listing data only
    python blog_scraper.py --pages 5        # only crawl the first 5 listing pages
    python blog_scraper.py --output posts.csv --input old_posts.csv

Requirements:
    pip install requests beautifulsoup4
"""

import argparse
import csv
import sys
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASE_LISTING_URL = "https://embarkingonvoyage.com/blog/"
USER_AGENT = "Mozilla/5.0 (compatible; BlogScraperBot/1.0)"
REQUEST_DELAY = 1.0

# Default cap on how many listing pages to crawl in one run.
MAX_PAGES = 150

# ---- Selectors matching embarkingonvoyage.com's WordPress theme ----
CARD_SELECTOR = "li.wp-block-post"
CARD_LINK_SELECTOR = "a.post-card__link"
TITLE_SELECTOR = ".post-card__info-title"
EXCERPT_SELECTOR = ".post-card__excerpt"
AUTHOR_SELECTOR = ".post-card__author-name"
DATE_SELECTOR = ".post-card__date"
READ_TIME_SELECTOR = ".post-card__read-time"
NEXT_PAGE_SELECTOR = "a.wp-block-query-pagination-next"

CSV_COLUMNS = [
    "title",
    "url",
    "author",
    "date",
    "date_display",
    "read_time",
    "excerpt",
]

# Cache robots.txt parsers per domain so we don't re-fetch it on every page.
# Use a sentinel to distinguish "not yet cached" from "cached as failed",
# since a plain None can't tell those apart on a dict.get().
_ROBOTS_CACHE = {}
_ROBOTS_FETCH_FAILED = object()


def can_fetch(url, user_agent=USER_AGENT):
    """Check robots.txt before fetching a URL (cached per domain)."""
    parsed = urlparse(url)
    domain_key = f"{parsed.scheme}://{parsed.netloc}"
    cached = _ROBOTS_CACHE.get(domain_key)

    if cached is None:
        robots_url = f"{domain_key}/robots.txt"
        rp = RobotFileParser()

        try:
            rp.set_url(robots_url)
            rp.read()
            _ROBOTS_CACHE[domain_key] = rp
            cached = rp
        except Exception:
            # If robots.txt cannot be read, cache the failure (don't retry
            # on every page) and allow the request.
            _ROBOTS_CACHE[domain_key] = _ROBOTS_FETCH_FAILED
            return True

    if cached is _ROBOTS_FETCH_FAILED:
        return True

    return cached.can_fetch(user_agent, url)


def get_soup(url, session):
    """Fetch a URL and return BeautifulSoup."""
    response = session.get(url, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def clean_text(el):
    """Clean text from a BeautifulSoup element."""
    if not el:
        return ""

    for span in el.select(".post-card__view-more"):
        span.decompose()

    return el.get_text(strip=True, separator=" ")


def normalize_url(url):
    """
    Normalize URLs so small differences don't create duplicates.
    """
    parsed = urlparse(url)

    # Remove fragments (#something)
    normalized = parsed._replace(fragment="").geturl()

    # Remove trailing slash except for root
    if parsed.path != "/":
        normalized = normalized.rstrip("/")

    return normalized


def extract_cards(soup, page_url):
    """Extract blog posts from a listing page."""
    posts = []

    cards = soup.select(CARD_SELECTOR)

    for card in cards:
        link_el = card.select_one(CARD_LINK_SELECTOR)

        if not link_el or not link_el.get("href"):
            continue

        url = urljoin(page_url, link_el["href"])
        url = normalize_url(url)

        title = clean_text(card.select_one(TITLE_SELECTOR))

        if not title:
            continue

        excerpt = clean_text(card.select_one(EXCERPT_SELECTOR))
        author = clean_text(card.select_one(AUTHOR_SELECTOR))
        read_time = clean_text(card.select_one(READ_TIME_SELECTOR))

        date_el = card.select_one(DATE_SELECTOR)

        date_iso = ""
        date_display = ""

        if date_el:
            date_iso = date_el.get("datetime", "")
            date_display = clean_text(date_el)

        posts.append({
            "title": title,
            "url": url,
            "author": author,
            "date": date_iso,
            "date_display": date_display,
            "read_time": read_time,
            "excerpt": excerpt,
        })

    return posts


def find_next_page(soup, current_url):
    """Find the next pagination URL."""
    next_el = soup.select_one(NEXT_PAGE_SELECTOR)

    if next_el and next_el.get("href"):
        return urljoin(current_url, next_el["href"])

    return None


def load_existing_csv(path):
    """
    Load existing archive CSV (used only to know which URLs are already
    known, so we don't re-report them as "new").

    Returns:
        existing_urls: set of normalized URLs already present
    """

    existing_urls = set()

    if not path.exists():
        print(f"Archive CSV does not exist yet: {path}")
        return existing_urls

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            url = row.get("url", "").strip()

            if not url:
                continue

            existing_urls.add(normalize_url(url))

    print(f"Existing archive records: {len(existing_urls)}")

    return existing_urls


def scrape_latest_pages(start_url, session, max_pages=10):
    """
    Scrape listing pages, following "next" links, up to max_pages.
    """

    all_posts = []
    seen_urls = set()

    current_url = start_url
    page_num = 1

    while current_url and page_num <= max_pages:

        if not can_fetch(current_url):
            print(
                f"Blocked by robots.txt: {current_url}",
                file=sys.stderr
            )
            break

        print(f"\nFetching listing page {page_num}/{max_pages}")
        print(current_url)

        try:
            soup = get_soup(current_url, session)
        except requests.RequestException as e:
            print(
                f"Failed to fetch listing page: {e}",
                file=sys.stderr
            )
            break

        page_posts = extract_cards(soup, current_url)

        print(f"Found {len(page_posts)} posts")

        for post in page_posts:
            url = post["url"]

            # Avoid duplicates between listing pages
            if url not in seen_urls:
                seen_urls.add(url)
                all_posts.append(post)

        next_url = find_next_page(soup, current_url)

        current_url = next_url
        page_num += 1

        if current_url and page_num <= max_pages:
            time.sleep(REQUEST_DELAY)

    print(f"\nTotal unique posts discovered: {len(all_posts)}")

    return all_posts


def write_new_posts(path, new_posts):
    """
    Write ONLY the newly discovered posts to the output CSV.

    This file is a hand-off buffer for a downstream ingestion step: it is
    always (re)written from scratch with just this run's new posts (never
    merged with the archive), so the downstream step can safely append it
    to the archive and clear it without re-processing old rows.

    A header-only file is written even when there are zero new posts, so
    downstream tooling can rely on the file always existing and being
    valid CSV.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(new_posts)

    if new_posts:
        print(f"\nWrote {len(new_posts)} new blog(s) to: {path}")
        for post in new_posts:
            print("Title: ", post.get("title"))
    else:
        print(f"\nNo new blogs found. Wrote header-only file to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check embarkingonvoyage.com's blog listing pages and write "
            "any posts not already present in the archive CSV to the "
            "output CSV."
        )
    )
    parser.add_argument(
        "--url",
        default=BASE_LISTING_URL,
        help="Blog listing URL"
    )

    parser.add_argument(
        "--pages",
        type=int,
        default=MAX_PAGES,
        help=f"Number of latest listing pages to check (default: {MAX_PAGES})"
    )

    parser.add_argument(
        "--output",
        default="data/blog_posts.csv",
        help="Output CSV to write newly discovered posts to (overwritten each run)"
    )

    parser.add_argument(
        "--input",
        default="data/old_blog_posts.csv",
        help="Existing archive CSV used to detect which posts are already known"
    )

    args = parser.parse_args()

    output_path = ROOT / Path(args.output)
    input_path = ROOT / Path(args.input)
    print("output path:", output_path)
    print("input path:", input_path)

    # ---------------------------------------------------------
    # 1. Load existing archive URLs
    # ---------------------------------------------------------

    existing_urls = load_existing_csv(input_path)

    # ---------------------------------------------------------
    # 2. Create HTTP session
    # ---------------------------------------------------------

    session = requests.Session()

    session.headers.update({
        "User-Agent": USER_AGENT
    })

    # ---------------------------------------------------------
    # 3. Scrape listing pages
    # ---------------------------------------------------------

    discovered_posts = scrape_latest_pages(
        args.url,
        session,
        max_pages=args.pages
    )

    # ---------------------------------------------------------
    # 4. Find only NEW posts
    # ---------------------------------------------------------

    new_posts = []

    for post in discovered_posts:

        url = normalize_url(post["url"])

        if url in existing_urls:
            continue

        new_posts.append(post)

        # Prevent duplicates among newly discovered posts
        existing_urls.add(url)

    # ---------------------------------------------------------
    # 5. Display new posts
    # ---------------------------------------------------------

    if new_posts:

        print("\nNEW BLOGS FOUND")
        print("=" * 60)

        for i, post in enumerate(new_posts, 1):
            print(f"\n{i}. {post['title']}")
            print(f"   URL: {post['url']}")
            print(f"   Author: {post['author']}")
            print(f"   Date: {post['date_display']}")

    # ---------------------------------------------------------
    # 6. Write new posts to the output buffer file
    # ---------------------------------------------------------

    write_new_posts(output_path, new_posts)


if __name__ == "__main__":
    main()