#!/usr/bin/env python3
"""
Blog scraper for embarkingonvoyage.com/blog

Crawls every paginated listing page (page/2/, page/3/, ... following the
"next" link automatically, so it keeps working even if the total page
count changes) and extracts each post card's title, excerpt, URL, author,
date, and read time. Optionally visits each individual post page to also
grab the full article body.

Usage:
    python blog_scraper.py                       # scrape all pages, listing data only
    python blog_scraper.py --pages 5              # only crawl the first 5 listing pages
    python blog_scraper.py --full-content          # also fetch full article text (slower)
    python blog_scraper.py --output posts.json      # write JSON instead of CSV

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

# Only check the latest 10 listing pages
MAX_PAGES = 10

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
_ROBOTS_CACHE = {}


def can_fetch(url, user_agent=USER_AGENT):
    """Check robots.txt before fetching a URL (cached per domain)."""
    parsed = urlparse(url)
    domain_key = f"{parsed.scheme}://{parsed.netloc}"
    print("domain key is::::::::::::::::", domain_key)
    rp = _ROBOTS_CACHE.get(domain_key)

    if rp is None:
        robots_url = f"{domain_key}/robots.txt"
        rp = RobotFileParser()

        try:
            rp.set_url(robots_url)
            rp.read()
        except Exception:
            # If robots.txt cannot be read, allow the request.
            rp = None

        _ROBOTS_CACHE[domain_key] = rp

    if rp is None:
        return True

    return rp.can_fetch(user_agent, url)


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
    Load existing CSV.

    Returns:
        existing_posts: list of existing rows
        existing_urls: set of normalized URLs
    """

    existing_posts = []
    existing_urls = set()

    if not path.exists():
        print(f"CSV does not exist yet: {path}")
        return existing_posts, existing_urls

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            url = row.get("url", "").strip()

            if not url:
                continue

            row["url"] = normalize_url(url)

            existing_posts.append({
                column: row.get(column, "")
                for column in CSV_COLUMNS
            })

            existing_urls.add(row["url"])

    print(f"Existing CSV records: {len(existing_posts)}")

    return existing_posts, existing_urls


def scrape_latest_pages(start_url, session, max_pages=10):
    """
    Scrape only the latest N listing pages.
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


def append_new_posts(path, existing_posts, new_posts):
    """
    Append new posts to the CSV while preserving existing records.
    """

    if not new_posts:
        print("\nNo new blogs found.")
        return

    # Make sure the output directory exists (important when running in Docker
    # with a mounted volume that might not have the folder pre-created).
    path.parent.mkdir(parents=True, exist_ok=True)

    # Combine existing + new
    all_posts = existing_posts + new_posts

    # Write complete CSV using the same column structure
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(all_posts)

    print(f"\nAdded {len(new_posts)} new blog(s).")
    print(f"Total blogs in CSV: {len(all_posts)}")
    print(f"Saved to: {path}") 

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check the latest 10 blog pages on "
            "embarkingonvoyage.com and append new posts "
            "to an existing CSV."
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
        help="Number of latest listing pages to check (default: 10)"
    )

    parser.add_argument(
        "--output",
        default="data/blog_posts.csv",
        help="Existing CSV file"
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    # ---------------------------------------------------------
    # 1. Load existing CSV
    # ---------------------------------------------------------

    existing_posts, existing_urls = load_existing_csv(output_path)

    # ---------------------------------------------------------
    # 2. Create HTTP session
    # ---------------------------------------------------------

    session = requests.Session()

    session.headers.update({
        "User-Agent": USER_AGENT
    })

    # ---------------------------------------------------------
    # 3. Check latest pages
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
    # 6. Append new posts
    # ---------------------------------------------------------

    append_new_posts(
        output_path,
        existing_posts,
        new_posts
    )


if __name__ == "__main__":
    main()