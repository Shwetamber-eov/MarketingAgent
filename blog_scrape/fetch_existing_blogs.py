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
import json
import sys
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

BASE_LISTING_URL = "https://embarkingonvoyage.com/blog/"
USER_AGENT = "Mozilla/5.0 (compatible; BlogScraperBot/1.0)"
REQUEST_DELAY = 1.0  # seconds between requests — be polite to the server

# ---- Selectors matching embarkingonvoyage.com's WordPress theme ----
CARD_SELECTOR = "li.wp-block-post"           # each post card wrapper on listing pages
CARD_LINK_SELECTOR = "a.post-card__link"     # the link that wraps a card's content
TITLE_SELECTOR = ".post-card__info-title"
EXCERPT_SELECTOR = ".post-card__excerpt"
AUTHOR_SELECTOR = ".post-card__author-name"
DATE_SELECTOR = ".post-card__date"           # has a datetime="" attribute
READ_TIME_SELECTOR = ".post-card__read-time"
NEXT_PAGE_SELECTOR = "a.wp-block-query-pagination-next"

# Best-effort selectors for the full article body on an individual post page.
# WordPress content usually lives in one of these — adjust if extraction comes back empty.
FULL_CONTENT_SELECTORS = [".entry-content", "article .wp-block-post-content", "article"]
def can_fetch(url, user_agent=USER_AGENT):
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True

def get_soup(url, session):
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def clean_text(el):
    if not el:
        return ""
    # strip the "Read More…" span out of the excerpt block if present
    for span in el.select(".post-card__view-more"):
        span.decompose()
    return el.get_text(strip=True, separator=" ")

def extract_cards(soup, page_url):
    posts = []
    cards = soup.select(CARD_SELECTOR)
    for card in cards:
        link_el = card.select_one(CARD_LINK_SELECTOR)
        if not link_el or not link_el.get("href"):
            continue
        url = urljoin(page_url, link_el["href"])

        title = clean_text(card.select_one(TITLE_SELECTOR))
        if not title:
            continue

        excerpt = clean_text(card.select_one(EXCERPT_SELECTOR))
        author = clean_text(card.select_one(AUTHOR_SELECTOR))
        read_time = clean_text(card.select_one(READ_TIME_SELECTOR))
        date_el = card.select_one(DATE_SELECTOR)
        date_iso = date_el.get("datetime", "") if date_el else ""
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
    next_el = soup.select_one(NEXT_PAGE_SELECTOR)
    if next_el and next_el.get("href"):
        return urljoin(current_url, next_el["href"])
    return None

def scrape_full_content(url, session):
    soup = get_soup(url, session)
    for sel in FULL_CONTENT_SELECTORS:
        el = soup.select_one(sel)
        if el:
            paragraphs = el.find_all("p")
            if paragraphs:
                return "\n\n".join(p.get_text(strip=True) for p in paragraphs)
    return ""


def scrape_all_pages(start_url, max_pages=None, fetch_full_content=False):
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_posts = []
    current_url = start_url
    page_num = 1

    while current_url:
        if max_pages and page_num > max_pages:
            break
        if not can_fetch(current_url):
            print(f"Blocked by robots.txt: {current_url}", file=sys.stderr)
            break

        print(f"Fetching listing page {page_num}: {current_url}")
        try:
            soup = get_soup(current_url, session)
        except requests.RequestException as e:
            print(f"  Failed: {e}", file=sys.stderr)
            break

        page_posts = extract_cards(soup, current_url)
        print(f"  Found {len(page_posts)} posts")
        all_posts.extend(page_posts)

        next_url = find_next_page(soup, current_url)
        current_url = next_url
        page_num += 1
        if current_url:
            time.sleep(REQUEST_DELAY)

    if fetch_full_content:
        print(f"\nFetching full content for {len(all_posts)} posts...")
        for i, post in enumerate(all_posts, 1):
            time.sleep(REQUEST_DELAY)
            try:
                post["content"] = scrape_full_content(post["url"], session)
                print(f"  [{i}/{len(all_posts)}] {post['title'][:60]}")
            except requests.RequestException as e:
                print(f"  [{i}/{len(all_posts)}] Failed: {e}", file=sys.stderr)
                post["content"] = ""

    return all_posts


def save_csv(posts, path):
    if not posts:
        print("No posts to save.")
        return
    keys = list(posts[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(posts)
    print(f"\nSaved {len(posts)} posts to {path}")


def save_json(posts, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(posts)} posts to {path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape all blog posts from embarkingonvoyage.com/blog")
    parser.add_argument("--url", default=BASE_LISTING_URL, help="Listing page to start from")
    parser.add_argument("--pages", type=int, default=None, help="Max number of listing pages to crawl (default: all ~102)")
    parser.add_argument("--full-content", action="store_true", help="Also visit each post to scrape the full article body (slower — ~600+ extra requests)")
    parser.add_argument("--output", default="blog_posts.csv", help="Output file path (.csv or .json)")
    args = parser.parse_args()

    posts = scrape_all_pages(args.url, max_pages=args.pages, fetch_full_content=args.full_content)

    if args.output.endswith(".json"):
        save_json(posts, args.output)
    else:
        save_csv(posts, args.output)


if __name__ == "__main__":
    main()