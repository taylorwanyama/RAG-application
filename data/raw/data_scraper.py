import json
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://tenders.go.ke"
API_URL = f"{BASE_URL}/api/active-tenders"

OUTPUT_DIR = Path("data/raw")
RAW_JSON_PATH = OUTPUT_DIR / "active_tenders_raw.json"
CSV_PATH = OUTPUT_DIR / "active_tenders.csv"


def create_session():
    """
    Create a requests session with automatic retries.
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def fetch_page(session, page, per_page=100):
    """
    Fetch one page of active tenders.
    """

    params = {
        "perpage": per_page,
        "page": page,
    }

    response = session.get(
        API_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def extract_tender(tender):
    pe = tender.get("pe") or {}
    method = tender.get("procurement_method") or {}
    category = tender.get("procurement_category") or {}
    addenda = tender.get("addenda") or []

    return {
        "id": tender.get("id"),
        "tender_ref": (tender.get("tender_ref") or "").strip(),
        "description": tender.get("title"),          
        "procuring_entity": pe.get("name"),
        "procurement_method": method.get("title"),
        "procurement_category": category.get("title"),
        "close_at": tender.get("close_at"),
        "published_at": tender.get("published_at"), 
        "addendum_added": tender.get("addendum_added"),
        "addenda_count": len(addenda),

        # Useful extras the API already provides
        "venue": tender.get("venue"),
        "tender_fee": tender.get("tender_fee"),
        "validity_in_days": tender.get("validity_in_days"),
        "county_id": pe.get("county_id"),
        "pe_city": pe.get("city"),
        "pe_email": pe.get("email"),
        "pe_telephone": pe.get("telephone"),
        "financial_year": (tender.get("financial_year") or {}).get("name"),
        "ocid": tender.get("ocid"),
        "terminated": tender.get("terminated"),
        "tender_document_urls": "; ".join(
            BASE_URL + d["url"] for d in (tender.get("documents") or []) if d.get("url")
        ),
    }
def scrape_active_tenders(test_mode=False):
    """
    Scrape active tenders.

    test_mode=True:
        Fetch only the first 5 tenders.

    test_mode=False:
        Fetch all active tenders.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = create_session()

    all_tenders = []
    raw_pages = []

    page = 1

    # First request
    if test_mode:
        per_page = 5
    else:
        per_page = 100

    print("=" * 80)
    print("KENYA PUBLIC TENDERS SCRAPER")
    print("=" * 80)

    print(f"API: {API_URL}")
    print(f"Mode: {'TEST' if test_mode else 'FULL'}")
    print()

    while True:

        print(f"Fetching page {page}...")

        try:
            payload = fetch_page(
                session=session,
                page=page,
                per_page=per_page,
            )

        except requests.RequestException as error:
            print(f"ERROR fetching page {page}: {error}")
            break

        raw_pages.append(payload)

        data = payload.get("data", [])

        if not data:
            print("No more tenders returned.")
            break

        all_tenders.extend(data)

        current_page = payload.get("current_page")
        last_page = payload.get("last_page")
        total = payload.get("total")

        print(
            f"  Retrieved: {len(data)}"
            f" | Total collected: {len(all_tenders)}"
            f" | Total available: {total}"
        )

        # Test mode stops after first request
        if test_mode:
            break

        # Stop when we've reached the final page
        if current_page >= last_page:
            break

        page += 1

        # Be polite to the server
        time.sleep(0.5)

    print()
    print("=" * 80)
    print("SCRAPING COMPLETE")
    print("=" * 80)

    print(f"Raw tenders collected: {len(all_tenders)}")

    # ------------------------------------------------------------------
    # Save raw API response
    # ------------------------------------------------------------------

    raw_output = {
        "source": API_URL,
        "total_collected": len(all_tenders),
        "pages_scraped": len(raw_pages),
        "pages": raw_pages,
    }

    with open(RAW_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(
            raw_output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Raw JSON saved to: {RAW_JSON_PATH}")

    # ------------------------------------------------------------------
    # Transform into tabular format
    # ------------------------------------------------------------------

    records = [
        extract_tender(tender)
        for tender in all_tenders
    ]

    df = pd.DataFrame(records)

    for col in ["close_at", "published_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Site times are Kenyan local time (EAT)
    now = pd.Timestamp.now(tz="Africa/Nairobi").tz_localize(None)
    hours_left = (df["close_at"] - now).dt.total_seconds() / 3600

    def fmt(h):
        if pd.isna(h): return None
        if h < 0: return "Closed"
        return f"{int(h)} hours" if h < 24 else f"{int(h // 24)} days"

    df["time_to_close"] = hours_left.map(fmt)

    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if "tender_ref" in df.columns:
        duplicate_refs = df["tender_ref"].duplicated().sum()

        print(f"Duplicate tender references: {duplicate_refs}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    df.to_csv(
        CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    print()
    print(f"CSV saved to: {CSV_PATH}")

    return df


if __name__ == "__main__":

    TEST_MODE = False # Set to True for testing

    df = scrape_active_tenders(
        test_mode=TEST_MODE
    )

    print()
    print("=" * 80)
    print("SAMPLE DATA")
    print("=" * 80)

    print(
        df.head().to_string(index=False)
    )