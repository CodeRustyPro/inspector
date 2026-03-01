"""
Fetch official MSHA and OSHA regulatory text from the eCFR public API
(api.ecfr.gov) and seed into VectorAI DB as a searchable regulations corpus.

This is NOT hardcoded summaries — it fetches actual CFR text verbatim
from the U.S. Electronic Code of Federal Regulations (api.ecfr.gov).

Regulations indexed:
  MSHA 30 CFR Part 56 (Metal/Nonmetal Surface Mine Safety):
    - 56.4200–56.4203, 56.4230   Fire prevention and control
    - 56.14100–56.14101          Pre-shift inspection, brakes
    - 56.14130–56.14132          ROPS, seatbelts, audible warning devices
    - 56.14200–56.14201          Illumination
  OSHA 29 CFR Part 1926 (Construction):
    - 1926.600–1926.602          Equipment and motor vehicles

Usage:
    python -m scripts.fetch_regulations          # fetch live + seed
    python -m scripts.fetch_regulations --force  # re-seed even if exists
"""

import sys, os, re, time, hashlib, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import xml.etree.ElementTree as ET

from backend.vectordb import get_store, REGULATIONS_COLLECTION
from backend.embeddings import embed_text

ECFR_BASE = "https://api.ecfr.gov/api/versioner/v1"
ECFR_DATE = "2025-01-01"   # Pinned date — update annually

# ─────────────────────────────────────────────────────────────────
# Target sections: (title, part, section_suffix, tag, keywords)
# section_suffix is appended to part to form the CFR section: e.g.
#   part=56, suffix="4230" → section param = "56.4230"
#   part=1926, suffix="600" → section param = "1926.600"
# ─────────────────────────────────────────────────────────────────
TARGET_SECTIONS = [
    # MSHA Part 56 — Fire Prevention and Control
    (30, 56,   "4200", "fire",             "fire prevention practices hazardous materials fire hazard mining"),
    (30, 56,   "4201", "fire",             "fire fighting training fire extinguisher operation personnel"),
    (30, 56,   "4202", "fire",             "fire control equipment location firefighting suppression"),
    (30, 56,   "4203", "fire",             "fire extinguisher type class ABC dry chemical capacity"),
    (30, 56,   "4230", "fire_extinguisher","fire extinguisher self-propelled mobile equipment excavator loader dozer"),
    # MSHA Part 56 — Pre-shift Inspection / Removal from Service
    (30, 56,   "14100","inspection",       "pre-shift inspection safety defects removal from service records tagging"),
    (30, 56,   "14101","brakes",           "brakes service brakes parking brakes self-propelled mobile equipment"),
    # MSHA Part 56 — ROPS / Seatbelts / Backup Alarms
    (30, 56,   "14130","rops",             "rollover protective structure ROPS canopy overhead guard mobile equipment"),
    (30, 56,   "14131","seatbelt",         "seatbelts seat belts ROPS equipped mobile equipment operator"),
    (30, 56,   "14132","backup_alarm",     "audible warning device backup alarm reverse travel obstructed vision"),
    # MSHA Part 56 — Illumination
    (30, 56,   "14200","lights",           "illumination lighting work areas equipment operations mines"),
    (30, 56,   "14201","lights",           "illumination standards minimum lighting levels foot candles"),
    # OSHA Part 1926 — Construction
    (29, 1926, "600",  "inspection",       "equipment general construction site safety requirements inspection"),
    (29, 1926, "601",  "inspection",       "motor vehicles construction site inspection brakes lights backup alarm"),
    (29, 1926, "602",  "inspection",       "material handling equipment construction excavator loader compactor"),
]


def _ecfr_url(title: int, part: int, section_suffix: str) -> str:
    """Build eCFR API URL. Section param must be '{part}.{suffix}' (e.g. '56.4230')."""
    section_param = f"{part}.{section_suffix}"
    return (
        f"{ECFR_BASE}/full/{ECFR_DATE}/title-{title}.xml"
        f"?part={part}&section={section_param}"
    )


def _clean_head(text: str) -> str:
    """Remove section number prefix and encoding artifacts from HEAD element."""
    # eCFR returns Â§ as encoding artifact for §
    text = text.replace("Â§", "§").replace("Â", "").strip()
    # Remove leading "§ 56.4230 " prefix to get just the title
    text = re.sub(r'^§?\s*[\d.]+\s*', '', text).strip().rstrip(".")
    return text


def _strip_tags(xml_text: str) -> str:
    """Remove all XML/HTML tags, collapse whitespace."""
    text = re.sub(r'<[^>]+>', ' ', xml_text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _parse_ecfr_xml(xml_bytes: bytes, title: int, part: int, section_suffix: str) -> list[dict]:
    """
    Parse eCFR XML for one section into paragraph-level chunks.
    eCFR XML structure:
      <DIV8 N="56.4230" TYPE="SECTION">
        <HEAD>§ 56.4230 Self-propelled equipment.</HEAD>
        <P>(a)(1) Whenever a fire...</P>
        <P>...</P>
      </DIV8>
    """
    section_param = f"{part}.{section_suffix}"
    citation = f"{title} CFR {section_param}"
    source_url = _ecfr_url(title, part, section_suffix)
    chunks = []

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"    XML parse error for {citation}: {e}")
        return []

    # Find the section title from HEAD element
    head_el = root.find(".//HEAD")
    section_title = _clean_head(head_el.text or "") if head_el is not None else section_param

    # Collect all paragraph-level text (P, FP, NOTE, EXTRACT)
    paragraphs = []
    for el in root.iter():
        if el.tag in ("P", "FP", "NOTE", "EXTRACT"):
            raw = ET.tostring(el, encoding="unicode", method="xml")
            text = _strip_tags(raw).strip()
            if text and len(text) > 15:
                paragraphs.append(text)

    if not paragraphs:
        # Fallback: take all text from the root, skip the HEAD
        if head_el is not None:
            root.remove(head_el)
        full = _strip_tags(ET.tostring(root, encoding="unicode")).strip()
        if full and len(full) > 15:
            paragraphs = [full[:2000]]

    for i, para in enumerate(paragraphs):
        # Embedding text: citation + title + paragraph (truncate to ~280 chars for CLIP 77-token limit)
        embed_str = f"{citation} {section_title}: {para}"[:280]
        chunks.append({
            "citation":      citation,
            "section_title": section_title,
            "paragraph_index": i,
            "text":          para,
            "full_citation": f"{citation} — {section_title}",
            "source":        "eCFR",
            "source_url":    source_url,
            "embed_text":    embed_str,
        })

    return chunks


def fetch_section(title: int, part: int, section_suffix: str,
                  tag: str, keywords: str) -> list[dict]:
    """Fetch one CFR section from eCFR API and return parsed chunks."""
    url = _ecfr_url(title, part, section_suffix)
    section_param = f"{part}.{section_suffix}"
    citation = f"{title} CFR {section_param}"

    try:
        resp = requests.get(url, timeout=15, headers={"Accept": "application/xml"})
        if resp.status_code == 404:
            print(f"    ⚠ {citation}: 404 — section not found in eCFR")
            return []
        resp.raise_for_status()

        chunks = _parse_ecfr_xml(resp.content, title, part, section_suffix)
        for c in chunks:
            c["tag"] = tag
            c["keywords"] = keywords

        if chunks:
            print(f"    ✓ {citation} — {len(chunks)} chunk(s): \"{chunks[0]['section_title']}\"")
        else:
            print(f"    ⚠ {citation}: parsed 0 chunks")
        return chunks

    except requests.RequestException as e:
        print(f"    ✗ {citation}: network error — {e}")
        return []


def _chunk_id(title: int, part: int, section_suffix: str, paragraph_index: int) -> int:
    """Stable integer ID for a regulation chunk. Range: 5000–9999."""
    raw = f"{title}_{part}_{section_suffix}_{paragraph_index}"
    h = int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)
    return 5000 + (h % 4999)


def fetch_and_seed(force: bool = False):
    store = get_store()

    existing = store.count(REGULATIONS_COLLECTION)
    if existing > 0 and not force:
        print(f"\n  Regulations already seeded ({existing} chunks). Use --force to re-seed.")
        return

    if force and existing > 0:
        print(f"\n  --force: clearing {existing} existing regulation chunks...")
        store.delete_collection(REGULATIONS_COLLECTION)
        store._ensure_collections()

    print(f"\n{'='*60}")
    print(f"FETCHING REGULATIONS FROM eCFR")
    print(f"  Source: {ECFR_BASE}  (date: {ECFR_DATE})")
    print(f"  Sections to fetch: {len(TARGET_SECTIONS)}")
    print(f"{'='*60}")

    all_chunks = []
    for title, part, section_suffix, tag, keywords in TARGET_SECTIONS:
        chunks = fetch_section(title, part, section_suffix, tag, keywords)
        all_chunks.extend(chunks)
        time.sleep(0.25)   # polite rate limiting

    if not all_chunks:
        print("\n  ✗ No regulation chunks fetched — check network / eCFR API availability")
        return

    print(f"\n  Embedding and indexing {len(all_chunks)} chunks into VectorAI DB...")
    seen_ids: set[int] = set()
    seeded = 0

    for chunk in all_chunks:
        # Parse title/part/suffix from citation "30 CFR 56.4230"
        parts = chunk["citation"].split()
        c_title = int(parts[0])
        c_section = parts[2]                        # e.g. "56.4230"
        c_part_str, c_suffix = c_section.split(".", 1)
        cid = _chunk_id(c_title, int(c_part_str), c_suffix, chunk["paragraph_index"])
        while cid in seen_ids:
            cid += 1
        seen_ids.add(cid)

        vec = embed_text(chunk["embed_text"])
        payload = {k: v for k, v in chunk.items() if k != "embed_text"}
        store.upsert(REGULATIONS_COLLECTION, id=cid, vector=vec, payload=payload)
        seeded += 1

    print(f"\n  ✅ Seeded {seeded} regulation chunks into '{REGULATIONS_COLLECTION}'")

    print(f"\n{'='*60}")
    print("SMOKE TEST — querying regulations collection")
    print(f"{'='*60}")
    test_queries = [
        ("fire extinguisher MSHA mining equipment",          "fire_extinguisher"),
        ("backup alarm audible warning reverse self-propelled", "backup_alarm"),
        ("rollover protective structure ROPS",               "rops"),
        ("pre-shift inspection safety defects removal",      "inspection"),
        ("service brakes parking brakes equipment",          "brakes"),
    ]
    for query_text, label in test_queries:
        vec = embed_text(query_text)
        results = store.search(REGULATIONS_COLLECTION, vec, top_k=2)
        print(f"\n  [{label}] '{query_text[:50]}'")
        for r in results:
            p = r.get("payload", {})
            print(f"    {r['score']:.3f} | {p.get('citation')} — {p.get('section_title','')}")
            print(f"           {p.get('text','')[:110]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch MSHA/OSHA regulations from eCFR into VectorAI DB"
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-seed even if regulations already exist")
    args = parser.parse_args()
    fetch_and_seed(force=args.force)
