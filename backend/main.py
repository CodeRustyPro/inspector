"""
Cat Inspect AI — FastAPI Backend v0.4
CLIP embeddings + VectorAI DB + Gemini 3 Flash

v0.4: Photo-over-time comparison, maintenance interval engine,
      OCR nameplate reading, institutional memory enrichment,
      MSHA/OSHA compliance flagging.
"""
import os, io, json, time, math
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import numpy as np

from backend.vectordb import get_store, INSPECTION_COLLECTION, PARTS_COLLECTION
from backend.embeddings import embed_image, embed_text
from backend.gemini_service import (
    analyze_inspection, identify_part, generate_report,
    ocr_nameplate, get_maintenance_schedule,
)

app = FastAPI(title="Cat Inspect AI", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

_next_id = {"inspection": 1000, "parts": 2000}


# ============================================================
# Score Normalization
# ============================================================

def normalize_score(raw_score: float, mode: str = "cross_modal") -> float:
    """
    Normalize CLIP similarity scores to intuitive 0-100% range.

    Raw CLIP cosine similarity ranges:
    - Text↔Text: 0.7-0.95 for related, 0.2-0.4 for unrelated
    - Image↔Text (cross-modal): 0.20-0.35 for related, 0.10-0.20 for unrelated
    """
    if mode == "cross_modal":
        normalized = 1 / (1 + math.exp(-20 * (raw_score - 0.25)))
    elif mode == "text_text":
        normalized = 1 / (1 + math.exp(-8 * (raw_score - 0.6)))
    else:  # hybrid — already fused, typically 0.2-0.65
        normalized = 1 / (1 + math.exp(-10 * (raw_score - 0.35)))
    return round(min(max(normalized, 0.0), 1.0), 3)


# ============================================================
# Hybrid Search Engine — the key innovation
# ============================================================

def hybrid_search(
    store,
    collection: str,
    image_embedding: list,
    text_query: str = "",
    component: str = "general",
    top_k: int = 5,
    image_weight: float = 0.4,
    text_weight: float = 0.6,
) -> list:
    """
    Fuse image-based and text-based vector searches for dramatically
    better retrieval quality.

    Cross-modal CLIP (image→text) tops out at ~0.35 similarity.
    Text→Text CLIP hits 0.7-0.95 for related content.

    By combining both signals, we get the best of both worlds:
    - Image captures visual cues the inspector might not verbalize
    - Text captures domain-specific terminology and context

    Returns enriched results with fused scores.
    """
    filter_field = "component" if component != "general" else None
    filter_value = component if component != "general" else None

    # Search 1: Image embedding (cross-modal)
    image_results = store.search(
        collection, query_vector=image_embedding, top_k=top_k * 2,
        filter_field=filter_field, filter_value=filter_value,
    )

    # Build a map of id → result
    result_map = {}
    for r in image_results:
        result_map[r["id"]] = {
            "id": r["id"],
            "image_score": r["score"],
            "text_score": 0.0,
            "payload": r["payload"],
        }

    # Search 2: Text embedding (same-modal, much higher precision)
    if text_query and text_query.strip():
        text_embedding = embed_text(text_query)
        text_results = store.search(
            collection, query_vector=text_embedding, top_k=top_k * 2,
            filter_field=filter_field, filter_value=filter_value,
        )
        for r in text_results:
            if r["id"] in result_map:
                result_map[r["id"]]["text_score"] = r["score"]
            else:
                result_map[r["id"]] = {
                    "id": r["id"],
                    "image_score": 0.0,
                    "text_score": r["score"],
                    "payload": r["payload"],
                }
    else:
        # No text query — image only, full weight
        image_weight = 1.0
        text_weight = 0.0

    # Fuse scores
    for rid, data in result_map.items():
        data["fused_score"] = (
            image_weight * data["image_score"] +
            text_weight * data["text_score"]
        )

    # Sort by fused score
    results = sorted(result_map.values(), key=lambda x: x["fused_score"], reverse=True)
    return results[:top_k]


# ============================================================
# Fleet Intelligence — Escalation Detection & Cost Prediction
# ============================================================

def detect_escalation_risk(similar_cases: list) -> dict:
    """
    Analyze fleet history for escalation patterns.
    If a similar past case was rated YELLOW and later escalated to RED,
    or had costly outcomes, flag it proactively.
    """
    escalations = []
    costs = []
    downtimes = []

    for case in similar_cases:
        p = case.get("payload", {})
        outcome = str(p.get("outcome", "")).lower()
        rating = p.get("rating", "")

        # Detect escalation patterns in outcomes
        if "escalat" in outcome or "catastroph" in outcome or "burst" in outcome:
            escalations.append({
                "equipment": p.get("equipment_model", "?"),
                "hours": p.get("hours", "?"),
                "original_rating": rating,
                "outcome": p.get("outcome", "?"),
                "finding": p.get("finding", "?"),
            })

        # Extract costs from outcomes
        if "$" in outcome:
            import re
            cost_match = re.findall(r'\$[\d,]+', outcome)
            for c in cost_match:
                try:
                    costs.append(int(c.replace("$", "").replace(",", "")))
                except:
                    pass

        # Extract downtime indicators
        for keyword in ["hrs", "hours", "days", "weeks"]:
            if keyword in outcome:
                downtimes.append(outcome)

    risk = {
        "has_escalation_risk": len(escalations) > 0,
        "escalation_cases": escalations,
        "cost_range": None,
        "downtime_examples": downtimes[:3],
    }

    if costs:
        risk["cost_range"] = {
            "min": min(costs),
            "max": max(costs),
            "avg": round(sum(costs) / len(costs)),
            "sample_size": len(costs),
        }

    return risk


# ============================================================
# Photo-Over-Time Detection
# ============================================================

def detect_comparison_cases(similar: list, equipment_model: str, component: str) -> list:
    """
    Find cases from the SAME equipment model AND same component.
    These represent prior inspections of the same part — enabling
    condition trending over time.
    """
    if component == "general":
        return []

    comparisons = []
    for s in similar:
        p = s.get("payload", {})
        if (p.get("equipment_model", "") == equipment_model and
                p.get("component", "") == component):
            comparisons.append(s)

    # Sort by hours (ascending) to show progression
    comparisons.sort(key=lambda x: x.get("payload", {}).get("hours", 0))
    return comparisons


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.get("/api/health")
async def health():
    store = get_store()
    return {
        "status": "ok",
        "vectordb": "Actian VectorAI DB" if store.use_cortex else "In-Memory Fallback",
        "inspection_count": store.count(INSPECTION_COLLECTION),
        "parts_count": store.count(PARTS_COLLECTION),
    }


@app.post("/api/inspect")
async def inspect_checkpoint(
    photo: UploadFile = File(...),
    voice_transcript: str = Form(default=""),
    component: str = Form(default="general"),
    equipment_model: str = Form(default="CAT 320"),
    equipment_hours: int = Form(default=5000),
):
    """
    Core inspection pipeline:
    1. CLIP embed photo
    2. Hybrid search (image + text) against VectorAI DB fleet history
    3. Detect escalation risks from similar cases
    4. Detect photo-over-time comparison cases
    5. Get maintenance schedule context
    6. Gemini analyzes photo + voice + history + risk + comparison context
    7. Store new inspection (memory grows)
    """
    start = time.time()
    store = get_store()
    image_bytes = await photo.read()

    # Step 1: CLIP embedding
    embedding = embed_image(image_bytes)
    embed_time = time.time() - start

    # Step 2: Hybrid search — THE KEY IMPROVEMENT
    text_query_parts = []
    if component != "general":
        text_query_parts.append(component.replace("_", " "))
    if voice_transcript:
        text_query_parts.append(voice_transcript)
    text_query_parts.append(f"{equipment_model} inspection")
    text_query = " ".join(text_query_parts)

    similar = hybrid_search(
        store, INSPECTION_COLLECTION,
        image_embedding=embedding,
        text_query=text_query,
        component=component,
        top_k=8,  # Increased top_k to 8 to have enough pool for auto-detect filtering
        image_weight=0.35,
        text_weight=0.65,
    )
    search_time = time.time() - start - embed_time

    # Step 3: Escalation detection
    escalation_risk = detect_escalation_risk(similar)

    # Step 4: Photo-over-time comparison
    comparison_cases = detect_comparison_cases(similar, equipment_model, component)

    # Step 5: Maintenance schedule
    maintenance = get_maintenance_schedule(equipment_hours)

    # Step 6: Gemini analysis — feed it ALL context
    gemini_cases = [
        {"score": s["fused_score"], "payload": s["payload"]}
        for s in similar
    ]
    gemini_comparisons = [
        {"score": c["fused_score"], "payload": c["payload"]}
        for c in comparison_cases
    ] if comparison_cases else None

    try:
        analysis = analyze_inspection(
            image_bytes=image_bytes,
            voice_transcript=voice_transcript,
            component_name=component,
            equipment_model=equipment_model,
            equipment_hours=equipment_hours,
            similar_cases=gemini_cases,
            escalation_risk=escalation_risk,
            comparison_cases=gemini_comparisons,
        )
    except Exception as e:
        analysis = {
            "is_valid_equipment_image": True,
            "detected_subject": "unknown",
            "rating": "YELLOW", "confidence": 0.0,
            "finding": f"AI analysis error: {str(e)}",
            "reasoning": "Gemini API call failed",
            "action": "Manual inspection required",
            "severity_score": 5, "historical_context": "N/A",
            "condition_trend": "Unable to assess.",
            "fleet_pattern_note": "",
            "compliance_flag": None,
            "detected_component": component,
        }
    gemini_time = time.time() - start - embed_time - search_time

    # --- Reject non-equipment images ---
    if not analysis.get("is_valid_equipment_image", True):
        return JSONResponse(
            status_code=422,
            content={
                "error": "invalid_image",
                "detected_subject": analysis.get("detected_subject", "non-equipment subject"),
                "message": (
                    f"This image does not appear to show construction equipment or equipment components. "
                    f"Detected: {analysis.get('detected_subject', 'non-equipment subject')}. "
                    f"Please photograph a machine component, hose, filter, track, or safety equipment."
                ),
            },
        )

    # Fix Component Auto-Detection Bug:
    # Use Gemini's detected_component for VectorDB fleet history relevance.
    detected_component = analysis.get("detected_component", component)
    if not detected_component or detected_component.lower() == "none" or detected_component.lower() == "general":
        detected_component = "general"
    
    # Filter 'similar' cases to those matching the detected component, if it's not general
    if detected_component != "general" and component == "general":
         # Strict component match — only show genuinely relevant similar findings
         filtered_similar = [s for s in similar if s["payload"].get("component") == detected_component]

         # Cross-modal gap fix: seed data is text-embedded, user photos are image-embedded.
         # CLIP cross-modal similarity (~0.20-0.35) often fails to surface the right records.
         # If component filtering leaves us empty, do a targeted TEXT search for this component.
         if not filtered_similar:
             comp_query = f"{detected_component.replace('_', ' ')} {equipment_model} inspection defect"
             comp_embedding = embed_text(comp_query)
             text_results = store.search(
                 INSPECTION_COLLECTION, query_vector=comp_embedding, top_k=5,
                 filter_field="component", filter_value=detected_component,
             )
             filtered_similar = [
                 {"id": r["id"], "fused_score": r["score"], "payload": r["payload"]}
                 for r in text_results
             ]

         filtered_similar = filtered_similar[:5]
         similar = filtered_similar
    else:
        similar = similar[:5] # ensure we only return top 5
    
    # Recalculate escalations and comparisons based on the filtered list (which now matches the detected part)
    if component == "general" and detected_component != "general":
        escalation_risk = detect_escalation_risk(similar)
        comparison_cases = detect_comparison_cases(similar, equipment_model, detected_component)

    # Step 7: Store in VectorAI DB (Inspection Memory grows)
    inspection_id = _next_id["inspection"]
    _next_id["inspection"] += 1

    store.upsert(
        INSPECTION_COLLECTION, id=inspection_id, vector=embedding,
        payload={
            "component": detected_component, # Use detected component here!
            "equipment_model": equipment_model,
            "hours": equipment_hours,
            "rating": analysis.get("rating", "YELLOW"),
            "finding": analysis.get("finding", ""),
            "action": analysis.get("action", ""),
            "voice_transcript": voice_transcript,
            "severity_score": analysis.get("severity_score", 5),
            "outcome": "pending",
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    )

    # Build enriched similar cases with comparison flags
    comparison_ids = {c["id"] for c in comparison_cases} if comparison_cases else set()

    return {
        "inspection_id": inspection_id,
        "analysis": analysis,
        "similar_cases": [
            {
                "raw_score": round(s["fused_score"], 4),
                "is_comparison": s["id"] in comparison_ids,
                "payload": s["payload"],
            }
            for s in similar
        ],
        "escalation_risk": escalation_risk,
        "maintenance_due": maintenance,
        "memory_stats": {
            "total_inspections": store.count(INSPECTION_COLLECTION),
            "this_inspection_id": inspection_id,
        },
        "timing": {
            "embed_ms": round(embed_time * 1000),
            "search_ms": round(search_time * 1000),
            "gemini_ms": round(gemini_time * 1000),
            "total_ms": round((time.time() - start) * 1000),
        },
    }


@app.post("/api/identify-part")
async def identify_part_endpoint(
    photo: UploadFile = File(...),
    equipment_model: str = Form(default="CAT 320"),
):
    store = get_store()
    image_bytes = await photo.read()
    embedding = embed_image(image_bytes)
    similar_parts = store.search(PARTS_COLLECTION, query_vector=embedding, top_k=5)

    try:
        result = identify_part(image_bytes, equipment_model, similar_parts)
    except Exception as e:
        result = {"part_name": "Unknown", "error": str(e)}

    return {
        "identification": result,
        "catalog_matches": [
            {
                "raw_score": round(s["score"], 4),
                "match_strength": normalize_score(s["score"], "cross_modal"),
                "part_name": s["payload"].get("part_name", "?"),
                "part_number": s["payload"].get("part_number", "?"),
                "service_info": s["payload"].get("service_info", ""),
            }
            for s in similar_parts
        ],
    }


@app.post("/api/ocr")
async def ocr_endpoint(
    photo: UploadFile = File(...),
    equipment_model: str = Form(default=""),
):
    """Read serial plate / nameplate from equipment photo using Gemini vision."""
    image_bytes = await photo.read()
    try:
        result = ocr_nameplate(image_bytes, equipment_model)
    except Exception as e:
        result = {"serial_number": "Error", "error": str(e), "confidence": 0.0}
    return {"ocr_result": result}


@app.get("/api/maintenance/{hours}")
async def maintenance_endpoint(hours: int):
    """Get maintenance schedule for given machine hours."""
    return get_maintenance_schedule(hours)


@app.get("/api/maintenance-schedule")
async def maintenance_schedule_endpoint(hours: int = 0):
    """Get maintenance schedule — query param variant used by frontend."""
    return get_maintenance_schedule(hours)


@app.post("/api/generate-report")
async def generate_report_endpoint(body: dict):
    findings_list = body.get("findings", [])
    equipment_model = body.get("equipment_model", "CAT 320")
    equipment_hours = int(body.get("equipment_hours", 5000))
    report_html = generate_report(findings_list, equipment_model, equipment_hours)
    return {"report": report_html, "format": "html"}


@app.post("/api/reset-inspections")
async def reset_user_inspections():
    """
    Clear only user-added inspections (IDs >= 1000) while keeping seed data (IDs 1-999).
    Resets the inspection counter. Parts catalog is unaffected.
    """
    store = get_store()
    store.reset_user_inspections()
    global _next_id
    _next_id["inspection"] = 1000
    return {
        "status": "reset",
        "inspections": store.count(INSPECTION_COLLECTION),
        "message": f"User inspections cleared. Seed data retained.",
    }


@app.post("/api/reseed")
async def reseed_database():
    """Delete ALL collections and re-seed from scratch with canonical seed data."""
    store = get_store()
    store.delete_collection(INSPECTION_COLLECTION)
    store.delete_collection(PARTS_COLLECTION)

    store._ensure_collections()

    from scripts.seed_data import INSPECTION_RECORDS, PARTS_CATALOG

    for rec in INSPECTION_RECORDS:
        vec = embed_text(rec["text_for_embedding"])
        payload = {k: v for k, v in rec.items() if k not in ("id", "text_for_embedding")}
        store.upsert(INSPECTION_COLLECTION, id=rec["id"], vector=vec, payload=payload)

    for part in PARTS_CATALOG:
        vec = embed_text(part["text_for_embedding"])
        payload = {k: v for k, v in part.items() if k not in ("id", "text_for_embedding")}
        store.upsert(PARTS_COLLECTION, id=part["id"], vector=vec, payload=payload)

    global _next_id
    _next_id["inspection"] = 1000

    return {
        "status": "reseeded",
        "inspections": store.count(INSPECTION_COLLECTION),
        "parts": store.count(PARTS_COLLECTION),
    }


@app.get("/api/stats")
async def get_stats():
    store = get_store()
    return {
        "inspection_count": store.count(INSPECTION_COLLECTION),
        "parts_count": store.count(PARTS_COLLECTION),
        "vectordb_type": "Actian VectorAI DB" if store.use_cortex else "In-Memory Fallback",
    }


@app.get("/api/memory-stats")
async def memory_stats():
    """Fleet count endpoint used by the frontend header."""
    store = get_store()
    return {"total_inspections": store.count(INSPECTION_COLLECTION)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)