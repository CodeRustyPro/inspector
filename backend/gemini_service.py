"""
Gemini AI service — google-genai SDK, Gemini 3 Flash.
Domain-specific prompts with Cat equipment knowledge baked in.

v0.4: Maintenance interval engine, OCR nameplate reading,
      MSHA/OSHA compliance flagging, institutional memory synthesis.
"""
import os, json, re
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
_client = None

def _get_client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client

def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: pass
        return None


# ============================================================
# Cat Equipment Domain Knowledge (baked into prompts)
# ============================================================

CAT_DOMAIN_KNOWLEDGE = """
CATERPILLAR EQUIPMENT INSPECTION STANDARDS:

RATING CRITERIA (per Cat Inspect walkaround methodology):
- GREEN (Satisfactory): Component functioning within normal parameters. No visible wear beyond expected for machine hours. All fluid levels and pressures within spec. No action required until next scheduled PM.
- YELLOW (Monitor/Schedule): Visible wear, minor degradation, or early-stage issue detected. Component still functional but trending toward failure. Schedule repair within next planned maintenance window (typically 250-500 hours). Document for trend tracking.
- RED (Critical/Action Required): Safety hazard, imminent failure risk, regulatory violation, or component has failed. Machine must not operate until repaired. Requires immediate work order and parts procurement.

MAINTENANCE INTERVALS (Cat SIS standard for excavators):
- 10 hours (daily): Walkaround inspection, fluid levels, track tension, safety equipment
- 250 hours (PM1): Fuel system water drain, swing bearing grease, linkage pin grease
- 500 hours (PM2): Engine oil + filter, hydraulic oil filter, fuel filters, air filter (check), belts
- 1000 hours (PM3): Hydraulic oil sample (S.O.S), coolant sample, swing gear oil
- 2000 hours (PM4): Hydraulic oil change, coolant change, valve adjustment
- 6000 hours (Major): Undercarriage assessment, structural inspection, pump flow test

COMMON FAILURE PATTERNS TO REFERENCE:
- Hydraulic hose failures progress: weeping → drip → stream → burst (catastrophic)
- Track component wear is measured in mm and compared to Cat published wear limits
- Oil analysis (Cat S.O.S Services): S.O.S is trend-based — decisions are made by comparing 3–5 sequential samples over time, not fixed single-sample PPM limits. Rising silicon trends indicate air/oil entry points (air filter, crankcase breather, hose joints); rising iron trends indicate wear metal accumulation. Consult Cat S.O.S Services lab report for sample-specific guidance.
- Hoses have a 6-year age limit regardless of visual condition
- Seized pins result from inadequate greasing — check auto-lube system
- Belt cracks >1mm depth require replacement within 100 operating hours
- Backup alarms, fire extinguishers, and lighting are MSHA/OSHA safety-critical items
- Final drive duo-cone seal failures: debris packing → seal displacement → gear oil leak

COST/DOWNTIME CONTEXT (helps prioritize):
- Hydraulic hose replacement: $200-800, 2-4 hours
- Hydraulic pump failure: $8,000-15,000, 2-3 days
- Cylinder rebuild: $5,000-12,000, 3-5 days
- Undercarriage rebuild: $25,000-50,000, 1-2 weeks
- Turbocharger: $3,000-6,000, 1-2 days
- Engine overhaul: $30,000-60,000, 2-4 weeks
- Final drive seal replacement: $1,500-3,000, 1-2 days
- Final drive complete failure: $15,000-25,000, 1-2 weeks

SAFETY-CRITICAL ITEMS (any RED rating triggers a compliance flag with the specific CFR citation): ROPS, seatbelts, brakes, backup alarms, fire extinguishers, lighting, steering, emergency stops, structural integrity (boom, frame).
Regulatory text is sourced live from the eCFR database and provided separately in each inspection prompt.
"""


# ============================================================
# Equipment Model Specs — structured data beyond Gemini opinion
# ============================================================

EQUIPMENT_SPECS = {
    "CAT 320": {
        "engine": "Cat C4.4 ACERT, 148 hp",
        "operating_weight": "49,603 lbs",
        "hydraulic_system_pressure": "5076 PSI",
        "hydraulic_oil_capacity": "58.7 gal",
        "engine_oil_capacity": "3.7 gal",
        "coolant_capacity": "5.8 gal",
        "fuel_tank": "79.3 gal",
        "max_dig_depth": "22 ft 2 in",
        "typical_rebuild_value": "$350,000-$500,000",
        "component_life_hours": {
            "hydraulic_hose": "6000 hrs or 6 years (whichever first)",
            "track_shoe": "3000-6000 hrs depending on application",
            "track_roller": "4000-8000 hrs",
            "engine_overhaul": "10000-15000 hrs",
            "hydraulic_pump": "8000-12000 hrs",
            "swing_bearing": "8000-12000 hrs",
            "undercarriage_rebuild": "5000-8000 hrs",
        },
    },
    "CAT 336": {
        "engine": "Cat C9.3B, 268 hp",
        "operating_weight": "79,600 lbs",
        "hydraulic_system_pressure": "5076 PSI",
        "hydraulic_oil_capacity": "103 gal",
        "engine_oil_capacity": "7.1 gal",
        "coolant_capacity": "10 gal",
        "fuel_tank": "152 gal",
        "max_dig_depth": "25 ft 3 in",
        "typical_rebuild_value": "$400,000-$600,000",
        "component_life_hours": {
            "hydraulic_hose": "6000 hrs or 6 years",
            "track_shoe": "3000-6000 hrs",
            "track_roller": "4000-8000 hrs",
            "engine_overhaul": "12000-18000 hrs",
            "hydraulic_pump": "10000-15000 hrs",
            "swing_bearing": "10000-14000 hrs",
            "undercarriage_rebuild": "5000-8000 hrs",
        },
    },
    "CAT 349": {
        "engine": "Cat C13, 400 hp",
        "operating_weight": "110,000 lbs",
        "hydraulic_system_pressure": "5076 PSI",
        "typical_rebuild_value": "$500,000-$750,000",
    },
    "CAT 950": {
        "engine": "Cat C7.1, 217 hp",
        "operating_weight": "39,905 lbs",
        "typical_rebuild_value": "$300,000-$450,000",
    },
}

SAFETY_CRITICAL_COMPONENTS = [
    "backup_alarm", "lights", "brakes", "seatbelt", "rops", "fire_extinguisher",
    "steering", "emergency_stop", "boom_structure", "frame", "cab_glass",
]


# ============================================================
# Live Regulation Retrieval (eCFR → VectorAI DB → Prompt)
# ============================================================

# Component → regulation tag mapping (tags match what was indexed in fetch_regulations.py)
# Used to filter the regulations collection so retrieval is precise even though CLIP
# was trained on image-text pairs, not regulatory text.
_COMPONENT_TAG_MAP = {
    "fire_extinguisher": "fire_extinguisher",
    "backup_alarm":      "backup_alarm",
    "brakes":            "brakes",
    "rops":              "rops",
    "seatbelt":          "rops",        # ROPS section (56.14130) covers seatbelts
    "lights":            "lights",
    "steering":          "inspection",  # General mechanical integrity — 56.14100
    "emergency_stop":    "inspection",
    "boom_structure":    "inspection",
    "frame":             "inspection",
    "cab_glass":         "inspection",
}


def get_regulation_context(component_name: str, top_k: int = 4) -> tuple[str, str]:
    """
    Retrieve actual CFR regulation text from VectorAI DB for the given component.
    Uses tag-filtered vector search so each component maps to its correct regulatory
    subpart (e.g. fire_extinguisher → 56.4230, not the generic inspection section).

    Returns (regulation_text_for_prompt, primary_citation_for_flag).

    Gracefully returns empty strings if the regulations collection is not seeded —
    run `python -m scripts.fetch_regulations` to populate it first.
    """
    try:
        # Lazy import to avoid circular deps at module load time
        from backend.vectordb import get_store, REGULATIONS_COLLECTION
        from backend.embeddings import embed_text as _embed

        store = get_store()
        if store.count(REGULATIONS_COLLECTION) == 0:
            return "", "MSHA 30 CFR 56.14100(c)"

        # Determine the regulation tag for this component
        _comp_lower = component_name.lower().replace(" ", "_")
        reg_tag = next(
            (tag for key, tag in _COMPONENT_TAG_MAP.items() if key in _comp_lower),
            None
        )

        # Natural-language query for vector search
        readable = component_name.replace("_", " ").strip()
        query = f"{readable} safety regulation mining equipment"
        vec = _embed(query)

        # Strategy: tag-filtered search first (precise), fallback to unfiltered
        results = []
        if reg_tag:
            results = store.search(
                REGULATIONS_COLLECTION, vec, top_k=top_k,
                filter_field="tag", filter_value=reg_tag
            )
        # Always append the general inspection regulation (56.14100) for safety-critical items
        inspection_results = store.search(
            REGULATIONS_COLLECTION, vec, top_k=2,
            filter_field="tag", filter_value="inspection"
        )
        # Merge: tagged results first, then inspection, deduplicate by citation
        seen_citations: list[str] = []
        merged = []
        for r in (results + inspection_results):
            c = r.get("payload", {}).get("citation", "")
            if c not in seen_citations:
                seen_citations.append(c)
                merged.append(r)
        if not merged:
            merged = store.search(REGULATIONS_COLLECTION, vec, top_k=top_k)

        if not merged:
            return "", "MSHA 30 CFR 56.14100(c)"

        lines = [
            "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "APPLICABLE REGULATIONS (verbatim from official eCFR database — ecfr.gov):",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        rendered_citations: list[str] = []
        primary_citation = "MSHA 30 CFR 56.14100(c)"

        for r in merged:
            p = r.get("payload", {})
            citation  = p.get("citation", "")
            sec_title = p.get("section_title", "")
            text      = p.get("text", "")
            source    = p.get("source_url", "https://www.ecfr.gov")

            if citation not in rendered_citations:
                rendered_citations.append(citation)
                lines.append(f"\n§ {citation} — {sec_title}")

            lines.append(f'  "{text}"')

            if not primary_citation or primary_citation == "MSHA 30 CFR 56.14100(c)":
                if reg_tag and p.get("tag") == reg_tag:
                    primary_citation = citation

        lines.append(f"\n  [Source: eCFR — {source}]")
        # Ensure 56.14100(c) is always cited as the removal-from-service anchor
        if "30 CFR 56.14100" not in "\n".join(rendered_citations):
            lines.append(
                "\n§ 30 CFR 56.14100(c) — Safety defects; examination, correction and records\n"
                '  "When defects make continued operation hazardous to persons, the defective items '
                "including self-propelled mobile equipment shall be taken out of service and placed in "
                "a designated area posted for that purpose, or a tag or other effective method of "
                'marking the defective items shall be used to prohibit further use until the defects are corrected."'
            )

        return "\n".join(lines), primary_citation

    except Exception as e:
        print(f"  [regulation_context] retrieval failed: {e}")
        return "", "MSHA 30 CFR 56.14100(c)"


# ============================================================
# Maintenance Interval Engine
# ============================================================

def get_maintenance_schedule(hours: int) -> dict:
    """
    Calculate upcoming maintenance milestones based on current machine hours.
    Returns the nearest PM with tasks and required parts.
    """
    PMS = [
        (250,  "PM1", "Fuel system water drain, swing bearing grease, linkage pin grease, belt inspection",
         []),
        (500,  "PM2", "Engine oil + filter change, hydraulic oil filter, fuel filters, air filter check",
         ["1R-1807 engine oil filter", "1R-0751 fuel filter", "4I-3948 hydraulic spin-on"]),
        (1000, "PM3", "S.O.S fluid sample, coolant sample, swing gear oil, valve lash adjustment",
         ["S.O.S sampling kit"]),
        (2000, "PM4", "Complete hydraulic oil change, coolant change, valve adjustment, ROPS inspection",
         ["094-4412 hydraulic return filter", "7Y-4748 suction screen", "Cat HYDO Advanced 10"]),
        (6000, "Major Service", "Undercarriage assessment, structural inspection, pump flow test, full fluid analysis",
         []),
    ]

    upcoming = []
    for interval, name, tasks, parts in PMS:
        next_due = ((hours // interval) + 1) * interval
        remaining = next_due - hours
        upcoming.append({
            "name": name,
            "interval_hours": interval,
            "due_at_hours": next_due,
            "hours_remaining": remaining,
            "tasks": tasks,
            "parts": parts,
        })

    upcoming.sort(key=lambda x: x["hours_remaining"])

    # The nearest milestone — check if multiple PMs align at the same hour
    nearest = upcoming[0]
    aligned = [pm for pm in upcoming if pm["due_at_hours"] == nearest["due_at_hours"]]
    if len(aligned) > 1:
        # Use the highest-level PM since it includes all lower tasks
        highest = max(aligned, key=lambda x: x["interval_hours"])
        highest["includes"] = [pm["name"] for pm in aligned]
        # Merge all parts from aligned PMs
        all_parts = []
        for pm in aligned:
            all_parts.extend(pm["parts"])
        highest["parts"] = list(dict.fromkeys(all_parts))  # dedupe preserving order
        nearest = highest

    urgency = "normal"
    if nearest["hours_remaining"] <= 25:
        urgency = "imminent"
    elif nearest["hours_remaining"] <= 100:
        urgency = "approaching"

    return {
        "current_hours": hours,
        "next_service": nearest,
        "urgency": urgency,
        "upcoming": upcoming[:3],
    }


# ============================================================
# OCR — Nameplate / Serial Plate Reading
# ============================================================

def ocr_nameplate(image_bytes: bytes, equipment_model: str = "") -> dict:
    """Read serial number, model, and date code from equipment nameplate photo."""
    client = _get_client()

    prompt = f"""You are a Caterpillar equipment identification specialist with expertise in reading nameplates, serial plates, and component identification tags on heavy machinery.

Equipment context (if known): {equipment_model}

INSTRUCTIONS:
1. Carefully examine the image for any visible text, numbers, barcodes, or identification markings
2. Extract ALL readable text — serial numbers, model numbers, date codes, part numbers
3. For Cat equipment, serial numbers typically follow: 3-letter prefix + 5 digits (e.g., DKS00001) or legacy format #L-#### (e.g., 1R-1807)
4. Note the condition of the plate itself — is it clear, worn, corroded, partially covered?

Respond ONLY with valid JSON (no markdown fences):
{{
    "serial_number": "The primary serial or identification number, or 'Not visible'",
    "model_number": "Equipment model/type designation if visible",
    "date_code": "Manufacturing or service date if visible",
    "part_number": "Part number if this is a component plate",
    "additional_text": "Any other readable text or markings",
    "confidence": 0.85,
    "plate_condition": "clear | worn | corroded | partially_obscured | damaged"
}}"""

    image_part = types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(parts=[types.Part(text=prompt), image_part])],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    result = _parse_json(response.text)
    if result is None:
        result = {
            "serial_number": "Unable to read",
            "confidence": 0.0,
            "raw_response": response.text[:500],
        }
    return result


# ============================================================
# Core Inspection Analysis (enhanced v0.4)
# ============================================================

def analyze_inspection(
    image_bytes: bytes,
    voice_transcript: str,
    component_name: str,
    equipment_model: str,
    equipment_hours: int,
    similar_cases: list,
    escalation_risk: dict = None,
    comparison_cases: list = None,
) -> dict:
    client = _get_client()

    # --- Fleet history context ---
    history_text = ""
    if similar_cases:
        history_text = "\n\nFLEET INSPECTION MEMORY (similar past cases from VectorAI DB — hybrid image+text search):\n"
        for i, case in enumerate(similar_cases[:5]):
            p = case.get("payload", {})
            history_text += (
                f"  Case {i+1} (relevance: {case.get('score', 0):.2f}): "
                f"{p.get('equipment_model', '?')} at {p.get('hours', '?')} hrs | "
                f"Component: {p.get('component', '?')} | "
                f"Rating: {p.get('rating', '?')} | "
                f"Finding: {p.get('finding', 'N/A')} | "
                f"Action taken: {p.get('action', 'N/A')} | "
                f"Outcome: {p.get('outcome', 'N/A')}\n"
            )

    # --- Escalation risk context ---
    escalation_text = ""
    if escalation_risk and escalation_risk.get("has_escalation_risk"):
        escalation_text = "\n\nFLEET BACKGROUND — Some past cases of this component type escalated:\n"
        for esc in escalation_risk.get("escalation_cases", [])[:3]:
            escalation_text += (
                f"  - {esc['equipment']} at {esc['hours']} hrs: "
                f"Was rated {esc['original_rating']}, outcome: {esc['outcome']}\n"
            )
        cost = escalation_risk.get("cost_range")
        if cost:
            escalation_text += f"  Historical repair cost range: ${cost['min']:,} - ${cost['max']:,} (avg ${cost['avg']:,}, n={cost['sample_size']})\n"
        escalation_text += "  Only mention this if the CURRENT PHOTO actually shows visible signs of that same failure mode. Do not apply escalation risk to a clean or merely dusty component.\n"

    # --- Photo-over-time comparison context ---
    comparison_text = ""
    if comparison_cases:
        comparison_text = "\n\n📸 PHOTO-OVER-TIME — Previous inspection of the SAME component on the SAME equipment:\n"
        for i, cc in enumerate(comparison_cases[:2]):
            p = cc.get("payload", {})
            comparison_text += (
                f"  Previous ({p.get('date', 'unknown date')}): "
                f"{p.get('equipment_model', '?')} at {p.get('hours', '?')} hrs | "
                f"Rating: {p.get('rating', '?')} | "
                f"Finding: {p.get('finding', 'N/A')}\n"
            )
        comparison_text += (
            "  COMPARE the current photo against these past findings. "
            "Note any progression of wear, degradation, or improvement since the last inspection. "
            "Estimate a rate of change if possible.\n"
        )

    # --- Maintenance context ---
    maint = get_maintenance_schedule(equipment_hours)
    next_pm = maint.get("next_service", {})
    maint_text = (
        f"\n\nMAINTENANCE CONTEXT: Next service is {next_pm.get('name', '?')} "
        f"due at {next_pm.get('due_at_hours', '?')} hrs "
        f"({next_pm.get('hours_remaining', '?')} hrs remaining). "
        f"Tasks: {next_pm.get('tasks', 'N/A')}\n"
    )

    # --- Equipment model specs (structured data — not Gemini's opinion) ---
    equip_spec_text = ""
    spec = EQUIPMENT_SPECS.get(equipment_model)
    if spec:
        equip_spec_text = f"\n\nEQUIPMENT DATA (from Cat SIS / manufacturer specs — cross-reference with visual findings):\n"
        equip_spec_text += f"  Model: {equipment_model}\n"
        for k, v in spec.items():
            if k == "component_life_hours":
                equip_spec_text += "  Expected component life:\n"
                for comp, life in v.items():
                    equip_spec_text += f"    - {comp.replace('_', ' ')}: {life}\n"
            else:
                equip_spec_text += f"  {k.replace('_', ' ').title()}: {v}\n"
        equip_spec_text += (
            f"  Current hours: {equipment_hours}\n"
            f"  NOTE: If the inspected component is approaching or past its expected life hours, factor this into your severity assessment.\n"
        )

    # --- Live regulation retrieval from eCFR-seeded VectorAI DB ---
    _comp_lower = component_name.lower()
    regulation_text, _reg_citation = get_regulation_context(component_name)

    is_safety_critical = any(
        sc in _comp_lower
        for sc in SAFETY_CRITICAL_COMPONENTS
    )
    compliance_instruction = ""
    if is_safety_critical:
        compliance_instruction = f"""
COMPLIANCE CHECK REQUIRED: This is a safety-critical component.
The applicable regulatory text has been retrieved verbatim from the eCFR database above.
If you rate this RED, you MUST include a compliance_flag citing the specific CFR section(s)
from the regulation text above. 30 CFR 56.14100(c) applies universally — equipment with
safety defects must be removed from service until repaired.
Required documentation: defect tag, supervisor notification, lock out/tag out, safety log entry.
"""

    prompt = f"""You are an expert Caterpillar equipment field inspector with 20+ years of experience. You have deep knowledge of Cat maintenance intervals, failure modes, and repair costs. You also serve as an institutional memory — sharing wisdom from fleet-wide inspection data with newer technicians.

{CAT_DOMAIN_KNOWLEDGE}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — IMAGE VALIDITY CHECK (evaluate this FIRST before anything else):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Look at the image carefully. Ask yourself: Is the PRIMARY SUBJECT of this photo construction/industrial equipment or an equipment component?

VALID images (set is_valid_equipment_image=true):
  - Heavy machinery: excavators, wheel loaders, dozers, articulated trucks, off-highway trucks
  - Equipment components: hydraulic hoses, cylinders, pumps, tracks, rollers, sprockets, idlers
  - Engine/drivetrain: filters, belts, turbochargers, cooling systems, oil levels
  - Safety items ON EQUIPMENT: fire extinguishers mounted in cab, backup alarms, ROPS, seatbelts
  - Ground engaging tools: bucket teeth, cutting edges, blade edges
  - Structural: boom welds, cracks, pin/bushing wear
  - Identification: nameplates, serial plates, component tags on equipment
  - Fluids/gauges: dipsticks, sight glasses, oil samples, service indicator lights

INVALID images (set is_valid_equipment_image=false):
  - Photographs of PEOPLE, faces, selfies, portraits — even in industrial settings
  - Pure landscapes, snow scenes, nature without visible equipment in the foreground
  - Animals, pets
  - Food, household items, personal belongings
  - Standard cars, bicycles, motorcycles (not heavy equipment)
  - Buildings or architecture with no equipment context
  - Abstract images, screenshots of documents (unless it's an equipment nameplate)
  - Completely blurry or entirely black/overexposed images
  - Any image where you genuinely cannot identify an equipment-related subject

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT INSPECTION:
- Equipment: {equipment_model}
- Machine hours: {equipment_hours}
- Component being inspected: {component_name}
- Inspector's voice note: "{voice_transcript if voice_transcript else 'No voice note provided'}"
{history_text}
{escalation_text}
{comparison_text}
{maint_text}
{equip_spec_text}
{regulation_text}
{compliance_instruction}

STEP 2 — INSTRUCTIONS (only if is_valid_equipment_image=true):
1. BASE YOUR FINDING STRICTLY ON WHAT YOU CAN SEE. Dust is dust. Oil film is oil film. A drip is a drip. Do not diagnose a failure mode unless you can see it. Do not upgrade "dusty" to "weeping seal" unless you can see an actual oil film in the photo.
2. RATE CONSERVATIVELY. If the component looks clean or merely dirty with no active leak, fluid loss, or structural damage — rate GREEN and say so. Only rate YELLOW if there is a visible early-stage defect (wear, minor seepage, early cracking). Only rate RED for clear failures, safety hazards, or regulatory violations.
3. Cross-reference the inspector's voice note — it may clarify what looks ambiguous in the photo.
4. Fleet history provides BACKGROUND CONTEXT only. Do NOT apply a fleet escalation pattern to the current inspection unless the current photo shows the same specific failure trigger (e.g., visible oil weeping, not just dust).
5. For part numbers, only cite numbers you are highly confident about; otherwise say "verify P/N via Cat SIS".
6. Note if a PM service is approaching and it would be a logical time to address this item.
7. If photo-over-time data exists, compare current condition to the previous inspection and note progression.
8. fleet_pattern_note: ONLY populate this if fleet history contains a case with the SAME visible defect. If the current photo shows dust/dirt only and fleet history shows a weeping seal — do NOT connect them. Set to empty string if there is no direct match.
9. Set detected_component to the most specific matching component key (see allowed values in schema).

Respond ONLY with valid JSON (no markdown fences, no extra text):
{{
    "is_valid_equipment_image": true,
    "detected_subject": "Concise description of the primary subject visible in the image (e.g. 'hydraulic hose with black bulging crack', 'CAT 320D serial nameplate', 'red fire extinguisher with white residue'). If invalid, describe what you actually see (e.g. 'person in snow', 'dog', 'food').",
    "detected_component": "Best matching component key — MUST be one of: hydraulic_hose|hydraulic_cylinder|hydraulic_pump|hydraulic_oil|track_shoe|track_tension|track_roller|final_drive|boom_pin|boom_structure|stick_pin|bucket_teeth|bucket_edge|swing_bearing|engine_oil|coolant_system|belt|air_filter|fuel_filter|turbocharger|transmission|lights|backup_alarm|fire_extinguisher|brakes|seatbelt|cab_glass|tire|alternator|starter_motor|general",
    "rating": "GREEN|YELLOW|RED",
    "confidence": 0.85,
    "finding": "Describe only what is DIRECTLY VISIBLE in the photo. State the actual condition (e.g. 'hoses are dusty and dry, no oil film visible', 'O-ring fitting shows minor oil weeping with dust accumulation'). Do not infer hidden failures. If is_valid_equipment_image=false, set to 'Invalid image — not construction equipment'.",
    "reasoning": "Explain your rating based on the visual evidence. Keep it grounded — if the component looks fine except for surface dirt, say that. Reference machine hours and PM schedule if relevant. If is_valid_equipment_image=false, briefly state why.",
    "action": "Practical next steps appropriate to the actual finding. For a dusty-but-intact component, 'clean at next scheduled service' is a valid action. Only recommend urgent intervention if the visual evidence warrants it. If is_valid_equipment_image=false, set to 'Please photograph actual equipment or equipment components.'",
    "severity_score": 5,
    "historical_context": "Summarize only directly relevant fleet history. If history shows a weeping seal case but current photo shows only dust, note they are different conditions. State 'No matching fleet history for this specific condition' if nothing directly applies.",
    "escalation_warning": "Only populate if the CURRENT PHOTO shows a visible failure trigger that fleet data has shown to escalate. If the photo shows a clean or dusty-only component, set to 'None'.",
    "cost_estimate": "Based on fleet history and Cat service data, estimated repair cost range and downtime. E.g. '$200-800, 2-4 hours' or 'N/A if insufficient data'.",
    "condition_trend": "If photo-over-time data exists: describe how condition changed since last inspection. E.g. 'Wear progressed ~15% since last inspection at 4,200 hrs (500 hrs ago). At current rate, replacement needed within 400 hrs.' If no comparison data: 'No previous inspection data for comparison.'",
    "fleet_pattern_note": "1–2 sentence factual summary of what the fleet history data above shows for this specific defect type — e.g. progression rates, repair outcomes, costs. Cite only data explicitly present in the fleet history cases above. Leave as empty string if no relevant cases or if is_valid_equipment_image=false.",
    "compliance_flag": null
}}

For compliance_flag, if a safety-critical component is rated RED, set it to:
{{
    "regulation": "The specific CFR section(s) from the regulation text above that apply to this component",
    "description": "Verbatim or close paraphrase of the key requirement from the regulation text above",
    "action_required": "Generate defect tag, notify supervisor, lock out/tag out per 30 CFR 56.14100(c), document in safety log"
}}
Otherwise keep compliance_flag as null.
If is_valid_equipment_image=false, keep compliance_flag as null and set severity_score=0, confidence=0.0, rating=YELLOW."""

    image_part = types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(parts=[types.Part(text=prompt), image_part])],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    result = _parse_json(response.text)
    if result is None:
        result = {
            "rating": "YELLOW", "confidence": 0.5,
            "finding": response.text[:300],
            "reasoning": "Could not parse structured response",
            "action": "Manual review recommended",
            "severity_score": 5,
            "historical_context": "N/A",
            "condition_trend": "No previous inspection data for comparison.",
            "fleet_pattern_note": "",
            "compliance_flag": None,
        }
    return result


def identify_part(image_bytes: bytes, equipment_model: str, similar_parts: list) -> dict:
    client = _get_client()

    catalog_text = ""
    if similar_parts:
        catalog_text = "\n\nPARTS CATALOG MATCHES (from VectorAI DB similarity search):\n"
        for i, part in enumerate(similar_parts[:5]):
            p = part.get("payload", {})
            catalog_text += (
                f"  Match {i+1} (sim: {part.get('score', 0):.2f}): "
                f"{p.get('part_name', '?')} — P/N: {p.get('part_number', '?')}, "
                f"Models: {p.get('compatible_models', [])}, "
                f"Info: {p.get('service_info', 'N/A')}\n"
            )

    prompt = f"""You are a Caterpillar parts specialist with extensive knowledge of Cat part numbers, applications, and service requirements.

Equipment context: {equipment_model}
{catalog_text}

INSTRUCTIONS:
1. Identify the specific part in the photo — be precise (not just "hydraulic hose" but "high pressure hydraulic hose assembly with JIC flare fittings")
2. Match against the catalog entries above — select the best match or indicate if no exact match
3. Assess the part's current condition based on the photo
4. Provide a specific, useful recommendation — not just "ok" but actionable service advice

Respond ONLY with valid JSON (no markdown):
{{
    "part_name": "Precise part name",
    "part_number": "Cat P/N from catalog or 'Not in catalog'",
    "confidence": 0.85,
    "condition": "Detailed condition assessment from photo — new, serviceable, worn, damaged, failed",
    "compatible_models": ["List of compatible Cat models"],
    "recommendation": "Specific service recommendation: installation instructions, replacement interval, what to inspect, storage requirements — provide real value to the technician",
    "service_info": "Relevant maintenance interval or service note for this part type"
}}"""

    image_part = types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(parts=[types.Part(text=prompt), image_part])],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    result = _parse_json(response.text)
    if result is None:
        result = {
            "part_name": "Unknown", "part_number": "Unknown", "confidence": 0.0,
            "condition": response.text[:200], "compatible_models": [],
            "recommendation": "Manual identification required", "service_info": "",
        }
    return result


def generate_report(findings: list, equipment_model: str, equipment_hours: int) -> str:
    client = _get_client()

    from datetime import datetime
    today = datetime.now().strftime("%B %d, %Y")

    maint = get_maintenance_schedule(equipment_hours)
    next_pm = maint.get("next_service", {})

    prompt = f"""Generate a professional equipment inspection report that a Cat dealer service manager would be proud to send to a customer.

Equipment: {equipment_model}
Current Hours: {equipment_hours}
Inspection Date: {today}
Next Scheduled Service: {next_pm.get('name', '?')} at {next_pm.get('due_at_hours', '?')} hrs ({next_pm.get('hours_remaining', '?')} hrs remaining)

Findings from this inspection:
{json.dumps(findings, indent=2)}

{CAT_DOMAIN_KNOWLEDGE}

REPORT FORMAT (use HTML tags for formatting — this will be rendered in a web browser):

<h2>EQUIPMENT INSPECTION REPORT</h2>
<p><strong>Equipment:</strong> {equipment_model} | <strong>Hours:</strong> {equipment_hours} | <strong>Date:</strong> {today}</p>
<p><strong>Next Service:</strong> {next_pm.get('name', '?')} due at {next_pm.get('due_at_hours', '?')} hrs ({next_pm.get('hours_remaining', '?')} hrs remaining)</p>
<p><strong>Overall Status:</strong> [worst rating color: 🟢 OPERATIONAL / 🟡 CAUTION — MONITOR / 🔴 ACTION REQUIRED — DO NOT OPERATE]</p>

<hr style="border:1px solid #E5E7EB; margin:24px 0;">

<h3>1. EXECUTIVE SUMMARY</h3>
<p style="color:#4B5563; line-height:1.6;">[2-3 sentences summarizing overall machine condition, most critical finding, and whether machine is cleared for operation. If compliance flags exist, note them here.]</p>

<h3>2. DETAILED FINDINGS</h3>
[For each finding, include:]
<div style="margin:16px 0;padding:16px;background:#FAFAFA;border-radius:8px;border:1px solid #E5E7EB;border-left:4px solid [rating color: #15803D for GREEN, #A16207 for YELLOW, #B91C1C for RED]">
<p style="margin-bottom:8px;"><strong>Component:</strong> [name] | <strong>Rating:</strong> [GREEN/YELLOW/RED] | <strong>Confidence:</strong> [X%]</p>
<p style="margin-bottom:8px;color:#4B5563;"><strong>Observation:</strong> [what was found]</p>
<p style="margin-bottom:8px;color:#4B5563;"><strong>Risk:</strong> [what happens if not addressed, including cost/downtime]</p>
<p style="margin-bottom:8px;color:#4B5563;"><strong>Historical Pattern:</strong> [reference fleet data if available]</p>
<p style="margin-bottom:0;color:#6B7280;font-size:13px;"><strong>Fleet Pattern:</strong> [1–2 sentence factual summary of what fleet history shows for this defect type, if relevant data exists — otherwise omit]</p>
</div>

<h3>3. REGULATORY COMPLIANCE</h3>
<p style="color:#4B5563; line-height:1.6;">[List any compliance flags. If none: "No regulatory violations identified in this inspection."]</p>

<h3>4. PRIORITIZED ACTION ITEMS</h3>
[Numbered list ordered by severity:]
<p style="color:#4B5563; line-height:1.6;">1. <strong>[IMMEDIATE/SCHEDULE/MONITOR]</strong> — [action with Cat P/N] — [cost/downtime estimate]</p>

<h3>5. NEXT SERVICE</h3>
<p style="color:#4B5563; line-height:1.6;">[Based on current hours, detail the next PM with required parts list and estimated cost]</p>

Write as a senior Cat dealer technician. Be specific, reference real intervals, provide genuine value."""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    text = response.text.strip()
    if text.startswith("```html"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


if __name__ == "__main__":
    print(f"Model: {GEMINI_MODEL}")
    print(f"API key set: {'Yes' if GEMINI_API_KEY else 'NO'}")

    # Test maintenance schedule
    for hrs in [450, 980, 1950, 5900]:
        m = get_maintenance_schedule(hrs)
        ns = m["next_service"]
        print(f"  {hrs} hrs → Next: {ns['name']} at {ns['due_at_hours']}hrs ({ns['hours_remaining']}hrs away) [{m['urgency']}]")

    if GEMINI_API_KEY:
        c = _get_client()
        r = c.models.generate_content(
            model=GEMINI_MODEL,
            contents="Say 'Gemini 3 Flash connected' and nothing else.",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        print(f"✅ Test: {r.text}")