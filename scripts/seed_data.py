"""
Seed VectorAI DB with realistic Cat equipment inspection history.

50+ records covering real Cat maintenance intervals, failure progressions,
actual part numbers, and realistic outcomes with cost/downtime data.

Usage: python -m scripts.seed_data
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.vectordb import get_store, INSPECTION_COLLECTION, PARTS_COLLECTION
from backend.embeddings import embed_text

INSPECTION_RECORDS = [
    # === HYDRAULIC SYSTEM - Failure progression: seep → drip → burst ===
    {"id": 1, "text_for_embedding": "hydraulic hose leak active oil drip from boom cylinder fitting high pressure line excavator",
     "component": "hydraulic_hose", "equipment_model": "CAT 320", "hours": 4200, "rating": "RED",
     "finding": "Active oil drip from boom cylinder hose fitting at JIC 37-degree flare connection, ~1 drop per 5 seconds. Hose shows external abrasion from contact with boom structure. System pressure drops visible during full extension.",
     "action": "Lock out/tag out. Replace hose assembly (verify P/N via Cat SIS against machine serial — boom hose P/N varies by serial range) and inspect adjacent routing for chafe protection. Torque new JIC fitting to 62 ft-lbs per Cat spec.",
     "outcome": "hose_replaced_same_day_4hrs_downtime", "severity_score": 9},

    {"id": 2, "text_for_embedding": "minor hydraulic oil seep weeping around hose connection fitting excavator boom no active drip",
     "component": "hydraulic_hose", "equipment_model": "CAT 320", "hours": 3100, "rating": "YELLOW",
     "finding": "Light oil film (weeping) around boom hose O-ring face seal. No active drip. O-ring may be hardening from heat exposure. Hose exterior shows early UV degradation.",
     "action": "Monitor daily. Schedule hose and face seal O-ring replacement within 250 hours (6V-8398 is a single ORFS O-ring, not a kit — verify full hose assembly P/N via Cat SIS). Clean area and mark for tracking.",
     "outcome": "escalated_to_red_after_300hrs_oring_failed", "severity_score": 6},

    {"id": 3, "text_for_embedding": "clean hydraulic hose good condition no leaks tight connections excavator boom",
     "component": "hydraulic_hose", "equipment_model": "CAT 320", "hours": 1500, "rating": "GREEN",
     "finding": "Hose assemblies in good condition. Connections tight, no visible weeping. No cracking, abrasion, or UV damage. Routing clear of pinch points. Recently replaced per 2000-hour PM.",
     "action": "No action. Next inspection at 2000 hours per Cat SIS maintenance interval.",
     "outcome": "no_issues_through_3000hrs", "severity_score": 1},

    {"id": 4, "text_for_embedding": "hydraulic cylinder rod deep scoring scratched chrome damage pitting seal failure",
     "component": "hydraulic_cylinder", "equipment_model": "CAT 336", "hours": 6800, "rating": "RED",
     "finding": "Deep circumferential scoring on boom cylinder rod chrome. 3 grooves, deepest ~0.5mm. Active oil bypass past rod seal. Contaminated oil or debris ingress past wiper seal.",
     "action": "Remove from service. Cylinder rebuild: re-chrome rod, replace seal kit P/N 215-9985. Flush hydraulic system and replace filters. Investigate contamination source.",
     "outcome": "cylinder_rebuilt_72hrs_$8500_repair", "severity_score": 9},

    {"id": 5, "text_for_embedding": "hydraulic oil level low sight glass below minimum mark tank excavator",
     "component": "hydraulic_oil", "equipment_model": "CAT 320", "hours": 5200, "rating": "YELLOW",
     "finding": "Hydraulic oil 3 inches below minimum on sight glass with cylinders retracted. Dark amber color acceptable for 500-hr oil. Consumption exceeds normal 0.5L/250hrs.",
     "action": "Top off with Cat HYDO Advanced 10W. Check cylinder seals, hose connections, and swing motor for hidden leaks. Submit oil for Cat S.O.S analysis.",
     "outcome": "slow_leak_found_at_swing_motor_seal", "severity_score": 5},

    {"id": 6, "text_for_embedding": "hydraulic pump whining noise cavitation high pitch sound under load reduced flow",
     "component": "hydraulic_pump", "equipment_model": "CAT 330", "hours": 7200, "rating": "RED",
     "finding": "Main pump producing high-pitched whining/cavitation under load. Boom raise time 12s vs 8s spec. Case drain flow exceeds 5 GPM limit.",
     "action": "Do not operate under load. Schedule pump replacement. Check suction line for restriction, tank screen, oil level and viscosity. S.O.S sample before teardown.",
     "outcome": "pump_replaced_48hrs_$12000", "severity_score": 10},

    {"id": 7, "text_for_embedding": "new hydraulic hose protective shipping caps on fittings unused replacement part",
     "component": "hydraulic_hose", "equipment_model": "CAT 320", "hours": 5000, "rating": "YELLOW",
     "finding": "New replacement hydraulic hose with plastic shipping caps still on JIC fittings. Caps not removed before installation. Contamination risk if caps fragment during forced connection.",
     "action": "Remove all shipping caps. Inspect fitting threads for debris. Clean with lint-free cloth. Verify correct P/N before installation. Torque to spec.",
     "outcome": "caps_removed_installed_correctly", "severity_score": 4},

    {"id": 8, "text_for_embedding": "hydraulic hose bulging swelling blister deformation about to burst high pressure",
     "component": "hydraulic_hose", "equipment_model": "CAT 349", "hours": 5800, "rating": "RED",
     "finding": "Visible 2-inch bulge on high-pressure boom hose. Inner tube delamination from reinforcement braid. Past 6-year age limit. Imminent burst risk under 5000 PSI system pressure.",
     "action": "IMMEDIATE shutdown. Do not pressurize. Replace hose assembly. Verify date code on replacement. Inspect all hoses of same vintage.",
     "outcome": "hose_replaced_emergency_prevented_catastrophic_failure", "severity_score": 10},

    # === UNDERCARRIAGE / TRACK ===
    {"id": 10, "text_for_embedding": "track shoe bent damaged deformed grouser plate undercarriage excavator",
     "component": "track_shoe", "equipment_model": "CAT 320", "hours": 5500, "rating": "YELLOW",
     "finding": "One track shoe with edge deformation on outboard grouser. Impact damage from rocky material. Track shoe height 28mm vs 38mm new — 26% worn.",
     "action": "Monitor wear rate. Rotate tracks if single-side pattern. Replace when height reaches 20mm per Cat undercarriage guide. Check track tension.",
     "outcome": "two_more_bent_within_500hrs_full_replacement_6200hrs", "severity_score": 5},

    {"id": 11, "text_for_embedding": "multiple track shoes bent cracked severe wear undercarriage needs full replacement",
     "component": "track_shoe", "equipment_model": "CAT 320", "hours": 7200, "rating": "RED",
     "finding": "8 of 49 shoes bent/cracking at grouser welds. Left-side accelerated wear indicates misalignment. Shoe height 15mm — below Cat minimum 20mm.",
     "action": "Replace complete track group. Inspect rollers, idlers, sprockets for secondary damage. Verify track frame alignment per Cat SIS.",
     "outcome": "full_undercarriage_rebuild_$35000", "severity_score": 8},

    {"id": 12, "text_for_embedding": "track chain tension loose sagging slack undercarriage adjustment needed",
     "component": "track_tension", "equipment_model": "CAT 336", "hours": 3800, "rating": "YELLOW",
     "finding": "Track sag 3.5 inches at midspan — exceeds 1-2 inch Cat spec. Track riding off sprocket during swing. Recoil spring grease needs checking.",
     "action": "Adjust via grease fitting until sag is 1-2 inches. If adjustment doesn't hold, inspect recoil spring and adjuster cylinder for internal leak.",
     "outcome": "tension_adjusted_held_1000hrs", "severity_score": 4},

    {"id": 13, "text_for_embedding": "track roller bottom roller leaking oil seal failure undercarriage",
     "component": "track_roller", "equipment_model": "CAT 320", "hours": 8100, "rating": "YELLOW",
     "finding": "Bottom roller #3 left side leaking from face seal. Oil film on housing and frame. Roller turns freely — bearing not yet damaged.",
     "action": "Schedule roller replacement at next PM (verify bottom roller P/N via Cat SIS against machine serial — varies by production date). Do not wait for seizure — seized roller damages track chain and carrier frame.",
     "outcome": "roller_replaced_8500hrs_no_secondary_damage", "severity_score": 5},

    {"id": 14, "text_for_embedding": "sprocket segment worn teeth thin hooked undercarriage drive",
     "component": "sprocket", "equipment_model": "CAT 320", "hours": 9000, "rating": "YELLOW",
     "finding": "Sprocket teeth showing hooked wear. Width 15mm vs 22mm new. Mismatched with newer chain — accelerating both component wear.",
     "action": "Replace sprocket segments P/N 6Y-4898 at next chain change. Worn sprockets with new chain causes rapid chain wear.",
     "outcome": "sprocket_replaced_with_new_chain_9500hrs", "severity_score": 5},

    # === ENGINE COMPARTMENT ===
    {"id": 20, "text_for_embedding": "engine coolant leak radiator hose clamp loose dripping antifreeze green",
     "component": "coolant_system", "equipment_model": "CAT 320", "hours": 4500, "rating": "RED",
     "finding": "Active coolant leak from upper radiator hose — clamp backed off from vibration. Surge tank below MIN. Green staining on engine block indicates ongoing loss.",
     "action": "Do not start. Tighten clamp. Top off with Cat ELC only — do not mix types. Pressure test at 15 PSI after repair.",
     "outcome": "clamp_tightened_no_recurrence", "severity_score": 7},

    {"id": 21, "text_for_embedding": "engine oil level low dark black dirty dipstick overdue oil change",
     "component": "engine_oil", "equipment_model": "CAT 336", "hours": 3200, "rating": "YELLOW",
     "finding": "Oil 1.5 quarts low. Very dark — overdue change (due at 3000 hrs, 500-hr interval). Gritty between fingers — soot loading.",
     "action": "Change oil and filter P/N 1R-0716 immediately. Submit S.O.S sample for fuel dilution, coolant contamination, wear metals. Reset service indicator.",
     "outcome": "oil_changed_SOS_showed_high_soot_clogged_filter", "severity_score": 5},

    {"id": 22, "text_for_embedding": "fan belt serpentine belt cracked worn glazed fraying engine",
     "component": "belt", "equipment_model": "CAT 320", "hours": 6000, "rating": "YELLOW",
     "finding": "Fan belt with hairline transverse cracks and glazing. 3 of 6 ribs cracked >1mm deep. Tension low — deflection exceeds 15mm spec.",
     "action": "Replace P/N 3N-8049 within 100 hours. Snapped belt = immediate overheating. Inspect tensioner pulley bearing.",
     "outcome": "belt_replaced_tensioner_also_worn", "severity_score": 6},

    {"id": 23, "text_for_embedding": "air filter dirty clogged debris restriction service indicator red engine",
     "component": "air_filter", "equipment_model": "CAT 320", "hours": 2800, "rating": "YELLOW",
     "finding": "Service indicator in RED. Heavy dust loading visible. Quarry conditions require 250-hr changes vs standard 500-hr.",
     "action": "Replace primary element P/N 131-8822. Inspect secondary element. Clean pre-cleaner bowl. Adjust PM to 250-hr intervals for this site.",
     "outcome": "filter_replaced_pm_interval_shortened", "severity_score": 4},

    {"id": 24, "text_for_embedding": "turbocharger black smoke power loss exhaust boost leak shaft play",
     "component": "turbocharger", "equipment_model": "CAT 330", "hours": 8500, "rating": "RED",
     "finding": "Excessive black smoke under load, noticeable power loss. Boost 18 PSI vs 28 PSI spec. Shaft radial play exceeds 0.003 inches. Oil at compressor outlet.",
     "action": "Remove from heavy production. Turbo rebuild/replacement. Check intake/exhaust piping for leaks. Inspect intercooler for oil. S.O.S for bearing material.",
     "outcome": "turbo_replaced_$4200_intercooler_cleaned", "severity_score": 9},

    {"id": 25, "text_for_embedding": "radiator fins bent blocked debris restricted airflow cooling overheating",
     "component": "radiator", "equipment_model": "CAT 320", "hours": 4000, "rating": "YELLOW",
     "finding": "Radiator ~40% blocked with debris and bent fins. Coolant running 5C above normal at 97C. AC condenser also partially blocked.",
     "action": "Clean core with low-pressure water (back to front). Straighten fins. Clean AC condenser. Install debris screen. Add to daily walkaround.",
     "outcome": "cleaned_temp_returned_to_normal_92C", "severity_score": 4},

    # === STRUCTURAL / BOOM ===
    {"id": 30, "text_for_embedding": "boom pin seized frozen stuck indicator flag broken loader pivot",
     "component": "boom_pin", "equipment_model": "CAT 950", "hours": 7500, "rating": "RED",
     "finding": "Boom-to-frame pin seized. Indicator flag sheared confirming no rotation during greasing. Bore galling visible. Insufficient daily greasing.",
     "action": "Do not operate — sudden failure causes boom drop. Press pin out. Replace pin P/N 8T-4778 and bushings. Implement daily grease log.",
     "outcome": "machine_down_2_days_bore_sleeved_$5200", "severity_score": 10},

    {"id": 31, "text_for_embedding": "weld crack stress fracture boom structural damage crack propagation",
     "component": "boom_structure", "equipment_model": "CAT 320", "hours": 9200, "rating": "RED",
     "finding": "Stress crack at boom weld joint cylinder mount, 2.5 inches. Propagating from weld toe into base metal. Paint cracking confirms growth over time.",
     "action": "Remove from service. Certified structural welding required — not field repair. NDT (dye penetrant) to determine full extent. Ship to Cat rebuild.",
     "outcome": "boom_repaired_NDT_tested_2_week_$7500", "severity_score": 10},

    {"id": 32, "text_for_embedding": "bucket teeth worn rounded dull cutting edge ground engaging tools",
     "component": "bucket_teeth", "equipment_model": "CAT 320", "hours": 4800, "rating": "YELLOW",
     "finding": "J-series teeth 60% worn. Tips rounded reducing efficiency 15-20%. Center tooth more worn than corners — operator technique issue. Adapters OK.",
     "action": "Replace teeth P/N 1U-3352 before next heavy digging. Worn teeth increase fuel 10-20%. Consider K-series for rock.",
     "outcome": "teeth_replaced_fuel_improved_12pct", "severity_score": 5},

    {"id": 33, "text_for_embedding": "bucket tooth missing broken off adapter exposed damaged retainer pin failure",
     "component": "bucket_teeth", "equipment_model": "CAT 336", "hours": 5100, "rating": "RED",
     "finding": "Corner tooth missing — retainer failure. Adapter impact-damaged. Lost tooth is foreign object hazard in material. Adjacent tooth loose.",
     "action": "Replace tooth and adapter. Check all retainers. Search material pile for lost tooth to prevent downstream damage.",
     "outcome": "tooth_adapter_replaced_retainers_checked_$380", "severity_score": 8},

    {"id": 34, "text_for_embedding": "stick cylinder pin bushing worn loose excessive play clearance",
     "component": "stick_pin", "equipment_model": "CAT 320", "hours": 6500, "rating": "YELLOW",
     "finding": "Stick-to-bucket pin 3mm radial play. Bushing wear visible. Pin shows linear marks. Auto-lube was disconnected.",
     "action": "Schedule pin/bushing replacement. Reconnect auto-lube. Current play operable but accelerating. Monitor weekly.",
     "outcome": "pin_bushings_replaced_7000hr_autolube_repaired", "severity_score": 5},

    # === WHEEL LOADER ===
    {"id": 40, "text_for_embedding": "transmission fluid leak seal weeping drivetrain underside loader",
     "component": "transmission", "equipment_model": "CAT 950", "hours": 5800, "rating": "YELLOW",
     "finding": "Minor transmission fluid weep at rear output seal. Not actively dripping. TO-4 fluid on belly pan. Present ~200 hours based on staining.",
     "action": "Monitor level every shift. Schedule seal replacement at next downtime — requires driveshaft disconnect. Do not delay past 500 hours.",
     "outcome": "seal_replaced_6200hrs_planned_PM_$1800", "severity_score": 5},

    {"id": 41, "text_for_embedding": "loader tire sidewall cut damage puncture wheel front rubber",
     "component": "tire", "equipment_model": "CAT 950", "hours": 3900, "rating": "YELLOW",
     "finding": "Front left tire (20.5R25) 2-inch sidewall cut from rebar. No cord exposed. Pressure holding at 42 PSI (spec 45). Tread 55% remaining.",
     "action": "Monitor daily. If cord visible, remove immediately — sidewall blowouts are dangerous. Maintain pressure. Log in tire management.",
     "outcome": "tire_replaced_at_next_rotation_800hrs_later", "severity_score": 4},

    {"id": 42, "text_for_embedding": "cab access step ladder rubber pad cracked peeling slip hazard safety",
     "component": "access_steps", "equipment_model": "CAT 950", "hours": 4100, "rating": "YELLOW",
     "finding": "Anti-slip pad on second step cracked and peeling. Steel surface exposed — slippery when wet. OSHA 3-point contact compromised.",
     "action": "Replace pad. Apply anti-slip tape as interim. Document in OSHA inspection log. Safety write-up required.",
     "outcome": "pad_replaced_tape_applied_same_day", "severity_score": 4},

    {"id": 43, "text_for_embedding": "loader bucket cutting edge worn thin bolt-on replacement needed",
     "component": "cutting_edge", "equipment_model": "CAT 966", "hours": 3500, "rating": "YELLOW",
     "finding": "Cutting edge 40% remaining. Scalloped wear between bolts. One bolt missing — material packing behind edge.",
     "action": "Replace edge P/N 4T-6381 within 200 hours. Replace missing bolt now. Consider heavy-duty edge for aggregate.",
     "outcome": "edge_replaced_upgraded_to_heavy_duty", "severity_score": 5},

    # === ELECTRICAL / SAFETY ===
    {"id": 50, "text_for_embedding": "backup alarm reverse warning not working silent broken safety critical MSHA",
     "component": "backup_alarm", "equipment_model": "CAT 320", "hours": 5500, "rating": "RED",
     "finding": "Backup alarm non-functional in reverse. Corroded wiring connector from water intrusion. MSHA/OSHA critical — requires 85dB at 15 feet.",
     "action": "Do not operate. Clean/replace connector. Test in all positions. Document in safety log. MSHA citable violation.",
     "outcome": "connector_replaced_alarm_functional_1hr", "severity_score": 9},

    {"id": 51, "text_for_embedding": "headlight work light broken cracked lens dim moisture night visibility",
     "component": "lights", "equipment_model": "CAT 336", "hours": 4700, "rating": "YELLOW",
     "finding": "Left front work light cracked with moisture intrusion. Output ~50% reduced. Day shift only until repaired.",
     "action": "Replace work light assembly. Check all other lenses — vibration failures occur in groups. Restrict to day operations.",
     "outcome": "light_replaced_all_inspected", "severity_score": 3},

    {"id": 52, "text_for_embedding": "fire extinguisher expired missing cab safety equipment MSHA inspection",
     "component": "fire_extinguisher", "equipment_model": "CAT 320", "hours": 5000, "rating": "RED",
     "finding": "Fire extinguisher inspection expired 8 months. Pressure gauge in yellow. MSHA requires annual inspection. Bracket loose.",
     "action": "Replace immediately with ABC dry chemical 5 lb minimum. Tighten bracket. Log in safety record. MSHA citable violation — 30 CFR 56.4230 requires self-propelled equipment to carry an accessible, serviceable fire extinguisher; 30 CFR 56.4200–56.4203 mandate type, capacity, and annual inspection. Remove from service under 30 CFR 56.14100(c) until compliant.",
     "outcome": "extinguisher_replaced_logged", "severity_score": 8},

    # === FLUIDS / FILTERS ===
    {"id": 60, "text_for_embedding": "fuel filter water separator bowl contamination diesel water accumulation",
     "component": "fuel_system", "equipment_model": "CAT 320", "hours": 3000, "rating": "YELLOW",
     "finding": "Water separator showing 1 inch of water. Indicator light ignored ~50 hours. Risk of injector damage ($800+ each). Likely from tanker delivery.",
     "action": "Drain separator. Check fuel tank bottom sample. Replace filter P/N 1R-0762. Brief operator on daily check. Investigate fuel source.",
     "outcome": "water_drained_fuel_source_changed_no_damage", "severity_score": 5},

    {"id": 61, "text_for_embedding": "engine oil analysis high silicon dirt ingestion contamination SOS lab results",
     "component": "engine_oil", "equipment_model": "CAT 330", "hours": 4800, "rating": "YELLOW",
     "finding": "S.O.S analysis trend: Silicon has risen from 18 → 27 → 45 ppm across three consecutive samples — a rising trend indicating dirt ingestion, not a single-point exceedance. Iron trending upward at 35 ppm. Cat S.O.S is trend-based; the pattern here warrants investigation of the air intake path.",
     "action": "Inspect complete intake: filter housing seal, hose clamps, turbo inlet, crankcase breather. Find dust entry before oil change. Replace filter.",
     "outcome": "loose_intake_clamp_found_silicon_normalized", "severity_score": 6},

    # === LARGE EQUIPMENT ===
    {"id": 70, "text_for_embedding": "articulated dump truck body pivot hinge pin wear bushing hauler",
     "component": "body_pivot", "equipment_model": "CAT 745", "hours": 8000, "rating": "YELLOW",
     "finding": "Body pivot pins 2mm radial play. Bushings 65% worn. Some grease starvation on lower bushing. Normal for 8000 hours in haul application.",
     "action": "Increase greasing to every 4 hours. Monitor monthly. Schedule replacement at 10,000 hr PM. Order now — 6 week lead time.",
     "outcome": "greasing_increased_replaced_at_10000hrs", "severity_score": 4},

    {"id": 80, "text_for_embedding": "dozer blade cutting edge worn thin end bits push arm",
     "component": "blade_edge", "equipment_model": "CAT D6", "hours": 4200, "rating": "YELLOW",
     "finding": "Blade edge 30% remaining. Center more worn — typical push pattern. One end bit worn through. Digging below grade wasting fuel.",
     "action": "Replace edge and end bits. Consider carbide-insert for abrasive material. Check tilt cylinder for bias.",
     "outcome": "edge_replaced_carbide_ordered_for_next", "severity_score": 5},

    {"id": 90, "text_for_embedding": "frozen hydraulic lines cold weather winter not preheated sluggish arctic operation",
     "component": "hydraulic_hose", "equipment_model": "CAT 320", "hours": 3200, "rating": "YELLOW",
     "finding": "Hydraulic oil extremely viscous — not pre-warmed in -15C. Functions sluggish. Cavitation risk to pump. Cat requires 15-min warm-up with HYDO 10W.",
     "action": "Warm at low idle 15 minutes before operating hydraulics. Install cold weather package (block heater + tank heater). Verify oil viscosity grade.",
     "outcome": "operator_briefed_block_heater_installed", "severity_score": 4},

    # === EXPANDED FLEET DATA — Fire Extinguisher (demo-critical) ===
    {"id": 100, "text_for_embedding": "fire extinguisher expired inspection tag date overdue gauge low pressure red zone safety cab",
     "component": "fire_extinguisher", "equipment_model": "CAT 336", "hours": 3200, "rating": "RED",
     "finding": "Fire extinguisher annual inspection tag expired 14 months. Gauge needle at bottom of yellow zone trending toward red. Handle corrosion visible. MSHA violation.",
     "action": "Replace immediately. Procure 10lb ABC dry chemical unit. Document in MSHA safety log. Machine must not operate until compliant per 30 CFR 56.4230.",
     "outcome": "extinguisher_replaced_MSHA_log_updated", "severity_score": 9},

    {"id": 101, "text_for_embedding": "fire extinguisher gauge borderline yellow zone bracket loose vibration safety cab mount",
     "component": "fire_extinguisher", "equipment_model": "CAT 349", "hours": 4500, "rating": "YELLOW",
     "finding": "Fire extinguisher gauge in low-green/borderline zone. Mounting bracket fastener loose from vibration — unit shifting in holder. Tag current but annual recharge due in 30 days.",
     "action": "Tighten bracket. Schedule professional recharge within 30 days. Monitor gauge weekly. Order replacement if gauge drops further.",
     "outcome": "bracket_tightened_recharge_scheduled", "severity_score": 5},

    {"id": 102, "text_for_embedding": "fire extinguisher good condition fully charged green gauge clean cab safety equipment",
     "component": "fire_extinguisher", "equipment_model": "CAT 320", "hours": 2000, "rating": "GREEN",
     "finding": "Fire extinguisher in excellent condition. Gauge solidly in green. Current annual inspection tag dated within 6 months. Pin and tamper seal intact. Clean, no corrosion.",
     "action": "No action required. Next annual inspection due in 6 months. Verify operator knows location and operation.",
     "outcome": "no_issues_passed_inspection", "severity_score": 1},

    {"id": 103, "text_for_embedding": "fire extinguisher missing absent not present cab safety empty bracket MSHA critical",
     "component": "fire_extinguisher", "equipment_model": "CAT 950", "hours": 6200, "rating": "RED",
     "finding": "Fire extinguisher completely missing from cab bracket. Empty mounting bracket only. Immediate MSHA/OSHA violation. Total loss of fire suppression capability.",
     "action": "Do NOT operate. Install certified extinguisher immediately. Issue MSHA defect tag. Supervisor notification required. Investigate how/when removed.",
     "outcome": "replacement_installed_same_shift_incident_report_filed", "severity_score": 10},

    # === EXPANDED — Hydraulic Hose States ===
    {"id": 104, "text_for_embedding": "hydraulic hose dirty dusty dry no leak oil film accumulation control valve area",
     "component": "hydraulic_hose", "equipment_model": "CAT 336", "hours": 4800, "rating": "GREEN",
     "finding": "Hydraulic hoses and fittings at control valve area show heavy accumulation of dry dust and debris. No active oil film, no weeping, no structural damage. Connections tight.",
     "action": "Clean the control valve and hose bundle area during the scheduled PM2 service (due in 200 hours) to allow proper visual inspection. No repair needed.",
     "outcome": "cleaned_at_PM_no_issues_found", "severity_score": 2},

    {"id": 105, "text_for_embedding": "hydraulic hose burst catastrophic failure oil spray high pressure line emergency boom cylinder",
     "component": "hydraulic_hose", "equipment_model": "CAT 320", "hours": 6200, "rating": "RED",
     "finding": "Boom cylinder high-pressure hose catastrophic burst. Complete loss of boom hydraulic function. Oil evacuated from system — estimated 15 gallons on ground. Hose past 6-year age limit.",
     "action": "Machine down. Environmental containment required. Replace hose assembly + flush system + new filters. S.O.S sample after repair. File environmental incident report.",
     "outcome": "hose_replaced_system_flushed_48hrs_$2800_environmental_cleanup_$1200", "severity_score": 10},

    {"id": 106, "text_for_embedding": "hydraulic hose UV cracking surface degradation aging sun damage dry rot approaching 6 year limit",
     "component": "hydraulic_hose", "equipment_model": "CAT 330", "hours": 3500, "rating": "YELLOW",
     "finding": "Hose outer cover shows UV degradation — fine surface cracking and discoloration. Date code indicates 5.2 years age. No reinforcement exposure yet. Within 6-year mandatory replacement window.",
     "action": "Schedule replacement within next 500 hours or before 6-year date code limit, whichever comes first. Mark hose with yellow tag. Order replacement.",
     "outcome": "replaced_at_5500hrs_before_age_limit", "severity_score": 5},

    # === EXPANDED — Final Drive / Undercarriage ===
    {"id": 107, "text_for_embedding": "track roller seized frozen locked bearing failure chain damage undercarriage catastrophic",
     "component": "track_roller", "equipment_model": "CAT 320", "hours": 6500, "rating": "RED",
     "finding": "Bottom roller #5 right side completely seized. Flat spot worn into roller surface — 8mm deep. Chain links wearing against housing. Secondary damage to carrier frame mounting.",
     "action": "Remove from service immediately. Replace roller + inspect chain links 22-28 for accelerated wear. Check carrier frame for cracks. Replace track if chain pitch elongation exceeds 3%.",
     "outcome": "roller_replaced_chain_within_spec_but_4_links_replaced_$3200", "severity_score": 8},

    {"id": 108, "text_for_embedding": "final drive seal failure duo-cone leak oil gear oil sprocket housing excavator",
     "component": "final_drive", "equipment_model": "CAT 336", "hours": 7000, "rating": "RED",
     "finding": "Final drive duo-cone seal failure. Active gear oil leak from sprocket hub. Approximately 1 quart loss per shift. Debris packing evident around seal face — likely caused seal displacement.",
     "action": "Remove from heavy production. Schedule final drive seal replacement ($1500-3000, 1-2 days). Top off gear oil each shift until repair. Clean debris from seal area.",
     "outcome": "seal_replaced_7500hrs_$2200_1.5_days_downtime", "severity_score": 8},

    {"id": 109, "text_for_embedding": "final drive debris packing dirt accumulation around seal sprocket hub early warning",
     "component": "final_drive", "equipment_model": "CAT 320", "hours": 4000, "rating": "YELLOW",
     "finding": "Debris accumulation packing tightly around final drive duo-cone seal area. No oil leak yet, but material is pressing against seal face. Left side worse than right.",
     "action": "Clean both final drive seal areas thoroughly. Add to daily walkaround — debris packing is the #1 cause of duo-cone seal displacement. Install rubber scraper guards if available.",
     "outcome": "cleaned_scraper_installed_seal_held_2000hrs_more", "severity_score": 4},

    # === EXPANDED — Safety Items ===
    {"id": 110, "text_for_embedding": "backup alarm weak low volume below threshold decibel MSHA safety reverse warning",
     "component": "backup_alarm", "equipment_model": "CAT 336", "hours": 5200, "rating": "RED",
     "finding": "Backup alarm sounding but significantly below required 85dB at 15 feet. Measured approximately 60dB. Speaker membrane deteriorated from water exposure. Partially functional but non-compliant.",
     "action": "Replace alarm assembly. Test at 15 feet with dB meter. MSHA requires 85dB or audible above ambient. Document test results in safety log.",
     "outcome": "alarm_replaced_tested_92dB_compliant", "severity_score": 8},

    {"id": 111, "text_for_embedding": "cab glass windshield star crack impact damage debris strike visibility safety",
     "component": "cab_glass", "equipment_model": "CAT 320", "hours": 3800, "rating": "YELLOW",
     "finding": "Windshield has 3-inch star crack from rock debris impact, lower right quadrant. Not in primary operator sight line. No water intrusion yet. Laminated glass holding.",
     "action": "Schedule windshield replacement. Apply clear repair resin to prevent propagation. Acceptable for continued operation if crack does not grow. Monitor daily.",
     "outcome": "windshield_replaced_at_next_PM_$1800", "severity_score": 4},

    # === EXPANDED — Engine ===
    {"id": 112, "text_for_embedding": "coolant system radiator leak active drip overheating high temperature warning engine",
     "component": "coolant_system", "equipment_model": "CAT 349", "hours": 5500, "rating": "RED",
     "finding": "Active coolant leak from radiator core — pinhole in tube. Temperature hitting 105C under load (max 100C). Coolant level dropping 1L per hour. Previous repair attempt with stop-leak visible.",
     "action": "Do not operate under load. Radiator core requires replacement, not repair. Flush entire cooling system — stop-leak product may have clogged heater core. Replace thermostat.",
     "outcome": "radiator_replaced_system_flushed_3_days_$4500", "severity_score": 9},

    {"id": 113, "text_for_embedding": "fan belt serpentine broken snapped engine overheat alternator no charge catastrophic",
     "component": "belt", "equipment_model": "CAT 336", "hours": 7500, "rating": "RED",
     "finding": "Serpentine belt snapped. Engine immediately overheated — water pump and fan not driven. Alternator not charging. Operator shut down within 2 minutes. Belt remnants tangled in fan shroud.",
     "action": "Install new belt P/N 3N-8049. Inspect fan shroud for damage. Check tensioner bearing and idler pulley. Monitor coolant temp for 4 hours after restart. S.O.S sample to check for overheat damage.",
     "outcome": "belt_replaced_tensioner_pulley_also_replaced_$280_4hrs", "severity_score": 9},

    {"id": 114, "text_for_embedding": "engine oil analysis SOS trending high iron copper wear metals contamination sample laboratory",
     "component": "engine_oil", "equipment_model": "CAT 320", "hours": 4500, "rating": "YELLOW",
     "finding": "S.O.S trend: Iron 55→72→98 ppm over last 3 samples (1500hr span). Copper rising 12→18→28 ppm — bearing material. Silicon stable at 9 ppm. Cat S.O.S lab flags this trend as 'monitor closely.' No abnormal noise or performance change yet.",
     "action": "Shorten sample interval to every 125 hours. Inspect valve train and main bearings at next PM. If iron exceeds 120 ppm or copper exceeds 40 ppm, plan for engine top-end inspection.",
     "outcome": "monitoring_continued_trend_stabilized_after_valve_adjustment", "severity_score": 6},

    # === EXPANDED — Structural ===
    {"id": 115, "text_for_embedding": "boom weld crack stress fracture lift cylinder mounting bracket structural failure",
     "component": "boom_structure", "equipment_model": "CAT 336", "hours": 10000, "rating": "RED",
     "finding": "3-inch crack at boom lift cylinder mounting bracket weld. Crack propagating from weld toe into base metal. Visible through paint. High-stress area — catastrophic boom failure risk under load.",
     "action": "Remove from service immediately. NDT dye penetrant inspection to map crack extent. Certified structural weld repair required — not field repair. Cat rebuild center recommended.",
     "outcome": "boom_shipped_to_rebuild_center_3_weeks_$12000", "severity_score": 10},

    {"id": 116, "text_for_embedding": "hydraulic hose clean good condition dry dust only no leak no damage control valve area excavator",
     "component": "hydraulic_hose", "equipment_model": "CAT 320", "hours": 2450, "rating": "GREEN",
     "finding": "The hydraulic hoses and fittings at the control valve show heavy accumulation of dry dust and debris. No active oil film, weeping, or structural damage. All connections tight. Recently serviced at PM1.",
     "action": "Clean the control valve and hose bundle area during the scheduled PM2 service (due in 50 hours) to allow proper visual inspection going forward.",
     "outcome": "pending", "severity_score": 2},

    # === EXPANDED — Miscellaneous ===
    {"id": 117, "text_for_embedding": "swing bearing play movement excessive clearance grease starvation upper structure rotation",
     "component": "swing_bearing", "equipment_model": "CAT 336", "hours": 8500, "rating": "YELLOW",
     "finding": "Swing bearing has ~2mm vertical play when boom fully extended. Grease purge showing dark contaminated grease — possible water intrusion. Swing gear gear tooth wear visible at access port.",
     "action": "Increase greasing frequency to every 50 hours. Flush old grease with Cat NLGI 2EP. Submit grease sample for water content. Plan for bearing replacement at 10,000 hrs ($15,000-25,000).",
     "outcome": "greasing_protocol_updated_bearing_replaced_at_10200hrs", "severity_score": 6},

    {"id": 118, "text_for_embedding": "seatbelt frayed webbing worn retractor slow sluggish safety restraint ROPS cab",
     "component": "seatbelt", "equipment_model": "CAT 320", "hours": 6000, "rating": "YELLOW",
     "finding": "Seatbelt webbing showing fraying at anchor point — 15% width reduction. Retractor mechanism sluggish. Belt locks properly but slow to retract. ROPS structure intact.",
     "action": "Replace seatbelt assembly. Do not splice or repair webbing. Verify retractor function after replacement. MSHA/OSHA requires functional restraint when ROPS equipped.",
     "outcome": "seatbelt_replaced_retractor_functional_$180", "severity_score": 6},

    {"id": 119, "text_for_embedding": "bucket teeth multiple missing broken retainer failure adapters damaged exposed ground engaging",
     "component": "bucket_teeth", "equipment_model": "CAT 320", "hours": 5500, "rating": "RED",
     "finding": "Two bucket teeth missing — both corner positions. Retainer pins failed. Adapters impact-damaged and mushroomed. Remaining teeth 70% worn. Lost teeth are FOD hazard in material.",
     "action": "Replace all teeth and both damaged adapters. Inspect all retainer pins. Search stockpile for lost teeth. Consider upgrading to K-series HD for rock application.",
     "outcome": "all_teeth_and_adapters_replaced_K_series_installed_$650", "severity_score": 8},
]


PARTS_CATALOG = [
    {"id": 2001, "text_for_embedding": "caterpillar J-series bucket tooth point adapter excavator digging standard",
     "part_name": "J-Series Bucket Tooth", "part_number": "1U3352",
     "compatible_models": ["CAT 320", "CAT 325", "CAT 330"], "category": "ground_engaging",
     "service_info": "Replace at 50% wear or rounded tip. Life: 200-400 hrs."},

    {"id": 2002, "text_for_embedding": "caterpillar K-series heavy duty rock bucket tooth tip large excavator",
     "part_name": "K-Series HD Tooth", "part_number": "1U3452RC",
     "compatible_models": ["CAT 336", "CAT 349", "CAT 352"], "category": "ground_engaging",
     "service_info": "Heavy rock chisel. 30% longer life in abrasive conditions."},

    {"id": 2003, "text_for_embedding": "caterpillar bolt-on cutting edge bucket loader excavator flat steel",
     "part_name": "Bolt-On Cutting Edge", "part_number": "4T6381",
     "compatible_models": ["CAT 320", "CAT 325", "CAT 950", "CAT 966"], "category": "ground_engaging",
     "service_info": "Replace below 25% thickness. Check bolt torque every 50 hrs."},

    {"id": 2004, "text_for_embedding": "caterpillar high pressure hydraulic hose assembly black rubber braided steel fittings boom cylinder",
     "part_name": "High Pressure Hydraulic Hose", "part_number": "Verify via Cat SIS (serial-specific)",
     "compatible_models": ["CAT 320", "CAT 325", "CAT 330"], "category": "hydraulic",
     "service_info": "Replace every 6 years max regardless of appearance. Check for abrasion, bulging, weeping. P/N varies by serial range — always verify via Cat SIS or dealer."},

    {"id": 2005, "text_for_embedding": "caterpillar track shoe grouser pad 600mm steel plate undercarriage excavator",
     "part_name": "Track Shoe (600mm)", "part_number": "6Y6340",
     "compatible_models": ["CAT 320", "CAT 325"], "category": "undercarriage",
     "service_info": "Replace below 20mm height (38mm new). Life: 3000-6000 hrs."},

    {"id": 2006, "text_for_embedding": "caterpillar bottom track roller sealed round metal undercarriage excavator",
     "part_name": "Track Roller (Bottom)", "part_number": "Verify via Cat SIS (serial-specific)",
     "compatible_models": ["CAT 320", "CAT 325", "CAT 330"], "category": "undercarriage",
     "service_info": "Replace at first sign of seal leak. Seized roller damages chain and carrier frame. Life: 4000-8000 hrs. P/N varies by frame width and production date — verify via Cat SIS."},

    {"id": 2007, "text_for_embedding": "caterpillar primary air filter element cylindrical paper engine intake",
     "part_name": "Primary Air Filter", "part_number": "131-8822",
     "compatible_models": ["CAT 320", "CAT 325"], "category": "filtration",
     "service_info": "Replace at indicator RED or 500 hrs (250 in dust). Never reuse."},

    {"id": 2008, "text_for_embedding": "caterpillar hydraulic oil return filter element canister spin-on",
     "part_name": "Hydraulic Oil Filter", "part_number": "1R-0741",
     "compatible_models": ["CAT 320", "CAT 325", "CAT 330", "CAT 336"], "category": "filtration",
     "service_info": "Every 500 hrs. Pre-fill with clean oil. Protects pump and valves."},

    {"id": 2009, "text_for_embedding": "caterpillar serpentine fan belt ribbed rubber engine accessory drive",
     "part_name": "Serpentine Belt", "part_number": "3N8049",
     "compatible_models": ["CAT 320", "CAT 325"], "category": "engine",
     "service_info": "Inspect every 500 hrs. Replace at cracking/glazing. Snapped = overheat."},

    {"id": 2010, "text_for_embedding": "caterpillar front idler assembly undercarriage track excavator large wheel",
     "part_name": "Front Idler Assembly", "part_number": "9W8690",
     "compatible_models": ["CAT 320", "CAT 325"], "category": "undercarriage",
     "service_info": "Check for flat spots, leaks, flange wear. Lasts 2 chain lives."},

    {"id": 2011, "text_for_embedding": "caterpillar sprocket drive segment bolt-on teeth undercarriage final drive",
     "part_name": "Sprocket Segment", "part_number": "6Y4898",
     "compatible_models": ["CAT 320", "CAT 325", "CAT 330"], "category": "undercarriage",
     "service_info": "Replace with new chain. Worn sprockets destroy new chain."},

    {"id": 2012, "text_for_embedding": "caterpillar wheel loader tire 20.5R25 L3 rubber large round",
     "part_name": "Loader Tire 20.5R25", "part_number": "Bridgestone VJT",
     "compatible_models": ["CAT 950", "CAT 962", "CAT 966"], "category": "tires",
     "service_info": "Monitor pressure daily (45 PSI). Rotate at 50% tread. Remove at cord exposure."},

    {"id": 2013, "text_for_embedding": "caterpillar engine oil filter spin-on canister yellow cap",
     "part_name": "Engine Oil Filter", "part_number": "1R-0716",
     "compatible_models": ["CAT 320", "CAT 325", "CAT 330"], "category": "filtration",
     "service_info": "Every 500 hrs with oil change. Pre-fill. Hand-tighten."},

    {"id": 2014, "text_for_embedding": "caterpillar fuel filter water separator element diesel primary",
     "part_name": "Fuel Filter / Water Sep", "part_number": "1R-0762",
     "compatible_models": ["CAT 320", "CAT 325", "CAT 330", "CAT 336"], "category": "filtration",
     "service_info": "Every 500 hrs. Drain water daily. Water kills injectors ($800+ each)."},

    {"id": 2015, "text_for_embedding": "caterpillar hydraulic cylinder seal kit o-ring wiper rod piston rebuild",
     "part_name": "Hydraulic Cylinder Seal Kit", "part_number": "215-9985",
     "compatible_models": ["CAT 320", "CAT 325", "CAT 330"], "category": "hydraulic",
     "service_info": "Install whenever cylinder opened. Never reuse seals. Keep in stock."},

    {"id": 2016, "text_for_embedding": "caterpillar excavator bucket pin hardened steel retainer connection",
     "part_name": "Bucket Pin (Hardened)", "part_number": "8T-4778",
     "compatible_models": ["CAT 320", "CAT 325", "CAT 330"], "category": "structural",
     "service_info": "Grease daily. Replace at 3mm play. Always replace bushings with pin."},
]


def seed():
    store = get_store()
    print(f"\n{'='*60}")
    print(f"SEEDING INSPECTION HISTORY ({len(INSPECTION_RECORDS)} records)")
    print(f"{'='*60}")
    existing = store.count(INSPECTION_COLLECTION)
    if existing > 0:
        print(f"  Already has {existing} records. Skipping. (Delete to re-seed)")
    else:
        for i, rec in enumerate(INSPECTION_RECORDS):
            print(f"  [{i+1}/{len(INSPECTION_RECORDS)}] {rec['text_for_embedding'][:60]}...")
            vec = embed_text(rec["text_for_embedding"])
            payload = {k: v for k, v in rec.items() if k not in ("id", "text_for_embedding")}
            store.upsert(INSPECTION_COLLECTION, id=rec["id"], vector=vec, payload=payload)
        print(f"  ✅ Seeded {len(INSPECTION_RECORDS)} inspection records")

    print(f"\n{'='*60}")
    print(f"SEEDING PARTS CATALOG ({len(PARTS_CATALOG)} parts)")
    print(f"{'='*60}")
    existing_parts = store.count(PARTS_COLLECTION)
    if existing_parts > 0:
        print(f"  Already has {existing_parts} parts. Skipping.")
    else:
        for i, part in enumerate(PARTS_CATALOG):
            print(f"  [{i+1}/{len(PARTS_CATALOG)}] {part['part_name']}...")
            vec = embed_text(part["text_for_embedding"])
            payload = {k: v for k, v in part.items() if k not in ("id", "text_for_embedding")}
            store.upsert(PARTS_COLLECTION, id=part["id"], vector=vec, payload=payload)
        print(f"  ✅ Seeded {len(PARTS_CATALOG)} parts")

    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    for q in ["hydraulic hose leaking oil boom", "track shoe bent undercarriage", "engine oil dirty low"]:
        vec = embed_text(q)
        results = store.search(INSPECTION_COLLECTION, vec, top_k=3)
        print(f"\n  '{q}'")
        for r in results:
            print(f"    {r['score']:.3f} | {r['payload'].get('rating')} | {r['payload'].get('finding','')[:70]}")
    print(f"\n✅ Done! Inspections: {store.count(INSPECTION_COLLECTION)} | Parts: {store.count(PARTS_COLLECTION)}")

if __name__ == "__main__":
    seed()