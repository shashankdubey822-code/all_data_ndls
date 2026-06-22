"""
openrouter_analyzer.py - OpenRouter AI integration for RAIL-SENSE
Component-aware LLM prompts for all 6 LHB Fiat Bogie components.
"""

# ─────────────────────────────────────────────────────────────
# Import required standard libraries and third-party modules
# ─────────────────────────────────────────────────────────────
import os         # For environment variable access
import json       # For parsing and generating JSON payloads
import base64     # For encoding image data to transmit to the API
import requests   # For making HTTP requests to the OpenRouter API
from io import BytesIO # For in-memory binary streams (e.g. image buffers)
from PIL import Image  # Pillow library for image processing and resizing

# ─────────────────────────────────────────────────────────────
# Environment and Configuration Variables
# ─────────────────────────────────────────────────────────────
# Fetch the OpenRouter API base URL, defaulting to the standard v1 endpoint
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Fetch the API key needed to authenticate requests to OpenRouter
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
# Fetch the site URL used in headers to identify the app to OpenRouter
APP_SITE_URL        = os.getenv("APP_SITE_URL", "http://localhost:8000")
# Fetch the application title used in headers to identify the app to OpenRouter
APP_TITLE           = os.getenv("APP_TITLE", "RAIL-SENSE")

# ─────────────────────────────────────────────────────────────
# LLM Model Configuration
# ─────────────────────────────────────────────────────────────
# Dictionary mapping logical model performance tiers to actual OpenRouter free model identifiers
FREE_MODELS = {
    "fastest":  "liquid/lfm-2.5-1.2b-instruct:free",    # Optimized for fastest response time
    "balanced": "openai/gpt-oss-20b:free",              # Balanced performance and reasoning
    "standard": "nvidia/nemotron-nano-9b-v2:free",      # Standard reliable model tier
    "vision":   "nvidia/nemotron-nano-12b-v2-vl:free",  # Multimodal model capable of image analysis
}
# Define the fallback model to use if a tier is missing or unrequested
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b:free")


# ─────────────────────────────────────────────────────────────
# Per-component checklist definitions
# ─────────────────────────────────────────────────────────────
COMPONENT_CHECKLISTS = {
    "coupler": {
        "fields": ["knuckle_lock", "coupler_alignment", "structural_integrity", "pin_assembly", "corrosion_wear"],
        "labels": {
            "knuckle_lock":         "Knuckle Lock",
            "coupler_alignment":    "Coupler Alignment",
            "structural_integrity": "Structural Integrity",
            "pin_assembly":         "Pin Assembly",
            "corrosion_wear":       "Corrosion & Wear",
        },
    },
    "axle_box": {
        "fields": ["grease_seal", "casing_integrity", "bolt_condition", "bearing_temperature", "contamination", "earthing_device"],
        "labels": {
            "grease_seal":      "Grease / Rear Seal",
            "casing_integrity": "Casing Integrity",
            "bolt_condition":   "Fastener / Bolt Condition",
            "bearing_temperature": "Bearing Heat Discoloration",
            "contamination":    "Oil / Grease Contamination",
            "earthing_device":  "Earthing Device / Carbon Brush",
        },
    },
    "brake_disk": {
        "fields": ["surface_condition", "thermal_cracks", "groove_depth", "mounting_bolts", "contamination"],
        "labels": {
            "surface_condition": "Disk Surface Condition",
            "thermal_cracks":    "Thermal / Radial Cracks",
            "groove_depth":      "Scoring / Groove Depth",
            "mounting_bolts":    "Mounting Bolt Integrity",
            "contamination":     "Oil / Fluid Contamination",
        },
    },
    "damper": {
        "fields": ["piston_rod", "oil_leakage", "mounting_bushings", "casing_damage", "stroke_travel"],
        "labels": {
            "piston_rod":        "Piston Rod Condition",
            "oil_leakage":       "Oil / Fluid Leakage",
            "mounting_bushings": "Silent Block / Bushings",
            "casing_damage":     "Casing / Barrel Damage",
            "stroke_travel":     "Stroke Travel Indicator",
        },
    },
    "spring": {
        "fields": ["coil_fracture", "permanent_set", "surface_corrosion", "seating_contact", "coil_spacing"],
        "labels": {
            "coil_fracture":     "Coil Fracture / Cracks",
            "permanent_set":     "Permanent Set / Sagging",
            "surface_corrosion": "Surface Corrosion / Pitting",
            "seating_contact":   "Spring Seat Contact",
            "coil_spacing":      "Inter-Coil Spacing",
        },
    },
    "wheel": {
        "fields": [
            "flange_height", "flange_thickness", "qr_value", "sharp_flange",
            "thin_flange", "tread_hollow", "shelling", "spalling",
            "wheel_diameter", "wheel_diameter_diff",
            "wheel_flat_length", "wheel_flat_depth",
            "thermal_crack", "rim_crack", "web_crack", "hub_crack",
        ],
        "labels": {
            "flange_height":       "Flange Height",
            "flange_thickness":    "Flange Thickness",
            "qr_value":           "qR Value (Flange Root)",
            "sharp_flange":       "Sharp Flange",
            "thin_flange":        "Thin Flange",
            "tread_hollow":       "Tread Hollow",
            "shelling":           "Shelling (Surface Fatigue)",
            "spalling":           "Spalling (Material Loss)",
            "wheel_diameter":     "Wheel Diameter",
            "wheel_diameter_diff":"Wheel Diameter Difference (Axle)",
            "wheel_flat_length":  "Wheel Flat Length",
            "wheel_flat_depth":   "Wheel Flat Depth",
            "thermal_crack":      "Thermal Crack",
            "rim_crack":          "Rim Crack",
            "web_crack":          "Web Crack",
            "hub_crack":          "Hub Crack",
        },
    },
}

COMPONENT_SYSTEM_PROMPTS = {
    "coupler":
        "You are a Senior Mechanical Inspector at Indian Railways (RDSO certified) specialising in "
        "CBC H-type Tightlock Couplers used in LHB coaches. "
        "Evaluate: knuckle lock, coupler body alignment, structural cracks, pin/lock lift assembly, rust.",
    "axle_box":
        "You are a Senior Mechanical Inspector at Indian Railways (RDSO certified) specialising in "
        "Axle Box assemblies on LHB Fiat Bogies. "
        "Evaluate: rear grease seal leakage, casing cracks, fastener torque condition, "
        "roller bearing thermal discoloration, oil/grease contamination, and critically — "
        "the Earthing Device (carbon brush/earthing disc assembly) condition. "
        "Per RDSO/2002/CG-01 & IRIEEN guidelines, the earthing brush must maintain contact "
        "to prevent stray traction currents from damaging bearings. Check brush wear, spring pressure, "
        "contact area cleanliness, and carbon brush holder integrity.",
    "brake_disk":
        "You are a Senior Mechanical Inspector at Indian Railways (RDSO certified) specialising in "
        "Brake Disk assemblies on LHB Fiat Bogies. "
        "Evaluate: disk surface scoring, radial thermal cracks, groove depth beyond limits, "
        "mounting bolt condition, and oil/fluid contamination.",
    "damper":
        "You are a Senior Mechanical Inspector at Indian Railways (RDSO certified) specialising in "
        "Hydraulic Dampers on LHB Fiat Bogies. "
        "Evaluate: piston rod surface condition, oil leakage from seals, silent block bushing cracks, "
        "barrel/casing physical damage, and available stroke travel.",
    "spring":
        "You are a Senior Mechanical Inspector at Indian Railways (RDSO certified) specialising in "
        "Helical Coil Springs on LHB Fiat Bogies. "
        "Evaluate: coil fractures or cracks, permanent set/sagging beyond RDSO limits, "
        "surface corrosion/pitting, spring seat seating quality, and inter-coil spacing.",
    "wheel": (
        "You are an expert Railway Wheel Inspection AI for LHB coaches at Indian Railways.\n"
        "You operate strictly under RDSO / IRS T-27 standards.\n\n"

        "=== ABSOLUTE RULES — NEVER VIOLATE ===\n"
        "1. NEVER generate fixed, assumed, or imagined values. Every finding must come ONLY from "
        "the Computer Vision data provided.\n"
        "2. If a parameter CANNOT be determined from CV data, set status to NOT_DETECTED and "
        "running_fitness to INSUFFICIENT_DATA.\n"
        "3. Use the EXACT limits below — do not alter or approximate them.\n"
        "4. Never produce identical reports for different images.\n\n"

        "=== EXACT RDSO PARAMETER LIMITS — USE PRECISELY ===\n\n"

        "1. FLANGE HEIGHT (FH) | New Wheel: 28.5 mm | Condemning Max: 35 mm\n"
        "   ≤ 32 mm           → GOOD    | FIT FOR RUNNING\n"
        "   > 32 mm ≤ 35 mm   → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   > 35 mm            → CRITICAL | NOT FIT FOR RUNNING | Action: Reprofile Wheel\n\n"

        "2. FLANGE THICKNESS (FT) | New Wheel: 29.4 mm | Minimum: 22 mm\n"
        "   ≥ 25 mm           → GOOD    | FIT FOR RUNNING\n"
        "   22–25 mm          → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   < 22 mm            → CRITICAL | NOT FIT FOR RUNNING | Action: Reprofile Wheel\n\n"

        "3. qR VALUE | Use latest RDSO configured limit\n"
        "   Within standard  → GOOD    | FIT FOR RUNNING\n"
        "   Near limit        → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   Out of limit      → CRITICAL | NOT FIT FOR RUNNING\n\n"

        "4. SHARP FLANGE\n"
        "   Not detected      → GOOD    | FIT FOR RUNNING\n"
        "   Minor sharpness   → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   Sharp detected    → CRITICAL | NOT FIT FOR RUNNING\n\n"

        "5. THIN FLANGE\n"
        "   Not detected      → GOOD    | FIT FOR RUNNING\n"
        "   Moderate wear     → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   Detected          → CRITICAL | NOT FIT FOR RUNNING\n\n"

        "6. TREAD HOLLOW\n"
        "   ≤ Standard limit  → GOOD    | FIT FOR RUNNING\n"
        "   Near limit        → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   Exceeds limit     → CRITICAL | NOT FIT FOR RUNNING\n\n"

        "7. SHELLING (Surface Fatigue)\n"
        "   Area < 5%         → GOOD    | FIT FOR RUNNING\n"
        "   Area 5–15%        → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   Area > 15%        → CRITICAL | NOT FIT FOR RUNNING\n\n"

        "8. SPALLING (Material Loss)\n"
        "   No spalling       → GOOD    | FIT FOR RUNNING\n"
        "   Localized         → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   Heavy material loss → CRITICAL | NOT FIT FOR RUNNING\n\n"

        "9. WHEEL DIAMETER | New: 915 mm | Condemning: 855 mm\n"
        "   > 880 mm          → GOOD    | FIT FOR RUNNING\n"
        "   855–880 mm        → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   < 855 mm          → CRITICAL | NOT FIT FOR RUNNING\n\n"

        "10. WHEEL DIAMETER DIFFERENCE (same axle) | Max: 0.5 mm\n"
        "   ≤ 0.5 mm          → GOOD    | FIT FOR RUNNING\n"
        "   0.5–1.0 mm        → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   > 1.0 mm          → CRITICAL | NOT FIT FOR RUNNING\n\n"

        "11. WHEEL FLAT LENGTH\n"
        "   0–20 mm           → GOOD    | FIT FOR RUNNING\n"
        "   20–50 mm          → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   > 50 mm           → CRITICAL | NOT FIT FOR RUNNING\n\n"

        "12. WHEEL FLAT DEPTH\n"
        "   0–1 mm            → GOOD    | FIT FOR RUNNING\n"
        "   1–2 mm            → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   > 2 mm            → CRITICAL | NOT FIT FOR RUNNING\n\n"

        "13. THERMAL CRACK\n"
        "   No crack          → GOOD    | FIT FOR RUNNING\n"
        "   Length < 10 mm    → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   Length > 10 mm    → CRITICAL | NOT FIT FOR RUNNING\n\n"

        "14. RIM CRACK\n"
        "   No crack          → GOOD    | FIT FOR RUNNING\n"
        "   Crack < 5 mm      → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   Crack ≥ 5 mm      → CRITICAL | REMOVE FROM SERVICE\n\n"

        "15. WEB CRACK\n"
        "   No crack          → GOOD    | FIT FOR RUNNING\n"
        "   Minor crack       → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   Structural crack  → CRITICAL | REMOVE FROM SERVICE\n\n"

        "16. HUB CRACK\n"
        "   No crack          → GOOD    | FIT FOR RUNNING\n"
        "   Small crack       → WARNING | FIT FOR RUNNING WITH MONITORING\n"
        "   Major crack       → CRITICAL | REMOVE FROM SERVICE\n\n"

        "=== DEFECT SCORE BANDS ===\n"
        "0–20 = GOOD | 21–40 = MINOR DEFECT | 41–60 = WARNING | 61–80 = HIGH RISK | 81–100 = CRITICAL\n"
        "Always explain: which detected defects contributed, why the score was assigned, "
        "which parameters triggered the risk level.\n\n"

        "=== FINAL DECISION — CHOOSE EXACTLY ONE ===\n"
        "Fit for Service | Monitor | Schedule Maintenance | "
        "Immediate Reprofiling Required | Remove From Service | Condemn Wheelset\n\n"

        "=== OUTPUT FORMAT PER PARAMETER ===\n"
        "For every detectable parameter output:\n"
        "  status           : GOOD / WARNING / CRITICAL / NOT_DETECTED\n"
        "  detected_value   : actual CV-detected measurement with units\n"
        "  standard_limit   : exact RDSO limit from table above\n"
        "  running_fitness  : FIT FOR RUNNING / FIT FOR RUNNING WITH MONITORING / "
        "NOT FIT FOR RUNNING / REMOVE FROM SERVICE / INSUFFICIENT_DATA\n"
        "  technical_justification : why this status was assigned based on CV data\n"
        "  maintenance_recommendation : specific action for workshop crew\n"
    ),
}


def _coach_info_str(coach_meta: dict) -> str:
    """Helper function to format coach metadata into a readable string for the LLM."""
    # Build a formatted string of the coach's metadata, falling back to safe defaults if missing
    return (
        f"Coach Number: {coach_meta.get('coach_number', 'N/A')} | "
        f"Type: {coach_meta.get('coach_type', 'LHB')} | "
        f"Depot: {coach_meta.get('depot', 'N/A')} | "
        f"Zone: {coach_meta.get('zone', 'N/A')} | "
        f"Inspector: {coach_meta.get('inspector_name', 'N/A')}"
    )


def _cv_summary(cnn_result: dict) -> str:
    """Helper function to summarize standard Computer Vision (CNN) results into a text block."""
    # Extract the bounding box coordinates, if available
    bbox = cnn_result.get("bbox")
    # Format the bounding box as a string, handling the case where it's missing
    bbox_str = (
        f"Defect at xmin={bbox['xmin']}, ymin={bbox['ymin']}, xmax={bbox['xmax']}, ymax={bbox['ymax']}"
        if bbox else "No bounding box detected."
    )
    # Combine various CV metrics (status, probabilities, specific levels) into a pipe-separated string
    return (
        f"CNN: {cnn_result.get('status', 'UNKNOWN')} | "
        f"Defect Prob: {cnn_result.get('defect_score', 0):.1f}% | "
        f"Confidence: {cnn_result.get('confidence', 0):.1f}% | "
        f"Rust: {cnn_result.get('rust_level', 0):.1f}% | "
        f"Oil Stain: {cnn_result.get('oil_level', 0):.1f}% | "
        f"Edge Anomaly: {cnn_result.get('edge_density', 0):.1f}% | "
        f"Alignment: {'NORMAL' if cnn_result.get('alignment_ok', True) else 'DEVIATION'} | "
        f"BBox: {bbox_str}"
    )


def _json_schema(component: str) -> str:
    """Helper function to generate the expected JSON schema string for the LLM prompt."""
    # Retrieve the specific checklist fields for the component, defaulting to coupler if not found
    fields = COMPONENT_CHECKLISTS.get(component, COMPONENT_CHECKLISTS["coupler"])["fields"]
    
    # Check if the component is 'wheel', which requires an extended, detailed JSON schema
    if component == "wheel":
        # Generate the JSON properties for each wheel field
        items = "\n".join(
            f'  "{f}": {{'
            f'"status": "GOOD/WARNING/CRITICAL/NOT_DETECTED", '
            f'"detected_value": "e.g. 21.3 mm", '
            f'"standard_limit": "e.g. >= 22 mm", '
            f'"running_fitness": "FIT FOR RUNNING / FIT FOR RUNNING WITH MONITORING / NOT FIT FOR RUNNING / REMOVE FROM SERVICE / INSUFFICIENT_DATA", '
            f'"technical_justification": "Why this status based on exact CV measurement vs RDSO limit", '
            f'"maintenance_recommendation": "Specific action"}},'
            for f in fields
        )
        # Wrap the fields in the top-level JSON structure tailored for wheels
        return (
            "{\n" + items + "\n"
            '  "defect_score": 0,\n'
            '  "defect_score_justification": "Which detected parameters drove this score and why",\n'
            '  "overall_diagnosis": "Full RDSO-standard technical diagnosis paragraph",\n'
            '  "risk_assessment": "LOW/MEDIUM/HIGH/CRITICAL",\n'
            '  "immediate_action": "Exact maintenance crew instruction per RDSO standard",\n'
            '  "final_decision": "Fit for Service / Monitor / Schedule Maintenance / Immediate Reprofiling Required / Remove From Service / Condemn Wheelset",\n'
            '  "workshop_code": "NONE / PERIODIC / URGENT / EMERGENCY"\n'
            "}"
        )
        
    # For all non-wheel components, use a simpler JSON schema structure
    items  = "\n".join(
        f'  "{f}": {{"status": "OK/WARNING/CRITICAL", "detail": "..."}},'
        for f in fields
    )
    # Wrap the simple fields in the top-level standard JSON structure
    return (
        "{\n" + items + "\n"
        '  "overall_diagnosis": "One paragraph diagnosis",\n'
        '  "risk_assessment": "LOW/MEDIUM/HIGH/CRITICAL",\n'
        '  "immediate_action": "Specific action for maintenance crew",\n'
        '  "final_decision": "FIT FOR RUN / MONITOR CLOSELY / UNFIT - SEND TO WORKSHOP",\n'
        '  "workshop_code": "NONE / PERIODIC / URGENT / EMERGENCY"\n'
        "}"
    )


def _wheel_cv_detail(cnn_result: dict) -> str:
    """Build a detailed CV data block specifically tailored for wheel inspection."""
    # Extract wheel-specific features, defaulting to an empty dictionary
    wheel = cnn_result.get("wheel_features", {})
    # Extract standard CNN result parameters, defaulting to 0 or True as appropriate
    defect_score  = cnn_result.get("defect_score", 0)
    rust          = cnn_result.get("rust_level", 0)
    edge          = cnn_result.get("edge_density", 0)
    oil           = cnn_result.get("oil_level", 0)
    alignment_ok  = cnn_result.get("alignment_ok", True)
    confidence    = cnn_result.get("confidence", 0)
    status        = cnn_result.get("status", "UNKNOWN")
    bbox          = cnn_result.get("bbox")

    # Interpret and map the numeric defect score into a categorized severity band
    if defect_score <= 20:   score_band = "GOOD"
    elif defect_score <= 40: score_band = "MINOR DEFECT"
    elif defect_score <= 60: score_band = "WARNING"
    elif defect_score <= 80: score_band = "HIGH RISK"
    else:                    score_band = "CRITICAL"

    # Construct the bounding box text using Grad-CAM pixel coordinates, if available
    bbox_str = (
        f"Defect region localised at xmin={bbox['xmin']} ymin={bbox['ymin']} "
        f"xmax={bbox['xmax']} ymax={bbox['ymax']} (pixel coords, Grad-CAM)"
        if bbox else "No defect region localised by Grad-CAM."
    )

    # Initialize a list of lines representing the computer vision output summary
    lines = [
        "=== COMPUTER VISION OUTPUT — WHEEL INSPECTION ===",
        f"CNN Model Status   : {status}",
        f"Model Confidence   : {confidence:.1f}%",
        f"Defect Score       : {defect_score:.1f}%  [{score_band}]",
        f"Rust/Corrosion     : {rust:.1f}%",
        f"Edge Anomaly Index : {edge:.1f}%  (high = structural surface irregularity)",
        f"Oil/Fluid Stain    : {oil:.1f}%",
        f"Wheel Alignment    : {'NORMAL' if alignment_ok else 'DEVIATION DETECTED'}",
        f"Grad-CAM BBox      : {bbox_str}",
    ]

    # If there are any wheel-specific CV features, append them as a separate block
    if wheel:
        lines.append("")
        lines.append("=== WHEEL-SPECIFIC CV MEASUREMENTS ===")
        # Iterate over the wheel features dictionary and format each key-value pair
        for k, v in wheel.items():
            lines.append(f"  {k:<30}: {v}")

    # Derive high-level defect signals or actionable advice from the numeric CV data
    lines.append("")
    lines.append("=== DERIVED DEFECT SIGNALS (use to populate parameters) ===")
    # Check for high defect probability
    if defect_score > 60:
        lines.append(f"  HIGH defect probability ({defect_score:.1f}%) — inspect: shelling, spalling, cracks, flat")
    # Check for significant surface corrosion
    if rust > 15:
        lines.append(f"  Significant corrosion detected ({rust:.1f}%) — check rim surface, web, hub")
    # Check for excessive edge anomalies
    if edge > 18:
        lines.append(f"  High edge anomaly ({edge:.1f}%) — possible flat, crack, or tread irregularity")
    # Check for wheel alignment deviation
    if not alignment_ok:
        lines.append("  Alignment deviation — check wheel diameter difference, flange geometry")
    # Check for oil or fluid leaks
    if oil > 10:
        lines.append(f"  Oil/fluid contamination ({oil:.1f}%) — check axle bearing seal")
    # Check if a defect region was localized
    if bbox:
        lines.append("  Localised defect region confirmed by Grad-CAM — likely surface defect or crack")

    # Join the list of lines into a single newline-separated string
    return "\n".join(lines)


def _build_text_prompt(cnn_result: dict, coach_meta: dict, component: str) -> str:
    """Helper function to compile the complete text prompt for the LLM."""
    # Get the specific system instructions for the given component (defaults to coupler)
    system = COMPONENT_SYSTEM_PROMPTS.get(component, COMPONENT_SYSTEM_PROMPTS["coupler"])
    # Generate the expected JSON structure string for this component
    schema = _json_schema(component)
    
    # If the component is a wheel, generate the detailed wheel CV block
    if component == "wheel":
        cv_block = _wheel_cv_detail(cnn_result)
    # Otherwise, generate the standard component CV summary block
    else:
        cv_block = _cv_summary(cnn_result)
        
    # Construct and return the full prompt string, combining instructions, coach info, CV data, and schema
    return (
        f"{system}\n\n"
        f"COACH INFORMATION:\n{_coach_info_str(coach_meta)}\n\n"
        f"COMPUTER VISION DATA:\n{cv_block}\n\n"
        f"INSTRUCTION: Generate the inspection report ONLY from the CV data above. "
        f"Do NOT invent measurements not present in the data. "
        f"Mark parameters as NOT_DETECTED if the CV data provides no evidence for them.\n\n"
        f"Respond ONLY with valid JSON matching exactly this schema:\n{schema}"
    )


def _build_vision_prompt(pil_img: Image.Image, coach_meta: dict, component: str) -> list:
    """Helper function to compile a multimodal (vision) prompt containing an image and text instructions."""
    # Create an in-memory bytes buffer to hold the image data
    buf = BytesIO()
    # Resize the image to 512x512 using Lanczos resampling and save it as a JPEG in the buffer
    pil_img.resize((512, 512), Image.LANCZOS).save(buf, format="JPEG", quality=85)
    # Encode the image bytes into a base64 string
    b64    = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    # Get the system instructions for the requested component
    system = COMPONENT_SYSTEM_PROMPTS.get(component, COMPONENT_SYSTEM_PROMPTS["coupler"])
    # Get the JSON schema, strip the trailing brace, and append a field for bounding box coordinate output
    schema = _json_schema(component).rstrip("}") + ',\n  "bbox": null\n}'
    
    # Return the message structure formatted for an OpenRouter Vision request
    return [{
        "role": "user",
        "content": [
            # Pass the base64 image data as an image URL block
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            # Pass the text instructions and schema alongside the image
            {
                "type": "text",
                "text": (
                    f"{system}\n\n"
                    f"COACH: {_coach_info_str(coach_meta)}\n\n"
                    f"Inspect the image carefully. If you detect a defect, provide its location as "
                    f"'bbox' with normalised 0-1000 coordinates.\n\n"
                    f"Respond ONLY with valid JSON:\n{schema}"
                ),
            },
        ],
    }]


def _call_openrouter(messages: list, model: str) -> str:
    """Helper function to execute the HTTP POST request to the OpenRouter AI API."""
    # Prepare the headers required by OpenRouter, including authentication and app identity
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer":  APP_SITE_URL,
        "X-Title":       APP_TITLE,
        "Content-Type":  "application/json",
    }
    # Prepare the JSON payload body for the API request
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  2048,   # Wheel reports are dense, requiring a higher token limit
        "temperature": 0.25,   # Keep the temperature low to ensure consistent and technical outputs
    }
    # Issue the POST request to OpenRouter with a 60-second timeout
    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers, json=payload, timeout=60,
    )
    # Raise an exception if the HTTP request returned an error status code
    resp.raise_for_status()
    # Extract and return the raw string content from the API response message
    return resp.json()["choices"][0]["message"]["content"]


def _parse_json_response(raw: str, component: str) -> dict:
    """Helper function to safely extract and parse a JSON dictionary from the LLM's string response."""
    # Strip any leading/trailing whitespace from the raw response
    raw = raw.strip()
    # If the response contains markdown code block formatting (```), attempt to strip it out
    if "```" in raw:
        # Find the index of the first opening brace
        start = raw.find("{")
        # Find the index immediately after the last closing brace
        end   = raw.rfind("}") + 1
        # Slice the string to only include the actual JSON object
        raw   = raw[start:end]
    
    try:
        # Parse the JSON string into a Python dictionary
        return json.loads(raw)
    except json.JSONDecodeError:
        # If parsing fails, generate a fallback response dictionary containing safe default values
        fields = COMPONENT_CHECKLISTS.get(component, COMPONENT_CHECKLISTS["coupler"])["fields"]
        return {
            # Fill every required field with an UNKNOWN status indicating a parse error occurred
            **{f: {"status": "UNKNOWN", "detail": "Parse error"} for f in fields},
            # Truncate the raw output to 500 characters and use it as the overall diagnosis
            "overall_diagnosis": raw[:500],
            "risk_assessment":   "UNKNOWN",
            "immediate_action":  "Manual inspection required",
            "final_decision":    "MONITOR CLOSELY",
            "workshop_code":     "PERIODIC",
            "bbox":              None,
        }


def _evaluate_wheel_features_direct(cnn_result: dict) -> dict:
    """
    Directly evaluate wheel_features measurements against RDSO/IRS T-27 limits.
    Returns a fully populated checklist bypassing the LLM — always works even if
    the OpenRouter API fails or returns unparseable JSON.
    """
    import re
    wheel = cnn_result.get("wheel_features", {})
    if not wheel:
        return {}

    def num(s):
        """Extract first float from a string like '31.5 mm'."""
        m = re.search(r'[\d.]+', str(s))
        return float(m.group()) if m else None

    def entry(status, detected, limit, fitness, justification, recommendation):
        """Build a standardised wheel parameter dict."""
        return {
            "status":                  status,
            "detected_value":          detected,
            "standard_limit":          limit,
            "running_fitness":         fitness,
            "technical_justification": justification,
            "maintenance_recommendation": recommendation,
        }

    cl = {}

    # ── 1. Flange Height ──────────────────────────────────────────
    fh = num(wheel.get("flange_height", ""))
    if fh is not None:
        v = wheel["flange_height"]
        if fh <= 32:
            cl["flange_height"] = entry("GOOD", v, "<= 32 mm (max 35 mm)", "FIT FOR RUNNING",
                f"FH {fh} mm is within RDSO safe limit of <=32 mm.", "Continue scheduled monitoring.")
        elif fh <= 35:
            cl["flange_height"] = entry("WARNING", v, "32–35 mm warning zone", "FIT FOR RUNNING WITH MONITORING",
                f"FH {fh} mm exceeds 32 mm warning threshold — approaching condemning limit of 35 mm.",
                "Schedule reprofiling at next available opportunity.")
        else:
            cl["flange_height"] = entry("CRITICAL", v, "Max 35 mm (RDSO condemning limit)", "NOT FIT FOR RUNNING",
                f"FH {fh} mm exceeds RDSO condemning limit of 35 mm.", "Reprofile wheel immediately. Do not run.")

    # ── 2. Flange Thickness ───────────────────────────────────────
    ft = num(wheel.get("flange_thickness", ""))
    if ft is not None:
        v = wheel["flange_thickness"]
        if ft >= 25:
            cl["flange_thickness"] = entry("GOOD", v, ">= 25 mm (min 22 mm)", "FIT FOR RUNNING",
                f"FT {ft} mm is above RDSO 25 mm threshold.", "Continue scheduled monitoring.")
        elif ft >= 22:
            cl["flange_thickness"] = entry("WARNING", v, "22–25 mm warning zone", "FIT FOR RUNNING WITH MONITORING",
                f"FT {ft} mm is in the warning zone (22–25 mm).", "Schedule reprofiling. Monitor closely.")
        else:
            cl["flange_thickness"] = entry("CRITICAL", v, "Min 22 mm (RDSO limit)", "NOT FIT FOR RUNNING",
                f"FT {ft} mm is below RDSO minimum of 22 mm.", "Reprofile wheel immediately. Do not run.")

    # ── 3. qR Value ───────────────────────────────────────────────
    qr = num(wheel.get("qr_value", ""))
    if qr is not None:
        v = wheel["qr_value"]
        if qr >= 7.0:
            cl["qr_value"] = entry("GOOD", v, ">= 6.5 mm", "FIT FOR RUNNING",
                f"qR {qr} mm is within RDSO standard.", "Continue scheduled monitoring.")
        elif qr >= 6.5:
            cl["qr_value"] = entry("WARNING", v, "Near limit (min 6.5 mm)", "FIT FOR RUNNING WITH MONITORING",
                f"qR {qr} mm is near RDSO minimum.", "Monitor closely. Schedule reprofiling.")
        else:
            cl["qr_value"] = entry("CRITICAL", v, "Min 6.5 mm (RDSO limit)", "NOT FIT FOR RUNNING",
                f"qR {qr} mm is below RDSO minimum.", "Reprofile wheel immediately.")

    # ── 4. Sharp Flange ───────────────────────────────────────────
    sf = str(wheel.get("sharp_flange", "")).lower()
    if "no" in sf or sf in ("", "none"):
        cl["sharp_flange"] = entry("GOOD", "Not detected", "No sharp flange permitted", "FIT FOR RUNNING",
            "No sharp flange observed.", "No action required.")
    else:
        cl["sharp_flange"] = entry("WARNING", wheel["sharp_flange"], "No sharp flange permitted",
            "FIT FOR RUNNING WITH MONITORING", "Minor sharpness detected on flange.", "Schedule wheel turning.")

    # ── 5. Thin Flange ────────────────────────────────────────────
    tf = str(wheel.get("thin_flange", "")).lower()
    if "no" in tf or tf in ("", "none"):
        cl["thin_flange"] = entry("GOOD", "Not detected", "No thin flange permitted", "FIT FOR RUNNING",
            "No thin flange observed.", "No action required.")
    else:
        cl["thin_flange"] = entry("CRITICAL", wheel["thin_flange"], "No thin flange permitted",
            "NOT FIT FOR RUNNING", "Thin flange detected.", "Reprofile wheel immediately.")

    # ── 6. Tread Hollow ───────────────────────────────────────────
    th = num(wheel.get("tread_hollow", ""))
    if th is not None:
        v = wheel["tread_hollow"]
        if th <= 5.0:
            cl["tread_hollow"] = entry("GOOD", v, "<= 5.0 mm", "FIT FOR RUNNING",
                f"Tread hollow {th} mm is within RDSO limit.", "Continue monitoring.")
        elif th <= 8.0:
            cl["tread_hollow"] = entry("WARNING", v, "5.0–8.0 mm warning zone", "FIT FOR RUNNING WITH MONITORING",
                f"Tread hollow {th} mm approaching limit.", "Schedule reprofiling.")
        else:
            cl["tread_hollow"] = entry("CRITICAL", v, "Max 5.0 mm (RDSO)", "NOT FIT FOR RUNNING",
                f"Tread hollow {th} mm exceeds RDSO limit.", "Reprofile immediately.")

    # ── 7. Shelling ───────────────────────────────────────────────
    sh_raw = str(wheel.get("shelling", ""))
    sh_pct = num(sh_raw)
    if sh_pct is not None and sh_pct == 0 or sh_raw.strip() == "0%":
        cl["shelling"] = entry("GOOD", "0%", "< 5% area", "FIT FOR RUNNING",
            "No shelling observed.", "No action required.")
    elif sh_pct is not None:
        v = sh_raw
        if sh_pct < 5:
            cl["shelling"] = entry("GOOD", v, "< 5% area", "FIT FOR RUNNING",
                f"Shelling area {sh_pct}% is within RDSO limit.", "Monitor at next inspection.")
        elif sh_pct <= 15:
            cl["shelling"] = entry("WARNING", v, "5–15% area", "FIT FOR RUNNING WITH MONITORING",
                f"Shelling area {sh_pct}% in warning zone.", "Schedule maintenance.")
        else:
            cl["shelling"] = entry("CRITICAL", v, "Max 5% area", "NOT FIT FOR RUNNING",
                f"Shelling area {sh_pct}% exceeds RDSO limit.", "Remove from service.")
    else:
        cl["shelling"] = entry("GOOD", "0%", "< 5% area", "FIT FOR RUNNING",
            "No significant shelling detected.", "Continue monitoring.")

    # ── 8. Spalling ───────────────────────────────────────────────
    sp = str(wheel.get("spalling", "")).lower()
    if sp in ("none", "", "no spalling"):
        cl["spalling"] = entry("GOOD", "None", "No spalling permitted", "FIT FOR RUNNING",
            "No spalling observed.", "No action required.")
    elif "localized" in sp:
        cl["spalling"] = entry("WARNING", wheel["spalling"], "No localized spalling",
            "FIT FOR RUNNING WITH MONITORING", "Localized spalling detected.", "Schedule workshop attention.")
    else:
        cl["spalling"] = entry("CRITICAL", wheel["spalling"], "No spalling permitted",
            "NOT FIT FOR RUNNING", "Significant spalling detected.", "Remove from service.")

    # ── 9. Wheel Diameter ─────────────────────────────────────────
    wd = num(wheel.get("wheel_diameter", ""))
    if wd is not None:
        v = wheel["wheel_diameter"]
        if wd > 880:
            cl["wheel_diameter"] = entry("GOOD", v, "> 880 mm (condemning: 855 mm)", "FIT FOR RUNNING",
                f"Wheel diameter {wd} mm is above 880 mm threshold.", "Continue monitoring.")
        elif wd >= 855:
            cl["wheel_diameter"] = entry("WARNING", v, "855–880 mm warning zone", "FIT FOR RUNNING WITH MONITORING",
                f"Wheel diameter {wd} mm in warning zone.", "Schedule reprofiling.")
        else:
            cl["wheel_diameter"] = entry("CRITICAL", v, "Min 855 mm (RDSO condemning)", "NOT FIT FOR RUNNING",
                f"Wheel diameter {wd} mm below condemning limit.", "Condemn wheelset.")

    # ── 10. Wheel Diameter Difference ─────────────────────────────
    wdd = num(wheel.get("wheel_diameter_diff", ""))
    if wdd is not None:
        v = wheel["wheel_diameter_diff"]
        if wdd <= 0.5:
            cl["wheel_diameter_diff"] = entry("GOOD", v, "<= 0.5 mm", "FIT FOR RUNNING",
                f"Diameter difference {wdd} mm is within RDSO limit.", "Continue monitoring.")
        elif wdd <= 1.0:
            cl["wheel_diameter_diff"] = entry("WARNING", v, "0.5–1.0 mm warning zone",
                "FIT FOR RUNNING WITH MONITORING", f"Diameter difference {wdd} mm in warning zone.",
                "Schedule workshop attention.")
        else:
            cl["wheel_diameter_diff"] = entry("CRITICAL", v, "Max 0.5 mm (RDSO)", "NOT FIT FOR RUNNING",
                f"Diameter difference {wdd} mm exceeds RDSO limit.", "Remove from service.")

    # ── 11. Wheel Flat Length ─────────────────────────────────────
    wfl = num(wheel.get("wheel_flat_length", ""))
    if wfl is not None:
        v = wheel["wheel_flat_length"]
        if wfl <= 20:
            cl["wheel_flat_length"] = entry("GOOD", v, "<= 20 mm", "FIT FOR RUNNING",
                f"Flat length {wfl} mm is within RDSO limit.", "Continue monitoring.")
        elif wfl <= 50:
            cl["wheel_flat_length"] = entry("WARNING", v, "20–50 mm warning zone",
                "FIT FOR RUNNING WITH MONITORING", f"Flat length {wfl} mm in warning zone.",
                "Schedule workshop attention.")
        else:
            cl["wheel_flat_length"] = entry("CRITICAL", v, "Max 20 mm / >50 mm critical",
                "NOT FIT FOR RUNNING", f"Flat length {wfl} mm critically exceeds RDSO limit.",
                "Remove from service immediately.")

    # ── 12. Wheel Flat Depth ──────────────────────────────────────
    wfd = num(wheel.get("wheel_flat_depth", ""))
    if wfd is not None:
        v = wheel["wheel_flat_depth"]
        if wfd <= 1.0:
            cl["wheel_flat_depth"] = entry("GOOD", v, "<= 1 mm", "FIT FOR RUNNING",
                f"Flat depth {wfd} mm is within RDSO limit.", "Continue monitoring.")
        elif wfd <= 2.0:
            cl["wheel_flat_depth"] = entry("WARNING", v, "1–2 mm warning zone",
                "FIT FOR RUNNING WITH MONITORING", f"Flat depth {wfd} mm in warning zone.",
                "Schedule workshop attention.")
        else:
            cl["wheel_flat_depth"] = entry("CRITICAL", v, "Max 1 mm (RDSO)", "NOT FIT FOR RUNNING",
                f"Flat depth {wfd} mm exceeds RDSO limit.", "Remove from service.")

    # ── 13–16. Cracks ─────────────────────────────────────────────
    crack_cfg = {
        "thermal_crack": ("Thermal",  10, "< 10 mm (>10 mm critical)",   "Max 10 mm (RDSO)"),
        "rim_crack":     ("Rim",        5, "< 5 mm (>= 5 mm critical)",   "Max 5 mm (RDSO)"),
        "web_crack":     ("Web",        0, "No crack permitted",           "No crack permitted"),
        "hub_crack":     ("Hub",        0, "No crack permitted",           "No crack permitted"),
    }
    for key, (label, limit_mm, warn_lim, crit_lim) in crack_cfg.items():
        raw_crack = str(wheel.get(key, "")).lower()
        if raw_crack in ("none", ""):
            cl[key] = entry("GOOD", "None", f"No {label.lower()} crack permitted", "FIT FOR RUNNING",
                f"No {label.lower()} crack detected.", "Continue scheduled inspection.")
        else:
            crack_len = num(wheel.get(key, ""))
            if limit_mm > 0 and crack_len is not None and crack_len < limit_mm:
                cl[key] = entry("WARNING", wheel[key], warn_lim, "FIT FOR RUNNING WITH MONITORING",
                    f"{label} crack {crack_len} mm — approaching condemning limit.", "Expedite inspection.")
            else:
                fitness = "REMOVE FROM SERVICE" if key in ("rim_crack", "web_crack", "hub_crack") else "NOT FIT FOR RUNNING"
                cl[key] = entry("CRITICAL", wheel.get(key, "Detected"), crit_lim, fitness,
                    f"{label} crack detected — RDSO condemning threshold reached.", "Remove from service immediately.")

    return cl


def analyze(
    cnn_result: dict,
    coach_meta: dict,
    component:  str = "coupler",
    model_key:  str = "balanced",
    pil_img:    Image.Image = None,
) -> dict:
    """Main function to trigger an AI analysis for a given component and its CV data."""
    # Ensure the OpenRouter API key is present before proceeding
    if not OPENROUTER_API_KEY:
        return {"error": "OpenRouter API key not configured"}

    # Resolve the requested model string using the FREE_MODELS mapping, or use the default
    model         = FREE_MODELS.get(model_key, DEFAULT_MODEL)
    # Determine whether a vision-capable multimodal request is needed based on the model key and image presence
    is_vision     = model_key == "vision" and pil_img is not None
    # Load the relevant checklist configuration for the target component
    checklist_cfg = COMPONENT_CHECKLISTS.get(component, COMPONENT_CHECKLISTS["coupler"])

    try:
        # If performing a vision analysis, construct the multimodal message prompt
        if is_vision:
            messages = _build_vision_prompt(pil_img, coach_meta, component)
        # Otherwise, build a standard text-based prompt using the available CV metrics
        else:
            messages = [{"role": "user", "content": _build_text_prompt(cnn_result, coach_meta, component)}]

        # Execute the API call to OpenRouter and receive the raw model completion string
        raw_response = _call_openrouter(messages, model)
        # Attempt to parse the resulting JSON string into a Python dictionary
        parsed       = _parse_json_response(raw_response, component)

        # Initialize the AI bounding box as None
        ai_bbox = None
        # If this was a vision task and the AI returned bounding box coordinates, map them to image pixels
        if is_vision and parsed.get("bbox") and pil_img:
            b = parsed["bbox"]
            w, h = pil_img.size
            # Calculate pixel coordinates by scaling the AI's 0-1000 normalized output against image dimensions
            ai_bbox = {
                "xmin": int(b.get("xmin", 0) / 1000 * w),
                "ymin": int(b.get("ymin", 0) / 1000 * h),
                "xmax": int(b.get("xmax", 1000) / 1000 * w),
                "ymax": int(b.get("ymax", 1000) / 1000 * h),
            }

        # Extract the final decision string from the parsed AI response
        decision  = parsed.get("final_decision", "")

        # Evaluate the status keyword depending on whether it's the wheel component (which uses specific terms)
        if component == "wheel":
            # Define keywords indicating a wheel condemnation or workshop action
            condemn_terms  = ["Condemn", "Remove From Service"]
            workshop_terms = ["Reprofiling", "Schedule Maintenance", "UNFIT"]
            monitor_terms  = ["Monitor"]
            
            # Map the parsed decision string into one of three statuses: UNFIT, MONITOR, or FIT
            ai_status = (
                "UNFIT"   if any(t.lower() in decision.lower() for t in condemn_terms + workshop_terms) else
                "MONITOR" if any(t.lower() in decision.lower() for t in monitor_terms) else
                "FIT"
            )
        else:
            # Map non-wheel component decisions to standard status keywords
            ai_status = (
                "UNFIT"   if "UNFIT"   in decision else
                "MONITOR" if "MONITOR" in decision else
                "FIT"
            )

        # ── Build checklist ───────────────────────────────────────────
        if component == "wheel":
            # Always build from rule-based direct evaluator first (reliable, offline)
            direct_cl = _evaluate_wheel_features_direct(cnn_result)

            ai_checklist = {}
            for f in checklist_cfg["fields"]:
                # Prefer LLM result if it has a valid status; else use direct eval
                raw_param  = parsed.get(f, {})
                use_direct = not isinstance(raw_param, dict) or raw_param.get("status") in ("UNKNOWN", None, "")

                if use_direct and f in direct_cl:
                    dp = direct_cl[f]
                    raw_status = dp.get("status", "UNKNOWN")
                else:
                    dp = raw_param if isinstance(raw_param, dict) else {}
                    raw_status = dp.get("status", "UNKNOWN")

                # Map status: GOOD and NOT_DETECTED both map to OK badge
                mapped = (
                    "OK"       if raw_status in ("GOOD", "NORMAL", "OK", "NOT_DETECTED") else
                    "WARNING"  if raw_status == "WARNING" else
                    "CRITICAL" if raw_status == "CRITICAL" else
                    "UNKNOWN"
                )

                # Build detail string from wheel-specific fields
                val   = dp.get("detected_value", "")
                limit = dp.get("standard_limit", "")
                fit   = dp.get("running_fitness", "")
                just  = dp.get("technical_justification", "")
                rec   = dp.get("maintenance_recommendation", "")

                detail_parts = []
                if val:   detail_parts.append(f"Detected: {val}")
                if limit: detail_parts.append(f"Limit: {limit}")
                if fit:   detail_parts.append(f"Fitness: {fit}")
                if just:  detail_parts.append(just)
                if rec:   detail_parts.append(f"Rec: {rec}")

                detail = " | ".join(detail_parts) if detail_parts else "-"
                ai_checklist[f] = {"status": mapped, "detail": detail}
        else:
            # For non-wheel components, use the LLM parameter dictionary directly
            ai_checklist = {
                f: parsed.get(f, {"status": "UNKNOWN", "detail": "-"})
                for f in checklist_cfg["fields"]
            }

        # Extract the AI's general diagnosis paragraph
        diagnosis = parsed.get("overall_diagnosis", "")
        # If this is a wheel analysis and a defect score justification was provided, prepend it to the diagnosis
        if component == "wheel" and parsed.get("defect_score_justification"):
            diagnosis = (
                f"[Defect Score {parsed.get('defect_score', '?')}%] "
                f"{parsed['defect_score_justification']}\n\n{diagnosis}"
            )

        # Construct and return the final analysis dictionary consolidating all calculated values
        return {
            "ai_model":         model,
            "ai_status":        ai_status,
            "ai_checklist":     ai_checklist,
            "checklist_labels": checklist_cfg["labels"],
            "ai_diagnosis":     diagnosis,
            "risk_assessment":  parsed.get("risk_assessment", "UNKNOWN"),
            "ai_action":        parsed.get("immediate_action", ""),
            "final_decision":   parsed.get("final_decision", ""),
            "workshop_code":    parsed.get("workshop_code", "NONE"),
            "ai_bbox":          ai_bbox,
        }

    except requests.exceptions.Timeout:
        # Return an error message if the OpenRouter API request takes too long
        return {"error": "OpenRouter request timed out. Using local result only."}
    except requests.exceptions.RequestException as e:
        # Return an error message if any network-related issue occurs during the API request
        return {"error": f"OpenRouter API error: {str(e)}"}
    except Exception as e:
        # Return a generic error message for any unhandled exceptions during the analysis flow
        return {"error": f"Analysis failed: {str(e)}"}
