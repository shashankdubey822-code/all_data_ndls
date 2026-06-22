"""
app.py - FastAPI main application for RAIL-SENSE
Multi-Component LHB Bogie AI Diagnostic System v2.0
"""

# Import standard library modules for OS interaction, unique ID generation, file operations
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path

# Load environment variables from a .env file located in the same directory as this script
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Import FastAPI framework components for API routing, file handling, and background tasks
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
# Import StaticFiles to serve static assets like images and frontend files
from fastapi.staticfiles import StaticFiles
# Import response types for returning files and JSON data
from fastapi.responses import FileResponse, JSONResponse
# Import CORS middleware to allow cross-origin requests from frontend to backend
from fastapi.middleware.cors import CORSMiddleware
# Import PIL for image processing and manipulation
from PIL import Image

# Import internal modules for database operations, AI model management, and OpenRouter analysis
import backend.database as db
import backend.model_manager as mm
import backend.openrouter_analyzer as ora

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
# Define the root directory of the project by going one level up from the current script's directory
PROJECT_ROOT = Path(__file__).parent.parent
# Define the dataset root (same as project root in this structure)
DATASET_ROOT = PROJECT_ROOT
# Define the directory where uploaded images will be temporarily/permanently saved
UPLOAD_DIR   = PROJECT_ROOT / "uploads"
# Define the directory where model weight files (.pth) are stored
MODEL_DIR    = PROJECT_ROOT / "backend"
# Define the SQLite database file path
DB_PATH      = PROJECT_ROOT / "backend" / "railsense.db"

# Create the uploads directory if it doesn't already exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# Set environment variables so internal modules can easily access these paths
os.environ["MODEL_PATH"]   = str(MODEL_DIR / "coupler_model.pth")
os.environ["DATASET_PATH"] = str(DATASET_ROOT)
os.environ["DB_PATH"]      = str(DB_PATH)

# Initialize the SQLite database (create tables if they don't exist)
db.init_db()
# Index the dataset folder to update the database with available training images
db.index_dataset(str(DATASET_ROOT))

# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
# Initialize the FastAPI application with metadata for OpenAPI docs
app = FastAPI(
    title="RAIL-SENSE LHB Bogie Diagnostic API",
    version="2.0.0",
    description="AI-powered multi-component inspection system for Indian Railways LHB Fiat Bogies",
)

# Add CORS middleware to permit cross-origin requests from any domain/method/header
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Define path to the frontend directory
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
# Mount the frontend directory to serve static CSS, JS, etc. at /static
app.mount("/static",  StaticFiles(directory=str(FRONTEND_DIR)), name="static")
# Mount the uploads directory so images can be accessed directly via URL at /uploads
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)),   name="uploads")

# Dynamically mount a /dataset/<component> route for each component's training image folder
for comp_key, comp_cfg in mm.COMPONENT_REGISTRY.items():
    comp_path = DATASET_ROOT / comp_cfg["folder"]
    # Check if the folder exists before attempting to mount
    if comp_path.is_dir():
        app.mount(
            f"/dataset/{comp_key}",
            StaticFiles(directory=str(comp_path)),
            name=f"dataset_{comp_key}",
        )


# Define the root endpoint to serve the main HTML frontend
@app.get("/")
async def root():
    # Return the index.html file to be rendered in the browser
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ─────────────────────────────────────────────────────────────
# POST /api/analyze
# ─────────────────────────────────────────────────────────────
# Define the endpoint to handle image uploads and execute component analysis
@app.post("/api/analyze")
async def analyze_image(
    # Accept multipart form data including the image file and associated metadata
    file:           UploadFile = File(...),
    component_type: str = Form("coupler"),
    coach_number:   str = Form(""),
    coach_type:     str = Form("LHB"),
    depot:          str = Form(""),
    zone:           str = Form(""),
    inspector_name: str = Form(""),
    notes:          str = Form(""),
    ai_model:       str = Form("balanced"),
    use_ai:         str = Form("true"),
):
    # Extract the file extension, defaulting to .jpg if missing
    ext       = Path(file.filename).suffix or ".jpg"
    # Generate a unique 12-character ID for the file name
    uid       = uuid.uuid4().hex[:12]
    fname     = f"{uid}{ext}"
    # Define the absolute path where the image will be saved
    save_path = UPLOAD_DIR / fname
    # Write the uploaded file chunks to the local disk safely
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Open the saved image using PIL and ensure it's in RGB format for model compatibility
    pil_img    = Image.open(save_path).convert("RGB")
    # Run the initial deep learning (CNN) inference to detect defects, rust, alignment, etc.
    cnn_result = mm.infer(pil_img, component=component_type)

    # Pack coach metadata into a dictionary to pass to the LLM analyzer
    coach_meta = {
        "coach_number":   coach_number,
        "coach_type":     coach_type,
        "depot":          depot,
        "zone":           zone,
        "inspector_name": inspector_name,
    }

    ai_result = {}
    # Check if the user opted to use the advanced AI (LLM) analysis
    if use_ai.lower() == "true":
        # Call the OpenRouter analyzer, passing CNN results and metadata
        ai_result = ora.analyze(
            cnn_result=cnn_result,
            coach_meta=coach_meta,
            component=component_type,
            model_key=ai_model,
            # If a vision model is selected, pass the PIL image for multimodal analysis
            pil_img=pil_img if ai_model == "vision" else None,
        )

    # Determine the final component status: prefer AI's status if available, fallback to CNN
    final_status = ai_result.get("ai_status") or cnn_result["status"]
    # Determine the final bounding box for the defect region
    final_bbox   = ai_result.get("ai_bbox")   or cnn_result.get("bbox")

    # Save the combined inspection results to the SQLite database
    inspection_id = db.save_inspection({
        "created_at":      datetime.now().isoformat(),
        "component_type":  component_type,
        "coach_number":    coach_number,
        "coach_type":      coach_type,
        "depot":           depot,
        "zone":            zone,
        "inspector_name":  inspector_name,
        "image_path":      f"/uploads/{fname}",
        "status":          final_status,
        "confidence":      cnn_result["confidence"],
        "defect_score":    cnn_result["defect_score"],
        "rust_level":      cnn_result["rust_level"],
        "alignment_ok":    cnn_result["alignment_ok"],
        "bbox":            final_bbox,
        "ai_model":        ai_result.get("ai_model", ""),
        "ai_diagnosis":    ai_result.get("ai_diagnosis", ""),
        "ai_checklist":    ai_result.get("ai_checklist", {}),
        "ai_action":       ai_result.get("ai_action", ""),
        "notes":           notes,
    })

    # Return a comprehensive JSON response to the frontend client
    return JSONResponse({
        "id":               inspection_id,
        "component_type":   component_type,
        # Look up the human-readable label for the component type
        "component_label":  mm.COMPONENT_REGISTRY.get(component_type, {}).get("label", component_type),
        "status":           final_status,
        "confidence":       cnn_result["confidence"],
        "defect_score":     cnn_result["defect_score"],
        "normal_score":     cnn_result["normal_score"],
        "rust_level":       cnn_result["rust_level"],
        "edge_density":     cnn_result["edge_density"],
        "oil_level":        cnn_result["oil_level"],
        "alignment_ok":     cnn_result["alignment_ok"],
        "bbox":             final_bbox,
        "image_url":        f"/uploads/{fname}",
        "ai_model":         ai_result.get("ai_model", ""),
        "ai_checklist":     ai_result.get("ai_checklist", {}),
        "checklist_labels": ai_result.get("checklist_labels", {}),
        "ai_diagnosis":     ai_result.get("ai_diagnosis", ""),
        "risk_assessment":  ai_result.get("risk_assessment", ""),
        "ai_action":        ai_result.get("ai_action", ""),
        "final_decision":   ai_result.get("final_decision", final_status),
        "workshop_code":    ai_result.get("workshop_code", "NONE"),
        "created_at":       datetime.now().isoformat(),
    })


# ─────────────────────────────────────────────────────────────
# GET /api/history
# ─────────────────────────────────────────────────────────────
# Define endpoint to fetch recent inspection history from the database
@app.get("/api/history")
async def get_history(limit: int = 50, component: str = None):
    # Pass limit and optional component filter to DB and return as JSON
    return JSONResponse(db.get_all_inspections(limit=limit, component=component))


# ─────────────────────────────────────────────────────────────
# GET /api/stats
# ─────────────────────────────────────────────────────────────
# Define endpoint to retrieve aggregated dashboard statistics
@app.get("/api/stats")
async def get_stats():
    # Return overall stats (e.g., total inspections, defect rates)
    return JSONResponse(db.get_stats())


# ─────────────────────────────────────────────────────────────
# GET /api/inspection/{id}
# ─────────────────────────────────────────────────────────────
# Define endpoint to fetch details of a specific inspection by its ID
@app.get("/api/inspection/{inspection_id}")
async def get_inspection(inspection_id: int):
    # Query database for the given ID
    row = db.get_inspection_by_id(inspection_id)
    # Return a 404 Not Found error if the record does not exist
    if not row:
        raise HTTPException(status_code=404, detail="Inspection not found")
    return JSONResponse(row)


# ─────────────────────────────────────────────────────────────
# GET /api/dataset
# ─────────────────────────────────────────────────────────────
# Define endpoint to list available training dataset images
@app.get("/api/dataset")
async def get_dataset(component: str = None, label: str = None):
    # Query the DB for indexed dataset images, optionally filtering by component/label
    images = db.get_dataset_images(component=component, label=label)
    # Loop over results and construct the direct URL for each image
    for img in images:
        comp  = img["component"]
        lbl   = img["label"]
        fname = img["filename"]
        img["url"] = f"/dataset/{comp}/{lbl}/{fname}"
    return JSONResponse(images)


# ─────────────────────────────────────────────────────────────
# GET /api/components
# ─────────────────────────────────────────────────────────────
# Define endpoint to list all supported bogie components
@app.get("/api/components")
async def get_components():
    # Construct a dictionary mapping component keys to their friendly labels and folder names
    return JSONResponse({
        k: {"label": v["label"], "folder": v["folder"]}
        for k, v in mm.COMPONENT_REGISTRY.items()
    })


# ─────────────────────────────────────────────────────────────
# POST /api/detect-component
# Runs image through ALL 6 CNN models → returns best match
# ─────────────────────────────────────────────────────────────
# Define endpoint to automatically guess the component type from an image
@app.post("/api/detect-component")
async def detect_component(file: UploadFile = File(...)):
    """
    Auto-detect which LHB bogie component is in the uploaded image.
    Runs inference on every trained model and returns the one with
    the highest confidence score.
    """
    # Extract file extension and generate a short unique ID for the auto-detect upload
    ext       = Path(file.filename).suffix or ".jpg"
    uid       = uuid.uuid4().hex[:8]
    fname     = f"detect_{uid}{ext}"
    save_path = UPLOAD_DIR / fname
    
    # Save the uploaded image locally
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Attempt to open and convert the image; return an HTTP 400 if it fails
    try:
        pil_img = Image.open(save_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    # Dictionary to keep track of inference scores across all models
    scores: dict = {}
    # Iterate through all registered bogie components
    for comp_key in mm.COMPONENT_REGISTRY:
        try:
            # Run the specific model for the current component
            result = mm.infer(pil_img, component=comp_key)
            # Use the normal_score as the "this is the correct component" proxy:
            # a trained model on the matching component should produce distinct patterns
            scores[comp_key] = {
                "confidence":   result["confidence"],
                "defect_score": result["defect_score"],
                "normal_score": result["normal_score"],
                "label":        mm.COMPONENT_REGISTRY[comp_key]["label"],
            }
        except Exception:
            # In case a model fails or isn't trained yet, assign zero scores
            scores[comp_key] = {
                "confidence": 0, "defect_score": 0,
                "normal_score": 0, "label": mm.COMPONENT_REGISTRY[comp_key]["label"],
            }

    # Best match = component whose model gives the LOWEST defect_score
    # (i.e. most "at home" with the image — its training distribution)
    best = min(scores.items(), key=lambda x: abs(x[1]["defect_score"] - 30))
    # Fallback: determine the model that yields the absolute highest confidence score
    best_conf = max(scores.items(), key=lambda x: x[1]["confidence"])
    # Pick the component associated with the highest confidence
    detected = best_conf[0]  # highest confidence wins

    # Return the identified component details and the scores from all models
    return JSONResponse({
        "detected_component": detected,
        "detected_label":     scores[detected]["label"],
        "confidence":         scores[detected]["confidence"],
        "all_scores":         scores,
        "image_url":          f"/uploads/{fname}",
    })



# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
# Dictionary acting as an in-memory store for tracking background training progress
_training_status: dict = {}


# Synchronous helper function to execute model training in the background
def _run_training(component: str):
    # Initialize the training state for the specific component
    _training_status[component] = {"running": True, "log": [], "done": False, "error": None}
    try:
        # Define a callback to append training logs to our status dict
        def cb(msg):
            _training_status[component]["log"].append(msg)
        # Clear out any old cached model before starting training
        mm._model_cache.pop(component, None)
        # Call the underlying manager to train the model for 30 epochs
        mm.train_model(component=component, epochs=30, progress_cb=cb)
        # Clear cache again and explicitly load the newly trained model weights
        mm._model_cache.pop(component, None)
        mm.load_model(component=component)
        # Mark training as successfully finished
        _training_status[component]["done"] = True
    except Exception as e:
        # Record any errors that happen during training
        _training_status[component]["error"] = str(e)
    finally:
        # Always mark training as no longer running, regardless of success or failure
        _training_status[component]["running"] = False


# Define endpoint to trigger model training for a specific component
@app.post("/api/train")
async def trigger_training(background_tasks: BackgroundTasks, component: str = "coupler"):
    # If training is already actively running for this component, return 409 Conflict
    if _training_status.get(component, {}).get("running"):
        return JSONResponse(
            {"message": f"Training for {component} already in progress"},
            status_code=409,
        )
    # Add the training job to FastAPI's background tasks queue
    background_tasks.add_task(_run_training, component)
    return JSONResponse({"message": f"Training started for {component}"})


# Define endpoint to poll the ongoing training status and view logs
@app.get("/api/train/status")
async def training_status(component: str = None):
    # If a specific component is requested, return its precise status
    if component:
        return JSONResponse(
            _training_status.get(
                component,
                # Default mock object if no training has ever started for this component
                {"running": False, "log": [], "done": False, "error": None},
            )
        )
    # Otherwise return the status dict for all components
    return JSONResponse(_training_status)


# ─────────────────────────────────────────────────────────────
# GET /api/models
# ─────────────────────────────────────────────────────────────
# Define endpoint to fetch information about configured AI LLM and CNN models
@app.get("/api/models")
async def get_models():
    return JSONResponse({
        # List the available free LLM models from OpenRouter
        "free_models":  ora.FREE_MODELS,
        # Get the default LLM model from env vars
        "default":      os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b:free"),
        # Retrieve the readiness/presence status of local CNN models
        "model_status": mm.get_model_status(),
    })


# If this script is run directly, start the Uvicorn ASGI server
if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app on all network interfaces (0.0.0.0) at port 8000 with auto-reload enabled
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
