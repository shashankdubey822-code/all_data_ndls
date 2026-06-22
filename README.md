# LHB Coach Maintenance Dashboard (NDLS Depot)
### AI-Powered Multi-Component Inspection & Diagnostics System for Indian Railways LHB Fiat Bogies

> [!NOTE]
> **Branding & Context**: This project is developed for the New Delhi (NDLS) Coach Care Depot, Indian Railways, powered by Counterpoint. It strictly references RDSO (Research Designs and Standards Organisation) and IRS (Indian Railway Standards) maintenance manuals to automate the detection, classification, and reporting of bogie defects.

---

## 📌 Project Overview

The **LHB Coach Maintenance Dashboard** is a state-of-the-art diagnostic system designed to automate visual inspection and compliance audits of Linke Hofmann Busch (LHB) Fiat Bogies. 

The system leverages a **hybrid AI approach**:
1. **Deep Learning (Computer Vision)**: Six local PyTorch CNN models (using a MobileNetV2 architecture) trained specifically on Indian Railways datasets classify defects for individual bogie components.
2. **Traditional Computer Vision**: OpenCV-driven image processing maps edge anomalies, rust/corrosion hotspots, and generates heatmaps to pinpoint exact physical defects.
3. **Generative AI (Large Language Models)**: Integrates with LLMs via **OpenRouter** (e.g., Gemini 1.5 Pro / GPT-4o) using component-specific RDSO system prompts. By combining the CV parameters, LLMs generate professional, standard-compliant technical justifications, reference the exact clauses of the RDSO manuals, and assign definitive maintenance actions.

---

## 🛠️ Key Features

- **6/6 Trained AI Models**: Local PyTorch models (`.pth` weights) are fully trained and integrated for the six critical components:
  - **CBC Coupler**: Identifies cracks in knuckles, wear on lock-lift assembly, and rust levels.
  - **Axle Box**: Evaluates grease leakage, axle box cover deformation, and mechanical wear.
  - **Brake Disk**: Audits thermal cracks, deep grooves, thickness wear, and bolt tightness.
  - **Damper**: Detects hydraulic oil leakage, damaged mounting bushes, and structural bends.
  - **Coil Spring**: Detects inner/outer coil breakage, cracks, and uneven coil-to-coil contact.
  - **Wheel**: Evaluates wheel flange height, thickness, qR value, wheel diameter, and shelling defects.
- **RDSO Standards Compliance**: The analysis logic dynamically references physical thresholds (e.g., flange thickness $< 22\text{ mm}$ or qR value $< 6.5\text{ mm}$ triggers an immediate **NOT FIT** status as per IRS T-27 standards).
- **Rich Dashboard UI**: Collapsible sidebar, dark-mode glassmorphic theme, responsive cards, and dynamic statistics showing total inspections, defective counts, and fit/unfit ratios.
- **Inspection History & Interactive Gallery**: Stores all runs in a local SQLite database, allowing users to search, filter by component or status, and view historical reports.
- **PDF Report Generation**: Instantly exports official Indian Railways inspection reports complete with metadata, CV stats, LLM recommendations, and timestamped signatures.

---

## 📁 Repository Structure

```
all_data_ndls_depo_fait_bogies/
│
├── README.md                                # Project documentation
├── backend/                                 # FastAPI Backend
│   ├── app.py                               # FastAPI routes & static file handlers
│   ├── database.py                          # SQLite DB init & history management
│   ├── model_manager.py                     # PyTorch model load & CV pipeline (MobileNetV2)
│   ├── openrouter_analyzer.py               # OpenRouter LLM context & prompt engineering
│   ├── requirements.txt                     # Python packages list
│   ├── railsense.db                         # Local SQLite Database file
│   └── *_model.pth                          # 6 Trained PyTorch weights (.pth)
│
├── frontend/                                # HTML/CSS/JS Frontend
│   ├── index.html                           # Dashboard structure & modals
│   ├── style.css                            # Glassmorphic dark styling & sidebar collapse logic
│   ├── app.js                               # Frontend logic, charts, API integration
│   ├── LHB_Coach_Maintenance_User_Manual.pdf # PDF User Manual
│   └── *.png / *.jpg                        # Logos and UI assets
│
└── uploads/                                 # Uploaded inspection images storage
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites
- Python 3.9+ (Python 3.10 recommended)
- CUDA-enabled GPU (optional, fallbacks to CPU automatically)

### 2. Clone and Setup Environment
Open your terminal/PowerShell in the repository directory:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (CMD):
.\venv\Scripts\activate.bat
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables
Create a file named `.env` in the `backend/` directory (or edit the existing one):
```ini
OPENROUTER_API_KEY=your_openrouter_api_key_here
PORT=8001
HOST=0.0.0.0
```
> **Note**: To retrieve LLM reports, an OpenRouter API key is required. If the key is missing or invalid, the backend will auto-fallback to rule-based mock responses, ensuring the app remains functional.

---

## 🚀 Running the Application

To start the FastAPI backend server:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8001 --reload
```

Once running:
- **Dashboard Interface**: Open your browser and navigate to `http://localhost:8001`
- **Interactive Swagger API Docs**: Navigate to `http://localhost:8001/docs`

---

## 📊 Core API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/api/models` | Checks the loading status of the 6 PyTorch models and returns filenames. |
| **GET** | `/api/stats` | Retrieves metrics for the dashboard (Total Inspections, Fit/Unfit counts). |
| **GET** | `/api/history` | Returns the list of all historical inspections stored in `railsense.db`. |
| **POST** | `/api/analyze` | Accepts an image upload + metadata, runs CV model classification + OpenCV defect isolation, triggers OpenRouter, and saves the run. |
| **GET** | `/api/pdf-report/{id}`| Generates a formatted PDF report for the given inspection record. |
| **POST** | `/api/delete/{id}` | Deletes a record from the database. |

---

## 🧠 AI Models and Computer Vision Pipeline

1. **Model Architecture**: The system utilizes **MobileNetV2** pre-trained on ImageNet and fine-tuned on custom image datasets representing bogie components.
2. **Defect Classification**: 
   - Normal vs Defective classification outputting confidence scores.
   - Outputs are routed to `model_manager.py` where a threshold checks if the confidence indicates a defect.
3. **OpenCV Post-Processing**:
   - Performs edge detection (`cv2.Canny`) and contour analysis to locate surface cracks, breaks, and oil leaks.
   - Highlights anomalies visually in the output image which is saved back to `uploads/`.

---

## 👥 Authors & License

- **Client**: Indian Railways.
- **Developer**: Counterpoint AI Systems Group.
- **License**: Proprietary for Indian Railways Internal Operations.
