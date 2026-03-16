"""
main.py — FastAPI Nutrition Recommendation API
────────────────────────────────────────────────
Start with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs      ← Swagger UI
    http://localhost:8000/redoc     ← ReDoc UI
    http://localhost:8000           ← Web frontend
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os, traceback
import csv
from datetime import datetime

from schemas import UserProfileRequest, NutritionPlanResponse
from recommender import get_recommendation, load_artifacts

# ── App setup ────────────────────────────────────────────────
# ── Lifespan handler ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-load model
    try:
        load_artifacts()
        print("ML model loaded successfully.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("   Run `python train_and_save_model.py` first.")
    yield
    # Shutdown: simple message
    print("Shutting down NutriAI API...")

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title       = "🥗 NutriAI — Premium Nutrition API",
    description = (
        "Personalised nutrition & diet recommendations for users based on "
        "their weight, eating habits, lifestyle, and health goals.\n\n"
        "Built on UCI Obesity Levels dataset with Gradient Boosting + KNN hybrid model."
    ),
    version     = "1.1.0",
    lifespan    = lifespan,
    contact     = {"name": "NutriAI Support", "email": "support@nutriai.io"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS/JS if any)
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Activity Logging ─────────────────────────────────────────
def log_api_request(endpoint: str, user_name: str = None, bmi: float = None):
    """Log API traffic to reports/api_requests.csv for analytics."""
    try:
        os.makedirs("reports", exist_ok=True)
        csv_path = os.path.join("reports", "api_requests.csv")
        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "endpoint", "user_name", "bmi"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                endpoint,
                user_name if user_name else "",
                bmi if bmi is not None else ""
            ])
    except Exception as e:
        print(f"Error logging API request: {e}")

# Startup logic moved to lifespan


# ── Routes ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Frontend"])
async def root():
    """Serve the web frontend."""
    html_path = os.path.join("templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Nutrition API is running. Visit <a href='/docs'>/docs</a></h2>")


@app.get("/health", tags=["System"])
async def health_check():
    """API health check."""
    try:
        arts = load_artifacts()
        return {
            "status"          : "healthy",
            "model_loaded"    : True,
            "model_type"      : type(arts["classifier"]).__name__,
            "training_samples": int(arts["X_scaled"].shape[0]),
        }
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "model_loaded": False})


@app.post(
    "/recommend",
    response_model=NutritionPlanResponse,
    tags=["Recommendation"],
    summary="Get personalised nutrition recommendation",
    response_description="Full nutrition plan with meal suggestions, alerts, and similar user profiles",
)
async def recommend(user: UserProfileRequest):
    """
    Submit your profile and receive a fully personalised nutrition plan including:

    - **BMI classification** with confidence score
    - **Daily calorie target & macro split**
    - **4-meal daily meal plan** (adjusted for diet preference)
    - **Foods to increase / avoid**
    - **Supplement recommendations**
    - **Personalised alerts** based on your habits
    - **Similar users** from collaborative filtering
    - **Weight goals** — how much to lose/gain for healthy BMI
    """
    try:
        log_api_request(endpoint="/recommend", user_name=user.name)
        result = get_recommendation(user)
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not trained yet. Run `python train_and_save_model.py` first. Error: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {traceback.format_exc()}")


@app.get("/bmi", tags=["Utilities"])
async def calculate_bmi(height_cm: float, weight_kg: float):
    """
    Quick BMI calculator. Returns BMI value, category, and weight targets.
    """
    if height_cm <= 0 or weight_kg <= 0:
        raise HTTPException(status_code=422, detail="Height and weight must be positive")
    h    = height_cm / 100
    bmi  = round(weight_kg / (h ** 2), 2)
    from nutrition_plans import BMI_THRESHOLDS, NUTRITION_PLANS
    category = "Unknown"
    for key, lo, hi in BMI_THRESHOLDS:
        if lo <= bmi < hi:
            category = key.replace("_", " ")
            break
    healthy_min = round(18.5 * h ** 2, 1)
    healthy_max = round(24.9 * h ** 2, 1)
    
    log_api_request(endpoint="/bmi", bmi=bmi)
    
    return {
        "height_cm"           : height_cm,
        "weight_kg"           : weight_kg,
        "bmi"                 : bmi,
        "category"            : category,
        "healthy_weight_range": f"{healthy_min}–{healthy_max} kg",
        "weight_to_lose_kg"   : max(0, round(weight_kg - healthy_max, 1)),
        "weight_to_gain_kg"   : max(0, round(healthy_min - weight_kg, 1)),
    }


@app.get("/plans", tags=["Utilities"])
async def list_plans():
    """List all available nutrition plan categories."""
    from nutrition_plans import NUTRITION_PLANS, BMI_THRESHOLDS
    return [
        {"key": key, "bmi_range": NUTRITION_PLANS[key]["bmi_range"],
         "calorie_target": NUTRITION_PLANS[key]["calorie_target"],
         "priority": NUTRITION_PLANS[key]["priority"]}
        for key, _, _ in BMI_THRESHOLDS
        if key in NUTRITION_PLANS
    ]


@app.get("/diets", tags=["Utilities"])
async def list_diet_options():
    """List all supported diet preferences."""
    from nutrition_plans import DIET_OVERRIDES
    return list(DIET_OVERRIDES.keys())


@app.get("/allergies", tags=["Utilities"])
async def list_allergies():
    """List all supported allergy types."""
    from nutrition_plans import ALLERGY_RESTRICTIONS
    return list(ALLERGY_RESTRICTIONS.keys())
