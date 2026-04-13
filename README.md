# Carbon Footprint AI Prediction System 🌍🌱

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-REST_API-lightgrey?style=flat-square&logo=flask)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-blue?style=flat-square)
![Chart.js](https://img.shields.io/badge/Chart.js-Data_Visualization-pink?style=flat-square&logo=chartdotjs)

An end-to-end Machine Learning web application designed to predict an individual's carbon footprint based on daily lifestyle and behavioral choices. It provides detailed category breakdown charts, compares the user against national and global averages, and leverages an AI-driven smart suggestion engine for actionable emission reduction.

---

## 🏗️ System Architecture

The project is built on a decoupled architecture, ensuring scalability, maintainability, and clear separation of concerns between the user interface and the machine learning model serving.

1. **Frontend**: Pure HTML5, CSS3 (with CSS variables for dynamic Dark/Light themes), and Vanilla JavaScript. Utilizes **Chart.js** for rendering interactive doughnut charts depicting category-wise emissions. Connectivity is handled via native `fetch` utilizing CORS.
2. **Backend**: A **Flask**-based RESTful API that handles input validation, orchestrates model inference, calculates deterministic breakdowns, and structures response payloads.
3. **Machine Learning Pipeline**: Trained on synthetically generated, realistically distributed datasets mirroring granular behavioral data. Utilizes **XGBoost Regression** coupled with an exhaustive Scikit-Learn preprocessing pipeline.

---

## 🧠 Machine Learning Engine & Pipeline

### 1. Data Generation (`backend/dataset.py`)
To bypass authenticated API limits without sacrificing structural integrity, the project dynamically generates a large synthetic dataset (5,000+ samples). 
- Generates 10+ core features covering Electricity usage, Transportation (Vehicle type, mileage, flights), Diet (Vegan to Non-vegetarian), Waste generation, Consumption (Grocery spend), Heating sources, and Digital footprint.
- Real-world proxies and emission factors are applied (e.g., specific kgCO2/kWh, flight haul tiering, dietary emission baselines) to construct the target variable: `total_footprint_tco2e`.
- Gaussian noise (`std=0.15`) is injected to simulate real-world variance and prevent model overfitting on deterministic rules.

### 2. Preprocessing & Feature Engineering (`backend/model.py`)
The pipeline securely handles real-world imperfect data using `sklearn.compose.ColumnTransformer`:
- **Numerical Features**: Imputed using `median` strategy and scaled via `StandardScaler` to ensure stability for distance-based and gradient-based algorithms.
- **Categorical Features**: Imputed using `most_frequent` and strictly One-Hot Encoded with `handle_unknown='ignore'` to prevent crashes on out-of-vocabulary inputs dynamically passed from the UI.

### 3. Model Selection & Evaluation
The training script systematically evaluations multiple regressors using cross-validation:
1. **Linear Regression** (Baseline reference)
2. **Random Forest Regressor** (Handling non-linearities and interactions)
3. **XGBoost Regressor** (Gradient boosted trees optimized for tabular regression)

*The pipeline automatically evaluates models based on **$R^2$ Score**, **Mean Absolute Error (MAE)**, and **Root Mean Squared Error (RMSE)**. The best-performing pipeline is serialized via `joblib` into the `model/` directory.*

---

## ⚙️ Backend API & Core Utilities (`backend/utils.py`, `backend/app.py`)

### `POST /predict`
Accepts a JSON payload containing user behavior metrics, processes it, and returns the inference alongside rich contextual data.

#### Deterministic Breakdown Generation
While the ML model predicts the holistic total emissions representing complex interactions, `compute_breakdown()` parses the incoming data against known environmental factors to split the footprint into logical categories: `Transport`, `Electricity`, `Diet`, `Goods`, `Waste`, and `Digital`.

#### Smart Suggestions Engine
Instead of generic "go green" tips, `generate_suggestions(breakdown)` determines actionable remediation paths:
- Dynamically sorts the user's computed breakdown by severity.
- Targets only the Top 3 highest emitting categories.
- Cross-references a multi-dimensional recommendation dictionary assessing base *actions*, *CO2 savings ratio potential*, and user *difficulty*.
- Ranks the tailored suggestions dynamically based on projected `tCO2e` offset.

#### Standardized Percentile & Global Grading
Applies standard deviation heuristic mapping alongside strict baselines:
- **India Average Baseline**: ~1.9 tCO2e/yr
- **World Average Baseline**: ~4.7 tCO2e/yr
- Calculates an interpretable A to F letter grade and a relative comparative percentile score.

---

## 🛠️ Setup & Installation Instructions

### Prerequisites
- Python 3.10+
- Modern Web Browser

### 1. Environment Initialization
Clone the repository and install all required pip packages.
```bash
git clone <repository_url>
cd "Carbon Footprint AI Prediction System"
pip install -r requirements.txt
```

### 2. Dataset Generation
Run the dataset synthetic generator to create `data/carbon_data.csv`.
```bash
python backend/dataset.py
```

### 3. Model Training Pipeline
Train the respective models and serialize the best performing pipeline objects.
```bash
python backend/model.py
```
*Outputs generated: `metrics.json` (benchmark stats) and `model/carbon_model.pkl` / `model/preprocessor.pkl`.*

### 4. Bootstrapping the Backend Server
Start the Flask application.
```bash
python backend/app.py
```
*API will default to binding to `http://localhost:5000`.*

### 5. Launch the Frontend
Access the UI safely using any modern web browser. Simply open `frontend/index.html`. Alternatively, serve it locally using:
```bash
python -m http.server 8000 --directory frontend
```

---

## 📁 Repository Structure

```text
├── backend/
│   ├── app.py          # Flask REST API Controller logic & routing
│   ├── dataset.py      # Synthetic data generator mapped to realistic footprints
│   ├── model.py        # ML training, preprocessing, cross-validation & serialization
│   └── utils.py        # Core logic: Inference wrapper, breakdowns, suggestions engine
├── data/               # Holds the dynamically generated synthetic CSV dataset
├── frontend/
│   ├── index.html      # UI Structure and layout 
│   ├── style.css       # Responsive, aesthetic CSS with CSS variables
│   └── app.js          # App logic, asynchronous fetch APIs, Chart.js implementations
├── model/              # Serialized `.pkl` models from Scikit-learn Pipeline
├── requirements.txt    # Python dependencies map
└── README.md           # Technical Documentation
```

---

## 🚀 Deployment (Production)

The frontend and backend must be independently deployed for optimal micro-services scale mapping.

1. **Backend API (e.g., Render Web Service, Heroku)**:
   - Configure the root directory for build step: `pip install -r requirements.txt`
   - Start command: `gunicorn backend.app:app`
   - Ensure CORS origins in `app.py` restrict to the production frontend domain instead of `*`.

2. **Frontend UI (e.g., Vercel, Netlify, Render Static Site)**:
   - Deploy the `frontend/` directory as a static site payload.
   - **Crucial Update**: Navigate into `app.js` and alter the `fetch()` route to target the live backend `https://<your-backend-url>/predict` rather than `http://localhost:5000`.
# carbon-footprint-ai
