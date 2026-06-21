# ParkingIQ — Traffic Impact Intelligence Web App

AI-driven parking enforcement intelligence platform. Upload a parking violations CSV to get:

- **Traffic Impact Score** — delay (minutes) & speed reduction (%) per violation
- **Dynamic Congestion Heatmap** — interactive Folium map weighted by impact
- **Enforcement Prioritization Index (EPI)** — 0–100 junction ranking
- **ML Model** — GradientBoosting predictor, reusable on any new city's data

---

## Setup (5 minutes)

### 1. Install Python 3.9+
Download from https://python.org/downloads

### 2. Open terminal in this folder
Right-click the `parkingiq` folder → Open in Terminal

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

---

## How to use

1. Open `http://localhost:5000`
2. Drag & drop your CSV file (or click to browse)
3. The model runs automatically — watch the progress bar
4. Explore:
   - **Dashboard** — 5 charts: hourly pattern, violation types, vehicle types, risk tiers, monthly trend
   - **Heatmap** — interactive map, click any junction for full stats
   - **Priority Index** — ranked enforcement table with EPI scores
5. Download any output: CSVs, charts PNG, heatmap HTML, report TXT

---

## CSV column requirements

| Column | Required | Example |
|---|---|---|
| `violation_type` | Yes | `["WRONG PARKING"]` |
| `vehicle_type` | Yes | `CAR`, `SCOOTER` |
| `latitude` | Yes | `12.9716` |
| `longitude` | Yes | `77.5946` |
| `created_datetime` | Yes | `2024-01-15T05:30:00` |
| `junction_name` | Optional | `BTP051 - Safina Plaza Junction` |
| `police_station` | Optional | `Upparpet` |

Missing optional columns are handled gracefully.

---

## Output files (auto-downloaded from UI)

| File | Description |
|---|---|
| `violation_impact_scores.csv` | Per-violation: impact score, delay_minutes, speed_reduction_pct, risk_tier |
| `junction_priority_index.csv` | Per-junction: EPI score, daily_hours_lost, priority_tier, avg_speed_red |
| `enforcement_report.txt` | Full text report with top zones, timing, vehicle targets |
| `analysis_charts.png` | 10-panel static chart dashboard |
| `congestion_heatmap.html` | Interactive Folium map — open in any browser |
| `impact_predictor.pkl` | Trained ML model — reuse with predict_new_data() |

---

## Project structure

```
parkingiq/
├── app.py              ← Flask server (routes, job queue)
├── model.py            ← ML pipeline (impact scores, EPI, charts, heatmap)
├── requirements.txt
├── templates/
│   └── index.html      ← Single-page UI
├── static/
│   ├── css/style.css
│   └── js/main.js
├── uploads/            ← Temporary CSV storage (auto-created)
└── outputs/            ← Analysis results (auto-created)
```

---

## EPI Formula

```
EPI = 40% × Avg Impact Score
    + 25% × Total Delay Burden
    + 15% × Critical/High Violation Share
    + 10% × Daily Recurrence
    + 10% × Peak-Hour Concentration
```

## Traffic Impact Formula

```
delay_minutes = 0.08 × severity × vehicle_size × peak_factor × (1 + 0.5 × is_junction)
speed_reduction% = 3.5% × severity × vehicle_size × peak_factor × (1 + 0.3 × is_junction)
impact_score (0-100) = 35% severity + 25% peak + 20% junction + 20% vehicle_size
```
