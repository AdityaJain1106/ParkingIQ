"""
ParkingIQ — Flask Backend
Handles CSV upload, runs the traffic impact pipeline, serves results.
"""

import os, re, json, uuid, threading, time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "parkingiq-secret-2024"
CORS(app)

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# In-memory job store
jobs = {}   # job_id -> {status, progress, message, result}

# ─── Import model logic ───────────────────────────────────────────────
from model import run_pipeline

# ─── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    job_id = str(uuid.uuid4())[:8]
    upload_path = UPLOAD_FOLDER / f"{job_id}_{f.filename}"
    f.save(str(upload_path))

    jobs[job_id] = {"status": "queued", "progress": 0, "message": "File uploaded", "result": None}

    # Run in background thread
    t = threading.Thread(target=run_job, args=(job_id, str(upload_path)))
    t.daemon = True
    t.start()

    return jsonify({"job_id": job_id})


def run_job(job_id: str, csv_path: str):
    out_dir = OUTPUT_FOLDER / job_id
    out_dir.mkdir(exist_ok=True)

    def progress(pct, msg):
        jobs[job_id]["progress"] = pct
        jobs[job_id]["message"]  = msg
        jobs[job_id]["status"]   = "running"

    try:
        progress(5,  "Loading and parsing CSV…")
        df, junctions = run_pipeline(
            csv_path, str(out_dir),
            progress_cb=progress
        )

        # Build JSON result for frontend
        result = build_result(df, junctions, out_dir)
        jobs[job_id]["result"]   = result
        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"]  = "Analysis complete"

    except Exception as e:
        import traceback
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = str(e)
        jobs[job_id]["error_detail"] = traceback.format_exc()


def build_result(df, junctions, out_dir):
    """Serialize dataframes to JSON-friendly dicts for the frontend."""
    import numpy as np

    def safe(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return round(float(v), 3)
        if isinstance(v, float) and (v != v): return None
        return v

    total   = len(df)
    approved = int((df.get("validation_status", df.get("risk_tier","")) == "approved").sum()) if "validation_status" in df.columns else 0

    # Metrics
    metrics = {
        "total_violations":   total,
        "avg_delay_min":      round(float(df["delay_minutes"].mean()), 3),
        "total_delay_hours":  round(float(df["delay_minutes"].sum() / 60), 1),
        "avg_speed_reduction":round(float(df["speed_reduction_pct"].mean()), 1),
        "max_speed_reduction":round(float(df["speed_reduction_pct"].max()), 1),
        "critical_count":     int((df["risk_tier"] == "CRITICAL").sum()),
        "high_count":         int((df["risk_tier"] == "HIGH").sum()),
        "junctions_scored":   len(junctions),
        "urgent_junctions":   int((junctions["priority_tier"] == "URGENT").sum()),
        "priority_junctions": int((junctions["priority_tier"] == "PRIORITY").sum()),
    }

    # Top 20 junctions
    junc_list = []
    for _, r in junctions.head(20).iterrows():
        junc_list.append({k: safe(v) for k, v in r.items()})

    # Hourly impact
    hourly = df.groupby("hour")["impact_score"].mean().reindex(range(24), fill_value=0)
    hourly_data = [round(float(v), 1) for v in hourly.values]

    # Violation type breakdown
    viol_counts = df["primary_violation"].value_counts().head(8)
    viol_data   = {"labels": list(viol_counts.index), "values": [int(v) for v in viol_counts.values]}

    # Vehicle type
    veh_counts = df["vehicle_type"].value_counts().head(8)
    veh_data   = {"labels": list(veh_counts.index), "values": [int(v) for v in veh_counts.values]}

    # Risk tier
    tier_counts = df["risk_tier"].astype(str).value_counts()
    tier_data   = {"labels": list(tier_counts.index), "values": [int(v) for v in tier_counts.values]}

    # Monthly
    df["month_str"] = df["_dt"].dt.to_period("M").astype(str)
    monthly = df.groupby("month_str").size().reset_index()
    monthly_data = {"labels": list(monthly["month_str"]), "values": [int(v) for v in monthly[0]]}

    # Geo points sample (max 3000)
    geo_sample = df[["lat","lon","impact_score"]].dropna().sample(min(3000, len(df)), random_state=42)
    geo_points  = geo_sample.values.tolist()

    # All junctions for map
    junc_map = []
    for _, r in junctions.iterrows():
        junc_map.append({
            "name":  r["junction_name"],
            "lat":   safe(r["lat"]),
            "lon":   safe(r["lon"]),
            "epi":   safe(r["enforcement_priority_index"]),
            "tier":  str(r["priority_tier"]),
            "delay": safe(r["total_delay_min"]),
            "speed": safe(r["avg_speed_red"]),
            "count": safe(r["total_violations"]),
            "rank":  safe(r["rank"]),
            "daily_hours": safe(r["daily_hours_lost"]),
            "critical": safe(r["critical_count"]),
        })

    # Files available for download
    files = {
        "violation_scores":    "violation_impact_scores.csv",
        "junction_priority":   "junction_priority_index.csv",
        "enforcement_report":  "enforcement_report.txt",
        "charts":              "analysis_charts.png",
        "heatmap":             "congestion_heatmap.html",
    }

    # Slot heatmap filenames for timeline slider
    heatmap_slots = {
        "all":       "congestion_heatmap.html",
        "night":     "heatmap_night.html",
        "morning":   "heatmap_morning.html",
        "afternoon": "heatmap_afternoon.html",
        "evening":   "heatmap_evening.html",
    }

    return {
        "metrics":       metrics,
        "junctions":     junc_list,
        "junc_map":      junc_map,
        "hourly":        hourly_data,
        "viol_data":     viol_data,
        "veh_data":      veh_data,
        "tier_data":     tier_data,
        "monthly_data":  monthly_data,
        "geo_points":    geo_points,
        "files":         files,
        "heatmap_slots": heatmap_slots,
    }


@app.route("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status":   job["status"],
        "progress": job["progress"],
        "message":  job["message"],
        "result":   job.get("result"),
    })


@app.route("/api/download/<job_id>/<filename>")
def download(job_id, filename):
    safe_name = Path(filename).name
    path = OUTPUT_FOLDER / job_id / safe_name
    if not path.exists():
        return jsonify({"error": "File not found"}), 404

    if safe_name.endswith(".html"):
        return send_file(str(path), as_attachment=False, mimetype="text/html")
    return send_file(str(path), as_attachment=False)


if __name__ == "__main__":
    print("\n🅿  ParkingIQ Server starting…")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
