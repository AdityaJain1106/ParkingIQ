"""
model.py — Traffic Impact Quantification Engine
Adapted from traffic_impact_model.py with progress_cb support for Flask.
"""

import re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

warnings.filterwarnings("ignore")

# ─── WEIGHTS ─────────────────────────────────────────────────────────────────
VIOLATION_SEVERITY = {
    "PARKING IN A MAIN ROAD": 1.00, "DOUBLE PARKING": 0.95,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 0.90,
    "PARKING NEAR ROAD CROSSING": 0.85,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 0.80,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 0.75,
    "PARKING OTHER THAN BUS STOP": 0.70,
    "NO PARKING": 0.60, "WRONG PARKING": 0.50,
    "PARKING ON FOOTPATH": 0.30, "HT V PROHIBITED": 0.25,
    "DEFECTIVE NUMBER PLATE": 0.05, "UNKNOWN": 0.40,
}
VEHICLE_SIZE = {
    "BUS (BMTC/KSRTC)": 3.0, "PRIVATE BUS": 2.8, "MAXI-CAB": 2.5,
    "TANKER": 2.4, "LGV": 2.0, "HMV": 2.0, "TIPPER": 1.8,
    "GOODS AUTO": 1.5, "PASSENGER AUTO": 1.2,
    "CAR": 1.0, "JEEP": 1.1, "MOTOR CYCLE": 0.6, "SCOOTER": 0.6,
    "UNKNOWN": 1.0,
}
PEAK_WEIGHT = {
    0:1.4,1:1.2,2:1.3,3:1.4,4:1.8,5:2.2,6:2.8,7:3.5,8:4.0,9:3.2,
    10:2.5,11:2.2,12:2.4,13:2.3,14:2.2,15:2.6,16:3.2,17:4.0,
    18:4.2,19:3.8,20:3.0,21:2.4,22:1.8,23:1.5,
}
BASE_DELAY  = 0.08
BASE_SPEED  = 0.035


def extract_primary(vt):
    types = re.findall(r'"([^"]+)"', str(vt))
    return types[0].strip().upper() if types else str(vt).strip().upper() or "UNKNOWN"


def load_and_prepare(path):
    use_cols = [
        "created_datetime",
        "violation_type",
        "latitude",
        "longitude",
        "vehicle_type",
        "junction_name",
        "police_station"
    ]

    df = pd.read_csv(
        path,
        usecols=lambda c: c in use_cols,
        low_memory=True
    )
    for col in ["created_datetime","created_dt","datetime"]:
        if col in df.columns:
            df["_dt"] = pd.to_datetime(df[col], utc=True, errors="coerce"); break
    else:
        df["_dt"] = pd.NaT

    df["hour"]        = df["_dt"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["_dt"].dt.dayofweek.fillna(0).astype(int)
    df["month"]       = df["_dt"].dt.month.fillna(1).astype(int)
    df["date"]        = df["_dt"].dt.date

    if "violation_type" in df.columns:
        df["primary_violation"] = df["violation_type"].apply(extract_primary)
    elif "primary_violation" not in df.columns:
        df["primary_violation"] = "UNKNOWN"

    df["lat"] = pd.to_numeric(df.get("latitude",  pd.Series(dtype=float)), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("longitude", pd.Series(dtype=float)), errors="coerce")
    df["lat"].fillna(df["lat"].median() if not df["lat"].isna().all() else 12.97, inplace=True)
    df["lon"].fillna(df["lon"].median() if not df["lon"].isna().all() else 77.58, inplace=True)

    if "vehicle_type"    not in df.columns: df["vehicle_type"]    = "UNKNOWN"
    if "junction_name"   not in df.columns: df["junction_name"]   = "No Junction"
    if "police_station"  not in df.columns: df["police_station"]  = "UNKNOWN"

    df["vehicle_type"]  = df["vehicle_type"].fillna("UNKNOWN").str.upper().str.strip()
    df["junction_name"] = df["junction_name"].fillna("No Junction").str.strip()
    df["police_station"]= df["police_station"].fillna("UNKNOWN").str.strip()
    return df


def compute_traffic_impact(df):
    df["severity_score"] = df["primary_violation"].map(VIOLATION_SEVERITY).fillna(0.4)
    df["vehicle_size"]   = df["vehicle_type"].map(VEHICLE_SIZE).fillna(1.0)
    df["peak_factor"]    = df["hour"].map(PEAK_WEIGHT).fillna(2.0)
    df["is_junction"]    = (df["junction_name"] != "No Junction").astype(float)

    df["delay_minutes"] = (
        BASE_DELAY * df["severity_score"] * df["vehicle_size"]
        * df["peak_factor"] * (1 + 0.5 * df["is_junction"])
    ).round(3)

    df["speed_reduction_pct"] = (
        BASE_SPEED * 100 * df["severity_score"] * df["vehicle_size"]
        * df["peak_factor"] * (1 + 0.3 * df["is_junction"])
    ).clip(0, 65).round(2)

    raw = (
        df["severity_score"] * 35 + (df["peak_factor"] / 4.2) * 25
        + df["is_junction"] * 20 + (df["vehicle_size"] / 3.0) * 20
    )
    scaler = MinMaxScaler(feature_range=(0, 100))
    df["impact_score"] = scaler.fit_transform(raw.values.reshape(-1, 1)).round(1)

    df["risk_tier"] = pd.cut(
        df["impact_score"], bins=[0,30,55,75,100],
        labels=["LOW","MEDIUM","HIGH","CRITICAL"], include_lowest=True
    )
    return df


def aggregate_junctions(df):
    jdf = df[df["junction_name"] != "No Junction"].copy()
    if len(jdf) == 0:
        df["_geo_cluster"] = df["lat"].round(3).astype(str)+"_"+df["lon"].round(3).astype(str)
        jdf = df.copy(); jdf["junction_name"] = jdf["_geo_cluster"]

    stats = jdf.groupby("junction_name").agg(
        lat=("lat","mean"), lon=("lon","mean"),
        total_violations=("impact_score","count"),
        avg_impact=("impact_score","mean"),
        max_impact=("impact_score","max"),
        total_delay_min=("delay_minutes","sum"),
        avg_delay_min=("delay_minutes","mean"),
        avg_speed_red=("speed_reduction_pct","mean"),
        max_speed_red=("speed_reduction_pct","max"),
        critical_count=("risk_tier", lambda x:(x=="CRITICAL").sum()),
        high_count=("risk_tier", lambda x:(x=="HIGH").sum()),
        peak_violations=("peak_factor", lambda x:(x>=3.0).sum()),
        unique_days=("date","nunique"),
    ).reset_index()

    stats["daily_avg_violations"] = (stats["total_violations"] / stats["unique_days"].clip(1)).round(1)

    def norm(s):
        mn,mx = s.min(),s.max()
        return pd.Series(np.zeros(len(s)),index=s.index) if mx==mn else (s-mn)/(mx-mn)

    stats["enforcement_priority_index"] = (
        norm(stats["avg_impact"])*40 + norm(stats["total_delay_min"])*25
        + norm(stats["critical_count"]+stats["high_count"])*15
        + norm(stats["daily_avg_violations"])*10 + norm(stats["peak_violations"])*10
    ).round(1)

    stats["priority_tier"] = pd.cut(
        stats["enforcement_priority_index"], bins=[0,25,50,75,100],
        labels=["ROUTINE","MODERATE","PRIORITY","URGENT"], include_lowest=True
    )
    stats["daily_hours_lost"] = (stats["total_delay_min"]/60).round(2)
    stats = stats.sort_values("enforcement_priority_index", ascending=False).reset_index(drop=True)
    stats.insert(0, "rank", stats.index+1)
    return stats


def train_impact_predictor(df, output_dir):
    if len(df) > 50000:
        df = df.sample(50000, random_state=42)
    
    le_viol = LabelEncoder(); le_veh = LabelEncoder(); le_ps = LabelEncoder()
    df["viol_enc"] = le_viol.fit_transform(df["primary_violation"])
    df["veh_enc"]  = le_veh.fit_transform(df["vehicle_type"])
    df["ps_enc"]   = le_ps.fit_transform(df["police_station"].fillna("UNKNOWN"))

    feat = ["hour","day_of_week","month","viol_enc","veh_enc","ps_enc",
            "is_junction","severity_score","vehicle_size","peak_factor"]
    X = df[feat].fillna(0).values
    y = df["impact_score"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=150, max_depth=6, learning_rate=0.08,
                                      subsample=0.8, random_state=42)
    model.fit(X_tr, y_tr)

    mae = mean_absolute_error(y_te, model.predict(X_te))
    r2  = r2_score(y_te, model.predict(X_te))

    arts = {"model": model, "encoders": {"violation":le_viol,"vehicle":le_veh,"police_station":le_ps},
            "feature_cols": feat, "mae": mae, "r2": r2}
    joblib.dump(arts, Path(output_dir) / "impact_predictor.pkl")
    return arts


def _build_single_folium_map(df_full, heat_data, junctions, c_lat, c_lon, slot_label="All time", total_all=None):
    """Build one Folium heatmap with slot-aware radius, popup stats, and top-bar banner."""
    import math

    n      = len(heat_data)
    n_all  = total_all or n

    # ── radius/blur scale with density so sparse slots still look bold ──
    if n > 50000:
        radius, blur, min_op = 14, 12, 0.4
    elif n > 10000:
        radius, blur, min_op = 18, 15, 0.5
    elif n > 1000:
        radius, blur, min_op = 22, 18, 0.55
    else:
        radius, blur, min_op = 28, 22, 0.65   # afternoon — make each point very visible

    m = folium.Map(location=[c_lat, c_lon], zoom_start=13, tiles="CartoDB dark_matter")

    bbox = heat_data[
        heat_data["lat"].between(c_lat - 0.25, c_lat + 0.25) &
        heat_data["lon"].between(c_lon - 0.25, c_lon + 0.25)
    ]
    if len(bbox) >= 10:
        heat_data = bbox

    if not heat_data.empty:
        # Normalise impact scores within THIS slot so colour range always full
        scores = heat_data["impact_score"].values.astype(float)
        mn, mx = scores.min(), scores.max()
        if mx > mn:
            scores = (scores - mn) / (mx - mn) * 100
        else:
            scores = scores * 0 + 80
        pts = [[row[0], row[1], s] for row, s in zip(heat_data[["lat","lon"]].values, scores)]

        HeatMap(
            pts,
            min_opacity=min_op,
            max_opacity=0.92,
            radius=radius,
            blur=blur,
            gradient={0.0:"#000080", 0.25:"#0000ff", 0.45:"#00ff00",
                      0.65:"#ffff00", 0.82:"#ff6600", 1.0:"#ff0000"}
        ).add_to(m)

        bounds = [
            [heat_data["lat"].min(), heat_data["lon"].min()],
            [heat_data["lat"].max(), heat_data["lon"].max()]
        ]
        m.fit_bounds(bounds)

    # ── Per-slot junction stats ──────────────────────────────────────────
    # Count violations per junction for THIS slot only
    slot_junc_counts = {}
    if "junction_name" in df_full.columns:
        sj = df_full[df_full["junction_name"] != "No Junction"]["junction_name"].value_counts()
        slot_junc_counts = sj.to_dict()

    tier_clr = {"URGENT": "red", "PRIORITY": "orange", "MODERATE": "blue", "ROUTINE": "green"}
    tier_hex  = {"URGENT":"#ff2222","PRIORITY":"#ff8800","MODERATE":"#4488ff","ROUTINE":"#22c77a"}

    for _, r in junctions.iterrows():
        clr      = tier_clr.get(str(r["priority_tier"]), "gray")
        hex_clr  = tier_hex.get(str(r["priority_tier"]), "#aaa")
        jname    = r["junction_name"]
        slot_cnt = slot_junc_counts.get(jname, 0)
        pct_slot = (slot_cnt / r["total_violations"] * 100) if r["total_violations"] > 0 else 0

        popup_html = f"""
<div style='font-family:monospace;background:#0d1117;color:#e0e0e0;
     padding:14px 16px;border-radius:8px;min-width:280px;
     border:1px solid #333;font-size:12px;line-height:1.8'>
  <div style='font-size:14px;font-weight:700;color:#ff6b35;margin-bottom:8px'>
    #{int(r["rank"])} {jname}
  </div>
  <div style='background:#161b22;border-radius:6px;padding:8px 10px;margin-bottom:8px'>
    <span style='color:{hex_clr};font-weight:700'>{r["priority_tier"]}</span>
    &nbsp;·&nbsp;
    <b style='color:#fff'>EPI {r["enforcement_priority_index"]:.1f}/100</b>
  </div>
  <table style='width:100%;border-collapse:collapse'>
    <tr><td style='color:#8b91a8'>All-time violations</td>
        <td style='text-align:right;color:#fff'><b>{int(r["total_violations"]):,}</b></td></tr>
    <tr><td style='color:#8b91a8'>This slot violations</td>
        <td style='text-align:right;color:#ff6b35'><b>{slot_cnt:,} ({pct_slot:.1f}%)</b></td></tr>
    <tr><td style='color:#8b91a8'>Daily avg violations</td>
        <td style='text-align:right;color:#fff'>{r["daily_avg_violations"]:.0f}/day</td></tr>
    <tr><td style='color:#8b91a8'>Avg delay / violation</td>
        <td style='text-align:right;color:#fff'>{r["avg_delay_min"]:.3f} min</td></tr>
    <tr><td style='color:#8b91a8'>Daily hours lost</td>
        <td style='text-align:right;color:#f5c33a'><b>{r["daily_hours_lost"]:.1f} hrs</b></td></tr>
    <tr><td style='color:#8b91a8'>Avg speed reduction</td>
        <td style='text-align:right;color:#fff'>{r["avg_speed_red"]:.1f}%</td></tr>
    <tr><td style='color:#8b91a8'>Max speed reduction</td>
        <td style='text-align:right;color:#fff'>{r["max_speed_red"]:.1f}%</td></tr>
    <tr><td style='color:#8b91a8'>Critical violations</td>
        <td style='text-align:right;color:#ff4444'><b>{int(r["critical_count"])}</b></td></tr>
    <tr><td style='color:#8b91a8'>High violations</td>
        <td style='text-align:right;color:#ff8800'>{int(r["high_count"])}</td></tr>
    <tr><td style='color:#8b91a8'>Peak-hour violations</td>
        <td style='text-align:right;color:#fff'>{int(r["peak_violations"])}</td></tr>
    <tr><td style='color:#8b91a8'>Unique active days</td>
        <td style='text-align:right;color:#fff'>{int(r["unique_days"])}</td></tr>
  </table>
  <div style='margin-top:8px;font-size:10px;color:#555;text-align:right'>
    Rank #{int(r["rank"])} of {len(junctions)} junctions
  </div>
</div>"""

        folium.Marker(
            [r["lat"], r["lon"]],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"#{int(r['rank'])} {jname} | EPI:{r['enforcement_priority_index']:.0f} | Slot:{slot_cnt:,}",
            icon=folium.Icon(color=clr, icon="map-marker", prefix="fa")
        ).add_to(m)

    # ── Stats legend + top banner ────────────────────────────────────────
    pct_of_all = (n / n_all * 100) if n_all > 0 else 100
    avg_imp    = heat_data["impact_score"].mean() if not heat_data.empty else 0

    # Top-right stats panel
    stats_panel = f"""
<div style='position:fixed;top:10px;right:10px;z-index:9999;
     background:rgba(10,10,20,0.94);padding:14px 16px;border-radius:10px;
     border:1px solid #333;color:#e0e0e0;font-family:monospace;font-size:12px;
     min-width:200px;line-height:1.9'>
  <div style='color:#ff6b35;font-weight:700;font-size:13px;margin-bottom:6px'>
    📊 {slot_label}
  </div>
  <div><span style='color:#8b91a8'>Violations</span>
       &nbsp;<b style='color:#fff;float:right'>{n:,}</b></div>
  <div><span style='color:#8b91a8'>% of all-time</span>
       &nbsp;<b style='color:#ff6b35;float:right'>{pct_of_all:.1f}%</b></div>
  <div><span style='color:#8b91a8'>Avg impact</span>
       &nbsp;<b style='color:#fff;float:right'>{avg_imp:.1f}/100</b></div>
  <hr style='border-color:#222;margin:6px 0'>
  <div style='color:#8b91a8;font-size:10px;margin-bottom:4px'>JUNCTION TIERS</div>
  <div><span style='color:#ff4444'>● URGENT</span>
       <span style='float:right;color:#888'>EPI 75–100</span></div>
  <div><span style='color:#ff8800'>● PRIORITY</span>
       <span style='float:right;color:#888'>EPI 50–75</span></div>
  <div><span style='color:#4488ff'>● MODERATE</span>
       <span style='float:right;color:#888'>EPI 25–50</span></div>
  <div><span style='color:#44cc44'>● ROUTINE</span>
       <span style='float:right;color:#888'>EPI 0–25</span></div>
  <hr style='border-color:#222;margin:6px 0'>
  <div style='color:#555;font-size:10px'>Click markers for full stats</div>
</div>"""

    m.get_root().html.add_child(folium.Element(stats_panel))
    folium.LayerControl().add_to(m)
    return m


def make_heatmap(df, junctions, output_dir):
    """Generate 5 Folium heatmaps: all-time + 4 time slots. Each visually distinct."""
    c_lat = float(df["lat"].median()) if not df["lat"].isna().all() else 12.97
    c_lon = float(df["lon"].median()) if not df["lon"].isna().all() else 77.58
    out   = Path(output_dir)

    base    = df[["lat","lon","impact_score","hour","junction_name"]].dropna(subset=["lat","lon","hour"])

if len(base) > 20000:
    base = base.sample(20000, random_state=42)
    total_n = len(base)

    slots = [
        ("congestion_heatmap.html", "All time · 24 hrs",          list(range(24))),
        ("heatmap_night.html",      "🌙 Night · 10 PM – 6 AM",    [22,23,0,1,2,3,4,5]),
        ("heatmap_morning.html",    "🌅 Morning · 6 AM – 12 PM",  list(range(6,12))),
        ("heatmap_afternoon.html",  "☀️ Afternoon · 12 PM – 5 PM", list(range(12,17))),
        ("heatmap_evening.html",    "🌆 Evening · 5 PM – 10 PM",  list(range(17,22))),
    ]

    for fname, label, hours in slots:
        subset     = base[base["hour"].isin(hours)] if hours != list(range(24)) else base
        heat_cols  = subset[["lat","lon","impact_score"]].copy()
        m = _build_single_folium_map(subset, heat_cols, junctions, c_lat, c_lon, label, total_all=total_n)
        m.save(str(out / fname))


def make_charts(df, junctions, output_dir):
    plt.style.use("dark_background")
    BG = "#0d1117"; PANEL = "#161b22"
    COLORS = ["#ff6b35","#4ecdc4","#45b7d1","#96ceb4","#ffeaa7","#fd79a8","#a29bfe","#55efc4"]

    fig = plt.figure(figsize=(22, 24), facecolor=BG)
    fig.suptitle("PARKING-INDUCED CONGESTION — INTELLIGENCE REPORT",
                 fontsize=17, fontweight="bold", color="white", y=0.98)

    # 1. Top 15 junctions EPI
    ax1 = fig.add_subplot(4, 3, (1, 2)); ax1.set_facecolor(PANEL)
    top15 = junctions.head(15)
    c_map = {"URGENT":"#ff2222","PRIORITY":"#ff8800","MODERATE":"#ffdd00","ROUTINE":"#44cc44"}
    colors = [c_map.get(str(t),"#aaa") for t in top15["priority_tier"]]
    bars = ax1.barh(range(len(top15)), top15["enforcement_priority_index"], color=colors, edgecolor="none", height=0.7)
    ax1.set_yticks(range(len(top15)))
    ax1.set_yticklabels([f"#{i+1} {n[:38]}" for i,n in enumerate(top15["junction_name"])], fontsize=8, color="#cdd9e5")
    ax1.set_xlabel("Enforcement Priority Index (EPI)", color="#8b949e")
    ax1.set_title("Enforcement Priority Index — Top 15 Junctions", color="white", fontsize=11, pad=10)
    [ax1.axvline(v, color=c_map[k], linestyle="--", alpha=0.4, lw=1) for k,v in [("URGENT",75),("PRIORITY",50),("MODERATE",25)]]
    ax1.tick_params(colors="#8b949e"); ax1.invert_yaxis()
    for bar, val in zip(bars, top15["enforcement_priority_index"]):
        ax1.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f"{val:.0f}", va="center", ha="left", fontsize=8, color="white")

    # 2. Tier donut
    ax2 = fig.add_subplot(4, 3, 3); ax2.set_facecolor(PANEL)
    tier_order = ["URGENT","PRIORITY","MODERATE","ROUTINE"]
    tier_c = ["#ff2222","#ff8800","#ffdd00","#44cc44"]
    tc = junctions["priority_tier"].value_counts()
    vals = [tc.get(t,0) for t in tier_order]
    ax2.pie(vals, labels=tier_order, autopct="%1.0f%%", colors=tier_c, startangle=90,
            wedgeprops={"edgecolor":BG,"linewidth":2}, textprops={"color":"white","fontsize":9})
    ax2.set_title("Junction Tier Distribution", color="white", fontsize=11, pad=10)

    # 3. Delay by violation type
    ax3 = fig.add_subplot(4, 3, 4); ax3.set_facecolor(PANEL)
    vd = df.groupby("primary_violation")["delay_minutes"].sum().sort_values(ascending=False).head(9)
    ax3.barh(range(len(vd)), vd.values, color=COLORS[0], edgecolor="none", height=0.65)
    ax3.set_yticks(range(len(vd))); ax3.set_yticklabels([v[:28] for v in vd.index], fontsize=7.5, color="#cdd9e5")
    ax3.set_xlabel("Total Delay (minutes)", color="#8b949e")
    ax3.set_title("Delay Burden by Violation Type", color="white", fontsize=10, pad=8)
    ax3.tick_params(colors="#8b949e"); ax3.invert_yaxis()

    # 4. Hourly pattern
    ax4 = fig.add_subplot(4, 3, 5); ax4.set_facecolor(PANEL)
    hourly = df.groupby("hour")["impact_score"].mean().reindex(range(24), fill_value=0)
    bc = ["#ff2222" if v>65 else "#ff8800" if v>45 else "#4ecdc4" for v in hourly]
    ax4.bar(range(24), hourly.values, color=bc, edgecolor="none")
    ax4.set_xticks(range(0,24,3)); ax4.set_xticklabels([f"{h:02d}h" for h in range(0,24,3)], fontsize=8, color="#cdd9e5")
    ax4.set_xlabel("Hour of Day", color="#8b949e"); ax4.set_ylabel("Avg Impact Score", color="#8b949e")
    ax4.set_title("Avg Impact Score by Hour", color="white", fontsize=10, pad=8)
    ax4.axhline(65, color="#ff2222", linestyle="--", alpha=0.5, lw=1)
    ax4.tick_params(colors="#8b949e")

    # 5. Vehicle impact
    ax5 = fig.add_subplot(4, 3, 6); ax5.set_facecolor(PANEL)
    vi = df.groupby("vehicle_type").agg(avg_impact=("impact_score","mean"), count=("impact_score","count")).reset_index()
    vi = vi[vi["count"]>100].sort_values("avg_impact", ascending=False).head(10)
    ax5.barh(range(len(vi)), vi["avg_impact"], color=COLORS[1], edgecolor="none", height=0.65)
    ax5.set_yticks(range(len(vi))); ax5.set_yticklabels(vi["vehicle_type"], fontsize=8, color="#cdd9e5")
    ax5.set_xlabel("Avg Impact Score", color="#8b949e")
    ax5.set_title("Avg Impact by Vehicle Type", color="white", fontsize=10, pad=8)
    ax5.tick_params(colors="#8b949e"); ax5.invert_yaxis()

    # 6. Speed reduction histogram
    ax6 = fig.add_subplot(4, 3, 7); ax6.set_facecolor(PANEL)
    ax6.hist(df["speed_reduction_pct"].clip(0,60), bins=40, color=COLORS[2], edgecolor=BG, alpha=0.9)
    ax6.axvline(df["speed_reduction_pct"].mean(), color="#ff6b35", linestyle="--", lw=1.5, label=f"Mean {df['speed_reduction_pct'].mean():.1f}%")
    ax6.set_xlabel("Speed Reduction (%)", color="#8b949e"); ax6.set_ylabel("Violations", color="#8b949e")
    ax6.set_title("Speed Reduction Distribution", color="white", fontsize=10, pad=8)
    ax6.legend(fontsize=8); ax6.tick_params(colors="#8b949e")

    # 7. Delay burden top 10 junctions
    ax7 = fig.add_subplot(4, 3, 8); ax7.set_facecolor(PANEL)
    t10j = junctions.head(10)
    ax7.bar(range(len(t10j)), t10j["total_delay_min"]/60, color=COLORS[3], edgecolor="none")
    ax7.set_xticks(range(len(t10j)))
    ax7.set_xticklabels([n.split(" - ")[-1][:16] for n in t10j["junction_name"]], rotation=40, ha="right", fontsize=7, color="#cdd9e5")
    ax7.set_ylabel("Delay (hours)", color="#8b949e")
    ax7.set_title("Cumulative Delay — Top 10 Junctions", color="white", fontsize=10, pad=8)
    ax7.tick_params(colors="#8b949e")

    # 8. Monthly trend
    ax8 = fig.add_subplot(4, 3, 9); ax8.set_facecolor(PANEL)
    df["month_str"] = df["_dt"].dt.to_period("M").astype(str)
    mo = df.groupby("month_str").agg(violations=("impact_score","count"), avg_impact=("impact_score","mean")).reset_index().sort_values("month_str")
    if len(mo) > 0:
        ax8b = ax8.twinx()
        ax8.bar(range(len(mo)), mo["violations"], color=COLORS[4], alpha=0.8, edgecolor="none")
        ax8b.plot(range(len(mo)), mo["avg_impact"], color=COLORS[0], marker="o", lw=2, ms=5)
        ax8.set_xticks(range(len(mo))); ax8.set_xticklabels(mo["month_str"], rotation=35, ha="right", fontsize=8, color="#cdd9e5")
        ax8.set_ylabel("Violations", color="#8b949e"); ax8b.set_ylabel("Avg Impact", color="#8b949e")
        ax8b.tick_params(colors="#8b949e")
    ax8.set_title("Monthly Violations & Impact Trend", color="white", fontsize=10, pad=8)
    ax8.tick_params(colors="#8b949e")

    # 9. Geo scatter
    ax9 = fig.add_subplot(4, 3, (10, 11)); ax9.set_facecolor(PANEL)
    samp = df.sample(min(2000, len(df)), random_state=42)
    sc = ax9.scatter(samp["lon"], samp["lat"], c=samp["impact_score"], cmap="RdYlGn_r", s=4, alpha=0.4, linewidths=0)
    for _, jr in junctions.head(15).iterrows():
        ax9.scatter(jr["lon"], jr["lat"], s=90, color="white", edgecolors="#ff6b35", linewidths=1.5, zorder=5)
        ax9.annotate(f"#{int(jr['rank'])}", (jr["lon"], jr["lat"]), fontsize=6, color="white", xytext=(3,3), textcoords="offset points")
    plt.colorbar(sc, ax=ax9, label="Impact Score", shrink=0.8)
    ax9.set_xlabel("Longitude", color="#8b949e"); ax9.set_ylabel("Latitude", color="#8b949e")
    ax9.set_title("Geo-Spatial Impact Map (top 15 junctions marked)", color="white", fontsize=10, pad=8)
    ax9.tick_params(colors="#8b949e")

    # 10. Violation × Hour heatmap
    ax10 = fig.add_subplot(4, 3, 12); ax10.set_facecolor(PANEL)
    dfj = df[df["junction_name"]!="No Junction"]
    if len(dfj) > 100:
        piv = dfj.groupby(["hour","primary_violation"])["impact_score"].mean().unstack(fill_value=0)
        top_v = dfj["primary_violation"].value_counts().head(6).index
        piv = piv[[c for c in top_v if c in piv.columns]]
        if not piv.empty:
            sns.heatmap(piv.T, ax=ax10, cmap="YlOrRd", linewidths=0.2, linecolor=BG,
                        xticklabels=[f"{h:02d}h" for h in piv.index],
                        yticklabels=[v[:22] for v in piv.columns],
                        cbar_kws={"shrink":0.8})
            ax10.tick_params(colors="#cdd9e5", labelsize=7)
            ax10.set_xlabel("Hour", color="#8b949e")
    ax10.set_title("Impact: Violation × Hour", color="white", fontsize=10, pad=8)

    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(str(Path(output_dir)/"analysis_charts.png"), dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()


def save_reports(df, junctions, output_dir):
    out = Path(output_dir)
    cols = [c for c in ["lat","lon","junction_name","vehicle_type","primary_violation",
                         "hour","day_of_week","impact_score","delay_minutes",
                         "speed_reduction_pct","risk_tier","peak_factor"] if c in df.columns]
    df[cols].to_csv(out/"violation_impact_scores.csv", index=False, encoding="utf-8")
    junctions.to_csv(out/"junction_priority_index.csv", index=False, encoding="utf-8")

    total = len(df)
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║    PARKING-INDUCED TRAFFIC IMPACT — ENFORCEMENT REPORT          ║
╚══════════════════════════════════════════════════════════════════╝

DATASET OVERVIEW
  Total violations analyzed : {total:,}
  Junctions scored          : {len(junctions)}
  URGENT junctions          : {int((junctions['priority_tier']=='URGENT').sum())}
  PRIORITY junctions        : {int((junctions['priority_tier']=='PRIORITY').sum())}

TRAFFIC IMPACT SUMMARY
  Avg delay per violation   : {df['delay_minutes'].mean():.3f} minutes
  Total delay burden        : {df['delay_minutes'].sum()/60:,.0f} hours
  Avg speed reduction       : {df['speed_reduction_pct'].mean():.1f}%
  Max speed reduction       : {df['speed_reduction_pct'].max():.1f}%
  Critical risk violations  : {(df['risk_tier']=='CRITICAL').sum():,} ({(df['risk_tier']=='CRITICAL').mean()*100:.1f}%)

TOP 10 ENFORCEMENT PRIORITY JUNCTIONS
{"─"*66}
{"#":<4} {"Junction":<42} {"EPI":>5} {"Tier":<10} {"Delay(hr)":>9}
{"─"*66}"""
    for _, row in junctions.head(10).iterrows():
        report += f"\n{int(row['rank']):<4} {row['junction_name'][:42]:<42} {row['enforcement_priority_index']:>5.1f} {str(row['priority_tier']):<10} {row['daily_hours_lost']:>9.1f}"

    report += f"""
{"─"*66}

RECOMMENDED ENFORCEMENT WINDOWS
  Morning rush  06:00–09:00  — Avg impact: {df[df['hour'].between(6,9)]['impact_score'].mean():.0f}/100
  Evening rush  17:00–19:00  — Avg impact: {df[df['hour'].between(17,19)]['impact_score'].mean():.0f}/100
  Night peak    22:00–06:00  — Avg impact: {df[df['hour'].isin(list(range(22,24))+list(range(0,6)))]['impact_score'].mean():.0f}/100

TOP VEHICLE TARGETS (by avg impact)\n"""
    vt = df.groupby("vehicle_type")["impact_score"].agg(["mean","count"]).reset_index()
    for _, row in vt[vt["count"]>100].sort_values("mean",ascending=False).head(6).iterrows():
        report += f"  {row['vehicle_type']:<30} avg impact: {row['mean']:.1f}/100  ({int(row['count']):,} violations)\n"

    with open(out/"enforcement_report.txt","w", encoding="utf-8") as f:
        f.write(report)


def run_pipeline(csv_path, output_dir, progress_cb=None):
    def p(pct, msg):
        if progress_cb: progress_cb(pct, msg)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    p(10, "Loading & parsing data…")
    df = load_and_prepare(csv_path)
    p(25, "Computing traffic impact scores…")
    df = compute_traffic_impact(df)
    p(45, "Aggregating junction statistics…")
    junctions = aggregate_junctions(df)
    p(60, "Training ML impact predictor…")
    train_impact_predictor(df, output_dir)
    p(70, "Building interactive heatmap…")
    make_heatmap(df, junctions, output_dir)
    p(83, "Generating analysis charts…")
    make_charts(df, junctions, output_dir)
    p(93, "Saving reports & CSVs…")
    save_reports(df, junctions, output_dir)
    p(100, "Done!")
    return df, junctions
