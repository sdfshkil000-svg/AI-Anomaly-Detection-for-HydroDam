import io
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# HydroSafe AI - Member 4: Anomaly Detection Agent
# Public agent interface required by the project:
#     detect_anomalies(readings) -> list[dict]
# ============================================================


# -----------------------------
# Core anomaly detection agent
# -----------------------------
def _robust_z_score(values: np.ndarray, median: float, mad: float) -> np.ndarray:
    """Robust z-score using Median Absolute Deviation."""
    if mad <= 1e-12:
        std = float(np.std(values))
        if std <= 1e-12:
            return np.zeros(len(values))
        return np.abs((values - median) / std)

    # 0.6745 makes MAD comparable to standard deviation for a normal distribution.
    return np.abs(0.6745 * (values - median) / mad)


def _severity_from_score(score: float) -> str:
    if score < 0.35:
        return "NORMAL"
    if score < 0.60:
        return "WATCH"
    if score < 0.80:
        return "WARNING"
    return "CRITICAL"


def _normalise_readings(readings) -> pd.DataFrame:
    """
    Accepts:
      1) list[dict] with sensor, timestamp, value
      2) pandas DataFrame with sensor/timestamp/value
      3) dict in the form {"readings": [...]}

    A sensor column is required. Value can be named:
      value, reading, current_value
    """
    if isinstance(readings, dict):
        readings = readings.get("readings", readings)

    if isinstance(readings, pd.DataFrame):
        df = readings.copy()
    else:
        df = pd.DataFrame(readings)

    if df.empty:
        return pd.DataFrame(columns=["sensor", "timestamp", "value"])

    # Flexible field mapping for easy integration with teammate data.
    rename_map = {}
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}

    for target, aliases in {
        "sensor": ["sensor", "sensor_id", "instrument", "instrument_id", "parameter"],
        "timestamp": ["timestamp", "time", "datetime", "date"],
        "value": ["value", "reading", "current_value", "measurement"],
    }.items():
        for alias in aliases:
            if alias in lower_to_original:
                rename_map[lower_to_original[alias]] = target
                break

    df = df.rename(columns=rename_map)

    missing = [c for c in ["sensor", "value"] if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + ". Expected sensor and value."
        )

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.now()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["sensor"] = df["sensor"].astype(str)

    df = df.dropna(subset=["sensor", "value"]).copy()
    df = df.sort_values(["sensor", "timestamp"]).reset_index(drop=True)
    return df


def detect_anomalies(readings) -> list[dict]:
    """
    HydroSafe Member 4 public function.

    Detects unusual sensor readings independently for each sensor.

    Input:
        list[dict] / DataFrame / {"readings": [...]}

    Required fields:
        sensor
        value (or reading/current_value)

    Optional:
        timestamp

    Output:
        list[dict] with:
          sensor, baseline, current_value, score, severity, confidence,
          deviation, timestamp
    """
    df = _normalise_readings(readings)

    if df.empty:
        return []

    results = []

    for sensor, group in df.groupby("sensor", sort=False):
        values = group["value"].to_numpy(dtype=float)

        # Historical baseline = robust median.
        baseline = float(np.median(values))
        mad = float(np.median(np.abs(values - baseline)))

        robust_scores = _robust_z_score(values, baseline, mad)

        # Also use a rolling median so a sudden change is detected even
        # when the long-term sensor distribution is broad.
        series = pd.Series(values)
        rolling_baseline = series.shift(1).rolling(
            window=min(20, max(5, len(series) - 1)), min_periods=3
        ).median()

        latest = float(values[-1])
        latest_robust = float(robust_scores[-1])

        if pd.notna(rolling_baseline.iloc[-1]):
            local_baseline = float(rolling_baseline.iloc[-1])
            local_scale = float(
                np.median(
                    np.abs(
                        series.iloc[max(0, len(series) - 20) : -1]
                        - local_baseline
                    )
                )
            )
            if local_scale > 1e-12:
                local_z = abs(0.6745 * (latest - local_baseline) / local_scale)
            else:
                local_z = 0.0
        else:
            local_baseline = baseline
            local_z = latest_robust

        # Combine long-term and local evidence.
        anomaly_strength = max(latest_robust / 6.0, local_z / 6.0)
        score = float(np.clip(anomaly_strength, 0.0, 1.0))

        # Confidence is higher when enough historical observations exist.
        history_factor = min(1.0, len(values) / 30.0)
        confidence = float(np.clip(0.55 + 0.40 * history_factor, 0.55, 0.95))

        severity = _severity_from_score(score)

        # A gentle change in confidence for extremely strong deviations.
        if score >= 0.8:
            confidence = min(0.99, confidence + 0.03)

        timestamp = group.iloc[-1]["timestamp"]
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.now()

        results.append(
            {
                "sensor": sensor,
                "baseline": round(baseline, 4),
                "current_value": round(latest, 4),
                "score": round(score, 4),
                "severity": severity,
                "confidence": round(confidence, 4),
                "deviation": round(latest - baseline, 4),
                "timestamp": pd.Timestamp(timestamp).isoformat(),
            }
        )

    # Most important anomalies first.
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# -----------------------------
# Demo data for the Streamlit UI
# -----------------------------
@st.cache_data
def make_demo_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sensors = {
        "PZ-023": 12.0,   # piezometer pressure
        "PZ-024": 11.5,
        "SEEP-01": 4.0,   # seepage
        "DISP-07": 2.5,   # displacement
        "TEMP-03": 24.0,  # temperature
    }

    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("h"), periods=180, freq="h"
    )

    rows = []
    for sensor, base in sensors.items():
        noise = rng.normal(0, max(abs(base) * 0.015, 0.03), len(timestamps))
        trend = np.linspace(0, base * 0.015, len(timestamps))
        values = base + noise + trend

        # Inject realistic prototype anomalies into two sensors.
        if sensor == "PZ-023":
            values[-1] = base * 1.55
        if sensor == "SEEP-01":
            values[-3:] += base * 0.75

        for ts, value in zip(timestamps, values):
            rows.append({"timestamp": ts, "sensor": sensor, "value": value})

    return pd.DataFrame(rows)


def _severity_counts(results):
    counts = {"NORMAL": 0, "WATCH": 0, "WARNING": 0, "CRITICAL": 0}
    for item in results:
        counts[item["severity"]] = counts.get(item["severity"], 0) + 1
    return counts


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="HydroSafe AI — Anomaly Detection",
    page_icon="💧",
    layout="wide",
)

st.title("💧 HydroSafe AI")
st.subheader("Member 4 — AI Anomaly Detection Agent")
st.caption(
    "Prototype: detects abnormal dam-monitoring sensor behaviour from historical "
    "and latest readings. The anomaly engine is deterministic and explainable."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Upload sensor CSV",
        type=["csv"],
        help="CSV should contain sensor and value columns. Timestamp is recommended.",
    )

    st.markdown("**Accepted column names**")
    st.code(
        "sensor,timestamp,value\n"
        "PZ-023,2026-09-01 10:00,12.4\n"
        "PZ-023,2026-09-01 11:00,12.6",
        language="csv",
    )

    use_demo = st.checkbox("Use HydroSafe demo data", value=uploaded is None)

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        data = _normalise_readings(raw)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
elif use_demo:
    data = make_demo_data()
else:
    st.info("Upload a CSV or enable demo data from the sidebar.")
    st.stop()

results = detect_anomalies(data)
result_df = pd.DataFrame(results)

if result_df.empty:
    st.warning("No valid sensor readings were found.")
    st.stop()

counts = _severity_counts(results)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sensors analysed", len(results))
c2.metric("Critical", counts["CRITICAL"])
c3.metric("Warning", counts["WARNING"])
c4.metric("Watch", counts["WATCH"])

st.divider()

left, right = st.columns([1.1, 1])

with left:
    st.markdown("### Anomaly results")

    display_df = result_df[
        [
            "sensor",
            "baseline",
            "current_value",
            "score",
            "severity",
            "confidence",
            "deviation",
        ]
    ].copy()

    display_df["score"] = display_df["score"].map(lambda x: f"{x:.2f}")
    display_df["confidence"] = display_df["confidence"].map(lambda x: f"{x:.0%}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

with right:
    st.markdown("### Highest-priority finding")
    top = results[0]

    st.metric(
        "Sensor",
        top["sensor"],
        delta=f"{top['current_value'] - top['baseline']:+.3f} from baseline",
    )

    st.write(f"**Severity:** `{top['severity']}`")
    st.write(f"**Anomaly score:** `{top['score']:.2f}`")
    st.write(f"**Confidence:** `{top['confidence']:.0%}`")

    if top["severity"] in ("WARNING", "CRITICAL"):
        st.warning(
            f"{top['sensor']} shows behaviour outside its historical baseline. "
            "Verify the instrument and compare with related monitoring parameters "
            "before taking a safety decision."
        )
    elif top["severity"] == "WATCH":
        st.info(
            f"{top['sensor']} deserves continued monitoring because its latest "
            "reading is beginning to depart from its normal pattern."
        )
    else:
        st.success("No strong anomaly was detected in the latest sensor readings.")

st.markdown("### Sensor trend")

sensor_list = sorted(data["sensor"].unique().tolist())
selected_sensor = st.selectbox("Select sensor", sensor_list)

sensor_history = data[data["sensor"] == selected_sensor].copy()
sensor_history["timestamp"] = pd.to_datetime(sensor_history["timestamp"])

st.line_chart(
    sensor_history.set_index("timestamp")[["value"]],
    use_container_width=True,
)

selected_result = next(
    item for item in results if item["sensor"] == selected_sensor
)

st.json(selected_result)

st.markdown("### Unified `anomalies` output")

st.code(
    json.dumps({"anomalies": results}, indent=2),
    language="json",
)

st.caption(
    "Safety note: this prototype is decision support only. An anomaly flag should "
    "be verified against instrument condition, data quality, thresholds, and "
    "engineering judgement."
)
