import sqlite3
from datetime import timedelta

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

st.set_page_config(page_title="Malaria Early Warning", layout="wide")

DB_PATH = "malaria_logs.db"

# ──────────────────────────────────────────────────────────────
# CONFIG: every threshold lives here so it can be defended/changed
# ──────────────────────────────────────────────────────────────
LAG_DAYS = 14            # rainfall accumulation window (breeding-site lag)
OUTLOOK_DAYS = 3         # forecast days added to the window
RAIN_MOD_MM = 40         # PILOT DEFAULT, calibrate against local case data
RAIN_HIGH_MM = 70        # PILOT DEFAULT, calibrate against local case data
HUMIDITY_HIGH = 80       # % mean relative humidity
TEMP_MIN, TEMP_MAX = 18, 34  # °C range suitable for transmission

RISK_ORDER = {"High": 0, "Moderate": 1, "Low": 2}
RISK_RGBA = {"High": [220, 38, 38, 180], "Moderate": [245, 158, 11, 180], "Low": [22, 163, 74, 180]}
RISK_HEX = {"High": "#dc2626", "Moderate": "#f59e0b", "Low": "#16a34a"}

# Demand multiplier applied to baseline consumption, per zone, by risk level
DEMAND_MULT = {"Low": 1.0, "Moderate": 1.4, "High": 2.2}
# Baseline daily consumption PER ZONE (placeholders: replace with HMIS/LMIS data)
BASELINE_DAILY = {"ACT (courses)": 45, "RDT kits": 120, "IV Artesunate (vials)": 3}
DEFAULT_STOCK = {"ACT (courses)": 8000, "RDT kits": 24000, "IV Artesunate (vials)": 600}

# Zones spread across Oyo State (approximate coordinates, verify before the demo)
ZONES = {
    "Ibadan": {"lat": 7.3775, "lon": 3.9470},
    "Oyo": {"lat": 7.8500, "lon": 3.9333},
    "Ogbomoso": {"lat": 8.1335, "lon": 4.2400},
    "Iseyin": {"lat": 7.9667, "lon": 3.6000},
    "Saki": {"lat": 8.6667, "lon": 3.4000},
    "Igbo-Ora": {"lat": 7.4333, "lon": 3.2833},
}

ACTIONS = {
    "High": [
        "Within 48h: clear drains and gutters, larvicide stagnant water in the zone.",
        "Distribute LLINs / verify net coverage in the highest-burden wards.",
        "Pre-position ACT and RDT stock at primary health centres.",
        "Alert PHC officers to expect a case surge; extend clinic hours.",
        "Broadcast community messaging (radio, WhatsApp groups, town criers).",
    ],
    "Moderate": [
        "Run a stagnant-water survey and schedule source reduction this week.",
        "Check ACT/RDT stock against the resupply lead time (see supply tracker).",
        "Issue a public advisory: sleep under nets, drain standing water.",
        "Re-assess in 72h when the new forecast arrives.",
    ],
    "Low": [
        "Routine surveillance and reporting continue.",
        "Use the window to restock and service sprayers/nets.",
        "Keep community messaging going at low intensity.",
    ],
}


# ──────────────────────────────────────────────────────────────
# One risk function used EVERYWHERE (sidebar, table, map, alerts)
# ──────────────────────────────────────────────────────────────
def assess_risk(rain_mm, humidity, temp_c):
    temp_ok = TEMP_MIN <= temp_c <= TEMP_MAX
    if rain_mm >= RAIN_HIGH_MM and humidity >= HUMIDITY_HIGH and temp_ok:
        return "High"
    if rain_mm >= RAIN_MOD_MM and temp_ok:
        return "Moderate"
    return "Low"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS patient_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, age INTEGER,
            zone TEXT, rdt_result TEXT, treatment TEXT, officer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS risk_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, zone TEXT, rain_window_mm REAL,
            humidity REAL, temp_c REAL, risk TEXT, simulated INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.commit()
    conn.close()


init_db()
def migrate_risk_logs():
    conn = sqlite3.connect(DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(risk_logs)")]
    if cols and "zone" not in cols:
        conn.execute("DROP TABLE risk_logs")
        conn.commit()
    conn.close()


init_db()
migrate_risk_logs()
init_db()
st.title("🦟 Malaria Early Warning & Prevention System")
st.markdown("**Forecast → Surveillance → Supply chain**, in one dashboard for community health intervention.")

# ──────────────────────────────────────────────────────────────
# Sidebar: simulation (uses the SAME assess_risk as the table)
# ──────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Live Risk Simulation")
st.sidebar.markdown("Override live weather to test the warning engine.")
sim_mode = st.sidebar.checkbox("🚀 Enable Demo Simulation Mode", value=True)
sim_rain = st.sidebar.slider(f"Rainfall over {LAG_DAYS}-day window (mm)", 0, 150, 65)
sim_hum = st.sidebar.slider("Humidity (%)", 20, 100, 85)
sim_temp = st.sidebar.slider("Temperature (°C)", 15, 40, 27)

sim_level = assess_risk(sim_rain, sim_hum, sim_temp)
st.sidebar.subheader("Simulation Result:")
if sim_level == "High":
    st.sidebar.error("🚨 Status: HIGH RISK TRIGGERED")
elif sim_level == "Moderate":
    st.sidebar.warning("⚠️ Status: Moderate Risk")
else:
    st.sidebar.success("✅ Status: Low Risk")


# ──────────────────────────────────────────────────────────────
# 1. Data: 4 weeks of history + 7-day forecast, one API call
# ──────────────────────────────────────────────────────────────
TODAY = pd.Timestamp.now(tz="Africa/Lagos").tz_localize(None).normalize()


def synthetic_hourly():
    """Placeholder data used only if the API is unreachable (clearly flagged in the UI)."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(TODAY - timedelta(days=28), periods=(28 + 7) * 24, freq="h")
    frames = []
    for name in ZONES:
        frames.append(
            pd.DataFrame(
                {
                    "time": idx,
                    "precip": rng.gamma(0.3, 1.2, len(idx)),
                    "rh": rng.normal(75, 8, len(idx)).clip(40, 100),
                    "temp": rng.normal(27, 3, len(idx)),
                    "Zone": name,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=3600, show_spinner="Fetching forecast…")
def load_hourly():
    names = list(ZONES)
    params = {
        "latitude": ",".join(str(ZONES[n]["lat"]) for n in names),
        "longitude": ",".join(str(ZONES[n]["lon"]) for n in names),
        "hourly": "precipitation,relative_humidity_2m,temperature_2m",
        "past_days": 28,
        "forecast_days": 7,
        "timezone": "Africa/Lagos",
    }
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict):
            payload = [payload]
        frames = []
        for name, p in zip(names, payload):
            h = pd.DataFrame(p["hourly"])
            h["Zone"] = name
            frames.append(h)
        df = pd.concat(frames, ignore_index=True)
        df["time"] = pd.to_datetime(df["time"])
        df = df.rename(
            columns={"precipitation": "precip", "relative_humidity_2m": "rh", "temperature_2m": "temp"}
        )
        df["precip"] = df["precip"].fillna(0)
        df[["rh", "temp"]] = df[["rh", "temp"]].ffill().bfill()
        return df, True
    except Exception:
        return synthetic_hourly(), False


hourly, is_live = load_hourly()
if not is_live:
    st.warning("⚠️ Weather API unreachable: showing synthetic placeholder data, not real conditions.")

daily = (
    hourly.assign(Date=hourly["time"].dt.normalize())
    .groupby(["Zone", "Date"])
    .agg(Rain_mm=("precip", "sum"), Temp_C=("temp", "mean"), Humidity=("rh", "mean"))
    .reset_index()
)
daily["Rain_14d_mm"] = daily.groupby("Zone")["Rain_mm"].transform(lambda s: s.rolling(LAG_DAYS).sum())

rows = []
for zone, g in daily.groupby("Zone"):
    past = g[(g["Date"] <= TODAY) & (g["Date"] > TODAY - timedelta(days=LAG_DAYS))]
    nxt = g[(g["Date"] > TODAY) & (g["Date"] <= TODAY + timedelta(days=OUTLOOK_DAYS))]
    window = pd.concat([past, nxt])
    rows.append(
        {
            "Zone": zone,
            f"Rain past {LAG_DAYS}d (mm)": round(past["Rain_mm"].sum(), 1),
            f"Rain next {OUTLOOK_DAYS}d (mm)": round(nxt["Rain_mm"].sum(), 1),
            "Rainfall_mm": round(window["Rain_mm"].sum(), 1),
            "Humidity": round(window["Humidity"].mean(), 0),
            "Temps_C": round(window["Temp_C"].mean(), 1),
        }
    )
df_weather = pd.DataFrame(rows)

if sim_mode:
    st.info("🎛️ **Simulation mode ON.** Sidebar values override live weather in every zone. Untick it to see real data.")
    df_weather["Rainfall_mm"] = float(sim_rain)
    df_weather["Humidity"] = float(sim_hum)
    df_weather["Temps_C"] = float(sim_temp)

df_weather["Risk"] = df_weather.apply(
    lambda r: assess_risk(r["Rainfall_mm"], r["Humidity"], r["Temps_C"]), axis=1
)
df_weather = df_weather.sort_values("Risk", key=lambda s: s.map(RISK_ORDER)).reset_index(drop=True)

# KPI bar
c1, c2, c3 = st.columns(3)
c1.metric("🚨 High Risk Zones", int((df_weather["Risk"] == "High").sum()))
c2.metric("⚠️ Moderate Risk Zones", int((df_weather["Risk"] == "Moderate").sum()))
c3.metric("✅ Low Risk Zones", int((df_weather["Risk"] == "Low").sum()))
st.write("---")

# ──────────────────────────────────────────────────────────────
# 2. Table
# ──────────────────────────────────────────────────────────────
st.subheader("Regional Weather Forecast & Risk Status")
show = df_weather.rename(
    columns={"Rainfall_mm": f"Window rain (mm)", "Humidity": "Humidity (%)", "Temps_C": "Temp (°C)"}
).copy()
show.index = range(1, len(show) + 1)
st.dataframe(show, use_container_width=True)
st.caption(
    f"Window rain = past {LAG_DAYS} days + next {OUTLOOK_DAYS}-day forecast. "
    f"Source: Open-Meteo, {len(ZONES)} zones across Oyo State."
)

# ──────────────────────────────────────────────────────────────
# 3. Dispatch alerts & next actions (never empty)
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# 3. 7-day risk calendar + alert dispatch preview
#    (replaces the old "Dispatch alerts & next actions" section)
# ──────────────────────────────────────────────────────────────
st.subheader("🗓️ 7-Day Risk Calendar")
st.caption(
    "Same rule as the table above, recomputed for each day "
    f"({LAG_DAYS}-day rainfall + {OUTLOOK_DAYS}-day outlook). "
    "Days late in the week have less forecast data behind them, so treat them as indicative."
)


def window_stats(g, d):
    w = g[(g["Date"] > d - timedelta(days=LAG_DAYS)) & (g["Date"] <= d + timedelta(days=OUTLOOK_DAYS))]
    return w["Rain_mm"].sum(), w["Humidity"].mean(), w["Temp_C"].mean()


cal_rows = []
for zone, g in daily.groupby("Zone"):
    base_rain = window_stats(g, TODAY)[0]
    for i in range(7):
        d = TODAY + timedelta(days=i)
        rain, hum, temp = window_stats(g, d)
        if sim_mode:
            # keep each zone's day-to-day shape, anchored to the slider value for today
            rain = max(0.0, rain + (sim_rain - base_rain))
            hum, temp = sim_hum, sim_temp
        cal_rows.append(
            {
                "Zone": zone,
                "Date": d,
                "Day": d.strftime("%a %d %b"),
                "Rain_mm": round(rain, 1),
                "Risk": assess_risk(rain, hum, temp),
            }
        )
cal = pd.DataFrame(cal_rows)
day_order = [(TODAY + timedelta(days=i)).strftime("%a %d %b") for i in range(7)]
zone_order = list(df_weather["Zone"])

base = alt.Chart(cal).encode(
    x=alt.X("Day:O", sort=day_order, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
    y=alt.Y("Zone:N", sort=zone_order, title=None),
)
heat = base.mark_rect(stroke="#0e1117", strokeWidth=2).encode(
    color=alt.Color(
        "Risk:N",
        scale=alt.Scale(
            domain=["Low", "Moderate", "High"],
            range=[RISK_HEX["Low"], RISK_HEX["Moderate"], RISK_HEX["High"]],
        ),
        legend=alt.Legend(title="Risk"),
    ),
    tooltip=["Zone", "Day", alt.Tooltip("Rain_mm:Q", title="Window rain (mm)"), "Risk"],
)
labels = base.mark_text(color="white", fontSize=12).encode(text=alt.Text("Rain_mm:Q", format=".0f"))
st.altair_chart((heat + labels).properties(height=45 * len(zone_order) + 30), use_container_width=True)
st.caption("Numbers in cells = window rainfall (mm).")

# zones that are not High today but are forecast to turn High this week
outlook = {}
for zone in zone_order:
    zc = cal[cal["Zone"] == zone].sort_values("Date")
    later_high = zc[(zc["Risk"] == "High") & (zc["Date"] > TODAY)]
    if zc.iloc[0]["Risk"] != "High" and not later_high.empty:
        outlook[zone] = later_high.iloc[0]["Day"]
if outlook:
    st.warning(
        "📈 **Escalation forecast:** "
        + "; ".join(f"{z} reaches HIGH on {d}" for z, d in outlook.items())
        + ". Act at least 48h ahead."
    )


# ── Alert dispatch preview ────────────────────────────────────
def build_message(row):
    z, rain, hum = row["Zone"], row["Rainfall_mm"], row["Humidity"]
    if row["Risk"] == "High":
        return (
            f"MALARIA ALERT - {z}: HIGH risk. {rain:.0f} mm rain in the last {LAG_DAYS} days + forecast, "
            f"humidity {hum:.0f}%. Within 48h: clear drains and larvicide standing water, "
            "distribute LLINs, pre-position ACT/RDT at PHCs. Reply ACK to confirm."
        )
    msg = (
        f"MALARIA ADVISORY - {z}: MODERATE risk ({rain:.0f} mm). Survey stagnant water this week, "
        "check ACT/RDT stock and advise net use. Re-check in 72h."
    )
    if z in outlook:
        msg += f" Forecast to reach HIGH on {outlook[z]}."
    return msg


def log_alert(zone, risk, message):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS alert_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, zone TEXT, risk TEXT, message TEXT,
            status TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        "INSERT INTO alert_log (zone, risk, message, status) VALUES (?, ?, ?, ?)",
        (zone, risk, message, "Queued (demo)"),
    )
    conn.commit()
    conn.close()


st.subheader("📣 Actionable Dispatch Alerts & Next Recommended Actions")
actionable = df_weather[df_weather["Risk"] != "Low"]
if st.button("📣 Dispatch alerts to all High/Moderate zones", disabled=actionable.empty):
    for _, r in actionable.iterrows():
        log_alert(r["Zone"], r["Risk"], build_message(r))
    st.success(f"{len(actionable)} alerts queued for ward health officers.")
st.caption(
    "Demo mode: alerts are logged, not sent. An SMS/WhatsApp gateway (e.g. Termii, Twilio) "
    "plugs in at `log_alert()`. Messages are editable before sending."
)

for _, row in df_weather.iterrows():
    z = row["Zone"]
    icon = {"High": "🚨", "Moderate": "⚠️", "Low": "✅"}[row["Risk"]]
    label = f"{icon} {z}: {row['Risk']} risk ({row['Rainfall_mm']} mm, {row['Humidity']:.0f}% RH)"
    with st.expander(label, expanded=(row["Risk"] == "High")):
        st.markdown("**Recommended actions**")
        for a in ACTIONS[row["Risk"]]:
            st.markdown(f"- {a}")
        if row["Risk"] != "Low":
            st.markdown(f"**Alert to Ward Health Officer, {z}**")
            edited = st.text_area(
                "Message", build_message(row), height=110, key=f"msg_{z}", label_visibility="collapsed"
            )
            if st.button(f"Send alert to {z}", key=f"send_{z}"):
                log_alert(z, row["Risk"], edited)
                st.success(f"✅ Alert queued for Ward Health Officer, {z}.")

with st.expander("🗂️ Alert dispatch log"):
    try:
        conn = sqlite3.connect(DB_PATH)
        alerts = pd.read_sql("SELECT * FROM alert_log ORDER BY id DESC", conn)
        conn.close()
        if alerts.empty:
            st.info("No alerts dispatched yet.")
        else:
            st.dataframe(alerts, use_container_width=True, hide_index=True)
    except Exception:
        st.info("No alerts dispatched yet.")

# ──────────────────────────────────────────────────────────────
# 4. Map with legend and zone names
# ──────────────────────────────────────────────────────────────
st.subheader("🗺️ Regional Risk Spatial View")
map_df = df_weather[["Zone", "Rainfall_mm", "Humidity", "Risk"]].copy()
map_df["lat"] = map_df["Zone"].map(lambda z: ZONES[z]["lat"])
map_df["lon"] = map_df["Zone"].map(lambda z: ZONES[z]["lon"])
map_df["color"] = map_df["Risk"].map(RISK_RGBA)

st.markdown(
    " &nbsp;&nbsp; ".join(
        f"<span style='color:{RISK_HEX[k]};font-size:20px'>●</span> {k} risk" for k in ["High", "Moderate", "Low"]
    ),
    unsafe_allow_html=True,
)
st.pydeck_chart(
    pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(latitude=7.95, longitude=3.8, zoom=7.2, pitch=0),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=9000,
                pickable=True,
                auto_highlight=True,
            ),
            pdk.Layer(
                "TextLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_text="Zone",
                get_size=14,
                get_color=[30, 30, 30, 255],
                get_pixel_offset=[0, -22],
            ),
        ],
        tooltip={"text": "{Zone}\nWindow rain: {Rainfall_mm} mm\nHumidity: {Humidity}%\nRisk: {Risk}"},
    ),
    use_container_width=True,
)

# ──────────────────────────────────────────────────────────────
# 5. Real trend: 4 weeks history + forecast
# ──────────────────────────────────────────────────────────────
st.subheader("📈 Rainfall Trend: Past 4 Weeks + 7-Day Forecast")
st.markdown(
    "Rainfall drives mosquito breeding **with a lag**, so we track the rolling "
    f"{LAG_DAYS}-day total against the risk thresholds. Right of the dashed line = forecast."
)
zones_pick = st.multiselect("Zones", list(ZONES), default=list(ZONES))
trend = daily[daily["Zone"].isin(zones_pick)]


def trend_chart(data, y, ytitle, hlines=None):
    base = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(f"{y}:Q", title=ytitle),
            color="Zone:N",
            tooltip=["Zone", "Date:T", alt.Tooltip(f"{y}:Q", format=".1f")],
        )
    )
    layers = [
        base,
        alt.Chart(pd.DataFrame({"Date": [TODAY]})).mark_rule(strokeDash=[5, 5], color="gray").encode(x="Date:T"),
    ]
    for val, col in hlines or []:
        layers.append(alt.Chart(pd.DataFrame({"y": [val]})).mark_rule(color=col).encode(y="y:Q"))
    return alt.layer(*layers).properties(height=340)


tab1, tab2 = st.tabs(["Daily rainfall", f"Rolling {LAG_DAYS}-day rainfall vs thresholds"])
with tab1:
    st.altair_chart(trend_chart(trend, "Rain_mm", "Rainfall (mm/day)"), use_container_width=True)
with tab2:
    st.altair_chart(
        trend_chart(
            trend.dropna(subset=["Rain_14d_mm"]),
            "Rain_14d_mm",
            f"{LAG_DAYS}-day rainfall (mm)",
            hlines=[(RAIN_MOD_MM, RISK_HEX["Moderate"]), (RAIN_HIGH_MM, RISK_HEX["High"])],
        ),
        use_container_width=True,
    )
    st.caption(f"Orange line = moderate threshold ({RAIN_MOD_MM} mm), red line = high threshold ({RAIN_HIGH_MM} mm).")

# ──────────────────────────────────────────────────────────────
# 6. Zone deep-dive
# ──────────────────────────────────────────────────────────────
st.subheader("🔍 Zone Deep-Dive Inspector")
selected_zone = st.selectbox("Select a zone:", df_weather["Zone"].unique())
zr = df_weather[df_weather["Zone"] == selected_zone].iloc[0]
d1, d2, d3 = st.columns(3)
d1.metric("Temperature", f"{zr['Temps_C']} °C")
d2.metric("Window Rainfall", f"{zr['Rainfall_mm']} mm")
d3.metric("Humidity", f"{zr['Humidity']:.0f} %")
if zr["Risk"] == "High":
    st.error(f"🚨 **CRITICAL ALERT in {selected_zone}:** high vector proliferation risk. Larviciding and sanitation required.")
elif zr["Risk"] == "Moderate":
    st.warning(f"⚠️ **MODERATE RISK in {selected_zone}:** monitor stagnant water and issue advisories.")
else:
    st.success(f"✅ **LOW RISK in {selected_zone}:** conditions below transmission thresholds.")

# ──────────────────────────────────────────────────────────────
# 7. CSV export
# ──────────────────────────────────────────────────────────────
st.subheader("📩 Export Regional Risk Report")
st.download_button(
    "Download Active Risk Report (CSV)",
    data=df_weather.to_csv(index=False).encode("utf-8"),
    file_name="malaria_risk_dispatch_report.csv",
    mime="text/csv",
)

# ──────────────────────────────────────────────────────────────
# 8. Clinical surveillance
# ──────────────────────────────────────────────────────────────
st.subheader("🏥 Clinical Surveillance & Patient Entry Portal")
st.markdown("Log RDT(RAPID DIAGNOSTIC TEST) results from primary health centres to track cases against weather risk.")

with st.form("patient_registration_form"):
    p1, p2 = st.columns(2)
    with p1:
        patient_id = st.text_input("Hospital name (e.g uch ibadan)")
        patient_name = st.text_input("Patient Full Name")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=25)
        zone_select = st.selectbox("Reporting Zone", list(ZONES))
    with p2:
        rdt_status = st.selectbox("RDT Result", ["Positive (+)", "Negative (-)"])
        treatment_given = st.selectbox(
            "Intervention", ["Artemether-Lumefantrine (AL)",
 "Artesunate-Amodiaquine (ASAQ)",
 "Injectable Artesunate (Severe)",
 "Pre-referral rectal artesunate",
 "Other antimalarial",
 "No antimalarial (RDT negative)",
 "Referred to higher facility"]
        )
        reporter_name = st.text_input("Reporting Health Officer")
    submitted = st.form_submit_button("💾 Submit Patient Record")

    if submitted:
        if not patient_id.strip() or not patient_name.strip():
            st.error("Patient ID and Patient Name are required.")
        else:
            try:
                conn = sqlite3.connect(DB_PATH)
                cols = [c[1] for c in conn.execute(
                    "PRAGMA table_info(patient_cases)").fetchall()]
                if "patient_name" not in cols:
                    conn.execute(
                        "ALTER TABLE patient_cases ADD COLUMN patient_name TEXT")
                conn.execute(
                    "INSERT INTO patient_cases (patient_id, patient_name, age, "
                    "zone, rdt_result, treatment, officer) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (patient_id.strip(), patient_name.strip(), patient_age,
                     zone_select, rdt_status, treatment_given, reporter_name),
                )
                conn.commit()
                conn.close()
                st.success(f"Logged patient {patient_id} for {zone_select}.")
            except Exception as e:
                st.error(f"Database error: {e}")

try:
    conn = sqlite3.connect(DB_PATH)
    cases_7d = pd.read_sql(
        "SELECT zone AS Zone, COUNT(*) AS Positives_7d FROM patient_cases "
        "WHERE rdt_result LIKE 'Positive%' AND timestamp >= datetime('now', '-7 days') GROUP BY zone",
        conn,
    )
    conn.close()
except Exception:
    cases_7d = pd.DataFrame(columns=["Zone", "Positives_7d"])


if st.checkbox("📊 View Patient Case Log"):
    conn = sqlite3.connect(DB_PATH)
    df_patients = pd.read_sql("SELECT * FROM patient_cases ORDER BY id DESC", conn)
    conn.close()
    if df_patients.empty:
        st.info("No patient records yet. Use the form above.")
    else:
        st.dataframe(df_patients, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# 9. Supply chain: stock vs demand projected from risk
# ──────────────────────────────────────────────────────────────
# ======================================================================
# MEDICAL SUPPLIES: live inventory block for app.py
# Replaces the old slider-based "Medical Supplies" section.
# Needs ledger.py (the new version) in the same folder as app.py.
# ======================================================================
import ledger
import pandas as pd
from datetime import date as _date

ledger.init_db()
_rerun = getattr(st, "rerun", None) or st.experimental_rerun   # works on old and new Streamlit

st.markdown("---")
st.subheader(" Medical Supplies: Live Inventory")
st.caption("Every recorded use is subtracted from stock. Days left is worked out from REAL average daily use "
           "over the last 14 days, not placeholder numbers.")

if "inv_flash" in st.session_state:                 # "Saved" message that survives the page refresh
    st.success(st.session_state.pop("inv_flash"))

# --- Forecast -> stock link ------------------------------------------------
if "Risk" in df_weather.columns:
    n_high = int(df_weather["Risk"].astype(str).str.contains("High").sum())
    if n_high:
        st.warning(f"🌧️ Forecast: {n_high} zone(s) at HIGH malaria risk. Demand usually rises after heavy rain, "
                   "so test a surge with the slider below.")

surge_pct = st.slider("What if demand rises by (%)? Test a wet-season surge", 0, 200, 0, step=10)
surge = surge_pct / 100
if surge_pct:
    st.caption(f"Showing a +{surge_pct}% demand surge scenario. Set the slider to 0 to see real usage only.")

inv_rows = ledger.stock_summary(14, 30, uplift=surge)     # one row per facility and drug
inv_total = ledger.commodity_totals(14, 30, uplift=surge) # one row per drug, all facilities added up

# --- Nothing recorded yet ---------------------------------------------------
if inv_rows.empty:
    st.info("No stock recorded yet. Open 'Stock Ledger' in the left menu to add a facility and its starting stock, "
            "or load demo data below to see how the inventory works.")
    if st.button("Load DEMO data (not real)"):
        try:
            ledger.load_demo_data()
            _rerun()
        except ValueError as e:
            st.warning(str(e))
else:
    # --- Almost-finished alerts ---------------------------------------------
    is_low = inv_rows["Status"].str.contains("Red|Stock-out")
    is_soon = inv_rows["Status"].str.contains("Amber")
    for r in inv_rows[is_low].to_dict("records"):
        st.error(f"🔴 **{r['Facility']}: {r['Commodity']} is almost finished.** {int(r['Balance']):,} left, "
                 f"about {r['Days left']} days of stock (stock-out around {r['Stock-out date'] or 'now'}). "
                 f"Suggested order: {int(r['Suggested order']):,}.")
    for r in inv_rows[is_soon].to_dict("records"):
        st.warning(f"🟠 **{r['Facility']}: {r['Commodity']} is running low.** {int(r['Balance']):,} left, "
                   f"about {r['Days left']} days. Order soon (suggested: {int(r['Suggested order']):,}).")
    if inv_rows["Status"].str.contains("Green").all():
        st.success("✅ No stock alerts: every item has 14 or more days of cover.")
    elif not is_low.any() and not is_soon.any():
        st.info("Some items have no usage recorded yet, so days left cannot be estimated for them.")

    # --- One card per drug (all facilities added up) -------------------------
    st.markdown("**All facilities combined**")
    cols = st.columns(len(inv_total))
    for col, r in zip(cols, inv_total.to_dict("records")):
        days = None if pd.isna(r["Days left"]) else float(r["Days left"])
        with col:
            st.metric(r["Commodity"], f"{int(r['Balance']):,} in stock",
                      f"~{days:.0f} days left" if days is not None else "no usage recorded yet",
                      delta_color="off")
            st.progress(min(1.0, days / 30) if days is not None else 0.0)
            if "Red" in r["Status"] or "Stock-out" in r["Status"]:
                st.error("🔴 ALMOST FINISHED: order now" if days else "🔴 STOCK-OUT")
            elif "Amber" in r["Status"]:
                st.warning("🟠 Running low: order soon")
            elif "Green" in r["Status"]:
                st.success("🟢 Healthy stock")
            else:
                st.info("⚪ No usage recorded yet")
            if r["Stock-out date"]:
                st.caption(f"Estimated stock-out: {r['Stock-out date']}. Suggested order: {int(r['Suggested order']):,}")

    with st.expander("Stock by facility / village"):
        st.dataframe(inv_rows, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Download stock report (CSV)", inv_rows.to_csv(index=False).encode("utf-8"),
                           "stock_report.csv", "text/csv")
        st.caption("Red = under 7 days of stock, amber = 7 to 13 days, green = 14 or more.")

    # --- Record usage right here ---------------------------------------------
    st.markdown("#### Record usage")
    _facs, _coms = ledger.list_facilities(), ledger.list_commodities()
    fac_map = {n: int(i) for n, i in zip(_facs["name"], _facs["id"])}
    com_map = dict(zip(_coms["name"], _coms["code"]))
    with st.form("quick_use", clear_on_submit=True):
        a, b, c, d = st.columns(4)
        facility = a.selectbox("Facility / village", list(fac_map))
        drug = b.selectbox("Drug / item", list(com_map))
        qty = c.number_input("Quantity used", min_value=1, value=1, step=1)
        span = d.number_input("Over how many days?", min_value=1, max_value=31, value=1, step=1,
                              help="1 = used today. Type 3 if this quantity was used over 3 days.")
        who = st.text_input("Your name")
        if st.form_submit_button("💾 Record usage and subtract from stock"):
            try:
                left = ledger.record_entry(_date.today(), fac_map[facility], com_map[drug], "used",
                                           int(qty), who, int(span))
                st.session_state["inv_flash"] = f"Saved. {drug} at {facility}: {left} left."
                _rerun()
            except ValueError as e:
                st.error(str(e))

    with st.expander("Recent stock movements"):
        recent = ledger.history(limit=10)[["date", "facility", "commodity", "type", "received", "used",
                                           "balance", "entered_by", "status"]]
        st.dataframe(recent, hide_index=True, use_container_width=True)

st.caption("Deliveries, starting stock and corrections: open 'Stock Ledger' in the left menu. "
           "Decision support only, not clinical advice.")
# ===================== END of Medical Supplies block ======================

# ──────────────────────────────────────────────────────────────
# 10. Risk log viewer (now actually populated)
# ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📊 Forecast Event Log (SQL backend)"):
    if st.button("Log this forecast snapshot"):
        conn = sqlite3.connect(DB_PATH)
        conn.executemany(
            "INSERT INTO risk_logs (zone, rain_window_mm, humidity, temp_c, risk, simulated) VALUES (?,?,?,?,?,?)",
            [
                (r.Zone, r.Rainfall_mm, r.Humidity, r.Temps_C, r.Risk, int(sim_mode))
                for r in df_weather.itertuples()
            ],
        )
        conn.commit()
        conn.close()
        st.success("Snapshot logged.")
    conn = sqlite3.connect(DB_PATH)
    df_logs = pd.read_sql("SELECT * FROM risk_logs ORDER BY id DESC", conn)
    conn.close()
    if df_logs.empty:
        st.info("No snapshots yet. Click the button above to log the current forecast.")
    else:
        st.dataframe(df_logs, use_container_width=True)
