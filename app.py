
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Agriculture Crop Pattern Mining",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Agriculture Crop Pattern Mining")
st.write("Interactive dashboard for exploring crop patterns, seasons, soil types, rainfall, and yield.")

DATA_PATH = Path(__file__).parent / "data" / "crop_data.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()

st.sidebar.header("🔍 Select Your Filters")

filtered = df.copy()

for col in ["Crop", "Season", "Soil_Type", "Region"]:
    if col in filtered.columns:
        values = sorted(filtered[col].dropna().astype(str).unique())
        selected = st.sidebar.selectbox(
            f"Select {col.replace('_', ' ')}",
            ["All"] + values
        )

        if selected != "All":
            filtered = filtered[
                filtered[col].astype(str) == selected
            ]

for col in ["Rainfall", "Temperature"]:
    if col in filtered.columns:
        numeric = pd.to_numeric(filtered[col], errors="coerce")
        valid = numeric.dropna()

        if not valid.empty:
            minimum = float(valid.min())
            maximum = float(valid.max())

            selected_range = st.sidebar.slider(
                f"{col} Range",
                min_value=minimum,
                max_value=maximum,
                value=(minimum, maximum)
            )

            filtered = filtered[
                numeric.between(
                    selected_range[0],
                    selected_range[1]
                )
            ]

st.subheader("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Dataset Records", len(df))
c2.metric("Filtered Records", len(filtered))
c3.metric(
    "Number of Crops",
    filtered["Crop"].nunique() if "Crop" in filtered.columns else 0
)

st.divider()

st.subheader("📋 Filtered Crop Data")
st.dataframe(filtered, use_container_width=True)

if filtered.empty:
    st.warning("No records match these filters. Please change your selections.")
    st.stop()

st.divider()

st.subheader("🌱 Crop Distribution")

if "Crop" in filtered.columns:
    crop_counts = filtered["Crop"].value_counts()
    st.bar_chart(crop_counts)

st.subheader("🌦️ Crop Distribution by Season")

if "Season" in filtered.columns and "Crop" in filtered.columns:
    season_data = pd.crosstab(
        filtered["Season"],
        filtered["Crop"]
    )
    st.bar_chart(season_data)

st.subheader("🪨 Soil Type Distribution")

if "Soil_Type" in filtered.columns:
    soil_counts = filtered["Soil_Type"].value_counts()
    st.bar_chart(soil_counts)

st.subheader("📈 Average Yield Analysis")

if "Yield" in filtered.columns:
    filtered["Yield"] = pd.to_numeric(
        filtered["Yield"],
        errors="coerce"
    )

    if filtered["Yield"].notna().any():
        average_yield = filtered["Yield"].mean()
        st.metric("Average Yield in Selected Records", f"{average_yield:.2f}")

        if "Crop" in filtered.columns:
            yield_by_crop = (
                filtered.groupby("Crop")["Yield"]
                .mean()
                .dropna()
                .sort_values(ascending=False)
            )

            st.bar_chart(yield_by_crop)

st.divider()

st.subheader("📥 Download Filtered Data")

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered CSV",
    data=csv_data,
    file_name="filtered_crop_data.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully!")