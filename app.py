import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px

st.set_page_config(page_title="Bengaluru House Price Predictor", page_icon="🏠", layout="wide")

# ---------------------------------------------------------------
# LOAD ARTIFACTS (cached so it only loads once)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("meta.json", "r") as f:
        meta = json.load(f)
    return model, meta

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

model, meta = load_artifacts()
df = load_data()

FEATURE_COLUMNS = meta["feature_columns"]
LOCATIONS = meta["locations"]
AREA_TYPES = meta["area_types"]

# ---------------------------------------------------------------
# PREDICT FUNCTION
# ---------------------------------------------------------------
def predict_price(location, area_type, total_sqft, bath, balcony, bhk, ready_to_move):
    x = pd.DataFrame(np.zeros((1, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    x.loc[0, "total_sqft"] = total_sqft
    x.loc[0, "bath"] = bath
    x.loc[0, "balcony"] = balcony
    x.loc[0, "bhk"] = bhk
    x.loc[0, "ready_to_move"] = 1 if ready_to_move else 0

    loc_col = f"loc_{location}"
    if loc_col in x.columns:
        x.loc[0, loc_col] = 1  # else falls into the dropped baseline location

    area_col = f"area_{area_type}"
    if area_col in x.columns:
        x.loc[0, area_col] = 1

    pred = model.predict(x)[0]
    return max(pred, 0)

# ---------------------------------------------------------------
# SIDEBAR NAV
# ---------------------------------------------------------------
st.sidebar.title("🏠 Navigation")
page = st.sidebar.radio("Go to", ["Predict Price", "Data Explorer", "Model Info"])

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **{meta['best_model']}**")
st.sidebar.caption(f"Test RMSE: **{meta['test_rmse']} lakh**")
st.sidebar.caption(f"Test R²: **{meta['test_r2']}**")

# ---------------------------------------------------------------
# PAGE 1: PREDICT
# ---------------------------------------------------------------
if page == "Predict Price":
    st.title("🏠 Bengaluru House Price Predictor")
    st.write("Enter property details below to estimate the market price (in ₹ Lakhs).")

    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("Location", LOCATIONS, index=LOCATIONS.index("other") if "other" in LOCATIONS else 0)
        area_type = st.selectbox("Area Type", AREA_TYPES)
        total_sqft = st.number_input("Total Area (sqft)", min_value=200.0, max_value=15000.0,
                                      value=float(meta["total_sqft_median"]), step=50.0)
        bhk = st.slider("BHK", 1, 10, 2)

    with col2:
        bath = st.slider("Bathrooms", 1, 10, int(meta["bath_median"]))
        balcony = st.slider("Balconies", 0, 4, int(meta["balcony_median"]))
        ready_to_move = st.checkbox("Ready to Move?", value=True)

    st.markdown("---")

    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        price = predict_price(location, area_type, total_sqft, bath, balcony, bhk, ready_to_move)
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated Price", f"₹ {price:.2f} Lakh")
        c2.metric("≈ In Crores", f"₹ {price/100:.2f} Cr")
        c3.metric("Price per sqft", f"₹ {(price*100000/total_sqft):,.0f}")
        st.info(f"Estimate range: ₹{max(price-meta['test_rmse'],0):.1f} – ₹{price+meta['test_rmse']:.1f} Lakh "
                f"(based on model's typical error of ±{meta['test_rmse']} lakh)")

# ---------------------------------------------------------------
# PAGE 2: DATA EXPLORER
# ---------------------------------------------------------------
elif page == "Data Explorer":
    st.title("📊 Data Explorer")
    st.write(f"Cleaned dataset: **{df.shape[0]} rows** (after outlier removal from original 13,320)")

    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Location Insights", "Raw Data"])

    with tab1:
        fig = px.histogram(df, x="price", nbins=60, title="Price Distribution (₹ Lakh)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(df, x="total_sqft", y="price", color="bhk",
                           title="Price vs Total Sqft (colored by BHK)",
                           labels={"total_sqft": "Total Sqft", "price": "Price (Lakh)"})
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        top_locations = df["location"].value_counts().head(15)
        fig3 = px.bar(x=top_locations.index, y=top_locations.values,
                       title="Top 15 Locations by Listing Count",
                       labels={"x": "Location", "y": "Count"})
        st.plotly_chart(fig3, use_container_width=True)

        avg_price_loc = df.groupby("location")["price"].mean().sort_values(ascending=False).head(15)
        fig4 = px.bar(x=avg_price_loc.index, y=avg_price_loc.values,
                       title="Top 15 Most Expensive Locations (avg price)",
                       labels={"x": "Location", "y": "Avg Price (Lakh)"})
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------
# PAGE 3: MODEL INFO
# ---------------------------------------------------------------
elif page == "Model Info":
    st.title("🤖 Model Information")
    st.write(f"**Best model selected:** {meta['best_model']}")

    st.subheader("Cross-Validation RMSE Comparison")
    cv_df = pd.DataFrame(list(meta["cv_results"].items()), columns=["Model", "CV RMSE (lakh)"]).sort_values("CV RMSE (lakh)")
    fig = px.bar(cv_df, x="Model", y="CV RMSE (lakh)", title="Lower is better")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Test RMSE", f"{meta['test_rmse']} lakh")
    c2.metric("Test MAE", f"{meta['test_mae']} lakh")
    c3.metric("Test R²", meta["test_r2"])

    st.subheader("Pipeline Summary")
    st.markdown("""
    1. **Cleaning**: dropped `society` (high nulls/cardinality), parsed `size`→`bhk`,
       converted `total_sqft` (handled ranges & units: Sq. Meter, Sq. Yards, Perch, Acres, Guntha, Cents, Grounds),
       grouped rare locations (≤10 listings) into `other`, converted `availability`→`ready_to_move` binary.
    2. **Missing values**: `bath`/`balcony` filled with group/overall median.
    3. **Outlier removal**: sqft/bhk ratio < 300, bath > bhk+2, per-location price-per-sqft outliers (mean ± 1 std),
       inconsistent BHK pricing within a location, and top 1% luxury price outliers.
    4. **Features**: total_sqft, bath, balcony, bhk, ready_to_move + one-hot encoded location & area_type.
    5. **Model selection**: 5-fold CV across Linear Regression, Lasso, Decision Tree, Random Forest, Gradient Boosting.
    6. **Winner**: Gradient Boosting Regressor, tuned (n_estimators=300, max_depth=5, lr=0.05, subsample=0.9).
    """)
