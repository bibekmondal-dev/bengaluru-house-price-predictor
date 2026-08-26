"""
Bengaluru House Price Prediction — full pipeline
Cleaning -> Feature Engineering -> Outlier Removal -> Model Training -> Save artifacts
"""
import pandas as pd
import numpy as np
import re
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------
df = pd.read_csv("Bengaluru_House_Data.csv")
print("Raw shape:", df.shape)

# ---------------------------------------------------------------
# 2. DROP LOW-VALUE / HIGH-NULL COLUMNS
# ---------------------------------------------------------------
# society: ~41% missing + 2688 unique values -> not usable
df = df.drop(columns=["society"])

# ---------------------------------------------------------------
# 3. CLEAN `size` -> bhk (integer)
# ---------------------------------------------------------------
df["size"] = df["size"].fillna(df["size"].mode()[0])
df["bhk"] = df["size"].apply(lambda x: int(str(x).split(" ")[0]))
df = df.drop(columns=["size"])

# ---------------------------------------------------------------
# 4. CLEAN `total_sqft` -> numeric sqft (handles ranges + units)
# ---------------------------------------------------------------
UNIT_TO_SQFT = {
    "Sq. Meter": 10.7639,
    "Sq. Yards": 9.0,
    "Perch": 272.25,
    "Acres": 43560.0,
    "Guntha": 1089.0,
    "Cents": 435.6,
    "Grounds": 2400.0,
}

def convert_sqft(x):
    x = str(x).strip()
    # range like "2100 - 2850"
    if "-" in x:
        parts = x.split("-")
        try:
            a, b = float(parts[0].strip()), float(parts[1].strip())
            return (a + b) / 2
        except ValueError:
            return np.nan
    # plain number
    try:
        return float(x)
    except ValueError:
        pass
    # number + unit, e.g. "34.46Sq. Meter"
    for unit, factor in UNIT_TO_SQFT.items():
        if unit in x:
            num = re.findall(r"[\d.]+", x)
            if num:
                return float(num[0]) * factor
    return np.nan

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna(subset=["total_sqft"])

# ---------------------------------------------------------------
# 5. CLEAN `location`
# ---------------------------------------------------------------
df["location"] = df["location"].fillna("other").apply(lambda x: str(x).strip())
location_counts = df["location"].value_counts()
rare_locations = location_counts[location_counts <= 10].index
df["location"] = df["location"].apply(lambda x: "other" if x in rare_locations else x)

# ---------------------------------------------------------------
# 6. CLEAN `availability` -> binary ready_to_move
# ---------------------------------------------------------------
df["ready_to_move"] = (df["availability"] == "Ready To Move").astype(int)
df = df.drop(columns=["availability"])

# ---------------------------------------------------------------
# 7. CLEAN `area_type` -> keep as category (will one-hot encode)
# ---------------------------------------------------------------
df["area_type"] = df["area_type"].fillna(df["area_type"].mode()[0]).apply(lambda x: str(x).strip())

# ---------------------------------------------------------------
# 8. FILL bath / balcony
# ---------------------------------------------------------------
df["bath"] = df["bath"].fillna(df.groupby("bhk")["bath"].transform("median"))
df["bath"] = df["bath"].fillna(df["bath"].median())
df["balcony"] = df["balcony"].fillna(df["balcony"].median())

# ---------------------------------------------------------------
# 9. FEATURE: price_per_sqft (for outlier detection only)
# ---------------------------------------------------------------
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

# ---------------------------------------------------------------
# 10. OUTLIER REMOVAL
# ---------------------------------------------------------------
before = df.shape[0]

# (a) unrealistic sqft per bedroom (< 300 sqft/room is basically impossible)
df = df[(df["total_sqft"] / df["bhk"]) >= 300]

# (b) bathrooms shouldn't exceed bhk + 2
df = df[df["bath"] <= (df["bhk"] + 2)]

# (c) remove price_per_sqft outliers *within each location* (mean +/- 1 std)
def remove_pps_outliers(data):
    out = pd.DataFrame()
    for loc, sub in data.groupby("location"):
        m, s = sub["price_per_sqft"].mean(), sub["price_per_sqft"].std()
        reduced = sub[(sub["price_per_sqft"] > (m - s)) & (sub["price_per_sqft"] <= (m + s))]
        out = pd.concat([out, reduced], ignore_index=True)
    return out

df = remove_pps_outliers(df)

# (d) for same location, a 2 BHK priced above a 3 BHK at similar/lower sqft is bad data
def remove_bhk_outliers(data):
    exclude_indices = np.array([])
    for loc, sub in data.groupby("location"):
        bhk_stats = {}
        for bhk, bhk_df in sub.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean": bhk_df["price_per_sqft"].mean(),
                "std": bhk_df["price_per_sqft"].std(),
                "count": bhk_df.shape[0],
            }
        for bhk, bhk_df in sub.groupby("bhk"):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats["count"] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df["price_per_sqft"] < stats["mean"]].index.values,
                )
    return data.drop(exclude_indices, errors="ignore")

df = df.reset_index(drop=True)
df = remove_bhk_outliers(df)

# (e) drop extreme luxury outliers (top 1% by price) — sparse, noisy, wreck RMSE
price_cap = df["price"].quantile(0.99)
df = df[df["price"] <= price_cap]

print(f"Outlier removal: {before} -> {df.shape[0]} rows (price cap: {price_cap:.0f} lakh)")

df = df.drop(columns=["price_per_sqft"])

# ---------------------------------------------------------------
# 11. FINAL FEATURE SET
# ---------------------------------------------------------------
features_df = df[["location", "area_type", "total_sqft", "bath", "balcony", "bhk", "ready_to_move", "price"]].copy()

# one-hot encode location & area_type
location_dummies = pd.get_dummies(features_df["location"], prefix="loc", drop_first=True)
area_dummies = pd.get_dummies(features_df["area_type"], prefix="area", drop_first=True)

X = pd.concat(
    [features_df[["total_sqft", "bath", "balcony", "bhk", "ready_to_move"]], location_dummies, area_dummies],
    axis=1,
)
y = features_df["price"]

feature_columns = X.columns.tolist()
print("Final feature matrix:", X.shape)

# ---------------------------------------------------------------
# 12. TRAIN / TEST SPLIT
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# ---------------------------------------------------------------
# 13. MODEL COMPARISON (5-fold CV, RMSE)
# ---------------------------------------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(alpha=1.0, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, random_state=RANDOM_STATE),
}

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    rmse = -scores.mean()
    results[name] = rmse
    print(f"{name:20s} CV RMSE: {rmse:.3f} (+/- {scores.std():.3f})")

best_name = min(results, key=results.get)
print("\nBest model (by CV RMSE):", best_name)

# ---------------------------------------------------------------
# 14. FINAL FIT + TEST EVALUATION
# ---------------------------------------------------------------
best_model = models[best_name]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"\nTest RMSE: {test_rmse:.3f} lakh")
print(f"Test MAE : {test_mae:.3f} lakh")
print(f"Test R2  : {test_r2:.4f}")

# refit best model on FULL data for deployment
best_model.fit(X, y)

# ---------------------------------------------------------------
# 15. SAVE ARTIFACTS
# ---------------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

locations_sorted = sorted(df["location"].unique().tolist())
area_types_sorted = sorted(df["area_type"].unique().tolist())

meta = {
    "feature_columns": feature_columns,
    "locations": locations_sorted,
    "area_types": area_types_sorted,
    "best_model": best_name,
    "test_rmse": round(test_rmse, 3),
    "test_mae": round(test_mae, 3),
    "test_r2": round(test_r2, 4),
    "cv_results": {k: round(v, 3) for k, v in results.items()},
    "total_sqft_median": float(df["total_sqft"].median()),
    "bath_median": float(df["bath"].median()),
    "balcony_median": float(df["balcony"].median()),
}
with open("meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# save cleaned dataframe too (handy for the Streamlit EDA tab)
df.to_csv("cleaned_data.csv", index=False)

print("\nSaved: model.pkl, meta.json, cleaned_data.csv")
