# ==========================================
# RETAIL DEMAND FORECASTING
# Model Training + Evaluation + Forecast + Save
# ==========================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 2. LOAD DATASET

df = pd.read_csv("retail_sales.csv")

df["date"] = pd.to_datetime(df["date"])

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())


# 3. SELECT ONE STORE AND ONE ITEM

store_data = df[
    (df["store_id"] == "store_1") &
    (df["item_id"] == "item_1")
].copy()

store_data = store_data.sort_values("date")
store_data = store_data.reset_index(drop=True)

print("Selected Data Shape:", store_data.shape)


# 4. EXPLORATORY DATA ANALYSIS

# Monthly sales
monthly_sales = store_data.groupby(
    store_data["date"].dt.to_period("M")
)["sales"].sum()

monthly_sales.plot(figsize=(12, 5))
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Yearly sales
yearly_sales = store_data.groupby(
    store_data["date"].dt.year
)["sales"].sum()

yearly_sales.plot(kind="bar", figsize=(8, 5))
plt.title("Yearly Sales")
plt.xlabel("Year")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()


# 5. FEATURE ENGINEERING

store_data["year"] = store_data["date"].dt.year
store_data["weekday"] = store_data["date"].dt.weekday
store_data["month"] = store_data["date"].dt.month

# Lag features
store_data["lag_1"] = store_data["sales"].shift(1)
store_data["lag_7"] = store_data["sales"].shift(7)

# Remove rows with missing lag values
store_data = store_data.dropna().reset_index(drop=True)


# 6. DEFINE FEATURES AND TARGET

features = [
    "price",
    "promo",
    "weekday",
    "month",
    "year",
    "lag_1",
    "lag_7"
]

X = store_data[features]
y = store_data["sales"]


# 7. TIME-BASED TRAIN, VALIDATION AND TEST SPLIT

# Train: 2019-2021
# Validation: 2022
# Test: 2023

train_data = store_data[store_data["date"].dt.year < 2022]
val_data = store_data[store_data["date"].dt.year == 2022]
test_data = store_data[store_data["date"].dt.year == 2023]

X_train = train_data[features]
y_train = train_data["sales"]

X_val = val_data[features]
y_val = val_data["sales"]

X_test = test_data[features]
y_test = test_data["sales"]

print("Train rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# 8. TRAIN DIFFERENT MODELS ON TRAINING DATA

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ),

    "Tuned Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}

validation_results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    val_mae = mean_absolute_error(
        y_val,
        val_predictions
    )

    validation_results[name] = val_mae

    print(name, ":", round(val_mae, 4))


# Baseline: previous day's sales
baseline_val_predictions = X_val["lag_1"]

baseline_val_mae = mean_absolute_error(
    y_val,
    baseline_val_predictions
)

validation_results["Baseline"] = baseline_val_mae

print("Baseline:", round(baseline_val_mae, 4))


# 9. COMPARE VALIDATION RESULTS

validation_df = pd.DataFrame(
    list(validation_results.items()),
    columns=["Model", "MAE"]
).sort_values("MAE")

print("\nValidation Model Comparison")
print(validation_df)

validation_df.plot(
    x="Model",
    y="MAE",
    kind="bar",
    figsize=(10, 5),
    legend=False
)

plt.title("Model Comparison on Validation Data")
plt.ylabel("MAE")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 10. TRAIN FINAL RANDOM FOREST MODEL

# Based on validation results, use Random Forest.
# Train on all data before 2023.

final_model = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)

X_final_train = store_data[
    store_data["date"].dt.year < 2023
][features]

y_final_train = store_data[
    store_data["date"].dt.year < 2023
]["sales"]

final_model.fit(X_final_train, y_final_train)


# 11. FINAL TEST EVALUATION

test_predictions = final_model.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

baseline_test_mae = mean_absolute_error(
    y_test,
    X_test["lag_1"]
)

print("\nFinal Test Results")
print("Baseline MAE:", round(baseline_test_mae, 4))
print("Random Forest MAE:", round(test_mae, 4))
print("Random Forest RMSE:", round(test_rmse, 4))


# 12. ACTUAL VS PREDICTED

plt.figure(figsize=(12, 5))

plt.plot(
    test_data["date"],
    y_test,
    label="Actual Sales"
)

plt.plot(
    test_data["date"],
    test_predictions,
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()


# 13. RESIDUAL ANALYSIS

residuals = y_test - test_predictions

plt.figure(figsize=(8, 5))

plt.scatter(
    test_predictions,
    residuals,
    alpha=0.5
)

plt.axhline(y=0, color="red", linestyle="--")

plt.title("Residual Plot")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()


# 14. FEATURE IMPORTANCE

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": final_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)

importance_df.plot(
    x="Feature",
    y="Importance",
    kind="bar",
    figsize=(9, 5),
    legend=False
)

plt.title("Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# 15. SAVE MODEL USING JOBLIB

model_artifact = {
    "model": final_model,
    "features": features,
    "store_id": "store_1",
    "item_id": "item_1",
    "last_date": store_data["date"].max(),
    "last_price": float(store_data.iloc[-1]["price"]),
    "last_promo": int(store_data.iloc[-1]["promo"])
}

joblib.dump(
    model_artifact,
    "retail_model.pkl"
)

print("\nModel saved successfully!")
print("File: retail_model.pkl")


# 16. LOAD SAVED MODEL (CHECK)

loaded_artifact = joblib.load("retail_model.pkl")

loaded_model = loaded_artifact["model"]

print("Saved model loaded successfully!")


# 17. GENERATE 7-DAY RECURSIVE FORECAST

history = store_data.set_index("date")["sales"].copy()

last_date = store_data["date"].max()

forecast_results = []

for i in range(1, 8):

    future_date = last_date + pd.Timedelta(days=i)

    lag_1 = history.loc[
        future_date - pd.Timedelta(days=1)
    ]

    lag_7 = history.loc[
        future_date - pd.Timedelta(days=7)
    ]

    input_data = pd.DataFrame([{
        "price": model_artifact["last_price"],
        "promo": model_artifact["last_promo"],
        "weekday": future_date.weekday(),
        "month": future_date.month,
        "year": future_date.year,
        "lag_1": lag_1,
        "lag_7": lag_7
    }])[features]

    prediction = loaded_model.predict(input_data)[0]

    forecast_results.append({
        "date": future_date,
        "predicted_sales": round(prediction, 4)
    })

    # Add prediction to history for recursive forecasting
    history.loc[future_date] = prediction


forecast_df = pd.DataFrame(forecast_results)

print("\n7-Day Forecast:")
print(forecast_df)


# 18. PLOT FORECAST

plt.figure(figsize=(10, 5))

plt.plot(
    forecast_df["date"],
    forecast_df["predicted_sales"],
    marker="o"
)

plt.title("7-Day Retail Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.xticks(rotation=30)
plt.grid(True)
plt.tight_layout()
plt.show()