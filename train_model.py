import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

# Load the datasets
disease_df = pd.read_csv("disease_data_monthly.csv")
climate_df = pd.read_csv("climate_data_monthly.csv")
population_df = pd.read_csv("population_data.csv")
healthcare_df = pd.read_csv("healthcare_data.csv")


# Merge datasets
merged_df = disease_df.merge(climate_df, on=["Month", "State", "District"])
merged_df = merged_df.merge(population_df, on=["State", "District"])
merged_df = merged_df.merge(healthcare_df, on=["State", "District"])

# Select features and target variable
features = [
    "AvgTemperature", "AvgHumidity", "TotalRainfall", "Population",
    "PopulationDensity", "UrbanizationRate", "Hospitals", "Beds", "Doctors", "Nurses"
]
target = "ReportedCases"

# Create a binary target variable: 1 for high outbreaks, 0 for low
merged_df["HighOutbreak"] = (merged_df[target] > 1000).astype(int)

# Split into features (X) and target (y)
X = merged_df[features]
y = merged_df["HighOutbreak"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Plot feature importances
feature_importances = model.feature_importances_
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Disease Outbreak Prediction")
plt.show()

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, "disease_outbreak_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully!")
