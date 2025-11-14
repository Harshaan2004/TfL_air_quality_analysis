"""
TfL Air Quality Analysis Script

This script analyses TfL air quality data with focus on Northern Line
as worst-case scenario for pathogen transmission risk assessment.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

# 1. Load Data
df = pd.read_csv('tfl_air_quality_data.csv')

print("--- Data Loaded Successfully ---")
print(f"Loaded {len(df)} data points (rows).")
print("\n")

# 2. Define Features (X) and Target (y)
# Adding 'Year' and 'Carriages_per_Train' for comprehensive analysis
FEATURES = ['Vent_per_Carriage_m3s', 'Line_Type', 'Year', 'Carriages_per_Train']
TARGET = 'Mean_Respirable_Dust_mg_m3'

X = df[FEATURES]
y = df[TARGET]

# 3. Preprocessing: Handle Categorical & Numerical Data
categorical_features = ['Line_Type']
numerical_features = ['Vent_per_Carriage_m3s', 'Year', 'Carriages_per_Train']

# Create a 'preprocessor' to handle the different feature types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# 4. Create the Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5. Assess Model Performance with Cross-Validation
# This is a more robust method for small datasets
# We use K-Folds cross-validation (splitting the data into 5 'folds')
# n_splits=5: N=24 will use ~20 for training and ~4 for testing, 5 times
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("--- Model Assessment (Cross-Validation) ---")
print(f"R-squared scores for 5 folds: {np.round(cv_scores, 2)}")
print(f"Mean Cross-Validated R-squared: {np.mean(cv_scores):.4f}")
print(f"Std. Dev. of R-squared: {np.std(cv_scores):.4f}")
print("\nThis R-squared score is a more honest and robust "
      "measure of model performance.")
print("This is the R-squared to report in dissertation.\n")


# 6. Analyse Feature Importance (The main goal)
# To get final coefficients, train the model on ALL data.
# This is valid because goal is *inference* (understanding) not *prediction*.
model.fit(X, y)

print("--- Model Coefficients (Feature Importance) ---")
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coefficients = model.named_steps['regressor'].coef_
intercept = model.named_steps['regressor'].intercept_

print(f"Model Base Intercept: {intercept:.4f}")
print("This is the 'starting' dust level before considering the features.\n")

print("Feature Coefficients:")
for name, coef in zip(feature_names, coefficients):
    clean_name = name.replace('num__', '').replace('cat__Line_Type_', '')
    print(f"  - {clean_name}: {coef:.4f}")

print("\n--- Interpretation ---")
print("This model assesses the *linear relationship* between "
      "features and dust level.")
print("A *negative* coefficient means that as the feature's value "
      "*increases*, the dust level *decreases*.")
print("A *positive* coefficient means that as the feature's value "
      "*increases*, the dust level *increases*.")
print("The *size* of the coefficient shows how strong that "
      "feature's influence is.")


# 7. Visualise the Results
print("\n--- Generating Visualisation ---")

plt.figure(figsize=(12, 7))
colors = {'Deep-Level': 'red', 'Sub-Surface': 'blue'}
scatter = plt.scatter(
    df['Vent_per_Carriage_m3s'],
    df['Mean_Respirable_Dust_mg_m3'],
    c=df['Line_Type'].map(colors),
    alpha=0.7,
    s=df['Year'].replace({2021: 50, 2023: 100, 2024: 150})  # Size shows year
)

plt.title('Ventilation Rate vs. Mean Respirable Dust in Driver Cabs '
          '(2021-2024)', fontsize=16)
plt.xlabel('Ventilation per Carriage (m³/s)', fontsize=12)
plt.ylabel('Mean Respirable Dust (mg/m³)', fontsize=12)

# Create legend for colors
color_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label=key,
               markerfacecolor=value, markersize=10)
    for key, value in colors.items()
]
# Create legend for sizes (Year)
size_handles = [
    plt.scatter([], [], s=s, label=year, c='gray')
    for year, s in {2021: 50, 2023: 100, 2024: 150}.items()
]

legend1 = plt.legend(handles=color_handles, title='Line Type', loc='upper left')
plt.gca().add_artist(legend1)
plt.legend(handles=size_handles, title='Year', loc='upper right')

plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ventilation_vs_dust_plot_v2.png')
print("Plot saved as 'ventilation_vs_dust_plot_v2.png'")
# plt.show()

print("\n=== DISSERTATION CONTEXT: CFD-ML INTEGRATION ===")
print("This analysis provides baseline data for:")
print("1. CFD Model Validation - Ventilation rates vs "
      "particle concentration")
print("2. ML Training Data - Real-world TfL measurements")
print("3. Risk Assessment Parameters - Line type and occupancy effects")
print("\nKey findings for aerosol transmission modelling:")
print(f"- Ventilation effectiveness: {coefficients[0]:.4f} "
      f"mg/m³ per m³/s increase")
# Coeffs order: ['Vent_per_Carriage_m3s', 'Year', 'Carriages_per_Train',
# 'Line_Type_Sub-Surface']
print(f"- Deep-level vs Sub-surface difference: {coefficients[3]:.4f} mg/m³")
print(f"- Temporal variation (Year effect): {coefficients[1]:.4f} "
      f"mg/m³ per year")

# Additional analysis for dissertation
print("\n=== RISK ASSESSMENT IMPLICATIONS ===")
# Calculate risk categories based on dust levels
df['Risk_Category'] = pd.cut(df['Mean_Respirable_Dust_mg_m3'],
                            bins=[0, 0.1, 0.2, 0.5],
                            labels=['Low', 'Medium', 'High'])

risk_summary = df.groupby(['Line_Type', 'Risk_Category']).size().unstack(fill_value=0)
print("Risk distribution by line type:")
print(risk_summary)

print("\n=== CFD VALIDATION PARAMETERS ===")
print("Recommended CFD simulation parameters based on data:")
vent_min = df['Vent_per_Carriage_m3s'].min()
vent_max = df['Vent_per_Carriage_m3s'].max()
print(f"- Ventilation range to test: {vent_min:.2f} - {vent_max:.2f} m³/s")
dust_min = df['Mean_Respirable_Dust_mg_m3'].min()
dust_max = df['Mean_Respirable_Dust_mg_m3'].max()
print(f"- Expected particle concentration range: "
      f"{dust_min:.2f} - {dust_max:.2f} mg/m³")
low_risk_vent = df[df['Mean_Respirable_Dust_mg_m3'] < 0.1]
critical_vent = low_risk_vent['Vent_per_Carriage_m3s'].min()
print(f"- Critical ventilation threshold for low risk: ~{critical_vent:.2f} m³/s")

print("\n=== NORTHERN LINE FOCUS: WORST-CASE SCENARIO ANALYSIS ===")
northern_data = df[df['Line'] == 'Northern'].copy()
print("Northern Line - Consistently highest dust levels "
      "(worst-case scenario):")
print(f"- Sample size: {len(northern_data)} measurements across "
      f"{northern_data['Year'].nunique()} years")
n_dust_min = northern_data['Mean_Respirable_Dust_mg_m3'].min()
n_dust_max = northern_data['Mean_Respirable_Dust_mg_m3'].max()
print(f"- Dust level range: {n_dust_min:.2f} - {n_dust_max:.2f} mg/m³")
n_dust_mean = northern_data['Mean_Respirable_Dust_mg_m3'].mean()
print(f"- Mean dust level: {n_dust_mean:.3f} mg/m³")
n_vent_rate = northern_data['Vent_per_Carriage_m3s'].iloc[0]
print(f"- Ventilation rate: {n_vent_rate:.2f} m³/s (consistently low)")

# Compare Northern Line to all others
other_lines = df[df['Line'] != 'Northern']
print("\nComparison with other lines:")
print(f"- Northern Line mean: {n_dust_mean:.3f} mg/m³")
other_mean = other_lines['Mean_Respirable_Dust_mg_m3'].mean()
print(f"- All other lines mean: {other_mean:.3f} mg/m³")
ratio = n_dust_mean / other_mean
print(f"- Northern Line is {ratio:.1f}x worse")

# Risk category analysis for Northern Line
northern_risk = northern_data['Risk_Category'].value_counts()
print("\nNorthern Line risk distribution:")
for risk, count in northern_risk.items():
    percentage = count/len(northern_data)*100
    print(f"- {risk} risk: {count} measurements ({percentage:.1f}%)")

print("\n=== CFD SIMULATION PARAMETERS FOR NORTHERN LINE ===")
print("Based on Northern Line worst-case scenario:")
print(f"- Current ventilation: {n_vent_rate:.2f} m³/s")
print(f"- Current dust levels: {n_dust_min:.2f} - {n_dust_max:.2f} mg/m³")
print("- Recommended CFD test scenarios:")
print(f"  * Baseline (current): {n_vent_rate:.2f} m³/s ventilation")
print("  * Improved scenario: 1.39 m³/s ventilation (Sub-Surface level)")
print("  * Passenger scenarios: 0, 2, 4-5 passengers (as planned)")

# Temporal analysis for Northern Line
print("\nNorthern Line temporal trends (pathogen transmission implications):")
northern_temporal = northern_data.groupby('Year')['Mean_Respirable_Dust_mg_m3'].mean()
for year, dust_level in northern_temporal.items():
    print(f"- {year}: {dust_level:.3f} mg/m³")

improvement_2021_2023 = ((northern_temporal[2021] - northern_temporal[2023])
                         / northern_temporal[2021] * 100)
regression_2023_2024 = ((northern_temporal[2024] - northern_temporal[2023])
                        / northern_temporal[2023] * 100)
print(f"- Improvement 2021→2023: {improvement_2021_2023:.1f}% reduction")
print(f"- Regression 2023→2024: {regression_2023_2024:.1f}% increase")
print("This shows air quality can fluctuate - important for risk modelling!")
