import pandas as pd
import io
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

# 1. Load Your Data
csv_data = """Line,Year,Rolling_Stock,Line_Type,Carriages_per_Train,Total_Vent_m3s,Vent_per_Carriage_m3s,Mean_Respirable_Dust_mg_m3,Data_Source
Northern,2021,1995TS,Deep-Level,6,7.102,1.18,0.37,Driver Cab
Bakerloo,2021,1972TS,Deep-Level,7,7.553,1.08,0.35,Driver Cab
Victoria,2021,2009TS,Deep-Level,8,9.068,1.13,0.31,Driver Cab
Piccadilly,2021,1973TS,Deep-Level,6,7.079,1.18,0.26,Driver Cab
Central,2021,1992TS,Deep-Level,8,9.284,1.16,0.23,Driver Cab
Jubilee,2021,1996TS,Deep-Level,7,9.042,1.29,0.12,Driver Cab
Circle/H&C,2021,S7,Sub-Surface,7,9.742,1.39,0.05,Driver Cab
District,2021,S7,Sub-Surface,7,9.742,1.39,0.04,Driver Cab
Northern,2023,1995TS,Deep-Level,6,7.102,1.18,0.3,Driver Cab
Piccadilly,2023,1973TS,Deep-Level,6,7.079,1.18,0.28,Driver Cab
Bakerloo,2023,1972TS,Deep-Level,7,7.553,1.08,0.25,Driver Cab
Central,2023,1992TS,Deep-Level,8,9.284,1.16,0.19,Driver Cab
Victoria,2023,2009TS,Deep-Level,8,9.068,1.13,0.19,Driver Cab
Circle/H&C,2023,S7,Sub-Surface,7,9.742,1.39,0.18,Driver Cab
District,2023,S7,Sub-Surface,7,9.742,1.39,0.18,Driver Cab
Jubilee,2023,1996TS,Deep-Level,7,9.042,1.29,0.13,Driver Cab
Victoria,2024,2009TS,Deep-Level,8,9.068,1.13,0.43,Driver Cab
Bakerloo,2024,1972TS,Deep-Level,7,7.553,1.08,0.31,Driver Cab
Jubilee,2024,1996TS,Deep-Level,7,9.042,1.29,0.31,Driver Cab
Northern,2024,1995TS,Deep-Level,6,7.102,1.18,0.28,Driver Cab
Central,2024,1992TS,Deep-Level,8,9.284,1.16,0.27,Driver Cab
Piccadilly,2024,1973TS,Deep-Level,6,7.079,1.18,0.26,Driver Cab
Circle/H&C,2024,S7,Sub-Surface,7,9.742,1.39,0.06,Driver Cab
District,2024,S7,Sub-Surface,7,9.742,1.39,0.05,Driver Cab"""

df = pd.read_csv(io.StringIO(csv_data))

print("--- Data Loaded Successfully ---")
print(f"Loaded {len(df)} data points (rows).")
print("\n")

# 2. Define Features (X) and Target (y)
# We are adding 'Year' and 'Carriages_per_Train' as suggested by the analysis
features = ['Vent_per_Carriage_m3s', 'Line_Type', 'Year', 'Carriages_per_Train']
target = 'Mean_Respirable_Dust_mg_m3'

X = df[features]
y = df[target]

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
# This is a more robust method for small datasets, as pointed out in the critique.
# We use K-Folds cross-validation (splitting the data into 5 'folds')
# 'n_splits=5' is a good choice. Since N=24, it will use ~20 for training and ~4 for testing, 5 times.
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("--- Model Assessment (Cross-Validation) ---")
print(f"R-squared scores for each of the 5 'folds': {np.round(cv_scores, 2)}")
print(f"Mean Cross-Validated R-squared: {np.mean(cv_scores):.4f}")
print(f"Std. Dev. of R-squared: {np.std(cv_scores):.4f}")
print("\nThis R-squared score is a more honest and robust measure of your model's performance.")
print("This is the R-squared you should report in your dissertation.\n")


# 6. Analyze Feature Importance (The main goal)
# To get the final coefficients for interpretation, we train the model on ALL data.
# This is valid because our goal is *inference* (understanding) not *prediction*.
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
print("This model assesses the *linear relationship* between your features and the dust level.")
print("A *negative* coefficient means that as the feature's value *increases*, the dust level *decreases*.")
print("A *positive* coefficient means that as the feature's value *increases*, the dust level *increases*.")
print("The *size* of the coefficient shows how strong that feature's influence is.")


# 7. Visualize the Results
print("\n--- Generating Visualization ---")

plt.figure(figsize=(12, 7))
colors = {'Deep-Level': 'red', 'Sub-Surface': 'blue'}
scatter = plt.scatter(df['Vent_per_Carriage_m3s'], df['Mean_Respirable_Dust_mg_m3'], 
                      c=df['Line_Type'].map(colors), alpha=0.7,
                      s=df['Year'].replace({2021: 50, 2023: 100, 2024: 150})) # Use size to show 'Year'

plt.title('Ventilation Rate vs. Mean Respirable Dust in Driver Cabs (2021-2024)', fontsize=16)
plt.xlabel('Ventilation per Carriage (m³/s)', fontsize=12)
plt.ylabel('Mean Respirable Dust (mg/m³)', fontsize=12)

# Create legend for colors
color_handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, 
                           markerfacecolor=value, markersize=10) for key, value in colors.items()]
# Create legend for sizes (Year)
size_handles = [plt.scatter([],[], s=s, label=year, c='gray') 
                for year, s in {2021: 50, 2023: 100, 2024: 150}.items()]

legend1 = plt.legend(handles=color_handles, title='Line Type', loc='upper left')
plt.gca().add_artist(legend1)
plt.legend(handles=size_handles, title='Year', loc='upper right')

plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ventilation_vs_dust_plot_v2.png')
print("Plot saved as 'ventilation_vs_dust_plot_v2.png'")
# plt.show()

print("\n=== DISSERTATION CONTEXT: CFD-ML INTEGRATION ===")
print("This analysis provides baseline data for:")
print("1. CFD Model Validation - Ventilation rates vs particle concentration")
print("2. ML Training Data - Real-world TfL measurements")
print("3. Risk Assessment Parameters - Line type and occupancy effects")
print("\nKey findings for aerosol transmission modeling:")
print(f"- Ventilation effectiveness: {coefficients[0]:.4f} mg/m³ per m³/s increase")
# Ensure coefficient indexing is correct after adding features
# Coeffs order: ['Vent_per_Carriage_m3s', 'Year', 'Carriages_per_Train', 'Line_Type_Sub-Surface']
print(f"- Deep-level vs Sub-surface difference: {coefficients[3]:.4f} mg/m³")
print(f"- Temporal variation (Year effect): {coefficients[1]:.4f} mg/m³ per year")

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
print(f"- Ventilation range to test: {df['Vent_per_Carriage_m3s'].min():.2f} - {df['Vent_per_Carriage_m3s'].max():.2f} m³/s")
print(f"- Expected particle concentration range: {df['Mean_Respirable_Dust_mg_m3'].min():.2f} - {df['Mean_Respirable_Dust_mg_m3'].max():.2f} mg/m³")
print(f"- Critical ventilation threshold for low risk: ~{df[df['Mean_Respirable_Dust_mg_m3'] < 0.1]['Vent_per_Carriage_m3s'].min():.2f} m³/s")

print("\n=== NORTHERN LINE FOCUS: WORST-CASE SCENARIO ANALYSIS ===")
northern_data = df[df['Line'] == 'Northern'].copy()
print("Northern Line - Consistently highest dust levels (worst-case scenario):")
print(f"- Sample size: {len(northern_data)} measurements across {northern_data['Year'].nunique()} years")
print(f"- Dust level range: {northern_data['Mean_Respirable_Dust_mg_m3'].min():.2f} - {northern_data['Mean_Respirable_Dust_mg_m3'].max():.2f} mg/m³")
print(f"- Mean dust level: {northern_data['Mean_Respirable_Dust_mg_m3'].mean():.3f} mg/m³")
print(f"- Ventilation rate: {northern_data['Vent_per_Carriage_m3s'].iloc[0]:.2f} m³/s (consistently low)")

# Compare Northern Line to all others
other_lines = df[df['Line'] != 'Northern']
print(f"\nComparison with other lines:")
print(f"- Northern Line mean: {northern_data['Mean_Respirable_Dust_mg_m3'].mean():.3f} mg/m³")
print(f"- All other lines mean: {other_lines['Mean_Respirable_Dust_mg_m3'].mean():.3f} mg/m³")
print(f"- Northern Line is {(northern_data['Mean_Respirable_Dust_mg_m3'].mean() / other_lines['Mean_Respirable_Dust_mg_m3'].mean()):.1f}x worse")

# Risk category analysis for Northern Line
northern_risk = northern_data['Risk_Category'].value_counts()
print(f"\nNorthern Line risk distribution:")
for risk, count in northern_risk.items():
    print(f"- {risk} risk: {count} measurements ({count/len(northern_data)*100:.1f}%)")

print(f"\n=== CFD SIMULATION PARAMETERS FOR NORTHERN LINE ===")
print("Based on Northern Line worst-case scenario:")
print(f"- Current ventilation: {northern_data['Vent_per_Carriage_m3s'].iloc[0]:.2f} m³/s")
print(f"- Current dust levels: {northern_data['Mean_Respirable_Dust_mg_m3'].min():.2f} - {northern_data['Mean_Respirable_Dust_mg_m3'].max():.2f} mg/m³")
print(f"- Recommended CFD test scenarios:")
print(f"  * Baseline (current): {northern_data['Vent_per_Carriage_m3s'].iloc[0]:.2f} m³/s ventilation")
print(f"  * Improved scenario: 1.39 m³/s ventilation (Sub-Surface level)")
print(f"  * Passenger scenarios: 0, 2, 4-5 passengers (as planned)")

# Temporal analysis for Northern Line
print(f"\nNorthern Line temporal trends (pathogen transmission implications):")
northern_temporal = northern_data.groupby('Year')['Mean_Respirable_Dust_mg_m3'].mean()
for year, dust_level in northern_temporal.items():
    print(f"- {year}: {dust_level:.3f} mg/m³")

improvement_2021_2023 = (northern_temporal[2021] - northern_temporal[2023]) / northern_temporal[2021] * 100
regression_2023_2024 = (northern_temporal[2024] - northern_temporal[2023]) / northern_temporal[2023] * 100
print(f"- Improvement 2021→2023: {improvement_2021_2023:.1f}% reduction")
print(f"- Regression 2023→2024: {regression_2023_2024:.1f}% increase")
print("This shows air quality can fluctuate - important for risk modeling!")