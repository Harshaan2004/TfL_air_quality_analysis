import pandas as pd
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Your Data
# I've pasted the exact CSV string you provided.
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

# Read the string data into a pandas DataFrame
df = pd.read_csv(io.StringIO(csv_data))

print("--- Data Loaded Successfully ---")
print(f"Loaded {len(df)} data points (rows).")
print(df.head())
print("\n")

# 2. Define Features (X) and Target (y)
# These are the columns your project plan aims to correlate
features = ['Vent_per_Carriage_m3s', 'Line_Type']
target = 'Mean_Respirable_Dust_mg_m3'

X = df[features]
y = df[target]

# 3. Preprocessing: Handle Categorical Data
# We need to convert 'Line_Type' (text) into numbers (0 or 1)
# 'Vent_per_Carriage_m3s' is numerical, so we'll 'passthrough'
categorical_features = ['Line_Type']
numerical_features = ['Vent_per_Carriage_m3s']

# Create a 'preprocessor' to handle the different feature types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features) # drop='first' avoids dummy variable trap
    ])

# 4. Create and Train the Model
# We create a 'pipeline' to chain the preprocessing and the regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Because the dataset is very small (N=24), we will train on all of it
# to understand the *existing* relationships.
# A train/test split would create a test set too small to be meaningful.
model.fit(X, y)

# 5. Assess the Model's Performance
print("--- Model Assessment ---")

# Make predictions on the same data to see how well the model fits
y_pred = model.predict(X)

# R-squared: "How much of the variation in dust does our model explain?"
# 1.0 is a perfect fit.
r2 = r2_score(y, y_pred)
print(f"Model R-squared (Goodness of Fit): {r2:.4f}")

# Root Mean Squared Error (RMSE): "How wrong is the model, on average?"
# Lower is better. Units are the same as the target (mg/m³).
rmse = mean_squared_error(y, y_pred) ** 0.5
print(f"Model RMSE (Average Error): {rmse:.4f} mg/m³")
print("\n")

# 6. Analyze Feature Importance (The most important part!)
print("--- Model Coefficients (Feature Importance) ---")

# Get the names of the features *after* preprocessing
# (e.g., 'Line_Type' becomes 'Line_Type_Sub-Surface')
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coefficients = model.named_steps['regressor'].coef_
intercept = model.named_steps['regressor'].intercept_

print(f"Model Base Intercept: {intercept:.4f}")
print("This is the 'starting' dust level before considering the features.\n")

print("Feature Coefficients:")
for name, coef in zip(feature_names, coefficients):
    # This cleans up the default scikit-learn names for readability
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

# Create a scatter plot of your main feature vs. your target
# Color the dots by the 'Line_Type' to see the difference
colors = {'Deep-Level': 'red', 'Sub-Surface': 'blue'}
plt.scatter(df['Vent_per_Carriage_m3s'], df['Mean_Respirable_Dust_mg_m3'], 
            c=df['Line_Type'].map(colors), alpha=0.7)

plt.title('Ventilation Rate vs. Mean Respirable Dust in Driver Cabs', fontsize=16)
plt.xlabel('Ventilation per Carriage (m³/s)', fontsize=12)
plt.ylabel('Mean Respirable Dust (mg/m³)', fontsize=12)

# Create a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', label=key, 
                      markerfacecolor=value, markersize=10) for key, value in colors.items()]
plt.legend(handles=handles, title='Line Type')

plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('ventilation_vs_dust_plot.png')
print("Plot saved as 'ventilation_vs_dust_plot.png'")
# plt.show() # Use this if running in a local environment