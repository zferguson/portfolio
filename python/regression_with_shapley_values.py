import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt

# Step 1: Simulate dataset
np.random.seed(42)
n = 3000
df = pd.DataFrame({
    'Tool_A_Adoption': np.random.binomial(1, 0.6, n),
    'Tool_B_Adoption': np.random.binomial(1, 0.4, n),
    'Training_Hours': np.random.normal(10, 2, n),
    'Client_Meetings': np.random.poisson(5, n),
    'Digital_Usage': np.random.uniform(0, 1, n),
    'Total_Assets': np.random.normal(5_000_000, 1_000_000, n)
})

# Step 2: Create target variable
df['Client_Satisfaction'] = (
    60
    + 10 * df['Tool_A_Adoption']
    + 5 * df['Tool_B_Adoption']
    + 0.8 * df['Training_Hours']
    + 1.2 * df['Client_Meetings']
    + 15 * df['Digital_Usage']
    + 0.000001 * df['Total_Assets']
    + np.random.normal(0, 5, n)
)

# Step 3: Define features and target
X = df.drop(columns='Client_Satisfaction')
y = df['Client_Satisfaction']

# Step 4: Scale Total_Assets only
scaler = StandardScaler()
X['Total_Assets'] = scaler.fit_transform(X[['Total_Assets']])

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 7: Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 8: SHAP values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Step 9: Visualize SHAP summary
shap.plots.beeswarm(shap_values)