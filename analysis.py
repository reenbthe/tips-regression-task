import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# 2. Inspect & Preprocess
# Convert 'sex', 'smoker', 'day', 'time' into numeric dummy variables
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('tip', axis=1)
y = df_encoded['tip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"Model Performance:\nRMSE: {rmse:.4f}\nR2 Score: {r2:.4f}")

# 5. Visualisation
# Get coefficients to see which variable has the most weight
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='magma')
plt.title('Variable Influence on Tip Amount')
plt.tight_layout()
plt.savefig('visualisation.png')