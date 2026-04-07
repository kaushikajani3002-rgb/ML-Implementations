import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset (FIXED PATH)
df = pd.read_csv("D:/7013-DS/ML/auto-mpg.csv")

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert horsepower to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'])

# Drop missing values
df = df.dropna()

# Drop non-numeric column
if 'car name' in df.columns:
    df = df.drop(columns=['car name'])

# Define features and target
X = df.drop(columns=['mpg'])
y = df['mpg']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

remaining_features = list(X.columns)
selected_features = []
best_score = -np.inf

print("Forward Feature Selection Progress:\n")

while remaining_features:
    scores = []

    for feature in remaining_features:
        features_to_test = selected_features + [feature]

        model = LinearRegression()
        model.fit(X_train[features_to_test], y_train)

        y_pred = model.predict(X_test[features_to_test])
        score = r2_score(y_test, y_pred)

        scores.append((score, feature))

    # Sort by highest R2
    scores.sort(reverse=True)
    current_best_score, best_feature = scores[0]

    if current_best_score > best_score:
        best_score = current_best_score
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        print(f"Added: {best_feature}, R2 score: {best_score:.4f}")
    else:
        break

print("\nFinal Selected Features:")
print(selected_features)
