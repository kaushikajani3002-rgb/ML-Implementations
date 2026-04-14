import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. Load and clean data
df = pd.read_csv("auto-mpg.csv")

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert horsepower to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'])

# Drop missing values
df = df.dropna()

# Remove non-numeric column
if 'car name' in df.columns:
    df = df.drop(columns=['car name'])


#2. define features and target
X= df.drop(columns=['mpg'])
y= df['mpg']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Backward feature selection
selected_features = list(X.columns)

#Train full model first
model = LinearRegression()
model.fit(X_train[selected_features], y_train)
y_pred = model.predict(X_test[selected_features])

best_score = r2_score(y_test, y_pred)

print("Backward Feature Elimination Process: \n")
print(f"Initial R2 score (All Features): {best_score:.4f} \n")

while len(selected_features) > 1:
    scores = []

    for feature in selected_features:
        features_to_test = selected_features.copy()
        features_to_test.remove(feature)

        model= LinearRegression()
        model.fit(X_train[features_to_test], y_train)

        y_pred = model.predict(X_test[features_to_test])
        score = r2_score(y_test, y_pred)

        scores.append((score, feature))

    #find removal that gives highest r2
    scores.sort(reverse=True)
    current_best_score, worst_feature = scores[0]

    if current_best_score >= best_score:
        best_score = current_best_score
        selected_features.remove(worst_feature)
        
        print(f"Removed: {worst_feature}, R2 Score: {best_score:.4f}")
    else:
        break

print("\n Final Selected Features: ")
print(selected_features)















