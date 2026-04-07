import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Load CSV file - We will call it 'df' so it matches the lines below
df = pd.read_csv("D:/7013-DS/ML/knn_dataset.csv")

# 2. Convert to numeric (errors='coerce' handles any weird text in the columns)
df['CGPA'] = pd.to_numeric(df['CGPA'], errors='coerce')
df['Communication'] = pd.to_numeric(df['Communication'], errors='coerce')
df['Aptitude'] = pd.to_numeric(df['Aptitude'], errors='coerce')
df['Pro_skill'] = pd.to_numeric(df['Pro_skill'], errors='coerce')
df['Job_offered'] = pd.to_numeric(df['Job_offered'], errors='coerce')

# Drop any rows that failed to convert (NaN values)
df = df.dropna()

# 3. Define X (Features) and y (Target)
# We must define these BEFORE we use them in get_dummies or scaling
X = df.drop('Job_offered', axis=1) 
y = df['Job_offered']

# 4. Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# 5. Scale features (KNN needs this!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=0
)

# 7. Train and evaluate KNN
for k in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    print(f"\n--- K = {k} ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy: {accuracy:.2f} %")
