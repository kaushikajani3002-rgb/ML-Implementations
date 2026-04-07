#PCA on Auto MPG Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Load dataset
df= pd.read_csv("D:/7013-DS/ML/auto-mpg.csv")

#Data Preprocessing
#Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

#Converts the horsepower column from string to numeric. Necessary because "?" values previously prevented it from being numeric.
df['horsepower'] = pd.to_numeric(df['horsepower'])

#drop rows with misssing values
df.dropna(inplace=True)

#PCA works only on numeric data. car name is text → dropped. The if statement avoids errors if the column doesn’t exist.
if 'car name' in df.columns:
    df.drop(columns=['car name'], inplace=True)

# Feature Selection
# Separate features and target (mpg)
X = df.drop(columns=['mpg'])
y = df['mpg']

#origin is categorical (1 = USA, 2 = Europe, 3 = Japan).
#Converts it into binary columns (dummy variables).
#drop_first=True prevents multicollinearity (dummy variable trap).
X = pd.get_dummies(X, columns=['origin'], drop_first=True)

#Feature scaling (Important for PCA)
scaler = StandardScaler()
#fit() → calculates mean & std. transform() → applies scaling
X_scaled = scaler.fit_transform(X)

#Apply PCA
#Example: reduce to 3 principal components
pca= PCA(n_components=3)

#fit(): find principal components, transform(): projects data into lower dimensional space.
X_pca = pca.fit_transform(X_scaled)

#Results
print("Original Shape: ", X_scaled.shape)
print("Reduced Shape", X_pca.shape)






















