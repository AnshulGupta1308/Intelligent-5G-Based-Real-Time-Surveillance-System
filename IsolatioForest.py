import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

X = np.load("normal_features.npy")

print("Training samples:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=400,
    contamination='auto',
    random_state=42
)

model.fit(X_scaled)

pickle.dump(model, open("iforest.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))

print("Model saved")