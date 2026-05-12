from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler   
from sklearn.calibration import CalibratedClassifierCV 
import joblib

from data_collector import full_X, full_Y


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(full_X)

model = CalibratedClassifierCV(LogisticRegression(), cv=5, method='sigmoid')
model.fit(X_train_scaled, full_Y)


joblib.dump(model, 'saved_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully!")
