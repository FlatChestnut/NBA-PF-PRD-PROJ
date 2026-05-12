from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler   
from sklearn.calibration import CalibratedClassifierCV 
import joblib
from xgboost import XGBClassifier

from data_collector import full_X, full_Y


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(full_X)

model = XGBClassifier(
    n_estimators=200,      # number of trees
    learning_rate=0.05,    # step size per tree
    max_depth=4,           # how deep each tree can go  
    random_state=42
)
model.fit(X_train_scaled, full_Y)


joblib.dump(model, 'saved_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully!")
