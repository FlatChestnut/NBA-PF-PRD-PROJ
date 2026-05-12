import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from app import stats

loaded_model = joblib.load('saved_model.pkl')
scaler = joblib.load('scaler.pkl')

stats_scaled = scaler.transform([stats])

prediction = loaded_model.predict(stats_scaled)[0]
confidence = loaded_model.predict_proba(stats_scaled)[0]

print(f"DEBUG - Raw Probs: {confidence}")
print(f"DEBUG - Scaled Stats: {stats_scaled}")

if prediction == 1:
    print(f"Your team is predicted to win with a confidence of {confidence[1] * 100:.2f}%")
else:
    print(f"Your team is predicted to lose with a confidence of {confidence[0] * 100:.2f}%")