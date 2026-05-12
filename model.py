import joblib
from app import stats

loaded_model = joblib.load('saved_model.pkl')
loaded_model().predict([stats])