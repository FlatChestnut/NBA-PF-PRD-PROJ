from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from data_collector import full_X, full_Y

i = 0
sum = 0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(full_X, full_Y, test_size=0.2, random_state=i*13)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(n_estimators=200, max_depth=10, learning_rate=0.02)
    model.fit(X_train_scaled, y_train)

    print(f"Accuracy: {model.score(X_test_scaled, y_test)}")
    sum+=model.score(X_test_scaled, y_test)

print(f"Average Accuracy: {sum/300}")