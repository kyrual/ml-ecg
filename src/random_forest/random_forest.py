from src.diagnostics_data import load_data, save_classification_report
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
    )

X_train, X_test, y_train, y_test, le = load_data(pca_components=4, return_pca=False)

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

save_classification_report(y_test, y_pred, le.classes_, "random_forest/rf_report.csv")
