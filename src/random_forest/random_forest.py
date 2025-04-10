from src.diagnostics_data import load_data, save_classification_report, plt_learning_curve
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report
    )

X_train, X_test, y_train, y_test, le = load_data(pca_components=4, return_pca=True)

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

save_classification_report(
    y_test, 
    y_pred, 
    le.classes_, 
    "random_forest/rf_report.csv", 
    )

plt_learning_curve(
    model=rf,
    X=X_train,
    y=y_train,
    title="Random Forest Learning Curve",
    scoring='accuracy',
    cv=3,
    train_sizes=np.linspace(0.1, 1.0, 10),
    path="random_forest/rf_learning_curve.png"
)