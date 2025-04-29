from src.diagnostics_data import load_data
from src.reports import save_classification_report, plt_learning_curve
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test, le = load_data(pca_components=4, return_pca=False, filter_classes=True, threshold=100)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=12,
    min_samples_leaf=8,
    max_features=0.2,
    random_state=0
    )
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

save_classification_report(
    y_test, 
    y_pred, 
    le.classes_, 
    "random_forest/rf_report_pca4.csv", 
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