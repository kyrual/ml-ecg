from src.diagnostics_data import load_data
from src.reports import save_classification_report, plt_learning_curve
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, f1_score

X_train, X_test, y_train, y_test, le = load_data(pca_components=4, return_pca=False, filter_classes=True, threshold=100)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    min_samples_split=6,
    min_samples_leaf=2,
    max_features=0.3,
    class_weight='balanced',
    random_state=0
    )
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

save_classification_report(
    y_test, 
    y_pred, 
    le.classes_, 
    "random_forest/rf_report_v3.csv", 
    )

plt_learning_curve(
    model=rf,
    X=X_train,
    y=y_train,
    title="Random Forest Learning Curve",
    scoring='accuracy',
    cv=3,
    train_sizes=np.linspace(0.1, 1.0, 10),
    path="random_forest/rf_learning_curve_v3.png"
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # or 'weighted'
f1 = f1_score(y_test, y_pred, average='macro')  # or 'weighted'

print(f"🎯 Accuracy:  {accuracy:.4f}")
print(f"💡 Precision: {precision:.4f}")
print(f"🌟 F1-score:  {f1:.4f}")

from sklearn.metrics import balanced_accuracy_score
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"⚖️ Balanced Accuracy: {balanced_acc:.4f}")