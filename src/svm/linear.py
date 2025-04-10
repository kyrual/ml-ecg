from src.diagnostics_data import load_data, save_classification_report, plt_learning_curve
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test, le = load_data(pca_components=4, return_pca=True)

c = 1.0 
svm = LinearSVC(C=1.0, max_iter=5000, class_weight='balanced') # balanced penalizes the model for getting it wrong
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

save_classification_report(
    y_test, 
    y_pred, 
    label_names=le.classes_, 
    filename="/svm/linear_report.csv", 
)

plt_learning_curve(
    model=svm,
    X=X_train,
    y=y_train,
    title="SVM(Linear) Learning Curve",
    scoring='balanced_accuracy',
    cv=3,
    train_sizes=np.linspace(0.1, 1.0, 50),
    path="svm/linear_learning_curve.png"
)