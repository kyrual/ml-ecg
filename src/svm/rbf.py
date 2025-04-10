from src.diagnostics_data import load_data, save_classification_report, plt_learning_curve
import pandas as pd
import numpy as np
from sklearn.svm import SVC

X_train, X_test, y_train, y_test, le = load_data(pca_components=4, return_pca=True)

# SVM classifier configuration
phi = 'rbf' 
c = 1.0 

svm = SVC(kernel=phi, decision_function_shape='ovr', C=c)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

save_classification_report(
    y_test, 
    y_pred, 
    label_names=le.classes_, 
    filename="/svm/rbf_report.csv", 
)

plt_learning_curve(
    model=svm,
    X=X_train,
    y=y_train,
    title="SVM(RBF) Learning Curve",
    scoring='balanced_accuracy',
    cv=3,
    train_sizes=np.linspace(0.1, 1.0, 50),
    path="svm/rbf_learning_curve.png"
)