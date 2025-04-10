from src.diagnostics_data import load_data, save_classification_report, plt_learning_curve
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# KNN classifier configuration
n = 7           # 3 neighbors
d = 'minkowski' # minkowski distance
p = 1           # power of 2 in minkowski distance (a.k.a, euclidean distance)
algo = 'auto'  # brute force sample comparison

X_train, X_test, y_train, y_test, le = load_data(pca_components=4, return_pca=True)

knn = KNeighborsClassifier(n_neighbors=n, metric=d, p=p, algorithm=algo)
knn.fit(X_train, y_train.ravel())

y_pred = knn.predict(X_test)

save_classification_report(
    y_test, 
    y_pred, 
    label_names=le.classes_, 
    filename="/knn/knn_report.csv", 
)

plt_learning_curve(
    model=knn,
    X=X_train,
    y=y_train,
    title="KNN Learning Curve",
    scoring='balanced_accuracy',
    cv=3,
    train_sizes=np.linspace(0.1, 1.0, 50),
    path="random_forest/rf_learning_curve.png"
)