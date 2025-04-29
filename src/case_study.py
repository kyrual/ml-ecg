from src.diagnostics_data import load_data
from src.reports import save_classification_report, plt_learning_curve
import pandas as pd
import numpy as np
import random
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

# adding random so that every demo is randomized
random_seed = random.randint(0, 42)
np.random.seed(random_seed)
random.seed(random_seed)
print(f"Random Seed: {random_seed}\n")

sample_classes = ['SB', 'AFIB', 'SA']
print(f"Selected Classes: {sample_classes}")

class_indices = [list(le.classes_).index(class_name) for class_name in sample_classes]

sampled_idxs = []
for idx in class_indices:
    class_sample_idxs = np.where(y_test == idx)[0]
    sampled = np.random.choice(class_sample_idxs, size=1, replace=False)
    sampled_idxs.extend(sampled)

X_sample = X_test[sampled_idxs]
y_true = y_test[sampled_idxs]
y_pred = rf.predict(X_sample)

true_labels = le.inverse_transform(y_true)
pred_labels = le.inverse_transform(y_pred)

for i in range(len(sampled_idxs)):
    print(f"Class   : {true_labels[i]}")
    print(f"Prediction: {pred_labels[i]}")

    probs = rf.predict_proba(X_sample)[i]
    print("Prediction Probabilities:")
    for j, prob in enumerate(probs):
        print(f"  {le.classes_[j]:<8}: {prob:.2f}")