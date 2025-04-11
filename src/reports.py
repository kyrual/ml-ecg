import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def save_classification_report(y_test, y_pred, label_names=None, filename="filename.csv", output_dir="./src/"):
    report_dict = classification_report(
        y_test, y_pred, 
        target_names=label_names, 
        output_dict=True,
        zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().round(2)

    column_order = ['precision', 'recall', 'f1-score', 'support']
    report_df = report_df[column_order]

    summary_rows = ['accuracy', 'macro avg', 'weighted avg']
    class_rows = [i for i in report_df.index if i not in summary_rows]
    report_df = report_df.loc[class_rows + summary_rows]

    report_df = report_df.reset_index()
    report_df = report_df.rename(columns={'index': 'class'})

    report_df.to_csv(f'{output_dir}/{filename}', index=False)

def plt_learning_curve(model, X, y, title="title", scoring='accuracy', cv=5, train_sizes=np.linspace(0.1, 1.0, 10), path=""):
    train_sizes, train_scores, valid_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        shuffle=True,
        random_state=0
    )

    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)

    train_std = np.std(train_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'r-', label='train')
    plt.plot(train_sizes, valid_mean, 'b-', label='validation')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid()

    plt.savefig(f'./src/{path}')
    plt.show()

def plot_predictions(pred, gt=None, title=""):
    return 