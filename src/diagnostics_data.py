import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

def load_data(pca_components=4, return_pca=True):
    print("Loading original data!")
    df = pd.read_excel("./data/Diagnostics.xlsx")

    print("Preprocessing data...")
    df_diagnostics = df.iloc[:, [1, 3, 4, 5, 6, 7, 12]].drop_duplicates()

    rhythm_le = LabelEncoder()
    y = rhythm_le.fit_transform(df_diagnostics['Rhythm'])

    num_cols = ['PatientAge', 'VentricularRate', 'AtrialRate', 'QRSDuration', 'QRSCount']
    feature_cols = num_cols + ['Gender']
    X = df_diagnostics[feature_cols].copy()
    X[num_cols] = X[num_cols].astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=0,
        stratify=y
    )

    preprocessor = ColumnTransformer([
        ('onehot_gender', OneHotEncoder(drop='first', sparse_output=False), ['Gender']),
        ('scale_numeric', StandardScaler(), num_cols)
    ])

    X_train_transf = preprocessor.fit_transform(X_train)
    X_test_transf = preprocessor.transform(X_test)

    if return_pca:
        print(f"Applying PCA with {pca_components} components")
        pca = PCA(n_components=pca_components)
        X_train_pca = pca.fit_transform(X_train_transf)
        X_test_pca = pca.transform(X_test_transf)

        print("Total variance retained:", round(np.sum(pca.explained_variance_ratio_), 4))

        return X_train_pca, X_test_pca, y_train, y_test, rhythm_le

    else:
        return X_train_transf, X_test_transf, y_train, y_test, rhythm_le
    
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