import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

def load_data(pca_components=4, return_pca=True, filter_classes=False, threshold=100):
    print("Loading original data!")
    df = pd.read_excel("./data/Diagnostics.xlsx")

    print("Preprocessing data...")
    df_diagnostics = df.iloc[:, [1, 3, 4, 5, 6, 7, 12]].drop_duplicates()

    if filter_classes:
        print(f"Filtering classes with fewer than {threshold} samples...")
        label_counts = df_diagnostics['Rhythm'].value_counts()
        valid_labels = label_counts[label_counts >= threshold].index

        df_diagnostics = df_diagnostics[df_diagnostics['Rhythm'].isin(valid_labels)]
    else:
        print("Keeping original dataset with all classes :)")

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
        print(f"Returning dataset without PCA")
        return X_train_transf, X_test_transf, y_train, y_test, rhythm_le