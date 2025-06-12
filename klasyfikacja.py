import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def liczCONFMATBIS(C):
    TP = np.diag(C)
    FP = np.sum(C, axis=0) - TP
    FN = np.sum(C, axis=1) - TP
    TN = np.sum(C) - (FP + FN + TP)

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (TN + FN)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    return {"TPR": TPR, "TNR": TNR, "PPV": PPV, "NPV": NPV, "ACC": ACC, "MCC": MCC}

iris = load_iris(as_frame=True)
df = iris.frame
print("Statystyki opisowe:\n", df.describe())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['sepal length (cm)'], df['sepal width (cm)'],
                df['petal length (cm)'], c=df['target'], cmap='viridis')
ax.set_xlabel('długość działki kielicha (cm)')
ax.set_ylabel('szerokość działki kielicha (cm)')
ax.set_zlabel('długość płatka (cm)')
plt.title("Wykres 3D cech Iris")
plt.show()
plt.close()

# === Wykres pudełkowy===
df = df.rename(columns={
    'sepal length (cm)': 'długość działki kielicha (cm)',
    'sepal width (cm)': 'szerokość działki kielicha (cm)',
    'petal length (cm)': 'długość płatka (cm)',
    'petal width (cm)': 'szerokość płatka (cm)'
})
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.iloc[:, :-1])
plt.title("Wykres pudełkowy cech Iris")
plt.show()
plt.close()

X = df.iloc[:, :-1]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

rf_params = {
    "n_estimators": 11,
    "max_depth": 5,
    "criterion": "gini",
    "random_state": 42
}

model = RandomForestClassifier(**rf_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

main_acc = accuracy_score(y_test, y_pred)
main_f1 = f1_score(y_test, y_pred, average='macro')
main_cm = pd.crosstab(y_test, y_pred, rownames=['Rzeczywiste'], colnames=['Przewidziane']).to_numpy()
main_metryki = liczCONFMATBIS(main_cm)

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names)
plt.title("Macierz pomyłek - Random Forest")
plt.show()
plt.close()

print(f"Accuracy: {main_acc:.2f}")
print(f"F1-score: {main_f1:.2f}")
print(f"TPR (czułość): {main_metryki['TPR']}")
print(f"TNR (specyficzność): {main_metryki['TNR']}")
print(f"MCC: {main_metryki['MCC']}")

# === Test 3 Konfiguracji - RANDOM FOREST ===
rf_configs = [
    {"n_estimators": 12, "max_depth": 5, "criterion": "gini"},
    {"n_estimators": 6, "max_depth": 15, "criterion": "entropy"},
    {"n_estimators": 200, "max_depth": None, "criterion": "gini"},
    #   drzewa              głebokość           kryterium
]

results_txt = "=== Statystyki Random Forest ===\n\n"
results_txt += "Konfiguracja: Główna\n"
results_txt += f"Parametry: {rf_params}\n"
results_txt += f"Accuracy: {main_acc:.2f}\n"
results_txt += f"F1-score: {main_f1:.2f}\n"
results_txt += f"TPR (czułość): {main_metryki['TPR'].round(2)}\n"
results_txt += f"TNR (specyficzność): {main_metryki['TNR'].round(2)}\n"
results_txt += f"MCC: {main_metryki['MCC'].round(2)}\n\n"

print("\n===== RANDOM FOREST - RÓŻNE KONFIGURACJE =====\n")
for i, params in enumerate(rf_configs):
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = pd.crosstab(y_test, y_pred, rownames=['Rzeczywiste'], colnames=['Przewidziane']).to_numpy()
    metryki = liczCONFMATBIS(cm)

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names)
    plt.title(f"Macierz pomyłek - Konfiguracja {i+1}")
    plt.show()
    plt.close()

    print(f"\nKonfiguracja {i+1}: {params}")
    print(f"Accuracy: {acc:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"TPR: {metryki['TPR']}")
    print(f"TNR: {metryki['TNR']}")
    print(f"MCC: {metryki['MCC']}")

    results_txt += f"Konfiguracja {i+1}: {params}\n"
    results_txt += f"Accuracy: {acc:.2f}\n"
    results_txt += f"F1-score: {f1:.2f}\n"
    results_txt += f"TPR: {metryki['TPR'].round(2)}\n"
    results_txt += f"TNR: {metryki['TNR'].round(2)}\n"
    results_txt += f"MCC: {metryki['MCC'].round(2)}\n\n"

with open("statystyki.txt", "w", encoding="utf-8") as f:
    f.write(results_txt)

# === MODELE SVM ===
svm_kernels = ['linear', 'rbf', 'poly']
results_txt += "\n=== Statystyki SVM ===\n\n"
print("\n===== SVM - RÓŻNE JĄDRA =====\n")

for kernel in svm_kernels:
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = pd.crosstab(y_test, y_pred, rownames=['Rzeczywiste'], colnames=['Przewidziane']).to_numpy()
    metryki = liczCONFMATBIS(cm)

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names)
    plt.title(f"Macierz pomyłek - SVM (kernel: {kernel})")
    plt.show()
    plt.close()

    print(f"\nSVM (kernel: {kernel})")
    print(f"Accuracy: {acc:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"TPR: {metryki['TPR']}")
    print(f"TNR: {metryki['TNR']}")
    print(f"MCC: {metryki['MCC']}")

    results_txt += f"SVM (kernel: {kernel})\n"
    results_txt += f"Accuracy: {acc:.2f}\n"
    results_txt += f"F1-score: {f1:.2f}\n"
    results_txt += f"TPR: {metryki['TPR'].round(2)}\n"
    results_txt += f"TNR: {metryki['TNR'].round(2)}\n"
    results_txt += f"MCC: {metryki['MCC'].round(2)}\n\n"

# === MODELE KNN ===
knn_values = [3, 5, 7]
results_txt += "\n=== Statystyki KNN ===\n\n"
print("\n===== KNN - RÓŻNE WARTOŚCI k =====\n")

for k in knn_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = pd.crosstab(y_test, y_pred, rownames=['Rzeczywiste'], colnames=['Przewidziane']).to_numpy()
    metryki = liczCONFMATBIS(cm)

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=iris.target_names)
    plt.title(f"Macierz pomyłek - KNN (k={k})")
    plt.show()
    plt.close()

    print(f"\nKNN (k={k})")
    print(f"Accuracy: {acc:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"TPR: {metryki['TPR']}")
    print(f"TNR: {metryki['TNR']}")
    print(f"MCC: {metryki['MCC']}")

    results_txt += f"KNN (k={k})\n"
    results_txt += f"Accuracy: {acc:.2f}\n"
    results_txt += f"F1-score: {f1:.2f}\n"
    results_txt += f"TPR: {metryki['TPR'].round(2)}\n"
    results_txt += f"TNR: {metryki['TNR'].round(2)}\n"
    results_txt += f"MCC: {metryki['MCC'].round(2)}\n\n"

with open("statystyki.txt", "w", encoding="utf-8") as f:
    f.write(results_txt)
