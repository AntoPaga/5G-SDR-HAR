import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


# =========================
# PATHS
# =========================

# Folder containing the dataset CSV
ROOT_DIR = "dataset"
DATASET_FILE = os.path.join(ROOT_DIR, "dataset.csv")

# Output folder for plots
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================
# LOAD & CLEAN DATA
# =========================

# Load dataset and sanitize column names
df = pd.read_csv(DATASET_FILE)
df.columns = df.columns.str.strip()

# Select feature columns (exclude label and segment index)
feature_cols = [c for c in df.columns if c not in ["label", "index"]]

# Ensure features are numeric; coerce errors to NaN
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

# Remove invalid values (inf/-inf) and drop rows with missing feature/label values
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["label"])

X = df[feature_cols].values
y = df["label"].astype(str).values

# Encode labels as integers
le = LabelEncoder()
y = le.fit_transform(y)

# =========================
# TRAIN / VAL / TEST SPLIT
# =========================

# 20% test, then 25% of remaining 80% as validation
# => 60% train, 20% val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# =========================
# MODEL: SVM (RBF) + STANDARDIZATION
# =========================

# Use a pipeline to avoid data leakage:
# scaling is fit only on training data and applied consistently to val/test
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced"
    ))
])

# Train the model
model.fit(X_train, y_train)

# =========================
# VALIDATION EVALUATION
# =========================

val_pred = model.predict(X_val)
print("Val Accuracy:", accuracy_score(y_val, val_pred))
print("Val Macro-F1 :", f1_score(y_val, val_pred, average="macro"))

# =========================
# TEST EVALUATION
# =========================

test_pred = model.predict(X_test)
acc = accuracy_score(y_test, test_pred)
f1m = f1_score(y_test, test_pred, average="macro")

print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Macro-F1:  {f1m:.4f}\n")
print(classification_report(y_test, test_pred, target_names=le.classes_))

# =========================
# CONFUSION MATRIX (ROW-NORMALIZED)
# =========================

cm = confusion_matrix(y_test, test_pred, normalize="true")

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    cm,
    annot=True,
    fmt=".2%",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    annot_kws={"size": 24},
    vmin=0,
    vmax=1,
)

# Make colorbar ticks readable
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=24)

plt.xlabel("Predicted Label", fontsize=24)
plt.ylabel("True Label", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# Save and show confusion matrix
cm_path_png = os.path.join(PLOTS_DIR, "svm_confusion_matrix_test.png")
cm_path_pdf = os.path.join(PLOTS_DIR, "svm_confusion_matrix_test.pdf")
plt.tight_layout()
plt.savefig(cm_path_png, dpi=300, bbox_inches="tight")
plt.savefig(cm_path_pdf, bbox_inches="tight")
plt.show()

print(f"Saved confusion matrix to: {cm_path_png}")
print(f"Saved confusion matrix to: {cm_path_pdf}")
