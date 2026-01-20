import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# PATHS
# =========================

# Dataset folder containing the extracted features CSV
ROOT_DIR = "dataset"
DATASET_FILE = os.path.join(ROOT_DIR, "dataset.csv")

# Output folder for plots
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================

# Load CSV containing extracted features and class labels
df = pd.read_csv(DATASET_FILE)

# Select all numeric feature columns (exclude label and segment index)
feature_cols = [col for col in df.columns if col not in ["label", "index"]]

X = df[feature_cols].values
y = df["label"].values

print("Number of features used:", len(feature_cols))
print("Features:", feature_cols)

# Encode class labels as integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# =========================
# SPLIT DATA
# =========================

# Train/Validation/Test split:
#  - 20% test
#  - from remaining 80%, 25% validation => 60% train, 20% val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# =========================
# FEATURE SCALING
# =========================

# Standardize features using training set statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# =========================
# MODEL DEFINITION
# =========================

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 256, 256),
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=32,
    learning_rate="constant",
    learning_rate_init=0.001,
    max_iter=300,
    shuffle=True,
    random_state=42,
    verbose=True,
)

# =========================
# MANUAL EPOCH TRAINING
# =========================

epoch_losses = []
epoch_val_scores = []

# Train using partial_fit to track loss/validation accuracy at each epoch
for epoch in range(mlp.max_iter):
    mlp.partial_fit(X_train, y_train, classes=np.unique(y))

    # Training loss (from MLP internal loss_)
    train_loss = mlp.loss_
    epoch_losses.append(train_loss)

    # Validation accuracy
    val_score = mlp.score(X_val, y_val)
    epoch_val_scores.append(val_score)

    print(
        f"Epoch {epoch + 1}/{mlp.max_iter} - Train Loss: {train_loss:.4f} - Val Acc: {val_score:.4f}"
    )

# =========================
# FINAL EVALUATION (TEST SET)
# =========================

test_accuracy = mlp.score(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy}\n")

y_test_pred = mlp.predict(X_test)

# Row-normalized confusion matrix (each row sums to 1)
conf_matrix_test = confusion_matrix(y_test, y_test_pred, normalize="true")

# =========================
# PLOT: CONFUSION MATRIX
# =========================

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    conf_matrix_test,
    annot=True,
    fmt=".2%",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    annot_kws={"size": 24},
    vmin=0,
    vmax=1,
)

# Increase colorbar tick size for readability
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=24)

plt.xlabel("Predicted Label", fontsize=24)
plt.ylabel("True Label", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# Save and show confusion matrix
cm_path_png = os.path.join(PLOTS_DIR, "confusion_matrix_test.png")
cm_path_pdf = os.path.join(PLOTS_DIR, "confusion_matrix_test.pdf")
plt.tight_layout()
plt.savefig(cm_path_png, dpi=300, bbox_inches="tight")
plt.savefig(cm_path_pdf, bbox_inches="tight")
plt.show()

print(f"Saved confusion matrix to: {cm_path_png}")
print(f"Saved confusion matrix to: {cm_path_pdf}")

# =========================
# CLASSIFICATION REPORT
# =========================

print("\nClassification Report - Test Set\n")
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

# =========================
# PLOT: TRAIN LOSS & VAL ACC
# =========================

plt.figure(figsize=(12, 5))
plt.plot(epoch_losses, label="Train Loss")
plt.plot(epoch_val_scores, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Training Loss & Validation Accuracy")
plt.legend()

# Save and show training curves
curves_path_png = os.path.join(PLOTS_DIR, "training_curves.png")
curves_path_pdf = os.path.join(PLOTS_DIR, "training_curves.pdf")
plt.tight_layout()
plt.savefig(curves_path_png, dpi=300, bbox_inches="tight")
plt.savefig(curves_path_pdf, bbox_inches="tight")
plt.show()

print(f"Saved training curves to: {curves_path_png}")
print(f"Saved training curves to: {curves_path_pdf}")
