import os
import random
import numpy as np
import pandas as pd

# Set random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
file_path = r"C:\Users\HP\OneDrive\Desktop\epilipsy project\dataset1.xlsx"
df = pd.read_excel(file_path)

# Handle missing values (fill with column mean)
if df.isnull().values.any():
    df.fillna(df.mean(), inplace=True)

# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode target labels if categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = y.astype(int)

# Print unique labels and check for class imbalance
unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# Compute class weights for imbalanced data
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Build the model
def build_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model(X_train.shape[1], num_classes)

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train_cat,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test_cat),
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=1)
print(f'Test accuracy: {test_acc}')

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_cat, axis=1)

# Accuracy
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f'Accuracy: {accuracy}')

# Classification report (precision, recall, F1-score)
target_names = [str(cls) for cls in label_encoder.classes_]
print(classification_report(y_test_classes, y_pred_classes, target_names=target_names))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test_classes, y_pred_classes))

# K-fold Cross-Validation with Logistic Regression for baseline
from sklearn.linear_model import LogisticRegression

kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
clf = LogisticRegression(max_iter=500, random_state=SEED)
cv_scores = []
for train_idx, test_idx in kfold.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    clf.fit(X_tr, y_tr)
    score = clf.score(X_te, y_te)
    cv_scores.append(score)
print(f'K-Fold Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean CV Accuracy: {np.mean(cv_scores)}')

# --- Model Explainability with SHAP (DeepExplainer) ---
import shap

# Use a small subset of background and test samples for memory efficiency
background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_test[:5])

# Visualize summary plot (run in Jupyter for interactive plot)
shap.summary_plot(shap_values, X_test[:5], feature_names=df.columns[:-1])

# --- Model Explainability with LIME ---
import lime
import lime.lime_tabular

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, feature_names=df.columns[:-1], class_names=target_names, discretize_continuous=True
)
i = 0  # Index of test sample to explain
exp = lime_explainer.explain_instance(X_test[i], model.predict, num_features=10)
exp.show_in_notebook(show_table=True)

# (Optional) Save model and scaler for deployment
# model.save('epilepsy_model.h5')
# import joblib
# joblib.dump(scaler, 'scaler.save')
