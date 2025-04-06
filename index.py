import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load the dataset
df = pd.read_csv('cleaned_dataset_draft4.csv')

# Separate features and target
X = df.drop('readmitted_binary', axis=1)
y = df['readmitted_binary']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define column groups
diag_cols = ['diag_1', 'diag_2', 'diag_3']
numerical_cols = ['time_in_hospital', 'num_medications', 'number_diagnoses', 'age_encoded']
categorical_cols = ['race', 'gender', 'admission_type_id', 'discharge_disposition_id', 'insulin']
binary_cols = ['change', 'diabetesMed'] 

# Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ('target_encoder', TargetEncoder(cols=diag_cols)),
    ('preprocessor', ColumnTransformer([
        ('one_hot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('scaler', StandardScaler(), numerical_cols + diag_cols)  # Include target-encoded diag cols
    ], remainder='passthrough')),  # Pass through binary cols
    ('classifier', LogisticRegression(
        class_weight='balanced',  # Handles class imbalance
        random_state=42,
        max_iter=1000  # Ensure convergence
    ))
])

# Train Logistic Regression
lr_pipeline.fit(X_train, y_train)

# Evaluate Logistic Regression
y_pred_lr = lr_pipeline.predict(X_test)
y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")
print(classification_report(y_test, y_pred_lr))

# Random Forest Pipeline
rf_pipeline = Pipeline([
    ('target_encoder', TargetEncoder(cols=diag_cols)),
    ('preprocessor', ColumnTransformer([
        ('one_hot', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ], remainder='passthrough')),  # Pass through numerical and binary cols
    ('classifier', RandomForestClassifier(
        class_weight='balanced',  # Handles class imbalance
        random_state=42,
        n_jobs=-1  # Use all cores
    ))
])

# Train Random Forest
rf_pipeline.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rf_pipeline.predict(X_test)
y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")
print(classification_report(y_test, y_pred_rf))




import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

plt.style.use('seaborn-v0_8') 

# ================== ROC Curves ==================
plt.figure(figsize=(10, 6))

# Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2,
         label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, color='green', lw=2,
         label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

# Baseline
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.show()

# ================== Confusion Matrices ==================
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Oranges', ax=ax[0],
           xticklabels=['Not Readmitted', 'Readmitted'],
           yticklabels=['Not Readmitted', 'Readmitted'])
ax[0].set_title('Logistic Regression')

# Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax[1],
           xticklabels=['Not Readmitted', 'Readmitted'],
           yticklabels=['Not Readmitted', 'Readmitted'])
ax[1].set_title('Random Forest')

plt.tight_layout()
plt.show()

# ================== Precision-Recall Curves ==================
plt.figure(figsize=(10, 6))

# Logistic Regression
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_proba_lr)
pr_auc_lr = auc(recall_lr, precision_lr)
plt.plot(recall_lr, precision_lr, color='darkorange',
         label=f'Logistic Regression (AUC = {pr_auc_lr:.2f})')

# Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_proba_rf)
pr_auc_rf = auc(recall_rf, precision_rf)
plt.plot(recall_rf, precision_rf, color='green',
         label=f'Random Forest (AUC = {pr_auc_rf:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc="upper right")
plt.show()