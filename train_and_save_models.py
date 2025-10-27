# train_and_save_models.py
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Buat folder static jika belum ada
os.makedirs('static', exist_ok=True)

# === 1. Muat dan Pra-pemrosesan Data ===
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.drop('id', axis=1)

# Handle missing value pada 'bmi'
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Encode fitur kategorikal
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna(df[col].mode()[0])  # isi NaN dengan modus
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Pisahkan fitur dan target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 2. Latih Dua Model ===
# Model A: Overfit (tanpa batasan)
model_overfit = DecisionTreeClassifier(random_state=42)
model_overfit.fit(X_train, y_train)

# Model B: Pruned (dengan max_depth=4)
model_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
model_pruned.fit(X_train, y_train)

# Evaluasi
print("=== Model Overfit ===")
print(classification_report(y_test, model_overfit.predict(X_test)))

print("\n=== Model Pruned ===")
print(classification_report(y_test, model_pruned.predict(X_test)))

# Simpan model terbaik (pruned)
joblib.dump(model_pruned, 'best_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
joblib.dump(X.columns.tolist(), 'feature_names.joblib')

# === 3. Simpan Visualisasi ===

# Distribusi target
plt.figure(figsize=(6, 4))
sns.countplot(x='stroke', data=df)
plt.title('Distribusi Stroke (0=Tidak, 1=Ya)')
plt.savefig('static/target_distribution.png')
plt.close()

# Confusion Matrix (model pruned)
y_pred = model_pruned.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tidak Stroke', 'Stroke'],
            yticklabels=['Tidak Stroke', 'Stroke'])
plt.title('Confusion Matrix (Model Pruned)')
plt.savefig('static/confusion_matrix.png')
plt.close()

# Feature Importance
importances = model_pruned.feature_importances_
feat_imp = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': importances}).sort_values('Pentingnya', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Pentingnya', y='Fitur', data=feat_imp, palette='viridis')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()

# Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(
    model_pruned,
    feature_names=X.columns,
    class_names=['Tidak Stroke', 'Stroke'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree (max_depth=4)')
plt.savefig('static/decision_tree.png', bbox_inches='tight')
plt.close()

print("✅ Semua model dan visualisasi berhasil disimpan!")
