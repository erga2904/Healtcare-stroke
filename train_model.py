# train_model.py (versi diperbaiki)
import os
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix

# --- Pastikan folder 'static' ada ---
os.makedirs('static', exist_ok=True)

# --- Muat dataset ---
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.drop('id', axis=1)

# --- Perbaiki cara handle missing value ---
# Gunakan cara yang aman di pandas >=2.0
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# --- Encode fitur kategorikal ---
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Handle nilai 'Other' atau NaN dengan mengisi modus jika perlu
    df[col] = df[col].fillna(df[col].mode()[0])  # isi NaN dengan modus
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --- Pisahkan fitur dan target ---
X = df.drop('stroke', axis=1)
y = df['stroke']

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Latih model Decision Tree (dengan pruning) ---
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# --- Simpan artefak ---
joblib.dump(model, 'stroke_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
joblib.dump(X.columns.tolist(), 'feature_names.joblib')

# === SIMPAN VISUALISASI ===

# 1. Distribusi target
plt.figure(figsize=(6, 4))
sns.countplot(x='stroke', data=df)
plt.title('Distribusi Pasien: Stroke (1) vs Tidak Stroke (0)')
plt.xlabel('stroke')
plt.ylabel('Jumlah')
plt.savefig('static/target_distribution.png')
plt.close()

# 2. Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Tidak Stroke', 'Stroke'],
    yticklabels=['Tidak Stroke', 'Stroke']
)
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.savefig('static/confusion_matrix.png')
plt.close()

# 3. Feature Importance
importances = model.feature_importances_
feat_imp = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': importances}).sort_values('Pentingnya', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Pentingnya', y='Fitur', data=feat_imp)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()

# 4. Visualisasi Pohon
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Tidak Stroke', 'Stroke'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree (max_depth=4)')
plt.savefig('static/decision_tree.png', bbox_inches='tight')
plt.close()

print("✅ Model dan visualisasi berhasil disimpan di folder 'static/'!")