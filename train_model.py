import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# -----------------------------
# 1. Load dataset
# -----------------------------
# You can get the dataset from: https://www.kaggle.com/datasets/ealaxi/paysim1
# File name: PS_20174392719_1491204439457_log.csv
print("📥 Loading dataset (PaySim synthetic transaction data)...")
df = pd.read_csv("./data/PS_20174392719_1491204439457_log.csv")

# -----------------------------
# 2. Keep relevant columns
# -----------------------------
cols = [
    'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'isFraud'
]
df = df[cols]

# -----------------------------
# 3. Encode and clean
# -----------------------------
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])  # e.g., TRANSFER, CASH_OUT, PAYMENT, etc.

# Replace NaN and inf
df = df.replace([float('inf'), float('-inf')], 0).fillna(0)

# -----------------------------
# 4. Split features/target
# -----------------------------
X = df.drop('isFraud', axis=1)
y = df['isFraud']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. Train model
# -----------------------------
print("🧠 Training XGBoost model...")
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate
# -----------------------------
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Save model and scaler
# -----------------------------
with open("./pickles/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("./pickles/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("./pickles/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\n✅ Model, scaler, and encoder saved successfully!")
