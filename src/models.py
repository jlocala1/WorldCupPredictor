import joblib
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
FEATURES_PATH = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'features.csv'

FEATURE_FILTERS = ['form_', 'h2h_', 'elo', 'fifa_', 'squad_', 'top26_',
                   'avg_value', 'caps', '_z', 'neutral']
TARGET_NAMES = ['away_win', 'draw', 'home_win']


def prepare_data(df):
    features = [c for c in df.columns if any(x in c for x in FEATURE_FILTERS)]
    print(f"Selected {len(features)} features: {sorted(features)}\n")

    train_df = df[df['split'] == 'train'].dropna(subset=['label'])
    val_df = df[df['split'] == 'val'].dropna(subset=['label'])

    X_train, Y_train = train_df[features], train_df['label']
    X_val, Y_val = val_df[features], val_df['label']

    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)

    return (
        X_train_scaled,
        X_val_scaled,
        Y_train,
        Y_val,
        features,
        imputer,
        scaler,
    )

def train_logistic_regression(X_train, Y_train):
    model = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42, class_weight='balanced')
    model.fit(X_train, Y_train)
    return model

def evaluate_model(model, X_val, Y_val, features):
    probs = model.predict_proba(X_val)
    preds = model.predict(X_val)
    print(f"Validation Log Loss: {log_loss(Y_val, probs):.4f}")
    print(classification_report(Y_val, preds))

    home_win_idx = list(model.classes_).index(2.0)
    elo_idx = features.index('home_elo') if 'home_elo' in features else None
    if elo_idx is not None:
        print(f"Elo weight for home win: {model.coef_[home_win_idx][elo_idx]:.4f}")
    importance = pd.DataFrame({'Feature': features, 'Weight': model.coef_[home_win_idx]})
    print(importance.sort_values(by='Weight', ascending=False).head(5).to_string(index=False))

def save_models(model, imputer, scaler, features):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / 'lr.pkl')
    joblib.dump(imputer, MODELS_DIR / 'imputer.pkl')
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    joblib.dump(model.classes_, MODELS_DIR / 'classes.pkl')
    joblib.dump(features, MODELS_DIR / 'feature_names.pkl')

if __name__ == "__main__":
    data = pd.read_csv(FEATURES_PATH)
    X_train, X_val, Y_train, Y_val, features, imputer, scaler = prepare_data(data)
    model = train_logistic_regression(X_train, Y_train)
    evaluate_model(model, X_val, Y_val, features)
    save_models(model, imputer, scaler, features)

