import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from collections import Counter

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'features.csv'

def load_model():
    model = joblib.load(MODELS_DIR / 'lr.pkl')
    imputer = joblib.load(MODELS_DIR / 'imputer.pkl')
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    features = joblib.load(MODELS_DIR / 'feature_names.pkl')
    classes = joblib.load(MODELS_DIR / 'classes.pkl')
    return model, imputer, scaler, features, classes

def run_match_sim(home_team, away_team, df, model, imputer, scaler, feature_cols, classes):
    h_data = df[df['home_team'] == home_team].iloc[-1:]
    a_data = df[df['away_team'] == away_team].iloc[-1:]
    if h_data.empty or a_data.empty:
        print(f"Warning: Missing data for {home_team} or {away_team}. Defaulting to home win.")
        return 2.0 # Default to home win if data is missing
    match_row = h_data.copy()
    for col in feature_cols:
        if 'away_' in col:
            match_row[col] = a_data[col].values[0]
    
    X = match_row[feature_cols]
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    probs = model.predict_proba(X_scaled)[0]
    outcome = np.random.choice(classes, p=probs)

    if outcome == 1.0:
        return np.random.choice([0.0, 2.0]) # Randomly choose home or away win for draw
    return outcome
    


def simulate_tournament(participants, df, model, imputer, scaler, feature_cols, classes, verbose=False):
    current_round = participants.copy()
    while len(current_round) > 1:
        winners = []
        for i in range(0, len(current_round), 2):
            home = current_round[i]
            away = current_round[i+1]
            result = run_match_sim(home, away, df, model, imputer, scaler, feature_cols, classes)
            winner = home if result == 2.0 else away
            winners.append(winner)
            if verbose:
                print(f"{home} vs {away} -> Winner: {winner}")
        
        current_round = winners
    return current_round[0]

def monte_carlo_simulation(participants, df, model, imputer, scaler, feature_cols, classes, iters=1000):
    print(f"Running Monte Carlo simulation with {iters} iterations...")
    results = []
    for i in range(iters):
        winner = simulate_tournament(participants, df, model, imputer, scaler, feature_cols, classes, verbose=False)
        results.append(winner)
        if (i+1) % (iters // 10) == 0:
            print(f"Progress {(i+1)/iters * 100:.0f}%")

    winner_counts = Counter(results) 
    stats = pd.DataFrame.from_dict(winner_counts, orient='index', columns=['Wins'])
    stats['Win Probability'] = (stats['Wins'] / iters) * 100
    return stats.sort_values(by='Win Probability', ascending=False)


if __name__ == "__main__":
    print("Loading model and data...")
    model, imputer, scaler, features, classes = load_model()
    df = pd.read_csv(DATA_PATH)
    participants = [
        "Argentina", "Mexico",
        "France", "Poland",
        "England", "Senegal",
        "Brazil", "South Korea"
    ]

    print(f"Simulating knockout stage for participants: {len(participants)} teams")
    results = monte_carlo_simulation(participants, df, model, imputer, scaler, features, classes, iters=1000)
    print(f"\n --- Simulation Results ---")
    print(results)