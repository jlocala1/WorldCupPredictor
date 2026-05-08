import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from collections import Counter
from elo import EloRating

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'features.csv'

def load_model():
    model = joblib.load(MODELS_DIR / 'lr_raw.pkl')
    fill_values = joblib.load(MODELS_DIR / 'fill_values.pkl')
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    features = joblib.load(MODELS_DIR / 'feature_names.pkl')
    classes = joblib.load(MODELS_DIR / 'classes.pkl')
    return model, fill_values, scaler, features, classes

def get_simulated_score(outcome):
    if outcome == 2.0:
        return (1, 0) # Home win
    elif outcome == 0.0:
        return (0, 1) # Away win
    else:
        return (1, 1) # Draw

def run_match_sim(home_team, away_team, df, model, elo_tracker, fill_values, scaler, feature_cols, classes):
    match_row = pd.DataFrame(np.nan, index=[0], columns=feature_cols)
    h_data = df[df['home_team'] == home_team].iloc[-1:]
    a_data = df[df['away_team'] == away_team].iloc[-1:]
    if h_data.empty or a_data.empty:
        print(f"Warning: Missing data for {home_team} or {away_team}. Defaulting to home win.")
        return 2.0 # Default to home win if data is missing
    for col in feature_cols:
        if col in h_data.columns and 'home_' in col:
            match_row.at[0, col] = h_data[col].values[0]
        elif col in a_data.columns and 'away_' in col:
            match_row.at[0, col] = a_data[col].values[0]

    match_row.at[0, 'home_elo'] = elo_tracker.get(home_team)
    match_row.at[0, 'away_elo'] = elo_tracker.get(away_team)
    match_row.at[0, 'elo_diff'] = match_row.at[0, 'home_elo'] - match_row.at[0, 'away_elo']

    for col in feature_cols:
        if col.endswith('_missing'):
            base_col = col.replace('_missing', '')
            match_row.at[0, col] = 1.0 if pd.isna(match_row.at[0, base_col]) else 0.0

    X_imputed = match_row.fillna(fill_values)
    X_scaled = scaler.transform(X_imputed)

    probs = model.predict_proba(X_scaled)[0]
    outcome = np.random.choice(classes, p=probs)
    h_score, a_score = get_simulated_score(outcome)
    elo_tracker.update(home_team, away_team, h_score, a_score, tournament="World Cup", neutral=True)

    if outcome == 1.0:
        return np.random.choice([0.0, 2.0]) # Randomly choose home or away win for draw
    return outcome

def simulate_tournament(participants, df, model, sim_tracker, fill_values, scaler, feature_cols, classes, verbose=False):
    current_round = participants.copy()
    while len(current_round) > 1:
        winners = []
        for i in range(0, len(current_round), 2):
            home = current_round[i]
            away = current_round[i+1]
            result = run_match_sim(home, away, df, model, sim_tracker, fill_values, scaler, feature_cols, classes)
            winner = home if result == 2.0 else away
            winners.append(winner)
            if verbose:
                print(f"{home} vs {away} -> Winner: {winner}")
        
        current_round = winners
    return current_round[0]

def monte_carlo_simulation(participants, df, model, fill_values, scaler, feature_cols, classes, iters=1000):
    baseline_elos = {}
    for team in participants:
        mask = (df['home_team'] == team) | (df['away_team'] == team)
        team_data = df[mask]
        if team_data.empty:
            print(f"Warning: No historical data for {team}. Assigning default Elo of 1500.")
            baseline_elos[team] = 1500
            continue
        last_row = team_data.iloc[-1]
        if last_row['home_team'] == team:
            baseline_elos[team] = last_row['home_elo']
        else:
            baseline_elos[team] = last_row['away_elo']
            
    print(f"Running Monte Carlo simulation with {iters} iterations...")
    results = []
    for i in range(iters):
        sim_tracker = EloRating(ratings=baseline_elos.copy())
        winner = simulate_tournament(participants, df, model, sim_tracker, fill_values, scaler, feature_cols, classes, verbose=False)
        results.append(winner)
        if (i+1) % (iters // 10) == 0:
           print(f"Progress {(i+1)/iters * 100:.0f}%")

    winner_counts = Counter(results) 
    stats = pd.DataFrame.from_dict(winner_counts, orient='index', columns=['Wins'])
    stats['Win Probability'] = (stats['Wins'] / iters) * 100
    return stats.sort_values(by='Win Probability', ascending=False)


if __name__ == "__main__":
    print("Loading model and data...")
    model, fill_values, scaler, features, classes = load_model()
    df = pd.read_csv(DATA_PATH)
    participants = [
        "Argentina", "Mexico",
        "France", "Poland",
        "England", "Senegal",
        "Brazil", "South Korea"
    ]

    print(f"Simulating knockout stage for participants: {len(participants)} teams")
    results = monte_carlo_simulation(participants, df, model, fill_values, scaler, features, classes, iters=1000)
    print(f"\n --- Simulation Results ---")
    print(results)