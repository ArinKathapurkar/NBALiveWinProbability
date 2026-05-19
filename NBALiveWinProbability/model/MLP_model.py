import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd 
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from pathlib import Path 
import joblib

ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "S23" : ROOT / "data" / "historical" / "pbp_2023_24.csv",
    "S24" : ROOT / "data" / "historical" / "pbp_2024_25.csv",
    "S25" : ROOT / "data" / "historical" / "pbp_2025_26.csv" 
}


TRAINING = {
    "scaler" : ROOT / "model" / "scaler.pkl",
    "checkpoint" : ROOT / "model" / "checkpoints" / "win_prob.pt"
}

TRAINING["scaler"].parent.mkdir(parents=True, exist_ok=True)
TRAINING["checkpoint"].parent.mkdir(parents=True, exist_ok=True)





class LiveProbability(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8,32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32,16)
        self.layer3 = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x 

model = LiveProbability()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
best_val_loss = float('inf')


def predict_win_probability(game_state: dict, model, scaler) -> float:
    """
    Predicts the probability that the HOME team wins.
    
    Input features (8):
        period, seconds_remaining_in_period, seconds_remaining_in_game,
        score_diff (home - away), home_fouls, away_fouls,
        foul_diff (home - away), possession (1=home, 0=away)
    
    Output:
        float in (0, 1) — probability of home team winning
    """
    scaler = joblib.load(TRAINING["scaler"])
    
    model = LiveProbability()
    model.load_state_dict(torch.load(TRAINING["checkpoint"]))
    model.eval()

    # build feature array in the same column order as training
    features = np.array([[
        game_state['period'],
        game_state['seconds_remaining_in_period'],
        game_state['seconds_remaining_in_game'],
        game_state['score_diff'],
        game_state['home_fouls'],
        game_state['away_fouls'],
        game_state['foul_diff'],
        game_state['possession']
    ]])

    features_scaled = scaler.transform(features)

    x = torch.tensor(features_scaled, dtype=torch.float32)

    # forward pass
    with torch.no_grad():
        prob = model(x)

    return {
        "home_team"         : game_state["home_team"],
        "away_team"         : game_state["away_team"],
        "home_win_prob"     : round(prob.item(), 4),
        "away_win_prob"     : round(1 - prob.item(), 4),
    }


if __name__ == "__main__":
    
    frames = [pd.read_csv(p) for p in PATHS.values() if p.exists()]

    if not frames:
        raise FileNotFoundError("No training data found — run fetch_historical.py first")

    data = pd.concat(frames, ignore_index=True)
    #print(f"Loaded {len(data)} rows from {len(frames)} seasons")

    '''
    print(data.shape)
    print(data.columns.tolist())
    print(data.isnull().sum())
    '''


    all_games = data['game_id'].unique()
    np.random.seed(42)
    np.random.shuffle(all_games)

    split = int(.8*len(all_games))
    train_ids = all_games[:split]
    val_ids = all_games[split:]

    train_df = data[data['game_id'].isin(train_ids)]
    val_df = data[data['game_id'].isin(val_ids)]



    feature_cols = ['period',
                    'seconds_remaining_in_period',
                    'seconds_remaining_in_game',
                    'score_diff',
                    'home_fouls',
                    'away_fouls',
                    'foul_diff',
                    'possession' 
                    ]


    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df['home_win'].to_numpy()

    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df['home_win'].to_numpy()


    '''------Standardization------'''
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)  
    joblib.dump(scaler, TRAINING["scaler"])



    '''------TENSORS------'''
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)



    '''------DATA LOADER------'''

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=512, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=512, shuffle=False)

        
    for epoch in range(100):
        model.train()
        train_loss = 0.0 
        
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
    
        
        model.eval()
        val_loss = 0.0 
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred  = model(X_batch)
                loss    = loss_fn(y_pred, y_batch)
                val_loss += loss.item()
        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch {epoch+1:2d} | train: {avg_train:.4f} | val: {avg_val:.4f}")
        
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), TRAINING["checkpoint"])
            print(f"  ↑ saved (val loss improved to {best_val_loss:.4f})")
        
    print("Training Complete")