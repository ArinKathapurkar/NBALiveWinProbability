import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from flask import Flask 
from flask_socketio import SocketIO
import torch, joblib


#---Loading the Model---
from model.MLP_model import LiveProbability, predict_win_probability, TRAINING

model = LiveProbability()
model.load_state_dict(torch.load(TRAINING["checkpoint"]))
model.eval()
scaler = joblib.load(TRAINING["scaler"])
#-----------------------


app = Flask(__name__)

from flask import send_from_directory

@app.route('/')
def index():
    frontend_dir = Path(__file__).resolve().parent.parent / 'frontend'
    return send_from_directory(str(frontend_dir), 'index.html')
    
    
socketio = SocketIO(app, cors_allowed_origins="*")

from data.fetch_live import get_todays_games, get_game_state

def compute_Live():
    while True:
        live_games = [g for g in get_todays_games() if g['status'] == 2]

        results = []
        for game in live_games:
            state = get_game_state(game['game_id'])
            if state:
                print(state)
                prob = predict_win_probability(state, model, scaler)
                results.append(prob)

        socketio.emit('probabilities', results)
        socketio.sleep(15)
        

        
# ── start background task on first connection ─────────────
@socketio.on('connect')
def on_connect():
    print("client connected")

socketio.start_background_task(compute_Live)

# ── run ───────────────────────────────────────────────────
if __name__ == "__main__":
    socketio.run(app, port=5001, debug=True)