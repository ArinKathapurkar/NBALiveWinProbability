from nba_api.live.nba.endpoints import scoreboard
from nba_api.live.nba.endpoints import boxscore
from nba_api.live.nba.endpoints import playbyplay

import time

# Today's Score Board
games = scoreboard.ScoreBoard()

# dictionary
games.get_dict()

def fetch__games():
    board = scoreboard.ScoreBoard()
    games = board.games.get_dict()  # list of game dicts
    
    result = []
    for game in games:
        result.append({
            'game_id': game['gameId'],
            'home_team': game['homeTeam']['teamTricode'],
            'away_team': game['awayTeam']['teamTricode'],
            'status': game['gameStatus'],       # 1=pre, 2=live, 3=final
            'status_text': game['gameStatusText'],  # "Q3 8:42", "Final", etc.
        })
    return result


def fetch_game_score(game_id: str) -> dict:
    box = boxscore.Boxscore()
    game = box.games.get_dict()
    
    home = game['homeTeam']    
    
    away = game['awayTeam']

    #collecting features: score differential, game clock, fouls given, free throw percentage, lead changes 
    score_delta = home['score'] - away['score'] #does this deal with negatives??
    clock = game['gameClock']
    lead_changes = home['statistics']['leadChanges']
    home_ft_percentage = home['statistics']['freeThrowsPercentage']
    away_ft_percentage = away['statistics']['freeThrowsPercentage']
    
    
def fetch_live_posession(gameid: str) --> str: 
    
    
    
    
    
    
    
    
    

    