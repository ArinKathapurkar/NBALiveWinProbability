import logging
import re
import time

from nba_api.live.nba.endpoints.scoreboard import ScoreBoard
from nba_api.live.nba.endpoints.boxscore import BoxScore
from nba_api.live.nba.endpoints.playbyplay import PlayByPlay

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_CLOCK_RE = re.compile(r'PT(\d+)M([\d.]+)S')

# NBA's CDN requires a Referer and current User-Agent; the library defaults get a 403.
_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


def get_todays_games() -> list[dict]:
    try:
        board = ScoreBoard(headers=_HEADERS)
        games = board.games.get_dict()
    except Exception:
        logger.exception("Failed to fetch scoreboard (no games today or API error)")
        return []
    result = []
    for game in games:
        result.append({
            'game_id': game['gameId'],
            'home_team': game['homeTeam']['teamTricode'],
            'away_team': game['awayTeam']['teamTricode'],
            'status': game['gameStatus'],
            'status_text': game['gameStatusText'],
        })
    return result


def parse_clock(clock_str: str) -> float:
    if not clock_str:
        return 0.0
    m = _CLOCK_RE.match(clock_str)
    if not m:
        return 0.0
    minutes = float(m.group(1))
    seconds = float(m.group(2))
    return minutes * 60 + seconds


def _get_possession(game_id: str, home_team_id: int, home_tricode: str, away_tricode: str) -> str | None:
    try:
        pbp = PlayByPlay(game_id, headers=_HEADERS)
        actions = pbp.actions.get_dict()
        if not actions:
            return None
        poss_team_id = actions[-1].get('possession')
        if poss_team_id is None:
            return None
        return home_tricode if poss_team_id == home_team_id else away_tricode
    except Exception:
        logger.exception("Failed to fetch possession for game_id=%s", game_id)
        return None


def get_game_state(game_id: str) -> dict | None:
    try:
        box = BoxScore(game_id, headers=_HEADERS)
        game = box.game.get_dict()

        home = game['homeTeam']
        away = game['awayTeam']

        home_score = int(home.get('score') or 0)
        away_score = int(away.get('score') or 0)
        home_fouls = int(home['statistics']['foulsPersonal'])
        away_fouls = int(away['statistics']['foulsPersonal'])

        # At period boundaries the team statistics can briefly show 0;
        # fall back to summing individual player fouls in that case.
        if home_fouls == 0 and away_fouls == 0:
            home_fouls = sum(int(p['statistics'].get('foulsPersonal', 0)) for p in home.get('players', []))
            away_fouls = sum(int(p['statistics'].get('foulsPersonal', 0)) for p in away.get('players', []))

        possession = _get_possession(
            game_id, home['teamId'], home['teamTricode'], away['teamTricode']
        )

        return {
            'game_id': game_id,
            'period': game['period'],
            'clock_seconds': parse_clock(game.get('gameClock', '')),
            'score_diff': home_score - away_score,
            'home_fouls': home_fouls,
            'away_fouls': away_fouls,
            'foul_diff': home_fouls - away_fouls,
            'home_team': home['teamTricode'],
            'away_team': away['teamTricode'],
            'possession': possession,
        }
    except Exception:
        logger.exception("Failed to fetch game state for game_id=%s", game_id)
        return None


def poll_live_games(interval_seconds: int = 15) -> None:
    while True:
        games = get_todays_games()
        live = [g for g in games if g['status'] == 2]
        logger.info("Found %d live game(s)", len(live))

        for game in live:
            state = get_game_state(game['game_id'])
            if state is not None:
                print(state)

        time.sleep(interval_seconds)


if __name__ == "__main__":
    poll_live_games()
