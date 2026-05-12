from time import sleep
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
import pandas as pd
from typing import cast


seasons = ['2025-26']
team_names = [
    'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
    'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets',
    'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers',
    'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
    'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks',
    'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns',
    'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors',
    'Utah Jazz', 'Washington Wizards'
]

all_data = []
for season in seasons:
    for team in team_names:
        team_id = teams.find_teams_by_full_name(team)[0]['id']
        data = teamgamelog.TeamGameLog(season=season, team_id=team_id).get_data_frames()[0]
        all_data.append(data)
        sleep(1)
        print(f"Done: {team} {season}")



df = cast(pd.DataFrame, pd.concat(all_data, ignore_index=True))


df = df.sort_values(['Team_ID', 'GAME_DATE']).reset_index(drop=True)

df["HOME"] = df["MATCHUP"].apply(
    lambda x: 1 if "vs." in x else 0
)

df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

df["PREV_DATE"] = df.groupby("Team_ID")["GAME_DATE"].shift(1)
df["REST_DAYS"] = (df["GAME_DATE"] - df["PREV_DATE"]).dt.days

df_opp = df.copy()

df = df.merge(df_opp, on="Game_ID", suffixes=("", "_OPP"))

df = df[df["Team_ID"] != df["Team_ID_OPP"]]

# Fix MIN parsing before using it
def parse_min(x):
    if pd.isna(x) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)  # already numeric, no parsing needed
    parts = str(x).split(":")
    if len(parts) == 2:
        return int(parts[0]) + int(parts[1]) / 60
    return float(x)

df["MIN"] = df["MIN"].apply(parse_min)
df["MIN_OPP"] = df["MIN_OPP"].apply(parse_min) 

# Compute single-game possession estimate
df["POSS"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]
df["POSS"] = df["POSS"].fillna(0)  # Handle any NaN values in possessions
df["POSS_OPP"] = df["FGA_OPP"] + 0.44 * df["FTA_OPP"] - df["OREB_OPP"] + df["TOV_OPP"]
df["POSS_OPP"] = df["POSS_OPP"].fillna(0)  # Handle any NaN values in possessions

# Roll all relevant columns (no sleep — no API call)
raw_cols = ["PTS", "FGA", "OREB", "TOV", "FTA", "AST", "POSS", "PTS_OPP", "POSS_OPP", "MIN"]
for col in raw_cols:
    df[f"{col}_ROLL"] = (
        df.groupby("Team_ID")[col]
        .transform(lambda x: x.shift(1).rolling(15, min_periods=3).sum())
    )
    df[f"{col}_ROLL_SEASON"] = (
        df.groupby("Team_ID")[col]
        .transform(lambda x: x.shift(1).rolling(82, min_periods=2).sum())
    )



# Derive features from consistently rolled values
df["OFF_RATING"] = (df["PTS_ROLL"] / df["POSS_ROLL"]) * 100
df["AST_TOV"] = df["AST_ROLL"] / df["TOV_ROLL"]
df["PACE"] = (df["POSS_ROLL"] / df["MIN_ROLL"]) * 48
df["DEF_RATING"] = (df["PTS_OPP_ROLL"] / df["POSS_OPP_ROLL"]) * 100

# Derive features for the whole season
df["OFF_RATING_SEASON"] = (df["PTS_ROLL_SEASON"] / df["POSS_ROLL_SEASON"]) * 100
df["DEF_RATING_SEASON"] = (df["PTS_OPP_ROLL_SEASON"] / df["POSS_OPP_ROLL_SEASON"]) * 100
df["AST_TOV_SEASON"] = df["AST_ROLL_SEASON"] / df["TOV_ROLL_SEASON"]
df["PACE_SEASON"] = (df["POSS_ROLL_SEASON"] / df["MIN_ROLL_SEASON"]) * 48


full_X = df[["Game_ID", "Team_ID", "WL", "HOME", "OFF_RATING", "AST_TOV", "PACE", "DEF_RATING", 
             "OFF_RATING_SEASON", "DEF_RATING_SEASON", "AST_TOV_SEASON", "PACE_SEASON",
             "REST_DAYS"]].dropna()
full_X = full_X.reset_index(drop=True)

# Merge each team's row with opponent's row on the same game
full_X = full_X.merge(full_X, on="Game_ID", suffixes=("", "_OPP"))
full_X = full_X[full_X["Team_ID"] != full_X["Team_ID_OPP"]]

# Compute differences
full_X["diff_OFF_RATING"] = full_X["OFF_RATING"] - full_X["OFF_RATING_OPP"]
full_X["diff_AST_TOV"]    = full_X["AST_TOV"]    - full_X["AST_TOV_OPP"]
full_X["diff_PACE"]       = full_X["PACE"]        - full_X["PACE_OPP"]
full_X["diff_DEF_RATING"] = full_X["DEF_RATING"]  - full_X["DEF_RATING_OPP"]
full_X["diff_REST_DAYS"] = full_X["REST_DAYS"]  - full_X["REST_DAYS_OPP"]




#Differences for the whole season
full_X["diff_OFF_RATING_SEASON"] = full_X["OFF_RATING_SEASON"] - full_X["OFF_RATING_SEASON_OPP"]
full_X["diff_DEF_RATING_SEASON"] = full_X["DEF_RATING_SEASON"] - full_X["DEF_RATING_SEASON_OPP"]
full_X["diff_AST_TOV_SEASON"] = full_X["AST_TOV_SEASON"] - full_X["AST_TOV_SEASON_OPP"]
full_X["diff_PACE_SEASON"] = full_X["PACE_SEASON"] - full_X["PACE_SEASON_OPP"]

# Extract y before dropping
full_Y = (full_X["WL"] == "W").astype(int).reset_index(drop=True)

full_X = full_X[["diff_OFF_RATING", "diff_AST_TOV", "diff_PACE", "diff_DEF_RATING", 
                 "diff_OFF_RATING_SEASON", "diff_DEF_RATING_SEASON", "diff_AST_TOV_SEASON", 
                 "diff_PACE_SEASON", "HOME", "diff_REST_DAYS"]].reset_index(drop=True)

print(f"Total rows: {len(full_X)}")
print(f"Total labels: {len(full_Y)}")