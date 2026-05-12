from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
import pandas as pd
from typing import cast

team1 = input("Enter the first team's name: ")
team2 = input("Enter the second team's name: ")

date1 = input("Enter the game date for the first team (YYYY-MM-DD): ")
sample_size1 = int(input("Enter the number of samples to use for training for the first team (e.g., 10): "))
game_date1 = str(date1).split("-")
season1 = game_date1[0] + "-" + str(int(game_date1[0]) + 1)[-2:]

date2 = input("Enter the game date for the second team (YYYY-MM-DD): ")
sample_size2 = int(input("Enter the number of samples to use for training for the second team (e.g., 10): "))
game_date2 = str(date2).split("-")
season2 = game_date2[0] + "-" + str(int(game_date2[0]) + 1)[-2:]

def parse_min(x):
    if pd.isna(x) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)  # already numeric, no parsing needed
    parts = str(x).split(":")
    if len(parts) == 2:
        return int(parts[0]) + int(parts[1]) / 60
    return float(x)

def get_team_stats(team_name, season, sample_size, target_date):

    stats = []

    target_date = pd.to_datetime(target_date)

    team_id = teams.find_teams_by_full_name(team_name)[0]['id']

    df = teamgamelog.TeamGameLog(
        season=season,
        team_id=team_id
    ).get_data_frames()[0]

    df = cast(pd.DataFrame, pd.concat([df], ignore_index=True))

    df["GAME_DATE"] = pd.to_datetime(
    df["GAME_DATE"],
    format="%b %d, %Y"
    )

    # only keep games before target date
    df = cast(
        pd.DataFrame,
        df[df["GAME_DATE"] <= target_date]
    )

 
    # chronological order for rolling
    df = cast(
        pd.DataFrame,
        df.sort_values("GAME_DATE").reset_index(drop=True)
    )

    # possessions
    df["MIN"] = df["MIN"].apply(parse_min)

    df["POSS"] = (
        df["FGA"]
        + 0.44 * df["FTA"]
        - df["OREB"]
        + df["TOV"]
    )

    df["POSS"] = df["POSS"].fillna(0)

    # opponent merge
    df_opp = df.copy()

    df = cast(
        pd.DataFrame,
        df.merge(df_opp, on="Game_ID", suffixes=("", "_OPP"))
    )
    

    raw_cols = [
        "PTS", "FGA", "OREB", "TOV",
        "FTA", "AST", "POSS",
        "PTS_OPP", "POSS_OPP"
    ]

    for col in raw_cols:
        df[f"{col}_ROLL"] = (
            df.groupby("Team_ID")[col]
            .transform(
                lambda x:
                    x.shift(1)
                     .rolling(sample_size, min_periods=3)
                     .sum()
            )
        )

    # advanced metrics
    df["OFF_RATING"] = (df["PTS_ROLL"] / df["POSS_ROLL"]) * 100
    df["AST_TOV"] = df["AST_ROLL"] / df["TOV_ROLL"]
    df["PACE"] = (df["POSS"] / df["MIN"]) * 48
    df["DEF_RATING"] = (
        df["PTS_OPP_ROLL"] / df["POSS_OPP_ROLL"]
    ) * 100

    df = cast(
    pd.DataFrame,
    df.dropna(subset=[
        "OFF_RATING",
        "AST_TOV",
        "PACE",
        "DEF_RATING"
    ])
    )

    if df.empty:
        print(df["OFF_RATING"])
        raise ValueError(
            f"Not enough games before {target_date} for {team_name}"
        )
        

    stats.append({
        "OFF_RATING": df["OFF_RATING"].iloc[-1],
        "AST_TOV": df["AST_TOV"].iloc[-1],
        "PACE": df["PACE"].iloc[-1],
        "DEF_RATING": df["DEF_RATING"].iloc[-1]
    })
    print(df[["GAME_DATE", "AST_ROLL", "TOV_ROLL", "OFF_RATING", "DEF_RATING"]])
    return stats

team1_stats = get_team_stats(team1, season1, sample_size1, date1)
team2_stats = get_team_stats(team2, season2, sample_size2, date2)
print(f"{team1} stats: {team1_stats}")
print(f"{team2} stats: {team2_stats}")
