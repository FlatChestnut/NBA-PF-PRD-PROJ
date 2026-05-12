from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
import pandas as pd
from typing import cast

team1 = input("Enter the your team's name: ")
team2 = input("Enter the opposing team's name: ")

def get_season(date_str):
    date = pd.to_datetime(date_str)
    year = date.year
    month = date.month
    if month < 7:
        start_year = year - 1
    else:
        start_year = year
    return f"{start_year}-{str(start_year + 1)[-2:]}"

date1 = input("Enter the game date for the first team (YYYY-MM-DD): ")
sample_size1 = int(input("Enter the number of samples to use for training for the first team (e.g., 10): "))
game_date1 = str(date1).split("-")
season1 = get_season(date1)

date2 = input("Enter the game date for the second team (YYYY-MM-DD): ")
sample_size2 = int(input("Enter the number of samples to use for training for the second team (e.g., 10): "))
game_date2 = str(date2).split("-")
season2 = get_season(date2)

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
        "PTS_OPP", "POSS_OPP", "MIN"
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
        df[f"{col}_SEASON"] = (
            df.groupby("Team_ID")[col]
            .transform(
                lambda x:
                    x.shift(1)
                     .rolling(82, min_periods=3)
                     .sum()
            )
        )


    df["OFF_RATING_SEASON"] = (df["PTS_SEASON"] / df["POSS_ROLL"]) * 100
    df["AST_TOV_SEASON"] = df["AST_SEASON"] / df["TOV_SEASON"]
    df["PACE_SEASON"] = (df["POSS_SEASON"] / df["MIN_SEASON"]) * 48
    df["DEF_RATING_SEASON"] = (
        df["PTS_OPP_SEASON"] / df["POSS_OPP_SEASON"]
    ) * 100

    # advanced metrics
    df["OFF_RATING"] = (df["PTS_ROLL"] / df["POSS_ROLL"]) * 100
    df["AST_TOV"] = df["AST_ROLL"] / df["TOV_ROLL"]
    df["PACE"] = (df["POSS"] / df["MIN_ROLL"]) * 48
    df["DEF_RATING"] = (
        df["PTS_OPP_ROLL"] / df["POSS_OPP_ROLL"]
    ) * 100
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["REST_DAYS"] = (df["GAME_DATE"].iloc[-1] - df["GAME_DATE"].iloc[-2]).days

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
        
    
    df["HOME"] = df["MATCHUP"].apply(
    lambda x: 1 if "vs." in x else 0
    )

    stats = [df["OFF_RATING"].iloc[-1], df["AST_TOV"].iloc[-1],
             df["PACE"].iloc[-1], df["DEF_RATING"].iloc[-1], 
             df["OFF_RATING_SEASON"].iloc[-1], df["DEF_RATING_SEASON"].iloc[-1], 
             df["AST_TOV_SEASON"].iloc[-1], df["PACE_SEASON"].iloc[-1], 
             df["HOME"].iloc[-1], df["REST_DAYS"].iloc[-1]]
    
    return stats

team1_stats = get_team_stats(team1, season1, sample_size1, date1)
team2_stats = get_team_stats(team2, season2, sample_size2, date2)
stats = [team1_stats[0] - team2_stats[0],
         team1_stats[1] - team2_stats[1],
         team1_stats[2] - team2_stats[2],
         team1_stats[3] - team2_stats[3],
         team1_stats[4] - team2_stats[4],
         team1_stats[5] - team2_stats[5],
         team1_stats[6] - team2_stats[6],
         team1_stats[7] - team2_stats[7],
         team1_stats[8] - team2_stats[8],
         team1_stats[9] - team2_stats[9]]
