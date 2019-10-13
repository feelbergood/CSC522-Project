import csv
import os
import pandas as pd

dirname = os.path.dirname(__file__)
player_playoffs = os.path.join(dirname, 'databasebasketball2.0/player_playoffs.txt')
df_player = pd.read_csv(player_playoffs, sep=",")
playoff_teams = {}

# NBA and ABA merged after 1976, only use data from 1976
for index, row in df_player[df_player['year'] >= 1976].iterrows():
    year = row['year']
    team = row['team']
    if year not in playoff_teams.keys():
        playoff_teams[year] = [team]
    else:
        if team not in playoff_teams[year]:
            playoff_teams[year].append(team)

team_season = os.path.join(dirname, 'databasebasketball2.0/team_season.txt')
df_team = pd.read_csv(team_season, sep=",")
df_team = df_team[df_team['year'] >= 1976].reset_index(drop=True)
df_team['class'] = 0

# In playoff: 1, not in playoff: 0
for index, row in df_team.iterrows():
    year = row['year']
    team = row['team']
    if team in playoff_teams[year]:
        df_team.loc[index, 'class'] = 1

# df_team.to_csv('output/team_seasons_classified.csv', index=False)
print(df_team.to_csv(index=False))