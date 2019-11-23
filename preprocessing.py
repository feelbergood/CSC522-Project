import csv
import os

import numpy as np
import pandas as pd

dirname = os.path.dirname(__file__)
player_playoffs = os.path.join(dirname, 'original_data/databasebasketball2.0/player_playoffs.txt')
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

team_season = os.path.join(dirname, 'original_data/databasebasketball2.0/team_season.txt')
df_team = pd.read_csv(team_season, sep=",")
df_team = df_team[df_team['year'] >= 1976].reset_index(drop=True)
df_team['class'] = 0

# In playoff: 1, not in playoff: 0
for index, row in df_team.iterrows():
    year = row['year']
    team = row['team']
    if team in playoff_teams[year]:
        df_team.loc[index, 'class'] = 1


# df_team.to_csv('preprocessed_data/team_seasons_classified_orig.csv', index=False)


def reduce_columns(df_input):
    df_output = df_input[['team', 'year', 'leag', 'o_fgm', 'o_fga', 'o_ftm', 'o_fta', 'o_oreb', 'o_dreb', 'o_reb', 'o_asts', 'o_pf', 'o_stl',
                          'o_to', 'o_blk', 'o_pts', 'd_fgm', 'd_fga', 'd_ftm', 'd_fta', 'd_oreb', 'd_dreb', 'd_reb',
                          'd_asts', 'd_pf', 'd_stl', 'd_to', 'd_blk', 'd_pts', 'pace', 'class']]
    return df_output


# reduce_columns(df_team).to_csv('preprocessed_data/team_seasons_classified_1.csv', index=False)

def reduce_rows(df_input):
    df_output = df_input[df_input.o_3pm != 0]
    df_output = df_output[df_output.o_3pa != 0]
    return df_output


# reduce_rows(df_team).to_csv('preprocessed_data/team_seasons_classified_2.csv', index=False)

def replace_with_mean(df_input):
    df_input['o_3pm'] = df_input['o_3pm'].replace(0, np.NaN)
    df_input['o_3pa'] = df_input['o_3pa'].replace(0, np.NaN)
    df_output = df_input.fillna(df_input.mean())
    return df_output


# replace_with_mean(df_team).to_csv('preprocessed_data/team_seasons_classified_3.csv', index=False)

def replace_with_median(df_input):
    df_input['o_3pm'] = df_input['o_3pm'].replace(0, np.NaN)
    df_input['o_3pa'] = df_input['o_3pa'].replace(0, np.NaN)
    df_output = df_input.fillna(df_input.median())
    return df_output


# replace_with_median(df_team).to_csv('preprocessed_data/team_seasons_classified_4.csv', index=False)

def class_count(df_input):
    seriesObj = df_input.apply(lambda x: True if x['class'] == 1 else False, axis=1)
    numOfOnes = len(seriesObj[seriesObj == True].index)
    print('class == 1 : ', numOfOnes)
    seriesObj = df_input.apply(lambda x: True if x['class'] == 0 else False, axis=1)
    numOfZeroes = len(seriesObj[seriesObj == True].index)
    print('class == 0 : ', numOfZeroes)


class_count(df_team)

