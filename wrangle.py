'''Aquire and prep chess game data'''

import os
import pandas as pd
import numpy as np

import re

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

################################################################# acquire main function#################################################################

def wrangle_chess_data(reprep = False):
    ''' Aquires and Prepares data for project'''

    if (os.path.isfile('chess_games_prepared.csv') == False) or (reprep == True):

        # read in data from csv
        df = pd.read_csv('games.csv')

        # get feature columns
        df = df[['rated','winner', 'increment_code', 'white_rating',
            'black_rating', 'opening_name']]

        # rename columns
        df = df.rename(columns={'winner': 'winning_pieces'})

        # ensuring no white space in values
        columns = ['winning_pieces','increment_code', 'opening_name','opening_name']

        for column in columns:
        
            df[column] = df[column].apply(lambda value: value.strip())

        # adding derived features
        df = add_features(df)

        df = df[['time_control_group']].join(pd.get_dummies(df,columns=['time_control_group']))

        # saving to csv
        df.to_csv('chess_games_prepared.csv', index = False)

    return pd.read_csv('chess_games_prepared.csv')

####################################Trian Validate Test Split#######################################################################

def split_my_data(df):
    '''Splits data and returns a train, validate, and test dataframe'''

    # split df into train_validate and test
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.upset)

    # split train_validate into train and validate
    train, validate =  train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.upset)

    # reset index for train validate and test
    train.reset_index(drop=True, inplace=True)
    validate.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train, validate, test

###################################################### scaling #######################################################################################

def scale_data(train, validate, test):
    "Adds scaled columns to split data"

    # Scaling continuous variables
    cols_to_scale = ['rating_difference', 'game_rating']

    # create df's for train validate and test with only columns that need to be scaled
    train_to_be_scaled = train[cols_to_scale]
    validate_to_be_scaled = validate[cols_to_scale]
    test_to_be_scaled = test[cols_to_scale]

    # create scaler object and fit that object on the train data
    scaler = sklearn.preprocessing.MinMaxScaler().fit(train_to_be_scaled)

    # transform data into an array using the scaler object 
    train_scaled = scaler.transform(train_to_be_scaled)
    validate_scaled = scaler.transform(validate_to_be_scaled)
    test_scaled = scaler.transform(test_to_be_scaled)

    # transform data into a dataframe
    train_scaled = pd.DataFrame(train_scaled, columns = cols_to_scale)
    validate_scaled = pd.DataFrame(validate_scaled, columns = cols_to_scale)
    test_scaled = pd.DataFrame(test_scaled, columns = cols_to_scale)

    # add _scaled to each column name in the scaled data
    for col in cols_to_scale:

        train_scaled = train_scaled.rename(columns={col: col + "_scaled"})
        validate_scaled = validate_scaled.rename(columns={col: col + "_scaled"})
        test_scaled = test_scaled.rename(columns={col: col + "_scaled"})

    # add scaled columns to their original dataframes
    train = train.join(train_scaled)
    validate = validate.join(validate_scaled)
    test = test.join(test_scaled)

    return train, validate, test

####################################################### feature engineering ###########################################################################

def get_time_block(value):
    '''convert time code to time in minutes'''

    # get both variables from the time code
    value = re.sub(r'\+', ' ', value)
    value = value.split(' ')
    value = int(value[0])

    # return time block
    if value >= 60:

        return "Standard"

    elif value <= 30 and value >= 15:

        return "Rapid"

    elif value <= 5 and value >= 3:

        return "Blitz"

    elif value <= 2:

        return "Bullet"

    else:

        return "Other"


def add_features(df):
    '''Adds features for exploration'''

    # add feature columns
    df['upset'] = (((df.white_rating > df.black_rating) & (df.winning_pieces == 'black')) |
                  ((df.white_rating < df.black_rating) & (df.winning_pieces == 'white')))

    df["rating_difference"] = abs(df.white_rating - df.black_rating)

    df["game_rating"] = (df.white_rating + df.black_rating) / 2
    df["game_rating"] = df["game_rating"].astype(int)

    df["lower_rated_white"] = (df.white_rating < df.black_rating)

    df['time_control_group'] = df.increment_code.apply(lambda value: get_time_block(value))

    return df