# Chess Upsets

# Project Description 

Chess is widely renowned as one of the most skill intensive games ever invented. Both sides begin the game with identical pieces in an identical position, there are no random elements (aside from assigning the first move), and the movement of those pieces during a game can result in over 121 million possible combinations in just the fist three moves. Because of this, the player with the most skill is likely to win the grand majority of chess games. I have decided to look into the different elements of a chess game to determine if any of them increase or decrease the chance of a player with lower skill defeating a player with greater skill.

# Project Goal

Discover drivers of upsets in chess games played on Lichess.org, and use those drivers to develop a machine learning model to predict whether a given game will end in upset. An upset is defined as a lower rated player defeating a higher rated player. This information could be used to further our understanding of which game elements contribute to or detract from a game’s skill intensity.

# Initial thoughts

My initial hypothesis is that drivers of upsets will be elements that either grant an outright advantage to one player or increase the likelihood of players making mistakes.

# The Plan

* Aquire data from Kaggle

* Prepare data
    * Create Engineered columns from existing data
        * upset
        * rating_difference
        * game_rating
        * lower_rated_white
        * time_control_group

* Explore data in search of drivers of upsets
    * Answer the following questions
        * How often do upsets occur?
        * Dose first move advantage effect upsets?
        * Does game rating (The average rating of both players in a game) have an effect on upsets?
        * Does time block effect upsets?
        * Does a game being rated effect upsets?

* Develop a Model to predict if a chess game will end in an upset 
    * Use drivers identified in explore to build predictive models of different types 
    * Evaluate tode models on train and validate data
    * Select the best model based on highest accuracy 
    * Evaluate the best model on test data

* Draw conclusions

# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|Rated| True or False, The game's result is reflected in each player's rating|
|Winning Pieces| The color of pieces the winning player was moving|
|White Rating| Rating of the player moving the white pieces using the Glicko-2 rating method for games played on Lichess|
|Black Rating| Rating of the player moving the white pieces using the Glicko-2 rating method for games played on Lichess|
|Rating Difference| The difference in rating between the players in the game|
|Game Rating| The average rating of the two players in the game|
|Lower Rated White| True or False, The lower rated player is moving the white pieces|
|Opening Name| The name of the opening played in the game|
|Time Control Group| The amount of time allotted to each player to make their moves, **Standard** (60 min or more), **Rapid** (30 - 15 min), **Blitz** (5 - 3 min), or **Bullet** (2 or less), **Other** (any other time limit)|
|Upset (Target)| True or False, The lower rated player won the game|
|Additional Features|Encoded values for 'Time Control Group' and 'Opening Name'|

# Steps to Reproduce 
1) Clone this repo.
2) Acquire the data from Kaggle at https://www.kaggle.com/datasnaek/chess.
3) Put the data in the file containing the cloned repo.
4) Run notebook.
