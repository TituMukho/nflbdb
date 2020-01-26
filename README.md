# NFL Big Data Bowl 2019

Predict Rushing Yards Gained/Lost in an Offensive Play for the NFL

This project is for MGMT 590- Predictive Analytics at Krannert School of Management, developed by Team : Abhinav Chanda, Pranay Khandelwal and Tirthankar Mukhopadhyay. This is a data science model and implementation of a stock market prediction for Walt Disney Co.

As part of the Kaggle NFL Big Data Bowl 2020, we developed a sequential neural network to predict the number of rushing yards an NFL team would gain/lose on each play.

The data provided was part of NFL's NextGen Stats. The training data was broken down play-wise into 22 rows. Each row corresponding to a single player on the field. The data points included fields such as player orientation, positional coordinates on the field, offensive formation, defensive formation, weather conditions, speed, weight, acceleration, etc.

We developed around 30 new features based on our understanding of the game. A few features we generated were momentum, yards gained vs distance to goal, bucketed jersey numbers, etc. We used an ensemble feature importance algorithm to select a total of 24 features from a total of around 70.

A TensorFlow based sequential neural network was used to generate the final predictions. We optimized our neural network using Bayesian Optimization techniques to determine the optimal number of layers, nodes, dropout, etc.
