import sys
#import math
import numpy as np
import matplotlib.pyplot as plt
#from sql import *
#from sql.aggregate import *
#from sql.conditionals import *
from numpy import genfromtxt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold


gameGenre = int(input("Enter game genre as one of the following:\nAction=0, Adventure=1, Casual=2, Indie=3, Massively Multiplayer=4, Racing=5, RPG=6, Simulation=7, Sports=8, Strategy=9\n> "))
#genreScore = 1.2 #float(input("Enter game's genre score\n> "))
#hourScore = 2.6 #float(input("Enter the genre's hour score\n> "))
if gameGenre == 0:
    genreScore = 3.7
    hourScore = 4.4
elif gameGenre == 1:
    genreScore = 1.1
    hourScore = 0.7
elif gameGenre == 2:
    genreScore = 1.2
    hourScore = 0.7
elif gameGenre == 3:
    genreScore = 0.9
    hourScore = 0.2
elif gameGenre == 4:
    genreScore = 0
    hourScore = 0
elif gameGenre == 5:
    genreScore = 0
    hourScore = 0
elif gameGenre == 6:
    genreScore = 1.1
    hourScore = 1.1
elif gameGenre == 7:
    genreScore = 0.6
    hourScore = 0.2
elif gameGenre == 8:
    genreScore = 0.2
    hourScore = 0
elif gameGenre == 9:
    genreScore = 1.2
    hourScore = 2.6 
else:
    genreScore = 3 #average genreScore value
    hourScore = 2 #average hourScore value

    
print("For the next 3 questions, please refer to Metacritic.com. If there is insufficient information to answer the question, enter \"5\"\n")
critScore = float(input("Enter the game's critical score, as a number from 0 to 10\n> "))
userScore = float(input("Enter the game's user score, as a number from 0 10 10\n> "))
pubScore = float(input("Enter the publisher's score, as a number from 0 to 10\n> "))

#library = Table(genfromtxt('library.csv', delimiter=' '))


library2 = genfromtxt('library2.csv', delimiter=',')

#print(library2)

# generates data & split it into X (training input) and y (target output)
X = library2[:, 0:5]
y = library2[:, 6]

#print(X)
#print(y)

neurons = 20  # <- number of neurons in the hidden layer
eta = 0.1       # <- the learning rate parameter

# here we create the MLP regressor
mlp =  MLPRegressor(hidden_layer_sizes=(neurons,), verbose=True, learning_rate_init=eta)
# here we train the MLP
mlp.fit(X, y)
while(mlp.score(X,y) < 0):
    mlp.fit(X,y)
# E_out in training
print("Training set score: %f" % mlp.score(X, y))

# now we generate new data as testing set and get E_out for testing set
Xtest = np.array([genreScore, hourScore, critScore, userScore, pubScore])
#print("Testing set score: %f" % mlp.score(X, y))
ypred = mlp.predict(Xtest)
fResult = float(ypred)
rResult = round(fResult)
print(ypred)
print("Final Score: %f" % rResult)


    
    


