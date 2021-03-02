#
# Given height and weight range and max reps to train give dog reccomendation
# data from: https://data.world/len/dog-size-intelligence-linked
#

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#read file
intelligence_data = pd.read_csv("dog_intelligence.csv")
breed_data = pd.read_csv("AKC Breed Info.csv")

#merge files
prediction_data = intelligence_data.merge(breed_data)

#setup input and output
X = prediction_data[["height_low_inches", "height_high_inches", "weight_low_lbs", "weight_high_lbs", "reps_upper"]]
y = prediction_data["Breed"]

#create model
model = DecisionTreeClassifier()
model.fit(X, y)

#get input
minHeight = int(input("What is the minimum height you want in inches? "))
maxHeight = int(input("What is the maximum height you want in inches? "))
minWeight = int(input("What is the minimum weight you want in pounds? "))
maxWeight = int(input("What is the maximum weight you want in pounds? "))
maxReps = int(input("What is the maximum times you want to repeat a command before they learn it? "))

#predict
prediction = model.predict([[minHeight, maxHeight, minWeight, maxWeight, maxReps]])
print("You should get a " + prediction[0])