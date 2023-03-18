from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


exercises = {
    "legs": ["Air Squat", "Back Squat", "Burpees", "Cross Jumps", "Cross Jumps Rotation", "Jogging", "Jumping Jacks", "Overhead Squat", "Pistol", "Quick Steps", "Sumo High Pull"],
    "abs": ["Back Squat", "Bicycle Crunch", "Burpees", "Circle Crunch", "Cross Jumps Rotation", "Overhead Squat", "Pike Walk", "Pistol", "Plank", "Push Up", "Situps"],
    "back": ["Back Squat", "Burpees", "Plank", "Sumo High Pull"],
    "chest": ["Burpees", "Push Up"],
    "arms": ["Bicep Curl", "Burpees", "Pike Walk", "Push Up"],
    "shoulders": ["Front Raises", "Overhead Squat", "Pike Walk", "Plank", "Push Up", "Sumo High Pull"],
}


breakfast = pd.read_csv("breakfast.csv")
lunch = pd.read_csv("lunch.csv")
dinner = pd.read_csv("dinner.csv")

dietplan = breakfast[["Diet Plan"]].values
for i in range(len(dietplan)):
    if dietplan[i] == "Lose Weight":
        dietplan[i] = 0
    elif dietplan[i] == "Maintain Weight":
        dietplan[i] = 1
    else:
        dietplan[i] = 2

breakfastlist = [food for food in breakfast][4:]
lunchlist = [food for food in lunch][4:]
dinnerlist = [food for food in dinner][4:]

breakfast["Diet Plan"] = dietplan
lunch["Diet Plan"] = dietplan
dinner["Diet Plan"] = dietplan

Xb = breakfast[['Height', 'Weight', 'BMI']].values
Yb = breakfast.drop(columns=['Height', 'Weight', 'BMI']).values

Xl = lunch[['Height', 'Weight', 'BMI']].values
Yl = lunch.drop(columns=['Height', 'Weight', 'BMI']).values

Xd = dinner[['Height', 'Weight', 'BMI']].values
Yd = dinner.drop(columns=['Height', 'Weight', 'BMI']).values

modelb = MultiOutputClassifier(RandomForestClassifier())

modell = MultiOutputClassifier(RandomForestClassifier())

modeld = MultiOutputClassifier(RandomForestClassifier())

Xb = np.array(Xb)
Yb = np.array(Yb)
Xb = Xb.astype('float32')
Yb = Yb.astype('int32')

Xl = np.array(Xl)
Yl = np.array(Yl)
Xl = Xl.astype('float32')
Yl = Yl.astype('int32')

Xd = np.array(Xd)
Yd = np.array(Yd)
Xd = Xd.astype('float32')
Yd = Yd.astype('int32')

modelb.fit(Xb, Yb)
modell.fit(Xl, Yl)
modeld.fit(Xd, Yd)


@app.get("/dietplan")
def getDietPlan(height: int, weight: int):
    bmi = round(weight / ((height / 100) ** 2), 2)
    inputdata = np.array([[height, weight, bmi]])
    breakfastfood = [int(i) for i in modelb.predict(inputdata)[0]]
    lunchfood = [int(i) for i in modell.predict(inputdata)[0]]
    dinnerfood = [int(i) for i in modeld.predict(inputdata)[0]]
    plan = breakfastfood.pop(0)
    lunchfood.pop(0)
    dinnerfood.pop(0)

    res = {}
    breakfastres = {}
    lunchres = {}
    dinnerres = {}

    dietplanlist = ["Lose Weight", "Maintain Weight", "Gain Weight"]

    for i in range(len(breakfastfood)):
        if breakfastfood[i] != 0:
            breakfastres[breakfastlist[i]] = breakfastfood[i]

    
    for i in range(len(lunchfood)):
        if lunchfood[i] != 0:
            lunchres[lunchlist[i]] = lunchfood[i]

    for i in range(len(dinnerfood)):
        if dinnerfood[i] != 0:
            dinnerres[dinnerlist[i]] = dinnerfood[i]
    
    res["Breakfast"] = breakfastres
    res["Lunch"] = lunchres
    res["Dinner"] = dinnerres
    res["Diet Plan"] = dietplanlist[plan]

    return res


@app.get("/exercise")
def getExercise(bodypart: str):
    return {"Exercises": exercises[bodypart]}