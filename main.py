from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = FastAPI()

bankloan = pd.read_csv("BankLoan.csv")
bankloan["AGE"] = bankloan["AGE"].astype("category")
train_dum = pd.get_dummies(bankloan, columns=["AGE"], drop_first=True)

y = train_dum["DEFAULTER"]
X = train_dum.drop(columns=["SN", "DEFAULTER"])

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X, y)


class Customer(BaseModel):
    AGE: int
    EMPLOY: float
    ADDRESS: float
    DEBTINC: float
    CREDDEBT: float
    OTHDEBT: float


@app.get("/")
def home():
    return {"message": "Bank Loan Default API is running"}


@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])
    data["AGE_2"] = int(customer.AGE == 2)
    data["AGE_3"] = int(customer.AGE == 3)
    data = data.reindex(columns=X.columns, fill_value=0)

    prob = model.predict_proba(data)[:, 1][0]
    return {"probability_of_default": round(float(prob), 3)}
