
# 1. Library imports
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pydantic import BaseModel
import joblib 


class Diamond(BaseModel):
    cut: int 
    colour: int 
    clarity: int 
    carat: float


class DiamondModel:
    
    def __init__(self):
        self.df = pd.read_csv('bling.csv')
        self.model_fname = 'diamond_model.pkl'
        try:
            self.model = joblib.load(self.model_fname)
        except Exception as _:
            self.model = self.train_model()
            joblib.dump(self.model, self.model_fname)

    
    def split_data(self):
        
        X = self.df[['cut', 'colour', 'clarity', 'carat']].copy()
        y = self.df.price

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        return X_train, X_test, y_train, y_test
    
    def train_model(self):
        
        X_train, _, y_train, _ = self.split_data()
        
        rf = RandomForestRegressor(
            n_estimators=60,
            bootstrap=True,
            random_state=0
        )
        model = rf.fit(X_train, y_train)
        
        return model

    def get_metrics(self):
        
        _, X_test, _, y_test = self.split_data()

        y_pred = self.model.predict(X_test)
    
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmspe = np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100
        
        return mae, mape, rmse, rmspe


    def predict_price(self, cut, colour, clarity, carat):
        
        diamond_specs = [[cut, colour, clarity, carat]]
        prediction = self.model.predict(diamond_specs)[0]
        
        return prediction
        