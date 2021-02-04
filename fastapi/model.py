
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
        self.df = pd.read_csv('diamonds.csv')
        self.model_fname = 'diamond_model.pkl'
        try:
            self.model = joblib.load(self.model_fname)
        except Exception as _:
            self.model = self.train_model()
            joblib.dump(self.model, self.model_fname)

    
    def pre_process(self):
        
        df = self.df.drop_duplicates(subset=['upc'], keep='first', ignore_index=True)
        
        # Label encode categorical data
        df.cut = df.cut.map(
            {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Ideal': 3, 'Super Ideal': 4}
        )

        df.colour = df.colour.map(
            {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
        )

        df.clarity = df.clarity.map(
            {'SI2': 0, 'SI1': 1, 'VS2': 2, 'VS1': 3, 'VVS2': 4, 'VVS1':5, 'IF': 6, 'FL': 7}
        )

        df = df[['cut', 'colour', 'clarity', 'carat', 'price']].copy()

        # Remove outliers
        Q1 = df.price.quantile(0.25)
        Q2 = df.price.quantile(0.5)
        Q3 = df.price.quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - (1.5 * IQR)
        upper_limit = Q3 + (1.5 * IQR)
        df = df[(df.price >= lower_limit) & (df.price <= upper_limit)].reset_index(drop=True)
        
        # Split dataset
        X = df.drop(['price'], axis=1)
        y = df.price

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        return X_train, X_test, y_train, y_test
    
    def train_model(self):
        
        X_train, _, y_train, _ = self.pre_process()
        
        rf = RandomForestRegressor(
            n_estimators=50,
            bootstrap=True,
            random_state=0
        )
        model = rf.fit(X_train, y_train)
        
        return model

    def get_metrics(self):
        
        _, X_test, _, y_test = self.pre_process()

        y_pred = self.model.predict(X_test)
    
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        # rmspe = rmse / y_test.mean()
        rmspe = np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100
        
        return mae, mape, rmse, rmspe


    def predict_price(self, cut, colour, clarity, carat):
        
        diamond_specs = [[cut, colour, clarity, carat]]
        prediction = self.model.predict(diamond_specs)[0]
        
        return prediction
        