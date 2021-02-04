import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('bling.csv')

st.title('Diamond Price Prediction')

st.write('Have fun predicting the price of your diamond!')

st.sidebar.write('Choose your hyperparameters.')

n_estimators = st.sidebar.slider('n_estimators', 1, 100)
max_depth = st.sidebar.slider('max_depth', 1, 10)

st.sidebar.write('Select an option from each of the four to get your prediction.')

cut = st.sidebar.selectbox(
    'Cut', 
    ('Fair', 'Good', 'Very Good', 'Ideal', 'Super Ideal')
)
cut_dict = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Ideal': 3, 'Super Ideal': 4}

colour = st.sidebar.selectbox(
    'Colour', 
    ('J', 'I', 'H', 'G', 'F', 'E', 'D')
)
colour_dict = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity = st.sidebar.selectbox(
    'Clarity', 
    ('SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL')
)
clarity_dict = {'SI2': 0,'SI1': 1, 'VS2': 2, 'VS1': 3, 'VVS2': 4, 'VVS1':5, 'IF': 6, 'FL': 7}

carat = st.sidebar.slider('Carat weight', 0.25, 1.00)


def split_data(df):
       
    X = df[['cut', 'colour', 'clarity', 'carat']].copy()
    y = df.price

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(df)


def train_model(n_estimators, max_depth):
        
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        bootstrap=True,
        random_state=0
    )
    model = rf.fit(X_train, y_train)
    
    return model

model = train_model(n_estimators, max_depth)

def get_metrics():
   
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmspe = np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100
    
    return mae, mape, rmse, rmspe

mae, mape, rmse, rmspe = get_metrics()

st.write(f'MAE: ${mae:.2f}')
st.write(f'MAPE: {mape:.2f}')
st.write(f'RMSE: ${rmse:.2f}')
st.write(f'RMSPE: {rmspe:.2f}')

def predict_price(cut, colour, clarity, carat):
        
    diamond_specs = [[cut, colour, clarity, carat]]
    prediction = model.predict(diamond_specs)[0]
    
    return prediction

prediction = predict_price(
    cut_dict[cut], colour_dict[colour], clarity_dict[clarity], carat
)

st.write(f'The diamond you have selected is approximately ${prediction:,.2f}.')
st.write(f'{carat} carat weight')
st.write(f'{cut} cut')
st.write(f'{colour} colour')
st.write(f'{clarity} clarity')
