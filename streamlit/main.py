import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import optuna

# Title
st.markdown('# Diamond Price Prediction')
st.markdown('---')
st.markdown('##')

# Introduction
col1, col2 = st.beta_columns([2, 3])

with col1:
    st.image('streamlit/Images/diamonds.jpg', use_column_width=True)
with col2:
    st.markdown('###')
    st.markdown('''
    The price of a diamond depends on its  
    *cut*, *colour*, *clarity* and *carat weight*,  
      
    a.k.a. the **4Cs of Diamonds**.
    ''')
st.markdown('##')

# Header
st.markdown('## Predict round cut diamond prices in **_real_** time!')

# Selection criteria
# Cut
st.markdown('### Cut')

cut = st.selectbox(
    'Select a colour from the dropdown menu', 
    ('Fair', 'Good', 'Very Good', 'Ideal', 'Super Ideal')
)

cut_dict = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Ideal': 3, 'Super Ideal': 4}

with st.beta_expander("Learn more about cut quality"):
    st.markdown("""
        ### Cut refers to the quality of the cut of the diamond.  

        Superior cuts will reflect the most light back giving a sense of 
        brilliance, fire and sparkle! Cut is arguably the most difficult to 
        measure. The grade is given from a combination of interplay of many 
        proportions and individual characteristics.
    """)
    st.markdown('#')
    st.image('streamlit/Images/diamond_cut.png', use_column_width=True)
    st.markdown('#')

# Colour
st.markdown('### Colour')

colour = st.selectbox(
    'Select a cut from the dropdown menu', 
    ('J', 'I', 'H', 'G', 'F', 'E', 'D')
)

colour_dict = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}

with st.beta_expander("Learn more about colour grade"):
    st.markdown("""
        ### Colour refers to how clear a diamond is.  

        A completely colourless diamond is the most rare and has a colour grade of **D**.  
        As the letter increases, the diamond will begin to have more yellowish hues.  

        This app allows you to choose diamonds with a color grade from  
        slightly tinted (**J**) to colourless (**D**).
    """)
    st.markdown('#')
    st.image('streamlit/Images/diamond_colour.jpg', output_format='JPEG', use_column_width=True)
    st.markdown('#')

# Clarity
st.markdown('### Clarity')

clarity = st.selectbox(
    'Select a clarity from the dropdown menu',
    ('SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL')
)

clarity_dict = {'SI2': 0, 'SI1': 1, 'VS2': 2, 'VS1': 3, 'VVS2': 4, 'VVS1':5, 'IF': 6, 'FL': 7}

with st.beta_expander("Learn more about clarity"):
    st.markdown("""
        ### Clarity refers to the amount of inclusions and blemishes  
        
        • Slightly Included (**SI2, SI1**)  
        *Inclusions noticible under 10x magnification*  

        • Very Slightly Included (**VS2, VS1**)  
        *Minor inclusions observed with effort under 10x magnification*  

        • Very, Very Slightly Included (**VVS2, VVS1**)  
        *Inclusions difficult for a skilled grader to see under 10x magnification*  

        • Internally Flawless (**IF**)  
        *No inclusions visible under 10x magnification*  

        • Flawless (**FL**)  
        *Inclusions or blemishes not visible under 10x magnification*  
    """)
    st.markdown('#')
    st.image('streamlit/Images/diamond_clarity.png', use_column_width=True)
    st.markdown('#')
    

# Carat
st.markdown('### Carat')

carat = st.slider('Select a carat weight on the slider', 0.25, 1.00)

with st.beta_expander("Learn more about carat weight"):
    st.markdown("""
        ### Carat refers to the weight of the diamond.  

        One carat is about 0.2 grams. Heavier diamonds are much more rare.
        As the carat weight increases, so does the price. The increase in 
        price is not proportionate to the increase in carat weight. A 1 
        carat diamond will be more than double the price of a 0.5 carat 
        diamond, all other things held equal.
    """)
    st.markdown('#')
    st.image('streamlit/Images/diamond_carat.jpg', output_format='JPEG')
    st.markdown('#')

st.markdown('#')
st.markdown('#')


@st.cache
def create_dataset():

    df = pd.read_csv('streamlit/bling.csv')
       
    X = df[['cut', 'colour', 'clarity', 'carat']].values
    y = df.price.values

    return X, y

X, y = create_dataset()


def train_model():

    rf = RandomForestRegressor(n_jobs=-1, random_state=0)
    study = joblib.load('streamlit/diamond_rf.pkl')
    rf.set_params(**study.best_params)
    model = rf.fit(X, y)
    
    return model

model = train_model()


#Prediction function
def predict_price(cut, colour, clarity, carat):
        
    diamond_specs = [[cut, colour, clarity, carat]]
    prediction = model.predict(diamond_specs)[0]
    
    return prediction

prediction = predict_price(
    cut_dict[cut], colour_dict[colour], clarity_dict[clarity], carat
)

# Live prediction
st.markdown('## The diamond you selected is approximately')
st.title(f'$ {prediction:,.2f}')
st.write('Price reported in CAD')
st.text('')
st.write(f'Cut: {cut} | Colour: {colour} | Clarity: {clarity} | Carat: {carat}')
