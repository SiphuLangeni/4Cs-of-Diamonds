import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


st.markdown("<h1 style='text-align: center; '>Diamond Price Prediction</h1>", unsafe_allow_html=True)

# st.image('diamonds01.jpg', use_column_width=True)

# st.image('carat-weight.jpg')

st.markdown('''
The price of diamonds is largely dependent on the *cut, colour, clarity* and *carat weight*  
a.k.a. the **4Cs of Diamonds**.  

This simple app allows you to predict the price of round cut diamonds.
''')

# st.sidebar.write('Choose your hyperparameters.')

# n_estimators = st.sidebar.slider('n_estimators', 1, 100)
# max_depth = st.sidebar.slider('max_depth', 1, 10)

st.write('Select an option from each of the four to get your prediction.')

st.markdown('### Cut')
cut = st.selectbox(
    'Select a colour from the dropdown menu', 
    ('Fair', 'Good', 'Very Good', 'Ideal', 'Super Ideal')
)
cut_dict = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Ideal': 3, 'Super Ideal': 4}
st.markdown('### Colour')
colour = st.selectbox(
    'Select a cut from the dropdown menu', 
    ('J', 'I', 'H', 'G', 'F', 'E', 'D')
)
colour_dict = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}

st.markdown('### Clarity')
clarity = st.selectbox(
    'Select a clarity from the dropdown menu',
    ('SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF', 'FL')
)
with st.beta_expander("Clarity explanation"):
    st.markdown("""
        ### The amount of inclusions and blemishes  
        
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
    st.image('clarity_range.png', use_column_width=True)
clarity_dict = {'SI2': 0,'SI1': 1, 'VS2': 2, 'VS1': 3, 'VVS2': 4, 'VVS1':5, 'IF': 6, 'FL': 7}

st.markdown('### Carat')
carat = st.slider('Select a carat weight on the slider', 0.25, 1.00)
with st.beta_expander("Carat weight explanation"):
    st.markdown("""
        ### The amount of inclusions and blemishes  
        
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
    # st.image('carat_weight.jpg')




@st.cache
def split_data():

    df = pd.read_csv('bling.csv')
       
    X = df[['cut', 'colour', 'clarity', 'carat']].copy()
    y = df.price

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data()


def train_model():
        
    rf = RandomForestRegressor(
        n_estimators=65,
        max_depth=5,
        bootstrap=True,
        random_state=0
    )
    model = rf.fit(X_train, y_train)
    
    return model

model = train_model()

def get_metrics():
   
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmspe = np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100
    
    return mae, mape, rmse, rmspe

mae, mape, rmse, rmspe = get_metrics()

# st.write(f'MAE: ${mae:.2f}')
# st.write(f'MAPE: {mape:.2f}')
# st.write(f'RMSE: ${rmse:.2f}')
# st.write(f'RMSPE: {rmspe:.2f}')

def predict_price(cut, colour, clarity, carat):
        
    diamond_specs = [[cut, colour, clarity, carat]]
    prediction = model.predict(diamond_specs)[0]
    
    return prediction

prediction = predict_price(
    cut_dict[cut], colour_dict[colour], clarity_dict[clarity], carat
)

def make_html(htag, alignment, text):
    '''Changes text to html for annotation widget user interface.
    :param text: Text for conversion to html.
    :type text: str
    :returns: HTML snippet
    :rtype: str
    '''
    # html = '"' + f"<{htag} style='text-align: {alignment};'>{text}</{htag}>" + '"'
    html = '"' + '<' + htag + " style='text-align: " + alignment + ";'>" + text + '</' + htag + '>' + '"'
    return html

test = make_html('h1', 'center', str(prediction))
st.markdown(make_html('h1', 'center', str(prediction)))
st.write(type(test))

# st.write(f'{carat} carat weight')
# st.write(f'Cut:\t{cut}')
# st.write(f'{colour} colour')
# st.write(f'{clarity} clarity')
# st.write(f'is approximately')
# st.title(f'${prediction:,.2f}')

st.markdown("<h2 style='text-align: center;'>The diamond you selected is approximately</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.beta_columns(4)
with col1:
    st.markdown("<h3 style='text-align: center; '>Cut</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; '>Cut</h2>", unsafe_allow_html=True)
with col2:
    st.markdown("<h3 style='text-align: center; '>Color</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; '>Color</h2>", unsafe_allow_html=True)

with col3:
    st.markdown("<h3 style='text-align: center; '>Clarity</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; '>Clarity</h2>", unsafe_allow_html=True)

with col4:
    st.markdown("<h3 style='text-align: center; '>Carat</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; '>Carat</h2>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; '>is approximately</h2>", unsafe_allow_html=True)
# st.markdown(style='text-align: center;' make_html(test), unsafe_allow_html=True)

