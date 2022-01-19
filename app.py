import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image



st.sidebar.header('User Input Features')

def user_input_features():
    Hour = st.sidebar.slider('Hour of a day', 0, 23,1)
    day = st.sidebar.slider('Day of a week', 0, 6, 1)
    month = st.sidebar.slider('Month of a year ', 0, 11, 1)
    Temperature = st.sidebar.slider('Temperature(°C)',-18, 40,1)
    Humidity = st.sidebar.slider('Humidity(%)', 0, 98, 1)
    Windspeed= st.sidebar.slider('Wind speed (m/s)', 0, 8, 1)
    Visibility= st.sidebar.slider('Visibility (10m)', 27, 2000, 1)
    Dewpointtemperature= st.sidebar.slider('Dew point temperature(°C)', -30, 28, 1)
    SolarRadiation= st.sidebar.slider('Solar Radiation (MJ/m2)', 0, 4, 1)
    Rainfall= st.sidebar.slider('Rainfall(mm)', 0, 35, 1)
    Snowfall=st.sidebar.slider('Snowfall (cm)', 0, 9, 1)
    Seasons= st.sidebar.selectbox('Type Of Season', ('Spring', 'Summer','Autumn','Winter'))
    Holiday= st.sidebar.selectbox('Holiday or not', ('No Holiday','Holiday'))
    Functioning_Day= st.sidebar.selectbox('Functioning Day', ('Yes','No'))
    
    

    data = {'Hour': Hour,
            'Temperature(°C)': Temperature,
            'Humidity(%)': Humidity,
            'Wind speed (m/s)': Windspeed,
            'Visibility (10m)': Visibility,
            'Dew point temperature(°C)': Dewpointtemperature,
            'Solar Radiation (MJ/m2)': SolarRadiation,
            'Rainfall(mm)': Rainfall,
            'Snowfall (cm)': Snowfall,
            'Seasons':Seasons,
            'Holiday':Holiday,
            'Functioning Day':Functioning_Day,
            'day': day,
            'month': month
            }
    features = pd.DataFrame(data, index=[0])
    return features



def predict_bike_count(given_value):
    pickle_in = open("Final_model","rb")
    pred=pickle.load(pickle_in)
    prediction=pred.predict(given_value)
    return prediction



def main():
    st.write("""
Created by Adithya Kolla

Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort.

It is important to make the rental bike available and accessible to the public  at the right time as it lessens the waitingtime.

Eventually, providing the city with a stable supply of rental bikes becomes a major concern.

Use the sidebar to select input features. Each feature defaults to its mean or mode, as appropriate.
""")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Bike Rent Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    input_df = user_input_features()
    df = pd.read_csv('SeoulBikeData.csv',encoding= 'unicode_escape')
    df['Date'] = pd.to_datetime(df['Date'])
    dates = df.Date
    days = [date.weekday() for date in dates]
    months = [date.month for date in dates]
    df['day'] = days
    df['month'] = months
    X = df.drop(columns=['Rented Bike Count', 'Date'])
    df1 = pd.concat([input_df, X], axis=0)
    df3=pd.get_dummies(df1)
    df3 = df3[:1]
    print(df3)
    result=""
    st.subheader('Prediction')
    if st.button("Predict"):
        result=predict_bike_count(df3)
    st.success('The Predicted Bike count for the given inputs is {}'.format(result))
    if st.button("About"):
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()