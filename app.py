import streamlit as st
import  pickle
import pandas as pd

st.title("Win Predictor")

teams=['Royal Challengers Bengaluru',
 'Mumbai Indians',
 'Kolkata Knight Riders',
 'Rajasthan Royals',
 'Chennai Super Kings',
 'Sunrisers Hyderabad',
 'Lucknow Super Giants',
 'Gujarat Titans',
 'Delhi Capitals',
 'Punjab Kings']

cities=['Bangalore', 'Chandigarh', 'Mumbai', 'Kolkata', 'Jaipur',
       'Chennai', 'Hyderabad', 'Cape Town', 'Port Elizabeth', 'Durban',
       'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Cuttack',
       'Ahmedabad', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Ranchi',
       'Delhi', 'Abu Dhabi', 'Pune', 'Rajkot', 'Kanpur', 'Indore',
       'Bengaluru', 'Dubai', 'Sharjah', 'Navi Mumbai', 'Lucknow',
       'Guwahati', 'Mohali']

pipe=pickle.load(open('pipe.pkl', 'rb'))

col1, col2 = st.columns(2)

with col1:
    batting_team=st.selectbox('Select The Batting Team', teams)
with col2:
    bowling_team=st.selectbox('Select The Bowing Team', teams)

selected_city=st.selectbox('Select The City', sorted(cities))

target=st.number_input('Target')

col3, col4,col5 = st.columns(3)
with col3:
    score=st.number_input('Score')
with col4:
    overs=st.number_input('Overs Completed')
with col5:
    wickets= st.number_input('Wickets')

if st.button('Predict'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
                             'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win * 100)) + "%")
    st.header(bowling_team + "- " + str(round(loss * 100)) + "%")