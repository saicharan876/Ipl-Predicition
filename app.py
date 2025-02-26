import streamlit as st
import pickle
import pandas as pd

st.title("Win Predictor")

teams = [
    'Royal Challengers Bengaluru', 'Mumbai Indians', 'Kolkata Knight Riders',
    'Rajasthan Royals', 'Chennai Super Kings', 'Sunrisers Hyderabad',
    'Lucknow Super Giants', 'Gujarat Titans', 'Delhi Capitals', 'Punjab Kings'
]

cities = [
    'Bangalore', 'Chandigarh', 'Mumbai', 'Kolkata', 'Jaipur', 'Chennai',
    'Hyderabad', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
    'East London', 'Johannesburg', 'Kimberley', 'Cuttack', 'Ahmedabad',
    'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Ranchi', 'Delhi', 'Abu Dhabi',
    'Pune', 'Rajkot', 'Kanpur', 'Indore', 'Bengaluru', 'Dubai', 'Sharjah',
    'Navi Mumbai', 'Lucknow', 'Guwahati', 'Mohali'
]

pipe = pickle.load(open('pipe.pkl', 'rb'))

st.subheader("Match Setup")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select The Batting Team', teams)
with col2:
    bowling_team = st.selectbox('Select The Bowling Team', teams)

selected_city = st.selectbox('Select The City', sorted(cities))
target = st.number_input('Target', min_value=0, step=1)

st.subheader("Current Match Status")

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

if st.button('Predict'):
    if batting_team == bowling_team:
        st.error("Please select different teams for batting and bowling.")
    else:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else float('inf')

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.subheader("Prediction Results")
        col6, col7 = st.columns(2)
        with col6:
            st.metric(label=batting_team, value=f"{round(win * 100)}%")
        with col7:
            st.metric(label=bowling_team, value=f"{round(loss * 100)}%")
