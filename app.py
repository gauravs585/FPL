import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary

# Step 1: Fetch FPL data
def fetch_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()
    
    players = pd.DataFrame(data['elements'])
    teams = pd.DataFrame(data['teams'])
    return players, teams

# Step 2: Preprocess data
def preprocess_data(players, teams):
    players = players[['id', 'web_name', 'element_type', 'team', 'now_cost', 'total_points', 'form', 'selected_by_percent', 'minutes']]
    players['position'] = players['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
    players['now_cost'] = players['now_cost'] / 10
    players = players.merge(teams[['id', 'strength']], left_on='team', right_on='id', how='left')
    players['form'] = players['form'].astype(float)
    players['selected_by_percent'] = players['selected_by_percent'].astype(float)
    players = players.dropna()
    return players

# Step 3: Train model
def train_model(players):
    features = ['form', 'selected_by_percent', 'minutes', 'strength']
    X = players[features]
    y = players['total_points']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Step 4: Predict points
def predict_points(model, scaler, players):
    features = ['form', 'selected_by_percent', 'minutes', 'strength']
    X_scaled = scaler.transform(players[features])
    players['predicted_points'] = model.predict(X_scaled)
    return players

# Step 5: Select optimal team
def select_team(players):
    prob = LpProblem("FPL_Team_Selection", LpMaximize)
    player_vars = {i: LpVariable(f"player_{i}", cat=LpBinary) for i in players.index}
    
    # Objective: Maximize predicted points
    prob += lpSum(player_vars[i] * players.loc[i, 'predicted_points'] for i in players.index)
    
    # Constraints
    prob += lpSum(player_vars[i] for i in players.index) == 15  # Total 15 players
    prob += lpSum(player_vars[i] * players.loc[i, 'now_cost'] for i in players.index) <= 100  # Budget
    
    # Position constraints (FPL rules)
    prob += lpSum(player_vars[i] for i in players[players['position'] == 'GK'].index) == 2
    prob += lpSum(player_vars[i] for i in players[players['position'] == 'DEF'].index) == 5
    prob += lpSum(player_vars[i] for i in players[players['position'] == 'MID'].index) == 5
    prob += lpSum(player_vars[i] for i in players[players['position'] == 'FWD'].index) == 3
    
    # Max 3 players from each real-life team
    for team_id in players['team'].unique():
        prob += lpSum(player_vars[i] for i in players[players['team'] == team_id].index) <= 3
    
    prob.solve()
    
    selected_players = players[[player_vars[i].value() == 1 for i in players.index]]
    return selected_players[['web_name', 'position', 'now_cost', 'predicted_points']]


# Streamlit App
st.title("Fantasy Premier League Optimal Team Predictor")
st.write("This app predicts the best starting 11 based on current FPL data, form, and performance.")

if st.button("Generate Optimal Team"):
    with st.spinner("Fetching latest FPL data..."):
        players, teams = fetch_fpl_data()
        players = preprocess_data(players, teams)
        model, scaler = train_model(players)
        players = predict_points(model, scaler, players)
        optimal_team = select_team(players)
    
    st.success("✅ Optimal Team Generated!")
    st.dataframe(optimal_team)

    # Total cost and points
    total_cost = optimal_team['now_cost'].sum()
    total_points = optimal_team['predicted_points'].sum()
    st.write(f"**Total Cost:** £{total_cost:.1f}m")
    st.write(f"**Predicted Total Points:** {total_points:.2f}")

