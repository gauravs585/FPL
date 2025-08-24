import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(page_title="FPL AI Optimizer", page_icon="⚽", layout="wide")

class FPLAnalyzer:
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.players_df = None
        self.teams_df = None
        self.fixtures_df = None
        self.current_gw = None
        
    def fetch_all_data(self):
        """Fetch comprehensive FPL data"""
        try:
            # Bootstrap data (players, teams, gameweeks)
            bootstrap_url = f"{self.base_url}/bootstrap-static/"
            bootstrap_response = requests.get(bootstrap_url)
            bootstrap_data = bootstrap_response.json()
            
            self.players_df = pd.DataFrame(bootstrap_data['elements'])
            self.teams_df = pd.DataFrame(bootstrap_data['teams'])
            self.gameweeks_df = pd.DataFrame(bootstrap_data['events'])
            
            # Current gameweek
            self.current_gw = next((gw['id'] for gw in bootstrap_data['events'] if gw['is_current']), 1)
            
            # Fixtures data
            fixtures_url = f"{self.base_url}/fixtures/"
            fixtures_response = requests.get(fixtures_url)
            self.fixtures_df = pd.DataFrame(fixtures_response.json())
            
            return True
        except Exception as e:
            st.error(f"Failed to fetch FPL data: {str(e)}")
            return False
    
    def fetch_user_team(self, team_id):
        """Fetch user's current FPL team"""
        try:
            team_url = f"{self.base_url}/entry/{team_id}/"
            team_response = requests.get(team_url)
            team_data = team_response.json()
            
            # Get current team picks
            picks_url = f"{self.base_url}/entry/{team_id}/event/{self.current_gw}/picks/"
            picks_response = requests.get(picks_url)
            picks_data = picks_response.json()
            
            # Get transfer history
            transfers_url = f"{self.base_url}/entry/{team_id}/transfers/"
            transfers_response = requests.get(transfers_url)
            transfers_data = transfers_response.json()
            
            return {
                'team_info': team_data,
                'current_picks': picks_data,
                'transfers': transfers_data
            }
        except Exception as e:
            st.error(f"Failed to fetch team data: {str(e)}")
            return None
    
    def preprocess_data(self):
        """Enhanced data preprocessing with additional features"""
        # Player data processing
        self.players_df['position'] = self.players_df['element_type'].map({
            1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'
        })
        self.players_df['price'] = self.players_df['now_cost'] / 10
        self.players_df['value'] = self.players_df['total_points'] / self.players_df['price']
        self.players_df['form_float'] = pd.to_numeric(self.players_df['form'], errors='coerce')
        self.players_df['selected_by_percent_float'] = pd.to_numeric(self.players_df['selected_by_percent'], errors='coerce')
        
        # Team strength merge
        team_strength = self.teams_df[['id', 'strength', 'strength_overall_home', 'strength_overall_away']]
        self.players_df = self.players_df.merge(
            team_strength, left_on='team', right_on='id', how='left', suffixes=('', '_team')
        )
        
        # Clean data
        self.players_df = self.players_df.dropna(subset=['form_float', 'selected_by_percent_float', 'minutes'])
        
    def calculate_fixture_difficulty(self, team_id, gameweeks=5):
        """Calculate upcoming fixture difficulty for a team"""
        upcoming_fixtures = self.fixtures_df[
            ((self.fixtures_df['team_h'] == team_id) | (self.fixtures_df['team_a'] == team_id)) &
            (self.fixtures_df['event'] >= self.current_gw) &
            (self.fixtures_df['event'] < self.current_gw + gameweeks)
        ].copy()
        
        if upcoming_fixtures.empty:
            return 3.0  # Neutral difficulty
        
        difficulties = []
        for _, fixture in upcoming_fixtures.iterrows():
            if fixture['team_h'] == team_id:
                # Home fixture - opponent's away strength
                opponent_strength = self.teams_df.loc[
                    self.teams_df['id'] == fixture['team_a'], 'strength_overall_away'
                ].iloc[0] if len(self.teams_df.loc[self.teams_df['id'] == fixture['team_a']]) > 0 else 3
            else:
                # Away fixture - opponent's home strength
                opponent_strength = self.teams_df.loc[
                    self.teams_df['id'] == fixture['team_h'], 'strength_overall_home'
                ].iloc[0] if len(self.teams_df.loc[self.teams_df['id'] == fixture['team_h']]) > 0 else 3
                
            difficulties.append(opponent_strength)
        
        return np.mean(difficulties)
    
    def predict_points(self):
        """Advanced points prediction using multiple features"""
        features = [
            'form_float', 'selected_by_percent_float', 'minutes', 'strength',
            'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
            'yellow_cards', 'red_cards', 'saves', 'bonus'
        ]
        
        # Add fixture difficulty
        fixture_difficulties = []
        for _, player in self.players_df.iterrows():
            difficulty = self.calculate_fixture_difficulty(player['team'])
            fixture_difficulties.append(difficulty)
        
        self.players_df['fixture_difficulty'] = fixture_difficulties
        features.append('fixture_difficulty')
        
        # Prepare features
        X = self.players_df[features].fillna(0)
        y = self.players_df['total_points']
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_scaled, y)
        
        # Predict next gameweek points
        self.players_df['predicted_points'] = model.predict(X_scaled)
        
        # Captain prediction (higher expected returns)
        self.players_df['captain_score'] = (
            self.players_df['predicted_points'] * 
            (1 + self.players_df['form_float'] / 10) *
            (1 - self.players_df['fixture_difficulty'] / 10)
        )
        
        return model, scaler
    
    def analyze_current_team(self, team_data):
        """Analyze user's current team and identify issues"""
        if not team_data:
            return None
            
        picks = pd.DataFrame(team_data['current_picks']['picks'])
        current_squad = self.players_df[self.players_df['id'].isin(picks['element'])].copy()
        
        # Add captain and vice-captain info
        captain_id = picks[picks['is_captain'] == True]['element'].iloc[0] if len(picks[picks['is_captain'] == True]) > 0 else None
        vice_captain_id = picks[picks['is_vice_captain'] == True]['element'].iloc[0] if len(picks[picks['is_vice_captain'] == True]) > 0 else None
        
        analysis = {
            'squad': current_squad,
            'total_value': current_squad['price'].sum(),
            'total_points': current_squad['total_points'].sum(),
            'captain_id': captain_id,
            'vice_captain_id': vice_captain_id,
            'bank': team_data['current_picks']['entry_history']['bank'] / 10,
            'transfers_available': team_data['current_picks']['entry_history']['event_transfers'],
            'transfer_cost': team_data['current_picks']['entry_history']['event_transfers_cost']
        }
        
        # Identify issues
        issues = []
        
        # Check for injured/suspended players
        injured = current_squad[
            (current_squad['chance_of_playing_next_round'] == 0) |
            (current_squad['status'] == 'i') |
            (current_squad['status'] == 's')
        ]
        if len(injured) > 0:
            issues.append(f" {len(injured)} player(s) injured/suspended")
        
        # Check for poor form players
        poor_form = current_squad[current_squad['form_float'] < 3.0]
        if len(poor_form) > 0:
            issues.append(f" {len(poor_form)} player(s) in poor form")
        
        # Check fixture difficulty
        difficult_fixtures = current_squad[current_squad['fixture_difficulty'] > 4.0]
        if len(difficult_fixtures) > 0:
            issues.append(f"{len(difficult_fixtures)} player(s) have difficult fixtures")
        
        analysis['issues'] = issues
        return analysis
    
    def recommend_transfers(self, team_analysis, max_transfers=2):
        """Recommend optimal transfers based on team analysis"""
        if not team_analysis:
            return []
        
        current_squad = team_analysis['squad']
        bank = team_analysis['bank']
        
        recommendations = []
        
        # Priority 1: Replace injured/suspended players
        injured = current_squad[
            (current_squad['chance_of_playing_next_round'] == 0) |
            (current_squad['status'] == 'i') |
            (current_squad['status'] == 's')
        ]
        
        for _, player in injured.iterrows():
            # Find replacements in same position within budget
            budget = bank + player['price']
            position_players = self.players_df[
                (self.players_df['element_type'] == player['element_type']) &
                (self.players_df['price'] <= budget) &
                (self.players_df['id'] != player['id'])
            ].sort_values('predicted_points', ascending=False)
            
            if len(position_players) > 0:
                best_replacement = position_players.iloc[0]
                recommendations.append({
                    'type': 'Emergency',
                    'out': player['web_name'],
                    'in': best_replacement['web_name'],
                    'cost_change': best_replacement['price'] - player['price'],
                    'points_gain': best_replacement['predicted_points'] - player['predicted_points'],
                    'reason': f"Player injured/suspended"
                })
        
        # Priority 2: Value upgrades (if transfers available)
        if len(recommendations) < max_transfers:
            # Find underperforming players
            poor_performers = current_squad[
                current_squad['form_float'] < 4.0
            ].sort_values('predicted_points')
            
            for _, player in poor_performers.head(max_transfers - len(recommendations)).iterrows():
                budget = bank + player['price']
                position_players = self.players_df[
                    (self.players_df['element_type'] == player['element_type']) &
                    (self.players_df['price'] <= budget) &
                    (self.players_df['predicted_points'] > player['predicted_points'] + 1) &
                    (self.players_df['id'] != player['id'])
                ].sort_values('predicted_points', ascending=False)
                
                if len(position_players) > 0:
                    best_replacement = position_players.iloc[0]
                    recommendations.append({
                        'type': 'Upgrade',
                        'out': player['web_name'],
                        'in': best_replacement['web_name'],
                        'cost_change': best_replacement['price'] - player['price'],
                        'points_gain': best_replacement['predicted_points'] - player['predicted_points'],
                        'reason': f"Better form and fixtures"
                    })
        
        return recommendations
    
    def recommend_captain(self, team_analysis):
        """Recommend captain and vice-captain"""
        if not team_analysis:
            return None
        
        current_squad = team_analysis['squad'].copy()
        current_squad = current_squad.sort_values('captain_score', ascending=False)
        
        # Filter out unlikely to play
        playing_squad = current_squad[
            current_squad['chance_of_playing_next_round'] >= 75
        ]
        
        if len(playing_squad) == 0:
            playing_squad = current_squad
        
        captain_rec = playing_squad.iloc[0]
        vice_captain_rec = playing_squad.iloc[1] if len(playing_squad) > 1 else playing_squad.iloc[0]
        
        return {
            'captain': {
                'name': captain_rec['web_name'],
                'predicted_points': captain_rec['predicted_points'],
                'captain_score': captain_rec['captain_score'],
                'confidence': min(100, captain_rec['captain_score'] * 10)
            },
            'vice_captain': {
                'name': vice_captain_rec['web_name'],
                'predicted_points': vice_captain_rec['predicted_points'],
                'captain_score': vice_captain_rec['captain_score']
            }
        }

# Streamlit App
def main():
    st.title("⚽ FPL AI Team Optimizer")
    st.markdown("### Your Complete Season Companion")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FPLAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for user input
    with st.sidebar:
        st.header("🔧 Settings")
        team_id = st.text_input("Enter your FPL Team ID", placeholder="123456")
        
        if st.button("🔄 Refresh Data", type="primary"):
            with st.spinner("Fetching latest FPL data..."):
                if analyzer.fetch_all_data():
                    analyzer.preprocess_data()
                    analyzer.predict_points()
                    st.success("✅ Data updated successfully!")
                else:
                    st.error("❌ Failed to update data")
    
    # Main content
    if analyzer.players_df is None:
        st.info("📥 Click 'Refresh Data' to load the latest FPL information")
        return
    
    # Team Analysis Section
    if team_id:
        st.header(" Your Team Analysis")
        
        with st.spinner("Analyzing your team..."):
            team_data = analyzer.fetch_user_team(team_id)
            team_analysis = analyzer.analyze_current_team(team_data)
        
        if team_analysis:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Team Value", f"£{team_analysis['total_value']:.1f}m")
            with col2:
                st.metric("Bank", f"£{team_analysis['bank']:.1f}m")
            with col3:
                st.metric("Free Transfers", team_analysis['transfers_available'])
            with col4:
                st.metric("Total Points", f"{team_analysis['total_points']}")
            
            # Issues
            if team_analysis['issues']:
                st.warning("**Team Issues Detected:**")
                for issue in team_analysis['issues']:
                    st.write(f"• {issue}")
            else:
                st.success("No major issues detected with your team!")
            
            # Transfer Recommendations
            st.subheader("Transfer Recommendations")
            recommendations = analyzer.recommend_transfers(team_analysis)
            
            if recommendations:
                for i, rec in enumerate(recommendations):
                    with st.expander(f"{rec['type']}: {rec['out']} → {rec['in']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Cost Change:** £{rec['cost_change']:.1f}m")
                            st.write(f"**Expected Points Gain:** +{rec['points_gain']:.1f}")
                        with col2:
                            st.write(f"**Reason:** {rec['reason']}")
            else:
                st.info("No urgent transfers recommended at this time.")
            
            # Captain Recommendations
            st.subheader("Captaincy Recommendation")
            captain_rec = analyzer.recommend_captain(team_analysis)
            
            if captain_rec:
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Captain:** {captain_rec['captain']['name']}")
                    st.write(f"Predicted Points: {captain_rec['captain']['predicted_points']:.1f}")
                    st.write(f"Confidence: {captain_rec['captain']['confidence']:.0f}%")
                
                with col2:
                    st.info(f"**Vice-Captain:** {captain_rec['vice_captain']['name']}")
                    st.write(f"Predicted Points: {captain_rec['vice_captain']['predicted_points']:.1f}")
    
    # General Analytics Section
    st.header("League Analytics")
    
    # Top performers by position
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Predicted Performers")
        top_players = analyzer.players_df.nlargest(10, 'predicted_points')[
            ['web_name', 'position', 'price', 'predicted_points', 'selected_by_percent']
        ]
        st.dataframe(top_players, use_container_width=True)
    
    with col2:
        st.subheader("💎 Best Value Players")
        analyzer.players_df['value_score'] = analyzer.players_df['predicted_points'] / analyzer.players_df['price']
        value_players = analyzer.players_df.nlargest(10, 'value_score')[
            ['web_name', 'position', 'price', 'predicted_points', 'value_score']
        ]
        st.dataframe(value_players, use_container_width=True)
    
    # Price change predictions
    st.subheader("Price Change Monitor")
    
    # Simulate price change probability (in real implementation, this would use historical data)
    analyzer.players_df['price_change_prob'] = (
        (analyzer.players_df['selected_by_percent_float'] - 5) * 
        analyzer.players_df['form_float'] / 100
    ).clip(-1, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Likely to Rise:**")
        rising = analyzer.players_df[analyzer.players_df['price_change_prob'] > 0.3].nlargest(5, 'price_change_prob')[
            ['web_name', 'price', 'selected_by_percent', 'form']
        ]
        st.dataframe(rising, use_container_width=True)
    
    with col2:
        st.write("**Likely to Fall:**")
        falling = analyzer.players_df[analyzer.players_df['price_change_prob'] < -0.3].nsmallest(5, 'price_change_prob')[
            ['web_name', 'price', 'selected_by_percent', 'form']
        ]
        st.dataframe(falling, use_container_width=True)

if __name__ == "__main__":
    main()
