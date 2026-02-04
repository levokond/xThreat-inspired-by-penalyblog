import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def parse_qualifiers(shots_df):
    """
    Parse qualifier columns from the qualifiers string field.
    
    Args:
        shots_df: DataFrame with shots data containing 'qualifiers' column
        
    Returns:
        DataFrame with added qualifier columns as boolean/numeric features
    """
    qualifier_columns = ['Assisted', 'BigChance', 'Blocked', 'BlockedX', 'BlockedY', 'DirectCorner', 
                         'DirectFreekick', 'FastBreak', 'FirstTouch', 'FromCorner', 'Head', 
                         'OneOnOne', 'OtherBodyPart', 'OwnGoal', 'Penalty', 'RegularPlay', 
                         'SetPiece', 'ThrowinSetPiece', 'Volley']
    
    shots_copy = shots_df.copy()
    
    for col in qualifier_columns:
        shots_copy[col] = False
    
    for idx, qualifiers_str in shots_copy['qualifiers'].items():
        qualifiers_list = ast.literal_eval(qualifiers_str)
        for q in qualifiers_list:
            display_name = q['type']['displayName']
            if display_name in qualifier_columns:
                if 'value' in q:
                    shots_copy.at[idx, display_name] = q['value']
                else:
                    shots_copy.at[idx, display_name] = True
    
    return shots_copy


def calculate_shot_features(df):
    """
    Calculate distance and angle features for shots.
    
    Args:
        df: DataFrame with shot data containing 'x' and 'y' coordinates
        
    Returns:
        DataFrame with added 'distance' and 'angle' columns
    """
    df = df.copy()
    
    # Calculate distance in meters
    df['x_meters'] = (100 - df['x']) * 105/100
    df['c'] = abs(df['y'] - 50) * 68/100
    
    df['distance'] = np.sqrt(df['x_meters']**2 + df['c']**2)
    
    # Calculate angle to goal
    df["angle"] = np.where(
        np.arctan(7.32 * df["x_meters"] / (df["x_meters"]**2 + df["c"]**2 - (7.32/2)**2)) > 0,
        np.arctan(7.32 * df["x_meters"] / (df["x_meters"]**2 + df["c"]**2 - (7.32/2)**2)),
        np.arctan(7.32 * df["x_meters"] / (df["x_meters"]**2 + df["c"]**2 - (7.32/2)**2)) + np.pi
    )
    
    return df


def calculate_open_play_xg(events_df):
    """
    Calculate expected goals (xG) for open play shots (including penalties).
    
    Args:
        events_df: DataFrame containing all events data with columns:
                   'is_shot', 'qualifiers', 'x', 'y', 'is_goal', 'player', 'player_id'
    
    Returns:
        DataFrame with columns: ['player', 'player_id', 'total_xg']
        sorted by total_xg descending
    """
    # Filter shots
    shots = events_df[events_df['is_shot'] == True].copy()
    
    # Parse qualifiers
    shots = parse_qualifiers(shots)
    
    # Filter for open play shots (exclude headers, set pieces, own goals)
    open_play_shots = shots[
        (shots['Head'] == False) & 
        (shots['FromCorner'] == False) & 
        (shots['DirectFreekick'] == False) & 
        (shots['SetPiece'] == False) & 
        (shots['ThrowinSetPiece'] == False) & 
        (shots['DirectCorner'] == False) & 
        (shots['OwnGoal'] == False)
    ].copy()
    
    # Handle missing goals
    open_play_shots['is_goal'] = open_play_shots['is_goal'].fillna(False)
    open_play_shots.fillna(0, inplace=True)
    
    # Calculate features
    open_play_shots = calculate_shot_features(open_play_shots)
    
    # Prepare training data
    X = open_play_shots[['distance', 'angle', 'Assisted', 'FastBreak', 
                         'FirstTouch', 'OneOnOne', 'Volley', 'Penalty']]
    y = open_play_shots['is_goal']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(C=100, max_iter=1000, penalty='l1', solver='liblinear')
    model.fit(X_train, y_train)
    
    # Predict on all data
    open_play_shots['goal_probability'] = model.predict_proba(X)[:, 1]
    
    # Manual adjustment for penalties
    open_play_shots.loc[open_play_shots['Penalty'] == 1, 'goal_probability'] = 0.8
    
    # Group by player
    xg_by_player = open_play_shots.groupby(['player', 'player_id'])['goal_probability'].sum().reset_index()
    xg_by_player.columns = ['player', 'player_id', 'total_xg']
    
    return xg_by_player.sort_values('total_xg', ascending=False).reset_index(drop=True)


def calculate_header_xg(events_df):
    """
    Calculate expected goals (xG) for headed shots.
    
    Args:
        events_df: DataFrame containing all events data with columns:
                   'is_shot', 'qualifiers', 'x', 'y', 'is_goal', 'player', 'player_id'
    
    Returns:
        DataFrame with columns: ['player', 'player_id', 'total_xg']
        sorted by total_xg descending
    """
    # Filter shots
    shots = events_df[events_df['is_shot'] == True].copy()
    
    # Parse qualifiers
    shots = parse_qualifiers(shots)
    
    # Filter for headed shots (exclude own goals)
    headed_shots = shots[
        (shots['Head'] == True) & 
        (shots['OwnGoal'] == False)
    ].copy()
    
    # Handle missing goals
    headed_shots['is_goal'] = headed_shots['is_goal'].fillna(False)
    headed_shots.fillna(0, inplace=True)
    
    # Calculate features
    headed_shots = calculate_shot_features(headed_shots)
    
    # Prepare training data
    X = headed_shots[['distance', 'angle', 'Assisted', 'FromCorner']]
    y = headed_shots['is_goal']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict on all data
    headed_shots['goal_probability'] = model.predict_proba(X)[:, 1]
    
    # Group by player
    xg_by_player = headed_shots.groupby(['player', 'player_id'])['goal_probability'].sum().reset_index()
    xg_by_player.columns = ['player', 'player_id', 'total_xg']
    
    return xg_by_player.sort_values('total_xg', ascending=False).reset_index(drop=True)


def calculate_total_xg(events_df):
    """
    Calculate total expected goals (xG) combining both open play and headers.
    
    Args:
        events_df: DataFrame containing all events data with columns:
                   'is_shot', 'qualifiers', 'x', 'y', 'is_goal', 'player', 'player_id'
    
    Returns:
        DataFrame with columns: ['player', 'player_id', 'open_play_xg', 'header_xg', 'total_xg']
        sorted by total_xg descending
    """
    # Calculate both types of xG
    open_play_xg = calculate_open_play_xg(events_df)
    header_xg = calculate_header_xg(events_df)
    
    # Rename columns for merging
    open_play_xg.columns = ['player', 'player_id', 'open_play_xg']
    header_xg.columns = ['player', 'player_id', 'header_xg']
    
    # Merge the two dataframes
    total_xg = pd.merge(open_play_xg, header_xg, on=['player', 'player_id'], how='outer')
    
    # Fill NaN values with 0 (players who only have one type of shot)
    total_xg['open_play_xg'] = total_xg['open_play_xg'].fillna(0)
    total_xg['header_xg'] = total_xg['header_xg'].fillna(0)
    
    # Calculate total xG
    total_xg['total_xg'] = total_xg['open_play_xg'] + total_xg['header_xg']
    
    return total_xg.sort_values('total_xg', ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('events.csv', low_memory=False)
    
    print("Total xG by Player:")
    total_xg = calculate_total_xg(df)
    print(total_xg.head(10))
