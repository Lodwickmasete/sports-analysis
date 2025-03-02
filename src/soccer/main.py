import math
import random
from typing import Dict, List, Tuple, Any

# Load the team data
def load_team_data() -> List[Dict[str, Any]]:
    # This would normally load from a file, but we'll hard-code the data here
    return [
        {
            "club": "Vondwe Xi Bullets",
            "played": 8,
            "won": 4,
            "drawn": 3,
            "lost": 1,
            "gf": 15,
            "ga": 5,
            "gd": "+10",
            "points": 15
        },
        {
            "club": "Musina United",
            "played": 9,
            "won": 4,
            "drawn": 3,
            "lost": 2,
            "gf": 12,
            "ga": 9,
            "gd": "+3",
            "points": 15
        },
        {
            "club": "Nkowankowa Continental",
            "played": 9,
            "won": 4,
            "drawn": 2,
            "lost": 3,
            "gf": 10,
            "ga": 9,
            "gd": "+1",
            "points": 14
        },
        {
            "club": "Tzaneen United FC",
            "played": 9,
            "won": 4,
            "drawn": 2,
            "lost": 3,
            "gf": 10,
            "ga": 9,
            "gd": "+1",
            "points": 14
        },
        {
            "club": "United Artists",
            "played": 9,
            "won": 3,
            "drawn": 5,
            "lost": 1,
            "gf": 6,
            "ga": 7,
            "gd": "-1",
            "points": 14
        },
        {
            "club": "Munaca FC",
            "played": 9,
            "won": 3,
            "drawn": 4,
            "lost": 2,
            "gf": 11,
            "ga": 5,
            "gd": "+6",
            "points": 13
        },
        {
            "club": "White Vultures",
            "played": 8,
            "won": 3,
            "drawn": 4,
            "lost": 1,
            "gf": 7,
            "ga": 5,
            "gd": "+2",
            "points": 13
        },
        {
            "club": "Mpheni Defenders",
            "played": 7,
            "won": 3,
            "drawn": 3,
            "lost": 1,
            "gf": 7,
            "ga": 2,
            "gd": "+5",
            "points": 12
        },
        {
            "club": "Black Leopards FC",
            "played": 9,
            "won": 2,
            "drawn": 3,
            "lost": 4,
            "gf": 11,
            "ga": 14,
            "gd": "-3",
            "points": 9
        },
        {
            "club": "Maruleng Celtic FC",
            "played": 7,
            "won": 2,
            "drawn": 2,
            "lost": 3,
            "gf": 4,
            "ga": 6,
            "gd": "-2",
            "points": 8
        },
        {
            "club": "Winners Park (Bellevue Village)",
            "played": 9,
            "won": 2,
            "drawn": 1,
            "lost": 6,
            "gf": 6,
            "ga": 11,
            "gd": "-5",
            "points": 7
        },
        {
            "club": "Thlothlokwe Golden Syrup FC",
            "played": 9,
            "won": 0,
            "drawn": 2,
            "lost": 7,
            "gf": 2,
            "ga": 19,
            "gd": "-17",
            "points": 2
        }
    ]
    

# Validate the integrity of team data
def validate_team_data(teams: List[Dict[str, Any]]) -> List[str]:
    """Validate team data for consistency and correctness"""
    issues = []
    
    for team in teams:
        # Check if games played matches the sum of results
        total_results = team["won"] + team["drawn"] + team["lost"]
        if team["played"] != total_results:
            issues.append(f"{team['club']}: Games played ({team['played']}) doesn't match sum of results ({total_results})")
        
        # Check if points calculation is correct (3 for win, 1 for draw)
        expected_points = (team["won"] * 3) + team["drawn"]
        if team["points"] != expected_points:
            issues.append(f"{team['club']}: Points ({team['points']}) doesn't match expected points ({expected_points})")
            
        # Check if goal difference is calculated correctly
        expected_gd = team["gf"] - team["ga"]
        actual_gd = int(team["gd"].replace("+", "")) if team["gd"] not in ["-0", "+0"] else 0
        if expected_gd != actual_gd:
            issues.append(f"{team['club']}: Goal difference ({team['gd']}) doesn't match expected GD ({expected_gd})")
    
    return issues

# Calculate league-wide statistics
def calculate_league_stats(teams: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate overall league statistics"""
    total_matches = sum(team["played"] for team in teams) / 2  # Each match counts twice (once for each team)
    total_goals = sum(team["gf"] for team in teams)
    total_wins = sum(team["won"] for team in teams)
    total_draws = sum(team["drawn"] for team in teams)
    
    return {
        "avg_goals_per_match": total_goals / max(1, total_matches),
        "win_percentage": total_wins / max(1, total_wins + total_draws + sum(team["lost"] for team in teams)),
        "draw_percentage": total_draws / max(1, total_matches * 2),
        "home_advantage_factor": 1.3  # Estimated from typical soccer leagues
    }

# Calculate advanced team metrics using Bayesian methods
def calculate_team_metrics(teams: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calculate comprehensive team performance metrics with Bayesian adjustments"""
    metrics = {}
    league_stats = calculate_league_stats(teams)
    
    # League averages for reference
    avg_goals_per_game = league_stats["avg_goals_per_match"]
    
    # Prior estimates (Bayesian approach)
    prior_matches = 3  # Weight of prior knowledge
    prior_win_rate = 0.33  # Expected win rate for an average team
    prior_draw_rate = 0.25  # Expected draw rate for an average team
    prior_goals_per_game = avg_goals_per_game / 2  # Expected goals per game for average team
    
    for team in teams:
        # Skip if team hasn't played any matches
        if team["played"] == 0:
            continue
            
        # Basic statistics with Bayesian adjustments
        win_rate = (team["won"] + (prior_win_rate * prior_matches)) / (team["played"] + prior_matches)
        draw_rate = (team["drawn"] + (prior_draw_rate * prior_matches)) / (team["played"] + prior_matches)
        loss_rate = (team["lost"] + ((1 - prior_win_rate - prior_draw_rate) * prior_matches)) / (team["played"] + prior_matches)
        
        # Goals scored/conceded with Bayesian adjustments
        goals_scored_per_game = (team["gf"] + (prior_goals_per_game * prior_matches)) / (team["played"] + prior_matches)
        goals_conceded_per_game = (team["ga"] + (prior_goals_per_game * prior_matches)) / (team["played"] + prior_matches)
        
        # Points per game
        points_per_game = team["points"] / team["played"]
        
        # Recent form (would ideally use actual recent results if available)
        # For now, we'll estimate form from win/loss ratio
        estimated_form = win_rate - loss_rate
        
        # Attacking strength (how many more goals team scores compared to average)
        offensive_strength = goals_scored_per_game / max(0.1, avg_goals_per_game / 2)
        
        # Defensive strength (inverse of conceded goals compared to average)
        defensive_strength = (avg_goals_per_game / 2) / max(0.1, goals_conceded_per_game)
        
        # Overall team strength (composite metric)
        overall_strength = (win_rate * 2.5) + (draw_rate * 0.5) + (offensive_strength * 1.2) + (defensive_strength * 1.2)
        
        # Home/Away performance adjustment
        # In reality, this would use actual home/away splits, but we'll estimate
        home_bias = random.uniform(0.9, 1.1)  # Random factor as we don't have home/away splits
        away_bias = random.uniform(0.9, 1.1)
        
        # Consistency factor (would ideally use variance in results if available)
        consistency = random.uniform(0.8, 1.2)  # Random factor as proxy
        
        metrics[team["club"]] = {
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
            "goals_scored_per_game": goals_scored_per_game,
            "goals_conceded_per_game": goals_conceded_per_game,
            "points_per_game": points_per_game,
            "offensive_strength": offensive_strength,
            "defensive_strength": defensive_strength,
            "overall_strength": overall_strength,
            "estimated_form": estimated_form,
            "home_bias": home_bias,
            "away_bias": away_bias,
            "consistency": consistency
        }
    
    return metrics

# Calculate Poisson probability for exact goal count
def poisson_probability(lambda_val: float, k: int) -> float:
    """Calculate the Poisson probability mass function"""
    # P(X = k) = (λ^k * e^-λ) / k!
    try:
        return (lambda_val ** k * math.exp(-lambda_val)) / math.factorial(k)
    except OverflowError:
        # Handle large factorials more safely
        log_p = k * math.log(lambda_val) - lambda_val - sum(math.log(i) for i in range(1, k + 1))
        return math.exp(log_p)

# Calculate match probabilities using Dixon-Coles model
def calculate_match_probabilities(home_team: str, away_team: str, team_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate match outcome probabilities using adjusted Poisson model"""
    if home_team not in team_metrics or away_team not in team_metrics:
        raise ValueError(f"Team not found: {home_team if home_team not in team_metrics else away_team}")
    
    # Get team metrics
    home_metrics = team_metrics[home_team]
    away_metrics = team_metrics[away_team]
    
    # Home advantage factor
    home_advantage = 1.3
    
    # Expected goals calculation
    home_attack = home_metrics["offensive_strength"] * home_advantage * home_metrics["home_bias"]
    home_defense = home_metrics["defensive_strength"]
    away_attack = away_metrics["offensive_strength"] * away_metrics["away_bias"]
    away_defense = away_metrics["defensive_strength"]
    
    # Adjusted expected goals using Dixon-Coles approach
    home_expected_goals = home_attack / away_defense
    away_expected_goals = away_attack / home_defense
    
    # Calculate low-scoring correction (for 0-0, 1-0, 0-1, 1-1 outcomes)
    # This adjusts for the fact that these scores are more common than Poisson would predict
    rho = -0.1  # Typical value from research
    
    # Calculate match outcome probabilities using Poisson distribution
    max_goals = 10  # Maximum goals to consider for each team
    
    # Initialize counters for home win, draw, away win
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0
    
    # Calculate all score probabilities up to max_goals
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            # Basic Poisson probabilities
            p_home = poisson_probability(home_expected_goals, home_goals)
            p_away = poisson_probability(away_expected_goals, away_goals)
            
            # Apply Dixon-Coles correction for low scores
            if home_goals <= 1 and away_goals <= 1:
                if home_goals == 0 and away_goals == 0:
                    correction = 1 - rho
                elif home_goals == 1 and away_goals == 0:
                    correction = 1 + rho
                elif home_goals == 0 and away_goals == 1:
                    correction = 1 + rho
                else:  # 1-1
                    correction = 1 - rho
                p_score = p_home * p_away * correction
            else:
                p_score = p_home * p_away
            
            # Update outcome probabilities
            if home_goals > away_goals:
                p_home_win += p_score
            elif home_goals == away_goals:
                p_draw += p_score
            else:
                p_away_win += p_score
    
    # Normalize probabilities to ensure they sum to 1
    total_prob = p_home_win + p_draw + p_away_win
    p_home_win /= total_prob
    p_draw /= total_prob
    p_away_win /= total_prob
    
    # Apply team form adjustment
    form_factor = 0.1
    form_diff = home_metrics["estimated_form"] - away_metrics["estimated_form"]
    p_home_win = min(0.95, max(0.05, p_home_win + (form_diff * form_factor)))
    p_away_win = min(0.95, max(0.05, p_away_win - (form_diff * form_factor)))
    p_draw = 1 - p_home_win - p_away_win
    
    # Round and convert to percentages
    return {
        "home_win_probability": round(p_home_win * 100, 2),
        "away_win_probability": round(p_away_win * 100, 2),
        "draw_probability": round(p_draw * 100, 2),
        "home_win_odds": round(1 / max(0.01, p_home_win), 2),
        "away_win_odds": round(1 / max(0.01, p_away_win), 2),
        "draw_odds": round(1 / max(0.01, p_draw), 2),
        "home_expected_goals": round(home_expected_goals, 2),
        "away_expected_goals": round(away_expected_goals, 2)
    }

# Calculate correct score probabilities
def calculate_correct_score_probabilities(home_team: str, away_team: str, team_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calculate probabilities for specific scorelines"""
    if home_team not in team_metrics or away_team not in team_metrics:
        raise ValueError(f"Team not found: {home_team if home_team not in team_metrics else away_team}")
    
    # Get team metrics
    home_metrics = team_metrics[home_team]
    away_metrics = team_metrics[away_team]
    
    # Home advantage factor
    home_advantage = 1.3
    
    # Expected goals calculation
    home_attack = home_metrics["offensive_strength"] * home_advantage * home_metrics["home_bias"]
    home_defense = home_metrics["defensive_strength"]
    away_attack = away_metrics["offensive_strength"] * away_metrics["away_bias"]
    away_defense = away_metrics["defensive_strength"]
    
    # Adjusted expected goals
    home_expected_goals = home_attack / away_defense
    away_expected_goals = away_attack / home_defense
    
    # Calculate low-scoring correction (for 0-0, 1-0, 0-1, 1-1 outcomes)
    rho = -0.1  # Typical value from research
    
    # Calculate score probabilities
    max_goals = 5  # Maximum goals to consider for each team
    score_probs = {}
    total_prob = 0
    
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            # Basic Poisson probabilities
            p_home = poisson_probability(home_expected_goals, home_goals)
            p_away = poisson_probability(away_expected_goals, away_goals)
            
            # Apply Dixon-Coles correction for low scores
            if home_goals <= 1 and away_goals <= 1:
                if home_goals == 0 and away_goals == 0:
                    correction = 1 - rho
                elif home_goals == 1 and away_goals == 0:
                    correction = 1 + rho
                elif home_goals == 0 and away_goals == 1:
                    correction = 1 + rho
                else:  # 1-1
                    correction = 1 - rho
                p_score = p_home * p_away * correction
            else:
                p_score = p_home * p_away
            
            score_key = f"{home_goals}-{away_goals}"
            score_probs[score_key] = p_score
            total_prob += p_score
    
    # Normalize and format probabilities
    formatted_probs = {}
    for score, prob in score_probs.items():
        normalized_prob = prob / total_prob
        formatted_probs[score] = round(normalized_prob * 100, 2)
    
    # Sort by probability (descending)
    return dict(sorted(formatted_probs.items(), key=lambda item: item[1], reverse=True))

# Main function to predict match outcome with comprehensive information
def predict_match(home_team: str, away_team: str) -> Dict[str, Any]:
    """Generate comprehensive match prediction"""
    teams = load_team_data()
    
    # Validate team data
    validation_issues = validate_team_data(teams)
    if validation_issues:
        print("Warning: Data validation issues detected:")
        for issue in validation_issues:
            print(f"  - {issue}")
    
    # Calculate team metrics
    team_metrics = calculate_team_metrics(teams)
    
    # Get outcome probabilities
    outcome_probs = calculate_match_probabilities(home_team, away_team, team_metrics)
    
    # Get correct score probabilities
    score_probs = calculate_correct_score_probabilities(home_team, away_team, team_metrics)
    
    # Get team data for display
    home_team_data = None
    away_team_data = None
    
    for team in teams:
        if team["club"] == home_team:
            home_team_data = team
        if team["club"] == away_team:
            away_team_data = team
    
    if not home_team_data or not away_team_data:
        raise ValueError(f"Team not found: {home_team if not home_team_data else away_team}")
    
    # Get top 5 most likely scores
    top_scores = dict(list(score_probs.items())[:5])
    
    # Calculate over/under probabilities
    over_under_2_5 = sum(prob for score, prob in score_probs.items() 
                         if sum(map(int, score.split('-'))) > 2.5)
    
    # Calculate both teams to score probability
    btts_yes = sum(prob for score, prob in score_probs.items() 
                   if int(score.split('-')[0]) > 0 and int(score.split('-')[1]) > 0)
    
    return {
        "match": f"{home_team} vs {away_team}",
        "home_team_record": f"W{home_team_data['won']}-D{home_team_data['drawn']}-L{home_team_data['lost']}",
        "away_team_record": f"W{away_team_data['won']}-D{away_team_data['drawn']}-L{away_team_data['lost']}",
        "home_team_stats": {
            "goals_for": home_team_data["gf"],
            "goals_against": home_team_data["ga"],
            "points": home_team_data["points"],
            "points_per_game": round(home_team_data["points"] / home_team_data["played"], 2)
        },
        "away_team_stats": {
            "goals_for": away_team_data["gf"],
            "goals_against": away_team_data["ga"],
            "points": away_team_data["points"],
            "points_per_game": round(away_team_data["points"] / away_team_data["played"], 2)
        },
        "outcome_probabilities": {
            "home_win": outcome_probs["home_win_probability"],
            "draw": outcome_probs["draw_probability"],
            "away_win": outcome_probs["away_win_probability"]
        },
        "betting_odds": {
            "home_win": outcome_probs["home_win_odds"],
            "draw": outcome_probs["draw_odds"],
            "away_win": outcome_probs["away_win_odds"]
        },
        "expected_goals": {
            "home": outcome_probs["home_expected_goals"],
            "away": outcome_probs["away_expected_goals"]
        },
        "most_likely_scores": top_scores,
        "market_probabilities": {
            "over_2.5_goals": round(over_under_2_5, 2),
            "under_2.5_goals": round(100 - over_under_2_5, 2),
            "btts_yes": round(btts_yes, 2),
            "btts_no": round(100 - btts_yes, 2)
        }
    }

# Function to display all available teams
def list_teams():
    """List all available teams with their current standings"""
    teams = load_team_data()
    
    # Sort teams by points and then goal difference
    sorted_teams = sorted(teams, 
                         key=lambda x: (x["points"], int(x["gd"].replace("+", "")), x["gf"]), 
                         reverse=True)
    
    print("\n===== TEAM STANDINGS =====")
    print(f"{'Pos':<4}{'Team':<35}{'P':<4}{'W':<4}{'D':<4}{'L':<4}{'GF':<4}{'GA':<4}{'GD':<6}{'Pts':<4}")
    print("-" * 75)
    
    for i, team in enumerate(sorted_teams, 1):
        print(f"{i:<4}{team['club']:<35}{team['played']:<4}{team['won']:<4}{team['drawn']:<4}" +
              f"{team['lost']:<4}{team['gf']:<4}{team['ga']:<4}{team['gd']:<6}{team['points']:<4}")
    
    team_names = [team["club"] for team in teams]
    return team_names

# Function to simulate a match based on probabilities
def simulate_match(home_team: str, away_team: str, num_simulations: int = 1000) -> Dict[str, Any]:
    """Run Monte Carlo simulations of the match"""
    teams = load_team_data()
    team_metrics = calculate_team_metrics(teams)
    
    # Get match probabilities
    outcome_probs = calculate_match_probabilities(home_team, away_team, team_metrics)
    score_probs = calculate_correct_score_probabilities(home_team, away_team, team_metrics)
    
    # Expected goals for Poisson simulation
    home_expected_goals = outcome_probs["home_expected_goals"]
    away_expected_goals = outcome_probs["away_expected_goals"]
    
    # Simulation results
    results = {
        "home_wins": 0,
        "draws": 0,
        "away_wins": 0,
        "score_distribution": {},
        "goals_distribution": {
            "home": [0] * 11,  # 0-10 goals
            "away": [0] * 11
        }
    }
    
    for _ in range(num_simulations):
        # Simulate scoreline using Poisson distribution
        home_goals = 0
        away_goals = 0
        
        # Use inverse transform sampling for more accurate simulation
        r = random.random()
        cumulative_prob = 0
        
        for score, prob in score_probs.items():
            cumulative_prob += prob / 100
            if r <= cumulative_prob:
                home_goals, away_goals = map(int, score.split('-'))
                break
        
        # Update results
        if home_goals > away_goals:
            results["home_wins"] += 1
        elif home_goals == away_goals:
            results["draws"] += 1
        else:
            results["away_wins"] += 1
        
        # Track score distribution
        score_key = f"{home_goals}-{away_goals}"
        results["score_distribution"][score_key] = results["score_distribution"].get(score_key, 0) + 1
        
        # Track goals distribution
        if home_goals <= 10:
            results["goals_distribution"]["home"][home_goals] += 1
        if away_goals <= 10:
            results["goals_distribution"]["away"][away_goals] += 1
    
    # Convert to percentages
    for key in ["home_wins", "draws", "away_wins"]:
        results[key] = round(results[key] / num_simulations * 100, 2)
    
    # Sort and format score distribution
    score_dist = {}
    for score, count in results["score_distribution"].items():
        score_dist[score] = round(count / num_simulations * 100, 2)
    
    # Sort by frequency
    score_dist = dict(sorted(score_dist.items(), key=lambda x: x[1], reverse=True))
    
    # Convert goals distribution to percentages
    for team in ["home", "away"]:
        for i in range(len(results["goals_distribution"][team])):
            results["goals_distribution"][team][i] = round(
                results["goals_distribution"][team][i] / num_simulations * 100, 2
            )
    
    # Get team stats
    home_team_data = None
    away_team_data = None
    
    for team in teams:
        if team["club"] == home_team:
            home_team_data = team
        if team["club"] == away_team:
            away_team_data = team
    
    return {
        "match": f"{home_team} vs {away_team}",
        "outcome_probabilities": {
            "home_win": results["home_wins"],
            "draw": results["draws"],
            "away_win": results["away_wins"]
        },
        "expected_goals": {
            "home": round(home_expected_goals, 2),
            "away": round(away_expected_goals, 2)
        },
        "most_likely_scores": dict(list(score_dist.items())[:5]),
        "goals_distribution": results["goals_distribution"],
        "team_records": {
            "home": f"W{home_team_data['won']}-D{home_team_data['drawn']}-L{home_team_data['lost']}",
            "away": f"W{away_team_data['won']}-D{away_team_data['drawn']}-L{away_team_data['lost']}"
        }
    }



# Function to analyze all matchups in the league (continued)
def analyze_all_matchups() -> Dict[str, List[Dict[str, Any]]]:
    """Generate predictions for all possible matches in the league"""
    teams = load_team_data()
    team_metrics = calculate_team_metrics(teams)
    
    team_names = [team["club"] for team in teams]
    
    matches = []
    
    for home_idx, home_team in enumerate(team_names):
        for away_idx, away_team in enumerate(team_names):
            if home_idx != away_idx:  # Skip matches against same team
                try:
                    outcome_probs = calculate_match_probabilities(home_team, away_team, team_metrics)
                    matches.append({
                        "match": f"{home_team} vs {away_team}",
                        "home_win": outcome_probs["home_win_probability"],
                        "draw": outcome_probs["draw_probability"],
                        "away_win": outcome_probs["away_win_probability"],
                        "home_expected_goals": outcome_probs["home_expected_goals"],
                        "away_expected_goals": outcome_probs["away_expected_goals"]
                    })
                except Exception as e:
                    print(f"Error analyzing {home_team} vs {away_team}: {str(e)}")
    
    # Group matches by team
    matches_by_team = {}
    for team in team_names:
        home_matches = [m for m in matches if m["match"].startswith(f"{team} vs")]
        away_matches = [m for m in matches if "vs " + team in m["match"]]
        matches_by_team[team] = {
            "home_matches": home_matches,
            "away_matches": away_matches,
            "all_matches": home_matches + away_matches
        }
    
    # Find most likely upsets
    upsets = []
    for match in matches:
        parts = match["match"].split(" vs ")
        home_team, away_team = parts[0], parts[1]
        
        # Get team rankings based on points
        home_team_points = next((team["points"] for team in teams if team["club"] == home_team), 0)
        away_team_points = next((team["points"] for team in teams if team["club"] == away_team), 0)
        
        # Define an upset as when a team with significantly fewer points has a good chance to win
        points_diff = home_team_points - away_team_points
        
        if points_diff > 10 and match["away_win"] > 30:
            # Away team is much weaker but has a good chance
            upsets.append({
                "match": match["match"],
                "upset_probability": match["away_win"],
                "points_difference": points_diff,
                "underdog": away_team
            })
        elif points_diff < -10 and match["home_win"] > 30:
            # Home team is much weaker but has a good chance
            upsets.append({
                "match": match["match"],
                "upset_probability": match["home_win"],
                "points_difference": -points_diff,
                "underdog": home_team
            })
    
    # Sort upsets by upset probability
    upsets = sorted(upsets, key=lambda x: x["upset_probability"], reverse=True)
    
    return {
        "all_matches": matches,
        "matches_by_team": matches_by_team,
        "potential_upsets": upsets[:10]  # Top 10 potential upsets
    }

# Function to simulate a full season
def simulate_season(num_simulations: int = 100) -> Dict[str, Any]:
    """Simulate the remainder of the season multiple times and calculate final standings"""
    teams = load_team_data()
    team_metrics = calculate_team_metrics(teams)
    team_names = [team["club"] for team in teams]
    
    # Create a copy of current standings to use as a base for simulations
    current_standings = {}
    for team in teams:
        current_standings[team["club"]] = {
            "points": team["points"],
            "gd": int(team["gd"].replace("+", "")) if team["gd"] not in ["-0", "+0"] else 0,
            "gf": team["gf"],
            "played": team["played"]
        }
    
    # Track final positions across all simulations
    final_positions = {team: [0] * len(team_names) for team in team_names}
    champions = {team: 0 for team in team_names}
    relegated = {team: 0 for team in team_names}
    
    # Calculate remaining matches
    total_matches = len(team_names) * (len(team_names) - 1)  # Each team plays every other team once
    matches_per_team = len(team_names) - 1  # Number of matches each team plays
    total_played = sum(team["played"] for team in teams) / 2  # Each match counts for 2 teams
    remaining_matches = total_matches - total_played
    
    # Check if we have fixtures data
    has_fixtures = False  # Set to True if you have actual fixture data
    
    if not has_fixtures:
        # Generate generic remaining fixtures
        # This is simplified and doesn't account for teams that may have played each other already
        fixtures = []
        for home_idx, home_team in enumerate(team_names):
            for away_idx, away_team in enumerate(team_names):
                if home_idx != away_idx:  # Skip matches against same team
                    home_played = current_standings[home_team]["played"]
                    if home_played < matches_per_team:
                        fixtures.append((home_team, away_team))
        
        # Randomly select from the fixtures to match the expected remaining matches
        if len(fixtures) > remaining_matches:
            fixtures = random.sample(fixtures, int(remaining_matches))
    
    # Run simulations
    for sim in range(num_simulations):
        # Create a copy of current standings for this simulation
        sim_standings = {team: current_standings[team].copy() for team in current_standings}
        
        # Simulate each remaining match
        for home_team, away_team in fixtures:
            # Get match probabilities
            try:
                outcome_probs = calculate_match_probabilities(home_team, away_team, team_metrics)
                
                # Simulate match outcome
                r = random.random() * 100
                
                if r < outcome_probs["home_win_probability"]:
                    # Home win
                    sim_standings[home_team]["points"] += 3
                    
                    # Simulate goals (simplified)
                    home_goals = max(1, min(5, round(random.gauss(outcome_probs["home_expected_goals"], 1))))
                    away_goals = max(0, min(home_goals - 1, round(random.gauss(outcome_probs["away_expected_goals"], 1))))
                    
                elif r < outcome_probs["home_win_probability"] + outcome_probs["draw_probability"]:
                    # Draw
                    sim_standings[home_team]["points"] += 1
                    sim_standings[away_team]["points"] += 1
                    
                    # Simulate goals for a draw
                    home_goals = max(0, min(3, round(random.gauss(outcome_probs["home_expected_goals"], 0.5))))
                    away_goals = home_goals
                    
                else:
                    # Away win
                    sim_standings[away_team]["points"] += 3
                    
                    # Simulate goals
                    away_goals = max(1, min(5, round(random.gauss(outcome_probs["away_expected_goals"], 1))))
                    home_goals = max(0, min(away_goals - 1, round(random.gauss(outcome_probs["home_expected_goals"], 1))))
                
                # Update goal difference and goals for
                sim_standings[home_team]["gd"] += (home_goals - away_goals)
                sim_standings[away_team]["gd"] += (away_goals - home_goals)
                sim_standings[home_team]["gf"] += home_goals
                sim_standings[away_team]["gf"] += away_goals
                
                # Update matches played
                sim_standings[home_team]["played"] += 1
                sim_standings[away_team]["played"] += 1
                
            except Exception as e:
                print(f"Error simulating {home_team} vs {away_team}: {str(e)}")
        
        # Sort teams by points, goal difference, and goals scored
        sorted_teams = sorted(sim_standings.items(), 
                             key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]), 
                             reverse=True)
        
        # Record final positions
        for pos, (team, _) in enumerate(sorted_teams):
            final_positions[team][pos] += 1
            
            # Record champions and relegated teams
            if pos == 0:
                champions[team] += 1
            elif pos >= len(team_names) - 3:  # Bottom 3 teams relegated
                relegated[team] += 1
    
    # Calculate average position and position probabilities
    avg_positions = {}
    position_probs = {}
    
    for team in team_names:
        # Average position
        total_pos = sum(pos * count for pos, count in enumerate(final_positions[team]))
        avg_positions[team] = round(total_pos / num_simulations + 1, 2)  # +1 because positions are 1-based
        
        # Position probabilities
        position_probs[team] = [round(count / num_simulations * 100, 2) for count in final_positions[team]]
    
    # Calculate championship and relegation probabilities
    champion_probs = {team: round(count / num_simulations * 100, 2) for team, count in champions.items()}
    relegation_probs = {team: round(count / num_simulations * 100, 2) for team, count in relegated.items()}
    
    # Sort results by average position
    sorted_positions = sorted(avg_positions.items(), key=lambda x: x[1])
    
    # Prepare detailed results
    detailed_results = []
    for team, avg_pos in sorted_positions:
        detailed_results.append({
            "team": team,
            "avg_position": avg_pos,
            "champion_prob": champion_probs[team],
            "relegation_prob": relegation_probs[team],
            "position_distribution": {f"Pos {i+1}": position_probs[team][i] for i in range(len(team_names))}
        })
    
    return {
        "detailed_results": detailed_results,
        "champion_probabilities": sorted(champion_probs.items(), key=lambda x: x[1], reverse=True),
        "relegation_probabilities": sorted(relegation_probs.items(), key=lambda x: x[1], reverse=True)
    }

# Function to estimate "value bets" based on model vs market odds
def find_value_bets(market_odds: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """Identify potential value bets by comparing model predictions with market odds"""
    teams = load_team_data()
    team_metrics = calculate_team_metrics(teams)
    
    value_bets = []
    
    for match, odds in market_odds.items():
        # Parse match string to get teams
        parts = match.split(" vs ")
        if len(parts) != 2:
            continue
            
        home_team, away_team = parts[0], parts[1]
        
        try:
            # Get model predictions
            pred = calculate_match_probabilities(home_team, away_team, team_metrics)
            
            # Calculate implied probabilities from market odds
            home_implied_prob = 1 / odds["home"] * 100
            draw_implied_prob = 1 / odds["draw"] * 100
            away_implied_prob = 1 / odds["away"] * 100
            
            # Calculate overround (bookmaker margin)
            overround = (home_implied_prob + draw_implied_prob + away_implied_prob) / 100
            
            # Calculate fair implied probabilities (removing overround)
            home_fair_implied = home_implied_prob / overround
            draw_fair_implied = draw_implied_prob / overround
            away_fair_implied = away_implied_prob / overround
            
            # Calculate edge (model probability - fair implied probability)
            home_edge = pred["home_win_probability"] - home_fair_implied
            draw_edge = pred["draw_probability"] - draw_fair_implied
            away_edge = pred["away_win_probability"] - away_fair_implied
            
            # Calculate expected value (probability * decimal odds - 1)
            home_ev = (pred["home_win_probability"] / 100 * odds["home"]) - 1
            draw_ev = (pred["draw_probability"] / 100 * odds["draw"]) - 1
            away_ev = (pred["away_win_probability"] / 100 * odds["away"]) - 1
            
            # Find the best value bet
            best_edge = max(home_edge, draw_edge, away_edge)
            best_ev = max(home_ev, draw_ev, away_ev)
            
            if best_edge > 5 and best_ev > 0.05:  # Threshold for value (5% edge, 5% EV)
                if best_edge == home_edge:
                    bet_type = "home"
                    model_prob = pred["home_win_probability"]
                    market_prob = home_fair_implied
                    market_odds = odds["home"]
                    ev = home_ev
                elif best_edge == draw_edge:
                    bet_type = "draw"
                    model_prob = pred["draw_probability"]
                    market_prob = draw_fair_implied
                    market_odds = odds["draw"]
                    ev = draw_ev
                else:
                    bet_type = "away"
                    model_prob = pred["away_win_probability"]
                    market_prob = away_fair_implied
                    market_odds = odds["away"]
                    ev = away_ev
                
                value_bets.append({
                    "match": match,
                    "bet_type": bet_type,
                    "model_probability": round(model_prob, 2),
                    "market_implied_probability": round(market_prob, 2),
                    "edge": round(best_edge, 2),
                    "market_odds": round(market_odds, 2),
                    "expected_value": round(ev * 100, 2),
                    "overround": round((overround - 1) * 100, 2)
                })
        
        except Exception as e:
            print(f"Error analyzing {match}: {str(e)}")
    
    # Sort by expected value
    return sorted(value_bets, key=lambda x: x["expected_value"], reverse=True)

# Function to generate a readable match report
def generate_match_report(home_team: str, away_team: str) -> str:
    """Generate a human-readable match report with predictions"""
    prediction = predict_match(home_team, away_team)
    
    # Format the report
    report = [
        f"=== MATCH PREDICTION: {home_team} vs {away_team} ===\n",
        f"Team Records:",
        f"  {home_team}: {prediction['home_team_record']}",
        f"  {away_team}: {prediction['away_team_record']}\n",
        f"Team Stats:",
        f"  {home_team}: {prediction['home_team_stats']['goals_for']} goals scored, {prediction['home_team_stats']['goals_against']} conceded",
        f"  {away_team}: {prediction['away_team_stats']['goals_for']} goals scored, {prediction['away_team_stats']['goals_against']} conceded\n",
        f"Match Outcome Probabilities:",
        f"  {home_team} Win: {prediction['outcome_probabilities']['home_win']}%",
        f"  Draw: {prediction['outcome_probabilities']['draw']}%",
        f"  {away_team} Win: {prediction['outcome_probabilities']['away_win']}%\n",
        f"Expected Goals:",
        f"  {home_team}: {prediction['expected_goals']['home']}",
        f"  {away_team}: {prediction['expected_goals']['away']}\n",
        f"Most Likely Scorelines:"
    ]
    
    # Add most likely scores
    for score, prob in prediction['most_likely_scores'].items():
        home_goals, away_goals = score.split('-')
        report.append(f"  {home_team} {home_goals}-{away_goals} {away_team}: {prob}%")
    
    report.extend([
        f"\nMarket Probabilities:",
        f"  Over 2.5 goals: {prediction['market_probabilities']['over_2.5_goals']}%",
        f"  Both Teams To Score: {prediction['market_probabilities']['btts_yes']}%"
    ])
    
    return "\n".join(report)

# Function to run a simple command-line interface
def main():
    """Run the prediction system with a simple CLI"""
    print("\n=== SOCCER MATCH PREDICTION SYSTEM ===")
    print("Based on statistical modeling and Bayesian inference")
    
    while True:
        print("\nOptions:")
        print("1. List all teams")
        print("2. Predict a match")
        print("3. Simulate a match")
        print("4. Generate season simulation")
        print("5. Quit")
        
        choice = input("\nChoose an option (1-5): ")
        
        if choice == "1":
            team_names = list_teams()
            
        elif choice == "2":
            team_names = list_teams()
            
            print("\nEnter team names from the list above:")
            home_team = input("Home team: ")
            away_team = input("Away team: ")
            
            if home_team in team_names and away_team in team_names:
                report = generate_match_report(home_team, away_team)
                print("\n" + report)
            else:
                print("Invalid team names. Please select from the list.")
        
        elif choice == "3":
            team_names = list_teams()
            
            print("\nEnter team names from the list above:")
            home_team = input("Home team: ")
            away_team = input("Away team: ")
            
            if home_team in team_names and away_team in team_names:
                num_sims = int(input("Number of simulations (default 1000): ") or "1000")
                sim_results = simulate_match(home_team, away_team, num_sims)
                
                print(f"\n=== MATCH SIMULATION: {home_team} vs {away_team} ===")
                print(f"Based on {num_sims} simulations\n")
                
                print("Outcome Probabilities:")
                print(f"  {home_team} Win: {sim_results['outcome_probabilities']['home_win']}%")
                print(f"  Draw: {sim_results['outcome_probabilities']['draw']}%")
                print(f"  {away_team} Win: {sim_results['outcome_probabilities']['away_win']}%\n")
                
                print("Expected Goals:")
                print(f"  {home_team}: {sim_results['expected_goals']['home']}")
                print(f"  {away_team}: {sim_results['expected_goals']['away']}\n")
                
                print("Most Likely Scorelines:")
                for score, prob in sim_results['most_likely_scores'].items():
                    home_goals, away_goals = score.split('-')
                    print(f"  {home_team} {home_goals}-{away_goals} {away_team}: {prob}%")
            else:
                print("Invalid team names. Please select from the list.")
        
        elif choice == "4":
            num_sims = int(input("Number of season simulations (default 100): ") or "100")
            print(f"\nRunning {num_sims} season simulations...")
            
            season_results = simulate_season(num_sims)
            
            print("\n=== SEASON SIMULATION RESULTS ===")
            print("Championship Probabilities:")
            for team, prob in season_results["champion_probabilities"][:5]:  # Top 5
                print(f"  {team}: {prob}%")
            
            print("\nRelegation Probabilities:")
            for team, prob in season_results["relegation_probabilities"][:5]:  # Top 5
                print(f"  {team}: {prob}%")
            
            print("\nDetailed Team Predictions:")
            for team_result in season_results["detailed_results"][:10]:  # Top 10
                print(f"  {team_result['team']}: Avg Position {team_result['avg_position']}, " +
                      f"Win League {team_result['champion_prob']}%, " +
                      f"Relegation {team_result['relegation_prob']}%")
        
        elif choice == "5":
            print("\nThank you for using the Soccer Match Prediction System!")
            break
            
        else:
            print("Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()