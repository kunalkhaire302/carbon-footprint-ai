import joblib
import pandas as pd
import numpy as np

# Load preprocessor and model globally to avoid loading on every request
try:
    preprocessor = joblib.load('model/preprocessor.pkl')
    model = joblib.load('model/carbon_model.pkl')
except FileNotFoundError:
    preprocessor = None
    model = None

# Averages for percentile calculation
INDIA_AVG = 1.9
WORLD_AVG = 4.7

def predict_emission(input_data):
    """
    Predicts the total carbon footprint using the loaded ML model.
    """
    if model is None or preprocessor is None:
        raise Exception("Model or Preprocessor not found. Please train models first.")
    
    # Needs to match the DataFrame structure generated in dataset.py
    df_input = pd.DataFrame([input_data])
    
    X_processed = preprocessor.transform(df_input)
    prediction = model.predict(X_processed)[0]
    
    return round(float(prediction), 2)

def compute_breakdown(input_data):
    """
    Calculates category-wise emissions deterministically to provide insights.
    Matches logic in dataset.py.
    """
    household_size = max(1, input_data.get('household_size', 1))
    
    elec = (input_data.get('electricity_usage_kwh', 0) * 12 * 0.4) / 1000.0 / household_size
    
    emission_factors = {'petrol': 0.2, 'diesel': 0.23, 'hybrid': 0.12, 'electric': 0.05, 'none': 0.0}
    veh_type = input_data.get('vehicle_type', 'none')
    veh_km = input_data.get('vehicle_km', 0) if veh_type != 'none' else 0
    transport = (emission_factors.get(veh_type, 0.0) * veh_km * 12) / 1000.0
    
    flight = (input_data.get('flights_short_haul', 0) * 0.15) + (input_data.get('flights_long_haul', 0) * 0.8)
    transport += flight # Combine vehicle and flights into transport
    
    diet_factors = {'vegan': 1.0, 'vegetarian': 1.3, 'pescatarian': 1.6, 'non-vegetarian': 2.3}
    diet = diet_factors.get(input_data.get('diet_type', 'non-vegetarian'), 2.3)
    
    waste = (input_data.get('waste_kg_weekly', 0) * 52 * 0.05) / 1000.0
    
    goods = (input_data.get('grocery_spend_monthly', 0) * 12 * 0.001) / household_size
    
    heating_factors = {'natural gas': 1.2, 'oil': 1.8, 'electricity': 0.8, 'none': 0.0}
    home = heating_factors.get(input_data.get('heating_source', 'none'), 0.0) / household_size
    home += elec # Combine heating and electricity into Home/Electricity
    
    digital = (input_data.get('internet_usage_hours', 0) * 0.03)
    
    return {
        "Transport": round(transport, 2),
        "Electricity": round(home, 2),
        "Diet": round(diet, 2),
        "Goods": round(goods, 2),
        "Waste": round(waste, 2),
        "Digital": round(digital, 2)
    }

def generate_suggestions(breakdown):
    """
    Generates smart, ranked suggestions based on max contributors to carbon footprint.
    """
    suggestions_db = {
        "Transport": [
            {"action": "Switch to public transport or EV", "savings_ratio": 0.4, "difficulty": "Hard"},
            {"action": "Carpool to work twice a week", "savings_ratio": 0.15, "difficulty": "Medium"},
            {"action": "Reduce long-haul flights by 1 per year", "savings_ratio": 0.3, "difficulty": "Medium"}
        ],
        "Electricity": [
            {"action": "Install solar panels or switch to green energy", "savings_ratio": 0.5, "difficulty": "Hard"},
            {"action": "Switch all bulbs to LEDs", "savings_ratio": 0.05, "difficulty": "Easy"},
            {"action": "Use smart thermostats, lower by 2°C in winter", "savings_ratio": 0.1, "difficulty": "Medium"}
        ],
        "Diet": [
            {"action": "Transition to a vegetarian/vegan diet", "savings_ratio": 0.4, "difficulty": "Medium"},
            {"action": "Have 3 meat-free days per week", "savings_ratio": 0.15, "difficulty": "Easy"},
            {"action": "Buy locally sourced, seasonal produce", "savings_ratio": 0.05, "difficulty": "Easy"}
        ],
        "Goods": [
            {"action": "Buy second-hand clothes & electronics", "savings_ratio": 0.2, "difficulty": "Easy"},
            {"action": "Reduce fast-fashion purchases", "savings_ratio": 0.15, "difficulty": "Easy"}
        ],
        "Waste": [
            {"action": "Compost organic waste", "savings_ratio": 0.3, "difficulty": "Medium"},
            {"action": "Recycle paper, plastic, and glass rigorously", "savings_ratio": 0.2, "difficulty": "Easy"}
        ],
        "Digital": [
            {"action": "Unsubscribe from junk emails and delete old files", "savings_ratio": 0.2, "difficulty": "Easy"},
            {"action": "Reduce video streaming quality from 4K to HD", "savings_ratio": 0.4, "difficulty": "Easy"}
        ]
    }
    
    generated = []
    
    # Sort categories by emission amount
    sorted_breakdown = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
    
    # Target top 3 contributors
    for i in range(min(3, len(sorted_breakdown))):
        category, amount = sorted_breakdown[i]
        
        # We don't suggest if the emission is already very low
        if amount < 0.2:
            continue
            
        for db_sug in suggestions_db.get(category, []):
            co2_saved = round(amount * db_sug["savings_ratio"], 2)
            if co2_saved > 0.05: # Only significant savings
                generated.append({
                    "action": db_sug["action"],
                    "co2_saved_tyr": co2_saved,
                    "category": category,
                    "difficulty": db_sug["difficulty"]
                })
                
    # Sort generated ascendingly by co2 saved
    generated = sorted(generated, key=lambda x: x["co2_saved_tyr"], reverse=True)
    
    # Add rank
    top_5 = generated[:5]
    for idx, sug in enumerate(top_5):
        sug["rank"] = idx + 1
        
    return top_5

def calculate_percentile_and_grade(total_footprint):
    """
    Calculates a proxy percentile and A-F letter grade.
    Assumes standard deviation for global is around 2.0.
    """
    z_score = (total_footprint - INDIA_AVG) / 1.5 
    # Just a simple heuristic percentile
    percentile = int(round(100 - (min(max(z_score, 0), 3) / 3 * 100)))
    percentile = max(min(percentile, 99), 1)
    
    if total_footprint < INDIA_AVG:
        grade = "A"
    elif total_footprint < INDIA_AVG * 1.5:
        grade = "B"
    elif total_footprint < WORLD_AVG:
        grade = "C"
    elif total_footprint < WORLD_AVG * 1.5:
        grade = "D"
    else:
        grade = "F"
        
    return percentile, grade
