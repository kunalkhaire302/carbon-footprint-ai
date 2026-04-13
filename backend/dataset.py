import os
import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples=5000, output_path='data/carbon_data.csv'):
    """
    Generates synthetic dataset for Individual Carbon Footprint Calculation.
    Features are mapped based on realistic distributions to compute the target
    variable (total_footprint_tco2e) roughly matching real-world logic.
    """
    np.random.seed(42)
    
    print("Generating synthetic dataset...")
    
    # Generate feature distributions
    electricity_usage_kwh = np.random.normal(250, 80, num_samples).clip(min=50) # skewed towards 200-300 kWh/month
    vehicle_types = ['petrol', 'diesel', 'electric', 'hybrid', 'none']
    vehicle_type = np.random.choice(vehicle_types, num_samples, p=[0.45, 0.25, 0.1, 0.1, 0.1])
    
    # KM driven depends heavily on vehicle type (none = 0)
    vehicle_km = np.where(vehicle_type != 'none', np.random.normal(800, 400, num_samples).clip(min=0), 0)
    
    flights_short_haul = np.random.poisson(1.5, num_samples)
    flights_long_haul = np.random.poisson(0.5, num_samples)
    
    diet_types = ['vegan', 'vegetarian', 'pescatarian', 'non-vegetarian']
    diet_type = np.random.choice(diet_types, num_samples, p=[0.05, 0.2, 0.05, 0.7])
    
    waste_kg_weekly = np.random.normal(15, 5, num_samples).clip(min=2)
    household_size = np.random.randint(1, 6, num_samples)
    grocery_spend_monthly = np.random.normal(300, 150, num_samples).clip(min=50)
    
    heating_sources = ['natural gas', 'electricity', 'oil', 'none']
    heating_source = np.random.choice(heating_sources, num_samples, p=[0.5, 0.3, 0.1, 0.1])
    
    internet_usage_hours = np.random.normal(6, 3, num_samples).clip(min=0.5, max=16)

    # Dictionary representing features
    data = {
        'electricity_usage_kwh': electricity_usage_kwh.round(1),
        'vehicle_type': vehicle_type,
        'vehicle_km': vehicle_km.round(1),
        'flights_short_haul': flights_short_haul,
        'flights_long_haul': flights_long_haul,
        'diet_type': diet_type,
        'waste_kg_weekly': waste_kg_weekly.round(1),
        'household_size': household_size,
        'grocery_spend_monthly': grocery_spend_monthly.round(2),
        'heating_source': heating_source,
        'internet_usage_hours': internet_usage_hours.round(1)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate target variable intuitively (tCO2e/year)
    # 1. Electricity (~0.4 kgCO2/kWh average)
    elec_emissions = (df['electricity_usage_kwh'] * 12 * 0.4) / 1000.0 / df['household_size']
    
    # 2. Vehicle (~0.2 kgCO2/km for petrol/diesel, 0.05 for EV, 0.12 for hybrid)
    emission_factors = {'petrol': 0.2, 'diesel': 0.23, 'hybrid': 0.12, 'electric': 0.05, 'none': 0.0}
    vehicle_emissions = (df['vehicle_type'].map(emission_factors) * df['vehicle_km'] * 12) / 1000.0
    
    # 3. Flights (short: ~0.15 tCO2e, long: ~0.8 tCO2e)
    flight_emissions = (df['flights_short_haul'] * 0.15) + (df['flights_long_haul'] * 0.8)
    
    # 4. Diet (base emissions tCO2e/yr: vegan 1.5, veg 1.7, pesc 1.9, non-veg 2.5)
    diet_factors = {'vegan': 1.0, 'vegetarian': 1.3, 'pescatarian': 1.6, 'non-vegetarian': 2.3}
    diet_emissions = df['diet_type'].map(diet_factors)
    
    # 5. Waste (~52 weeks * kg * 0.05 kgCO2e/kg waste)
    waste_emissions = (df['waste_kg_weekly'] * 52 * 0.05) / 1000.0
    
    # 6. Goods & services (approx 0.001 tCO2e per $)
    goods_emissions = (df['grocery_spend_monthly'] * 12 * 0.001) / df['household_size']
    
    # 7. Heating (assuming standard average use in winter, split by house size)
    heating_emissions = df['heating_source'].map({'natural gas': 1.2, 'oil': 1.8, 'electricity': 0.8, 'none': 0.0}) / df['household_size']
    
    # 8. Digital carbon proxy (data centers + device usage: ~0.03 tCO2e per daily hour over year)
    digital_emissions = (df['internet_usage_hours'] * 0.03)
    
    # Introduce some noise to make it realistic for the model
    noise = np.random.normal(0, 0.15, num_samples)
    
    df['total_footprint_tco2e'] = (elec_emissions + vehicle_emissions + flight_emissions + 
                                   diet_emissions + waste_emissions + goods_emissions + 
                                   heating_emissions + digital_emissions + noise).round(2).clip(lower=0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated and saved to {output_path} ({num_samples} samples)")
    
    # Performing basic Exploratory Data Analysis (EDA)
    print("\\n--- EDA Summary ---")
    print(f"Data Shape: {df.shape}")
    print("\\nMissing Values:")
    print(df.isnull().sum())
    print("\\nFeature Summary:")
    print(df.describe())
    
    return df

if __name__ == '__main__':
    generate_synthetic_data()
