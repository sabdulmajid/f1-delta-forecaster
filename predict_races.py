#!/usr/bin/env python3
"""
F1 Race Prediction Script - predicts pace deltas for upcoming races.
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

def log_message(message, level="INFO"):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    colors = {
        "INFO": "\033[94m",     # Blue
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",    # Red
        "SUCCESS": "\033[92m"   # Green
    }
    reset_color = "\033[0m"
    
    color = colors.get(level, "")
    print(f"{color}[{timestamp}] {level}: {message}{reset_color}", flush=True)

def get_upcoming_races():
    try:
        import fastf1
        
        current_year = 2025
        schedule = fastf1.get_event_schedule(current_year)
        
        races = schedule[schedule['EventFormat'] != 'testing'].copy()
        races['EventDate'] = pd.to_datetime(races['EventDate'])
        
        today = pd.Timestamp.now()
        upcoming = races[races['EventDate'] > today].head(5)
        
        return upcoming[['EventName', 'Location', 'EventDate']].to_dict('records')
        
    except Exception as e:
        log_message(f"Could not fetch upcoming races: {e}", "WARNING")
        return [
            {"EventName": "Next Grand Prix", "Location": "TBD", "EventDate": "2025-08-01"}
        ]

def load_trained_model():
    try:
        from training.lightning_module import F1LightningModule
        
        checkpoint_dirs = [
            "models/checkpoints/lightning_logs/version_0/checkpoints",
            "models/checkpoints"
        ]
        
        checkpoint_path = None
        for dir_path in checkpoint_dirs:
            if Path(dir_path).exists():
                ckpt_files = list(Path(dir_path).glob("*.ckpt"))
                if ckpt_files:
                    best_ckpts = [f for f in ckpt_files if 'last' not in f.name]
                    checkpoint_path = str(best_ckpts[0] if best_ckpts else ckpt_files[0])
                    break
        
        if not checkpoint_path:
            raise ValueError("No trained model checkpoint found")
        
        log_message(f"Loading model from: {checkpoint_path}")
        model = F1LightningModule.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        return model
        
    except Exception as e:
        log_message(f"Failed to load model: {e}", "ERROR")
        raise

def create_race_scenarios():
    scenarios = []
    
    compounds = ["SOFT", "MEDIUM", "HARD"]
    tyre_ages = [5, 15, 25, 35]
    
    weather_conditions = [
        {"name": "Cool", "air_temp": 20.0},
        {"name": "Moderate", "air_temp": 25.0},
        {"name": "Hot", "air_temp": 35.0}
    ]
    
    driving_styles = [
        {"name": "Conservative", "aggression": 0.3},
        {"name": "Balanced", "aggression": 0.6},
        {"name": "Aggressive", "aggression": 0.9}
    ]
    
    for compound in compounds:
        for tyre_age in tyre_ages:
            for weather in weather_conditions:
                for style in driving_styles:
                    scenario = {
                        "compound": compound,
                        "tyre_age": tyre_age,
                        "weather": weather,
                        "driving_style": style,
                        "scenario_name": f"{compound}_{tyre_age}laps_{weather['name']}_{style['name']}"
                    }
                    scenarios.append(scenario)
    
    return scenarios

def create_synthetic_sequence(scenario, sequence_length=5):
    """Create a synthetic sequence based on scenario parameters."""
    sequences = []
    
    for i in range(sequence_length):
        # Base speed influenced by tyre age and driving style
        base_speed = 260 - (scenario["tyre_age"] * 0.5)  # Degradation
        aggression_factor = scenario["driving_style"]["aggression"]
        speed_mean = base_speed + (aggression_factor * 20)
        
        # Temperature effects
        temp_factor = (scenario["weather"]["air_temp"] - 25) / 25
        speed_mean *= (1 - temp_factor * 0.02)  # Hot weather = slower
        
        # Add some variation
        speed_variation = np.random.normal(0, 10)
        speed_mean += speed_variation
        
        features = [
            speed_mean,  # speed_mean
            speed_mean + np.random.uniform(15, 30),  # speed_max
            speed_mean - np.random.uniform(10, 20),  # speed_min
            np.random.uniform(8, 15),  # speed_std
            
            60 + aggression_factor * 25,  # throttle_mean
            100,  # throttle_max
            0.3 + aggression_factor * 0.3,  # throttle_time_full
            
            aggression_factor * 15,  # brake_mean
            aggression_factor * 0.4,  # brake_time_active
            np.random.uniform(0, 0.8),  # drs_active
            
            np.random.randint(0, 4),  # gear_changes
            i * 5,  # turns_since_start
            scenario["tyre_age"],  # tyre_life
            
            scenario["weather"]["air_temp"],  # air_temp
            scenario["weather"]["air_temp"] + 10,  # track_temp
            50.0,  # humidity
        ]
        
        # Add compound one-hot encoding
        compound_features = [0, 0, 0]  # [HARD, MEDIUM, SOFT]
        if scenario["compound"] == "HARD":
            compound_features[0] = 1
        elif scenario["compound"] == "MEDIUM":
            compound_features[1] = 1
        else:  # SOFT
            compound_features[2] = 1
        
        features.extend(compound_features)
        sequences.append(features)
    
    return np.array(sequences)

def predict_race_scenarios(model, scenarios):
    """Make predictions for all race scenarios."""
    import torch
    
    predictions = []
    
    log_message(f"Making predictions for {len(scenarios)} scenarios...")
    
    for scenario in scenarios:
        try:
            # Create synthetic sequence
            sequence = create_synthetic_sequence(scenario)
            
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(sequence_tensor).item()
            
            result = {
                "scenario": scenario["scenario_name"],
                "compound": scenario["compound"],
                "tyre_age": scenario["tyre_age"],
                "weather": scenario["weather"]["name"],
                "driving_style": scenario["driving_style"]["name"],
                "predicted_delta": prediction
            }
            predictions.append(result)
            
        except Exception as e:
            log_message(f"Prediction failed for scenario {scenario['scenario_name']}: {e}", "WARNING")
            continue
    
    return predictions

def analyze_predictions(predictions):
    """Analyze and summarize predictions."""
    df = pd.DataFrame(predictions)
    
    analysis = {
        "summary_stats": {
            "total_predictions": len(predictions),
            "mean_delta": df["predicted_delta"].mean(),
            "std_delta": df["predicted_delta"].std(),
            "min_delta": df["predicted_delta"].min(),
            "max_delta": df["predicted_delta"].max()
        },
        "compound_analysis": df.groupby("compound")["predicted_delta"].agg(['mean', 'std']).to_dict(),
        "tyre_age_analysis": df.groupby("tyre_age")["predicted_delta"].agg(['mean', 'std']).to_dict(),
        "weather_analysis": df.groupby("weather")["predicted_delta"].agg(['mean', 'std']).to_dict(),
        "driving_style_analysis": df.groupby("driving_style")["predicted_delta"].agg(['mean', 'std']).to_dict()
    }
    
    return analysis

def generate_prediction_report(upcoming_races, predictions, analysis):
    """Generate a comprehensive prediction report."""
    report = []
    
    report.append("# F1 Race Prediction Report")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Upcoming races
    report.append("## Upcoming Races")
    for race in upcoming_races:
        report.append(f"- **{race['EventName']}** at {race['Location']} on {race['EventDate']}")
    report.append("")
    
    # Summary statistics
    stats = analysis["summary_stats"]
    report.append("## Prediction Summary")
    report.append(f"- Total scenarios analyzed: {stats['total_predictions']}")
    report.append(f"- Average pace delta: {stats['mean_delta']:.3f} seconds")
    report.append(f"- Range: {stats['min_delta']:.3f} to {stats['max_delta']:.3f} seconds")
    report.append("")
    
    # Key insights
    report.append("## Key Insights")
    
    # Compound insights
    compound_means = {k: v['mean'] for k, v in analysis["compound_analysis"].items()}
    best_compound = min(compound_means.keys(), key=lambda x: compound_means[x])
    worst_compound = max(compound_means.keys(), key=lambda x: compound_means[x])
    
    report.append(f"- **Best compound**: {best_compound} (avg delta: {compound_means[best_compound]:.3f}s)")
    report.append(f"- **Worst compound**: {worst_compound} (avg delta: {compound_means[worst_compound]:.3f}s)")
    
    # Weather insights  
    weather_means = {k: v['mean'] for k, v in analysis["weather_analysis"].items()}
    best_weather = min(weather_means.keys(), key=lambda x: weather_means[x])
    
    report.append(f"- **Optimal weather**: {best_weather} conditions")
    report.append("")
    
    # Strategy recommendations
    report.append("## Strategy Recommendations")
    
    # Find best overall scenarios
    df = pd.DataFrame(predictions)
    best_scenarios = df.nsmallest(5, "predicted_delta")
    
    report.append("**Top 5 predicted strategies:**")
    for i, (_, scenario) in enumerate(best_scenarios.iterrows(), 1):
        report.append(f"{i}. {scenario['compound']} tyres, {scenario['tyre_age']} laps, "
                     f"{scenario['weather']} weather, {scenario['driving_style']} style "
                     f"(Δ: {scenario['predicted_delta']:.3f}s)")
    
    return "\n".join(report)

def main():
    log_message("Starting F1 Race Predictions")
    log_message("=" * 50)
    
    try:
        # Check if training is complete
        if not Path("TRAINING_COMPLETE.txt").exists():
            log_message("Model training not complete. Run train_full.py first.", "ERROR")
            return False
        
        # Get upcoming races
        upcoming_races = get_upcoming_races()
        log_message(f"Found {len(upcoming_races)} upcoming races")
        
        # Load trained model
        model = load_trained_model()
        log_message("Model loaded successfully")
        
        # Create race scenarios
        scenarios = create_race_scenarios()
        log_message(f"Created {len(scenarios)} prediction scenarios")
        
        # Make predictions
        predictions = predict_race_scenarios(model, scenarios)
        log_message(f"Generated {len(predictions)} predictions")
        
        # Analyze results
        analysis = analyze_predictions(predictions)
        
        # Generate report
        report = generate_prediction_report(upcoming_races, predictions, analysis)
        
        # Save outputs
        os.makedirs("predictions", exist_ok=True)
        
        # Save detailed predictions
        pd.DataFrame(predictions).to_csv("predictions/race_predictions.csv", index=False)
        
        # Save analysis
        with open("predictions/analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Save report
        with open("predictions/PREDICTION_REPORT.md", "w") as f:
            f.write(report)
        
        log_message("Predictions saved to predictions/ directory")
        log_message("=" * 50)
        log_message("✅ Race predictions completed successfully")
        
        # Print summary
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(report)
        
        return True
        
    except Exception as e:
        log_message(f"Prediction failed: {e}", "ERROR")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
