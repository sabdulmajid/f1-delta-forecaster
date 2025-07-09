import fastf1
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import os
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings('ignore')

class F1DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        cache_dir = self.raw_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        try:
            fastf1.Cache.enable_cache(str(cache_dir))
            print(f"✓ FastF1 cache enabled at: {cache_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Could not enable FastF1 cache: {e}")
        
        self.processed_sessions = 0
        self.failed_sessions = 0
        self.total_laps_processed = 0
    
    def load_race_data(self, year: int, race: str) -> pd.DataFrame:
        try:
            session = fastf1.get_session(year, race, 'R')
            session.load()
            return session
        except Exception as e:
            print(f"Error loading {year} {race}: {e}")
            return None
    
    def extract_micro_sector_features(self, session) -> pd.DataFrame:
        features_list = []
        
        for driver_number in session.drivers:
            try:
                driver_laps = session.laps.pick_driver(driver_number)
                if len(driver_laps) == 0:
                    continue
                    
                for lap_idx, lap in driver_laps.iterrows():
                    telemetry = lap.get_car_data().add_distance()
                    
                    if len(telemetry) == 0:
                        continue
                    
                    # Create 100m micro-sectors
                    distance_bins = np.arange(0, telemetry['Distance'].max(), 100)
                    
                    for i, dist_start in enumerate(distance_bins[:-1]):
                        dist_end = distance_bins[i + 1]
                        sector_data = telemetry[
                            (telemetry['Distance'] >= dist_start) & 
                            (telemetry['Distance'] < dist_end)
                        ]
                        
                        if len(sector_data) == 0:
                            continue
                        features = {
                            'year': session.event.year,
                            'race': session.event['EventName'],
                            'driver': driver_number,
                            'lap_number': lap['LapNumber'],
                            'micro_sector': i,
                            'distance_start': dist_start,
                            'distance_end': dist_end,
                            
                            # Speed features
                            'speed_mean': sector_data['Speed'].mean(),
                            'speed_max': sector_data['Speed'].max(),
                            'speed_min': sector_data['Speed'].min(),
                            'speed_std': sector_data['Speed'].std(),
                            
                            # Throttle features
                            'throttle_mean': sector_data['Throttle'].mean(),
                            'throttle_max': sector_data['Throttle'].max(),
                            'throttle_time_full': (sector_data['Throttle'] == 100).sum() / len(sector_data),
                            
                            # Brake features
                            'brake_mean': sector_data['Brake'].mean() if 'Brake' in sector_data.columns else 0,
                            'brake_time_active': (sector_data['Brake'] > 0).sum() / len(sector_data) if 'Brake' in sector_data.columns else 0,
                            
                            # DRS
                            'drs_active': sector_data['DRS'].mean() if 'DRS' in sector_data.columns else 0,
                            
                            # Gear changes
                            'gear_changes': (sector_data['nGear'].diff().abs() > 0).sum() if 'nGear' in sector_data.columns else 0,
                            
                            # Track position
                            'turns_since_start': i,  # Proxy for track position
                            
                            # Lap information
                            'lap_time': lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else np.nan,
                            'sector1_time': lap['Sector1Time'].total_seconds() if pd.notna(lap['Sector1Time']) else np.nan,
                            'sector2_time': lap['Sector2Time'].total_seconds() if pd.notna(lap['Sector2Time']) else np.nan,
                            'sector3_time': lap['Sector3Time'].total_seconds() if pd.notna(lap['Sector3Time']) else np.nan,
                            
                            # Tyre information
                            'compound': lap['Compound'],
                            'tyre_life': lap['TyreLife'],
                            'fresh_tyre': lap['FreshTyre'],
                            
                            # Position and gaps
                            'position': lap['Position'],
                            'grid_position': lap.get('GridPosition', 20),
                            
                            # Track status
                            'track_status': 1,  # Default to green flag
                            
                            # Weather 
                            'air_temp': 25.0,  # Default air temp
                            'track_temp': 35.0,  # Default track temp  
                            'humidity': 50.0,  # Default humidity
                            'rainfall': False,
                        }
                        
                        features_list.append(features)
                        
            except Exception as e:
                print(f"Error processing driver {driver_number}: {e}")
                continue
        
        return pd.DataFrame(features_list)
    
    def calculate_pace_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pace delta (target variable) for each lap."""
        df = df.copy()
        df['next_lap_time'] = df.groupby(['driver'])['lap_time'].shift(-1)
        df['pace_delta'] = df['next_lap_time'] - df['lap_time']
        
        # Remove outliers (laps > 2 minutes or < 1 minute)
        df = df[(df['lap_time'] > 60) & (df['lap_time'] < 120)]
        df = df[(df['next_lap_time'] > 60) & (df['next_lap_time'] < 120)]
        
        return df
    
    def create_sequence_dataset(self, df: pd.DataFrame, sequence_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        # Select feature columns
        feature_cols = [
            'speed_mean', 'speed_max', 'speed_min', 'speed_std',
            'throttle_mean', 'throttle_max', 'throttle_time_full',
            'brake_mean', 'brake_time_active', 'drs_active',
            'gear_changes', 'turns_since_start', 'tyre_life',
            'air_temp', 'track_temp', 'humidity'
        ]
        
        # Clean data first
        df = df.dropna(subset=['compound', 'pace_delta'])
        df = df[df['compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]  # Valid compounds only
        
        # Add categorical features (one-hot encoded)
        compound_dummies = pd.get_dummies(df['compound'], prefix='compound')
        df = pd.concat([df, compound_dummies], axis=1)
        feature_cols.extend(compound_dummies.columns.tolist())
        
        sequences = []
        targets = []
        
        # Group by driver to create sequences
        for driver in df['driver'].unique():
            driver_data = df[df['driver'] == driver].sort_values('lap_number')
            
            if len(driver_data) < sequence_length + 1:
                continue
            
            for i in range(len(driver_data) - sequence_length):
                # Get sequence of features
                sequence_data = driver_data.iloc[i:i + sequence_length][feature_cols]
                target = driver_data.iloc[i + sequence_length]['pace_delta']
                
                # Check for valid data
                if pd.isna(target) or sequence_data.isna().any().any():
                    continue
                    
                sequence = sequence_data.values
                sequences.append(sequence)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def process_race_season(self, year: int, races: List[str] = None) -> Dict:
        """Process an entire season of race data."""
        if races is None:
            # Get all races for the year
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'] != 'testing']['EventName'].tolist()
        
        all_features = []
        
        print(f"Processing {year} season...")
        for race in races:
            try:
                session = self.load_race_data(year, race)
                if session is None:
                    continue
                
                features = self.extract_micro_sector_features(session)
                if len(features) > 0:
                    all_features.append(features)
                    
            except Exception as e:
                print(f"Error processing {race}: {e}")
                continue
        
        if not all_features:
            raise ValueError(f"No data extracted for {year}")
        
        # Combine all race data
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Calculate pace deltas
        combined_df = self.calculate_pace_delta(combined_df)
        
        # Create sequences
        X, y = self.create_sequence_dataset(combined_df)
        
        # Save processed data
        output_path = self.processed_dir / f"f1_data_{year}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({
                'features': X,
                'targets': y,
                'raw_data': combined_df,
                'year': year,
                'races': races
            }, f)
        
        print(f"Saved processed data to {output_path}")
        print(f"Dataset shape: {X.shape}, Targets shape: {y.shape}")
        
        return {
            'features': X,
            'targets': y,
            'raw_data': combined_df,
            'year': year,
            'races': races
        }

def main():
    parser = argparse.ArgumentParser(description='Process F1 data for tyre degradation forecasting')
    parser.add_argument('--year', type=int, default=2023, help='F1 season year')
    parser.add_argument('--races', nargs='+', default=None, help='Specific races to process')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    
    args = parser.parse_args()
    
    loader = F1DataLoader(args.data_dir)
    
    try:
        data = loader.process_race_season(args.year, args.races)
        print("Data processing completed successfully!")
        print(f"Features shape: {data['features'].shape}")
        print(f"Targets shape: {data['targets'].shape}")
        
    except Exception as e:
        print(f"Error during data processing: {e}")

if __name__ == "__main__":
    main()
