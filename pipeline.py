import os
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import logging
from datetime import datetime

from model import CNN_GRU_Attention


class TimeSeriesSegmentation:
    def __init__(self):
        base_dir = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parent))
        # Define the base directories
        self.data_dir = Path(os.environ.get("DATA_DIR", base_dir / "data/segmentation_data/fllanza_data"))
        self.labels_dir = Path(os.environ.get("LABELS_DIR", base_dir / "data/segmentation_labels/segmentation_labels"))
        
        # Define phase columns
        self.voltage_columns = ["U1", "U2", "U3"]
        self.current_columns = ["I1", "I2", "I3"]
        
        # Initialize data containers
        self.labels_J03 = None
        self.labels_J04 = None
        self.measurements_J03 = None
        self.measurements_J04 = None
        self.rms_df = None
        self.rms_df2 = None
        
    def load_labels(self, label_file):
        """Function to load labels into a separate dataframe"""
        labels_df = pd.read_csv(self.labels_dir / label_file)
        labels_df_cleaned = labels_df.dropna().copy()  # Ensure it's a copy
        labels_df_cleaned["sample_id"] = labels_df_cleaned["sample_id"].astype(int)
        return labels_df_cleaned  # Return the cleaned labels DataFrame

    def load_measurements(self, segment_folder):
        """Function to load measurement data into a separate dataframe"""
        combined_data = []

        # Loop through all sample folders
        for sample_folder in self.data_dir.iterdir():
            if sample_folder.is_dir():
                sample_id = sample_folder.name  # Extract sample ID from folder name
                measurement_file = sample_folder / segment_folder / "measurements.csv"
                
                if measurement_file.exists():
                    df = pd.read_csv(measurement_file)
                    df["Time"] = pd.to_numeric(df["Time"])  # Ensure 'Time' column is numeric
                    df["sample_id"] = int(sample_id)  # Store sample_id
                    combined_data.append(df)
                else:
                    print(f"Measurements file not found for Sample ID {sample_id} in {segment_folder}.")

        # Concatenate all data
        if combined_data:
            measurements_df = pd.concat(combined_data, ignore_index=True)
            measurements_df["sample_id"] = measurements_df["sample_id"].astype(int)
            return measurements_df
        else:
            return pd.DataFrame()

    def apply_robust_scaling(self, df, sample_id, columns):
        """Function to apply RobustScaler for a specific sample_id"""
        sample_data = df[df['sample_id'] == sample_id].copy()  # Filter data for the sample
        
        scaler = RobustScaler()  # Initialize RobustScaler
        df.loc[df['sample_id'] == sample_id, columns] = scaler.fit_transform(sample_data[columns])  # Apply scaling

    def normalize_using_first_two_cycles_amplitude(self, df, columns):
        """
        Normalize each sample using the amplitude (max - min) of the first two cycles (0.04s).
        """
        normalized_df = df.copy()

        for sample_id in df["sample_id"].unique():
            sample_df = df[df["sample_id"] == sample_id].copy()

            # Define the first two cycles as the first 0.04 seconds
            start_time = sample_df["Time"].min()
            time_threshold = start_time + 0.04
            first_two_cycles = sample_df[sample_df["Time"] <= time_threshold]

            if not first_two_cycles.empty:
                # Compute amplitude (max - min) for each column
                amplitude = first_two_cycles[columns].max() - first_two_cycles[columns].min()

                # Avoid division by zero (if amplitude is 0, set to 1)
                amplitude.replace(0, 1, inplace=True)

                # Normalize the entire sample
                normalized_df.loc[df["sample_id"] == sample_id, columns] = sample_df[columns] / amplitude

        return normalized_df

    def calculate_rms_by_segments(self, df, voltage_columns, current_columns, segment_size=40):
        """Function to calculate RMS for each segment (40 samples)"""
        rms_values = []

        # Iterate through unique sample IDs
        for sample_id in df["sample_id"].unique():
            sample_data = df[df["sample_id"] == sample_id].reset_index(drop=True)

            num_samples = len(sample_data)

            # Segment the data in chunks of 'segment_size'
            for start_idx in range(1, num_samples, segment_size):
                end_idx = min(start_idx + segment_size, num_samples)
                segment_df = sample_data.iloc[start_idx:end_idx]

                # Compute RMS for this segment
                rms_segment = {"sample_id": sample_id, "Segment": start_idx // segment_size + 1}

                for phase in voltage_columns:
                    rms_segment[f"RMS_{phase}"] = np.sqrt(np.mean(segment_df[phase] ** 2))

                for phase in current_columns:
                    rms_segment[f"RMS_{phase}"] = np.sqrt(np.mean(segment_df[phase] ** 2))

                # Store the end time of the segment
                rms_segment["Time"] = segment_df.iloc[-1]["Time"]

                rms_values.append(rms_segment)

        return pd.DataFrame(rms_values)

    def compute_derivative(self, rms_df, voltage_columns, current_columns):
        new_df = rms_df.copy()  # Create a copy to avoid chained assignment issues
        
        # Process each unique sample_id separately
        for sample_id in new_df["sample_id"].unique():
            # Get data for this sample_id
            sample_mask = new_df["sample_id"] == sample_id
            sample_data = new_df[sample_mask]
            
            # Get indices for this sample in the original dataframe
            sample_indices = sample_data.index
            
            for phase in voltage_columns + current_columns:
                rms_col = f"RMS_{phase}"
                dy_dx_col = f"d_{phase}/dt"   # First derivative
                d2y_dx2_col = f"d2_{phase}/dt2"  # Second derivative
                
                # Compute derivatives only for this sample
                dy_dx_values = np.gradient(sample_data[rms_col], sample_data["Time"])
                d2y_dx2_values = np.gradient(dy_dx_values, sample_data["Time"])
                
                # Assign computed values back to the original dataframe at the correct indices
                new_df.loc[sample_indices, dy_dx_col] = dy_dx_values
                new_df.loc[sample_indices, d2y_dx2_col] = d2y_dx2_values
        
        return new_df  # Return the updated DataFrame

    def normalize_derivatives(self, df, voltage_columns, current_columns, method='z-score'):
        new_df = df.copy()
        
        # Columns to normalize
        derivative_columns = []
        for phase in voltage_columns + current_columns:
            derivative_columns.extend([f"d_{phase}/dt", f"d2_{phase}/dt2"])
        
        if method == 'z-score':
            # Z-score normalization (standard scaling)
            for col in derivative_columns:
                new_df[f'norm_{col}'] = (new_df[col] - new_df[col].mean()) / new_df[col].std()
        
        elif method == 'min-max':
            # Min-Max scaling to [0, 1] range
            for col in derivative_columns:
                min_val = new_df[col].min()
                max_val = new_df[col].max()
                new_df[f'norm_{col}'] = (new_df[col] - min_val) / (max_val - min_val)
        
        elif method == 'robust':
            # Robust scaling using median and interquartile range
            for col in derivative_columns:
                median = new_df[col].median()
                iqr = new_df[col].quantile(0.75) - new_df[col].quantile(0.25)
                new_df[f'norm_{col}'] = (new_df[col] - median) / iqr
        
        else:
            raise ValueError("Invalid normalization method. Choose 'z-score', 'min-max', or 'robust'.")
        
        return new_df

    def create_dynamic_labels(self, rms_df, labels_df):
        """Function to create labels dynamically according to rms_df"""
        # Initialize empty lists to store start_t and end_t for each row
        start_times = []
        end_times = []

        # Iterate through each sample_id in the rms_df
        for sample_id in rms_df["sample_id"].unique():
            # Get all labels for the current sample_id
            sample_labels = labels_df[labels_df["sample_id"] == sample_id]

            # Get the time intervals for the sample_id
            sample_times = sample_labels[['start_t', 'end_t']].values

            # Sort the time intervals in case they are not in order
            sample_times = sorted(sample_times, key=lambda x: x[0])

            # Get the rows in rms_df corresponding to this sample_id
            sample_rms = rms_df[rms_df["sample_id"] == sample_id].reset_index(drop=True)

            # Iterate through the rows of sample_rms and assign the correct start_t and end_t
            label_idx = 0
            for idx, row in sample_rms.iterrows():
                # Assign the current start_t and end_t to this row
                start_times.append(sample_times[label_idx][0])
                end_times.append(sample_times[label_idx][1])

                # Check if we need to switch to the next label (after the end_t time is passed)
                if row["Time"] > sample_times[label_idx][1]:
                    # Switch to the next label if available
                    if label_idx + 1 < len(sample_times):
                        label_idx += 1

        # Add the start and end times as new columns to the rms_df
        rms_df["start_t"] = start_times
        rms_df["end_t"] = end_times

        return rms_df

    def create_sequences_with_indices(self, X, y, time, time_steps=50):
        """Step 1: Create Sequences and Track Indices"""
        X_seq, y_seq, time_seq, indices = [], [], [], []
        for i in range(0,len(X) - time_steps,time_steps):
            X_seq.append(X[i:i + time_steps])
            y_seq.append(y[i + 1])
            time_seq.append(time[i + time_steps])
            indices.append(i)  # Store the original index
        return np.array(X_seq), np.array(y_seq), np.array(time_seq), np.array(indices)

    def create_sequences_with_indices_alt(self, X, y, time, time_steps=50):
        X_seq, y_seq, time_seq = [], [], []
        for i in range(0, len(X) , time_steps):
            X_seq.append(X[i:i + time_steps])
            y_seq.append(y[i + 1])
            time_seq.append(time[i:i + time_steps])
        return np.array(X_seq), np.array(y_seq), np.array(time_seq)

    def create_windows_for_test(self, X, y, time, time_steps=50):
        X_seq, y_seq, time_seq = [], [], []
        for i in range(0, len(X), time_steps):
            X_seq.append(X[i:i + time_steps])
            y_seq.append(y[i +  1])
            time_seq.append(time[i:i + time_steps])
        return np.array(X_seq), np.array(y_seq), np.array(time_seq)

    def prepare_data(self):
        """Main data preparation pipeline"""
        # Load J03 and J04 labels
        self.labels_J03 = self.load_labels("J03_labels.csv")
        self.labels_J04 = self.load_labels("J04_labels.csv")

        # Load J03 and J04 measurement data
        self.measurements_J03 = self.load_measurements("J03")
        self.measurements_J04 = self.load_measurements("J04")

        # Display both separate dataframes
        print("Labels DataFrame:")
        print(self.labels_J03.head())

        print("\nMeasurements DataFrame:")
        print(self.measurements_J03.head())

        # Sort by 'sample_id' first, then by 'Time' within each sample_id
        self.measurements_J03 = self.measurements_J03.sort_values(by=["sample_id", "Time"]).reset_index(drop=True)
        # Sort by 'sample_id' first, then by 'Time' within each sample_id
        self.measurements_J04 = self.measurements_J04.sort_values(by=["sample_id", "Time"]).reset_index(drop=True)

        # Apply RobustScaler to measurements_J03
        sample_ids_J03 = self.measurements_J03['sample_id'].unique()

        for sample_id in sample_ids_J03:
            self.apply_robust_scaling(self.measurements_J03, sample_id, self.voltage_columns)

        # Apply RobustScaler to measurements_J04
        sample_ids_J04 = self.measurements_J04['sample_id'].unique()

        for sample_id in sample_ids_J04:
            self.apply_robust_scaling(self.measurements_J04, sample_id, self.voltage_columns)

        # Apply normalization to current signals
        self.measurements_J03 = self.normalize_using_first_two_cycles_amplitude(self.measurements_J03, self.current_columns)
        self.measurements_J04 = self.normalize_using_first_two_cycles_amplitude(self.measurements_J04, self.current_columns)

        # Calculate RMS on the filtered signals
        self.rms_df = self.calculate_rms_by_segments(self.measurements_J03, self.voltage_columns, self.current_columns)
        self.rms_df2 = self.calculate_rms_by_segments(self.measurements_J04, self.voltage_columns, self.current_columns)

        # Sort and reset index
        self.rms_df = self.rms_df.sort_values(by=["sample_id", "Time"]).reset_index(drop=True)
        self.rms_df2 = self.rms_df2.sort_values(by=["sample_id", "Time"]).reset_index(drop=True)

        # Apply derivative computation to both datasets
        self.rms_df = self.compute_derivative(self.rms_df, self.voltage_columns, self.current_columns)
        self.rms_df2 = self.compute_derivative(self.rms_df2, self.voltage_columns, self.current_columns)

        # Normalize derivatives
        self.rms_df = self.normalize_derivatives(self.rms_df, self.voltage_columns, self.current_columns, method='min-max')
        self.rms_df2 = self.normalize_derivatives(self.rms_df2, self.voltage_columns, self.current_columns, method='min-max')

        # Create the dynamic labels for rms_df based on the labels in labels_J03
        rms_df_with_labels = self.create_dynamic_labels(self.rms_df, self.labels_J03)
        rms_df_with_labels2 = self.create_dynamic_labels(self.rms_df2, self.labels_J04)

        # Count the number of rows per sample_id
        sample_sizes = rms_df_with_labels.groupby('sample_id').size()

        # Filter out sample_ids with more than 100 rows
        samples_to_keep = sample_sizes[sample_sizes <= 100].index

        # Filter the original DataFrame to keep only these samples
        rms_df_filtered = rms_df_with_labels[rms_df_with_labels['sample_id'].isin(samples_to_keep)]



        # Count the number of rows per sample_id
        sample_sizes = rms_df_with_labels2.groupby('sample_id').size()

        # Filter out sample_ids with more than 100 rows
        samples_to_keep = sample_sizes[sample_sizes <= 100].index

        # Filter the original DataFrame to keep only these samples
        rms_df_filtered2 = rms_df_with_labels2[rms_df_with_labels2['sample_id'].isin(samples_to_keep)]

        # Display the first few rows with the new labels
        print(rms_df_with_labels.head())

        return rms_df_filtered, rms_df_filtered2

    def prepare_training_data(self, rms_df_filtered, rms_df_filtered2):
        """Prepare training data for all phases"""
        # Define features for each phase
        phase1_voltage_features = ['RMS_U1', 'd_U1/dt']
        phase2_voltage_features = ['RMS_U2', 'd_U2/dt']
        phase3_voltage_features = ['RMS_U3', 'd_U3/dt']

        phase1_current_features = ['RMS_I1', 'd_I1/dt']
        phase2_current_features = ['RMS_I2', 'd_I2/dt']
        phase3_current_features = ['RMS_I3', 'd_I3/dt']

        labels = ['start_t', 'end_t']

        # Extract each phase's features separately
        # Phase 1
        X1_voltage_phase1 = rms_df_filtered[phase1_voltage_features].values
        X1_current_phase1 = rms_df_filtered[phase1_current_features].values
        X2_voltage_phase1 = rms_df_filtered2[phase1_voltage_features].values
        X2_current_phase1 = rms_df_filtered2[phase1_current_features].values

        # Phase 2
        X1_voltage_phase2 = rms_df_filtered[phase2_voltage_features].values
        X1_current_phase2 = rms_df_filtered[phase2_current_features].values
        X2_voltage_phase2 = rms_df_filtered2[phase2_voltage_features].values
        X2_current_phase2 = rms_df_filtered2[phase2_current_features].values

        # Phase 3
        X1_voltage_phase3 = rms_df_filtered[phase3_voltage_features].values
        X1_current_phase3 = rms_df_filtered[phase3_current_features].values
        X2_voltage_phase3 = rms_df_filtered2[phase3_voltage_features].values
        X2_current_phase3 = rms_df_filtered2[phase3_current_features].values

        # Labels remain the same
        y1 = rms_df_filtered[labels].values
        y2 = rms_df_filtered2[labels].values
        time_values1 = rms_df_filtered['Time'].values
        time_values2 = rms_df_filtered2['Time'].values

        # For demonstration, I'll continue with Phase 1 data splitting
        # You can repeat this process for phases 2 and 3
        test_size = 2500

        # Phase 1 Data Preparation
        # Split Phase 1 Voltage Data
        X_train1_v_phase1, X_test1_v_phase1 = X1_voltage_phase1[:-test_size], X1_voltage_phase1[-test_size:]
        X_train2_v_phase1, X_test2_v_phase1 = X2_voltage_phase1[:-test_size], X2_voltage_phase1[-test_size:]

        # Split Phase 1 Current Data
        X_train1_c_phase1, X_test1_c_phase1 = X1_current_phase1[:-test_size], X1_current_phase1[-test_size:]
        X_train2_c_phase1, X_test2_c_phase1 = X2_current_phase1[:-test_size], X2_current_phase1[-test_size:]

        # Labels and Time Values
        y_train1, y_test1 = y1[:-test_size], y1[-test_size:]
        y_train2, y_test2 = y2[:-test_size], y2[-test_size:]
        time_train1, time_test1 = time_values1[:-test_size], time_values1[-test_size:]
        time_train2, time_test2 = time_values2[:-test_size], time_values2[-test_size:]

        # Concatenate Phase 1 Training Data
        X_train_v_phase1 = np.concatenate([X_train1_v_phase1, X_train2_v_phase1], axis=0)
        X_train_c_phase1 = np.concatenate([X_train1_c_phase1, X_train2_c_phase1], axis=0)
        y_train = np.concatenate([y_train1, y_train2], axis=0)
        time_train = np.concatenate([time_train1, time_train2], axis=0)

        # Concatenate Phase 1 Testing Data
        X_test_v_phase1 = np.concatenate([X_test1_v_phase1, X_test2_v_phase1], axis=0)
        X_test_c_phase1 = np.concatenate([X_test1_c_phase1, X_test2_c_phase1], axis=0)
        y_test = np.concatenate([y_test1, y_test2], axis=0)
        time_test = np.concatenate([time_test1, time_test2], axis=0)

        # Split into Train and Validation for Phase 1
        X_train_v_phase1_final, X_val_v_phase1, y_train_final, y_val, time_train_final, time_val = train_test_split(
            X_train_v_phase1, y_train, time_train, test_size=2500, random_state=42, shuffle=False
        )
        X_train_c_phase1_final, X_val_c_phase1, _, _, _, _ = train_test_split(
            X_train_c_phase1, y_train, time_train, test_size=2500, random_state=42, shuffle=False
        )

        # Apply Windowing to Phase 1 Voltage Data
        X_train_v_phase1_seq, y_train_seq, time_train_seq, _ = self.create_sequences_with_indices(
            X_train_v_phase1_final, y_train_final, time_train_final, time_steps=100
        )
        X_val_v_phase1_seq, y_val_seq, time_val_seq, _ = self.create_sequences_with_indices(
            X_val_v_phase1, y_val, time_val, time_steps=100
        )

        # Apply Windowing to Phase 1 Current Data
        X_train_c_phase1_seq, _, _, _ = self.create_sequences_with_indices(
            X_train_c_phase1_final, y_train_final, time_train_final, time_steps=100
        )
        X_val_c_phase1_seq, _, _, _ = self.create_sequences_with_indices(
            X_val_c_phase1, y_val, time_val, time_steps=100
        )

        # Phase 2 Data Preparation
        # Split Phase 2 Voltage Data
        X_train1_v_phase2, X_test1_v_phase2 = X1_voltage_phase2[:-test_size], X1_voltage_phase2[-test_size:]
        X_train2_v_phase2, X_test2_v_phase2 = X2_voltage_phase2[:-test_size], X2_voltage_phase2[-test_size:]

        # Split Phase 2 Current Data
        X_train1_c_phase2, X_test1_c_phase2 = X1_current_phase2[:-test_size], X1_current_phase2[-test_size:]
        X_train2_c_phase2, X_test2_c_phase2 = X2_current_phase2[:-test_size], X2_current_phase2[-test_size:]

        # Concatenate Phase 2 Training Data
        X_train_v_phase2 = np.concatenate([X_train1_v_phase2, X_train2_v_phase2], axis=0)
        X_train_c_phase2 = np.concatenate([X_train1_c_phase2, X_train2_c_phase2], axis=0)

        # Concatenate Phase 2 Testing Data
        X_test_v_phase2 = np.concatenate([X_test1_v_phase2, X_test2_v_phase2], axis=0)
        X_test_c_phase2 = np.concatenate([X_test1_c_phase2, X_test2_c_phase2], axis=0)

        # Split into Train and Validation for Phase 2
        X_train_v_phase2_final, X_val_v_phase2, y_train_final, y_val, time_train_final, time_val = train_test_split(
            X_train_v_phase2, y_train, time_train, test_size=2500, random_state=42, shuffle=False
        )
        X_train_c_phase2_final, X_val_c_phase2, _, _, _, _ = train_test_split(
            X_train_c_phase2, y_train, time_train, test_size=2500, random_state=42, shuffle=False
        )

        # Apply Windowing to Phase 2 Voltage Data
        X_train_v_phase2_seq, y_train_seq, time_train_seq, _ = self.create_sequences_with_indices(
            X_train_v_phase2_final, y_train_final, time_train_final, time_steps=100
        )
        X_val_v_phase2_seq, y_val_seq, time_val_seq, _ = self.create_sequences_with_indices(
            X_val_v_phase2, y_val, time_val, time_steps=100
        )

        # Apply Windowing to Phase 2 Current Data
        X_train_c_phase2_seq, _, _, _ = self.create_sequences_with_indices(
            X_train_c_phase2_final, y_train_final, time_train_final, time_steps=100
        )
        X_val_c_phase2_seq, _, _, _ = self.create_sequences_with_indices(
            X_val_c_phase2, y_val, time_val, time_steps=100
        )

        # Phase 3 Data Preparation
        # Split Phase 3 Voltage Data
        X_train1_v_phase3, X_test1_v_phase3 = X1_voltage_phase3[:-test_size], X1_voltage_phase3[-test_size:]
        X_train2_v_phase3, X_test2_v_phase3 = X2_voltage_phase3[:-test_size], X2_voltage_phase3[-test_size:]

        # Split Phase 3 Current Data
        X_train1_c_phase3, X_test1_c_phase3 = X1_current_phase3[:-test_size], X1_current_phase3[-test_size:]
        X_train2_c_phase3, X_test2_c_phase3 = X2_current_phase3[:-test_size], X2_current_phase3[-test_size:]

        # Concatenate Phase 3 Training Data
        X_train_v_phase3 = np.concatenate([X_train1_v_phase3, X_train2_v_phase3], axis=0)
        X_train_c_phase3 = np.concatenate([X_train1_c_phase3, X_train2_c_phase3], axis=0)

        # Concatenate Phase 3 Testing Data
        X_test_v_phase3 = np.concatenate([X_test1_v_phase3, X_test2_v_phase3], axis=0)
        X_test_c_phase3 = np.concatenate([X_test1_c_phase3, X_test2_c_phase3], axis=0)

        # Split into Train and Validation for Phase 3
        X_train_v_phase3_final, X_val_v_phase3, y_train_final, y_val, time_train_final, time_val = train_test_split(
            X_train_v_phase3, y_train, time_train, test_size=2500, random_state=42, shuffle=False
        )
        X_train_c_phase3_final, X_val_c_phase3, _, _, _, _ = train_test_split(
            X_train_c_phase3, y_train, time_train, test_size=2500, random_state=42, shuffle=False
        )

        # Apply Windowing to Phase 3 Voltage Data
        X_train_v_phase3_seq, y_train_seq, time_train_seq, _ = self.create_sequences_with_indices(
            X_train_v_phase3_final, y_train_final, time_train_final, time_steps=100
        )
        X_val_v_phase3_seq, y_val_seq, time_val_seq, _ = self.create_sequences_with_indices(
            X_val_v_phase3, y_val, time_val, time_steps=100
        )

        # Apply Windowing to Phase 3 Current Data
        X_train_c_phase3_seq, _, _, _ = self.create_sequences_with_indices(
            X_train_c_phase3_final, y_train_final, time_train_final, time_steps=100
        )
        X_val_c_phase3_seq, _, _, _ = self.create_sequences_with_indices(
            X_val_c_phase3, y_val, time_val, time_steps=100
        )

        # Prepare test sequences
        # Process each phase separately for validation data
        # Phase 1
        X_val_v_phase1_seq, y_val_phase1_seq, time_val_phase1_seq = self.create_sequences_with_indices_alt(
            X_val_v_phase1, y_val, time_val, time_steps=100
        )
        X_val_c_phase1_seq, y_val_c_phase1_seq, time_val_c_phase1_seq = self.create_sequences_with_indices_alt(
            X_val_c_phase1, y_val, time_val, time_steps=100)
        # Phase 2
        X_val_v_phase2_seq, y_val_phase2_seq, time_val_phase2_seq = self.create_sequences_with_indices_alt(
            X_val_v_phase2, y_val, time_val, time_steps=100
        )
        X_val_c_phase2_seq, y_val_c_phase2_seq, time_val_c_phase2_seq = self.create_sequences_with_indices_alt(
            X_val_c_phase2, y_val, time_val, time_steps=100
        )

        # Phase 3
        X_val_v_phase3_seq, y_val_phase3_seq, time_val_phase3_seq = self.create_sequences_with_indices_alt(
            X_val_v_phase3, y_val, time_val, time_steps=100
        )
        X_val_c_phase3_seq, y_val_c_phase3_seq, time_val_c_phase3_seq = self.create_sequences_with_indices_alt(
            X_val_c_phase3, y_val, time_val, time_steps=100
        )

        # Process each phase separately for test data
        # Phase 1
        X_test_v_phase1_seq, y_test_phase1_seq, time_test_phase1_seq = self.create_windows_for_test(
            X_test_v_phase1, y_test, time_test, time_steps=100
        )
        X_test_c_phase1_seq, y_test_c_phase1_seq, time_test_c_phase1_seq = self.create_windows_for_test(
            X_test_c_phase1, y_test, time_test, time_steps=100
        )

        # Phase 2
        X_test_v_phase2_seq, y_test_phase2_seq, time_test_phase2_seq = self.create_windows_for_test(
            X_test_v_phase2, y_test, time_test, time_steps=100
        )
        X_test_c_phase2_seq, y_test_c_phase2_seq, time_test_c_phase2_seq = self.create_windows_for_test(
            X_test_c_phase2, y_test, time_test, time_steps=100
        )

        # Phase 3
        X_test_v_phase3_seq, y_test_phase3_seq, time_test_phase3_seq = self.create_windows_for_test(
            X_test_v_phase3, y_test, time_test, time_steps=100
        )
        X_test_c_phase3_seq, y_test_c_phase3_seq, time_test_c_phase3_seq = self.create_windows_for_test(
            X_test_c_phase3, y_test, time_test, time_steps=100
        )

        # Return all the prepared data
        return {
            # Training data for voltage phases
            'X_train_v_phase1_seq': X_train_v_phase1_seq,
            'X_train_v_phase2_seq': X_train_v_phase2_seq,
            'X_train_v_phase3_seq': X_train_v_phase3_seq,
            
            # Training data for current phases
            'X_train_c_phase1_seq': X_train_c_phase1_seq,
            'X_train_c_phase2_seq': X_train_c_phase2_seq,
            'X_train_c_phase3_seq': X_train_c_phase3_seq,
            
            # Validation data for voltage phases
            'X_val_v_phase1_seq': X_val_v_phase1_seq,
            'X_val_v_phase2_seq': X_val_v_phase2_seq,
            'X_val_v_phase3_seq': X_val_v_phase3_seq,
            
            # Validation data for current phases
            'X_val_c_phase1_seq': X_val_c_phase1_seq,
            'X_val_c_phase2_seq': X_val_c_phase2_seq,
            'X_val_c_phase3_seq': X_val_c_phase3_seq,
            
            # Test data for voltage phases
            'X_test_v_phase1_seq': X_test_v_phase1_seq,
            'X_test_v_phase2_seq': X_test_v_phase2_seq,
            'X_test_v_phase3_seq': X_test_v_phase3_seq,
            
            # Test data for current phases
            'X_test_c_phase1_seq': X_test_c_phase1_seq,
            'X_test_c_phase2_seq': X_test_c_phase2_seq,
            'X_test_c_phase3_seq': X_test_c_phase3_seq,
            
            # Labels
            'y_train_seq': y_train_seq,
            'y_val_phase1_seq': y_val_phase1_seq,
            'y_val_phase2_seq': y_val_phase2_seq,
            'y_val_phase3_seq': y_val_phase3_seq,
            'y_test_phase1_seq': y_test_phase1_seq,
            'y_test_phase2_seq': y_test_phase2_seq,
            'y_test_phase3_seq': y_test_phase3_seq,

            # Time values for evaluation and visualization
            'time_train_seq': time_train_seq,
            'time_val_phase1_seq': time_val_phase1_seq,
            'time_val_phase2_seq': time_val_phase2_seq,
            'time_val_phase3_seq': time_val_phase3_seq,
            'time_test_phase1_seq': time_test_phase1_seq,
            'time_test_phase2_seq': time_test_phase2_seq,
            'time_test_phase3_seq': time_test_phase3_seq,
        }

    # IoU Function (for evaluation)
    def calculate_iou(self,pred, target):
        pred_start, pred_end = pred[:, 0], pred[:, 1]
        true_start, true_end = target[:, 0], target[:, 1]
        
        # Calculate intersection and union
        intersection = torch.clamp(torch.min(pred_end, true_end) - torch.max(pred_start, true_start), min=0)
        union = (true_end - true_start) + (pred_end - pred_start) - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        return iou.mean().item()

    # IoU Loss Function
    def iou_loss(self, pred, target):
        pred_start, pred_end = pred[:, 0], pred[:, 1]
        true_start, true_end = target[:, 0], target[:, 1]
        intersection = torch.clamp(torch.min(pred_end, true_end) - torch.max(pred_start, true_start), min=0)
        union = (true_end - true_start) + (pred_end - pred_start) - intersection
        iou = intersection / (union + 1e-6)  
        return torch.mean(1 - iou)  

    # Combined Loss (L1 + IoU Loss)
    def combined_loss(self, pred, target, alpha=0.5):
        l1_loss = nn.SmoothL1Loss()(pred, target)
        iou_loss_value = self.iou_loss(pred, target)
        return alpha * l1_loss + (1 - alpha) * iou_loss_value

    # Training Function
    def train_model(self, model, train_loader, val_loader, num_epochs=50, patience=10, model_name="model", device="cuda"):
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
        best_val_loss = float("inf")
        best_iou = 0.0
        best_model_path = f"models/best_{model_name}.pth"
        no_improve_count = 0
        
        # Lists to track metrics
        train_losses = []
        val_losses = []
        val_ious = []
        
        for epoch in range(num_epochs):
            model.train()  
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()  
                y_pred = model(X_batch)
                loss = self.combined_loss(y_pred, y_batch)
                loss.backward()  
                optimizer.step()  
                train_loss += loss.item()  

            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            model.eval()  
            val_loss = 0.0
            val_iou = 0.0
            with torch.no_grad():  
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    loss = self.combined_loss(y_pred, y_batch)
                    val_loss += loss.item()
                    
                    # Calculate IoU for this batch
                    batch_iou = self.calculate_iou(y_pred, y_batch)
                    val_iou += batch_iou

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            val_iou /= len(val_loader)
            val_ious.append(val_iou)
            
            scheduler.step()  
            
            # Print epoch metrics
            print(f"Epoch [{epoch+1}/{num_epochs}] - {model_name} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iou = val_iou
                torch.save(model.state_dict(), best_model_path)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                print(f"Early stopping triggered for {model_name} after {epoch+1} epochs.")
                break
        
        print(f"Best validation loss: {best_val_loss:.4f}, Best IoU: {best_iou:.4f}, Model saved to {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        
        # Return model and training metrics
        return model, {'train_losses': train_losses, 'val_losses': val_losses, 'val_ious': val_ious}

    # Training Setup
    def setup_and_train(self, X_train_data, y_train_data, X_val_data, y_val_data, model, model_name, batch_size=16, device="cuda"):
        X_train_tensor = torch.tensor(X_train_data, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_data, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_data, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_data, dtype=torch.float32).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train the model
        model, metrics = self.train_model(model, train_loader, val_loader, model_name=model_name, device=device)
        
        return model, metrics
    
    # Evaluation Function (Updated to return predictions and IoU scores)
    def evaluate_model(self, model, X_test_data, y_test_data, batch_size=16, device="cuda"):
        model.eval()
        
        X_test_tensor = torch.tensor(X_test_data, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test_data, dtype=torch.float32).to(device)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        test_loss = 0.0
        test_iou = 0.0
        all_predictions = []
        all_ious = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = self.combined_loss(y_pred, y_batch)
                test_loss += loss.item()
                
                # Calculate IoU for each sample in batch
                pred_start, pred_end = y_pred[:, 0], y_pred[:, 1]
                true_start, true_end = y_batch[:, 0], y_batch[:, 1]
                
                # Calculate intersection and union for each sample
                intersection = torch.clamp(torch.min(pred_end, true_end) - torch.max(pred_start, true_start), min=0)
                union = (true_end - true_start) + (pred_end - pred_start) - intersection
                
                # Calculate IoU for each sample
                sample_ious = intersection / (union + 1e-6)
                all_ious.extend(sample_ious.cpu().numpy())
                
                # Store predictions
                all_predictions.append(y_pred.cpu())
    
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        
        test_loss /= len(test_loader)
        test_iou = sum(all_ious) / len(all_ious)
        
        return test_iou, all_predictions, np.array(all_ious)

    # Visualization Function
    def visualize_predictions(self, model, training_data, num_samples=50):
        # Ensure model is evaluated for each phase separately for both voltage and current
        y_pred_tensors = {}
        ious_scores = {}
        
        # Evaluate voltage phases
        print("\n=== Evaluating Voltage Phases ===")
        for phase in [1, 2, 3]:
            # Get test data for this phase
            X_test_seq = training_data[f"X_test_v_phase{phase}_seq"]
            y_test_seq = training_data[f"y_test_phase{phase}_seq"]
            
            print(f"Evaluating model on Voltage Phase {phase}...")
            mean_iou, y_pred_tensors[f'v_phase{phase}'], ious_scores[f'v_phase{phase}'] = self.evaluate_model(
                model, X_test_seq, y_test_seq, device=device
            )
            print(f"Mean IoU for Voltage Phase {phase}: {mean_iou:.4f}")
        
        # Evaluate current phases
        print("\n=== Evaluating Current Phases ===")
        for phase in [1, 2, 3]:
            # Get test data for this phase
            X_test_seq = training_data[f"X_test_c_phase{phase}_seq"]
            y_test_seq = training_data[f"y_test_phase{phase}_seq"]
            
            print(f"Evaluating model on Current Phase {phase}...")
            mean_iou, y_pred_tensors[f'c_phase{phase}'], ious_scores[f'c_phase{phase}'] = self.evaluate_model(
                model, X_test_seq, y_test_seq, device=device
            )
            print(f"Mean IoU for Current Phase {phase}: {mean_iou:.4f}")
        
        # Convert test labels to numpy
        y_test_tensors = {
            phase: training_data[f"y_test_phase{phase}_seq"]
            for phase in [1, 2, 3]
        }
        
        # Loop through each test sample and plot VOLTAGE phases
        for test_sample in range(min(num_samples, len(training_data["X_test_v_phase1_seq"]))):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)  # 1 row, 3 columns
            
            for i, phase in enumerate([1, 2, 3]):
                # Fetch phase-specific data
                X_test_seq = training_data[f"X_test_v_phase{phase}_seq"]
                time_test_seq = training_data[f"time_test_phase{phase}_seq"]
                y_pred_tensor = y_pred_tensors[f'v_phase{phase}']
                y_test_tensor = y_test_tensors[phase]
                
                X_sample = X_test_seq[test_sample]  # Shape: (time_steps, features)
                
                # If time sequence is available, use it; otherwise create a synthetic time axis
                if time_test_seq is not None:
                    time_sample = time_test_seq[test_sample]
                else:
                    time_sample = np.arange(X_sample.shape[0])
                
                # Get predicted and true start/end times
                pred_start_time = y_pred_tensor[test_sample][0]
                pred_end_time = y_pred_tensor[test_sample][1]
                orig_start_time = y_test_tensor[test_sample][0]
                orig_end_time = y_test_tensor[test_sample][1]
                
                # Plot on the respective subplot
                ax = axes[i]
                num_sensors = X_sample.shape[1] - 1  # Number of features in input data
                
                for sensor_idx in range(min(3, num_sensors)):  # Plot at most 3 sensors for clarity
                    ax.plot(time_sample, X_sample[:, sensor_idx])
                
                # Highlight predicted event window
                ax.axvspan(pred_start_time, pred_end_time, color='gold', alpha=0.3, label="Predicted Event")
                ax.axvline(orig_start_time, color='red', linestyle='dashed', label="True Start")
                ax.axvline(orig_end_time, color='green', linestyle='dashed', label="True End")
                
                # Calculate IoU for this sample
                sample_iou = ious_scores[f'v_phase{phase}'][test_sample]
                
                # Subplot-specific labels
                ax.set_xlabel('Time')
                if i == 0:
                    ax.set_ylabel('Voltage')
                ax.set_title(f'Voltage Phase {phase} (IoU: {sample_iou:.4f})')
                ax.legend()
                ax.grid(True)
            
            # Main title for the entire figure
            plt.suptitle(f'Test Sample {test_sample+1} - Voltage Data Across Phases')
            plt.tight_layout()
            plt.savefig(f'outputs/sample_{test_sample+1}_voltage_prediction.png')  # Save figure
    
        # Loop through each test sample and plot CURRENT phases
        for test_sample in range(min(num_samples, len(training_data["X_test_c_phase1_seq"]))):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)  # 1 row, 3 columns
            
            for i, phase in enumerate([1, 2, 3]):
                # Fetch phase-specific data
                X_test_seq = training_data[f"X_test_c_phase{phase}_seq"]
                time_test_seq = training_data[f"time_test_phase{phase}_seq"]
                y_pred_tensor = y_pred_tensors[f'c_phase{phase}']
                y_test_tensor = y_test_tensors[phase]
                
                X_sample = X_test_seq[test_sample]  # Shape: (time_steps, features)
                
                # If time sequence is available, use it; otherwise create a synthetic time axis
                if time_test_seq is not None:
                    time_sample = time_test_seq[test_sample]
                else:
                    time_sample = np.arange(X_sample.shape[0])
                
                # Get predicted and true start/end times
                pred_start_time = y_pred_tensor[test_sample][0]
                pred_end_time = y_pred_tensor[test_sample][1]
                orig_start_time = y_test_tensor[test_sample][0]
                orig_end_time = y_test_tensor[test_sample][1]
                
                # Plot on the respective subplot
                ax = axes[i]
                num_sensors = X_sample.shape[1] - 1 # Number of features in input data
                
                for sensor_idx in range(min(3, num_sensors)):  # Plot at most 3 sensors for clarity
                    ax.plot(time_sample, X_sample[:, sensor_idx], label=f'Sensor {sensor_idx+1}')
                
                # Highlight predicted event window
                ax.axvspan(pred_start_time, pred_end_time, color='gold', alpha=0.3, label="Predicted Event")
                ax.axvline(orig_start_time, color='red', linestyle='dashed', label="True Start")
                ax.axvline(orig_end_time, color='green', linestyle='dashed', label="True End")
                
                # Calculate IoU for this sample
                sample_iou = ious_scores[f'c_phase{phase}'][test_sample]
                
                # Subplot-specific labels
                ax.set_xlabel('Time')
                if i == 0:
                    ax.set_ylabel('Current')
                ax.set_title(f'Current Phase {phase} (IoU: {sample_iou:.4f})')
                ax.legend()
                ax.grid(True)
            
            # Main title for the entire figure
            plt.suptitle(f'Test Sample {test_sample+1} - Current Data Across Phases')
            plt.tight_layout()
            plt.savefig(f'outputs/sample_{test_sample+1}_current_prediction.png')  # Save figure
    

    def integrate_with_existing_pipeline():
        print("\n" + "="*60)
        print("STEP 6: SETTING UP FASTAPI VISUALIZATION SERVER")
        print("="*60)
        
        try:
            # Import the setup function
            from fast_api import setup_fastapi_environment
            
            # Setup the FastAPI environment with your training data
            setup_fastapi_environment(training_data)
            
            # Optional: Start the server automatically
            import subprocess
            import time
            
            print("\nStarting FastAPI server...")
            process = subprocess.Popen([
                "python", "-m", "uvicorn", 
                "fastapi_server:app", 
                "--reload", 
                "--host", "0.0.0.0", 
                "--port", "8000"
            ])
            
            # Give the server a moment to start
            time.sleep(3)
            
            print("✓ FastAPI server started successfully!")
            print("🌐 Visit http://localhost:8000 to view visualizations")
            print("📊 Interactive visualization interface is now available")
            
            # Keep the server running
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Shutting down server...")
                process.terminate()
                
        except Exception as e:
            print(f"✗ Error setting up FastAPI server: {e}")
            print("You can still run the server manually:")
            print("  python visualization_server.py")

if __name__ == "__main__":
    print("="*60)
    print("STARTING TIME SERIES SEGMENTATION PIPELINE")
    print("="*60)
    
    # Set parameters
    batch_size = 32
    input_shape = 2  # Assuming input dimension is 2
    hidden_dim = 64
    num_layers = 3
    dropout_rate = 0.1
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")
    
    # Initialize the TimeSeriesSegmentation class
    print("\n" + "="*50)
    print("STEP 1: INITIALIZING TimeSeriesSegmentation CLASS")
    print("="*50)
    ts_seg = TimeSeriesSegmentation()
    print("✓ TimeSeriesSegmentation class initialized successfully")
    print(f"  - Data directory: {ts_seg.data_dir}")
    print(f"  - Labels directory: {ts_seg.labels_dir}")
    print(f"  - Voltage columns: {ts_seg.voltage_columns}")
    print(f"  - Current columns: {ts_seg.current_columns}")
    
    # Step 1: Prepare data (loads labels and measurements, applies preprocessing)
    print("\n" + "="*50)
    print("STEP 2: PREPARING DATA (LOAD & PREPROCESS)")
    print("="*50)
    try:
        rms_df_with_labels, rms_df_with_labels2 = ts_seg.prepare_data()
        print("✓ Data preparation completed successfully")
        print(f"  - RMS DataFrame 1 shape: {rms_df_with_labels.shape}")
        print(f"  - RMS DataFrame 2 shape: {rms_df_with_labels2.shape}")
        print(f"  - RMS DataFrame 1 columns: {list(rms_df_with_labels.columns)}")
        print(f"  - Sample of RMS DataFrame 1:")
        print(rms_df_with_labels.head(3))
    except Exception as e:
        print(f"✗ Error in data preparation: {e}")
        exit(1)
    
    # Step 2: Prepare training data (split and create sequences)
    print("\n" + "="*50)
    print("STEP 3: PREPARING TRAINING DATA (SPLIT & SEQUENCE)")
    print("="*50)
    try:
        training_data = ts_seg.prepare_training_data(rms_df_with_labels, rms_df_with_labels2)
        print("✓ Training data preparation completed successfully")
        
        # Print shapes of all prepared data
        for key, value in training_data.items():
            if isinstance(value, np.ndarray):
                print(f"  - {key}: {value.shape}")
        
        print(f"  - Total training data keys: {len(training_data.keys())}")
        
    except Exception as e:
        print(f"✗ Error in training data preparation: {e}")
        exit(1)
    
    # Create the model (you'll need to import your CNN_GRU_Attention model)
    print("\n" + "="*50)
    print("STEP 4: MODEL INITIALIZATION")
    print("="*50)
    try:
        # Assuming you have CNN_GRU_Attention class defined somewhere
        model = CNN_GRU_Attention(input_shape, hidden_dim=hidden_dim, num_layers=num_layers, dropout_rate=dropout_rate).to(device)
        print("  - Model parameters would be:")
        print(f"    * Input shape: {input_shape}")
        print(f"    * Hidden dim: {hidden_dim}")
        print(f"    * Num layers: {num_layers}")
        print(f"    * Dropout rate: {dropout_rate}")
    except Exception as e:
        print(f"✗ Error in model initialization: {e}")
        model = None
    
    # Sequential Training: Voltage Phases
    print("\n" + "="*60)
    print("STEP 5: SEQUENTIAL TRAINING - VOLTAGE PHASES")
    print("="*60)

    # Save initial weights for comparison later
    torch.save(model.state_dict(), "models/initial_weights.pth")
    
    # === Sequential Training: First all voltage phases ===
    print("\n=== Sequential Training: Voltage Phases ===\n")
    
    phase_metrics = {}
    
    # Train on voltage data for each phase sequentially
    for phase in [1, 2, 3]:
        print(f"\n=== Training on Voltage Phase {phase} ===\n")
        
        # Get data for this phase (you'll need to replace these with your actual data variables)
        X_train_data = training_data[f"X_train_v_phase{phase}_seq"]
        y_train_data = training_data["y_train_seq"]
        X_val_data = training_data[f"X_val_v_phase{phase}_seq"]
        y_val_data = training_data[f"y_val_phase{phase}_seq"]
        
        # Train model on this phase
        model, metrics = ts_seg.setup_and_train(
            X_train_data, y_train_data, 
            X_val_data, y_val_data, 
            model, f"voltage_phase{phase}", 
            batch_size=batch_size, 
            device=device
        )
        
        # Save metrics
        phase_metrics[f"voltage_phase{phase}"] = metrics
        
        # Save checkpoint after each phase
        torch.save(model.state_dict(), f"models/model_after_voltage_phase{phase}.pth")
    
    # After training on all voltage phases, save this state
    torch.save(model.state_dict(), "models/model_after_all_voltage_phases.pth")
    
    # === Sequential Training: Now all current phases ===
    print("\n=== Sequential Training: Current Phases ===\n")
    
    # Train on current data for each phase sequentially
    for phase in [1, 2, 3]:
        print(f"\n=== Training on Current Phase {phase} ===\n")
        
        # Get data for this phase
        X_train_data = training_data[f"X_train_c_phase{phase}_seq"]
        y_train_data = training_data["y_train_seq"]
        X_val_data = training_data[f"X_val_c_phase{phase}_seq"]
        y_val_data = training_data[f"y_val_phase{phase}_seq"]
            
        # Train model on this phase
        model, metrics = ts_seg.setup_and_train(
            X_train_data, y_train_data, 
            X_val_data, y_val_data, 
            model, f"current_phase{phase}", 
            batch_size=batch_size, 
            device=device
        )
        
        # Save metrics
        phase_metrics[f"current_phase{phase}"] = metrics
        
        # Save checkpoint after each phase
        torch.save(model.state_dict(), f"models/model_after_current_phase{phase}.pth")
    
    # Final model weights
    torch.save(model.state_dict(), "models/final_model.pth")

    print("\n=== Visualizing Predictions on Test Data ===\n")
    ts_seg.visualize_predictions(model, training_data, num_samples=50)

    ts_seg.integrate_with_existing_pipeline()


    # Add this code to the end of your existing pipeline (after the visualize_predictions line)

# ============================================================================
# FASTAPI VISUALIZATION SERVER INTEGRATION
# ============================================================================

print("\n" + "="*60)
print("STEP 6: SETTING UP FASTAPI VISUALIZATION SERVER")
print("="*60)

import pickle
import os
import subprocess
import time
from pathlib import Path

def setup_fastapi_visualization_server():
    """Setup and start the FastAPI visualization server"""
    
    # 1. Create necessary directories
    os.makedirs("models", exist_ok=True)
    print("✓ Models directory verified")
    
    # 2. Save training data for FastAPI access
    try:
        with open("training_data.pkl", "wb") as f:
            pickle.dump(training_data, f)
        print("✓ Training data saved for FastAPI access")
    except Exception as e:
        print(f"⚠️  Warning: Could not save training data: {e}")
    
    # 3. Verify model files exist
    model_files = ["models/final_model.pth"]
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"⚠️  Warning: Missing model files: {missing_files}")
    else:
        print("✓ Model files verified")
    
    # 4. Create a simple FastAPI server file if it doesn't exist
    if not os.path.exists("run_server.py"):
        create_simple_server_file()
        print("✓ FastAPI server file created")
    
    # 5. Create requirements file
    requirements = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0", 
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "python-multipart==0.0.6"
    ]
    
    with open("requirements_viz.txt", "w") as f:
        f.write("\n".join(requirements))
    print("✓ Requirements file created")
    
    print("\n" + "="*50)
    print("🎉 FASTAPI SETUP COMPLETED!")
    print("="*50)
    print("To start the visualization server:")
    print("1. Install requirements: pip install -r requirements_viz.txt")
    print("2. Run server: python visualization_server.py")
    print("3. Visit: http://localhost:8000")
    print("="*50)
    
    # Ask user if they want to start the server now
    try:
        start_now = input("\nStart visualization server now? (y/n): ").lower().strip()
        if start_now == 'y':
            start_fastapi_server()
    except KeyboardInterrupt:
        print("\nSkipping server startup...")

def create_simple_server_file():
    """Create a simplified FastAPI server file"""
    server_code = '''
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pickle
import os
from datetime import datetime

app = FastAPI(title="Time Series Visualization")

# Global variables
model = None
training_data = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup():
    global model, training_data
    
    # Load training data
    if os.path.exists("training_data.pkl"):
        with open("training_data.pkl", "rb") as f:
            training_data = pickle.load(f)
        print("✓ Training data loaded")
    
    # Load model (you'll need to import your model class)
    try:
        # Replace with your actual model import and initialization
        # from your_module import CNN_GRU_Attention
        # model = CNN_GRU_Attention(input_shape=2, hidden_dim=64, num_layers=3, dropout_rate=0.1)
        # model.load_state_dict(torch.load("models/final_model.pth", map_location=device))
        # model.to(device)
        # model.eval()
        print("⚠️  Model loading commented out - update with your model class")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=\"\"\"
    <!DOCTYPE html>
    <html>
    <head>
        <title>Time Series Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            h1 { color: #333; text-align: center; }
            .controls { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 5px; }
            select, input, button { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #007bff; color: white; cursor: pointer; }
            button:hover { background: #0056b3; }
            .plot-container { margin-top: 20px; text-align: center; }
            .plot-container img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Time Series Segmentation Results</h1>
            
            <div class="controls">
                <label>Phase Type:</label>
                <select id="phaseType">
                    <option value="voltage">Voltage</option>
                    <option value="current">Current</option>
                </select>
                
                <label>Phase Number:</label>
                <select id="phaseNum">
                    <option value="1">Phase 1</option>
                    <option value="2">Phase 2</option>
                    <option value="3">Phase 3</option>
                </select>
                
                <label>Samples:</label>
                <input type="number" id="numSamples" value="50" min="10" max="200">
                
                <button onclick="generatePlot()">Generate Plot</button>
            </div>
            
            <div id="loading" style="display:none; text-align:center;">
                <p>🔄 Generating visualization...</p>
            </div>
            
            <div id="plotContainer" class="plot-container"></div>
        </div>

        <script>
            async function generatePlot() {
                const phaseType = document.getElementById('phaseType').value;
                const phaseNum = document.getElementById('phaseNum').value;
                const numSamples = document.getElementById('numSamples').value;
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('plotContainer').innerHTML = '';
                
                try {
                    const response = await fetch(`/plot?phase_type=${phaseType}&phase_num=${phaseNum}&num_samples=${numSamples}`);
                    const result = await response.json();
                    
                    if (result.plot) {
                        document.getElementById('plotContainer').innerHTML = 
                            `<img src="data:image/png;base64,${result.plot}" alt="Visualization">`;
                    }
                } catch (error) {
                    document.getElementById('plotContainer').innerHTML = 
                        `<p style="color:red;">Error: ${error.message}</p>`;
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    \"\"\")

@app.get("/plot")
async def generate_plot(phase_type: str = "voltage", phase_num: int = 1, num_samples: int = 50):
    global model, training_data
    
    if training_data is None:
        raise HTTPException(status_code=500, detail="Training data not loaded")
    
    try:
        # Get data based on phase
        if phase_type == "voltage":
            X_data = training_data.get(f"X_test_v_phase{phase_num}_seq", np.random.randn(num_samples, 10, 2))
            y_data = training_data.get(f"y_test_phase{phase_num}_seq", np.random.randn(num_samples, 1))
        else:
            X_data = training_data.get(f"X_test_c_phase{phase_num}_seq", np.random.randn(num_samples, 10, 2))
            y_data = training_data.get(f"y_test_phase{phase_num}_seq", np.random.randn(num_samples, 1))
        
        # Limit samples
        X_data = X_data[:num_samples]
        y_data = y_data[:num_samples]
        
        # Generate predictions (mock if model not loaded)
        if model is not None:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_data).to(device)
                predictions = model(X_tensor)
                predictions = torch.sigmoid(predictions).cpu().numpy()
        else:
            # Mock predictions for demonstration
            predictions = np.random.sigmoid(np.random.randn(*y_data.shape))
        
        # Create plot
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Time Series Results - {phase_type.title()} Phase {phase_num}', fontsize=14)
        
        # Plot 1: Predictions vs Ground Truth
        sample_range = min(20, len(predictions))
        x_range = range(sample_range)
        ax1.plot(x_range, predictions[:sample_range].flatten(), 'b-', label='Predictions', linewidth=2)
        ax1.plot(x_range, y_data[:sample_range].flatten(), 'r--', label='Ground Truth', linewidth=2)
        ax1.set_title('Predictions vs Ground Truth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution
        ax2.hist(predictions.flatten(), bins=30, alpha=0.7, label='Predictions', color='blue')
        ax2.hist(y_data.flatten(), bins=30, alpha=0.7, label='Ground Truth', color='red')
        ax2.set_title('Value Distributions')
        ax2.legend()
        
        # Plot 3: Scatter
        ax3.scatter(y_data.flatten(), predictions.flatten(), alpha=0.6)
        ax3.plot([y_data.min(), y_data.max()], [y_data.min(), y_data.max()], 'r--')
        ax3.set_xlabel('Ground Truth')
        ax3.set_ylabel('Predictions')
        ax3.set_title('Scatter Plot')
        
        # Plot 4: Errors
        errors = predictions.flatten() - y_data.flatten()
        ax4.hist(errors, bins=30, alpha=0.7, color='green')
        ax4.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
        ax4.set_title('Prediction Errors')
        ax4.legend()
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Calculate metrics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        return {
            "plot": plot_data,
            "mae": float(mae),
            "rmse": float(rmse),
            "num_samples": len(predictions),
            "phase_type": phase_type,
            "phase_num": phase_num
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": training_data is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Time Series Visualization Server...")
    print("Visit http://localhost:8000 to view the interface")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
'''
    
    with open("visualization_server.py", "w") as f:
        f.write(server_code)

def start_fastapi_server():
    """Start the FastAPI server"""
    try:
        print("\n🚀 Starting FastAPI server...")
        print("📊 Visualization interface will be available at: http://localhost:8000")
        print("⏹️  Press Ctrl+C to stop the server")
        
        # Start the server
        process = subprocess.Popen([
            "python", "visualization_server.py"
        ])
        
        # Wait a moment for server to start
        time.sleep(2)
        print("✅ Server started successfully!")
        
        # Keep server running until interrupted
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
            process.terminate()
            process.wait()
            print("✅ Server stopped")
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("You can start it manually by running: python visualization_server.py")

# Run the setup
try:
    setup_fastapi_visualization_server()
except Exception as e:
    print(f"❌ Error in FastAPI setup: {e}")
    print("You can set up manually using the created files.")

print("\n" + "="*60)
print("PIPELINE COMPLETED WITH FASTAPI INTEGRATION")
print("="*60)
    