import os
import numpy as np
import pandas as pd
from .pham_turn_detection import PhamTurnDetection

def extract_turns_from_dataset(csv_path, sf=64.0, tasks_to_process=[2]):
    """
    Loads a dataset, flags turns using PhamTurnDetection with an 'isTurn' column,
    and returns the complete DataFrame.
    """
    if not os.path.exists(csv_path):
        print(f"Error: The file {csv_path} does not exist.")
        return None

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Initialize isTurn column to 0 for all rows
    df['isTurn'] = 0
    
    # Constants for Pham library
    TRACKING_SYSTEM = "imu"
    TRACKED_POINT = "LowerBack"
    
    walking_mask = df["taskID"].isin(tasks_to_process)
    
    if not walking_mask.any():
        print(f"No data found for tasks {tasks_to_process}")
        return df

    # Group by subject and session within the walking tasks
    grouped = df[walking_mask].groupby(["subjectID", "sessionID"])

    for (sub_id, sess_id), group_df in grouped:
        # Get indices to update the original dataframe later
        indices = group_df.index
        
        motion_data_group = pd.DataFrame({
            f"{TRACKED_POINT}_ACCEL_x": group_df["acc_x"].astype(float) * 9.81,
            f"{TRACKED_POINT}_ACCEL_y": group_df["acc_y"].astype(float) * 9.81,
            f"{TRACKED_POINT}_ACCEL_z": group_df["acc_z"].astype(float) * 9.81,
            f"{TRACKED_POINT}_GYRO_x": np.rad2deg(group_df["gyro_x"].astype(float)),
            f"{TRACKED_POINT}_GYRO_y": np.rad2deg(group_df["gyro_y"].astype(float)),
            f"{TRACKED_POINT}_GYRO_z": np.rad2deg(group_df["gyro_z"].astype(float)),
        })

        # Clean micro-gaps
        motion_data_group = interpolate_short_nans_and_zero_long(
            motion_data_group, 
            max_interp_gap=int(0.1 * sf)
        )

        pham = PhamTurnDetection()
        try:
            pham.detect(
                accel_data=motion_data_group[[f"{TRACKED_POINT}_ACCEL_x", f"{TRACKED_POINT}_ACCEL_y", f"{TRACKED_POINT}_ACCEL_z"]],
                gyro_data=motion_data_group[[f"{TRACKED_POINT}_GYRO_x", f"{TRACKED_POINT}_GYRO_y", f"{TRACKED_POINT}_GYRO_z"]],
                gyro_vertical=f"{TRACKED_POINT}_GYRO_x", 
                sampling_freq_Hz=int(sf),
                tracking_system=TRACKING_SYSTEM,
                tracked_point=TRACKED_POINT,
                plot_results=False,
            )

            # Update isTurn column based on detection flags
            if len(pham.flags_start_90) > 0:
                for start, end in zip(pham.flags_start_90, pham.flags_end_90):
                    turn_indices = indices[start:end]
                    df.loc[turn_indices, 'isTurn'] = 1
                    
        except Exception as e:
            print(f"Error on Sub {sub_id}, Sess {sess_id}: {e}")

    # Desired column ordering (added isTurn, removed turn_id)
    cols = ['subjectID', 'sessionID', 'taskID', 'timestamp', 'isTurn',
            'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    final_df = df[[c for c in cols if c in df.columns]]
    
    turns_count = df['isTurn'].sum()
    print(f"Extraction complete: {turns_count} rows flagged as turns out of {len(df)}.")
    return final_df

def interpolate_short_nans_and_zero_long(data, max_interp_gap):
    cleaned = data.copy()
    for col in cleaned.columns:
        x = cleaned[col].to_numpy(copy=True)
        isnan = np.isnan(x)
        if not isnan.any(): continue
        nan_idx = np.where(isnan)[0]
        splits = np.split(nan_idx, np.where(np.diff(nan_idx) != 1)[0] + 1)
        for seg in splits:
            if len(seg) <= max_interp_gap:
                start, end = seg[0], seg[-1]
                left, right = start - 1, end + 1
                if left >= 0 and right < len(x):
                    x[start:end+1] = np.interp(np.arange(start, end+1), [left, right], [x[left], x[right]])
                else: x[start:end+1] = 0.0
            else: x[seg] = 0.0
        cleaned[col] = x
    return cleaned