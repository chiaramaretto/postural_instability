import os
import numpy as np
import pandas as pd
from pham_turn_detection import PhamTurnDetection

# =========================================================
# Utility: interpolate short NaNs, zero long NaNs
# =========================================================
def interpolate_short_nans_and_zero_long(data, max_interp_gap):
    cleaned = data.copy()
    for col in cleaned.columns:
        x = cleaned[col].to_numpy(copy=True)
        isnan = np.isnan(x)
        if not isnan.any():
            continue
        nan_idx = np.where(isnan)[0]
        splits = np.split(nan_idx, np.where(np.diff(nan_idx) != 1)[0] + 1)
        for seg in splits:
            if len(seg) <= max_interp_gap:
                start, end = seg[0], seg[-1]
                left, right = start - 1, end + 1
                if left >= 0 and right < len(x):
                    x[start:end + 1] = np.interp(
                        np.arange(start, end + 1), [left, right], [x[left], x[right]]
                    )
                else:
                    x[start:end + 1] = 0.0
            else:
                x[seg] = 0.0
        cleaned[col] = x
    return cleaned

# =========================================================
# USER PARAMETERS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "cleaned_data", "omnia_park_sensor.csv")
SAMPLING_FREQUENCY = 64.0
TRACKING_SYSTEM = "imu"
TRACKED_POINT = "LowerBack"
TASKS_TO_KEEP = [2]

# OUTPUT_CSV è uguale a CSV_PATH per sovrascrivere
OUTPUT_CSV = CSV_PATH
TEMP_CSV = OUTPUT_CSV + ".tmp"

# =========================================================
# Load CSV
# =========================================================
print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

all_extracted_turns = []

# =========================================================
# Processing Loop
# =========================================================
df_filtered = df[df["taskID"].isin(TASKS_TO_KEEP)].reset_index(drop=True)
grouped = df_filtered.groupby(["subjectID", "sessionID"])

for (sub_id, sess_id), group_df in grouped:
    print(f"Processing Subject {sub_id}, Session {sess_id}...")
    original_rows = group_df.reset_index(drop=True)

    motion_data_group = pd.DataFrame({
        f"{TRACKED_POINT}_ACCEL_x": original_rows["acc_x"].astype(float) * 9.81,
        f"{TRACKED_POINT}_ACCEL_y": original_rows["acc_y"].astype(float) * 9.81,
        f"{TRACKED_POINT}_ACCEL_z": original_rows["acc_z"].astype(float) * 9.81,
        f"{TRACKED_POINT}_GYRO_x": np.rad2deg(original_rows["gyro_x"].astype(float)),
        f"{TRACKED_POINT}_GYRO_y": np.rad2deg(original_rows["gyro_y"].astype(float)),
        f"{TRACKED_POINT}_GYRO_z": np.rad2deg(original_rows["gyro_z"].astype(float)),
    })

    motion_data_group = interpolate_short_nans_and_zero_long(
        motion_data_group,
        max_interp_gap=int(0.1 * SAMPLING_FREQUENCY),
    )

    pham = PhamTurnDetection()
    try:
        pham.detect(
            accel_data=motion_data_group[[f"{TRACKED_POINT}_ACCEL_x", 
                                          f"{TRACKED_POINT}_ACCEL_y", 
                                          f"{TRACKED_POINT}_ACCEL_z"]],
            gyro_data=motion_data_group[[f"{TRACKED_POINT}_GYRO_x", 
                                         f"{TRACKED_POINT}_GYRO_y", 
                                         f"{TRACKED_POINT}_GYRO_z"]],
            gyro_vertical=f"{TRACKED_POINT}_GYRO_x",
            sampling_freq_Hz=int(SAMPLING_FREQUENCY),
            tracking_system=TRACKING_SYSTEM,
            tracked_point=TRACKED_POINT,
            plot_results=False,
        )

        if len(pham.flags_start_90) == 0:
            print(f"   --> No turns detected for Subject {sub_id}")
            continue

        for turn_idx, (start, end) in enumerate(zip(pham.flags_start_90, pham.flags_end_90)):
            turn_segment = original_rows.iloc[start:end].copy()
            turn_segment['turn_id'] = turn_idx + 1
            all_extracted_turns.append(turn_segment)

        print(f"   --> SUCCESS: Extracted {len(pham.flags_start_90)} turns.")

    except Exception as e:
        print(f"   --> Error processing Subject {sub_id}: {e}")

# =========================================================
# Save Final Result 
# =========================================================
if all_extracted_turns:
    final_df = pd.concat(all_extracted_turns, ignore_index=True)
    
    cols_to_save = ['subjectID', 'sessionID', 'taskID', 'turn_id', 'timestamp', 
                    'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    final_df = final_df[[c for c in cols_to_save if c in final_df.columns]]
    
    try:
        final_df.to_csv(TEMP_CSV, index=False)
        os.replace(TEMP_CSV, OUTPUT_CSV)
        
        print(f"\nSUCCESS! File '{OUTPUT_CSV}' sovrascritto con i soli turni.")
        print(f"Totale righe: {len(final_df)}")
    except Exception as e:
        print(f"\nERRORE durante il salvataggio: {e}")
        if os.path.exists(TEMP_CSV):
            os.remove(TEMP_CSV)
else:
    print("\nNo turns detected. L'originale non è stato modificato.")