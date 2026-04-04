# Postural Instability Analysis in Parkinson’s Disease from a Single Lower-Back Sensor: A Deep Learning Cross-Dataset Approach
The different datasets are in different formats, unit of measurements and sampling rates.
Re-organise all datasets (each in a separate CSV file) so that they all share the same format:

1. Use consistent columns name. E.g., acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, subjectID, taskID
2. Resample all datasets to a fixed sampling rate (e.g., 64 Hz or 128 Hz).
3. Adjust axes orientation so that each axis represents the same direction in all datasets (e.g., decide that the x-axis always represent the vertical positive or negative direction; the y-axis represents always the antero-posterior direction, positive backward; etc...).
4. For each dataset, make sure the subjectID is correctly associate across sensor data and clinical data.
5. Collect a minimum set of demographic (e.g., age, gender) and clinical (e.g., disease duration, H&Y stage, UPDRS-III, postural stability scale) information.