import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os

celestrak_data = pd.read_csv('/mnt/ionosphere-data/celestrak/kp_ap_timeseries.csv')

print(celestrak_data)

#plt a figure with two subplots histogram of the Kp and Ap values

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].hist(celestrak_data['Kp'].dropna(), edgecolor='black', alpha=0.7)
axs[0].set_title('Histogram of Kp Values')
axs[0].set_xlabel('Kp Value')
axs[0].set_yscale('log')  # Use logarithmic scale for better visibility
axs[0].set_ylabel('Frequency')
axs[1].hist(celestrak_data['Ap'].dropna(), bins = np.logspace(np.log10(1), np.log10(300), 20), edgecolor='black', alpha=0.7)
axs[1].set_xscale('log')  # Use logarithmic scale for better visibility
axs[1].set_yscale('log')  # Use logarithmic scale for better visibility
axs[1].set_title('Histogram of Ap Values')
axs[1].set_xlabel('Ap Value')
axs[1].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

#Create label for the Kp timeseries
# Where kp<5 put G0, when kp>5 put G1, when kp>6 put G2, when kp>7 put G3, when kp>8 put G4, when kp>9 put G5
def label_kp(kp_value):
    if kp_value < 5:
        return 'G0'
    elif kp_value < 6:
        return 'G1'
    elif kp_value < 7:
        return 'G2'
    elif kp_value < 8:
        return 'G3'
    elif kp_value < 9:
        return 'G4'
    else:
        return 'G5'
celestrak_data['Kp_Label'] = celestrak_data['Kp'].apply(label_kp)
# Save the updated dataframe with Kp labels to a new CSV file

print(celestrak_data.head())

#celestrak_data.to_csv('/mnt/ionosphere-data/celestrak/kp_ap_timeseries_with_labels.csv', index=False)

# old function with threshold - PART 1
def find_strong_kp_intervals(data, initial_threshold=4.5, min_hours=9, avg_threshold=6):
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.sort_values('Datetime').reset_index(drop=True)

    # Step 1: Flag where Kp > initial threshold (e.g., 3)
    data['above_initial'] = data['Kp'] > initial_threshold

    # Step 2: Group consecutive values above threshold
    data['group'] = (data['above_initial'] != data['above_initial'].shift()).cumsum()

    results = []
    for group_id, group_df in data.groupby('group'):
        if not group_df['above_initial'].iloc[0]:
            continue  # skip groups not above threshold

        # Calculate duration
        duration_hours = (group_df['Datetime'].iloc[-1] - group_df['Datetime'].iloc[0]).total_seconds() / 3600

        if duration_hours >= min_hours:
            avg_kp = group_df['Kp'].mean()
            max_kp = group_df['Kp'].max()

            if avg_threshold <= avg_kp < avg_threshold + 1: #avg_kp > avg_threshold condition to be inclusive
            
                results.append({
                    'Start': group_df['Datetime'].iloc[0],
                    'End': group_df['Datetime'].iloc[-1],
                    'Duration_hours': duration_hours,
                    'Average_Kp': avg_kp,
                    'Max_Kp': max_kp
                })

    return pd.DataFrame(results)


kp_intervals_sliding = find_strong_kp_intervals(celestrak_data, initial_threshold=4.5, min_hours=9, avg_threshold=6)
kp_intervals_sliding = kp_intervals_sliding[kp_intervals_sliding['Start'] > '2010-01-01']
print("High average Kp intervals after 2010:")
print(kp_intervals_sliding)
print(len(kp_intervals_sliding), "intervals found.")


# Output directory
output_dir = '/mnt/ionosphere-data/events/'

# Defined thresholds and windows
thresholds = [5, 6, 7, 8, 9]               # 5 to 9 inclusive
window_sizes = [3, 6, 9, 12]            # Only these 4 window lengths

# Loop through combinations
for avg_threshold in thresholds:
    for window_hours in window_sizes:
        kp_intervals_sliding = find_strong_kp_intervals(celestrak_data, initial_threshold=4.5, min_hours=window_hours, avg_threshold=avg_threshold)

        # Filter to post-2010 events
        df = kp_intervals_sliding[kp_intervals_sliding['Start'] > '2010-01-01']

        if df.empty:
            print(f"No events for G{avg_threshold}H{window_hours}, skipping.")
            continue

        # Save as G{threshold}H{window}.csv
        filename = f'G{avg_threshold-4}H{window_hours}.csv'
        filepath = os.path.join(output_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} events to {filename}")

#based on label - PART 2

data_kp_with_labels = pd.read_csv('/mnt/ionosphere-data/celestrak/kp_ap_timeseries_with_labels.csv')

display(data_kp_with_labels)

def classify_storm_events_by_label(df):
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

    # Flag if this is a non-G0 label
    df['is_storm'] = df['Kp_Label'] != 'G0'

    # Group consecutive storm entries
    df['group'] = (df['is_storm'] != df['is_storm'].shift()).cumsum()

    results = []

    for _, group_df in df.groupby('group'):
        if not group_df['is_storm'].iloc[0]:
            continue  # skip non-storm periods

        start_time = group_df['Datetime'].iloc[0]
        end_time = group_df['Datetime'].iloc[-1]
        duration_hours = len(group_df) * 3  # Assuming 3-hour intervals

        max_label = group_df['Kp_Label'].max()  # 'G1', 'G2', etc. (alphabetical but we can order it)

        results.append({
            'Start': start_time,
            'End': end_time,
            'Duration_hours': duration_hours,
            'Max_Kp_Label': max_label
        })

    return pd.DataFrame(results)

#usage
storm_intervals = classify_storm_events_by_label(data_kp_with_labels)
# Filter to post-2010 events
storm_events_df = storm_intervals[storm_intervals['Start'] > '2010-01-01']
display(storm_events_df)

# Output directory
output_dir = '/mnt/ionosphere-data/events/'

# Define storm levels and durations to categorize
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
window_sizes = [3, 6, 9, 12]  # minimum durations (in hours)

# Loop through combinations
for label in labels:
    for window_hours in window_sizes:
        # Filter by label and duration
        df = storm_events_df[
            (storm_events_df['Max_Kp_Label'] == label) &
            (storm_events_df['Duration_hours'] >= window_hours) &
            (storm_events_df['Start'] > '2010-01-01')
        ]

        if df.empty:
            print(f"No events for {label}H{window_hours}, skipping.")
            continue

        # Save as G{label_number}H{window}.csv
        label_number = int(label[1])  # e.g., "G3" → 3
        filename = f'G{label_number}H{window_hours}_labels.csv'
        filepath = os.path.join(output_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} events to {filename}")
