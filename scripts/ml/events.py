import pandas as pd

# SDOML-lite: 2010-05-13T00:00:00 - 2024-07-27T00:00:00

# NOAA Space Weather Scales
# Geomagnetic Storms
# https://www.swpc.noaa.gov/noaa-scales-explanation
#
# G0: Unsettled (Kp < 4)
# G1: Minor (5 <= Kp < 6)
# G2: Moderate (6 <= Kp < 7)
# G3: Strong (7 <= Kp < 8)
# G4: Severe (8 <= Kp < 9)
# G5: Extreme (Kp >= 9)

# The Mestici Scale
# G1H9: G1 and higher geomagnetic storms, with a duration of at least 9 hours
# G2H9: G2 and higher geomagnetic storms, with a duration of at least 9 hours
# G3H9: G3 and higher geomagnetic storms, with a duration of at least 9 hours
# G4H9: G4 and higher geomagnetic storms, with a duration of at least 9 hours
# G5H9: G5 and higher geomagnetic storms, with a duration of at least 9 hours
# The events are based on the Mestici Scale, which is a measure of geomagnetic storm activity.

# Event name prefix; Start Time YYYY-MM-DDTHH:MM:SS; End Time YYYY-MM-DDTHH:MM:SS; max Kp
events = []
events.append(('G2H9', '2011-08-05T18:00:00', '2011-08-06T03:00:00', 7.7))
events.append(('G2H9', '2011-10-24T18:00:00', '2011-10-25T06:00:00', 7.3))
events.append(('G2H9', '2012-03-09T00:00:00', '2012-03-09T15:00:00', 8))
events.append(('G2H9', '2013-03-17T06:00:00', '2013-03-17T21:00:00', 6.7))
events.append(('G2H9', '2015-06-22T12:00:00', '2015-06-23T12:00:00', 8.3))
events.append(('G2H9', '2015-09-11T06:00:00', '2015-09-11T15:00:00', 7))
events.append(('G2H9', '2017-09-07T21:00:00', '2017-09-08T21:00:00', 8.3))
events.append(('G2H9', '2017-11-07T18:00:00', '2017-11-08T03:00:00', 6.3))
events.append(('G2H9', '2021-11-03T21:00:00', '2021-11-04T12:00:00', 7.7))
events.append(('G2H9', '2023-03-23T09:00:00', '2023-03-24T06:00:00', 8))
events.append(('G2H9', '2023-04-23T12:00:00', '2023-04-24T09:00:00', 8.3))
events.append(('G2H9', '2023-11-05T09:00:00', '2023-11-05T21:00:00', 7.3))
events.append(('G2H9', '2024-05-10T15:00:00', '2024-05-12T03:00:00', 9))
events.append(('G2H9', '2024-06-28T09:00:00', '2024-06-28T18:00:00', 7.7))
events.append(('G2H9', '2024-08-11T21:00:00', '2024-08-13T00:00:00', 8))
events.append(('G2H9', '2024-10-10T15:00:00', '2024-10-11T12:00:00', 8.7))
events.append(('G2H9', '2025-01-01T03:00:00', '2025-01-01T18:00:00', 8))
events.append(('G2H9', '2025-04-16T09:00:00', '2025-04-16T21:00:00', 7.7))
events.append(('G2H9', '2025-06-01T00:00:00', '2025-06-01T18:00:00', 7.7))

events = pd.DataFrame(events, columns=['prefix', 'start', 'end', 'max_kp'])

# sort by start
events = events.sort_values(by='start')

# Pre-start and post-end buffers to include, in multiples of the start-end duration
pre_start = 0.5
post_end = 1



format = '%Y-%m-%dT%H:%M:%S'
events['date_start'] = pd.to_datetime(events['start'], format=format)
events['date_end'] = pd.to_datetime(events['end'], format=format)
events['duration'] = events['date_end'] - events['date_start']
# Adjust start and end dates based on pre_start and post_end
events['date_start_adjusted'] = events['date_start'] - pre_start * events['duration']
events['date_end_adjusted'] = events['date_end'] + post_end * events['duration']

# Round down date_start to nearest :00, :15, :30, :45
events['date_start'] = events['date_start'].dt.floor('15min')

# count the number of times each unique prefix appears
events['event_id'] = events.groupby('prefix').cumcount() + 1

EventCatalog = {}
for prefix in events['prefix'].unique():
    events_with_prefix = events[events['prefix'] == prefix]
    num_events = len(events_with_prefix)
    for i in range(num_events):
        event = events_with_prefix.iloc[i]
        # event_id = prefix + '-' + str(i+1).zfill(len(str(num_events)))
        date_start = event['date_start'].isoformat()
        date_end = event['date_end'].isoformat()
        date_start_adjusted = event['date_start_adjusted'].isoformat()
        date_end_adjusted = event['date_end_adjusted'].isoformat()
        duration = event['duration']
        event_id = prefix + '-' + date_start
        max_kp = event['max_kp']
        EventCatalog[event_id] = date_start, date_end, date_start_adjusted, date_end_adjusted, duration, max_kp

for event, val in EventCatalog.items():
    print(event, val[0], val[1], val[2], val[3], val[4], val[5])

# event_id                 date_start          date_end            date_start_adjusted date_end_adjusted   duration        max_kp
# G2H9-2011-08-05T18:00:00 2011-08-05T18:00:00 2011-08-06T03:00:00 2011-08-05T13:30:00 2011-08-06T12:00:00 0 days 09:00:00 7.7
# G2H9-2011-10-24T18:00:00 2011-10-24T18:00:00 2011-10-25T06:00:00 2011-10-24T12:00:00 2011-10-25T18:00:00 0 days 12:00:00 7.3
# G2H9-2012-03-09T00:00:00 2012-03-09T00:00:00 2012-03-09T15:00:00 2012-03-08T16:30:00 2012-03-10T06:00:00 0 days 15:00:00 8.0
# G2H9-2013-03-17T06:00:00 2013-03-17T06:00:00 2013-03-17T21:00:00 2013-03-16T22:30:00 2013-03-18T12:00:00 0 days 15:00:00 6.7
# G2H9-2015-06-22T12:00:00 2015-06-22T12:00:00 2015-06-23T12:00:00 2015-06-22T00:00:00 2015-06-24T12:00:00 1 days 00:00:00 8.3
# G2H9-2015-09-11T06:00:00 2015-09-11T06:00:00 2015-09-11T15:00:00 2015-09-11T01:30:00 2015-09-12T00:00:00 0 days 09:00:00 7.0
# G2H9-2017-09-07T21:00:00 2017-09-07T21:00:00 2017-09-08T21:00:00 2017-09-07T09:00:00 2017-09-09T21:00:00 1 days 00:00:00 8.3
# G2H9-2017-11-07T18:00:00 2017-11-07T18:00:00 2017-11-08T03:00:00 2017-11-07T13:30:00 2017-11-08T12:00:00 0 days 09:00:00 6.3
# G2H9-2021-11-03T21:00:00 2021-11-03T21:00:00 2021-11-04T12:00:00 2021-11-03T13:30:00 2021-11-05T03:00:00 0 days 15:00:00 7.7
# G2H9-2023-03-23T09:00:00 2023-03-23T09:00:00 2023-03-24T06:00:00 2023-03-22T22:30:00 2023-03-25T03:00:00 0 days 21:00:00 8.0
# G2H9-2023-04-23T12:00:00 2023-04-23T12:00:00 2023-04-24T09:00:00 2023-04-23T01:30:00 2023-04-25T06:00:00 0 days 21:00:00 8.3
# G2H9-2023-11-05T09:00:00 2023-11-05T09:00:00 2023-11-05T21:00:00 2023-11-05T03:00:00 2023-11-06T09:00:00 0 days 12:00:00 7.3
# G2H9-2024-05-10T15:00:00 2024-05-10T15:00:00 2024-05-12T03:00:00 2024-05-09T21:00:00 2024-05-13T15:00:00 1 days 12:00:00 9.0
# G2H9-2024-06-28T09:00:00 2024-06-28T09:00:00 2024-06-28T18:00:00 2024-06-28T04:30:00 2024-06-29T03:00:00 0 days 09:00:00 7.7
# G2H9-2024-08-11T21:00:00 2024-08-11T21:00:00 2024-08-13T00:00:00 2024-08-11T07:30:00 2024-08-14T03:00:00 1 days 03:00:00 8.0
# G2H9-2024-10-10T15:00:00 2024-10-10T15:00:00 2024-10-11T12:00:00 2024-10-10T04:30:00 2024-10-12T09:00:00 0 days 21:00:00 8.7
# G2H9-2025-01-01T03:00:00 2025-01-01T03:00:00 2025-01-01T18:00:00 2024-12-31T19:30:00 2025-01-02T09:00:00 0 days 15:00:00 8.0
# G2H9-2025-04-16T09:00:00 2025-04-16T09:00:00 2025-04-16T21:00:00 2025-04-16T03:00:00 2025-04-17T09:00:00 0 days 12:00:00 7.7
# G2H9-2025-06-01T00:00:00 2025-06-01T00:00:00 2025-06-01T18:00:00 2025-05-31T15:00:00 2025-06-02T12:00:00 0 days 18:00:00 7.7