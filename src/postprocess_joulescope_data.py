#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
import sys
import pandas as pd
from pyjls import Reader

def read_joulescope_data(filepath, start_time, stop_time, signal_id=3):

    with Reader(filepath) as reader:
        # signal = reader.signal_lookup('current')
        # data = reader.fsr(signal.signal_id, 0, 1000) 
        # breakpoint()
        # print(f'{type(data)} | {data.shape} | {np.mean(data)}')
        # print(reader.signals)

        # Find first timestamp
        start_sample_id = reader.timestamp_to_sample_id(3, start_time)
        stop_sample_id = reader.timestamp_to_sample_id(3, stop_time)
        signal = reader.fsr(signal_id, start_sample_id, stop_sample_id - start_sample_id)
        print(signal)

    return 0

def string2utc(time_string):
    # Define the format string
    # %Y-%m-%d %H:%M:%S.%f%z will handle the date, time, microseconds, and timezone
    format_string = "%Y-%m-%d %H:%M:%S.%f%z"

    # Truncate to 6 digits of microseconds
    truncated_time_string = time_string[:26] + time_string[29:]

    # Parse the truncated string
    dt = datetime.strptime(truncated_time_string, format_string)

    # Convert the datetime object to a Unix timestamp
    unix_timestamp = dt.timestamp()

    return unix_timestamp


def read_jsl_data(filepath):
    df = pd.read_csv(filepath)

    # Get energy consumption for each inference
    for index, row in df.iterrows():

        # Read .jsl file
        # start_time = string2utc(row["created_at"])*1e6
        # stop_time = string2utc(row["stopped_at"])*1e6
        # energy_consumption_kwh = read_joulescope_data(joulescope_data_filepath, start_time, stop_time)

        continue

    return None

def read_joulescope_statistics_data(filepath, start_time, stop_time):

    pass
    


if __name__ == '__main__':



    joulescope_statistics_folderpath = sys.argv[1]
    llm_data_filepaths = sys.argv[2:]

    # Find all joulescope data files
    joulescope_data_filepaths = []
    for root, dirs, files in os.walk(joulescope_statistics_folderpath):
        for file in files:
            if file.endswith(".csv"):
                joulescope_data_filepaths.append(os.path.join(root, file))

    # Find timestamps from the joulescope data filepaths, which are in this format:
    #   20250122_093117-JS110-000436.csv
    joulescope_timestamps = []
    for joulescope_data_filepath in joulescope_data_filepaths:
        timestamp = joulescope_data_filepath.split("-")[0]
        # Turn the timestamps into datetime objects
        joulescope_timestamps.append(datetime.strptime(timestamp, "%Y%m%d_%H%M%S"))

    for llm_data_filepath in llm_data_filepaths:

        df = pd.read_csv(llm_data_filepath)

        # Find which joulescope data file to use
        # Use the first timestamp of the llm data to find the corresponding joulescope data, which is the closest one back in time
        llm_timestamp = datetime.strptime(df["created_at"][0], "%Y-%m-%d %H:%M:%S.%f%z")
        closest_joulescope_timestamp = min(joulescope_timestamps, key=lambda x: abs(x - llm_timestamp))

        # Get energy consumption for each inference
        # for index, row in df.iterrows():
