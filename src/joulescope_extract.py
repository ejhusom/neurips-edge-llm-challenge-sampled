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

def read_joulescope_statistics_data(filepath, start_time, stop_time):

    df = pd.read_csv(filepath)



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

if __name__ == '__main__':

    llm_data_filepath = sys.argv[1]
    joulescope_data_filepath = sys.argv[2]
    df = pd.read_csv(llm_data_filepath)

    # Get energy consumption for each inference
    for index, row in df.iterrows():
        print(row["created_at"])
        print(row["stopped_at"])

        # Read .jsl file
        # start_time = string2utc(row["created_at"])*1e6
        # stop_time = string2utc(row["stopped_at"])*1e6
        # energy_consumption_kwh = read_joulescope_data(joulescope_data_filepath, start_time, stop_time)

        # Read .csv file 
        start_time = string2utc(row["created_at"])
        stop_time = string2utc(row["stopped_at"])
        energy_consumption_kwh = read_joulescope_statistics_data(joulescope_data_filepath, start_time, stop_time)
        df.loc[index, "energy_consumption_kwh"] = energy_consumption_kwh
