#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from datetime import datetime

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


if __name__ == '__main__':

    # joulescope_statistics_folderpath = sys.argv[1]
    # llm_data_filepaths = sys.argv[2:]

    # Parse arguments
    parser = argparse.ArgumentParser(description="Postprocess joulescope data")
    parser.add_argument("joulescope_statistics_folderpath", type=str, help="Path to the folder containing the joulescope statistics")
    parser.add_argument("llm_data_filepaths", type=str, nargs="+", help="Path to the LLM data files")
    parser.add_argument("--output_folderpath", type=str, default="./postprocessed_data", help="Path to the output folder")
    parser.add_argument("--idle_power", type=float, default=0.0, help="Idle power consumption in Watts")

    args = parser.parse_args()


    # Create the output folder if it does not exist
    if not os.path.exists(args.output_folderpath):
        os.makedirs(args.output_folderpath)

    # Find all joulescope data files
    joulescope_data_filepaths = []
    for root, dirs, files in os.walk(args.joulescope_statistics_folderpath):
        for file in files:
            if file.endswith(".csv"):
                joulescope_data_filepaths.append(os.path.join(root, file))

    # Find timestamps from the joulescope data filepaths, which are in this format:
    #   20250122_093117-JS110-000436.csv
    joulescope_filepath_dict = {}
    for joulescope_data_filepath in joulescope_data_filepaths:
        basename = os.path.basename(joulescope_data_filepath)
        timestamp = basename.split("-")[0]
        # Turn the timestamps into datetime objects
        datetime_timestamp = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        joulescope_filepath_dict[datetime_timestamp] = joulescope_data_filepath
    
    report = ""

    for llm_data_filepath in args.llm_data_filepaths:

        df = pd.read_csv(llm_data_filepath)
        df["energy_consumption_joules"] = None
        df["energy_consumption_joules_without_idle_subtracted"] = None
        df["tokens_per_second"] = df["eval_count"] / (df["eval_duration"] * 10**-9)

        # Find which joulescope data file to use
        # Use the first timestamp of the llm data to find the corresponding joulescope data, which is the closest one back in time. Only back in time, not forward.
        first_llm_timestamp = df["created_at"][0]
        llm_timestamp = pd.to_datetime(first_llm_timestamp, format="%Y-%m-%d %H:%M:%S.%f%z").replace(tzinfo=None)
        # Find all joulescope_timestamps preceeding llm_timestamp
        preceeding_joulescope_timestamps = [timestamp for timestamp in joulescope_filepath_dict.keys() if timestamp < llm_timestamp]
        # Find the closest one
        closest_joulescope_timestamp = max(preceeding_joulescope_timestamps)
        # Find the corresponding joulescope data filepath
        current_joulescope_data_filepath = joulescope_filepath_dict[closest_joulescope_timestamp]

        print(llm_data_filepath)
        print(current_joulescope_data_filepath)

        joulescope_data = pd.read_csv(current_joulescope_data_filepath)
        joulescope_data["timestamp"] = closest_joulescope_timestamp + pd.to_timedelta(joulescope_data["#time"], unit="s")

        # Get energy consumption for each inference
        for index, row in df.iterrows():
            start_time = pd.to_datetime(row["created_at"]). replace(tzinfo=None)
            stop_time = pd.to_datetime(row["stopped_at"]). replace(tzinfo=None)
            duration = stop_time - start_time

            if pd.isnull(stop_time):
                try:
                    stop_time = pd.to_datetime(df.loc[index+1, "created_at"]). replace(tzinfo=None)
                except:
                    continue


            closest_idx_start = (joulescope_data["timestamp"] - start_time).abs().idxmin()
            closest_idx_stop = (joulescope_data["timestamp"] - stop_time).abs().idxmin()

            energy_consumption_start = joulescope_data.loc[closest_idx_start, "energy"]
            energy_consumption_stop = joulescope_data.loc[closest_idx_stop, "energy"]
            energy_consumption = energy_consumption_stop - energy_consumption_start

            df.loc[index, "energy_consumption_joules_without_idle_subtracted"] = energy_consumption

            # Subtract idle power
            energy_consumption -= args.idle_power * duration.total_seconds()

            if energy_consumption < 0:
                energy_consumption = 0.0
                report += f"Negative energy consumption for {llm_data_filepath} at index {index}, timestamp {start_time} to {stop_time}\n"

            df.loc[index, "energy_consumption_joules"] = energy_consumption


        df["idle_power_watts"] = args.idle_power

        # df.to_csv(llm_data_filepath.replace("llm_responses", "llm_responses_with_energy_consumption"), index=False)
        output_filepath = os.path.join(args.output_folderpath, os.path.basename(llm_data_filepath))
        df.to_csv(output_filepath, index=False)
        print(f"Saved {llm_data_filepath}")
    
    with open(os.path.join(args.output_folderpath, "report.txt"), "w") as f:
        f.write(report)