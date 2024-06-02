# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# Helper functions to adjust the timestamps of our data
# while keeping the order of the selected events and
# the relative distance from one event to the other
import pandas as pd


def date_adjustment(
    sample: pd.Timestamp,
    data_max: pd.Timestamp,
    new_max: pd.Timestamp,
    old_data_period: pd.Timedelta,
    new_data_period: pd.Timedelta,
) -> pd.Timestamp:
    """
    Adjust a specific sample's date according to the original and new time periods

    :param sample: The sample's timestamp
    :param data_max: The original data's max timestamp
    :param new_max: The new data's max timestamp
    :param old_data_period: The original data's time period
    :param new_data_period: The new data's time period

    :returns: The adjusted timestamp
    """
    sample_dates_scale = (data_max - sample) / old_data_period
    sample_delta = new_data_period * sample_dates_scale
    new_sample_ts = new_max - sample_delta
    return new_sample_ts


def adjust_data_timespan(
    dataframe: pd.DataFrame,
    timestamp_col: str = "timestamp",
    new_period: str = "2d",
    new_max_date_str: str = "now",
):
    """
    Adjust the dataframe timestamps to the new time period

    :param dataframe: The dataframe to adjust
    :param timestamp_col: The timestamp column name
    :param new_period: The new time period
    :param new_max_date_str: The new max date

    :returns: The adjusted dataframe
    """
    # Calculate old time period
    data_min = dataframe.timestamp.min()
    data_max = dataframe.timestamp.max()
    old_data_period = data_max - data_min

    # Set new time period
    new_time_period = pd.Timedelta(new_period)
    new_max = pd.Timestamp(new_max_date_str)
    new_min = new_max - new_time_period
    new_data_period = new_max - new_min

    # Apply the timestamp change
    df = dataframe.copy()
    df[timestamp_col] = df[timestamp_col].apply(
        lambda x: date_adjustment(
            x, data_max, new_max, old_data_period, new_data_period
        )
    )
    df.sort_values(by="timestamp", axis=0, inplace=True)
    return df
