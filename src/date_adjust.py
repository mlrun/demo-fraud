# Helper functions to adjust the timestamps of our data
# while keeping the order of the selected events and
# the relative distance from one event to the other


def date_adjustment(sample, data_max, new_max, old_data_period, new_data_period):
    """
    Adjust a specific sample's date according to the original and new time periods
    """
    sample_dates_scale = (data_max - sample) / old_data_period
    sample_delta = new_data_period * sample_dates_scale
    new_sample_ts = new_max - sample_delta
    return new_sample_ts


def adjust_data_timespan(
    dataframe,
    timestamp_col="timestamp",
    new_period="2d",
    new_max_date_str="now",
):
    """
    Adjust the dataframe timestamps to the new time period
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
    return df
