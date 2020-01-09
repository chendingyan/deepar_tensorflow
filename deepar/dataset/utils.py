import pandas as pd
import logging

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

def robust_timedelta(offset=1, freq="D"):
    """ robust pandas timedelta wrapper to bypass 'Y' and 'M' args
    
    Keyword Arguments:
        freq {str} -- frequency of timedelta (default: {'D'})
        offset {int} -- multiplier of frequency (default: {1})
    
    Returns:
        pd Timedelta -- timedelta object 
    """

    if freq == "YS":
        freq = "D"
        offset *= 365
    elif freq == "MS" or freq == "M":
        freq = "D"
        offset *= 31
    else:
        freq = freq
    return pd.Timedelta(offset, freq)

def robust_reindex(df, min_date, max_date, freq="D"):
    """ reindex dataframe with tolerance and method (interpolation) kwargs
        handles special case weekly PHEM data robustly (where we want to allow 
        tolerance in re-indexing)
    
    Arguments:
        df {pd df} -- pandas df to reindex
        min_date {pd DateTime} -- minimum date of reindexing
        max_date {pd DateTime} -- maximum date of reindexing
    
    Keyword Arguments:
        freq {str} -- frequency of reindexing (default: {'D'})
    
    Returns:
        pd df -- robustly reindexed dataframe
    """

    # special case for PHEM weekly data, we allow tolerance in reindexing so weekly
    # data can be aligned correctly
    if freq == "W":
        tolerance = "1" + freq
        method = "nearest"
    else:
        tolerance = None
        method = None

    return df.reindex(
        pd.date_range(min_date, max_date, freq=freq, normalize=True),
        tolerance=tolerance,
        method=method,
    )

