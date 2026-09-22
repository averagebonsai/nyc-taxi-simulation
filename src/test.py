from pathlib import Path

import pandas as pd


JANUARY_2026_TUESDAYS_AND_WEDNESDAYS = frozenset(
    {
        "2026-01-06",
        "2026-01-07",
        "2026-01-13",
        "2026-01-14",
        "2026-01-20",
        "2026-01-21",
        "2026-01-27",
        "2026-01-28",
    }
)
TIMESTAMP_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def sum_january_2026_2am_ridership(input_path: str | Path) -> int:
    """Return ridership from 2:00 to 3:00 AM on the specified January dates."""
    total_ridership = 0

    for chunk in pd.read_csv(
        input_path,
        usecols=["transit_timestamp", "ridership"],
        chunksize=250_000,
    ):
        timestamps = pd.to_datetime(
            chunk["transit_timestamp"], format=TIMESTAMP_FORMAT, errors="raise"
        )
        is_target_ride = (
            timestamps.dt.strftime("%Y-%m-%d").isin(
                JANUARY_2026_TUESDAYS_AND_WEDNESDAYS
            )
            & timestamps.dt.hour.eq(2)
        )
        total_ridership += pd.to_numeric(
            chunk.loc[is_target_ride, "ridership"], errors="coerce"
        ).sum()

    return int(total_ridership)


def find_rows(input):
    """
    Calculates base prices ("edge weights") for each (start, destination) pair / each edge. 
    Uses Tuesday 2am - 4am as a baseline. 
    Here, base price for each (start, destination) pair is just the average price across all rides in this region.
    """
    df = pd.read_parquet(input)
    relevant_df = df[(df['day_of_week'].isin([2,3])) & (df['request_hour'].isin([2,3]))]
    return relevant_df.shape


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    ridership_path = (
        SCRIPT_DIR.parent
        / "data"
        / "MTA_Subway_Hourly_Ridership__Beginning_2025_20260920.csv"
    )
    print(sum_january_2026_2am_ridership(ridership_path)) #43625

    taxi_path = (
        SCRIPT_DIR.parent
        / "data"
        / "new.parquet"
    )

    print(find_rows(taxi_path)) #82742
