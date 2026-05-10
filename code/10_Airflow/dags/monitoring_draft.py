
import pandas as pd
from datetime import datetime, timedelta

from evidently.ui.workspace import CloudWorkspace


from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot

PATH = "https://renergies99-bucket.s3.eu-west-3.amazonaws.com/public/"
rte = "prod/eCO2mix_RTE_Annuel-Definitif.csv"

def load_rte(PATH,rte):
    """
    Loads the rte data from a csv
    """
    url = f"{PATH}{rte}"
    data_whole = pd.read_csv(url)
    return data_whole

def split_data(data_whole):
    """
    Splits a DataFrame into three time-based subsets for EvidentlyAI drift monitoring.

    Args:
        data_whole: Input DataFrame with a date column.

    Returns:
        current_data:   Last 31 days  → the "production" data to monitor
        reference_data: The year prior to that window → the baseline
        past_data:      Everything older → historical archive
    """
    today = datetime.now()
    current_start    = today - timedelta(days=31)
    reference_end    = current_start
    reference_start  = reference_end - timedelta(days=365)

    current_data   = data_whole[data_whole["Date"] >  current_start]
    reference_data = data_whole[(data_whole["Date"] >  reference_start) & 
                        (data_whole["Date"] <= reference_end)]
    past_data      = data_whole[data_whole["Date"] <= reference_start]

    print(f"Latest date      : {today.date()}")
    print(f"Current window   : {(current_start).date()} → {today.date()}  ({len(current_data)} rows)")
    print(f"Reference window : {reference_start.date()} → {reference_end.date()}  ({len(reference_data)} rows)")
    print(f"Past data        : up to {reference_start.date()}  ({len(past_data)} rows)")

    return current_data, reference_data, past_data

             

past = pd.read_csv(f"{PATH}/02-sample-evidently-app/data/employee_performance_2022.csv")
reference = pd.read_csv(f"{PATH}/02-sample-evidently-app/data/employee_performance_2023.csv")
current = pd.read_csv(f"{PATH}/02-sample-evidently-app/data/employee_performance_2024.csv")

parsed_past = Dataset.from_pandas(
    past,
    data_definition=DataDefinition()
)

parsed_reference = Dataset.from_pandas(
    reference,
    data_definition=DataDefinition()
)

parsed_current = Dataset.from_pandas(
    current,
    data_definition=DataDefinition()
)

