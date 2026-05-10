import os
import tempfile
from datetime import datetime, timedelta

import boto3
import pandas as pd
from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago

from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

S3_BUCKET       = "your-s3-bucket"
S3_PREFIX       = "drift-reports"
DRIFT_THRESHOLD = 0.3   # retrain if >30% of columns have drifted
DATE_COL        = "Date"

PATH = "your/data/path/"
RTE  = "your_file.csv"

DEFAULT_ARGS = {
    "owner":             "airflow",
    "retries":           1,
    "retry_delay":       timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_rte(PATH, rte):
    """
    Loads the rte data from a csv.
    """
    url = f"{PATH}{rte}"
    data_whole = pd.read_csv(url)
    return data_whole


def split_dataframe_for_evidently(df: pd.DataFrame, date_col: str = DATE_COL):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    today           = datetime.now()
    current_start   = today - timedelta(days=31)
    reference_end   = current_start
    reference_start = reference_end - timedelta(days=365)

    current_data   = df[df[date_col] >  current_start]
    reference_data = df[(df[date_col] >  reference_start) &
                        (df[date_col] <= reference_end)]
    past_data      = df[df[date_col] <= reference_start]

    return current_data, reference_data, past_data


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def task_run_drift_analysis(ti=None):
    """
    Runs the Evidently drift report + test suite.
    Pushes the HTML report path and the should_retrain flag to XCom.
    """
    df = load_rte(PATH, RTE)
    current_data, reference_data, _ = split_dataframe_for_evidently(df)

    run_stamp = datetime.now().strftime("%Y_%m")

    # --- Full report (HTML artifact) ---
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference_data, current_data=current_data)

    report_filename = f"drift_report_{run_stamp}.html"
    report_path     = os.path.join(tempfile.gettempdir(), report_filename)
    report.save_html(report_path)

    # --- Test suite (retrain decision) ---
    test_suite = TestSuite(tests=[
        TestShareOfDriftedColumns(lte=DRIFT_THRESHOLD),
    ])
    test_suite.run(reference_data=reference_data, current_data=current_data)

    results        = test_suite.as_dict()
    should_retrain = not results["summary"]["all_passed"]
    drift_share    = (
        results["summary"]["by_status"].get("FAIL", 0)
        / results["summary"]["total"]
    )

    print(f"Drift share     : {drift_share:.1%}")
    print(f"Should retrain  : {should_retrain}")

    ti.xcom_push(key="report_path",     value=report_path)
    ti.xcom_push(key="report_filename", value=report_filename)
    ti.xcom_push(key="should_retrain",  value=should_retrain)
    ti.xcom_push(key="drift_share",     value=drift_share)


def task_branch(ti=None):
    """Routes to retraining or skip based on the drift decision."""
    should_retrain = ti.xcom_pull(
        task_ids="run_drift_analysis", key="should_retrain"
    )
    return "retrain_model" if should_retrain else "skip_retraining"



def task_skip_retraining(ti=None):
    drift_share = ti.xcom_pull(
        task_ids="run_drift_analysis", key="drift_share"
    )
    print(f"No retraining needed. Drift share: {drift_share:.1%}")


def task_save_report(ti=None):
    """
    Saves the HTML drift report (which embeds the full metrics + decision)
    to S3, then optionally logs to Evidently Cloud.
    """
    report_path     = ti.xcom_pull(task_ids="run_drift_analysis", key="report_path")
    report_filename = ti.xcom_pull(task_ids="run_drift_analysis", key="report_filename")
    should_retrain  = ti.xcom_pull(task_ids="run_drift_analysis", key="should_retrain")
    drift_share     = ti.xcom_pull(task_ids="run_drift_analysis", key="drift_share")

    run_stamp = datetime.now().strftime("%Y_%m")
    s3_key    = f"{S3_PREFIX}/{run_stamp}/{report_filename}"

    # --- Save to S3 ---
    s3 = boto3.client("s3")
    s3.upload_file(report_path, S3_BUCKET, s3_key)

    # Attach retraining decision as S3 object metadata
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": s3_key},
        Key=s3_key,
        MetadataDirective="REPLACE",
        Metadata={
            "drift_share":    str(round(drift_share, 4)),
            "should_retrain": str(should_retrain),
            "run_date":       run_stamp,
        },
    )
    print(f"Report saved to s3://{S3_BUCKET}/{s3_key}")

    # --- (Optional) Log to Evidently Cloud ---
    # from evidently.ui.workspace.cloud import CloudWorkspace
    #
    # ws = CloudWorkspace(
    #     token="YOUR_EVIDENTLY_CLOUD_TOKEN",   # store in Airflow Variables or Secrets
    #     url="https://app.evidently.cloud",
    # )
    # project = ws.get_project("YOUR_PROJECT_ID")
    #
    # report.upload(project)        # uploads the Report object directly
    # test_suite.upload(project)    # uploads the TestSuite results


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="model_drift_monitoring",
    description="Monthly drift analysis with conditional retraining and S3 archiving",
    default_args=DEFAULT_ARGS,
    start_date=days_ago(1),
    schedule_interval="0 6 1 * *",  # 06:00 on the 1st of every month
    catchup=False,
    tags=["monitoring", "drift", "evidently"],
) as dag:

    run_drift_analysis = PythonOperator(
        task_id="run_drift_analysis",
        python_callable=task_run_drift_analysis,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=task_branch,
    )

    retrain_model = TriggerDagRunOperator(
        task_id="retrain_model",
        trigger_dag_id="your_retraining_dag_id",   # replace with your retraining DAG's dag_id
        wait_for_completion=False,                  # set True to wait for retraining to finish
        conf={"triggered_by": "drift_monitoring"},  # passed to the retraining DAG via dag_run.conf
    )

    skip_retraining = PythonOperator(
        task_id="skip_retraining",
        python_callable=task_skip_retraining,
    )

    save_report = PythonOperator(
        task_id="save_report",
        python_callable=task_save_report,
        trigger_rule="none_skipped",  # runs regardless of which branch was taken
    )

    run_drift_analysis >> branch >> [retrain_model, skip_retraining] >> save_report