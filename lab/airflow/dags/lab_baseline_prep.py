
"""Week 11 Mini-Assignment: Scheduled Airflow ETL and ML pipeline.

Two datasets are cleaned in parallel, merged, stored in PostgreSQL,
read back for analysis, and published for the agent workflow.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import os
import shutil

import pandas as pd
from airflow.decorators import dag, task
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

REPO = Path("/repo")

CUSTOMERS = REPO / "datasets" / "airflow" / "customers.csv"
TRANSACTIONS = REPO / "datasets" / "airflow" / "transactions.csv"

RUNS = REPO / "artifacts" / "airflow" / "runs"
PUBLISH = REPO / "datasets" / "airflow" / "merged_latest.csv"

DATABASE_URL = os.getenv(
    "WEEK11_DATABASE_URL",
    "postgresql+psycopg2://airflow:airflow@airflow-db:5432/week11_ml",
)

TARGET = "is_fraud"
JOIN_KEY = "customer_id"
SUBSET_ROWS = 2000

DEFAULT_ARGS = {
    "retries": 2,
    "retry_delay": timedelta(seconds=5),
}


@dag(
    dag_id="lab_baseline_prep",
    description="Week 11: Parallel ETL, PostgreSQL and ML",
    start_date=datetime(2026, 11, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["student", "week11", "etl", "ml"],
)
def lab_baseline_prep():

    # -----------------------------------------------------------------------
    # 1. Validate datasets independently
    # -----------------------------------------------------------------------

    @task
    def validate_customers():
        if not CUSTOMERS.is_file():
            raise FileNotFoundError(CUSTOMERS)

        columns = set(pd.read_csv(CUSTOMERS, nrows=0).columns)

        required = {JOIN_KEY, "loan_status"}

        if not required.issubset(columns):
            raise ValueError("Missing required customer columns")

        return {"path": str(CUSTOMERS)}

    @task
    def validate_transactions():
        if not TRANSACTIONS.is_file():
            raise FileNotFoundError(TRANSACTIONS)

        columns = set(pd.read_csv(TRANSACTIONS, nrows=0).columns)

        required = {
            "transaction_id", JOIN_KEY,
            "timestamp", "amount", TARGET
        }

        if not required.issubset(columns):
            raise ValueError("Missing required transaction columns")

        return {"path": str(TRANSACTIONS)}

    # -----------------------------------------------------------------------
    # 2. Clean both datasets in parallel
    # -----------------------------------------------------------------------

    @task
    def clean_customers(meta: dict, **context):
        df = pd.read_csv(meta["path"])
        original_rows = len(df)

        df[JOIN_KEY] = df[JOIN_KEY].astype("string").str.strip()

        df = df.dropna(subset=[JOIN_KEY])
        df = df[df[JOIN_KEY] != ""]

        # Customer IDs must be unique.
        df = df.drop_duplicates(subset=[JOIN_KEY])

        # Retain customers with missing loan status.
        df["loan_status"] = (
            df["loan_status"].fillna("Unknown")
        )

        df["customer_since"] = pd.to_datetime(
            df["customer_since"], errors="coerce"
        )

        for col in ["age", "annual_income", "credit_score"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(
            subset=["age", "annual_income", "credit_score"]
        )

        # Basic domain checks.
        df = df[
            (df["age"] > 0)
            & (df["age"] <= 120)
            & (df["annual_income"] >= 0)
            & (df["credit_score"] >= 0)
        ]

        out = RUNS / context["ts_nodash"] / "customers_clean.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

        print(
            f"Customers: {original_rows} -> {len(df)} rows"
        )

        return {"path": str(out), "rows": len(df)}

    @task
    def clean_transactions(meta: dict, **context):
        df = pd.read_csv(meta["path"])
        original_rows = len(df)

        for col in ["transaction_id", JOIN_KEY]:
            df[col] = df[col].astype("string").str.strip()

        df = df.dropna(
            subset=["transaction_id", JOIN_KEY, TARGET]
        )

        df = df[
            (df["transaction_id"] != "")
            & (df[JOIN_KEY] != "")
        ]

        # Distinct transactions may belong to the same customer.
        df = df.drop_duplicates(subset=["transaction_id"])

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )

        df["amount"] = pd.to_numeric(
            df["amount"], errors="coerce"
        )

        df[TARGET] = pd.to_numeric(
            df[TARGET], errors="coerce"
        )

        df = df.dropna(
            subset=["timestamp", "amount", TARGET]
        )

        df = df[
            (df["amount"] >= 0)
            & (df[TARGET].isin([0, 1]))
        ]

        df[TARGET] = df[TARGET].astype(int)

        out = RUNS / context["ts_nodash"] / "transactions_clean.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

        print(
            f"Transactions: {original_rows} -> {len(df)} rows"
        )

        return {"path": str(out), "rows": len(df)}

    # -----------------------------------------------------------------------
    # 3. Merge cleaned data
    # -----------------------------------------------------------------------

    @task
    def merge_csvs(left_meta: dict, right_meta: dict, **context):
        customers = pd.read_csv(left_meta["path"])
        transactions = pd.read_csv(right_meta["path"])

        merged = transactions.merge(
            customers,
            on=JOIN_KEY,
            how="left",
            validate="many_to_one",
            indicator=True,
            suffixes=("", "_customer"),
        )

        unmatched = int(
            (merged["_merge"] == "left_only").sum()
        )

        print("Unmatched transactions:", unmatched)

        # Only fully matched records enter the ML dataset.
        merged = merged[merged["_merge"] == "both"].copy()
        merged = merged.drop(columns=["_merge"])

        if merged.empty:
            raise ValueError("No matched transactions")

        if merged["transaction_id"].duplicated().any():
            raise ValueError("Duplicate transaction IDs after merge")

        out = RUNS / context["ts_nodash"] / "merged.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out, index=False)

        return {
            "merged_path": str(out),
            "n_merged": len(merged),
            "unmatched": unmatched,
        }

    # -----------------------------------------------------------------------
    # 4. Load the full cleaned dataset into PostgreSQL
    # -----------------------------------------------------------------------

    @task
    def load_to_postgres(merge_meta: dict):
        df = pd.read_csv(merge_meta["merged_path"])

        engine = create_engine(DATABASE_URL)

        try:
            with engine.begin() as conn:
                df.to_sql(
                    "cleaned_transactions",
                    conn,
                    if_exists="replace",
                    index=False,
                    chunksize=500,
                    method="multi",
                )

                row_count = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM cleaned_transactions"
                    )
                ).scalar_one()
        finally:
            engine.dispose()

        if row_count != len(df):
            raise ValueError("PostgreSQL row count mismatch")

        print("Rows stored in PostgreSQL:", row_count)

        return {
            "table": "cleaned_transactions",
            "rows": int(row_count),
        }

    # -----------------------------------------------------------------------
    # 5. Read data back from PostgreSQL
    # -----------------------------------------------------------------------

    @task
    def read_from_postgres(db_meta: dict, **context):
        engine = create_engine(DATABASE_URL)

        try:
            with engine.connect() as conn:
                df = pd.read_sql_query(
                    text(
                        "SELECT * FROM cleaned_transactions"
                    ),
                    conn,
                )
        finally:
            engine.dispose()

        if len(df) != db_meta["rows"]:
            raise ValueError("Database read-back count mismatch")

        out = RUNS / context["ts_nodash"] / "database_read.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

        return {
            "merged_path": str(out),
            "n_rows": len(df),
        }

    # -----------------------------------------------------------------------
    # 6. Create a reproducible subset for classroom ML
    # -----------------------------------------------------------------------

    @task
    def make_subset(db_meta: dict, **context):
        df = pd.read_csv(db_meta["merged_path"])

        n = min(SUBSET_ROWS, len(df))

        subset = (
            df.sample(n=n, random_state=42)
            if n < len(df)
            else df
        )

        out = RUNS / context["ts_nodash"] / "subset.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(out, index=False)

        return {
            "merged_path": str(out),
            "n_subset": len(subset),
        }

    # -----------------------------------------------------------------------
    # 7. Profile numeric features and potential leakage
    # -----------------------------------------------------------------------

    @task
    def profile_light(subset_meta: dict):
        df = pd.read_csv(subset_meta["merged_path"])

        drops = [
            c for c in df.columns
            if c != TARGET
            and (
                c.endswith("_id")
                or c in {
                    "timestamp",
                    "fraud_probability",
                    "fraud_history",
                    "customer_risk",
                }
            )
        ]

        return {
            "recommended_drops": drops,
            "n_rows": len(df),
        }

    # -----------------------------------------------------------------------
    # 8. Train and evaluate a baseline model
    # -----------------------------------------------------------------------

    @task
    def train_baseline(subset_meta: dict, profile_meta: dict):
        df = pd.read_csv(subset_meta["merged_path"])

        drops = profile_meta["recommended_drops"]

        X = (
            df.drop(
                columns=[TARGET, *drops],
                errors="ignore"
            )
            .select_dtypes(include="number")
            .fillna(0)
        )

        y = df[TARGET].astype(int)

        if X.shape[1] == 0:
            raise ValueError("No numeric features available")

        if y.nunique() < 2 or y.value_counts().min() < 2:
            raise ValueError("Insufficient samples for both classes")

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # Scale numeric features for Logistic Regression.
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
            ),
        )

        clf.fit(X_train, y_train)

        predictions = clf.predict(X_val)

        accuracy = float(
            accuracy_score(y_val, predictions)
        )

        f1 = float(
            f1_score(
                y_val, predictions, zero_division=0
            )
        )

        print("Validation Accuracy:", accuracy)
        print("Validation F1:", f1)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_features": X.shape[1],
        }

    # -----------------------------------------------------------------------
    # 9. Publish dataset for LangGraph workflow
    # -----------------------------------------------------------------------

    @task
    def publish_merged(subset_meta: dict, baseline: dict):
        PUBLISH.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(
            subset_meta["merged_path"], PUBLISH
        )

        return {
            "published_latest": str(PUBLISH),
            "merged_dataset_path": subset_meta["merged_path"],
            "baseline_val_accuracy": baseline["accuracy"],
            "baseline_val_f1": baseline["f1"],
            "target_column": TARGET,
        }

    # -----------------------------------------------------------------------
    # DAG wiring
    # -----------------------------------------------------------------------

    # Parallel branches: customers and transactions.
    customers = clean_customers(validate_customers())
    transactions = clean_transactions(validate_transactions())

    merged = merge_csvs(customers, transactions)

    database = load_to_postgres(merged)
    db_data = read_from_postgres(database)

    demo = make_subset(db_data)
    prof = profile_light(demo)
    base = train_baseline(demo, prof)

    publish_merged(demo, base)


# Register the DAG.
lab_baseline_prep()
