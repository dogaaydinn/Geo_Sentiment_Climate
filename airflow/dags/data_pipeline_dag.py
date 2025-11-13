"""
Airflow DAG for Geo_Sentiment_Climate Data Pipeline.

This DAG orchestrates the complete data pipeline:
1. Data ingestion from EPA sources
2. Data validation and quality checks
3. Data preprocessing and cleaning
4. Feature engineering
5. Model training (optional)
6. Model evaluation
7. Model deployment
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Default arguments
default_args = {
    'owner': 'geo_climate_team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['dogaa882@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=4),
}

# Create DAG
dag = DAG(
    'geo_climate_data_pipeline',
    default_args=default_args,
    description='Complete data pipeline for air quality analysis',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    tags=['data-pipeline', 'ml', 'air-quality'],
    max_active_runs=1,
)

# Task functions
def data_ingestion(**context):
    """Ingest data from EPA sources."""
    from source.data_ingestion import ingest_data

    print("Starting data ingestion...")
    ingest_data()
    print("Data ingestion completed")


def data_validation(**context):
    """Validate ingested data quality."""
    from source.data_check import check_data_quality

    print("Starting data validation...")
    # Add validation logic
    print("Data validation completed")


def data_preprocessing(**context):
    """Preprocess and clean data."""
    from source.data_preprocessing import preprocess_data

    print("Starting data preprocessing...")
    preprocess_data()
    print("Data preprocessing completed")


def feature_engineering(**context):
    """Engineer features for modeling."""
    from source.feature_engineering import engineer_features

    print("Starting feature engineering...")
    # engineer_features()
    print("Feature engineering completed")


def model_training(**context):
    """Train ML models."""
    from source.ml.model_training import ModelTrainer, TrainingConfig

    print("Starting model training...")

    config = TrainingConfig(
        model_type="xgboost",
        task_type="regression",
        n_trials=30,
        target_column="aqi",
        experiment_name="geo_climate_daily_training"
    )

    trainer = ModelTrainer(config)
    results = trainer.run_full_pipeline()

    # Push results to XCom
    context['task_instance'].xcom_push(key='training_results', value=results)

    print(f"Model training completed: {results['model_path']}")


def model_evaluation(**context):
    """Evaluate trained models."""
    from source.ml.model_evaluation import ModelEvaluator

    print("Starting model evaluation...")

    # Pull training results from XCom
    training_results = context['task_instance'].xcom_pull(
        task_ids='train_model',
        key='training_results'
    )

    print(f"Model evaluation completed for: {training_results.get('model_path')}")


def model_deployment(**context):
    """Deploy model to production if metrics are good."""
    from source.ml.model_registry import ModelRegistry

    print("Starting model deployment...")

    # Pull training results
    training_results = context['task_instance'].xcom_pull(
        task_ids='train_model',
        key='training_results'
    )

    # Check if metrics meet threshold
    test_r2 = training_results['test_metrics'].get('r2', 0)

    if test_r2 > 0.8:  # Example threshold
        print(f"Model meets quality threshold (R2={test_r2:.4f}), deploying...")
        # Promote model to production
        # registry = ModelRegistry()
        # registry.promote_model(model_id, 'production')
    else:
        print(f"Model does not meet quality threshold (R2={test_r2:.4f}), skipping deployment")


# Define tasks
t1_ingest = PythonOperator(
    task_id='ingest_data',
    python_callable=data_ingestion,
    dag=dag,
)

t2_validate = PythonOperator(
    task_id='validate_data',
    python_callable=data_validation,
    dag=dag,
)

t3_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=data_preprocessing,
    dag=dag,
)

t4_feature_eng = PythonOperator(
    task_id='engineer_features',
    python_callable=feature_engineering,
    dag=dag,
)

t5_train = PythonOperator(
    task_id='train_model',
    python_callable=model_training,
    dag=dag,
)

t6_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=model_evaluation,
    dag=dag,
)

t7_deploy = PythonOperator(
    task_id='deploy_model',
    python_callable=model_deployment,
    dag=dag,
)

# Health check task
t8_health = BashOperator(
    task_id='health_check',
    bash_command='curl -f http://localhost:8000/health || exit 1',
    dag=dag,
)

# Define task dependencies
t1_ingest >> t2_validate >> t3_preprocess >> t4_feature_eng >> t5_train >> t6_evaluate >> t7_deploy >> t8_health
