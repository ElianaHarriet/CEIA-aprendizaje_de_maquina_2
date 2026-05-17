import datetime

from airflow.decorators import dag, task

default_args = {
    "owner": "CEIA - FIUBA",
    "depends_on_past": False,
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


@dag(
    dag_id="train_movielens",
    description="DAG para entrenamiento del modelo.",
    tags=["train", "MovieLens"],
    default_args=default_args,
    catchup=False,
)
def train_movielens():
    @task(task_id="train_model")
    def train_task():
        print("Training...")
        return None

    train_task()


dag = train_movielens()
