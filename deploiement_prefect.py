
"""
deploiement_prefect.py
Crée les déploiements planifiés des flows (1 fois par jour) avec Prefect 3.x.
"""
from pipeline_prefect import flow_all, flow_train, flow_evaluate

if __name__ == "__main__":
    flow_all.serve(
        name="ml-pipeline-all",
        cron="0 2 * * *",
        tags=["mlops", "full-pipeline"],
    )
    flow_train.serve(
        name="ml-pipeline-train",
        cron="0 3 * * *",
        tags=["mlops", "training"],
    )
    flow_evaluate.serve(
        name="ml-pipeline-evaluate",
        cron="0 4 * * *",
        tags=["mlops", "evaluation"],
    )
