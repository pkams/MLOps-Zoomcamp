from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow_location="score.py",
    name="ride_duration_prediction",
    schedule=CronSchedule(cron="0 3 2 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"],
    parameters={
        "taxy_type":"green",
        "run_id": "b5eb4a60a4934bba8ccd86bf71d3f9fa"
    },
    flow_storage = "7caf68ca-235b-42c9-ae39-c370f6c7b6d9"
)