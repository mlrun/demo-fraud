# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import mlrun
import os
from mlrun.datastore.datastore_profile import (
    DatastoreProfileRedis,
    DatastoreProfileKafkaStream,
    register_temporary_client_datastore_profile,
)


def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    """
    Creating the project for this demo. This function is expected to be called automatically when
    calling the function `mlrun.get_or_create_project`.

    :returns: a fully prepared project for this demo.
    """
    # Set the project git source:
    source = project.get_param(key="source")
    if not source:
        source = "git://github.com/mlrun/demo-fraud.git"
    print(f"Project Source: {source}")
    project.set_source(source=source, pull_at_runtime=True)

    if project.get_param("pre_load_data"):
        print("pre_load_data")

    # Refresh MLRun hub to the most up-to-date version:
    mlrun.get_run_db().get_hub_catalog(source_name="default", force_refresh=True)

    # Set the functions:

    project.set_function(
        func="src/get_vector.py",
        name="get-vector",
        handler="get_offline_features",
        kind="job",
    ).save()
    project.set_function(f"db://{project.name}/get-vector", name="get-vector")

    _set_function(
        project=project,
        func="hub://feature_selection",
        name="feature-selection",
        kind="job",
    )

    _set_function(
        project=project,
        func="hub://auto_trainer",
        name="train",
        kind="job",
    )
    _set_function(
        project=project,
        func="hub://auto_trainer",
        name="evaluate",
        kind="job",
    )

    _set_function(
        project=project,
        func="hub://v2_model_server",
        name="serving",
        kind="serving",
    )

    # Set the training workflow:
    project.set_workflow("main", "src/train_workflow.py", embed=True)

    # Set data source for feature store
    _set_datasource(project)

    # Save and return the project:
    project.save()
    return project


def _set_function(
    project: mlrun.projects.MlrunProject,
    func: str,
    name: str,
    kind: str,
    node_name: str = None,
    image: str = None,
):
    # Set the given function:
    with_repo = not func.startswith("hub://")
    mlrun_function = project.set_function(
        func=func,
        name=name,
        kind=kind,
        with_repo=with_repo,
        image=image,
    )
    if node_name:
        mlrun_function.with_node_selection(node_name=node_name)
    # Save:
    mlrun_function.save()

    project.set_function(f"db://{project.name}/{name}", name=name)


def _set_datasource(project: mlrun.projects.MlrunProject):
    # If running on community edition - use redis and kafka.
    tsdb_profile = mlrun.datastore.DatastoreProfileV3io(
        name="fraud-tsdb", v3io_access_key=mlrun.mlconf.get_v3io_access_key()
    )
    stream_profile = mlrun.datastore.DatastoreProfileV3io(
        name="fraud-stream", v3io_access_key=mlrun.mlconf.get_v3io_access_key()
    )

    if mlrun.mlconf.is_ce_mode():
        redis_uri = os.environ.get("REDIS_URI", None)
        redis_user = os.environ.get("REDIS_USER", "")
        redis_password = os.environ.get("REDIS_PASSWORD", "")
        kafka_host = os.environ.get(
            "KAFKA_SERVICE_HOST",
            f"kafka-stream.{os.environ.get('MLRUN_NAMESPACE', 'mlrun')}.svc.cluster.local",
        )
        kafka_port = os.environ.get("KAFKA_SERVICE_PORT", "9092")
        assert (
            redis_uri is not None
        ), "ERROR - When running on community edition, redis endpoint is required to run fraud-demo."
        assert (
            kafka_host is not None
        ), "ERROR - When running on community edition, kafka endpoint is required to run fraud-demo."

        project.set_secrets(
            {
                "REDIS_URI": redis_uri,
                "REDIS_USER": redis_user,
                "REDIS_PASSWORD": redis_password,
                "KAFKA_SERVICE_HOST": kafka_host,
                "KAFKA_SERVICE_PORT": kafka_port,
            }
        )

        # Redis datastore-profile
        tsdb_profile = DatastoreProfileRedis(
            name="fraud-tsdb",
            endpoint_url=redis_uri,
            username=redis_user,
            password=redis_password,
        )
        # Kafka datastore-profile
        stream_profile = DatastoreProfileKafkaStream(
            name="fraud-stream",
            brokers=f"{kafka_host}:{kafka_port}",
            topics=[],
        )
        project.params["online_target"] = "ds://fraud-tsdb"
        for fs in ["transactions", "events", "labels"]:
            project.params[fs] = os.path.join(mlrun.mlconf.artifact_path, fs + ".pq")

        # dealing with kafka
        kafka_uri = f"{kafka_host}:{kafka_port}"
        project.params["transaction_stream"] = (
            f"kafka://{kafka_uri}?topic=transactions"
        )
        project.params["events_stream"] = f"kafka://{kafka_uri}?topic=events"

        register_temporary_client_datastore_profile(tsdb_profile)
        register_temporary_client_datastore_profile(stream_profile)

    else:
        project.params["transaction_stream"] = (
            f"v3io:///projects/{project.name}/streams/transaction"
        )
        project.params["events_stream"] = (
            f"v3io:///projects/{project.name}/streams/events"
        )

    project.register_datastore_profile(tsdb_profile)
    project.register_datastore_profile(stream_profile)

    for key, value in project.params.items():
        project.params[key] = value.replace("{{run.project}}", project.name)
