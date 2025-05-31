# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import mlrun
from kfp import dsl
import os

from mlrun.model import HyperParamOptions
from mlrun.datastore.datastore_profile import DatastoreProfileKafkaSource, DatastoreProfileTDEngine


# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(
    name="Fraud Detection Pipeline",
    description="Detecting fraud from a transactions dataset",
)
def pipeline(vector_name="transactions-fraud", features=[], label_column="is_error"):
    """
    This pipeline will train a model to detect fraud from a transactions dataset.
    :param vector_name: The name of the feature vector to use
    :param features: A list of features to use
    :param label_column: The name of the label column

    :returns: None
    """
    
    # Get the project
    project = mlrun.get_current_project()  

    # Get FeatureVector
    get_vector_func = project.get_function("get-vector")
    get_vector_run = project.run_function(
        get_vector_func,
        name="get-vector",
        params={
            "feature_vector": vector_name,
            "features": features,
            "label_feature": label_column,
            "target": {"name": "parquet", "kind": "parquet"},
            "update_stats": True,
        },
        outputs = [
            "feature_vector"
        ]
    )
    
    # Feature selection
    feature_selection_func = project.get_function("feature-selection")
    feature_selection_run = project.run_function(
        feature_selection_func,
        name="feature-selection",
        params={
            "output_vector_name": "short",
            "label_column": project.get_param("label_column", "label"),
            "k": 18,
            "min_votes": 2,
            "ignore_type_errors": True,
        },
        inputs={
            "df_artifact": project.get_artifact_uri(vector_name, "feature-vector")
        },
        outputs=[
            "feature_scores",
            "selected_features_count",
            "top_features_vector",
            "selected_features",
        ],
    ).after(get_vector_run)

    # train with hyper-paremeters
    train_func = project.get_function("train")
    train_run = project.run_function(
        train_func,
        name="train",
        handler="train",
        params={
            "sample": -1,
            "label_column": project.get_param("label_column", "label"),
            "test_size": 0.10,
        },
        hyperparams={
            "model_name": [
                "transaction_fraud_rf",
                "transaction_fraud_xgboost",
                "transaction_fraud_adaboost",
            ],
            "model_class": [
                "sklearn.ensemble.RandomForestClassifier",
                "sklearn.linear_model.LogisticRegression",
                "sklearn.ensemble.AdaBoostClassifier",
            ],
        },
        hyper_param_options=HyperParamOptions(
            strategy="list", selector="max.accuracy"
        ),
        inputs={"dataset": feature_selection_run.outputs["top_features_vector"]},
        outputs=["model", "test_set"],
    ).after(feature_selection_run)

    # test and visualize your model
    test_func = project.get_function("evaluate")
    test_run = mlrun.run_function(
        test_func,
        name="evaluate",
        handler="evaluate",
        params={
            "label_columns": project.get_param("label_column", "label"),
            "model": train_run.outputs["model"],
            "drop_columns": project.get_param("label_column", "label"),
        },
        inputs={"dataset": train_run.outputs["test_set"]},
    ).after(train_run)

    # Create a serverless function from the hub, add a feature enrichment router
    # This will enrich and impute the request with data from the feature vector
    serving_func = project.get_function("serving")
    serving_func.set_topology(
        "router",
        mlrun.serving.routers.EnrichmentModelRouter(
            feature_vector_uri="short", impute_policy={"*": "$mean"}
        ),
        exist_ok=True,
    )

    # Enable model monitoring
    serving_func.set_tracking()

    if mlrun.mlconf.is_ce_mode():
        # Use default service
        tsdb_profile = DatastoreProfileTDEngine(name="fraud-monitoring-tsdb",
                                        user='root',
                                        password='taosdata',
                                        host=f"tdengine.{os.environ.get('MLRUN_NAMESPACE', 'mlrun')}.svc.cluster.local",
                                        port='6041')
        project.register_datastore_profile(tsdb_profile)

        kafka_host = os.environ.get('KAFKA_SERVICE_HOST', f"kafka-stream.{os.environ.get('MLRUN_NAMESPACE', 'mlrun')}.svc.cluster.local")
        kafka_port = os.environ.get('KAFKA_SERVICE_PORT', '9092')

        stream_profile = DatastoreProfileKafkaSource(
            name='fraud-monitoring-stream',
            brokers=f"{kafka_host}:{kafka_port}",
            topics=[],
        )

        project.set_model_monitoring_credentials(
            tsdb_profile_name=tsdb_profile.name,
            stream_profile_name=stream_profile.name,
            replace_creds=True
        )

    else:
        project.set_model_monitoring_credentials(
            tsdb_profile_name='fraud-tsdb',
            stream_profile_name='fraud-stream',
            replace_creds=True
        )

    serving_func.save()
    # deploy the model server, pass a list of trained models to serve
    deploy = project.deploy_function(
        serving_func,
        models=[{"key": "fraud", "model_path": train_run.outputs["model"]}],
    ).after(train_run)
