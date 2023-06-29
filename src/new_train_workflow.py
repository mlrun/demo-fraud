# Copyright 2019 Iguazio
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

from mlrun.model import HyperParamOptions


# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(name="Fraud Detection Pipeline",
    description="Detecting fraud from a transactions dataset",)
def pipeline(
        vector_name="transactions-fraud",
        features=[],
        label_column="is_error",
):
    project = mlrun.get_current_project()
    # Get FeatureVector
    get_vector = mlrun.run_function(
        "hub://get_offline_features",
        name="get_vector",
        params={'feature_vector': vector_name,
                'features': features,
                'label_feature': label_column,
                "entity_timestamp_column": "timestamp",
                'target': {'name': 'parquet', 'kind': 'parquet'},
                "update_stats": True},
        outputs=["feature_vector"],
    )

    
    # Feature selection
    feature_selection = mlrun.run_function(
        "hub://feature_selection",
        name="feature-selection",
        params={
            "output_vector_name": "short",
            "label_column": project.get_param("label_column", "label"),
            "k": 18,
            "min_votes": 2,
            "ignore_type_errors": True,
        },
        inputs={
            "df_artifact": project.get_artifact_uri(get_vector.outputs['feature_vector'], "feature-vector")
        },
        outputs=[
            "feature_scores",
            "selected_features_count",
            "top_features_vector",
            "selected_features",
        ],
    )

    # train with hyper-paremeters
    train = mlrun.run_function(
        "hub://auto_trainer",
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
        hyper_param_options=HyperParamOptions(strategy="list", selector="max.accuracy"),
        inputs={"dataset": feature_selection.outputs["top_features_vector"]},
        outputs=["model", "test_set"],
    )

    # test and visualize your model
    test = mlrun.run_function(
        "hub://auto_trainer",
        name="evaluate",
        handler="evaluate",
        params={
            "label_columns": project.get_param("label_column", "label"),
            "model": train.outputs["model"],
            "drop_columns": project.get_param("label_column", "label"),
        },
        inputs={"dataset": train.outputs["test_set"]},
    )

    # Create a serverless function from the hub, add a feature enrichment router
    # This will enrich and impute the request with data from the feature vector
    serving_function = mlrun.import_function("hub://v2_model_server", 
                                             new_name="serving")
    serving_function.set_topology(
        "router",
        mlrun.serving.routers.EnrichmentModelRouter(
            feature_vector_uri="short",
            impute_policy={"*": "$mean"}),
            exist_ok=True
    )
    # Enable model monitoring
    serving_function.set_tracking()
    serving_function.save()

    # deploy your model as a serverless function, you can pass a list of models to serve
    deploy = mlrun.deploy_function(
        serving_function,
        models=[{"key": "fraud", "model_path": train.outputs["model"]}],
    )
