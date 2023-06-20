import mlrun
from kfp import dsl
from mlrun.model import HyperParamOptions

from mlrun import (
    build_function,
    deploy_function,
    import_function,
    run_function,
)


@dsl.pipeline(
    name="Fraud Detection Pipeline",
    description="Detecting fraud from a transactions dataset",
)
def kfpipeline(vector_name="transactions-fraud"):
    project = mlrun.get_current_project()

    feature_selection_fn = mlrun.import_function('hub://feature_selection')
    # Feature selection
    feature_selection = run_function(
        feature_selection_fn,
        name="feature_selection",
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
    )

    # train with hyper-paremeters
    train = run_function(
        "train",
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
        hyper_param_options=HyperParamOptions(selector="max.accuracy"),
        inputs={"dataset": feature_selection.outputs["top_features_vector"]},
        outputs=["model", "test_set"],
    )

    # test and visualize your model
    test = run_function(
        "train",
        name="evaluate",
        handler="evaluate",
        params={
            "label_columns": project.get_param("label_column", "label"),
            "model": train.outputs["model"],
            "drop_columns": project.get_param("label_column", "label"),
        },
        inputs={"dataset": train.outputs["test_set"]},
    )

    # route your serving model to use enrichment
    funcs["serving"].set_topology(
        "router",
        "mlrun.serving.routers.EnrichmentModelRouter",
        name="EnrichmentModelRouter",
        feature_vector_uri="transactions-fraud-short",
        impute_policy={"*": "$mean"},
        exist_ok=True,
    )

    # deploy your model as a serverless function, you can pass a list of models to serve
    deploy = deploy_function(
        "serving",
        models=[{"key": "fraud", "model_path": train.outputs["model"]}],
    )
