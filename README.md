# Feature store end-to-end demo
This demo showcases financial fraud prevention using the MLRun feature store to define complex features that help identify 
fraud. Fraud prevention specifically is a challenge because it requires processing raw transaction and events in real-time, and 
being able to quickly respond and block transactions before they occur.

To address this, you create a development pipeline and a production pipeline. Both pipelines share the same feature 
engineering and model code, but serve data very differently. Furthermore, you automate the data and model monitoring 
process, identify drift and trigger retraining in a CI/CD pipeline. This process is described in the diagram below:

![Feature store demo diagram - fraud prevention](images/feature_store_demo_diagram.png)

The project implementation consists of the following steps:
1. Exploring and analyzing the data (EDA).
2. Building the data ingestion and preparation pipeline.
3. Building the model training and validation pipeline.
4. Developing the application serving pipeline (intercept requests, process data, inference, and so on).
5. Monitoring the data and model (drift, and so on.). 
6. Addressing continuous operations and CI/CD.

The data preparation step will be implemented in two ways: using standard Python packages and using a feature store.












This demo shows the usage of MLRun and the feature store. 

> - This demo works with the online feature store, which is currently not part of the Open Source default deployment.

The demo showcases:

- [**Explore and analyze the data (EDA)**](01-exploratory-data-analysis.ipynb)
- [**Prepare and train offline data**](02-interactive-data-preparation.ipynb)
- [**Ingest and prepare data**](03-ingest-with-feature-store.ipynb)
- [**Build an automated ML pipeline**](04-train-test-pipeline.ipynb)
- [**Model serving**](05-real-time-serving-pipeline.ipynb)

Fraud prevention specifically is a challenge as it requires processing raw transaction and events in real-time and being able to
quickly respond and block transactions before they occur. Consider, for example, a case where you would like to evaluate the
average transaction amount. When training the model, it is common to take a DataFrame and just calculate the average. However,
when dealing with real-time/online scenarios, this average has to be calculated incrementally.

In this demo we will learn how to **Ingest** different data sources to our **Feature Store**. Specifically, we will consider 2 types of data: 

- **Transactions**: Monetary activity between 2 parties to transfer funds.
- **Events**: Activity that done by the party, such as login or password change.

![](./images/feature_store_demo_diagram.png)

We will walk through creation of ingestion pipeline for each data source with all the needed preprocessing and validation. We will run the pipeline locally within the notebook and then launch a real-time function to **ingest live data** or schedule a cron to run the task when needed.

Following the ingestion, we will create a feature vector, select the most relevant features and create a final model. We will then deploy the model and showcase the feature vector and model serving.

## Demo flow

1. Exploratory data analysis (EDA)

- **Notebook**: [01-exploratory-data-analysis.ipynb](01-exploratory-data-analysis.ipynb)
- **Description**: Load and analyze datasets for structure, statistical distribution, categories, and missing values. Use pre-baked functions from the MLRun hub to perform EDA and modeling.
  - Explore the transactions dataset
  - Explore the user events dataset
  - Merge transactions, events datasets
  - Data Analysis with MLRun
- **MLRun hub functions:**
  - [Describe](https://www.mlrun.org/hub/functions/master/describe/)  

2. Interactive data preparation

- **Notebook**: [02-interactive-data-preparation.ipynb](02-interactive-data-preparation.ipynb)
- **Description**: Prepare three datasets: credit transactions, user events, and fraud labels, by applying transformations. User events are processed to create categorical features that capture activities like logins or password changes, which may indicate fraud. The datasets are then combined, and a target label column is generated to train and evaluate a basic model using sklearn.
- **Key steps:**
  - Preparing the credit transaction dataset
  - Preparing the user events (activities) dataset
  - Extracting labels and training a model
  - Train the model

3. Data ingestion and preparation using the MLRun feature store

- **Notebook**: [03-ingest-with-feature-store.ipynb](03-ingest-with-feature-store.ipynb)
- **Description**: 
- **Key steps:**
  - MLRun installation, create the project
  - Building the credit transactions data pipeline (feature set)
  - Defining the transactions feature set
  - Building the user events data pipeline (feature set)
  - Building the target labels data pipeline (feature set)
  - Ingest data into the feature store

4. Model training and validation pipeline

- **Notebook**: [04-train-test-pipeline.ipynb](04-train-test-pipeline.ipynb)
- **Description**: Generate feature vectors with multiple features from one or more feature sets and feed them into an automated ML training and testing pipeline to create high-quality models.
- **Key steps:**
  - Creating and evaluating a feature vector
  - Build and run an automated training and validation pipeline
  - Run the ML pipeline
  - Test the model endpoint
- **Key files:**
  - [train_workflow.py](./src/train_workflow.py)
- **MLRun hub functions:**
  - [Feature selection](https://www.mlrun.org/hub/functions/master/feature-selection/)
  - [Auto trainer](https://www.mlrun.org/hub/functions/master/auto_trainer/)
  - [V2 model server](https://www.mlrun.org/hub/functions/master/v2-model-server/)

5. Real-time application pipeline

- **Notebook**: [Real-time application pipeline](05-real-time-serving-pipeline.ipynb)
- **Description**: Define an application pipeline that accepts a user request, enriches the request with real-time features from the feature store, and feeds the features into a three-legged ensemble that uses the newly trained models.
- **Key steps:**
  - Defining a custom serving class
  - Build an application pipeline with enrichment and ensemble
  - Test the application pipeline locally
  - Deploying the function on the Kubernetes cluster
  - Test the server
- **Key files:**
  - [serving.py](.src/serving.py)