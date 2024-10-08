import mlrun
import mlrun.feature_store as fstore

from mlrun.datastore.targets import ParquetTarget


def get_offline_features(feature_vector, features, label_feature):
    
    fv = fstore.FeatureVector(feature_vector, 
                          features, 
                          label_feature=label_feature,
                          description='Predicting a fraudulent transaction')

    data = fv.get_offline_features(target=ParquetTarget())
    
    return data