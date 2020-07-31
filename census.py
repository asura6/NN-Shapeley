import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.datasets import dump_svmlight_file
from sklearn.utils import resample
import xgboost as xgb
import shap

# Encode label-data into unique numbers
def encode_dataset(dset):
    label_encoder = preprocessing.LabelEncoder() 
    for key in dset.columns:
        dset[key] = label_encoder.fit_transform(dset[key])
    return dset

"""
Read the dataset, returns data and test_data
encode_labels turns labels into unique numbers
drop_nominal_data removes data which is not ordinal
"""
def read_data(encode_labels=True, drop_nominal_data=False):
    columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','50K']
    # Get the data files from https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
    # Download adult.data and adult.test
    data = pd.read_csv('adult.data', names=columns, sep=',')
    test_data = pd.read_csv('adult.test', names=columns, sep=',', skiprows=1)

    if encode_labels:
        data = encode_dataset(data)
        test_data = encode_dataset(test_data)

    if drop_nominal_data:
        nominal_columns = ['workclass','education','marital-status','occupation','relationship','race','native-country']
        data = data.drop(labels=nominal_columns,axis=1)
        test_data = test_data.drop(labels=nominal_columns,axis=1)

    return data, test_data

"""
Upsample unbalanced dataset to have as many minority classes as majority classes
Assumes class is found in the last column in the dataset and that the classes are either 1 or 0
"""
def upsample_dataset(data, ratio=1):
    # Check which class is minority
    count_0 = np.sum(data.iloc[:,-1] == 0)
    count_1 = np.sum(data.iloc[:,-1] == 1)

    # Split into minority and majority class
    if count_0 > count_1:
        df_majority = data[data.iloc[:,-1]==0]
        df_minority = data[data.iloc[:,-1]==1]
    else:
        df_majority = data[data.iloc[:,-1]==1]
        df_minority = data[data.iloc[:,-1]==0]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
        replace=True,     # sample with replacement
        n_samples=int(len(df_majority)*ratio))    # to match majority class
        
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1)
    return df_upsampled

"""
Write and return handles to svmlight files to reduce memory when training with xgboost
"""
def create_svmlight_datasets(X, Y, name='dtrain.svm'):
    dump_svmlight_file(X, Y, name, zero_based=True)
    dset_svm = xgb.DMatrix(name, feature_names=list(X.columns))
    return dset_svm

"""
Train and evaluate xgboost on the test_dataset
"""
def run_xgboost(X_train, Y_train, X_test, Y_test):
    # Prepare xgboost datasets
    dtrain_svm = create_svmlight_datasets(X_train, Y_train, name='dtrain.svm')
    dtest_svm = create_svmlight_datasets(X_test, Y_test, name='dtest.svm')

    # Fit the model on the training data
    param = {
        'max_depth': 30,  # the maximum depth of each tree
        'eta': 0.25,  # the training step for each iteration
        'objective':'binary:logistic',
        'num_class': 1}  # the number of classes that exist in this datset]]
    num_rounds = 100  # the number of training iterations    
    bst = xgb.train(param, dtrain_svm, num_rounds)

    # Add feature names to make analysis simpler
    bst.feature_names = list(X_train.columns)

    # Get the test predictions
    preds = bst.predict(dtest_svm)
    # Convert the predictions from fractinal values to boolean values
    best_preds = np.asarray([int(np.round(line)) for line in preds])
    # Return the fitted model and the best predictions
    return bst, best_preds


if __name__ == '__main__':
    data, test_data = read_data(encode_labels=True, drop_nominal_data=False)

    data = upsample_dataset(data, ratio=1) # Upsamle to balanced dataset

    # Split into X, Y
    X_train = data.iloc[:,:-1]
    Y_train = data.iloc[:,-1]
    X_test = test_data.iloc[:,:-1]
    Y_test = test_data.iloc[:,-1]

    # Train and evaluate using xgboost
    bst, xgboost_test_predictions = run_xgboost(X_train, Y_train, X_test, Y_test)
    print(f"Model has {np.sum(xgboost_test_predictions==Y_test)/len(xgboost_test_predictions)*100:.3} % success rate")

    # Make plots of different importance scores
    for score in ['weight','gain','cover']:
        xgb.plot_importance(bst, importance_type=score, show_values=False, xlabel=score)
        plt.savefig(f'xkb_{score}.png')

    """ Analyze using Shapeley values """
    # This hack is needed for the current version of xgboost with SHAP
    model_bytearray = bst.save_raw()[4:]
    def myfun(self=None):
        return model_bytearray
    bst.save_raw = myfun

    # Get the explainer for the xgboost model and calculate the shapeley-values
    explainer = shap.TreeExplainer(bst)
    shap_values = explainer.shap_values(X_test)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X_test)
    shap.dependence_plot("occupation", shap_values, X_test, interaction_index=None, show=False)