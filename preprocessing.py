from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
import data
import pandas as pd
import numpy as np
import warnings


# We need to standardize them together
def standardize_features(train_features: pd.DataFrame, test_features: pd.DataFrame):
    """
    Process the training data from the raw data frame that was read from csv using data.get_training_features

    :param appendix: can be used to retrieve csv from file rather than being passed a dataframe
    :param features: the reshaped data frame
    :param export_csv: if True, a csv with the normalized features will be saved
    :return: the standardized data as dataframe!
    """

    scaler = RobustScaler()

    large_set = pd.concat([train_features, test_features])

    matrix = scaler.fit_transform(large_set)
    large_set.loc[:] = matrix

    train_features_imputed = large_set.loc[train_features.index]
    test_features_imputed = large_set.loc[test_features.index]

    return train_features_imputed, test_features_imputed


# We should impute them together to get a more general mean for missing values
def impute_features(train_features: pd.DataFrame, test_features: pd.DataFrame):

    large_set = pd.concat([train_features, test_features])

    imp = SimpleImputer(strategy='median', missing_values=np.nan)
    matrix = imp.fit_transform(large_set)
    large_set.loc[:] = matrix

    train_features_imputed = large_set.loc[train_features.index]
    test_features_imputed = large_set.loc[test_features.index]

    return train_features_imputed, test_features_imputed
    


def verify_mean(processed_features, target_mean=0):
    """
    Compute mean along axis to verify we have target_mean

    :param processed_features: the features as array or dataframe
    :param target_mean: the mean you want to have. Default=0
    """
    counter = 0
    averages = np.average(processed_features.values, axis=0)
    for average in averages:
        if np.abs(average - target_mean) > 1e-3:
            counter += 1

    if counter != 0:
        warnings.warn('%.d columns have not target mean' % counter)


def verify_std(processed_features, target_std=1):
    """
    Compute the std along the axis to verify all features have target_std

    :param processed_features: the feautres as array or dataframe
    :param target_std: the std you want to have. Default = 0
    """
    counter = 0
    averages = np.std(processed_features.values, axis=0)
    for average in averages:
        if np.abs(average - target_std) > 1e-3:
            counter += 1

    if counter != 0:
        warnings.warn('%.d columns have not target standard deviation' % counter)

# this only takes one set of features because we impute every patient by themselves.


def prepare_features(features: pd.DataFrame = None, appendix: str = None, read_from_file= False):
    """
    Flatten the measurements for every patient


    :param features: features as dataframe
    :param appendix: for the filename when the features are being saved

    :return: the prepared features as dataframe
    """

    if read_from_file:
        df = pd.read_csv('transformed_data/reshaped_features_' + appendix + '.csv', index_col='pid')
        return df

    pids = features['pid'].drop_duplicates()

    labels_per_patient = features.columns.drop(['Age','pid', 'Time'])
    # get one row per patient
    ages = features.drop_duplicates(subset='pid')
    ages.set_index('pid', inplace=True)
    # now only keep one column, the age.
    ages = ages.loc[:,['Age']]

    # From the task description we know that every patient has 12 timestamps but let's check

    data_for_patient_1 = features.loc[features['pid'] == pids[0], :]

    number_of_timestamps = data_for_patient_1.shape[0]

    labels = ['Age']

    for index in range(number_of_timestamps):
        new_labels = [ label + "_" + str(index) for label in labels_per_patient.values]
        labels.extend(new_labels)

    new_features = pd.DataFrame(index=pids, columns=labels)



    for patient in pids.values:
        imp = SimpleImputer(strategy='mean', missing_values=np.nan)
        # first, retrieve the rows that are for this patient
        sub_frame = features.loc[features['pid'] == patient]
        # now lets find how many nans we have for every measurement
        sum_of_nans = sub_frame.isna().sum()
        # for every measuremnt, determine if we need to impute in the first place:
        # if we have all nans, this column will be dropped which is not what we want.
        impute_indices = [False if sum==number_of_timestamps else True for sum in sum_of_nans]
        mask = pd.array(impute_indices, dtype="boolean")
        # this has only the columns where not all the values are nan, this is where we will impute
        frame_to_impute = sub_frame.iloc[:,mask]
        imputed_matrix = imp.fit_transform(frame_to_impute)
        imputed_frame = pd.DataFrame(imputed_matrix, index=frame_to_impute.index, columns=frame_to_impute.columns)
        # now insert the imputed columns in the old dataframe. We will still have nans here, but they will be imputed
        # at a later stage when all patients are combined.
        new_sub_frame = sub_frame.copy()
        for index,label in enumerate(sub_frame.columns):
            if impute_indices[index]:
                new_sub_frame.loc[:,label] = imputed_frame[label]
        # sort and flatten for thi spatient
        sub_frame = new_sub_frame
        sorted = sub_frame.sort_values('Time')
        sorted = sorted.drop(['pid', 'Time', 'Age'], axis='columns')
        matrix = sorted.values
        vector = matrix.flatten()
        vector = np.insert(vector,0,ages.at[patient,'Age'])
        # insert this patient into the dataframe
        new_features.loc[patient,:] = vector



    return new_features


# raw_data = data.get_training_features()
# prepared_features = prepare_features(features=raw_data, appendix = 'train', read_from_file=False)





