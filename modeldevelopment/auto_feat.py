import logging
import traceback

import numpy as np
from aif360.datasets import StandardDataset
# from autofeat import AutoFeatClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Making modeldevelopment as a package and using it for importing functions and library
# import modeldevelopment.settings as settings

import settings

logger = logging.getLogger(__name__)


class MakeDataSet:

    def __init__(self):

        self.dataframe = None
        self.ignore_columns = settings.COlUMN_TO_DROP + settings.Y_COLUMN
        self.cat_columns = []
        self.num_columns = []
        self.privileged_attr = None

    def get_cat_num_columns(self):
        """
        get categorical_names
        """
        try:
            if settings.COlUMN_TO_DROP:
                self.dataframe.drop(settings.COlUMN_TO_DROP,
                                    axis=1, inplace=True)

            data = self.dataframe

            for col in data.columns:
                if (data[col].nunique() <= 51) and (col not in self.ignore_columns):
                    self.cat_columns.append(col)
                else:
                    if col not in self.ignore_columns:
                        self.num_columns.append(col)

            logger.info(f"List of Numerical Columns : {self.num_columns}")
            logger.info(f"List of Categorical Columns : {self.cat_columns}")
        except Exception as e:
            logger.error(e)

    def handle_missing_values(self):
        """
        missing values
        """
        try:
            df = self.dataframe

            # set the imputed value for all features
            filling_values = dict()
            for each_column in self.num_columns + self.cat_columns:
                if df[each_column].dtypes in ["int32", "int64", "float64", "float32"]:
                    filling_values[each_column] = df[each_column].median()
                elif df[each_column].dtypes in ["object"]:
                    filling_values[each_column] = "Missing value"

            logger.info(
                f"List of  Columns with Missing values :\n{df.isnull().sum()}")

            # Fill the missing values
            self.dataframe.fillna(filling_values, inplace=True)
        except Exception as e:
            logger.error(e)


    def decode_dataset(self, data_frame):
        """
        data_frame:
        """
        try:

            self.dataframe = data_frame

            self.get_cat_num_columns()
            self.handle_missing_values()

            # self.dataframe.to_csv(
            #     "./data/cleaned data/processed_data.csv", index=False)

            data_encoded = self.dataframe.copy()
            categorical_names = {}
            encoders = {}

            # Use Label Encoder for categorical columns (including target column)
            # print(self.cat_columns)
            for feature in self.cat_columns:
                le = LabelEncoder()
                le.fit(data_encoded[feature])

                data_encoded[feature] = le.transform(data_encoded[feature])
                

                categorical_names[feature] = le.classes_
                encoders[feature] = le

            
#             with open('lable_encoders.pickle', 'wb') as handle:
#                 pickle.dump(encoders, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # data_encoded[self.cat_columns] = data_encoded[self.cat_columns].astype('category')
            # data_encoded[self.num_columns] = data_encoded[self.num_columns].astype(float)

            self.privileged_attr = np.where(
                categorical_names[settings.PRIVILEGED_ATTRIBUTE] == '0')[0]
            print(self.privileged_attr)

#             with open('categorical_names.pickle', 'wb') as handle:
#                 pickle.dump(categorical_names, handle, protocol=pickle.HIGHEST_PROTOCOL)


            logging.info("Converting Pandas dataframe to AIF360 dataframe format...")

            dataset_orig = StandardDataset(data_encoded,
                                           label_name=settings.Y_COLUMN[0],
                                           favorable_classes=settings.FAVORABLE_CLASS,
                                           protected_attribute_names=[settings.PRIVILEGED_ATTRIBUTE],
                                           privileged_classes=[self.privileged_attr],)
                                        #    categorical_features = cat_columns_wo_privileged_attr)  # Commenting this out to avoid creating one hot encoded features

            return dataset_orig

        except Exception as e:
            logger.error(traceback.format_exc())
            print(traceback.format_exc())

    def get_scaled_data(self, train, valid, test):
        try:
            scalar = settings.SCALAR()
            dataset_copy_train = train.copy()
            dataset_copy_test = test.copy(deep=True)
            dataset_copy_valid = valid.copy(deep=True)
            
            self.get_cat_num_columns()
            print(self.num_columns)
            print(dataset_copy_train[self.num_columns].head())

            dataset_copy_train[self.num_columns] = scalar.fit_transform(
                dataset_copy_train[self.num_columns])
            dataset_copy_test[self.num_columns] = scalar.transform(
                dataset_copy_test[self.num_columns])
            dataset_copy_valid[self.num_columns] = scalar.transform(
                dataset_copy_valid[self.num_columns])

            return dataset_copy_train, dataset_copy_test, dataset_copy_valid

        except Exception as e:
            print(traceback.format_exc())
            logger.error(e)

    def train_valid_test_split(self, data_frame):
        """
        dataframe
        """
        try:

            # dataset_orig_train, dataset_orig_vt = data_frame.split(
            #     [0.7], shuffle=False)
            # dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split(
            #     [0.7], shuffle=False)
            self.dataframe = data_frame
            train, validate, test = np.split(data_frame, [int(.7*len(data_frame)), int(.9*len(data_frame))])
            
            train_scaled, validate_scaled, test_scaled = self.get_scaled_data(train, validate, test)
            
            dataset_orig_train = self.decode_dataset(train_scaled)
            dataset_orig_test = self.decode_dataset(validate_scaled)
            dataset_orig_valid = self.decode_dataset(test_scaled)

            if settings.AUTO_FEATURES:
                pass
                # af_model = AutoFeatClassifier(verbose=1,
                #                               feateng_steps=1,

                #                               )

                # X_train_feature_creation = af_model.fit_transform(
                #     dataset_orig_train.convert_to_dataframe()[0].drop(
                #         settings.Y_COLUMN, axis=1),
                #     dataset_orig_train.labels.ravel())

                # X_test_feature_creation = af_model.transform(
                #     dataset_orig_test.convert_to_dataframe()[0].drop(settings.Y_COLUMN, axis=1))

                # X_valid_feature_creation = af_model.transform(
                #     dataset_orig_valid.convert_to_dataframe()[0].drop(settings.Y_COLUMN, axis=1))

                # X_train_feature_creation[settings.Y_COLUMN[0]
                # ] = dataset_orig_train.labels.ravel()
                # X_test_feature_creation[settings.Y_COLUMN[0]
                # ] = dataset_orig_test.labels.ravel()
                # X_valid_feature_creation[settings.Y_COLUMN[0]
                # ] = dataset_orig_valid.labels.ravel()
                # print(X_train_feature_creation)
                # print(X_train_feature_creation.shape)

                # dataset_orig_train = StandardDataset(X_train_feature_creation,
                #                                      label_name=settings.Y_COLUMN[0],
                #                                      favorable_classes=settings.FAVORABLE_CLASS,
                #                                      protected_attribute_names=[
                #                                          settings.PRIVILEGED_ATTRIBUTE],
                #                                      privileged_classes=[self.privileged_attr])

                # dataset_orig_test = StandardDataset(X_test_feature_creation,
                #                                     label_name=settings.Y_COLUMN[0],
                #                                     favorable_classes=settings.FAVORABLE_CLASS,
                #                                     protected_attribute_names=[
                #                                         settings.PRIVILEGED_ATTRIBUTE],
                #                                     privileged_classes=[self.privileged_attr])

                # dataset_orig_valid = StandardDataset(X_valid_feature_creation,
                #                                      label_name=settings.Y_COLUMN[0],
                #                                      favorable_classes=settings.FAVORABLE_CLASS,
                #                                      protected_attribute_names=[
                #                                          settings.PRIVILEGED_ATTRIBUTE],
                #                                      privileged_classes=[self.privileged_attr])

            dataset_orig_train.convert_to_dataframe()[0].to_csv("./data/model results datasets/Train_data.csv",
                                                                index=False)
            dataset_orig_test.convert_to_dataframe()[0].to_csv("./data/model results datasets/Test_data.csv",
                                                               index=False)
            dataset_orig_valid.convert_to_dataframe()[0].to_csv("./data/model results datasets/Valid_data.csv",
                                                                index=False)

            return dataset_orig_train, dataset_orig_valid, dataset_orig_test

        except Exception as e:
            print(traceback.format_exc())
            logger.error(e)
