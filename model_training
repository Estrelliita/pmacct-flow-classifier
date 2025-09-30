from enum import Enum

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

class DataHandler:
    """Handles df operations: concatenation."""
    def __init__(self):
        #Change this with the path to your additional csv
        self.additional_dataset_path = "C:\\Users\\estre\\Documentos\\School\\Robotics\\Networks\\pmacct_ml_gui\\training_sets\\Combined_set.csv" 

    def concat_dataset(self, user_df: pd.DataFrame) -> pd.DataFrame:
        """Concatenates the user's DataFrame with an additional dataset (if provided)."""
        if self.additional_dataset_path:
            try:
                additional_df = pd.read_csv(self.additional_dataset_path)
                return pd.concat([user_df, additional_df], ignore_index=True)
            except FileNotFoundError:
                print(f"Warning: Additional dataset not found at {self.additional_dataset_path}. Returning original data.")
        return user_df
    
class Features(Enum):
  SRC_IP = "SRC_IP"
  DST_IP = "DST_IP"
  SRC_PORT = "SRC_PORT"
  DST_PORT = "DST_PORT"
  PROTOCOL = "PROTOCOL"
  SAMPLING_RATE = "SAMPLING_RATE"
  PACKETS = "PACKETS"
  BYTES = "BYTES"
  FLOWS = "FLOWS"
  TIMESTAMP_START = "TIMESTAMP_START"
  TIMESTAMP_END = "TIMESTAMP_END"
  FLOW_DURATION = "FLOW_DURATION"
  CITY = "CITY"
  STATE = "STATE"
  COUNTRY = "COUNTRY"

class MLModel(Enum):
   KNN = "knn"
   DECISION_TREE = "decisiontree"
   BEST = "best"

class FeatureProcessor:
    def __init__(self, selected_features_enums: list[Features]):
      self.selected_features = [feature.value for feature in selected_features_enums]
      self.all_categorical_features = ['SRC_IP', 'DST_IP', 'PROTOCOL', 'CITY', 'STATE', 'COUNTRY']
      self.all_numerical_features = ['SRC_PORT', 'DST_PORT', 'SAMPLING_RATE', 'PACKETS', 'FLOWS', 'BYTES', 'FLOW_DURATION']

    def categorize_features(self) -> tuple[list[str], list[str]]:
        """
        Categorizes selected Features Enums into categorical, and numerical.
        """
 
        categorical_features = []
        numerical_features = []
    
        for feature_string in self.selected_features:
            if feature_string in self.all_categorical_features:
                categorical_features.append(feature_string)
            elif feature_string in self.all_numerical_features:
                numerical_features.append(feature_string)

        return categorical_features, numerical_features
    def preprocess_features(self, user_df: pd.DataFrame, model_type: MLModel) -> pd.DataFrame:
        """Preprocesses features for the specified model type."""

        categorical_features, numerical_features = self.categorize_features()
        user_df = user_df.copy()

        # One-hot encode categorical features
        if categorical_features:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Important to handle unseen values
            encoded_data = encoder.fit_transform(user_df[categorical_features])
            encoded_cols = encoder.get_feature_names_out(categorical_features)
            encoded_df = pd.DataFrame(encoded_data, index=user_df.index, columns=encoded_cols)
            user_df = pd.concat([user_df, encoded_df], axis=1).drop(columns=categorical_features)
            numerical_features.extend(encoded_cols)  # Add encoded cols to numerical

        # Scale numerical features (only for KNN)
        if model_type == MLModel.KNN and numerical_features:
            scaler = StandardScaler()
            user_df[numerical_features] = scaler.fit_transform(user_df[numerical_features])

        return user_df

      
   
class ModelTrainer:
    def __init__(self, model_type: MLModel):
        self.model_type = model_type

    def train_and_evaluate(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, hyperparameters: dict) -> tuple[float, str]:
        """Trains and evaluates the model, returning accuracy and a classification report."""

        if self.model_type == MLModel.KNN:
            model = KNeighborsClassifier(**hyperparameters)
        elif self.model_type == MLModel.DECISION_TREE:
            model = DecisionTreeClassifier(**hyperparameters)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def grid_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict, float, str]:
        """Performs RandomizedSearchCV and returns best parameters, accuracy, and report."""

        if self.model_type == MLModel.KNN:
            model = KNeighborsClassifier()
            param_grid = {
                'weights': ['uniform', 'distance'],
                'n_neighbors': [1, 3, 5, 10, 15, 20],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'] 
            }
        elif self.model_type == MLModel.DECISION_TREE:
            model = DecisionTreeClassifier()
            param_grid = {
               'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 10, 20, 30, 40, 50]
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=5,
            verbose=0,
            random_state=42,
            n_jobs=-1,
        )
        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        report = classification_report(y_train, y_train) #Classification report for the train data

        return best_params, accuracy, report

   
def run_classification(user_df: pd.DataFrame, selected_features: list[Features], model_type: MLModel, hyperparameters: dict, use_additional_data: bool = False, grid_search: bool = False):
    data_handler = DataHandler()
    if use_additional_data:
        user_df = data_handler.concat_dataset(user_df)

    feature_processor = FeatureProcessor(selected_features)
    processed_df = feature_processor.preprocess_features(user_df, model_type)
    X = processed_df.drop(columns=['CLASS'], errors='ignore')  # Handle missing 'CLASS'
    y = user_df['CLASS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_trainer = ModelTrainer(model_type)

    results = {}

    if grid_search:
        model_trainer = ModelTrainer(model_type)
        best_params, accuracy, report = model_trainer.grid_search(X_train, y_train)
        results['best_hyperparameters'] = best_params
        results['grid_search_accuracy'] = accuracy
        results['grid_search_report'] = report
        # Train and evaluate with best params
        accuracy, report = model_trainer.train_and_evaluate(X_train, X_test, y_train, y_test, best_params)
        results['accuracy'] = accuracy
        results['classification_report'] = report
    else:
        accuracy, report = model_trainer.train_and_evaluate(X_train, X_test, y_train, y_test, hyperparameters)
        results['accuracy'] = accuracy
        results['classification_report'] = report
    return results
        
            
