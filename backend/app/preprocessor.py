import pandas as pd
import numpy as np
import json
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Define target columns
target_columns = [
    "Life Stabilization",
    "Employment Assistance Services",
    "Retention Services",
    "Specialized Services",
    "Employer Financial Supports",
    "Enhanced Referrals for Skills Development",
    "Outcome"
]

df_train = None
df_new = None
numerical_cols = None
object_cols = None
num_imputer = None
obj_imputer = None
preprocessor = None
model = None
model_return_to_work = None
X_new_transformed = None
predicted_values = None

def load_and_dropna(csv_path):
    """Load a CSV file and drop columns where all values are missing."""
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=1, how='all')
    return df

def impute_missing_values(df):
    """Impute missing values in a DataFrame and return additional information."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()

    num_imputer = SimpleImputer(strategy='most_frequent')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    obj_imputer = SimpleImputer(strategy='constant', fill_value='missing')
    df[object_cols] = obj_imputer.fit_transform(df[object_cols])
    
    return df, numerical_cols, object_cols, num_imputer, obj_imputer

def preprocess_features(df, target_columns=None, is_training_data=True):
    """Preprocess features using OneHotEncoding for categorical variables."""
    if is_training_data and target_columns is not None:
        X = df.drop(target_columns, axis=1)
    else:
        X = df.copy()
    
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')

    X_transformed = preprocessor.fit_transform(X)

    if is_training_data and target_columns is not None:
        return X_transformed, df[target_columns], preprocessor
    else:
        return X_transformed, preprocessor

def split_data(X, y, test_size=0.20, random_state=42):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def compute_overall_accuracy(model, inputs, targets, name=''):
    """Compute overall accuracy for training and validation."""
    preds = model.predict(inputs)
    accuracies = []

    for i, col in enumerate(targets.columns):
        accuracy = accuracy_score(targets[col], preds[:, i])
        accuracies.append(accuracy)

    overall_accuracy = np.mean(accuracies)
    print(f"Overall {name} Accuracy: {overall_accuracy*100:.2f}%")

    return preds

def evaluate_triggered_interventions(trigger_mappings, filepath):
    """Evaluates which interventions are triggered based on the answers."""
    df = pd.read_csv(filepath, header=None)

    ca_numbers = df.iloc[0, :]
    answers = df.iloc[1, :]

    triggered_interventions = []
    triggering_ca_numbers = []

    for ca, triggers in trigger_mappings.items():
        ca_index = ca_numbers[ca_numbers == ca].index.tolist()
        if ca_index:
            ca_answer = answers.iloc[ca_index[0]]
            for answer, interventions in triggers.items():
                if ca == "CA108":
                    try:
                        if int(ca_answer) <= 5:
                            triggered_interventions.extend(interventions)
                            triggering_ca_numbers.extend([ca for _ in interventions])
                    except ValueError:
                        pass
                elif ca_answer in answer:
                    triggered_interventions.extend(interventions)
                    triggering_ca_numbers.extend([ca for _ in interventions])

    unique_triggered_interventions = list(set(triggered_interventions))
    unique_triggering_ca_numbers = list(set(triggering_ca_numbers))

    return unique_triggered_interventions, unique_triggering_ca_numbers

def load_trigger_mappings(filepath):
    """Load trigger mappings from a JSON file."""
    with open(filepath, 'r') as file:
        trigger_mappings = json.load(file)
    return trigger_mappings

def format_readable_triggered_interventions(triggered_interventions):
    """Formats the triggered interventions into a readable string."""
    interventions, ca_numbers = triggered_interventions
    output_lines = []
    
    for intervention in interventions:
        output_lines.append(f"- {intervention}")

    output_lines.append("")
    ca_questions_str = ', '.join(ca_numbers)
    output_lines.append(f"Triggered by CA Questions: {ca_questions_str}")

    return output_lines

def train_model(csv_file_path):
    global df_train, numerical_cols, object_cols, num_imputer, obj_imputer, preprocessor, model
    
    # Load and prepare training data
    df_train = load_and_dropna(csv_file_path)
    df_train, numerical_cols, object_cols, num_imputer, obj_imputer = impute_missing_values(df_train)
    X_transformed, y, preprocessor = preprocess_features(df_train, target_columns=target_columns, is_training_data=True)
    X_train, X_test, y_train, y_test = split_data(X_transformed, y)

    # Train the model
    model = RandomForestClassifier(n_jobs =-1, random_state = 42)
    model.fit(X_train,y_train)

def make_prediction(csv_file_path):
    global X_new_transformed, df_new, predicted_values

    # Load new data
    df_new = pd.read_csv(csv_file_path)

    # Impute missing values in df_new using the imputers fitted on df_train
    df_new[numerical_cols] = num_imputer.transform(df_new[numerical_cols])  # numerical_cols determined from df_train
    df_new[object_cols] = obj_imputer.transform(df_new[object_cols])  # object_cols determined from df_train

    # Apply preprocessing transformations
    X_new_transformed = preprocessor.transform(df_new)  # 'preprocessor' fitted on df_train

    # Predict target values using the trained model
    predicted_targets = model.predict(X_new_transformed)
    predicted_values = predicted_targets[0]

    # Initialize a dictionary to store target-prediction pairs
    prediction_dict = {}

    # Mapping each target with its corresponding prediction and storing in the dictionary
    for target, prediction in zip(target_columns, predicted_values):
        # Using .strip() to clean up any leading/trailing whitespace or newline characters in the prediction
        prediction_dict[target] = prediction.strip()

    # Convert the dictionary to JSON format
    prediction_json = json.dumps(prediction_dict)

    return prediction_json

def get_probability():
    outcome_index = target_columns.index("Outcome")
    outcome_classes = model.classes_[outcome_index]
    index_return_to_work = list(outcome_classes).index('Return to Work')

    # Predict probabilities
    probabilities = model.predict_proba(X_new_transformed)
    probability_return_to_work = probabilities[outcome_index][0][index_return_to_work]

    return probability_return_to_work

def train_new_model_for_outcome_only():
    global model_return_to_work, preprocessor

    # Load and prepare training data
    target_columns_outcome = ["Outcome"]
    X_transformed, y, preprocessor = preprocess_features(df_train, target_columns=target_columns_outcome, is_training_data=True)
    X_train, X_test, y_train, y_test = split_data(X_transformed, y)

    # Train the model
    model_return_to_work = RandomForestClassifier(n_jobs =-1, random_state = 42)
    model_return_to_work.fit(X_train,y_train)

def get_probabilities_for_each_intervention():
    train_new_model_for_outcome_only()

    # List to store the results: each entry will be [column, intervention, probability]
    results = []

    # Make a copy of the original DataFrame and set interventions to 'None'
    df_base = df_new.copy()
    df_base[target_columns[:-1]] = 'None'

    # Insert predicted interventions into the copied DataFrame
    for col, intervention in zip(target_columns[:-1], predicted_values):
        if intervention == 'missing':  # Skip if there is no intervention
            continue

        df_modified = df_base.copy()
        df_modified[col] = intervention

        X_modified_preprocessed = preprocessor.transform(df_modified.drop('Outcome', axis=1))
        probabilities = model_return_to_work.predict_proba(X_modified_preprocessed)
        probability_return_to_work = probabilities[0][1]

        results.append([col, intervention, probability_return_to_work])

    return results

def get_overall_probability_with_all_interventions():
    train_new_model_for_outcome_only()

    df_modified = df_new.copy()
    # Insert all the predicted interventions into the copied DataFrame
    for col, pred in zip(target_columns[:-1], predicted_values):
        df_modified[col] = pred
    
    # Make prediction on Outcome
    X_modified_preprocessed = preprocessor.transform(df_modified.drop('Outcome', axis=1))
    probabilities_with_all_interventions = model_return_to_work.predict_proba(X_modified_preprocessed)
    probability_return_to_work_with_all_interventions = probabilities_with_all_interventions[0][1]

    return probability_return_to_work_with_all_interventions

def get_triggered_interventions(filepath):
    trigger_mappings = load_trigger_mappings('trigger_mappings.json')
    triggered_interventions = evaluate_triggered_interventions(trigger_mappings, filepath)
    result = format_readable_triggered_interventions(triggered_interventions)
    return result
