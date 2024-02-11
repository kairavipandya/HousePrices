import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to load the dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Function to preprocess the dataset
def preprocess_data(data):
    # Check for missing values
    print("Checking for missing values...")
    missing_values = data.isnull().sum()
    if missing_values.any():
        print("Warning: Missing values found! Filling missing values with 0.")
        data.fillna(0, inplace=True)
    else:
        print("No missing values found.")
    return data

# Function to split data into features (X) and target variable (y)
def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

# Function to train the model
def train_model(X_train, y_train):
    print("Training the model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

# Function to make predictions
def make_predictions(model, X_test):
    print("Making predictions...")
    return model.predict(X_test)

# Function to display evaluation results
def display_results(mae, mse, r2):
    print("Evaluation Results:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

# Main function
def main():
    # Load the dataset
    print("Loading the dataset...")
    file_path = 'boston.csv'
    house_data = load_dataset(file_path)
    
    # Preprocess the dataset
    house_data = preprocess_data(house_data)
    
    # Split data into features (X) and target variable (y)
    X, y = split_data(house_data, 'MEDV')
    
    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    mae, mse, r2 = evaluate_model(model, X_test, y_test)
    
    # Display evaluation results
    display_results(mae, mse, r2)
    
    # Make predictions
    predictions = make_predictions(model, X_test)
    print("Sample predictions:")
    print(predictions[:5])

if __name__ == "__main__":
    main()
