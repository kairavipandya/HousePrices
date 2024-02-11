# House Price Prediction

## Overview

This project aims to predict house prices based on various features such as crime rate, zoning, and room count. We use the Boston Housing Dataset for this purpose.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project consists of the following files:

- `app.py`: The main Python script containing the data loading, preprocessing, model training, evaluation, and prediction functions.
- `boston.csv`: The dataset used for training and testing the model.
- `README.md`: This file, providing an overview of the project, instructions for installation and usage, and other details.

## Getting Started

To get started with this project, follow the steps below:

### Prerequisites

- Python 3
- pandas
- scikit-learn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/house-price-prediction.git
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn
   ```

## Usage

1. Run the `app.py` script:
   ```bash
   python app.py
   ```
2. The script will load the dataset, preprocess it, split it into training and testing sets, train the model, evaluate its performance, and make predictions.
3. Evaluation metrics (Mean Absolute Error, Mean Squared Error, and R-squared) will be displayed in the console.
4. Sample predictions will also be printed.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
