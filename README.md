# Predicting Credit Card Customer Churn

This project is an end-to-end machine learning pipeline to predict customer churn for a credit card company. By identifying customers who are likely to churn, the company can take proactive steps to retain them, which is often more cost-effective than acquiring new customers.

The project demonstrates a complete MLOps workflow, including:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training and Tuning
- Model Evaluation
- Experiment Tracking with MLflow

## Methodology

The process begins with analyzing the dataset to understand the characteristics of churning vs. non-churning customers. Then, features are preprocessed, and three different classification algorithms are trained and evaluated.

### Exploratory Data Analysis (EDA)

To understand the data, several visualizations are generated and saved in the `images/eda` directory:
- Churn distribution histogram
- Customer age distribution
- Marital status distribution
- Total transaction count distribution
- Correlation heatmap of numerical features

### Feature Engineering

The raw data is transformed to be suitable for machine learning models. The preprocessing pipeline, defined in `churn_library.py`, includes:
- **Categorical Features**: One-hot encoding is applied to categorical columns.
- **Numerical Features**: Numerical columns are scaled using `StandardScaler`.

### Model Training and Evaluation

Three different algorithms are trained to predict customer churn. The performance of each model is tracked using MLflow.

1.  **Logistic Regression**: A linear model used as a baseline for its simplicity and interpretability. It provides a good starting point for evaluating more complex models.

2.  **Random Forest**: An ensemble of decision trees. This model can capture complex, non-linear relationships in the data. It is tuned using `RandomizedSearchCV` to find the optimal hyperparameters.

3.  **LightGBM**: A high-performance gradient boosting framework. LightGBM is known for its speed and efficiency, especially with large datasets. It is also tuned using `RandomizedSearchCV`.

The models are evaluated based on a classification report, which includes precision, recall, and F1-score for the churn class. The results, including feature importance plots for the tree-based models, are saved in the `images/results` directory and logged as artifacts in MLflow.

## Project Structure

- `churn_library.py`: Core Python script containing functions for data import, EDA, feature engineering, and model training.
- `churn_notebook.ipynb`: Jupyter notebook for initial exploration and prototyping.
- `test_unit.py`: Unit tests for the `churn_library.py`.
- `config.yaml`: Configuration file for paths, model parameters, and other settings.
- `Dockerfile`: For containerizing the application.
- `Makefile`: For easy execution of common development tasks.

### Folders

- `data/`: Contains the raw CSV data.
- `images/`: Stores plots from EDA and model results.
- `logs/`: Contains log files.
- `models/`: Stores the serialized (pickled) trained models.
- `mlruns/`: Directory for MLflow experiment tracking.

## Getting Started

It is recommended to use a virtual environment. This project uses `uv` for dependency management.

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install uv
    uv pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is not present, you might need to generate it from `pyproject.toml` or install dependencies from it directly if using a modern package manager like Poetry or PDM).*

## Usage

To run the full pipeline (EDA, feature engineering, and model training):

```bash
python churn_library.py
```

This will:
- Generate EDA plots in `images/eda/`.
- Train the models and save them in the `models/` directory.
- Save model performance plots in `images/results/`.
- Create an MLflow experiment and log the runs in the `mlruns/` directory.

### MLflow UI

To view the experiment results, run the MLflow UI:
```bash
mlflow ui
```
Then navigate to `http://localhost:5000` in your browser.

## Development

This project includes tools for maintaining code quality.

### Testing

To run the unit tests:
```bash
make test
```
or
```bash
pytest
```

### Linting and Formatting

To check for code style issues and format the code:
```bash
make lint
make format
```

## Docker

To build and run the application in a Docker container:

1.  **Build the Docker image:**
    ```bash
    docker build -t customer-churn .
    ```

2.  **Run the main script inside the container:**
    ```bash
    docker run customer-churn
    ```

## Tech Stack

- **Python**
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, LightGBM
- **Experiment Tracking**: MLflow
- **Testing**: Pytest
- **Containerization**: Docker

## Resources

- [Udacity](https://www.udacity.com/)
- [Machine Learning DevOps Engineer](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821)