# 🛡️ Credit Scoring model

## Table of Contents

- [Overview](#overview)
- [Technologies](#technologies)
- [Folder Organization](#folder-organization)
- [Setup](#setup)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)

## Overview: Key Functionalities


### Data Summarization Overview


## 1. Exploratory Data Analysis (EDA)
- **Data Overview**: Structure, number of rows, columns, and data types.
- **Summary Statistics**: Analyze central tendency and distribution.
- **Numerical Features Distribution**: Visualizations for patterns, skewness, and outliers.
- **Categorical Features Distribution**: Frequency and variability insights.
- **Correlation Analysis**: Relationships between numerical features.
- **Missing Values Identification**: Determine imputation strategies.
- **Outlier Detection**: Use box plots for outlier analysis.

## 2. Feature Engineering
- **Create Aggregate Features**:
    - - - Total Transaction Amount
    - - - Average Transaction Amount
    - - - Transaction Count
    - - - Standard Deviation of Transaction Amounts
- **Extract Features**:
   - - - Transaction Hour, Day, Month, Year
- **Encode Categorical Variables**:
   - - - One-Hot Encoding and Label Encoding.
- **Handle Missing Values**: Use imputation or removal strategies.
- **Normalize/Standardize Numerical Features**: Apply normalization or standardization techniques.
- **Feature Engineering Libraries**:
   - - - Xverse
   - - - Weight of Evidence (WoE)

## 3.  Default Estimator and WoE Binning
- Construct a default estimator using RFMS formalism to classify users.
- Assign good and bad labels to users.
- Perform Weight of Evidence (WoE) binning for further analysis.  
## 4. Modeling
- **Model Selection and Training**:
   - - - Split the data into training and testing sets.
   - - - Choose at least two models: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting Machines (GBM).
   - - - Train the models and perform hyperparameter tuning.
- **Model Evaluation**:
   - - - Assess model performance using accuracy, precision, recall, F1 score, and ROC-AUC.

## 5. Model Serving API Call
-**Create a REST API**:
   - - - i used Flask
   - - - Load the trained machine-learning model.


# Model Development 
 
 - **Project Overview**: This project focuses on predicting store sales using both machine learning and deep learning techniques.
  
- **Machine Learning Approach**:
  - Utilizes a Random Forest Regressor with sklearn pipelines for modular and reproducible modeling.
  - Chooses an interpretable loss function to evaluate model performance.
  - Analyzes feature importance and estimates confidence intervals for predictions.
  - Serializes models with timestamps for tracking daily predictions.

- **Deep Learning Approach**:
  - Builds a Long Short-Term Memory (LSTM) model using TensorFlow and Keras.
  - Transforms time series data into a supervised learning format to predict future sales.
  - Ensures efficient execution in Google Colab.

# Tools & Libraries Used

1. **Programming Language**: [![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=yellow)](https://www.python.org/)
2. **Data Manipulation**: [![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
3. **Numerical Computation**: [![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
4. **Data Visualization (Basic)**: [![Matplotlib](https://img.shields.io/badge/Matplotlib-005C7F?style=flat&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
5. **Data Visualization (Advanced)**: [![Seaborn](https://img.shields.io/badge/Seaborn-30B6D7?style=flat&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
6. **Machine Learning**: [![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
7. **Gradient Boosting**: [![XGBoost](https://img.shields.io/badge/XGBoost-3EBB00?style=flat&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
8. **Feature Engineering**: [![Xverse](https://img.shields.io/badge/Xverse-7B68EE?style=flat&logo=java&logoColor=white)](https://xverse.io/)
9. **WoE and IV Calculations**: [![woe](https://img.shields.io/badge/woe-FB0F2D?style=flat&logoColor=white)](https://pypi.org/project/woe/)
10. **Evaluation Metrics**: [![Metrics from Scikit-learn](https://img.shields.io/badge/Scikit--learn%20Metrics-AB8C3A?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics)
11. **REST API**: [![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/) 
12. **CI/CD**: [![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat&logo=githubactions&logoColor=white)](https://github.com/features/actions)
13. **Cloud Platform**: [![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazonaws&logoColor=white)](https://aws.amazon.com/) 
14. **Version Control**: [![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)](https://git-scm.com/)
15. **Environment Management**: [![Virtualenv](https://img.shields.io/badge/Virtualenv-3C873A?style=flat&logoColor=white)](https://virtualenv.pypa.io/en/latest/)
## Folder Organization

```

📁.dvc
└──
    └── 📁cache
    └── 📁tmp
    └── 📜.gitignore
    └── 📃config
    └── 📃config.local

📁.github
└──
    └── 📁workflows
         └── 📃unittests.yml
└── 📁notebooks
         └── 📓ExploratoryDataAnalysis.ipynb
         └── 📓featureEngineering.ipynb
         └── 📓model_training.ipynb
└── 📁scripts
         └── 📃__init__.py
         └── 📃eda.py
         └── 📃featureextraction.py
         └── 📃hyperparameterTunning.py
         └── 📃modelcreation.py
└── 💻src
    └── 📁dashboard-div
                    └── 📁static
                            └── 📝styles.css
                    └── 📁templates
                            └── 📝index.html
                    └── 📝app.py
└── ⌛tests
         └── 📃__init__.py

└── 📜.gitignore
└── 📰README.md
└── 🔋requirements.txt
└── 📇templates.py

```

### Folder Structure: A Deep Dive

- **📁.github**: This folder contains GitHub-related configurations, including CI/CD workflows.

  - **📁workflows**: Contains the CI/CD pipeline definitions.
    - **📃blank.yml**: Configuration for Continuous Integration.
    - **📃unittests.yml**: Configuration for running unit tests.

- ## 📁notebooks: This directory holds Jupyter notebooks and related Python files.

### **📓ExploratoryDataAnalysis.ipynb**

**Overview**: This notebook performs an in-depth analysis on Exploratory data anaysis
### **📓featureEngineering.ipynb**

**Overview**: works on feature extraction

### **📓model_training.ipynb**

**Overview**: This notebook is dedicated to model training and hyper parameter tunning



- **📁scripts**: Contains Python scripts used throughout the project.

  - ## Modules Overview

This directory contains essential Python modules for analyzing and processing customer engagement data. Each module serves a specific purpose in the data analysis pipeline.

### **Modules**

- **📃 `__init__.py`**: Initializes the package and allows importing of modules.

- **📃 `eda.py`**: a module for a exploratory data analysis

### **Usage**

These modules are designed to be used in conjunction with each other to streamline the data analysis process, from data preparation and cleaning to in-depth analysis and model creation.

- **💻src**: The main source code of the project, including the Streamlit dashboard and other related files.

  - **📁dashboard-div**: Holds the code for the dashboard.
    - **📝app.py**: Main application file for the dashboard.
    - **📝README.md**: Documentation specific to the dashboard component.

- **⌛tests**: Contains test files, including unit and integration tests.

  - \***\*init**.py\*\*: Initialization file for the test module.

- **📜.gitignore**: Specifies files and directories to be ignored by Git.

- **📰README.md**: The main documentation for the entire project.

- **🔋requirements.txt**: Lists the Python dependencies required to run the project.

- **📇templates.py**: Contains templates used within the project, possibly for generating or processing data.

## Setup

1. Clone the repo

```bash
git clone https://github.com/Bereket-07/Credit-Scoring-Model.git
```

2. Change directory

```bash
cd User_Analysis_and_Engagement
```

3. Install all dependencies

```bash
pip install -r requirements.txt
```

4. change directory to run the Flask app locally.

```bash
cd src\dashboard-div
```

5. Start the Flask app

```bash
Python app.py
```

## Contributing

We welcome contributions to this project! To get started, please follow these guidelines:

### How to Contribute

1. **Fork the repository**: Click the "Fork" button at the top right of this page to create your own copy of the repository.
2. **Clone your fork**: Clone the forked repository to your local machine.
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
3. **Create a new branch**: Create a new branch for your feature or bugfix.
   ```bash
   git checkout -b feature/your-feature
   ```
4. **Make your changes**: Implement your feature or fix the bug. Ensure your code adheres to the project's coding standards and style.
5. **Commit your changes**: Commit your changes with a descriptive message.
   ```bash
   git add .
   git commit -m 'Add new feature or fix bug'
   ```
6. **Push your branch**: Push your branch to your forked repository.
   ```bash
   git push origin feature/your-feature
   ```
7. **Create a Pull Request**: Go to the repository on GitHub, switch to your branch, and click the `New Pull Request` button. Provide a detailed description of your changes and submit the pull request.

## Additional Information

- **Bug Reports**: If you find a bug, please open an issue in the repository with details about the problem.

- **Feature Requests**: If you have ideas for new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License

### Summary

The MIT License is a permissive free software license originating at the Massachusetts Institute of Technology (MIT). It is a simple and easy-to-understand license that places very few restrictions on reuse, making it a popular choice for open source projects.

By using this project, you agree to include the original copyright notice and permission notice in any copies or substantial portions of the software.
