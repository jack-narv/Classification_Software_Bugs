# Classification_Software_Bugs
Classification of Software Bugs Using Supervised Learning Models

This project implements a pipeline for software bug classification using machine learning models, combining data collection, preprocessing, feature engineering, and model training. In the main.py file we can find the following:

1. Data Collection
  - GitHub API Integration: Public repositories (python/cpython, pandas-dev/pandas, etc.) are queried using the GitHub API to retrieve commit data.
  - Commit Analysis:
    - The script parses commit messages and associated files.
    - Extracts Python code files for static analysis.
    - Features are generated using libraries like Radon (for complexity metrics) and Flake8 (for linting issues).
      
2. Feature Extraction
For each code file, key metrics are computed:
  - Maintainability Index: Evaluates code maintainability.
  - Cyclomatic Complexity: Measures the complexity of control flow.
  - Code Statistics: Includes number of lines, functions, classes, comments, and dependencies.
    
Bug types are classified from commit messages and code content using regex and static analysis tools.

3. Data Preprocessing
  - Imputation: Handles missing values in the dataset using the mean strategy.
  - Outlier Handling: Identifies outliers using the Interquartile Range (IQR) method, imputing extreme values with the median.
  - Logarithmic Transformation: Applies log transformation to skewed variables (e.g., complexity) for normalization.
  - Normalization: Standardizes numerical features using StandardScaler.
    
4. Class Balancing
Tackles class imbalance with:
  - SMOTE (Synthetic Minority Over-sampling Technique): Generates synthetic samples for minority classes.
  - Undersampling: Reduces samples in majority classes.
Combines both techniques to balance the dataset for training.

5. Machine Learning Models
Random Forest Classifier
  - A robust ensemble model that constructs multiple decision trees during training and outputs the mode of classes for classification.
  - Hyperparameter Tuning: Conducted using RandomizedSearchCV to find the optimal values for parameters such as:
    - Number of trees (n_estimators)
    - Maximum tree depth (max_depth)
    - Minimum samples for split (min_samples_split) and leaf (min_samples_leaf).

XGBoost Classifier
  - A powerful gradient-boosting framework optimized for speed and performance.
  - Handles missing values natively and supports class imbalance with scale_pos_weight.
  - Hyperparameter Tuning: Explores parameters like:
    - Learning rate (learning_rate)
    - Maximum tree depth (max_depth)
    - Subsample ratio (subsample).
      
6. Evaluation
Both models are evaluated using:
  - Accuracy: Measures overall correct predictions.
  - Precision, Recall, F1-Score: Focuses on performance per class, especially for imbalanced datasets.
  - Confusion Matrix: Visualizes true positives, false positives, false negatives, and true negatives for each class.
    
7. Results
A comparative analysis of RandomForestClassifier and XGBoostClassifier is performed, highlighting:
Model precision and recall for underrepresented bug types.
Effectiveness of preprocessing techniques like outlier handling and normalization.

8. Output
  - Best Model Export: The pipeline saves the trained XGBoost and RandomForestClassifier model (bug_type_xgboost_model.pkl, bug_type_classifier_model.pkl)
    for further deployment or analysis. Visualizations: Includes histograms, confusion matrices, and performance plots to summarize the results.

  - Visualizations: Includes histograms, confusion matrices, and performance graphs to summarize results.

------------------------------------------------------------- FLOW DESCRIPTION --------------------------------------------------------------------------

1. fetch_balanced_commits_from_multiple_repos:
Collects commit data from multiple GitHub repositories and ensures balanced sampling of commits across bug types.

  - Connects to the GitHub API to fetch commits from specified repositories.
  - Extracts Python files from each commit for further analysis.
  - Ensures the dataset has an equal representation of different bug types by limiting the number of commits for each bug type.

2. analyze_code:
Analyzes the content of Python files in commits to extract code metrics and identify potential bugs.

  - Parses Python code using ast to compute metrics like:
    - Maintainability Index
    - Cyclomatic Complexity
    - Number of functions, classes, comments, and lines
  - Uses Radon and Flake8 for static code analysis.
  - Identifies potential bug types (e.g., SyntaxError, TypeError) using regex and static analysis.
  - Output: Returns a dictionary of code metrics and the detected bug type (if any).
    
3. process_commits:
Aggregates metrics and bug types for all analyzed commits into a structured dataset.

  - Iterates through commits and applies analyze_code to each relevant file.
  - Compiles results into a DataFrame with columns for code metrics and bug types.
  - Handles errors in code analysis by recording issues in a separate error log.
    
4. process_outliers:
Identifies and handles outliers in numerical variables using the Interquartile Range (IQR) method.

  - Calculates the lower and upper bounds based on IQR.
  - Filters outliers from the dataset or returns the bounds for later imputation.
  - Output: A filtered DataFrame with outlier thresholds.
    
5. variable_logarithm_complexity:
Applies logarithmic transformation to the complexity variable to reduce skewness.

  - Transforms the complexity variable using log1p to handle zero values safely.
  - Adds a new column (complexity_log) to the dataset.
    
6. imput_outliers:
Replaces outlier values with the median of the respective variable.

  - Uses the lower and upper bounds calculated by process_outliers.
  - Imputes values outside these bounds with the variable's median.
    
7. balance_dataset:
Balances the dataset by addressing class imbalances.

  - SMOTE: Generates synthetic samples for underrepresented classes.
  - Undersampling: Reduces the number of samples in overrepresented classes.
  - Returns a balanced dataset ready for training.
    
8. normalize_features:
Standardizes numerical variables for consistent scaling.

  - Uses StandardScaler to normalize the features.
  - Ensures the dataset is scaled for optimal model performance.
    
9. hyperparameter_tuning:
Optimizes model hyperparameters using RandomizedSearchCV.

  - Defines a range of hyperparameters for the machine learning model (e.g., XGBoost, RandomForest).
  - Uses cross-validation to find the best combination of hyperparameters.
  - Returns the best model configuration.
    
10. train_model:
Trains the machine learning model and evaluates its performance.

  - Handles preprocessing, including:
    - Missing value imputation.
    - Data balancing.
    - Normalization.
  - Splits the dataset into training and testing sets.
  - Trains the model using the optimized hyperparameters.
  - Evaluates the model with metrics like accuracy, precision, recall, F1-score, and confusion matrix.
  - Saves the trained model for deployment or further analysis.
    
11. data_pipeline
Orchestrates the entire workflow.

  - Calls the tasks in sequence:
    - Fetching and analyzing commits.
    - Preprocessing and balancing the dataset.
    - Training and evaluating the machine learning model.
  - Outputs evaluation metrics and saves the final model.

------------------------------------------------------------------------------------------------------------------------------------

The bug_commit_dataset_with_types file is the dataset with the 20 thousand bug records that we are going to work with in the collab.

------------------------------------------------------------------------------------------------------------------------------------

The CLASSIFICATION_SOFTWARE_BUGS_ML.ipynb notebook loads the bug_commit_dataset_with_types dataset and works on outlier processing,
logarithmic transformation, normalization, and training the RandomForestClassifier and XGBoost models.

------------------------------------------------------------------------------------------------------------------------------------
The test.py file does the following:

  - Loads a pre-trained machine learning model (bug_classifier_model.pkl or bug_type_classifier_model.pkl).
  - Allows users to upload Python files for analysis.
  - Analyzes the file's content to compute key metrics such as complexity, number of lines, and number of functions.
  - Uses these metrics as features to predict whether the file contains a bug.
  - Displays the results in an interactive and user-friendly interface.

------------------------------------------------------------------------------------------------------------------------------------

The link to the lightning study is as follows: https://lightning.ai/live-session/94790b8f-dca8-4393-9a00-5ec84b0ebbf4
