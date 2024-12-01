# Classification_Software_Bugs
Classification of Software Bugs Using Supervised Learning Models

This project implements a pipeline for software bug classification using machine learning models, combining data collection, preprocessing, feature engineering, and model training. Below is an overview of the workflow:

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


