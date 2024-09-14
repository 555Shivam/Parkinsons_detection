# Parkinson's Disease Prediction using K-Nearest Neighbors (KNN)

This project is an implementation of a K-Nearest Neighbors (KNN) classifier to predict the existence of Parkinson's disease based on medical data. The dataset used for this project contains various features that help distinguish Parkinson's disease from non-Parkinson's cases. Additionally, dimensionality reduction and hyperparameter tuning are applied to improve the classifier's performance.

## Dataset

The dataset used in this project is named `parkinsons.data`. It contains the following:
- **Features**: Various medical metrics that may correlate with the presence of Parkinson's.
- **Target**: `status` (1 = Parkinson's, 0 = No Parkinson's)

The dataset is read from a CSV file and processed using pandas.

## Project Structure

- **Libraries Used**: 
  - `numpy` and `pandas` for data manipulation.
  - `matplotlib` and `seaborn` for data visualization.
  - `scikit-learn` for building the KNN model, splitting data, and tuning hyperparameters.
  - `TSNE` for dimensionality reduction.

## Steps Involved

1. **Data Loading**: 
   - The dataset is loaded using `pandas.read_csv()`.
   - The index column is set to `name`, which uniquely identifies each data sample.

2. **Data Visualization**:
   - A correlation heatmap is plotted to visualize the relationship between features.
   - A t-SNE visualization is used to reduce the dataset to 2 dimensions and show how well the classes can be separated.

3. **Data Splitting**:
   - The data is split into training and testing sets using `train_test_split()` with 70% training data and 30% testing data.

4. **Model Training**:
   - A K-Nearest Neighbors (KNN) model is built using `KNeighborsClassifier()`.
   - The model is trained on the training data and predictions are made on the test data.

5. **Model Evaluation**:
   - The confusion matrix, classification report, and accuracy score are used to evaluate the model's performance.
   - A confusion matrix plot is generated to show the classification results.

6. **Hyperparameter Tuning**:
   - Grid search is used to optimize the KNN model's `n_neighbors` parameter.
   - The best parameter values and recall score are obtained using `GridSearchCV()`.

7. **Final Predictions**:
   - The tuned model is used to make final predictions on the test data.
   - The recall score is calculated to assess the model's sensitivity to identifying Parkinson's cases.

## Files

- **main_script.py**: Contains the code for data loading, processing, training, and evaluation.
- **parkinsons.data**: Dataset used for this project.

## Instructions

1. Install the necessary dependencies:
   ```bash
   mamba install scikit-learn=1.2.1 seaborn
   pip install dtreeviz
