# Diabetes-Prediction
## Background Introduction
Diabetes is one of the most common chronic diseases globally, affecting tens of millions of people each year. It is characterized by the body's inability to effectively regulate glucose levels in the blood. This is usually because the pancreas does not produce enough insulin or because cells in the body do not respond adequately to insulin. Insulin is a hormone that helps cells absorb glucose from the blood to obtain energy. Diabetes is typically characterized by the body not producing enough insulin or not being able to use insulin effectively as needed.
The task of this assignment is to predict the category of the test set based on health indicator information data after training the model with training data.
The target is divided into three categories: 0 for non-diabetic, 1 for pre-diabetes, and 2 for diabetes.
## Dataset Description
This assignment's data contains 22 fields, among which the target field is the prediction objective. The dataset is located at `dataset.csv`

Specific field descriptions are shown in the following table:
| Field         | Description                             |
|:--------------|:----------------------------------------|
| Id           | Sample ID                               |
| HighBP       | High Blood Pressure                     |
| HighChol     | High Cholesterol Check                  |
| CholCheck    | Cholesterol Check                       |
| BMI          | Body Mass Index                         |
| Smoker       | Smoker                                  |
| Stroke       | Stroke                                  |
| HeartDiseaseorAttack | Heart Disease or Attack        |
| PhysActivity | Physical Activity                       |
| Fruits       | Fruit Consumption                       |
| Veggies      | Vegetable Consumption                   |
| HvyAlcoholConsump | Heavy Alcohol Consumption         |
| AnyHealthcare | Any Healthcare                          |
| NoDocbcCost  | Whether medical costs affect doctor visits |
| GenHlth      | General Health                          |
| MentHlth     | Mental Health                           |
| PhysHlth     | Physical Health                         |
| DiffWalk     | Difficulty Walking/Climbing Stairs      |
| Sex          | Gender                                  |
| Age          | Age                                     |
| Education    | Education Level                         |
| Income       | Income Level                            |
| target       | 0 for non-diabetic, 1 for pre-diabetes, 2 for diabetic |

## Data Processing
During data processing, we adopted the following methods:
### 1. Handling Missing Values
- **Strategy**: Fill missing values using the mean value of each feature grouped by the target variable (target).
- **Result**: The dataset is complete overall, with no missing values.
### 2. Handling Outliers
- **Strategy**: Divide the data into three groups according to the target variable, detect outliers using a combination of Z-score and IQR methods, and replace outliers with their upper or lower limit values.
- **Purpose**: To minimize the impact of data imbalance on model performance.
### 3. Data Normalization
- **Binary Variables**: Identified 14 binary variables, which were not normalized.
- **Continuous Variables**: Identified 7 continuous variables, and normalization was performed differently based on model requirements:
  1. **Linear Models (such as Logistic Regression)**: Standardize all continuous variables (Z-score).
  2. **Tree-based Models (such as Decision Trees, Random Forests, XGBoost, LightGBM)**: Do not normalize the variables.
  3. **Instance-based Models (such as KNN, K-Means)**: Normalize all numerical features (min-max) to limit the range to [0,1].
  4. **Multilayer Perceptron (MLP)**: Normalize all numerical features (min-max) to limit the range to [0, 1].
### 4. Balancing Data Through Sampling Methods
#### Oversampling Methods
1. **Random Oversampling**:
   - Simple but may lead to overfitting.
2. **SMOTE (Synthetic Minority Over-sampling Technique)**:
   - Generates synthetic samples through interpolation, avoiding overfitting.
3. **ADASYN (Adaptive Synthetic Sampling Method)**:
   - Generates synthetic data for hard-to-classify samples, enhancing model generalization ability.
4. **Borderline SMOTE**:
   - Generates synthetic samples targeting minority class samples near decision boundaries.
5. **KMeans SMOTE**:
   - Combines K-Means clustering and SMOTE to generate more contextually relevant synthetic samples.
#### Hybrid Sampling Methods
1. **SMOTEENN (SMOTE + Edited Nearest Neighbors)**:
   - Combines SMOTE and ENN to optimize synthetic samples.
2. **SMOTETomek (SMOTE + Tomek Links)**:
   - Combines SMOTE and Tomek links to remove close opposing class samples, clarifying decision boundaries.
## Selection of Post-Sampling Classifiers for Diabetes Prediction
| Category  | Algorithm    | Advantages    | Disadvantages     |
|---------|--------------|---------------|-------------------|
| Linear Models       | Multiclass Logistic Regression            | Easy to implement and interpret. Efficient training. Suitable for binary classification.  | Assumes linear relationships between variables. Not suitable for complex relationships.   |
| Linear Models       | Multi-class Support Vector Machine (SVM) | Effective in high-dimensional space. Memory efficient. Has kernel functionality.  | Requires careful parameter tuning. Not suitable for large datasets.   |
| Tree-based Models   | Random Forest Classifier    | Handles overfitting well. Suitable for large datasets. Provides feature importance. | Prediction speed can be slow. Complex and difficult to interpret. |
| Tree-based Models    | XGBoost    | Fast training speed, built-in parallelization and distributed computing support; handles missing values, implicit feature selection process; offers flexibility with parameters and regularization, performing excellently across various tasks. | Many parameters and complex, requiring fine-tuning; difficult to explain; can still overfit with smaller or overly complex datasets. |
| Tree-based Models     | AdaBoost Classifier     | Increases classification accuracy. Flexible combination with any learning algorithm.  | Sensitive to noisy data and outliers. Can overfit very complex datasets.  |
| Tree-based Models     | Gradient Boosting   | Efficient and flexible. Can optimize different loss functions.  | Prone to overfitting without proper adjustments. Training is time-consuming. |
| Tree-based Models    | Ensemble Learning (Random Forest, KNN, AdaBoost, Bagging) | Complements the strengths of multiple ensemble strategies and base models, enhancing overall robustness and prediction accuracy; reduces overfitting risk through model diversity; adapts to different data distributions and task scenarios | Model complexity significantly increases, leading to high computational costs during training and prediction; parameter tuning and fusion strategy design are challenging; model interpretability further decreases, requiring more data and resources |
| Instance-based Models | K Nearest Neighbors (KNN)  | Does not make assumptions about data. Simple and effective. Suitable for any type of data.  | High computational cost. Performance depends on the number of dimensions.  |
| Neural Network Models   | MLP Classifier  | Capable of modeling complex nonlinear relationships and can handle large datasets well.  | Requires substantial computational resources and can easily overfit without appropriate regularization.|

## Model Training and Evaluation Metrics
### Validation Protocol:
In model training, we used an 80 − 20 train-test split within each fold, where 80% of the data was used for training and 20% was reserved for testing. Additionally, we used a 5-fold cross-validation method to reduce the risk of overfitting evaluation and better assess the stability and generalization capability of the model performance.
### Evaluation Metrics:
Given the severe imbalance in the original data, we used F1-score (Macro) as the metric for evaluating model performance.
The formula for calculating the F1-score is as follows
$$F1_{score}= \frac{1}{n} \sum_{i=1}^{n} F1_i $$
where
$$F1_i = 2 \times \frac{P_i \times R_i}{P_i+R_i}$$  

The available models are as follows:
1. Support Vector Machine (SVM)
2. Logistic Regression
3. K-Nearest Neighbors (KNN)
4. Decision Tree
5. Ensemble Learning

## Program Execution
The specific program code is located in `main.ipynb`
