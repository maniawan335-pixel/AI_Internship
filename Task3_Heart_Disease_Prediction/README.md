
Objective
Binary classification model to predict heart disease risk using UCI dataset.

Dataset
Source: heart_disease_uci.csv

Features: Age, sex, chest pain type, blood pressure, cholesterol, etc.

Target: num (0 = No disease, 1 = Disease)

Preprocessing
Dropped columns: id, ca, thal, slope

Converted target to binary (0/1)

Filled missing values:

Numerical cols: median

Categorical cols: mode

Scaled numerical features (StandardScaler)

One-hot encoded categorical features

Model
Algorithm: Logistic Regression

Train/Test Split: 80/20

Max iterations: 1000

Evaluation Metrics
Accuracy: ~80-85%

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Key Libraries
python
pandas, numpy, matplotlib, seaborn
scikit-learn (ColumnTransformer, OneHotEncoder, StandardScaler, LogisticRegression)
Run
bash
pip install pandas numpy matplotlib seaborn scikit-learn
python heart_disease_prediction.py
Sample Output
text
Accuracy: 0.85
Confusion Matrix: [[25 5]
                   [4 26]]
Visualizations
Scatter plot: trestbps vs age (colored by disease)

Confusion Matrix heatmap
