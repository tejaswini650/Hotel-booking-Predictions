# Converted from mlcode.ipynb
# Run this file in VS Code using: python mlcode.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

df=pd.read_csv(r"C:\\Users\\USER\\Downloads\\classification\\hotel_bookings_updated_2024.csv")
df

df.shape

df.info()

df.isnull().sum() # missing values

df['children']=df['children'].fillna(df['children'].median())# numerical data,mode--->categorical data
df['country']=df['country'].fillna(df['country'].mode()[0])  # fill the missing values

df.drop(columns=['company'],inplace=True)
df.drop(columns=['agent'],inplace=True)

df.isnull().sum()

df.duplicated().sum() # duplicate values

df["reservation_status"].value_counts()

# ### Handling outliers

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Select numerical columns
num_cols = df.select_dtypes(include="number").columns.tolist()

# Exclude target and identifier columns from outlier analysis
exclude_cols = ["is_canceled", "agent", "company"]
features = [col for col in num_cols if col not in exclude_cols]

# Create multiple rows of boxplots
n_cols = 3
n_rows = math.ceil(len(features) / n_cols)

plt.figure(figsize=(18, 5 * n_rows))

for i, feature in enumerate(features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(y=df[feature])
    plt.title(f"Boxplot of {feature}")
    plt.ylabel(feature)

plt.tight_layout()
plt.show()

# ### Remove outliers

# 1. Select numerical columns
num_cols = df.select_dtypes(include="number").columns

# 2. Exclude target and identifier columns
exclude_cols = ["is_canceled", "agent", "company"]
features = [col for col in num_cols if col not in exclude_cols]

# 3. Calculate IQR limits and remove outlier rows
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

# 4. Check final dataset shape
print("Dataset shape after removing outliers:", df.shape)

leakage_columns = [
    "is_canceled",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list"
]

df1 = df.drop(columns=leakage_columns, errors="ignore")

print(df1.shape)
print(df1.columns.tolist())

# ### splitting the data

# Independent variables
X = df1.drop("reservation_status", axis=1)

# Dependent variable
y = df1["reservation_status"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print(y.value_counts())
print("\nPercentage:")
print(y.value_counts(normalize=True) * 100)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# ### numerical and categorical data

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import accuracy_score, classification_report

num_cols = X_train.select_dtypes(include=['number']).columns

cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

print("Numerical:", len(num_cols))
print("Categorical:", len(cat_cols))
print("Numerical columns:",num_cols)
print("Categorical columns:",cat_cols)

# ### Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_cols = X_train.select_dtypes(include='number').columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(
        handle_unknown='infrequent_if_exist',
        min_frequency=100
    ))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

num_pipeline

cat_pipeline

preprocessor

# ### PCA

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same preprocessing to testing data
X_test_processed = preprocessor.transform(X_test)

print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=False)

X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)
print("Training data shape:", X_train_scaled.shape)
print("Number of columns:", X_train_scaled.shape[1])

from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(
    n_components=100,
    svd_solver='arpack',
    random_state=42
)

# Fit PCA on training data
X_train_pca = pca.fit_transform(X_train_processed)

# Transform testing data
X_test_pca = pca.transform(X_test_processed)

# Display results
print("Training shape:", X_train_pca.shape)
print("Testing shape:", X_test_pca.shape)

print("Explained variance:",
      pca.explained_variance_ratio_.sum())

# ### Filter Method

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=50)

X_train_selected = selector.fit_transform(
    X_train_processed, y_train
)

X_test_selected = selector.transform(
    X_test_processed
)
print("Before:", X_train_processed.shape)
print("After:", X_train_selected.shape)
print("Before Filter Method:", X_train_processed.shape[1])
print("After Filter Method:", X_train_selected.shape[1])

# ### Base Model

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=False)

X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

from sklearn.linear_model import LogisticRegression

# Create base model
base_model = LogisticRegression(max_iter=1000)

# Train the model
base_model.fit(X_train_scaled, y_train)

# test the model
y_pred = base_model.predict(X_test_scaled)

from sklearn.metrics import accuracy_score, classification_report

print("Base Model Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ### Ensemle Learning models by pipeline

from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    BaggingClassifier,
    #BoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

# 1. Bagging Pipeline
bagging_pipeline = Pipeline([
    ("preprocessor", (preprocessor)),
    ("classifier", BaggingClassifier(
        #estimator=DecisionTreeClassifier(),
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])
bagging_pipeline

# 2. Boosting Pipeline (AdaBoost)
boosting_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    ))
])
boosting_pipeline

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Create the complete pipeline
random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])
random_forest_pipeline

# 4. AdaBoost Pipeline
adaboost_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", AdaBoostClassifier(
        n_estimators=100,
        random_state=42
    ))
])
adaboost_pipeline

# 5. Gradient Boosting Pipeline
gradient_boosting_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ))
])
gradient_boosting_pipeline

#6. DECISION TREE PIPELEINE
decision_tree_pipeline=Pipeline([
     ("preprocessor", clone(preprocessor)),
    ("classifier", DecisionTreeClassifier(
        #n_estimators=100,
        random_state=42
    ))
])
decision_tree_pipeline

boosting_pipeline.fit(X_train, y_train)

random_forest_pipeline.fit(X_train, y_train)

adaboost_pipeline.fit(X_train, y_train)

gradient_boosting_pipeline.fit(X_train, y_train)
decision_tree_pipeline.fit(X_train,y_train)

print("All models trained successfully!")

# ### HyperParameter Tuning

import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.model_selection import RandomizedSearchCV

# BOOSTING - OPTIMIZED HYPERPARAMETER TUNING

param_grid = {
    'classifier__n_estimators': [50, 100, 150, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.5, 1.0]
}

random = RandomizedSearchCV(
    estimator=adaboost_pipeline,
    param_distributions=param_grid,
   ## n_iter=8,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()

random_result = random.fit(X_train, y_train)

random_time = time.time() - start_time

print("Boosting Best Accuracy:", random_result.best_score_)
print("Best Parameters:", random_result.best_params_)
print("Execution Time:", round(random_time, 2), "seconds")

best_boosting = random_result.best_estimator_

from sklearn.model_selection import RandomizedSearchCV
import time

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', None]
}

random = RandomizedSearchCV(
    random_forest_pipeline,
    param_distributions=param_grid,
    #n_iter=10,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()
random_result = random.fit(X_train, y_train)
random_time = time.time() - start_time

print("Random Forest Best Accuracy:", random_result.best_score_)
print("Best Parameters:", random_result.best_params_)
print("Execution Time:", round(random_time, 2), "seconds")

best_random_forest = random_result.best_estimator_

from sklearn.model_selection import RandomizedSearchCV
import time

param_grid = {
    'classifier__n_estimators': [50, 100, 150, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.5, 1.0]
}

random = RandomizedSearchCV(
    adaboost_pipeline,
    param_distributions=param_grid,
    #n_iter=10,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()
random_result = random.fit(X_train, y_train)
random_time = time.time() - start_time

print("AdaBoost Best Accuracy:", random_result.best_score_)
print("Best Parameters:", random_result.best_params_)
print("Execution Time:", round(random_time, 2), "seconds")

best_adaboost = random_result.best_estimator_

from sklearn.model_selection import RandomizedSearchCV
import time

param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__learning_rate': [0.05, 0.1, 0.2],
    'classifier__max_depth': [2, 3],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

random = RandomizedSearchCV(
    estimator=gradient_boosting_pipeline,
    param_distributions=param_grid,
    #n_iter=2,
    cv=2,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()

random_result = random.fit(X_train, y_train)

random_time = time.time() - start_time

print("Gradient Boosting Best Accuracy:", random_result.best_score_)
print("Best Parameters:", random_result.best_params_)
print("Execution Time:", round(random_time, 2), "seconds")

best_gradient_boosting = random_result.best_estimator_

# 6. Define hyperparameters
param_grid = {
    'classifier__max_depth': [3, 4, 5, 6, 7, 8, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_features': [None, 'sqrt', 'log2']
}

# RandomizedSearchCV

dt = DecisionTreeClassifier(random_state=42)
random = RandomizedSearchCV(
    estimator=decision_tree_pipeline,
    param_distributions=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

start_time = time.time()
random_result = random.fit(X_train, y_train)
random_time = time.time() - start_time

print("\nRandomizedSearchCV Best Accuracy: %.4f" % random_result.best_score_)
print("Best Params:", random_result.best_params_)
print("Execution Time: %.2f seconds" % random_time)
best_decision_tree=random_result.best_estimator_

best_models = {
    "Random Forest": best_random_forest,
    "Gradient Boosting": best_gradient_boosting,
    "AdaBoost": best_adaboost,
    "Decision Tree": best_decision_tree
}
best_models

# ### Evaluate the model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

results = []

for name, model in best_models.items():

    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, average='macro', zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, average='macro', zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, average='macro', zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print(results_df.sort_values(
    by='F1 Score', ascending=False
))

import matplotlib.pyplot as plt

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", figsize=(10, 5))

plt.title("Comparison of Tuned Ensemble Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ### Cross Validation

from sklearn.model_selection import cross_val_score, StratifiedKFold
import time

# Random Forest settings
random_forest_pipeline.set_params(
    classifier__n_estimators=50,
    classifier__n_jobs=1
)

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

start_time = time.time()

cv_scores = cross_val_score(
    random_forest_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
    n_jobs=2,
    error_score="raise"
)

print("Cross-validation F1 scores:", cv_scores)
print("Mean F1 Score:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())
print("Execution Time:", round(time.time() - start_time, 2), "seconds")

from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd

cv_results = []

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, model in best_models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        error_score="raise"
    )

    cv_results.append({
        "Model": name,
        "Mean F1 Score": scores.mean(),
        "Std": scores.std()
    })

cv_results_df = pd.DataFrame(cv_results)

cv_results_df = cv_results_df.sort_values(
    by="Mean F1 Score",
    ascending=False
).reset_index(drop=True)

print(cv_results_df)

# ### compare  the algorithms

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)
best_model_name = results_df.loc[0, "Model"]

print("Best Model:", best_model_name)
print("Best F1 Score:", results_df.loc[0, "F1 Score"])

best_model = best_models[best_model_name]

# ### confusion matrix and classification report

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Predict test data
best_model = best_random_forest
y_pred = best_model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

best_model_name = cv_results_df.loc[0, "Model"]

best_model = best_models[best_model_name]

print("Best Model:", best_model_name)
print("Best CV F1 Score:", cv_results_df.loc[0, "Mean F1 Score"])
print("Standard Deviation:", cv_results_df.loc[0, "Std"])

y_probability = best_model.predict_proba(X_test)

print("Prediction Probabilities:")
print(y_probability[:10])

print("Model classes:", best_model.classes_)
print("Target unique values:", y_train.unique())

import pandas as pd

y_probability = best_model.predict_proba(X_test)

probability_df = pd.DataFrame(
    y_probability,
    columns=best_model.classes_
)

probability_df["Predicted Status"] = best_model.predict(X_test)

probability_df["Actual Status"] = y_test.values

print(probability_df.head(10))

# ### save the model

import joblib

joblib.dump(
    best_model,
    "hotel_reservation_model.pkl"
)

print("Model saved successfully!")

best_model = joblib.load("hotel_reservation_model.pkl")
print("Model loaded successfully!")

# ### Test predictions on new booking data

new_booking = pd.DataFrame([{
    'hotel': 'City Hotel',
    'lead_time': 50,
    'arrival_date_year': 2017,
    'arrival_date_month': 'August',
    'arrival_date_week_number': 32,
    'arrival_date_day_of_month': 10,
    'stays_in_weekend_nights': 2,
    'stays_in_week_nights': 3,
    'adults': 2,
    'children': 0,
    'babies': 0,
    'meal': 'BB',
    'country': 'PRT',
    'market_segment': 'Online TA',
    'distribution_channel': 'TA/TO',
    'is_repeated_guest': 0,
    'previous_cancellations': 0,
    'previous_bookings_not_canceled': 0,
    'reserved_room_type': 'A',
    'deposit_type': 'No Deposit',
    'customer_type': 'Transient',
    'adr': 100.0,
    'required_car_parking_spaces': 0,
    'total_of_special_requests': 1,
    'city': 'Hyderabad'
}])

# Match new booking columns with training columns
new_booking = new_booking.reindex(columns=X_train.columns)

# Predict reservation status
prediction = best_model.predict(new_booking)

print("Predicted Reservation Status:", prediction[0])

# Display prediction probabilities
probabilities = best_model.predict_proba(new_booking)[0]

for status, probability in zip(best_model.classes_, probabilities):
    print(f"{status}: {probability * 100:.2f}%")