FILE 1

# Distributed-Machine-Learning

# 💻 Banking Data Analysis with Apache Spark – Data Parallelism Project

## 📌 Objective

The objective of this project is to demonstrate how **data parallelism** in Apache Spark can be used to process large-scale banking data efficiently. The dataset (`bank.csv`) contains client information related to marketing campaigns for term deposit subscriptions.

## 🗂️ Dataset Description

The dataset includes the following features:

- `age`: Age of the client
- `job`: Type of job
- `marital`: Marital status
- `education`: Education level
- `default`: Has credit in default?
- `balance`: Account balance
- `housing`: Has housing loan?
- `loan`: Has personal loan?
- `contact`: Type of communication
- `day`: Last contact day
- `month`: Last contact month
- `duration`: Last contact duration
- `campaign`: Number of contacts in the current campaign
- `pdays`: Days since the client was last contacted
- `previous`: Number of contacts before this campaign
- `poutcome`: Outcome of previous campaign
- `y`: Client subscribed to a term deposit (target variable)

## ⚙️ Technologies Used

- Apache Spark
- PySpark (Spark SQL, MLlib)
- Google Colab (Cloud Notebook Environment)
- Python

## 🚀 Project Structure and Tasks

### 1. 📊 Data Preparation and Partitioning

- Loaded the dataset into a Spark DataFrame.
- Partitioned the dataset using the `repartition()` method based on the `job` column to improve parallel processing.

### 2. 🧠 Parallel Data Analysis

- Calculated **average account balance per job category** using Spark’s parallel `groupBy()` and `avg()` functions.
- Identified the **top 5 age groups with the most personal loans**, using derived age group buckets and distributed aggregation.

### 3. 🤖 Model Training with Parallelism

- Used **Logistic Regression** to predict whether a client will subscribe to a term deposit.
- Partitioned the data into training and test sets.
- Encoded categorical features using `StringIndexer`, and assembled features using `VectorAssembler`.
- Trained the model using Spark MLlib with automatic parallelization.

### 4. 📈 Model Evaluation

- Evaluated the model using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

### 5. 📡 Resource Monitoring

- Used the `psutil` library to track CPU and memory usage before and after data processing.
- Observed increased CPU activity during parallel operations and efficient memory usage throughout.

### 6. 📅 Task Management and Scheduling

- Used Spark's internal DAG scheduler and ML pipelines to manage complex, multi-step workflows.
- Ensured preprocessing and model training tasks were optimized for execution in parallel using Spark’s lazy evaluation.

## ✅ Key Insights

- The age group **30–39** had the highest number of loan holders.
- Clients with **retired** or **student** job titles had the highest average account balances.
- Logistic Regression was successfully trained and evaluated using distributed processing.

## 📂 File Structure





#🚀 2ND FILE: MACHINE LEARNING WITH SPARK ML
      


                      
# 🧠 Term Deposit Subscription Prediction using Spark ML

This project uses **Apache Spark MLlib** to build a binary classification model that predicts whether a client will subscribe to a term deposit, based on personal and financial data from the `bank.csv` dataset. It also includes preprocessing, exploratory analysis, model evaluation, and hyperparameter tuning in a distributed environment.

---

## 📁 Dataset Overview

The dataset contains banking information for marketing campaigns. Key features include:

| Feature        | Description |
|----------------|-------------|
| `age`          | Age of the client |
| `job`          | Type of job |
| `marital`      | Marital status |
| `education`    | Education level |
| `default`      | Credit default status |
| `balance`      | Account balance |
| `housing`      | Housing loan status |
| `loan`         | Personal loan status |
| `contact`      | Contact communication type |
| `month`        | Last contact month |
| `duration`     | Last contact duration (in seconds) |
| `campaign`     | Number of contacts during this campaign |
| `pdays`        | Days since last contact |
| `previous`     | Number of previous contacts |
| `poutcome`     | Outcome of previous campaign |
| `y`            | Target (Yes/No – subscribed to term deposit) |

---

## 🎯 Project Objective

To build and evaluate a machine learning pipeline in Apache Spark to:
- Preprocess and transform data
- Train and validate a **Logistic Regression** model
- Evaluate model performance using multiple metrics
- Tune hyperparameters for better accuracy
- Identify important features influencing client behavior

---

## 🔄 Workflow Summary

### 1️⃣ Data Loading & EDA
- Loaded `bank.csv` into a Spark DataFrame
- Displayed schema and inspected raw rows

### 2️⃣ Data Preprocessing
- Handled missing values (none found)
- Identified and kept outliers
- Converted categorical columns to numeric using `StringIndexer`

### 3️⃣ Feature Engineering
- Combined all features into a single vector using `VectorAssembler`

### 4️⃣ Model Training
- Used **Logistic Regression** for binary classification
- Split dataset (80% training / 20% testing)

### 5️⃣ Model Evaluation
- Accuracy: **89.17%**
- Precision: **87.07%**
- Recall: **89.17%**
- F1 Score: **87.07%**

### 6️⃣ Hyperparameter Tuning
- Used `ParamGridBuilder` and `CrossValidator`
- Best model accuracy after tuning: **89.06%**

### 7️⃣ Feature Importance (Top Influencers)
| Feature              | Effect |
|----------------------|--------|
| `poutcome_indexed`   | Strong Positive |
| `housing_indexed`    | Moderate Positive |
| `marital_indexed`    | Slight Positive |
| `loan_indexed`       | Strong Negative |
| `contact_indexed`    | Mild Negative |

---

## 📊 Key Insights

- Past campaign outcomes (`poutcome`) are the most reliable predictor of subscription.
- Clients with a housing loan are more likely to subscribe.
- Personal loans negatively impact likelihood of subscription.
- Several features like age, education, and balance had little effect in this model.

---

## 🛠 Technologies Used

- Apache Spark 3.x
- PySpark (Spark MLlib)
- Google Colab / Local Environment
- Python 3.11
- Pandas, Matplotlib (optional for visualization)

---

## 📂 How to Run

1. Open `Google Colab` or your local Jupyter environment.
2. Upload the `bank.csv` dataset.
3. Execute the steps in order:
   - Data loading
   - Preprocessing
   - Feature engineering
   - Model training & evaluation
   - Hyperparameter tuning

---



---

## ✅ Final Verdict

This Spark ML project successfully demonstrates how distributed machine learning can be used in real banking environments to **predict customer behavior**, **optimize marketing efforts**, and **derive strategic insights** from large datasets.

---


