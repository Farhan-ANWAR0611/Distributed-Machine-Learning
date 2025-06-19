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

