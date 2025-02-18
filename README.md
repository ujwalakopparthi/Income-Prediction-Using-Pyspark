# Income-Prediction-Using-Pyspark

💼 Income Prediction with PySpark 🔥

Welcome to the Income Prediction project! This project uses Apache Spark and various machine learning algorithms to predict whether an individual's income exceeds $50K based on various features like age, workclass, education, occupation, and more. 💵
🛠️ Technologies Used

Apache Spark: A fast and general-purpose cluster-computing system for big data processing 🔥
PySpark: Python API for Spark 💻
ngrok: Exposing local servers to the internet 🌐
Matplotlib & Seaborn: Visualization of results 📊
Scikit-learn: Metrics and evaluation 🧑‍🏫
📊 Dataset

The dataset used in this project is adult.csv, which contains information about individuals, such as:
Age 👶👴
Workclass 🧑‍💼
Education 🎓
Occupation 💼
Income 💰
The task is to predict whether the income of an individual is greater than $50K based on these features.
🏗️ Project Structure

Data Cleaning 🧹: Handle missing values, duplicates, and apply transformations.
Feature Engineering 🔧: Create new features like age_category.
Modeling 💡: Implement multiple machine learning models:
Logistic Regression (LR) 🤖
Decision Tree (DT) 🌳
Support Vector Machine (SVM) 🧑‍💼
Random Forest (RF) 🌲
Naive Bayes (NB) 📈
Gradient Boosting (GBT) 🚀
Model Evaluation 📉: Performance metrics include accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix.
⚙️ Setup Instructions

Install the required libraries:
pip install pyspark findspark pyngrok matplotlib seaborn scikit-learn
Initialize PySpark:
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("IncomePrediction").getOrCreate()
Load the dataset:
df = spark.read.csv('adult.csv', header=True, inferSchema=True)
Run the notebook! 📓
📈 Results

The project evaluates the performance of different models using metrics like:
Accuracy ✅
Precision 🎯
Recall 🔍
F1 Score 🏆
ROC-AUC 📊
🔧 How to Run the Project

Clone this repository:
git clone https://github.com/ujwalakopparthi/Income-Prediction-Using-Pyspark.git
Navigate to the project directory:
cd income-prediction
Run the notebook:
jupyter notebook income_prediction.ipynb

🎉 Contribution

Feel free to fork this project and contribute to improving it! 🚀
Hope this helps! Let me know if you'd like any changes. 😊
