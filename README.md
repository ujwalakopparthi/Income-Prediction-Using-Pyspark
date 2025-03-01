# Income Prediction with PySpark

Welcome to the **Income Prediction** project! This project leverages Apache Spark and various machine learning algorithms to predict whether an individual's income exceeds $50K based on features like age, workclass, education, occupation, and more.

---

## 💼 Project Overview

The goal of this project is to predict whether an individual's income exceeds $50K based on various personal features. The dataset used is **adult.csv**, containing details such as:

- Age 👶👴
- Workclass 🧑‍💼
- Education 🎓
- Occupation 💼
- Income 💰 (target variable)

The prediction task is a binary classification: predicting whether the income is greater than $50K.

---

## 🛠️ Technologies Used

- **Apache Spark**: A fast, general-purpose cluster-computing system for big data processing.
- **PySpark**: Python API for Apache Spark, used for distributed data processing.
- **ngrok**: Exposing local servers to the internet (if required for the project).
- **Matplotlib & Seaborn**: For data visualization and plotting.
- **Scikit-learn**: For machine learning models, evaluation metrics, and performance analysis.

---

## 📊 Dataset

The dataset used in this project is **adult.csv**, which contains the following features:

- **Age**: Age of the individual.
- **Workclass**: Type of employment.
- **Education**: Level of education.
- **Occupation**: Type of occupation.
- **Income**: Whether the individual earns more than $50K or not (target variable).

---

## 🏗️ Project Structure

1. **Data Cleaning 🧹**:
   - Handle missing values and duplicates.
   - Apply necessary transformations to prepare the data for modeling.

2. **Feature Engineering 🔧**:
   - Create new features like **age_category** to enhance model performance.

3. **Modeling 💡**:
   - Implement various machine learning models:
     - Logistic Regression (LR) 🤖
     - Decision Tree (DT) 🌳
     - Support Vector Machine (SVM) 🧑‍💼
     - Random Forest (RF) 🌲
     - Naive Bayes (NB) 📈
     - Gradient Boosting (GBT) 🚀

4. **Model Evaluation 📉**:
   - Evaluate model performance using metrics such as:
     - Accuracy ✅
     - Precision 🎯
     - Recall 🔍
     - F1 Score 🏆
     - ROC-AUC 📊
     - Confusion Matrix 📉

---

## ⚙️ Setup Instructions

### 1. Install Required Libraries

You can install the necessary libraries with the following command:

```bash
pip install pyspark findspark pyngrok matplotlib seaborn scikit-learn
```
2. Initialize PySpark
python:
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("IncomePrediction").getOrCreate()

3. Load the Dataset
python:
df = spark.read.csv('adult.csv', header=True, inferSchema=True)

4. Run the Project
```bash
jupyter notebook income_prediction.ipynb
```
## 📈 Results

This project evaluates the performance of different machine learning models using the following metrics:

- **Accuracy**: Measures the proportion of correctly classified instances.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all actual positives.
- **F1 Score**: The weighted average of Precision and Recall, providing a balance between them.
- **ROC-AUC**: The Area Under the Receiver Operating Characteristic Curve, indicating the model's ability to distinguish between classes.

---

##🔧 How to Run the Project
1. Clone the repository:
```bash
git clone https://github.com/ujwalakopparthi/Income-Prediction-Using-Pyspark.git
```
2. Navigate to the project directory:
```bash
cd income-prediction
```
3. Open the Jupyter notebook:
```bash
jupyter notebook income_prediction.ipynb
```



## 🎉 Contribution

Feel free to fork this project and contribute to improving it! 🚀

If you have any suggestions, improvements, or bug fixes, please:

- Open an issue if you encounter any problems.
- Submit a pull request with your changes.

All contributions are welcome, and any help is greatly appreciated!

## 🎯 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.


