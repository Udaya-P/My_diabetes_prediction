# My_diabetes_prediction
## 📉 Diabetes Prediction Using Machine Learning

This project demonstrates how to **predict whether a person is diabetic** using machine learning models, trained on the popular **PIMA Indian Diabetes Dataset**. Two notebooks are provided — each showcasing different ML approaches.

---

### 📁 Project Files

* `Diabetes prediction.ipynb`
  Uses:

  * `KNeighborsClassifier`
  * `DecisionTreeClassifier`
  * `MLPClassifier (Neural Network)`
  * Includes accuracy comparison of the models

* `My_diabetes_prediction.ipynb`
  Uses:

  * `SVC (Support Vector Classifier)`
  * Contains a **simple user input pipeline** for real-time prediction

---

### 🧠 Features Used

The dataset consists of 8 health-related features:

1. Number of Pregnancies
2. Glucose Level
3. Blood Pressure
4. Skin Thickness
5. Insulin Level
6. BMI (Body Mass Index)
7. Diabetes Pedigree Function
8. Age

The target variable is:

* `1`: Diabetic
* `0`: Not Diabetic

---

### 🚀 How Prediction Works

You can input your own data in this format:

```python
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
```

Prediction code:

```python
# Convert to numpy array
input_arr = np.asarray(input_data)

# Reshape as a single sample
input_reshaped = input_arr.reshape(1, -1)

# Standardize input based on training scaler
std_data = scaler.transform(input_reshaped)

# Make prediction
prediction = classifier.predict(std_data)

# Output result
if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
```

📅 Replace `classifier` with your trained model (e.g., `svm_model`, `knn_model`, etc.)

---

### 📊 Model Performance

| Model                    | Description                            |
| ------------------------ | -------------------------------------- |
| `SVC`                    | Achieved \~70% accuracy                |
| `KNeighborsClassifier`   | Easy to implement, distance-based      |
| `DecisionTreeClassifier` | Fast, interpretable                    |
| `MLPClassifier`          | Deep learning model for classification |

📌 Note: Accuracy may vary depending on preprocessing and test splits.

---

### 📦 Requirements

Install dependencies:

Typical packages:

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`

---

### 📈 Future Improvements

* Deploy as a web app using Streamlit or Flask

 
---

### 👤 Author

Udaya Pragna Gangavaram
Feel free to fork or contribute!
