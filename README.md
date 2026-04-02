## **LOAN APPROVAL PREDICTION USING RANDOM FOREST AND SHAP**

### **PROBLEM STATEMENT**
Banks face challenges in processing the growing number of loan applications, leading to delays and biased decisions.This creates a need for an automated system to ensure faster and reliable decision making by explaining reason behind the loan approvals or rejections

### **OBJECTIVE**
To build an automated machine learning model for predicting loan approvals, while providing explainable insights into the key factors influencing decisions that
enhances transparency for both banks and customers.

### **PROJECT WORKFLOW**
<p align="left">
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/580c13ff-0b75-4b1d-bfe9-0ab48d90c3ae" />

### **DATASET OVERVIEW**
- The dataset used in this project is collected from Kaggle and contains information about loan applicants.
- It consists of 50,000 records and 20 features, providing a large dataset for training the model.
- The dataset includes both numerical and categorical variables related to applicant financial details.
- Key features include age, income, loan amount, credit-related ratios and other financial indicators.
- The target variable in the dataset is loan status, where 1 indicates loan approval and 0 indicates rejection

### **TOOLS USED**
- Python – Main programming language for data processing and modeling
- Pandas – Data manipulation and analysis
- NumPy – Numerical computations
- Matplotlib & Seaborn – Data visualization
- Scikit-learn – Machine learning models 
- SHAP – Explainable AI for feature importance and model interpretation

### **DATA PREPROCESSING**
- The dataset was  preprocessed to ensure it is clean and suitable for model training.
- Categorical variables  were converted into numerical format using encoding techniques.
- Irrelevant columns such as customer ID were excluded from the model training process.
- Divided dataset into training and testing sets with 80% training, 20% testing
- The dataset was then prepared and structured properly before feeding it into the machine learning model.

### **EXPLORATORY DATA ANALYSIS**

### **CORRELATION MATRIX**

### **EVALUATION MATRIX**
- The performance of the model was evaluated using multiple metrics to ensure reliability and accuracy.
- Accuracy was used to measure the overall correctness of the model that it achieved
- Precision measures how many of the predicted approvals or rejections were actually correct.
- Recall measures the model’s ability to correctly identify actual approved and rejected cases.
- F1-score provides a balance between precision and recall, ensuring a fair evaluation of the model.

### **MODEL PERFORMANCE**
- In this project, the Random Forest Classifier was used to predict loan approval.
- The model was trained and evaluated using test data to measure its performance.
- The model achieved an overall accuracy of 92%, indicating strong predictive capability.
- For rejected loans (Class 0), the model achieved a precision of 0.92 and recall of 0.89.
- For approved loans (Class 1), the model achieved a precision of 0.91 and recall of 0.93.
- The F1-score of around 0.91–0.92 shows a good balance between precision and recall.
- Random Forest was chosen because it handles large datasets efficiently and achieved more accuracy than other models

### **HYPERPARAMETER TUNING**
- Hyperparameter tuning was performed to improve the performance of the Random Forest model.
- Different combinations of parameters were tested to find the optimal settings for the model.
- Techniques such as Grid Search was used to identify the best parameter values.
- After tuning, the model showed improved accuracy and better generalization on unseen data.
- This step helped in reducing overfitting and enhancing the overall performance of the model.

### **SHAP ANALYSIS**
- To improve model transparency, SHAP (SHapley Additive Explanations) was used to explain the predictions.
- SHAP helps identify how each feature contributes to the final decision of loan approval or rejection.
- It provides local explanations, showing the impact of each feature for an individual prediction.
- Positive SHAP values indicate features that support loan approval, while negative values indicate features that contribute to rejection.
- It also enables users to know why a particular loan was approved or rejected.

### **DEPLOYMENT**
- The trained machine learning model was deployed using a Streamlit web application to make it accessible and user-friendly.
- The application allows users to input applicant details such as income, loan amount, and employment status through an interactive interface.
- Based on the input values, the model instantly predicts whether the loan will be approved or rejected.
- SHAP analysis was integrated into the application to provide explanations for each prediction 
- The app visually shows how each feature positively or negatively impacts the final decision

### **FUTURE WORK**
- Use a larger and more diverse dataset to make the model more robust and generalizable 
- Deploy the application on a cloud platform for wider accessibility and real-time usage 
- Improve model performance by using advanced algorithms such as Gradient Boosting or XGBoost 
- Advanced machine learning or deep learning models can be explored to further improve prediction accuracy.
- The application can be integrated with real-time banking systems for practical implementation.






