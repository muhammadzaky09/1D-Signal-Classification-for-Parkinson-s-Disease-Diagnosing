#Parkinson's Disease Diagnosis using 1D Signal Classification
# Methodology
Speech recordings are 1D Audio Signal and chosen as the source of data in this paper. In order to build the classifying model, a dataset that contains both speech recording of the patients and also the control group is used. The patients (188 total) come from both genders (107 men and 81 women) ranging from age 33 to 87. On the other hand, the control group consists of 64 healthy individuals (23 men and 41 women) with ages varying from 41 to 82.

The model was made using Scikit-Learn, which is a Python’s Library dedicated for Machine Learning. Specifically, Random Forest Classifier, a popular supervised machine learning algorithm, is used. Multiple decision trees provide opinion on how the data is classified and choose the most popular result (Majority Voting).

## 1. Data Exploration
Data Exploration is conducted to get to know the nature of the data. It also helps plan on what to do with the data.
![image](https://github.com/muhammadzaky09/1D-Signal-Classification-for-Parkinson-s-Disease-Diagnosing/assets/88239996/f5987125-12e5-4442-b1f5-74a736a99915)
According to the aforementioned graph, patients with Parkinson’s disease have a higher standard deviation of delta-delta log energy than healthy individuals. This finding may be important if speech characteristics are used as features to categorize Parkinson’s disease. The attributes or characteristics of the data that are used to make predictions or categorize data are known as features in machine learning. Parkinson’s patients may have some underlying differences in their speech patterns, which can be used as a feature in the classification model, as evidenced by the higher standard deviation of delta-delta log energy in these patients.

## 2. Fitting and Instantiating Model
Data standardization or preprocessing here is done by using StandardScaler class from the sklearn library which performs standardization by subtracting the mean value of each feature and then dividing it by its standard deviation. This process ensures that the transformed data has a mean of 0 and a standard deviation of 1. Then, we extract the input and target variables from df using iloc function. The variable 'x' represents the input data, which contains the speech signal features of the patients, and has a shape of (n_samples, 754), where n_samples is the number of samples in the data set. The variable 'y' represents the target variable, which contains the labels indicating whether each sample is positive or negative for Parkinson's disease, and has a shape of (n_samples,).

## 3. Data Splitting & Variable Putting
Now, after reading on the best way and proportion to split the data, the processed data is split into Training and Testing one with a composition of 20% Testing Data and 80% Training Data. In this paper, we don’t use Validation Data. Also, data is split into input and target variables. The input variables are the features that the model will use to make predictions, while the target variable is the variable that the model will predict. In this study, the target variable is the presence or absence of Parkinson’s disease, while the input variables are the speech attributes extracted from the patient’s recordings.

## 4. Training the Model
With the help of Random Forest Classifier, we train the model to be able to classify the data. In this case, the n-estimator’s hyperparameter or the number of trees is set to 100 as a means to balance between better prediction and computational time. Then, the fit() function is used to train the data after the model is initialized. The fit() function is a method in the Random Forest Classifier class that trains the model on the input data and target variables. During the training process, the model learns the underlying patterns and relationships in the data and adjusts its parameters to minimize the prediction error.

# Experimental Results
## 1. Model Testing
After training, we can predict the model on Testing Data. Afterwards, we can check the accuracy of the model using actual and predicted data.

### Metrics
```bash
- Accuracy: 88.16%
- Precision: 0.8809523809523809
- Recall: 0.9736842105263158
- R2 Score: 0.368421052631579
- F1 Score: 0.925
- MAE: 0.11842105263157894
- MSE: 0.11842105263157894
```
Above values show that the accuracy of the testing model is 88.16% correct compared to the training model. This shows a comparatively high level of correctness, thus we don’t need to make any changes necessary to the hyperparameter or the algorithm as a whole. The precision of the model is 0.8809, which indicates the proportion of true positive cases out of all positive cases predicted by the model. In other words, the model correctly identified 88.09% of the cases as positive for Parkinson’s disease out of all the cases predicted as positive by the model.

The recall of the model is 0.9737, which indicates the proportion of true positive cases out of all actual positive cases. The model correctly identified 97.37% of the actual positive cases of Parkinson’s disease. The R2 score of the model is 0.3684, which indicates the proportion of variance in the target variable (Parkinson’s disease presence) that is explained by the model. A higher R2 score indicates a better fit of the model to the data. The F1 score of the model is 0.925, which is the harmonic mean of precision and recall. It is a measure of the model’s overall accuracy, taking into account both precision and recall. The MAE (Mean Absolute Error) and MSE (Mean Squared Error) are both 0.1184, which indicate the average difference between the actual and predicted values. A lower MAE and MSE indicate better performance of the model.
![image](https://github.com/muhammadzaky09/1D-Signal-Classification-for-Parkinson-s-Disease-Diagnosing/assets/88239996/7f664f7c-18d3-4301-b1ab-3a2badd934fb)

The testing model can be visualized using the confusion matrix plot so it can be easily comprehended. 111 data are classified as True Positive, 23 as True Negative, 15 as False Positive, and 3 as False Negative. As the false negative is categorized as the most dangerous of them, the model has shown quite good results.

# Discussion
Speech Impairment Analysis as a method of early detection of Parkinson’s disease is one of the problems that could be solved using Machine Learning. Random Forest Classifier is chosen because of its reliability and higher percentage of accuracy and precision. Dataset that is chosen is Parkinson’s Disease Classification Dataset. In order to build the classifying model, a dataset that contains both speech recording of the patients and also the control group. The patients (188 total) come from both genders (107 men and 81 women) ranging from age 33 to 87. On the other hand, the control group consists of 64 healthy individuals (23 men and 41 women) with ages varying from 41 to 82.

The results of the classifying model also showed a quite good result, with the accuracy of 88.16% and 88.09% precision. The recall of the model is 0.97. The R2 score of the model is 0.3684, which indicates the proportion of variance in the target variable (Parkinson’s disease presence) that is explained by the model. A higher R2 score indicates a better fit of the model to the data. The F1 score of the model is 0.925, which is the harmonic mean of precision and recall. It is a measure of the model’s overall accuracy, taking into account both precision and recall. The MAE (Mean Absolute Error) and MSE (Mean Squared Error) are both 0.1184, which indicate the average difference between the actual and predicted values. A lower MAE and MSE indicate better performance of the model.

The testing model can be visualized using the confusion matrix plot so it can be easily comprehended. 111 data are classified as True Positive, 23 as True Negative, 15 as False Positive, and 3 as False Negative. As the false negative is categorized as the most dangerous of them, the model has shown quite good results.
