# IoT Network Intrusion Detection System UNSW-NB15
Network Intrusion Detection based on various Machine learning and Deep learning algorithms using UNSW-NB15 Dataset

## Prerequisites
-   Sklearn
-   Pandas
-   Numpy
- imblearn
-   Matplotlib
-   Pickle
- missingno
-  Tensor
## Running the Notebook
The notebook can be run on
-   Google Colaboratory
-   Jupyter Notebook

## Instructions
- To run the code , user must have the required dataset and the libraries installed on their system.
- There are 3 `ipynb` files:
1.  **Pre-Processing + Binary Classification - mean imputation.ipynb**:
This file has all the code for dataset pre processing and classification using all models have been performed using mean imputed dataset.
2.  **BinaryClassification - regression imputation.ipynb**:
This file implements the classification on regression imputed dataset using all models.
3.  **intrusion detection system-MultiClass.ipynb**:
This file implements the multi class classification on both mean and regression imputed dataset.
- Upload the notebook on Jupyter Notebook or Google Colaboratory.
- Upload the Dataset on drive and mount drive to gain access of the dataset.
- To run complete code at once press <kbd>Ctrl</kbd> + <kbd>F9</kbd>
- To run any specific segment of code, select that code cell and press <kbd>Shift</kbd>+<kbd>Enter</kbd> or <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Enter</kbd>
- The code should be executed in the given order for best results without encountering any errors.

## Dataset 
-  [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- The total number of records is **2,540,044**.
- These records are divided into four `.csv` files named:
1. UNSW-NB15_1.csv,
2. UNSW-NB15_2.csv,
3. UNSW-NB15_3.csv,
4. UNSW-NB15_4.csv
These `.csv` files do not have feature names.
- Total **49** features with the class label both in binary and multi class. These features are described in `NUSW-NB15_features.csv` file.

## Machine/Deep Learning Models

- Random Forest Classifier 
- Linear Support Vector Machine
- Non-Linear Support Vector Machine 
- Artificial neural Networks

## Data Preprocessing
- Concatenated the 4 csv files into one and assigned feature names to each column which were extracted from the **NUSW-NB15_features.csv** file.
- Dataset has **2540043 instances** and **49 attributes.**
- Dataset has **87.35 % normal** and **12.65% abnormal** class. To cater this problem undersampling has been done by reducing the number of instances of normal class by **50%** and rest remains the same.
- Undersampled dataset is comprised of **77.54% of normal** class and **22.46% of abnormal class.**
- After dropping the object datatype attributes , dataset has **1430663** instances and **44 attributes.**
- To cater missing values , two techniques have been used.
1- **Mean imputation** 
2- **Regression imputation** 
- **Random Forest Classifier** has been used for **feature selection**. Top features whose threshold was **greater than 0.05** were discarded to avoid overfitting.
- The final pre processed dataset has **1430663 instances** and **40 attributes**. 
- Both mean imputed pre processed dataset and regression imputed dataset were stored on google drive.
- For further computation and analysis these datasets were directly used from the drive.

## Normalization
- We have used **MinMaxScaler** for normalizing the dataset but this normalized dataset is used for ANN and SVM models only.

## Splitting Dataset
- Randomly Splitting the dataset into **80% for training** and **20% for testing**

## Hyperparameter Tuning 
- **Bayesian Search** has been used to find the optimal hyperparameters for the **random forest model and Linear Support Vector Machine**.
- Results of bayesian Search were:
1- RandomForest Model:
    ```
    - n_estimators = 200
    - criterion = entropy
    - max_features = sqrt
    - min_samples_split = 5
    ```

  2- Linear Support Vector Machine :
  ```
  - penalty = l1
  - loss = squared_hinge
  - tol = 2.4004455788496022e-11
  - C = 3.9728931339630273
     ```
- These parameters were used to train the random forest and Linear SVM models.


## Classification 

### Binary Classification
 - **RandomForest** 
 HyperParameters for RF
   ```
   - n_estimators = 200 
   - criterion = entropy 
   - max_features = sqrt 
   - min_samples_split = 5
   ```

 - **Linear SVM**
 HyperParameters for linear SVM 
    ```
    - penalty = l1
   - loss = squared_hinge
   - tol = 2.4004455788496022e-11
   - C = 3.9728931339630273
   - dual = False 
   ```
 
     
 - **Non-Linear SVM**

   - Radial Basis Function (Rbf) is used as kernel
  
 - **ANN**
   HyperParameters for ANN:
   ```
   1- Epoch = 5
   2- Batch Size = 32
   3- Hidden Size = 100
   4- Learning rate = 0.001
   5- Loss function = Cross entropy 
   6- optimizer = Adam 
   ``` 
#### Results:
1- **Mean imputed Dataset:**
 - RandomForest Model: 
   - Accuracy =`99.43278125906484%`
 - Linear SVM:
    - Accuracy = `90.90667626593227%`
 - Non-linear SVM: 
    - Accuracy = `98.11870703484045%`
 - ANN:
   - Accuracy = `98.85%`


  2- **Regression imputed Dataset:**
 - RandomForest Model: 
   - Accuracy = `99.43138330776247%`
  - Linear SVM:
    - Accuracy = `90.91471448592088%`
 - Non-linear SVM:
    - Accuracy = `97.0726899728448%`
 - ANN:
   - Accuracy = `98.83%`



### Multi-class Classification 
 - **RandomForest** 
   HyperParameters for RF
   ```
   - n_estimators = 200 
   - criterion = entropy 
   - max_features = sqrt 
   - min_samples_split = 5
   ```

  - **Linear SVM**
   HyperParameters for linear SVM 
    ```
    - penalty = l1
    - loss = squared_hinge
    - tol = 2.4004455788496022e-11
    - C = 3.9728931339630273
    - dual = False 
 - **Non-Linear SVM**
   - Radial Base Function (Rbf) is used as kernel
   
  
 - **ANN**
   - **One hot encoder** is used to convert the categorical multi class labels into integers to make it compatible for the neural network model.
   - HyperParameters for ANN were:
   ```
   1- Epoch = 10
   2- Batch Size = 8
   3- Hidden Size = 64
   4- Loss function = Cross entropy 
   5- optimizer = Adam 
   ```

#### Results:
1- **Mean imputed Dataset:**
 - RandomForest Model: 
   - Accuracy = `97.05905994764672%`
 - Linear SVM:
   - Accuracy = `90.85215616513999%`
 - Non-linear SVM:
    - Accuracy = `95.47937497597271%`
 - ANN:
   - Accuracy = `96.20421272624968`


  2- **Regression imputed Dataset:**
 - RandomForest Model: 
   - Accuracy = `97.48089175313578%`
  - Linear SVM:
    -  Accuracy = `92.3696323038587%`
 - Non-linear SVM:
    - Accuracy = `95.53704046719532%`
 - ANN:
   - Accuracy = `96.22657994708754%`


## Issues Encountered 
1.  **Limited RAM:** The free version of Google Colaboratory has a restricted amount of available random access memory (RAM). This caused problems when working with UNSW-NB15 large dataset and when working with memory intensive operations like GridSearch which is an exhaustive search operation. 
 
2. The free version of Google Colab does not allow for **continuous background code execution.** This limitation poses difficulties for lengthy computations, such as GridSearch and SVM, as the Colab session becomes inactive after a certain period of time.
3. Linear SVM was not giving satisfactory results even after finding the optimal hyperparameters through Bayesian Search.

## Changes Incorporated
1. Used Bayesian Search for hyperparameter tuning instead of GridSearch. Grid Search is an exhaustive search and is computationally very expensive. 
2. Instead of using just Linear SVM , non Linear SVM was also used which gave better results than linear SVM.
3. Standardised the data through normalization technique using MinMaxScaler.

## Citations

- Intrusion detection in internet of things using supervised machine learning based on application and transport layer features using UNSW‐NB15 data‐set by Muhammad Ahmad1, Qaiser Riaz1*, Muhammad Zeeshan1 and Muhammad Safeer Khan3



