# ML_Project29-XgBoost

### House Price Prediction with XGBoost Regression
This project predicts house prices using the XGBoost regression algorithm. The model is trained on a dataset containing various house features and their corresponding sale prices.

### Data
The project expects a CSV file named melb_data.csv containing house attributes and their sale prices. This dataset is publicly available online or you can replace it with your own dataset following a similar format.

### Key Features
XGBoost Regression: This project utilizes XGBoost, a powerful machine learning algorithm known for its accuracy and efficiency in regression tasks.

Feature Selection: The code focuses on relevant features like number of rooms, distance to a specific location, land size, building area, year built, and price. You can explore incorporating additional features based on your data.

Train-Test Split: The data is split into training and testing sets to train the model and evaluate its performance on unseen data.

Evaluation Metrics: Mean Absolute Error (MAE) and R-squared (R2) are used to assess the model's prediction accuracy.
Running the Project

Prerequisites: Ensure you have Python installed with libraries like pandas, scikit-learn, XGBoost, seaborn, and matplotlib. You can install them using pip:
```
pip install pandas scikit-learn xgboost seaborn matplotlib
```
Replace Data Path (Optional): If your data file is named differently or located elsewhere, update the filename in df = pd.read_csv('melb_data.csv')

Run the Script: Execute the Python script containing the code you provided.

### Understanding the Code
Data Loading (In[36]) - Loads the CSV data into a pandas DataFrame named df.

Data Exploration (In[37] - In[40])

Displays the first few rows of the data (df.head())

Selects relevant features for house price prediction (df[['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt', 'Price']])

Splits data into features (X) and target variable (y)

Splits data into training and testing sets (X_train, X_test, y_train, y_test) using train_test_split

XGBoost Model Training (In[50] - In[51])

Creates an XGBRegressor object with defined hyperparameters (n_estimators, learning_rate, n_jobs)

Trains the model on the training data (X_train, y_train)

Prediction and Evaluation (In[52] - In[53])

Generates price predictions for the testing data (X_test) using the trained model

Calculates Mean Absolute Error (MAE) and R-squared (R2) to evaluate the model's performance

### Further Enhancements

Experiment with different hyperparameter tuning techniques to potentially improve model accuracy.

Explore incorporating additional features or feature engineering techniques to capture more relevant information.

Visualize the distribution of errors to understand model biases.

Compare XGBoost performance with other machine learning algorithms for house price prediction.

### This project provides a foundation for house price prediction using XGBoost regression. Feel free to modify and extend the code to fit your specific data and requirements.
