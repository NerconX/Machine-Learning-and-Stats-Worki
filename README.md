<h1> Statistical and Machine Learning Projects </h1>
multiple projects done in school with machine learning

# Project 1 - Multiphase Flow Pattern Classification using K-Nearest Neighbors
This project implements a k-Nearest Neighbors (k-NN) classifier to analyze and classify multiphase flow patterns using the BD12Experiments6FP dataset. The analysis includes parameter optimization and performance evaluation. <br> 

##Overview <br> 
The code performs the following main tasks:<br> 

### Data Loading and Preprocessing: <br> 
  Loads the BD12Experiments6FP.csv dataset <br> 
  Separates features and target variables <br> 
  Splits data into training (80%) and testing (20%) sets <br> 
  Applies standard scaling to normalize the features <br> 

### Initial KNN Classification: <br>
  Implements basic k-NN classifier <br>
  Evaluates initial performance metrics <br>
  Generates confusion matrix visualization <br>
  
### Parameter Optimization: <br>
  Uses GridSearchCV to find optimal k value <br>
  Tests k values from 1 to 50 <br>
  Implements cross-validation <br>
  Applies best parameters to improve model performance <br>


### Performance Comparison: <br>
  Compares accuracy before and after optimization <br>
  Visualizes results using bar charts <br>
  Analyzes confusion matrices <br>
  Reports precision, recall, and F1 scores <br>

### Results <br>
The optimization process improved the model's performance: <br>
<br>
Initial accuracy: 87% <br>
Optimized accuracy: 90% <br>
Overall improvement: 3% <br>

The optimized model showed better precision and F1 scores across all classes, indicating more reliable predictions.
Requirements

Python 3.x
Required libraries:

numpy
pandas
scikit-learn
matplotlib
seaborn



Usage

Ensure all required libraries are installed
Place the BD12Experiments6FP.csv file in the same directory as the notebook
Run the Jupyter notebook cells sequentially
View results and visualizations

Files

hw1GM.ipynb: Main Jupyter notebook containing all code and analysis
BD12Experiments6FP.csv: Dataset containing multiphase flow pattern data
README.md: This file

Implementation Details
The implementation follows these steps:

Data splitting: 80-20 train-test split
Feature scaling using StandardScaler
Initial k-NN implementation with default parameters
Grid search for optimal k value
Performance evaluation using multiple metrics
Visualization of results using confusion matrices and bar charts
