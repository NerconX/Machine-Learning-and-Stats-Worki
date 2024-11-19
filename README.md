<h1> Statistical and Machine Learning Projects </h1>
multiple projects done in school with machine learning

<!DOCTYPE html>
<html>
<head>
<title>Multiphase Flow Pattern Classification</title>
</head>
<body>

<h1>Project 1 - Multiphase Flow Pattern Classification using K-Nearest Neighbors</h1>

<p>This project implements a k-Nearest Neighbors (k-NN) classifier to analyze and classify multiphase flow patterns using the <code>BD12Experiments6FP.csv</code> dataset. The analysis includes parameter optimization and performance evaluation.</p>

<h2>Overview</h2>
<p>The code performs the following main tasks:</p>

<h3>1. Data Loading and Preprocessing:</h3>
<ul>
 <li>Loads the BD12Experiments6FP.csv dataset</li>
 <li>Separates features and target variables</li>
 <li>Splits data into training (80%) and testing (20%) sets</li>
 <li>Applies standard scaling to normalize the features</li>
</ul>

<h3>2. Initial KNN Classification:</h3>
<ul>
 <li>Implements basic k-NN classifier</li>
 <li>Evaluates initial performance metrics</li>
 <li>Generates confusion matrix visualization</li>
</ul>

<h3>3. Parameter Optimization:</h3>
<ul>
 <li>Uses GridSearchCV to find optimal k value</li>
 <li>Tests k values from 1 to 50</li>
 <li>Implements cross-validation</li>
 <li>Applies best parameters to improve model performance</li>
</ul>

<h3>4. Performance Comparison:</h3>
<ul>
 <li>Compares accuracy before and after optimization</li>
 <li>Visualizes results using bar charts</li>
 <li>Analyzes confusion matrices</li>
 <li>Reports precision, recall, and F1 scores</li>
</ul>

<h2>Results</h2>
<p>The optimization process improved the model's performance:</p>
<ul>
 <li>Initial accuracy: 87%</li>
 <li>Optimized accuracy: 90%</li>
 <li>Overall improvement: 3%</li>
</ul>
<p>The optimized model showed better precision and F1 scores across all classes, indicating more reliable predictions.</p>

<h2>Requirements</h2>
<ul>
 <li>Python 3.x</li>
 <li>Required libraries:
   <ul>
     <li>numpy</li>
     <li>pandas</li>
     <li>scikit-learn</li>
     <li>matplotlib</li>
     <li>seaborn</li>
   </ul>
 </li>
</ul>

<h2>Usage</h2>
<ol>
 <li>Ensure all required libraries are installed</li>
 <li>Place the BD12Experiments6FP.csv file in the same directory as the notebook</li>
 <li>Run the Jupyter notebook cells sequentially</li>
 <li>View results and visualizations</li>
</ol>

<h2>Files</h2>
<ul>
 <li><code>hw1GM.ipynb</code>: Main Jupyter notebook containing all code and analysis</li>
 <li><code>BD12Experiments6FP.csv</code>: Dataset containing multiphase flow pattern data</li>
</ul>

<h2>Implementation Details</h2>
<p>The implementation follows these steps:</p>
<ol>
 <li>Data splitting: 80-20 train-test split</li>
 <li>Feature scaling using StandardScaler</li>
 <li>Initial k-NN implementation with default parameters</li>
 <li>Grid search for optimal k value</li>
 <li>Performance evaluation using multiple metrics</li>
 <li>Visualization of results using confusion matrices and bar charts</li>
</ol>

</body>
</html>
