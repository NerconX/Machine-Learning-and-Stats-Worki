<h1> Statistical and Machine Learning Projects </h1>
multiple projects done in school with machine learning
<!------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->
<!DOCTYPE html>
<html>
<!------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->
<body>
<h1>Project 1 - Multiphase Flow Pattern Classification using K-Nearest Neighbors</h1>

<p>This project implements a k-Nearest Neighbors (k-NN) classifier to analyze and classify multiphase flow patterns using the <code>BD12Experiments6FP.csv</code> dataset. The analysis includes parameter optimization and performance evaluation.</p>

<h2>Overview</h2>
<p>The code performs the following main tasks:</p>

<h3>1. Data Loading and Preprocessing:</h3>
<ul>
 <li>Loads the <code>BD12Experiments6FP.csv</code> dataset</li>
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
 <li>Uses <code>GridSearchCV</code> to find optimal k value</li>
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
 <li>Place the <code>BD12Experiments6FP.csv</code> file in the same directory as the notebook</li>
 <li>Run the Jupyter notebook cells sequentially</li>
 <li>View results and visualizations</li>
</ol>

<h2>Files</h2>
<ul>
 <li><code>Project1.ipynb</code>: Main Jupyter notebook containing all code and analysis</li>
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

<!------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->

<body>
<h1>Project 2 - Breast Cancer Classification using Logistic Regression</h1>
<p>This project implements a Logistic Regression classifier to analyze and classify breast cancer tumors as benign or malignant using the Breast Cancer Wisconsin dataset. The analysis includes parameter optimization and performance evaluation.</p>

<h2>Overview</h2>
<p>The code performs the following main tasks:</p>

<h3>1. Data Loading and Preprocessing:</h3>
<ul>
 <li>Loads the Breast Cancer Wisconsin dataset</li>
 <li>Separates features and target variables</li>
 <li>Splits data into training and testing sets</li>
 <li>Applies standard scaling to normalize the features</li>
</ul>

<h3>2. Initial Logistic Regression:</h3>
<ul>
 <li>Implements basic logistic regression classifier</li>
 <li>Evaluates initial performance metrics</li>
 <li>Generates confusion matrix visualization</li>
 <li>Calculates precision, recall, and F1 scores</li>
</ul>

<h3>3. Parameter Optimization:</h3>
<ul>
 <li>Uses GridSearchCV for hyperparameter tuning</li>
 <li>Tests different values for regularization parameter C</li>
 <li>Evaluates different penalty types (l1, l2)</li>
 <li>Implements cross-validation</li>
</ul>

<h3>4. Performance Comparison:</h3>
<ul>
 <li>Compares accuracy before and after optimization</li>
 <li>Analyzes confusion matrices</li>
 <li>Visualizes results using bar charts</li>
 <li>Reports detailed classification metrics</li>
</ul>

<h2>Results</h2>
<p>The optimization process showed minimal impact on model performance:</p>
<ul>
 <li>Initial accuracy: 97%</li>
 <li>Optimized accuracy: 96%</li>
 <li>Slight decrease in accuracy: -1%</li>
</ul>
<p>The high initial accuracy suggests the base model was already well-optimized for this classification task.</p>

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
 <li>Load <code>Project2.ipynb</code> in Jupyter Notebook/Lab</li>
 <li>Run all cells sequentially</li>
 <li>Review results and visualizations</li>
</ol>

<h2>Files</h2>
<ul>
 <li><code>Project2.ipynb</code>: Main Jupyter notebook containing all code and analysis</li>
 <li>Uses built-in sklearn breast cancer dataset</li>
</ul>

<h2>Implementation Details</h2>
<p>The implementation follows these steps:</p>
<ol>
 <li>Dataset loading and feature extraction</li>
 <li>Train-test data splitting</li>
 <li>Feature standardization</li>
 <li>Initial logistic regression model implementation</li>
 <li>Grid search for optimal parameters</li>
 <li>Comprehensive performance evaluation</li>
 <li>Results visualization through confusion matrices</li>
</ol>

<h2>Key Findings</h2>
<ul>
 <li>The initial model achieved high accuracy (97%)</li>
 <li>Parameter optimization did not significantly improve performance</li>
 <li>Both models showed strong classification capabilities for benign and malignant tumors</li>
 <li>The optimized model found best performance with L1 penalty and C=1</li>
</ul>

<h2>Acknowledgments</h2>
<ul>
 <li>Breast Cancer Wisconsin Dataset contributors</li>
</ul>

<!------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->

<h1>Project 3 - Breast Cancer Classification using Hierarchical and K-Means Clustering</h1>
<p>This project implements both hierarchical and k-means clustering algorithms to analyze and classify breast cancer data using the Breast Cancer Wisconsin dataset. The analysis includes multiple linkage criteria for hierarchical clustering and parameter optimization for k-means clustering.</p>

<h2>Overview</h2>
<p>The code performs the following main tasks:</p>

<h3>1. Data Loading and Preprocessing:</h3>
<ul>
 <li>Loads the Breast Cancer Wisconsin dataset from sklearn</li>
 <li>Separates features and target variables</li>
 <li>Applies standard scaling to normalize the features</li>
 <li>Prepares data for both clustering approaches</li>
</ul>

<h3>2. Hierarchical Clustering Analysis:</h3>
<ul>
 <li>Implements clustering with multiple linkage criteria:
   <ul>
     <li>Ward linkage</li>
     <li>Complete linkage</li>
     <li>Average linkage</li>
     <li>Single linkage</li>
   </ul>
 </li>
 <li>Generates dendrograms for visual analysis</li>
 <li>Compares clustering structures across different methods</li>
</ul>

<h3>3. K-Means Clustering Analysis:</h3>
<ul>
 <li>Implements k-means clustering with optimization</li>
 <li>Evaluates performance using multiple metrics:
   <ul>
     <li>Inertia (Elbow method)</li>
     <li>Silhouette score</li>
     <li>Calinski-Harabasz score</li>
   </ul>
 </li>
 <li>Visualizes clustering results using PCA</li>
</ul>

<h3>4. Performance Evaluation:</h3>
<ul>
 <li>Compares clustering results with actual labels</li>
 <li>Generates confusion matrices</li>
 <li>Analyzes cluster quality metrics</li>
 <li>Provides visual representations of results</li>
</ul>

<h2>Results</h2>
<p>The analysis provides insights into the natural groupings within the breast cancer data:</p>
<ul>
 <li>Hierarchical clustering reveals the hierarchical structure of the data</li>
 <li>K-means clustering achieves optimal performance with k=2 clusters</li>
 <li>Results align well with the binary nature of the diagnosis (malignant/benign)</li>
</ul>

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
     <li>scipy</li>
   </ul>
 </li>
</ul>

<h2>Usage</h2>
<ol>
 <li>Ensure all required libraries are installed</li>
 <li>Run the Python script or Jupyter notebook</li>
 <li>View the generated visualizations and analysis results</li>
 <li>Check the summary metrics printed at the end of execution</li>
</ol>

<h2>Files</h2>
<ul>
 
 <li><code>Project3.ipynb</code>: Main Python script containing all code and analysis</li>
 <li>Dataset: Built-in Breast Cancer Wisconsin dataset from sklearn</li>
</ul>

<h2>Implementation Details</h2>
<p>The implementation follows these steps:</p>
<ol>
 <li>Data preprocessing with StandardScaler</li>
 <li>Hierarchical clustering implementation with four linkage methods</li>
 <li>K-means clustering with parameter optimization</li>
 <li>Performance evaluation using multiple metrics</li>
 <li>Visualization of results using:
   <ul>
     <li>Dendrograms</li>
     <li>PCA plots</li>
     <li>Confusion matrices</li>
     <li>Evaluation metric plots</li>
   </ul>
 </li>
</ol>

<!---------------------------------------------------------------------------------------------------------------------------------------4---------------------------------------------------------------------------------------------------------------------->
<h1>Project 4 - Parkinson's Disease Classification using Decision Trees and Random Forests</h1>
<p>This project implements Decision Tree (DT) and Random Forest (RF) classifiers to analyze and classify Parkinson's Disease patients using the <code>DB_Voice_Features.csv</code> dataset. The analysis includes parameter optimization, handling class imbalance, and performance evaluation.</p>

<h2>Overview</h2>
<p>The code performs the following main tasks:</p>

<h3>1. Data Loading and Preprocessing:</h3>
<ul>
 <li>Loads the <code>DB_Voice_Features.csv</code> dataset</li>
 <li>Separates features and target variables</li>
 <li>Splits data into training (80%) and testing (20%) sets</li>
 <li>Removes non-numeric columns ('name' and 'status')</li>
</ul>

<h3>2. Initial Model Implementation:</h3>
<ul>
 <li>Implements base Decision Tree classifier</li>
 <li>Implements base Random Forest classifier</li>
 <li>Evaluates initial performance metrics</li>
 <li>Generates performance visualization</li>
</ul>

<h3>3. Parameter Optimization:</h3>
<ul>
 <li>Uses <code>GridSearchCV</code> to find optimal parameters</li>
 <li>For Decision Tree:
   <ul>
     <li>Tests different max_depth values</li>
     <li>Optimizes min_samples_split and min_samples_leaf</li>
     <li>Evaluates different criterion options (gini, entropy)</li>
   </ul>
 </li>
 <li>For Random Forest:
   <ul>
     <li>Tests different n_estimators values</li>
     <li>Optimizes max_depth and min_samples parameters</li>
     <li>Implements cross-validation</li>
   </ul>
 </li>
</ul>

<h3>4. Performance Comparison:</h3>
<ul>
 <li>Evaluates model performance before and after optimization</li>
 <li>Analyzes impact of class balancing</li>
 <li>Compares Decision Tree vs Random Forest results</li>
 <li>Reports comprehensive performance metrics</li>
</ul>

<h2>Results</h2>
<p>The optimization and balancing processes improved the models' performance:</p>

<ul>
 <li>Decision Tree:
   <ul>
     <li>Initial accuracy: 92.31%</li>
     <li>Initial F1 Score: 95.38%</li>
     <li>Post-balancing F1 Score: 88.89%</li>
   </ul>
 </li>
 <li>Random Forest:
   <ul>
     <li>Optimized accuracy: 95.00%</li>
     <li>Post-balancing improvements across metrics</li>
     <li>Final F1 Score: 94.74%</li>
   </ul>
 </li>
</ul>

<p>The Random Forest model showed superior performance after optimization and balancing, indicating more reliable predictions across all metrics.</p>

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
 <li>Place the <code>DB_Voice_Features.csv</code> file in the same directory as the script</li>
 <li>Run the Python script</li>
 <li>View results and performance metrics</li>
</ol>

<h2>Files</h2>
<ul>
 <li><code>Project4.ipynb</code>: Main Python script containing all code and analysis</li>
 <li><code>DB_Voice_Features.csv</code>: Dataset containing voice feature measurements for PD classification</li>
</ul>

<h2>Implementation Details</h2>
<p>The implementation follows these steps:</p>
<ol>
 <li>Data preprocessing:
   <ul>
     <li>Loading and cleaning dataset</li>
     <li>Feature selection and extraction</li>
     <li>80-20 train-test split</li>
   </ul>
 </li>
 <li>Initial model implementation:
   <ul>
     <li>Base Decision Tree deployment</li>
     <li>Base Random Forest deployment</li>
     <li>Initial performance assessment</li>
   </ul>
 </li>
 <li>Parameter optimization:
   <ul>
     <li>Grid search implementation</li>
     <li>Cross-validation for both models</li>
     <li>Hyperparameter tuning</li>
   </ul>
 </li>
 <li>Class imbalance handling:
   <ul>
     <li>Minority class upsampling</li>
     <li>Model retraining with balanced data</li>
     <li>Performance reassessment</li>
   </ul>
 </li>
 <li>Final evaluation:
   <ul>
     <li>Comprehensive metric analysis</li>
     <li>Model comparison</li>
     <li>Results visualization</li>
   </ul>
 </li>
</ol>

<!---------------------------------------------------------------------------------------------------------------------------------------5---------------------------------------------------------------------------------------------------------------------->

<body>
<h1>Project 5 - Glioblastoma Single Cell RNA-seq Classification using SVM and MLP</h1>
<p>This project implements Support Vector Machine (SVM) and Multi-Layer Perceptron (MLP) classifiers to analyze and classify single-cell RNA sequencing data from Glioblastoma patients. The analysis includes handling class imbalance, parameter optimization, and comprehensive performance evaluation.</p>

<h2>Overview</h2>
<p>The code performs the following main tasks:</p>

<h3>1. Data Loading and Preprocessing:</h3>
<ul>
    <li>Loads the Glioblastoma Single Cell RNA-seq dataset</li>
    <li>Performs exploratory data analysis</li>
    <li>Visualizes class distribution</li>
    <li>Checks for missing values</li>
    <li>Applies standard scaling to normalize features</li>
</ul>

<h3>2. Initial Model Implementation:</h3>
<ul>
    <li>Implements basic SVM classifier</li>
    <li>Implements basic MLP classifier</li>
    <li>Evaluates initial performance metrics</li>
    <li>Generates classification reports</li>
    <li>Calculates accuracy, precision, recall, and F1 scores</li>
</ul>

<h3>3. Parameter Optimization:</h3>
<ul>
    <li>Uses GridSearchCV for hyperparameter tuning</li>
    <li>For SVM:
        <ul>
            <li>Tests different values for C parameter</li>
            <li>Evaluates different kernels (linear, rbf)</li>
            <li>Tests various gamma values</li>
        </ul>
    </li>
    <li>For MLP:
        <ul>
            <li>Optimizes hidden layer sizes</li>
            <li>Tests different activation functions</li>
            <li>Adjusts learning rate alpha</li>
        </ul>
    </li>
</ul>

<h3>4. Class Imbalance Handling:</h3>
<ul>
    <li>Implements SMOTE for balanced sampling</li>
    <li>Retrains models on balanced dataset</li>
    <li>Compares performance before and after balancing</li>
</ul>

<h3>5. Performance Comparison:</h3>
<ul>
    <li>Compares accuracy across all model variants</li>
    <li>Analyzes classification reports</li>
    <li>Evaluates impact of parameter optimization</li>
    <li>Assesses effectiveness of imbalance handling</li>
</ul>

<h2>Results</h2>
<p>The optimization and balancing process showed significant improvements:</p>

<ul>
    <li>SVM:
        <ul>
            <li>Initial accuracy: 100%</li>
            <li>Maintained perfect accuracy after optimization</li>
            <li>Possible indication of overfitting</li>
        </ul>
    </li>
    <li>MLP:
        <ul>
            <li>Initial accuracy: 97%</li>
            <li>Optimized accuracy: 99%</li>
            <li>Improved performance after parameter tuning</li>
        </ul>
    </li>
</ul>

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
            <li>imbalanced-learn (for SMOTE)</li>
        </ul>
    </li>
</ul>

<h2>Usage</h2>
<ol>
    <li>Ensure all required libraries are installed</li>
    <li>Load <code>Project5.ipynb</code> in Jupyter Notebook/Lab</li>
    <li>Run all cells sequentially</li>
    <li>Review results and visualizations</li>
</ol>

<h2>Files</h2>
<ul>
    <li><code>Project5.ipynb</code>: Main Jupyter notebook containing all code and analysis</li>
    <li><code>Data_Glioblastoma5Patients_SC.csv</code>: Dataset containing single-cell RNA-seq data</li>
</ul>

<h2>Implementation Details</h2>
<p>The implementation follows these steps:</p>
<ol>
    <li>Dataset loading and exploratory analysis</li>
    <li>Data preprocessing and scaling</li>
    <li>Initial model training (SVM and MLP)</li>
    <li>Grid search optimization</li>
    <li>Class imbalance handling using SMOTE</li>
    <li>Comprehensive performance evaluation</li>
    <li>Results visualization and comparison</li>
</ol>

<h2>Key Findings</h2>
<ul>
    <li>Class imbalance present in the original dataset</li>
    <li>SVM achieved perfect classification but potential overfitting</li>
    <li>MLP showed improvement after optimization</li>
    <li>Parameter tuning more effective for MLP than SVM</li>
    <li>SMOTE successfully addressed class imbalance issues</li>
</ul>

<h2>Acknowledgments</h2>
<ul>
    <li>Source of Glioblastoma Single Cell RNA-seq dataset</li>
    <li>Contributors to scikit-learn and imbalanced-learn libraries</li>
</ul>

<h2>Note on Results for Project 5</h2>
<p>The perfect accuracy achieved by the SVM classifier suggests potential overfitting and should be interpreted with caution. The MLP classifier showed more realistic performance improvements through optimization, indicating it might be more suitable for generalization to new data.</p>
</body>

<!---------------------------------------------------------------------------------------------------------------------------------------6---------------------------------------------------------------------------------------------------------------------->
<body>
<h1>Project 6 - Well Log Analysis and Vp Prediction using Machine Learning Models</h1>
<p>This project implements and compares four machine learning algorithms (MLP, SVM, Decision Tree, and Random Forest) in their ability to predict P-wave velocity (Vp) from well log data. The analysis is deliberately split into pre- and post-optimization phases to demonstrate the impact of parameter tuning on model performance.</p>

<h2>Analysis Overview</h2>
<p>The project was structured in two parts to examine the evolution of model performance:</p>

<h3>1. Pre-Optimization Phase:</h3>
<ul>
    <li>Initial Implementation:
        <ul>
            <li>MLP: Single hidden layer (100 neurons), default parameters</li>
            <li>SVM: RBF kernel, default C=1.0, epsilon=0.1</li>
            <li>Decision Tree: Default parameters</li>
            <li>Random Forest: 100 estimators</li>
        </ul>
    </li>
    <li>Baseline Performance:
        <ul>
            <li>MLP showed good initial performance with low MSE</li>
            <li>SVM demonstrated varied performance (R² 0.529 to 0.803)</li>
            <li>Decision Tree achieved high accuracy with R² close to 1</li>
            <li>Random Forest showed excellent performance with high R² values</li>
        </ul>
    </li>
</ul>

<h3>2. Post-Optimization Phase:</h3>
<ul>
    <li>Parameter Optimization:
        <ul>
            <li>MLP: Tested various hidden layer sizes, alpha values, and learning rates</li>
            <li>SVM: Optimized C, gamma, and epsilon parameters</li>
            <li>Decision Tree: Tuned max_depth, min_samples_split, and min_samples_leaf</li>
            <li>Random Forest: Adjusted n_estimators, max_depth, and min_samples_split</li>
        </ul>
    </li>
    <li>Optimization Impact:
        <ul>
            <li>MLP: Performance decreased, suggesting possible overfitting</li>
            <li>SVM: Significant improvement in both R² and MSE</li>
            <li>Decision Tree: Maintained high performance with slight improvements</li>
            <li>Random Forest: Maintained robust performance across parameter changes</li>
        </ul>
    </li>
</ul>

<h2>Key Findings</h2>
<ul>
    <li>Model Behavior:
        <ul>
            <li>Random Forest proved most robust and stable</li>
            <li>SVM showed highest improvement from optimization</li>
            <li>MLP demonstrated sensitivity to parameter changes</li>
            <li>Decision Tree maintained consistent performance</li>
        </ul>
    </li>
    <li>Parameter Sensitivity:
        <ul>
            <li>Random Forest: Least sensitive to parameter changes</li>
            <li>SVM: Most benefited from optimization</li>
            <li>MLP: Most sensitive to parameter adjustments</li>
            <li>Decision Tree: Moderately affected by optimization</li>
        </ul>
    </li>
</ul>

<h2>Methodology Benefits</h2>
<ul>
    <li>Educational Value:
        <ul>
            <li>Clear demonstration of optimization impact</li>
            <li>Understanding of model behavior patterns</li>
            <li>Insight into parameter sensitivity</li>
            <li>Real-world performance implications</li>
        </ul>
    </li>
    <li>Practical Insights:
        <ul>
            <li>Identification of most stable models</li>
            <li>Understanding of optimization necessity</li>
            <li>Resource allocation guidance</li>
            <li>Model selection criteria establishment</li>
        </ul>
    </li>
</ul>

<h2>Usage</h2>
<ol>
    <li>Ensure all required libraries are installed</li>
    <li>Place the well log data files (*.npy) in the working directory:
        <ul>
            <li><code>19A.npy</code></li>
            <li><code>BT2.npy</code></li>
            <li><code>F1B.npy</code></li>
            <li><code>SR.npy</code></li>
        </ul>
    </li>
    <li>Run both pre and post optimization scripts sequentially:
        <ul>
            <li>Part 1: Pre-optimization baseline analysis</li>
            <li>Part 2: Parameter optimization and comparison</li>
        </ul>
    </li>
    <li>View performance metrics and visualizations:
        <ul>
            <li>Depth vs. Vp plots</li>
            <li>MSE and R² scores</li>
            <li>Parameter optimization results</li>
        </ul>
    </li>
</ol>

<h2>Files</h2>
<ul>
    <li>Code Files:
        <ul>
            <li><code>Project6 Part1.ipynb</code>: Pre-optimization implementation with default parameters</li>
            <li><code>Project6 Part2.ipynb</code>: Post-optimization implementation with GridSearchCV</li>
        </ul>
    </li>
    <li>Data Files:
        <ul>
            <li><code>19A.npy</code>: Well log data from well 19A</li>
            <li><code>BT2.npy</code>: Well log data from well BT2</li>
            <li><code>F1B.npy</code>: Well log data from well F1B</li>
            <li><code>SR.npy</code>: Well log data from well SR</li>
        </ul>
    </li>
    <li>Output Files:
        <ul>
            <li>Visualization plots for each model and well combination</li>
            <li>Performance metrics in CSV format</li>
            <li>Optimization results summary</li>
        </ul>
    </li>
</ul>

<h2>Implementation Impact</h2>
<p>The two-phase approach revealed several important insights:</p>
<ul>
    <li>Performance Variations:
        <ul>
            <li>Not all models benefit equally from optimization</li>
            <li>Some models perform well with default parameters</li>
            <li>Optimization can sometimes degrade performance</li>
            <li>Model stability varies significantly</li>
        </ul>
    </li>
    <li>Practical Implications:
        <ul>
            <li>Random Forest: Best choice for robust performance</li>
            <li>SVM: Requires optimization for best results</li>
            <li>MLP: Needs careful parameter tuning</li>
            <li>Decision Tree: Good baseline performer</li>
        </ul>
    </li>
</ul>

<h2>Notes on Pre vs Post Optimization Results</h2>
<ul>
    <li>MLP Performance:
        <ul>
            <li>Pre: Good performance with low MSE across all wells</li>
            <li>Post: Decreased performance suggesting optimization led to overfitting</li>
        </ul>
    </li>
    <li>SVM Performance:
        <ul>
            <li>Pre: Moderate performance with R² ranging from 0.529 to 0.803</li>
            <li>Post: Significant improvement in both R² and MSE values</li>
        </ul>
    </li>
    <li>Decision Tree Performance:
        <ul>
            <li>Pre: High accuracy with R² values close to 1</li>
            <li>Post: Maintained high performance with minor improvements</li>
        </ul>
    </li>
    <li>Random Forest Performance:
        <ul>
            <li>Pre: Excellent performance with high R² values</li>
            <li>Post: Consistently high performance across different parameter settings</li>
            <li>Note: Longest computational time among all models</li>
        </ul>
    </li>
</ul>

<h2>Conclusions</h2>
<p>The project demonstrated that:</p>
<ul>
    <li>Parameter optimization's impact varies significantly across models</li>
    <li>Default parameters can sometimes provide optimal performance</li>
    <li>Model selection should consider optimization requirements</li>
    <li>Understanding baseline performance is crucial for effective model development</li>
</ul>


</body>
</html>
