By: Jordan D. Bailey
# QABF-ML
Machine learning code for predicting behavioral function
Download the QABF-Final.py file and/or the QABF-crossval.py file along with the RealdataIntensity.csv file. 
The csv file should be saved in the same directory as either python file. 
In order to run you need python 3.0+ and the following python libraries: sci-kit learn, keras, numpy, pandas
These can be more easily obtained by downloading and installing Anaconda or Mini-conda: https://www.anaconda.com
Run the file by opening your terminal (Mac) or or command prompt (Windows) and typing the following:

python QABF-Finaly.py 
OR
python QABF-crossval.py




-- Cleaned Up LDA and Logistic R File QABF Predictions.R -- 
Last updated: 2 Sep 2019
By: Mark J. Rzeszutek

All of the R code that generated the data found in experiment 3 of "Evaluation of an Artificial Neural Network for Supplementing Behavioral Assessment"
can be found in the R file named "Cleaned Up LDA and Logistic R File QABF Predictions". 

To run all code with no extra steps, download the project file, R file, and QABFData.csv into a folder. Then, open the project file in RStudio, open the R file, press ctrl+shift+s, and the entire
R script will run and assign all variables. 

If the MASS package is not installed, you will be required to install. You can do so with install.package("MASS"). 

To run interactively, run lines in order using ctrl+enter. At least the variables from cell 1 must be run for the code to work.  
Cells can also be run on their own.

After a variable is assigned, calling it in the console or clicking on it in the environment viewer will call it. 

The code is organized in the following way. If the file is opened in RStudio, each section is commented into a cell for easy navigation.  
1. Data upload/cleaning/variable defining for convenience
2. LDA multiclass function and results
3. Logistic endorsement LOOCV and results
4. Logistic intensity LOOCV and results
5. Logistic combined endorsement and intensity LOOCV and results
6. Logistic with all predictors LOOCV and results
7. LDA Binary Accuracy
8. Full logistic models

Variables and results are named to correspond with the logical outcomes of the relevant sections. 

For full models, p-values and parameters can be obtained by using the summary(x) function, where x is the name of the
model of interest. 
