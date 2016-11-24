# class-project: Predicting bad loans using deep learning.

%% Introduction

The lending club has posted its loan information dataset online, which allows freely downloaded by any interested parties. In this class project, I would like to take them as the dataset to explore the deep learning and ensemble learning for the project.

%% Install the packages used

h2o R package and related are the major part of installation. The execution environment needs java pre-installed.

Run the script

 Rscript jlt245_class_project_bad_loan_predict_packages_install.R



%% Data loading and Cleaning

The datasets are from https://www.lendingclub.com/info/download-data.action, they are three in the project folder:

LoanStats_2016Q1.csv
LoanStats_2016Q2.csv
LoanStats_2016Q3.csv

Run the script:

 Rscript jlt245_class_project_bad_loan_predict_load_data.R

will merge them and generate a single raw data file:

 jlt245_class_project_bad_loan_predict_raw_data.csv

They are totally 330867 records with 111 dimensions or fields.

%% Feature Selection

Obviously we are not going to process all those 111 dimensions in the raw dataset. What I did was to use a R script to convert fields into features that we will process the learning algorithms against them and narrow down the dimensions.

Run the script:

 Rscript jlt245_class_project_bad_loan_predict_feature_selection.R

to generate the interested loan data with 20 features. The first is "bad_loan" indicator, others are the features of data to predict "bad_loan". The final outcome is the loan dataset: jlt245_class_project_bad_loan_predict_loan_data.csv.

Be aware that we need some features with converted data from original raw set such as:

  bad_loan = ifelse(loan_status=="Charged Off", 1, 0),
  issue_d = mdy(issue_d),
  earliest_cr_line = mdy(earliest_cr_line),
  time_history = as.numeric(issue_d - earliest_cr_line),
  revol_util = as.numeric(sub("%", "", revol_util)),
  emp_listed = as.numeric(!is.na(emp_title) * 1),
  empty_desc = as.numeric(is.na(desc)),
  emp_na = ifelse(emp_length == "n/a", 1, 0),
  emp_length = ifelse(emp_length == "< 1 year" | emp_length == "n/a", 0, emp_length) and emp_length = as.numeric(gsub("\\D", "", emp_length)),
  delinq_ever = as.numeric(!is.na(mths_since_last_delinq)),
  home_ownership = ifelse(home_ownership == "NONE", "OTHER", home_ownership))

The 20 features I have chosen for this project are:

bad_loan, loan_amnt, empty_desc, emp_listed, emp_na, emp_length, verification_status, home_ownership, annual_inc, purpose, time_history, inq_last_6mths, open_acc, pub_rec, revol_util, dti, total_acc, delinq_2yrs, delinq_ever, int_rate

There are always possibilities to improve the prediction by experimenting the various selection of features. 

Also PCA analysis may help such process too.

%% Deep Learning

With the chosen dataset of 20 dimensions, we can run deep learning algorithms in h2o R package.

The deep learning algorithm I am using is an ANN with multiple hidden layers in between, for example:

# perform a 5-fold cross-validation deep learning model and validate on a test set
model <- h2o.deeplearning (
  x = x,
  y = y,
  training_frame = train,
  validation_frame = test,
  distribution = "multinomial",
  activation = "RectifierWithDropout",
  hidden = c(64, 128, 64),
  input_dropout_ratio = 0.2,
  l1 = 1e-5,
  epochs = 10,
  nfolds = 5
)

That is a 5-fold CV deep learning model with three hidden layers, 1st of 64 nodes, 2nd is 128 nodes and 3rd is 64 layers. The ratio of train and test sets is 3:1.


Run the script:

 Rscript jlt245_class_project_bad_loan_predict_deep_learning.R

 **be aware that the run time may take a vew hours to complete the above script.**

Comparisons of performances:

[1] 5-fold CV, hidden = 64, 128, 64 nodes

  H2OBinomialMetrics: deeplearning

  MSE:  0.003277576
  RMSE:  0.05725012
  LogLoss:  0.02733859
  Mean Per-Class Error:  0.4864729
  AUC:  0.7231921
  Gini:  0.4463842

[2] 10-fold CV, hidden = 64, 128, 64 nodes

  H2OBinomialMetrics: deeplearning

  MSE:  0.003278861
  RMSE:  0.05726134
  LogLoss:  0.03098881
  Mean Per-Class Error:  0.4469392
  AUC:  0.7248728
  Gini:  0.4497456

 * 10-fold CV seems to improve a bit, but not that much in this case: AUC from 0.7232 to 0.7249 *


[3] 5-fold CV, hidden = 64, 128, 2, 128, 64 nodes

  H2OBinomialMetrics: deeplearning

  MSE:  0.003276932
  RMSE:  0.0572445
  LogLoss:  0.02568059
  Mean Per-Class Error:  0.5
  AUC:  0.5
  Gini:  0

 * Not good for 5 hidden layers in this case *


%% Ensemble Method

Ensemble method are learning approach that train multiple learners and combine them for use, with
Boosting and Bagging as representatives in order to achieve better performance.

h2o R packages provide a rich set of ML algorithms to use, I will try two Ensembles, one with 
popular 4 algorithms: GLM, Random Forest, GBM and Deep Learning. The other I will repeat these four
with more parameter changes for each.

Run the script:
 
 Rscript jlt245_class_project_bad_loan_predict_ensemble.R


The results are:

  Base learner performance, sorted by specified metric:
                     learner       AUC
  4 h2o.deeplearning.wrapper 0.5256778
  2 h2o.randomForest.wrapper 0.5853513
  1          h2o.glm.wrapper 0.6195399
  3          h2o.gbm.wrapper 0.6588160

  H2O Ensemble Performance on <newdata>:
  ----------------
  Family: binomial

  Ensemble performance (AUC): 0.635881662244555

and further algorithms case, the results are:

  Base learner performance, sorted by specified metric:
                learner       AUC
  14 h2o.deeplearning.2 0.5263948
  13 h2o.deeplearning.1 0.5401991
  15 h2o.deeplearning.3 0.5490740
  9           h2o.gbm.3 0.6138080
  1           h2o.glm.1 0.6183734
  2           h2o.glm.2 0.6195399
  3           h2o.glm.3 0.6208864
  4  h2o.randomForest.1 0.6322672
  6  h2o.randomForest.3 0.6417271
  5  h2o.randomForest.2 0.6447206
  8           h2o.gbm.2 0.6488665
  7           h2o.gbm.1 0.6501414
  10          h2o.gbm.4 0.6532700
  11          h2o.gbm.5 0.6532700
  12          h2o.gbm.6 0.6552727


  H2O Ensemble Performance on <newdata>:
  ----------------
  Family: binomial

  Ensemble performance (AUC): 0.644141126382818

 * as can be seen that Ensemble performance is above the average of the performance from all base algorithms *


%% Conclusion

A few key points of achieving better performance are:

1. feature selection - I did try some additional fields or narrow down to less fields to come up different 
results.
2. model selection - deep learning comes with some alternatives such as distribution, hidden layers etc that
impact the performance. In addition, different algorithms run against different datasets may achieve different
performance.
3. Ensemble method - seems to work out an average performance among the choosen base algorithms.

 

%% References

[1] https://h2o-release.s3.amazonaws.com/h2o/rel-turan/4/docs-website/h2o-docs/booklets/R_Vignette.pdf
[2] https://h2o-release.s3.amazonaws.com/h2o/rel-tibshirani/8/docs-website/h2o-docs/booklets/DeepLearning_Vignette.pdf
[3] http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html
[4] http://www.dataversity.net/efficient-machine-learning-h2o-r-python-part-1/
[5] Ensemble Methods (Foundations and Algorithms) by Zhi-Hua Zhou
[6] http://www.stat.berkeley.edu/~ledell/R/h2oEnsemble.pdf
[7] https://rdrr.io/cran/h2o/man/h2o.prcomp.html
[8] https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/


