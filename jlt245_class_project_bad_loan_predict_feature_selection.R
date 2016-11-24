setwd("~/class-project")

library(readr)
library(dplyr)
library(lubridate)

# Import the raw data
loan_csv <- "jlt245_class_project_bad_loan_predict_raw_data.csv"

data <- read.csv(loan_csv)

dim(data)

# "id","member_id","loan_amnt","funded_amnt","funded_amnt_inv","term","int_rate","installment","grade",
# "sub_grade","emp_title","emp_length","home_ownership","annual_inc","verification_status","issue_d",
# "loan_status","pymnt_plan","url","desc","purpose","title","zip_code","addr_state","dti","delinq_2yrs",
# "earliest_cr_line","inq_last_6mths","mths_since_last_delinq","mths_since_last_record","open_acc","pub_rec",
# "revol_bal","revol_util","total_acc","initial_list_status","out_prncp","out_prncp_inv","total_pymnt","total_pymnt_inv",
# "total_rec_prncp","total_rec_int","total_rec_late_fee","recoveries","collection_recovery_fee","last_pymnt_d",
# "last_pymnt_amnt","next_pymnt_d","last_credit_pull_d","collections_12_mths_ex_med","mths_since_last_major_derog",
# "policy_code","application_type","annual_inc_joint","dti_joint","verification_status_joint","acc_now_delinq",
# "tot_coll_amt","tot_cur_bal","open_acc_6m","open_il_6m","open_il_12m","open_il_24m","mths_since_rcnt_il","total_bal_il",
# "il_util","open_rv_12m","open_rv_24m","max_bal_bc","all_util","total_rev_hi_lim","inq_fi","total_cu_tl","inq_last_12m",
# "acc_open_past_24mths","avg_cur_bal","bc_open_to_buy","bc_util","chargeoff_within_12_mths","delinq_amnt",
# "mo_sin_old_il_acct","mo_sin_old_rev_tl_op","mo_sin_rcnt_rev_tl_op","mo_sin_rcnt_tl","mort_acc","mths_since_recent_bc",
# "mths_since_recent_bc_dlq","mths_since_recent_inq","mths_since_recent_revol_delinq","num_accts_ever_120_pd",
# "num_actv_bc_tl","num_actv_rev_tl","num_bc_sats","num_bc_tl","num_il_tl","num_op_rev_tl","num_rev_accts",
# "num_rev_tl_bal_gt_0","num_sats","num_tl_120dpd_2m","num_tl_30dpd","num_tl_90g_dpd_24m","num_tl_op_past_12m",
# "pct_tl_nvr_dlq","percent_bc_gt_75","pub_rec_bankruptcies","tax_liens","tot_hi_cred_lim","total_bal_ex_mort",
# "total_bc_limit","total_il_high_credit_limit"
#

# feature selection and data conversions for the features we are interested in.
loan_data <- data %>%
  mutate(bad_loan = ifelse(loan_status=="Charged Off", 1, 0),
         issue_d = mdy(issue_d),
         earliest_cr_line = mdy(earliest_cr_line),
         time_history = as.numeric(issue_d - earliest_cr_line),
         revol_util = as.numeric(sub("%", "", revol_util)),
         emp_listed = as.numeric(!is.na(emp_title) * 1),
         empty_desc = as.numeric(is.na(desc)),
         emp_na = ifelse(emp_length == "n/a", 1, 0),
         emp_length = ifelse(emp_length == "< 1 year" | emp_length == "n/a", 0, emp_length),
         emp_length = as.numeric(gsub("\\D", "", emp_length)),
         delinq_ever = as.numeric(!is.na(mths_since_last_delinq)),
         home_ownership = ifelse(home_ownership == "NONE", "OTHER", home_ownership)) %>%
  select(bad_loan, loan_amnt, empty_desc, emp_listed, emp_na, emp_length, verification_status, home_ownership,
         annual_inc, purpose, time_history, inq_last_6mths, open_acc, pub_rec, revol_util, dti, total_acc,
         delinq_2yrs, delinq_ever, int_rate)

(ldd <- dim(loan_data))
colnames(loan_data)

##
# Principal components analysis of an H2O data frame using the power method to calculate the singular value decomposition of the Gram matrix.
##
# The output for PCA includes the following:
#   Model parameters
# Output (model category, model summary, scoring history, training metrics, validation metrics, iterations)
# Importance of components
# Training metrics
# Rotation
# Preview POJO
library(h2o)
h2o.init(nthreads = -1, max_mem_size = "16G")

data.hex <- as.h2o(loan_data)

# try different transform "STANDARDIZE", "NORMALIZE"
# try pca_method as "GramSVD" and "NONE"
pca_model <- h2o.prcomp(training_frame = data.hex, k = ldd[2], transform = "STANDARDIZE", pca_method = "GramSVD")

pca_model@model

pca_model@model$importance
# Importance of components: 
#   pc1      pc2      pc3      pc4      pc5      pc6      pc7      pc8
# Standard deviation     1.533944 1.475700 1.177659 1.144607 1.027045 1.004207 1.000312 0.998317
# Proportion of Variance 0.119685 0.110769 0.070544 0.066640 0.053654 0.051294 0.050897 0.050694
# Cumulative Proportion  0.119685 0.230454 0.300998 0.367637 0.421291 0.472585 0.523482 0.574176
# pc9     pc10     pc11     pc12     pc13     pc14     pc15     pc16
# Standard deviation     0.964081 0.935521 0.931754 0.891024 0.872536 0.868213 0.816371 0.797537
# Proportion of Variance 0.047277 0.044517 0.044159 0.040383 0.038725 0.038342 0.033900 0.032354
# Cumulative Proportion  0.621453 0.665970 0.710130 0.750513 0.789237 0.827579 0.861479 0.893832
# pc17     pc18     pc19     pc20
# Standard deviation     0.572536 0.508518 0.501796 0.334634
# Proportion of Variance 0.016673 0.013153 0.012808 0.005696
# Cumulative Proportion  0.910506 0.923659 0.936467 0.942163

write.csv(loan_data, file="jlt245_class_project_bad_loan_predict_loan_data.csv", row.names = FALSE)

h2o.shutdown(prompt=FALSE)


# references:
# [1] https://rdrr.io/cran/h2o/man/h2o.prcomp.html
# [2] https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/

