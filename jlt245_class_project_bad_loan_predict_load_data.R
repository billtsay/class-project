setwd("~/class-project")

library(lubridate)

library(readr)
library(dplyr)

filenames <- c("LoanStats_2016Q1.csv", "LoanStats_2016Q2.csv", "LoanStats_2016Q3.csv")

# skip the first line in each file. The first line is just a comment of LC.
data_list <- lapply(filenames, function (x) read_csv(file=x, skip=1))

data_frame <- do.call(rbind, data_list)
dim(data_frame)

write.csv(data_frame, file = "jlt245_class_project_bad_loan_predict_raw_data.csv", row.names = FALSE)





