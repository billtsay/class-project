setwd("~/class-project")

library(h2o)
h2o.init(nthreads = -1, max_mem_size = "16G")

# Import the data
loan_csv <- "jlt245_class_project_bad_loan_predict_loan_data.csv"

data <- h2o.importFile(loan_csv)  
dim(data)

data$bad_loan <- as.factor(data$bad_loan)  #encode the binary repsonse as a factor

# Partition the data into train and test sets
splits <- h2o.splitFrame(data, seed = 1111)
train <- splits[[1]]
test <- splits[[2]]

dim(train)
dim(test)

# Identify response and predictor variables
y <- "bad_loan"
x <- setdiff(names(data), y)

# encode the response column as categorical for multinomial classification
train[,y] <- as.factor(train[,y])
test[,y] <- as.factor(test[,y])

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

predictions <- predict(object = model, newdata = test)
perf <- h2o.performance(model, test)
perf


# perform a 10-fold cross-validation deep learning model and validate on a test set
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
  nfolds = 10
)

predictions <- predict(object = model, newdata = test)
perf <- h2o.performance(model, test)
perf

# perform a 5 hidden layers with 5-fold cross-validation deep learning model and validate on a test set
model <- h2o.deeplearning (
  x = x,
  y = y,
  training_frame = train,
  validation_frame = test,
  distribution = "multinomial",
  activation = "RectifierWithDropout",
  hidden = c(64, 128, 2, 128, 64),
  input_dropout_ratio = 0.2,
  l1 = 1e-5,
  epochs = 10,
  nfolds = 5
)

predictions <- predict(object = model, newdata = test)
perf <- h2o.performance(model, test)
perf


# Train Deep Learning model and variables on test set
# and save the variable importances
model_vi <- h2o.deeplearning (
  x = x,
  y = y,
  training_frame = train,
  validation_frame = test,
  distribution = "multinomial",
  activation = "RectifierWithDropout",
  hidden = c(64, 128, 64),
  input_dropout_ratio = 0.2,
  sparse = TRUE,
  l1 = 1e-5,
  variable_importances = TRUE,
  epochs = 10
)

model_vi

varimp <- h2o.varimp(model_vi)
varimp

hidden_opt <- list(c(128, 128), c(64, 128, 64), c(128, 256, 128))
l1_opt <- c(1e-4, 1e-6)
hyper_params <- list(hidden = hidden_opt, l1 = l1_opt)

model_grid <- h2o.grid(
  "deeplearning",
  hyper_params = hyper_params,
  x = x,
  y = y,
  distribution = "multinomial",
  training_frame = train,
  validation_frame = test,
  score_interval = 2,
  epochs = 1024,
  stopping_rounds = 3,
  stopping_tolerance = 0.05,
  stopping_metric = "misclassification"
)

model_grid

for (model_id in model_grid@model_ids) {
  model <- h2o.getModel(model_id)
  mse <- h2o.mse(model, valid = TRUE)
  print (sprintf("Test set MSE: %f", mse))
}

h2o.shutdown(prompt=FALSE)


# references:
# [1] https://h2o-release.s3.amazonaws.com/h2o/rel-turan/4/docs-website/h2o-docs/booklets/R_Vignette.pdf
# [2] https://h2o-release.s3.amazonaws.com/h2o/rel-tibshirani/8/docs-website/h2o-docs/booklets/DeepLearning_Vignette.pdf
# [3] http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html
# [4] http://www.dataversity.net/efficient-machine-learning-h2o-r-python-part-1/



