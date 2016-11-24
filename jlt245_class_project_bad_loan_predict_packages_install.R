# sudo yum install libxml2-devel libxslt-devlop
#

setwd("~/class-project")
install.packages(c("devtools", "stringr"), repos = "http://cran.us.r-project.org")
library(devtools)

install.packages(c("xml2", "rvest"), repos = "http://cran.us.r-project.org")

install.packages(c("tidyverse", "reshape", "data.table"), repos = "http://cran.us.r-project.org")
install.packages(c("h2o", "ROCR"), repos = "http://cran.us.r-project.org")

install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")
