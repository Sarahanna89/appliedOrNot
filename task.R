## needed packages
library(mlr)
library(randomForest)

## import both dataframes
job_desc <- read.csv("job_desc.csv")
user <- read.csv("user.csv")


##########################
## descriptive analysis ##
##########################

## first overview of data (in particular variable(-names) and scaling)
str(job_desc)

summary(job_desc)
table(job_desc$company)
levels(job_desc$user_id)                 # conspicuous: user only visited once                           
hist(job_desc$salary)
sort(table(job_desc$job_title_full))     # too many levels

str(user)
levels(user$user_id)                                           
par(ask=TRUE) 
lapply(user[,3:58], hist)  # conspicuous: all features equally distributed
# no logarithm needed

## first data preprocessing: target "has_applied" is binary and should be a factor
user$has_applied <- factor(user$has_applied, levels = c("0","1"), labels = c("no", "yes"))
table(user$has_applied)
table(user$has_applied)/2000
 
## combine datasets (by "user_id")
data <- merge(user, job_desc, by = "user_id")

## 'user_id' irrelevant for further analysis
data <- data[, !(colnames(data) %in% "user_id")]
str(data)

## number of missing values
sapply(data, function(x) sum(is.na(x)))

## pairwise correlations of numeric features
cor_matrix <- cor(data[, sapply(data, is.numeric)], use = "pairwise.complete.obs")
range(cor_matrix - diag(nrow(cor_matrix)))
#[1] -0.08605054  0.10615617

## no high correlations between features
## methods for dimension reduction (PCA) not very helpful 



########################
## Data preprocessing ##
########################

##### new variables ############################################################

## create new targets from "job_title_full" with more specific information
job <- as.character(data$job_title_full)
job

## Experience needed for job: Junior/Senior/Lead
## extract first word of string (if present: job titles without this information start with a space, in this case "NA")
job_title_experience <- factor(ifelse(grepl("^\\w", job), gsub("^(\\w+) .*", "\\1", job), "NA"))
table(job_title_experience)

## accentuation of "m/f/d" present or not 
job_title_mfd <- factor(grepl("m/f/d", tolower(job)))
table(job_title_mfd)

## type/name of job position (without experience status):
## split at "-" and filter text before
temp <- sapply(strsplit(job, "-"), "[", 1)
## delete experience information (if present)
job_title_type <- trimws(gsub("^\\w* (.*)$", "\\1", temp))
## delete m/f/d information
job_title_type <- factor(gsub(" *[(]*[mM]/[fF]/[dD][)]*", "", job_title_type))
table(job_title_type)

## some extra information in job_title_full:
## split at "-" and filter text after
temp2 <- trimws(sapply(strsplit(job, "-"), "[", 2))
## delete m/f/d information
job_title_extra <- gsub(" *[(]*[mM]/[fF]/[dD][)]*", "", temp2)
## "NA" as class
job_title_extra[is.na(job_title_extra)] <- "NA"
job_title_extra <- factor(job_title_extra)
table(job_title_extra)

## add new variables
data <- cbind(data, job_title_experience, job_title_mfd, job_title_type, job_title_extra) 

## remove "job_title_full"
data <- data[, !(names(data) %in% "job_title_full")]


#### handling of missing values ################################################

## Challenging:
## many missing values (probably not missing at random, no further information
## on anonymized features): problematic for many machine learning models 
## Idea: (dummies) binary variables and use information of interactions

## filter numeric features with missing values
numeric_NA <- sapply(data, function(x) is.numeric(x) && sum(is.na(x)))
names_numeric_NA <- names(data)[numeric_NA]

data_ext <- data
## create binary feature (0: missing, 1: not missing) for every respective
## numeric feature
dummy_data <- NULL 
interaction_with_dummy_data <- NULL                 
for(i in names_numeric_NA) {
  dummy <- as.numeric(!is.na(data[,i]))
  dummy_data <- cbind(dummy_data, dummy)
  data_ext[, i][is.na(data_ext[, i])] <- 0  
  # after that: set NA to 0 -> is equivalent to interaction of original variable and dummy 
   names(data_ext)[names(data_ext) == i] <- paste0("interaction_", i, "_available")
}
colnames(dummy_data) <- paste(names_numeric_NA, "available", sep = "_")
dummy_data <- data.frame(dummy_data)

## add new features to dataframe
data_ext <- cbind(data_ext, dummy_data)
data_ext <- data_ext[, c("has_applied", sort(names(data_ext)[-1]))]

## sort columns 
str(data_ext)
data_ext <- data_ext[, c(1:2, 57:61, 3:56, 62:117)]



###############
## modelling ##
###############

## define classification task on data_ext
task_data_ext <- makeClassifTask(id = "data_ext", data = data_ext, target = "has_applied")

## define some initial learners: predict.type = "prob" for AUC
lrn_rpart <- makeLearner("classif.rpart", predict.type = "prob")
lrn_lda <- makeLearner("classif.lda", predict.type = "prob")
lrn_LogReg <- makeLearner("classif.logreg", predict.type="prob")
lrn_RF <- makeLearner("classif.randomForest", predict.type = "prob", importance = TRUE)
lrn_nnet <- makeLearner("classif.nnet", predict.type = "prob")
lrn_boosting <- makeLearner("classif.ada", predict.type = "prob")
lrn_boosting2 <- makeLearner("classif.boosting", predict.type = "prob")
lrn_boosting2 <- setHyperPars(lrn_boosting2, coeflearn = "Freund")

## define resampling strategy (10-fold cross validation) and fixed training 
## and test sets (set of integer vectors) for every fold 
set.seed(2020)
rdesc_cv10_data <- makeResampleDesc(method = "CV", iters = 10)
rinst_cv10_data <- makeResampleInstance(rdesc_cv10_data, task_data_ext)

## measures
measures <- list(acc, auc)

## initial benchmark on task_data_ext (no tuning of parameters in first step)
set.seed(2020)
benchmark_data_ext <- benchmark(learners = list(lrn_rpart, lrn_lda, lrn_LogReg, lrn_RF, lrn_nnet, lrn_boosting, lrn_boosting2), 
                                tasks = task_data_ext, 
                                resamplings = rinst_cv10_data, 
                                measures = measures)
#   task.id           learner.id acc.test.mean auc.test.mean
#1 data_ext        classif.rpart        0.6655     0.6644501
#2 data_ext          classif.lda        0.6490     0.6958887
#3 data_ext       classif.logreg        0.6500     0.6956917
#4 data_ext classif.randomForest        0.6720     0.7567295 -> tuning
#5 data_ext         classif.nnet        0.6245     0.6397298
#6 data_ext          classif.ada        0.6860     0.7500161 -> tuning
#7 data_ext     classif.boosting        0.6430     0.6912086


## test ROC curve for rpart
mod_RF <- train(learner = lrn_RF, task = task_data_ext)
p_RF <- predict(mod_RF, task = task_data_ext) 
ROC_RF <- generateThreshVsPerfData(p_RF, measures = list(fpr, tpr, mmce, auc))
plotROCCurves(ROC_RF)



###################
## Random Forest ##
###################

## Focus on Random Forest

#### importance of variables ###################################################
mod_RF <- train(learner = lrn_RF, task = task_data_ext)
getFeatureImportance(mod_RF)
varImpPlot(getLearnerModel(mod_RF), main = "Random Forest")


#### ROC curves resampling #####################################################

set.seed(2020)
resample_RF <- resample(learner = lrn_RF, task = task_data_ext, 
                        resampling = rinst_cv10_data, 
                        measures = list(fpr, tpr, mmce, acc, auc), show.info = TRUE)

par(mfrow = c(1,2))
ROC_resample_RF <- generateThreshVsPerfData(resample_RF, list(fpr, tpr), aggregate = FALSE)
ROC_resample_RF_aggr <- generateThreshVsPerfData(resample_RF, list(fpr, tpr), aggregate = TRUE)
plotROCCurves(ROC_resample_RF)                                  
plotROCCurves(ROC_resample_RF_aggr)  


#### tuning ####################################################################

## Random Forest seems to be good:
## further tuning of parameters for optimizing AUC

getLearnerParamSet(lrn_RF)
#                     Type  len   Def   Constr Req Tunable Trafo
#ntree             integer    -   500 1 to Inf   -    TRUE     -
#mtry              integer    -     - 1 to Inf   -    TRUE     -
#replace           logical    -  TRUE        -   -    TRUE     -
#classwt     numericvector <NA>     - 0 to Inf   -    TRUE     -
#cutoff      numericvector <NA>     -   0 to 1   -    TRUE     -
#strata            untyped    -     -        -   -   FALSE     -
#sampsize    integervector <NA>     - 1 to Inf   -    TRUE     -
#nodesize          integer    -     1 1 to Inf   -    TRUE     -
#maxnodes          integer    -     - 1 to Inf   -    TRUE     -
#importance        logical    - FALSE        -   -    TRUE     -
#localImp          logical    - FALSE        -   -    TRUE     -
#proximity         logical    - FALSE        -   -   FALSE     -
#oob.prox          logical    -     -        -   Y   FALSE     -
#norm.votes        logical    -  TRUE        -   -   FALSE     -
#do.trace          logical    - FALSE        -   -   FALSE     -
#keep.forest       logical    -  TRUE        -   -   FALSE     -
#keep.inbag        logical    - FALSE        -   -   FALSE     -

## important parameters: ntree (number of trees), 
## mtry (number of variables randomly sampled as candidates at each split)

## nested cross validation (try to avoid overfitting): estimate error/AUC
## use AUC as optimization criteria

## use 3-fold cross-validation in inner loop (tuning of parameters)
## and 10-fold cross-validation in outer loop (determination of AUC) 
set.seed(2020)
rdesc_cv3_data <- makeResampleDesc("CV", iters = 3)

## use grid Search on parameters "mtry" and "ntree"
parameter_to_tune_RF <- makeParamSet(
  makeIntegerParam("mtry", lower = 2, upper = 11),
  makeIntegerParam("ntree", lower = 500, upper = 2000)
)  

ctrl_RF <- makeTuneControlGrid(resolution = c(mtry = 10, ntree = 4)) 

wrapper_RF_tune <- makeTuneWrapper(lrn_RF, resampling = rdesc_cv3_data, 
                                   par.set = parameter_to_tune_RF, measures = list(auc), 
                                   control = ctrl_RF, show.info = TRUE)


## benchmark experiment (AUC with tuning)
set.seed(2020)
benchmark_RF_tune <- benchmark(wrapper_RF_tune, tasks = task_data_ext, 
                               resamplings = rinst_cv10_data, measures = measures, 
                               show.info = TRUE)
benchmark_RF_tune
#   task.id                 learner.id acc.test.mean auc.test.mean
#1 data_ext classif.randomForest.tuned         0.663     0.7576018


## tuning (3-fold cross-validation -> as in inner loop above)
set.seed(2020)
lrn_RF_tune <- tuneParams(lrn_RF, task = task_data_ext, resampling = rdesc_cv3_data,
                          par.set = parameter_to_tune_RF, control = ctrl_RF,
                          show.info = TRUE)







