library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)

train = vroom("AmazonEmployeeAccess/train.csv")
test = vroom("AmazonEmployeeAccess/test.csv")
train$ACTION = as.factor(train$ACTION)

my_recipe = recipe(ACTION ~ ., data = train) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> 
  step_other(all_nominal_predictors(), threshold = 0.001) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

rf_model = rand_forest(mtry = tune(),
                       min_n = tune(),
                       trees = 50) |> 
  set_mode("classification") |> 
  set_engine("ranger")

wf = workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(rf_model)

tuning_grid = grid_regular(mtry(range = c(1, 8)),
                           min_n(),
                           levels = 5)

folds = vfold_cv(train, v = 10, repeats = 1)

CV_results = wf |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune = CV_results |> 
  select_best()

final_wf = wf |> 
  finalize_workflow(bestTune) |> 
  fit(data = train)

predictions = predict(final_wf,
                      new_data = test,
                      type = "prob")

kaggle_submission = bind_cols(test["id"], predictions[".pred_1"]) |> 
  rename("Id" = id, "Action" = .pred_1)
#kaggle_submission

vroom_write(x = kaggle_submission, 
            file = "AmazonEmployeeAccess/rfpred.csv",
            delim = ",")




