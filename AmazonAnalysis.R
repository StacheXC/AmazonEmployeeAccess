library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)

train = vroom("AmazonEmployeeAccess/train.csv")
test = vroom("AmazonEmployeeAccess/test.csv")
train$ACTION = as.factor(train$ACTION)

my_recipe = recipe(ACTION ~ ., data = train) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> 
  step_other(all_nominal_predictors(), threshold = 0.1) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |> 
  step_zv() |> 
  step_normalize(all_numeric_predictors()) |> 
  step_pca(all_predictors(), threshold = 0.8)

#prep = prep(my_recipe)
#baked = bake(prep, new_data = train)
#baked

# Penalized Regression
logRegModel = logistic_reg(mixture = tune(), penalty = tune()) |> 
  set_engine("glmnet")

wf = workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(logRegModel)

tuning_grid = grid_regular(penalty(),
                           mixture(),
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

#predictions

kaggle_submission = bind_cols(test["id"], predictions[".pred_1"]) |> 
  rename("Id" = id, "Action" = .pred_1)
#kaggle_submission

vroom_write(x = kaggle_submission, 
            file = "logregpred2.csv",
            delim = ",")




