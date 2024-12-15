library(tidymodels)
library(vroom)
library(embed)

train = vroom("AmazonEmployeeAccess/train.csv")
test = vroom("AmazonEmployeeAccess/test.csv")
train$ACTION = as.factor(train$ACTION)

my_recipe = recipe(ACTION ~ ., data = train) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> 
  step_other(all_nominal_predictors(), threshold = 0.001) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |> 
  step_normalize(all_predictors()) |> 
  step_pca(all_predictors(), threshold = 0.8) |> 
  step_normalize(all_predictors())

prep = prep(my_recipe)
baked = bake(prep, new_data = train)
baked

linear_model = svm_linear(cost=0.000977) |> 
  set_mode("classification") |> 
  set_engine("kernlab")

nb_wf = workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(linear_model) |> 
  fit(data = train)

predictions = predict(nb_wf,
                      new_data = test,
                      type = "prob")






tuning_grid = grid_regular(cost(),
                           levels = 5)

folds = vfold_cv(train, v = 10, repeats = 1)

CV_results = nb_wf |> 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

bestTune = CV_results |>  
  select_best()

final_wf = nb_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data = train)

predictions = predict(final_wf,
                      new_data = test,
                      type = "prob")

kaggle_submission = bind_cols(test["id"], predictions[".pred_1"]) |> 
  rename("Id" = id, "Action" = .pred_1)
#kaggle_submission

vroom_write(x = kaggle_submission, 
            file = "AmazonEmployeeAccess/svmpred.csv",
            delim = ",")











