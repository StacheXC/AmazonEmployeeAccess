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
  step_normalize(all_predictors())

model = parsnip::bart(trees = 50) |> 
  set_engine("dbarts") |> 
  set_mode("classification")

wf = workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(model) |> 
  fit(data = train)

predictions = predict(wf,
                      new_data = test,
                      type = "prob")

kaggle_submission = bind_cols(test["id"], predictions[".pred_1"]) |> 
  rename("Id" = id, "Action" = .pred_1)

vroom_write(x = kaggle_submission, 
            file = "AmazonEmployeeAccess/bartpred.csv",
            delim = ",")




