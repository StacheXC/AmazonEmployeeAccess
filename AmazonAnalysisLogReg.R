library(vroom)
library(tidymodels)
library(tidyverse)
library(embed)

train = vroom("train.csv")
test = vroom("test.csv")
train$ACTION = as.factor(train$ACTION)

my_recipe = recipe(ACTION ~ ., data = train) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> 
  step_other(all_nominal_predictors(), threshold = 0.1) |> 
  step_dummy(all_nominal_predictors())

#prep = prep(my_recipe)
#baked = bake(prep, new_data = train)
#baked

# Logistic Regression
logRegModel = logistic_reg() |> 
  set_engine("glm")

wf = workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(logRegModel) |> 
  fit(data = train)

predictions = predict(wf,
                      new_data = test,
                      type = "prob")

#predictions

kaggle_submission = bind_cols(test["id"], predictions[".pred_1"]) |> 
  rename("Id" = id, "Action" = .pred_1)
#kaggle_submission

vroom_write(x = kaggle_submission, 
            file = "logregpred1.csv",
            delim = ",")
