library(vroom)
library(tidymodels)
library(tidyverse)
library(ggmosaic)
library(embed)

df = vroom("train.csv")
df

ggplot(df, aes(x = ACTION)) + 
  geom_bar()

ggplot(df, aes(x = RESOURCE, y = as.factor(ACTION))) + 
  geom_boxplot()

my_recipe = recipe(ACTION ~ ., data = df) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> 
  step_other(all_predictors(), threshold = .001) |> 
  step_dummy(all_nominal_predictors())

prep = prep(my_recipe)
baked = bake(prep, new_data = df)

baked


