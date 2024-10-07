library(vroom)
library(tidymodels)
library(tidyverse)

df = vroom("train.csv")
df

df$RESOURCE |> hist()
