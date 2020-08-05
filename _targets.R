library(targets)
library(tarchetypes)
source("R/functions.R")
options(tidyverse.quiet = TRUE)

# Uncomment below to use local multicore computing
# when running tar_make_clustermq().
options(clustermq.scheduler = "multicore")

# Uncomment below to deploy targets to parallel jobs
# on a Sun Grid Engine cluster when running tar_make_clustermq().
# options(clustermq.scheduler = "sge", clustermq.template = "sge.tmpl")

tar_option_set(
  packages = c(
    "keras",
    "recipes",
    "rmarkdown",
    "rsample",
    "tidyverse",
    "yardstick"
  )
)
tar_pipeline(
  tar_target(
    data_file,
    "data/customer_churn.csv",
    format = "file",
    deployment = "local"
  ),
  tar_target(
    data,
    split_data(data_file),
    format = "qs",
    deployment = "local"
  ),
  tar_target(
    recipe,
    prepare_recipe(data),
    format = "qs",
    deployment = "local"
  ),
  tar_target(
    units,
    c(16, 32),
    deployment = "local"
  ),
  tar_target(
    act,
    c("relu", "sigmoid"),
    deployment = "local"
  ),
  tar_target(
    run,
    test_model(data, recipe, units1 = units, act1 = act),
    pattern = cross(units, act),
    format = "fst_tbl"
  ),
  tar_target(
    best_run,
    run %>%
      top_n(1, accuracy) %>%
      head(1),
    format = "fst_tbl",
    deployment = "local"
  ),
  tar_target(
    best_model,
    train_best_model(best_run, recipe),
    format = "keras"
  ),
  tar_render(report, "report.Rmd")
)
