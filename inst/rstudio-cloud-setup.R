install.packages(c(
  "fst",
  "keras",
  "recipes",
  "remotes",
  "rmarkdown",
  "rsample",
  "tidyverse",
  "visNetwork",
  "yardstick"
))
remotes::install_github("wlandau/targets")
remotes::install_github("wlandau/tarchetypes")
reticulate::install_miniconda("miniconda")
line <- paste0("WORKON_HOME=", file.path(getwd(), "virtualenvs"))
writeLines(line, ".Renviron")
rstudioapi::restartSession()
reticulate::virtualenv_create("r-reticulate", python = "miniconda/bin/python")
keras::install_keras(
  method = "virtualenv",
  conda = "miniconda/bin/conda",
  envname = "r-reticulate"
)
