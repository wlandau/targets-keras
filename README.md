
# `targets` R package Keras model example

[![Launch RStudio
Cloud](https://img.shields.io/badge/RStudio-Cloud-blue)](https://rstudio.cloud/project/1430828/)

The goal of this workflow is find the Keras model that best predicts
customer attrition (“churn”) on a subset of the [IBM Watson Telco
Customer Churn
dataset](https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/).
(See [this RStudio Blog post by Matt
Dancho](https://blogs.rstudio.com/ai/posts/2018-01-11-keras-customer-churn/)
for a thorough walkthrough of the use case.) Here fit multiple Keras
models to the dataset with different tuning parameters, pick the one
with the highest classification test accuracy, and produce a trained
model for the best set of tuning parameters we find.

## The `targets` pipeline

The [`targets`](https://github.com/wlandau/targets) R package manages
the workflow. It automatically skips steps of the pipeline when the
results are already up to date, which is critical for machine learning
tasks that take a long time to run. It also helps users understand and
communicate this work with tools like the interactive dependency graph
below.

``` r
library(targets)
tar_visnetwork()
```

![](./images/graph.png)

## How to access

You can try out this example project as long as you have a browser and
an internet connection. [Click
here](https://rstudio.cloud/project/1430828/) to navigate your browser
to an RStudio Cloud instance. Alternatively, you can clone or download
this code repository and install the R packages [listed
here](https://github.com/wlandau/targets-minimal/blob/03835c2aa4679dcf3f28c623a06d7505b18bee17/DESCRIPTION#L25-L30).

## How to run

In the R console, call the
[`tar_make()`](https://wlandau.github.io/targets/reference/tar_make.html)
function to run the pipeline. Then, call `tar_read(hist)` to retrieve
the histogram. Experiment with [other
functions](https://wlandau.github.io/targets/reference/index.html) such
as
[`tar_visnetwork()`](https://wlandau.github.io/targets/reference/tar_visnetwork.html)
to learn how they work.

## File structure

The files in this example are organized as follows.

``` r
├── run.sh
├── run.R
├── _targets.R
├── sge.tmpl
├── R/
├──── functions.R
├── data/
├──── customer_churn.csv
└── report.Rmd
```

| File                                                                                                    | Purpose                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`run.sh`](https://github.com/wlandau/targets-keras/blob/main/run.sh)                                   | Shell script to run [`run.R`](https://github.com/wlandau/targets-keras/blob/main/run.R) in a persistent background process. Works on Unix-like systems. Helpful for long computations on servers.                                                                                                                                                                                                           |
| [`run.R`](https://github.com/wlandau/targets-keras/blob/main/run.R)                                     | R script to run `tar_make()` or `tar_make_clustermq()` (uncomment the function of your choice.)                                                                                                                                                                                                                                                                                                             |
| [`_targets.R`](https://github.com/wlandau/targets-keras/blob/main/_targets.R)                           | The special R script that declares the [`targets`](https://github.com/wlandau/targets) pipeline. See `tar_script()` for details.                                                                                                                                                                                                                                                                            |
| [`sge.tmpl`](https://github.com/wlandau/targets-keras/blob/main/sge.tmpl)                               | A [`clustermq`](https://github.com/mschubert/clustermq) template file to deploy targets in parallel to a Sun Grid Engine cluster.                                                                                                                                                                                                                                                                           |
| [`R/functions.R`](https://github.com/wlandau/targets-keras/blob/main/R/functions.R)                     | An R script with user-defined functions. Unlike [`_targets.R`](https://github.com/wlandau/targets-keras/blob/main/_targets.R), there is nothing special about the name or location of this script. In fact, for larger projects, it is good practice to partition functions into multiple files.                                                                                                            |
| [`data/customer_churn.csv`](https://github.com/wlandau/targets-keras/blob/main/data/customer_churn.csv) | A subset of the [IBM Watson Telco Customer Churn dataset](https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/)                                                                                                                                                                                                                        |
| [`report.Rmd`](https://github.com/wlandau/targets-keras/blob/main/report.Rmd)                           | An R Markdown report summarizing the results of the analysis. For more information on how to include R Markdown reports as reproducible components of the pipeline, see the `tar_render()` function from the [`tarchetypes`](https://wlandau.github.io/tarchetypes) package and the [literate programming chapter of the manual](https://wlandau.github.io/targets-manual/files.html#literate-programming). |

## High-performance computing

You can run this project locally on your laptop or remotely on a
cluster. You have several choices, and they each require modifications
to [`run.R`](https://github.com/wlandau/targets-keras/blob/main/run.R)
and
[`_targets.R`](https://github.com/wlandau/targets-keras/blob/main/_targets.R).

| Mode            | When to use                        | Instructions for [`run.R`](https://github.com/wlandau/targets-keras/blob/main/run.R) | Instructions for [`_targets.R`](https://github.com/wlandau/targets-keras/blob/main/_targets.R) |
| --------------- | ---------------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Sequential      | Low-spec local machine or Windows. | Uncomment `tar_make()`                                                               | No action required.                                                                            |
| Local multicore | Local machine with a Unix-like OS. | Uncomment `tar_make_clustermq()`                                                     | Uncomment `options(clustermq.scheduler = "multicore")`                                         |
| Sun Grid Engine | Sun Grid Engine cluster.           | Uncomment `tar_make_clustermq()`                                                     | Uncomment `options(clustermq.scheduler = "sge", clustermq.template = "sge.tmpl")`              |
