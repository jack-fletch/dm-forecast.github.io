This repo contains all the data and code necessary to replicate the analyses performed in this project.

There are 3 folders and 3 files in this repo.

-   `data` contains all of the data required to create the forecasts. This data comes directly from https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

-   `pics` contains many images that go into the paper and the presentation. Most of these can be replicated using the code (though some of them cannot, and instead I provide their source in the paper).

-   `code` contains all of the code needed to preprocess all of the data and make the forecasts.

    -   There are 7 scripts. Scripts 1-4 preprocess all of the data and produce the forecasts while the scripts prepended with `z_` are used for some of the plots that are in the paper and the presentation. In addition, there is a script with several functions used in training and preprocessing.

    -   Note that `z_paper_plots_R.R` uses the the csv written by `1_data_pipeline.py`. This file is extremely large, and therefore it will not automatically be written unless you uncomment that line of code.

-   `paper.html` is the compiled html version of the paper.

-   `paper.qmd` is the uncompiled version of the paper. There are a few plots in the paper whose code is in this uncompiled version (and can't be seen anywhere else). If you want to 
compile this file yourself you will need to have `sales_clean.csv` in your `code` directory (which
is created from `1_data_pipeline.py` if you uncomment that line of code at the end of the file).

-   `ref.bib` this is the bibtex file with all of my references