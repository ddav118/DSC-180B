# DSC-180B
Project
├── run.py             <- run.py with calls to functions in src.
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── temp           <- Intermediate data that has been transformed.
│   ├── out            <- The final, canonical data sets for modeling.
├── notebooks          <- Jupyter notebooks (presentation only).
├── references         <- Data dictionaries, explanatory materials.
├── src                <- Source code for use in this project.
    ├── data           <- Scripts to download or generate data.
    │   └── make_dataset.py
    ├── features       <- Scripts to turn raw data into features for modeling.
    │   └── build_features.py
    ├── models         <- Scripts to train models and make predictions.
    │   ├── predict_model.py
    │   └── train_model.py
    └── visualization  <- Scripts to create exploratory and results-oriented viz.
        └── visualize.py
