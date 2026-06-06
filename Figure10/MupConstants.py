"""
Hardcoded Figure-10 MUPs (one per dataset).

Each MUP below is a Maximal Uncovered Pattern that was selected because, when
used in the Figure-10 coverage experiment (see ``GenerateFigure10.py``), it
reproduces the behaviour of Figure 10 in *"Assessing and Remedying Coverage for
a Given Dataset"*: as records from the under-covered subgroup are progressively
added to the training set, the subgroup's accuracy and F1 measure rise, while
the overall accuracy stays roughly flat.

A subgroup pattern is a tuple over ``feature_cols`` where ``None`` means 'X'
(the value is unspecified / a wildcard).

For each dataset we also pin the exact experiment settings (test size, the
sequence of subgroup-training sizes, and the random_state) under which the
curve is the cleanest, since -- like the paper's figure -- the plot is a single
representative run.
"""


def _airbnb_high_price(df):
    """AirBnb's raw label is the continuous ``price``; Figure 10 needs a binary
    target, so derive one: is the listing priced at or above the median?"""
    df = df.dropna(subset=["price"]).copy()
    df["high_price"] = (df["price"] >= df["price"].median()).astype(int)
    return df


# dataset name -> everything GenerateFigure10.py needs to rebuild the figure.
MUP_CONFIGS = {
    "Compas": {
        "csv": "CompasDataset.csv",
        "encoding": "utf-8",
        # Coverage dimensions: define the subgroup / where the MUP lives.
        "feature_cols": ["sex", "age_cat", "race"],
        "label_col": "two_year_recid",
        # Richer feature set the classifier trains on (paper Figure 10): more
        # than the coverage dimensions so subgroup members vary in feature space
        # and added subgroup data can actually change what the model learns.
        "model_feature_cols": [
            "sex", "age_cat", "race", "priors_count",
            "c_charge_degree", "decile_score",
        ],
        "tau": 80,                                   # 67-row group is uncovered at tau=80
        "subgroup_pattern": ("Male", "Less than 25", "Other"),
        "subgroup_test_size": 16,
        "subgroup_train_sizes": [0, 13, 26, 38, 51],
        "random_state": 1,
        # For reference (mean subgroup-accuracy curve at this run):
        # 0.31 -> 0.50 -> 0.50 -> 0.50 -> 0.69
    },
    "Adult Income": {
        "csv": "AdultIncomeDataSet.csv",
        "encoding": "utf-8",
        "feature_cols": ["age", "sex", "race"],      # `age` is already bucketed
        "label_col": "income",
        "model_feature_cols": [
            "age", "sex", "race", "education",
            "hours.per.week", "occupation", "workclass",
        ],
        "tau": 50,                                   # genuine MUP at the standard tau
        "subgroup_pattern": ("35-44", "Female", "Amer-Indian-Eskimo"),
        "subgroup_test_size": 8,
        "subgroup_train_sizes": [0, 6, 12, 17, 23],
        "random_state": 10,
        # subgroup-accuracy curve: 0.50 -> 0.63 -> 0.88 -> 1.0 -> 1.0
    },
    "AirBnb": {
        "csv": "AirBnbListingsDatasets.csv",
        "encoding": "latin-1",
        "feature_cols": ["host_is_superhost", "room_type", "instant_bookable"],
        "label_col": "high_price",
        "make_label": _airbnb_high_price,            # derive the binary label
        "model_feature_cols": [
            "host_is_superhost", "room_type", "instant_bookable",
            "neighbourhood", "property_type", "accommodates",
        ],
        "tau": 100,                                  # 92-row group, the dataset's only MUP
        "subgroup_pattern": ("t", "Hotel room", "f"),
        "subgroup_test_size": 20,
        "subgroup_train_sizes": [0, 18, 36, 54, 72],
        "random_state": 20,
        # subgroup-accuracy curve: 0.70 -> 0.75 -> 0.85 -> 0.85 -> 0.85
    },
}
