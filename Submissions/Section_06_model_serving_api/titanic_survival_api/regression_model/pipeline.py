from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder, OneHotEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from regression_model.config.core import config
from regression_model.processing import features as pp
from regression_model.processing.features import ExtractLetterTransformer


titanic_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string 'missing'
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing",
                fill_value="Missing",
                variables=config.model_config.CATEGORICAL_VARIABLES,
            ),
        ),
        # add missing indicator to numerical variables
        ("missing_indicator", AddMissingIndicator(variables=config.model_config.NUMERICAL_VARIABLES)),
        # impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median", variables=config.model_config.NUMERICAL_VARIABLES
            ),
        ),
        # Extract first letter from config.model_config.CABIN
        ("extract_letter", ExtractLetterTransformer(column_name=config.model_config.CABIN)),
        # == CATEGORICAL ENCODING ======
        # remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        (
            "rare_label_encoder",
            RareLabelEncoder(tol=0.05, n_categories=2, variables=config.model_config.CATEGORICAL_VARIABLES),
        ),
        # encode categorical variables using one hot encoding into k-1 variables
        (
            "categorical_encoder",
            OneHotEncoder(drop_last=True, variables=config.model_config.CATEGORICAL_VARIABLES),
        ),
        # scale using standardization
        ("scaler", StandardScaler()),
        # logistic regression (use C=0.0005 and random_state=0)
        (
            "Logit",
            LogisticRegression(
                random_state=0,
                max_iter=1000,
                penalty="l1",
                C=0.0005,
                solver="liblinear",
            ),
        ),
    ]
)