import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Modulo de prototipo opcional para a melhoria proposta no notebook.
# O modelo principal do projeto continua sendo gerado pelo notebook.


RANDOM_STATE = 42

BASE_NUMERIC_FEATURES = [
    "guests_model",
    "bedrooms_model",
    "beds_model",
    "bathrooms_model",
    "bath_per_bedroom",
    "beds_per_guest",
    "guests_per_bedroom",
    "lat",
    "lon",
    "distance_to_market_center_km",
    "title_len",
    "title_word_count",
    "description_len",
    "is_superhost_int",
    "is_new_listing_int",
    "has_studio",
    "has_loft",
    "has_luxury",
    "has_view",
    "has_metro",
    "has_parking",
    "has_pool",
    "has_gym",
    "has_wifi",
    "has_air_conditioning",
    "has_beach",
    "has_balcony",
]

VARIABLE_NUMERIC_FEATURES = [
    "capacity_score",
    "guests_x_bedrooms",
    "guests_x_bathrooms",
    "bedrooms_x_bathrooms",
    "capacity_pressure",
    "sleeping_density",
    "is_large_unit",
    "is_compact_unit",
    "large_unit_with_few_bathrooms",
]

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + VARIABLE_NUMERIC_FEATURES
CATEGORICAL_FEATURES = [
    "neighborhood_key",
    "listing_obj_type",
    "market_segment",
    "variable_segment",
    "capacity_tier",
    "bedroom_tier",
    "bathroom_tier",
]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def strip_accents(value):
    if pd.isna(value):
        return ""
    text = str(value).lower()
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_key(value):
    return re.sub(r"\s+", " ", strip_accents(value)).strip()


def _tier_from_values(values, bins, labels):
    series = pd.Series(values)
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(str).to_numpy()


def add_variable_model_features(df):
    out = df.copy()
    out["neighborhood_key"] = out["airbnb_neighborhood"].map(normalize_key)

    out["capacity_tier"] = _tier_from_values(
        out["guests_model"],
        bins=[0, 2, 4, 6, 8, 16],
        labels=["1_2", "3_4", "5_6", "7_8", "9_16"],
    )
    out["bedroom_tier"] = _tier_from_values(
        out["bedrooms_model"],
        bins=[0, 1, 2, 3, 10],
        labels=["1", "2", "3", "4_plus"],
    )
    out["bathroom_tier"] = _tier_from_values(
        out["bathrooms_model"],
        bins=[0, 1, 2, 3, 10],
        labels=["1", "2", "3", "4_plus"],
    )

    out["is_large_unit"] = (
        out["guests_model"].ge(6) | out["bedrooms_model"].ge(3) | out["bathrooms_model"].ge(2.5)
    ).astype(int)
    out["is_compact_unit"] = (
        out["guests_model"].le(2) & out["bedrooms_model"].le(1) & out["bathrooms_model"].le(1)
    ).astype(int)
    out["large_unit_with_few_bathrooms"] = (out["guests_model"].ge(6) & out["bathrooms_model"].le(1)).astype(int)

    out["capacity_score"] = (
        out["guests_model"]
        + out["bedrooms_model"] * 2.0
        + out["bathrooms_model"] * 1.5
        + out["beds_model"] * 0.5
    )
    out["guests_x_bedrooms"] = out["guests_model"] * out["bedrooms_model"]
    out["guests_x_bathrooms"] = out["guests_model"] * out["bathrooms_model"]
    out["bedrooms_x_bathrooms"] = out["bedrooms_model"] * out["bathrooms_model"]
    out["capacity_pressure"] = out["guests_model"] / (out["bedrooms_model"] * 2 + out["bathrooms_model"]).clip(lower=1)
    out["sleeping_density"] = out["guests_model"] / out["beds_model"].clip(lower=1)

    is_beach = out["market_segment"].eq("beach_premium") | out["has_beach"].eq(1)
    is_extended = out["market_segment"].eq("extended_market")
    is_urban = out["market_segment"].eq("urban_core")
    is_large = is_urban & (
        out["bedrooms_model"].ge(3) | out["guests_model"].ge(6) | out["bathrooms_model"].ge(2.5)
    )
    is_family = is_urban & (
        out["bedrooms_model"].eq(2) | out["guests_model"].between(4, 5) | out["bathrooms_model"].ge(1.5)
    )

    out["variable_segment"] = np.select(
        [is_beach, is_extended, is_large, is_family],
        ["beach_premium", "extended_market", "urban_large", "urban_family"],
        default="urban_compact",
    )
    return out


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def make_regressor(loss="absolute_error", quantile=None, min_samples_leaf=20):
    params = {
        "loss": loss,
        "learning_rate": 0.045,
        "max_iter": 650,
        "max_leaf_nodes": 31,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": 0.03,
        "early_stopping": True,
        "random_state": RANDOM_STATE,
    }
    if quantile is not None:
        params["quantile"] = quantile
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", HistGradientBoostingRegressor(**params))])


def make_classifier():
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.045,
                    max_iter=350,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=0.04,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    abs_error = np.abs(y_pred - y_true)
    ape = abs_error / np.clip(y_true, 1e-9, None) * 100
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(ape)),
        "MdAPE": float(np.median(ape)),
        "R2_log": float(r2_score(np.log(y_true), np.log(np.clip(y_pred, 1e-9, None)))),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def build_premium_thresholds(train_df, min_neighborhood_rows=30):
    neighborhood_stats = (
        train_df.groupby("neighborhood_key")
        .agg(n=("base_price", "size"), p75=("base_price", lambda x: x.quantile(0.75)))
        .reset_index()
    )
    segment_stats = (
        train_df.groupby("variable_segment")
        .agg(n=("base_price", "size"), p75=("base_price", lambda x: x.quantile(0.75)))
        .reset_index()
    )
    usable_neighborhoods = neighborhood_stats.loc[neighborhood_stats["n"].ge(min_neighborhood_rows)]
    return {
        "min_neighborhood_rows": min_neighborhood_rows,
        "neighborhood_p75": usable_neighborhoods.set_index("neighborhood_key")["p75"].to_dict(),
        "segment_p75": segment_stats.set_index("variable_segment")["p75"].to_dict(),
        "global_p75": float(train_df["base_price"].quantile(0.75)),
    }


def assign_premium_target(df, thresholds):
    neighborhood_threshold = df["neighborhood_key"].map(thresholds["neighborhood_p75"])
    segment_threshold = df["variable_segment"].map(thresholds["segment_p75"])
    threshold = neighborhood_threshold.fillna(segment_threshold).fillna(thresholds["global_p75"])
    return df["base_price"].ge(threshold).astype(int)


def _reference_summary(df, keys):
    if keys:
        grouped = df.groupby(keys)
    else:
        grouped = [((), df)]

    rows = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(keys, key)}
        row.update(
            {
                "reference_n": int(len(group)),
                "reference_median": float(group["base_price"].median()),
                "reference_p75": float(group["base_price"].quantile(0.75)),
                "reference_p90": float(group["base_price"].quantile(0.90)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_reference_tables(train_df):
    specs = {
        "neighborhood_structural": ["neighborhood_key", "bedrooms_model", "capacity_tier", "bathroom_tier"],
        "neighborhood_bedrooms": ["neighborhood_key", "bedrooms_model"],
        "neighborhood": ["neighborhood_key"],
        "segment_structural": ["variable_segment", "bedrooms_model", "capacity_tier", "bathroom_tier"],
        "segment_bedrooms": ["variable_segment", "bedrooms_model"],
        "segment": ["variable_segment"],
        "global": [],
    }
    return {name: _reference_summary(train_df, keys) for name, keys in specs.items()}


REFERENCE_RULES = [
    ("neighborhood_structural", ["neighborhood_key", "bedrooms_model", "capacity_tier", "bathroom_tier"], 8),
    ("neighborhood_bedrooms", ["neighborhood_key", "bedrooms_model"], 8),
    ("neighborhood", ["neighborhood_key"], 20),
    ("segment_structural", ["variable_segment", "bedrooms_model", "capacity_tier", "bathroom_tier"], 12),
    ("segment_bedrooms", ["variable_segment", "bedrooms_model"], 10),
    ("segment", ["variable_segment"], 1),
    ("global", [], 1),
]


def _match_reference(row, reference_tables):
    for source, keys, min_rows in REFERENCE_RULES:
        table = reference_tables[source]
        match = table
        for key in keys:
            match = match.loc[match[key].eq(row[key])]
        if not match.empty and int(match.iloc[0]["reference_n"]) >= min_rows:
            result = match.iloc[0].to_dict()
            result["comparable_source"] = source
            return result
    global_row = reference_tables["global"].iloc[0].to_dict()
    global_row["comparable_source"] = "global"
    return global_row


def comparable_predictions(df, artifact):
    rows = [_match_reference(row, artifact["reference_tables"]) for _, row in df.iterrows()]
    out = pd.DataFrame(rows, index=df.index)
    return out.rename(
        columns={
            "reference_n": "comparable_n",
            "reference_median": "comparable_median",
            "reference_p75": "comparable_p75",
            "reference_p90": "comparable_p90",
        }
    )


def predict_variable_price(feature_df, artifact):
    features = add_variable_model_features(feature_df)
    x = features[artifact["feature_cols"]]

    global_p50 = np.exp(artifact["global_models"]["p50"].predict(x))
    global_p75 = np.exp(artifact["global_models"]["p75"].predict(x))
    global_p90 = np.exp(artifact["global_models"]["p90"].predict(x))

    segment_p50 = global_p50.copy()
    has_segment_model = np.zeros(len(features), dtype=bool)
    for segment, model in artifact["segment_models"].items():
        mask = features["variable_segment"].eq(segment).to_numpy()
        if mask.any():
            segment_p50[mask] = np.exp(model.predict(x.loc[mask]))
            has_segment_model[mask] = True

    comparable = comparable_predictions(features, artifact)
    comparable_weight = np.select(
        [
            comparable["comparable_n"].ge(40),
            comparable["comparable_n"].ge(15),
            comparable["comparable_n"].ge(8),
        ],
        [0.05, 0.025, 0.0],
        default=0.0,
    )
    segment_weight = np.where(has_segment_model, 0.10, 0.0)
    global_weight = np.full(len(features), 0.85)
    total_weight = global_weight + segment_weight + comparable_weight

    p50_blend = (
        global_p50 * global_weight
        + segment_p50 * segment_weight
        + comparable["comparable_median"].to_numpy() * comparable_weight
    ) / total_weight

    p75_anchor = np.maximum(global_p75, p50_blend)
    p90_anchor = np.maximum(global_p90, p75_anchor)
    p75_anchor = np.maximum(
        p75_anchor * 0.75 + comparable["comparable_p75"].to_numpy() * 0.25,
        p50_blend,
    )
    p90_anchor = np.maximum(
        p90_anchor * 0.75 + comparable["comparable_p90"].to_numpy() * 0.25,
        p75_anchor,
    )

    premium_probability = artifact["premium_classifier"].predict_proba(x)[:, 1]
    premium_strength = np.clip((premium_probability - 0.80) / 0.25, 0, 1)
    premium_anchor = np.where(premium_probability >= 0.85, p90_anchor, p75_anchor)
    recommended_price = p50_blend + premium_strength * 0.20 * (premium_anchor - p50_blend)

    # Guardrails keep the point estimate inside the learned interval.
    recommended_price = np.clip(recommended_price, p50_blend * 0.75, p90_anchor)

    result = features.copy()
    result["predicted_base_price"] = recommended_price
    result["predicted_p50"] = p50_blend
    result["predicted_p75"] = p75_anchor
    result["predicted_p90"] = p90_anchor
    result["global_p50"] = global_p50
    result["segment_p50"] = segment_p50
    result["premium_probability"] = premium_probability
    result["has_segment_model"] = has_segment_model
    for col in ["comparable_source", "comparable_n", "comparable_median", "comparable_p75", "comparable_p90"]:
        result[col] = comparable[col].to_numpy()
    return result
