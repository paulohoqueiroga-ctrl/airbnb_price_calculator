from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from variable_price_model import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    add_variable_model_features,
    assign_premium_target,
    build_premium_thresholds,
    build_reference_tables,
    make_classifier,
    make_regressor,
    predict_variable_price,
    regression_metrics,
)


# Prototipo opcional, nao parte do resultado principal do projeto.
# O notebook gera o modelo principal em models/base_price_model.joblib.
# Este script materializa a melhoria proposta de modelo variavel para exploracao futura.


ROOT = Path(__file__).resolve().parents[1]
MODELING_DATASET = ROOT / "outputs" / "modeling_dataset.csv"
MODEL_PATH = ROOT / "models" / "variable_price_model.joblib"
TABLE_DIR = ROOT / "outputs" / "tables"
PREDICTION_PATH = ROOT / "outputs" / "variable_model_holdout_predictions.csv"
SUMMARY_PATH = ROOT / "outputs" / "variable_model_summary.json"


def make_split(model_df):
    y = model_df["base_price"].copy()
    strat_bins = pd.qcut(np.log(y), q=5, labels=False, duplicates="drop")
    train_idx, test_idx = train_test_split(
        model_df.index,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=strat_bins,
    )
    return model_df.loc[train_idx].copy(), model_df.loc[test_idx].copy()


def fit_segment_models(train_df, min_rows=250):
    models = {}
    segment_counts = train_df["variable_segment"].value_counts().to_dict()
    for segment, n_rows in segment_counts.items():
        if n_rows < min_rows:
            continue
        segment_train = train_df.loc[train_df["variable_segment"].eq(segment)].copy()
        min_samples_leaf = max(10, min(25, int(n_rows // 40)))
        model = make_regressor(loss="absolute_error", min_samples_leaf=min_samples_leaf)
        model.fit(segment_train[FEATURE_COLS], np.log(segment_train["base_price"]))
        models[segment] = model
    return models, segment_counts


def evaluate_predictions(test_df, prediction_df):
    y_true = test_df["base_price"].to_numpy()
    metric_rows = []
    for name, column in [
        ("global_p50", "global_p50"),
        ("segment_p50_blend", "predicted_p50"),
        ("variable_ensemble", "predicted_base_price"),
        ("variable_p75", "predicted_p75"),
        ("variable_p90", "predicted_p90"),
    ]:
        metric_rows.append({"model": name, **regression_metrics(y_true, prediction_df[column].to_numpy())})
    return pd.DataFrame(metric_rows).sort_values("MAE").reset_index(drop=True)


def build_segment_reports(test_df, prediction_df):
    evaluation = test_df.copy()
    evaluation["predicted_base_price"] = prediction_df["predicted_base_price"].to_numpy()
    evaluation["premium_probability"] = prediction_df["premium_probability"].to_numpy()
    evaluation["comparable_source"] = prediction_df["comparable_source"].to_numpy()
    evaluation["error"] = evaluation["predicted_base_price"] - evaluation["base_price"]
    evaluation["abs_error"] = evaluation["error"].abs()
    evaluation["ape"] = evaluation["abs_error"] / evaluation["base_price"] * 100
    evaluation["price_band"] = pd.qcut(
        evaluation["base_price"],
        q=4,
        labels=["low", "mid_low", "mid_high", "high"],
    )

    by_price_band = (
        evaluation.groupby("price_band", observed=False)
        .agg(
            n=("id", "size"),
            actual_median=("base_price", "median"),
            pred_median=("predicted_base_price", "median"),
            MAE=("abs_error", "mean"),
            MAPE=("ape", "mean"),
            Bias=("error", "mean"),
            premium_prob_median=("premium_probability", "median"),
        )
        .reset_index()
    )

    by_variable_segment = (
        evaluation.groupby("variable_segment")
        .agg(
            n=("id", "size"),
            actual_median=("base_price", "median"),
            pred_median=("predicted_base_price", "median"),
            MAE=("abs_error", "mean"),
            MAPE=("ape", "mean"),
            Bias=("error", "mean"),
        )
        .sort_values("n", ascending=False)
        .reset_index()
    )

    by_bedrooms = (
        evaluation.groupby("bedrooms_model")
        .agg(
            n=("id", "size"),
            actual_median=("base_price", "median"),
            pred_median=("predicted_base_price", "median"),
            MAE=("abs_error", "mean"),
            MAPE=("ape", "mean"),
            Bias=("error", "mean"),
        )
        .reset_index()
    )
    return evaluation, by_price_band, by_variable_segment, by_bedrooms


def build_scenario_impacts(model_df, artifact):
    rows = []
    max_guests = artifact["structural_caps"]["guests"]
    eligible = model_df.loc[model_df["guests_model"].lt(max_guests)].copy()
    base_pred = predict_variable_price(eligible, artifact)["predicted_base_price"].to_numpy()
    variant = eligible.copy()
    variant["guests_model"] = (variant["guests_model"] + 1).clip(upper=max_guests)
    variant["guests"] = variant["guests_model"]
    variant["bath_per_bedroom"] = variant["bathrooms_model"] / variant["bedrooms_model"].clip(lower=1)
    variant["beds_per_guest"] = variant["beds_model"] / variant["guests_model"].clip(lower=1)
    variant["guests_per_bedroom"] = variant["guests_model"] / variant["bedrooms_model"].clip(lower=1)
    variant_pred = predict_variable_price(variant, artifact)["predicted_base_price"].to_numpy()
    delta = variant_pred - base_pred
    rows.append(
        {
            "scenario": "+1 guest, demais variaveis constantes",
            "n": int(len(delta)),
            "median_delta_rs": float(np.median(delta)),
            "mean_delta_rs": float(np.mean(delta)),
            "median_delta_pct": float(np.median(delta / base_pred * 100)),
            "mean_delta_pct": float(np.mean(delta / base_pred * 100)),
        }
    )

    for raw_col, model_col, cap, label in [
        ("bedrooms", "bedrooms_model", artifact["structural_caps"]["bedrooms"], "+1 bedroom"),
        ("bathrooms_num", "bathrooms_model", artifact["structural_caps"]["bathrooms"], "+1 bathroom"),
    ]:
        eligible = model_df.loc[model_df[model_col].lt(cap)].copy()
        base_pred = predict_variable_price(eligible, artifact)["predicted_base_price"].to_numpy()
        variant = eligible.copy()
        variant[model_col] = (variant[model_col] + 1).clip(upper=cap)
        if raw_col in variant:
            variant[raw_col] = variant[model_col]
        variant["bath_per_bedroom"] = variant["bathrooms_model"] / variant["bedrooms_model"].clip(lower=1)
        variant["beds_per_guest"] = variant["beds_model"] / variant["guests_model"].clip(lower=1)
        variant["guests_per_bedroom"] = variant["guests_model"] / variant["bedrooms_model"].clip(lower=1)
        variant_pred = predict_variable_price(variant, artifact)["predicted_base_price"].to_numpy()
        delta = variant_pred - base_pred
        rows.append(
            {
                "scenario": f"{label}, demais variaveis constantes",
                "n": int(len(delta)),
                "median_delta_rs": float(np.median(delta)),
                "mean_delta_rs": float(np.mean(delta)),
                "median_delta_pct": float(np.median(delta / base_pred * 100)),
                "mean_delta_pct": float(np.mean(delta / base_pred * 100)),
            }
        )
    return pd.DataFrame(rows)


def main():
    print("Carregando dataset modelado...")
    model_df = pd.read_csv(MODELING_DATASET)
    model_df = add_variable_model_features(model_df)
    train_df, test_df = make_split(model_df)

    print(f"Treino: {len(train_df):,} linhas | Holdout: {len(test_df):,} linhas")
    print("Treinando modelos globais p50/p75/p90...")
    global_models = {
        "p50": make_regressor(loss="absolute_error", min_samples_leaf=20),
        "p75": make_regressor(loss="quantile", quantile=0.75, min_samples_leaf=20),
        "p90": make_regressor(loss="quantile", quantile=0.90, min_samples_leaf=20),
    }
    for model in global_models.values():
        model.fit(train_df[FEATURE_COLS], np.log(train_df["base_price"]))

    print("Treinando modelos por segmento...")
    segment_models, segment_counts = fit_segment_models(train_df)
    print("Segmentos com modelo proprio:", ", ".join(sorted(segment_models)) or "nenhum")

    print("Treinando classificador de cauda/premium...")
    thresholds = build_premium_thresholds(train_df)
    premium_target = assign_premium_target(train_df, thresholds)
    premium_classifier = make_classifier()
    premium_classifier.fit(train_df[FEATURE_COLS], premium_target)

    artifact = {
        "artifact_type": "variable_price_ensemble",
        "model_name": "variable_segment_quantile_comparable_v1",
        "global_models": global_models,
        "segment_models": segment_models,
        "premium_classifier": premium_classifier,
        "premium_thresholds": thresholds,
        "reference_tables": build_reference_tables(train_df),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_cols": FEATURE_COLS,
        "segment_counts": segment_counts,
        "trained_segment_models": sorted(segment_models),
        "structural_caps": {"guests": 16, "bedrooms": 10, "beds": 30, "bathrooms": 10},
        "notes": [
            "Ponto principal combina p50 global, modelo por segmento, comparaveis locais e lift premium.",
            "p75/p90 sao guardrails de cauda, nao resultados esperados fixos.",
            "Use scripts/test_variable_model.py para diagnosticos em apartamentos novos.",
        ],
    }

    print("Avaliando holdout...")
    prediction_df = predict_variable_price(test_df, artifact)
    metrics_df = evaluate_predictions(test_df, prediction_df)
    evaluation_df, by_price_band, by_variable_segment, by_bedrooms = build_segment_reports(test_df, prediction_df)
    scenario_impacts = build_scenario_impacts(test_df, artifact)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(TABLE_DIR / "variable_model_metrics.csv", index=False)
    by_price_band.to_csv(TABLE_DIR / "variable_model_segment_price_band.csv", index=False)
    by_variable_segment.to_csv(TABLE_DIR / "variable_model_segment_variable.csv", index=False)
    by_bedrooms.to_csv(TABLE_DIR / "variable_model_segment_bedrooms.csv", index=False)
    scenario_impacts.to_csv(TABLE_DIR / "variable_model_scenario_impacts.csv", index=False)
    evaluation_df.to_csv(PREDICTION_PATH, index=False)

    joblib.dump(artifact, MODEL_PATH)

    summary = {
        "model_name": artifact["model_name"],
        "n_modeling_rows": int(len(model_df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "segment_counts_train": {k: int(v) for k, v in segment_counts.items()},
        "trained_segment_models": sorted(segment_models),
        "metrics": metrics_df.to_dict(orient="records"),
        "scenario_impacts": scenario_impacts.to_dict(orient="records"),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nMetricas do novo sistema:")
    print(metrics_df.round(3).to_string(index=False))
    print("\nMetricas por faixa de preco:")
    print(by_price_band.round(3).to_string(index=False))
    print("\nImpactos contrafactuais:")
    print(scenario_impacts.round(3).to_string(index=False))
    print("\nArtefatos salvos:")
    print(f"- {MODEL_PATH}")
    print(f"- {TABLE_DIR / 'variable_model_metrics.csv'}")
    print(f"- {PREDICTION_PATH}")
    print(f"- {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
