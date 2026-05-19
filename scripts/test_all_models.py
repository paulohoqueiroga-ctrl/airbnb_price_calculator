from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from test_model import (
    add_features,
    build_consistent_new_apartments,
    build_guest_only_sensitivity_cases,
    build_stress_cases,
    predict_base_price,
)
from variable_price_model import predict_variable_price


# Runner comparativo opcional entre o modelo principal do notebook e o prototipo variavel.
# Nao e necessario para reproduzir o resultado principal.


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_PATH = ROOT / "models" / "base_price_model.joblib"
VARIABLE_MODEL_PATH = ROOT / "models" / "variable_price_model.joblib"
BASE_METRICS_PATH = ROOT / "outputs" / "tables" / "model_metrics.csv"
VARIABLE_METRICS_PATH = ROOT / "outputs" / "tables" / "variable_model_metrics.csv"
OUTPUT_PATH = ROOT / "outputs" / "tables" / "all_models_new_apartment_tests.csv"


def load_artifacts():
    missing = [path for path in [BASE_MODEL_PATH, VARIABLE_MODEL_PATH] if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Artefato(s) ausente(s): {missing_list}")
    return joblib.load(BASE_MODEL_PATH), joblib.load(VARIABLE_MODEL_PATH)


def cases_to_dataframe(cases):
    raw_rows = []
    meta_rows = []
    for case in cases:
        raw_rows.append({key: value for key, value in case.items() if not key.startswith("_")})
        meta_rows.append({"suite": case["_suite"], "case": case["_case"]})
    return pd.DataFrame(raw_rows), pd.DataFrame(meta_rows)


def build_all_cases():
    return build_consistent_new_apartments() + build_guest_only_sensitivity_cases() + build_stress_cases()


def run_unified_case_battery(base_artifact, variable_artifact):
    raw_df, meta_df = cases_to_dataframe(build_all_cases())

    base_pred, base_features = predict_base_price(raw_df, base_artifact)
    variable_pred = predict_variable_price(base_features, variable_artifact)

    result = pd.concat(
        [
            meta_df.reset_index(drop=True),
            raw_df.reset_index(drop=True),
            base_features[
                [
                    "guests_model",
                    "bedrooms_model",
                    "beds_model",
                    "bathrooms_model",
                    "market_segment",
                    "distance_to_market_center_km",
                ]
            ].reset_index(drop=True),
        ],
        axis=1,
    )
    result["base_model_price"] = base_pred
    result["variable_model_price"] = variable_pred["predicted_base_price"].to_numpy()
    result["variable_p50"] = variable_pred["predicted_p50"].to_numpy()
    result["variable_p75"] = variable_pred["predicted_p75"].to_numpy()
    result["variable_p90"] = variable_pred["predicted_p90"].to_numpy()
    result["premium_probability"] = variable_pred["premium_probability"].to_numpy()
    result["variable_segment"] = variable_pred["variable_segment"].to_numpy()
    result["comparable_source"] = variable_pred["comparable_source"].to_numpy()
    result["comparable_n"] = variable_pred["comparable_n"].to_numpy()

    result["variable_minus_base"] = result["variable_model_price"] - result["base_model_price"]
    result["variable_vs_base_pct"] = result["variable_minus_base"] / result["base_model_price"] * 100
    for price_col in ["base_model_price", "variable_model_price"]:
        result[f"{price_col}_delta_vs_suite_base"] = result[price_col] - result.groupby("suite")[price_col].transform(
            "first"
        )
        result[f"{price_col}_pct_vs_suite_base"] = (
            result[price_col] / result.groupby("suite")[price_col].transform("first") - 1
        ) * 100
    return result


def assert_unified_invariants(result):
    price_cols = ["base_model_price", "variable_model_price", "variable_p50", "variable_p75", "variable_p90"]
    values = result[price_cols].to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError("Ha previsoes infinitas ou NaN no teste unificado.")
    if not (values > 0).all():
        raise AssertionError("Todas as previsoes precisam ser positivas.")
    if not result["variable_p75"].ge(result["variable_p50"]).all():
        raise AssertionError("Guardrail violado: variable_p75 menor que variable_p50.")
    if not result["variable_p90"].ge(result["variable_p75"]).all():
        raise AssertionError("Guardrail violado: variable_p90 menor que variable_p75.")

    by_case = result.set_index("case")
    for model_col in ["base_model_price", "variable_model_price"]:
        checks = [
            ("liberdade_4p_2q", "liberdade_2p_1q"),
            ("liberdade_8p_4q", "liberdade_4p_2q"),
            ("pinheiros_4p_2q", "pinheiros_2p_1q"),
            ("riviera_8p_4q", "riviera_4p_2q"),
        ]
        for larger_case, smaller_case in checks:
            if by_case.loc[larger_case, model_col] <= by_case.loc[smaller_case, model_col]:
                raise AssertionError(f"{model_col}: {larger_case} deveria ficar acima de {smaller_case}.")

    if by_case.loc["guests_only_20p", "guests_model"] != 16:
        raise AssertionError("Cap de guests esperado: guests=20 precisa virar guests_model=16.")
    if by_case.loc["cap_stress_20p_15q_10c_1b", "bedrooms_model"] != 10:
        raise AssertionError("Cap de bedrooms esperado: bedrooms=15 precisa virar bedrooms_model=10.")


def print_metric_snapshot():
    print("\nMetricas de holdout salvas:")
    if BASE_METRICS_PATH.exists():
        base_metrics = pd.read_csv(BASE_METRICS_PATH)
        base_final = base_metrics.loc[base_metrics["model"].eq("robust_segmented_hgb_log_mae")]
        if base_final.empty:
            base_final = base_metrics.head(1)
        print("\nModelo original:")
        print(base_final.round(3).to_string(index=False))
    else:
        print(f"- Metricas do modelo original nao encontradas: {BASE_METRICS_PATH}")

    if VARIABLE_METRICS_PATH.exists():
        variable_metrics = pd.read_csv(VARIABLE_METRICS_PATH)
        print("\nModelo variavel:")
        print(variable_metrics.round(3).to_string(index=False))
    else:
        print(f"- Metricas do modelo variavel nao encontradas: {VARIABLE_METRICS_PATH}")


def print_unified_table(result):
    cols = [
        "suite",
        "case",
        "airbnb_neighborhood",
        "guests",
        "guests_model",
        "bedrooms",
        "bedrooms_model",
        "bathrooms_model",
        "market_segment",
        "variable_segment",
        "base_model_price",
        "variable_model_price",
        "variable_p50",
        "variable_p75",
        "variable_p90",
        "premium_probability",
        "comparable_source",
        "comparable_n",
        "variable_minus_base",
        "variable_vs_base_pct",
    ]
    print("\nBateria unificada de new_apartment:")
    print(result[cols].round(2).to_string(index=False))


def main():
    base_artifact, variable_artifact = load_artifacts()
    print("Modelos carregados:")
    print(f"- original: {base_artifact['model_name']}")
    print(f"- variavel: {variable_artifact['model_name']}")

    result = run_unified_case_battery(base_artifact, variable_artifact)
    assert_unified_invariants(result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print_metric_snapshot()
    print_unified_table(result)
    print("\nTeste unificado passou.")
    print(f"Resultado salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
