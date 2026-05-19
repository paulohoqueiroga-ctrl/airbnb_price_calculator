from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from test_model import add_features, build_consistent_new_apartments, build_guest_only_sensitivity_cases, build_stress_cases
from variable_price_model import predict_variable_price


# Teste do prototipo opcional de modelo variavel.
# Para reproduzir a versão principal do projeto, use o notebook ou scripts/test_model.py.


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_PATH = ROOT / "models" / "base_price_model.joblib"
VARIABLE_MODEL_PATH = ROOT / "models" / "variable_price_model.joblib"
OUTPUT_PATH = ROOT / "outputs" / "tables" / "variable_model_new_apartment_tests.csv"


def load_artifacts():
    if not VARIABLE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo variavel nao encontrado em {VARIABLE_MODEL_PATH}. "
            "Rode: .venv/bin/python scripts/train_variable_model.py"
        )
    return joblib.load(BASE_MODEL_PATH), joblib.load(VARIABLE_MODEL_PATH)


def cases_to_dataframe(cases):
    raw_rows = []
    meta_rows = []
    for case in cases:
        raw_rows.append({key: value for key, value in case.items() if not key.startswith("_")})
        meta_rows.append({"suite": case["_suite"], "case": case["_case"]})
    return pd.DataFrame(raw_rows), pd.DataFrame(meta_rows)


def run_case_battery(base_artifact, variable_artifact):
    cases = build_consistent_new_apartments() + build_guest_only_sensitivity_cases() + build_stress_cases()
    raw_df, meta_df = cases_to_dataframe(cases)
    feature_df = add_features(raw_df, base_artifact)
    prediction_df = predict_variable_price(feature_df, variable_artifact)

    result = pd.concat([meta_df.reset_index(drop=True), prediction_df.reset_index(drop=True)], axis=1)
    result["delta_vs_suite_base"] = result["predicted_base_price"] - result.groupby("suite")[
        "predicted_base_price"
    ].transform("first")
    result["pct_vs_suite_base"] = (
        result["predicted_base_price"] / result.groupby("suite")["predicted_base_price"].transform("first") - 1
    ) * 100
    return result


def assert_basic_invariants(result):
    price_cols = ["predicted_base_price", "predicted_p50", "predicted_p75", "predicted_p90"]
    values = result[price_cols].to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError("Ha previsoes infinitas ou NaN na bateria de teste.")
    if not (values > 0).all():
        raise AssertionError("Todas as previsoes precisam ser positivas.")
    if not result["predicted_p75"].ge(result["predicted_p50"]).all():
        raise AssertionError("Guardrail violado: predicted_p75 menor que predicted_p50.")
    if not result["predicted_p90"].ge(result["predicted_p75"]).all():
        raise AssertionError("Guardrail violado: predicted_p90 menor que predicted_p75.")

    by_case = result.set_index("case")
    if by_case.loc["liberdade_4p_2q", "predicted_base_price"] <= by_case.loc["liberdade_2p_1q", "predicted_base_price"]:
        raise AssertionError("Cenario Liberdade 4p/2q deveria ficar acima de 2p/1q.")
    if by_case.loc["liberdade_8p_4q", "predicted_base_price"] <= by_case.loc["liberdade_4p_2q", "predicted_base_price"]:
        raise AssertionError("Cenario Liberdade 8p/4q deveria ficar acima de 4p/2q.")
    if by_case.loc["riviera_8p_4q", "predicted_base_price"] <= by_case.loc["riviera_4p_2q", "predicted_base_price"]:
        raise AssertionError("Cenario Riviera 8p/4q deveria ficar acima de 4p/2q.")
    if by_case.loc["guests_only_20p", "guests_model"] != 16:
        raise AssertionError("Cap de guests esperado: guests=20 precisa virar guests_model=16.")


def print_result_table(result):
    cols = [
        "suite",
        "case",
        "airbnb_neighborhood",
        "variable_segment",
        "guests",
        "guests_model",
        "bedrooms",
        "bedrooms_model",
        "bathrooms_model",
        "predicted_base_price",
        "predicted_p50",
        "predicted_p75",
        "predicted_p90",
        "premium_probability",
        "comparable_source",
        "comparable_n",
        "delta_vs_suite_base",
        "pct_vs_suite_base",
    ]
    print("\nBateria do modelo variavel para new_apartment:")
    print(result[cols].round(2).to_string(index=False))


def main():
    base_artifact, variable_artifact = load_artifacts()
    result = run_case_battery(base_artifact, variable_artifact)
    assert_basic_invariants(result)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print_result_table(result)
    print("\nTestes de invariantes passaram.")
    print(f"Resultado salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
