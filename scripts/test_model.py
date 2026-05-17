from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tabas_base_price_model.joblib"
MODELING_DATASET = ROOT / "outputs" / "modeling_dataset.csv"
RAW_NEIGHBORHOOD_TABLE = ROOT / "outputs" / "tables" / "raw_neighborhood_prices.csv"


def parse_bathroom(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if "half" in text:
        return 0.5
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def strip_accents(value):
    if pd.isna(value):
        return ""
    text = str(value).lower()
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_key(value):
    return re.sub(r"\s+", " ", strip_accents(value)).strip()


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def add_features(raw_df, artifact):
    out = raw_df.copy()

    out["bathrooms_num"] = out["bathrooms"].apply(parse_bathroom)
    out["distance_to_market_center_km"] = haversine_km(
        artifact["market_center_lat"],
        artifact["market_center_lon"],
        out["lat"],
        out["lon"],
    )

    text_norm = (
        out["ad_title"].fillna("")
        + " "
        + out["kicker_content_message"].fillna("")
        + " "
        + out["airbnb_neighborhood"].fillna("")
    ).map(strip_accents)

    keyword_patterns = {
        "has_studio": r"\bstudio\b",
        "has_loft": r"\bloft\b",
        "has_luxury": r"luxo|luxury|alto padrao|high standard|sofistic",
        "has_view": r"vista|view|mar",
        "has_metro": r"metro|subway",
        "has_parking": r"garagem|parking|vaga",
        "has_pool": r"piscina|pool",
        "has_gym": r"academia|gym|fitness",
        "has_wifi": r"wi-?fi|internet",
        "has_air_conditioning": r"ar cond|air cond|a/c|ar c\.?",
        "has_beach": r"praia|beach|rivier|pe na areia|bertioga|sao lourenco",
        "has_balcony": r"varanda|balcony|terrace|terraco",
    }
    for col, pattern in keyword_patterns.items():
        out[col] = text_norm.str.contains(pattern, regex=True, na=False).astype(int)

    out["title_len"] = out["ad_title"].fillna("").str.len()
    out["title_word_count"] = out["ad_title"].fillna("").str.split().str.len()
    out["description_len"] = out["kicker_content_message"].fillna("").str.len()
    out["is_superhost_int"] = out["is_superhost"].astype(int)
    out["is_new_listing_int"] = out["is_new_listing"].astype(int)
    out["beds"] = out["beds"].astype(float)

    out["market_segment"] = np.select(
        [out["has_beach"].eq(1), out["distance_to_market_center_km"].gt(40)],
        ["beach_premium", "extended_market"],
        default="urban_core",
    )

    # Caps robustos: valores acima desses limites ainda sao aceitos, mas o
    # modelo usa a versao capada para evitar extrapolacao instavel.
    caps = artifact.get("structural_caps", {"guests": 16, "bedrooms": 10, "beds": 30, "bathrooms": 10})
    out["guests_model"] = out["guests"].clip(1, caps.get("guests", 16))
    out["bedrooms_model"] = out["bedrooms"].clip(1, caps.get("bedrooms", 10))
    out["beds_model"] = out["beds"].clip(1, caps.get("beds", 30))
    out["bathrooms_model"] = out["bathrooms_num"].clip(0.5, caps.get("bathrooms", 10))
    out["bath_per_bedroom"] = out["bathrooms_model"] / out["bedrooms_model"].clip(lower=1)
    out["beds_per_guest"] = out["beds_model"] / out["guests_model"].clip(lower=1)
    out["guests_per_bedroom"] = out["guests_model"] / out["bedrooms_model"].clip(lower=1)

    return out


def predict_base_price(raw_df, artifact):
    features = add_features(raw_df, artifact)
    prediction = np.exp(artifact["pipeline"].predict(features[artifact["feature_cols"]]))
    return prediction, features


def neighborhood_fallback(raw_df):
    if not RAW_NEIGHBORHOOD_TABLE.exists():
        return None

    raw_neighborhood = pd.read_csv(RAW_NEIGHBORHOOD_TABLE)
    raw_neighborhood["neighborhood_key"] = raw_neighborhood["airbnb_neighborhood"].map(normalize_key)
    input_key = normalize_key(raw_df.iloc[0]["airbnb_neighborhood"])

    exact = raw_neighborhood.loc[raw_neighborhood["neighborhood_key"].eq(input_key)]
    if exact.empty and "rivier" in input_key:
        exact = raw_neighborhood.loc[raw_neighborhood["neighborhood_key"].str.contains("rivier", na=False)]
    if exact.empty:
        return None

    n = exact["n"].sum()
    median_reference = np.average(exact["median_price"], weights=exact["n"])
    mean_reference = np.average(exact["mean_price"], weights=exact["n"])
    names = ", ".join(exact["airbnb_neighborhood"].head(4).tolist())
    return {
        "matched_neighborhoods": names,
        "n": int(n),
        "median_reference": float(median_reference),
        "mean_reference": float(mean_reference),
    }


def predict_with_diagnostics(raw_df, artifact, trained_neighborhood_counts):
    prediction, features = predict_base_price(raw_df, artifact)

    issues = []
    distance = float(features.iloc[0]["distance_to_market_center_km"])
    segment = features.iloc[0]["market_segment"]
    neighborhood = raw_df.iloc[0]["airbnb_neighborhood"]
    neighborhood_count = int(trained_neighborhood_counts.get(neighborhood, 0))
    max_distance = float(artifact.get("max_training_distance_km", 150))

    searchable_text = normalize_key(
        " ".join(
            [
                str(raw_df.iloc[0].get("airbnb_neighborhood", "")),
                str(raw_df.iloc[0].get("ad_title", "")),
                str(raw_df.iloc[0].get("kicker_content_message", "")),
            ]
        )
    )

    if distance > max_distance:
        issues.append(f"distancia {distance:.1f} km acima do limite treinado de {max_distance:.0f} km")
    if neighborhood_count == 0:
        issues.append(f"bairro/regiao '{neighborhood}' nao aparece na amostra robusta de treino")
    elif neighborhood_count < 10:
        issues.append(f"bairro/regiao '{neighborhood}' tem apenas {neighborhood_count} exemplo(s) no treino robusto")
    if re.search(r"\bbertioga\b|rivier|sao lourenco|praia|beach|pe na areia", searchable_text):
        issues.append(f"sinal de praia/premium detectado; segmento do modelo = {segment}")

    caps = artifact.get("structural_caps", {"guests": 16, "bedrooms": 10, "beds": 30, "bathrooms": 10})
    if float(raw_df.iloc[0].get("bedrooms", 0)) > caps.get("bedrooms", 10):
        issues.append(f"quartos acima do cap robusto {caps.get('bedrooms', 10)}")
    if float(features.iloc[0].get("bathrooms_num", 0)) > caps.get("bathrooms", 10):
        issues.append(f"banheiros acima do cap robusto {caps.get('bathrooms', 10)}")
    if float(raw_df.iloc[0].get("beds", 0)) > caps.get("beds", 30):
        issues.append(f"camas acima do cap robusto {caps.get('beds', 30)}")

    # Detecta texto de Bertioga/Riviera com coordenadas no nucleo de Sao Paulo.
    if segment == "beach_premium" and distance < 20 and re.search(r"\bbertioga\b|rivier", searchable_text):
        issues.append("coordenadas parecem Sao Paulo urbano, mas texto/bairro indica Bertioga/Riviera")

    fallback = neighborhood_fallback(raw_df) if issues else None
    return prediction, features, issues, fallback


def print_model_feature_snapshot(raw_df, features):
    raw = raw_df.iloc[0]
    feat = features.iloc[0]
    print("\nSnapshot das features usadas pelo modelo:")
    print(f"- bedrooms bruto: {raw.get('bedrooms')} -> bedrooms_model: {feat.get('bedrooms_model')}")
    print(f"- guests bruto: {raw.get('guests')} -> guests_model: {feat.get('guests_model')}")
    print(f"- beds bruto: {raw.get('beds')} -> beds_model: {feat.get('beds_model')}")
    print(f"- bathrooms bruto: {raw.get('bathrooms')} -> bathrooms_model: {feat.get('bathrooms_model')}")
    print(f"- market_segment: {feat.get('market_segment')}")
    print(f"- distancia_km: {feat.get('distance_to_market_center_km'):.1f}")
    print("\nObservacao: o pipeline nao usa 'bedrooms' bruto diretamente.")
    print("Ele usa 'bedrooms_model', que so muda quando voce roda add_features()/predict_with_diagnostics().")


def compare_bedroom_variants(artifact, trained_neighborhood_counts):
    print("\nComparacao controlada: Liberdade variando quartos")
    base = {
        "airbnb_neighborhood": "Liberdade",
        "ad_title": "Apartamento completo na Liberdade perto do metro",
        "listing_obj_type": "REGULAR",
        "kicker_content_message": "Entire rental unit in Liberdade, perto do metro, wifi e ar condicionado",
        "is_superhost": True,
        "is_new_listing": False,
        "guests": 2,
        "bedrooms": 1,
        "beds": 1,
        "bathrooms": "1",
        "lat": -23.557,
        "lon": -46.635,
    }
    rows = []
    for bedrooms in [1, 2, 3, 6, 10, 12]:
        row = base.copy()
        row["bedrooms"] = bedrooms
        row["guests"] = min(16, max(2, bedrooms * 2))
        row["beds"] = bedrooms
        raw = pd.DataFrame([row])
        pred, features, issues, _ = predict_with_diagnostics(raw, artifact, trained_neighborhood_counts)
        rows.append(
            {
                "bedrooms_raw": bedrooms,
                "bedrooms_model": features.iloc[0]["bedrooms_model"],
                "guests_model": features.iloc[0]["guests_model"],
                "beds_model": features.iloc[0]["beds_model"],
                "preco_previsto": pred[0],
                "diagnosticos": "; ".join(issues),
            }
        )
    print(pd.DataFrame(rows).round(2).to_string(index=False))


def main():
    print("Carregando modelo. A primeira execucao pode levar alguns segundos...", flush=True)
    artifact = joblib.load(MODEL_PATH)
    print(f"Modelo carregado: {artifact['model_name']}")

    modeling_df = pd.read_csv(MODELING_DATASET)
    trained_neighborhood_counts = modeling_df["airbnb_neighborhood"].value_counts().to_dict()

    sample = modeling_df.sample(10, random_state=42)
    sample_pred = np.exp(artifact["pipeline"].predict(sample[artifact["feature_cols"]]))
    sample_mae = np.mean(np.abs(sample["base_price"].to_numpy() - sample_pred))
    print("\nSanity check em 10 linhas da base modelada:")
    print(f"MAE da amostra: R$ {sample_mae:,.2f}")
    print(
        pd.DataFrame(
            {
                "id": sample["id"].to_numpy(),
                "bairro": sample["airbnb_neighborhood"].to_numpy(),
                "segmento": sample["market_segment"].to_numpy(),
                "base_price_real": sample["base_price"].to_numpy(),
                "base_price_previsto": sample_pred,
            }
        ).round(2).to_string(index=False)
    )

    # Edite este bloco para testar um novo imovel.
    new_apartment = pd.DataFrame(
        [
            {
                "airbnb_neighborhood": "Liberdade",
                "ad_title": "Apartamento completo na Liberdade perto do metro",
                "listing_obj_type": "REGULAR",
                "kicker_content_message": "Entire rental unit in Liberdade, perto do metro, wifi e ar condicionado",
                "is_superhost": True,
                "is_new_listing": False,
                "guests": 2,
                "bedrooms": 1,
                "beds": 1,
                "bathrooms": "1",
                "lat": -23.557,
                "lon": -46.635,
            }
        ]
    )

    pred, features, issues, fallback = predict_with_diagnostics(new_apartment, artifact, trained_neighborhood_counts)
    print("\nPredicao para o novo exemplo:")
    print(f"Preco base estimado: R$ {pred[0]:,.2f} por diaria")
    print(f"Segmento: {features.iloc[0]['market_segment']}")
    print(f"Distancia ao centro robusto do mercado: {features.iloc[0]['distance_to_market_center_km']:.1f} km")
    print_model_feature_snapshot(new_apartment, features)

    if issues:
        print("\nDiagnosticos:")
        for issue in issues:
            print(f"- {issue}")
        if fallback:
            print("\nReferencia por comparaveis brutos:")
            print(f"- Match: {fallback['matched_neighborhoods']}")
            print(f"- Anuncios: {fallback['n']}")
            print(f"- Mediana observada: R$ {fallback['median_reference']:,.2f}")
            print(f"- Media observada: R$ {fallback['mean_reference']:,.2f}")

    compare_bedroom_variants(artifact, trained_neighborhood_counts)


if __name__ == "__main__":
    main()
