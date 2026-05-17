from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_base_price_modeling_tabas.ipynb"


def add_markdown(cells, text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def add_code(cells, text):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }

    cells = []

    add_markdown(
        cells,
        """
        # Tabas - Base Price Modeling for Short-Term Rentals

        Este notebook resolve o case de Data Scientist da Tabas usando as duas bases locais:

        - `airbnb_apart (1).csv`: atributos estruturais e metadados dos anúncios.
        - `airbnb_prices (1).csv`: cotações por janela de estadia, preços e avaliações.

        A solução estima um **preço base estrutural por diária**. O foco é capturar valor de imóvel, localização e qualidade observável, sem transformar o problema em precificação dinâmica por data, evento, feriado ou ocupação.
        """,
    )

    add_markdown(
        cells,
        """
        ## 0. Leitura dos requisitos do case

        O enunciado pede explicitamente:

        1. Limpeza e pré-processamento das bases.
        2. Análise exploratória cobrindo distribuição de preço, outliers, extremos, preço por bairro e outros insights relevantes.
        3. Decisão clara sobre `check_in`, `check_out`, `los` e `reference_date`, para não modelar sazonalidade/dinâmica.
        4. Engenharia de atributos, incluindo tratamento geográfico.
        5. Treino de modelo para estimar preço base.
        6. Validação com métricas justificadas.
        7. Avaliação por segmentos.
        8. Interpretação dos drivers de preço, incluindo impacto aproximado de +1 quarto, +1 banheiro e mudança de localização.
        9. Discussão de estabilidade/generalização, limitações, produção e extensões para pricing dinâmico.

        A estratégia abaixo segue esses pontos como uma trilha auditável.
        """,
    )

    add_code(
        cells,
        r"""
        from pathlib import Path
        import json
        import re
        import unicodedata
        import warnings

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        from IPython.display import display
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import HuberRegressor, Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        warnings.filterwarnings("ignore")
        pd.set_option("display.max_columns", 120)
        pd.set_option("display.width", 160)
        sns.set_theme(style="whitegrid", context="notebook")

        RANDOM_STATE = 42
        ROOT = Path.cwd()
        if not (ROOT / "airbnb_apart (1).csv").exists() and (ROOT.parent / "airbnb_apart (1).csv").exists():
            ROOT = ROOT.parent
        DATA_APART = ROOT / "airbnb_apart (1).csv"
        DATA_PRICES = ROOT / "airbnb_prices (1).csv"
        OUT = ROOT / "outputs"
        FIG_DIR = OUT / "figures"
        TABLE_DIR = OUT / "tables"
        MODEL_DIR = ROOT / "models"

        for path in [OUT, FIG_DIR, TABLE_DIR, MODEL_DIR]:
            path.mkdir(parents=True, exist_ok=True)

        print(f"Workspace: {ROOT}")
        print(f"Arquivos encontrados: {DATA_APART.name}, {DATA_PRICES.name}")
        """,
    )

    add_markdown(
        cells,
        """
        ## 1. Carga e auditoria inicial

        A primeira etapa verifica tamanho, tipos, nulos, duplicatas e cobertura do `id` entre as bases. Isso define a unidade correta de modelagem e evita vazamento por repetição de linhas de preço.
        """,
    )

    add_code(
        cells,
        r"""
        apart_raw = pd.read_csv(DATA_APART).drop(columns=["Unnamed: 0"], errors="ignore")
        prices_raw = pd.read_csv(DATA_PRICES).drop(columns=["Unnamed: 0"], errors="ignore")

        id_intersection = set(apart_raw["id"]).intersection(set(prices_raw["id"]))

        audit_rows = []
        for name, data in [("airbnb_apart", apart_raw), ("airbnb_prices", prices_raw)]:
            audit_rows.append(
                {
                    "dataset": name,
                    "rows": len(data),
                    "columns": data.shape[1],
                    "unique_ids": data["id"].nunique(),
                    "duplicated_rows": int(data.duplicated().sum()),
                    "memory_mb": round(data.memory_usage(deep=True).sum() / 1e6, 1),
                }
            )

        audit = pd.DataFrame(audit_rows)
        coverage = pd.DataFrame(
            [
                {"metric": "ids em airbnb_apart", "value": apart_raw["id"].nunique()},
                {"metric": "ids em airbnb_prices", "value": prices_raw["id"].nunique()},
                {"metric": "ids com dados nas duas bases", "value": len(id_intersection)},
                {"metric": "apartamentos sem preço observado", "value": apart_raw["id"].nunique() - len(id_intersection)},
                {"metric": "ids de preço sem metadados", "value": prices_raw["id"].nunique() - len(id_intersection)},
            ]
        )

        missing_apart = apart_raw.isna().mean().sort_values(ascending=False).rename("missing_rate").reset_index().rename(columns={"index": "column"})
        missing_prices = prices_raw.isna().mean().sort_values(ascending=False).rename("missing_rate").reset_index().rename(columns={"index": "column"})

        audit.to_csv(TABLE_DIR / "data_audit.csv", index=False)
        coverage.to_csv(TABLE_DIR / "id_coverage.csv", index=False)
        missing_apart.to_csv(TABLE_DIR / "missing_airbnb_apart.csv", index=False)
        missing_prices.to_csv(TABLE_DIR / "missing_airbnb_prices.csv", index=False)

        display(audit)
        display(coverage)
        display(missing_apart.head(10))
        display(missing_prices.head(12))
        """,
    )

    add_markdown(
        cells,
        """
        ### Observações da auditoria

        - A base de preços tem várias linhas por anúncio; portanto, a validação precisa ser por anúncio, não por linha de cotação.
        - Nem todos os anúncios têm preço observado. O modelo usa apenas anúncios com alvo observável, mas uma implementação em produção aceitaria novos imóveis desde que tenham os atributos estruturais necessários.
        - `cleaning_fee` e `airbnb_service_fee` estão completamente ausentes nesta extração; por isso não são usados para o alvo.
        - `bathrooms` vem como texto e precisa ser convertido para número.
        """,
    )

    add_code(
        cells,
        r"""
        def parse_bathroom(value):
            # Converte strings como '1', '1.5' e 'Half-bath' para número.
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


        def haversine_km(lat1, lon1, lat2, lon2):
            # Distância geodésica aproximada em km.
            radius_km = 6371.0088
            phi1 = np.radians(lat1)
            phi2 = np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lon2 - lon1)
            a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
            return 2 * radius_km * np.arcsin(np.sqrt(a))


        def regression_metrics(y_true, y_pred):
            y_pred = np.clip(np.asarray(y_pred), 1, None)
            return {
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
                "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
                "MdAPE": np.median(np.abs((y_true - y_pred) / y_true)) * 100,
                "R2_log": r2_score(np.log(y_true), np.log(y_pred)),
                "Bias": float(np.mean(y_pred - y_true)),
            }


        def rounded_display(data, decimals=2):
            display(data.round(decimals) if hasattr(data, "round") else data)
        """,
    )

    add_markdown(
        cells,
        """
        ## 2. Construção do alvo estrutural

        A base de preços tem janelas de estadia (`check_in`, `check_out`, `los`) e datas de consulta (`reference_date`). Esses campos capturam efeitos dinâmicos: sazonalidade, feriados, antecedência, eventos e descontos por duração. Como o objetivo é **preço base estrutural**, eles não entram como features do modelo final.

        Uso essas variáveis apenas para construir e auditar o alvo:

        1. Removo linhas de preço extremamente improváveis usando os percentis 0,1% e 99,9% de `price_per_night`.
        2. Agrego por `id` com a **mediana** de `price_per_night`, que é mais robusta que média contra picos de calendário e erros de scrape.
        3. Exijo pelo menos 5 cotações por anúncio para reduzir ruído no alvo.
        4. Faço filtros de sanidade geográfica e estrutural, removendo contaminações claras e mantendo a cauda válida de praia/premium.
        5. Não faço corte final do alvo agregado; a robustez vem do uso de `log(base_price)`, perda absoluta e segmentação explícita de mercado.
        """,
    )

    add_code(
        cells,
        r"""
        prices = prices_raw.copy()
        for col in ["check_in", "check_out", "reference_date", "extraction_date"]:
            prices[col] = pd.to_datetime(prices[col], errors="coerce")

        positive_prices = prices.loc[prices["price_per_night"].gt(0)].copy()
        row_price_low, row_price_high = positive_prices["price_per_night"].quantile([0.001, 0.999])
        prices_clean = positive_prices.loc[positive_prices["price_per_night"].between(row_price_low, row_price_high)].copy()

        los_diagnostic = (
            positive_prices.groupby("los")["price_per_night"]
            .describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            .reset_index()
        )

        checkin_diagnostic = (
            positive_prices.groupby(["los", "check_in"], as_index=False)["price_per_night"]
            .median()
            .rename(columns={"price_per_night": "median_price_per_night"})
        )

        base_target = (
            prices_clean.groupby("id")
            .agg(
                base_price=("price_per_night", "median"),
                price_iqr=("price_per_night", lambda s: s.quantile(0.75) - s.quantile(0.25)),
                n_price_obs=("price_per_night", "size"),
                los3_share=("los", lambda s: float((s == 3).mean())),
                review_count=("review_count", "max"),
            )
            .reset_index()
        )

        df = apart_raw.merge(base_target, on="id", how="inner")
        df["bathrooms_num"] = df["bathrooms"].apply(parse_bathroom)

        market_center_lat = df["lat"].median()
        market_center_lon = df["lon"].median()
        df["distance_to_market_center_km"] = haversine_km(market_center_lat, market_center_lon, df["lat"], df["lon"])

        target_build_summary = pd.DataFrame(
            [
                {"step": "linhas de preço originais", "rows": len(prices_raw), "unique_ids": prices_raw["id"].nunique()},
                {"step": "linhas com price_per_night positivo", "rows": len(positive_prices), "unique_ids": positive_prices["id"].nunique()},
                {"step": "linhas após filtro p0.1-p99.9", "rows": len(prices_clean), "unique_ids": prices_clean["id"].nunique()},
                {"step": "alvo agregado por anúncio", "rows": len(base_target), "unique_ids": base_target["id"].nunique()},
                {"step": "merge alvo + atributos", "rows": len(df), "unique_ids": df["id"].nunique()},
            ]
        )

        target_build_summary.to_csv(TABLE_DIR / "target_build_summary.csv", index=False)
        los_diagnostic.to_csv(TABLE_DIR / "los_price_diagnostic.csv", index=False)
        checkin_diagnostic.to_csv(TABLE_DIR / "checkin_price_diagnostic.csv", index=False)

        print(f"Filtro de preço por linha: {row_price_low:,.2f} a {row_price_high:,.2f}")
        print(f"Centro robusto do mercado: lat={market_center_lat:.6f}, lon={market_center_lon:.6f}")
        display(target_build_summary)
        rounded_display(los_diagnostic)
        """,
    )

    add_markdown(
        cells,
        """
        ## 3. EDA: preço, outliers, bairro e geografia

        A EDA abaixo é feita antes do filtro final de modelagem para deixar visíveis os problemas do mercado observado: cauda de preço, anúncios extremos e coordenadas fora do núcleo geográfico.
        """,
    )

    add_code(
        cells,
        r"""
        target_distribution = df["base_price"].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.995]).to_frame("base_price")
        geo_distribution = df["distance_to_market_center_km"].describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99, 0.995]).to_frame("distance_to_market_center_km")

        neighborhood_prices_raw = (
            df.groupby("airbnb_neighborhood")
            .agg(n=("id", "size"), median_price=("base_price", "median"), mean_price=("base_price", "mean"), median_distance_km=("distance_to_market_center_km", "median"))
            .query("n >= 30")
            .sort_values("median_price", ascending=False)
            .reset_index()
        )

        extreme_listings = (
            df.sort_values("base_price", ascending=False)
            [["id", "airbnb_neighborhood", "ad_title", "guests", "bedrooms", "beds", "bathrooms", "lat", "lon", "distance_to_market_center_km", "base_price", "n_price_obs"]]
            .head(25)
        )

        remote_locations = (
            df.loc[df["distance_to_market_center_km"] > 40]
            .groupby("airbnb_neighborhood")
            .agg(n=("id", "size"), median_distance_km=("distance_to_market_center_km", "median"), median_price=("base_price", "median"))
            .sort_values("n", ascending=False)
            .reset_index()
        )

        target_distribution.to_csv(TABLE_DIR / "raw_target_distribution.csv")
        geo_distribution.to_csv(TABLE_DIR / "geo_distance_distribution.csv")
        neighborhood_prices_raw.to_csv(TABLE_DIR / "raw_neighborhood_prices.csv", index=False)
        extreme_listings.to_csv(TABLE_DIR / "extreme_listings.csv", index=False)
        remote_locations.to_csv(TABLE_DIR / "remote_location_groups.csv", index=False)

        display(target_distribution.round(2))
        display(geo_distribution.round(2))
        print("Bairros/regiões com maior mediana de preço, antes do filtro final:")
        display(neighborhood_prices_raw.head(15).round(2))
        print("Anúncios extremos:")
        display(extreme_listings.round(2))
        print("Grupos geográficos acima de 40 km do centro robusto (diagnóstico para segmentação):")
        display(remote_locations.head(20).round(2))
        """,
    )

    add_code(
        cells,
        r"""
        # A distribuição tem cauda longa. Em escala linear o miolo fica espremido
        # perto de zero; por isso usamos eixo X em log para o diagnóstico bruto.
        plt.figure(figsize=(10, 5))
        sns.histplot(df["base_price"], bins=80, log_scale=(True, False), color="#276fbf")
        plt.title("Distribuição do alvo agregado por anúncio")
        plt.xlabel("Mediana de price_per_night por anúncio (R$) - escala log")
        plt.ylabel("Quantidade de anúncios")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "eda_raw_target_distribution.png", dpi=160)
        plt.show()

        # Visualização complementar: zoom no corpo da distribuição até p99.
        price_p99 = df["base_price"].quantile(0.99)
        plt.figure(figsize=(10, 5))
        sns.histplot(df.loc[df["base_price"] <= price_p99, "base_price"], bins=80, color="#276fbf")
        plt.title("Distribuição do alvo agregado por anúncio - zoom até p99")
        plt.xlabel("Mediana de price_per_night por anúncio (R$)")
        plt.ylabel("Quantidade de anúncios")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "eda_raw_target_distribution_zoom_p99.png", dpi=160)
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.histplot(np.log(df["base_price"]), bins=80, color="#2f8f5b")
        plt.title("Distribuição do log do preço base")
        plt.xlabel("log(base_price)")
        plt.ylabel("Quantidade de anúncios")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "eda_log_target_distribution.png", dpi=160)
        plt.show()

        plot_neighborhood = neighborhood_prices_raw.head(18).sort_values("median_price")
        plt.figure(figsize=(10, 7))
        sns.barplot(data=plot_neighborhood, x="median_price", y="airbnb_neighborhood", color="#4b8bbe")
        plt.title("Maiores medianas de preço por bairro/região (n >= 30)")
        plt.xlabel("Mediana de preço base (R$)")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "eda_price_by_neighborhood_raw.png", dpi=160)
        plt.show()

        plt.figure(figsize=(8, 6))
        sample_geo = df.sample(min(len(df), 7000), random_state=RANDOM_STATE)
        sns.scatterplot(data=sample_geo, x="lon", y="lat", hue="distance_to_market_center_km", size="base_price", sizes=(8, 80), alpha=0.45, palette="viridis")
        plt.title("Coordenadas dos anúncios e distância do centro robusto")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "eda_geo_scatter.png", dpi=160)
        plt.show()
        """,
    )

    add_markdown(
        cells,
        """
        ### Decisões de limpeza após a EDA

        A EDA mostra três problemas relevantes:

        - A cauda de preço é muito longa, com diárias agregadas acima de vários milhares de reais. Parte disso é luxo, mas parte parece efeito de dados e janelas específicas.
        - Há coordenadas muito distantes do núcleo de São Paulo, apesar de `city` ser constante. Removo apenas contaminações fortes acima de 150 km, mantendo Bertioga/Riviera como segmento de praia/premium.
        - Existem inconsistências estruturais, como número de quartos muito alto para `Entire rental unit`. Faço filtros amplos de sanidade e caps robustos para não treinar o modelo em anúncios multiunidade ou erros de scrape.
        """,
    )

    add_markdown(
        cells,
        """
        ## 4. Feature engineering e amostra final de modelagem

        Features usadas no modelo final:

        - Capacidade e estrutura: hóspedes, quartos, camas, banheiros e razões derivadas.
        - Geografia: bairro Airbnb, latitude, longitude e distância ao centro robusto do mercado.
        - Metadados do anúncio: superhost, novo anúncio e tipo do objeto.
        - Texto simples do título/descrição: flags para studio, luxo, vista, metrô, garagem, piscina, academia, Wi-Fi, ar-condicionado, praia e varanda.

        Ratings e reviews foram analisados, mas não entram no modelo final escolhido: eles podem melhorar pouco a previsão, porém reduzem portabilidade para imóveis novos da Tabas, onde esses sinais podem não existir.
        """,
    )

    add_code(
        cells,
        r"""
        def add_text_features(data):
            out = data.copy()
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
            return out


        def add_structural_features(data):
            out = add_text_features(data)
            out["is_superhost_int"] = out["is_superhost"].astype(int)
            out["is_new_listing_int"] = out["is_new_listing"].astype(int)
            out["beds"] = out["beds"].astype(float)
            out["market_segment"] = np.select(
                [out["has_beach"].eq(1), out["distance_to_market_center_km"].gt(40)],
                ["beach_premium", "extended_market"],
                default="urban_core",
            )
            out["guests_model"] = out["guests"].clip(1, 16)
            out["bedrooms_model"] = out["bedrooms"].clip(1, 10)
            out["beds_model"] = out["beds"].clip(1, 30)
            out["bathrooms_model"] = out["bathrooms_num"].clip(0.5, 10)
            out["bath_per_bedroom"] = out["bathrooms_model"] / out["bedrooms_model"].clip(lower=1)
            out["beds_per_guest"] = out["beds_model"] / out["guests_model"].clip(lower=1)
            out["guests_per_bedroom"] = out["guests_model"] / out["bedrooms_model"].clip(lower=1)
            return out


        model_df = add_structural_features(df)

        filter_report = []
        filter_report.append({"step": "merge alvo + atributos", "rows": len(model_df)})

        model_df = model_df.loc[model_df["n_price_obs"] >= 5].copy()
        filter_report.append({"step": "mínimo de 5 cotações por anúncio", "rows": len(model_df)})

        model_df = model_df.loc[model_df["distance_to_market_center_km"] <= 150].copy()
        filter_report.append({"step": "raio geográfico <= 150 km, mantendo praia/premium", "rows": len(model_df)})

        structural_mask = (
            model_df["bedrooms"].between(1, 10)
            & model_df["guests"].between(1, 16)
            & model_df["bathrooms_num"].between(0.5, 10)
            & (model_df["beds"].isna() | model_df["beds"].between(1, 30))
        )
        model_df = model_df.loc[structural_mask].copy()
        filter_report.append({"step": "sanidade estrutural", "rows": len(model_df)})

        target_low = model_df["base_price"].min()
        target_high = model_df["base_price"].max()
        filter_report.append({"step": "sem corte final de alvo; robustez via log-preço e perda absoluta", "rows": len(model_df)})

        filter_report = pd.DataFrame(filter_report)
        final_target_distribution = model_df["base_price"].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_frame("base_price")

        filter_report.to_csv(TABLE_DIR / "model_filter_report.csv", index=False)
        final_target_distribution.to_csv(TABLE_DIR / "final_target_distribution.csv")
        model_df.to_csv(OUT / "modeling_dataset.csv", index=False)

        print(f"Faixa final do alvo sem corte adicional: min={target_low:.2f}, max={target_high:.2f}")
        display(filter_report)
        display(final_target_distribution.round(2))
        """,
    )

    add_markdown(
        cells,
        """
        ## 5. Modelagem

        A validação é feita por anúncio, com split treino/teste estratificado por faixas de log-preço. Isso evita que o mesmo `id` apareça em treino e teste por causa das múltiplas linhas de preço.

        Modelos comparados:

        - Baseline: mediana de preço por bairro, com fallback para mediana global.
        - Huber log-linear: robusto e interpretável, útil como benchmark.
        - Ridge log-linear: benchmark linear estável.
        - Random Forest / Extra Trees em log-preço: árvores com bagging.
        - HistGradientBoosting em log-preço com perda absoluta: modelo final escolhido por melhor MAE e boa robustez.

        O alvo é treinado em `log(base_price)`, e as previsões são convertidas de volta para reais. Isso reduz assimetria e evita que a cauda alta domine o treino.
        """,
    )

    add_code(
        cells,
        r"""
        numeric_features = [
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
        categorical_features = ["airbnb_neighborhood", "listing_obj_type", "market_segment"]
        feature_cols = numeric_features + categorical_features

        y = model_df["base_price"].copy()
        strat_bins = pd.qcut(np.log(y), q=5, labels=False, duplicates="drop")
        train_idx, test_idx = train_test_split(model_df.index, test_size=0.20, random_state=RANDOM_STATE, stratify=strat_bins)

        train_df = model_df.loc[train_idx].copy()
        test_df = model_df.loc[test_idx].copy()
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df["base_price"]
        y_test = test_df["base_price"]

        preprocessor_tree = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), numeric_features),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
                        ]
                    ),
                    categorical_features,
                ),
            ],
            verbose_feature_names_out=False,
        )

        preprocessor_scaled = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
                        ]
                    ),
                    categorical_features,
                ),
            ],
            verbose_feature_names_out=False,
        )

        candidate_models = {
            "baseline_neighborhood": None,
            "huber_log": Pipeline(steps=[("preprocess", preprocessor_scaled), ("model", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=1000))]),
            "ridge_log": Pipeline(steps=[("preprocess", preprocessor_scaled), ("model", Ridge(alpha=3.0, random_state=RANDOM_STATE))]),
            "random_forest_log": Pipeline(
                steps=[
                    ("preprocess", preprocessor_tree),
                    ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1)),
                ]
            ),
            "extra_trees_log": Pipeline(
                steps=[
                    ("preprocess", preprocessor_tree),
                    ("model", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1)),
                ]
            ),
            "robust_segmented_hgb_log_mae": Pipeline(
                steps=[
                    ("preprocess", preprocessor_tree),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            loss="absolute_error",
                            learning_rate=0.05,
                            max_iter=500,
                            max_leaf_nodes=31,
                            min_samples_leaf=20,
                            l2_regularization=0.02,
                            early_stopping=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        }

        fitted_models = {}
        predictions = {}
        metric_rows = []

        neighborhood_median = train_df.groupby("airbnb_neighborhood")["base_price"].median()
        global_median = train_df["base_price"].median()

        for model_name, model in candidate_models.items():
            if model is None:
                y_pred = X_test["airbnb_neighborhood"].map(neighborhood_median).fillna(global_median).to_numpy()
            else:
                model.fit(X_train, np.log(y_train))
                y_pred = np.exp(model.predict(X_test))
                fitted_models[model_name] = model
            predictions[model_name] = y_pred
            metric_rows.append({"model": model_name, **regression_metrics(y_test, y_pred)})

        metrics_df = pd.DataFrame(metric_rows).sort_values("MAE").reset_index(drop=True)
        metrics_df.to_csv(TABLE_DIR / "model_metrics.csv", index=False)
        display(metrics_df.round(3))
        """,
    )

    add_markdown(
        cells,
        """
        ## 6. Avaliação e consistência por segmentos

        Métricas usadas:

        - **MAE**: erro absoluto médio em reais por diária; é a métrica principal porque é interpretável e robusta.
        - **RMSE**: penaliza grandes erros, útil para monitorar cauda alta.
        - **MAPE / MdAPE**: erro percentual médio e mediano, úteis para comparar imóveis baratos e caros.
        - **R2 no log**: mede explicação no espaço em que o modelo foi treinado.
        - **Bias**: mostra tendência de sobrepreço ou subpreço.
        """,
    )

    add_code(
        cells,
        r"""
        best_model_name = "robust_segmented_hgb_log_mae"
        best_model = fitted_models[best_model_name]
        best_pred = predictions[best_model_name]

        evaluation_df = test_df.copy()
        evaluation_df["predicted_base_price"] = best_pred
        evaluation_df["error"] = evaluation_df["predicted_base_price"] - evaluation_df["base_price"]
        evaluation_df["abs_error"] = evaluation_df["error"].abs()
        evaluation_df["ape"] = evaluation_df["abs_error"] / evaluation_df["base_price"] * 100
        evaluation_df["price_band"] = pd.qcut(evaluation_df["base_price"], q=4, labels=["low", "mid_low", "mid_high", "high"])

        segment_price_band = (
            evaluation_df.groupby("price_band")
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

        segment_bedrooms = (
            evaluation_df.groupby("bedrooms")
            .agg(n=("id", "size"), actual_median=("base_price", "median"), MAE=("abs_error", "mean"), MAPE=("ape", "mean"), Bias=("error", "mean"))
            .reset_index()
        )

        top_neighborhoods = evaluation_df["airbnb_neighborhood"].value_counts().head(12).index
        segment_neighborhood = (
            evaluation_df.loc[evaluation_df["airbnb_neighborhood"].isin(top_neighborhoods)]
            .groupby("airbnb_neighborhood")
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

        segment_market = (
            evaluation_df.groupby("market_segment")
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

        evaluation_df[["id", "base_price", "predicted_base_price", "error", "abs_error", "ape", "airbnb_neighborhood", "bedrooms", "bathrooms_num", "guests"]].to_csv(
            OUT / "holdout_predictions.csv", index=False
        )
        segment_price_band.to_csv(TABLE_DIR / "segment_price_band.csv", index=False)
        segment_bedrooms.to_csv(TABLE_DIR / "segment_bedrooms.csv", index=False)
        segment_neighborhood.to_csv(TABLE_DIR / "segment_neighborhood.csv", index=False)
        segment_market.to_csv(TABLE_DIR / "segment_market_robust.csv", index=False)

        print("Modelo final:", best_model_name)
        display(metrics_df.loc[metrics_df["model"].eq(best_model_name)].round(3))
        print("Segmentos por faixa de preço:")
        display(segment_price_band.round(2))
        print("Segmentos de mercado:")
        display(segment_market.round(2))
        print("Segmentos por número de quartos:")
        display(segment_bedrooms.round(2))
        print("Principais bairros no holdout:")
        display(segment_neighborhood.round(2))
        """,
    )

    add_code(
        cells,
        r"""
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=evaluation_df["base_price"], y=evaluation_df["predicted_base_price"], alpha=0.45, s=24, color="#276fbf")
        max_axis = max(evaluation_df["base_price"].max(), evaluation_df["predicted_base_price"].max())
        plt.plot([0, max_axis], [0, max_axis], color="#333333", linewidth=1.2)
        plt.title("Preço real vs. previsto no holdout")
        plt.xlabel("Preço base observado (R$)")
        plt.ylabel("Preço base previsto (R$)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "model_actual_vs_predicted.png", dpi=160)
        plt.show()

        plt.figure(figsize=(9, 5))
        sns.boxplot(data=evaluation_df, x="price_band", y="ape", color="#70a288")
        plt.ylim(0, min(150, evaluation_df["ape"].quantile(0.98)))
        plt.title("Erro percentual absoluto por faixa de preço")
        plt.xlabel("Faixa de preço")
        plt.ylabel("APE (%)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "model_ape_by_price_band.png", dpi=160)
        plt.show()
        """,
    )

    add_markdown(
        cells,
        """
        ## 7. Interpretação dos drivers

        A interpretação combina:

        - Importância por permutação no holdout, medindo quanto o MAE piora quando cada feature é embaralhada.
        - Cenários contrafactuais para estimar impacto médio de +1 quarto e +1 banheiro mantendo o restante constante.
        - Cenário de localização para um apartamento representativo de 1 quarto e 1 banheiro, trocando bairro e coordenadas para medianas dos bairros.
        """,
    )

    add_code(
        cells,
        r"""
        def negative_mae_original_scale(estimator, X, y_log):
            y_true = np.exp(y_log)
            y_pred = np.exp(estimator.predict(X))
            return -mean_absolute_error(y_true, y_pred)


        permutation = permutation_importance(
            best_model,
            X_test,
            np.log(y_test),
            scoring=negative_mae_original_scale,
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        feature_importance = (
            pd.DataFrame(
                {
                    "feature": X_test.columns,
                    "importance_mae": permutation.importances_mean,
                    "importance_std": permutation.importances_std,
                }
            )
            .sort_values("importance_mae", ascending=False)
            .reset_index(drop=True)
        )
        feature_importance.to_csv(TABLE_DIR / "feature_importance.csv", index=False)

        display(feature_importance.head(20).round(3))

        plot_imp = feature_importance.head(15).sort_values("importance_mae")
        plt.figure(figsize=(9, 6))
        sns.barplot(data=plot_imp, x="importance_mae", y="feature", color="#4b8bbe")
        plt.title("Importância por permutação: aumento de MAE ao embaralhar feature")
        plt.xlabel("Aumento aproximado de MAE (R$)")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "model_permutation_importance.png", dpi=160)
        plt.show()
        """,
    )

    add_code(
        cells,
        r"""
        def update_derived_features(data):
            out = data.copy()
            out["bedrooms_model"] = out["bedrooms_model"].clip(1, 10)
            out["bathrooms_model"] = out["bathrooms_model"].clip(0.5, 10)
            out["beds_model"] = out["beds_model"].clip(1, 30)
            out["guests_model"] = out["guests_model"].clip(1, 16)
            out["bath_per_bedroom"] = out["bathrooms_model"] / out["bedrooms_model"].clip(lower=1)
            out["beds_per_guest"] = out["beds_model"] / out["guests_model"].clip(lower=1)
            out["guests_per_bedroom"] = out["guests_model"] / out["bedrooms_model"].clip(lower=1)
            return out


        scenario_rows = []

        bedroom_mask = test_df["bedrooms_model"] < 10
        x0 = test_df.loc[bedroom_mask, feature_cols].copy()
        x1 = update_derived_features(x0.assign(bedrooms_model=x0["bedrooms_model"] + 1))
        p0 = np.exp(best_model.predict(x0))
        p1 = np.exp(best_model.predict(x1[feature_cols]))
        delta = p1 - p0
        scenario_rows.append(
            {
                "scenario": "+1 bedroom, demais variáveis constantes",
                "n": int(bedroom_mask.sum()),
                "median_delta_rs": float(np.median(delta)),
                "mean_delta_rs": float(np.mean(delta)),
                "median_delta_pct": float(np.median(delta / p0 * 100)),
                "mean_delta_pct": float(np.mean(delta / p0 * 100)),
            }
        )

        bathroom_mask = test_df["bathrooms_model"] < 10
        x0 = test_df.loc[bathroom_mask, feature_cols].copy()
        x1 = update_derived_features(x0.assign(bathrooms_model=x0["bathrooms_model"] + 1))
        p0 = np.exp(best_model.predict(x0))
        p1 = np.exp(best_model.predict(x1[feature_cols]))
        delta = p1 - p0
        scenario_rows.append(
            {
                "scenario": "+1 bathroom, demais variáveis constantes",
                "n": int(bathroom_mask.sum()),
                "median_delta_rs": float(np.median(delta)),
                "mean_delta_rs": float(np.mean(delta)),
                "median_delta_pct": float(np.median(delta / p0 * 100)),
                "mean_delta_pct": float(np.mean(delta / p0 * 100)),
            }
        )

        scenario_impacts = pd.DataFrame(scenario_rows)
        scenario_impacts.to_csv(TABLE_DIR / "scenario_impacts.csv", index=False)
        display(scenario_impacts.round(2))
        """,
    )

    add_code(
        cells,
        r"""
        # Apartamento representativo para simular mudança de localização.
        representative = {col: train_df[col].median() for col in numeric_features}
        representative.update({col: train_df[col].mode().iloc[0] for col in categorical_features})
        representative = pd.DataFrame([representative])
        representative["guests_model"] = 2
        representative["bedrooms_model"] = 1
        representative["beds_model"] = 1
        representative["bathrooms_model"] = 1
        representative = update_derived_features(representative)

        neighborhood_stats = (
            train_df.groupby("airbnb_neighborhood")
            .agg(
                train_n=("id", "size"),
                med_lat=("lat", "median"),
                med_lon=("lon", "median"),
                med_dist=("distance_to_market_center_km", "median"),
                observed_median=("base_price", "median"),
            )
            .query("train_n >= 40")
        )

        location_rows = []
        for neighborhood, row in neighborhood_stats.iterrows():
            scenario = representative.copy()
            scenario["airbnb_neighborhood"] = neighborhood
            scenario["lat"] = row["med_lat"]
            scenario["lon"] = row["med_lon"]
            scenario["distance_to_market_center_km"] = row["med_dist"]
            predicted = float(np.exp(best_model.predict(scenario[feature_cols]))[0])
            location_rows.append(
                {
                    "airbnb_neighborhood": neighborhood,
                    "train_n": int(row["train_n"]),
                    "observed_median": float(row["observed_median"]),
                    "predicted_representative_price": predicted,
                }
            )

        location_effects = pd.DataFrame(location_rows)
        location_effects_no_generic = location_effects.loc[location_effects["airbnb_neighborhood"].str.lower() != "são paulo"].copy()
        location_effects_no_generic = location_effects_no_generic.sort_values("predicted_representative_price", ascending=False).reset_index(drop=True)
        location_effects_no_generic.to_csv(TABLE_DIR / "location_effects.csv", index=False)

        print("Maiores preços previstos para apartamento representativo, excluindo label genérico 'São Paulo':")
        display(location_effects_no_generic.head(15).round(2))
        print("Menores preços previstos para apartamento representativo:")
        display(location_effects_no_generic.tail(15).round(2))
        """,
    )

    add_markdown(
        cells,
        """
        ## 8. Artefatos de produção e resumo executivo

        O objeto salvo em `models/tabas_base_price_model.joblib` contém o pipeline de pré-processamento tabular e o modelo final. A engenharia de features anterior ao pipeline (`bathrooms_num`, distância, `market_segment`, flags de texto e razões estruturais) está documentada neste notebook e deve ser empacotada como transformação versionada em produção.

        Atenção para testes manuais: o pipeline final não usa `bedrooms`, `guests`, `beds` e `bathrooms` brutos diretamente. Ele usa `bedrooms_model`, `guests_model`, `beds_model` e `bathrooms_model`, gerados antes do `predict`. Portanto, mudar apenas `bedrooms` em um dataframe já processado não altera a previsão; é necessário refazer a feature engineering. Também há caps robustos: por exemplo, 12 quartos vira `bedrooms_model = 10`, para evitar extrapolação instável fora da amostra.
        """,
    )

    add_code(
        cells,
        r"""
        production_model = best_model
        production_model.fit(model_df[feature_cols], np.log(model_df["base_price"]))

        model_artifact = {
            "model_name": best_model_name,
            "pipeline": production_model,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_cols": feature_cols,
            "market_center_lat": float(market_center_lat),
            "market_center_lon": float(market_center_lon),
            "row_price_low": float(row_price_low),
            "row_price_high": float(row_price_high),
            "target_low": float(target_low),
            "target_high": float(target_high),
            "max_training_distance_km": 150.0,
            "structural_caps": {"guests": 16, "bedrooms": 10, "beds": 30, "bathrooms": 10},
        }
        joblib.dump(model_artifact, MODEL_DIR / "tabas_base_price_model.joblib")

        best_metrics = metrics_df.loc[metrics_df["model"].eq(best_model_name)].iloc[0].to_dict()
        summary = {
            "final_model": best_model_name,
            "n_modeling_rows": int(len(model_df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "row_price_filter": {"p001": float(row_price_low), "p999": float(row_price_high)},
            "target_range": {"min": float(target_low), "max": float(target_high)},
            "segment_counts": model_df["market_segment"].value_counts().to_dict(),
            "market_center": {"lat": float(market_center_lat), "lon": float(market_center_lon)},
            "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in best_metrics.items()},
            "top_features": feature_importance.head(10).to_dict(orient="records"),
            "scenario_impacts": scenario_impacts.to_dict(orient="records"),
        }
        with open(OUT / "case_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("Artefatos salvos:")
        print(f"- {MODEL_DIR / 'tabas_base_price_model.joblib'}")
        print(f"- {OUT / 'case_summary.json'}")
        print(f"- {TABLE_DIR}")
        print(f"- {FIG_DIR}")
        display(pd.DataFrame([summary["metrics"]]).round(3))
        """,
    )

    add_markdown(
        cells,
        """
        ## 9. Conclusões

        - O preço base foi definido no nível do anúncio, usando mediana robusta das cotações por `id`.
        - `check_in`, `check_out`, `los` e `reference_date` foram excluídos das features finais para evitar modelagem dinâmica; eles foram usados apenas na auditoria/construção do alvo.
        - O modelo final é interpretável por importâncias, cenários contrafactuais e avaliação por segmentos.
        - A cauda alta foi mantida no treinamento, mas imóveis premium/luxo e praia têm maior erro e devem ter monitoramento/governança separados.
        - Para produção, a recomendação é versionar a transformação de features, monitorar drift por cidade/bairro/faixa de preço e reestimar o alvo com novas cotações periodicamente.
        """,
    )

    notebook.cells = cells
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook criado em {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
