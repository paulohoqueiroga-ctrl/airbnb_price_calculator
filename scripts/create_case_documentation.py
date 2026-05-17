import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def money(value):
    return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(value):
    return f"{float(value):.2f}%".replace(".", ",")


def num(value, decimals=2):
    return f"{float(value):.{decimals}f}".replace(".", ",")


def intfmt(value):
    return f"{int(value):,}".replace(",", ".")


def markdown_table(df, columns=None, max_rows=None, money_cols=None, pct_cols=None, round_cols=None):
    data = df.copy()
    if columns:
        data = data[columns]
    if max_rows:
        data = data.head(max_rows)

    money_cols = set(money_cols or [])
    pct_cols = set(pct_cols or [])
    round_cols = set(round_cols or [])

    formatted = []
    for _, row in data.iterrows():
        out_row = []
        for col in data.columns:
            value = row[col]
            if pd.isna(value):
                out_row.append("")
            elif col in money_cols:
                out_row.append(money(value))
            elif col in pct_cols:
                out_row.append(pct(value))
            elif col in round_cols:
                out_row.append(num(value))
            else:
                out_row.append(str(value).replace("|", "\\|"))
        formatted.append(out_row)

    headers = [str(c) for c in data.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def read_csv(name, **kwargs):
    return pd.read_csv(TABLES / name, **kwargs)


def main():
    summary = json.loads((OUT / "case_summary.json").read_text(encoding="utf-8"))
    audit = read_csv("data_audit.csv")
    coverage = read_csv("id_coverage.csv")
    filter_report = read_csv("model_filter_report.csv")
    raw_target = read_csv("raw_target_distribution.csv", index_col=0)
    final_target = read_csv("final_target_distribution.csv", index_col=0)
    metrics = read_csv("model_metrics.csv")
    seg_price = read_csv("segment_price_band.csv")
    seg_market_path = TABLES / "segment_market_robust.csv"
    seg_market = pd.read_csv(seg_market_path) if seg_market_path.exists() else pd.DataFrame()
    seg_bed = read_csv("segment_bedrooms.csv")
    seg_neigh = read_csv("segment_neighborhood.csv")
    importance = read_csv("feature_importance.csv")
    scenario = read_csv("scenario_impacts.csv")
    location = read_csv("location_effects.csv")
    los = read_csv("los_price_diagnostic.csv")
    raw_neigh = read_csv("raw_neighborhood_prices.csv")
    remote = read_csv("remote_location_groups.csv")

    best = metrics.iloc[0]
    scenario_bed = scenario[scenario["scenario"].str.contains("bedroom")].iloc[0]
    scenario_bath = scenario[scenario["scenario"].str.contains("bathroom")].iloc[0]
    loc_no_generic = location[location["airbnb_neighborhood"].str.lower() != "são paulo"].copy()
    loc_top = loc_no_generic.sort_values("predicted_representative_price", ascending=False).head(8)
    loc_bottom = loc_no_generic.sort_values("predicted_representative_price").head(8)

    raw_target_table = raw_target.reset_index().rename(columns={"index": "stat"})
    final_target_table = final_target.reset_index().rename(columns={"index": "stat"})

    text = f"""
# Tabas - Documentação do Case de Base Price

Esta documentação acompanha o notebook [`notebooks/01_base_price_modeling_tabas.ipynb`](../notebooks/01_base_price_modeling_tabas.ipynb) e explica o que foi feito, por que cada decisão foi tomada e onde cada ponto do enunciado foi atendido.

## Entregáveis gerados

- Notebook executado: [`notebooks/01_base_price_modeling_tabas.ipynb`](../notebooks/01_base_price_modeling_tabas.ipynb)
- Modelo treinado: [`models/tabas_base_price_model.joblib`](../models/tabas_base_price_model.joblib)
- Predições de holdout: [`outputs/holdout_predictions.csv`](../outputs/holdout_predictions.csv)
- Tabelas: [`outputs/tables/`](../outputs/tables/)
- Figuras: [`outputs/figures/`](../outputs/figures/)
- Resumo estruturado: [`outputs/case_summary.json`](../outputs/case_summary.json)

## Checklist do enunciado

| Ponto do case | Como foi atendido |
|---|---|
| Data cleaning e preprocessing | Conversão de datas, parsing de banheiros, remoção de índice artificial, filtros de preço, filtros geográficos e filtros estruturais. |
| Exploratory analysis | Distribuição de preço, cauda extrema, preço por bairro, diagnóstico geográfico, LOS e segmentos premium/estendidos. |
| Tratamento de `check_in`, `check_out`, `los`, `reference_date` | Usados apenas para construir/auditar o alvo; excluídos das features finais para não modelar sazonalidade. |
| Feature engineering | Capacidade, banheiros, razões estruturais, lat/lon, distância ao centro robusto, bairro, flags textuais e metadados do anúncio. |
| Geographic information | Bairro Airbnb + latitude/longitude + distância ao centro robusto; remoção apenas de contaminações fortes acima de 150 km e segmentação de praia/premium. |
| Model training | Comparação de baseline por bairro, Huber, Ridge, Random Forest, Extra Trees e HistGradientBoosting. |
| Validation and evaluation | Split por anúncio, estratificado por log-preço; métricas MAE, RMSE, MAPE, MdAPE, R2_log e Bias. |
| Segment consistency | Avaliação por faixa de preço, número de quartos e principais bairros. |
| Interpretation | Importância por permutação, impacto de +1 quarto, +1 banheiro e mudança de localização. |
| Production readiness | Pipeline salvo, features listadas, filtros e limitações documentados; recomendações de monitoramento e reprocessamento. |
| Bonus dynamic pricing | Proposta de extensão para multiplicadores dinâmicos sobre o preço base. |

## 1. Enquadramento do problema

O objetivo foi estimar o preço base estrutural por diária de um imóvel. A unidade de modelagem escolhida foi o anúncio (`id`), não cada linha de preço, porque a base `airbnb_prices` contém múltiplas cotações por anúncio e essas cotações carregam efeitos de calendário e duração da estadia.

A decisão central foi separar dois problemas:

- **Preço base estrutural:** valor esperado por imóvel dado estrutura, localização e atributos observáveis.
- **Preço dinâmico:** ajustes por sazonalidade, eventos, feriados, antecedência, ocupação e LOS.

Este case resolve apenas o primeiro. Por isso, `check_in`, `check_out`, `los` e `reference_date` não foram usados como preditores finais.

## 2. Dados e cobertura

{markdown_table(audit)}

Cobertura do merge:

{markdown_table(coverage)}

Pontos relevantes:

- `airbnb_prices` tem muitas linhas por anúncio, então treinar por linha causaria vazamento e super-representaria anúncios com mais cotações.
- `airbnb_apart` tem 19.068 anúncios, mas apenas 11.494 têm preço observado na base de preços.
- `cleaning_fee` e `airbnb_service_fee` estão totalmente ausentes nesta extração e foram excluídos.
- `bathrooms` veio como texto e foi convertido para número.

## 3. Construção do alvo estrutural

O alvo foi construído em cinco passos:

1. Manter apenas `price_per_night > 0`.
2. Remover linhas extremas fora de p0,1 e p99,9 de `price_per_night`: {money(summary["row_price_filter"]["p001"])} a {money(summary["row_price_filter"]["p999"])}.
3. Agregar por anúncio com a mediana de `price_per_night`.
4. Exigir pelo menos 5 cotações por anúncio.
5. Aplicar filtros finais de modelagem: sanidade estrutural ampla, raio de 150 km e segmentação `urban_core`, `extended_market` e `beach_premium`, sem corte final do alvo agregado.

Relatório de filtros:

{markdown_table(filter_report)}

Distribuição do alvo antes dos filtros finais:

{markdown_table(raw_target_table, money_cols=["base_price"])}

Distribuição do alvo final de modelagem:

{markdown_table(final_target_table, money_cols=["base_price"])}

O alvo final ficou com {intfmt(summary["n_modeling_rows"])} anúncios, treino com {intfmt(summary["n_train"])} e teste com {intfmt(summary["n_test"])}.
Faixa final do alvo agregado: {money(summary["target_range"]["min"])} a {money(summary["target_range"]["max"])}.

## 4. Decisão sobre variáveis temporais e LOS

A base contém estadias de 3 e 15 noites. A mediana de `price_per_night` varia por `los`, o que confirma que duração de estadia tem componente dinâmico/comercial.

{markdown_table(los[["los", "count", "mean", "50%", "95%", "99%"]], money_cols=["mean", "50%", "95%", "99%"])}

Decisão:

- `los`, `check_in`, `check_out` e `reference_date` foram usados para diagnosticar o alvo e medir variação de janelas.
- Eles foram excluídos das features finais para evitar que o modelo aprendesse preço de temporada ou promoção.
- A mediana por anúncio foi escolhida por ser robusta a picos de calendário e descontos pontuais.

## 5. EDA e insights

Principais achados:

- A distribuição bruta de preço tem cauda muito longa: mediana {money(raw_target.loc["50%", "base_price"])}, p95 {money(raw_target.loc["95%", "base_price"])}, p99 {money(raw_target.loc["99%", "base_price"])} e máximo {money(raw_target.loc["max", "base_price"])}.
- Parte dos maiores preços aparece em regiões de praia ou fora do núcleo urbano, como Bertioga e Riviera de São Lourenço; esses casos foram mantidos como segmento `beach_premium` quando geograficamente coerentes.
- Há contaminação geográfica apesar de `city` ser constante, então localização precisava ser tratada explicitamente.
- O label genérico `São Paulo` em `airbnb_neighborhood` é heterogêneo; por isso lat/lon são importantes.

Bairros/regiões com maior mediana antes dos filtros finais:

{markdown_table(raw_neigh[["airbnb_neighborhood", "n", "median_price", "mean_price", "median_distance_km"]], max_rows=12, money_cols=["median_price", "mean_price"], round_cols=["median_distance_km"])}

Principais grupos acima de 40 km do centro robusto, usados para decidir segmentação e remoção apenas acima de 150 km:

{markdown_table(remote[["airbnb_neighborhood", "n", "median_distance_km", "median_price"]], max_rows=12, money_cols=["median_price"], round_cols=["median_distance_km"])}

Como isso influenciou a modelagem:

- Usei log do preço como alvo de treino para reduzir assimetria.
- Usei perda absoluta no boosting para reduzir sensibilidade a outliers.
- Mantive Bertioga/Riviera e demais caudas válidas como `beach_premium`.
- Removi apenas contaminações geográficas fortes acima de 150 km.
- Não houve corte final p99 do alvo agregado; a cauda é tratada com log-preço, perda absoluta e caps robustos de features.

## 6. Features usadas

Features numéricas:

- `guests`, `bedrooms`, `beds`, `bathrooms_num`
- `bath_per_bedroom`, `beds_per_guest`, `guests_per_bedroom`
- `lat`, `lon`, `distance_to_market_center_km`
- tamanho do título e descrição
- flags de texto: studio, loft, luxo, vista, metrô, garagem, piscina, academia, Wi-Fi, ar-condicionado, praia e varanda
- `is_superhost`, `is_new_listing`

Features categóricas:

- `airbnb_neighborhood`
- `listing_obj_type`

Reviews e ratings foram avaliados, mas não entraram no modelo final para manter portabilidade para imóveis novos da Tabas. O ganho observado não compensou a dependência de informação que pode não existir em produção.

## 7. Modelos e escolha final

Métricas no holdout:

{markdown_table(metrics[["model", "MAE", "RMSE", "MAPE", "MdAPE", "R2_log", "Bias"]], money_cols=["MAE", "RMSE", "Bias"], pct_cols=["MAPE", "MdAPE"], round_cols=["R2_log"])}

Modelo final: **{summary["final_model"]}**.

Justificativa:

- Melhor MAE no holdout: {money(best["MAE"])}.
- MdAPE de {pct(best["MdAPE"])}, indicando erro percentual mediano aceitável para um mercado ruidoso.
- Usa perda absoluta em log-preço, boa combinação de robustez e flexibilidade.
- Supera o baseline por bairro em MAE, MAPE e R2_log.

## 8. Consistência por segmentos

Por faixa de preço:

{markdown_table(seg_price[["price_band", "n", "actual_median", "pred_median", "MAE", "MAPE", "Bias"]], money_cols=["actual_median", "pred_median", "MAE", "Bias"], pct_cols=["MAPE"])}

Por segmento de mercado:

{markdown_table(seg_market[["market_segment", "n", "actual_median", "pred_median", "MAE", "MAPE", "Bias"]], money_cols=["actual_median", "pred_median", "MAE", "Bias"], pct_cols=["MAPE"]) if not seg_market.empty else "Tabela nao gerada."}

Por número de quartos:

{markdown_table(seg_bed[["bedrooms", "n", "actual_median", "MAE", "MAPE", "Bias"]], money_cols=["actual_median", "MAE", "Bias"], pct_cols=["MAPE"])}

Principais bairros do holdout:

{markdown_table(seg_neigh[["airbnb_neighborhood", "n", "actual_median", "pred_median", "MAE", "MAPE", "Bias"]], max_rows=12, money_cols=["actual_median", "pred_median", "MAE", "Bias"], pct_cols=["MAPE"])}

Leitura:

- O modelo é mais estável no miolo da distribuição.
- A faixa alta e `beach_premium` ainda têm maior erro e bias negativo, mas agora entram no treino e são identificados explicitamente.
- Imóveis de 3+ quartos têm menos exemplos e maior erro; esse segmento deve ter monitoramento separado.

## 9. Interpretação dos drivers

Top features por importância de permutação:

{markdown_table(importance[["feature", "importance_mae", "importance_std"]], max_rows=12, money_cols=["importance_mae", "importance_std"])}

Os principais drivers são localização (`lon`, `lat`, distância e bairro), banheiros, quartos e capacidade. Isso é coerente com o objetivo estrutural: preço base é principalmente função de localização e configuração do imóvel.

Impactos contrafactuais estimados no holdout:

- +1 quarto, mantendo demais variáveis constantes: mediana de {money(scenario_bed["median_delta_rs"])}, ou {pct(scenario_bed["median_delta_pct"])}.
- +1 banheiro, mantendo demais variáveis constantes: mediana de {money(scenario_bath["median_delta_rs"])}, ou {pct(scenario_bath["median_delta_pct"])}.

Mudança de localização para um apartamento representativo de 1 quarto e 1 banheiro, excluindo o label genérico `São Paulo`:

Maiores preços previstos:

{markdown_table(loc_top[["airbnb_neighborhood", "train_n", "observed_median", "predicted_representative_price"]], money_cols=["observed_median", "predicted_representative_price"])}

Menores preços previstos:

{markdown_table(loc_bottom[["airbnb_neighborhood", "train_n", "observed_median", "predicted_representative_price"]], money_cols=["observed_median", "predicted_representative_price"])}

## 10. Estabilidade, generalização e limitações

O modelo é razoavelmente generalizável para o mercado principal observado porque:

- Usa split por anúncio.
- Remove variáveis explicitamente dinâmicas.
- Usa filtros replicáveis.
- Usa features disponíveis para novos imóveis.
- Mantém modelo flexível, mas não excessivamente complexo.

Limitações importantes:

- Qualidade real do imóvel é parcialmente observada apenas via texto; não há fotos, metragem, andar, condomínio, vista real, padrão de mobiliário ou amenidades detalhadas.
- Alguns bairros têm poucos exemplos.
- A cauda premium é subestimada e precisa de camada especializada ou revisão humana.
- O filtro de 150 km foi calibrado para remover contaminações fortes e manter caudas válidas de praia; em múltiplas cidades deve ser parametrizado por mercado.
- O alvo é uma estimativa do mercado Airbnb, não necessariamente o preço ótimo de receita da Tabas.

## 11. Como operacionalizar na Tabas

Fluxo recomendado:

1. Receber atributos estruturais do imóvel e coordenadas.
2. Aplicar a mesma feature engineering versionada do notebook.
3. Prever preço base com o pipeline salvo.
4. Aplicar guardrails: mínimo/máximo por cidade, bairro, quartos e faixa de confiança.
5. Encaminhar para revisão manual imóveis fora de distribuição: luxo extremo, distância atípica acima do raio treinado, coordenadas inconsistentes ou previsão muito acima dos comparáveis.
6. Monitorar MAE/MAPE por cidade, bairro, quartos e faixa de preço.
7. Reestimar o alvo periodicamente com novas cotações de mercado.

Artefato atual: `models/tabas_base_price_model.joblib`.

### Como testar corretamente

O pipeline salvo espera as features de modelagem já derivadas. Para testar um imóvel bruto, use `scripts/test_model.py` ou replique a função `add_features()` antes de chamar `predict`.

Ponto importante: o modelo não usa `bedrooms`, `guests`, `beds` e `bathrooms` brutos diretamente. Ele usa:

- `bedrooms_model`
- `guests_model`
- `beds_model`
- `bathrooms_model`

Essas features são criadas com caps robustos para evitar extrapolação. Exemplo: `bedrooms = 12` vira `bedrooms_model = 10`. Se você altera apenas `bedrooms` em um dataframe já processado e chama `artifact["pipeline"].predict(...)`, a previsão pode ficar igual porque `bedrooms_model` não foi recalculado.

Use este comando para ver uma comparação controlada de Liberdade variando quartos:

```powershell
.\\.venv\\Scripts\\python scripts\\test_model.py
```

## 12. Extensão futura para dynamic pricing

Para evoluir de preço base para preço dinâmico, a arquitetura recomendada é multiplicativa:

`preço final = preço base estrutural × fator sazonal × fator evento × fator antecedência × fator LOS × fator ocupação × restrições comerciais`

Próximos componentes:

- Modelo de sazonalidade por cidade/bairro e dia da semana.
- Calendário de eventos e feriados.
- Curvas de antecedência e last-minute.
- Elasticidade por LOS.
- Otimização com ocupação, pickup e inventário Tabas.
- Experimentos A/B ou backtesting com receita, ADR, RevPAR e ocupação.

## 13. Reprodutibilidade

Ambiente usado:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python -m pip install -r requirements.txt
```

Para recriar o notebook:

```powershell
.\\.venv\\Scripts\\python scripts\\create_case_notebook.py
```

O notebook já foi executado nesta entrega; para reexecutar, abra `notebooks/01_base_price_modeling_tabas.ipynb` no Jupyter/VS Code usando o kernel do `.venv`.
"""

    (DOCS / "case_documentation.md").write_text(text.strip() + "\n", encoding="utf-8")
    print(DOCS / "case_documentation.md")


if __name__ == "__main__":
    main()
