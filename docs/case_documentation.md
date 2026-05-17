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

| dataset | rows | columns | unique_ids | duplicated_rows | memory_mb |
| --- | --- | --- | --- | --- | --- |
| airbnb_apart | 19068 | 16 | 19068 | 0 | 12.4 |
| airbnb_prices | 1090726 | 22 | 11494 | 0 | 431.9 |

Cobertura do merge:

| metric | value |
| --- | --- |
| ids em airbnb_apart | 19068 |
| ids em airbnb_prices | 11494 |
| ids com dados nas duas bases | 11494 |
| apartamentos sem preço observado | 7574 |
| ids de preço sem metadados | 0 |

Pontos relevantes:

- `airbnb_prices` tem muitas linhas por anúncio, então treinar por linha causaria vazamento e super-representaria anúncios com mais cotações.
- `airbnb_apart` tem 19.068 anúncios, mas apenas 11.494 têm preço observado na base de preços.
- `cleaning_fee` e `airbnb_service_fee` estão totalmente ausentes nesta extração e foram excluídos.
- `bathrooms` veio como texto e foi convertido para número.

## 3. Construção do alvo estrutural

O alvo foi construído em cinco passos:

1. Manter apenas `price_per_night > 0`.
2. Remover linhas extremas fora de p0,1 e p99,9 de `price_per_night`: R$ 101,53 a R$ 63.077,00.
3. Agregar por anúncio com a mediana de `price_per_night`.
4. Exigir pelo menos 5 cotações por anúncio.
5. Aplicar filtros finais de modelagem: sanidade estrutural ampla, raio de 150 km e segmentação `urban_core`, `extended_market` e `beach_premium`, sem corte final do alvo agregado.

Relatório de filtros:

| step | rows |
| --- | --- |
| merge alvo + atributos | 11492 |
| mínimo de 5 cotações por anúncio | 10789 |
| raio geográfico <= 150 km, mantendo praia/premium | 10664 |
| sanidade estrutural | 10618 |
| sem corte final de alvo; robustez via log-preço e perda absoluta | 10618 |

Distribuição do alvo antes dos filtros finais:

| stat | base_price |
| --- | --- |
| count | R$ 11.492,00 |
| mean | R$ 627,97 |
| std | R$ 2.005,65 |
| min | R$ 101,67 |
| 1% | R$ 133,47 |
| 5% | R$ 164,37 |
| 25% | R$ 219,92 |
| 50% | R$ 278,00 |
| 75% | R$ 421,38 |
| 95% | R$ 1.832,33 |
| 99% | R$ 6.482,38 |
| 99.5% | R$ 11.094,87 |
| max | R$ 56.551,00 |

Distribuição do alvo final de modelagem:

| stat | base_price |
| --- | --- |
| count | R$ 10.618,00 |
| mean | R$ 623,07 |
| std | R$ 2.062,42 |
| min | R$ 101,67 |
| 1% | R$ 133,14 |
| 5% | R$ 163,67 |
| 25% | R$ 218,60 |
| 50% | R$ 274,57 |
| 75% | R$ 408,05 |
| 95% | R$ 1.816,60 |
| 99% | R$ 6.427,32 |
| max | R$ 56.551,00 |

O alvo final ficou com 10.618 anúncios, treino com 8.494 e teste com 2.124.
Faixa final do alvo agregado: R$ 101,67 a R$ 56.551,00.

## 4. Decisão sobre variáveis temporais e LOS

A base contém estadias de 3 e 15 noites. A mediana de `price_per_night` varia por `los`, o que confirma que duração de estadia tem componente dinâmico/comercial.

| los | count | mean | 50% | 95% | 99% |
| --- | --- | --- | --- | --- | --- |
| 3.0 | 102980.0 | R$ 512,62 | R$ 321,67 | R$ 1.347,50 | R$ 2.901,31 |
| 15.0 | 987746.0 | R$ 1.174,69 | R$ 274,47 | R$ 4.519,00 | R$ 13.383,00 |

Decisão:

- `los`, `check_in`, `check_out` e `reference_date` foram usados para diagnosticar o alvo e medir variação de janelas.
- Eles foram excluídos das features finais para evitar que o modelo aprendesse preço de temporada ou promoção.
- A mediana por anúncio foi escolhida por ser robusta a picos de calendário e descontos pontuais.

## 5. EDA e insights

Principais achados:

- A distribuição bruta de preço tem cauda muito longa: mediana R$ 278,00, p95 R$ 1.832,33, p99 R$ 6.482,38 e máximo R$ 56.551,00.
- Parte dos maiores preços aparece em regiões de praia ou fora do núcleo urbano, como Bertioga e Riviera de São Lourenço; esses casos foram mantidos como segmento `beach_premium` quando geograficamente coerentes.
- Há contaminação geográfica apesar de `city` ser constante, então localização precisava ser tratada explicitamente.
- O label genérico `São Paulo` em `airbnb_neighborhood` é heterogêneo; por isso lat/lon são importantes.

Bairros/regiões com maior mediana antes dos filtros finais:

| airbnb_neighborhood | n | median_price | mean_price | median_distance_km |
| --- | --- | --- | --- | --- |
| Bertioga | 119 | R$ 2.426,33 | R$ 5.436,78 | 71,57 |
| Riviera De Sao Lourenco | 285 | R$ 2.074,67 | R$ 4.345,56 | 70,10 |
| Guadalajara | 87 | R$ 537,73 | R$ 896,83 | 7868,53 |
| Morumbi | 58 | R$ 438,02 | R$ 629,09 | 6,80 |
| Vila Olímpia | 226 | R$ 402,00 | R$ 602,77 | 4,33 |
| Alvinópolis | 155 | R$ 370,67 | R$ 514,30 | 0,83 |
| Higienópolis | 32 | R$ 364,80 | R$ 480,19 | 2,38 |
| Paraíso | 65 | R$ 340,67 | R$ 554,88 | 1,63 |
| Jardim Paulista | 450 | R$ 340,00 | R$ 632,13 | 0,96 |
| Pinheiros | 464 | R$ 324,00 | R$ 456,45 | 2,77 |
| Itaim Bibi | 522 | R$ 312,27 | R$ 481,39 | 6,40 |
| Butantã | 74 | R$ 301,10 | R$ 655,32 | 5,02 |

Principais grupos acima de 40 km do centro robusto, usados para decidir segmentação e remoção apenas acima de 150 km:

| airbnb_neighborhood | n | median_distance_km | median_price |
| --- | --- | --- | --- |
| Riviera De Sao Lourenco | 285 | 70,10 | R$ 2.074,67 |
| Bertioga | 118 | 71,58 | R$ 2.438,58 |
| Guadalajara | 87 | 7868,53 | R$ 537,73 |
| Campinas | 83 | 84,69 | R$ 255,80 |
| Americana | 48 | 114,61 | R$ 252,12 |
| Zapopan | 29 | 7872,99 | R$ 833,53 |
| Centro | 24 | 114,61 | R$ 142,32 |
| Lapa | 15 | 80,57 | R$ 168,60 |
| São Lourenço | 15 | 71,80 | R$ 3.539,47 |
| Ourinhos | 14 | 334,12 | R$ 210,27 |
| Hortolândia | 12 | 93,21 | R$ 269,93 |
| Cidade Universitária | 11 | 92,58 | R$ 235,10 |

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

| model | MAE | RMSE | MAPE | MdAPE | R2_log | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| robust_segmented_hgb_log_mae | R$ 353,96 | R$ 2.362,68 | 23,87% | 16,61% | 0,52 | R$ -282,65 |
| huber_log | R$ 364,26 | R$ 2.334,61 | 27,18% | 20,34% | 0,48 | R$ -282,37 |
| random_forest_log | R$ 366,24 | R$ 2.345,48 | 28,74% | 20,55% | 0,54 | R$ -222,65 |
| ridge_log | R$ 370,03 | R$ 2.289,62 | 31,72% | 24,63% | 0,50 | R$ -234,93 |
| extra_trees_log | R$ 375,94 | R$ 2.322,06 | 31,12% | 21,16% | 0,52 | R$ -217,25 |
| baseline_neighborhood | R$ 400,34 | R$ 2.419,06 | 32,94% | 25,38% | 0,28 | R$ -310,24 |

Modelo final: **robust_segmented_hgb_log_mae**.

Justificativa:

- Melhor MAE no holdout: R$ 353,96.
- MdAPE de 16,61%, indicando erro percentual mediano aceitável para um mercado ruidoso.
- Usa perda absoluta em log-preço, boa combinação de robustez e flexibilidade.
- Supera o baseline por bairro em MAE, MAPE e R2_log.

## 8. Consistência por segmentos

Por faixa de preço:

| price_band | n | actual_median | pred_median | MAE | MAPE | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| low | 531 | R$ 191,33 | R$ 226,67 | R$ 46,13 | 26,30% | R$ 45,01 |
| mid_low | 531 | R$ 243,67 | R$ 245,69 | R$ 30,78 | 12,73% | R$ 12,82 |
| mid_high | 531 | R$ 323,40 | R$ 291,29 | R$ 64,02 | 19,03% | R$ -17,30 |
| high | 531 | R$ 671,33 | R$ 447,50 | R$ 1.274,92 | 37,40% | R$ -1.171,12 |

Por segmento de mercado:

| market_segment | n | actual_median | pred_median | MAE | MAPE | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| beach_premium | 84 | R$ 2.322,75 | R$ 2.133,11 | R$ 3.811,44 | 38,64% | R$ -3.314,74 |
| extended_market | 50 | R$ 266,07 | R$ 232,58 | R$ 298,90 | 36,19% | R$ -249,52 |
| urban_core | 1990 | R$ 268,80 | R$ 261,98 | R$ 209,40 | 22,93% | R$ -155,49 |

Por número de quartos:

| bedrooms | n | actual_median | MAE | MAPE | Bias |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 1510.0 | R$ 249,73 | R$ 144,50 | 20,96% | R$ -102,83 |
| 2.0 | 455.0 | R$ 389,07 | R$ 313,88 | 28,12% | R$ -230,22 |
| 3.0 | 99.0 | R$ 1.118,00 | R$ 988,39 | 37,34% | R$ -820,83 |
| 4.0 | 53.0 | R$ 2.729,20 | R$ 4.447,39 | 43,40% | R$ -3.846,92 |
| 5.0 | 6.0 | R$ 4.416,77 | R$ 9.536,77 | 42,32% | R$ -9.192,06 |
| 6.0 | 1.0 | R$ 832,23 | R$ 17,89 | 2,15% | R$ -17,89 |

Principais bairros do holdout:

| airbnb_neighborhood | n | actual_median | pred_median | MAE | MAPE | Bias |
| --- | --- | --- | --- | --- | --- | --- |
| São Paulo | 682 | R$ 273,17 | R$ 261,29 | R$ 254,72 | 23,67% | R$ -200,82 |
| Itaim Bibi | 101 | R$ 309,87 | R$ 318,36 | R$ 146,82 | 20,72% | R$ -76,36 |
| Vila Mariana | 101 | R$ 236,67 | R$ 236,42 | R$ 164,05 | 18,50% | R$ -127,26 |
| Pinheiros | 91 | R$ 347,00 | R$ 318,11 | R$ 247,30 | 22,34% | R$ -206,35 |
| Jardim Paulista | 85 | R$ 333,83 | R$ 313,95 | R$ 149,95 | 28,03% | R$ -51,73 |
| República | 71 | R$ 217,50 | R$ 227,07 | R$ 56,97 | 19,82% | R$ -9,88 |
| Bela Vista | 70 | R$ 270,07 | R$ 254,17 | R$ 489,88 | 25,35% | R$ -431,04 |
| Santo Amaro | 59 | R$ 316,00 | R$ 259,53 | R$ 508,65 | 20,18% | R$ -474,15 |
| Barra Funda | 56 | R$ 276,83 | R$ 255,10 | R$ 210,65 | 24,21% | R$ -172,53 |
| Riviera De Sao Lourenco | 54 | R$ 2.048,00 | R$ 2.065,07 | R$ 3.539,71 | 36,30% | R$ -2.999,35 |
| Consolação | 48 | R$ 309,82 | R$ 273,00 | R$ 166,15 | 22,00% | R$ -124,72 |
| Brás | 47 | R$ 257,00 | R$ 290,40 | R$ 71,77 | 20,44% | R$ -21,83 |

Leitura:

- O modelo é mais estável no miolo da distribuição.
- A faixa alta e `beach_premium` ainda têm maior erro e bias negativo, mas agora entram no treino e são identificados explicitamente.
- Imóveis de 3+ quartos têm menos exemplos e maior erro; esse segmento deve ter monitoramento separado.

## 9. Interpretação dos drivers

Top features por importância de permutação:

| feature | importance_mae | importance_std |
| --- | --- | --- |
| bathrooms_model | R$ 63,53 | R$ 2,14 |
| lon | R$ 27,00 | R$ 0,94 |
| lat | R$ 25,64 | R$ 1,04 |
| bedrooms_model | R$ 23,27 | R$ 1,26 |
| has_beach | R$ 17,01 | R$ 0,63 |
| distance_to_market_center_km | R$ 8,80 | R$ 0,41 |
| guests_model | R$ 7,60 | R$ 0,73 |
| is_superhost_int | R$ 1,47 | R$ 0,92 |
| has_metro | R$ 1,34 | R$ 0,63 |
| title_word_count | R$ 1,30 | R$ 0,13 |
| airbnb_neighborhood | R$ 0,87 | R$ 0,31 |
| description_len | R$ 0,79 | R$ 0,19 |

Os principais drivers são localização (`lon`, `lat`, distância e bairro), banheiros, quartos e capacidade. Isso é coerente com o objetivo estrutural: preço base é principalmente função de localização e configuração do imóvel.

Impactos contrafactuais estimados no holdout:

- +1 quarto, mantendo demais variáveis constantes: mediana de R$ 14,52, ou 4,97%.
- +1 banheiro, mantendo demais variáveis constantes: mediana de R$ 119,55, ou 42,49%.

Mudança de localização para um apartamento representativo de 1 quarto e 1 banheiro, excluindo o label genérico `São Paulo`:

Maiores preços previstos:

| airbnb_neighborhood | train_n | observed_median | predicted_representative_price |
| --- | --- | --- | --- |
| Morumbi | 43 | R$ 418,83 | R$ 350,10 |
| Vila Olímpia | 181 | R$ 402,33 | R$ 336,67 |
| Pinheiros | 332 | R$ 313,20 | R$ 314,29 |
| Alvinópolis | 109 | R$ 359,00 | R$ 307,02 |
| Butantã | 51 | R$ 301,87 | R$ 300,64 |
| Jardim Paulista | 327 | R$ 336,00 | R$ 297,07 |
| Tatuapé | 127 | R$ 276,20 | R$ 289,33 |
| Lapa | 48 | R$ 239,63 | R$ 280,71 |

Menores preços previstos:

| airbnb_neighborhood | train_n | observed_median | predicted_representative_price |
| --- | --- | --- | --- |
| Água Rasa | 55 | R$ 139,47 | R$ 138,10 |
| Brás | 163 | R$ 273,50 | R$ 196,31 |
| Tucuruvi | 50 | R$ 201,40 | R$ 201,85 |
| Ipiranga | 78 | R$ 218,30 | R$ 209,41 |
| Vila Anita | 42 | R$ 248,20 | R$ 209,45 |
| Santa Cecília | 87 | R$ 269,80 | R$ 209,56 |
| Sé | 53 | R$ 195,67 | R$ 211,80 |
| Bertioga | 84 | R$ 2.706,77 | R$ 212,38 |

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
.\.venv\Scripts\python scripts\test_model.py
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
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Para recriar o notebook:

```powershell
.\.venv\Scripts\python scripts\create_case_notebook.py
```

O notebook já foi executado nesta entrega; para reexecutar, abra `notebooks/01_base_price_modeling_tabas.ipynb` no Jupyter/VS Code usando o kernel do `.venv`.
