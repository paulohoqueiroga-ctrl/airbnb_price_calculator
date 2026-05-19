# Portfolio Base Price Modeling

Projeto de portfólio de ciência de dados. O objetivo é estimar o preço base estrutural por diária de imóveis usando dados de mercado do Airbnb, sem modelar dinâmica de temporada, eventos, feriados, ocupação ou last-minute.

## Artefatos

- Notebook executado: `notebooks/01_base_price_modeling.ipynb`
- Documentação do projeto: `docs/project_documentation.md`
- Modelo gerado pelo notebook: `models/base_price_model.joblib`
- Métricas, tabelas e predições: `outputs/`
- Scripts auxiliares de documentação/teste: `scripts/`

## Como Rodar

Crie o ambiente e instale as dependências:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Teste o modelo principal:

```bash
.venv/bin/python scripts/test_model.py
```

Regenere a documentação:

```bash
.venv/bin/python scripts/create_project_documentation.py
```

## Arquitetura de Modelagem

O modelo principal resolve preço base estrutural, não preço dinâmico. As variáveis temporais (`check_in`, `check_out`, `los`, `reference_date`) são usadas para construir e auditar o alvo, mas não entram como preditores finais.

O notebook gera o artefato final do projeto (`base_price_model.joblib`), um baseline robusto em log-preço com avaliação por segmentos e interpretação por importâncias e cenários contrafactuais.

Como melhoria proposta, uma evolução de produção poderia adicionar:

- segmentos de imóvel (`urban_compact`, `urban_family`, `urban_large`, `beach_premium`, `extended_market`);
- features de capacidade e interações estruturais;
- modelos p50, p75 e p90;
- classificador de cauda/premium;
- camada de comparáveis locais por bairro, estrutura e segmento.

Essa evolução fica como proposta futura, não como requisito para reproduzir a versão principal.

## Observações de Produção

- O pipeline espera feature engineering antes da predição. Para imóveis brutos, use as funções em `scripts/test_model.py`.
- Valores estruturais têm caps robustos para evitar extrapolação fora da amostra, por exemplo `guests > 16` vira `guests_model = 16`.
- A cauda premium ainda tem bias negativo e deve ir para revisão humana ou calibração adicional quando preço, bairro, estrutura ou comparáveis indicarem risco.
- Para múltiplas cidades, o centro geográfico, filtros de distância e segmentos devem ser parametrizados por mercado.

## Validação

Comandos usados para validação local:

```bash
.venv/bin/python -m py_compile scripts/*.py
.venv/bin/python scripts/test_model.py
```

O notebook está executado, sem erros salvos, e a documentação resume métricas, limitações, interpretação e próximos passos.
