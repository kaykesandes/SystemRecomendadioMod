# Sistema de Recomendação de Livros (Goodbooks‑10k)

Aplicação web em Streamlit para recomendar livros a partir dos títulos que você gostou. O projeto traz:
- Recomendação baseada em conteúdo (TF‑IDF de título, autores e tags) com similaridade do cosseno
- Interface para escolher livros que você curtiu e receber recomendações
- Exploração rápida dos dados (head dos CSVs e contagens)
- Treinamento opcional de um modelo colaborativo simples (Matrix Factorization via SGD) com acompanhamento de épocas e curva de erro
- Visualizações 2D dos embeddings (PCA e t‑SNE)

## Requisitos
- Python 3.10+
- Dependências no `requirements.txt`:
  - `streamlit`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `seaborn`, `matplotlib`

Instalação:
```
pip install -r requirements.txt
```

## Dados esperados
Coloque os CSVs do Goodbooks‑10k na pasta `data/`:
- `books.csv`
- `book_tags.csv`
- `tags.csv`
- `ratings.csv` (opcional, necessário para treinar o modelo colaborativo)

Observações sobre colunas:
- Este app é tolerante a variações comuns de esquema. Em `books.csv`, ele detecta automaticamente a coluna de ID entre `goodreads_book_id`, `book_id` ou `best_book_id` para alinhar com `book_tags.csv` (que usa `goodreads_book_id`).
- Em `ratings.csv`, espera‑se `user_id`, `book_id`, `rating`.

## Como executar
1) Inicie a aplicação Streamlit:
```
streamlit run app.py
```
2) Abra o link exibido no terminal (ex.: `http://localhost:8501`).

## Como usar
1) Exploração de Dados
	- A seção “Exploração de Dados” mostra `head()` de `books`, `book_tags`, `tags` e (se presente) `ratings`.
	- Exibe também `user_id.unique()` e `book_id.unique()` a partir de `ratings`.

2) Parâmetros e Filtros
	- Ajuste a quantidade de recomendações (Top‑K).
	- Filtre por intervalo de ano de publicação e nota média mínima.

3) Seleção de Livros Gostados
	- No campo de busca, selecione os livros que você já leu e gostou.
	- Clique em “Recomendar” para ver os resultados ordenados por similaridade.

4) Treinamento (opcional) – Matrix Factorization (MF)
	- Na sidebar, expanda “Treinamento (MF ‑ SGD)”.
	- Defina: dimensão do embedding (k), épocas, learning rate, regularização, amostragem de interações e batch size.
	- Clique em “Treinar MF agora”. O app:
	  - Reindexa `user_id` e `book_id`
	  - Treina via SGD minimizando MSE
	  - Mostra o esquema do modelo (número de usuários/itens, k, épocas, LR, etc.)
	  - Plota a curva de MSE por época
	- Após treinar, os embeddings de itens ficam disponíveis para visualização (PCA/t‑SNE) e podem substituir o TF‑IDF nas visualizações.

5) Visualizações do Modelo
	- Escolha a fonte dos embeddings: “MF (se treinado)” ou “TF‑IDF”.
	- PCA (2D): scatter plot com amostra de até ~2000 pontos para performance.
	- t‑SNE (2D) opcional: dentro do expansor, com controles de perplexity e learning rate (usa PCA para 50D antes do t‑SNE quando necessário).

## Como funciona a recomendação (Content‑Based)
1. Construção do corpus por livro: `titulo + autores + top tags` (top‑N por frequência em `book_tags`/`tags`).
2. Vetorização TF‑IDF (stopwords em inglês, até 50k features).
3. Dado o conjunto de livros curtidos, calcula‑se o centróide TF‑IDF e a similaridade do cosseno com todos os livros.
4. Ordenação por similaridade e remoção dos livros já curtidos.

## “Esquema de dados do modelo” (quando MF estiver treinado)
Após o treinamento, a seção exibe:
- `n_users`, `n_items`, `embedding_dim (k)`, `epochs`, `lr`, `reg`, `batch_size`, `samples_trained`
- Gráfico da perda MSE por época

## Estrutura do projeto (simplificada)
```
.
├── app.py                  # App Streamlit
├── requirements.txt        # Dependências
├── README.md               # Este guia
└── data/                   # Coloque aqui os CSVs
	 ├── books.csv
	 ├── book_tags.csv
	 ├── tags.csv
	 └── ratings.csv (opcional)
```

## Dicas de performance
- O TF‑IDF e as visualizações podem ser pesados: o app amostra até ~2000 pontos nas projeções 2D.
- Para treinos rápidos no MF, use amostras de `ratings` menores e poucas épocas; aumente conforme necessário.

## Solução de problemas
- Erro de permissão salvando dados em `/data`: garanta que os CSVs estejam em `./data/` (pasta local do projeto).
- Coluna `goodreads_book_id` ausente em `books.csv`: o app detecta automaticamente `book_id`/`best_book_id` como fallback.
- Falta de pacotes (`seaborn`/`matplotlib`): rode `pip install -r requirements.txt`.

## Próximos passos (opcionais)
- Exibir capas (`image_url`) nos resultados.
- Persistir embeddings treinados em disco para reutilização sem re‑treinar.
- Adicionar avaliação offline (RMSE hold‑out, métricas de ranking como HR@K, NDCG@K).

---

Projeto para estudos e demonstração de sistemas de recomendação com Goodbooks‑10k.
