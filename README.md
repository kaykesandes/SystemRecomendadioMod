# SystemRecomendation

## App de recomendação (Goodbooks-10k)

Este projeto cria um sistema de recomendação de livros baseado em conteúdo usando o dataset Goodbooks-10k. Interface feita com Streamlit.

### Estrutura de dados esperada
Coloque os CSVs na pasta `data/`:
- `books.csv`
- `book_tags.csv`
- `tags.csv`
- `ratings.csv` (opcional)

### Como rodar
1. Crie/ative o ambiente virtual (opcional) e instale dependências:
```
pip install -r requirements.txt
```
2. Inicie o app:
```
streamlit run app.py
```

### Observações
- O modelo usa TF-IDF em título, autor e principais tags de cada livro.
- A recomendação é gerada a partir dos livros que você marcou como “gostei”, calculando similaridade do cosseno.
# SystemRecomendadioMod
