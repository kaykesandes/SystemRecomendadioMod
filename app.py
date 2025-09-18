import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data(show_spinner=False)
def load_data():
    books = pd.read_csv("./data/books.csv")
    book_tags = pd.read_csv("./data/book_tags.csv")
    tags = pd.read_csv("./data/tags.csv")
    ratings_path = "./data/ratings.csv"
    ratings = None
    try:
        ratings = pd.read_csv(ratings_path)
    except Exception:
        pass
    return books, book_tags, tags, ratings


def build_content_corpus(books: pd.DataFrame, book_tags: pd.DataFrame, tags: pd.DataFrame) -> pd.Series:
    """Cria um corpus textual por livro usando título, autores e top tags.

    Suporta duas variantes comuns do Goodbooks-10k:
    - books.csv contém coluna "goodreads_book_id" (padrão original)
    - books.csv não tem "goodreads_book_id"; nesse caso usamos "book_id" como o id do Goodreads
    """
    # Detecta coluna que representa o id do Goodreads no arquivo de livros
    if "goodreads_book_id" in books.columns:
        gr_col = "goodreads_book_id"
    elif "book_id" in books.columns:
        # Em alguns dumps, "book_id" é o id do Goodreads
        gr_col = "book_id"
    elif "best_book_id" in books.columns:
        gr_col = "best_book_id"
    else:
        raise KeyError("Não foi possível identificar a coluna de Goodreads ID em books.csv")

    # Une tags em texto por livro (book_tags usa a coluna 'goodreads_book_id')
    tag_names = tags.set_index("tag_id")["tag_name"]
    bt = book_tags.copy()
    bt["tag_name"] = bt["tag_id"].map(tag_names)
    bt = bt.sort_values(["goodreads_book_id", "count"], ascending=[True, False])
    N = 10  # top N tags por livro
    top_bt = bt.groupby("goodreads_book_id").head(N)
    tag_text = top_bt.groupby("goodreads_book_id")["tag_name"].apply(lambda s: " ".join(map(str, s)))

    # Monta corpus: título + autores + tags
    books = books.copy()
    books["authors"] = books.get("authors", "").fillna("")
    books["title"] = books.get("title", "").fillna("")
    gr_values = books[gr_col].astype(int)
    tag_text = tag_text.reindex(gr_values).fillna("")
    corpus = books["title"].str.lower() + " " + books["authors"].str.lower() + " " + tag_text.values
    return corpus


@st.cache_resource(show_spinner=False)
def build_vectorizer_and_matrix(corpus: pd.Series):
    vectorizer = TfidfVectorizer(max_features=50000, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


def recommend_by_liked_items(books, X, liked_indices, top_k=10):
    if len(liked_indices) == 0:
        return pd.DataFrame()
    # Vetor médio dos itens curtidos
    liked_vectors = X[liked_indices]
    centroid = liked_vectors.mean(axis=0)
    # mean() em sparse retorna np.matrix; convertemos para ndarray (1, n_features)
    centroid = np.asarray(centroid).ravel()[np.newaxis, :]
    sims = cosine_similarity(centroid, X).ravel()
    # remove itens já curtidos
    sims[liked_indices] = -np.inf
    idx = np.argpartition(-sims, range(min(top_k, len(sims))))[:top_k]
    idx = idx[np.argsort(-sims[idx])]
    # Seleciona diretamente colunas do books (evita depender de nomes de IDs)
    select_cols = [c for c in ["title", "authors", "original_publication_year", "average_rating"] if c in books.columns]
    recs = books.iloc[idx][select_cols].copy()
    recs["similarity"] = sims[idx]
    return recs


def main():
    st.set_page_config(page_title="Book Recommender", layout="wide")
    st.title("Sistema de Recomendação de Livros")
    st.caption("Dataset: Goodbooks-10k")

    with st.spinner("Carregando dados..."):
        books, book_tags, tags, ratings = load_data()

    # Construção do corpus e matriz de features
    with st.spinner("Preparando modelo (TF-IDF)..."):
        corpus = build_content_corpus(books, book_tags, tags)
        vectorizer, X = build_vectorizer_and_matrix(corpus)

    # Sidebar: parâmetros
    with st.sidebar:
        st.header("Parâmetros")
        top_k = st.slider("Quantidade de recomendações", 5, 30, 10, 1)
        st.divider()
        st.subheader("Filtros opcionais")
        min_year = int(books["original_publication_year"].dropna().min())
        max_year = int(books["original_publication_year"].dropna().max())
        year_range = st.slider("Ano de publicação (intervalo)", min_year, max_year, (min_year, max_year))
        min_avg_rating = st.slider("Avaliação média mínima (Goodreads)", 0.0, 5.0, 0.0, 0.1)

        st.divider()
        with st.expander("Treinamento (MF - SGD)"):
            mf_k = st.slider("Dimensão do embedding (k)", 8, 128, 32, 8)
            mf_epochs = st.slider("Épocas", 1, 20, 5, 1)
            mf_lr = st.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3, format_func=lambda x: f"{x:g}")
            mf_reg = st.select_slider("Regularização (L2)", options=[0.0, 1e-5, 1e-4, 1e-3, 1e-2], value=1e-4, format_func=lambda x: f"{x:g}")
            sample_size = st.slider("Amostra de interações (ratings)", 10000, 300000, 100000, 10000)
            batch_size = st.select_slider("Batch size", options=[256, 512, 1024, 2048, 4096], value=1024)
            seed = 42
            train_button = st.button("Treinar MF agora")

    # Utilidades de treinamento MF
    def prepare_ratings(ratings_df: pd.DataFrame):
        df = ratings_df[["user_id", "book_id", "rating"]].dropna().copy()
        # Reindexar ids para 0..N-1
        u_codes, u_uniques = pd.factorize(df["user_id"], sort=True)
        i_codes, i_uniques = pd.factorize(df["book_id"], sort=True)
        y = df["rating"].astype(np.float32).values
        return u_codes.astype(np.int32), i_codes.astype(np.int32), y, len(u_uniques), len(i_uniques)

    def train_mf_sgd(u_idx, i_idx, y, n_users, n_items, k=32, epochs=5, lr=1e-3, reg=1e-4, batch_size=1024, seed=42):
        rng = np.random.default_rng(seed)
        U = rng.normal(0, 0.1, size=(n_users, k)).astype(np.float32)
        V = rng.normal(0, 0.1, size=(n_items, k)).astype(np.float32)
        n = len(y)
        losses = []
        idx_all = np.arange(n)
        for ep in range(1, epochs + 1):
            rng.shuffle(idx_all)
            for start in range(0, n, batch_size):
                sl = idx_all[start:start + batch_size]
                uu = u_idx[sl]
                ii = i_idx[sl]
                r = y[sl]
                Uu = U[uu]
                Vi = V[ii]
                pred = np.sum(Uu * Vi, axis=1)
                err = pred - r
                # Gradientes
                gradU = err[:, None] * Vi + reg * Uu
                gradV = err[:, None] * Uu + reg * Vi
                U[uu] -= lr * gradU
                V[ii] -= lr * gradV
            # MSE ao fim da época (amostra para rapidez)
            eval_idx = idx_all[: min(50000, n)]
            mse = float(np.mean((np.sum(U[u_idx[eval_idx]] * V[i_idx[eval_idx]], axis=1) - y[eval_idx]) ** 2))
            losses.append(mse)
        return U, V, losses

    # Exploração de dados
    st.header("Exploração de Dados")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("books.csv (head)")
        st.dataframe(books.head())
        # contagens únicas
        uniq_books = books["book_id"].nunique() if "book_id" in books.columns else None
        if uniq_books is not None:
            st.caption(f"book_id.unique(): {uniq_books}")
    with col2:
        st.subheader("book_tags.csv (head)")
        st.dataframe(book_tags.head())
        uniq_gr = book_tags["goodreads_book_id"].nunique() if "goodreads_book_id" in book_tags.columns else None
        if uniq_gr is not None:
            st.caption(f"goodreads_book_id.unique() em book_tags: {uniq_gr}")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("tags.csv (head)")
        st.dataframe(tags.head())
    with col4:
        st.subheader("ratings.csv (head)")
        if ratings is not None:
            st.dataframe(ratings.head())
            # contagens de usuários e livros únicos nos ratings
            uniq_users = ratings["user_id"].nunique() if "user_id" in ratings.columns else None
            uniq_books_r = ratings["book_id"].nunique() if "book_id" in ratings.columns else None
            if uniq_users is not None:
                st.caption(f"user_id.unique() em ratings: {uniq_users}")
            if uniq_books_r is not None:
                st.caption(f"book_id.unique() em ratings: {uniq_books_r}")
        else:
            st.info("ratings.csv não encontrado (opcional)")

    # Seleção de livros curtidos
    st.subheader("Quais livros você gostou?")
    titles = books["title"].fillna("") + " — " + books["authors"].fillna("")
    selected = st.multiselect(
        "Digite e selecione títulos que você curtiu",
        options=list(titles),
        default=[],
    )

    # Mapear seleção para índices
    title_to_index = {t: i for i, t in enumerate(titles)}
    liked_indices = [title_to_index[s] for s in selected if s in title_to_index]

    if st.button("Recomendar"):
        recs = recommend_by_liked_items(books, X, liked_indices, top_k=top_k)

        # Aplicar filtros opcionais diretamente nas colunas já presentes
        if not recs.empty:
            if "original_publication_year" in recs.columns:
                recs = recs[
                    recs["original_publication_year"].fillna(min_year).between(year_range[0], year_range[1])
                ]
            if "average_rating" in recs.columns:
                recs = recs[recs["average_rating"].fillna(0) >= min_avg_rating]

        st.subheader("Recomendações")
        if recs.empty:
            st.info("Nenhuma recomendação encontrada. Tente selecionar mais livros ou relaxar os filtros.")
        else:
            display_cols = [c for c in ["title", "authors", "similarity", "original_publication_year", "average_rating"] if c in recs.columns]
            st.dataframe(
                recs[display_cols].rename(columns={
                    "title": "Título",
                    "authors": "Autor(es)",
                    "similarity": "Similaridade",
                    "original_publication_year": "Ano",
                    "average_rating": "Nota média",
                }),
                use_container_width=True,
            )

    # Treinamento MF quando solicitado
    if ratings is not None and train_button:
        with st.spinner("Treinando MF (SGD)..."):
            u_idx, i_idx, y, n_users, n_items = prepare_ratings(ratings)
            # Amostragem para acelerar
            n_all = len(y)
            take = min(sample_size, n_all)
            rng = np.random.default_rng(42)
            sel = rng.choice(n_all, size=take, replace=False)
            U, V, losses = train_mf_sgd(
                u_idx[sel], i_idx[sel], y[sel],
                n_users, n_items,
                k=mf_k, epochs=mf_epochs, lr=mf_lr, reg=mf_reg, batch_size=batch_size, seed=42
            )
            st.session_state["book_embeddings"] = V  # shape: (n_items, k)
            st.session_state["mf_losses"] = losses
            st.session_state["mf_schema"] = {
                "n_users": n_users,
                "n_items": n_items,
                "embedding_dim": mf_k,
                "epochs": mf_epochs,
                "lr": mf_lr,
                "reg": mf_reg,
                "batch_size": batch_size,
                "samples_trained": int(take),
            }

    # Seção: Esquema de dados do modelo
    st.header("Esquema de Dados do Modelo")
    if "mf_schema" in st.session_state:
        schema = st.session_state["mf_schema"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Usuários", schema["n_users"])
        c2.metric("Itens (livros)", schema["n_items"])
        c3.metric("Dimensão (k)", schema["embedding_dim"])
        c4.metric("Épocas", schema["epochs"])
        st.json(schema)
    else:
        st.caption("Treine o modelo MF para ver o esquema e métricas.")

    # Métrica de treino por época (se houver)
    if "mf_losses" in st.session_state:
        st.subheader("Curva de Treinamento (MSE por época)")
        fig_l, ax_l = plt.subplots(figsize=(6, 3))
        ax_l.plot(range(1, len(st.session_state["mf_losses"]) + 1), st.session_state["mf_losses"], marker="o")
        ax_l.set_xlabel("Época")
        ax_l.set_ylabel("MSE (amostra)")
        ax_l.grid(True, alpha=0.3)
        st.pyplot(fig_l)

    # Seção: Visualização de embeddings (PCA / t-SNE)
    st.header("Visualização de Embeddings")
    viz_source = st.radio(
        "Fonte dos embeddings para visualizar",
        options=["MF (se treinado)", "TF-IDF"],
        index=0 if "book_embeddings" in st.session_state else 1,
        horizontal=True,
    )

    try:
        max_points = 2000
        if viz_source == "MF (se treinado)" and "book_embeddings" in st.session_state:
            E = st.session_state["book_embeddings"]
            n_items = E.shape[0]
            idx_sample = np.random.choice(n_items, size=min(max_points, n_items), replace=False)
            emb_sample = E[idx_sample]
        else:
            # Fallback TF-IDF (livros x vocabulário)
            n_samples = X.shape[0]
            idx_sample = np.random.choice(n_samples, size=min(max_points, n_samples), replace=False)
            emb_sample = X[idx_sample].toarray()

        st.subheader("PCA (2D)")
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(emb_sample)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], s=10, ax=ax)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        st.pyplot(fig)

        with st.expander("t-SNE (2D) - opcional"):
            perplexity = st.slider("Perplexity", 5, 50, 30, 1)
            lr_tsne = st.select_slider("Learning rate", options=[10, 50, 100, 200, 500, 1000], value=200)
            run_tsne = st.button("Gerar t-SNE")
            if run_tsne:
                # Reduz primeiro com PCA para 50D se necessário
                X_tsne = emb_sample
                if X_tsne.shape[1] > 50:
                    X_tsne = PCA(n_components=50, random_state=42).fit_transform(X_tsne)
                tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr_tsne, init="random", random_state=42)
                tsne_res = tsne.fit_transform(X_tsne)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], s=10, ax=ax2)
                ax2.set_xlabel("t-SNE 1")
                ax2.set_ylabel("t-SNE 2")
                st.pyplot(fig2)
    except Exception as e:
        st.warning(f"Não foi possível gerar visualizações: {e}")


if __name__ == "__main__":
    main()
