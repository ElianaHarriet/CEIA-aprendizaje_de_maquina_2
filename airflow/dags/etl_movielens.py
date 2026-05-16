"""
etl_movielens.py — DAG de ETL para el dataset MovieLens 25M.

Descarga el dataset desde GroupLens y lo transforma en splits train/test
listos para entrenamiento. Cada task es una etapa independiente que lee
y escribe en MinIO (S3), lo que permite:
  - Reintentar una etapa fallida sin reejecutar las anteriores.
  - Paralelizar etapas independientes en el futuro.
  - Inspeccionar los artefactos intermedios en MinIO.

Flujo de datos:
    download_data()
        └─ s3://data/raw/{ratings,movies,genome-scores}.csv
                ↓
    sample_and_save_ratings()
        └─ s3://data/interim/ratings_sampled.csv
                ↓ (estas dos pueden correr en paralelo en el futuro)
    compute_movie_features()          compute_genome_pca()
        └─ s3://data/interim/             └─ s3://data/interim/
           movie_features.csv                genome_pca.csv
                ↓                                 ↓
    compute_user_features()
        └─ s3://data/interim/user_features.csv
                ↓
    merge_and_split()
        └─ s3://data/final/{X_train,X_test,y_train,y_test}.npy
           s3://data/final/feature_names.txt
"""

import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Pipeline — MovieLens 25M

Descarga el dataset MovieLens 25M desde GroupLens y construye 50 features
organizadas en etapas independientes. Cada etapa persiste su output en MinIO,
permitiendo reintentos granulares y futura paralelización.

**Etapas:**
1. `download_data` — Descarga ml-25m.zip y sube CSVs crudos a `s3://data/raw/`
2. `sample_and_save_ratings` — Samplea usuarios/ratings y guarda en `s3://data/interim/`
3. `compute_movie_features` — Stats por película + géneros + año → `s3://data/interim/`
4. `compute_genome_pca` — PCA de 1200 genome tags → 20 componentes → `s3://data/interim/`
5. `compute_user_features` — Stats por usuario → `s3://data/interim/`
6. `merge_and_split` — Merge, interacciones, escala y split → `s3://data/final/`

**Output final:** X_train.npy, X_test.npy, y_train.npy, y_test.npy, feature_names.txt
"""

default_args = {
    "owner": "CEIA - FIUBA",
    "depends_on_past": False,
    "schedule_interval": None,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=90),
}


@dag(
    dag_id="etl_movielens",
    description="ETL para MovieLens 25M: descarga, feature engineering y split train/test.",
    doc_md=markdown_text,
    tags=["ETL", "MovieLens"],
    default_args=default_args,
    catchup=False,
)
def etl_movielens():

    @task.virtualenv(
        task_id="download_data",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def download_data():
        """
        Descarga el dataset MovieLens 25M y sube los CSVs necesarios a MinIO.

        Pasos:
            1. Descarga ml-25m.zip (~250 MB) desde files.grouplens.org.
            2. Extrae el contenido en /tmp/movielens/. GroupLens extrae en
               una subcarpeta ml-25m/ — la aplanamos un nivel arriba.
            3. Sube ratings.csv, movies.csv y genome-scores.csv a s3://data/raw/
               usando awswrangler, que usa las variables AWS_* del docker-compose
               para conectarse a MinIO en vez de AWS S3 real.
            4. Limpia /tmp para liberar espacio en el worker.

        Por qué subimos a S3 en vez de dejar en disco local:
            Con CeleryExecutor las tasks pueden correr en workers distintos.
            S3 es el único storage compartido entre todos los workers.
        """
        import shutil
        import urllib.request
        import zipfile
        from pathlib import Path

        import awswrangler as wr
        import pandas as pd

        DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
        TMP_DIR = Path("/tmp/movielens")
        ZIP_PATH = Path("/tmp/ml-25m.zip")
        REQUIRED_FILES = ["ratings.csv", "movies.csv", "genome-scores.csv"]

        # 1. Descarga
        print(f"Descargando dataset desde {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Descarga completada.")

        # 2. Extracción y aplanado de subcarpeta
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(TMP_DIR)
        extracted_subdir = TMP_DIR / "ml-25m"
        if extracted_subdir.exists():
            for f in extracted_subdir.iterdir():
                shutil.move(str(f), str(TMP_DIR / f.name))
            extracted_subdir.rmdir()

        # 3. Subir CSVs a MinIO
        for filename in REQUIRED_FILES:
            s3_path = f"s3://data/raw/{filename}"
            print(f"Subiendo {filename} → {s3_path} ...")
            df = pd.read_csv(TMP_DIR / filename)
            wr.s3.to_csv(df=df, path=s3_path, index=False)
            print(f"  {len(df):,} filas subidas.")

        # 4. Limpieza
        shutil.rmtree(TMP_DIR)
        ZIP_PATH.unlink(missing_ok=True)
        print("Descarga y subida a MinIO completadas.")

    @task.virtualenv(
        task_id="sample_and_save_ratings",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def sample_and_save_ratings():
        """
        Samplea un subconjunto de usuarios y ratings y lo persiste en MinIO.
            ratings.csv es el CSV más grande (~650 MB). Samplearlo una vez y
            guardarlo en S3 evita que las tasks siguientes lo descarguen completo
            y repitan el mismo sampleo — cada una trabajaría sobre el mismo
            subconjunto reproducible.

        Estrategia de sampleo:
            1. Selecciona N_USERS usuarios al azar sin reemplazo.
            2. Filtra ratings para quedarse solo con esos usuarios.
            3. Si el total aún supera N_RATINGS, samplea filas aleatoriamente.
            Esto mantiene la distribución de usuarios intacta mientras acota
            el tamaño total del dataset.

        Output: s3://data/interim/ratings_sampled.csv
        """
        import awswrangler as wr
        import numpy as np

        RANDOM_STATE = 42
        N_USERS = 20_000
        N_RATINGS = 1_000_000

        print("Leyendo ratings.csv desde s3://data/raw/ ...")
        ratings = wr.s3.read_csv("s3://data/raw/ratings.csv")
        print(f"  Total ratings crudos: {len(ratings):,}")

        rng = np.random.default_rng(RANDOM_STATE)
        all_users = ratings["userId"].unique()
        sampled_users = rng.choice(
            all_users, size=min(N_USERS, len(all_users)), replace=False
        )
        ratings = ratings[ratings["userId"].isin(sampled_users)]

        if len(ratings) > N_RATINGS:
            ratings = ratings.sample(n=N_RATINGS, random_state=RANDOM_STATE)

        ratings = ratings.reset_index(drop=True)
        print(f"  Ratings tras sampleo: {len(ratings):,} de {len(all_users):,} usuarios únicos")

        wr.s3.to_csv(df=ratings, path="s3://data/interim/ratings_sampled.csv", index=False)
        print("ratings_sampled.csv guardado en s3://data/interim/")

    @task.virtualenv(
        task_id="compute_movie_features",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def compute_movie_features():
        """
        Calcula features estadísticas por película y las persiste en MinIO.

        Lee ratings sampleados y movies.csv para construir:
            movie_avg_rating      : promedio de rating de la película.
            movie_rating_count_log: log(1 + n_ratings). El log aplana la distribución
                                    sesgada — blockbusters con millones de ratings vs
                                    películas de nicho con decenas.
            movie_rating_std      : desvío estándar. Alta std = película polarizante.
            year                  : año extraído del título con regex.
                                    Películas sin año reciben la mediana del dataset.
            [20 columnas de género]: binarias (1 = la película tiene ese género).

        Las estadísticas de película deben reflejar el subconjunto de usuarios
        que el modelo va a ver durante el entrenamiento. Usar el CSV completo
        introduciría información de usuarios que no están en el train set.

        Output: s3://data/interim/movie_features.csv
        """
        import awswrangler as wr
        import numpy as np

        GENRES = [
            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
            "Western", "(no genres listed)",
        ]

        print("Leyendo datos desde MinIO ...")
        ratings = wr.s3.read_csv("s3://data/interim/ratings_sampled.csv")
        movies = wr.s3.read_csv("s3://data/raw/movies.csv")

        # Stats por película calculadas sobre el subconjunto sampleado
        stats = (
            ratings.groupby("movieId")["rating"]
            .agg(
                movie_avg_rating="mean",
                movie_rating_count="count",
                movie_rating_std="std",
            )
            .fillna(0)
        )
        stats["movie_rating_count_log"] = np.log1p(stats["movie_rating_count"])

        # Año extraído del título: "Toy Story (1995)" → 1995
        movies = movies.copy()
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype(float)
        movies["year"] = movies["year"].fillna(movies["year"].median())

        # Columna binaria por género
        for genre in GENRES:
            movies[genre] = movies["genres"].str.contains(genre, regex=False).astype(int)

        movie_feats = stats[
            ["movie_avg_rating", "movie_rating_count_log", "movie_rating_std"]
        ].join(movies.set_index("movieId")[["year"] + GENRES], how="left")

        wr.s3.to_csv(df=movie_feats.reset_index(), path="s3://data/interim/movie_features.csv", index=False)
        print(f"movie_features.csv guardado: {len(movie_feats):,} películas, {movie_feats.shape[1]} columnas.")

    @task.virtualenv(
        task_id="compute_genome_pca",
        requirements=["awswrangler==3.6.0", "scikit-learn==1.3.2"],
        system_site_packages=True,
    )
    def compute_genome_pca():
        """
        Reduce la matriz de genome tags a 20 componentes via PCA.

        genome-scores.csv tiene ~1200 tags por película con relevancia entre 0 y 1.
        La matriz resultante es densa: ~13.000 películas × 1.200 tags.

        PCA comprime esa información en N_GENOME_COMPONENTS componentes ortogonales
        que capturan la mayor parte de la varianza (en la práctica >80%). Esto:
            - Reduce la dimensionalidad de 1200 → 20 features.
            - Elimina redundancia entre tags correlacionados.
            - Preserva la información semántica sobre el "estilo" de cada película.

        Esta task puede correr en paralelo con compute_movie_features:
            Solo necesita genome-scores.csv (del raw), no los ratings sampleados.
            En una versión futura del DAG se puede hacer:
                [compute_movie_features(), compute_genome_pca()] >> compute_user_features()

        Output: s3://data/interim/genome_pca.csv
        """
        import awswrangler as wr
        from sklearn.decomposition import PCA

        RANDOM_STATE = 42
        N_GENOME_COMPONENTS = 20

        print("Leyendo genome-scores.csv desde MinIO ...")
        genome_scores = wr.s3.read_csv("s3://data/raw/genome-scores.csv")

        # Pivot: filas = películas, columnas = tagIds, valores = relevancia
        pivot = (
            genome_scores
            .pivot(index="movieId", columns="tagId", values="relevance")
            .fillna(0)
        )
        print(f"Matriz genome: {pivot.shape[0]:,} películas × {pivot.shape[1]:,} tags")

        pca = PCA(n_components=N_GENOME_COMPONENTS, random_state=RANDOM_STATE)
        components = pca.fit_transform(pivot.values)
        explained = pca.explained_variance_ratio_.sum()
        print(f"Varianza explicada acumulada: {explained:.1%}")

        cols = [f"genome_pca_{i}" for i in range(N_GENOME_COMPONENTS)]
        import pandas as pd
        genome_pca_df = pd.DataFrame(components, index=pivot.index, columns=cols)

        wr.s3.to_csv(
            df=genome_pca_df.reset_index(),
            path="s3://data/interim/genome_pca.csv",
            index=False,
        )
        print(f"genome_pca.csv guardado: {len(genome_pca_df):,} películas, {N_GENOME_COMPONENTS} componentes.")

    @task.virtualenv(
        task_id="compute_user_features",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True,
    )
    def compute_user_features():
        """
        Calcula features estadísticas por usuario y las persiste en MinIO.

        Lee los ratings sampleados (no el CSV completo) para calcular:
            user_avg_rating         : promedio de ratings del usuario.
            user_avg_rating_centered: promedio del usuario menos el promedio global.
                                      Captura si el usuario es "generoso" (positivo)
                                      o "exigente" (negativo) respecto al promedio global.
            user_rating_count_log   : log(1 + n_ratings). Usuarios con más historial
                                      tienen preferencias más confiables y estables.
            user_rating_std         : desvío estándar. Alta std = criterio heterogéneo;
                                      baja std = el usuario siempre da notas similares.

        Output: s3://data/interim/user_features.csv
        """
        import awswrangler as wr
        import numpy as np

        print("Leyendo ratings sampleados desde MinIO ...")
        ratings = wr.s3.read_csv("s3://data/interim/ratings_sampled.csv")

        global_mean = ratings["rating"].mean()
        print(f"  Promedio global de ratings: {global_mean:.3f}")

        stats = (
            ratings.groupby("userId")["rating"]
            .agg(
                user_avg_rating="mean",
                user_rating_count="count",
                user_rating_std="std",
            )
            .fillna(0)
        )
        stats["user_rating_count_log"] = np.log1p(stats["user_rating_count"])
        stats["user_avg_rating_centered"] = stats["user_avg_rating"] - global_mean

        user_feats = stats[[
            "user_avg_rating",
            "user_avg_rating_centered",
            "user_rating_count_log",
            "user_rating_std",
        ]]

        wr.s3.to_csv(
            df=user_feats.reset_index(),
            path="s3://data/interim/user_features.csv",
            index=False,
        )
        print(f"user_features.csv guardado: {len(user_feats):,} usuarios.")

    @task.virtualenv(
        task_id="merge_and_split",
        requirements=[
            "awswrangler==3.6.0",
            "scikit-learn==1.3.2",
            "numpy>=1.24",
        ],
        system_site_packages=True,
    )
    def merge_and_split():
        """
        Combina todos los artefactos intermedios, calcula features de interacción,
        escala y genera el split train/test final.

        Pasos:
            1. Lee ratings_sampled, movie_features, genome_pca y user_features desde S3.
            2. Merge: cada fila = un rating, con features de su película y su usuario.
            3. Calcula dos features de interacción:
               - user_deviation_from_movie_avg: diferencia entre el promedio del usuario
                 y el promedio de la película. Captura si el usuario sobrevalúa o subvalúa.
               - genre_cosine_similarity: similitud coseno entre el vector de géneros de
                 la película (binario) y la preferencia de géneros del usuario (continuo).
                 Se implementa como multiplicación elemento a elemento + suma por fila
                 (equivalente al dot product sin usar @, que haría el producto cruzado
                 de todas las combinaciones posibles en vez de solo los pares correctos).
            4. Define el target binario: 1 si rating >= 4.0, 0 si no.
            5. Escala con StandardScaler ajustado SOLO en train (evita data leakage).
            6. Sube X_train, X_test, y_train, y_test como .npy y feature_names.txt a S3.

        .npy en vez de .csv para los splits:
            Los arrays de features ya son numéricos y densos. .npy es más rápido de
            leer/escribir que CSV y no tiene ambigüedad de tipos al deserializar.
            El DAG de entrenamiento los carga con np.load() directamente.

        Output: s3://data/final/{X_train,X_test,y_train,y_test}.npy + feature_names.txt
        """
        import io

        import awswrangler as wr
        import boto3
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        RANDOM_STATE = 42
        TEST_SIZE = 0.2
        N_GENOME_COMPONENTS = 20
        GENRES = [
            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
            "Western", "(no genres listed)",
        ]

        def save_numpy_to_s3(arr, bucket, key):
            """
            Sube un array numpy a MinIO en formato .npy usando BytesIO como buffer.

            numpy.save() necesita un file-like object. Como S3 no es un filesystem,
            usamos io.BytesIO en RAM como intermediario:
                array → np.save() → BytesIO → boto3.put_object() → S3
            """
            buffer = io.BytesIO()
            np.save(buffer, arr)
            buffer.seek(0)
            boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

        # 1. Leer todos los artefactos intermedios desde MinIO
        print("Leyendo artefactos desde s3://data/interim/ ...")
        ratings = wr.s3.read_csv("s3://data/interim/ratings_sampled.csv")
        movie_feats = wr.s3.read_csv("s3://data/interim/movie_features.csv").set_index("movieId")
        genome_pca = wr.s3.read_csv("s3://data/interim/genome_pca.csv").set_index("movieId")
        user_feats = wr.s3.read_csv("s3://data/interim/user_features.csv").set_index("userId")

        # 2. Merge: cada fila combina un rating con features de su película y usuario
        df = ratings[["userId", "movieId", "rating"]].copy()
        df = df.merge(movie_feats, on="movieId", how="left")
        df = df.merge(genome_pca, on="movieId", how="left")
        df = df.merge(user_feats, on="userId", how="left")
        print(f"  DataFrame tras merge: {len(df):,} filas")

        # 3a. Feature de interacción: desviación del usuario respecto al promedio de la película
        df["user_deviation_from_movie_avg"] = df["user_avg_rating"] - df["movie_avg_rating"]

        # 3b. Feature de interacción: similitud coseno entre géneros de película y preferencias del usuario
        # Preferencia de género del usuario = promedio de los géneros de todas sus películas calificadas
        user_genre_pref = df.groupby("userId")[GENRES].mean()
        pref_cols = {g: f"user_pref_{g}" for g in GENRES}
        df = df.merge(user_genre_pref.rename(columns=pref_cols), on="userId", how="left")

        movie_vecs = df[GENRES].values.astype(float)
        user_vecs = df[list(pref_cols.values())].values.astype(float)
        dot = (movie_vecs * user_vecs).sum(axis=1)           # suma de productos elemento a elemento = dot product
        norm_m = np.linalg.norm(movie_vecs, axis=1)
        norm_u = np.linalg.norm(user_vecs, axis=1)
        denom = norm_m * norm_u
        df["genre_cosine_similarity"] = np.where(denom > 0, dot / denom, 0.0)

        # 4. Definir columnas finales y target binario
        # Target: 1 si rating >= 4.0 ("le gustó"), 0 si no.
        # Transforma regresión → clasificación binaria.
        genome_cols = [f"genome_pca_{i}" for i in range(N_GENOME_COMPONENTS)]
        feature_cols = (
            GENRES
            + ["movie_avg_rating", "movie_rating_count_log", "movie_rating_std", "year"]
            + genome_cols
            + ["user_avg_rating", "user_avg_rating_centered", "user_rating_count_log", "user_rating_std"]
            + ["genre_cosine_similarity", "user_deviation_from_movie_avg"]
        )
        df["target"] = (df["rating"] >= 4.0).astype(int)
        df = df.dropna(subset=feature_cols + ["target"])

        X = df[feature_cols].values.astype(float)
        y = df["target"].values
        print(f"Dataset final: {X.shape[0]:,} instancias, {X.shape[1]} features")
        print(f"Balance: positivos={y.mean():.1%}  negativos={(1 - y.mean()):.1%}")

        # 5. Split estratificado y escalado
        # IMPORTANTE: fit() del scaler solo sobre train para evitar data leakage.
        # Si fitteáramos sobre todo el dataset, la media y std del test contaminarían
        # el modelo — estaría "viendo el futuro" durante el entrenamiento.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print(f"Train: {X_train.shape}  Test: {X_test.shape}")

        # 6. Subir splits a MinIO
        print("Subiendo splits a s3://data/final/ ...")
        save_numpy_to_s3(X_train, "data", "final/X_train.npy")
        save_numpy_to_s3(X_test,  "data", "final/X_test.npy")
        save_numpy_to_s3(y_train, "data", "final/y_train.npy")
        save_numpy_to_s3(y_test,  "data", "final/y_test.npy")

        # feature_names.txt permite al DAG de entrenamiento loguear los nombres en MLflow
        boto3.client("s3").put_object(
            Bucket="data",
            Key="final/feature_names.txt",
            Body="\n".join(feature_cols).encode(),
        )
        print("Splits guardados exitosamente en MinIO.")

    # Encadenamiento secuencial — cada task espera a que la anterior termine exitosamente.
    # compute_movie_features y compute_genome_pca podrían correrse en paralelo en el futuro
    # cambiando: [compute_movie_features(), compute_genome_pca()] >> compute_user_features()
    (
        download_data()
        >> sample_and_save_ratings()
        >> compute_movie_features()
        >> compute_genome_pca()
        >> compute_user_features()
        >> merge_and_split()
    )


dag = etl_movielens()
