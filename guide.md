# Olist Delivery Risk — versione 2 settimane (14 × 3h)

*Guida pratica per imparare git, uv, Python, SQL (DuckDB), Spark, scikit-learn, PyTorch, Keras, MLflow, Docker, S3 e FastAPI costruendo un unico mini-progetto.*

## Come usare questa guida

**Obiettivo:** non produrre un portfolio perfetto, ma capire cosa fa ogni tecnologia, come si avvia e a cosa serve. Quindi: poca teoria, molta pratica guidata, codice minimo.

**Budget:** 14 giorni × 3 ore = 42 ore. Se lavori solo nei giorni feriali, il piano diventa 3 settimane: non comprimere i giorni, spalmali.

**Struttura di ogni giornata (3h):**

- 20 min: leggi il "concetto chiave" e la documentazione indicata.
- 2h: passi guidati.
- 40 min: scrivi 5 righe in `NOTES.md` (cosa ho capito, cosa mi ha bloccato), fai commit e tag.

**Regole:**

1. Se un passo ti blocca per più di 25 minuti, annota il blocco e passa avanti. Il tempo è il vincolo, non la completezza.
2. Prima prova da solo, poi leggi gli **Hint**.
3. Un commit per passo concluso, un tag a fine giornata (`day01`, `day02`…).
4. Chiedi aiuto a Claude incollando l'errore completo e il comando che hai lanciato.

**Cosa ho tagliato rispetto alla guida lunga:** distanza haversine/geolocalizzazione, calibrazione avanzata, tuning con run annidate, test di parità feature, monitoring/PSI, SHAP, Streamlit, Dependabot. AWS reale è solo bonus finale.

## Il progetto in 6 righe

- **Dataset:** Olist Brazilian E-Commerce (Kaggle), \~100k ordini, 9 CSV relazionali.
- **Problema:** al momento dell'acquisto, prevedere se l'ordine arriverà **in ritardo** rispetto alla data stimata.
- **Target:** `is_late = order_delivered_customer_date > order_estimated_delivery_date` (solo ordini consegnati).
- **Decisione di business:** se il rischio è alto, l'operations interviene (costo `C_int` = 2 €); un ritardo non gestito costa `C_late` = 15 € (valori inventati e configurabili).
- **Split temporale:** train = passato, validation = più recente, test = ultimo periodo. Mai split casuale.
- **Output:** modello nel registry MLflow (alias `champion`) servito da un'API FastAPI in Docker.

## Mappa delle 14 giornate

| Giorno | Tema | Tecnologie |
| --- | --- | --- |
| 1 | Setup + business case | git, uv, Python |
| 2 | Container + storage S3 locale | Docker, S3 (SeaweedFS), boto3 |
| 3–4 | Esplorazione dati e tabella per ordine | SQL (DuckDB) |
| 5–6 | ETL e feature | Spark |
| 7–8 | Baseline e soglia di costo | scikit-learn |
| 9–10 | Reti neurali | PyTorch, Keras |
| 11–12 | Tracking e registry | MLflow |
| 13 | API in container | FastAPI, Docker |
| 14 | Stack, CI, README | compose, GitHub Actions |

---

## Giorno 1 — Setup (git + uv + Python) e business case

**Concetto chiave:** `uv` sostituisce `pip + venv + pip-tools`: `pyproject.toml` dichiara le dipendenze, `uv.lock` fissa le versioni esatte (riproducibilità), `.venv/` è l'ambiente. Si usa sempre `uv run ...` per eseguire dentro l'ambiente.

**Passi**

1. Installa uv (istruzioni ufficiali Astral). Poi:

   ```bash
   uv init --package --name olist-delivery-risk
   uv python pin 3.12
   uv add pandas numpy scikit-learn duckdb pyarrow boto3 matplotlib
   uv add --group dev ruff pytest
   ```
2. `git init -b main`, crea `.gitignore` con `.venv/`, `__pycache__/`, `data/`, `mlruns/`, `mlflow.db`, `.env`, `*.pkl`. Crea il repo su GitHub e collegalo con `git remote add origin ...`.
3. Aggiungi in `pyproject.toml` la sezione `[tool.ruff]` con `line-length = 100`. Verifica con `uv run ruff check .` e `uv run ruff format .`.
4. Scrivi `docs/01_business_case.md` (max 1 pagina): chi usa il modello, quale decisione cambia, **prediction time = momento dell'acquisto**, target, metrica di business (costo atteso totale), valori di `C_int` e `C_late`.
5. In `src/olist_delivery_risk/cost.py` scrivi `expected_cost(y_true, y_pred, c_int=2, c_late=15)`: costo = (n. predetti positivi × `c_int`) + (n. falsi negativi × `c_late`). Scrivi `tests/test_cost.py` con 2 casi calcolati a mano. Lancia `uv run pytest`.
6. Lavora su un branch `feat/setup`, apri una Pull Request e fai merge: ti serve per prendere il ritmo del workflow.
7. Scarica il dataset da Kaggle (account gratuito) in `data/raw/`. **Non committarlo.**

**Hint**

- Differenza tra `uv add` (modifica pyproject e lock) e `uv sync` (allinea l'ambiente al lock): prova entrambi e leggi il diff di `uv.lock`.
- Se `expected_cost` ti sembra ambigua: chi è un falso positivo? Un ordine puntuale su cui intervieni inutilmente, costa `c_int`. Un vero positivo costa comunque `c_int` (intervento), ma evita `c_late`.
- Perché accuracy è una cattiva metrica con \~8% di ritardi? Scrivilo in 3 righe nelle note.

**Done:** `uv run pytest` verde, business case scritto, 1 PR mergiata. Tag `day01`.

---

## Giorno 2 — Docker e storage S3 locale

**Concetti chiave**

- *Docker*: **immagine** (ricetta + file), **container** (istanza in esecuzione), **volume** (dati persistenti), **porta** (`host:container`).
- *S3*: object storage con **bucket** e **key** (`cartella/file`, ma le cartelle sono solo prefissi). Un data lake tipico ha zone `raw` (immutabile) → `processed`.
- *Docker Compose*: descrive più servizi in un unico `docker-compose.yml`, avviati con `docker compose up`.

**Passi**

1. Installa Docker Desktop (o Docker Engine). Verifica con `docker run hello-world`.
2. Crea `docker-compose.yml`:

   ```yaml
   services:
     s3:
       image: chrislusf/seaweedfs
       command: server -s3 -dir=/data -s3.port=8333
       ports: ["8333:8333"]
       volumes: ["s3_data:/data"]
   volumes:
     s3_data:
   ```

   (MinIO non è più una scelta sicura; SeaweedFS è S3-compatibile. Verifica comando e porta nella sua documentazione.)
3. Avvia: `docker compose up -d s3`, poi `docker compose ps` e `docker compose logs s3`.
4. Crea `.env` (e `.env.example` committato) con `S3_ENDPOINT_URL=http://localhost:8333`, `AWS_ACCESS_KEY_ID=local`, `AWS_SECRET_ACCESS_KEY=local`, `AWS_DEFAULT_REGION=eu-west-1`.
5. In `src/olist_delivery_risk/storage.py` crea `get_s3_client()` con boto3: `endpoint_url` da `S3_ENDPOINT_URL` (se vuoto → AWS reale) e `Config(s3={"addressing_style": "path"})` quando l'endpoint è custom. Aggiungi `upload_file` e `list_keys(bucket, prefix)`.
6. `scripts/bootstrap_buckets.py`: crea i bucket `olist-raw` e `olist-processed` **in modo idempotente** (se esistono già, nessun errore).
7. Carica i 9 CSV in `s3://olist-raw/olist/ingest_date=AAAA-MM-GG/` e verifica con `list_keys`.

**Hint**

- Per caricare le variabili da `.env`: `uv add python-dotenv` e `load_dotenv()`, oppure `uv run --env-file .env ...`.
- Se ricevi `connection refused`, il container non è ancora pronto: aggiungi un piccolo retry.
- Dopo `docker compose down` i dati restano grazie al volume; con `down -v` li perdi: provalo e osserva.

**Done:** i 9 CSV sono visibili con `list_keys`. Tag `day02`.

---

## Giorno 3 — SQL con DuckDB (esplorazione)

**Concetto chiave:** DuckDB è un database analitico *embedded*: nessun server, interroga direttamente CSV e Parquet. Perfetto per esplorare.

**Setup**

```python
import duckdb

duckdb.sql("""
  SELECT order_status, COUNT(*) AS n
  FROM read_csv_auto('data/raw/olist_orders_dataset.csv')
  GROUP BY 1 ORDER BY n DESC
""").show()
```

Salva le query in `sql/01_exploration.sql` (commentate) e lanciale da Python.

**Esercizi guidati** (in ordine)

1. Ordini per `order_status`. Quanti sono `delivered`?
2. Ordini per mese (`date_trunc('month', order_purchase_timestamp)`). In quali mesi i dati sono "densi"? Servirà per lo split.
3. **Late rate globale** sui soli consegnati: `AVG((delivered > estimated)::INT)`.
4. Late rate per **stato del cliente** (join `orders` con `customers`), con `HAVING COUNT(*) >= 100`.
5. Giorni di consegna: mediana, p90, p99 con `quantile_cont(...)`. Confrontali con i giorni stimati.
6. **Impatto business:** review score medio per ordini in ritardo vs puntuali (join con `order_reviews`). Usa il risultato per giustificare `C_late` nel business case.

**Hint**

- Differenza di date in giorni: `date_diff('day', a, b)`. Le colonne data vanno convertite con `CAST(... AS TIMESTAMP)`.
- Una `JOIN` è corretta solo se sai quante righe aspettarti: prima di ogni join, conta le righe.
- Per capire gli errori SQL, riduci la query al solo pezzo che fallisce con `LIMIT 5`.

**Done:** sai dire late rate globale e i 2–3 driver principali (stato, mese…). Tag `day03`.

---

## Giorno 4 — SQL avanzato: una riga per ordine

**Concetti chiave:** CTE (`WITH`), **window function** (`OVER`), **fan-out dei join**: unire `orders` a `order_items` moltiplica le righe (un ordine ha più item) e falsa ogni aggregazione.

**Esercizi guidati**

1. Dimostra il fan-out: confronta `COUNT(*)` di `orders` con quello di `orders JOIN order_items`.
2. Con una CTE aggrega `order_items` **per ordine**: `n_items`, `SUM(price)`, `SUM(freight_value)`, `SUM(product_weight_g)` (join con `products`), `COUNT(DISTINCT seller_id)`.
3. Unisci la CTE a `orders` e `customers`. Verifica `COUNT(*) = COUNT(DISTINCT order_id)`.
4. Scegli una regola per gli ordini con più seller (es. il seller dell'item più costoso) e annotala nelle note.
5. **Window function:** late rate mensile e media mobile a 3 mesi: `AVG(x) OVER (ORDER BY mese ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)`. C'è un picco (Black Friday 2017)?
6. **Data quality:** trova (a) NULL nelle date di consegna per ordini `delivered`, (b) consegna prima dell'acquisto, (c) prezzi o pesi pari a 0. Decidi cosa scartare.
7. Esporta la tabella finale: `COPY (SELECT ...) TO 'data/processed/orders_duckdb.parquet' (FORMAT PARQUET)`.

**Hint**

- `ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY price DESC)` ti aiuta a scegliere l'item principale: filtra poi `rn = 1`.
- Un grafico di late rate mensile con matplotlib (da `.df()` di DuckDB) vale più di mille numeri: fanne uno e salvalo in `docs/img/`.

**Done:** query `sql/02_order_table.sql` che produce 1 riga per ordine; Parquet generato. Tag `day04`.

---

## Giorno 5 — Spark (parte 1): capire come funziona

**Concetti chiave**

- Spark è un motore distribuito; qui gira in locale (`local[*]`). Con 100k righe è **sovradimensionato**: lo usi per imparare, ed è onesto dirlo nel README.
- **Lazy evaluation:** `select`, `filter`, `join` sono *trasformazioni* (costruiscono un piano); `count`, `show`, `write` sono *azioni* (eseguono).
- **Shuffle:** spostamento dati tra partizioni (join, groupBy): è la parte costosa.

**Setup consigliato: Docker (niente Java sul tuo PC)** `docker/spark.Dockerfile`:

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group spark --no-install-project
ENV PATH="/app/.venv/bin:$PATH"
```

Prima: `uv add --group spark pyspark`. Poi:

```bash
docker build -f docker/spark.Dockerfile -t olist-spark .
docker run --rm -v $PWD:/app -p 4040:4040 olist-spark python src/olist_delivery_risk/etl/explore.py
```

**Esercizi guidati** (`etl/explore.py`)

1. Crea la `SparkSession` (`SparkSession.builder.master("local[*]").appName("olist").getOrCreate()`).
2. Leggi `orders` con **schema esplicito** (`StructType`) e non con `inferSchema`. Perché è meglio? Scrivi 2 motivi nelle note.
3. Converti le date in `timestamp` (`F.to_timestamp`), filtra `delivered`, calcola `is_late` e la late rate: deve coincidere con quella di DuckDB (**verifica incrociata**).
4. Fai un `groupBy("order_status").count()` e lancia `.explain()`: leggi il piano, trova lo shuffle (`Exchange`).
5. Metti `time.sleep(60)` a fine script, apri `http://localhost:4040` e guarda i job/stage nella Spark UI.

**Hint**

- Se lo script dice `file not found`: i percorsi sono quelli *dentro* il container (`/app/data/raw/...`).
- `df.show(5, truncate=False)` e `df.printSchema()` sono i tuoi occhi.
- Pensa: cosa NON succede finché non chiami un'azione? Provalo togliendo il `.show()`.

**Done:** late rate Spark = late rate DuckDB. Tag `day05`.

---

## Giorno 6 — Spark (parte 2): ETL, feature, split temporale

**Obiettivo:** script `etl/build_features.py` che da CSV produce Parquet pronto per il ML.

**Tabella finale (una riga per ordine consegnato)**

- `order_id`, `purchase_ts`, `is_late` (target)
- numeriche: `total_price`, `total_freight`, `n_items`, `total_weight_g`, `estimated_days` (stimata − acquisto), `freight_ratio` (freight/prezzo)
- temporali: `dow`, `hour`, `month`
- categoriche: `customer_state`, `main_category`
- storica: `seller_late_rate_hist`
- `split`: `train` / `val` / `test`

**Esercizi guidati**

1. Replica con DataFrame API la logica SQL del giorno 4 (aggregazione item per ordine). Aggiungi un `assert` sull'unicità di `order_id`.
2. Aggiungi le feature numeriche e temporali con funzioni **built-in** (`F.dayofweek`, `F.hour`, `F.datediff`).
3. **Feature storica del seller, in modo corretto (il pezzo più istruttivo):** per ogni ordine, la late rate del seller calcolata **solo sugli ordini già consegnati prima dell'acquisto corrente**. Self-join con condizione `prev.delivered_ts < curr.purchase_ts`, poi `groupBy` e media. Con smoothing: `(late + k*p0)/(n + k)`, con `p0` = late rate globale e `k` ≈ 20.
4. **Split temporale:** guardando la distribuzione mensile del giorno 4, scegli due date di taglio (es. train fino a metà marzo 2018, val fino a fine maggio, test il resto). Controlla che le dimensioni siano ragionevoli.
5. Scrivi Parquet **partizionato per `split`** in `data/processed/features/` e caricalo su `s3://olist-processed/features/v1/` con `storage.py`.
6. Scrivi `docs/feature_dictionary.md`: per ogni feature, "disponibile al prediction time? sì/no".

**Hint**

- Perché l'ordine A precedente non basta: è precedente per *acquisto*, ma potrebbe essere ancora in viaggio quando compri l'ordine B. Se lo usi, stai leggendo il futuro (**leakage**).
- Self-join pesante? `F.broadcast()` sulla tabella piccola e `explain()` per cercare `BroadcastHashJoin`.
- Se la parte storica ti blocca oltre i 40 minuti, usa la versione "naive" (tutti gli ordini passati) e annota che è leakage: è una scelta lecita per imparare.

**Done:** Parquet su S3 e locale, con le tre split non vuote. Tag `day06`.

---

## Giorno 7 — scikit-learn (parte 1): baseline con Pipeline

**Concetti chiave**

- `Pipeline` + `ColumnTransformer`: preprocessing e modello in **un solo oggetto**, salvato e riusato identico in produzione (niente *train/serve skew*).
- Con classi sbilanciate **non** si guarda l'accuracy: ROC-AUC, **PR-AUC** (average precision), Brier score, costo atteso.

**Esercizi guidati** (`training/sklearn_baseline.py` o notebook)

1. Carica il Parquet con pandas e separa `train`/`val`/`test` dalla colonna `split`. Separa `X` e `y`; **non** includere `order_id`, `purchase_ts`, `split` nelle feature.
2. Costruisci il `ColumnTransformer`: numeriche → `SimpleImputer(strategy="median")`; categoriche → `OneHotEncoder(handle_unknown="ignore")`.
3. Addestra 3 modelli sulla train, valutali sulla validation: `DummyClassifier(strategy="prior")`, `LogisticRegression(class_weight="balanced", max_iter=1000)`, `HistGradientBoostingClassifier()`.
4. Crea una tabella con ROC-AUC, PR-AUC, Brier. Quanto vale la PR-AUC "a caso"? (Uguale alla frazione di positivi.)
5. Con `permutation_importance` sulla validation guarda le 5 feature più importanti. Hanno senso di business? Una feature con importanza esagerata è un sospetto di leakage.
6. Confronta HGB con e senza `seller_late_rate_hist`: cosa cambia?

**Hint**

- Il preprocessing si fa `fit` solo sulla train: nella pipeline ci pensa lei.
- `HistGradientBoostingClassifier` gestisce NaN, ma le stringhe vanno codificate: oppure usa `categorical_features="from_dtype"` con colonne `category`.
- `average_precision_score` è la PR-AUC; `brier_score_loss` richiede probabilità, non classi.

**Done:** tabella dei 3 modelli sulla validation. Tag `day07`.

---

## Giorno 8 — scikit-learn (parte 2): soglia basata sul costo

**Concetto chiave:** il modello restituisce una probabilità; la **decisione** (intervenire sì/no) dipende da una soglia. La soglia giusta non è 0.5: è quella che minimizza il costo atteso con i tuoi `C_int` e `C_late`.

**Esercizi guidati**

1. Calcola `p_val = pipe.predict_proba(X_val)[:, 1]`.
2. Per soglie da 0.01 a 0.99, calcola `expected_cost` (giorno 1) sulla **validation**. Grafico costo vs soglia; scegli il minimo.
3. Applica la soglia scelta al **test, una sola volta**. Calcola costo e risparmio rispetto a due baseline: *mai intervenire* (costo = `n_late × C_late`) e *intervieni su tutti*.
4. **Sensibilità:** ripeti con `C_late/C_int` = 3, 7.5, 20. Come si sposta la soglia? (Grafico.)
5. **Calibrazione:** `CalibrationDisplay.from_predictions(y_val, p_val)` e prova `CalibratedClassifierCV(method="isotonic")`: migliora il Brier?
6. Salva la pipeline con `joblib.dump` e scrivi un test: `predict_proba` su un DataFrame di 1 riga funziona.

**Hint**

- Soglia scelta sul test = errore classico: il test serve per una sola misura finale.
- Se il risparmio è negativo o minimo, non è un fallimento: è un risultato. Chiediti se i costi assunti rendono l'intervento conveniente.
- Scrivi nel README 3 limiti di questa stima (costi inventati, efficacia dell'intervento ignota, drift nel tempo).

**Done:** soglia ottima, risparmio sul test, pipeline salvata. Tag `day08`.

---

## Giorno 9 — PyTorch: MLP con embedding e training loop

**Concetti chiave:** tensori, `Dataset`/`DataLoader`, `nn.Module`, `nn.Embedding` (vettore appreso per ogni categoria), loop di training manuale: forward → loss → `backward()` → `optimizer.step()`.

**Setup (CPU-only, immagini più leggere)** in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
```

Poi `uv add torch`.

**Esercizi guidati** (`training/torch_mlp.py`)

1. Prepara: numeriche scalate e imputate (riusa il preprocessing del giorno 7 solo per le numeriche); `customer_state` e `main_category` come interi (`0` = sconosciuto).
2. `Dataset` che restituisce `(x_num, x_cat, y)` e due `DataLoader` (train con `shuffle=True`, batch 512).
3. `nn.Module`: embedding per ciascuna categorica (dimensione \~ `min(50, n//2)`), concatenazione con le numeriche, MLP `Linear(…,64)→ReLU→Dropout→Linear(64,32)→ReLU→Linear(32,1)`.
4. Loop a mano: `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)`, `AdamW`, 20 epoche, valutazione per epoca su validation (PR-AUC), early stopping con salvataggio del miglior `state_dict`.
5. Fissa i seed (`torch.manual_seed`) e verifica la riproducibilità tra due run.
6. Calcola ROC-AUC, PR-AUC, Brier e costo (con la soglia ottima sulla validation) per confrontarli col baseline.

**Hint**

- Ricorda `model.train()` nel training e `model.eval()` + `torch.no_grad()` in validazione.
- Il modello emette *logit*: per la probabilità serve `torch.sigmoid`.
- Se la loss non scende: abbassa il learning rate (1e-3 → 3e-4) e controlla che non ci siano NaN nelle feature.
- Aspettativa realistica: il boosting spesso vince sui dati tabellari. Documentare il perché è un buon risultato.

**Done:** rete addestrata con metriche confrontabili. Tag `day09`.

---

## Giorno 10 — Keras e confronto finale

**Concetto chiave:** Keras 3 è un'API di alto livello che può usare PyTorch come backend: **non serve installare TensorFlow**.

**Setup:** `uv add keras`. Imposta la variabile **prima** dell'import:

```python
import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
```

**Esercizi guidati** (`training/keras_mlp.py`)

1. Riusa la matrice numerica del preprocessing sklearn (one-hot incluso: in Keras per semplicità niente embedding).
2. Modello: `Dense(64, relu) → Dropout(0.2) → Dense(32, relu) → Dense(1, sigmoid)`; `compile(optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.AUC(curve="PR", name="auc_pr")])`.
3. `fit` con `class_weight={0:1, 1:n_neg/n_pos}` e `EarlyStopping(monitor="val_auc_pr", mode="max", restore_best_weights=True)`. Plotta le learning curve.
4. **Confronto finale (45 min):** tabella con LogReg, HGB, PyTorch, Keras: ROC-AUC, PR-AUC, Brier, costo atteso, tempo di training. Scrivi 8 righe: *perché ha vinto chi ha vinto?* Confronta anche quante righe di codice ti sono servite in Keras vs PyTorch.

**Hint**

- Il `fit` di Keras vuole `validation_data=(X_val, y_val)`.
- Differenza di filosofia da annotare: Keras nasconde il loop, PyTorch ti obbliga a scriverlo (più controllo, più codice).
- Salva il confronto in `docs/results.md`: servirà nel README e nel registry.

**Done:** tabella con 4 modelli. Tag `day10`.

---

## Giorno 11 — MLflow: tracking degli esperimenti

**Concetti chiave:** *experiment* → *run* → **params** (iperparametri), **metrics** (risultati, anche per step/epoca), **artifacts** (file: grafici, modelli), **tags** (metadati). Scopo: ogni risultato è riproducibile e confrontabile.

**Setup server in Docker** `docker/mlflow.Dockerfile`:

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir mlflow boto3
```

Servizio nel compose:

```yaml
  mlflow:
    build: {context: ., dockerfile: docker/mlflow.Dockerfile}
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://s3:8333
      AWS_ACCESS_KEY_ID: local
      AWS_SECRET_ACCESS_KEY: local
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --artifacts-destination s3://mlflow
      --serve-artifacts
    volumes: ["mlflow_data:/mlflow"]
    depends_on: [s3]
    ports: ["5000:5000"]
```

Aggiungi il volume `mlflow_data` e il bucket `mlflow` al bootstrap. Poi `uv add mlflow`, `MLFLOW_TRACKING_URI=http://localhost:5000` nel `.env`, apri `http://localhost:5000`.

**Esercizi guidati**

1. Esperimento `olist-late-delivery`. Scrivi una piccola funzione `log_run(model_family, params, metrics, ...)` che usa `mlflow.start_run`, `log_params`, `log_metrics`, `set_tags` (con `git_sha` e `model_family`).
2. Logga HGB e LogReg con `mlflow.sklearn.log_model(pipe, name="model", signature=..., input_example=...)` (`infer_signature` da `mlflow.models`).
3. Logga la rete PyTorch con `mlflow.pytorch.log_model` e **le metriche per epoca** con `log_metric(..., step=epoch)`.
4. Logga Keras con `log_metric` per epoca e il modello come artifact (`mlflow.keras.log_model` oppure salva il file `.keras` con `log_artifact`).
5. Salva come artifact il grafico costo-soglia e la calibration curve.
6. Nella UI seleziona le 4 run e usa *Compare*. Salva uno screenshot in `docs/img/`.

**Hint**

- Il `git_sha`: `subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()`.
- Se il log del modello va in errore con S3: controlla che `--serve-artifacts` sia attivo e che il bucket `mlflow` esista.
- Se vedi errori di host non consentito dall'interno di un container, consulta le opzioni di sicurezza della tua versione di MLflow.
- Dove sono finiti fisicamente i file? Guarda il bucket `mlflow` con `list_keys`: è il collegamento tra S3 e MLflow.

**Done:** 4 run visibili e confrontabili in UI. Tag `day11`.

---

## Giorno 12 — MLflow Model Registry e promozione

**Concetti chiave:** il *registry* dà un nome e versioni ai modelli. Un **alias** (es. `champion`) è un puntatore mobile a una versione: l'API chiederà `models:/olist-late-delivery@champion` e non saprà mai quale versione sia.

**Esercizi guidati**

1. Registra il miglior modello (per costo atteso o PR-AUC sulla validation): `mlflow.register_model(f"runs:/{run_id}/model", "olist-late-delivery")`.
2. Salva la **soglia ottima** come tag della model version (`MlflowClient().set_model_version_tag(...)`), così l'API non la ricalcola.
3. Assegna l'alias: `client.set_registered_model_alias("olist-late-delivery", "champion", version)`.
4. Verifica il caricamento: `mlflow.pyfunc.load_model("models:/olist-late-delivery@champion").predict(X_val.head())`.
5. **Gate di promozione** `registry/promote.py`: trova la miglior run, confronta con la metrica del champion attuale (`get_model_version_by_alias`), promuove **solo se migliora di un margine** (es. 1% di costo); altrimenti `sys.exit(1)` con spiegazione.
6. **Valutazione finale (una volta):** sul champion calcola le metriche sul **test** e loggale in una run `release`.

**Hint**

- Per leggere la metrica del champion: `client.get_run(mv.run_id).data.metrics`.
- Per testare `promote.py` senza server, passa il client come parametro e usa un finto client nei test (un semplice oggetto con gli stessi metodi).
- Nota il vantaggio: puoi cambiare modello in produzione cambiando un alias, senza ricostruire l'API.

**Done:** `load_model("models:/...@champion")` funziona; `promote.py` testato. Tag `day12`.

---

## Giorno 13 — FastAPI in Docker

**Concetti chiave:** FastAPI + **Pydantic** (validazione automatica input/output), **lifespan** (il modello si carica una sola volta all'avvio), `/health` (il processo vive) vs `/ready` (il modello è pronto).

**Semplificazione:** l'API riceve direttamente le feature dell'ordine già calcolate (come nella tabella ETL). In produzione dovresti ricalcolarle online, ma qui l'obiettivo è imparare il serving. Scrivilo nei limiti del README.

**Setup:** `uv add --group serving fastapi "uvicorn[standard]"`; `uv add --group dev httpx`.

**Esercizi guidati** (`src/olist_delivery_risk/api/main.py`)

1. `OrderFeatures` (Pydantic) con le colonne del modello: tipi, `Field(ge=0)` su prezzo/freight, stati ammessi, ecc. `PredictionOutput`: `late_probability`, `at_risk`, `threshold`, `model_version`.
2. Nel `lifespan` carica `mlflow.pyfunc.load_model(os.getenv("MODEL_URI", "models:/olist-late-delivery@champion"))` e leggi la soglia dal tag della versione. Se il registry non risponde: `ready = False`, niente crash.
3. Endpoint `GET /health`, `GET /ready` (503 se non pronto), `POST /predict`, `GET /model-info`.
4. Avvia in locale: `uv run uvicorn olist_delivery_risk.api.main:app --reload` e prova `http://localhost:8000/docs` (Swagger automatico).
5. Test con `TestClient`: caso valido, input invalido (422), modello non pronto (503). Usa un modello finto (una pipeline sklearn piccola creata nella fixture).
6. Dockerfile multi-stage `docker/api.Dockerfile`: stage `builder` con uv (`uv sync --frozen --no-dev --group serving --no-install-project`), stage finale `python:3.12-slim` con utente non-root, `CMD ["uvicorn", ..., "--host", "0.0.0.0"]`. Build e run con `docker run -p 8000:8000 --env-file .env ...`.

**Hint**

- Dentro un container `localhost` non è l'host: per raggiungere MLflow nel compose usa `http://mlflow:5000`.
- Se champion è sklearn, l'immagine ha bisogno di `mlflow`, `scikit-learn`, `pandas`: aggiungili al gruppo `serving`. Se fosse la rete PyTorch servirebbe anche `torch` (CPU) e l'immagine sarebbe molto più grande: misura con `docker images`.
- Un 422 non è un bug: è Pydantic che fa il suo lavoro.

**Done:** `/predict` risponde in locale e in container; test API verdi. Tag `day13`.

---

## Giorno 14 — Stack completo, CI e README

**Parte 1 — Compose completo (1h)**

1. Aggiungi al compose il servizio `api` con `MLFLOW_TRACKING_URI=http://mlflow:5000` e `depends_on` con `condition: service_started` (o healthcheck sul mlflow).
2. Crea un `Makefile` con target `up`, `etl`, `train`, `promote`, `test`, `lint`, `demo` (sequenza completa).
3. `scripts/smoke_test.sh`: `curl` a `/health`, `/ready` e a una `/predict` di esempio; esce con codice ≠ 0 se qualcosa non va.
4. **Prova della verità:** clona il repo in una cartella nuova e segui solo il README. Quanti passi manuali servono? Riducili.

**Parte 2 — CI con GitHub Actions (45 min)** `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push: {branches: [main]}
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6   # verifica l'ultima major
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest -m "not spark and not integration"
```

Aggiungi il badge al README. Facoltativo: job `docker build` dell'immagine API.

**Parte 3 — README (1h)** Sezioni minime: problema e valore (con il risparmio atteso sul test e le assunzioni), diagramma architetturale, tabella dei 4 modelli, **cosa ho imparato** (leakage trovato, Spark vs DuckDB, boosting vs rete), quickstart in 5 comandi, limiti e prossimi passi.

**Hint**

- Se `pytest` in CI fallisce per marker sconosciuti, dichiarali in `[tool.pytest.ini_options] markers = [...]`.
- Il README è la parte più letta: investici l'ultima ora anche se non hai finito tutto.

**Done:** badge verde, `make demo` funzionante (o con i limiti documentati). Tag `v1.0`.

---

## Bonus (solo se avanzano tempo e energie)

- **AWS S3 reale:** apri un account solo a progetto finito, attiva MFA sul root, imposta un budget alert da 1 $, crea un utente IAM con policy limitata al tuo bucket, lascia `S3_ENDPOINT_URL` **vuoto** e rilancia upload ed ETL: se `storage.py` è fatto bene, non cambia una riga di codice. Poi cancella tutto. Verifica prima le condizioni attuali del free tier (oggi a crediti, non più "12 mesi gratis").
- **Spark su S3 con `s3a://`:** richiede `hadoop-aws` con versione allineata all'Hadoop della tua versione di Spark: è una trappola classica.
- **Benchmark pandas vs Spark** replicando il dataset ×50.
- **ONNX:** esporta la rete PyTorch e confronta le predizioni (tolleranza 1e-5).

---

## Se sei in ritardo: cosa tagliare (in quest'ordine)

1. Keras (tieni solo PyTorch), poi il confronto sarà a 3 modelli.
2. Feature storica del seller "corretta" (usa la naive e dichiara il limite).
3. Calibrazione e analisi di sensibilità.
4. CI: lascia solo lint + test.

**Non tagliare mai:** split temporale, soglia basata sui costi, confronto di PR-AUC, test, README con limiti.

## Errori comuni da evitare

- Split casuale su dati temporali (metriche gonfiate).
- Join senza controllare il fan-out.
- Scegliere la soglia sul test.
- Preprocessing fuori dalla pipeline (train/serve skew).
- Committare `data/`, `.env` o credenziali.
- Usare `latest` come tag Docker nelle versioni finali.
- Perdere ore su un errore: timebox di 25 minuti, annota e vai avanti.
