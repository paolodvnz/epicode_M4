# EPICODE Modulo 4 — Deep Learning

Progetti didattici su reti neurali con TensorFlow/Keras.
Ogni progetto è un Jupyter Notebook autonomo con analisi, visualizzazioni e commenti esplicativi.

---

## Progetti

### 1. Analisi Avanzata della Regolarizzazione

**File:** [`project_1_m4.ipynb`]

Confronto sistematico di quattro architetture su un dataset sintetico ad alta dimensionalità (100 feature, solo 15 informative, rumore gaussiano elevato), per dimostrare come le tecniche di regolarizzazione permettano a una rete neurale di ignorare il rumore e concentrarsi sul segnale reale.

**Pipeline:**
1. Setup e seed globale (riproducibilità completa)
2. Generazione dataset con `make_regression` e analisi delle correlazioni — identificazione empirica delle 15 feature informative
3. Preprocessing — train/val/test split (70/15/15) e `StandardScaler` fittato solo su train (no leakage)
4. **Baseline** — rete densa 256→128→64→1 senza regolarizzazione, overfitting marcato (gap train/val ~160× su MSE)
5. **Modelli regolarizzati** — stessa architettura base con Dropout (rate=0.3), L2 (λ=1e-3) e Batch Normalization, confronto curve di validation
6. **Feature importance** via norma L2 dei pesi del primo layer
7. Tabella riepilogativa sul test set e conclusioni

**Librerie:** `tensorflow/keras`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`

---

### 2. Computer Vision per la "Smart Agriculture"

**File:** [`project_2_m4.ipynb`]

Transfer Learning con ResNet50 per classificare 5 tipologie di colture / stati del terreno (Grano, Mais, Soia, Riso, Terreno vuoto) da immagini aeree RGB 224×224. Dataset standalone con dati simulati (immagini random + label one-hot), senza dipendenze da file esterni.

**Pipeline:**
1. Setup e seed globale — costanti, `COLOR`, `rcParams`, riproducibilità completa
2. Generazione dati simulati — `generate_dummy_data()`: immagini float32 `[0,1]`, label one-hot, distribuzione classi bilanciata
3. **Architettura (Feature Extraction)** — ResNet50 pre-addestrata su ImageNet (`include_top=False`, `trainable=False`) + `GlobalAveragePooling2D` → `Dropout(0.4)` → `Dense(5, softmax)`; `model.summary()` con confronto parametri totali (~23.6M) vs addestrabili (~10K)
4. **Training** — Adam (`lr=1e-4`), `categorical_crossentropy`, callbacks standard (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint), 5 epoche di prova
5. Visualizzazione — curve loss/accuracy train vs. validation con linea baseline casuale (20%)
6. Valutazione — classification report, confusion matrix, griglia campioni corretti (verde) / errati (rosso)
7. **(Bonus) Hyperparameter Tuning** — Keras Tuner `RandomSearch` su `dropout_rate` ∈ {0.2, 0.3, 0.4, 0.5} e `learning_rate` ∈ {1e-3, 1e-4, 1e-5}, 6 trial × 2 epoche; confronto best model vs baseline

**Librerie:** `tensorflow/keras`, `keras-tuner`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`

---

### 3. Model Swapping: da MobileNetV2 a VGG16

**File:** [`project_3_m4.ipynb`]

Sostituzione dell'architettura MobileNetV2 con **VGG16** per la classificazione zero-shot su ImageNet-1K. Il notebook testa il modello su 4 soggetti (elefante africano, auto sportiva, pianoforte a coda, aquila calva) scaricati da Wikimedia Commons, e analizza le Top-5 predizioni evidenziando le differenze tra classi con firma visiva univoca e classi ambigue per ImageNet.

**Pipeline:**
1. Setup — import, costanti, `rcParams`
2. **Caricamento modello** — VGG16 `weights='imagenet'`, `include_top=True`, ~138M parametri; confronto architetturale VGG16 vs MobileNetV2 (parametri, preprocessing, dimensione pesi)
3. **Download e visualizzazione** — `download_image()` con header User-Agent, `preprocess_image()` con `preprocess_input` specifico di VGG16; griglia 2×2 di anteprima prima della classificazione
4. **Classificazione** — `classify_image()` + `plot_results()`: grafico affiancato immagine / barre orizzontali Top-5 (Top-1 in rosso, 2–5 in blu)
5. **Analisi dei risultati reali** — confidenza per soggetto: aquila 99.91%, pianoforte 99.18%, elefante 87.30%, Ferrari 67.91% (distribuzione frammentata con `grille` e `golfcart`); nota critica sul preprocessing VGG16 vs MobileNetV2

**Librerie:** `tensorflow/keras`, `numpy`, `matplotlib`, `Pillow`

---
