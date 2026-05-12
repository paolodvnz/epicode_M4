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
