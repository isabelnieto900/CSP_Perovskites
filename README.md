# CSP_Perovskites

Predicción del sistema cristalino de perovskitas a partir de propiedades elementales de los elementos de los sitios A y B.

Versión modularizada del código de
[L. Sreekumar, S. Dileep, S. G. Nair y P. P. Nair, *"Machine Learning-Driven Crystal Systems Predictions of Perovskites From Elemental Properties"*, IEEE Access, vol. 14, 2026](https://doi.org/10.1109/ACCESS.2026.3661599)
(repositorio original: [medackan/CSP_Perovskites](https://github.com/medackan/CSP_Perovskites)).

## Estructura

```
CSP_Perovskites/
  data/
    raw/Master_File.csv        # dataset original: 5279 perovskitas, 7 sistemas cristalinos
    processed/                 # datasets generados por notebooks/01_preprocessing.ipynb
      legacy/                  # M4.csv y M7.csv originales (SMOTE aplicado antes del split)
  src/csp_perovskites/         # paquete Python
    config.py                  # rutas, constantes, listas de features
    io.py                      # lectura/escritura de CSV y figuras
    preprocessing.py           # limpieza, decorrelación, codificación multietiqueta
    datasets.py                # splits M7, M4 y binarios (split -> SMOTE solo en train)
    models.py                  # los 12 clasificadores y sus grillas de hiperparámetros
    evaluation.py              # métricas, corridas repetidas, Wilcoxon, multietiqueta
    shap_analysis.py           # importancia SHAP y selección de features
    plotting.py                # figuras del paper
  notebooks/
    01_preprocessing.ipynb     # distribución de clases, decorrelación, construcción de datasets
    02_M4.ipynb                # 12 modelos en M4, Wilcoxon, SHAP, M4*
    03_M7.ipynb                # 12 modelos en M7, Wilcoxon, SHAP, M7*, comparación M4 vs M7
    04_binary_multilabel.ipynb # binary relevance por sistema cristalino y métricas multietiqueta
    legacy/                    # notebooks originales sin modificar
  tests/                       # tests de pytest
  results/                     # salidas (figuras, tablas, SHAP); no se versiona
```

## Instalación

Requisitos: **Python 3.12 o superior** (las versiones fijadas en `requirements.txt` lo necesitan) y `git`.

### Linux, macOS o WSL

```bash
git clone https://github.com/isabelnieto900/CSP_Perovskites.git
cd CSP_Perovskites

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"

python -m ipykernel install --user --name csp-perovskites --display-name "Python (csp-perovskites)"
```

### Windows (PowerShell)

```powershell
git clone https://github.com/isabelnieto900/CSP_Perovskites.git
cd CSP_Perovskites

py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"

python -m ipykernel install --user --name csp-perovskites --display-name "Python (csp-perovskites)"
```

Si PowerShell bloquea la activación, ejecuta antes `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

`requirements.txt` instala exactamente las versiones con las que se probó el proyecto. `pip install -e ".[dev]"` instala el paquete `csp_perovskites` en modo editable (los cambios en `src/` se reflejan sin reinstalar) junto con Jupyter y pytest.

Para comprobar la instalación:

```bash
python -m pytest -q
```

Deben pasar todos los tests (unos 20 segundos).

## Uso

Activa siempre el entorno antes de trabajar (`source .venv/bin/activate` o `.venv\Scripts\Activate.ps1`).

### Notebooks

Abre los notebooks en VS Code, Cursor o Jupyter (`jupyter lab`) y selecciona el kernel **Python (csp-perovskites)**. Ejecútalos en orden:

| Notebook | Qué hace | Duración aproximada |
|---|---|---|
| `01_preprocessing` | Figuras de distribución, decorrelación y datasets en `data/processed/` | < 1 min |
| `02_M4` | 12 modelos en M4, Wilcoxon, SHAP y M4* | horas con los parámetros por defecto |
| `03_M7` | Igual para M7 y comparación M4 vs M7 | horas con los parámetros por defecto |
| `04_binary_multilabel` | Clasificación binaria por sistema cristalino y métricas multietiqueta | 10-20 min |

Los notebooks 02, 03 y 04 leen directamente `data/raw/Master_File.csv`, así que no dependen de haber corrido el 01. La comparación M4 vs M7 del 03 necesita haber corrido antes el 02.

### Parámetros

Cada notebook tiene una celda **Parámetros**. Para una prueba rápida:

```python
N_RUNS = 2                 # corridas con splits distintos (paper: 10)
CV = 3                     # folds de GridSearchCV (paper: 5)
MODELS = ["Naive Bayes", "LightGBM", "Extra Trees", "Random Forest"]  # None = los 12
SHAP_MAX_SAMPLES = 200     # submuestreo del test para SHAP
```

Otros parámetros:

- `SCALER`: `"minmax"` (paper), `"standard"` o `None`.
- `DROP_POLYMORPHS_FROM_TEST` (solo 03): `True` evalúa M7 solo con compuestos de una sola etiqueta.
- `REDUCED_FEATURES` (02 y 03): lista del paper por defecto; `shap_top10` usa la calculada en la corrida.
- `TUNE` (04): `True` aplica GridSearchCV a cada clasificador binario.

### Desde la terminal

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_preprocessing.ipynb

# corridas largas en segundo plano, sin límite de tiempo por celda
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/02_M4.ipynb > logs_M4.txt 2>&1 &
```

### Resultados

- `results/figures/`: figuras PNG (400 dpi).
- `results/tables/`: métricas por corrida (`*_runs.csv`), media y desviación (`*_summary.csv`), mejores hiperparámetros, comparación completo vs reducido, precisión binaria y métricas multietiqueta.
- `results/shap/`: importancias SHAP por modelo y combinadas, y gráficos.

### Uso como librería

```python
from csp_perovskites import io, preprocessing as pp, datasets as ds, models, evaluation as ev

df = pp.clean_master(io.load_master())
df_decor, _ = pp.decorrelate(df)

result = ev.run_repeated(df_decor, ds.make_m4_split,
                         models.get_classifiers(["LightGBM", "Extra Trees"]), n_runs=3)
ev.summarize(result.runs, as_text=True)
```
