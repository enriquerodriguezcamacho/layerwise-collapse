# Layer-Wise Information Compression and Signal Preservation as a Collapse Diagnostic

Proyecto de la asignatura *Métodos Diferenciales* (Máster en IA, ICAI).

## Hipótesis

La pérdida progresiva de preservación de señal en las capas profundas, y no un único
colapso de rango abrupto, constituye un **colapso representacional gradual** que explica
por qué las capas intermedias suelen superar a las finales en tareas downstream.

## Objetivos

1. **Replicar** las métricas de calidad de representación por capa (entropía de prompt,
   entropía de dataset, curvatura, rango efectivo, métricas de invarianza a augmentations)
   en transformers (Pythia, Llama) y state-space models (Mamba).
2. **Contrastar** si las capas que sobre-comprimen coinciden con el inicio del colapso de
   rango y con gradientes evanescentes en las proyecciones query/key.
3. **Mitigar**: explorar early stopping o regularización por capa en el cuello de botella
   de compresión sin sacrificar el rendimiento de la capa final.

## Estructura

```
configs/              Configuraciones YAML de experimentos
src/layercollapse/    Paquete principal
  models.py           Carga de modelos y extracción de hidden states por capa
  metrics.py          Métricas de representación (entropía matricial, rango efectivo, curvatura)
  gradients.py        Normas de gradiente en W_Q / W_K por capa (objetivo ii)
  data.py             Carga de datasets / prompts
scripts/              Puntos de entrada (extracción de métricas, entrenamiento)
  slurm/              Plantillas de jobs para la DGX
notebooks/            Exploración y figuras
results/              Salidas (no se versionan, solo la carpeta)
tests/                Tests unitarios de las métricas
docs/                 Notas, plan y bibliografía
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Uso rápido

```bash
# Métricas por capa sobre un modelo pequeño (funciona en CPU)
python scripts/extract_metrics.py --config configs/default.yaml

# Tests
pytest
```

## Referencia

Skean, O., Arefin, M. R., Zhao, D., Patel, N., Naghiyev, J., LeCun, Y., & Shwartz-Ziv, R. (2025).
*Layer by layer: Uncovering hidden representations in language models.* ICML 2025.
