# Plan de trabajo

## Fase 1 — Replicación (sin entrenamiento, solo inferencia)
- [ ] Métricas por capa en Pythia-70M/160M/410M y Mamba-130M/370M (en local)
- [ ] Escalar a Pythia-1.4B/2.8B y Llama 3 8B (DGX)
- [ ] Evolución durante el entrenamiento usando los checkpoints intermedios de Pythia
- [ ] Rendimiento downstream por capa (probing lineal / tareas tipo MTEB)

## Fase 2 — Diagnóstico (forward + backward, sin actualizar pesos)
- [ ] Normas de gradiente en W_Q / W_K por capa
- [ ] Localizar la capa de máxima compresión y comparar con el inicio del colapso de rango

## Fase 3 — Mitigación (requiere entrenar)
- [ ] Baseline: fine-tuning / preentrenamiento continuado de un modelo pequeño
- [ ] Regularizador de entropía por capa en el cuello de botella
- [ ] Early stopping guiado por la métrica de compresión
- [ ] Comparar rendimiento de la capa final frente al baseline

## Notas
