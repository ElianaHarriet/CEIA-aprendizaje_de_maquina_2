# Modelo base — Predicción de Ratings en MovieLens 25M

Este directorio contiene el trabajo práctico final desarrollado en la materia **Aprendizaje de Máquina I (CEIA - FIUBA)**, que sirve como modelo base para la implementación MLOps del presente repositorio.

## Descripción del modelo

El modelo aborda una tarea de **clasificación binaria**: predecir si un usuario calificará una película con 4 o más estrellas (`rating >= 4.0 -> 1`) utilizando el dataset [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/).

Las features utilizadas incluyen información de contenido de la película, perfil de comportamiento del usuario y señales del tag genome.

## Autoría

Este trabajo fue desarrollado por:

- **Jose Miguel Silva Pavón**
- **Pablo Santiago Rodríguez Castro**
- **Damian Nicolas Smilovich**

**Pablo Santiago Rodríguez Castro** y **Damian Nicolas Smilovich** forman parte del grupo de trabajo del presente proyecto de MLOps.

## Contenido

| Archivo | Descripción |
|---|---|
| `movielens_project.ipynb` | Notebook completo con EDA, ingeniería de features, entrenamiento y evaluación de modelos |

## Referencia

Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 1–19.
