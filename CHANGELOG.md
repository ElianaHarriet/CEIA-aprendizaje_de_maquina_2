# Changelog

Todos los cambios importantes de este proyecto serán documentados en este archivo.

## [Unreleased] - 2026-05-31

### Documentation

* Reorganización y mejora del `README.md`.
* Creación de `GUIDELINES.md` para documentar lineamientos del proyecto.
* Incorporación de los archivos `CHANGELOG.md` y `TODO.md`.

---

## [0.4.0] - 2026-05-25

### Changed

* Refactorización de la configuración de Airflow para parametrizar rutas y eliminar hardcodeos.
* Mejora de la portabilidad y mantenibilidad de los DAGs.

### Added

* Variables y conexiones de Airflow en formato JSON.
* Automatización de la importación de variables y conexiones.

### Removed

* Eliminación del backend legacy.
* Eliminación de archivos YAML legacy para variables y conexiones.

---

## [0.3.0] - 2026-05-18

### Added

* DAG para entrenamiento del modelo MovieLens (`train_movielens.py`).
* Primer DAG de entrenamiento en estado preliminar (WIP).
* Primer boceto del DAG de ETL.

### Changed

* Actualización de dependencias necesarias para construir la imagen de Airflow.
* Mejoras en la documentación del proyecto.

---

## [0.2.0] - 2026-05-10

### Added

* Implementación inicial del pipeline de Machine Learning para clasificación de ratings de MovieLens 25M.
* Scripts para:

  * ETL.
  * Ingeniería de características.
  * Entrenamiento.
  * Predicción.
* Carpeta `modelo/` con notebook base del trabajo práctico.
* Documentación de autoría e integrantes.
* Inclusión del repositorio académico como submódulo.

### Changed

* Configuración inicial del proyecto utilizando `uv`.
* Configuración de calidad de código mediante `ruff`.
* Actualización del `.gitignore`.
* Incorporación de estructura de directorios `data/`.