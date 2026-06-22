# Raw LAS collection

Исходные материалы хранятся без изменения значений и единиц. Для каждой скважины
используйте каталог `<basin>/<well_id>/` с подпапками `las`, `reports`, `tops`, `core`,
`cuttings`, `stratigraphy`, `interpreted` и `metadata`. Обязательны LAS и
`metadata/well_metadata.yaml`; остальные материалы добавляются при наличии.

Проверка структуры:

```powershell
python -m synthetic_well_logs validate-dataset-structure --root data/raw_las
```

Каталог `basin_001/WELL_001` — минимальный валидный пример. Ограничения лицензии
фиксируются в metadata; добавление файла в репозиторий не означает разрешение на его
распространение.
