SELECT
  table_catalog,
  table_schema,
  table_name,
  column_name,
  ordinal_position,
  data_type,
  is_nullable
FROM dev.information_schema.columns
WHERE table_catalog = 'dev'
  AND table_schema = 'gold'
  AND table_name IN (
    'dimentsoeseries',
    'factentsoetimeserieslatest',
    'factentsoetimeseriesvintages'
  )
ORDER BY table_name, ordinal_position
LIMIT 1024
