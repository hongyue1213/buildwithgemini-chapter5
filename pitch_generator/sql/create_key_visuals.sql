-- Copyright 2026 Google LLC
-- BigQuery Object Table DDL for Key Visuals (Step 2a)
-- Wraps Cloud Storage gs:// URIs into ObjectRef using OBJ.MAKE_REF and OBJ.FETCH_METADATA.

CREATE OR REPLACE TABLE bwg.key_visuals AS
SELECT
  campaign,
  concept,
  OBJ.FETCH_METADATA(
    OBJ.MAKE_REF(uri, 'asia-east1.pitch-connection')
  ) AS key_visual
FROM UNNEST([
  STRUCT('cats' AS campaign, 'Flying skateboards for cats' AS concept,
         'gs://qwiklabs-gcp-04-031dce32b81b-bwg/key-visuals/cats.png' AS uri),
  STRUCT('bike', 'A commuter bike built for rainy cities',
         'gs://qwiklabs-gcp-04-031dce32b81b-bwg/key-visuals/bike.png')
]);
