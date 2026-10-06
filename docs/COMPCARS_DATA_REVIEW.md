# CompCars findings for team review

Yuchen MENG, 11 September 2026. Original team manifests remain unchanged.

- All 345,581 extracted files matched supplied archive-index size/CRC checks.
  This is extraction consistency, not an independent publisher signature.
- All 75,366 task-manifest images decoded and official split membership matched.
- **721 web rows** have unavailable official body type 0 but are named convertible
  (369 train, 352 test). Genuine convertible is type 12. The notebook-generation
  cause has not been established; please review missing-value handling before
  indexing the type-name array. Missing targets should be masked/excluded only
  for body training, retaining images for other valid targets.
- The web subset contains two exact-byte duplicate groups, one crossing official
  train/test. Pilots quarantine training copies of test content and conflicting
  labels, and keep one representative per remaining exact-byte group. No deletion.
- Surveillance metadata confirms **ten colours**, including champagne. The earlier
  nine-class prose summary should be reconciled with the supplied README/MAT file.

Audit scripts produce detailed local CSV/JSON for review. Per-image lists and raw
photos are intentionally not published here. These controls do not establish
near-duplicate or physical-vehicle disjointness. Proposed shared manifest changes
still need team review; no agreement or client permission is implied.
