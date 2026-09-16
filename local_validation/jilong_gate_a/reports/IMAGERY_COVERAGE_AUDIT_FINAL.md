# Final imagery coverage audit

Coverage semantics are explicit: `point_covered` is point containment; `aoi_any_overlap` means any intersection; `aoi_full_coverage` means the full defined AOI is inside the image.

```json
{
  "pre_event_0p5m": {
    "official_port": {
      "point_covered": false,
      "aoi_any_overlap": true,
      "aoi_full_coverage": false
    },
    "vector_gap": {
      "point_covered": true,
      "aoi_any_overlap": true,
      "aoi_full_coverage": true
    },
    "old_diagnostic_entry": {
      "point_covered": true,
      "aoi_any_overlap": true,
      "aoi_full_coverage": false
    },
    "full_intended_reach": {
      "point_covered": true,
      "aoi_any_overlap": true,
      "aoi_full_coverage": false
    },
    "image_crs": "GEOGCS[\"GCS_CGCS_2000\",DATUM[\"D_CGCS_2000\",SPHEROID[\"CGCS2000\",6378137,298.257222101004,AUTHORITY[\"EPSG\",\"1024\"]]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AXIS[\"Latitude\",NORTH],AXIS[\"Longitude\",EAST]]",
    "image_resolution": [
      4.499999999999939e-06,
      4.499999999999917e-06
    ]
  },
  "post_event_planet_3m": {
    "official_port": {
      "point_covered": true,
      "aoi_any_overlap": true,
      "aoi_full_coverage": true
    },
    "vector_gap": {
      "point_covered": true,
      "aoi_any_overlap": true,
      "aoi_full_coverage": true
    },
    "old_diagnostic_entry": {
      "point_covered": false,
      "aoi_any_overlap": false,
      "aoi_full_coverage": false
    },
    "full_intended_reach": {
      "point_covered": false,
      "aoi_any_overlap": true,
      "aoi_full_coverage": false
    },
    "image_crs": "EPSG:32645",
    "image_resolution": [
      3.0,
      3.0
    ]
  }
}
```
