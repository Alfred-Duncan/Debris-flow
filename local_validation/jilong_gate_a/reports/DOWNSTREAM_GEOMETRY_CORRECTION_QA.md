# Downstream geometry correction QA

`HUMAN_IMAGE_REVIEW = APPROVED_FOR_LOCAL_CHANNEL_CORRECTION`.

The rejected P0--P2 official line is **1878.66 m**; its 12.5 m maximum adverse rise over 250 m is **200.0 m**. The approved physical-channel correction is **3684.46 m** and terminates at S4 rather than returning to P2. Its QA-masked native-12.5-m maximum 250 m rise is **148.0 m**; independent 8 m DEM gives **86.0 m**.

The maximum/mean lateral separation from the rejected line is 1258.0/443.4 m. Tie-in is S4 `(340571.826, 3129640.776)`; no earlier S4-side official geometry was retained because it was not image/DEM consistent.

The shadowed trunk has local disagreement between native 12.5 m and 8 m data. Raw 12.5 m values are preserved in the final CSV. Values differing by more than 120 m from a colocated valid 8 m sample are marked `dem_valid=false` for slope QA only; this is an evidence flag, not a DEM edit.
