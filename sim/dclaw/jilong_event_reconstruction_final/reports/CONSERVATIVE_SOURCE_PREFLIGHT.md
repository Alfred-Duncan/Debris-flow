# Conservative source preflight

The source area is exactly three 64 m cells (12,288 m²). The fixed unit-Froude closure supplies only injected momentum, never volume.

| case | V (m3) | T (s) | Qpeak (m3/s) | Usrc peak (m/s) | final F(T) (m3) | pass |
|---|---:|---:|---:|---:|---:|---|
| C1 | 1000000 | 90 | 17453.292520 | 10.478462 | 1000000 | True |
| C2 | 1000000 | 180 | 8726.646260 | 8.316761 | 1000000 | True |
| C3 | 2000000 | 90 | 34906.585040 | 13.202035 | 2000000 | True |
| C4 | 2000000 | 180 | 17453.292520 | 10.478462 | 2000000 | True |
| C5 | 3000000 | 90 | 52359.877560 | 15.112557 | 3000000 | True |
| C6 | 3000000 | 180 | 26179.938780 | 11.994845 | 3000000 | True |

All cases have exactly three valid cells, finite and nonnegative increments, downstream tangents, and Usrc peak <40 m/s.
