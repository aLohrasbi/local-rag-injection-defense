| Model | Defense | Canary rate (95% CI) | Acc strict (95% CI) | Acc lenient | FPR | Retr-hit | Gen p50 (s) | Gen p95 (s) | Total p50 (s) | Total p95 (s) | Refusal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b | none | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 3.0 | 3.9 | 3.0 | 3.9 | 0.00 |
| qwen2.5-1.5b | delimiter | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 2.8 | 3.2 | 2.8 | 3.2 | 0.00 |
| qwen2.5-1.5b | filter | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 2.9 | 3.7 | 2.9 | 3.7 | 0.00 |
| qwen2.5-1.5b | both | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 2.7 | 3.2 | 2.7 | 3.2 | 0.00 |
| qwen2.5-7b | none | 0.18 [0.05, 0.48] | n/a | 0.00 | n/a | 0.61 | 5.8 | 8.4 | 5.8 | 8.5 | 0.00 |
| qwen2.5-7b | delimiter | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 5.7 | 7.3 | 5.7 | 7.4 | 0.00 |
| qwen2.5-7b | filter | 0.09 [0.02, 0.38] | n/a | 0.00 | n/a | 0.61 | 5.7 | 8.5 | 5.7 | 8.5 | 0.00 |
| qwen2.5-7b | both | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 5.7 | 7.4 | 5.7 | 7.4 | 0.00 |

### Paired Exact McNemar Tests (vs. `none`, Holm-corrected)

| Model | Defense | Paired N | Fixed (b) | Broke (c) | p (exact) | p (Holm) |
|---|---|---|---|---|---|---|
| qwen2.5-1.5b | delimiter | 11 | 0 | 0 | 1.000 | 1.000 |
| qwen2.5-1.5b | filter | 11 | 0 | 0 | 1.000 | 1.000 |
| qwen2.5-1.5b | both | 11 | 0 | 0 | 1.000 | 1.000 |
| qwen2.5-7b | delimiter | 11 | 2 | 0 | 0.500 | 1.000 |
| qwen2.5-7b | filter | 11 | 1 | 0 | 1.000 | 1.000 |
| qwen2.5-7b | both | 11 | 2 | 0 | 0.500 | 1.000 |
