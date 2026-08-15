# Table 1: KV-Cache Memory Consumption & Compression Ratios

| Method | Context Length | Original (KB) | Compressed (KB) | Ratio | Saved (%) |
| --- | --- | --- | --- | --- | --- |
| FP16 Baseline | 2048 | 65536.0 | 65536.0 | 1.00x | 0.0% |
| Fixed 4-bit | 2048 | 65536.0 | 17408.0 | 3.76x | 73.4% |
| Fixed 3-bit | 2048 | 65536.0 | 13312.0 | 4.92x | 79.7% |
| Fixed 2-bit | 2048 | 65536.0 | 9216.0 | 7.11x | 85.9% |
| AdaptiveKV (Threshold) | 2048 | 65536.0 | 15847.5 | 4.14x | 75.8% |
| AdaptiveKV (Budget 25%) | 2048 | 65536.0 | 19456.0 | 3.37x | 70.3% |
| FP16 Baseline | 512 | 16384.0 | 16384.0 | 1.00x | 0.0% |
| Fixed 4-bit | 512 | 16384.0 | 4352.0 | 3.76x | 73.4% |
| Fixed 3-bit | 512 | 16384.0 | 3328.0 | 4.92x | 79.7% |
| Fixed 2-bit | 512 | 16384.0 | 2304.0 | 7.11x | 85.9% |
| AdaptiveKV (Threshold) | 512 | 16384.0 | 3854.5 | 4.25x | 76.5% |
| AdaptiveKV (Budget 25%) | 512 | 16384.0 | 4864.0 | 3.37x | 70.3% |
