# Generator-rationale disclosure ablation evaluation

## Conclusion

Use **without rationale** as the current default for the judge. Showing the generator's rationale produced a slightly higher raw acceptance rate, but a full visual audit of every accepted item found lower precision and one fewer valid output.

This experiment ablates whether the **judge sees** `generator_rationale`; it does not ablate whether the generator produces a rationale or otherwise reasons during generation.

## Quantitative comparison

| Measure | Without rationale | With rationale |
|---|---:|---:|
| Packets | 50 | 50 |
| Accepted by pipeline | 25 (50%) | 27 (54%) |
| Generation attempts | 122 | 117 |
| Judged attempts | 120 | 117 |
| Evidence-groundedness PASS | 81/120 (67.5%) | 82/117 (70.1%) |
| Visually verified valid accepted items | **21/25 (84.0%)** | **20/27 (74.1%)** |
| Visually verified false positives | **4/25 (16.0%)** | **7/27 (25.9%)** |
| Valid output yield per input packet | **21/50 (42.0%)** | **20/50 (40.0%)** |

The final pipeline outcomes were paired as follows:

- accepted in both: 17
- accepted only without rationale: 8
- accepted only with rationale: 10
- rejected in both: 15

The raw 4-point acceptance difference is not significant under an exact paired McNemar test (`p = 0.815`). After visual correction, the paired valid-yield outcomes were 12 valid in both, 9 valid only without rationale, 8 valid only with rationale, and 21 valid in neither (`p = 1.0`). Wilson intervals for accepted-item precision also overlap widely: without rationale 65.3-93.6%, with rationale 55.3-86.8%.

## Video audit

I audited all 52 accepted QA items across the 35 unique accepted clip pairs. For each 30-second pair, I inspected the complete 1-fps frame sequence stored beside the original videos and opened full-resolution frames for ambiguous cases.

The visually rejected accepted items were:

### Without rationale

- `DAY2_18360000_A2_A4_0-1`: the asker's view already shows the orange drink during the toast, so the claimed information gap is not real.
- `DAY4_15383000_A1_A4_0-1`: the asker repeatedly sees the pink flowers while clearing the table.
- `DAY5_15283000_A3_A4_0-1`: the asker directly sees the person holding the cardboard box.
- `DAY6_16390000_A1_A2_0-1`: the asker directly sees the seated person and projector screen.

### With rationale

- `DAY4_16533000_A1_A4_0-1`: the person looks at a whiteboard on its stand but never picks it up, contrary to the question.
- `DAY4_18220000_A1_A3_0-1`: the microwave item is a white disposable paper plate, not a glass bowl covered in plastic wrap.
- `DAY5_15283000_A3_A4_0-1`: the asker directly sees the person holding the cardboard box.
- `DAY6_11073000_A2_A4_0-1`: the asker's own view shows the person and brown upholstered seat.
- `DAY6_11133000_A4_A5_0-1`: the cards are on an outdoor patio table, not a living-room table.
- `DAY7_15050000_A1_A2_0-1`: the large white box is standing on the table; the seated person is not holding it.
- `DAY7_19220000_A1_A2_0-1`: the black camera/mount is placed high on a wall or cabinet edge, not on top of the microwave.

## Interpretation and next experiment

The raw acceptance gain should not be read as a causal benefit from rationale disclosure. Only 3 of 50 first-attempt generator outputs were exact matches across arms; the other 47 differed despite the shared seed, and retry paths diverged further after different feedback. All three exact first-attempt matches received the same reject outcome in both arms. Thus most observed differences combine generation sampling variance, different retry trajectories, and the judge treatment.

For a clean causal test, generate each candidate once, cache it, and judge the exact same candidate twice: once with and once without the rationale, using randomized condition order. If the research question is whether rationale improves generation itself, that needs a separate generator-prompt ablation rather than this judge-disclosure switch.
