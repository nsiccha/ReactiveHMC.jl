# Gallery

The ReactiveHMC benchmark dashboard runs eight HMC/NUTS implementations
through a configurable sweep over dimension and condition number on a
diagonal-MVN target, and reports per-gradient throughput
(`overhead_vs_plain`, `grads_per_sec`, `us_per_grad`) for each method.
The views below are the live dashboard during development; in the
deployed docs they're the most recent recording committed under
`docs/src/public/live-reactivehmc/`.

ReactiveHMC's "demo surface" is one comparative benchmark, not a
multi-item gallery — there's no per-item `web/gallery/*.jl` here.
The two recorded routes give the explorer + a worked-example table.

## Benchmark explorer

The interactive Vega-Lite explorer (Method × dim × κ sweep). Default
view: `overhead_vs_plain` vs `dim`, colored by method, faceted by κ,
log axes.

```@raw html
<div class="htmxo-embed-fullwidth">
<div class="htmxo-embed" data-hx-base="live-reactivehmc/" hx-trigger="load" hx-swap="innerHTML">
  <em>Loading ReactiveHMC explorer…</em>
</div>
</div>
```

## Single-config comparison table

The 11-column table for one `(dim, κ, stepsize, n_steps, seed)`
configuration — defaults to `dim=10, κ=100, stepsize=0.5, n_steps=10,
seed=42`. Use query params on the embedded URL to vary.

```@raw html
<div class="htmxo-embed-fullwidth">
<div class="htmxo-embed" data-hx-base="live-reactivehmc/table" hx-trigger="load" hx-swap="innerHTML">
  <em>Loading ReactiveHMC comparison table…</em>
</div>
</div>
```

## Refresh the recording

To re-record the dashboard for the docs:

```
GET /record_gallery               # uses cached benchmark results
GET /record_gallery?force=true    # invalidate cache and re-record
```

The recording dumps both full-page and HX-shape variants of `/` and
`/table` into `docs/src/public/live-reactivehmc/`. Commit the result
and CI deploys it.
