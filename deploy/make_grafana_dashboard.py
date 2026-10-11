"""Generate deploy/grafana/dashboards/gateway.json from a short panel list.

    python deploy/make_grafana_dashboard.py

Grafana dashboard JSON is long and repetitive; this keeps the definition readable.
"""

import json
from pathlib import Path

OUT = Path(__file__).parent / "grafana" / "dashboards" / "gateway.json"
DS = {"type": "prometheus", "uid": "prometheus"}

# (title, unit, [(PromQL, legend)], width, height, kind)
PANELS = [
    ("Requests per second by serving model", "reqps",
     [('sum by (served_model) (rate(gateway_requests_total[1m]))', "{{served_model}}")], 12, 8, "timeseries"),
    ("Latency (successful requests)", "s",
     [('histogram_quantile(0.5, sum by (le) (rate(gateway_request_latency_seconds_bucket[5m])))', "p50"),
      ('histogram_quantile(0.95, sum by (le) (rate(gateway_request_latency_seconds_bucket[5m])))', "p95")],
     12, 8, "timeseries"),
    ("Error rate (after retries and fallbacks)", "percentunit",
     [('sum(rate(gateway_requests_total{outcome="error",status!~"401|429"}[5m])) '
       '/ clamp_min(sum(rate(gateway_requests_total{status!~"401|429"}[5m])), 1e-9)', "errors")], 8, 8, "timeseries"),
    ("Fallback rate", "percentunit",
     [('sum(rate(gateway_requests_total{fallback="true"}[5m])) '
       '/ clamp_min(sum(rate(gateway_requests_total{cache_hit="false",status!~"401|429"}[5m])), 1e-9)', "fallbacks")],
     8, 8, "timeseries"),
    ("Response cache hit rate", "percentunit",
     [('sum(rate(gateway_requests_total{cache_hit="true"}[5m])) '
       '/ clamp_min(sum(rate(gateway_requests_total{status!~"401|429"}[5m])), 1e-9)', "hits")], 8, 8, "timeseries"),
    ("Spend per hour by key (USD)", "currencyUSD",
     [('sum by (api_key_id) (increase(gateway_cost_usd_total[1h]))', "{{api_key_id}}")], 12, 8, "timeseries"),
    ("Upstream attempts by result", "reqps",
     [('sum by (model, result) (rate(gateway_upstream_attempts_total[1m]))', "{{model}} {{result}}")], 12, 8, "timeseries"),
    ("Open circuits", "none",
     [('sum by (model) (gateway_circuit_open)', "{{model}}")], 8, 6, "stat"),
    ("Cache savings, last 24h (USD)", "currencyUSD",
     [('sum by (kind) (increase(gateway_cache_savings_usd_total[24h]))', "{{kind}}")], 8, 6, "stat"),
    ("Anthropic quota remaining (requests per minute)", "none",
     [('min by (model) (gateway_upstream_quota_remaining{kind="requests"})', "{{model}}")], 12, 8, "timeseries"),
    ("Anthropic quota remaining (input tokens per minute)", "none",
     [('min by (model) (gateway_upstream_quota_remaining{kind="input-tokens"})', "{{model}}")], 12, 8, "timeseries"),
    ("Time to first token p95 by model", "s",
     [('histogram_quantile(0.95, sum by (le, served_model) (rate(gateway_time_to_first_token_seconds_bucket[5m])))',
       "{{served_model}}")], 8, 6, "stat"),
]


def build() -> dict:
    panels, x, y, row_height = [], 0, 0, 0
    for i, (title, unit, queries, w, h, kind) in enumerate(PANELS, start=1):
        if x + w > 24:
            x, y = 0, y + row_height
            row_height = 0
        panels.append({
            "id": i, "type": kind, "title": title, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "fieldConfig": {"defaults": {"unit": unit, **({"min": 0} if kind == "timeseries" else {})}, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}} if kind == "timeseries" else {},
            "targets": [{"refId": chr(65 + j), "datasource": DS, "expr": expr, "legendFormat": legend}
                        for j, (expr, legend) in enumerate(queries)],
        })
        x += w
        row_height = max(row_height, h)
    return {
        "uid": "llm-gateway", "title": "LLM Gateway", "schemaVersion": 39, "version": 1,
        "time": {"from": "now-1h", "to": "now"}, "refresh": "10s", "tags": ["llm-gateway"], "panels": panels,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
