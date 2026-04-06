# Terminal Monitoring Suite

This project uses terminal-based tools for live monitoring of the AWS infrastructure. All scripts are located in `scripts/monitoring/`.

## 1. Live Application Logs (CloudWatch Tail)

We use the `cw` tool to tail CloudWatch logs in real-time.

**Tail ECS Streamer Logs:**
```bash
cw tail -f /ecs/tick-streamer --region ap-southeast-1
```

**Tail Lambda Scraper Logs:**
```bash
cw tail -f /aws/lambda/news-scraper-TradingViewScraper-xtdIdfZ9nANw --region ap-southeast-1
```

## 2. Live Performance Dashboards

These scripts fetch live metrics. Run them using the `watch` command to create a constantly updating dashboard.

> **⚠️ IMPORTANT:** You must use the `-c` or `--color` flag with the `watch` command so that it prints the colors correctly in your terminal!

**Monitor ECS (CPU/Memory):**
```bash
watch -c -n 10 ./scripts/monitoring/monitor-ecs.sh
```

**Monitor Lambda (Invocations/Errors/Duration):**
```bash
watch -c -n 30 ./scripts/monitoring/monitor-lambda.sh
```

**Monitor Kinesis Health (Throughput/Errors):**
```bash
watch -c -n 10 ./scripts/monitoring/monitor-kinesis-health.sh
```

## 3. Live Kinesis Data Tailing

To view the raw JSON data flowing through your Kinesis streams in real-time:

```bash
# Tail BTC ticks
./scripts/monitoring/tail-kinesis.sh tick-stream-btcusdt

# Tail ETH ticks
./scripts/monitoring/tail-kinesis.sh tick-stream-ethusdt
```
