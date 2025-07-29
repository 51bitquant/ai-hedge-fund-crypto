# AI Hedge Fund Crypto Nomad Configuration

This directory contains the Nomad job configuration for deploying the AI Hedge Fund Crypto trading bot on a Nomad cluster.

## Prerequisites

- Nomad 1.0+
- Docker
- Optional: Vault for secrets management

## Building the Docker Image

Before deploying to Nomad, you need to build the Docker image:

```bash
# From the repository root
docker build -t ai-hedge-fund-crypto:latest .
```

## Deploying to Nomad

To deploy the job to Nomad:

```bash
nomad job run nomad/ai-hedge-fund-crypto.nomad.hcl
```

## Configuration

The Nomad job uses metadata to configure the application. You can override these values when running the job:

```bash
nomad job run -meta MODE=live -meta SIGNAL_TICKERS=BTCUSDT,ETHUSDT,SOLUSDT nomad/ai-hedge-fund-crypto.nomad.hcl
```

### Available Configuration Options

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `MODE` | Application mode (backtest or live) | `backtest` |
| `START_DATE` | Start date for backtest | `2025-04-20` |
| `END_DATE` | End date for backtest | `2025-05-01` |
| `PRIMARY_INTERVAL` | Primary interval | `1h` |
| `INITIAL_CASH` | Initial cash amount | `100000` |
| `MARGIN_REQUIREMENT` | Margin requirement | `0.0` |
| `SHOW_REASONING` | Show reasoning | `false` |
| `SHOW_AGENT_GRAPH` | Show agent graph | `false` |
| `SIGNAL_INTERVALS` | Signal intervals (comma-separated) | `30m,1h,4h` |
| `SIGNAL_TICKERS` | Tickers to trade (comma-separated) | `BTCUSDT,ETHUSDT` |
| `SIGNAL_STRATEGIES` | Strategies to use (comma-separated) | `MacdStrategy` |
| `MODEL_NAME` | LLM model name | `gpt-4o-mini` |
| `MODEL_PROVIDER` | LLM provider | `openai` |
| `MODEL_BASE_URL` | LLM base URL (optional) | - |

## Secrets Management

The job is configured to use Vault for secrets management. You need to create a Vault policy named `ai-hedge-fund-crypto` and store the following secrets:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `OPENAI_API_KEY`
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

If you're not using Vault, you can modify the job file to use environment variables directly.

## Scaling

To scale the application, update the `count` parameter in the job file:

```hcl
group "trading-bot" {
  count = 3  # Run 3 instances
  # ...
}
```

Then update the job:

```bash
nomad job run nomad/ai-hedge-fund-crypto.nomad.hcl
```

## Monitoring

The job exposes a metrics port that can be used for monitoring. You can access it at the allocated port on the host machine.