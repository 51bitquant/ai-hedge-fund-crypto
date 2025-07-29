# AI Hedge Fund Crypto Helm Chart

This Helm chart deploys the AI Hedge Fund Crypto trading bot on a Kubernetes cluster.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+

## Installing the Chart

To install the chart with the release name `my-release`:

```bash
# Clone the repository
git clone https://github.com/ABoringBusiness/ai-hedge-fund-crypto.git
cd ai-hedge-fund-crypto

# Install the chart
helm install my-release ./helm/ai-hedge-fund-crypto
```

## Configuration

The following table lists the configurable parameters of the AI Hedge Fund Crypto chart and their default values.

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Image repository | `ai-hedge-fund-crypto` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `image.tag` | Image tag | `latest` |
| `config.mode` | Application mode (backtest or live) | `backtest` |
| `config.startDate` | Start date for backtest | `2025-04-20` |
| `config.endDate` | End date for backtest | `2025-05-01` |
| `config.primaryInterval` | Primary interval | `1h` |
| `config.initialCash` | Initial cash amount | `100000` |
| `config.marginRequirement` | Margin requirement | `0.0` |
| `config.showReasoning` | Show reasoning | `false` |
| `config.showAgentGraph` | Show agent graph | `false` |
| `config.signals.intervals` | Signal intervals | `["30m", "1h", "4h"]` |
| `config.signals.tickers` | Tickers to trade | `["BTCUSDT", "ETHUSDT"]` |
| `config.signals.strategies` | Strategies to use | `["MacdStrategy"]` |
| `config.model.name` | LLM model name | `gpt-4o-mini` |
| `config.model.provider` | LLM provider | `openai` |
| `secrets.binanceApiKey` | Binance API key | `""` |
| `secrets.binanceApiSecret` | Binance API secret | `""` |
| `secrets.openaiApiKey` | OpenAI API key | `""` |
| `secrets.groqApiKey` | Groq API key | `""` |
| `secrets.openrouterApiKey` | OpenRouter API key | `""` |
| `secrets.anthropicApiKey` | Anthropic API key | `""` |
| `secrets.googleApiKey` | Google API key | `""` |
| `persistence.enabled` | Enable persistence | `false` |
| `persistence.storageClass` | Storage class | `""` |
| `persistence.accessMode` | Access mode | `ReadWriteOnce` |
| `persistence.size` | Storage size | `1Gi` |

## Example: Installing with Custom Values

Create a `values.yaml` file:

```yaml
config:
  mode: live
  initialCash: 50000
  signals:
    tickers: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    strategies: ["MacdStrategy", "RSIStrategy"]
  model:
    name: "gpt-4o"
    provider: "openai"

secrets:
  binanceApiKey: "your-binance-api-key"
  binanceApiSecret: "your-binance-api-secret"
  openaiApiKey: "your-openai-api-key"

persistence:
  enabled: true
  size: 5Gi
```

Then install the chart:

```bash
helm install my-release ./helm/ai-hedge-fund-crypto -f values.yaml
```

## Scaling

To scale the deployment:

```bash
kubectl scale deployment my-release-ai-hedge-fund-crypto --replicas=3
```

Or update the `replicaCount` in your values file and upgrade the release:

```bash
helm upgrade my-release ./helm/ai-hedge-fund-crypto -f values.yaml
```