job "ai-hedge-fund-crypto" {
  datacenters = ["dc1"]
  type        = "service"

  group "trading-bot" {
    count = 1

    update {
      max_parallel     = 1
      min_healthy_time = "30s"
      healthy_deadline = "5m"
      auto_revert      = true
    }

    restart {
      attempts = 3
      interval = "5m"
      delay    = "25s"
      mode     = "delay"
    }

    network {
      port "metrics" {
        to = 8080
      }
    }

    task "bot" {
      driver = "docker"

      config {
        image = "ai-hedge-fund-crypto:latest"
        ports = ["metrics"]
        
        # Mount the config file
        volumes = [
          "local/config.yaml:/app/config.yaml"
        ]
      }

      template {
        data = <<EOH
# Rendered configuration file
mode: {{ env "NOMAD_META_MODE" }}
start_date: {{ env "NOMAD_META_START_DATE" }}
end_date: {{ env "NOMAD_META_END_DATE" }}
primary_interval: {{ env "NOMAD_META_PRIMARY_INTERVAL" }}
initial_cash: {{ env "NOMAD_META_INITIAL_CASH" }}
margin_requirement: {{ env "NOMAD_META_MARGIN_REQUIREMENT" }}
show_reasoning: {{ env "NOMAD_META_SHOW_REASONING" }}
show_agent_graph: {{ env "NOMAD_META_SHOW_AGENT_GRAPH" }}
signals:
  intervals: {{ env "NOMAD_META_SIGNAL_INTERVALS" | split "," | toJSON }}
  tickers: {{ env "NOMAD_META_SIGNAL_TICKERS" | split "," | toJSON }}
  strategies: {{ env "NOMAD_META_SIGNAL_STRATEGIES" | split "," | toJSON }}
model:
  name: {{ env "NOMAD_META_MODEL_NAME" }}
  provider: {{ env "NOMAD_META_MODEL_PROVIDER" }}
  {{- if env "NOMAD_META_MODEL_BASE_URL" }}
  base_url: {{ env "NOMAD_META_MODEL_BASE_URL" }}
  {{- end }}
EOH
        destination = "local/config.yaml"
      }

      env {
        BINANCE_API_KEY     = "{{ env \"NOMAD_VAULT_SECRET_BINANCE_API_KEY\" }}"
        BINANCE_API_SECRET  = "{{ env \"NOMAD_VAULT_SECRET_BINANCE_API_SECRET\" }}"
        OPENAI_API_KEY      = "{{ env \"NOMAD_VAULT_SECRET_OPENAI_API_KEY\" }}"
        GROQ_API_KEY        = "{{ env \"NOMAD_VAULT_SECRET_GROQ_API_KEY\" }}"
        OPENROUTER_API_KEY  = "{{ env \"NOMAD_VAULT_SECRET_OPENROUTER_API_KEY\" }}"
        ANTHROPIC_API_KEY   = "{{ env \"NOMAD_VAULT_SECRET_ANTHROPIC_API_KEY\" }}"
        GOOGLE_API_KEY      = "{{ env \"NOMAD_VAULT_SECRET_GOOGLE_API_KEY\" }}"
      }

      resources {
        cpu    = 500
        memory = 512
      }

      # Optional: Vault integration for secrets
      vault {
        policies = ["ai-hedge-fund-crypto"]
      }
    }

    # Optional: Add a volume for persistent data
    volume "data" {
      type      = "host"
      source    = "ai-hedge-fund-crypto-data"
      read_only = false
    }

    volume_mount {
      volume      = "data"
      destination = "/app/data"
      read_only   = false
    }
  }

  meta {
    # Default configuration values
    MODE                = "backtest"
    START_DATE          = "2025-04-20"
    END_DATE            = "2025-05-01"
    PRIMARY_INTERVAL    = "1h"
    INITIAL_CASH        = "100000"
    MARGIN_REQUIREMENT  = "0.0"
    SHOW_REASONING      = "false"
    SHOW_AGENT_GRAPH    = "false"
    SIGNAL_INTERVALS    = "30m,1h,4h"
    SIGNAL_TICKERS      = "BTCUSDT,ETHUSDT"
    SIGNAL_STRATEGIES   = "MacdStrategy"
    MODEL_NAME          = "gpt-4o-mini"
    MODEL_PROVIDER      = "openai"
    # MODEL_BASE_URL      = "https://api.openai.com/v1"
  }
}