# Datos del Proyecto

Los archivos CSV de datos no se incluyen en el repositorio Git por su tamaño (~1.3GB).

## Estructura esperada

```
data/
├── rates/                          # Candles OHLCV
│   ├── eurusd_m30_rates.csv
│   ├── eurusd_h1_rates.csv
│   ├── eurusd_h2_rates.csv
│   ├── eurusd_h3_rates.csv
│   ├── eurusd_h4_rates.csv
│   ├── eurusd_d1_rates.csv
│   └── ... (8 pares × 6 timeframes = 48 archivos)
├── indicators/                     # Indicadores técnicos pre-calculados
│   ├── technical_indicator_eurusd.csv
│   ├── technical_indicator_gbpusd.csv
│   └── ... (8 archivos, ~130-175MB cada uno)
├── trades/
│   └── paper_trades.csv            # Historial de trades (~260 trades)
├── analysis/
│   └── signal_discoveries.csv      # Señales descubiertas
└── sample/                         # Subset pequeño (incluido en Git)
    ├── eurusd_h1_rates_sample.csv  # Primeras 1000 filas
    └── technical_indicator_eurusd_sample.csv
```

## Pares FOREX incluidos

| Par | Descripción |
|-----|-------------|
| EURUSD | Euro / Dólar estadounidense |
| GBPUSD | Libra esterlina / Dólar |
| USDJPY | Dólar / Yen japonés |
| EURJPY | Euro / Yen japonés |
| USDCAD | Dólar / Dólar canadiense |
| EURCAD | Euro / Dólar canadiense |
| USDCHF | Dólar / Franco suizo |
| EURGBP | Euro / Libra esterlina |

## Timeframes

| Código | Periodo |
|--------|---------|
| M30 | 30 minutos |
| H1 | 1 hora |
| H2 | 2 horas |
| H3 | 3 horas |
| H4 | 4 horas |
| D1 | 1 día |

## Cómo obtener los datos

### Opción 1: Desde backup local (si tienes acceso)

```bash
# Copiar rates
cp /mnt/e/PostgreSQL/markets/{eurusd,gbpusd,usdjpy,eurjpy,usdcad,eurcad,usdchf,eurgbp}_{m30,h1,h2,h3,h4,d1}_rates.csv data/rates/

# Copiar indicadores
cp /mnt/e/PostgreSQL/ai_model/technical_indicator_{eurusd,gbpusd,usdjpy,eurjpy,usdcad,eurcad,usdchf,eurgbp}.csv data/indicators/

# Copiar trades
cp /mnt/e/PostgreSQL/ai_model/paper_trades.csv data/trades/
cp /mnt/e/PostgreSQL/ai_model/signal_discoveries.csv data/analysis/
```

### Opción 2: Usar datos de muestra

El directorio `data/sample/` contiene subsets pequeños (1000 filas) que permiten
ejecutar el stack y los notebooks sin necesidad de los datasets completos.

## Formato de los archivos

### Rates (OHLCV)
```csv
rate_time,open,high,low,close,volume,readable_date
1577836800,1.12130,1.12230,1.12050,1.12180,12345,2020-01-01 00:00:00
```

### Indicadores técnicos
```csv
symbol,timeframe,rate_time,sma_20,sma_50,sma_200,ema_12,ema_26,rsi_14,macd,atr_14,...
EURUSD,H1,1577836800,1.121,1.120,1.118,1.121,1.120,55.2,0.0012,0.0015,...
```

### Paper trades
```csv
id,symbol,direction,entry_time,entry_price,exit_time,exit_price,sl_price,tp_price,size,pnl_pips,exit_reason,entry_model,signal_timeframe
1,EURUSD,long,2026-02-13 10:00:00,1.0780,2026-02-13 14:30:00,1.0810,1.0750,1.0810,0.1,30.0,tp_hit,EMA_RSI_long_H1,H1
```
