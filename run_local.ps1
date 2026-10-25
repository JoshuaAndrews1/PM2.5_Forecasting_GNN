$env:PM25_CONFIG = Join-Path $PSScriptRoot 'config.local.yaml'
& (Join-Path $PSScriptRoot '..\PM2.5_Forecasting_GNN-main\.venv37\Scripts\python.exe') (Join-Path $PSScriptRoot 'train.py')
