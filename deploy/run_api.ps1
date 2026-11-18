# Laptop 1 — start the PathSnap API on the LAN (run from the repo root).
#   .\deploy\run_api.ps1 -Model artifacts\model.joblib -Trips "C:\path\to\eval_trips"
param(
  [string]$Model = "artifacts\model.joblib",
  [string]$Trips = "data\eval_trips",
  [string]$References = "",
  [string]$Decoder = "classifier_argmax",
  [int]$Port = 8000
)
$env:PATHSNAP_MODEL = $Model
$env:PATHSNAP_TRIPS_DIR = $Trips
$env:PATHSNAP_REFERENCE_DIR = $(if ($References) { $References } else { $Trips })
$env:PATHSNAP_DECODER = $Decoder
Write-Host "Laptop 1 IPv4 addresses (give one of these to Laptop 2):"
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254*" } |
  ForEach-Object { Write-Host ("  http://{0}:{1}" -f $_.IPAddress, $Port) }
python -m uvicorn api.main:app --host 0.0.0.0 --port $Port
