# Laptop 1, once, in an *Administrator* PowerShell: allow inbound API traffic on private networks.
param([int]$Port = 8000)
New-NetFirewallRule -DisplayName "PathSnap API $Port" -Direction Inbound -Protocol TCP `
  -LocalPort $Port -Action Allow -Profile Private,Domain
