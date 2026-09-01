# Copy to secrets.py and fill in. secrets.py is gitignored.
WIFI_SSID = "your-2.4GHz-network"
WIFI_PASSWORD = "your-password"

# Where the node POSTs windows. Must be the laptop's LAN IP -- "localhost"
# would resolve to the ESP32 itself. Find it with `ipconfig getifaddr en0`.
SERVER_URL = "http://192.168.1.100:8000/readings"  # <- your laptop, port, and path
