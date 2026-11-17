#!/usr/bin/env sh
# Unduh Plotly ke app/static/vendor/ agar aplikasi bisa jalan tanpa internet (butuh npm sekali saja).
set -e
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
(cd "$tmp" && npm init -y >/dev/null && npm install plotly.js-dist-min@2 >/dev/null)
mkdir -p app/static/vendor
cp "$tmp/node_modules/plotly.js-dist-min/plotly.min.js" app/static/vendor/plotly.min.js
rm -rf "$tmp"
echo "OK: app/static/vendor/plotly.min.js"
