#!/usr/bin/env bash
# Regenerate the basemap's glyph PBFs (MapLibre's signed-distance-field font format) from
# Libre Caslon Text (SIL OFL 1.1, https://github.com/google/fonts/tree/main/ofl/librecaslontext).
#
#   climate_twins/static/atlas/glyphs/build.sh [scratch dir]
#
# Google Fonts ships the face as two variable fonts (upright and italic, wght 400-700), so fontTools
# first cuts the static Regular, Bold and Italic instances; build_pbf_glyphs (Stadia Maps, Rust) then
# renders every 256-codepoint range of each into "<family> <style>/<start>-<end>.pbf", the font stack
# names style.json and map.js ask for. Ranges the font doesn't cover come out (nearly) empty; they're
# kept so that a stray non-Latin label gets blank glyphs rather than a failed request.
# Needs curl, uv and cargo; everything else goes in the scratch dir (default: a temp dir).
set -euo pipefail

out="$(cd "$(dirname "$0")" && pwd)"
work="${1:-$(mktemp -d)}"
mkdir -p "$work/vf" "$work/ttf" "$work/pbf"
base=https://raw.githubusercontent.com/google/fonts/main/ofl/librecaslontext

curl -fsSL -o "$work/vf/upright.ttf" "$base/LibreCaslonText%5Bwght%5D.ttf"
curl -fsSL -o "$work/vf/italic.ttf" "$base/LibreCaslonText-Italic%5Bwght%5D.ttf"
instance() { uvx --from fonttools fonttools varLib.instancer "$work/vf/$1.ttf" "wght=$2" --static --update-name-table -o "$work/ttf/$3.ttf"; }
instance upright 400 "Libre Caslon Text Regular"
instance upright 700 "Libre Caslon Text Bold"
instance italic 400 "Libre Caslon Text Italic"

cargo install --quiet --root "$work/cargo" build_pbf_glyphs
"$work/cargo/bin/build_pbf_glyphs" --overwrite "$work/ttf" "$work/pbf"

for stack in "Libre Caslon Text Regular" "Libre Caslon Text Bold" "Libre Caslon Text Italic"; do
  rm -rf "$out/$stack"
  cp -R "$work/pbf/$stack" "$out/$stack"
done
echo "Glyphs written to $out"
