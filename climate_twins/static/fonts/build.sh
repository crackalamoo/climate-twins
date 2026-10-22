#!/usr/bin/env bash
# Regenerate the panel's web fonts from Libre Caslon Text (SIL OFL 1.1,
# https://github.com/google/fonts/tree/main/ofl/librecaslontext).
#
#   climate_twins/static/fonts/build.sh [scratch dir]
#
# Like atlas/glyphs/build.sh, fontTools cuts static Regular, Bold and Italic from Google Fonts'
# variable fonts, then subsets each into the latin and latin-ext woff2 files style.css's @font-face
# rules name (same unicode ranges). Every OpenType feature is kept: prebuilt subsets (fontsource's)
# drop onum, the old-style figures the panel sets text in.
# The static Regular itself, whole, goes to climate_twins/fonts/ for the link-preview card
# (preview.py), out of the site's static files.
# Needs curl and uv; everything else goes in the scratch dir (default: a temp dir).
set -euo pipefail

out="$(cd "$(dirname "$0")" && pwd)"
work="${1:-$(mktemp -d)}"
mkdir -p "$work/vf" "$work/ttf"
base=https://raw.githubusercontent.com/google/fonts/main/ofl/librecaslontext

curl -fsSL -o "$work/vf/upright.ttf" "$base/LibreCaslonText%5Bwght%5D.ttf"
curl -fsSL -o "$work/vf/italic.ttf" "$base/LibreCaslonText-Italic%5Bwght%5D.ttf"
instance() { uvx --from fonttools fonttools varLib.instancer "$work/vf/$1.ttf" "wght=$2" --static -o "$work/ttf/$3.ttf"; }
instance upright 400 400-normal
instance upright 700 700-normal
instance italic 400 400-italic

latin="U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+0304,U+0308,U+0329,U+2000-206F,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD"
latin_ext="U+0100-02BA,U+02BD-02C5,U+02C7-02CC,U+02CE-02D7,U+02DD-02FF,U+0304,U+0308,U+0329,U+1D00-1DBF,U+1E00-1E9F,U+1EF2-1EFF,U+2020,U+20A0-20AB,U+20AD-20C0,U+2113,U+2C60-2C7F,U+A720-A7FF"
for face in 400-normal 700-normal 400-italic; do
  for subset in latin latin-ext; do
    ranges=$([ "$subset" = latin ] && echo "$latin" || echo "$latin_ext")
    uvx --from fonttools --with brotli pyftsubset "$work/ttf/$face.ttf" --unicodes="$ranges" \
      --layout-features='*' --flavor=woff2 --output-file="$out/libre-caslon-text-$subset-$face.woff2"
  done
done

cp "$work/ttf/400-normal.ttf" "$out/../../fonts/libre-caslon-text-regular.ttf"
echo "Fonts written to $out"
