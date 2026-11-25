#!/bin/sh
# Recreates the library's tutorial-style textures procedurally (ImageMagick 7).
# Run from this folder: sh make_textures.sh. Output is committed; rerun only to change them.
set -e
S=128
# Soft round glow with a hot core (flares, sparks, glows).
magick -size ${S}x${S} radial-gradient:white-black -evaluate pow 2.2 -alpha copy -fill white -colorize 100 flare.png
# Four-point star flare (muzzle flash core, firework glints, impact).
magick -size ${S}x${S} xc:black \
  \( -size ${S}x6 gradient:black-white -size ${S}x6 gradient:white-black +append -resize ${S}x6! \) -gravity center -compose lighten -composite \
  \( -size 6x${S} xc:white \) -gravity center -composite -blur 0x2 \
  \( -size ${S}x${S} radial-gradient:white-black -evaluate pow 3 \) -compose lighten -composite \
  -colorspace gray -alpha copy -fill white -colorize 100 star.png
# Shockwave ring: thin bright annulus with soft edges.
magick -size 256x256 radial-gradient:white-black -fx 'dd=1-u; exp(-pow((dd-0.82)/0.06,2))' -alpha copy -fill white -colorize 100 ring.png
# Smoke/dust puff: fractal noise in a soft round mask.
magick -seed 3 -size ${S}x${S} plasma:grey50-grey50 -colorspace gray -blur 0x2 -auto-level \
  \( -size ${S}x${S} radial-gradient:white-black -evaluate pow 1.5 \) -compose multiply -composite -auto-level \
  -alpha copy -fill white -colorize 100 puff.png
# Spark streak (velocity-aligned sparks, raindrops): tall thin glow.
magick -size 32x128 radial-gradient:white-black -evaluate pow 2 -alpha copy -fill white -colorize 100 streak.png
# Muzzle flash petals: 5 flame tongues around a hot centre, 2x2 flipbook variations.
for i in 0 1 2 3; do
  magick -seed $((10+i)) -size ${S}x${S} plasma:grey40-grey60 -colorspace gray -blur 0x3 -auto-level \
    -virtual-pixel black -distort Polar 0 -resize ${S}x${S}! \
    \( -size ${S}x${S} xc:black -fill white -draw "translate 64,64 rotate $((i*17)) path 'M 0,-60 L 9,-8 L 57,-18 L 10,6 L 35,52 L 0,14 L -35,52 L -10,6 L -57,-18 L -9,-8 Z'" -blur 0x5 \) \
    -compose multiply -composite \
    \( -size ${S}x${S} radial-gradient:white-black -evaluate pow 2 \) -compose screen -composite -auto-level \
    -alpha copy -fill white -colorize 100 /tmp/mf_$i.png
done
magick montage /tmp/mf_0.png /tmp/mf_1.png /tmp/mf_2.png /tmp/mf_3.png -background none -tile 2x2 -geometry +0+0 muzzle_2x2.png
# Fireball flipbook 4x4: billowing noise that grows, brightens then breaks up.
for i in $(seq 0 15); do
  r=$((30 + i*2)); c=$((100 - i*5))
  magick -seed $((40+i)) -size ${S}x${S} plasma:grey30-grey70 -colorspace gray -blur 0x2 -auto-level \
    \( -size ${S}x${S} radial-gradient:white-black -evaluate pow 1.2 -level 0%,$((40+i*3))% \) -compose multiply -composite \
    -level $((i*3))%,100% -alpha copy -fill white -colorize 100 /tmp/fb_$i.png
done
magick montage $(seq -f /tmp/fb_%g.png 0 15) -background none -tile 4x4 -geometry +0+0 fireball_4x4.png
# Swirl (portal): noise twisted around the centre, in a ring mask.
magick -seed 7 -size 256x256 plasma:grey20-grey80 -colorspace gray -blur 0x4 -auto-level -swirl 540 \
  \( -size 256x256 radial-gradient:white-black -fx 'dd=1-u; exp(-pow((dd-0.55)/0.28,2))*min(1,u*6)' \) -compose multiply -composite -level 0%,70% \
  -alpha copy -fill white -colorize 100 swirl.png
rm -f /tmp/mf_*.png /tmp/fb_*.png
