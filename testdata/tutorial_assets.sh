#!/bin/sh
# Source manifest for testdata/tutorial_assets/ (git-ignored, never
# committed): third-party files from the Niagara tutorials that the
# effects/library batch-2 effects follow. Run from the repository root:
#   sh testdata/tutorial_assets.sh
# effects/library/tutorial_assets is a symlink to that folder.
#
# Files                         Source                                             Terms
# fireworks/Flare1-4.png        Enrique Ventura, "Unreal 5.3 - Advanced Niagara    shared free in the video
#                               fireworks - Chapter 3" (PA9sQV8uI8g), Google Drive description for tutorial use
#                               links in the video description
# frames/<effect>.jpg           YouTube thumbnail (maxresdefault) of each          reference only (comparison),
# frames/<effect>_hq{1,2,3}.jpg tutorial, and YouTube's auto frames at 25/50/75 %  copyright of the creators
#
# No other tutorial offers its assets for free (project files are on
# Patreon/FAB); those textures are recreated in effects/library/textures
# (make_textures.sh) and effects/library/materials (*.wesl).
set -e
D=testdata/tutorial_assets
mkdir -p $D/fireworks $D/frames
gd() { curl -sL -o "$2" "https://drive.google.com/uc?export=download&id=$1"; }
gd 1lNncVFCdTFyDbFSmk2m7jeMYyoVtcTlh $D/fireworks/Flare1.png
gd 1-5JeeKeaIpbovMIiEROOrr6QCbVo6nH0 $D/fireworks/Flare2.png
gd 1oxfU9GyjoQvFNb_nEK6JV8q3f01Y47CW $D/fireworks/Flare3.png
gd 1VyTSLzUtIi1NkOVhqSSKBjXHJ_zh9SBc $D/fireworks/Flare4.png
for p in shockwave_ring:_EWOPaYytF4 lightning_strike:NwdUdaotzNA portal_vortex:uNU0xCeIL2I \
         stylized_tornado:360__bcB9cM rain_splash:QgEKhdV0xrA rain_base:T1o0Yu1VyyY \
         muzzle_flash:SGoNF1UTD3I footstep_dust:4LevtPm4i_I heal_aura:R1W_8Z1trsY \
         fireworks_show:PA9sQV8uI8g meteor_rain:-Cdn0_98PXM; do
  n=${p%%:*}; id=${p#*:}
  curl -sf -o $D/frames/$n.jpg "https://i.ytimg.com/vi/$id/maxresdefault.jpg" \
    || curl -sf -o $D/frames/$n.jpg "https://i.ytimg.com/vi/$id/hqdefault.jpg"
  for k in 1 2 3; do curl -sf -o $D/frames/${n}_hq$k.jpg "https://i.ytimg.com/vi/$id/hq$k.jpg" || true; done
done
echo "fetched into $D"
