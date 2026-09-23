#!/usr/bin/env bash
# Resume the 27-shoe pipeline at 11-B. sneaker_b33 and sneaker_4 are held back:
# their fitted anatomy leaves the frozen outer envelope (617 and 194 vertices
# outside), which 11-B rejects outright. 11-B preflights every shoe before
# processing any, so leaving them in aborts the whole batch.
set -u
source "$(dirname "$0")/env.sh"
tmux kill-session -t pipeline27b 2>/dev/null || true
tmux new-session -d -s pipeline27b "cd $FF_REPO/FootShellGaussian &&   /home/ab5298/anaconda3/envs/Shell/bin/python -m anatomical_coordinates.pipeline.run_to_11d     --root /home/ab5298/Outputs/FootShellGaussian/pipeline27 --gpus 2 --jobs 32 --leg numpy --from-stage 11b     --shoes adidas_substance aj_12_basketball_sneakers birkenstock_arizona_sandal canvas_shoe crocs crocs_by_speedyart_studio crocs_shoe duinn_shoes_womens_hiking_sandal_sport leather_boots nike_air_jordan pb129_shoe_low priest_karol_wojtyas_sports_shoes rtfktchallenge_golden_by_franz_vega sandal_1 sandals_0001 shoes_mockup_asset_vans_skate_old_skool_shoes sneaker_1 sneaker_2 sneaker_3 sneaker_5 sneaker_6 sneaker_7 sneaker_8 sneakers_seen ww_ii_german_jack_boots  2>&1 | tee /home/ab5298/Outputs/FootShellGaussian/pipeline27/logs/pipeline27_11b.log; echo DONE; exec bash"
echo "started pipeline27b"
