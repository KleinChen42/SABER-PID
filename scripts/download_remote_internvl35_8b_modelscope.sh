#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hera/pid_reliability_benchmark
MODEL="$ROOT/models/InternVL3_5-8B-modelscope"
BASE=https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-8B/resolve/master
LOG="$ROOT/logs/internvl35_8b_modelscope_download.log"
mkdir -p "$MODEL" "$ROOT/logs"
exec >> "$LOG" 2>&1
echo "INTERNVL_DOWNLOAD_START $(date -Is)"

download() {
  local name="$1"
  local expected="$2"
  local size="$3"
  local url="$BASE/$name"
  echo "FILE_START name=$name expected_bytes=$size sha256=$expected $(date -Is)"
  aria2c --allow-overwrite=false --auto-file-renaming=false --file-allocation=none \
    --continue=true --max-connection-per-server=8 --split=8 --min-split-size=8M \
    --timeout=30 --connect-timeout=30 --max-tries=20 --retry-wait=5 \
    --dir="$MODEL" --out="$name" "$url"
  actual=$(sha256sum "$MODEL/$name" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "HASH_FAIL name=$name actual=$actual expected=$expected"
    return 1
  fi
  echo "FILE_END name=$name size=$(stat -c %s "$MODEL/$name") sha256=$actual $(date -Is)"
}

download config.json e849152d81a9eb158475d493d6c5ba1fc935d35ed01429bbd0182582416bb963 2481
download configuration_intern_vit.py 6530df52748e4f7766568825e1d11f55dfa6cc421990020a640c548dc6cd2646 5546
download configuration_internvl_chat.py e9fdff47225fd380d940a4a2b6818b303fd2a5fdddf44fce0a75b05a4beb1539 4700
download conversation.py b1b9f280e9cb9211e519e2b8071002b57cc742aadb2f49b05fa674ee3c7d83ee 15309
download generation_config.json d599f7f3bee654c13b4b53652d239efb4bf80fbf4b4e46f57318f41b981bae9a 69
download merges.txt 8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5 1671853
download modeling_intern_vit.py a33ec11c517041c7fcbaaff3d736d846498468efaf4a64400a89d55132d5f638 18151
download modeling_internvl_chat.py ec972dd3d1c7f5da06455034d5cfe54968a9555dff60804edee9a7fb5759db34 16518
download preprocessor_config.json af09ffbe57900d5ae07f93f4c20693559a272e32fa6c5f816132cb71985aac98 666
download processor_config.json 3511f4ae65c2fa5c1ab6a0fb2bb2fe767d6f0ee53ba412953b33acc892327c2a 72
download model.safetensors.index.json 96c2010f8ba2a6b1687753a52fe797667ab8a8de958104e409e7e65439074af5 68446
download special_tokens_map.json 13a6dea522a937ef05172912c66214162c75165e723a3d6cf4ae2d09752013bf 744
download tokenizer.json 6581c44164d273d4222df982905a7e0450dcf3a4a7ebe98f9ec53e4de05beffe 11424300
download tokenizer_config.json 01fa09d7bd718d09a7d795aec4fb310f2be928b5273fd16faf64037f0e3993d5 7164
download video_preprocessor_config.json b7f9c784a27d30ddb3fc78fca353ab6ce982c2a263b702e2547302e8c1e0087a 1345
download vocab.json ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 2776833
download chat_template.jinja cabb34545cf3d04f4d256439f6b8e9f6fcf61edaaee3b8cb5c0937a7ea6206d7 475
download added_tokens.json e5c2531bd62d25dd2a7401203a20f7ffb0976a026299cb4d6ff2d559d7ac3e28 892
download model-00001-of-00004.safetensors 8948bf3bae28444c3a5f9120429acff1121ff3716b5f5e4a5041d511cab0bd12 4982437672
download model-00002-of-00004.safetensors 0646e5f2b3a811150998a7cb47bbbd38d5e812b2f2be365a2f6fe52c5c789c13 3848612416
download model-00003-of-00004.safetensors e834ec95e94830b3da4a23df06abc296ba48756589eec359bdbfcc541358721a 4999903176
download model-00004-of-00004.safetensors 8c33a67f6ea2cc5c35fd631a763d70b52e4b132e12ca331c1767c0a95326840 3225776168

echo "INTERNVL_DOWNLOAD_END $(date -Is)"
