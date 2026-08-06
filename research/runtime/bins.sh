#!/bin/sh
# Build the 3 frozen rust binaries into a PERSISTENT dir (never /tmp — wiped on VM restart).
# feature_builder <- master rust_ingest; build_samples + grid_sim_exitdbg <- lib from
# branch claude/husdc-rev1 + frozen bin sources scripts/{build_samples_husdc,grid_sim_exitdbg}.rs.
# Usage: bins.sh <repo_dir> [out_dir=$HOME/research_bins]
# After building, run the parity ritual before production (README).
set -e
REPO=${1:?repo dir}; OUT=${2:-$HOME/research_bins}
command -v cc >/dev/null || { sudo apt-get -qq update && sudo apt-get -qq install -y build-essential; }
[ -x "$HOME/.cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y -q --default-toolchain stable
. "$HOME/.cargo/env"
mkdir -p "$OUT"

# 1) feature_builder from master rust_ingest
cd "$REPO/rust_ingest"
CARGO_TARGET_DIR="$OUT/fb_target" cargo build --release --bin feature_builder

# 2) husdc pair: husdc-rev1 lib + frozen master bin sources
HS="$OUT/husdc_src"; rm -rf "$HS"; mkdir -p "$HS"
git -C "$REPO" archive claude/husdc-rev1 rust_ingest | tar -x -C "$HS"
cp "$REPO/scripts/build_samples_husdc.rs" "$HS/rust_ingest/src/bin/build_samples.rs"
cp "$REPO/scripts/grid_sim_exitdbg.rs" "$HS/rust_ingest/src/bin/grid_sim_exitdbg.rs"
cd "$HS/rust_ingest"
CARGO_TARGET_DIR="$OUT/husdc_target" cargo build --release --bin build_samples --bin grid_sim_exitdbg

ls -la "$OUT/fb_target/release/feature_builder" \
      "$OUT/husdc_target/release/build_samples" \
      "$OUT/husdc_target/release/grid_sim_exitdbg"
echo "[bins.sh DONE] export FB_BIN=$OUT/fb_target/release/feature_builder BS_BIN=$OUT/husdc_target/release/build_samples GRID_BIN=$OUT/husdc_target/release/grid_sim_exitdbg"
