#!/usr/bin/env bash
# Configure the RaspAudio MIC ULTRA 3 / WM8960 HAT for LUHKAS audio_node.
#
# This is intentionally safe to run on every audio_node start. It only touches
# the WM8960 card after the startup script has detected that card in ALSA.
set -euo pipefail

CARD="${AUDIO_WM8960_CARD:-}"
if [ -z "$CARD" ]; then
  CARD="$(awk '
    /^[[:space:]]*[0-9]+[[:space:]]+\[/ && ($0 ~ /wm8960soundcard|wm8960-soundcard/) {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
      print $1
      exit
    }
  ' /proc/asound/cards)"
fi

if [ -z "$CARD" ]; then
  echo "[audio_node] RaspAudio MIC ULTRA 3 / WM8960 card not found; skipping hardware config" >&2
  exit 0
fi

echo "[audio_node] configuring RaspAudio MIC ULTRA 3 / WM8960 card ${CARD}"

# Onboard microphones are wired through LINPUT1/RINPUT1 on the MIC ULTRA 3.
amixer -c "$CARD" sset "Left Boost Mixer LINPUT1" on >/dev/null
amixer -c "$CARD" sset "Left Boost Mixer LINPUT2" off >/dev/null
amixer -c "$CARD" sset "Left Boost Mixer LINPUT3" off >/dev/null
amixer -c "$CARD" sset "Left Input Boost Mixer LINPUT1" 0 >/dev/null
amixer -c "$CARD" sset "Left Input Boost Mixer LINPUT2" 0 >/dev/null
amixer -c "$CARD" sset "Left Input Boost Mixer LINPUT3" 0 >/dev/null
amixer -c "$CARD" sset "Left Input Mixer Boost" on >/dev/null

amixer -c "$CARD" sset "Right Boost Mixer RINPUT1" on >/dev/null
amixer -c "$CARD" sset "Right Boost Mixer RINPUT2" off >/dev/null
amixer -c "$CARD" sset "Right Boost Mixer RINPUT3" off >/dev/null
amixer -c "$CARD" sset "Right Input Boost Mixer RINPUT1" 0 >/dev/null
amixer -c "$CARD" sset "Right Input Boost Mixer RINPUT2" 0 >/dev/null
amixer -c "$CARD" sset "Right Input Boost Mixer RINPUT3" 0 >/dev/null
amixer -c "$CARD" sset "Right Input Mixer Boost" on >/dev/null

# 0 = Left ADC on left channel, right ADC on right channel.
amixer -c "$CARD" cset numid=41 0 >/dev/null
amixer -c "$CARD" sset "ADC High Pass Filter" on >/dev/null

# Requested final capture gain.
amixer -c "$CARD" sset Capture 100% >/dev/null
amixer -c "$CARD" sset "ADC PCM" 100% >/dev/null
amixer -c "$CARD" sset "ALC Function" Off >/dev/null
amixer -c "$CARD" sset "Noise Gate" off >/dev/null

# Speaker path.
amixer -c "$CARD" sset Playback 100% >/dev/null
amixer -c "$CARD" sset Speaker 100% >/dev/null
amixer -c "$CARD" sset "Left Output Mixer PCM" on >/dev/null
amixer -c "$CARD" sset "Right Output Mixer PCM" on >/dev/null
amixer -c "$CARD" sset "PCM Playback -6dB" off >/dev/null

if command -v sudo >/dev/null 2>&1; then
  sudo alsactl store "$CARD" >/dev/null || true
else
  alsactl store "$CARD" >/dev/null || true
fi
