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

CAPTURE_VOLUME="${AUDIO_CAPTURE_VOLUME:-70%}"
ADC_PCM_VOLUME="${AUDIO_ADC_PCM_VOLUME:-70%}"
ALC_FUNCTION="${AUDIO_ALC_FUNCTION:-Stereo}"
ALC_MODE="${AUDIO_ALC_MODE:-Limiter}"
ALC_TARGET="${AUDIO_ALC_TARGET:-8}"
ALC_MAX_GAIN="${AUDIO_ALC_MAX_GAIN:-3}"
ALC_MIN_GAIN="${AUDIO_ALC_MIN_GAIN:-0}"
NOISE_GATE="${AUDIO_NOISE_GATE:-off}"
NOISE_GATE_THRESHOLD="${AUDIO_NOISE_GATE_THRESHOLD:-4}"

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

# Speech recognition is much more sensitive to clipping than playback.
# The old 100%/100% path put both capture stages at +30dB, which is
# enough to flatten loud syllables and make Vosk choose the wrong words.
amixer -c "$CARD" sset Capture "$CAPTURE_VOLUME" >/dev/null
amixer -c "$CARD" sset "ADC PCM" "$ADC_PCM_VOLUME" >/dev/null
amixer -c "$CARD" sset "ALC Function" "$ALC_FUNCTION" >/dev/null
amixer -c "$CARD" sset "ALC Mode" "$ALC_MODE" >/dev/null
amixer -c "$CARD" sset "ALC Target" "$ALC_TARGET" >/dev/null
amixer -c "$CARD" sset "ALC Max Gain" "$ALC_MAX_GAIN" >/dev/null
amixer -c "$CARD" sset "ALC Min Gain" "$ALC_MIN_GAIN" >/dev/null
amixer -c "$CARD" sset "Noise Gate Threshold" "$NOISE_GATE_THRESHOLD" >/dev/null
amixer -c "$CARD" sset "Noise Gate" "$NOISE_GATE" >/dev/null

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
