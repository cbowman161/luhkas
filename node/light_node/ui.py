"""Light-node UI sections."""
from __future__ import annotations


def ui_sections() -> list[str]:
    return [
        """    <div class="card">
      <div class="card-title">Light</div>
      <div class="row"><span class="lbl">Auto low-light</span><button id="btn-camera_light_auto_enabled" onclick="setting('camera_light_auto_enabled')">-</button></div>
      <div class="row"><span class="lbl">Camera light</span><button id="btn-camera_light_enabled" onclick="setting('camera_light_enabled')">-</button></div>
      <div class="srow">
        <div class="slbls"><span>Light brightness</span><span id="val-camera_light_brightness">-</span></div>
        <input type="range" id="sld-camera_light_brightness" min="0" max="255" step="5"
          oninput="sld(this,'camera_light_brightness',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Darkness trigger</span><span id="val-camera_light_trigger_threshold">-</span></div>
        <input type="range" id="sld-camera_light_trigger_threshold" min="10" max="160" step="5"
          oninput="sld(this,'camera_light_trigger_threshold',0,'/settings')">
      </div>
      <div class="row">
        <span class="lbl">Ambient light</span>
        <span id="ambient-light-level" class="val">-</span>
      </div>
    </div>""",
    ]
