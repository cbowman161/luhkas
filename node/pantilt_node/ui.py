"""Pan/tilt-node UI sections."""
from __future__ import annotations


def ui_sections() -> list[str]:
    return [
        """    <div class="card" id="manual-card">
      <div class="card-title">Camera Control</div>
      <div class="dpad">
        <button class="dp dp-up" id="dp-up"
          onmousedown="startPT(0,1)" onmouseup="stopPT()" onmouseleave="stopPT()"
          ontouchstart="startPT(0,1);event.preventDefault()" ontouchend="stopPT()">▲</button>
        <button class="dp dp-left" id="dp-left"
          onmousedown="startPT(-1,0)" onmouseup="stopPT()" onmouseleave="stopPT()"
          ontouchstart="startPT(-1,0);event.preventDefault()" ontouchend="stopPT()">◀</button>
        <button class="dp dp-ctr" onclick="centerCamera()">⌖</button>
        <button class="dp dp-right" id="dp-right"
          onmousedown="startPT(1,0)" onmouseup="stopPT()" onmouseleave="stopPT()"
          ontouchstart="startPT(1,0);event.preventDefault()" ontouchend="stopPT()">▶</button>
        <button class="dp dp-dn" id="dp-dn"
          onmousedown="startPT(0,-1)" onmouseup="stopPT()" onmouseleave="stopPT()"
          ontouchstart="startPT(0,-1);event.preventDefault()" ontouchend="stopPT()">▼</button>
      </div>
      <div style="text-align:center;font-size:.65rem;color:#444;margin-top:2px">Arrow keys · ⌖ to center</div>
      <div class="srow" style="margin-top:4px">
        <div class="slbls"><span>Pan step</span><span id="pan-step-val">5</span></div>
        <input type="range" id="pan-step-sld" min="1" max="200" step="1" value="5"
          oninput="PAN_STEP=parseInt(this.value);q('pan-step-val').textContent=this.value">
      </div>
      <div class="srow">
        <div class="slbls"><span>Tilt step</span><span id="tilt-step-val">5</span></div>
        <input type="range" id="tilt-step-sld" min="1" max="30" step="1" value="5"
          oninput="TILT_STEP=parseInt(this.value);q('tilt-step-val').textContent=this.value">
      </div>
    </div>""",
        """    <div class="card">
      <div class="card-title">Pan-Tilt</div>
      <div class="row"><span class="lbl">Search camera</span><button id="btn-search_movement_enabled" onclick="setting('search_movement_enabled')">-</button></div>
      <div class="row"><span class="lbl">Edge reacquire</span><button id="btn-edge_reacquire_enabled" onclick="setting('edge_reacquire_enabled')">-</button></div>
      <div class="srow">
        <div class="slbls"><span>Max command</span><span id="val-max_command">-</span></div>
        <input type="range" id="sld-max_command" min="10" max="300" step="5"
          oninput="sld(this,'max_command',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Min command</span><span id="val-min_command">-</span></div>
        <input type="range" id="sld-min_command" min="1" max="30" step="1"
          oninput="sld(this,'min_command',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Max step (ramp rate)</span><span id="val-max_command_step">-</span></div>
        <input type="range" id="sld-max_command_step" min="1" max="100" step="1"
          oninput="sld(this,'max_command_step',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Command interval (s)</span><span id="val-command_interval_seconds">-</span></div>
        <input type="range" id="sld-command_interval_seconds" min="0.05" max="1.0" step="0.05"
          oninput="sld(this,'command_interval_seconds',2,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Settle enter (°)</span><span id="val-settle_enter_degrees">-</span></div>
        <input type="range" id="sld-settle_enter_degrees" min="0.5" max="20.0" step="0.5"
          oninput="sld(this,'settle_enter_degrees',1,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Settle exit (°)</span><span id="val-settle_exit_degrees">-</span></div>
        <input type="range" id="sld-settle_exit_degrees" min="0.5" max="25.0" step="0.5"
          oninput="sld(this,'settle_exit_degrees',1,'/settings')">
      </div>
      <hr class="divider">
      <div class="srow">
        <div class="slbls"><span>Pan min (°)</span><span id="val-estimated_pan_min">-</span></div>
        <input type="range" id="sld-estimated_pan_min" min="-180" max="0" step="5"
          oninput="sld(this,'estimated_pan_min',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Pan max (°)</span><span id="val-estimated_pan_max">-</span></div>
        <input type="range" id="sld-estimated_pan_max" min="0" max="180" step="5"
          oninput="sld(this,'estimated_pan_max',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Tilt min (°)</span><span id="val-estimated_tilt_min">-</span></div>
        <input type="range" id="sld-estimated_tilt_min" min="-90" max="0" step="5"
          oninput="sld(this,'estimated_tilt_min',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Tilt max (°)</span><span id="val-estimated_tilt_max">-</span></div>
        <input type="range" id="sld-estimated_tilt_max" min="0" max="90" step="5"
          oninput="sld(this,'estimated_tilt_max',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Pan scale</span><span id="val-pan_estimate_scale">-</span></div>
        <input type="range" id="sld-pan_estimate_scale" min="0.1" max="5.0" step="0.1"
          oninput="sld(this,'pan_estimate_scale',1,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Tilt scale</span><span id="val-tilt_estimate_scale">-</span></div>
        <input type="range" id="sld-tilt_estimate_scale" min="0.1" max="5.0" step="0.1"
          oninput="sld(this,'tilt_estimate_scale',1,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Pan limit margin (°)</span><span id="val-pan_limit_margin">-</span></div>
        <input type="range" id="sld-pan_limit_margin" min="0" max="60" step="5"
          oninput="sld(this,'pan_limit_margin',0,'/settings')">
      </div>
    </div>""",
    ]
