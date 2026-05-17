"""Rover-node UI sections."""
from __future__ import annotations


def ui_sections() -> list[str]:
    return [
        """    <div class="card">
      <div class="card-title">Controller</div>
      <div class="row"><span class="lbl">Manual control</span><button id="btn-manual_controller_enabled" onclick="setting('manual_controller_enabled')">-</button></div>
      <div class="row"><span class="lbl">USB gamepad</span><span id="gamepad-status" class="val">not connected</span></div>
      <div class="row"><span class="lbl">Last action</span><span id="gamepad-action" class="val">-</span></div>
    </div>""",
        """    <div class="card">
      <div class="card-title">Follow Tuning</div>
      <div class="row"><span class="lbl">Follow wheels</span><button id="btn-follow_enabled" onclick="tog('follow_enabled','/tracking','follow')">-</button></div>
      <div class="row"><span class="lbl">Wheel drive</span><button id="btn-wheel_enabled" onclick="setting('wheel_enabled')">-</button></div>
      <div class="srow">
        <div class="slbls"><span>Forward speed</span><span id="val-follow_forward_speed">-</span></div>
        <input type="range" id="sld-follow_forward_speed" min="100" max="800" step="50"
          oninput="sld(this,'follow_forward_speed',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Steer gain</span><span id="val-follow_steer_gain">-</span></div>
        <input type="range" id="sld-follow_steer_gain" min="0.5" max="10.0" step="0.5"
          oninput="sld(this,'follow_steer_gain',1,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Follow bbox ratio</span><span id="val-follow_target_bbox_ratio">-</span></div>
        <input type="range" id="sld-follow_target_bbox_ratio" min="0.10" max="0.60" step="0.02"
          oninput="sld(this,'follow_target_bbox_ratio',2,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Close bbox ratio</span><span id="val-close_target_bbox_ratio">-</span></div>
        <input type="range" id="sld-close_target_bbox_ratio" min="0.30" max="0.90" step="0.05"
          oninput="sld(this,'close_target_bbox_ratio',2,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Follow deadzone</span><span id="val-follow_deadzone_ratio">-</span></div>
        <input type="range" id="sld-follow_deadzone_ratio" min="0.01" max="0.20" step="0.01"
          oninput="sld(this,'follow_deadzone_ratio',2,'/settings')">
      </div>
    </div>""",
        """    <div class="card">
      <div class="card-title">Collision Avoidance</div>
      <div class="row">
        <span class="lbl">Enabled</span>
        <button id="btn-collision_avoidance_enabled" onclick="tog('collision_avoidance_enabled','/collision','enabled')">-</button>
      </div>
      <div class="row"><span class="lbl">Status</span><span id="collision-badge" class="badge clear">CLEAR</span></div>
      <div class="srow">
        <div class="slbls"><span>Height threshold</span><span id="val-collision_height_threshold">-</span></div>
        <input type="range" id="sld-collision_height_threshold" min="0.10" max="0.80" step="0.05"
          oninput="sld(this,'collision_height_threshold',2,'/collision','height_threshold')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Center zone</span><span id="val-collision_center_zone_fraction">-</span></div>
        <input type="range" id="sld-collision_center_zone_fraction" min="0.30" max="1.00" step="0.05"
          oninput="sld(this,'collision_center_zone_fraction',2,'/collision','center_zone_fraction')">
      </div>
    </div>""",
    ]
