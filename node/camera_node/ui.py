"""Camera-node UI sections."""
from __future__ import annotations


def ui_sections() -> list[str]:
    return [
        """    <div class="card">
      <div class="card-title">Behavior</div>
      <div class="row">
        <span class="lbl">Guard mode</span>
        <button id="btn-guard_mode" onclick="tog('guard_mode','/guard','enabled')">-</button>
      </div>
      <div class="row">
        <span class="lbl">State</span>
        <span id="bhv-badge" class="bhv bhv-idle">IDLE</span>
      </div>
      <div class="row">
        <span class="lbl">Time in state</span>
        <span id="bhv-time" class="val">-</span>
      </div>
      <div class="row" id="bhv-alerts-row" style="display:none">
        <span class="lbl">Guard alerts</span>
        <span id="bhv-alerts" class="val">0</span>
      </div>
    </div>""",
        """    <div class="card">
      <div class="card-title">Face</div>
      <div class="row"><span class="lbl">Detection</span><button id="btn-face_detection_enabled" onclick="setting('face_detection_enabled')">-</button></div>
      <div class="row"><span class="lbl">Recognition</span><button id="btn-face_recognition_enabled" onclick="setting('face_recognition_enabled')">-</button></div>
      <div class="row"><span class="lbl">Auto-capture refs</span><button id="btn-auto_reference_capture_enabled" onclick="setting('auto_reference_capture_enabled')">-</button></div>
      <div class="srow">
        <div class="slbls"><span>Auto-capture min conf</span><span id="val-auto_reference_min_confidence">-</span></div>
        <input type="range" id="sld-auto_reference_min_confidence" min="0.10" max="0.80" step="0.05"
          oninput="sld(this,'auto_reference_min_confidence',2,'/settings')">
      </div>
    </div>""",
        """    <div class="card">
      <div class="card-title">Vision</div>
      <div class="row">
        <span class="lbl">Pose estimation</span>
        <button id="btn-pose_enabled" onclick="setting('pose_enabled')">-</button>
      </div>
      <div class="row">
        <span class="lbl">Filter ghosts by pose</span>
        <button id="btn-pose_filter_persons" onclick="setting('pose_filter_persons')">-</button>
      </div>
      <div class="srow">
        <div class="slbls"><span>Pose interval (frames)</span><span id="val-pose_interval_frames">-</span></div>
        <input type="range" id="sld-pose_interval_frames" min="1" max="10" step="1"
          oninput="sld(this,'pose_interval_frames',0,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>Pose score threshold</span><span id="val-pose_score_threshold">-</span></div>
        <input type="range" id="sld-pose_score_threshold" min="0.10" max="0.90" step="0.05"
          oninput="sld(this,'pose_score_threshold',2,'/settings')">
      </div>
      <div class="srow">
        <div class="slbls"><span>JPEG quality</span><span id="val-jpeg_quality">-</span></div>
        <input type="range" id="sld-jpeg_quality" min="20" max="95" step="5"
          oninput="sld(this,'jpeg_quality',0,'/settings')">
      </div>
    </div>""",
        """    <div class="card">
      <div class="card-title">Target</div>
      <div class="row"><span class="lbl">State</span><span id="tgt-state" class="val">-</span></div>
      <div class="row"><span class="lbl">Identity</span><span id="tgt-identity" class="val">-</span></div>
      <div class="row"><span class="lbl">ID</span><span id="tgt-id" class="val">-</span></div>
      <div class="row"><span class="lbl">Faces</span><span id="face-queue" class="val">-</span></div>
      <div class="row"><span class="lbl">Asking</span><span id="face-asking" class="val">-</span></div>
    </div>""",
        """    <div class="card">
      <div class="card-title">Detections <span id="det-count" style="color:#555;font-weight:400"></span></div>
      <div id="det-list"><span class="none">waiting...</span></div>
    </div>""",
    ]
