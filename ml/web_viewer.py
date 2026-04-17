"""
Mobile-friendly Flask web viewer for active learning rounds.

Serves a gallery of segmentation overlays (max-projection, coloured mask on grey base)
and lets the user tap the best result on their phone. The selection is relayed back to
ActiveLearner via a threading.Event / queue mechanism so the main Python loop can block
until the user has chosen.

Usage (internal — called by run_active_learning.py):
    from ml.web_viewer import Viewer
    viewer = Viewer(host="0.0.0.0", port=5050)
    viewer.start()          # launches Flask in a background thread
    print(viewer.url)       # share this with your phone
    winner = viewer.present_round(1, 10, overlays_b64, param_labels)
    viewer.stop()
"""

import logging
import queue
import socket
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML template (single-file, no external CDN dependencies)
# ---------------------------------------------------------------------------

_GALLERY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
  <title>Nucleolus Seg — Round {round_num}/{total_rounds}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111;
      color: #eee;
      min-height: 100vh;
      padding: 12px;
    }}
    h1 {{
      font-size: 1.1rem;
      text-align: center;
      margin-bottom: 6px;
      color: #adf;
    }}
    p.hint {{
      font-size: 0.82rem;
      text-align: center;
      color: #888;
      margin-bottom: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
    }}
    .card {{
      background: #1e1e1e;
      border-radius: 10px;
      overflow: hidden;
      border: 3px solid transparent;
      cursor: pointer;
      transition: border-color 0.15s, transform 0.1s;
    }}
    .card:active {{ transform: scale(0.97); }}
    .card.selected {{ border-color: #4af; }}
    .card img {{
      width: 100%;
      display: block;
    }}
    .card .label {{
      font-size: 0.68rem;
      padding: 6px 8px;
      color: #aaa;
      white-space: pre-wrap;
      word-break: break-all;
    }}
    #confirm-btn {{
      display: block;
      width: 100%;
      margin-top: 16px;
      padding: 14px;
      font-size: 1.1rem;
      font-weight: bold;
      background: #246;
      color: #fff;
      border: none;
      border-radius: 10px;
      cursor: pointer;
    }}
    #confirm-btn:disabled {{
      background: #333;
      color: #666;
      cursor: not-allowed;
    }}
    #stop-btn {{
      display: block;
      width: 100%;
      margin-top: 8px;
      padding: 10px;
      font-size: 0.9rem;
      background: #333;
      color: #aaa;
      border: none;
      border-radius: 10px;
      cursor: pointer;
    }}
    #status {{
      margin-top: 12px;
      text-align: center;
      font-size: 0.85rem;
      color: #6cf;
    }}
  </style>
</head>
<body>
  <h1>Round {round_num} / {total_rounds} — Pick the best segmentation</h1>
  <p class="hint">Tap an image to select it, then press Confirm.</p>
  <div class="grid" id="grid">
    {cards_html}
  </div>
  <button id="confirm-btn" disabled onclick="submitChoice()">Confirm selection</button>
  <button id="stop-btn" onclick="submitStop()">Stop after this round</button>
  <div id="status"></div>

  <script>
    var selected = null;
    function selectCard(idx) {{
      document.querySelectorAll('.card').forEach(function(c) {{
        c.classList.remove('selected');
      }});
      document.getElementById('card-' + idx).classList.add('selected');
      selected = idx;
      document.getElementById('confirm-btn').disabled = false;
    }}
    function submitChoice() {{
      if (selected === null) return;
      document.getElementById('confirm-btn').disabled = true;
      document.getElementById('status').textContent = 'Submitting…';
      fetch('/choose', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{winner: selected, stop: false}})
      }}).then(function(r) {{ return r.json(); }})
        .then(function(d) {{
          document.getElementById('status').textContent = d.message || 'Saved!';
        }});
    }}
    function submitStop() {{
      if (selected === null) {{
        document.getElementById('status').textContent = 'Please select an image first.';
        return;
      }}
      document.getElementById('status').textContent = 'Stopping after this round…';
      fetch('/choose', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{winner: selected, stop: true}})
      }}).then(function(r) {{ return r.json(); }})
        .then(function(d) {{
          document.getElementById('status').textContent = d.message || 'Done.';
        }});
    }}
  </script>
</body>
</html>
"""

_CARD_HTML = """\
<div class="card" id="card-{idx}" onclick="selectCard({idx})">
  <img src="data:image/png;base64,{b64}" alt="Candidate {idx}">
  <div class="label">{label_text}</div>
</div>
"""

_WAITING_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nucleolus Seg — Waiting</title>
  <style>
    body {{ font-family: sans-serif; background: #111; color: #eee;
            display: flex; align-items: center; justify-content: center;
            height: 100vh; flex-direction: column; gap: 16px; }}
    h2 {{ color: #adf; }}
    p  {{ color: #888; font-size: 0.9rem; }}
  </style>
  <meta http-equiv="refresh" content="3">
</head>
<body>
  <h2>Preparing next round…</h2>
  <p>This page will refresh automatically.</p>
</body>
</html>
"""

_DONE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nucleolus Seg — Done</title>
  <style>
    body {{ font-family: sans-serif; background: #111; color: #eee;
            display: flex; align-items: center; justify-content: center;
            height: 100vh; flex-direction: column; gap: 16px; }}
    h2 {{ color: #4f8; }}
    p  {{ color: #888; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h2>All rounds complete!</h2>
  <p>Optimal parameters have been saved to config.yaml.</p>
  <p>You can close this tab.</p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Viewer class
# ---------------------------------------------------------------------------

class Viewer:
    """
    Lightweight Flask-based viewer for mobile preference collection.

    Thread model:
      - Flask runs in a daemon thread (_flask_thread).
      - present_round() blocks the main thread using _choice_queue.get().
      - The browser's POST /choose handler puts the result into the queue.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5050):
        self.host = host
        self.port = port
        self._choice_queue: queue.Queue = queue.Queue()
        self._current_page: Optional[str] = None  # rendered HTML for GET /
        self._done = False
        self.stop_requested = False
        self._flask_thread: Optional[threading.Thread] = None
        self._app = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Flask server in a background daemon thread."""
        try:
            from flask import Flask, jsonify, request, Response
        except ImportError:
            raise ImportError("Flask is required for the web viewer: pip install flask")

        app = Flask(__name__)
        self._app = app

        # Suppress Flask/Werkzeug request logs (keep our own logger clean)
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

        @app.route("/")
        def index():
            if self._done:
                return Response(_DONE_HTML, mimetype="text/html")
            if self._current_page is None:
                return Response(_WAITING_HTML, mimetype="text/html")
            return Response(self._current_page, mimetype="text/html")

        @app.route("/choose", methods=["POST"])
        def choose():
            data = request.get_json(force=True)
            winner = int(data.get("winner", 0))
            stop = bool(data.get("stop", False))
            self._choice_queue.put((winner, stop))
            return jsonify({"message": "Selection recorded! Next round loading…"})

        self._flask_thread = threading.Thread(
            target=lambda: app.run(host=self.host, port=self.port, debug=False, use_reloader=False),
            daemon=True,
        )
        self._flask_thread.start()
        logger.info("Web viewer started at %s", self.url)

    def stop(self) -> None:
        """Signal that all rounds are done."""
        self._done = True
        self._current_page = None

    @property
    def url(self) -> str:
        """Local URL (for display; use LAN IP for phone access)."""
        return f"http://{self.host}:{self.port}"

    @property
    def lan_url(self) -> str:
        """Best-guess LAN URL accessible from a phone on the same WiFi."""
        try:
            lan_ip = socket.gethostbyname(socket.gethostname())
            if lan_ip.startswith("127."):
                # Try harder
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                lan_ip = s.getsockname()[0]
                s.close()
        except Exception:
            lan_ip = "YOUR_LAN_IP"
        return f"http://{lan_ip}:{self.port}"

    # ------------------------------------------------------------------
    # Round interaction
    # ------------------------------------------------------------------

    def present_round(
        self,
        round_num: int,
        total_rounds: int,
        overlays_b64: List[str],
        param_labels: List[Dict[str, Any]],
    ) -> int:
        """
        Render and serve the gallery for this round; block until user selects.

        Returns
        -------
        int
            Index of the winner candidate (0-based).
        """
        cards_html = ""
        for idx, (b64, params) in enumerate(zip(overlays_b64, param_labels)):
            label_text = "\n".join(f"{k}: {v}" for k, v in params.items())
            cards_html += _CARD_HTML.format(idx=idx, b64=b64, label_text=label_text)

        self._current_page = _GALLERY_HTML.format(
            round_num=round_num,
            total_rounds=total_rounds,
            cards_html=cards_html,
        )

        logger.info(
            "Round %d ready — open %s on your phone", round_num, self.lan_url
        )

        # Block until user submits choice
        winner_index, stop = self._choice_queue.get()
        self.stop_requested = stop
        self._current_page = None   # show "preparing next round" page
        return winner_index
