from pathlib import Path
from arbor_monitor.webui.launcher import start_session_webui
import threading
s=Path(__file__).resolve().parents[1]/".arbor/sessions/house_prices_keyless_20261013"
v=start_session_webui(s,preferred=8765)
if v is None:raise RuntimeError("No available local port")
print(v.url,flush=True)
threading.Event().wait()
