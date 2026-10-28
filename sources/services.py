"""Background services shared by the whole app (created once, started by app.py)."""
import logging
import threading

from sources.camera import CameraService
from sources.dashcam import DashcamService
from sources.sim_proto import ProtoDemoSimulator
from sources.sim_real import RealFleetSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

camera = CameraService()
real_sim = RealFleetSimulator()
proto_demo = ProtoDemoSimulator(camera)
dashcam = DashcamService(camera, real_sim)
real_sim.dashcam = dashcam

_started = False
_lock = threading.Lock()


def start():
    global _started
    with _lock:
        if _started:
            return
        _started = True
    for service in (camera, real_sim, proto_demo, dashcam):
        service.start()
