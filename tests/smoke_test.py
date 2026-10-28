"""Smoke test: seeds data, runs the simulators, then calls every page callback in both modes.

    docker compose run --rm dashboard python -m tests.smoke_test
"""
import json
import os
import sys
import tempfile
import time
import traceback
from contextvars import copy_context

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="smoke_")   # own database: the seed below wipes it

import dash
import plotly.graph_objects as go
from dash._callback_context import context_value
from dash._utils import AttributeDict
from plotly.utils import PlotlyJSONEncoder

import config

config.CAMERA_SCAN_SUBNETS = []          # no network scan during tests

import app  # noqa: E402,F401  (builds the Dash app and registers every page)
from core import db, network  # noqa: E402
from pages import camera as cam_page, models, obstacles as ob_page, overview, planner, stats, system, tracking  # noqa: E402
from sources import services  # noqa: E402
from tools import seed  # noqa: E402

failures = []


def check(name, fn, *args, trigger=None):
    def run():
        if trigger:
            context_value.set(AttributeDict(triggered_inputs=[{"prop_id": trigger, "value": 1}]))
        return fn(*args)
    try:
        out = copy_context().run(run)
        skip = type(dash.no_update)
        payload = [None if isinstance(o, skip) else o for o in out] if isinstance(out, tuple) else out
        json.dumps(payload, cls=PlotlyJSONEncoder)  # everything must serialise like Dash does
        for item in out if isinstance(out, tuple) else (out,):
            if isinstance(item, go.Figure):
                item.to_plotly_json()
        print(f"  ok   {name}")
        return out
    except Exception as exc:  # PreventUpdate is fine
        if type(exc).__name__ == "PreventUpdate":
            print(f"  ok   {name} (no update)")
            return None
        failures.append(name)
        print(f"  FAIL {name}: {exc!r}")
        traceback.print_exc()
        return None


def main():
    db.init()
    t0 = time.time()
    print("seed:", seed.seed_all(), f"{time.time() - t0:.1f}s")
    services.real_sim.start()
    services.proto_demo.start()
    time.sleep(6)

    for mode in (config.REAL, config.PROTO):
        print(f"\n== mode {mode}")
        net = network.get_network(mode)
        check("app.live_status", app.live_status, 1, mode)
        check("overview.ov_camera", overview.ov_camera, mode)
        check("overview.ov_live", overview.ov_live, 1, mode)
        check("overview.ov_lists", overview.ov_lists, 1, mode)

        check("tracking.tr_live", tracking.tr_live, 1, mode, None, True)
        data, rows = check("tracking.tr_trips", tracking.tr_trips, 1, mode, None, None) or ([], [])
        assert rows, "the latest finished trip should be pre-selected"
        check("tracking.tr_trips(keep selection)", tracking.tr_trips, 2, mode, rows, data)
        if data:
            store, vmax, _v = check("tracking.tr_load", tracking.tr_load, rows, data, None, mode)
            check("tracking.tr_render", tracking.tr_render, vmax // 2, store)
            check("tracking.tr_render(end)", tracking.tr_render, vmax, store)
            check("tracking.tr_playback(play)", tracking.tr_playback, 1, None, True, 0, vmax, "4",
                  trigger="tr-play.n_clicks")
            check("tracking.tr_playback(step)", tracking.tr_playback, 1, 3, False, 0, vmax, "4",
                  trigger="tr-replay-int.n_intervals")
            check("tracking.tr_export", tracking.tr_export, 1, store)
        check("tracking.tr_select", tracking.tr_select, [1], None,
              trigger='{"id":"CAR-01","type":"tr-car"}.n_clicks')

        opts = check("planner.pl_options", planner.pl_options, mode, 0)
        start, end = opts[2], opts[3]
        for alpha in (0, 1.5, 5):
            check(f"planner.pl_compute(alpha={alpha})", planner.pl_compute, start, end, alpha, True, 0, mode)
        node = next(iter(net.nodes))
        check("planner.pl_click", planner.pl_click, {"points": [{"customdata": node}]}, "start", mode)
        if mode == config.PROTO:
            traj = check("planner.pl_trajectories", planner.pl_trajectories, 1.5, 0, None, mode)
            check("planner.pl_set_active", planner.pl_set_active, 1, [traj[0][0]["id"]])
            nodes, edges, trajs = check("planner.pl_load_track", planner.pl_load_track, mode, 0)
            check("planner.pl_add_rows", planner.pl_add_rows, 1, None, None, nodes, edges, trajs,
                  trigger="pl-add-node.n_clicks")
            check("planner.pl_save(valid)", planner.pl_save, 1, None, nodes, edges, trajs, trigger="pl-save.n_clicks")
            bad = [dict(e) for e in edges] + [{"u": "A", "v": "ZZ", "via": ""}]
            check("planner.pl_save(invalid)", planner.pl_save, 1, None, nodes, bad, trajs, trigger="pl-save.n_clicks")
            check("planner.pl_save(reset)", planner.pl_save, None, 1, nodes, edges, trajs, trigger="pl-reset.n_clicks")
            check("planner.pl_markers", planner.pl_markers, 1)

        check("obstacles.ob_class_options", ob_page.ob_class_options, mode)
        res = check("obstacles.ob_refresh", ob_page.ob_refresh, 1, mode, "review", ["critical", "high", "medium", "low"],
                    [], "7d", 0, None)
        check("obstacles.ob_refresh(all)", ob_page.ob_refresh, 1, mode, "all", ["critical"], [], "all", 0, None)
        if res and res[0]:
            oid = res[0][0]["id"]
            check("obstacles.ob_detail", ob_page.ob_detail, [oid], None, 0, mode)
            check("obstacles.ob_action", ob_page.ob_action, 1, None, None, None, [oid], mode, trigger="ob-ack.n_clicks")
        check("obstacles.ob_buttons", ob_page.ob_buttons, None)

        check("stats.st_live", stats.st_live, 1, mode)
        check("stats.st_history", stats.st_history, 1, mode)
        check("models.md_cards", models.md_cards, mode)
        check("models.md_live", models.md_live, 1, mode)
        check("system.sys_events", system.sys_events, 1, mode, ["info", "warning", "critical"], None)

        check("camera.cam_mode", cam_page.cam_mode, mode, True, True)
        check("camera.cam_live", cam_page.cam_live, 1, mode)
        check("camera.cam_record", cam_page.cam_record, None, 1, mode, trigger="tick.n_intervals")
        check("camera.cam_real_live", cam_page.cam_real_live, 1, mode)

    print("\n== mode-independent")
    check("camera.cam_real_source(invalid)", cam_page.cam_real_source, 1, None, None, None, "not/a/folder",
          trigger="cam-real-use.n_clicks")
    check("camera.cam_connect(invalid)", cam_page.cam_connect, 1, None, None, "999.1.1.1", trigger="cam-connect.n_clicks")
    check("camera.cam_snapshots", cam_page.cam_snapshots, None, 1, trigger="tick-slow.n_intervals")
    check("system.sys_devices", system.sys_devices, 1, "devices")
    weights = check("system.sys_weights", system.sys_weights, "settings")
    check("system.sys_save_weights", system.sys_save_weights, 1, None, weights, trigger="sys-weights-save.n_clicks")
    check("system.sys_recompute", system.sys_save_weights, None, 1, weights, trigger="sys-recompute.n_clicks")
    jpeg, _ = services.camera.latest_jpeg()
    print(f"  ok   camera placeholder jpeg ({len(jpeg)} bytes)" if jpeg[:2] == b"\xff\xd8" else "  FAIL placeholder")

    client = app.server.test_client()
    for url in ("/", "/_dash-layout", "/_dash-dependencies", "/assets/10_theme.css"):
        r = client.get(url)
        print(f"  {'ok  ' if r.status_code == 200 else 'FAIL'} GET {url} -> {r.status_code}")
        if r.status_code != 200:
            failures.append(url)

    print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
