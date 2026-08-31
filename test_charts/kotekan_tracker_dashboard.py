#!/usr/bin/env python3
"""CHARTS Kotekan Beam Tracker Live Plotly / Dash Web UI Dashboard.

Interactive web application for human visualization, live 2D/3D sky mapping,
antenna array health monitoring, and real-time beam steering over Kotekan's REST API.

Run:
    python kotekan_tracker_dashboard.py --port 8050 --rest-port 12048
Then open:
    http://localhost:8050
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dash
from dash import dcc, html, callback, Output, Input, State, ctx
import numpy as np
import plotly.graph_objects as go

# Setup paths
_script_dir = Path(__file__).resolve().parent
_kotekan_root = _script_dir.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from constants import (
    CHARTS_ALTITUDE_M,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_SPACING_M,
)
from kotekan_tracker_control import KotekanTrackerClient
from plot_beam_patterns_and_outputs import (
    compute_array_factor_2d,
    compute_beam_cuts,
    get_antenna_positions,
)

# Catalog of primary astronomical radio sources
ASTRONOMICAL_TARGETS = {
    "Vela Pulsar (PSR J0835-4510)": {"ra": 128.836, "dec": -45.176, "color": "#00FFCC"},
    "Sagittarius A* (Galactic Center)": {"ra": 266.417, "dec": -29.008, "color": "#FF7043"},
    "Centaurus A (NGC 5128)": {"ra": 201.365, "dec": -43.019, "color": "#AB47BC"},
    "Crab Pulsar (PSR J0534+2200)": {"ra": 83.633, "dec": 22.014, "color": "#FFEE58"},
    "Carén Local Zenith Field": {"ra": 24.346, "dec": CHARTS_LATITUDE_DEG, "color": "#42A5F5"},
}

BEAM_COLORS = [
    "#00E676", "#00B0FF", "#FF5252", "#FFD600",
    "#E040FB", "#FF6E40", "#69F0AE", "#40C4FF"
]


def create_app(rest_host: str = "127.0.0.1", rest_port: int = 12048) -> dash.Dash:
    """Builds and configures the Dash application."""
    client = KotekanTrackerClient(host=rest_host, port=rest_port)
    app = dash.Dash(__name__, title="CHARTS Beam Tracker Dashboard")

    app.layout = html.Div(
        style={
            "backgroundColor": "#0D1117",
            "color": "#C9D1D9",
            "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            "minHeight": "100vh",
            "padding": "16px 24px",
        },
        children=[
            # Header Bar
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "borderBottom": "1px solid #30363D",
                    "paddingBottom": "12px",
                    "marginBottom": "16px",
                },
                children=[
                    html.Div([
                        html.H1(
                            "CHARTS Kotekan Beam Tracker Live Control",
                            style={"fontSize": "22px", "fontWeight": "bold", "color": "#FFFFFF", "margin": "0 0 4px 0"},
                        ),
                        html.Div(
                            f"Carén Observatory (Lat {CHARTS_LATITUDE_DEG:.4f}°, Lon {CHARTS_LONGITUDE_DEG:.4f}°, Alt {CHARTS_ALTITUDE_M:.1f} m) | CUDA V5 Engine",
                            style={"fontSize": "13px", "color": "#8B949E"},
                        ),
                    ]),
                    html.Div(id="telemetry-badges", style={"display": "flex", "gap": "10px", "alignItems": "center"}),
                ],
            ),

            # Main Grid Layout: Left Column (Sky Map & Health) + Right Column (Controls)
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1.6fr 1fr", "gap": "20px"},
                children=[
                    # Left Column: Sky Map + Antenna Health Matrix
                    html.Div([
                        # Panel: Sky Map
                        html.Div(
                            style={
                                "backgroundColor": "#161B22",
                                "border": "1px solid #30363D",
                                "borderRadius": "8px",
                                "padding": "14px",
                                "marginBottom": "18px",
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                    children=[
                                        html.H3("Live Topocentric Sky Map & Celestial Footprints", style={"margin": "0", "fontSize": "15px", "color": "#FFFFFF"}),
                                        html.Span("Hemisphere Projection (l, m Space)", style={"fontSize": "12px", "color": "#8B949E"}),
                                    ],
                                ),
                                dcc.Graph(id="live-skymap-graph", config={"displayModeBar": False}, style={"height": "460px"}),
                            ],
                        ),

                        # Panel: Antenna Health Matrix
                        html.Div(
                            style={
                                "backgroundColor": "#161B22",
                                "border": "1px solid #30363D",
                                "borderRadius": "8px",
                                "padding": "14px",
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
                                    children=[
                                        html.H3("Antenna Array Health & Masking Matrix", style={"margin": "0", "fontSize": "15px", "color": "#FFFFFF"}),
                                        html.Div(id="antenna-summary-text", style={"fontSize": "13px", "color": "#58A6FF"}),
                                    ],
                                ),
                                dcc.Graph(id="antenna-grid-graph", config={"displayModeBar": False}, style={"height": "220px"}),
                                html.Div(
                                    style={"display": "flex", "gap": "10px", "marginTop": "10px", "alignItems": "center"},
                                    children=[
                                        html.Span("Toggle Mask ID:", style={"fontSize": "13px"}),
                                        dcc.Input(id="mask-ant-input", type="number", min=0, max=255, value=0, style={"width": "70px", "backgroundColor": "#0D1117", "color": "#FFF", "border": "1px solid #30363D", "padding": "4px 8px", "borderRadius": "4px"}),
                                        html.Button("Disable (Mask)", id="btn-mask-ant", style={"backgroundColor": "#DA3633", "color": "#FFF", "border": "none", "padding": "6px 12px", "borderRadius": "4px", "cursor": "pointer", "fontSize": "12px", "fontWeight": "bold"}),
                                        html.Button("Enable (Unmask)", id="btn-unmask-ant", style={"backgroundColor": "#238636", "color": "#FFF", "border": "none", "padding": "6px 12px", "borderRadius": "4px", "cursor": "pointer", "fontSize": "12px", "fontWeight": "bold"}),
                                        html.Button("⚡ Auto-Mask Dead", id="btn-auto-mask", style={"backgroundColor": "#8957E5", "color": "#FFF", "border": "none", "padding": "6px 12px", "borderRadius": "4px", "cursor": "pointer", "fontSize": "12px", "fontWeight": "bold"}),
                                        html.Span(id="mask-action-feedback", style={"fontSize": "12px", "color": "#3FB950", "marginLeft": "auto"}),
                                    ],
                                ),
                            ],
                        ),
                    ]),

                    # Right Column: Live Steering Controls & Active Beams Table
                    html.Div([
                        # Panel: Dynamic Beam Steering Control
                        html.Div(
                            style={
                                "backgroundColor": "#161B22",
                                "border": "1px solid #30363D",
                                "borderRadius": "8px",
                                "padding": "16px",
                                "marginBottom": "18px",
                            },
                            children=[
                                html.H3("Dynamic Beam Steering Controller", style={"margin": "0 0 12px 0", "fontSize": "15px", "color": "#FFFFFF"}),

                                # Beam Slot Selection
                                html.Div(
                                    style={"marginBottom": "14px"},
                                    children=[
                                        html.Label("Target Beam Slot:", style={"fontSize": "12px", "color": "#8B949E", "display": "block", "marginBottom": "4px"}),
                                        dcc.Dropdown(
                                            id="steer-beam-id",
                                            options=[{"label": f"Beam #{i}", "value": i} for i in range(8)],
                                            value=0,
                                            clearable=False,
                                            style={"backgroundColor": "#0D1117", "color": "#000"},
                                        ),
                                    ],
                                ),

                                # Steering Mode: Direction Cosines vs Celestial
                                dcc.Tabs(
                                    id="steering-mode-tabs",
                                    value="celestial",
                                    style={"height": "34px", "marginBottom": "14px"},
                                    children=[
                                        dcc.Tab(
                                            label="Celestial Target (RA/Dec)",
                                            value="celestial",
                                            style={"padding": "6px", "fontSize": "12px", "backgroundColor": "#0D1117", "color": "#8B949E"},
                                            selected_style={"padding": "6px", "fontSize": "12px", "backgroundColor": "#161B22", "color": "#58A6FF", "borderTop": "2px solid #58A6FF"},
                                        ),
                                        dcc.Tab(
                                            label="Direction Cosines (l, m)",
                                            value="direction",
                                            style={"padding": "6px", "fontSize": "12px", "backgroundColor": "#0D1117", "color": "#8B949E"},
                                            selected_style={"padding": "6px", "fontSize": "12px", "backgroundColor": "#161B22", "color": "#58A6FF", "borderTop": "2px solid #58A6FF"},
                                        ),
                                    ],
                                ),

                                # Celestial Form
                                html.Div(
                                    id="tab-celestial-form",
                                    children=[
                                        html.Label("Astronomical Presets:", style={"fontSize": "12px", "color": "#8B949E", "display": "block", "marginBottom": "4px"}),
                                        dcc.Dropdown(
                                            id="preset-target-dropdown",
                                            options=[{"label": k, "value": k} for k in ASTRONOMICAL_TARGETS.keys()],
                                            value="Vela Pulsar (PSR J0835-4510)",
                                            clearable=False,
                                            style={"backgroundColor": "#0D1117", "color": "#000", "marginBottom": "12px"},
                                        ),
                                        html.Div(
                                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginBottom": "14px"},
                                            children=[
                                                html.Div([
                                                    html.Label("Right Ascension (RA °):", style={"fontSize": "12px", "color": "#8B949E"}),
                                                    dcc.Input(id="input-ra-deg", type="number", value=128.836, step=0.001, style={"width": "100%", "backgroundColor": "#0D1117", "color": "#FFF", "border": "1px solid #30363D", "padding": "6px", "borderRadius": "4px"}),
                                                ]),
                                                html.Div([
                                                    html.Label("Declination (Dec °):", style={"fontSize": "12px", "color": "#8B949E"}),
                                                    dcc.Input(id="input-dec-deg", type="number", value=-45.176, step=0.001, style={"width": "100%", "backgroundColor": "#0D1117", "color": "#FFF", "border": "1px solid #30363D", "padding": "6px", "borderRadius": "4px"}),
                                                ]),
                                            ],
                                        ),
                                    ],
                                ),

                                # Direction Cosines Form
                                html.Div(
                                    id="tab-direction-form",
                                    style={"display": "none"},
                                    children=[
                                        html.Div(
                                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginBottom": "10px"},
                                            children=[
                                                html.Div([
                                                    html.Label("l0 (East):", style={"fontSize": "12px", "color": "#8B949E"}),
                                                    dcc.Input(id="input-l0", type="number", value=0.05, step=0.001, style={"width": "100%", "backgroundColor": "#0D1117", "color": "#FFF", "border": "1px solid #30363D", "padding": "6px", "borderRadius": "4px"}),
                                                ]),
                                                html.Div([
                                                    html.Label("m0 (North):", style={"fontSize": "12px", "color": "#8B949E"}),
                                                    dcc.Input(id="input-m0", type="number", value=-0.02, step=0.001, style={"width": "100%", "backgroundColor": "#0D1117", "color": "#FFF", "border": "1px solid #30363D", "padding": "6px", "borderRadius": "4px"}),
                                                ]),
                                            ],
                                        ),
                                        html.Div(
                                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "marginBottom": "14px"},
                                            children=[
                                                html.Div([
                                                    html.Label("dl/dt rate:", style={"fontSize": "12px", "color": "#8B949E"}),
                                                    dcc.Input(id="input-dl", type="number", value=1e-5, step=1e-6, style={"width": "100%", "backgroundColor": "#0D1117", "color": "#FFF", "border": "1px solid #30363D", "padding": "6px", "borderRadius": "4px"}),
                                                ]),
                                                html.Div([
                                                    html.Label("dm/dt rate:", style={"fontSize": "12px", "color": "#8B949E"}),
                                                    dcc.Input(id="input-dm", type="number", value=0.0, step=1e-6, style={"width": "100%", "backgroundColor": "#0D1117", "color": "#FFF", "border": "1px solid #30363D", "padding": "6px", "borderRadius": "4px"}),
                                                ]),
                                            ],
                                        ),
                                    ],
                                ),

                                # Action Button & Feedback
                                html.Button(
                                    "Steer Beam via Live REST",
                                    id="btn-steer-beam",
                                    style={
                                        "width": "100%",
                                        "backgroundColor": "#238636",
                                        "color": "#FFFFFF",
                                        "border": "none",
                                        "padding": "10px",
                                        "borderRadius": "6px",
                                        "fontWeight": "bold",
                                        "fontSize": "13px",
                                        "cursor": "pointer",
                                    },
                                ),
                                html.Div(id="steer-action-feedback", style={"marginTop": "8px", "fontSize": "12px", "color": "#58A6FF", "textAlign": "center"}),
                            ],
                        ),

                        # Panel: Active Beam Capacity & State Table
                        html.Div(
                            style={
                                "backgroundColor": "#161B22",
                                "border": "1px solid #30363D",
                                "borderRadius": "8px",
                                "padding": "16px",
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
                                    children=[
                                        html.H3("Active Multi-Beam Capacity", style={"margin": "0", "fontSize": "15px", "color": "#FFFFFF"}),
                                        html.Div([
                                            html.Span("Set Beams: ", style={"fontSize": "12px", "color": "#8B949E"}),
                                            dcc.Dropdown(
                                                id="active-beams-dropdown",
                                                options=[{"label": f"{i} Beams", "value": i} for i in range(1, 9)],
                                                value=2,
                                                clearable=False,
                                                style={"display": "inline-block", "width": "100px", "color": "#000"},
                                            ),
                                        ]),
                                    ],
                                ),
                                html.Div(id="active-beams-table", style={"fontSize": "12px"}),
                            ],
                        ),
                    ]),
                ],
            ),

            # Bottom Full-Width Section: Synthesized Beampattern & Sidelobe Hierarchy
            html.Div(
                style={
                    "backgroundColor": "#161B22",
                    "border": "1px solid #30363D",
                    "borderRadius": "8px",
                    "padding": "16px",
                    "marginTop": "20px",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"},
                        children=[
                            html.H3("Live Synthesized Beampattern, Main Lobe & Sidelobe Hierarchy", style={"margin": "0", "fontSize": "16px", "color": "#FFFFFF"}),
                            html.Span("Physical Array Factor & Sidelobe Profiles | -3 dB HPBW & -13 dB Levels", style={"fontSize": "12px", "color": "#8B949E"}),
                        ],
                    ),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "1.2fr 1fr", "gap": "16px"},
                        children=[
                            dcc.Graph(id="live-beampattern-2d", config={"displayModeBar": False}, style={"height": "360px"}),
                            dcc.Graph(id="live-beampattern-cuts", config={"displayModeBar": False}, style={"height": "360px"}),
                        ],
                    ),
                ],
            ),

            # Auto Refresh Timer (1 Second interval)
            dcc.Interval(id="telemetry-interval", interval=1000, n_intervals=0),
        ],
    )

    # ------------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------------

    @app.callback(
        Output("input-ra-deg", "value"),
        Output("input-dec-deg", "value"),
        Input("preset-target-dropdown", "value"),
    )
    def update_preset_fields(preset_name):
        if preset_name in ASTRONOMICAL_TARGETS:
            t = ASTRONOMICAL_TARGETS[preset_name]
            return t["ra"], t["dec"]
        return 0.0, 0.0

    @app.callback(
        Output("tab-celestial-form", "style"),
        Output("tab-direction-form", "style"),
        Input("steering-mode-tabs", "value"),
    )
    def toggle_form_visibility(tab_val):
        if tab_val == "celestial":
            return {"display": "block"}, {"display": "none"}
        return {"display": "none"}, {"display": "block"}

    @app.callback(
        Output("steer-action-feedback", "children"),
        Input("btn-steer-beam", "n_clicks"),
        State("steering-mode-tabs", "value"),
        State("steer-beam-id", "value"),
        State("input-ra-deg", "value"),
        State("input-dec-deg", "value"),
        State("input-l0", "value"),
        State("input-m0", "value"),
        State("input-dl", "value"),
        State("input-dm", "value"),
        prevent_initial_call=True,
    )
    def handle_beam_steer(n_clicks, mode, beam_id, ra, dec, l0, m0, dl, dm):
        if not n_clicks:
            return ""
        try:
            if mode == "celestial":
                # Compute current LST
                now_s = time.time()
                lst_hours = (now_s / 3600.0 * 1.0027379 + (CHARTS_LONGITUDE_DEG / 15.0)) % 24.0
                msg = client.steer_radec(int(beam_id), float(ra), float(dec), float(lst_hours))
                return f"[OK] Beam #{beam_id} steered to RA {ra:.3f}°, Dec {dec:.3f}°"
            else:
                msg = client.steer_lm(int(beam_id), float(l0), float(m0), float(dl), float(dm))
                return f"[OK] Beam #{beam_id} steered to l0={l0:.4f}, m0={m0:.4f}"
        except Exception as e:
            return f"[ERROR] Failed to steer beam: {e}"

    @app.callback(
        Output("mask-action-feedback", "children"),
        Input("btn-mask-ant", "n_clicks"),
        Input("btn-unmask-ant", "n_clicks"),
        Input("btn-auto-mask", "n_clicks"),
        State("mask-ant-input", "value"),
        prevent_initial_call=True,
    )
    def handle_antenna_masking(mask_clicks, unmask_clicks, auto_clicks, ant_id):
        triggered_id = ctx.triggered_id
        if not triggered_id:
            return ""
        try:
            if triggered_id == "btn-auto-mask":
                h5_file = "/home/fernando/charts/data/260816T013722Z_CHARTS_hdf5/baseband_virtual.h5"
                msg = client.auto_mask(h5_path=h5_file)
                return "⚡ Auto-Mask applied (dead lines isolated)"
            if ant_id is None:
                return ""
            enable = (triggered_id == "btn-unmask-ant")
            msg = client.mask_antenna(int(ant_id), enabled=enable)
            state_str = "UNMASKED (ACTIVE)" if enable else "MASKED (DEAD)"
            return f"Antenna #{ant_id} is now {state_str}"
        except Exception as e:
            return f"[ERROR] {e}"

    @app.callback(
        Output("active-beams-dropdown", "value"),
        Input("active-beams-dropdown", "value"),
        prevent_initial_call=True,
    )
    def handle_active_beam_change(new_count):
        if new_count:
            try:
                client.enable_beams(int(new_count))
            except Exception:
                pass
        return new_count

    @app.callback(
        Output("telemetry-badges", "children"),
        Output("live-skymap-graph", "figure"),
        Output("antenna-grid-graph", "figure"),
        Output("antenna-summary-text", "children"),
        Output("active-beams-table", "children"),
        Output("live-beampattern-2d", "figure"),
        Output("live-beampattern-cuts", "figure"),
        Input("telemetry-interval", "n_intervals"),
    )
    def update_live_telemetry(n_intervals):
        # Default state
        t_start = time.perf_counter()
        is_connected = False
        status = {}
        latency_ms = 0.0

        try:
            status = client.get_status()
            is_connected = True
            latency_ms = (time.perf_counter() - t_start) * 1000.0
        except Exception:
            is_connected = False

        # 1. Telemetry Badges
        conn_badge = html.Span(
            f"● REST ONLINE ({latency_ms:.1f} ms)" if is_connected else "● REST OFFLINE",
            style={
                "backgroundColor": "#238636" if is_connected else "#DA3633",
                "color": "#FFF",
                "padding": "4px 10px",
                "borderRadius": "12px",
                "fontSize": "12px",
                "fontWeight": "bold",
            },
        )
        active_beams_count = status.get("num_active_beams", 0)
        total_elements = status.get("total_elements", 64)
        active_antennas = status.get("active_antennas", total_elements)
        masked_antennas = status.get("masked_antennas", 0)

        badges = [
            conn_badge,
            html.Span(f"Beams: {active_beams_count}/8 Active", style={"backgroundColor": "#1F6FEB", "color": "#FFF", "padding": "4px 10px", "borderRadius": "12px", "fontSize": "12px"}),
            html.Span(f"Antennas: {active_antennas}/{total_elements} Alive", style={"backgroundColor": "#30363D", "color": "#C9D1D9", "padding": "4px 10px", "borderRadius": "12px", "fontSize": "12px"}),
        ]

        # 2. Live Topocentric Sky Map Figure
        fig_sky = go.Figure()
        theta = np.linspace(0, 2 * np.pi, 200)
        fig_sky.add_trace(go.Scatter(x=np.sin(theta), y=np.cos(theta), mode="lines", line=dict(color="#8B949E", dash="dash", width=1.5), name="Horizon (0°)", hoverinfo="none"))
        fig_sky.add_trace(go.Scatter(x=np.cos(np.radians(30)) * np.sin(theta), y=np.cos(np.radians(30)) * np.cos(theta), mode="lines", line=dict(color="#30363D", dash="dot", width=1), name="El 30°", hoverinfo="none"))
        fig_sky.add_trace(go.Scatter(x=np.cos(np.radians(60)) * np.sin(theta), y=np.cos(np.radians(60)) * np.cos(theta), mode="lines", line=dict(color="#30363D", dash="dot", width=1), name="El 60°", hoverinfo="none"))
        fig_sky.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", marker=dict(color="#FFEE58", symbol="cross", size=12), text=["Zenith"], textposition="top center", name="Zenith (n=1.0)"))

        # Astronomical Source Overlays
        now_s = time.time()
        lst_deg = ((now_s / 3600.0 * 1.0027379 + (CHARTS_LONGITUDE_DEG / 15.0)) * 15.0) % 360.0
        for name, data in ASTRONOMICAL_TARGETS.items():
            ha_rad = math.radians(lst_deg - data["ra"])
            dec_rad = math.radians(data["dec"])
            lat_rad = math.radians(CHARTS_LATITUDE_DEG)
            l = -math.cos(dec_rad) * math.sin(ha_rad)
            m = math.sin(dec_rad) * math.cos(lat_rad) - math.cos(dec_rad) * math.sin(lat_rad) * math.cos(ha_rad)
            n = math.sin(dec_rad) * math.sin(lat_rad) + math.cos(dec_rad) * math.cos(lat_rad) * math.cos(ha_rad)
            if n > 0:
                short_n = name.split("(")[0].strip()
                fig_sky.add_trace(go.Scatter(x=[l], y=[m], mode="markers+text", marker=dict(color=data["color"], symbol="circle-open", size=10, line=dict(width=2)), text=[short_n], textposition="bottom center", name=short_n))

        # Active Beams
        trajectories = status.get("trajectories", [])
        for b_idx in range(min(active_beams_count, len(trajectories))):
            b = trajectories[b_idx]
            l0 = b.get("l0", b.get("source_l0", 0.0))
            m0 = b.get("m0", b.get("source_m0", 0.0))
            color = BEAM_COLORS[b_idx % len(BEAM_COLORS)]
            fig_sky.add_trace(go.Scatter(
                x=[l0], y=[m0], mode="markers+text",
                marker=dict(color=color, size=16, line=dict(color="#FFF", width=1.5)),
                text=[f"B#{b_idx}"], textposition="middle right",
                name=f"Beam #{b_idx} (l={l0:.2f}, m={m0:.2f})"
            ))

        fig_sky.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            margin=dict(l=10, r=10, t=20, b=20),
            xaxis=dict(title="East-West Direction Cosine (l)", range=[-1.15, 1.15], zeroline=True, zerolinecolor="#30363D", gridcolor="#21262D"),
            yaxis=dict(title="North-South Direction Cosine (m)", range=[-1.15, 1.15], zeroline=True, zerolinecolor="#30363D", gridcolor="#21262D", scaleanchor="x", scaleratio=1),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=10)),
        )

        # 3. Antenna Health Matrix Figure (8x8 Grid)
        grid_dim = int(math.ceil(math.sqrt(total_elements)))
        grid_matrix = np.ones((grid_dim, grid_dim))
        bad_elements = set(status.get("bad_elements", []))

        custom_text = []
        for r in range(grid_dim):
            row_text = []
            for c in range(grid_dim):
                elem_id = r * grid_dim + c
                if elem_id < total_elements:
                    if elem_id in bad_elements:
                        grid_matrix[r, c] = 0
                        row_text.append(f"Ant #{elem_id}: MASKED (DEAD)")
                    else:
                        grid_matrix[r, c] = 1
                        row_text.append(f"Ant #{elem_id}: ACTIVE (ALIVE)")
                else:
                    grid_matrix[r, c] = 0.5
                    row_text.append("N/A")
            custom_text.append(row_text)

        fig_ant = go.Figure(data=go.Heatmap(
            z=grid_matrix,
            text=custom_text,
            hoverinfo="text",
            colorscale=[[0.0, "#DA3633"], [0.5, "#30363D"], [1.0, "#238636"]],
            showscale=False,
        ))
        fig_ant.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False),
        )

        ant_summary = f"{active_antennas} Connected | {masked_antennas} Masked"

        # 4. Active Beams Summary Table
        table_rows = []
        for b_idx in range(len(trajectories)):
            b = trajectories[b_idx]
            is_active = (b_idx < active_beams_count)
            l0 = b.get("l0", b.get("source_l0", 0.0))
            m0 = b.get("m0", b.get("source_m0", 0.0))
            n0 = b.get("n0", b.get("source_n0", 1.0))
            cel = b.get("celestial_target", {})
            target_str = f"RA {cel.get('ra_deg', 0.0):.2f}°, Dec {cel.get('dec_deg', 0.0):.2f}°" if cel.get("is_set", False) else "Manual (l, m)"

            status_tag = html.Span("ACTIVE", style={"color": "#3FB950", "fontWeight": "bold"}) if is_active else html.Span("IDLE", style={"color": "#8B949E"})
            row = html.Tr([
                html.Td(f"#{b_idx}", style={"padding": "6px 8px", "fontWeight": "bold"}),
                html.Td(status_tag, style={"padding": "6px 8px"}),
                html.Td(f"{l0:+.4f}", style={"padding": "6px 8px", "fontFamily": "monospace"}),
                html.Td(f"{m0:+.4f}", style={"padding": "6px 8px", "fontFamily": "monospace"}),
                html.Td(f"{n0:+.4f}", style={"padding": "6px 8px", "fontFamily": "monospace"}),
                html.Td(target_str, style={"padding": "6px 8px", "color": "#58A6FF"}),
            ], style={"borderBottom": "1px solid #21262D"})
            table_rows.append(row)

        table = html.Table(
            style={"width": "100%", "borderCollapse": "collapse", "marginTop": "6px"},
            children=[
                html.Thead(html.Tr([
                    html.Th("Beam", style={"textAlign": "left", "padding": "6px 8px", "color": "#8B949E"}),
                    html.Th("State", style={"textAlign": "left", "padding": "6px 8px", "color": "#8B949E"}),
                    html.Th("l0 (East)", style={"textAlign": "left", "padding": "6px 8px", "color": "#8B949E"}),
                    html.Th("m0 (North)", style={"textAlign": "left", "padding": "6px 8px", "color": "#8B949E"}),
                    html.Th("n0 (Zenith)", style={"textAlign": "left", "padding": "6px 8px", "color": "#8B949E"}),
                    html.Th("Celestial Target", style={"textAlign": "left", "padding": "6px 8px", "color": "#8B949E"}),
                ], style={"borderBottom": "1px solid #30363D"})),
                html.Tbody(table_rows),
            ],
        )

        # 5. Live 2D Beampattern Contour & Sidelobes
        mask_arr = np.ones(total_elements, dtype=np.uint8)
        for bad_id in bad_elements:
            if 0 <= bad_id < total_elements:
                mask_arr[bad_id] = 0

        target_b0 = trajectories[0] if trajectories else {}
        b0_l0 = target_b0.get("l0", target_b0.get("source_l0", 0.05))
        b0_m0 = target_b0.get("m0", target_b0.get("source_m0", -0.02))

        ant_pos = get_antenna_positions(total_elements, DEFAULT_SPACING_M)
        L_g, M_g, P_dB = compute_array_factor_2d(ant_pos, mask_arr, 400.0 * 1e6, b0_l0, b0_m0, grid_res=120)

        fig_beam_2d = go.Figure(data=go.Contour(
            x=L_g[0, :], y=M_g[:, 0], z=P_dB,
            colorscale="Plasma",
            contours=dict(start=-35, end=0, size=2.5, showlines=True),
            colorbar=dict(title="dB", thickness=12, len=0.9),
            hoverinfo="x+y+z",
        ))
        fig_beam_2d.add_trace(go.Scatter(
            x=[b0_l0], y=[b0_m0], mode="markers",
            marker=dict(symbol="cross", size=12, color="#00FFCC", line=dict(width=2)),
            name="Main Beam Peak", hoverinfo="name"
        ))
        fig_beam_2d.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            margin=dict(l=10, r=10, t=20, b=20),
            xaxis=dict(title="Direction Cosine l (East)", range=[-1.05, 1.05], zeroline=False),
            yaxis=dict(title="Direction Cosine m (North)", range=[-1.05, 1.05], zeroline=False, scaleanchor="x", scaleratio=1),
            showlegend=False,
        )

        # 6. Sidelobe Cross-Section Cuts
        l_c, P_ew, m_c, P_ns = compute_beam_cuts(ant_pos, mask_arr, 400.0 * 1e6, b0_l0, b0_m0, n_pts=200)
        fig_cuts = go.Figure()
        fig_cuts.add_trace(go.Scatter(x=l_c, y=P_ew, mode="lines", line=dict(color="#00FFCC", width=2), name="East-West Cut (l)"))
        fig_cuts.add_trace(go.Scatter(x=m_c, y=P_ns, mode="lines", line=dict(color="#FF7043", width=2, dash="dot"), name="North-South Cut (m)"))
        fig_cuts.add_trace(go.Scatter(x=[-1.0, 1.0], y=[-3.0, -3.0], mode="lines", line=dict(color="#FFEE58", dash="dash", width=1), name="HPBW (-3 dB)"))
        fig_cuts.add_trace(go.Scatter(x=[-1.0, 1.0], y=[-13.26, -13.26], mode="lines", line=dict(color="#FF5252", dash="dash", width=1), name="First Sidelobe (-13.2 dB)"))

        fig_cuts.update_layout(
            template="plotly_dark",
            paper_bgcolor="#161B22",
            plot_bgcolor="#161B22",
            margin=dict(l=10, r=10, t=20, b=20),
            xaxis=dict(title="Direction Cosine Displacement", range=[-0.8, 0.8], gridcolor="#21262D"),
            yaxis=dict(title="Normalized Power (dB)", range=[-38, 2], gridcolor="#21262D"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        )

        return badges, fig_sky, fig_ant, ant_summary, table, fig_beam_2d, fig_cuts

    return app


def main():
    parser = argparse.ArgumentParser(description="CHARTS Kotekan Beam Tracker Live Plotly UI Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web dashboard host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Web dashboard port (default: 8050)")
    parser.add_argument("--rest-host", type=str, default="127.0.0.1", help="Kotekan REST host (default: 127.0.0.1)")
    parser.add_argument("--rest-port", type=int, default=12048, help="Kotekan REST port (default: 12048)")
    args = parser.parse_args()

    app = create_app(rest_host=args.rest_host, rest_port=args.rest_port)
    print("=" * 80)
    print(" CHARTS KOTEKAN BEAM TRACKER PLOTLY DASHBOARD")
    print("=" * 80)
    print(f" -> Web Dashboard URL : http://{args.host}:{args.port}")
    print(f" -> Kotekan REST API  : http://{args.rest_host}:{args.rest_port}")
    print("=" * 80)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
