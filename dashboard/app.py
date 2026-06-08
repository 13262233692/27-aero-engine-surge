import gc
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import threading
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from buffer.lockfree_ring import LockFreeRingBuffer
from processing.cwt_analyzer import CWTAnalyzer
from processing.ekf_predictor import CompressorEKF
from processing.surge_line import SurgeLine
from simulator.pressure_simulator import PressureSimulator


SAMPLE_RATE = 100000
BUFFER_CAPACITY = SAMPLE_RATE * 10
CWT_TIME_BINS = 300
CWT_FREQ_BINS = 64
UPDATE_INTERVAL_MS = 500
SURGE_MARGIN_THRESHOLD = 1.5


ring_buffer = LockFreeRingBuffer(BUFFER_CAPACITY, dtype=np.float32)
ring_buffer.sample_rate = SAMPLE_RATE

cwt_analyzer = CWTAnalyzer(
    sample_rate=SAMPLE_RATE,
    freq_min=200.0,
    freq_max=45000.0,
    n_scales=CWT_FREQ_BINS,
    log_scale=True,
)

ekf = CompressorEKF(dt=0.01)
surge_line = SurgeLine(W_design=25.0, PR_design=12.5, W_surge_min=10.0)

simulator = PressureSimulator(
    sample_rate=SAMPLE_RATE,
    noise_level=0.5,
)

_latest_stats = {"total": 0, "utilization": 0, "latest_ts": 0.0, "surge_intensity": 0.0}
_ekf_state = {"W": 25.0, "PR": 12.5, "margin": 0.0, "bleed_active": False, "severity": 0.0}
_compressor_history_W = []
_compressor_history_PR = []
_MAX_HISTORY = 300


def simulator_loop():
    def on_batch(data, ts, W, PR):
        ring_buffer.write(data, ts)
        ekf.predict()
        ekf.update(np.array([W, PR]))
        _compressor_history_W.append(W)
        _compressor_history_PR.append(PR)
        if len(_compressor_history_W) > _MAX_HISTORY:
            _compressor_history_W.pop(0)
            _compressor_history_PR.pop(0)
    simulator.generate_continuous(batch_duration=0.01, callback=on_batch)


def _stat_badge(label, value, color, id=None):
    props = {
        "style": {
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "gap": "2px",
        },
        "children": [
            html.Span(label, style={"color": "#8b949e", "fontSize": "10px", "textTransform": "uppercase"}),
            html.Span(value, style={"color": color, "fontSize": "14px", "fontWeight": "700", "fontFamily": "monospace"}),
        ],
    }
    if id is not None:
        props["id"] = id
    return html.Div(**props)


def _panel_header(title, subtitle):
    return html.Div(
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "padding": "4px 8px",
            "borderBottom": "1px solid #21262d",
            "marginBottom": "4px",
        },
        children=[
            html.Span(title, style={"color": "#e6edf3", "fontSize": "13px", "fontWeight": "600"}),
            html.Span(subtitle, style={"color": "#8b949e", "fontSize": "11px", "fontFamily": "monospace"}),
        ],
    )


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Aero Engine Surge Monitor",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": "#0a0a1a", "minHeight": "100vh", "padding": "0"},
    children=[
        html.Div(
            style={
                "backgroundColor": "#0d1117",
                "borderBottom": "2px solid #1f6feb",
                "padding": "12px 24px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
            },
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "16px"},
                    children=[
                        html.Div(
                            style={
                                "width": "12px",
                                "height": "12px",
                                "borderRadius": "50%",
                                "backgroundColor": "#3fb950",
                                "boxShadow": "0 0 8px #3fb950",
                            }
                        ),
                        html.H1(
                            "涡扇发动机喘振监控大屏",
                            style={
                                "color": "#e6edf3",
                                "fontSize": "22px",
                                "fontWeight": "600",
                                "margin": "0",
                                "letterSpacing": "2px",
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "gap": "32px"},
                    children=[
                        _stat_badge("采样率", "100 kHz", "#58a6ff"),
                        _stat_badge("传感器", "CASCADE-01", "#3fb950"),
                        _stat_badge("缓冲区", "0%", "#d29922", id="buffer-badge"),
                        _stat_badge("喘振裕度", "--", "#3fb950", id="margin-badge"),
                        _stat_badge("放气阀", "CLOSED", "#8b949e", id="bleed-badge"),
                        _stat_badge("状态", "NORMAL", "#3fb950", id="status-badge"),
                    ],
                ),
            ],
        ),
        html.Div(
            style={
                "padding": "12px 16px",
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gridTemplateRows": "180px 1fr 1fr",
                "gap": "12px",
                "height": "calc(100vh - 70px)",
            },
            children=[
                html.Div(
                    style={"gridColumn": "1 / -1"},
                    children=[
                        _panel_header("实时压力波形", "P-CH1"),
                        dcc.Graph(
                            id="pressure-waveform",
                            style={"height": "150px"},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        _panel_header("CWT 频谱热力图 (Morlet)", "对数能量密度 dB"),
                        dcc.Graph(
                            id="cwt-heatmap",
                            style={"height": "calc(100% - 40px)"},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        _panel_header("3D 压气机性能图 / 喘振边界", "W-PR-N Map"),
                        dcc.Graph(
                            id="compressor-3d-map",
                            style={"height": "calc(100% - 40px)"},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        _panel_header("频带能量追踪", "dB"),
                        dcc.Graph(
                            id="band-energy",
                            style={"height": "calc(100% - 40px)"},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        _panel_header("EKF 状态轨迹 / 放气阀指令", "预测步长: 20"),
                        dcc.Graph(
                            id="ekf-trajectory",
                            style={"height": "calc(100% - 40px)"},
                            config={"displayModeBar": False},
                        ),
                    ],
                ),
            ],
        ),
        dcc.Interval(id="update-timer", interval=UPDATE_INTERVAL_MS, n_intervals=0),
    ],
)


@app.callback(
    [
        Output("pressure-waveform", "figure"),
        Output("cwt-heatmap", "figure"),
        Output("band-energy", "figure"),
        Output("compressor-3d-map", "figure"),
        Output("ekf-trajectory", "figure"),
        Output("buffer-badge", "children"),
        Output("margin-badge", "children"),
        Output("bleed-badge", "children"),
        Output("status-badge", "children"),
    ],
    [Input("update-timer", "n_intervals")],
)
def update_dashboard(n):
    data, ts = ring_buffer.get_latest_second()

    if len(data) == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return (
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            ["缓冲区", "0%"], ["喘振裕度", "--"], ["放气阀", "CLOSED"], ["状态", "WAITING"],
        )

    fig_waveform = _build_waveform(data, ts)

    n_sub = min(len(data), 2000)
    step = max(1, len(data) // n_sub)
    data_sub = data[::step].copy()

    del data
    del ts

    heatmap_data = cwt_analyzer.compute_heatmap_data(data_sub, time_offset=0.0)
    del data_sub

    if heatmap_data:
        heatmap_data = cwt_analyzer.downsample_heatmap(
            heatmap_data,
            target_time_bins=CWT_TIME_BINS,
            target_freq_bins=CWT_FREQ_BINS,
        )

    fig_heatmap = _build_heatmap(heatmap_data)
    fig_band = _build_band_energy(heatmap_data)

    if heatmap_data is not None:
        del heatmap_data

    fig_3d = _build_3d_compressor_map()
    fig_ekf = _build_ekf_trajectory()

    bleed_cmd = ekf.compute_bleed_valve_command(surge_line, threshold=SURGE_MARGIN_THRESHOLD)
    margin = bleed_cmd["margin"]
    severity = bleed_cmd["severity"]
    bleed_active = bleed_cmd["active"]
    valve_pct = bleed_cmd["valve_open_pct"]

    util_pct = ring_buffer.utilization
    margin_color = "#3fb950" if margin > 3.0 else ("#d29922" if margin > 1.5 else "#f85149")
    bleed_color = "#f85149" if bleed_active else "#8b949e"
    status_val = "SURGE WARN" if severity > 0.3 else ("CAUTION" if margin < 3.0 else "NORMAL")
    status_color = "#f85149" if severity > 0.3 else ("#d29922" if margin < 3.0 else "#3fb950")

    _latest_stats["utilization"] = util_pct
    _latest_stats["total"] = ring_buffer.total_written
    _latest_stats["latest_ts"] = ring_buffer.latest_timestamp
    _latest_stats["surge_intensity"] = severity
    _ekf_state["margin"] = margin
    _ekf_state["bleed_active"] = bleed_active
    _ekf_state["severity"] = severity

    gc.collect()

    return (
        fig_waveform, fig_heatmap, fig_band, fig_3d, fig_ekf,
        ["缓冲区", f"{util_pct}%"],
        ["喘振裕度", f"{margin:.1f}"],
        ["放气阀", f"{'OPEN ' + str(int(valve_pct)) + '%' if bleed_active else 'CLOSED'}"],
        ["状态", status_val],
    )


def _build_waveform(data, ts):
    fig = go.Figure()
    n_display = min(len(data), 5000)
    step = max(1, len(data) // n_display)
    d = data[::step].copy()
    t = ts[::step].copy() if len(ts) == len(data) else np.arange(len(d), dtype=np.float64) / SAMPLE_RATE
    fig.add_trace(
        go.Scattergl(
            x=t.tolist(),
            y=d.tolist(),
            mode="lines",
            line=dict(color="#58a6ff", width=1),
            name="P1",
        )
    )
    del d
    del t
    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=50, r=10, t=10, b=30),
        xaxis=dict(color="#8b949e", gridcolor="#21262d", title=dict(text="Time (s)", font=dict(size=10)), tickfont=dict(size=9)),
        yaxis=dict(color="#8b949e", gridcolor="#21262d", title=dict(text="Pressure (kPa)", font=dict(size=10)), tickfont=dict(size=9)),
        font=dict(family="monospace"),
    )
    return fig


def _build_heatmap(heatmap_data):
    fig = go.Figure()
    if heatmap_data is None:
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", xaxis={"visible": False}, yaxis={"visible": False})
        return fig
    power = heatmap_data["power"]
    freqs = heatmap_data["freqs"]
    times = heatmap_data["times"]
    z_data = power.tolist()
    x_data = times.tolist()
    y_data = freqs.tolist()
    del power
    del freqs
    del times
    fig.add_trace(
        go.Heatmap(
            z=z_data, x=x_data, y=y_data,
            colorscale=[
                [0.0, "#000033"], [0.15, "#000080"], [0.3, "#0040ff"],
                [0.45, "#00c0ff"], [0.6, "#00ff80"], [0.75, "#ffff00"],
                [0.9, "#ff4000"], [1.0, "#ffffff"],
            ],
            showscale=True,
            colorbar=dict(title=dict(text="dB", font=dict(size=9, color="#8b949e")), tickfont=dict(size=8, color="#8b949e"), thickness=10, len=0.9),
            hovertemplate="Freq: %{y:.0f} Hz<br>Time: %{x:.3f} s<br>Power: %{z:.1f} dB<extra></extra>",
        )
    )
    del z_data
    del x_data
    del y_data
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", margin=dict(l=60, r=10, t=10, b=30),
        xaxis=dict(color="#8b949e", gridcolor="#21262d", title=dict(text="Time (s)", font=dict(size=10)), tickfont=dict(size=9)),
        yaxis=dict(color="#8b949e", gridcolor="#21262d", title=dict(text="Frequency (Hz)", font=dict(size=10)), tickfont=dict(size=9), type="log"),
        font=dict(family="monospace"),
    )
    return fig


def _build_band_energy(heatmap_data):
    fig = go.Figure()
    if heatmap_data is None:
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", xaxis={"visible": False}, yaxis={"visible": False})
        return fig
    power = heatmap_data["power"]
    freqs = heatmap_data["freqs"]
    band_defs = [
        ("0.2-1 kHz", 200, 1000, "#ff7b72"),
        ("1-5 kHz", 1000, 5000, "#ffa657"),
        ("5-15 kHz", 5000, 15000, "#d2a8ff"),
        ("15-30 kHz", 15000, 30000, "#58a6ff"),
        ("30-45 kHz", 30000, 45000, "#3fb950"),
    ]
    band_values = []
    for label, f_lo, f_hi, color in band_defs:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if np.any(mask):
            band_power = np.mean(power[mask, :], axis=0)
            mean_db = float(np.mean(band_power))
            del band_power
            del mask
        else:
            mean_db = 0.0
        band_values.append((label, mean_db, color))
    del power
    del freqs
    for label, mean_db, color in band_values:
        fig.add_trace(go.Bar(name=label, x=[label], y=[mean_db], marker_color=color, marker_line_width=0, width=0.6))
    del band_values
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", margin=dict(l=50, r=10, t=10, b=40),
        barmode="group", showlegend=False,
        xaxis=dict(color="#8b949e", tickfont=dict(size=9)),
        yaxis=dict(color="#8b949e", gridcolor="#21262d", title=dict(text="Mean Power (dB)", font=dict(size=10)), tickfont=dict(size=9)),
        font=dict(family="monospace"),
    )
    return fig


def _build_3d_compressor_map():
    fig = go.Figure()

    speed_lines = surge_line.get_compressor_map_speed_lines(n_speeds=6, n_points=40)

    for sl in speed_lines:
        W_list = sl["W"].tolist()
        PR_list = sl["PR"].tolist()
        N_val = sl["N_pct"]

        fig.add_trace(
            go.Scatter3d(
                x=W_list,
                y=PR_list,
                z=[N_val] * len(W_list),
                mode="lines",
                line=dict(color="#30363d", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        W_eff_list = sl["W_eff"].tolist()
        PR_eff_list = sl["PR_eff"].tolist()
        fig.add_trace(
            go.Scatter3d(
                x=W_eff_list,
                y=PR_eff_list,
                z=[N_val] * len(W_eff_list),
                mode="lines",
                line=dict(color="#21262d", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        del sl

    W_surge, PR_surge = surge_line.get_surge_line_points(n_points=80)
    W_surge_list = W_surge.tolist()
    PR_surge_list = PR_surge.tolist()
    del W_surge
    del PR_surge

    for N_pct in [70, 80, 90, 100]:
        fig.add_trace(
            go.Scatter3d(
                x=W_surge_list,
                y=PR_surge_list,
                z=[N_pct] * len(W_surge_list),
                mode="lines",
                line=dict(color="#f85149", width=4),
                showlegend=False,
                hovertemplate="SURGE LINE<br>W: %{x:.1f} kg/s<br>PR: %{y:.2f}<extra></extra>",
            )
        )

    del W_surge_list
    del PR_surge_list

    current_state = ekf.state
    W_cur = float(current_state[0])
    PR_cur = float(current_state[1])
    del current_state

    trajectory = ekf.predict_trajectory(n_steps=20)
    traj_W = [float(t[0]) for t in trajectory]
    traj_PR = [float(t[1]) for t in trajectory]
    del trajectory

    n_hist = min(len(_compressor_history_W), 60)
    if n_hist > 0:
        hist_W = _compressor_history_W[-n_hist:]
        hist_PR = _compressor_history_PR[-n_hist:]
        hist_W_list = list(hist_W)
        hist_PR_list = list(hist_PR)
        fig.add_trace(
            go.Scatter3d(
                x=hist_W_list,
                y=hist_PR_list,
                z=[100] * len(hist_W_list),
                mode="lines",
                line=dict(color="#58a6ff", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        del hist_W_list
        del hist_PR_list

    fig.add_trace(
        go.Scatter3d(
            x=traj_W,
            y=traj_PR,
            z=[100] * len(traj_W),
            mode="lines+markers",
            line=dict(color="#ffa657", width=3, dash="dash"),
            marker=dict(size=3, color="#ffa657"),
            showlegend=False,
            hovertemplate="PREDICTED<br>W: %{x:.1f} kg/s<br>PR: %{y:.2f}<extra></extra>",
        )
    )
    del traj_W
    del traj_PR

    bleed_cmd = ekf.compute_bleed_valve_command(surge_line, threshold=SURGE_MARGIN_THRESHOLD)
    margin = bleed_cmd["margin"]
    point_color = "#f85149" if margin < SURGE_MARGIN_THRESHOLD else "#3fb950"
    point_size = 12 if margin < SURGE_MARGIN_THRESHOLD else 8

    fig.add_trace(
        go.Scatter3d(
            x=[W_cur],
            y=[PR_cur],
            z=[100],
            mode="markers",
            marker=dict(size=point_size, color=point_color, symbol="diamond", line=dict(width=2, color="white")),
            showlegend=False,
            hovertemplate="CURRENT<br>W: %{x:.1f} kg/s<br>PR: %{y:.2f}<extra></extra>",
        )
    )

    if margin < SURGE_MARGIN_THRESHOLD:
        W_surge_pt, PR_surge_pt = surge_line.get_surge_line_points(n_points=20)
        fig.add_trace(
            go.Scatter3d(
                x=W_surge_pt.tolist(),
                y=PR_surge_pt.tolist(),
                z=[100.5] * len(W_surge_pt),
                mode="lines",
                line=dict(color="#f85149", width=6),
                opacity=0.8,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        del W_surge_pt
        del PR_surge_pt

    fig.update_layout(
        paper_bgcolor="#0d1117",
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(title=dict(text="Mass Flow W (kg/s)", font=dict(size=10, color="#8b949e")), color="#8b949e", gridcolor="#21262d", tickfont=dict(size=8)),
            yaxis=dict(title=dict(text="Pressure Ratio PR", font=dict(size=10, color="#8b949e")), color="#8b949e", gridcolor="#21262d", tickfont=dict(size=8)),
            zaxis=dict(title=dict(text="Speed N (%)", font=dict(size=10, color="#8b949e")), color="#8b949e", gridcolor="#21262d", tickfont=dict(size=8)),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=10, b=10),
        font=dict(family="monospace"),
    )
    return fig


def _build_ekf_trajectory():
    fig = go.Figure()

    W_surge, PR_surge = surge_line.get_surge_line_points(n_points=100)
    fig.add_trace(
        go.Scatter(
            x=W_surge.tolist(),
            y=PR_surge.tolist(),
            mode="lines",
            line=dict(color="#f85149", width=3, dash="dash"),
            name="Surge Line",
            hovertemplate="SURGE LINE<br>W: %{x:.1f}<br>PR: %{y:.2f}<extra></extra>",
        )
    )
    del W_surge
    del PR_surge

    n_hist = min(len(_compressor_history_W), 200)
    if n_hist > 1:
        hist_W = _compressor_history_W[-n_hist:]
        hist_PR = _compressor_history_PR[-n_hist:]
        fig.add_trace(
            go.Scatter(
                x=list(hist_W),
                y=list(hist_PR),
                mode="lines",
                line=dict(color="#58a6ff", width=1.5),
                name="History",
                hoverinfo="skip",
            )
        )

    current_state = ekf.state
    W_cur = float(current_state[0])
    PR_cur = float(current_state[1])
    del current_state

    trajectory = ekf.predict_trajectory(n_steps=20)
    traj_W = [float(t[0]) for t in trajectory]
    traj_PR = [float(t[1]) for t in trajectory]
    del trajectory

    fig.add_trace(
        go.Scatter(
            x=traj_W,
            y=traj_PR,
            mode="lines+markers",
            line=dict(color="#ffa657", width=2, dash="dot"),
            marker=dict(size=5, color="#ffa657", symbol="triangle-up"),
            name="EKF Predicted",
            hovertemplate="PREDICTED<br>W: %{x:.1f}<br>PR: %{y:.2f}<extra></extra>",
        )
    )
    del traj_W
    del traj_PR

    bleed_cmd = ekf.compute_bleed_valve_command(surge_line, threshold=SURGE_MARGIN_THRESHOLD)
    margin = bleed_cmd["margin"]
    point_color = "#f85149" if margin < SURGE_MARGIN_THRESHOLD else "#3fb950"
    marker_symbol = "x" if margin < SURGE_MARGIN_THRESHOLD else "diamond"

    fig.add_trace(
        go.Scatter(
            x=[W_cur],
            y=[PR_cur],
            mode="markers",
            marker=dict(size=14, color=point_color, symbol=marker_symbol, line=dict(width=2, color="white")),
            name="Current",
            hovertemplate="CURRENT<br>W: %{x:.1f}<br>PR: %{y:.2f}<extra></extra>",
        )
    )

    if bleed_cmd["active"]:
        fig.add_hrect(
            y0=surge_line(W_cur),
            y1=surge_line(W_cur) + 0.5,
            fillcolor="rgba(248, 81, 73, 0.15)",
            line_width=0,
        )
        fig.add_annotation(
            x=W_cur - 1.5,
            y=PR_cur + 0.3,
            text=f"BLEED VALVE {int(bleed_cmd['valve_open_pct'])}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="#f85149",
            font=dict(size=11, color="#f85149", family="monospace"),
            bordercolor="#f85149",
            borderwidth=1,
            borderpad=4,
            bgcolor="#1a0505",
        )

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=55, r=10, t=10, b=35),
        xaxis=dict(
            color="#8b949e",
            gridcolor="#21262d",
            title=dict(text="Mass Flow W (kg/s)", font=dict(size=10)),
            tickfont=dict(size=9),
            range=[8, 35],
        ),
        yaxis=dict(
            color="#8b949e",
            gridcolor="#21262d",
            title=dict(text="Pressure Ratio PR", font=dict(size=10)),
            tickfont=dict(size=9),
            range=[8, 18],
        ),
        showlegend=False,
        font=dict(family="monospace"),
    )
    return fig


def run_dashboard(host="0.0.0.0", port=8051, with_simulator=True):
    if with_simulator:
        sim_thread = threading.Thread(target=simulator_loop, daemon=True)
        sim_thread.start()
        print("Pressure simulator started (100 kHz) with EKF + Surge Line")

    print(f"Dashboard starting at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_dashboard()
