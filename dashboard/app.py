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
from simulator.pressure_simulator import PressureSimulator


SAMPLE_RATE = 100000
BUFFER_CAPACITY = SAMPLE_RATE * 10
CWT_TIME_BINS = 300
CWT_FREQ_BINS = 64
UPDATE_INTERVAL_MS = 500


ring_buffer = LockFreeRingBuffer(BUFFER_CAPACITY, dtype=np.float32)
ring_buffer.sample_rate = SAMPLE_RATE

cwt_analyzer = CWTAnalyzer(
    sample_rate=SAMPLE_RATE,
    freq_min=200.0,
    freq_max=45000.0,
    n_scales=CWT_FREQ_BINS,
    log_scale=True,
)

simulator = PressureSimulator(
    sample_rate=SAMPLE_RATE,
    noise_level=0.5,
)

_latest_stats = {"total": 0, "utilization": 0, "latest_ts": 0.0, "surge_intensity": 0.0}


def simulator_loop():
    def on_batch(data, ts):
        ring_buffer.write(data, ts)
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
                        _stat_badge("状态", "NORMAL", "#3fb950", id="status-badge"),
                    ],
                ),
            ],
        ),
        html.Div(
            style={
                "padding": "16px",
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gridTemplateRows": "auto 1fr",
                "gap": "16px",
                "height": "calc(100vh - 70px)",
            },
            children=[
                html.Div(
                    style={"gridColumn": "1 / -1"},
                    children=[
                        _panel_header("实时压力波形", "P-CH1"),
                        dcc.Graph(
                            id="pressure-waveform",
                            style={"height": "200px"},
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
                        _panel_header("频带能量追踪", "dB"),
                        dcc.Graph(
                            id="band-energy",
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
        Output("buffer-badge", "children"),
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
        return empty_fig, empty_fig, empty_fig, ["缓冲区", "0%"], ["状态", "WAITING"]

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

    util_pct = ring_buffer.utilization
    status_val = "SURGE WARN" if _latest_stats.get("surge_intensity", 0) > 0.3 else "NORMAL"

    _latest_stats["utilization"] = util_pct
    _latest_stats["total"] = ring_buffer.total_written
    _latest_stats["latest_ts"] = ring_buffer.latest_timestamp

    gc.collect()

    return fig_waveform, fig_heatmap, fig_band, ["缓冲区", f"{util_pct}%"], ["状态", status_val]


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
        xaxis=dict(
            color="#8b949e",
            gridcolor="#21262d",
            title=dict(text="Time (s)", font=dict(size=10)),
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            color="#8b949e",
            gridcolor="#21262d",
            title=dict(text="Pressure (kPa)", font=dict(size=10)),
            tickfont=dict(size=9),
        ),
        font=dict(family="monospace"),
    )
    return fig


def _build_heatmap(heatmap_data):
    fig = go.Figure()

    if heatmap_data is None:
        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
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
            z=z_data,
            x=x_data,
            y=y_data,
            colorscale=[
                [0.0, "#000033"],
                [0.15, "#000080"],
                [0.3, "#0040ff"],
                [0.45, "#00c0ff"],
                [0.6, "#00ff80"],
                [0.75, "#ffff00"],
                [0.9, "#ff4000"],
                [1.0, "#ffffff"],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="dB", font=dict(size=9, color="#8b949e")),
                tickfont=dict(size=8, color="#8b949e"),
                thickness=10,
                len=0.9,
            ),
            hovertemplate="Freq: %{y:.0f} Hz<br>Time: %{x:.3f} s<br>Power: %{z:.1f} dB<extra></extra>",
        )
    )

    del z_data
    del x_data
    del y_data

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=60, r=10, t=10, b=30),
        xaxis=dict(
            color="#8b949e",
            gridcolor="#21262d",
            title=dict(text="Time (s)", font=dict(size=10)),
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            color="#8b949e",
            gridcolor="#21262d",
            title=dict(text="Frequency (Hz)", font=dict(size=10)),
            tickfont=dict(size=9),
            type="log",
        ),
        font=dict(family="monospace"),
    )
    return fig


def _build_band_energy(heatmap_data):
    fig = go.Figure()

    if heatmap_data is None:
        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
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
        fig.add_trace(
            go.Bar(
                name=label,
                x=[label],
                y=[mean_db],
                marker_color=color,
                marker_line_width=0,
                width=0.6,
            )
        )

    del band_values

    fig.update_layout(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=50, r=10, t=10, b=40),
        barmode="group",
        showlegend=False,
        xaxis=dict(
            color="#8b949e",
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            color="#8b949e",
            gridcolor="#21262d",
            title=dict(text="Mean Power (dB)", font=dict(size=10)),
            tickfont=dict(size=9),
        ),
        font=dict(family="monospace"),
    )
    return fig


def run_dashboard(host="0.0.0.0", port=8050, with_simulator=True):
    if with_simulator:
        sim_thread = threading.Thread(target=simulator_loop, daemon=True)
        sim_thread.start()
        print("Pressure simulator started (100 kHz)")

    print(f"Dashboard starting at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    run_dashboard()
