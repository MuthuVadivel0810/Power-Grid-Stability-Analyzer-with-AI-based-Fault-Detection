"""
=============================================================
  POWER GRID STABILITY ANALYZER
  Module: Plotly Chart Builders  (dashboard/charts.py)

  All interactive Plotly figures used in the dashboard:
    build_topology_fig()       – network graph (go.Scatter)
    build_voltage_bar()        – bus voltage profile
    build_line_flow_bar()      – P/Q flow per line
    build_waveform_fig()       – 3-phase Va/Vb/Vc + Ia
    build_sag_heatmap()        – post-fault voltage sag grid
    build_confusion_matrix()   – ML evaluation heatmap
    build_feature_importance() – horizontal bar chart
    build_probability_gauge()  – single-class confidence gauge
    build_class_probabilities()– stacked probability bars
=============================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from Dashboard.Utils import PLOTLY_LAYOUT, FAULT_COLORS, CLASS_NAMES

BUS_TYPE_COLORS = {"Slack": "#E74C3C", "PV": "#2ECC71", "PQ": "#3498DB"}

# ── Fixed node positions for IEEE 5-bus ───────────────────────────────────────
BUS_POS = {1:(0.0,0.8), 2:(0.5,1.0), 3:(0.0,0.2), 4:(0.5,0.0), 5:(1.0,0.5)}


# ==============================================================================
def build_topology_fig(bus_data, line_data, V=None,
                       highlight_bus=None, fault_type=None):
    """
    Interactive NetworkX-style grid topology.
    Nodes sized by voltage magnitude, highlighted fault bus in red.
    """
    fig = go.Figure()

    # ── Edges ─────────────────────────────────────────────────────────────────
    for _, line in line_data.iterrows():
        i, j = int(line["from_bus"]), int(line["to_bus"])
        x0, y0 = BUS_POS[i]; x1, y1 = BUS_POS[j]
        Z = round((line["R_pu"]**2 + line["X_pu"]**2)**0.5, 4)
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(color="#A8D8EA", width=3),
            hoverinfo="text",
            text=f"Line {int(line['line_id'])}: Bus{i}→Bus{j}<br>Z={Z} pu",
            showlegend=False,
        ))
        # Edge label (impedance)
        mx, my = (x0+x1)/2, (y0+y1)/2
        fig.add_annotation(x=mx, y=my, text=f"Z={Z}",
                           showarrow=False, font=dict(size=9, color="#F9CA24"),
                           bgcolor="rgba(26,26,46,0.7)")

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for _, bus in bus_data.iterrows():
        bid  = int(bus["bus_id"])
        x, y = BUS_POS[bid]
        V_mag = float(V[bid-1]) if V is not None else float(bus["V_mag"])
        is_fault = (highlight_bus is not None and bid == highlight_bus)

        color  = "#FF0000" if is_fault else BUS_TYPE_COLORS.get(bus["bus_type"], "#888")
        size   = 48 if is_fault else 38
        symbol = "star" if is_fault else "circle"
        label  = (f"⚡ FAULT: {fault_type.replace('_',' ')}<br>"
                  if is_fault else "") + \
                 f"<b>Bus-{bid}</b> ({bus['bus_type']})<br>" \
                 f"|V| = {V_mag:.4f} pu<br>" \
                 f"P_load = {bus['P_load_pu']} pu<br>" \
                 f"Q_load = {bus['Q_load_pu']} pu"

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=size, color=color,
                        line=dict(width=2, color="white"),
                        symbol=symbol),
            text=f"Bus-{bid}", textposition="top center",
            textfont=dict(color="white", size=11, family="monospace"),
            hovertext=label, hoverinfo="text",
            name=f"Bus-{bid} ({bus['bus_type']})",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="⚡ IEEE 5-Bus Power Grid Topology",
                   font=dict(size=16, color="white")),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=420,
        legend=dict(bgcolor="rgba(22,33,62,0.8)", bordercolor="#555",
                    font=dict(color="white", size=10)),
        showlegend=True,
    )
    return fig


# ==============================================================================
def build_voltage_bar(bus_data, V, theta_deg):
    """Grouped bar: |V| and θ per bus, with limit lines."""
    buses  = [f"Bus-{int(b)}" for b in bus_data["bus_id"]]
    V_list = list(V)
    T_list = list(theta_deg)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Voltage Magnitude |V| (pu)",
                                        "Voltage Angle θ (deg)"])

    colors_v = ["#E74C3C" if (v < 0.95 or v > 1.05)
                else "#2ECC71" for v in V_list]

    fig.add_trace(go.Bar(
        x=buses, y=V_list, marker_color=colors_v,
        text=[f"{v:.4f}" for v in V_list], textposition="outside",
        textfont=dict(color="white", size=10), name="|V| (pu)",
    ), row=1, col=1)

    fig.add_hline(y=0.95, line_dash="dash", line_color="#E67E22",
                  annotation_text="0.95 pu min", row=1, col=1)
    fig.add_hline(y=1.05, line_dash="dash", line_color="#F9CA24",
                  annotation_text="1.05 pu max", row=1, col=1)

    fig.add_trace(go.Bar(
        x=buses, y=T_list, marker_color="#3498DB",
        text=[f"{t:.3f}°" for t in T_list], textposition="outside",
        textfont=dict(color="white", size=10), name="θ (deg)",
    ), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                      title="Bus Voltage Profile — Load Flow Results")
    fig.update_yaxes(range=[0.88, 1.12], row=1, col=1)
    return fig


# ==============================================================================
def build_line_flow_bar(line_data, line_results):
    """Grouped bar: P and Q flow per line."""
    lines  = line_results["From→To"].tolist()
    P_vals = line_results["P_ij (pu)"].tolist()
    Q_vals = line_results["Q_ij (pu)"].tolist()
    losses = [abs(v) for v in line_results["P_loss (pu)"].tolist()]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Active & Reactive Flow (pu)",
                                        "Active Power Losses (pu)"])

    fig.add_trace(go.Bar(name="P flow", x=lines, y=P_vals,
                         marker_color="#2ECC71", opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(name="Q flow", x=lines, y=Q_vals,
                         marker_color="#3498DB", opacity=0.85), row=1, col=1)
    fig.add_trace(go.Bar(name="P loss", x=lines, y=losses,
                         marker_color="#E74C3C", opacity=0.85), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=370, barmode="group",
                      title="Transmission Line Power Flows & Losses")
    return fig


# ==============================================================================
def build_waveform_fig(waveform, fault_type):
    """Full 3-phase voltage + current waveform with fault window shading."""
    t_ms = waveform.t * 1000

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["3-Phase Voltages (pu)",
                                        "3-Phase Currents (pu)"],
                        shared_xaxes=True, vertical_spacing=0.12)

    # ── Voltages ───────────────────────────────────────────────────────────────
    for sig, col, name in [
        (waveform.Va, "#E74C3C", "Va"),
        (waveform.Vb, "#2ECC71", "Vb"),
        (waveform.Vc, "#3498DB", "Vc"),
    ]:
        fig.add_trace(go.Scatter(x=t_ms, y=sig, mode="lines",
                                 line=dict(color=col, width=1.6),
                                 name=name), row=1, col=1)

    # ── Currents ───────────────────────────────────────────────────────────────
    for sig, col, name in [
        (waveform.Ia, "#E67E22", "Ia"),
        (waveform.Ib, "#9B59B6", "Ib"),
        (waveform.Ic, "#1ABC9C", "Ic"),
    ]:
        fig.add_trace(go.Scatter(x=t_ms, y=sig, mode="lines",
                                 line=dict(color=col, width=1.6),
                                 name=name), row=2, col=1)

    # ── Fault window shading ───────────────────────────────────────────────────
    if fault_type != "No_Fault":
        for row in [1, 2]:
            fig.add_vrect(x0=40, x1=80,
                          fillcolor="rgba(249,202,36,0.07)",
                          line=dict(color="#F9CA24", width=1.5, dash="dash"),
                          row=row, col=1)
        fig.add_annotation(
            x=60, y=1.6, text=f"⚡ {fault_type.replace('_',' ')}",
            showarrow=False, font=dict(color="#F9CA24", size=11),
            bgcolor="rgba(26,26,46,0.85)", row=1, col=1,
            xref="x", yref="y",
        )

    fig.update_layout(**PLOTLY_LAYOUT, height=520,
                      title=f"3-Phase Waveforms — {fault_type.replace('_',' ')}",
                      legend=dict(orientation="h", y=-0.08,
                                  font=dict(color="white")))
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1,
                     color="white")
    fig.update_yaxes(title_text="Voltage (pu)", row=1, col=1, color="white")
    fig.update_yaxes(title_text="Current (pu)", row=2, col=1, color="white")
    return fig


# ==============================================================================
def build_sag_heatmap(fault_result, bus_data):
    """Post-fault bus voltage heatmap — prefault vs postfault."""
    buses   = [f"Bus-{i+1}" for i in range(len(bus_data))]
    pre     = list(fault_result.V_bus_prefault)
    post    = list(fault_result.V_bus_postfault)
    sag     = list(fault_result.voltage_sag)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Pre-fault |V|",  x=buses, y=pre,
                         marker_color="#2ECC71", opacity=0.8))
    fig.add_trace(go.Bar(name="Post-fault |V|", x=buses, y=post,
                         marker_color="#E74C3C", opacity=0.85))
    fig.add_trace(go.Scatter(name="Voltage Sag",x=buses, y=sag,
                             mode="lines+markers",
                             line=dict(color="#F9CA24", width=2.5),
                             marker=dict(size=8, color="#F9CA24"),
                             yaxis="y2"))

    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="group", height=380,
        title=f"Post-Fault Bus Voltage Profile "
              f"(Fault: {fault_result.fault_type.replace('_',' ')} @ Bus-{fault_result.fault_bus})",
        yaxis =dict(title="Voltage Magnitude (pu)", range=[0, 1.15]),
        yaxis2=dict(title="Voltage Sag (pu)", overlaying="y",
                    side="right", showgrid=False,
                    tickfont=dict(color="#F9CA24"),
                    titlefont=dict(color="#F9CA24")),
        legend=dict(font=dict(color="white"), bgcolor="rgba(22,33,62,0.8)"),
    )
    fig.add_hline(y=0.95, line_dash="dash", line_color="#E67E22")
    return fig


# ==============================================================================
def build_class_probabilities(probs: dict, predicted: str):
    """Horizontal bar chart of all class probabilities."""
    classes = list(probs.keys())
    values  = list(probs.values())
    colors  = ["#E74C3C" if c == predicted else "#3498DB" for c in classes]

    fig = go.Figure(go.Bar(
        x=values, y=classes, orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=0.5)),
        text=[f"{v:.2f}%" for v in values], textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=300,
        title="Fault Class Probabilities (%)",
        xaxis=dict(range=[0, 115], title="Probability (%)", color="white"),
        yaxis=dict(color="white"),
    )
    return fig


# ==============================================================================
def build_feature_importance_fig(importance_df, top_n=15):
    """Horizontal bar chart — top N feature importances."""
    df   = importance_df.head(top_n).sort_values("importance")
    cols = ["#E74C3C" if v > 0.08 else "#3498DB" if v > 0.04
            else "#2ECC71" for v in df["importance"]]

    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"], orientation="h",
        marker=dict(color=cols), error_x=dict(array=df["std"].tolist(),
                                               color="white", thickness=1.5),
        text=[f"{v:.4f}" for v in df["importance"]],
        textposition="outside", textfont=dict(color="white", size=9),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=480,
        title=f"Top-{top_n} Feature Importances — Random Forest",
        xaxis=dict(title="Importance Score", color="white"),
        yaxis=dict(color="white"),
    )
    return fig


# ==============================================================================
def build_confusion_matrix_fig(cm_norm, class_labels):
    """Heatmap for confusion matrix."""
    fig = px.imshow(cm_norm, x=class_labels, y=class_labels,
                    color_continuous_scale="Blues",
                    zmin=0, zmax=1,
                    labels=dict(x="Predicted", y="True", color="Recall"))
    fig.update_traces(text=[[f"{cm_norm[i][j]:.2f}"
                             for j in range(len(class_labels))]
                            for i in range(len(class_labels))],
                      texttemplate="%{text}", textfont=dict(size=12))
    fig.update_layout(**PLOTLY_LAYOUT, height=420,
                      title="Confusion Matrix (Normalised)")
    return fig


# ==============================================================================
def build_roc_fig(roc_data: dict):
    """Multi-class ROC curves."""
    colors = ["#2ECC71", "#E74C3C", "#3498DB", "#F9CA24", "#9B59B6"]
    fig    = go.Figure()

    for i, (cls, d) in enumerate(roc_data.items()):
        name = CLASS_NAMES.get(cls, str(cls))
        fig.add_trace(go.Scatter(
            x=d["fpr"], y=d["tpr"], mode="lines",
            line=dict(color=colors[i], width=2.2),
            name=f"{name} (AUC={d['auc']:.4f})",
            fill="tozeroy" if i == 0 else None,
            fillcolor=f"rgba(46,204,113,0.05)" if i == 0 else None,
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#7F8C8D", dash="dash", width=1.2),
        name="Random (AUC=0.5)", showlegend=True,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=420,
        title="ROC Curves — One-vs-Rest (All 5 Classes)",
        xaxis=dict(title="False Positive Rate", color="white",
                   range=[0, 1]),
        yaxis=dict(title="True Positive Rate", color="white",
                   range=[0, 1.02]),
        legend=dict(font=dict(color="white", size=9),
                    bgcolor="rgba(22,33,62,0.8)"),
    )
    return fig