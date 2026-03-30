"""PDF report generation for EV Cost-Per-Mile Simulator."""

import io
import tempfile
from pathlib import Path

from ev_model import SimConfig, get_cost_components, render_stacked_bar


def _render_chart_image(fig, width=700, height=350):
    """Render a Plotly figure to PNG bytes. Returns None on failure."""
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png", width=width, height=height)
    except Exception:
        return None


def _build_pdf(title, sections, chart_images=None):
    """Build a PDF from title, text sections, and optional chart images.

    sections: list of (heading, content) tuples
    chart_images: list of (label, png_bytes_or_None) tuples
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Charts
    if chart_images:
        for label, img_bytes in chart_images:
            if img_bytes is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                pdf.image(tmp_path, w=170)
                Path(tmp_path).unlink(missing_ok=True)
                pdf.ln(2)
            else:
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 6, f"[Chart unavailable: {label}]", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)

    # Text sections
    for heading, content in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for line in content:
            pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    return bytes(pdf.output())


def generate_pdf(sim_df, config: SimConfig, depots):
    """Generate a single-simulation PDF report. Returns bytes."""
    components = get_cost_components(sim_df)
    total_rev_miles = sim_df["distance_miles"].sum()
    total_cost = sum(components.values())
    avg_cpm = total_cost / total_rev_miles if total_rev_miles > 0 else 0

    # Chart
    fig = render_stacked_bar(sim_df)
    chart_img = _render_chart_image(fig)

    # Metrics section
    dh = sim_df["deadhead_miles"].sum()
    total_miles = total_rev_miles + dh
    metrics = [
        f"Total rides: {len(sim_df):,}",
        f"Revenue miles: {total_rev_miles:,.0f}",
        f"Deadhead miles: {dh:,.0f} ({dh/total_miles*100:.1f}%)" if total_miles > 0 else "Deadhead miles: 0",
        f"Average cost per mile: ${avg_cpm:.3f}",
        f"Total cost: ${total_cost:,.2f}",
        f"Charge events: {len(sim_df[sim_df['deadhead_miles'] > 0]):,}",
    ]

    # Cost breakdown
    breakdown = []
    for name, val in sorted(components.items(), key=lambda x: x[1], reverse=True):
        cpm = val / total_rev_miles if total_rev_miles > 0 else 0
        breakdown.append(f"  {name}: ${cpm:.3f}/mi (${val:,.2f} total)")

    # Depot info
    depot_lines = []
    for i, d in enumerate(depots):
        depot_lines.append(
            f"  {d.get('name', f'Depot {i+1}')}: "
            f"({d['lat']:.4f}, {d['lon']:.4f}), "
            f"{d.get('stalls', 0)} stalls"
        )

    # Assumptions
    assumptions = [
        f"  Battery: {config.battery_kwh} kWh",
        f"  Efficiency: {config.efficiency} mi/kWh",
        f"  Charge threshold: {config.charge_threshold}%",
        f"  Electricity: ${config.electricity_offpeak:.2f} off-peak / ${config.electricity_peak:.2f} peak",
        f"  Vehicle: ${config.purchase_price:,} / {config.lifetime_miles:,} mi lifetime",
        f"  Tolls: ${config.toll_per_mile:.2f}/mi, Insurance: ${config.insurance_per_mile:.2f}/mi",
    ]

    return _build_pdf(
        title="EV Cost Per Mile Report",
        sections=[
            ("Summary Metrics", metrics),
            ("Cost Breakdown (per revenue mile)", breakdown),
            ("Depot Locations", depot_lines),
            ("Assumptions", assumptions),
        ],
        chart_images=[("Cost breakdown chart", chart_img)],
    )


def generate_comparison_pdf(sim_a, sim_b, config: SimConfig, depots_a, depots_b):
    """Generate a comparison PDF report (2 pages). Returns bytes."""
    comp_a = get_cost_components(sim_a)
    comp_b = get_cost_components(sim_b)
    rev_a = sim_a["distance_miles"].sum()
    rev_b = sim_b["distance_miles"].sum()
    cpm_a = sum(comp_a.values()) / rev_a if rev_a > 0 else 0
    cpm_b = sum(comp_b.values()) / rev_b if rev_b > 0 else 0

    # Charts
    fig_a = render_stacked_bar(sim_a, height=300)
    fig_b = render_stacked_bar(sim_b, height=300)
    img_a = _render_chart_image(fig_a, width=600, height=300)
    img_b = _render_chart_image(fig_b, width=600, height=300)

    # Page 1: Side-by-side summary
    summary = [
        f"Config A: ${cpm_a:.3f}/mi  |  Config B: ${cpm_b:.3f}/mi",
        f"Difference: ${cpm_b - cpm_a:+.3f}/mi",
        "",
    ]

    # Per-component comparison
    all_keys = list(dict.fromkeys(list(comp_a.keys()) + list(comp_b.keys())))
    comparison_lines = []
    for k in all_keys:
        va = comp_a.get(k, 0) / rev_a if rev_a > 0 else 0
        vb = comp_b.get(k, 0) / rev_b if rev_b > 0 else 0
        diff = vb - va
        comparison_lines.append(f"  {k}: A=${va:.3f}  B=${vb:.3f}  delta=${diff:+.3f}")

    # Depot info
    depot_a_lines = [f"  {d.get('name', f'Depot {i+1}')}: ({d['lat']:.4f}, {d['lon']:.4f}), {d.get('stalls', 0)} stalls, ${d.get('lease_per_stall', config.lease_per_stall):,}/stall" for i, d in enumerate(depots_a)]
    depot_b_lines = [f"  {d.get('name', f'Depot {i+1}')}: ({d['lat']:.4f}, {d['lon']:.4f}), {d.get('stalls', 0)} stalls, ${d.get('lease_per_stall', config.lease_per_stall):,}/stall" for i, d in enumerate(depots_b)]

    return _build_pdf(
        title="EV Cost Per Mile Comparison Report",
        sections=[
            ("Summary", summary),
            ("Config A — Cost Breakdown", []),
            ("Config B — Cost Breakdown", []),
            ("Component Comparison (per revenue mile)", comparison_lines),
            ("Config A — Depots", depot_a_lines),
            ("Config B — Depots", depot_b_lines),
        ],
        chart_images=[
            ("Config A cost breakdown", img_a),
            ("Config B cost breakdown", img_b),
        ],
    )
