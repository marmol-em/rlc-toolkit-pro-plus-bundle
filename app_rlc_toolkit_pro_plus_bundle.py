# ==========================================================
# R-L-C TRANSMISSION LINE TOOLKIT — PRO+ (with Bundled Conductors)
# ----------------------------------------------------------
# Developed for: EE 421 - Power System Analysis (Lab)
# Activity: Power System Modelling Part 1 (Parameters)
# Author: Ella Mae M. Marmol (with assistant support)
# ----------------------------------------------------------
# Highlights
#   • Resistance with temperature correction (per km & total)
#   • Inductance & Capacitance (per km & total)
#   • Auto-computed GMR & GMD
#   • Switchable Bundled Conductors (n = 1..6) with bundle spacing
#   • Frequency-aware XL and BC (per km)
#   • Beginner-first UI + clean result cards + geometry sketch + trend plots
#   • Fully SI (Ω·m, m, mm, km, H, F)
# ==========================================================

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------- #
# Physical constants (SI)
# ----------------------------- #
MU0 = 4 * math.pi * 1e-7          # H/m
EPS0 = 8.854187817e-12            # F/m
TEMP_CONST = {"Copper": 234.5, "Aluminum": 228.1}

# ----------------------------- #
# Page config + light CSS for cards
# ----------------------------- #
st.set_page_config(page_title="R-L-C Toolkit (Pro+ Bundle)", page_icon="⚡", layout="wide")

# ==== Global Unit System ====
st.sidebar.header("🌍 Units")
unit_system = st.sidebar.selectbox(
    "Unit system",
    ["SI (m, mm, km)", "US (ft, inch, mile, kcmil)"],
    index=0,
    key="units_system"
)
show_dual = st.sidebar.checkbox("Show dual units in results (SI + US)", value=True, key="units_dual")

# Conversions
INCH_TO_M    = 0.0254
FT_TO_M      = 0.3048
MILE_TO_M    = 1609.344
KCMIL_TO_MM2 = 0.5067075  # 1 kcmil = 0.5067075 mm²
MM2_TO_M2    = 1e-6

def to_si_radius(val, unit_system):
    # radius: SI=mm, US=inch
    return (val/1000.0) if unit_system.startswith("SI") else (val * INCH_TO_M)

def to_si_spacing(val, unit_system):
    # spacing/height/coordinates: SI=m, US=ft
    return val if unit_system.startswith("SI") else (val * FT_TO_M)

def to_si_length(val, unit_system):
    # line length: SI=km, US=mile
    return (val*1000.0) if unit_system.startswith("SI") else (val * MILE_TO_M)

def to_si_area(val, unit_system):
    # metallic area: SI=mm², US=kcmil
    if unit_system.startswith("SI"):
        return val * MM2_TO_M2
    mm2 = val * KCMIL_TO_MM2
    return mm2 * MM2_TO_M2

def to_si_rho(val, rho_unit):  # resistivity: Ω·m or Ω·ft
    return val if rho_unit == "Ω·m" else (val * FT_TO_M)

def fmt_dual(si_value, si_label, us_value=None, us_label=None):
    if show_dual and us_value is not None:
        return f"{si_value} {si_label}  |  {us_value} {us_label}"
    return f"{si_value} {si_label}"

st.markdown(
    """
<style>
.result-card {border-radius: 16px; padding: 14px 16px; margin-bottom: 10px; background:#f7f9fc; border:1px solid #e9eef5}
.result-title {font-weight:600; font-size:0.95rem; color:#334155; margin-bottom:6px}
.result-value {font-weight:800; font-size:1.15rem; color:#0f172a}
.section-note {font-size:0.9rem; color:#475569}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------- #
# Helper functions
# ----------------------------- #

def pretty_card(label: str, value: str):
    """Render a soft card for a result metric."""
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-title">{label}</div>
          <div class="result-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Resistance helpers --- #

def rho_corrected_constantC(rho1_ohm_m: float, T1: float, T2: float, C_const: float) -> float:
    """Temperature-corrected resistivity using the 'constant C' model.
    ρ(T) = ρ_ref * (T + C) / (T_ref + C)
    """
    return rho1_ohm_m * (T2 + C_const) / (T1 + C_const)

# --- Geometry helpers --- #

def gmr_from_radius(radius_m: float, user_gmr_m: float = 0.0) -> float:
    """GMR = user value, else 0.7788*r for a solid round conductor."""
    return 0.7788 * radius_m if user_gmr_m == 0 else user_gmr_m

def gmd_single(d_m: float) -> float:
    """GMD for single-phase two-wire line (spacing d)."""
    return d_m

def gmd_three(xA, yA, xB, yB, xC, yC):
    """GMD for three-phase (transposed average): (Dab * Dbc * Dca)^(1/3). Also returns the three spacings."""
    Dab = math.hypot(xA - xB, yA - yB)
    Dbc = math.hypot(xB - xC, yB - yC)
    Dca = math.hypot(xC - xA, yC - yA)
    return (Dab * Dbc * Dca) ** (1 / 3), Dab, Dbc, Dca

# --- Bundled conductor helpers --- #

def bundle_eq_gmr(gmr_single_m: float, bundle_n: int, bundle_spacing_m: float) -> float:
    """Equivalent GMR for a symmetrical bundle of n subconductors with adjacent spacing d_b.
    Common approximation: GMR_eq = (gmr_single * d_b^(n-1))^(1/n)
    Works for n >= 1. For n == 1, returns gmr_single.
    """
    if bundle_n <= 1:
        return gmr_single_m
    return (gmr_single_m * (bundle_spacing_m ** (bundle_n - 1))) ** (1.0 / bundle_n)

def bundle_eq_radius(radius_m: float, bundle_n: int, bundle_spacing_m: float) -> float:
    """Equivalent radius for capacitance when using a symmetrical bundle.
    r_eq = (r * d_b^(n-1))^(1/n)  (consistent with standard approximations)
    """
    if bundle_n <= 1:
        return radius_m
    return (radius_m * (bundle_spacing_m ** (bundle_n - 1))) ** (1.0 / bundle_n)

# --- Parameter formulas --- #

def L_per_m_from_GMD_GMR(GMD_m: float, GMR_m: float) -> float:
    """Inductance per meter: L = (μ0 / 2π) ln(GMD / GMR)"""
    return (MU0 / (2 * math.pi)) * math.log(GMD_m / GMR_m)

def C_per_m_single(d_m: float, r_m: float, h_m: float) -> float:
    """Capacitance per meter for single-phase using image method with height (optional)."""
    Deq = math.sqrt(d_m**2 + (2 * h_m)**2) if h_m and h_m > 0 else d_m
    return (2 * math.pi * EPS0) / math.log(Deq / r_m)

def C_per_m_three(GMD_m: float, r_m: float, yA: float, yB: float, yC: float) -> float:
    """Capacitance per meter for three-phase using image method (height via 2*y)."""
    Dp = (2 * yA * 2 * yB * 2 * yC) ** (1 / 3)
    Deq = math.sqrt(GMD_m * Dp)
    return (2 * math.pi * EPS0) / math.log(Deq / r_m)

# ----------------------------- #
# Sidebar: global beginner guide
# ----------------------------- #
with st.sidebar:
    st.header("🧭 How to use (Beginner)")
    st.markdown(
        """
1) Pick a tab (Resistance, Inductance, Capacitance).  
2) Enter your line data (units shown beside fields).  
3) Read the result cards (per-km & total).  
4) Go to **Summary** to download a CSV for your report.
        """
    )
    st.markdown("---")
    st.markdown("**Tip:** If unsure, keep defaults and change one input at a time.")

# ----------------------------- #
# Header
# ----------------------------- #
st.title("⚡ R–L–C Transmission Line Toolkit — Pro+ (Bundled)")
st.caption("Beginner-friendly • Research-grade formulas • Beautiful outputs")

# ----------------------------- #
# Tabs
# ----------------------------- #
tabR, tabL, tabC, tabS = st.tabs(["🧮 Resistance", "🌀 Inductance", "⚡ Capacitance", "📊 Summary & Plots"])

# ==========================================================
# Resistance Tab
# ==========================================================
with tabR:
    st.subheader("🧮 Resistance (with temperature correction)")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        material = st.selectbox("Conductor material", ["Copper", "Aluminum"])
# dynamic labels
AREA_LABEL   = "Metallic area (mm²)" if unit_system.startswith("SI") else "Metallic area (kcmil)"
LENGTH_LABEL_R = "Line length (km)" if unit_system.startswith("SI") else "Line length (mile)"
RHO_UNIT_OPTS = ["Ω·m", "Ω·ft"]

rho1 = st.number_input("ρ₁ at T₁ (value)", value=1.724e-8 if material == "Copper" else 2.826e-8, format="%.6e")
rho_unit = st.selectbox("Resistivity unit", RHO_UNIT_OPTS, index=0, key="res_rho_unit")
area_in = st.number_input(AREA_LABEL, value=300.0, min_value=0.1)
length_in_R = st.number_input(LENGTH_LABEL_R, value=10.0, min_value=0.001, key="res_len")
    with c2:
        T1 = st.number_input("Reference temperature T₁ (°C)", value=20.0)
        T2 = st.number_input("Operating temperature T₂ (°C)", value=50.0)
        st.caption(f"Temperature constant θ = **{TEMP_CONST[material]} °C** for {material}")

    area_m2 = to_si_area(area_in, unit_system)
    rho1_SI = to_si_rho(rho1, rho_unit)
rho2 = rho_corrected_constantC(rho1_SI, T1, T2, TEMP_CONST[material])

    R_per_km = rho2 / area_m2 * 1000.0
    length_m_R = to_si_length(length_in_R, unit_system)
R_total = (rho2 / area_m2) * length_m_R

    st.markdown("#### Results")
    a, b, c = st.columns(3)
    with a:
        pretty_card("Corrected ρ₂", f"{rho2:.4e} Ω·m")
    with b:
        pretty_card("R per km", f"{R_per_km:.6f} Ω/km")
    with c:
        pretty_card("Total R", f"{R_total:.6f} Ω")

    st.markdown('<div class="section-note">R increases with temperature; larger area reduces R.</div>', unsafe_allow_html=True)

# ==========================================================
# Inductance Tab (with Bundled switch)
# ==========================================================
with tabL:
    st.subheader("🌀 Inductance (per-km & total) + GMR/GMD + Bundled switch + Sketch")
    st.markdown("Select system type, set geometry, optionally enable bundled conductors, and (optionally) adjust frequency for Xₗ.")

    # Inputs
    sys_type_L = st.radio("System", ["Single-phase (two-wire)", "Three-phase (transposed)"], horizontal=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        RADIUS_LABEL   = "Conductor radius (mm)" if unit_system.startswith("SI") else "Conductor radius (inch)"
radius_in = st.number_input(RADIUS_LABEL, value=10.0, min_value=0.1, key="ind_radius")
r_m = to_si_radius(radius_in, unit_system)
    with c2:
        user_gmr = st.number_input("GMR (m) — 0 = auto (0.7788·r)", value=0.0, min_value=0.0, format="%.6f", key="ind_gmr")
        gmr_single = gmr_from_radius(r_m, user_gmr)
    with c3:
        LENGTH_LABEL_L = "Line length (km)" if unit_system.startswith("SI") else "Line length (mile)"
length_in_L = st.number_input(LENGTH_LABEL_L, value=10.0, min_value=0.001, key="ind_len")
        freq = st.number_input("Frequency (Hz) for Xₗ", value=60.0, min_value=1.0, key="ind_freq")

    # Bundled switch
    bundle_mode = st.toggle("Enable Bundled Conductors (per phase)", key="ind_bundle")
    if bundle_mode:
        bundle_n = st.number_input("Number of subconductors (n)", min_value=2, max_value=6, value=4, step=1, key="ind_n")
        DB_LABEL = "Spacing between subconductors d_b (m)" if unit_system.startswith("SI") else "Spacing between subconductors d_b (ft)"
bundle_spacing_in = st.number_input(DB_LABEL, min_value=0.05, value=0.4, step=0.05, key="ind_db")
db_m = to_si_spacing(bundle_spacing_in, unit_system)
GMR_used = bundle_eq_gmr(gmr_single, bundle_n, db_m)
        st.caption("Using **GMR_eq** for bundle: GMR_eq = (GMR_single × d_b^(n−1))^(1/n)")
    else:
        bundle_n = 1
        bundle_spacing_m = 0.0
        GMR_used = gmr_single

    # Geometry
    if sys_type_L.startswith("Single"):
        SPACING_LABEL  = "Spacing d (m) between the two conductors" if unit_system.startswith("SI") else "Spacing d (ft) between the two conductors"
d_in = st.number_input(SPACING_LABEL, value=2.0, min_value=0.01, key="ind_d")
d_m = to_si_spacing(d_in, unit_system)
GMD = gmd_single(d_m)
        Dab = Dbc = Dca = None
        y_for_plot = 10.0  # for a simple sketch
    else:
        st.markdown("**Phase coordinates (m):**")
        cA, cB, cC = st.columns(3)
        with cA:
            xA = st.number_input("xA", value=0.0, format="%.3f", key="ind_xA")
            yA = st.number_input("yA", value=20.0, min_value=0.1, format="%.3f", key="ind_yA")
        with cB:
            xB = st.number_input("xB", value=8.0, format="%.3f", key="ind_xB")
            yB = st.number_input("yB", value=20.0, min_value=0.1, format="%.3f", key="ind_yB")
        with cC:
            xC = st.number_input("xC", value=16.0, format="%.3f", key="ind_xC")
            yC = st.number_input("yC", value=20.0, min_value=0.1, format="%.3f", key="ind_yC")
        # convert coords to SI if US
xA, yA = to_si_spacing(xA, unit_system), to_si_spacing(yA, unit_system)
xB, yB = to_si_spacing(xB, unit_system), to_si_spacing(yB, unit_system)
xC, yC = to_si_spacing(xC, unit_system), to_si_spacing(yC, unit_system)
GMD, Dab, Dbc, Dca = gmd_three(xA, yA, xB, yB, xC, yC)
        st.caption(f"Phase spacings: Dab={Dab:.3f} m • Dbc={Dbc:.3f} m • Dca={Dca:.3f} m")

    # Calculations
    L_per_m = L_per_m_from_GMD_GMR(GMD, GMR_used)
    L_per_km = L_per_m * 1000.0
    length_m_L = to_si_length(length_in_L, unit_system)
L_total = L_per_m * length_m_L
    X_L = 2 * math.pi * freq * L_per_km  # Ω/km

    # Cards
    st.markdown("#### Results")
    a, b, c = st.columns(3)
    with a:
        pretty_card("GMR used", f"{GMR_used:.6f} m" + ("  (bundle)" if bundle_mode else ""))
    with b:
        pretty_card("GMD", f"{GMD:.6f} m")
    with c:
        pretty_card("L per km", f"{L_per_km:.6e} H/km")
    d1, d2 = st.columns(2)
    with d1:
        pretty_card("Total L", f"{L_total:.6e} H")
    with d2:
        pretty_card("Xₗ per km @ f", f"{X_L:.6f} Ω/km")

    # Geometry sketch (with simple bundle dots if enabled)
    st.markdown("#### Geometry sketch")
    fig = plt.figure()
    ax = plt.gca()
    if sys_type_L.startswith("Single"):
        # draw two conductors at same height for visualization
        ax.scatter([0, d], [y_for_plot, y_for_plot])
        ax.text(0, y_for_plot + 0.6, "Cond 1", ha="center")
        ax.text(d, y_for_plot + 0.6, "Cond 2", ha="center")
        ax.hlines(y=y_for_plot, xmin=0, xmax=d, linestyles="dashed")
        if bundle_mode:
            # bundle shown as small offsets around each conductor center
            rad = bundle_spacing_m / 2.0
            for xc in [0, d]:
                for k in range(bundle_n):
                    ang = 2 * math.pi * k / bundle_n
                    bx = xc + rad * math.cos(ang)
                    by = y_for_plot + rad * math.sin(ang)
                    ax.scatter(bx, by, s=10)
        plt.title("Single-phase geometry")
    else:
        # base phase centers
        ax.scatter([xA, xB, xC], [yA, yB, yC])
        ax.text(xA, yA + 0.6, "A", ha="center")
        ax.text(xB, yB + 0.6, "B", ha="center")
        ax.text(xC, yC + 0.6, "C", ha="center")
        if bundle_mode:
            rad = bundle_spacing_m / 2.0
            for xc, yc in [(xA, yA), (xB, yB), (xC, yC)]:
                for k in range(bundle_n):
                    ang = 2 * math.pi * k / bundle_n
                    bx = xc + rad * math.cos(ang)
                    by = yc + rad * math.sin(ang)
                    ax.scatter(bx, by, s=10)
        plt.title("Three-phase geometry (phase centers & bundles)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True, linewidth=0.4)
    st.pyplot(fig)

    # Trend: L vs spacing (quick exploratory)
    st.markdown("#### Explore: Inductance vs spacing")
    smin = st.slider("Sweep start (m)", 0.5, 50.0, 2.0, key="L_smin")
    smax = st.slider("Sweep end (m)", smin + 0.5, 100.0, max(16.0, smin + 4.0), key="L_smax")
    pts = st.slider("Points", 5, 100, 25, key="L_pts")

    sweep = np.linspace(smin, smax, pts)
    if sys_type_L.startswith("Single"):
        L_vals = [L_per_m_from_GMD_GMR(gmd_single(x), GMR_used) * 1000.0 for x in sweep]
        xlab = "Spacing d (m)"
    else:
        L_vals = []
        for x in sweep:
            # simple symmetric family: (0,20), (x,20), (2x,20)
            GMD_demo, *_ = gmd_three(0, 20, x, 20, 2 * x, 20)
            L_vals.append(L_per_m_from_GMD_GMR(GMD_demo, GMR_used) * 1000.0)
        xlab = "Horizontal spacing parameter (m)"

    fig2 = plt.figure()
    plt.plot(sweep, L_vals)
    plt.xlabel(xlab)
    plt.ylabel("L per km (H/km)")
    plt.title("Inductance vs spacing (with current bundle settings)")
    plt.grid(True, linewidth=0.4)
    st.pyplot(fig2)

    # Persist for Summary
    st.session_state["L_per_km"] = L_per_km
    st.session_state["L_total"] = L_total

# ==========================================================
# Capacitance Tab (with Bundled switch)
# ==========================================================
with tabC:
    st.subheader("⚡ Capacitance (per-km & total) + GMD′ + Bundled switch + Plots")
    st.markdown("Pick system, geometry, height, optionally enable bundled conductors, and see B_C @ frequency.")

    sys_type_C = st.radio("System", ["Single-phase (two-wire)", "Three-phase (transposed)"], horizontal=True, key="cap_sys")

    c1, c2, c3 = st.columns(3)
    with c1:
        RADIUS_LABEL_C = "Conductor radius (mm)" if unit_system.startswith("SI") else "Conductor radius (inch)"
radius_in_c = st.number_input(RADIUS_LABEL_C, value=10.0, min_value=0.1, key="cap_rmm")
r_m_c = to_si_radius(radius_in_c, unit_system)
    with c2:
        HEIGHT_LABEL = "Average conductor height above ground (m)" if unit_system.startswith("SI") else "Average conductor height above ground (ft)"
height_in = st.number_input(HEIGHT_LABEL, value=12.0, min_value=0.0)
height_m = to_si_spacing(height_in, unit_system)
    with c3:
        LENGTH_LABEL_C = "Line length (km)" if unit_system.startswith("SI") else "Line length (mile)"
length_in_C = st.number_input(LENGTH_LABEL_C, value=10.0, min_value=0.001, key="cap_len")
        freq_c = st.number_input("Frequency (Hz) for B_C", value=60.0, min_value=1.0, key="cap_f")

    # Bundled switch for capacitance (affects equivalent radius)
    bundle_mode_c = st.toggle("Enable Bundled Conductors (per phase)", key="cap_bundle")
    if bundle_mode_c:
        bundle_n_c = st.number_input("Number of subconductors (n)", min_value=2, max_value=6, value=4, step=1, key="cap_n")
        DB_LABEL_C = "Spacing between subconductors d_b (m)" if unit_system.startswith("SI") else "Spacing between subconductors d_b (ft)"
bundle_spacing_in_c = st.number_input(DB_LABEL_C, min_value=0.05, value=0.4, step=0.05, key="cap_db")
db_m_c = to_si_spacing(bundle_spacing_in_c, unit_system)
r_used = bundle_eq_radius(r_m_c, bundle_n_c, db_m_c)
        st.caption("Using **r_eq** for bundle: r_eq = (r × d_b^(n−1))^(1/n)")
    else:
        bundle_n_c = 1
        bundle_spacing_m_c = 0.0
        r_used = r_m_c

    # Geometry and calculation
    length_m_C = to_si_length(length_in_C, unit_system)

    if sys_type_C.startswith("Single"):
        SPACING_LABEL_C = "Spacing d (m)" if unit_system.startswith("SI") else "Spacing d (ft)"
dC_in = st.number_input(SPACING_LABEL_C, value=2.5, min_value=0.01, key="cap_d")
dC_m = to_si_spacing(dC_in, unit_system)
GMD_eff = dC_m
C_per_m = C_per_m_single(dC_m, r_used, height_m)
    else:
        st.markdown("**Phase coordinates (m):**")
        cA, cB, cC = st.columns(3)
        with cA:
            xA = st.number_input("xA", value=0.0, format="%.3f", key="cap_xA")
            yA = st.number_input("yA", value=20.0, min_value=0.1, format="%.3f", key="cap_yA")
        with cB:
            xB = st.number_input("xB", value=8.0, format="%.3f", key="cap_xB")
            yB = st.number_input("yB", value=20.0, min_value=0.1, format="%.3f", key="cap_yB")
        with cC:
            xC = st.number_input("xC", value=16.0, format="%.3f", key="cap_xC")
            yC = st.number_input("yC", value=20.0, min_value=0.1, format="%.3f", key="cap_yC")

        # convert to SI if US
xA, yA = to_si_spacing(xA, unit_system), to_si_spacing(yA, unit_system)
xB, yB = to_si_spacing(xB, unit_system), to_si_spacing(yB, unit_system)
xC, yC = to_si_spacing(xC, unit_system), to_si_spacing(yC, unit_system)
GMD_eff, Dab_c, Dbc_c, Dca_c = gmd_three(xA, yA, xB, yB, xC, yC)
C_per_m = C_per_m_three(GMD_eff, r_used, yA, yB, yC)
        st.caption(f"Phase spacings: Dab={Dab_c:.3f} m • Dbc={Dbc_c:.3f} m • Dca={Dca_c:.3f} m")

    C_per_km = C_per_m * 1000.0
    C_total = C_per_m * length_m_C
    B_C = 2 * math.pi * freq_c * C_per_km  # S/km

    # Cards
    st.markdown("#### Results")
    a, b, c = st.columns(3)
    with a:
        pretty_card("Effective GMD (for C)", f"{GMD_eff:.6f} m")
    with b:
        pretty_card("C per km", f"{C_per_km:.6e} F/km")
    with c:
        pretty_card("Total C", f"{C_total:.6e} F")
    pretty_card("B_C per km @ f", f"{B_C:.6e} S/km")

    # Trend plots
    st.markdown("#### Explore: Capacitance trends")
    if sys_type_C.startswith("Single"):
        # Sweep height
        hmin = st.slider("Height sweep start (m)", 1.0, 40.0, 6.0, key="cap_hmin")
        hmax = st.slider("Height sweep end (m)", hmin + 0.5, 80.0, max(30.0, hmin + 4.0), key="cap_hmax")
        pts = st.slider("Points", 5, 100, 25, key="cap_pts")
        hs = np.linspace(hmin, hmax, pts)
        Cv = [C_per_m_single(dC, r_used, h) * 1000.0 for h in hs]  # F/km
        figC = plt.figure()
        plt.plot(hs, Cv)
        plt.xlabel("Height h (m)")
        plt.ylabel("C per km (F/km)")
        plt.title("Capacitance vs height (single-phase)")
        plt.grid(True, linewidth=0.4)
        st.pyplot(figC)
    else:
        # Sweep spacing family
        dmin = st.slider("Spacing sweep start (m)", 0.5, 50.0, 2.0, key="cap_dmin")
        dmax = st.slider("Spacing sweep end (m)", dmin + 0.5, 100.0, max(16.0, dmin + 4.0), key="cap_dmax")
        pts = st.slider("Points ", 5, 100, 25, key="cap_pts2")
        ds = np.linspace(dmin, dmax, pts)
        Cv = []
        for dv in ds:
            GMD_demo, *_ = gmd_three(0, 20, dv, 20, 2 * dv, 20)
            Cv.append(C_per_m_three(GMD_demo, r_used, 20, 20, 20) * 1000.0)
        figC2 = plt.figure()
        plt.plot(ds, Cv)
        plt.xlabel("Horizontal spacing parameter (m)")
        plt.ylabel("C per km (F/km)")
        plt.title("Capacitance vs spacing (three-phase)")
        plt.grid(True, linewidth=0.4)
        st.pyplot(figC2)

    # Persist for Summary
    st.session_state["C_per_km"] = C_per_km
    st.session_state["C_total"] = C_total

# ==========================================================
# Summary Tab
# ==========================================================
with tabS:
    st.subheader("📊 Summary & Export")
    st.markdown("All computed values (most recent from each tab).")

    # Bring values if they exist
    R_per_km = locals().get("R_per_km", float("nan"))
    R_total = locals().get("R_total", float("nan"))
    L_per_km = st.session_state.get("L_per_km", float("nan"))
    L_total = st.session_state.get("L_total", float("nan"))
    C_per_km = st.session_state.get("C_per_km", float("nan"))
    C_total = st.session_state.get("C_total", float("nan"))

    # Surge impedance (if L & C available)
    Zc = float("nan")
    if (not math.isnan(L_per_km)) and (not math.isnan(C_per_km)) and L_per_km > 0 and C_per_km > 0:
        Zc = math.sqrt(L_per_km / C_per_km)

    c1, c2, c3 = st.columns(3)
    with c1:
        pretty_card("Resistance (Ω/km)", f"{R_per_km:.6f}")
    with c2:
        pretty_card("Inductance (H/km)", f"{L_per_km:.6e}")
    with c3:
        pretty_card("Capacitance (F/km)", f"{C_per_km:.6e}")

    d1, d2, d3 = st.columns(3)
    with d1:
        pretty_card("Total R (Ω)", f"{R_total:.6f}")
    with d2:
        pretty_card("Total L (H)", f"{L_total:.6e}")
    with d3:
        pretty_card("Total C (F)", f"{C_total:.6e}")

    pretty_card("Surge Impedance √(L/C) (Ω)", f"{Zc:.2f}" if not math.isnan(Zc) else "—")

    df = pd.DataFrame([
        {
            "R_per_km_ohm_per_km": R_per_km,
            "R_total_ohm": R_total,
            "L_per_km_H_per_km": L_per_km,
            "L_total_H": L_total,
            "C_per_km_F_per_km": C_per_km,
            "C_total_F": C_total,
            "Surge_Impedance_Ohm": Zc if not math.isnan(Zc) else "",
        }
    ])

    st.dataframe(df, use_container_width=True)
    st.download_button(
        "📥 Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="RLC_summary.csv",
        mime="text/csv",
    )

    st.markdown(
        '<div class="section-note">For your report: include one screenshot per tab, plus the CSV table here. '\
        'Briefly discuss sensitivities shown in the plots, and explain the bundling impact (L↓, C↑).</div>',
        unsafe_allow_html=True,
    )

