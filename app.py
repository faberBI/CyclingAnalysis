"""
app.py — Piattaforma di analisi ciclismo (Streamlit)
====================================================
Tre modalità, separate e chiare:
  📄 SINGOLA USCITA   → analisi del file selezionato (curva della uscita, zone,
                        tipo/difficoltà, W'bal, durability e intervalli di quella uscita).
  📊 STAGIONE         → TUTTE le attività aggregate: curva di potenza su tutto lo
                        storico + FTP, CP/W', MAP, VO2max, durability aggregata,
                        classificazione, forma nel tempo (PMC) e trend.
  📆 PIANIFICAZIONE   → analisi settimanale + periodizzazione verso una gara.

Differenziatore vs Strava/TrainingPeaks: ogni numero ha un badge di affidabilità.
Avvio: streamlit run app.py
"""
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import cycling_analytics as ca
import training_intelligence as ti
from cycling_analytics import Confidence

st.set_page_config(page_title="Cycling Lab", page_icon="🚴", layout="wide")

# --------------------------------------------------------------------------- #
#  Badge affidabilità                                                         #
# --------------------------------------------------------------------------- #
BADGE = {
    Confidence.MEASURED:  ("#0A45FA", "MISURATO",  "dai dati, nessuna assunzione"),
    Confidence.ESTIMATED: ("#E08A00", "STIMATO",   "equazione di popolazione, ±5-15%"),
    Confidence.MODELED:   ("#B0392E", "MODELLATO", "non misurabile senza laboratorio"),
}
def badge(conf):
    color, label, _ = BADGE[conf]
    return (f"<span style='background:{color};color:white;padding:2px 8px;border-radius:10px;"
            f"font-size:.68rem;font-weight:700;letter-spacing:.3px'>{label}</span>")

def metric_card(col, title, m, big=True):
    val = f"{m.value:,.0f}" if big else f"{m.value:,.2f}"
    col.markdown(
        f"<div style='line-height:1.25'><div style='font-size:.8rem;color:#5b6470'>{title}</div>"
        f"<div style='font-size:1.7rem;font-weight:800;color:#0C1623'>{val} "
        f"<span style='font-size:.9rem;font-weight:500;color:#8a939f'>{m.unit}</span></div>"
        f"{badge(m.confidence)}"
        f"<div style='font-size:.68rem;color:#8a939f;margin-top:3px'>{m.method}</div></div>",
        unsafe_allow_html=True)

def state_card(col, title, value, sub, color="#0C1623", size="1.7rem"):
    col.markdown(
        f"<div style='line-height:1.25'><div style='font-size:.8rem;color:#5b6470'>{title}</div>"
        f"<div style='font-size:{size};font-weight:800;color:{color}'>{value}</div>"
        f"<div style='font-size:.7rem;color:#8a939f;margin-top:3px'>{sub}</div></div>",
        unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
#  Dati demo                                                                  #
# --------------------------------------------------------------------------- #
def demo_ride():
    np.random.seed(7)
    def b(w, s, n=15): return np.random.normal(w, n, s).clip(0)
    p = np.concatenate([b(150,600), b(1180,5,40), b(180,120), b(620,60,25), b(200,180),
                        b(415,300,20), b(220,300), b(325,1200,18), b(205,3600,25)])
    hr = (0.35*p + 96 + np.random.normal(0,4,len(p))).clip(80,190)
    return pd.DataFrame({"t": np.arange(len(p)), "power": p, "hr": hr})

def demo_week():
    today = date.today()
    return pd.DataFrame([
        {"data": today-timedelta(days=6), "TSS": 95,  "IF": 0.72, "%low":90, "%mid":10, "%high":0},
        {"data": today-timedelta(days=5), "TSS": 30,  "IF": 0.50, "%low":100,"%mid":0,  "%high":0},
        {"data": today-timedelta(days=4), "TSS": 110, "IF": 0.98, "%low":40, "%mid":50, "%high":10},
        {"data": today-timedelta(days=3), "TSS": 85,  "IF": 0.75, "%low":85, "%mid":15, "%high":0},
        {"data": today-timedelta(days=2), "TSS": 130, "IF": 1.05, "%low":50, "%mid":20, "%high":30},
        {"data": today-timedelta(days=1), "TSS": 40,  "IF": 0.55, "%low":100,"%mid":0,  "%high":0},
    ])

# --------------------------------------------------------------------------- #
#  SIDEBAR: profilo + sorgente dati                                          #
# --------------------------------------------------------------------------- #
st.sidebar.title("🚴 Cycling Lab")
st.sidebar.caption("Analisi con affidabilità dichiarata")

st.sidebar.subheader("Profilo atleta")
mass = st.sidebar.number_input("Peso (kg)", 40.0, 120.0, 70.0, 0.5)
height = st.sidebar.number_input("Altezza (cm)", 140, 210, 178)
age = st.sidebar.number_input("Età", 15, 80, 34)
sex = st.sidebar.selectbox("Sesso", ["M", "F"])
with st.sidebar.expander("Dati fisiologici (migliorano l'affidabilità)"):
    hrmax = st.number_input("HR max misurata (0=stima)", 0, 220, 192)
    hrrest = st.number_input("HR riposo (0=n/d)", 0, 100, 48)
    lthr = st.number_input("LTHR da test (0=n/d)", 0, 220, 168)
    vo2lab = st.number_input("VO2max da lab (0=stima)", 0.0, 90.0, 0.0, 0.1)
    fatmax_lab = st.number_input("FatMax %VO2max lab (0=stima)", 0.0, 90.0, 0.0, 1.0)
athlete = ca.Athlete(mass_kg=mass, height_cm=height, age=age, sex=sex,
                     hr_max=hrmax or None, hr_rest=hrrest or None, lthr=lthr or None,
                     vo2max_lab=vo2lab or None, fatmax_pct_vo2max=fatmax_lab or None)

st.sidebar.subheader("Sorgente dati (singola uscita)")
src = st.sidebar.selectbox("Da dove", ["Ride demo", "File FIT (Garmin/Polar/Wahoo)",
                                       "File CSV", "intervals.icu (API)"])
raw = None
try:
    if src == "Ride demo":
        raw = demo_ride()
    elif src.startswith("File FIT"):
        up = st.sidebar.file_uploader("File .fit", type=["fit"])
        if up: raw = ca.load_fit(up)
    elif src == "File CSV":
        up = st.sidebar.file_uploader("File .csv", type=["csv"])
        if up: raw = ca.load_csv(up)
    else:
        key = st.sidebar.text_input("API key intervals.icu", type="password")
        aid = st.sidebar.text_input("ID atleta (es. i382978, oppure 0)", value="0",
                                    help="Impostazioni sviluppatore. '0' = atleta della chiave.")
        if key and aid:
            st.session_state["icu_creds"] = {"key": key, "aid": aid}
            acts = ca.list_intervals_activities(aid, key)
            if acts:
                labels = {f"{a['date']} · {a['name'] or a['type']}  ({a['id']})": a["id"] for a in acts}
                choice = st.sidebar.selectbox("Scegli un'uscita da analizzare", list(labels.keys()))
                raw = ca.load_intervals_icu(labels[choice], key)
            else:
                st.sidebar.info("Nessuna attività trovata negli ultimi mesi.")
except Exception as e:
    st.sidebar.error(f"Errore caricamento: {e}")

# --------------------------------------------------------------------------- #
#  Elaborazione della SINGOLA uscita (se presente)                           #
# --------------------------------------------------------------------------- #
df = power = hr = None
mmp = pd.Series(dtype=float)
cp_res = ftp_res = map_m = lm = None
ftp_val = None
if raw is not None:
    df = ca.to_1hz(raw)
    power = df["power"].values if "power" in df.columns else None
    hr = df["hr"].values if "hr" in df.columns else None
    has_power = power is not None and np.nansum(power) > 0
    if has_power:
        mmp = ca.mean_maximal_power(power)
        if len(mmp[(mmp.index >= 120) & (mmp.index <= 1200)]) >= 3:
            cp_res = ca.critical_power(mmp, model="3param")
            ftp_res = ca.estimate_ftp(mmp, cp=cp_res["cp"].value)
            map_m = ca.maximal_aerobic_power(mmp)
            lm = ca.load_metrics(power, ftp_res["ftp_recommended"].value)
        ftp_val = ftp_res["ftp_recommended"].value if ftp_res else None
else:
    has_power = False

# --------------------------------------------------------------------------- #
#  SELETTORE MODALITÀ                                                         #
# --------------------------------------------------------------------------- #
mode = st.radio("Vista", ["📄 Singola uscita", "📊 Stagione (tutte le attività)", "📆 Pianificazione"],
                horizontal=True, label_visibility="collapsed")

# ========================================================================== #
#  MODALITÀ 1 — SINGOLA USCITA                                               #
# ========================================================================== #
if mode.startswith("📄"):
    if raw is None:
        st.info("Seleziona una sorgente nella sidebar (o usa la ride demo) per analizzare un'uscita.")
        st.stop()
    if not has_power:
        st.warning("Nessun dato di potenza in questa uscita: le metriche di potenza non sono disponibili.")

    st.title("Analisi singola uscita")
    c = st.columns(4)
    c[0].metric("Durata", f"{len(df)/60:.0f} min")
    if has_power:
        c[1].metric("Lavoro", f"{np.nansum(power)/1000:.0f} kJ")
        c[2].metric("Potenza media", f"{np.nanmean(power):.0f} W")
        c[3].metric("Potenza max", f"{np.nanmax(power):.0f} W")
    st.caption("VO2max, classificazione e curva di potenza su TUTTE le uscite sono nella scheda "
               "📊 Stagione — lì hanno senso, qui no (una singola uscita non li rappresenta).")

    t = st.tabs(["📈 Curva della uscita", "🎯 Soglie & Zone", "🔬 Analisi sessione", "🔥 Metabolismo"])

    # -- Curva della uscita --
    with t[0]:
        st.markdown(f"### Curva di potenza di questa uscita {badge(Confidence.MEASURED)}",
                    unsafe_allow_html=True)
        if has_power and len(mmp):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mmp.index, y=mmp.values, mode="lines+markers",
                                     line=dict(color="#0A45FA", width=3)))
            if cp_res:
                fig.add_hline(y=cp_res["cp"].value, line_dash="dash", line_color="#B0392E",
                              annotation_text=f"CP {cp_res['cp'].value:.0f} W")
            fig.update_xaxes(type="log", title="Durata (scala log)",
                             tickvals=[1,5,15,60,300,1200,3600,10800],
                             ticktext=["1s","5s","15s","1m","5m","20m","1h","3h"])
            fig.update_yaxes(title="Potenza (W)")
            fig.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Questa è la curva della SINGOLA uscita, non il tuo profilo. "
                       "Per il profilo vero vai su 📊 Stagione.")
        else:
            st.info("Servono dati di potenza.")

    # -- Soglie & Zone (della uscita) --
    with t[1]:
        if ftp_res and cp_res:
            st.caption("NB: FTP/CP qui sono stimati SOLO da questa uscita. Il valore attendibile "
                       "è quello stagionale (scheda 📊 Stagione).")
            cc = st.columns(4)
            metric_card(cc[0], "FTP (da questa uscita)", ftp_res["ftp_recommended"])
            metric_card(cc[1], "Critical Power", cp_res["cp"])
            metric_card(cc[2], "W' (anaerobico)", ca.Metric(cp_res["w_prime"].value/1000, "kJ",
                        cp_res["w_prime"].confidence, cp_res["w_prime"].method), big=False)
            metric_card(cc[3], "MAP", map_m)
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                pz = ca.power_zones(ftp_val)
                st.markdown(f"#### Zone di potenza {badge(pz['confidence'])}", unsafe_allow_html=True)
                tiz = ca.time_in_zones(power, pz["zones"])
                st.dataframe(pd.DataFrame([{"Zona":n, "Range (W)":f"{lo}-{hi if hi<9000 else '∞'}",
                            "Min":f"{tiz[n]/60:.0f}"} for n,(lo,hi) in pz["zones"].items()]),
                            hide_index=True, use_container_width=True)
            with col2:
                hz = ca.hr_zones(athlete)
                st.markdown(f"#### Zone cardiache {badge(hz['confidence'])}", unsafe_allow_html=True)
                st.caption(hz["method"])
                tiz_hr = ca.time_in_zones(hr, hz["zones"]) if hr is not None else {n:0 for n in hz["zones"]}
                st.dataframe(pd.DataFrame([{"Zona":n, "Range (bpm)":f"{lo}-{hi if hi<900 else '∞'}",
                            "Min":f"{tiz_hr[n]/60:.0f}"} for n,(lo,hi) in hz["zones"].items()]),
                            hide_index=True, use_container_width=True)
            st.divider()
            st.markdown("#### Carico dell'uscita")
            lc = st.columns(4)
            metric_card(lc[0], "Normalized Power", lm["normalized_power"])
            metric_card(lc[1], "Intensity Factor", lm["intensity_factor"], big=False)
            metric_card(lc[2], "TSS", lm["tss"])
            metric_card(lc[3], "Variability Index", lm["variability_index"], big=False)
        else:
            st.info("Per soglie e zone servono ≥3 sforzi massimali tra 2 e 20 min in questa uscita.")

    # -- Analisi sessione --
    with t[2]:
        if lm and cp_res:
            wt = ti.classify_workout(power, ftp_val, len(power), lm["variability_index"].value)
            wb = ti.w_bal(power, cp_res["cp"].value, cp_res["w_prime"].value)
            diff = ti.session_difficulty(lm["tss"].value, lm["intensity_factor"].value,
                                         len(power), wb["depleted_pct"].value)
            st.markdown(f"### {wt['type']} {badge(wt['confidence'])}", unsafe_allow_html=True)
            st.caption(f"Rilevato dalla distribuzione nelle zone · alta intensità {wt['high_intensity_pct']}%")
            d1, d2, d3 = st.columns(3)
            d1.markdown(f"**Difficoltà**<br><span style='font-size:1.6rem;font-weight:800'>"
                        f"{diff['label']}</span> ({diff['score_1to5']}/5)", unsafe_allow_html=True)
            d2.markdown(f"**Intensità**<br><span style='font-size:1.6rem;font-weight:800'>"
                        f"{diff['intensity']}</span>", unsafe_allow_html=True)
            d3.markdown(f"**Fabbisogno recupero**<br><span style='font-size:1.6rem;font-weight:800'>"
                        f"{diff['recovery_demand']}</span>", unsafe_allow_html=True)
            st.divider()
            cL, cR = st.columns([2, 1])
            with cL:
                st.markdown("#### W' balance — quanto ti sei avvicinato al limite")
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=wb["series"]/1000, mode="lines",
                                         line=dict(color="#0A45FA", width=1.5)))
                fig.add_hline(y=0, line_dash="dot", line_color="#B0392E")
                fig.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0),
                                  xaxis_title="Tempo (s)", yaxis_title="W'bal (kJ)")
                st.plotly_chart(fig, use_container_width=True)
            with cR:
                st.markdown("#### Segnali")
                metric_card(st, "W' consumato (picco)", wb["depleted_pct"], big=False)
                if hr is not None:
                    metric_card(st, "Decoupling aerobico", ti.aerobic_decoupling(power, hr), big=False)
                    st.caption("<5% = buona durabilità (solo su uscite steady).")

            st.divider()
            st.markdown("#### 🔋 Durability di questa uscita (potenza da stanco)")
            kj_thr = st.slider("Soglia kJ per 'da stanco'", 500, 4000, 2000, 250, key="dur_single")
            dur = ca.durability(power, kj_threshold=kj_thr)
            if dur["reached"]:
                st.dataframe(pd.DataFrame([{"Durata":{5:"5s",15:"15s",60:"1min",300:"5min",1200:"20min"}.get(d,f"{d}s"),
                            "Da fresco (W)":v["fresh"],"Da stanco (W)":v["fatigued"],"Caduta %":f"{v['drop_pct']}%"}
                            for d,v in dur["per_duration"].items()]), hide_index=True, use_container_width=True)
                st.caption(f"kJ totali: {dur['total_kj']:.0f}. Caduta bassa = ottima resistenza. "
                           "Su singola uscita è indicativo; la durability aggregata è in 📊 Stagione.")
            else:
                st.info(f"Uscita troppo corta ({dur['total_kj']:.0f} kJ < soglia {kj_thr}). Abbassa la soglia.")

            st.divider()
            st.markdown("#### 🎯 Intervalli rilevati automaticamente")
            ivs = ti.detect_intervals(power, ftp_val)
            if ivs:
                st.dataframe(pd.DataFrame([{"#":k,"Inizio":f"{iv['start_s']//60}:{iv['start_s']%60:02d}",
                            "Durata":f"{iv['duration_s']//60}:{iv['duration_s']%60:02d}","Media (W)":iv["avg_power"],
                            "% FTP":f"{iv['pct_ftp']}%","Picco (W)":iv["peak_power"]} for k,iv in enumerate(ivs,1)]),
                            hide_index=True, use_container_width=True)
                st.caption(f"{len(ivs)} sforzi sopra il 102% FTP per ≥20 s.")
            else:
                st.info("Nessuno sforzo intenso rilevato (uscita perlopiù aerobica costante).")
        else:
            st.info("Servono potenza e stima CP per l'analisi della sessione.")

    # -- Metabolismo --
    with t[3]:
        st.markdown("### Dispendio energetico di questa uscita")
        kcal = (ca.calories_from_power(power) if has_power else
                ca.calories_from_hr(hr, athlete) if hr is not None else None)
        if kcal:
            ec = st.columns(3)
            metric_card(ec[0], "Calorie bruciate", kcal)
            if map_m:
                sub = ca.substrate_split(power, map_m.value, athlete, total_kcal=kcal.value)
                metric_card(ec[1], "Da carboidrati", sub["carb_g"])
                metric_card(ec[2], "Da grassi", sub["fat_g"])
                st.markdown(f"#### Ripartizione substrati {badge(sub['pct_fat'].confidence)}",
                            unsafe_allow_html=True)
                fig = go.Figure(go.Bar(x=[sub["pct_carb"].value, sub["pct_fat"].value],
                    y=["Carboidrati","Grassi"], orientation="h", marker_color=["#0A45FA","#E08A00"],
                    text=[f"{sub['pct_carb'].value:.0f}%", f"{sub['pct_fat'].value:.0f}%"], textposition="inside"))
                fig.update_layout(height=160, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="% energia")
                st.plotly_chart(fig, use_container_width=True)
                if sub["pct_fat"].confidence == Confidence.MODELED:
                    st.warning("⚠️ Split MODELLATO da intensità. Il valore vero richiede RER da metabolimetro.")
                fm = ca.fatmax(map_m.value, athlete)
                st.markdown(f"**FatMax:** {fm.value:.0f} W {badge(fm.confidence)} · _{fm.note}_",
                            unsafe_allow_html=True)

                # Curva di ottimizzazione del consumo di grassi
                foc = ca.fat_oxidation_curve(map_m.value, athlete)
                fc = foc["curve"]
                st.markdown(f"#### 🔥 Ottimizzazione consumo grassi {badge(foc['confidence'])}",
                            unsafe_allow_html=True)
                figf = go.Figure()
                figf.add_trace(go.Scatter(x=fc["watt"], y=fc["fat_g_min"], name="Grassi (g/min)",
                                          line=dict(color="#E08A00", width=3), fill="tozeroy",
                                          fillcolor="rgba(224,138,0,0.12)"))
                figf.add_trace(go.Scatter(x=fc["watt"], y=fc["cho_g_min"], name="Carboidrati (g/min)",
                                          line=dict(color="#0A45FA", width=2, dash="dot")))
                figf.add_vline(x=foc["fatmax_watt"], line_dash="dash", line_color="#B0392E",
                               annotation_text=f"max grassi ~{foc['fatmax_watt']:.0f}W")
                figf.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                   xaxis_title="Potenza (W)", yaxis_title="Ossidazione (g/min)",
                                   legend=dict(orientation="h", y=1.15))
                st.plotly_chart(figf, use_container_width=True, key="fatox")
                st.info(f"Bruci il massimo di grassi intorno a **{foc['fatmax_watt']:.0f} W** "
                        f"(~{foc['fatmax_pct']:.0f}% della MAP, ~{foc['fatmax_fat_g_min']:.2f} g/min): "
                        "è l'intensità da tenere per il fondo brucia-grassi e le uscite lunghe. "
                        f"_{foc['note']}_")
            st.divider()
            st.markdown("### Fabbisogno calorico")
            act = st.select_slider("Attività (vita non sportiva)", options=[1.2,1.375,1.55,1.725],
                value=1.55, format_func=lambda x:{1.2:"Sedentario",1.375:"Leggero",1.55:"Moderato",1.725:"Attivo"}[x])
            de = ca.daily_energy(athlete, kcal.value, activity_factor=act)
            dc = st.columns(2)
            metric_card(dc[0], "Metabolismo basale", de["bmr"])
            metric_card(dc[1], "Fabbisogno oggi", de["tdee"])
            st.markdown(f"#### Piano nutrizionale {badge(Confidence.ESTIMATED)}", unsafe_allow_html=True)
            iff = lm["intensity_factor"].value if lm else 0.7
            fp = ca.fueling_plan(len(power) if has_power else len(df), iff, athlete)
            st.markdown(f"- **Prima:** {fp['pre_workout']}")
            st.markdown(f"- **Durante:** {fp['during_workout']}")
            st.markdown(f"- **Dopo:** {fp['post_workout']}")
            st.caption(fp["note"])
        else:
            st.info("Servono dati di potenza o HR.")

# ========================================================================== #
#  MODALITÀ 2 — STAGIONE (TUTTE LE ATTIVITÀ)                                 #
# ========================================================================== #
elif mode.startswith("📊"):
    st.title("Stagione — tutte le attività")
    st.caption("La curva di potenza costruita su TUTTE le uscite, e da lì FTP, VO2max, CP/W', "
               "durability aggregata e classificazione. Più forma nel tempo e trend.")
    creds = st.session_state.get("icu_creds")
    if not creds:
        st.info("Seleziona la sorgente **intervals.icu (API)** nella sidebar e inserisci ID atleta + "
                "API key. L'analisi di stagione legge automaticamente tutto lo storico.")
        st.stop()

    cfg = st.columns(3)
    win = cfg[0].selectbox("Finestra", ["Ultimi 90 giorni","Ultimi 6 mesi","Ultimi 12 mesi","Tutto"], index=0)
    days = {"Ultimi 90 giorni":90,"Ultimi 6 mesi":183,"Ultimi 12 mesi":365,"Tutto":3650}[win]
    maxact = cfg[1].slider("Max attività (più = più lento)", 10, 200, 60)
    kj_thr = cfg[2].slider("Soglia kJ per durability", 500, 4000, 2000, 250)

    if st.button("🔄 Analizza tutto lo storico", type="primary"):
        try:
            with st.spinner("Scarico l'elenco delle attività..."):
                acts = ca.list_intervals_activities(creds["aid"], creds["key"], days_back=days, limit=3000)
            st.session_state["season_acts"] = acts
            npw = len([a for a in acts if a.get("has_power")][:maxact])
            prog = st.progress(0.0, text=f"Scarico e analizzo {npw} attività...")
            res = ti.analyze_season_from_intervals(athlete, creds["aid"], creds["key"], acts,
                    max_activities=maxact, kj_threshold=kj_thr,
                    progress_cb=lambda f: prog.progress(min(f, 1.0)))
            st.session_state["season"] = res
            # Wellness (HRV / HR a riposo / sonno) per il recupero autonomico
            try:
                well = ca.load_intervals_wellness(creds["aid"], creds["key"], days_back=60)
                st.session_state["readiness"] = ti.wellness_readiness(well)
            except Exception:
                st.session_state["readiness"] = None
            prog.empty()
        except Exception as e:
            st.error(f"Errore durante il caricamento: {e}")

    acts = st.session_state.get("season_acts")
    res = st.session_state.get("season")
    if not acts:
        st.info("Premi **Analizza tutto lo storico** per iniziare.")
        st.stop()

    season = res["season_curve"] if res else pd.Series(dtype=float)

    # --- calcolo UNA VOLTA: forma/recupero (PMC) e profilo (curva aggregata) ---
    pmc = ti.pmc_from_activities(acts)
    state = None
    if len(pmc):
        ctl_now = float(pmc["ctl"].iloc[-1]); atl_now = float(pmc["atl"].iloc[-1])
        tsb_now = float(pmc["tsb"].iloc[-1])
        state = {"ctl": ctl_now, "atl": atl_now, "tsb": tsb_now,
                 "rec_days": ti.recovery_forecast(ctl_now, atl_now)}
    profile = None
    if res and len(season) and len(season[(season.index >= 120) & (season.index <= 1200)]) >= 3:
        _cp = ca.critical_power(season, model="3param")
        _ftp = ca.estimate_ftp(season, cp=_cp["cp"].value)
        _map = ca.maximal_aerobic_power(season)
        _vo2 = ca.estimate_vo2max(athlete, _map.value)
        _fm = ca.fatmax(_map.value, athlete)
        profile = {"cp": _cp, "ftp": _ftp, "map": _map, "vo2": _vo2, "fatmax": _fm}

    stabs = st.tabs(["🧭 Cruscotto", "📈 Curva & Profilo", "🔋 Durability",
                     "🏆 Classificazione", "📉 Trend"])

    # -- Cruscotto: stato + profilo a colpo d'occhio --
    with stabs[0]:
        readiness = st.session_state.get("readiness")
        st.markdown("#### Stato di forma e recupero")
        if state:
            tsb = state["tsb"]
            fcol = ("#2e9e5b" if tsb > 5 else "#0C1623" if tsb > -10
                    else "#E08A00" if tsb > -30 else "#B0392E")
            base_rd = state["rec_days"]
            adj_rd = ti.adjusted_recovery_days(base_rd, readiness)
            sc = st.columns(4)
            state_card(sc[0], "Forma (TSB)", f"{tsb:+.0f}", ti.tsb_label(tsb), color=fcol)
            state_card(sc[1], "Recupero", ti.recovery_status(tsb), "stato (da carico)", size="1.3rem")
            rd_sub = "riposo per TSB ≥ +5"
            rd_color = "#0C1623"
            if readiness and readiness.get("have_data") and adj_rd != base_rd:
                rd_sub = f"carico {base_rd} + wellness +{adj_rd-base_rd}"
                rd_color = "#B0392E"
            state_card(sc[2], "Recupero in giorni",
                       "già fresco" if adj_rd == 0 else f"~{adj_rd} gg", rd_sub, color=rd_color)
            state_card(sc[3], "Fitness / Fatica", f"{state['ctl']:.0f} / {state['atl']:.0f}", "CTL / ATL")
        else:
            st.info("Dati insufficienti per lo stato di forma.")

        # Recupero autonomico (wellness)
        if readiness and readiness.get("have_data"):
            ov = readiness["overall"]
            ocolor = {"verde": "#2e9e5b", "ambra": "#E08A00", "rosso": "#B0392E"}.get(ov, "#0C1623")
            st.markdown("**Recupero autonomico (wellness)**")
            wc = st.columns(4)
            state_card(wc[0], "Prontezza", ov.upper(), "HRV + HR riposo + sonno", color=ocolor, size="1.3rem")
            state_card(wc[1], "HRV", readiness["hrv"], "vs baseline")
            state_card(wc[2], "HR a riposo", readiness["rhr"], "vs baseline")
            state_card(wc[3], "Sonno", readiness["sleep"], "ultimi giorni")
            st.caption(f"{readiness['flag']} {badge(Confidence.ESTIMATED)} — HRV-guided (soglie di popolazione).",
                       unsafe_allow_html=True)
        else:
            st.caption(f"Recupero stimato dal SOLO carico (TSB) {badge(Confidence.ESTIMATED)}. Con dati "
                       "wellness (HRV / HR a riposo / sonno) su intervals.icu il recupero si affina.",
                       unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Profilo (dalla curva di potenza su tutte le uscite)")
        if profile:
            pc = st.columns(5)
            metric_card(pc[0], "FTP", profile["ftp"]["ftp_recommended"])
            metric_card(pc[1], "VO2max", profile["vo2"]["vo2max"])
            metric_card(pc[2], "W'", ca.Metric(profile["cp"]["w_prime"].value/1000, "kJ",
                        profile["cp"]["w_prime"].confidence, profile["cp"]["w_prime"].method), big=False)
            metric_card(pc[3], "MAP", profile["map"])
            metric_card(pc[4], "FatMax", profile["fatmax"])
            st.caption("Curva di potenza, CP e dispersione FTP → scheda 📈 Curva & Profilo · "
                       "Durability → 🔋 · Che corridore sei → 🏆")
        else:
            st.info("Servono sforzi massimali 2-20 min nella finestra scelta per il profilo (FTP/VO2max/W'). "
                    "Allarga la finestra e ri-analizza.")

        st.divider()
        st.markdown("#### Volume del periodo")
        n = len(acts)
        tot_tss = sum(a.get("load") or 0 for a in acts)
        tot_h = sum(a.get("moving_time") or 0 for a in acts) / 3600
        tot_km = sum(a.get("distance") or 0 for a in acts) / 1000
        m = st.columns(4)
        m[0].metric("Attività", n)
        m[1].metric("Ore totali", f"{tot_h:.0f}")
        m[2].metric("Distanza", f"{tot_km:,.0f} km")
        m[3].metric("TSS totale", f"{tot_tss:,.0f}")

        if len(pmc):
            st.markdown("#### Forma nel tempo — tutta la storia (CTL / ATL / TSB)")
            px = pd.to_datetime(pmc["day"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=px, y=pmc["ctl"], name="Fitness (CTL)", line=dict(color="#0A45FA", width=2.5)))
            fig.add_trace(go.Scatter(x=px, y=pmc["atl"], name="Fatica (ATL)", line=dict(color="#B0392E", width=1.2, dash="dot")))
            fig.add_trace(go.Scatter(x=px, y=pmc["tsb"], name="Forma (TSB)", line=dict(color="#2e9e5b", width=1.2), yaxis="y2"))
            fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(title="CTL/ATL"),
                              yaxis2=dict(title="TSB", overlaying="y", side="right"), legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True, key="pmc_season")

        # Distribuzione di intensità (polarizzazione, Seiler 80/20)
        if res and profile and res.get("power_hist"):
            ftp_s = profile["ftp"]["ftp_recommended"].value
            pol = ti.polarization_from_hist(res["power_hist"]["edges"],
                                            res["power_hist"]["counts"], ftp_s)
            st.markdown(f"#### Distribuzione di intensità — polarizzazione {badge(pol['confidence'])}",
                        unsafe_allow_html=True)
            figpol = go.Figure()
            figpol.add_trace(go.Bar(x=[pol["pct_low"]], y=["stagione"], orientation="h",
                name="Low Z1-2", marker_color="#2e9e5b", text=f"{pol['pct_low']}%", textposition="inside"))
            figpol.add_trace(go.Bar(x=[pol["pct_mid"]], y=["stagione"], orientation="h",
                name="Mid soglia", marker_color="#E08A00", text=f"{pol['pct_mid']}%", textposition="inside"))
            figpol.add_trace(go.Bar(x=[pol["pct_high"]], y=["stagione"], orientation="h",
                name="High Z5+", marker_color="#B0392E", text=f"{pol['pct_high']}%", textposition="inside"))
            figpol.update_layout(barmode="stack", height=130, margin=dict(l=0, r=0, t=10, b=0),
                                 xaxis_title="% del tempo pedalato", yaxis=dict(visible=False),
                                 legend=dict(orientation="h", y=1.5))
            st.plotly_chart(figpol, use_container_width=True, key="polarization")
            st.caption(f"**{pol['label']}** · Low {pol['low_h']:.0f}h / Mid {pol['mid_h']:.0f}h / "
                       f"High {pol['high_h']:.0f}h. Modello 3-zone (Seiler): <80% FTP / 80-105% / >105%. "
                       "L'ideale polarizzato è ~80% low e poca 'zona grigia'.")

        # Heatmap del carico (calendario stile GitHub, TSS/giorno)
        if len(pmc):
            st.markdown("#### Costanza del carico — TSS per giorno")
            cal = pmc[["day", "tss"]].copy()
            cal["day"] = pd.to_datetime(cal["day"])
            start_monday = cal["day"].min() - pd.Timedelta(days=int(cal["day"].min().dayofweek))
            cal["wk"] = ((cal["day"] - start_monday).dt.days // 7).astype(int)
            cal["wd"] = cal["day"].dt.dayofweek
            nwk = int(cal["wk"].max()) + 1
            z = np.full((7, nwk), np.nan)
            hover = np.empty((7, nwk), dtype=object)
            for _, rr in cal.iterrows():
                z[int(rr["wd"]), int(rr["wk"])] = rr["tss"]
                hover[int(rr["wd"]), int(rr["wk"])] = f"{rr['day'].strftime('%d %b %Y')}: {rr['tss']:.0f} TSS"
            figcal = go.Figure(go.Heatmap(
                z=z, customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                colorscale=[[0, "#ebedf0"], [0.001, "#c6e48b"], [0.35, "#7bc96f"],
                            [0.7, "#239a3b"], [1, "#196127"]],
                xgap=2, ygap=2, showscale=True, zmin=0,
                y=["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]))
            figcal.update_layout(height=210, margin=dict(l=0, r=0, t=10, b=0),
                                 yaxis=dict(autorange="reversed"), xaxis=dict(visible=False))
            st.plotly_chart(figcal, use_container_width=True, key="heatmap")
            st.caption("Ogni cella è un giorno (verde più scuro = più carico). Le celle chiare sono "
                       "riposo/stop: colpo d'occhio su costanza e buchi.")
            st.markdown(f"#### Curva di potenza aggregata su {res['used']} attività {badge(Confidence.MEASURED)}",
                        unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=season.index, y=season.values, mode="lines", line=dict(color="#0A45FA", width=3)))
            fig.update_xaxes(type="log", title="Durata (scala log)", tickvals=[1,5,15,60,300,1200,3600,10800],
                             ticktext=["1s","5s","15s","1m","5m","20m","1h","3h"])
            fig.update_yaxes(title="Potenza (W)")
            fig.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="curve_season")
            if profile:
                cp = profile["cp"]; ftp = profile["ftp"]; mapm = profile["map"]; vo2 = profile["vo2"]
                st.markdown("#### Profilo (dalla curva aggregata)")
                cA = st.columns(5)
                metric_card(cA[0], "FTP stagionale", ftp["ftp_recommended"])
                metric_card(cA[1], "Critical Power", cp["cp"])
                metric_card(cA[2], "W'", ca.Metric(cp["w_prime"].value/1000, "kJ",
                            cp["w_prime"].confidence, cp["w_prime"].method), big=False)
                metric_card(cA[3], "MAP", mapm)
                metric_card(cA[4], "VO2max", vo2["vo2max"])
                cB = st.columns(5)
                metric_card(cB[0], "FatMax", profile["fatmax"])
                if not athlete.vo2max_lab:
                    st.caption("⚠️ VO2max è una STIMA da potenza (±10-15%). Il valore vero serve il metabolimetro.")
                with st.expander("Tutte le stime di FTP (dispersione)"):
                    for k, mm in ftp.items():
                        if k == "ftp_recommended": continue
                        st.markdown(f"- **{mm.value:.0f} W** — {mm.method} {badge(mm.confidence)}",
                                    unsafe_allow_html=True)
            else:
                st.info("Nella finestra scelta mancano sforzi massimali 2-20 min per stimare CP/FTP/VO2max. Allarga la finestra.")
        else:
            st.info("Nessuna curva disponibile. Premi 'Analizza tutto lo storico'.")

    # -- Durability aggregata --
    with stabs[2]:
        st.markdown(f"#### Durability aggregata — potenza da stanco {badge(Confidence.MEASURED)}",
                    unsafe_allow_html=True)
        st.caption(f"Curva fresca vs curva DOPO {res['kj_threshold'] if res else kj_thr:.0f} kJ, aggregata su "
                   f"{res['n_durable'] if res else 0} uscite abbastanza lunghe. Il differenziatore vs Strava/TP.")
        durab = res["durability"] if res else {}
        if durab:
            st.dataframe(pd.DataFrame([{"Durata":{5:"5s",15:"15s",60:"1min",300:"5min",1200:"20min"}.get(d,f"{d}s"),
                        "Da fresco (W)":v["fresh"],"Da stanco (W)":v["fatigued"],"Caduta %":f"{v['drop_pct']}%"}
                        for d,v in durab.items()]), hide_index=True, use_container_width=True)
            fresh_x = list(durab.keys())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fresh_x, y=[durab[d]["fresh"] for d in fresh_x],
                                     mode="lines+markers", name="Da fresco", line=dict(color="#0A45FA", width=3)))
            fig.add_trace(go.Scatter(x=fresh_x, y=[durab[d]["fatigued"] for d in fresh_x],
                                     mode="lines+markers", name="Da stanco", line=dict(color="#B0392E", width=3, dash="dot")))
            fig.update_xaxes(type="log", title="Durata (s)", tickvals=[5,15,60,300,1200],
                             ticktext=["5s","15s","1m","5m","20m"])
            fig.update_yaxes(title="Potenza (W)")
            fig.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0), legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True, key="durability_season")
            st.caption("Caduta bassa = reggi bene la fatica. CAVEAT: attendibile solo se dopo la soglia kJ "
                       "hai fatto sforzi intensi in quelle uscite.")
        else:
            st.info("Nessuna uscita ha superato la soglia kJ scelta. Abbassala e ri-analizza, "
                    "oppure servono uscite più lunghe.")

    # -- Classificazione (aggregata) --
    with stabs[3]:
        if len(season):
            rt = ca.rider_type_full(season, mass)
            st.markdown(f"## {rt['primary']} {badge(rt['confidence'])}", unsafe_allow_html=True)
            st.write(rt["reasoning"])
            q = rt.get("qualities", {})
            if q:
                labels = {"sprint":"Sprint (5s)","anaerobico":"Anaerobico (1m)","vo2max":"VO2max (5m)","soglia":"Soglia (20m)"}
                order = [k for k in labels if k in q]
                fig = go.Figure(go.Scatterpolar(r=[q[k]["score"] for k in order],
                    theta=[labels[k] for k in order], fill="toself", line_color="#0A45FA"))
                fig.update_layout(polar=dict(radialaxis=dict(range=[0,1.05], visible=True)),
                                  height=350, margin=dict(l=40,r=40,t=20,b=20), showlegend=False)
                c1, c2 = st.columns([1,1])
                c1.plotly_chart(fig, use_container_width=True, key="radar_season")
                c1.caption("Scala 0 (amatore) → 1 (top-10 TdF) per qualità.")
                with c2:
                    st.markdown("#### Confronto con le categorie")
                    cl = ca.classify_category(season, mass)
                    dl = {5:"Sprint 5s",60:"Anaerobico 1min",300:"VO2max 5min",1200:"Soglia 20min"}
                    st.dataframe(pd.DataFrame([{"Qualità":dl.get(d,f"{d}s"),"W/kg":i["w_kg"],
                                "Watt":i["watt"],"Livello":i["category_label"]} for d,i in cl["per_duration"].items()]),
                                hide_index=True, use_container_width=True)
            st.warning("⚠️ Livelli amatoriali = Power Profile di Coggan (robusti). Livelli pro / top-20 GT / "
                       "top-10 Tour = STIME da letteratura e analisi salite (SRM, VAM), NON da laboratorio: "
                       "ordini di grandezza.")

            # --- Confronto divertente con i pro + Pogačar ---
            st.divider()
            st.markdown("#### 🏆 Tu vs i professionisti (e Pogačar 👑)")
            pcmp = ca.pro_comparison(season, mass)
            if pcmp["rows"]:
                dur_opts = {r["durata"]: r for r in pcmp["rows"]}
                sel = st.selectbox("Durata da confrontare", list(dur_opts.keys()),
                                   index=len(dur_opts) - 1)
                r = dur_opts[sel]
                cats = [("🚴 Tu", r["tu"], "#0A45FA"), ("🟢 Continental", r["continental"], "#2e9e5b"),
                        ("🔵 Professional", r["professional"], "#0088CC"),
                        ("🟣 World Tour", r["world_tour"], "#7A3FF2"),
                        ("👑 Pogačar", r["pogacar"], "#E0A400")]
                figp = go.Figure(go.Bar(x=[c[1] for c in cats], y=[c[0] for c in cats],
                    orientation="h", marker_color=[c[2] for c in cats],
                    text=[f"{c[1]:.1f}" for c in cats], textposition="outside"))
                figp.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                                   xaxis_title=f"W/kg — {sel}", yaxis=dict(autorange="reversed"))
                st.plotly_chart(figp, use_container_width=True, key="procomp")
                pct = r["pct_pogacar"]
                if pct is not None:
                    emoji = ("🌱" if pct < 50 else "🚴" if pct < 70 else "💪" if pct < 85
                             else "🔥" if pct <= 100 else "🤯👑")
                    msg = ("continua a spingere!" if pct < 50 else "buon amatore" if pct < 70
                           else "sei forte!" if pct < 85 else "quasi da pro!" if pct <= 100
                           else "hai battuto Pogi?!")
                    st.success(f"{emoji} Sul **{sel}** sei al **{pct}% di Pogačar** — {msg}")
                st.caption("⚠️ " + pcmp["note"])

            # --- Dove sei tra gli amatori: low / middle / top ---
            st.divider()
            st.markdown("#### Dove sei tra gli amatori")
            if profile:
                ftp_wkg = profile["ftp"]["ftp_recommended"].value / mass
                am = ca.classify_amateur(ftp_wkg)
                emj = {"Amatore base (low)": "🌱", "Amatore intermedio (mid)": "🚴",
                       "Amatore avanzato (top)": "💪", "Agonista / Elite amat.": "🏆"}.get(am["tier"], "🚴")
                st.markdown(f"### {emj} {am['tier']} · {ftp_wkg:.2f} W/kg FTP {badge(am['confidence'])}",
                            unsafe_allow_html=True)
                figa = go.Figure()
                band_colors = ["#e8f0eb", "#bfe0cc", "#7fc79f", "#3fae6e"]
                for (name, lo, hi), col in zip(am["bands"], band_colors):
                    figa.add_vrect(x0=max(lo, 2.0), x1=min(hi, 5.0), fillcolor=col, opacity=0.75,
                                   line_width=0, annotation_text=name.split(" (")[0],
                                   annotation_position="top left", annotation=dict(font_size=9))
                figa.add_vline(x=min(ftp_wkg, 5.0), line_color="#B0392E", line_width=3,
                               annotation_text=f"tu {ftp_wkg:.2f}")
                figa.add_trace(go.Scatter(x=[2.0, 5.0], y=[0, 0], mode="lines",
                                          line=dict(width=0), showlegend=False))
                figa.update_layout(height=150, margin=dict(l=0, r=0, t=28, b=0),
                                   xaxis_title="FTP W/kg", xaxis_range=[2.0, 5.0],
                                   yaxis=dict(visible=False, range=[-1, 1]))
                st.plotly_chart(figa, use_container_width=True, key="amateur")
                st.caption("Fasce (uomini, indicative): base(low) <3.1 · intermedio(mid) 3.1-3.8 · "
                           "avanzato(top) 3.8-4.5 · agonista >4.5 W/kg.")
            else:
                st.info("Serve la FTP stagionale (curva aggregata) per la classifica amatori.")
        else:
            st.info("Nessuna curva disponibile.")

    # -- Trend --
    with stabs[4]:
        trends = res["trends"] if res else None
        if trends is not None and len(trends):
            st.markdown("#### Trend nel tempo (il film, non la foto)")
            st.caption("Stai migliorando? eFTP e VO2max per uscita, Efficiency Factor (potenza/HR: "
                       "sale se la fitness aerobica migliora) e decoupling (più basso = più efficiente).")
            specs = [("eftp", "eFTP stimata", "#0A45FA", "W"),
                     ("vo2max", "VO2max stimata", "#2e9e5b", "mL/kg/min"),
                     ("ef", "Efficiency Factor (NP/HR)", "#7A3FF2", "W/bpm"),
                     ("decoupling", "Decoupling Pw:Hr", "#E08A00", "%")]
            for i in range(0, len(specs), 2):
                cols = st.columns(2)
                for col, (field, title, color, unit) in zip(cols, specs[i:i+2]):
                    with col:
                        sub = trends.dropna(subset=[field]) if field in trends.columns else trends.iloc[0:0]
                        if len(sub) >= 2:
                            fig = go.Figure(go.Scatter(x=sub["date"], y=sub[field], mode="markers+lines",
                                line=dict(color=color, width=1.5), marker=dict(size=5)))
                            fig.update_layout(height=230, margin=dict(l=0, r=0, t=32, b=0),
                                title=dict(text=f"{title} ({unit})", font=dict(size=13)), showlegend=False)
                            st.plotly_chart(fig, use_container_width=True, key=f"trend_{field}")
                        else:
                            st.caption(f"{title}: dati insufficienti.")
            st.caption("L'Efficiency Factor è più significativo sulle uscite aerobiche steady; "
                       "sulle sessioni a intervalli è più rumoroso.")
        else:
            st.info("Nessun trend disponibile. Premi 'Analizza tutto lo storico'.")

# ========================================================================== #
#  MODALITÀ 3 — PIANIFICAZIONE                                               #
# ========================================================================== #
else:
    st.title("Pianificazione")
    ptabs = st.tabs(["📅 Settimana & consiglio", "📆 Periodizzazione verso la gara"])

    # -- Settimana & consiglio --
    with ptabs[0]:
        st.caption("Inserisci gli allenamenti della settimana (modificabile). In produzione si popola "
                   "dallo storico. Nessun dato salvato in questa demo.")
        if "week_df" not in st.session_state:
            st.session_state.week_df = demo_week()
        edited = st.data_editor(st.session_state.week_df, num_rows="dynamic",
            use_container_width=True, hide_index=True,
            column_config={"data": st.column_config.DateColumn("Data"),
                           "TSS": st.column_config.NumberColumn("TSS", min_value=0),
                           "IF": st.column_config.NumberColumn("IF", format="%.2f"),
                           "%low": st.column_config.NumberColumn("% Z1-2"),
                           "%mid": st.column_config.NumberColumn("% Z3-4"),
                           "%high": st.column_config.NumberColumn("% Z5+")})
        seed = st.slider("Fitness di partenza (CTL iniziale)", 0, 100, 55)
        target = st.radio("Obiettivo prossima sessione", ["limiter", "strength"], horizontal=True,
            format_func=lambda x: "Allena il punto debole" if x=="limiter" else "Asseconda il punto di forza")
        sessions = [ti.Session(day=(r["data"] if isinstance(r["data"], date) else pd.to_datetime(r["data"]).date()),
                    tss=float(r["TSS"]), if_=float(r["IF"]), frac_low=r["%low"]/100, frac_mid=r["%mid"]/100,
                    frac_high=r["%high"]/100) for _, r in edited.iterrows() if pd.notna(r["TSS"])]
        if sessions:
            pmc = ti.training_load(sessions, seed_ctl=seed, seed_atl=seed)
            ws = ti.weekly_summary(sessions)
            st.markdown("#### Fitness / Fatica / Forma (PMC)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pmc["day"], y=pmc["ctl"], name="Fitness (CTL)", line=dict(color="#0A45FA", width=2.5)))
            fig.add_trace(go.Scatter(x=pmc["day"], y=pmc["atl"], name="Fatica (ATL)", line=dict(color="#B0392E", width=1.5, dash="dot")))
            fig.add_trace(go.Bar(x=pmc["day"], y=pmc["tsb"], name="Forma (TSB)", marker_color="#9aa4b2", opacity=0.5, yaxis="y2"))
            fig.update_layout(height=330, margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(title="CTL/ATL"),
                              yaxis2=dict(title="TSB", overlaying="y", side="right"), legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True, key="pmc_week")
            tsb_now = pmc["tsb"].iloc[-1]
            mm = st.columns(3)
            mm[0].metric("TSS settimana", ws["tss_week"])
            mm[1].metric("Forma oggi (TSB)", f"{tsb_now:+.0f}", ti.tsb_label(tsb_now))
            mm[2].metric("Distribuzione", "", ws["distribution"])
            rt_rec = ca.rider_type_full(st.session_state["season"]["season_curve"], mass) \
                if st.session_state.get("season") and len(st.session_state["season"]["season_curve"]) \
                else (ca.rider_type_full(mmp, mass) if (has_power and len(mmp)) else {})
            rec = ti.recommend_next_workout(pmc, ws, rt_rec, ftp_val or 250, target=target,
                                            readiness=st.session_state.get("readiness"))
            st.markdown("#### 🎯 Allenamento consigliato per la prossima uscita")
            r = rec["recommended"]
            st.success(f"**{r['name']}** — {r['prescription']}  · _{r['expected_tss']}_")
            st.markdown(f"**Perché:** {rec['rationale']}")
            if rec["alternatives"]:
                st.markdown(f"**Alternativa:** {rec['alternatives'][0]['name']} — {rec['alternatives'][0]['prescription']}")
            st.caption("⚠️ " + rec["disclaimer"])
        else:
            st.info("Aggiungi almeno un allenamento.")

    # -- Periodizzazione --
    with ptabs[1]:
        st.caption("Pianifica a ritroso dalla data gara: fasi Base→Build→Taper, e proiezione del PMC "
                   "per arrivare con la freschezza giusta.")
        wk_df = st.session_state.get("week_df", demo_week())
        _sess = [ti.Session(day=(r["data"] if isinstance(r["data"], date) else pd.to_datetime(r["data"]).date()),
                 tss=float(r["TSS"])) for _, r in wk_df.iterrows() if pd.notna(r.get("TSS"))]
        if st.session_state.get("season"):
            _pmc0 = ti.pmc_from_activities(st.session_state.get("season_acts", []))
            ctl_def = int(round(_pmc0["ctl"].iloc[-1])) if len(_pmc0) else 55
            atl_def = int(round(_pmc0["atl"].iloc[-1])) if len(_pmc0) else 55
        elif _sess:
            _p = ti.training_load(_sess, seed_ctl=55, seed_atl=55)
            ctl_def, atl_def = int(round(_p["ctl"].iloc[-1])), int(round(_p["atl"].iloc[-1]))
        else:
            ctl_def, atl_def = 50, 50
        i1, i2, i3 = st.columns(3)
        race = i1.date_input("Data gara", value=date.today()+timedelta(weeks=10))
        event = i2.selectbox("Tipo di evento", list(ti.EVENT_PROFILES.keys()), index=1)
        ramp = i3.slider("Rampa CTL / settimana (sicura 3-6)", 2, 8, 5)
        j1, j2 = st.columns(2)
        cur_ctl = j1.number_input("Fitness attuale (CTL)", 0, 160, ctl_def)
        cur_atl = j2.number_input("Fatica attuale (ATL)", 0, 160, atl_def)
        rt_for = ca.rider_type_full(st.session_state["season"]["season_curve"], mass) \
            if st.session_state.get("season") and len(st.session_state["season"]["season_curve"]) \
            else (ca.rider_type_full(mmp, mass) if (has_power and len(mmp)) else {})
        plan = ti.periodized_plan(date.today(), race, current_ctl=cur_ctl, current_atl=cur_atl,
                                  event=event, ftp=ftp_val or 250, safe_ramp=ramp, rider_type=rt_for)
        if "error" in plan:
            st.error(plan["error"])
        else:
            v = plan["verdict"]
            (st.success if v.startswith("✅") else st.warning)(v)
            ps = plan["phase_structure"]
            s = st.columns(4)
            s[0].metric("Settimane alla gara", plan["weeks_until"])
            s[1].metric("Struttura", f"{ps['base_weeks']}B / {ps['build_weeks']}Bu / {ps['taper_weeks']}T")
            s[2].metric("Picco CTL previsto", plan["peak_ctl"])
            s[3].metric("TSB al via", f"{plan['race_day_tsb']:+.0f}", f"target {plan['tsb_target'][0]}..{plan['tsb_target'][1]}")
            proj = plan["projection"]; proj_x = pd.to_datetime(proj["day"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=proj_x, y=proj["ctl"], name="Fitness (CTL)", line=dict(color="#0A45FA", width=2.5)))
            fig.add_trace(go.Scatter(x=proj_x, y=proj["atl"], name="Fatica (ATL)", line=dict(color="#B0392E", width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=proj_x, y=proj["tsb"], name="Forma (TSB)", line=dict(color="#2e9e5b", width=1.5), yaxis="y2"))
            lo, hi = plan["tsb_target"]
            fig.add_hrect(y0=lo, y1=hi, fillcolor="#2e9e5b", opacity=0.10, line_width=0, yref="y2")
            fig.add_vline(x=pd.Timestamp(race), line_dash="dash", line_color="#0C1623", annotation_text="GARA")
            fig.update_layout(height=340, margin=dict(l=0,r=0,t=10,b=0), yaxis=dict(title="CTL/ATL"),
                              yaxis2=dict(title="TSB", overlaying="y", side="right"), legend=dict(orientation="h", y=1.16))
            st.plotly_chart(fig, use_container_width=True, key="periodization")
            tw = plan["this_week"]
            st.markdown(f"#### 🎯 Questa settimana — Fase **{tw['phase']}**")
            st.success(f"**{tw['session']['name']}** — {tw['session']['prescription']}  · _{tw['session']['expected_tss']}_")
            st.caption(tw["focus"])
            with st.expander("Piano settimana per settimana"):
                st.dataframe(pd.DataFrame([{"Sett":w["week"],"Fase":w["phase"],"CTL target":w["target_ctl"],
                            "TSS target":w["weekly_tss"],"Focus":w["focus"],"Seduta":w["session"]["name"]}
                            for w in plan["weeks"]]), hide_index=True, use_container_width=True)
            for wn in plan["warnings"]:
                st.warning("⚠️ " + wn)
            st.caption("⚠️ " + plan["disclaimer"])

# ---- footer legenda ------------------------------------------------------- #
st.divider()
st.markdown("<div style='font-size:.75rem'>" +
            " · ".join(f"{badge(c)} {BADGE[c][2]}" for c in Confidence) + "</div>",
            unsafe_allow_html=True)
