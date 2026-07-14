"""
app.py — Piattaforma di analisi ciclismo (Streamlit)
====================================================
Differenziatore vs Strava/TrainingPeaks: accanto a OGNI numero un badge di
affidabilita' (misurato / stimato / modellato).

Avvio:  streamlit run app.py
Dipendenze: streamlit plotly fitparse scipy pandas numpy requests
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
#  Badge affidabilita'                                                        #
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
#  SIDEBAR                                                                    #
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

st.sidebar.subheader("Sorgente dati")
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
        act = st.sidebar.text_input("Activity ID (es. i123456)")
        if key and act: raw = ca.load_intervals_icu(act, key)
except Exception as e:
    st.sidebar.error(f"Errore caricamento: {e}")

if raw is None:
    st.info("Seleziona una sorgente e carica i dati (o usa la ride demo).")
    st.stop()

# --------------------------------------------------------------------------- #
#  ELABORAZIONE SESSIONE                                                      #
# --------------------------------------------------------------------------- #
df = ca.to_1hz(raw)
power = df["power"].values if "power" in df else None
hr = df["hr"].values if "hr" in df else None
has_power = power is not None and np.nansum(power) > 0
if not has_power:
    st.warning("Nessun dato di potenza: molte metriche useranno la HR (meno affidabili).")

mmp = ca.mean_maximal_power(power) if has_power else pd.Series(dtype=float)
cp_res = ftp_res = map_m = lm = None
if has_power and len(mmp[(mmp.index>=120)&(mmp.index<=1200)]) >= 3:
    cp_res = ca.critical_power(mmp, model="3param")
    ftp_res = ca.estimate_ftp(mmp, cp=cp_res["cp"].value)
    map_m = ca.maximal_aerobic_power(mmp)
    lm = ca.load_metrics(power, ftp_res["ftp_recommended"].value)
ftp_val = ftp_res["ftp_recommended"].value if ftp_res else None

# --------------------------------------------------------------------------- #
#  HEADER                                                                     #
# --------------------------------------------------------------------------- #
st.title("Report allenamento")
c = st.columns(4)
c[0].metric("Durata", f"{len(df)/60:.0f} min")
if has_power:
    c[1].metric("Lavoro", f"{np.nansum(power)/1000:.0f} kJ")
    c[2].metric("Potenza media", f"{np.nanmean(power):.0f} W")
    c[3].metric("Potenza max", f"{np.nanmax(power):.0f} W")

tabs = st.tabs(["📈 Curva di potenza", "🎯 Soglie & Zone", "🔬 Analisi sessione",
                "❤️ Stato di forma", "🔥 Metabolismo & Nutrizione",
                "🏆 Classificazione", "📅 Settimana & Piano", "📆 Periodizzazione"])

# ---- 1. CURVA DI POTENZA -------------------------------------------------- #
with tabs[0]:
    st.markdown(f"### Curva di potenza {badge(Confidence.MEASURED)}", unsafe_allow_html=True)
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
        fig.update_layout(height=430, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        rows = [{"Durata": lbl, "Watt": f"{mmp[d]:.0f}", "W/kg": f"{mmp[d]/mass:.2f}"}
                for d, lbl in {1:"1s",5:"5s",15:"15s",60:"1min",300:"5min",1200:"20min",3600:"1h"}.items()
                if d in mmp.index]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Servono dati di potenza.")

# ---- 2. SOGLIE & ZONE ----------------------------------------------------- #
with tabs[1]:
    if ftp_res and cp_res:
        st.markdown("### Soglie")
        cc = st.columns(4)
        metric_card(cc[0], "FTP", ftp_res["ftp_recommended"])
        metric_card(cc[1], "Critical Power", cp_res["cp"])
        metric_card(cc[2], "W' (anaerobico)", ca.Metric(cp_res["w_prime"].value/1000, "kJ",
                    cp_res["w_prime"].confidence, cp_res["w_prime"].method), big=False)
        metric_card(cc[3], "MAP", map_m)
        with st.expander("Tutte le stime di FTP (dispersione)"):
            for k, m in ftp_res.items():
                if k == "ftp_recommended": continue
                st.markdown(f"- **{m.value:.0f} W** — {m.method} {badge(m.confidence)}"
                            + (f" · _{m.note}_" if m.note else ""), unsafe_allow_html=True)
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
        st.markdown("#### Carico")
        lc = st.columns(4)
        metric_card(lc[0], "Normalized Power", lm["normalized_power"])
        metric_card(lc[1], "Intensity Factor", lm["intensity_factor"], big=False)
        metric_card(lc[2], "TSS", lm["tss"])
        metric_card(lc[3], "Variability Index", lm["variability_index"], big=False)
    else:
        st.info("Per soglie e zone servono ≥3 sforzi massimali tra 2 e 20 min.")

# ---- 3. ANALISI SESSIONE (NEW) -------------------------------------------- #
with tabs[2]:
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
        st.caption("Intensità e fabbisogno di recupero sono assi distinti: una sessione può "
                   "essere intensa ma corta (recupero rapido).")

        st.divider()
        cL, cR = st.columns([2, 1])
        with cL:
            st.markdown("#### W' balance — quanto ti sei avvicinato al limite")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=wb["series"]/1000, mode="lines",
                                     line=dict(color="#0A45FA", width=1.5)))
            fig.add_hline(y=0, line_dash="dot", line_color="#B0392E")
            fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                              xaxis_title="Tempo (s)", yaxis_title="W'bal (kJ)")
            st.plotly_chart(fig, use_container_width=True)
        with cR:
            st.markdown("#### Segnali")
            metric_card(st, "W' consumato (picco)", wb["depleted_pct"], big=False)
            if hr is not None:
                dec = ti.aerobic_decoupling(power, hr)
                metric_card(st, "Decoupling aerobico", dec, big=False)
                st.caption("<5% = buona durabilità.")
        with st.expander("Evidenze (tempo per zona, %)"):
            st.write(wt["evidence_pct"])
    else:
        st.info("Servono potenza e stima CP per l'analisi della sessione.")

# ---- 4. STATO DI FORMA (VO2max) ------------------------------------------- #
with tabs[3]:
    if map_m:
        vo2 = ca.estimate_vo2max(athlete, map_m.value)
        st.markdown("### VO2max")
        vc = st.columns([1, 2])
        metric_card(vc[0], "VO2max", vo2["vo2max"])
        with vc[1]:
            st.markdown("**Metodi indipendenti** (quanto concordano):")
            for k, m in vo2.items():
                if k == "vo2max": continue
                st.markdown(f"- **{m.value:.1f}** {m.unit} — {m.method} {badge(m.confidence)}",
                            unsafe_allow_html=True)
            if not athlete.vo2max_lab:
                st.warning("⚠️ STIME da potenza. La VO2max vera si misura solo con analisi "
                           "dei gas espirati. Inserisci il valore di lab nella sidebar per usarlo.")
        v = vo2["vo2max"].value
        lvl = ("elite" if v>70 else "molto buono" if v>60 else "buono" if v>50
               else "nella media" if v>40 else "da migliorare")
        st.info(f"VO2max ~{v:.0f} mL/kg/min → livello aerobico: **{lvl}**")
    else:
        st.info("Serve la MAP (miglior 5 min o test rampa) per stimare la VO2max.")

# ---- 5. METABOLISMO & NUTRIZIONE ------------------------------------------ #
with tabs[4]:
    st.markdown("### Dispendio energetico")
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
                text=[f"{sub['pct_carb'].value:.0f}%", f"{sub['pct_fat'].value:.0f}%"],
                textposition="inside"))
            fig.update_layout(height=160, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="% energia")
            st.plotly_chart(fig, use_container_width=True)
            if sub["pct_fat"].confidence == Confidence.MODELED:
                st.warning("⚠️ Split MODELLATO da intensità (curva di popolazione). Il valore vero "
                           "richiede RER da metabolimetro; il tuo crossover reale può variare di 10-15 %VO2max.")
            fm = ca.fatmax(map_m.value, athlete)
            st.markdown(f"**FatMax:** {fm.value:.0f} W {badge(fm.confidence)} · _{fm.note}_",
                        unsafe_allow_html=True)
        st.divider()
        st.markdown("### Fabbisogno calorico")
        act = st.select_slider("Attività (vita non sportiva)", options=[1.2,1.375,1.55,1.725],
            value=1.55, format_func=lambda x:{1.2:"Sedentario",1.375:"Leggero",
            1.55:"Moderato",1.725:"Attivo"}[x])
        de = ca.daily_energy(athlete, kcal.value, activity_factor=act)
        dc = st.columns(2)
        metric_card(dc[0], "Metabolismo basale", de["bmr"])
        metric_card(dc[1], "Fabbisogno oggi (BMR+attività+allenamento)", de["tdee"])
        st.markdown(f"#### Piano nutrizionale {badge(Confidence.ESTIMATED)}", unsafe_allow_html=True)
        iff = lm["intensity_factor"].value if lm else 0.7
        fp = ca.fueling_plan(len(power) if has_power else len(df), iff, athlete)
        st.markdown(f"- **Prima:** {fp['pre_workout']}")
        st.markdown(f"- **Durante:** {fp['during_workout']}")
        st.markdown(f"- **Dopo:** {fp['post_workout']}")
        st.caption(fp["note"])
    else:
        st.info("Servono dati di potenza o HR.")

# ---- 6. CLASSIFICAZIONE (upgraded) ---------------------------------------- #
with tabs[5]:
    if has_power and len(mmp):
        rt = ca.rider_type_full(mmp, mass)
        st.markdown(f"## {rt['primary']} {badge(rt['confidence'])}", unsafe_allow_html=True)
        st.write(rt["reasoning"])
        q = rt.get("qualities", {})
        if q:
            labels = {"sprint":"Sprint (5s)","anaerobico":"Anaerobico (1m)",
                      "vo2max":"VO2max (5m)","soglia":"Soglia (20m)"}
            order = [k for k in labels if k in q]
            fig = go.Figure(go.Scatterpolar(r=[q[k]["score"] for k in order],
                theta=[labels[k] for k in order], fill="toself", line_color="#0A45FA"))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0,1.05], visible=True)),
                              height=350, margin=dict(l=40,r=40,t=20,b=20), showlegend=False)
            c1, c2 = st.columns([1,1])
            c1.plotly_chart(fig, use_container_width=True)
            c1.caption("Scala 0 (amatore) → 1 (top-10 TdF) per qualità.")
            with c2:
                st.markdown("#### Confronto con le categorie")
                cl = ca.classify_category(mmp, mass)
                dl = {5:"Sprint 5s",60:"Anaerobico 1min",300:"VO2max 5min",1200:"Soglia 20min"}
                st.dataframe(pd.DataFrame([{"Qualità":dl.get(d,f"{d}s"),"W/kg":i["w_kg"],
                            "Watt":i["watt"],"Livello":i["category_label"]}
                            for d,i in cl["per_duration"].items()]),
                            hide_index=True, use_container_width=True)
        st.warning("⚠️ ONESTÀ SUI BENCHMARK: i livelli amatoriali derivano dal Power Profile di "
                   "Coggan (robusti). I livelli pro / top-20 Grande Giro / top-10 Tour sono STIME da "
                   "letteratura e analisi di potenza sulle salite (SRM, VAM), NON da laboratorio. "
                   "Ordini di grandezza. Inoltre la classificazione è valida solo su una curva "
                   "STAGIONALE con sforzi massimali reali a tutte le durate — una singola uscita "
                   "quasi mai li contiene tutti (usa più uscite).")
    else:
        st.info("Servono dati di potenza per la classificazione.")

# ---- 7. SETTIMANA & PIANO (NEW) ------------------------------------------- #
with tabs[6]:
    st.markdown("### Andamento settimanale → allenamento consigliato")
    st.caption("Inserisci gli allenamenti della settimana (modificabile). In produzione questa "
               "tabella si popola dal tuo storico. Nessun dato viene salvato in questa demo.")
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

    seed = st.slider("Fitness di partenza (CTL iniziale, se hai storico pregresso)", 0, 100, 55)
    target = st.radio("Obiettivo della prossima sessione", ["limiter", "strength"], horizontal=True,
                      format_func=lambda x: "Allena il punto debole" if x=="limiter"
                      else "Asseconda il punto di forza")

    sessions = [ti.Session(day=(r["data"] if isinstance(r["data"], date) else pd.to_datetime(r["data"]).date()),
                           tss=float(r["TSS"]), if_=float(r["IF"]),
                           frac_low=r["%low"]/100, frac_mid=r["%mid"]/100, frac_high=r["%high"]/100)
                for _, r in edited.iterrows() if pd.notna(r["TSS"])]

    if sessions:
        pmc = ti.training_load(sessions, seed_ctl=seed, seed_atl=seed)
        ws = ti.weekly_summary(sessions)

        st.markdown("#### Fitness / Fatica / Forma (PMC)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pmc["day"], y=pmc["ctl"], name="Fitness (CTL)",
                                 line=dict(color="#0A45FA", width=2.5)))
        fig.add_trace(go.Scatter(x=pmc["day"], y=pmc["atl"], name="Fatica (ATL)",
                                 line=dict(color="#B0392E", width=1.5, dash="dot")))
        fig.add_trace(go.Bar(x=pmc["day"], y=pmc["tsb"], name="Forma (TSB)",
                             marker_color="#9aa4b2", opacity=0.5, yaxis="y2"))
        fig.update_layout(height=330, margin=dict(l=0,r=0,t=10,b=0),
                          yaxis=dict(title="CTL / ATL"),
                          yaxis2=dict(title="TSB", overlaying="y", side="right"),
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)

        tsb_now = pmc["tsb"].iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("TSS settimana", ws["tss_week"])
        m2.metric("Forma oggi (TSB)", f"{tsb_now:+.0f}", ti.tsb_label(tsb_now))
        m3.metric("Distribuzione", "", ws["distribution"])

        rt_for_rec = ca.rider_type_full(mmp, mass) if (has_power and len(mmp)) else {}
        rec = ti.recommend_next_workout(pmc, ws, rt_for_rec, ftp_val or 250, target=target)

        st.markdown("#### 🎯 Allenamento consigliato per la prossima uscita")
        r = rec["recommended"]
        st.success(f"**{r['name']}** — {r['prescription']}  · _{r['expected_tss']}_")
        st.markdown(f"**Perché:** {rec['rationale']}")
        if rec["alternatives"]:
            alt = rec["alternatives"][0]
            st.markdown(f"**Alternativa:** {alt['name']} — {alt['prescription']}")
        st.caption("⚠️ " + rec["disclaimer"])
    else:
        st.info("Aggiungi almeno un allenamento alla tabella.")

# ---- 8. PERIODIZZAZIONE (NEW) --------------------------------------------- #
with tabs[7]:
    st.markdown("### Periodizzazione verso la gara")
    st.caption("Pianifica a ritroso dalla data gara: fasi Base→Build→Taper (scarico 3:1), rampa "
               "di CTL sicura, e proiezione del PMC per arrivare con la freschezza giusta.")

    # CTL/ATL attuali stimati dalla tabella settimanale (scheda precedente)
    wk_df = st.session_state.get("week_df", demo_week())
    _sess = [ti.Session(day=(r["data"] if isinstance(r["data"], date) else pd.to_datetime(r["data"]).date()),
                        tss=float(r["TSS"])) for _, r in wk_df.iterrows() if pd.notna(r.get("TSS"))]
    if _sess:
        _pmc0 = ti.training_load(_sess, seed_ctl=55, seed_atl=55)
        ctl_def, atl_def = int(round(_pmc0["ctl"].iloc[-1])), int(round(_pmc0["atl"].iloc[-1]))
    else:
        ctl_def, atl_def = 50, 50

    i1, i2, i3 = st.columns(3)
    race = i1.date_input("Data gara", value=date.today() + timedelta(weeks=10))
    event = i2.selectbox("Tipo di evento", list(ti.EVENT_PROFILES.keys()), index=1)
    ramp = i3.slider("Rampa CTL / settimana (sicura 3-6)", 2, 8, 5)
    j1, j2 = st.columns(2)
    cur_ctl = j1.number_input("Fitness attuale (CTL)", 0, 160, ctl_def,
                              help="Preso dalla scheda Settimana & Piano; modificabile.")
    cur_atl = j2.number_input("Fatica attuale (ATL)", 0, 160, atl_def)

    plan = ti.periodized_plan(date.today(), race, current_ctl=cur_ctl, current_atl=cur_atl,
                              event=event, ftp=ftp_val or 250, safe_ramp=ramp,
                              rider_type=ca.rider_type_full(mmp, mass) if (has_power and len(mmp)) else {})

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
        s[3].metric("TSB al via (previsto)", f"{plan['race_day_tsb']:+.0f}",
                    f"target {plan['tsb_target'][0]}..{plan['tsb_target'][1]}")

        proj = plan["projection"]
        proj_x = pd.to_datetime(proj["day"])          # plotly gestisce datetime, non date
        st.markdown("#### Proiezione Fitness / Fatica / Forma fino alla gara")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=proj_x, y=proj["ctl"], name="Fitness (CTL)",
                                 line=dict(color="#0A45FA", width=2.5)))
        fig.add_trace(go.Scatter(x=proj_x, y=proj["atl"], name="Fatica (ATL)",
                                 line=dict(color="#B0392E", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=proj_x, y=proj["tsb"], name="Forma (TSB)",
                                 line=dict(color="#2e9e5b", width=1.5), yaxis="y2"))
        lo, hi = plan["tsb_target"]
        fig.add_hrect(y0=lo, y1=hi, fillcolor="#2e9e5b", opacity=0.10, line_width=0, yref="y2")
        fig.add_vline(x=pd.Timestamp(race), line_dash="dash", line_color="#0C1623",
                      annotation_text="GARA")
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(title="CTL / ATL"),
                          yaxis2=dict(title="TSB (forma)", overlaying="y", side="right"),
                          legend=dict(orientation="h", y=1.16))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("La banda verde è la freschezza (TSB) target per l'evento; la linea GARA "
                   "mostra dove atterri con questo piano.")

        tw = plan["this_week"]
        st.markdown(f"#### 🎯 Questa settimana — Fase **{tw['phase']}**")
        st.success(f"**{tw['session']['name']}** — {tw['session']['prescription']}  · _{tw['session']['expected_tss']}_")
        st.caption(tw["focus"])

        with st.expander("Piano settimana per settimana"):
            st.dataframe(pd.DataFrame([{"Sett": w["week"], "Fase": w["phase"],
                        "CTL target": w["target_ctl"], "TSS target": w["weekly_tss"],
                        "Focus": w["focus"], "Seduta chiave": w["session"]["name"]}
                        for w in plan["weeks"]]), hide_index=True, use_container_width=True)

        for wn in plan["warnings"]:
            st.warning("⚠️ " + wn)
        st.caption("⚠️ " + plan["disclaimer"])

# ---- footer legenda ------------------------------------------------------- #
st.divider()
st.markdown("<div style='font-size:.75rem'>" +
            " · ".join(f"{badge(c)} {BADGE[c][2]}" for c in Confidence) +
            "</div>", unsafe_allow_html=True)
