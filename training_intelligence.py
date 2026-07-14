"""
training_intelligence.py
========================
Analisi della singola sessione + analisi settimanale + raccomandazione.

Stessa filosofia del motore: ogni output dichiara l'affidabilita'.
Tutto qui e' ESTIMATED/MODELED: sono euristiche su principi allenanti consolidati,
NON un piano periodizzato verso una gara. Utile come bussola, non come coach.

Riferimenti:
- Skiba et al. 2012/2015; Clarke & Skiba 2013 (W' balance)
- Banister / Coggan (Performance Management Chart: CTL, ATL, TSB)
- Seiler (allenamento polarizzato 80/20)
- Friel (decoupling aerobico Pw:Hr)
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
import numpy as np
import pandas as pd

from cycling_analytics import Confidence, Metric, COGGAN_POWER_ZONES, time_in_zones, power_zones


# --------------------------------------------------------------------------- #
#  1. W' BALANCE  — quanto sei andato vicino al limite (difficolta' fisiologica)
# --------------------------------------------------------------------------- #
def w_bal(power: np.ndarray, cp: float, w_prime: float) -> dict:
    """
    Bilancio di W' istante per istante (modello differenziale Clarke-Skiba).
    Sopra CP: si svuota. Sotto CP: si ricarica (esponenziale, limitato a W').
    Il MINIMO raggiunto = quanto ti sei avvicinato all'esaurimento anaerobico.
    W'bal negativo = hai superato la capacita' del modello (sforzo sovra-massimale
    o CP/W' sottostimati).
    """
    p = np.nan_to_num(power, nan=0.0)
    bal = np.empty(len(p))
    cur = w_prime
    for i, pw in enumerate(p):
        if pw >= cp:
            cur -= (pw - cp)
        else:
            cur += (cp - pw) * (w_prime - cur) / w_prime
        cur = min(cur, w_prime)
        bal[i] = cur
    min_bal = float(bal.min())
    depleted_pct = (w_prime - min_bal) / w_prime * 100
    return {
        "series": bal,
        "min_wbal": Metric(min_bal, "J", Confidence.ESTIMATED, "W'bal min (Clarke-Skiba)",
                           "quanto sei sceso nel serbatoio anaerobico"),
        "depleted_pct": Metric(min(depleted_pct, 100), "%", Confidence.ESTIMATED,
                               "% di W' consumato al punto piu' profondo",
                               ">90% = sforzo quasi massimale; >100% (clip) = supra-CP oltre modello"),
    }


# --------------------------------------------------------------------------- #
#  2. DECOUPLING AEROBICO (Pw:Hr) — deriva cardiaca / durabilita'             #
# --------------------------------------------------------------------------- #
def aerobic_decoupling(power: np.ndarray, hr: np.ndarray) -> Metric:
    """
    Confronta l'efficienza (potenza/HR) tra prima e seconda meta' della sessione.
    >5% = decoupling: la HR sale rispetto alla potenza (fatica / poca durabilita'
    / sforzo troppo intenso per il fondo). Significativo soprattutto su uscite
    aerobiche steady; su allenamenti a intervalli e' meno interpretabile.
    """
    p = np.nan_to_num(power, nan=0.0)
    h = np.asarray(hr, dtype=float)
    n = min(len(p), len(h))
    half = n // 2
    def ef(a, b):
        m = ~np.isnan(b)
        return a[m].mean() / b[m].mean() if b[m].mean() else np.nan
    ef1 = ef(p[:half], h[:half])
    ef2 = ef(p[half:n], h[half:n])
    dec = (ef1 - ef2) / ef1 * 100 if ef1 else 0.0
    return Metric(float(dec), "%", Confidence.ESTIMATED, "Pw:Hr prima vs seconda meta'",
                  "<5% buona durabilita'; >5% deriva cardiaca")


# --------------------------------------------------------------------------- #
#  3. TIPO DI ALLENAMENTO — riconoscimento automatico                         #
# --------------------------------------------------------------------------- #
def classify_workout(power: np.ndarray, ftp: float, duration_s: int,
                     variability_index: float) -> dict:
    """
    Riconosce il tipo di sessione dalla distribuzione nelle zone + struttura.
    Euristica trasparente (come fanno TrainingPeaks/intervals.icu): mostra anche
    le evidenze (frazione di tempo per zona) cosi' l'utente puo' giudicare.
    """
    pz = power_zones(ftp)["zones"]
    tiz = time_in_zones(power, pz)
    total = sum(tiz.values()) or 1
    frac = {k: v / total for k, v in tiz.items()}

    z1 = frac.get("Z1 Recupero attivo", 0)
    z2 = frac.get("Z2 Fondo", 0)
    z3 = frac.get("Z3 Tempo/Medio", 0)
    z4 = frac.get("Z4 Soglia", 0)
    z5 = frac.get("Z5 VO2max", 0)
    z6 = frac.get("Z6 Anaerobico", 0)
    z7 = frac.get("Z7 Neuromuscolare", 0)
    high = z5 + z6 + z7
    mins = duration_s / 60

    if duration_s < 1200 and (high > 0.15 or z4 > 0.2):
        wtype = "Attivazione / Sessione breve intensa"
    elif z1 > 0.65 and (z1 + z2) > 0.85:
        wtype = "Recupero attivo"
    elif z5 > 0.08 or z6 > 0.05:
        wtype = "VO2max / Intervalli intensi"
    elif z4 > 0.15:
        wtype = "Soglia" + (" (a intervalli)" if variability_index > 1.12 else " (steady)")
    elif z3 > 0.20:
        wtype = "Tempo / Sweet-spot"
    elif (z1 + z2) > 0.75:
        wtype = ("Fondo lungo (endurance)" if mins > 150 else "Fondo (endurance)")
    elif variability_index > 1.2 and (z6 + z7) > 0.03:
        wtype = "Misto / Fartlek (stop&go)"
    else:
        wtype = "Misto"

    return {
        "type": wtype,
        "confidence": Confidence.ESTIMATED,
        "evidence_pct": {k: round(v * 100, 1) for k, v in frac.items() if v > 0.01},
        "high_intensity_pct": round(high * 100, 1),
    }


# --------------------------------------------------------------------------- #
#  4. DIFFICOLTA' DELLA SESSIONE                                              #
# --------------------------------------------------------------------------- #
def session_difficulty(tss: float, intensity_factor: float, duration_s: int,
                       wbal_depleted_pct: Optional[float] = None) -> dict:
    """
    Difficolta' come composito di intensita' (IF), carico (TSS) e profondita'
    anaerobica (W'bal). Restituisce un livello 1-5 + il fabbisogno di recupero.
    Euristica su bande di uso comune.
    """
    # intensita' da IF
    if intensity_factor < 0.65:   intensity_lvl, intensity_lbl = 1, "Bassa"
    elif intensity_factor < 0.80: intensity_lvl, intensity_lbl = 2, "Moderata"
    elif intensity_factor < 0.90: intensity_lvl, intensity_lbl = 3, "Impegnativa"
    elif intensity_factor < 1.00: intensity_lvl, intensity_lbl = 4, "Dura"
    else:                         intensity_lvl, intensity_lbl = 5, "Molto dura"

    # spinta verso l'alto se sei sceso molto nel W'bal
    score = intensity_lvl
    if wbal_depleted_pct is not None and wbal_depleted_pct > 80:
        score = min(5, score + 1)
    # e se il TSS e' molto alto per la durata
    if tss > 300:
        score = min(5, score + 1)
    score = max(1, min(5, round(score)))
    labels = {1: "Facile", 2: "Moderata", 3: "Impegnativa", 4: "Dura", 5: "Molto dura"}

    # fabbisogno di recupero da TSS
    if tss < 150:   recovery = "basso (<24 h)"
    elif tss < 300: recovery = "~1 giorno"
    elif tss < 450: recovery = "~2 giorni"
    else:           recovery = "diversi giorni"

    return {
        "score_1to5": score,
        "label": labels[score],
        "intensity": intensity_lbl,
        "recovery_demand": recovery,
        "confidence": Confidence.ESTIMATED,
    }


# --------------------------------------------------------------------------- #
#  5. CARICO NEL TEMPO: CTL / ATL / TSB  (Fitness / Fatica / Forma)           #
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    day: date
    tss: float
    duration_s: int = 0
    if_: float = 0.0
    wtype: str = ""
    # distribuzione zone opzionale per la polarizzazione settimanale
    frac_low: float = 0.0     # Z1-Z2
    frac_mid: float = 0.0     # Z3-Z4
    frac_high: float = 0.0    # Z5+


def training_load(sessions: list[Session],
                  ctl_tc: int = 42, atl_tc: int = 7,
                  seed_ctl: float = 0.0, seed_atl: float = 0.0) -> pd.DataFrame:
    """
    Performance Management Chart. TSS giornaliero -> CTL (fitness, EMA 42gg),
    ATL (fatica, EMA 7gg), TSB (forma = CTL_ieri - ATL_ieri).
    ONESTA': CTL/ATL affidabili servono ~6 settimane di storico. Su pochi giorni
    e' solo un trend direzionale. Seed a 0 se non hai storico (sottostima iniziale).
    """
    if not sessions:
        return pd.DataFrame()
    days = sorted(sessions, key=lambda s: s.day)
    start, end = days[0].day, days[-1].day
    daily = {}
    for s in sessions:
        daily[s.day] = daily.get(s.day, 0.0) + s.tss

    rows, ctl, atl = [], seed_ctl, seed_atl
    d = start
    while d <= end:
        tss = daily.get(d, 0.0)
        tsb = ctl - atl                          # forma = valori di ieri
        ctl = ctl + (tss - ctl) / ctl_tc
        atl = atl + (tss - atl) / atl_tc
        rows.append({"day": d, "tss": tss, "ctl": ctl, "atl": atl, "tsb": tsb})
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def weekly_summary(sessions: list[Session]) -> dict:
    """
    Sintesi degli ultimi 7 giorni: TSS totale, n. sessioni, distribuzione di
    intensita' (polarizzazione), monotonia e strain (Foster).
    """
    if not sessions:
        return {}
    last_day = max(s.day for s in sessions)
    week = [s for s in sessions if (last_day - s.day).days < 7]
    tss_total = sum(s.tss for s in week)
    n = len(week)

    # polarizzazione: media pesata (per TSS) delle frazioni low/mid/high
    wsum = sum(s.tss for s in week) or 1
    low = sum(s.frac_low * s.tss for s in week) / wsum
    mid = sum(s.frac_mid * s.tss for s in week) / wsum
    high = sum(s.frac_high * s.tss for s in week) / wsum

    if low > 0.75 and mid < 0.2:
        distribution = "Polarizzata (80/20) — buono"
    elif mid > 0.35:
        distribution = "Troppa 'zona grigia' (tempo/soglia) — rischio stagnazione"
    elif high > 0.35:
        distribution = "Molto intensa — attenzione al recupero"
    else:
        distribution = "Prevalentemente aerobica"

    # monotonia e strain (Foster): media/deviazione dei TSS giornalieri
    by_day = {}
    for s in week:
        by_day[s.day] = by_day.get(s.day, 0.0) + s.tss
    daily_vals = [by_day.get(last_day - timedelta(days=i), 0.0) for i in range(7)]
    mean_d = np.mean(daily_vals)
    std_d = np.std(daily_vals) or 1e-6
    monotony = mean_d / std_d
    strain = tss_total * monotony

    return {
        "tss_week": round(tss_total),
        "sessions": n,
        "distribution": distribution,
        "frac_low": round(low, 2), "frac_mid": round(mid, 2), "frac_high": round(high, 2),
        "monotony": round(monotony, 2),
        "strain": round(strain),
        "confidence": Confidence.ESTIMATED,
    }


# --------------------------------------------------------------------------- #
#  6. RACCOMANDAZIONE: l'allenamento 'top' per domani                         #
# --------------------------------------------------------------------------- #
def tsb_label(tsb: float) -> str:
    """Interpretazione della forma (TSB) secondo bande d'uso comune."""
    if tsb > 15:    return "molto fresco (scarico/taper)"
    if tsb > 5:     return "fresco"
    if tsb > -10:   return "ottimale per allenarsi"
    if tsb > -30:   return "in fatica produttiva (carico)"
    return "molto affaticato (rischio sovraccarico)"


def _session_template(kind: str, ftp: float) -> dict:
    """Struttura concreta della sessione consigliata, ancorata alla FTP."""
    t = {
        "recupero": ("Recupero attivo", f"45-60 min in Z1 (<{round(0.55*ftp)} W)",
                     "~25-35 TSS"),
        "fondo": ("Fondo aerobico", f"90-150 min in Z2 ({round(0.56*ftp)}-{round(0.75*ftp)} W)",
                  "~90-140 TSS"),
        "lungo": ("Lungo endurance", f"3-4 h in Z2, con durabilita' ({round(0.6*ftp)}-{round(0.72*ftp)} W)",
                  "~180-260 TSS"),
        "sweetspot": ("Sweet-spot", f"3-4x 12 min all'88-93% FTP ({round(0.88*ftp)}-{round(0.93*ftp)} W)",
                      "~70-95 TSS"),
        "soglia": ("Soglia", f"3-4x 10 min al 95-100% FTP ({round(0.95*ftp)}-{round(ftp)} W)",
                   "~80-100 TSS"),
        "vo2max": ("VO2max", f"5-6x 4 min al 110-118% FTP ({round(1.10*ftp)}-{round(1.18*ftp)} W), rec 4 min",
                   "~70-90 TSS"),
        "anaerobico": ("Anaerobico / Sprint", f"6-8x 30 s all-out (>{round(1.5*ftp)} W), rec 4-5 min",
                       "~50-70 TSS"),
    }
    name, prescription, load = t[kind]
    return {"kind": kind, "name": name, "prescription": prescription, "expected_tss": load}


def recommend_next_workout(pmc: pd.DataFrame, weekly: dict, rider_type: dict,
                           ftp: float, target: str = "limiter") -> dict:
    """
    Propone la sessione successiva combinando FORMA (TSB), BILANCIO settimanale
    (polarizzazione, intensita' recente) e TIPO di corridore.

    target: "limiter" (default, allena il punto debole -> piu' guadagno) oppure
            "strength" (asseconda la specializzazione).

    Euristica su principi consolidati. NON sostituisce una periodizzazione verso
    un obiettivo/gara: e' una bussola per la prossima uscita.
    """
    tsb = float(pmc["tsb"].iloc[-1]) if len(pmc) else 0.0
    dist = weekly.get("distribution", "")
    high = weekly.get("frac_high", 0.0)
    monotony = weekly.get("monotony", 0.0)

    reasons, alternatives = [], []

    # 1) FATICA profonda o monotonia alta -> recupero
    if tsb < -25 or monotony > 2.5:
        primary = _session_template("recupero", ftp)
        reasons.append(f"Forma bassa (TSB {tsb:+.0f})" +
                       (" e monotonia elevata" if monotony > 2.5 else "") +
                       ": priorita' al recupero per assorbire il carico.")
        alternatives.append(_session_template("fondo", ftp))

    # 2) Troppa intensita' recente -> aerobico
    elif high > 0.30:
        primary = _session_template("fondo", ftp)
        reasons.append("Molta intensita' negli ultimi 7 giorni: serve volume aerobico "
                       "per bilanciare (evita altra alta intensita' ravvicinata).")
        alternatives.append(_session_template("recupero", ftp))

    # 3) Fresco e non troppa intensita' -> sessione di qualita' mirata
    elif tsb > -10:
        # scegli la qualita' in base al target e al tipo di corridore
        rel = rider_type.get("relative_strengths", {})
        if rel:
            weakest = min(rel, key=rel.get)
            strongest = max(rel, key=rel.get)
            focus = weakest if target == "limiter" else strongest
        else:
            focus = "vo2max"
        kind = {"sprint": "anaerobico", "anaerobico": "anaerobico",
                "vo2max": "vo2max", "soglia": "soglia"}.get(focus, "soglia")
        primary = _session_template(kind, ftp)
        why = ("allena il punto debole (piu' margine di crescita)"
               if target == "limiter" else "consolida il tuo punto di forza")
        reasons.append(f"Forma adeguata (TSB {tsb:+.0f}) e intensita' recente sotto controllo: "
                       f"sessione di qualita' che {why}.")
        if "grigia" in dist:
            reasons.append("La settimana e' troppo in 'zona grigia': meglio polarizzare "
                           "con uno stimolo netto piuttosto che altro tempo.")
        alternatives.append(_session_template("sweetspot", ftp))

    # 4) Forma intermedia -> mantenimento aerobico / sweet-spot
    else:
        primary = _session_template("sweetspot", ftp)
        reasons.append(f"Forma {tsb_label(tsb)} (TSB {tsb:+.0f}): stimolo sostenibile alla "
                       "soglia bassa senza aggiungere troppa fatica.")
        alternatives.append(_session_template("fondo", ftp))

    return {
        "recommended": primary,
        "alternatives": alternatives,
        "rationale": " ".join(reasons),
        "based_on": {"tsb": round(tsb, 1), "weekly_distribution": dist,
                     "rider_type": rider_type.get("primary", "")},
        "confidence": Confidence.ESTIMATED,
        "disclaimer": ("Euristica su principi allenanti (polarizzazione, recupero da TSB). "
                       "Non e' una periodizzazione verso una gara specifica."),
    }
