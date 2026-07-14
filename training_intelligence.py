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


def pmc_from_activities(activities: list[dict]) -> pd.DataFrame:
    """
    Costruisce il PMC (CTL/ATL/TSB) su TUTTO lo storico a partire dall'elenco
    attivita' di intervals.icu (ognuna con 'date' e 'load'=TSS). Seed a 0: con lo
    storico completo il CTL si costruisce correttamente dall'inizio.
    """
    sessions = []
    for a in activities:
        d = a.get("date")
        if not d:
            continue
        try:
            day = pd.to_datetime(d).date()
        except Exception:
            continue
        sessions.append(Session(day=day, tss=float(a.get("load") or 0)))
    if not sessions:
        return pd.DataFrame()
    return training_load(sessions)


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


def recovery_status(tsb: float) -> str:
    """Stato di recupero attuale dal TSB (facet del bilancio forma/fatica)."""
    if tsb >= 5:    return "Pienamente recuperato"
    if tsb >= -10:  return "Recuperato"
    if tsb >= -20:  return "Recupero parziale"
    return "Recupero necessario"


def recovery_forecast(ctl: float, atl: float, target_tsb: float = 5,
                      max_days: int = 21) -> int:
    """
    Giorni di RIPOSO stimati per tornare 'freschi' (TSB >= target_tsb).
    Simula giorni a TSS=0: ATL decade (tc 7gg) piu' in fretta di CTL (tc 42gg),
    quindi il TSB risale. 0 = gia' fresco; max_days = oltre l'orizzonte.
    Stima su modello (Banister/Coggan) SUL SOLO CARICO. Il recupero autonomico
    (HRV/HR riposo/sonno) puo' allungarlo: vedi wellness_readiness().
    """
    if ctl - atl >= target_tsb:
        return 0
    c, a = ctl, atl
    for d in range(1, max_days + 1):
        c += (0 - c) / 42
        a += (0 - a) / 7
        if c - a >= target_tsb:
            return d
    return max_days


def wellness_readiness(wellness_df) -> dict:
    """
    Prontezza autonomica dai dati di benessere di intervals.icu.
    Confronta HRV/HR-riposo/sonno recenti (media ultimi 3 gg) con una baseline
    (~ultimi 42 gg) e restituisce stato per metrica + complessivo (verde/ambra/rosso),
    una penalita' in giorni sul recupero e un flag per la seduta.

    Basato su principi di HRV-guided training (Seiler; HRV4Training; Plews et al.):
    HRV soppressa, HR a riposo elevata o sonno scarso -> ridurre carico, allungare
    recupero. STIMA: soglie di popolazione, il tuo range individuale puo' differire.
    """
    status = {"hrv": "n/d", "rhr": "n/d", "sleep": "n/d", "overall": "n/d",
              "recovery_penalty_days": 0, "flag": "", "have_data": False,
              "confidence": Confidence.ESTIMATED}
    if wellness_df is None or len(wellness_df) == 0 or "hrv" not in wellness_df:
        return status
    df = wellness_df.dropna(subset=["hrv"]).sort_values("date")
    if len(df) < 7:
        return status
    status["have_data"] = True

    # HRV: z-score dei recenti rispetto alla baseline (che ESCLUDE i giorni recenti)
    recent_hrv = df["hrv"].tail(3).mean()
    base = df["hrv"].iloc[:-3].tail(42)
    mu, sd = base.mean(), (base.std() or 1e-6)
    z = (recent_hrv - mu) / sd
    if z < -1.0:
        status["hrv"], hrv_pen = "soppressa", 2
    elif z < -0.3:
        status["hrv"], hrv_pen = "sotto la norma", 1
    else:
        status["hrv"], hrv_pen = "nella norma", 0

    # HR a riposo: elevazione rispetto alla baseline (esclusi i recenti)
    rhr_pen = 0
    if "resting_hr" in df and df["resting_hr"].notna().sum() >= 7:
        rr = df.dropna(subset=["resting_hr"])
        recent_rhr = rr["resting_hr"].tail(3).mean()
        base_rhr = rr["resting_hr"].iloc[:-3].tail(42).mean()
        if recent_rhr > base_rhr + 5:
            status["rhr"], rhr_pen = "elevata", 1
        elif recent_rhr > base_rhr + 2:
            status["rhr"], rhr_pen = "leggermente elevata", 1
        else:
            status["rhr"] = "normale"

    # Sonno
    sleep_pen = 0
    if "sleep_hours" in df and df["sleep_hours"].notna().sum() >= 3:
        sh = df.dropna(subset=["sleep_hours"])["sleep_hours"].tail(3).mean()
        if sh < 6:
            status["sleep"], sleep_pen = "scarso", 1
        elif sh < 7:
            status["sleep"], sleep_pen = "sotto la norma", 0
        else:
            status["sleep"] = "buono"

    pen = hrv_pen + rhr_pen + sleep_pen
    status["recovery_penalty_days"] = pen
    if pen >= 3:
        status["overall"] = "rosso"
        status["flag"] = "Segnali di scarso recupero: oggi meglio riposo o solo aerobico leggero."
    elif pen >= 1:
        status["overall"] = "ambra"
        status["flag"] = "Recupero incompleto: modera l'intensità, evita sedute molto dure."
    else:
        status["overall"] = "verde"
        status["flag"] = "Buon recupero: via libera all'allenamento."
    return status


def adjusted_recovery_days(base_days: int, readiness: dict) -> int:
    """Recupero in giorni = stima da carico (TSB) + penalita' da wellness (HRV/RHR/sonno)."""
    if not readiness or not readiness.get("have_data"):
        return base_days
    return base_days + int(readiness.get("recovery_penalty_days", 0))


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
                           ftp: float, target: str = "limiter",
                           readiness: dict = None) -> dict:
    """
    Propone la sessione successiva combinando FORMA (TSB), BILANCIO settimanale
    (polarizzazione, intensita' recente), TIPO di corridore e — se disponibile —
    la PRONTEZZA autonomica da wellness (HRV/HR-riposo/sonno).

    target: "limiter" (allena il punto debole) o "strength" (asseconda la forza).
    readiness: output di wellness_readiness() (opzionale). 'rosso' forza il recupero
               a prescindere dal TSB; 'ambra' declassa le sedute molto intense.

    Euristica su principi consolidati (polarizzazione, recupero da TSB, HRV-guided
    training). NON sostituisce una periodizzazione verso una gara.
    """
    tsb = float(pmc["tsb"].iloc[-1]) if len(pmc) else 0.0
    dist = weekly.get("distribution", "")
    high = weekly.get("frac_high", 0.0)
    monotony = weekly.get("monotony", 0.0)
    r_overall = readiness.get("overall") if readiness and readiness.get("have_data") else None

    reasons, alternatives = [], []

    # 0) WELLNESS ROSSO: il recupero autonomico ha la precedenza sul TSB
    if r_overall == "rosso":
        primary = _session_template("recupero", ftp)
        reasons.append(readiness["flag"] +
                       f" (HRV {readiness['hrv']}, HR riposo {readiness['rhr']}, sonno {readiness['sleep']}).")
        alternatives.append(_session_template("fondo", ftp))

    # 1) FATICA profonda o monotonia alta -> recupero
    elif tsb < -25 or monotony > 2.5:
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

    # WELLNESS AMBRA: declassa le sedute molto intense (VO2/anaerobico) a sweet-spot
    if r_overall == "ambra" and primary["kind"] in ("vo2max", "anaerobico"):
        alternatives.insert(0, primary)
        primary = _session_template("sweetspot", ftp)
        reasons.append(readiness["flag"] + " Ho abbassato l'intensità della seduta prevista.")

    return {
        "recommended": primary,
        "alternatives": alternatives,
        "rationale": " ".join(reasons),
        "based_on": {"tsb": round(tsb, 1), "weekly_distribution": dist,
                     "rider_type": rider_type.get("primary", ""),
                     "wellness": r_overall or "n/d"},
        "confidence": Confidence.ESTIMATED,
        "disclaimer": ("Euristica su principi allenanti (polarizzazione, recupero da TSB, "
                       "HRV-guided). Non e' una periodizzazione verso una gara specifica."),
    }


# --------------------------------------------------------------------------- #
#  7. PERIODIZZAZIONE VERSO UNA GARA                                          #
# --------------------------------------------------------------------------- #
# Pianifica a ritroso dalla data gara: fasi Base->Build->Taper (scarico 3:1),
# rampa di CTL sicura, e PROIEZIONE in avanti del PMC per verificare di arrivare
# con la freschezza (TSB) giusta.
#
# ONESTA': e' un modello su principi consolidati (Friel/Bompa fasi; Coggan/TP
# target CTL/TSB; Seiler polarizzazione). NON e' coaching individualizzato: non
# conosce la tua storia, il tuo tasso di risposta, lo stress di vita, i dettagli
# dell'evento. I target di TSB a fine gara sono regole del pollice (alcuni atleti
# picccano a +5, altri a +25). Raggiungere i numeri di carico non garantisce la
# forma: devi comunque fare bene le sedute. Modello a singolo picco.

# tsb = banda di freschezza a fine taper; taper_weeks = settimane di scarico;
# focus = enfasi delle qualita' nel Build.  (regole del pollice, non scienza esatta)
EVENT_PROFILES = {
    "Granfondo / Ciclistica":   {"tsb": (5, 15),  "taper_weeks": 2,
                                 "focus": "endurance lunga + soglia + durabilità"},
    "Gara in linea (1 giorno)": {"tsb": (10, 20), "taper_weeks": 2,
                                 "focus": "soglia + VO2max + strappi"},
    "Cronometro (TT)":          {"tsb": (15, 25), "taper_weeks": 2,
                                 "focus": "soglia/FTP sostenuta"},
    "Criterium":                {"tsb": (5, 15),  "taper_weeks": 1,
                                 "focus": "anaerobico + VO2max + ripetibilità"},
    "Corsa a tappe":            {"tsb": (5, 12),  "taper_weeks": 2,
                                 "focus": "volume + soglia ripetuta + durabilità"},
}

# frazione del TSS settimanale per giorno (microciclo tipo)
_MICROCYCLE = [0.0, 0.18, 0.14, 0.20, 0.0, 0.30, 0.18]   # Lun..Dom
_CTL_TC = 42
_F7 = 1 - (1 - 1 / _CTL_TC) ** 7                          # ~0.1552: quota di 'inseguimento' in 7 gg

def _weekly_tss_for_ramp(ctl_start: float, ramp: float) -> float:
    """TSS settimanale per far salire il CTL di `ramp` punti in 7 giorni."""
    d = ctl_start + ramp / _F7        # TSS medio giornaliero necessario
    return max(0.0, 7 * d)

def _phase_focus(phase: str, event_focus: str) -> str:
    return {
        "Base":  "Volume aerobico (Z2), polarizzato 80/20, con lungo settimanale",
        "Build": f"Intensità mirata: {event_focus}",
        "Taper": "Riduci il volume, mantieni stimoli brevi (aperture/richiami), arriva fresco",
        "Scarico": "Settimana di scarico: ~50% del carico per assorbire l'adattamento",
    }[phase]

def _phase_session(phase: str, event: str, ftp: float, rider_type: dict) -> dict:
    if phase == "Base":
        return _session_template("lungo", ftp)
    if phase == "Scarico":
        return _session_template("fondo", ftp)
    if phase == "Taper":
        return _session_template("vo2max", ftp)   # brevi richiami, volume ridotto
    # Build: seduta secondo l'evento (o il limitante)
    kind = {"Granfondo / Ciclistica": "soglia", "Gara in linea (1 giorno)": "vo2max",
            "Cronometro (TT)": "soglia", "Criterium": "anaerobico",
            "Corsa a tappe": "soglia"}.get(event, "soglia")
    return _session_template(kind, ftp)


def periodized_plan(current_date: date, race_date: date, current_ctl: float,
                    current_atl: Optional[float] = None,
                    event: str = "Gara in linea (1 giorno)",
                    ftp: float = 250, safe_ramp: float = 5.0,
                    rider_type: Optional[dict] = None) -> dict:
    """
    Piano periodizzato dalla data odierna alla gara + proiezione del PMC.
    safe_ramp = incremento di CTL/settimana desiderato (sicuro ~3-6; >8 rischioso).
    """
    rider_type = rider_type or {}
    if current_atl is None:
        current_atl = current_ctl
    prof = EVENT_PROFILES.get(event, EVENT_PROFILES["Gara in linea (1 giorno)"])
    tsb_lo, tsb_hi = prof["tsb"]

    days_to_race = (race_date - current_date).days
    warnings = []
    if days_to_race <= 0:
        return {"error": "La data gara deve essere futura."}
    W = max(1, (days_to_race + 6) // 7)          # settimane (arrotonda per eccesso)

    # --- struttura fasi ---
    if W <= 2:
        # A <2 settimane non si guadagna fitness utile: solo affinamento/freschezza.
        taper = W
        loading = 0
        base_count = 0
        warnings.append("Meno di ~2 settimane alla gara: non c'è tempo per costruire fitness "
                        "(gli adattamenti richiedono settimane). Piano di solo affinamento e freschezza.")
    else:
        taper = prof["taper_weeks"]
        taper = min(taper, W - 1)
        loading = W - taper
        base_count = max(0, round(loading * 0.55)) if loading > 2 else 0

    # --- piano settimanale (TSS + CTL intesa) ---
    weeks = []
    ctl_int = current_ctl
    last_build_weekly = 7 * current_ctl
    li = 0
    for i in range(W):
        in_taper = i >= (W - taper)
        if in_taper:
            phase = "Taper"
            ti_ = i - (W - taper)                 # 0-based nel taper
            frac = ([0.6, 0.4] if taper == 2 else [0.5])[min(ti_, taper - 1)]
            wk = frac * last_build_weekly
            ctl_int += -3
        else:
            recovery = (li + 1) % 4 == 0 and li < loading - 1     # scarico 3:1
            if recovery:
                phase = "Scarico"
                wk = 0.55 * last_build_weekly
                ctl_int += -1.5
            else:
                phase = "Base" if li < base_count else "Build"
                ramp = safe_ramp * (0.85 if phase == "Base" else 1.0)
                wk = _weekly_tss_for_ramp(ctl_int, ramp)
                ctl_int += ramp
                last_build_weekly = wk
            li += 1
        if wk > 800:
            warnings.append(f"Settimana {i+1}: TSS target {wk:.0f} molto alto per un amatore — "
                            "valuta un safe_ramp più basso.")
        weeks.append({"week": i + 1,
                      "start": current_date + timedelta(days=7 * i),
                      "phase": phase,
                      "focus": _phase_focus(phase, prof["focus"]),
                      "target_ctl": round(ctl_int),
                      "weekly_tss": round(wk),
                      "session": _phase_session(phase, event, ftp, rider_type)})

    # --- proiezione PMC in avanti (giorno per giorno) ---
    sessions = []
    for wk in weeks:
        for dofs, frac in enumerate(_MICROCYCLE):
            day = wk["start"] + timedelta(days=dofs)
            if day > race_date:
                break
            sessions.append(Session(day=day, tss=wk["weekly_tss"] * frac))
    proj = training_load(sessions, seed_ctl=current_ctl, seed_atl=current_atl)

    # TSB previsto il giorno gara
    race_row = proj[proj["day"] == race_date]
    race_tsb = float(race_row["tsb"].iloc[0]) if len(race_row) else float(proj["tsb"].iloc[-1])
    peak_ctl = float(proj["ctl"].max())

    if race_tsb < tsb_lo:
        verdict = f"⚠️ Arrivi troppo affaticato (TSB previsto {race_tsb:+.0f}, target {tsb_lo}..{tsb_hi}). Allunga o accentua il taper."
    elif race_tsb > tsb_hi:
        verdict = f"⚠️ Arrivi troppo scarico (TSB previsto {race_tsb:+.0f}, target {tsb_lo}..{tsb_hi}). Accorcia il taper o tieni più carico."
    else:
        verdict = f"✅ Arrivi in forma: TSB previsto {race_tsb:+.0f} (target {tsb_lo}..{tsb_hi})."

    # settimana corrente (la prima del piano)
    this_week = weeks[0]

    return {
        "weeks_until": W,
        "days_to_race": days_to_race,
        "phase_structure": {"base_weeks": base_count,
                            "build_weeks": loading - base_count,
                            "taper_weeks": taper},
        "weeks": weeks,
        "projection": proj,                 # DataFrame day/ctl/atl/tsb
        "race_day_tsb": round(race_tsb, 1),
        "tsb_target": (tsb_lo, tsb_hi),
        "peak_ctl": round(peak_ctl),
        "verdict": verdict,
        "this_week": this_week,
        "warnings": warnings,
        "confidence": Confidence.ESTIMATED,
        "disclaimer": ("Modello di pianificazione su principi consolidati (fasi Friel/Bompa, "
                       "target CTL/TSB Coggan/TrainingPeaks, polarizzazione Seiler). Non è coaching "
                       "individualizzato né tiene conto della tua storia/risposta. Modello a singolo "
                       "picco; i target di TSB sono regole del pollice."),
    }


# --------------------------------------------------------------------------- #
#  8. AUTO-DETECTION DEGLI INTERVALLI                                         #
# --------------------------------------------------------------------------- #
def detect_intervals(power, ftp: float, min_seconds: int = 20,
                     threshold_pct: float = 1.02, gap_seconds: int = 15) -> list[dict]:
    """
    Rileva automaticamente gli sforzi 'hard' dell'uscita: tratti con potenza sopra
    threshold_pct*FTP per almeno min_seconds, unendo micro-cali < gap_seconds.
    Utile per capire la STRUTTURA dell'allenamento senza inserirla a mano.
    """
    import numpy as _np
    p = _np.nan_to_num(_np.asarray(power, dtype=float), nan=0.0)
    above = (p >= threshold_pct * ftp).astype(int)
    edges = _np.diff(_np.concatenate([[0], above, [0]]))
    starts = list(_np.where(edges == 1)[0])
    ends = list(_np.where(edges == -1)[0])            # esclusivo
    merged = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] < gap_seconds:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if e - s >= min_seconds:
            seg = p[s:e]
            avg = float(seg.mean())
            out.append({"start_s": int(s), "duration_s": int(e - s),
                        "avg_power": round(avg), "pct_ftp": round(avg / ftp * 100),
                        "peak_power": round(float(seg.max()))})
    return out


# --------------------------------------------------------------------------- #
#  9. ANALISI STAGIONE + TREND NEL TEMPO (da intervals.icu)                   #
# --------------------------------------------------------------------------- #
def analyze_season_from_intervals(athlete, athlete_id: str, api_key: str,
                                  activities: list[dict], progress_cb=None,
                                  max_activities: int = 60,
                                  kj_threshold: float = 2000) -> dict:
    """
    Un'unica passata sugli streams delle attivita' (una chiamata /streams ciascuna):
    costruisce (1) la curva di potenza stagionale, (2) la curva 'da stanco' aggregata
    (durability su piu' uscite), (3) la serie temporale di eFTP, VO2max stimata e
    decoupling per uscita. Ritorna un dict con tutto.
    """
    import time as _time
    import cycling_analytics as ca
    acts = [a for a in activities if a.get("has_power")][:max_activities]
    curves, fatigued_curves, rows, used, n_durable = [], [], [], 0, 0
    dur_durations = [5, 15, 60, 300, 1200]
    n = len(acts) or 1
    for i, a in enumerate(acts):
        try:
            df1 = ca.to_1hz(ca.load_intervals_icu(a["id"], api_key))
            if "power" in df1.columns and float(df1["power"].sum()) > 0:
                power = df1["power"].values
                mmp_a = ca.mean_maximal_power(power)
                curves.append(mmp_a)
                # curva da stanco: MMP del tratto dopo kj_threshold kJ
                cum = np.cumsum(power) / 1000.0
                idx = int(np.searchsorted(cum, kj_threshold))
                if idx < len(power):
                    fatigued_curves.append(ca.mean_maximal_power(power[idx:], dur_durations))
                    n_durable += 1
                eftp = (0.95 * mmp_a[1200] if 1200 in mmp_a.index
                        else 0.90 * mmp_a[480] if 480 in mmp_a.index else np.nan)
                vo2 = (ca.estimate_vo2max(athlete, mmp_a[300])["vo2max"].value
                       if 300 in mmp_a.index else np.nan)
                dec = np.nan
                if "hr" in df1.columns and df1["hr"].notna().any():
                    dec = aerobic_decoupling(power, df1["hr"].values).value
                rows.append({"date": a.get("date"),
                             "eftp": round(float(eftp)) if eftp == eftp else np.nan,
                             "vo2max": round(float(vo2), 1) if vo2 == vo2 else np.nan,
                             "decoupling": round(float(dec), 1) if dec == dec else np.nan,
                             "tss": a.get("load") or np.nan})
                used += 1
        except Exception:
            pass
        if progress_cb:
            progress_cb((i + 1) / n)
        _time.sleep(0.12)

    season = (pd.concat(curves, axis=1).max(axis=1).rename("mmp_watt")
              if curves else pd.Series(dtype=float))
    season_fat = (pd.concat(fatigued_curves, axis=1).max(axis=1)
                  if fatigued_curves else pd.Series(dtype=float))
    # durability aggregata: caduta fresco -> stanco sulle curve stagionali
    durability = {}
    for d in dur_durations:
        f = season.get(d)
        g = season_fat.get(d) if len(season_fat) else None
        if f and g:
            durability[d] = {"fresh": round(f), "fatigued": round(g),
                             "drop_pct": round((f - g) / f * 100, 1)}

    trends = pd.DataFrame(rows)
    if len(trends):
        trends["date"] = pd.to_datetime(trends["date"], errors="coerce")
        trends = trends.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return {"season_curve": season, "season_fatigued": season_fat,
            "durability": durability, "n_durable": n_durable,
            "kj_threshold": kj_threshold, "trends": trends, "used": used}
