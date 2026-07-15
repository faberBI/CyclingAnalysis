"""
data_quality.py
===============
Trust layer: se vendi ACCURATEZZA, devi difendere l'INPUT.

Un ride con 30 s di potenza fantasma inquina NP, TSS e curva senza che nessuno lo
sappia. Qui rileviamo i problemi PRIMA di calcolare le metriche, e li dichiariamo
con un badge coerente col resto della piattaforma (buono / attenzione / scarso).

Controlli:
  - dropout di potenza (buchi/zeri prolungati in movimento)
  - spike non fisiologici (salti improvvisi, valori assurdi)
  - HR/cadenza fuori range fisiologico
  - indoor / rullo (quota piatta o assente, velocita' 'virtuale')
  - copertura dati (frazione di campioni validi)

Volutamente NON modifica i dati: separa la diagnosi dalla pulizia (to_1hz interpola;
questo giudica se ci si puo' fidare del risultato).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from cycling_analytics import Confidence

# soglie fisiologiche/di plausibilita' (ordini di grandezza, non dogmi)
POWER_ABS_MAX = 2500.0        # W: oltre = quasi certo artefatto (anche i velocisti stanno sotto)
POWER_JUMP = 900.0           # W: salto 1->1 s oltre = spike sospetto
HR_MIN, HR_MAX = 25, 230     # bpm plausibili
CAD_MAX = 250                # rpm plausibili


def _flag(level, msg):
    return {"level": level, "msg": msg}


def assess_data_quality(df: pd.DataFrame, dt: float = 1.0) -> dict:
    """
    Valuta la qualita' di un DataFrame (grezzo o a 1 Hz). Ritorna un punteggio 0-100,
    un livello (buono/attenzione/scarso), la lista dei flag e alcune metriche di
    copertura. Il livello mappa su un badge nell'app.

    Il punteggio parte da 100 e sottrae penalita' per gravita' del problema.
    """
    n = len(df)
    flags, penalty = [], 0.0
    stats = {"samples": int(n)}
    has_power = "power" in df.columns and df["power"].notna().any()

    # ---- POTENZA: dropout, spike, valori assurdi ----
    if has_power:
        p = pd.to_numeric(df["power"], errors="coerce").to_numpy(dtype=float)
        moving = p > 0                                   # proxy 'in movimento/pedalando'
        nan_frac = float(np.isnan(p).mean())
        # dropout = zeri/NaN dentro tratti attivi (buchi), non le soste vere
        active = np.where(moving)[0]
        dropout_frac = 0.0
        if active.size > 10:
            span = p[active.min():active.max() + 1]
            dropout_frac = float((np.nan_to_num(span) == 0).mean())
        stats["power_dropout_pct"] = round(dropout_frac * 100, 1)
        stats["power_nan_pct"] = round(nan_frac * 100, 1)
        if dropout_frac > 0.15:
            flags.append(_flag("scarso", f"Molti buchi di potenza in movimento "
                                         f"({dropout_frac*100:.0f}%): NP/TSS inaffidabili.")); penalty += 35
        elif dropout_frac > 0.05:
            flags.append(_flag("attenzione", f"Alcuni buchi di potenza ({dropout_frac*100:.0f}%): "
                                             "interpolati, ma occhio a NP/TSS.")); penalty += 12

        finite = p[np.isfinite(p)]
        n_over = int((finite > POWER_ABS_MAX).sum())
        if n_over:
            flags.append(_flag("attenzione", f"{n_over} campioni di potenza >{POWER_ABS_MAX:.0f} W "
                                             "(probabili spike): considera un filtro.")); penalty += min(15, n_over)
        jumps = int((np.abs(np.diff(np.nan_to_num(p))) > POWER_JUMP).sum())
        stats["power_spikes"] = n_over + jumps
        if jumps > n * 0.01:
            flags.append(_flag("attenzione", f"{jumps} salti di potenza >{POWER_JUMP:.0f} W/s: "
                                             "possibile dropout del misuratore o spike.")); penalty += 8
    else:
        flags.append(_flag("attenzione", "Nessuna potenza: molte metriche saranno stimate da HR "
                                         "o non disponibili.")); penalty += 5

    # ---- HR / CADENZA fuori range ----
    if "hr" in df.columns and df["hr"].notna().any():
        h = pd.to_numeric(df["hr"], errors="coerce").to_numpy(dtype=float)
        bad_hr = int(((h < HR_MIN) | (h > HR_MAX)).sum())
        if bad_hr > 5:
            flags.append(_flag("attenzione", f"{bad_hr} valori di HR fuori range "
                                             f"({HR_MIN}-{HR_MAX} bpm): sensore ballerino.")); penalty += 6
    if "cadence" in df.columns and df["cadence"].notna().any():
        c = pd.to_numeric(df["cadence"], errors="coerce").to_numpy(dtype=float)
        if int((c > CAD_MAX).sum()) > 3:
            flags.append(_flag("attenzione", f"Cadenza >{CAD_MAX} rpm: artefatti del sensore.")); penalty += 3

    # ---- INDOOR / RULLO ----
    indoor = detect_indoor(df)
    stats["indoor"] = indoor["indoor"]
    if indoor["indoor"]:
        flags.append(_flag("info", f"Probabile indoor/rullo ({indoor['reason']}): "
                                   "l'analisi altimetrica/aero non si applica."))

    # ---- COPERTURA ----
    if n < 60:
        flags.append(_flag("scarso", "Meno di 60 s di dati: risultati non significativi.")); penalty += 40

    score = max(0, round(100 - penalty))
    level = "buono" if score >= 85 else "attenzione" if score >= 60 else "scarso"
    return {
        "score": score,
        "level": level,
        "flags": flags,
        "stats": stats,
        "indoor": indoor["indoor"],
        "confidence": Confidence.MEASURED,
        "summary": {"buono": "Dati affidabili.",
                    "attenzione": "Dati usabili ma con riserve: leggi i flag.",
                    "scarso": "Dati problematici: le metriche possono essere fuorvianti."}[level],
    }


def detect_indoor(df: pd.DataFrame) -> dict:
    """
    Distingue rullo/indoor da uscita su strada. Segnali: quota assente o piatta
    (std < 2 m), oppure velocita' assente/troppo costante mentre c'e' potenza
    (velocita' 'virtuale' del rullo). Non e' un problema di qualita', ma cambia
    quali analisi hanno senso (niente salite/VAM/aero indoor).
    """
    reasons = []
    alt_flat = False
    if "altitude" in df.columns and df["altitude"].notna().any():
        alt = pd.to_numeric(df["altitude"], errors="coerce").dropna()
        if alt.std() < 2.0:
            alt_flat = True; reasons.append("quota piatta")
    else:
        reasons.append("nessuna quota")

    speed_const = False
    if "speed" in df.columns and df["speed"].notna().any():
        sp = pd.to_numeric(df["speed"], errors="coerce").dropna()
        mv = sp[sp > 1]
        if len(mv) > 30 and (mv.std() / (mv.mean() or 1)) < 0.05:
            speed_const = True; reasons.append("velocita' quasi costante (virtuale)")

    indoor = (alt_flat or "nessuna quota" in reasons) and \
             ("power" in df.columns and df["power"].notna().any())
    # rafforza il verdetto se anche la velocita' e' virtuale
    indoor = indoor or (speed_const and alt_flat)
    return {"indoor": bool(indoor), "reason": ", ".join(reasons) or "n/d"}
