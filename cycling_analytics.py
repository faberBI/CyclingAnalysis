"""
cycling_analytics.py
====================
Motore di analisi fisiologica per ciclismo.

Principio guida: OGNI metrica esce con un'etichetta di affidabilita'.

    Confidence.MEASURED  -> calcolato direttamente dai dati, nessuna assunzione
                            fisiologica (curva di potenza, calorie da potenza,
                            NP/TSS/CP/W'). Errore ~ solo qualita' del power meter.
    Confidence.ESTIMATED -> richiede assunzioni o equazioni di popolazione
                            (FTP da test, VO2max, kcal da HR). Errore tipico 5-15%.
    Confidence.MODELED   -> NON misurabile senza laboratorio (metabolimetro).
                            Restituiamo un modello di popolazione. Solo indicativo.
                            (FatMax, split grassi/carbo senza RER).

Questo e' il differenziatore rispetto a Strava/TrainingPeaks: la trasparenza
sull'incertezza. Non spacciamo una stima per una misura.

Riferimenti principali:
- Monod & Scherrer 1965; Morton 2006 (Critical Power / W')
- Allen, Coggan & McGregor, "Training and Racing with a Power Meter" (NP, IF, TSS, zone, power profile)
- ACSM Guidelines (equazione cicloergometro per VO2)
- Storer et al. 1990 (VO2max da cicloergometro)
- Keytel et al. 2005 (kcal da frequenza cardiaca)
- Jeukendrup & Wallis 2005; Frayn 1983 (ossidazione substrati da scambi respiratori)
- Mifflin-St Jeor 1990 (metabolismo basale)
- Pinot & Grappe 2011/2014 (Record Power Profile pro); van Erp & Sanders; Leo et al. 2022 (fisiologia pro peloton)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# --------------------------------------------------------------------------- #
#  Affidabilita'                                                              #
# --------------------------------------------------------------------------- #
class Confidence(str, Enum):
    MEASURED = "misurato"      # dai dati, nessuna assunzione
    ESTIMATED = "stimato"      # equazione di popolazione / test da campo
    MODELED = "modellato"      # non misurabile senza laboratorio


@dataclass
class Metric:
    """Contenitore uniforme: valore + come e' stato ottenuto + quanto ci fidiamo.

    ci / sd sono OPZIONALI (default None) per retro-compatibilita': quando presenti
    quantificano l'incertezza reale del numero (non solo l'etichetta di confidence).
    - sd  : deviazione standard stimata del valore (1 sigma).
    - ci  : intervallo (lo, hi) al 95% (tipicamente valore +/- 1.96*sd).
    Il differenziatore diventa cosi' DIMOSTRABILE: '250 +/- 8 W', non solo 'stimato'.
    """
    value: float
    unit: str
    confidence: Confidence
    method: str
    note: str = ""
    sd: Optional[float] = None
    ci: Optional[tuple[float, float]] = None

    @property
    def ci_str(self) -> str:
        """Rappresentazione compatta dell'intervallo, '' se assente."""
        if self.ci is None:
            return ""
        lo, hi = self.ci
        return f"{lo:.0f}–{hi:.0f} {self.unit}".strip()

    def __repr__(self):
        base = f"{self.value:.2f} {self.unit}"
        if self.ci is not None:
            base += f" (95% CI {self.ci[0]:.1f}–{self.ci[1]:.1f})"
        return f"{base} [{self.confidence.value}: {self.method}]"


# --------------------------------------------------------------------------- #
#  Profilo atleta                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Athlete:
    mass_kg: float
    height_cm: float
    age: int
    sex: str = "M"                      # "M" | "F"
    hr_max: Optional[int] = None        # misurata > formula. Se None: 220-eta' (impreciso)
    hr_rest: Optional[int] = None
    lthr: Optional[int] = None          # HR alla soglia lattacida (da test 20-30 min)
    # Se hai un test da laboratorio, passali: rendono MODELED -> MEASURED
    vo2max_lab: Optional[float] = None      # mL/kg/min misurato al metabolimetro
    fatmax_pct_vo2max: Optional[float] = None  # %VO2max misurato al FatMax test

    @property
    def hr_max_effective(self) -> tuple[int, Confidence]:
        if self.hr_max:
            return self.hr_max, Confidence.MEASURED
        # Tanaka 2001 (208 - 0.7*eta') meno peggio di 220-eta', ma resta popolazione
        return round(208 - 0.7 * self.age), Confidence.ESTIMATED


# --------------------------------------------------------------------------- #
#  1. PREPROCESSING — porta qualunque ride a 1 Hz, colonne standard           #
# --------------------------------------------------------------------------- #
STD_COLS = ["t", "power", "hr", "cadence", "speed", "altitude", "lat", "lon"]


def to_1hz(df: pd.DataFrame, time_col: str = "t") -> pd.DataFrame:
    """
    Ricampiona a 1 secondo. La curva di potenza e tutto il resto assumono 1 Hz.
    `t` puo' essere secondi (int/float), timestamp epoch o datetime/stringhe ISO.
    Robusto a colonne 'object' (testo/misto): tutte le colonne numeriche vengono
    forzate con to_numeric (i valori non validi diventano NaN e vengono interpolati).
    Gap brevi -> interpolati; gap lunghi (stop) -> power=0.
    """
    d = df.copy()

    # --- asse temporale in secondi dall'inizio (robusto) ---
    tcol = d[time_col]
    if pd.api.types.is_datetime64_any_dtype(tcol):
        sec = (tcol - tcol.iloc[0]).dt.total_seconds()
    else:
        num = pd.to_numeric(tcol, errors="coerce")
        if num.notna().sum() >= max(2, 0.5 * len(num)):        # secondi / epoch numerici
            sec = num - num.dropna().iloc[0]
        else:                                                  # prova stringhe datetime ISO
            dt = pd.to_datetime(tcol, errors="coerce", utc=True)
            if dt.notna().sum() < 2:
                raise ValueError("Colonna tempo non interpretabile (né numerica né datetime).")
            sec = (dt - dt.dropna().iloc[0]).dt.total_seconds()

    d = d.drop(columns=[time_col], errors="ignore")
    d["_sec"] = pd.to_numeric(sec, errors="coerce")
    d = d.dropna(subset=["_sec"])
    if len(d) == 0:
        raise ValueError("Nessun timestamp valido nei dati.")
    d["_sec"] = d["_sec"].round().astype(int)
    d = d.drop_duplicates("_sec").set_index("_sec")
    full = pd.RangeIndex(0, int(d.index.max()) + 1)
    d = d.reindex(full)

    # --- forza numerico su TUTTE le colonne dati prima di interpolare ---
    for c in ["power", "hr", "cadence", "speed", "altitude", "lat", "lon"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "power" in d.columns:
        d["power"] = d["power"].interpolate(limit=3).fillna(0).clip(lower=0)
    for c in ["hr", "cadence", "speed", "altitude"]:
        if c in d.columns:
            d[c] = d[c].interpolate(limit=5)
    # lat/lon: interpola i buchi brevi ma NON riempire con 0 (0,0 = bug GPS al largo)
    for c in ["lat", "lon"]:
        if c in d.columns:
            d[c] = d[c].interpolate(limit=5)
    d.index.name = "t"
    return d


# --------------------------------------------------------------------------- #
#  2. CURVA DI POTENZA (Mean Maximal Power)  —  MEASURED                       #
# --------------------------------------------------------------------------- #
# Durate standard: da 1 s a 3 h. Puoi passarne di custom.
DEFAULT_DURATIONS = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 120, 180, 300, 480,
                     600, 720, 900, 1200, 1800, 2400, 3600, 5400, 7200, 10800]


def mean_maximal_power(power: np.ndarray,
                       durations: list[int] = DEFAULT_DURATIONS) -> pd.Series:
    """
    Miglior potenza media su ogni finestra di durata `d` (secondi), su tutta la ride.
    Implementazione O(n) per durata via somma cumulata. Power a 1 Hz.
    Questo e' un dato MISURATO: nessuna assunzione fisiologica.
    """
    p = np.nan_to_num(np.asarray(power, dtype=float), nan=0.0)
    n = len(p)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    out = {}
    for d in durations:
        if d > n:
            continue
        window_sums = cs[d:] - cs[:-d]
        out[d] = float(window_sums.max() / d)
    return pd.Series(out, name="mmp_watt")


def season_power_curve(rides_power: list[np.ndarray],
                       durations: list[int] = DEFAULT_DURATIONS) -> pd.Series:
    """
    Curva di potenza aggregata su piu' allenamenti (max punto-a-punto).
    IMPORTANTE: la classificazione del corridore e i benchmark hanno senso SOLO
    su una curva stagionale, perche' una singola uscita raramente contiene sforzi
    massimali a TUTTE le durate (nessuno fa uno sprint puro e 3h all-out lo stesso giorno).
    """
    curves = [mean_maximal_power(p, durations) for p in rides_power]
    return pd.concat(curves, axis=1).max(axis=1).rename("mmp_watt")


# --------------------------------------------------------------------------- #
#  3. CRITICAL POWER & W'  —  MEASURED (dato modello valido)                   #
# --------------------------------------------------------------------------- #
# Modello iperbolico: P = CP + W'/t   <=>   W(=P*t) = W' + CP*t
# CP  = potenza sostenibile (asintoto), proxy fisiologico della soglia
# W'  = capacita' di lavoro sopra CP (Joule), il tuo "serbatoio anaerobico"
#
# Validita': usare sforzi MASSIMALI tra ~2 e ~20 min. Sotto i 2' e sopra i ~20-30'
# il modello 2-parametri sbaglia (serve il 3-parametri di Morton). Filtriamo.

def _cp_linear(t, cp, w_prime):        # W = W' + CP*t
    return w_prime + cp * t

def _cp_hyperbolic(t, cp, w_prime):    # P = CP + W'/t
    return cp + w_prime / t

def _cp_3param(t, cp, w_prime, pmax):  # Morton 3-par: aggiunge potenza istantanea max
    return w_prime / (t - w_prime / (pmax - cp)) + cp


def critical_power(mmp: pd.Series,
                   fit_range=(120, 1200),
                   model: str = "3param") -> dict[str, Metric]:
    """
    Stima CP e W' dai punti massimali della curva di potenza nel range valido.
    model: "linear" (robusto, 2-par), "hyperbolic" (2-par), "3param" (Morton, migliore
    se hai anche uno sprint <15 s per ancorare Pmax).
    """
    pts = mmp[(mmp.index >= fit_range[0]) & (mmp.index <= fit_range[1])]
    if len(pts) < 3:
        raise ValueError("Servono >=3 sforzi massimali tra 2 e 20 min per stimare CP/W'.")
    t = pts.index.values.astype(float)
    p = pts.values.astype(float)
    cp_sd = w_sd = None                      # deviazioni standard dei parametri (1 sigma)

    if model == "linear":
        w = p * t
        A = np.vstack([t, np.ones_like(t)]).T   # colonne: [t, 1] -> [CP, W']
        beta, *_ = np.linalg.lstsq(A, w, rcond=None)
        cp, w_prime = beta
        # covarianza OLS: sigma^2 * (A'A)^-1, sigma^2 = SSR/(n-2)
        resid = w - A @ beta
        dof = max(len(t) - 2, 1)
        sigma2 = float(resid @ resid) / dof
        cov = sigma2 * np.linalg.inv(A.T @ A)
        cp_sd, w_sd = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
        method = "modello lineare 2-par (W = W' + CP*t)"
        pmax = None
    elif model == "hyperbolic":
        (cp, w_prime), cov = curve_fit(_cp_hyperbolic, t, p,
                                       p0=[p.min(), 20000], maxfev=10000)
        if cov is not None and np.all(np.isfinite(cov)):
            cp_sd, w_sd = float(np.sqrt(abs(cov[0, 0]))), float(np.sqrt(abs(cov[1, 1])))
        method = "modello iperbolico 2-par (P = CP + W'/t)"
        pmax = None
    else:  # 3param
        # Il 3-par serve SOLO se nel fit ci sono sforzi brevi (<30 s) che vincolano Pmax.
        # Altrimenti diverge: fallback automatico all'iperbolico.
        has_short = t.min() <= 30
        pmax = None
        if has_short:
            p0 = [p.min() * 0.9, 20000, float(mmp.get(1, p.max() * 1.5))]
            try:
                (cp, w_prime, pmax), cov = curve_fit(
                    _cp_3param, t, p, p0=p0, maxfev=20000,
                    bounds=([50, 1000, 500], [700, 60000, 3000]))
                method = "modello 3-par Morton"
                if not (cp < pmax < 3000):   # sanity check fisiologico
                    raise ValueError("Pmax non fisiologico")
                if cov is not None and np.all(np.isfinite(cov)):
                    cp_sd, w_sd = float(np.sqrt(abs(cov[0, 0]))), float(np.sqrt(abs(cov[1, 1])))
            except Exception:
                pmax = None
        if pmax is None:
            (cp, w_prime), cov = curve_fit(_cp_hyperbolic, t, p,
                                           p0=[p.min(), 20000], maxfev=10000)
            if cov is not None and np.all(np.isfinite(cov)):
                cp_sd, w_sd = float(np.sqrt(abs(cov[0, 0]))), float(np.sqrt(abs(cov[1, 1])))
            method = ("iperbolico 2-par (3-par non applicabile: "
                      "manca uno sforzo <30 s nel range di fit)")

    def _ci(val, sd):
        return (val - 1.96 * sd, val + 1.96 * sd) if sd is not None else None

    res = {
        "cp": Metric(float(cp), "W", Confidence.MEASURED, method,
                     "Asintoto potenza-durata. Proxy della soglia sostenibile.",
                     sd=cp_sd, ci=_ci(float(cp), cp_sd)),
        "w_prime": Metric(float(w_prime), "J", Confidence.MEASURED, method,
                          "Lavoro erogabile sopra CP. ~15-25 kJ tipico; sprinter piu' alto.",
                          sd=w_sd, ci=_ci(float(w_prime), w_sd)),
    }
    if pmax is not None:
        res["pmax"] = Metric(float(pmax), "W", Confidence.ESTIMATED, method,
                             "Potenza istantanea massima teorica (estrapolata).")
    return res


# --------------------------------------------------------------------------- #
#  4. FTP & MAP  —  ESTIMATED                                                  #
# --------------------------------------------------------------------------- #
def estimate_ftp(mmp: pd.Series, cp: Optional[float] = None) -> dict[str, Metric]:
    """
    FTP con metodi multipli. Nessuno e' 'la verita'': il gold standard e' la soglia
    lattacida in lab. Restituiamo tutte le stime disponibili cosi' l'utente vede la dispersione.
    """
    out = {}
    if 1200 in mmp.index:  # test 20 min
        out["ftp_20min"] = Metric(mmp[1200] * 0.95, "W", Confidence.ESTIMATED,
            "95% del miglior 20 min (Coggan)", "Il classico test da campo.")
    if 480 in mmp.index:   # test 8 min
        out["ftp_8min"] = Metric(mmp[480] * 0.90, "W", Confidence.ESTIMATED,
            "90% del miglior 8 min (Carmichael)", "Piu' breve, tende a sovrastimare.")
    if 3600 in mmp.index and mmp[3600] > 0:  # 'miglior 60 min' della ride
        out["ftp_60min"] = Metric(mmp[3600], "W", Confidence.ESTIMATED,
            "miglior 60 min della ride",
            "ATTENZIONE: vale come FTP solo se e' stato un test massimale sull'ora. "
            "In un'uscita normale NON e' FTP. Non usato per la raccomandazione.")
    if cp is not None:
        out["ftp_from_cp"] = Metric(cp * 0.97, "W", Confidence.ESTIMATED,
            "~97% di CP (da modello potenza-durata)",
            "Il metodo piu' principiato se hai sforzi massimali a piu' durate.")
    # Raccomandazione: CP (model-based) > test 20 min > test 8 min.
    # Il 60-min-da-ride-qualunque e' escluso di proposito.
    priority = ["ftp_from_cp", "ftp_20min", "ftp_8min"]
    for key in priority:
        if key in out:
            rec = out[key]
            # dispersione tra TUTTI i metodi disponibili = incertezza onesta sulla FTP
            estimates = [m.value for k, m in out.items()
                         if k in ("ftp_from_cp", "ftp_20min", "ftp_8min")]
            if len(estimates) >= 2:
                lo, hi = min(estimates), max(estimates)
                out["ftp_recommended"] = Metric(rec.value, "W", rec.confidence,
                    rec.method, rec.note + f" Range tra i metodi: {lo:.0f}–{hi:.0f} W.",
                    sd=(hi - lo) / 2, ci=(lo, hi))
            else:
                out["ftp_recommended"] = rec
            break
    return out


def maximal_aerobic_power(mmp: pd.Series, ramp_final_watt: Optional[float] = None
                          ) -> Metric:
    """
    MAP = potenza alla VO2max. Meglio da test rampa; in mancanza usiamo il miglior 5 min
    come proxy (leggermente sotto la vera MAP)."""
    if ramp_final_watt:
        return Metric(ramp_final_watt, "W", Confidence.ESTIMATED,
                      "potenza finale test rampa", "Metodo di riferimento per la MAP.")
    if 300 in mmp.index:
        return Metric(mmp[300], "W", Confidence.ESTIMATED,
                      "proxy = miglior 5 min", "Sottostima leggermente la MAP vera.")
    raise ValueError("Impossibile stimare MAP: manca sia il test rampa sia un 5 min.")


# --------------------------------------------------------------------------- #
#  5. VO2MAX  —  ESTIMATED (MEASURED solo se hai il metabolimetro)            #
# --------------------------------------------------------------------------- #
def estimate_vo2max(athlete: Athlete, map_watt: float) -> dict[str, Metric]:
    """
    ATTENZIONE ONESTA': la VO2max VERA si misura solo con analisi dei gas espirati.
    Da potenza otteniamo STIME (errore tipico +/-10-15%). Diamo 2 metodi indipendenti.
    Se hai un valore di lab, lo usiamo come MEASURED e ignoriamo le stime.
    """
    out = {}
    if athlete.vo2max_lab:
        out["vo2max"] = Metric(athlete.vo2max_lab, "mL/kg/min", Confidence.MEASURED,
            "metabolimetro (lab)", "Misura diretta degli scambi respiratori.")
        return out

    m = athlete.mass_kg
    # Metodo A - equazione ACSM cicloergometro, valutata alla MAP
    vo2_acsm = 1.8 * (map_watt * 6.12) / m + 7.0        # mL/kg/min
    out["vo2max_acsm"] = Metric(vo2_acsm, "mL/kg/min", Confidence.ESTIMATED,
        "ACSM cicloergometro alla MAP", "Assume MAP = potenza alla VO2max.")
    # Metodo B - Storer et al. 1990 (regressione su cicloergometro)
    if athlete.sex.upper() == "M":
        vo2_abs = 10.51 * map_watt + 6.35 * m - 10.49 * athlete.age + 519.3
    else:
        vo2_abs = 9.39 * map_watt + 7.7 * m - 5.88 * athlete.age + 136.7
    out["vo2max_storer"] = Metric(vo2_abs / m, "mL/kg/min", Confidence.ESTIMATED,
        "Storer et al. 1990", "Regressione di popolazione specifica per ciclismo.")
    # Media delle stime come valore mostrato; la DISPERSIONE tra i due metodi
    # indipendenti e' una misura onesta dell'incertezza (oltre all'errore intrinseco
    # ~+/-10-15% di ciascuna equazione). Prendiamo il piu' ampio tra i due.
    vals = [out["vo2max_acsm"].value, out["vo2max_storer"].value]
    mean_v = float(np.mean(vals))
    spread = float(abs(vals[0] - vals[1]) / 2)           # meta'-range tra i metodi
    intrinsic = mean_v * 0.12                             # ~12% errore tipico di popolazione
    band = float(np.hypot(spread, intrinsic))            # combina le due fonti di incertezza
    out["vo2max"] = Metric(mean_v, "mL/kg/min", Confidence.ESTIMATED,
        "media ACSM + Storer", "Stima; per un valore vero serve test in laboratorio.",
        sd=band, ci=(mean_v - band, mean_v + band))
    return out


# --------------------------------------------------------------------------- #
#  6. ZONE  —  ESTIMATED (dipendono dall'ancora: HRmax/LTHR/FTP)              #
# --------------------------------------------------------------------------- #
def hr_zones(athlete: Athlete) -> dict:
    """
    Preferenza metodologica: LTHR (Coggan) > Riserva HR/Karvonen > %HRmax.
    220-eta' e' sconsigliato: errore fino a +/-12 bpm. Usa HRmax o LTHR misurate.
    """
    if athlete.lthr:  # zone Coggan su LTHR - le piu' affidabili per ciclismo
        L = athlete.lthr
        z = {
            "Z1 Recupero":      (0, round(0.81 * L)),
            "Z2 Fondo":         (round(0.81 * L), round(0.89 * L)),
            "Z3 Tempo":         (round(0.90 * L), round(0.93 * L)),
            "Z4 Soglia":        (round(0.94 * L), round(0.99 * L)),
            "Z5a VO2 basso":    (round(1.00 * L), round(1.02 * L)),
            "Z5b VO2 alto":     (round(1.03 * L), round(1.06 * L)),
            "Z5c Anaerobico":   (round(1.06 * L), 999),
        }
        return {"method": "zone Coggan su LTHR (misurata)", "confidence": Confidence.ESTIMATED, "zones": z}

    hrmax, conf = athlete.hr_max_effective
    if athlete.hr_rest:  # Karvonen (riserva) - migliore di %HRmax puro
        rest = athlete.hr_rest
        def k(p): return round(rest + p * (hrmax - rest))
        z = {
            "Z1 Recupero": (k(0.50), k(0.60)),
            "Z2 Fondo":    (k(0.60), k(0.70)),
            "Z3 Tempo":    (k(0.70), k(0.80)),
            "Z4 Soglia":   (k(0.80), k(0.90)),
            "Z5 VO2max":   (k(0.90), hrmax),
        }
        return {"method": f"riserva HR/Karvonen (HRmax {conf.value})", "confidence": conf, "zones": z}

    # ultimo fallback: %HRmax
    z = {
        "Z1 Recupero": (round(0.50*hrmax), round(0.60*hrmax)),
        "Z2 Fondo":    (round(0.60*hrmax), round(0.70*hrmax)),
        "Z3 Tempo":    (round(0.70*hrmax), round(0.80*hrmax)),
        "Z4 Soglia":   (round(0.80*hrmax), round(0.90*hrmax)),
        "Z5 VO2max":   (round(0.90*hrmax), hrmax),
    }
    return {"method": f"%HRmax (HRmax {conf.value}) - il meno preciso", "confidence": conf, "zones": z}


COGGAN_POWER_ZONES = [  # (nome, low%FTP, high%FTP)
    ("Z1 Recupero attivo", 0.00, 0.55),
    ("Z2 Fondo",           0.56, 0.75),
    ("Z3 Tempo/Medio",     0.76, 0.90),
    ("Z4 Soglia",          0.91, 1.05),
    ("Z5 VO2max",          1.06, 1.20),
    ("Z6 Anaerobico",      1.21, 1.50),
    ("Z7 Neuromuscolare",  1.51, 9.99),
]

def power_zones(ftp: float) -> dict:
    z = {name: (round(lo * ftp), round(hi * ftp)) for name, lo, hi in COGGAN_POWER_ZONES}
    return {"method": "zone Coggan su FTP (7 zone)", "confidence": Confidence.ESTIMATED, "zones": z}


def time_in_zones(series: np.ndarray, zones: dict) -> dict[str, float]:
    """Secondi trascorsi in ciascuna zona (per il grafico di distribuzione)."""
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    out = {}
    for name, (lo, hi) in zones.items():
        out[name] = float(((s >= lo) & (s < hi)).sum())
    return out


# --------------------------------------------------------------------------- #
#  7. METRICHE DI CARICO (NP, IF, TSS, VI)  —  MEASURED                        #
# --------------------------------------------------------------------------- #
def normalized_power(power: np.ndarray) -> float:
    """NP = media mobile 30 s della potenza, elevata a 4, media, radice quarta."""
    p = pd.Series(np.nan_to_num(power, nan=0.0))
    roll = p.rolling(30, min_periods=1).mean()
    return float((roll.pow(4).mean()) ** 0.25)

def load_metrics(power: np.ndarray, ftp: float) -> dict[str, Metric]:
    p = np.nan_to_num(power, nan=0.0)
    dur_s = len(p)
    avg = float(p.mean())
    np_ = normalized_power(p)
    intf = np_ / ftp
    tss = (dur_s * np_ * intf) / (ftp * 3600) * 100
    work_kj = float(p.sum()) / 1000.0
    return {
        "avg_power": Metric(avg, "W", Confidence.MEASURED, "media aritmetica", ""),
        "normalized_power": Metric(np_, "W", Confidence.MEASURED, "algoritmo Coggan 30 s", ""),
        "intensity_factor": Metric(intf, "", Confidence.ESTIMATED, "NP/FTP", "dipende dalla FTP scelta"),
        "tss": Metric(tss, "TSS", Confidence.ESTIMATED, "Training Stress Score", "dipende dalla FTP scelta"),
        "variability_index": Metric(np_ / avg if avg else 0, "", Confidence.MEASURED, "NP/media",
                                    "vicino a 1 = sforzo regolare; alto = stop&go"),
        "work": Metric(work_kj, "kJ", Confidence.MEASURED, "integrale della potenza", ""),
    }


def best_np_window(power: np.ndarray, window_s: int) -> Optional[dict]:
    """
    Trova la finestra di durata `window_s` con la NP (potenza ponderata) piu' alta e
    ne restituisce NP, media e VI (=NP/media). Serve alla FTP-da-NP sugli sforzi lunghi
    e al flag di 'quanto era steady' un test.

    NP di ogni finestra = ( media su window_s di (media mobile 30 s)^4 )^(1/4).
    Implementazione O(n) con rolling di pandas. Ritorna None se la ride e' piu' corta.
    """
    p = pd.Series(np.nan_to_num(np.asarray(power, dtype=float), nan=0.0))
    if len(p) < window_s:
        return None
    r30 = p.rolling(30, min_periods=1).mean()
    np_win = (r30.pow(4).rolling(window_s).mean()) ** 0.25   # NP della finestra che finisce in i
    mean_win = p.rolling(window_s).mean()
    end = int(np_win.idxmax())
    np_v, mean_v = float(np_win.iloc[end]), float(mean_win.iloc[end])
    vi = np_v / mean_v if mean_v else 1.0
    return {"np": np_v, "mean": mean_v, "vi": vi,
            "start_s": max(0, end - window_s + 1), "end_s": end, "window_s": window_s}


def ftp_from_np_long(power: np.ndarray, window_s: int = 3600) -> Optional[Metric]:
    """
    FTP stimata dalla NP (potenza ponderata) del miglior sforzo ~lungo (default 60 min).

    RAZIONALE (perche' qui va bene e sul test 20 min no): su uno sforzo tipo-gara di
    ~1 h, variabile, la NP e' un proxy di FTP migliore della media grezza — ed e' gia'
    ~la durata della FTP, quindi NESSUNO sconto 0.95. Sul test 20 min invece la NP
    gonfia la FTP (disuguaglianza di Jensen) e va evitata: li' si usa la media.

    ONESTA': vale SOLO se quell'ora e' stata sostenuta e dura. Il VI della finestra dice
    quanto era steady; con VI alto (partenza/arrivo mossi) la stima e' meno affidabile.
    """
    bw = best_np_window(power, window_s)
    if bw is None:
        return None
    mins = window_s // 60
    return Metric(bw["np"], "W", Confidence.ESTIMATED,
                  f"NP del miglior ~{mins} min",
                  f"Valida SOLO se quell'ora e' stata uno sforzo tipo-gara sostenuto "
                  f"(VI finestra {bw['vi']:.2f}). In un'uscita facile NON e' FTP.")


# --------------------------------------------------------------------------- #
#  8. CALORIE & SUBSTRATI                                                      #
# --------------------------------------------------------------------------- #
def calories_from_power(power: np.ndarray, gross_efficiency: float = 0.24
                        ) -> Metric:
    """
    MEASURED (alta affidabilita' per il ciclismo con potenza).
    Lavoro meccanico (kJ) / efficienza lorda / 4.184 = kcal.
    Con GE~0.24: kcal ~= kJ (comodo e valido). GE tipica 0.20-0.25.
    """
    work_kj = float(np.nan_to_num(power, nan=0.0).sum()) / 1000.0
    kcal = work_kj / gross_efficiency / 4.184
    return Metric(kcal, "kcal", Confidence.MEASURED,
                  f"lavoro/GE (GE={gross_efficiency:.2f})",
                  "Il ciclismo con power meter e' uno dei pochi casi in cui le kcal sono affidabili.")


def calories_from_hr(hr: np.ndarray, athlete: Athlete, vo2max: Optional[float] = None
                     ) -> Metric:
    """
    Fallback per uscite SENZA potenza. Equazione di Keytel et al. 2005 (usa HR, eta',
    massa, sesso; opzionale VO2max). Meno affidabile della via meccanica: ESTIMATED.
    """
    hr = np.asarray(hr, dtype=float)
    hr = hr[~np.isnan(hr)]
    mins = len(hr) / 60.0
    mean_hr = hr.mean()
    if athlete.sex.upper() == "M":
        kcal_min = (-55.0969 + 0.6309*mean_hr + 0.1988*athlete.mass_kg + 0.2017*athlete.age) / 4.184
    else:
        kcal_min = (-20.4022 + 0.4472*mean_hr - 0.1263*athlete.mass_kg + 0.0740*athlete.age) / 4.184
    return Metric(max(kcal_min, 0) * mins, "kcal", Confidence.ESTIMATED,
                  "Keytel et al. 2005 (da HR)",
                  "Usato solo se manca la potenza; errore maggiore.")


def substrate_split(power: np.ndarray, map_watt: float, athlete: Athlete,
                    rer: Optional[np.ndarray] = None,
                    total_kcal: Optional[float] = None) -> dict[str, Metric]:
    """
    Ripartizione energia grassi vs carboidrati.

    - Se hai RER (da metabolimetro): MEASURED. Usiamo Jeukendrup & Wallis 2005.
    - Altrimenti: MODELED. Stimiamo %CHO come funzione logistica dell'intensita'
      (%MAP ~ %VO2max), con crossover ~ zona fondo/tempo. E' un modello di POPOLAZIONE:
      il tuo crossover reale puo' differire di 10-15 %VO2max. Solo indicativo.
    """
    p = np.nan_to_num(power, nan=0.0)

    if rer is not None:  # via misurata
        # richiederebbe anche VO2, VCO2 assoluti; qui semplifichiamo su %CHO da RER
        rer = np.clip(np.asarray(rer, dtype=float), 0.70, 1.00)
        pct_cho = (rer - 0.70) / (1.00 - 0.70)          # 0.70->0% CHO, 1.00->100% CHO
        conf, method = Confidence.MEASURED, "da RER misurato (Jeukendrup-Wallis)"
    else:                # via modellata
        intensity = np.clip(p / map_watt, 0, 1.4)       # frazione di MAP ~ %VO2max
        x0 = athlete.fatmax_pct_vo2max/100 if athlete.fatmax_pct_vo2max else 0.62
        k = 9.0                                         # pendenza del crossover
        pct_cho = 1.0 / (1.0 + np.exp(-k * (intensity - x0)))
        conf, method = Confidence.MODELED, "logistica su %MAP (modello di popolazione)"

    # pesa la percentuale per l'energia spesa a ogni istante (proxy: potenza)
    w = p / p.sum() if p.sum() else np.ones_like(p) / len(p)
    mean_cho = float(np.sum(pct_cho * w))
    mean_fat = 1.0 - mean_cho

    if total_kcal is None:
        total_kcal = float(p.sum()) / 1000.0 / 0.24 / 4.184
    cho_kcal, fat_kcal = total_kcal * mean_cho, total_kcal * mean_fat
    return {
        "pct_carb": Metric(mean_cho * 100, "%", conf, method, ""),
        "pct_fat":  Metric(mean_fat * 100, "%", conf, method, ""),
        "carb_g":   Metric(cho_kcal / 4.0, "g", conf, method, "1 g CHO ~ 4 kcal"),
        "fat_g":    Metric(fat_kcal / 9.5, "g", conf, method, "1 g grasso ~ 9.5 kcal"),
    }


def fatmax(map_watt: float, athlete: Athlete) -> Metric:
    """
    FatMax = intensita' di massima ossidazione dei grassi.
    MODELED senza laboratorio: la sua determinazione VERA richiede un test incrementale
    con metabolimetro (curva di ossidazione grassi). Restituiamo la stima del modello
    (o il valore di lab se fornito). Da trattare come indicativo, NON come misura.
    """
    if athlete.fatmax_pct_vo2max:
        pct = athlete.fatmax_pct_vo2max
        return Metric(map_watt * pct/100, "W", Confidence.MEASURED,
                      f"test FatMax lab ({pct:.0f}% VO2max)", "Potenza al FatMax misurato.")
    # popolazione: FatMax tipico ~ 55-65% VO2max nei ben allenati
    pct = 62.0
    return Metric(map_watt * pct/100, "W", Confidence.MODELED,
                  "stima popolazione (~62% VO2max)",
                  "Individuale variabile: serve test incrementale con analisi gas.")


# --------------------------------------------------------------------------- #
#  9. FABBISOGNO CALORICO & FUELING  —  ESTIMATED                              #
# --------------------------------------------------------------------------- #
def daily_energy(athlete: Athlete, exercise_kcal: float,
                 activity_factor: float = 1.5) -> dict[str, Metric]:
    """
    BMR con Mifflin-St Jeor (piu' accurata di Harris-Benedict).
    Fabbisogno = BMR*fattore_attivita' (vita non sportiva) + kcal dell'allenamento.
    """
    m, h, a = athlete.mass_kg, athlete.height_cm, athlete.age
    bmr = 10*m + 6.25*h - 5*a + (5 if athlete.sex.upper() == "M" else -161)
    tdee = bmr * activity_factor + exercise_kcal
    return {
        "bmr": Metric(bmr, "kcal", Confidence.ESTIMATED, "Mifflin-St Jeor", "metabolismo basale"),
        "tdee": Metric(tdee, "kcal", Confidence.ESTIMATED,
                       f"BMR*{activity_factor} + allenamento", "fabbisogno giornaliero totale"),
    }


def fueling_plan(duration_s: float, intensity_if: float, athlete: Athlete) -> dict:
    """
    Linee guida evidence-based (ACSM / Jeukendrup) — range, non numeri magici.
    """
    hours = duration_s / 3600.0
    m = athlete.mass_kg

    # CHO durante (g/h)
    if hours < 0.75:
        during = "0 g/h (o mouth-rinse); riserve sufficienti"
    elif hours <= 2.5:
        during = "30-60 g/h (glucosio/maltodestrine)"
    else:
        during = "60-90 g/h (mix glucosio:fruttosio 2:1)"

    pre = f"{1*m:.0f}-{3*m:.0f} g CHO nelle 1-3 h prima ({1:.0f}-{3:.0f} g/kg)"
    post_cho = f"{1.0*m:.0f}-{1.2*m:.0f} g CHO nelle prime 1-2 h ({1.0}-{1.2} g/kg/h)"
    post_pro = f"{0.3*m:.0f} g proteine (0.3 g/kg) per la sintesi proteica"

    return {
        "confidence": Confidence.ESTIMATED,
        "method": "linee guida ACSM/Jeukendrup",
        "pre_workout": pre,
        "during_workout": during,
        "post_workout": f"{post_cho}; + {post_pro}",
        "note": "Piu' rilevante per uscite >90 min o alta intensita'. Personalizza in base a tolleranza gastrica.",
    }


# --------------------------------------------------------------------------- #
# 10. CLASSIFICAZIONE CORRIDORE                                                #
# --------------------------------------------------------------------------- #
# Benchmark W/kg per durate chiave. NOTA DI ONESTA' INTELLETTUALE:
# - I livelli fino a "elite amatoriale" derivano dal Power Profile di Coggan (dati robusti).
# - I livelli PRO derivano da letteratura (Pinot-Grappe RPP, van Erp/Sanders, Leo 2022) e,
#   per i tier "top-20 Grande Giro" / "top-10 Tour", da STIME di potenza sulle salite
#   (SRM trapelati, analisi VAM/Portoleau-Grappe): NON esiste un dataset di laboratorio
#   pulito per questi tier. Trattali come ordini di grandezza, non come cutoff esatti.
# Valori indicativi maschili; per le donne scala ~ -15% circa (o passa tabella dedicata).
RIDER_BENCHMARKS_M = {
    # durata_s : {categoria: W/kg}
    5: {   # sprint neuromuscolare
        "amatore":            11.0, "cat1_elite_amat":   16.0, "continental":       18.0,
        "professional":       20.0, "world_tour":        22.0, "top20_grande_giro": 23.0,
        "top10_tdf":          23.5,
    },
    60: {  # capacita' anaerobica
        "amatore":             7.5, "cat1_elite_amat":    9.5, "continental":       10.5,
        "professional":       11.5, "world_tour":        12.5, "top20_grande_giro": 13.0,
        "top10_tdf":          13.3,
    },
    300: { # potenza aerobica massima (~VO2max)
        "amatore":             4.3, "cat1_elite_amat":    5.3, "continental":        6.0,
        "professional":        6.6, "world_tour":         7.0, "top20_grande_giro":  7.3,
        "top10_tdf":           7.6,
    },
    1200: {# soglia / potenza salita 20 min (proxy FTP)
        "amatore":             3.2, "cat1_elite_amat":    4.2, "continental":        4.9,
        "professional":        5.5, "world_tour":         6.0, "top20_grande_giro":  6.3,
        "top10_tdf":           6.6,
    },
}
CATEGORY_ORDER = ["amatore", "cat1_elite_amat", "continental", "professional",
                  "world_tour", "top20_grande_giro", "top10_tdf"]
CATEGORY_LABELS = {
    "amatore": "Amatore", "cat1_elite_amat": "Elite amatoriale / Cat.1",
    "continental": "Continental", "professional": "Professional (Pro Team)",
    "world_tour": "World Tour", "top20_grande_giro": "Top-20 Grande Giro",
    "top10_tdf": "Top-10 Tour de France",
}


def classify_category(mmp: pd.Series, mass_kg: float) -> dict:
    """
    Per ogni durata chiave, colloca l'atleta nella categoria piu' alta il cui benchmark
    e' superato. Restituisce W/kg, categoria e percentile-like per durata.
    """
    out = {}
    for dur, table in RIDER_BENCHMARKS_M.items():
        if dur not in mmp.index:
            continue
        wkg = mmp[dur] / mass_kg
        reached = "sotto amatore"
        for cat in CATEGORY_ORDER:
            if wkg >= table[cat]:
                reached = cat
        out[dur] = {
            "w_kg": round(wkg, 2),
            "watt": round(mmp[dur]),
            "category": reached,
            "category_label": CATEGORY_LABELS.get(reached, reached),
        }
    return {
        "per_duration": out,
        "confidence": Confidence.ESTIMATED,
        "note": ("Livelli amatoriali: Coggan (robusti). Livelli pro/GT/Tour: stime da "
                 "letteratura e analisi salite, NON dataset di lab. Ordini di grandezza."),
    }


def rider_phenotype(mmp: pd.Series, mass_kg: float) -> dict:
    """
    Tipo di corridore dalla FORMA del profilo: confronta la forza relativa alle varie
    durate (5s sprint / 60s anaerobico / 5min VO2 / 20min soglia) rispetto ai benchmark,
    e vede dove eccelli RELATIVAMENTE a te stesso.
    """
    key = {5: "sprint", 60: "anaerobico", 300: "vo2max", 1200: "soglia"}
    scores = {}
    for dur, label in key.items():
        if dur in mmp.index and dur in RIDER_BENCHMARKS_M:
            wkg = mmp[dur] / mass_kg
            # punteggio = quanto sopra/sotto la mediana della scala benchmark
            scale = RIDER_BENCHMARKS_M[dur]
            lo, hi = scale["amatore"], scale["top10_tdf"]
            scores[label] = (wkg - lo) / (hi - lo)   # 0..1+ posizione nella scala pro
    if not scores:
        return {"phenotype": "dati insufficienti", "scores": {}}

    # normalizza rispetto alla media dell'atleta -> punti forti relativi
    mean_s = np.mean(list(scores.values()))
    rel = {k: v - mean_s for k, v in scores.items()}
    strong = max(rel, key=rel.get)
    phenotype = {
        "sprint": "Velocista (sprinter)",
        "anaerobico": "Passista veloce / Puncheur",
        "vo2max": "Finisseur / Puncheur da VO2max",
        "soglia": "Passista / Scalatore da cronometro",
    }[strong]
    # scalatore vs cronoman: se soglia forte, distingue su rapporto peso
    return {
        "phenotype": phenotype,
        "dominant_quality": strong,
        "scores_0to1_pro_scale": {k: round(v, 2) for k, v in scores.items()},
        "relative_strengths": {k: round(v, 2) for k, v in rel.items()},
        "confidence": Confidence.ESTIMATED,
    }


# --------------------------------------------------------------------------- #
# 11. LOADER MULTI-SORGENTE  (FIT / CSV / intervals.icu)                       #
# --------------------------------------------------------------------------- #
# Denominatore comune: qualunque sorgente -> DataFrame con colonne STD_COLS,
# poi passa da to_1hz(). FIT copre Garmin/Polar/Wahoo (tutti esportano .fit).

COL_ALIASES = {
    "t":        ["timestamp", "time", "secs", "elapsed", "seconds", "datetime"],
    "power":    ["power", "watts", "pwr", "power_w"],
    "hr":       ["hr", "heartrate", "heart_rate", "bpm", "heart rate"],
    "cadence":  ["cadence", "cad", "rpm"],
    "speed":    ["speed", "velocity_smooth", "kph", "enhanced_speed", "velocity"],
    "altitude": ["altitude", "elevation", "alt", "enhanced_altitude", "ele"],
    "lat":      ["lat", "latitude", "position_lat", "pos_lat"],
    "lon":      ["lon", "lng", "long", "longitude", "position_long", "position_lon", "pos_long"],
}

def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mappa nomi colonna eterogenei (Garmin/intervals.icu/export vari) allo standard."""
    lower = {c.lower().strip(): c for c in df.columns}
    out = pd.DataFrame()
    for std, aliases in COL_ALIASES.items():
        for a in aliases:
            if a in lower:
                out[std] = df[lower[a]]
                break
    if "t" not in out:                      # nessuna colonna tempo -> assume 1 Hz
        out["t"] = np.arange(len(df))
    return out

def parse_records(records: list[dict]) -> pd.DataFrame:
    """Nucleo puro e testabile: lista di record (dict) -> DataFrame standard."""
    df = pd.DataFrame(records)
    return _resolve_columns(df)

def load_csv(file) -> pd.DataFrame:
    """CSV da Garmin Connect / intervals.icu / export generici, con alias colonne."""
    return _resolve_columns(pd.read_csv(file))

def _semicircles_to_deg(v):
    """FIT registra lat/lon in 'semicircles' (int32). Converte in gradi decimali."""
    if v is None:
        return None
    return float(v) * (180.0 / 2 ** 31)

def load_fit(file) -> pd.DataFrame:
    """
    File .fit (Garmin, Polar, Wahoo, ...). Legge i messaggi 'record'.
    Gestisce campi mancanti e developer fields (alcuni power meter li usano).
    Estrae anche la posizione GPS (semicircles -> gradi) per la mappa del percorso.
    """
    from fitparse import FitFile
    fit = FitFile(file)
    rows = []
    for rec in fit.get_messages("record"):
        d = {f.name: f.value for f in rec}
        rows.append({
            "timestamp": d.get("timestamp"),
            "power":     d.get("power", d.get("Power")),
            "hr":        d.get("heart_rate"),
            "cadence":   d.get("cadence"),
            "speed":     d.get("enhanced_speed", d.get("speed")),
            "altitude":  d.get("enhanced_altitude", d.get("altitude")),
            "lat":       _semicircles_to_deg(d.get("position_lat")),
            "lon":       _semicircles_to_deg(d.get("position_long")),
        })
    return parse_records(rows)

def load_intervals_icu(activity_id: str, api_key: str) -> pd.DataFrame:
    """
    Streams (serie temporali a 1 Hz) di un'attivita' da intervals.icu.
    Auth: HTTP Basic, username letterale 'API_KEY', password = la tua chiave.
    L'activity_id e' nell'URL dell'attivita' (es. .../activities/i12345 -> "i12345").
    Endpoint: GET /api/v1/activity/{id}/streams

    NB Cloudflare: intervals.icu e' dietro Cloudflare, che blocca lo User-Agent di
    default di requests/urllib. Serve uno UA da browser (impostato sotto), altrimenti
    ricevi 403. Limiti chiave personale: 5000 richieste/giorno (solo uso personale;
    per multi-utente serve OAuth + Bearer token). Docs: https://intervals.icu/api-docs.html
    """
    import requests
    url = f"https://intervals.icu/api/v1/activity/{activity_id}/streams"
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0 Safari/537.36")}
    r = requests.get(url, auth=("API_KEY", api_key), headers=headers,
                     params={"types": "time,watts,heartrate,cadence,velocity_smooth,altitude,latlng"},
                     timeout=30)
    r.raise_for_status()
    streams = {s["type"]: s["data"] for s in r.json()}
    latlng = streams.get("latlng") or []
    lat = [p[0] if isinstance(p, (list, tuple)) and len(p) == 2 else None for p in latlng]
    lon = [p[1] if isinstance(p, (list, tuple)) and len(p) == 2 else None for p in latlng]
    df = pd.DataFrame({
        "t":        streams.get("time", list(range(len(streams.get("watts", []))))),
        "power":    streams.get("watts"),
        "hr":       streams.get("heartrate"),
        "cadence":  streams.get("cadence"),
        "speed":    streams.get("velocity_smooth"),
        "altitude": streams.get("altitude"),
        "lat":      lat or None,
        "lon":      lon or None,
    })
    return _resolve_columns(df)


def list_intervals_activities(athlete_id: str, api_key: str,
                              days_back: int = 120, limit: int = 40) -> list[dict]:
    """
    Elenca le attivita' recenti di un atleta su intervals.icu (per sceglierne una).
    athlete_id: il tuo ID atleta (es. "i382978") oppure "0" = atleta della chiave.
    Ritorna una lista di {id, name, date, type} ordinata dalla piu' recente.
    Endpoint: GET /api/v1/athlete/{id}/activities?oldest=...&newest=...
    NB: stesso schema auth + User-Agent browser (Cloudflare) di load_intervals_icu.
    Non testato contro l'API live da qui: se i nomi dei campi differissero, segnala
    cosa restituisce e allineo il parsing.
    """
    import requests
    from datetime import date, timedelta
    url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/activities"
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0 Safari/537.36")}
    params = {"oldest": (date.today() - timedelta(days=days_back)).isoformat(),
              "newest": date.today().isoformat()}
    r = requests.get(url, auth=("API_KEY", api_key), headers=headers,
                     params=params, timeout=30)
    r.raise_for_status()
    acts = r.json()
    rows = [{"id": a.get("id"),
             "name": a.get("name") or "",
             "date": (a.get("start_date_local") or a.get("start_date") or "")[:10],
             "type": a.get("type") or "",
             "load": a.get("icu_training_load") or a.get("training_load") or 0,
             "moving_time": a.get("moving_time") or a.get("elapsed_time") or 0,
             "distance": a.get("distance") or 0,
             "has_power": bool(a.get("icu_average_watts") or a.get("average_watts"))}
            for a in acts if a.get("id")]
    rows.sort(key=lambda x: x["date"], reverse=True)
    return rows[:limit]


def season_power_curve_from_intervals(athlete_id: str, api_key: str,
                                      activity_ids: list[str],
                                      progress_cb=None) -> tuple[pd.Series, int]:
    """
    Costruisce la CURVA DI POTENZA STAGIONALE scaricando gli streams di piu' attivita'
    e prendendo il massimo punto-a-punto (season_power_curve). Una chiamata /streams
    per attivita': fa pacing (~8 req/s) per rispettare il limite di 10/s di intervals.icu.
    Ritorna (curva MMP aggregata, numero di attivita' effettivamente usate).
    """
    curves, used = [], 0
    n = len(activity_ids) or 1
    for i, aid in enumerate(activity_ids):
        try:
            df1 = to_1hz(load_intervals_icu(aid, api_key))
            if "power" in df1 and float(df1["power"].sum()) > 0:
                curves.append(mean_maximal_power(df1["power"].values))
                used += 1
        except Exception:
            pass                      # attivita' senza potenza / errore singolo: salta
        if progress_cb:
            progress_cb((i + 1) / n)
        time.sleep(0.12)              # ~8 richieste/secondo
    if not curves:
        return pd.Series(dtype=float), 0
    season = pd.concat(curves, axis=1).max(axis=1).rename("mmp_watt")
    return season, used


# --------------------------------------------------------------------------- #
# 12. TASSONOMIA COMPLETA DEL CORRIDORE                                        #
# --------------------------------------------------------------------------- #
# Discriminatore chiave scalatore vs cronoman/rouleur: W/kg (relativo, conta in
# salita) vs Watt assoluti (contano sul piano/crono). Confronto il rango del
# corridore sulle due scale. I benchmark assoluti sono approssimativi (uomini).
ABS_BENCH = {
    5:    {"amatore":800,"cat1_elite_amat":1100,"continental":1300,"professional":1450,
           "world_tour":1600,"top20_grande_giro":1700,"top10_tdf":1750},  # sprint (W)
    1200: {"amatore":210,"cat1_elite_amat":300,"continental":360,"professional":400,
           "world_tour":430,"top20_grande_giro":450,"top10_tdf":470},     # soglia 20' (W)
}

def _scale_rank(value, table) -> float:
    """Posizione 0..1 di un valore sulla scala amatore->top10_tdf."""
    lo, hi = table["amatore"], table["top10_tdf"]
    return (value - lo) / (hi - lo)

def rider_type_full(mmp: pd.Series, mass_kg: float) -> dict:
    """
    Tipo di corridore 'generale': Velocista, Puncheur/Finisseur, Scalatore,
    Cronoman/Passista da fuga, Passista completo.
    Logica: forza relativa alle varie durate (su scala W/kg pro) + conferme sui
    Watt assoluti per distinguere sprinter veri e scalatori da cronomen.
    """
    key = {5: "sprint", 60: "anaerobico", 300: "vo2max", 1200: "soglia"}
    q = {}
    for dur, name in key.items():
        if dur in mmp.index and dur in RIDER_BENCHMARKS_M:
            wkg = mmp[dur] / mass_kg
            q[name] = {
                "wkg": round(wkg, 2), "watt": round(mmp[dur]),
                "score": round(_scale_rank(wkg, RIDER_BENCHMARKS_M[dur]), 2),
            }
    if not q:
        return {"primary": "dati insufficienti", "qualities": {}}

    mean_s = np.mean([v["score"] for v in q.values()])
    rel = {k: round(v["score"] - mean_s, 2) for k, v in q.items()}
    dominant = max(rel, key=rel.get)

    # conferme assolute
    sprint_abs = _scale_rank(mmp[5], ABS_BENCH[5]) if 5 in mmp.index else 0
    thr_abs = _scale_rank(mmp[1200], ABS_BENCH[1200]) if 1200 in mmp.index else 0
    thr_wkg = q.get("soglia", {}).get("score", 0)

    reasoning = []
    if dominant == "sprint" and sprint_abs > 0.35:
        primary = "Velocista (sprinter)"
        reasoning.append(f"Sprint dominante ({q['sprint']['wkg']} W/kg, {q['sprint']['watt']} W assoluti).")
    elif dominant in ("anaerobico", "vo2max"):
        primary = "Puncheur / Finisseur"
        reasoning.append("Picco di forza sugli sforzi brevi-intensi (1-5 min): strappi e finali mossi.")
    elif dominant == "soglia":
        if thr_wkg - thr_abs > 0.12:
            primary = "Scalatore"
            reasoning.append(f"Soglia forte in RELATIVO ({q['soglia']['wkg']} W/kg): il vantaggio emerge quando conta il peso (salita).")
        elif thr_abs - thr_wkg > 0.12:
            primary = "Cronoman / Passista da fuga"
            reasoning.append(f"Soglia forte in ASSOLUTO ({q['soglia']['watt']} W): il vantaggio emerge sul piano e a crono.")
        else:
            primary = "Passista completo"
            reasoning.append("Soglia solida sia in relativo sia in assoluto: adatto a fughe e ritmi prolungati.")
    else:
        primary = "Passista completo (all-rounder)"
        reasoning.append("Profilo bilanciato senza un picco netto.")

    secondary = sorted(rel, key=rel.get, reverse=True)
    secondary = [s for s in secondary if s != dominant][:1]
    sec_label = {"sprint":"spunto veloce","anaerobico":"capacità anaerobica",
                 "vo2max":"potenza aerobica","soglia":"tenuta alla soglia"}
    if secondary:
        reasoning.append(f"Qualità secondaria: {sec_label.get(secondary[0], secondary[0])}.")

    return {
        "primary": primary,
        "reasoning": " ".join(reasoning),
        "qualities": q,          # wkg, watt, score(0-1 scala pro) per durata
        "relative_strengths": rel,
        "confidence": Confidence.ESTIMATED,
    }


# --------------------------------------------------------------------------- #
# 13. DURABILITY / RESISTENZA ALLA FATICA  —  MEASURED                         #
# --------------------------------------------------------------------------- #
def durability(power, durations=(5, 15, 60, 300, 1200), kj_threshold: float = 2000):
    """
    Il differenziatore vs Strava/TrainingPeaks: la curva di potenza DA STANCO.
    Confronta la potenza massimale 'da fresco' (tutta l'uscita) con quella calcolata
    SOLO dopo aver accumulato `kj_threshold` kJ di lavoro. La caduta % a ogni durata
    misura quanto reggi con le gambe pesanti (fatigue resistance).

    CAVEAT onesto: ha senso solo se DOPO la soglia hai fatto sforzi (quasi) massimali;
    se dopo 2000 kJ hai solo pedalato piano, la 'caduta' e' un artefatto. Su singola
    uscita e' indicativo; il quadro vero emerge aggregando piu' uscite lunghe.
    """
    p = np.nan_to_num(np.asarray(power, dtype=float), nan=0.0)
    cum_kj = np.cumsum(p) / 1000.0
    idx = int(np.searchsorted(cum_kj, kj_threshold))
    out = {"kj_threshold": kj_threshold, "reached": idx < len(p),
           "total_kj": round(float(cum_kj[-1]), 0) if len(cum_kj) else 0,
           "confidence": Confidence.MEASURED, "per_duration": {}}
    if not out["reached"]:
        return out
    fresh = mean_maximal_power(p, list(durations))
    fatigued = mean_maximal_power(p[idx:], list(durations))
    for dsec in durations:
        f, g = fresh.get(dsec), fatigued.get(dsec)
        if f and g:
            out["per_duration"][dsec] = {"fresh": round(f), "fatigued": round(g),
                                         "drop_pct": round((f - g) / f * 100, 1)}
    return out


# --------------------------------------------------------------------------- #
# 14. WELLNESS (HRV / HR a riposo / sonno) da intervals.icu                    #
# --------------------------------------------------------------------------- #
def load_intervals_wellness(athlete_id: str, api_key: str, days_back: int = 60) -> pd.DataFrame:
    """
    Dati di benessere giornalieri da intervals.icu: HRV (rMSSD), HR a riposo, sonno.
    Endpoint: GET /api/v1/athlete/{id}/wellness?oldest=...&newest=...
    Ritorna un DataFrame con colonne date, hrv, resting_hr, sleep_hours (assenti se il
    campo non e' presente). Stesso auth + User-Agent browser (Cloudflare) degli altri
    loader. NON testato contro l'API live da qui.
    """
    import requests
    from datetime import date, timedelta
    url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness"
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")}
    params = {"oldest": (date.today() - timedelta(days=days_back)).isoformat(),
              "newest": date.today().isoformat()}
    r = requests.get(url, auth=("API_KEY", api_key), headers=headers, params=params, timeout=30)
    r.raise_for_status()
    rows = []
    for w in r.json():
        ss = w.get("sleepSecs")
        rows.append({"date": w.get("id"),
                     "hrv": w.get("hrv"),
                     "resting_hr": w.get("restingHR"),
                     "sleep_hours": (ss / 3600.0) if ss else np.nan})
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 15. CURVA DI OSSIDAZIONE DEI GRASSI (ottimizzazione consumo grassi)          #
# --------------------------------------------------------------------------- #
def fat_oxidation_curve(map_watt: float, athlete: "Athlete", n: int = 60) -> dict:
    """
    Curva di ossidazione grassi vs intensita' (MODELLATO). Per una griglia di potenze
    stima g/min di grassi e carboidrati con lo stesso modello logistico di
    substrate_split + il dispendio energetico a quella potenza. Il PICCO della curva
    dei grassi = FatMax (potenza di massimo consumo di grassi): a che intensita'
    ottimizzi il consumo di grassi.

    A bassa intensita' bruci soprattutto grassi ma poca energia totale -> pochi g/min;
    ad alta intensita' spendi tanto ma quasi solo carboidrati -> pochi grassi. Il
    massimo sta nel mezzo.

    ONESTA': senza analisi dei gas espirati e' un modello di POPOLAZIONE; il tuo FatMax
    reale puo' variare di 10-15 %VO2max. Per il valore vero serve un test incrementale
    con metabolimetro (o passa athlete.fatmax_pct_vo2max se l'hai misurato).
    """
    watts = np.linspace(0.20 * map_watt, 1.10 * map_watt, n)
    x0 = athlete.fatmax_pct_vo2max / 100 if athlete.fatmax_pct_vo2max else 0.62
    k = 9.0
    rows = []
    for w in watts:
        intensity = float(np.clip(w / map_watt, 0, 1.4))
        pct_cho = 1.0 / (1.0 + np.exp(-k * (intensity - x0)))
        pct_fat = 1.0 - pct_cho
        kcal_min = (w * 60 / 1000) / 0.24 / 4.184        # dispendio energetico a quella potenza
        rows.append({"watt": round(w),
                     "pct_intensity": round(intensity * 100),
                     "fat_g_min": kcal_min * pct_fat / 9.5,
                     "cho_g_min": kcal_min * pct_cho / 4.0})
    df = pd.DataFrame(rows)
    imax = int(df["fat_g_min"].idxmax())
    return {"curve": df,
            "fatmax_watt": float(df.loc[imax, "watt"]),
            "fatmax_pct": float(df.loc[imax, "pct_intensity"]),
            "fatmax_fat_g_min": float(df.loc[imax, "fat_g_min"]),
            "confidence": Confidence.MODELED,
            "note": "Modello di popolazione: per il FatMax vero serve un test con metabolimetro."}


# --------------------------------------------------------------------------- #
# 16. CONFRONTO CON I PRO (+ Pogacar) E CLASSIFICA AMATORI                     #
# --------------------------------------------------------------------------- #
# Stime W/kg di Pogacar da analisi di potenza sulle salite (Portoleau/Grappe-style),
# NON dati ufficiali di laboratorio. Da prendere per quello che sono: ordini di
# grandezza per un confronto divertente.
POGACAR_WKG = {5: 22.5, 60: 13.0, 300: 7.8, 1200: 6.9}

def pro_comparison(mmp: pd.Series, mass_kg: float) -> dict:
    """Confronto W/kg dell'atleta vs Continental / Professional / World Tour / Pogacar."""
    durs = {5: "Sprint 5s", 60: "1 min", 300: "5 min", 1200: "20 min"}
    rows = []
    for d, label in durs.items():
        if d in mmp.index and d in RIDER_BENCHMARKS_M:
            wkg = mmp[d] / mass_kg
            pog = POGACAR_WKG.get(d)
            rows.append({"durata": label, "dur_s": d, "tu": round(wkg, 2),
                         "continental": RIDER_BENCHMARKS_M[d]["continental"],
                         "professional": RIDER_BENCHMARKS_M[d]["professional"],
                         "world_tour": RIDER_BENCHMARKS_M[d]["world_tour"],
                         "pogacar": pog,
                         "pct_pogacar": round(wkg / pog * 100) if pog else None})
    return {"rows": rows, "confidence": Confidence.ESTIMATED,
            "note": ("Valori pro e Pogacar sono STIME da analisi di potenza sulle salite, "
                     "non dati ufficiali. Solo per confronto indicativo (e divertente).")}

# Fasce amatoriali su FTP/20-min W/kg (uomini, approssimative). low/middle/top.
AMATEUR_BANDS = [
    ("Amatore base (low)",       0.0, 3.1),
    ("Amatore intermedio (mid)", 3.1, 3.8),
    ("Amatore avanzato (top)",   3.8, 4.5),
    ("Agonista / Elite amat.",   4.5, 99.0),
]

def classify_amateur(ftp_wkg: float) -> dict:
    """Colloca l'atleta tra gli amatori: base(low) / intermedio(mid) / avanzato(top)."""
    tier = "n/d"
    for name, lo, hi in AMATEUR_BANDS:
        if lo <= ftp_wkg < hi:
            tier = name
            break
    # posizione 0-1 sulla scala amatoriale complessiva (~2.0 -> 4.5 W/kg)
    pos = min(1.0, max(0.0, (ftp_wkg - 2.0) / (4.5 - 2.0)))
    return {"tier": tier, "ftp_wkg": round(ftp_wkg, 2), "position": pos,
            "bands": AMATEUR_BANDS, "confidence": Confidence.ESTIMATED}
