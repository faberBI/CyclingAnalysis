"""
physics.py
==========
Analisi FISICA del ciclismo: sfrutta quota + velocita' (finora caricate ma inutilizzate).

Contenuti:
  1. Preprocessing cinematico: velocita' -> m/s, distanza, pendenza (smussata).
  2. VAM (velocita' ascensionale media) e rilevamento automatico delle salite.
  3. Modello di potenza fisico (gravita' + rotolamento + aerodinamica + inerzia).
  4. Stima di potenza SENZA power meter (da pendenza/velocita'/massa).
  5. Metodo Chung "virtual elevation": stima CdA e Crr dai dati di una uscita.

Filosofia coerente col motore: la VAM e la distanza sono MISURATE (solo geometria);
la potenza-da-fisica e CdA/Crr sono ESTIMATED (dipendono da massa, densita' aria,
assunzione vento=0). Ogni output porta la sua Confidence.

Riferimenti:
- Martin et al. 1998 (modello di potenza per ciclismo su strada)
- Chung R., "Estimating CdA with a power meter" (virtual elevation / 'aerolab')
- di Prampero et al.; Grappe (VAM come proxy di potenza in salita)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from cycling_analytics import Metric, Confidence

G = 9.80665                      # accelerazione di gravita' (m/s^2)
DRIVETRAIN_EFF = 0.976           # efficienza trasmissione (perdite catena ~2.4%)


# --------------------------------------------------------------------------- #
#  0. DENSITA' DELL'ARIA                                                       #
# --------------------------------------------------------------------------- #
def air_density(altitude_m: float = 0.0, temp_c: float = 15.0) -> float:
    """
    Densita' dell'aria (kg/m^3) con l'atmosfera standard (barometrica) + gas perfetti.
    A 0 m / 15 C ~ 1.225. Cala ~12% ogni 1000 m: conta per l'aerodinamica in quota.
    """
    p0, t0, L, R, M = 101325.0, 288.15, 0.0065, 8.31447, 0.0289644
    p = p0 * (1 - L * altitude_m / t0) ** (G * M / (R * L))
    t_k = temp_c + 273.15
    return float(p * M / (R * t_k))


# --------------------------------------------------------------------------- #
#  1. CINEMATICA: velocita' -> m/s, distanza, pendenza                        #
# --------------------------------------------------------------------------- #
def speed_to_ms(speed: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Normalizza la velocita' a m/s con euristica sull'unita' (i loader mescolano
    km/h e m/s). Se la mediana dei valori in movimento supera ~30 -> quasi certo km/h.
    Ritorna (array m/s, unita' rilevata) cosi' la scelta e' trasparente.
    """
    s = np.asarray(speed, dtype=float)
    moving = s[s > 1.0]
    med = np.nanmedian(moving) if moving.size else np.nanmedian(s)
    if med > 30:                          # 30 m/s = 108 km/h: impossibile su media -> e' km/h
        return s / 3.6, "km/h"
    return s, "m/s"


def distance_from_speed(speed_ms: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Distanza cumulata (m) integrando la velocita' a passo dt (1 Hz di default)."""
    v = np.nan_to_num(np.asarray(speed_ms, dtype=float), nan=0.0).clip(min=0)
    return np.cumsum(v) * dt


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Media mobile centrata robusta ai NaN (per smussare il rumore barometrico)."""
    s = pd.Series(x, dtype=float)
    return s.rolling(win, min_periods=1, center=True).mean().to_numpy()


def grade_series(altitude_m: np.ndarray, distance_m: np.ndarray,
                 smooth_win: int = 20) -> np.ndarray:
    """
    Pendenza (frazione: 0.08 = 8%) come d(quota)/d(distanza), con quota smussata.
    Il barometro ha rumore ~1-2 m: senza smussare la pendenza istantanea e' inservibile.
    Clip a +/-40% per tagliare artefatti (tunnel/GPS drift).
    """
    alt = _smooth(np.asarray(altitude_m, dtype=float), smooth_win)
    dist = np.asarray(distance_m, dtype=float)
    d_alt = np.gradient(alt)
    d_dist = np.gradient(dist)
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.where(d_dist > 0.1, d_alt / d_dist, 0.0)
    return np.clip(np.nan_to_num(grade), -0.40, 0.40)


# --------------------------------------------------------------------------- #
#  2. VAM + RILEVAMENTO SALITE  —  MEASURED (sola geometria)                   #
# --------------------------------------------------------------------------- #
def vam(elev_gain_m: float, duration_s: float) -> float:
    """VAM = metri di dislivello positivo per ora (m/h). Proxy classico di potenza in salita."""
    return float(elev_gain_m / (duration_s / 3600.0)) if duration_s > 0 else 0.0


def _vam_level(v: float) -> str:
    """Interpretazione indicativa della VAM (Grappe-style). Ordini di grandezza."""
    if v >= 1800: return "livello Tour, salita chiave (elite assoluto)"
    if v >= 1600: return "World Tour in salita"
    if v >= 1400: return "professionale / elite"
    if v >= 1100: return "agonista ben allenato"
    if v >= 800:  return "amatore evoluto"
    return "amatore / ritmo controllato"


def detect_climbs(altitude_m: np.ndarray, distance_m: np.ndarray,
                  power: np.ndarray | None = None, mass_kg: float | None = None,
                  min_gain_m: float = 30.0, min_grade: float = 0.03,
                  min_len_m: float = 500.0, gap_m: float = 200.0) -> dict:
    """
    Rileva le salite di una uscita e per ognuna calcola lunghezza, dislivello,
    pendenza media, durata, VAM e (se c'e' la potenza) W e W/kg medi.

    Euristica: tratti con pendenza smussata >= min_grade, uniti se separati da
    discese/pianori < gap_m, tenuti solo se dislivello >= min_gain_m, lunghezza
    >= min_len_m. VAM e geometria sono MEASURED.

    Categoria (stile ciclismo) da un indice di difficolta' = dislivello^2 / lunghezza,
    la stessa idea del punteggio FIETS/climb-by-bike (ordini di grandezza).
    """
    alt = _smooth(np.asarray(altitude_m, dtype=float), 20)
    dist = np.asarray(distance_m, dtype=float)
    grade = grade_series(alt, dist, smooth_win=20)
    up = grade >= min_grade

    # unisci tratti in salita separati da brevi non-salita (< gap_m di distanza)
    segs, i, n = [], 0, len(up)
    while i < n:
        if up[i]:
            j = i
            while j < n:
                if up[j]:
                    j += 1
                else:
                    k = j
                    while k < n and not up[k] and (dist[k] - dist[j]) < gap_m:
                        k += 1
                    if k < n and up[k]:
                        j = k                     # il pianoro era breve: continua la salita
                    else:
                        break
            segs.append((i, j)); i = j
        else:
            i += 1

    climbs = []
    for a, b in segs:
        b = min(b, n - 1)
        length = float(dist[b] - dist[a])
        gain = float(alt[b] - alt[a])
        if gain < min_gain_m or length < min_len_m:
            continue
        dur = float(b - a)                        # 1 Hz -> secondi = campioni
        avg_grade = gain / length if length else 0.0
        v = vam(gain, dur)
        difficulty = gain * gain / length if length else 0.0   # indice tipo FIETS
        cat = ("HC (fuori categoria)" if difficulty > 80 else
               "1a categoria" if difficulty > 50 else
               "2a categoria" if difficulty > 30 else
               "3a categoria" if difficulty > 15 else
               "4a categoria" if difficulty > 8 else "non categorizzata")
        entry = {"start_s": int(a), "duration_s": int(dur), "length_m": round(length),
                 "elev_gain_m": round(gain, 1), "avg_grade_pct": round(avg_grade * 100, 1),
                 "vam": round(v), "vam_level": _vam_level(v),
                 "category": cat, "difficulty_index": round(difficulty, 1)}
        if power is not None:
            seg_p = np.nan_to_num(np.asarray(power, dtype=float)[a:b], nan=0.0)
            if seg_p.size:
                entry["avg_power"] = round(float(seg_p.mean()))
                if mass_kg:
                    entry["w_kg"] = round(float(seg_p.mean()) / mass_kg, 2)
        climbs.append(entry)

    climbs.sort(key=lambda c: c["elev_gain_m"], reverse=True)
    return {"climbs": climbs, "n_climbs": len(climbs),
            "total_gain_m": round(sum(c["elev_gain_m"] for c in climbs), 1),
            "confidence": Confidence.MEASURED,
            "note": ("VAM e geometria sono misurate (solo dati). La categoria e' un "
                     "indice di difficolta' indicativo, non un albo ufficiale.")}


# --------------------------------------------------------------------------- #
#  3. MODELLO DI POTENZA FISICO                                               #
# --------------------------------------------------------------------------- #
def power_from_kinematics(speed_ms: np.ndarray, grade: np.ndarray,
                          total_mass_kg: float, cda: float = 0.32,
                          crr: float = 0.005, rho: float = 1.225,
                          dt: float = 1.0, wind_ms: np.ndarray | float = 0.0,
                          efficiency: float = DRIVETRAIN_EFF) -> np.ndarray:
    """
    Potenza AI PEDALI (W) richiesta a ogni istante dal modello fisico:
        P = [ m*g*slope*v + Crr*m*g*v + 0.5*rho*CdA*(v+w)^2*v + m*a*v ] / eta
    (piccola-angolo: slope ~ sin ~ tan; cos ~ 1). Il termine inerziale usa a=dv/dt.
    Utile per: (a) stimare la potenza SENZA power meter, (b) validare il metodo Chung.
    """
    v = np.nan_to_num(np.asarray(speed_ms, dtype=float), nan=0.0).clip(min=0)
    s = np.nan_to_num(np.asarray(grade, dtype=float), nan=0.0)
    w = np.asarray(wind_ms, dtype=float) if np.ndim(wind_ms) else float(wind_ms)
    a = np.gradient(v) / dt                       # accelerazione (m/s^2)
    f_grav = total_mass_kg * G * s
    f_roll = crr * total_mass_kg * G              # cos~1
    f_aero = 0.5 * rho * cda * (v + w) ** 2
    f_inertia = total_mass_kg * a
    p_wheel = (f_grav + f_roll + f_aero + f_inertia) * v
    return np.clip(p_wheel / efficiency, 0, None)


def estimate_power_no_meter(speed_ms, grade, total_mass_kg, cda=0.32, crr=0.005,
                            rho=1.225, dt=1.0) -> dict:
    """
    Stima della potenza per uscite SENZA misuratore (solo GPS+quota+massa).
    ESTIMATED: dipende da CdA/Crr assunti e assume vento nullo. Allarga la
    piattaforma a chi non ha un power meter, con onesta' sull'incertezza.
    """
    p = power_from_kinematics(speed_ms, grade, total_mass_kg, cda, crr, rho, dt)
    return {"power": p,
            "avg_power": Metric(float(np.mean(p)), "W", Confidence.ESTIMATED,
                                "modello fisico (no power meter)",
                                "Assume CdA/Crr tipici e vento nullo. Errore maggiore in pianura/vento."),
            "work_kj": Metric(float(np.sum(p)) / 1000.0 * dt, "kJ", Confidence.ESTIMATED,
                              "integrale della potenza modellata", ""),
            "assumptions": {"cda": cda, "crr": crr, "rho": round(rho, 3),
                            "mass_kg": total_mass_kg},
            "confidence": Confidence.ESTIMATED}


# --------------------------------------------------------------------------- #
#  4. METODO CHUNG — stima CdA e Crr dai dati  —  ESTIMATED                    #
# --------------------------------------------------------------------------- #
def _virtual_elevation(power, speed_ms, total_mass_kg, cda, crr, rho, dt,
                       efficiency=DRIVETRAIN_EFF):
    """
    Profilo di quota 'virtuale' implicato da potenza+velocita' dati CdA/Crr.
    Inverte il modello per la pendenza istantanea e la integra in quota.
        slope = ( eta*P/v - Crr*m*g - 0.5*rho*CdA*v^2 - m*a ) / (m*g)
        dh    = slope * v * dt
    Con CdA/Crr GIUSTI il profilo virtuale ricalca quello barometrico misurato.
    """
    v = np.nan_to_num(np.asarray(speed_ms, dtype=float), nan=0.0).clip(min=0)
    p = np.nan_to_num(np.asarray(power, dtype=float), nan=0.0)
    a = np.gradient(v) / dt
    vsafe = np.where(v > 0.5, v, np.nan)          # a bassa velocita' la stima di slope esplode
    wheelP = efficiency * p
    slope = (wheelP / vsafe - crr * total_mass_kg * G
             - 0.5 * rho * cda * vsafe ** 2 - total_mass_kg * a) / (total_mass_kg * G)
    slope = np.nan_to_num(slope, nan=0.0)
    dh = slope * v * dt
    return np.cumsum(dh)


def chung_cda_crr(power, speed_ms, altitude_m, total_mass_kg,
                  rho: float = 1.225, dt: float = 1.0,
                  cda_bounds=(0.15, 0.60), crr_bounds=(0.002, 0.012)) -> dict:
    """
    Stima CdA (m^2) e Crr adattando il profilo di quota VIRTUALE a quello MISURATO
    (metodo Chung / 'virtual elevation'). E' un calcolo fisico dai dati, ma dipende
    da massa e densita' dell'aria e assume vento nullo -> ESTIMATED.

    Migliore su percorsi ad anello o botta-e-ritorno con velocita' variabile e poco
    vento. Ritorna anche l'RMSE del fit (quota) come indicatore di qualita'.
    """
    alt = _smooth(np.asarray(altitude_m, dtype=float), 10)
    alt = alt - alt[0]                             # riferisci a 0 come la quota virtuale

    def resid(x):
        cda, crr = x
        ve = _virtual_elevation(power, speed_ms, total_mass_kg, cda, crr, rho, dt)
        return ve - alt

    x0 = [0.30, 0.005]
    lower = [cda_bounds[0], crr_bounds[0]]
    upper = [cda_bounds[1], crr_bounds[1]]
    sol = least_squares(resid, x0, bounds=(lower, upper), method="trf", max_nfev=2000)
    cda, crr = float(sol.x[0]), float(sol.x[1])
    rmse = float(np.sqrt(np.mean(sol.fun ** 2)))

    # incertezza approssimata dei parametri dalla Jacobiana del fit
    cda_sd = crr_sd = None
    try:
        J = sol.jac
        dof = max(len(sol.fun) - 2, 1)
        cov = np.linalg.inv(J.T @ J) * (float(sol.fun @ sol.fun) / dof)
        cda_sd, crr_sd = float(np.sqrt(abs(cov[0, 0]))), float(np.sqrt(abs(cov[1, 1])))
    except Exception:
        pass

    quality = ("buona" if rmse < 5 else "discreta" if rmse < 15 else
               "scarsa (percorso poco adatto o vento?)")
    return {
        "cda": Metric(cda, "m^2", Confidence.ESTIMATED, "Chung virtual elevation",
                      "Area frontale aerodinamica. ~0.20 aero/crono, ~0.30-0.40 in gruppo.",
                      sd=cda_sd, ci=(cda - 1.96 * cda_sd, cda + 1.96 * cda_sd) if cda_sd else None),
        "crr": Metric(crr, "", Confidence.ESTIMATED, "Chung virtual elevation",
                      "Coeff. di rotolamento. ~0.004 asfalto liscio/buoni tubolari, ~0.008+ ruvido.",
                      sd=crr_sd, ci=(crr - 1.96 * crr_sd, crr + 1.96 * crr_sd) if crr_sd else None),
        "rmse_elevation_m": round(rmse, 2),
        "fit_quality": quality,
        "confidence": Confidence.ESTIMATED,
        "note": ("Assume vento nullo, massa e densita' aria corrette. Attendibile su "
                 "anello/botta-e-ritorno con velocita' varia. RMSE alto = dati poco adatti."),
    }


# --------------------------------------------------------------------------- #
#  5. ENTRY POINT: analisi fisica completa di una uscita                      #
# --------------------------------------------------------------------------- #
def analyze_physics(df: pd.DataFrame, total_mass_kg: float,
                    temp_c: float = 15.0, dt: float = 1.0) -> dict:
    """
    Orchestratore: da un DataFrame a 1 Hz (colonne power/speed/altitude) produce
    distanza, pendenza, salite+VAM e — se c'e' la potenza — la stima CdA/Crr Chung.
    Robusto ai dati mancanti: salta ciò che non può calcolare e lo dichiara.
    """
    out = {"available": {}, "confidence": Confidence.MEASURED}
    has_alt = "altitude" in df.columns and df["altitude"].notna().any()
    has_speed = "speed" in df.columns and df["speed"].notna().any()
    has_power = "power" in df.columns and float(np.nansum(df["power"].values)) > 0

    if not has_alt:
        out["available"]["reason"] = "manca la quota: nessuna analisi altimetrica possibile."
        return out

    alt = df["altitude"].to_numpy(dtype=float)
    if has_speed:
        v_ms, unit = speed_to_ms(df["speed"].to_numpy(dtype=float))
        dist = distance_from_speed(v_ms, dt)
        out["speed_unit_detected"] = unit
    else:                                          # senza velocita': distanza non ricostruibile
        v_ms, dist = None, None

    if dist is not None:
        grade = grade_series(alt, dist)
        climbs = detect_climbs(alt, dist,
                               power=df["power"].to_numpy(dtype=float) if has_power else None,
                               mass_kg=total_mass_kg if has_power else None)
        out["available"]["climbs"] = True
        out["climbs"] = climbs
        out["total_gain_m"] = climbs["total_gain_m"]

        if has_power:
            mean_alt = float(np.nanmean(alt))
            rho = air_density(mean_alt, temp_c)
            out["air_density"] = round(rho, 3)
            try:
                out["aero"] = chung_cda_crr(df["power"].to_numpy(dtype=float),
                                            v_ms, alt, total_mass_kg, rho=rho, dt=dt)
                out["available"]["aero"] = True
            except Exception as e:
                out["available"]["aero"] = False
                out["aero_error"] = str(e)
    else:
        out["available"]["climbs"] = False
        out["available"]["reason"] = "manca la velocita': distanza/pendenza non ricostruibili."
    return out
