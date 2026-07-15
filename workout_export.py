"""
workout_export.py
=================
Chiude il loop consiglio -> ESECUZIONE: prende una sessione strutturata e la esporta
nei formati che rullo e ciclocomputer sanno leggere.

Formati:
  .zwo  Zwift (XML)                      — testo, nessuna dipendenza
  .mrc  ERG/MRC in %FTP (GoldenCheetah, molti rulli)   — testo
  .erg  ERG in Watt assoluti             — testo (serve la FTP)
  .fit  workout Garmin/Wahoo (head unit) — via 'fit-tool' (import lazy, opzionale)

Rappresentazione: un Workout = lista di 'segmenti', ciascuno o uno Step singolo
(riscaldamento a rampa, tratto steady, defaticamento) o un Interval (ripetute on/off).
Le potenze sono FRAZIONI di FTP (0.88 = 88% FTP): indipendenti dall'atleta finche'
non servono i Watt (allora si moltiplica per la FTP).

I 'kind' rispecchiano _session_template di training_intelligence, cosi' il workout
consigliato dall'app diventa un file scaricabile con un click.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import xml.sax.saxutils as _xml


# --------------------------------------------------------------------------- #
#  Schema                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    duration_s: int
    power_lo: float                       # frazione di FTP (0.88 = 88%)
    power_hi: Optional[float] = None      # se valorizzato -> rampa power_lo -> power_hi
    role: str = "steady"                  # warmup | cooldown | steady | active | rest
    cadence: Optional[int] = None
    label: str = ""

    @property
    def is_ramp(self) -> bool:
        return self.power_hi is not None and abs(self.power_hi - self.power_lo) > 1e-6


@dataclass
class Interval:
    repeat: int
    on: Step
    off: Step


Segment = Union[Step, Interval]


@dataclass
class Workout:
    name: str
    description: str
    segments: list[Segment] = field(default_factory=list)
    ftp: Optional[float] = None
    author: str = "Cycling Lab"
    sport: str = "bike"


# --------------------------------------------------------------------------- #
#  Costruttore: kind + FTP -> Workout strutturato                             #
# --------------------------------------------------------------------------- #
def build_structured_workout(kind: str, ftp: float) -> Workout:
    """
    Traduce un 'kind' (gli stessi di _session_template) in un workout eseguibile,
    con riscaldamento e defaticamento sensati. Le potenze restano in %FTP.
    """
    warm = Step(600, 0.50, 0.75, role="warmup", label="Riscaldamento")
    cool = Step(300, 0.55, 0.40, role="cooldown", label="Defaticamento")

    presets: dict[str, tuple[str, str, list[Segment]]] = {
        "recupero": ("Recupero attivo", "Z1 rigenerante, gambe leggere.",
                     [Step(2400, 0.50, role="steady", label="Z1 recupero")]),
        "fondo": ("Fondo aerobico", "Volume aerobico costante in Z2.",
                  [Step(5400, 0.65, role="steady", label="Z2 fondo")]),
        "lungo": ("Lungo endurance", "Uscita lunga in Z2 per la durabilita'.",
                  [Step(10800, 0.66, role="steady", label="Z2 lungo")]),
        "sweetspot": ("Sweet-spot", "Blocchi all'88-93% FTP, alto stimolo/basso stress.",
                      [Interval(4, Step(720, 0.90, role="active", label="Sweet-spot"),
                                Step(300, 0.55, role="rest", label="Recupero"))]),
        "soglia": ("Soglia", "Ripetute alla soglia (95-100% FTP).",
                   [Interval(4, Step(600, 0.98, role="active", label="Soglia"),
                             Step(300, 0.55, role="rest", label="Recupero"))]),
        "vo2max": ("VO2max", "Ripetute VO2max (110-118% FTP), recuperi lunghi.",
                   [Interval(5, Step(240, 1.14, role="active", label="VO2max"),
                             Step(240, 0.50, role="rest", label="Recupero"))]),
        "anaerobico": ("Anaerobico / Sprint", "Sprint quasi massimali, recuperi ampi.",
                       [Interval(7, Step(30, 1.60, role="active", label="Sprint"),
                                 Step(270, 0.50, role="rest", label="Recupero"))]),
    }
    if kind not in presets:
        raise ValueError(f"kind sconosciuto: {kind!r}. Validi: {sorted(presets)}")
    name, desc, main = presets[kind]
    segs: list[Segment] = [warm, *main, cool] if kind not in ("recupero",) else main
    return Workout(name=name, description=desc, segments=segs, ftp=ftp)


# --------------------------------------------------------------------------- #
#  Utility                                                                    #
# --------------------------------------------------------------------------- #
def flatten(w: Workout) -> list[Step]:
    """Espande gli Interval in una sequenza piatta di Step (on, off, on, off, ...)."""
    steps: list[Step] = []
    for seg in w.segments:
        if isinstance(seg, Interval):
            for _ in range(seg.repeat):
                steps.append(seg.on)
                steps.append(seg.off)
        else:
            steps.append(seg)
    return steps


def total_duration_s(w: Workout) -> int:
    return int(sum(s.duration_s for s in flatten(w)))


def estimate_tss(w: Workout) -> float:
    """
    TSS stimato del workout: somma su ogni step di IF^2 * ore * 100, con IF ~ %FTP
    medio dello step (media lo/hi per le rampe). Coerente con la definizione Coggan.
    """
    tss = 0.0
    for s in flatten(w):
        intensity = s.power_lo if not s.is_ramp else (s.power_lo + s.power_hi) / 2
        tss += (intensity ** 2) * (s.duration_s / 3600.0) * 100.0
    return round(tss, 1)


def summary(w: Workout) -> dict:
    dur = total_duration_s(w)
    return {"name": w.name, "duration_min": round(dur / 60), "duration_s": dur,
            "tss": estimate_tss(w), "n_steps": len(flatten(w))}


# --------------------------------------------------------------------------- #
#  1. ZWIFT (.zwo)                                                            #
# --------------------------------------------------------------------------- #
def to_zwo(w: Workout) -> str:
    """
    Genera un file .zwo (Zwift). Potenze come frazione di FTP. Gli Interval diventano
    <IntervalsT>, le rampe <Warmup>/<Cooldown>/<Ramp>, i tratti fissi <SteadyState>.
    """
    def esc(x): return _xml.escape(str(x))

    def cad(s: Step) -> str:
        return f' Cadence="{s.cadence}"' if s.cadence else ""

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<workout_file>",
             f"  <author>{esc(w.author)}</author>",
             f"  <name>{esc(w.name)}</name>",
             f"  <description>{esc(w.description)}</description>",
             "  <sportType>bike</sportType>", "  <tags/>", "  <workout>"]
    for seg in w.segments:
        if isinstance(seg, Interval):
            on, off = seg.on, seg.off
            lines.append(
                f'    <IntervalsT Repeat="{seg.repeat}" '
                f'OnDuration="{on.duration_s}" OffDuration="{off.duration_s}" '
                f'OnPower="{on.power_lo:.3f}" OffPower="{off.power_lo:.3f}"'
                f'{cad(on)}/>')
        else:
            s = seg
            if s.role == "warmup" and s.is_ramp:
                lines.append(f'    <Warmup Duration="{s.duration_s}" '
                             f'PowerLow="{s.power_lo:.3f}" PowerHigh="{s.power_hi:.3f}"{cad(s)}/>')
            elif s.role == "cooldown" and s.is_ramp:
                lines.append(f'    <Cooldown Duration="{s.duration_s}" '
                             f'PowerLow="{s.power_lo:.3f}" PowerHigh="{s.power_hi:.3f}"{cad(s)}/>')
            elif s.is_ramp:
                lines.append(f'    <Ramp Duration="{s.duration_s}" '
                             f'PowerLow="{s.power_lo:.3f}" PowerHigh="{s.power_hi:.3f}"{cad(s)}/>')
            else:
                lines.append(f'    <SteadyState Duration="{s.duration_s}" '
                             f'Power="{s.power_lo:.3f}"{cad(s)}/>')
    lines += ["  </workout>", "</workout_file>", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  2. ERG / MRC (.erg = Watt, .mrc = %FTP)                                    #
# --------------------------------------------------------------------------- #
def _course_points(w: Workout, as_watts: bool) -> list[tuple[float, float]]:
    """
    Punti (minuti, valore) del profilo. Ogni step -> due punti (inizio e fine) per
    disegnare il rettangolo/rampa. Valore = Watt (as_watts) o %FTP (0-100).
    """
    pts, t = [], 0.0
    ftp = w.ftp or 0.0
    for s in flatten(w):
        lo = s.power_lo
        hi = s.power_hi if s.is_ramp else s.power_lo
        if as_watts:
            v0, v1 = lo * ftp, hi * ftp
        else:
            v0, v1 = lo * 100.0, hi * 100.0
        t0 = t / 60.0
        t1 = (t + s.duration_s) / 60.0
        pts.append((t0, v0))
        pts.append((t1, v1))
        t += s.duration_s
    return pts


def _erg_mrc(w: Workout, as_watts: bool) -> str:
    if as_watts and not w.ftp:
        raise ValueError("Per l'export .erg (Watt) serve la FTP nel Workout.")
    unit_hdr = "MINUTES WATTS" if as_watts else "MINUTES PERCENT"
    fmt = "{:.2f}\t{:.0f}" if as_watts else "{:.2f}\t{:.1f}"
    head = ["[COURSE HEADER]", "VERSION = 2", "UNITS = ENGLISH",
            f"DESCRIPTION = {w.description}", f"FILE NAME = {w.name}",
            f"FTP = {w.ftp:.0f}" if w.ftp else "FTP = 0",
            unit_hdr, "[END COURSE HEADER]", "[COURSE DATA]"]
    body = [fmt.format(t, v) for t, v in _course_points(w, as_watts)]
    return "\n".join(head + body + ["[END COURSE DATA]", ""])


def to_erg(w: Workout) -> str:
    """ERG in Watt assoluti (richiede FTP). Per rulli in modalita' ERG."""
    return _erg_mrc(w, as_watts=True)


def to_mrc(w: Workout) -> str:
    """MRC in %FTP (indipendente dalla FTP). GoldenCheetah e molti rulli."""
    return _erg_mrc(w, as_watts=False)


# --------------------------------------------------------------------------- #
#  3. FIT workout (Garmin / Wahoo head unit) — dipendenza opzionale           #
# --------------------------------------------------------------------------- #
def to_fit(w: Workout) -> bytes:
    """
    Genera un file .fit di tipo WORKOUT (per ciclocomputer Garmin/Wahoo).
    Richiede la libreria 'fit-tool' (import lazy). Le potenze diventano target in
    Watt assoluti (serve la FTP). Se la libreria manca, solleva un errore chiaro.
    """
    if not w.ftp:
        raise ValueError("Per l'export .fit serve la FTP (i target sono in Watt).")
    try:
        from fit_tool.fit_file_builder import FitFileBuilder
        from fit_tool.profile.messages.file_id_message import FileIdMessage
        from fit_tool.profile.messages.workout_message import WorkoutMessage
        from fit_tool.profile.messages.workout_step_message import WorkoutStepMessage
        from fit_tool.profile.profile_type import (
            FileType, Manufacturer, Sport, WorkoutStepDuration,
            WorkoutStepTarget, Intensity)
    except Exception as e:                       # libreria assente
        raise RuntimeError(
            "Export .fit non disponibile: manca la libreria 'fit-tool' "
            "(pip install fit-tool). I formati .zwo / .erg / .mrc funzionano comunque."
        ) from e

    steps = flatten(w)
    builder = FitFileBuilder(auto_define=True)

    from datetime import datetime, timezone
    created_ms = round(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    fid = FileIdMessage()
    fid.type = FileType.WORKOUT
    fid.manufacturer = Manufacturer.DEVELOPMENT.value
    fid.product = 0
    fid.time_created = created_ms
    fid.serial_number = 0x10000000
    builder.add(fid)

    wm = WorkoutMessage()
    wm.workout_name = w.name[:15]
    wm.sport = Sport.CYCLING
    wm.num_valid_steps = len(steps)
    builder.add(wm)

    def watt_target(frac: float) -> int:
        # FIT: 0-1000 => %FTP ; 1000+watt => Watt assoluti. Usiamo i Watt (non ambiguo).
        return int(round(frac * w.ftp)) + 1000

    for i, s in enumerate(steps):
        sm = WorkoutStepMessage()
        sm.message_index = i
        sm.workout_step_name = (s.label or s.role)[:15]
        sm.intensity = (Intensity.WARMUP if s.role == "warmup" else
                        Intensity.COOLDOWN if s.role == "cooldown" else
                        Intensity.REST if s.role == "rest" else Intensity.ACTIVE)
        sm.duration_type = WorkoutStepDuration.TIME
        sm.duration_value = int(s.duration_s * 1000)          # ms
        sm.target_type = WorkoutStepTarget.POWER
        sm.target_value = 0                                    # 0 => usa custom range
        lo = min(s.power_lo, s.power_hi) if s.is_ramp else s.power_lo
        hi = max(s.power_lo, s.power_hi) if s.is_ramp else s.power_lo
        sm.custom_target_power_low = watt_target(lo)
        sm.custom_target_power_high = watt_target(hi)
        builder.add(sm)

    return builder.build().to_bytes()


# --------------------------------------------------------------------------- #
#  Dispatcher comodo per l'app                                                #
# --------------------------------------------------------------------------- #
def export(w: Workout, fmt: str):
    """Ritorna il contenuto nel formato richiesto: 'zwo'|'mrc'|'erg' -> str, 'fit' -> bytes."""
    fmt = fmt.lower().lstrip(".")
    if fmt == "zwo": return to_zwo(w)
    if fmt == "mrc": return to_mrc(w)
    if fmt == "erg": return to_erg(w)
    if fmt == "fit": return to_fit(w)
    raise ValueError(f"Formato non supportato: {fmt}")
