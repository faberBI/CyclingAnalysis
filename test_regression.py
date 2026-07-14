"""
test_regression.py — suite di regressione sui calcoli fisiologici e sulla logica.
================================================================================
Per una piattaforma che vende accuratezza, i numeri NON devono cambiare in silenzio.
Ogni test fissa un input noto e l'output atteso (calcolato a mano o da formula).

Esecuzione:
    pytest test_regression.py -v
    (oppure)  python test_regression.py     # runner integrato

Molti valori sono ancore esatte:
- NP di potenza costante = la costante; TSS di 1h a IF=1 = 100.
- kcal ≈ kJ di lavoro (efficienza lorda 0.24).
- Mifflin-St Jeor, zone Coggan, W'bal, CTL/ATL: formule deterministiche.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
import pytest

import cycling_analytics as ca
import training_intelligence as ti
from cycling_analytics import Confidence

TOL = 1e-6
ATH = ca.Athlete(mass_kg=70, height_cm=178, age=34, sex="M",
                 hr_max=192, hr_rest=48, lthr=170)


# ===================== PREPROCESSING ======================================== #
def test_to_1hz_coerces_object_columns():
    # colonne con testo/None (il bug storico) non devono rompere
    df = pd.DataFrame({"t": range(6), "power": [200, None, "210", 220, None, 230],
                       "hr": ["140", None, 142, None, "", 145]})
    out = ca.to_1hz(df)
    assert not out["power"].isna().any()        # power riempita
    assert out["power"].dtype.kind == "f"       # numerica
    assert len(out) == 6

def test_to_1hz_parses_iso_strings():
    ts = pd.date_range("2026-07-14 08:00:00", periods=5, freq="s").astype(str)
    out = ca.to_1hz(pd.DataFrame({"t": ts, "power": [100, 150, 200, 250, 300]}))
    assert len(out) == 5 and out["power"].iloc[-1] == 300


# ===================== CURVA DI POTENZA ===================================== #
def test_mmp_known_signal():
    p = np.array([100]*100 + [500]*10 + [100]*100, dtype=float)
    mmp = ca.mean_maximal_power(p, [1, 10, 20])
    assert mmp[1] == pytest.approx(500)          # picco 1s
    assert mmp[10] == pytest.approx(500)         # 10s tutti a 500
    assert mmp[20] == pytest.approx(300)         # 10x500 + 10x100 su 20s = 300

def test_season_curve_takes_pointwise_max():
    a = ca.mean_maximal_power(np.array([600]*5 + [100]*100, dtype=float), [5, 60])
    b = ca.mean_maximal_power(np.array([100]*5 + [300]*100, dtype=float), [5, 60])
    season = pd.concat([a, b], axis=1).max(axis=1)
    assert season[5] == pytest.approx(600)       # 5s dal primo
    assert season[60] > 250                       # 60s dal secondo


# ===================== CP / W' ============================================== #
def test_critical_power_recovers_params():
    # punti che stanno ESATTAMENTE su P = CP + W'/t con CP=250, W'=20000
    cp_true, w_true = 250.0, 20000.0
    durs = [180, 300, 600, 900, 1200]
    mmp = pd.Series({t: cp_true + w_true / t for t in durs})
    res = ca.critical_power(mmp, model="hyperbolic")
    assert res["cp"].value == pytest.approx(250, abs=1.0)
    assert res["w_prime"].value == pytest.approx(20000, abs=200)
    assert res["cp"].confidence == Confidence.MEASURED


# ===================== FTP / MAP ============================================ #
def test_ftp_from_20min():
    mmp = pd.Series({1200: 300.0})
    ftp = ca.estimate_ftp(mmp)
    assert ftp["ftp_20min"].value == pytest.approx(285)   # 95% di 300

def test_map_proxy_5min():
    mmp = pd.Series({300: 400.0})
    assert ca.maximal_aerobic_power(mmp).value == pytest.approx(400)


# ===================== VO2MAX =============================================== #
def test_vo2max_acsm_formula():
    # VO2(ml/kg/min) = 1.8*(MAP*6.12)/mass + 7 ; MAP=400, mass=70 -> 69.95
    vo2 = ca.estimate_vo2max(ATH, 400.0)
    assert vo2["vo2max_acsm"].value == pytest.approx(1.8*(400*6.12)/70 + 7, abs=0.01)
    assert vo2["vo2max"].confidence == Confidence.ESTIMATED

def test_vo2max_lab_overrides():
    a = ca.Athlete(70, 178, 34, "M", vo2max_lab=62.0)
    vo2 = ca.estimate_vo2max(a, 400.0)
    assert vo2["vo2max"].value == 62.0
    assert vo2["vo2max"].confidence == Confidence.MEASURED   # misurato, non stimato


# ===================== ZONE ================================================= #
def test_power_zones_coggan():
    z = ca.power_zones(250)["zones"]
    assert z["Z2 Fondo"] == (140, 188)           # 56%-75% di 250
    assert z["Z4 Soglia"] == (228, 262)          # 91%-105% (round bancario)

def test_hr_zones_lthr():
    z = ca.hr_zones(ATH)["zones"]                 # LTHR=170
    assert z["Z2 Fondo"] == (138, 151)
    assert z["Z4 Soglia"] == (160, 168)


# ===================== CARICO (NP/IF/TSS/VI) ================================ #
def test_load_metrics_constant_hour():
    p = np.full(3600, 200.0)
    lm = ca.load_metrics(p, ftp=200)
    assert lm["normalized_power"].value == pytest.approx(200, abs=0.5)
    assert lm["intensity_factor"].value == pytest.approx(1.0, abs=0.005)
    assert lm["tss"].value == pytest.approx(100, abs=0.5)       # 1h a IF=1 = 100
    assert lm["variability_index"].value == pytest.approx(1.0, abs=0.005)
    assert lm["work"].value == pytest.approx(720)               # 200W*3600s


# ===================== CALORIE / SUBSTRATI ================================= #
def test_calories_from_power():
    p = np.full(3600, 250.0)                       # 900 kJ
    kcal = ca.calories_from_power(p, gross_efficiency=0.24)
    assert kcal.value == pytest.approx(900/0.24/4.184, abs=0.5)  # ~896
    assert kcal.confidence == Confidence.MEASURED

def test_substrate_split_sums_to_100():
    p = np.full(1800, 200.0)
    sub = ca.substrate_split(p, map_watt=350, athlete=ATH, total_kcal=500)
    assert sub["pct_carb"].value + sub["pct_fat"].value == pytest.approx(100, abs=0.1)
    assert sub["pct_fat"].confidence == Confidence.MODELED        # senza RER = modellato


# ===================== FABBISOGNO =========================================== #
def test_bmr_mifflin():
    de = ca.daily_energy(ATH, exercise_kcal=0, activity_factor=1.0)
    assert de["bmr"].value == pytest.approx(10*70 + 6.25*178 - 5*34 + 5)   # 1647.5


# ===================== DURABILITY =========================================== #
def test_durability_detects_drop():
    fresh = np.concatenate([np.full(300, 400.0), np.full(3000, 200.0)])
    fatig = np.concatenate([fresh, np.full(300, 300.0)])   # 5min più debole dopo tanto kJ
    thr = fatig.sum()/1000 * 0.5
    d = ca.durability(fatig, durations=(300,), kj_threshold=thr)
    assert d["reached"] is True
    assert d["per_duration"][300]["drop_pct"] > 0

def test_durability_not_reached():
    d = ca.durability(np.full(100, 200.0), kj_threshold=99999)
    assert d["reached"] is False


# ===================== W' BAL =============================================== #
def test_wbal_depletion_above_cp():
    # 100s costanti a 50W sopra CP -> consuma 5000 J
    p = np.full(100, 300.0)
    wb = ti.w_bal(p, cp=250, w_prime=20000)
    assert wb["min_wbal"].value == pytest.approx(15000, abs=1)   # 20000 - 50*100
    assert wb["depleted_pct"].value == pytest.approx(25, abs=0.1)


# ===================== INTERVAL DETECTION ================================== #
def test_detect_intervals_single():
    p = np.concatenate([np.full(100, 150.0), np.full(60, 400.0), np.full(100, 150.0)])
    ivs = ti.detect_intervals(p, ftp=250, min_seconds=20)
    assert len(ivs) == 1
    assert ivs[0]["duration_s"] == 60
    assert ivs[0]["avg_power"] == pytest.approx(400)


# ===================== CTL/ATL/TSB ========================================= #
def test_training_load_single_session():
    s = [ti.Session(day=date(2026, 7, 14), tss=100.0)]
    pmc = ti.training_load(s, seed_ctl=0, seed_atl=0)
    # CTL = 0 + (100-0)/42 ; ATL = 100/7 ; TSB (giorno) = seed - seed = 0
    assert pmc["ctl"].iloc[-1] == pytest.approx(100/42, abs=1e-6)
    assert pmc["atl"].iloc[-1] == pytest.approx(100/7, abs=1e-6)
    assert pmc["tsb"].iloc[-1] == pytest.approx(0, abs=1e-6)


# ===================== RECUPERO ============================================ #
def test_recovery_status_bands():
    assert ti.recovery_status(10) == "Pienamente recuperato"
    assert ti.recovery_status(-5) == "Recuperato"
    assert ti.recovery_status(-15) == "Recupero parziale"
    assert ti.recovery_status(-25) == "Recupero necessario"

def test_recovery_forecast_fresh_is_zero():
    assert ti.recovery_forecast(60, 50) == 0        # TSB=+10 già fresco

def test_recovery_forecast_needs_rest():
    d = ti.recovery_forecast(60, 90)                 # TSB=-30
    assert 1 <= d <= 21


# ===================== WELLNESS ============================================ #
def _wellness(hrv_recent, hrv_base=80.0, rhr_recent=48, rhr_base=48, sleep=8.0, n=30):
    """Costruisce un DataFrame wellness: baseline stabile + valori recenti diversi."""
    days = [date(2026,7,14) - timedelta(days=i) for i in range(n)][::-1]
    hrv = [hrv_base]*(n-3) + [hrv_recent]*3
    rhr = [rhr_base]*(n-3) + [rhr_recent]*3
    slp = [8.0]*(n-3) + [sleep]*3
    return pd.DataFrame({"date": pd.to_datetime(days), "hrv": hrv,
                         "resting_hr": rhr, "sleep_hours": slp})

def test_wellness_green_when_normal():
    r = ti.wellness_readiness(_wellness(hrv_recent=80))
    assert r["overall"] == "verde" and r["recovery_penalty_days"] == 0

def test_wellness_red_when_hrv_crashed():
    # HRV recente molto sotto la baseline (80±~0) -> soppressa
    r = ti.wellness_readiness(_wellness(hrv_recent=55, rhr_recent=56, sleep=5.0))
    assert r["hrv"] == "soppressa"
    assert r["overall"] == "rosso"
    assert r["recovery_penalty_days"] >= 3

def test_wellness_insufficient_data():
    df = pd.DataFrame({"date": pd.to_datetime([date(2026,7,14)]), "hrv": [70.0]})
    r = ti.wellness_readiness(df)
    assert r["have_data"] is False

def test_adjusted_recovery_adds_penalty():
    r = ti.wellness_readiness(_wellness(hrv_recent=55, rhr_recent=56, sleep=5.0))
    base = 3
    assert ti.adjusted_recovery_days(base, r) == base + r["recovery_penalty_days"]

def test_adjusted_recovery_no_wellness_unchanged():
    assert ti.adjusted_recovery_days(4, {"have_data": False}) == 4


# ===================== RACCOMANDAZIONE + WELLNESS ========================== #
def _fresh_pmc():
    # settimana leggera -> TSB positivo (forma buona)
    s = [ti.Session(day=date(2026,7,14)-timedelta(days=i), tss=30.0) for i in range(10)]
    return ti.training_load(s, seed_ctl=50, seed_atl=45)

def test_recommend_red_forces_recovery():
    pmc = _fresh_pmc()
    ws = {"distribution": "aerobica", "frac_high": 0.0, "monotony": 1.0}
    red = ti.wellness_readiness(_wellness(hrv_recent=55, rhr_recent=56, sleep=5.0))
    rec = ti.recommend_next_workout(pmc, ws, {}, 250, readiness=red)
    assert rec["recommended"]["kind"] == "recupero"   # rosso -> recupero anche se fresco

def test_recommend_green_allows_quality():
    pmc = _fresh_pmc()
    ws = {"distribution": "aerobica", "frac_high": 0.0, "monotony": 1.0}
    green = ti.wellness_readiness(_wellness(hrv_recent=80))
    rec = ti.recommend_next_workout(pmc, ws, {}, 250, readiness=green)
    assert rec["recommended"]["kind"] in ("vo2max", "soglia", "anaerobico", "sweetspot")


# ===================== CLASSIFICAZIONE ===================================== #
def test_classify_category_wkg():
    # 20min a 4.9 W/kg (70kg -> 343W) deve raggiungere almeno 'continental'
    mmp = pd.Series({1200: 343.0})
    cl = ca.classify_category(mmp, mass_kg=70)
    assert cl["per_duration"][1200]["w_kg"] == pytest.approx(4.9, abs=0.01)
    assert cl["per_duration"][1200]["category"] in ("continental", "professional",
                                                    "world_tour", "top20_grande_giro", "top10_tdf")


# ===================== PERIODIZZAZIONE ===================================== #
def test_periodized_plan_structure():
    today = date(2026,7,14)
    plan = ti.periodized_plan(today, today+timedelta(weeks=12), current_ctl=55, current_atl=50,
                              event="Granfondo / Ciclistica", ftp=280, safe_ramp=5)
    assert plan["weeks_until"] == 12
    ps = plan["phase_structure"]
    assert ps["base_weeks"] + ps["build_weeks"] + ps["taper_weeks"] == 12
    assert isinstance(plan["race_day_tsb"], float)

def test_periodized_plan_too_close_is_all_taper():
    today = date(2026,7,14)
    plan = ti.periodized_plan(today, today+timedelta(days=8), current_ctl=60, current_atl=75,
                              event="Cronometro (TT)", ftp=290, safe_ramp=5)
    ps = plan["phase_structure"]
    assert ps["base_weeks"] == 0 and ps["build_weeks"] == 0    # niente build a <2 settimane


# ===================== FAT OX / PRO / AMATORI ============================= #
def test_fat_oxidation_curve_has_interior_peak():
    foc = ca.fat_oxidation_curve(map_watt=350, athlete=ATH)
    df = foc["curve"]
    imax = int(df["fat_g_min"].idxmax())
    assert 0 < imax < len(df) - 1                     # picco interno, non ai bordi
    assert 40 <= foc["fatmax_pct"] <= 75              # FatMax a intensità plausibile
    assert foc["confidence"] == Confidence.MODELED    # senza gas = modellato
    # ai bordi l'ossidazione grassi è minore che al picco
    assert df["fat_g_min"].iloc[0] < foc["fatmax_fat_g_min"]
    assert df["fat_g_min"].iloc[-1] < foc["fatmax_fat_g_min"]

def test_pro_comparison_pct_pogacar():
    # 20min a 6.9 W/kg (= Pogačar) -> 100%
    mmp = pd.Series({1200: 6.9 * 70})
    pc = ca.pro_comparison(mmp, mass_kg=70)
    row = [r for r in pc["rows"] if r["dur_s"] == 1200][0]
    assert row["tu"] == pytest.approx(6.9, abs=0.01)
    assert row["pct_pogacar"] == 100

def test_classify_amateur_bands():
    assert ca.classify_amateur(2.8)["tier"].startswith("Amatore base")
    assert ca.classify_amateur(3.4)["tier"].startswith("Amatore intermedio")
    assert ca.classify_amateur(4.1)["tier"].startswith("Amatore avanzato")
    assert ca.classify_amateur(4.8)["tier"].startswith("Agonista")
    # posizione monotona crescente
    assert ca.classify_amateur(2.8)["position"] < ca.classify_amateur(4.1)["position"]


def test_polarization_from_hist():
    edges = np.arange(0, 1610, 10)
    counts = np.zeros(len(edges) - 1)
    def add(w, s): counts[np.searchsorted(edges, w) - 1] += s
    add(150, 3600); add(260, 600); add(340, 300)     # low, mid, high con FTP=300
    pol = ti.polarization_from_hist(edges, counts, ftp=300)
    assert pol["pct_low"] + pol["pct_mid"] + pol["pct_high"] == pytest.approx(100, abs=1)
    assert pol["pct_low"] > 70                         # dominante low
    assert pol["low_h"] == pytest.approx(1.0, abs=0.01)  # 3600s = 1h
    assert "Polarizzata" in pol["label"]


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
