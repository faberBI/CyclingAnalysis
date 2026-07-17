"""
test_new_features.py — regressione sui moduli NUOVI (fisica, data-quality, longitudinale).
Stessa filosofia di test_regression.py: input noti, output ancorati a formule/geometria.

    pytest test_new_features.py -v
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
import pytest

import cycling_analytics as ca
import training_intelligence as ti
import physics as ph
import data_quality as dq
from cycling_analytics import Confidence

ATH = ca.Athlete(mass_kg=70, height_cm=178, age=34, sex="M", hr_max=192, hr_rest=48, lthr=170)


# ===================== CONFIDENCE INTERVALS (punto 2) ====================== #
def test_cp_reports_confidence_interval():
    # dati con rumore -> CI non degenere; il valore resta corretto
    rng = np.random.default_rng(0)
    durs = [120, 180, 300, 600, 900, 1200]
    mmp = pd.Series({t: 250 + 20000 / t + rng.normal(0, 3) for t in durs})
    res = ca.critical_power(mmp, model="hyperbolic")
    assert res["cp"].ci is not None
    lo, hi = res["cp"].ci
    assert lo < res["cp"].value < hi                 # il valore sta dentro il suo CI
    assert res["cp"].sd is not None and res["cp"].sd > 0

def test_vo2max_has_uncertainty_band():
    vo2 = ca.estimate_vo2max(ATH, 400.0)
    m = vo2["vo2max"]
    assert m.ci is not None and m.ci[0] < m.value < m.ci[1]
    assert m.sd and m.sd > 0

def test_vo2max_lab_still_no_regression():
    a = ca.Athlete(70, 178, 34, "M", vo2max_lab=62.0)
    assert ca.estimate_vo2max(a, 400.0)["vo2max"].confidence == Confidence.MEASURED

def test_ftp_recommended_has_range_when_multiple_methods():
    mmp = pd.Series({480: 300.0, 1200: 300.0})       # 8min e 20min -> due stime
    ftp = ca.estimate_ftp(mmp)
    rec = ftp["ftp_recommended"]
    assert rec.ci is not None and rec.ci[0] < rec.ci[1]


# ===================== FISICA: cinematica (punto 1) ======================== #
def test_speed_to_ms_detects_kmh():
    kmh = np.full(100, 36.0)                          # 36 km/h = 10 m/s
    ms, unit = ph.speed_to_ms(kmh)
    assert unit == "km/h" and ms[0] == pytest.approx(10.0)

def test_speed_to_ms_keeps_ms():
    ms_in = np.full(100, 9.0)
    ms, unit = ph.speed_to_ms(ms_in)
    assert unit == "m/s" and ms[0] == pytest.approx(9.0)

def test_distance_from_speed():
    v = np.full(100, 10.0)                            # 10 m/s * 100 s = 1000 m
    d = ph.distance_from_speed(v)
    assert d[-1] == pytest.approx(1000.0)

def test_air_density_drops_with_altitude():
    assert ph.air_density(0) == pytest.approx(1.225, abs=0.02)
    assert ph.air_density(2000) < ph.air_density(0)   # meno densa in quota


# ===================== FISICA: VAM & salite ================================ #
def test_vam_known():
    assert ph.vam(400, 1200) == pytest.approx(1200)   # 400 m in 20 min = 1200 m/h

def test_detect_climbs_single_clean_climb():
    # 300 s piano + 1200 s di salita all'8% a 4.17 m/s -> gain 400 m, VAM ~1200
    v = np.full(1500, 4.17)
    dist = ph.distance_from_speed(v)
    alt = np.zeros(1500)
    slope_per_s = 0.08 * 4.17                          # m di quota al secondo in salita
    alt[300:] = slope_per_s * np.arange(1500 - 300)
    res = ph.detect_climbs(alt, dist, min_gain_m=30)
    assert res["n_climbs"] == 1
    c = res["climbs"][0]
    assert c["elev_gain_m"] == pytest.approx(400, abs=25)
    assert c["avg_grade_pct"] == pytest.approx(8, abs=1.0)
    assert c["vam"] == pytest.approx(1200, abs=120)

def test_detect_climbs_ignores_flat():
    v = np.full(600, 8.0)
    dist = ph.distance_from_speed(v)
    alt = np.zeros(600)                                # tutto piano
    assert ph.detect_climbs(alt, dist)["n_climbs"] == 0


# ===================== FISICA: modello di potenza ========================== #
def test_power_from_kinematics_flat_matches_formula():
    v = np.full(200, 8.0)                              # costante -> accel ~0 all'interno
    grade = np.zeros(200)
    cda, crr, rho, m = 0.32, 0.005, 1.225, 80.0
    p = ph.power_from_kinematics(v, grade, m, cda, crr, rho)
    expected = (crr * m * ph.G + 0.5 * rho * cda * 8.0 ** 2) * 8.0 / ph.DRIVETRAIN_EFF
    assert p[100] == pytest.approx(expected, rel=0.02)  # campione interno


# ===================== FISICA: Chung CdA/Crr (ancora chiave) =============== #
def test_chung_recovers_known_cda_crr():
    # genero un ride con CdA/Crr NOTI col modello forward, poi verifico che Chung li ritrovi
    n = 1800
    tt = np.arange(n)
    # sweep di velocita' AMPIO (4-16 m/s): condizione d'uso reale del metodo Chung,
    # serve a separare CdA (~v^2) da Crr (costante).
    v = 10.0 + 6.0 * np.sin(2 * np.pi * tt / 300)
    grade = 0.02 + 0.015 * np.sin(2 * np.pi * tt / 450)  # pendenza sempre >0 -> P>0
    m, cda_true, crr_true, rho = 80.0, 0.28, 0.006, 1.20
    power = ph.power_from_kinematics(v, grade, m, cda_true, crr_true, rho)
    assert power.min() > 5                              # nessun clipping a 0: modello invertibile
    dist = ph.distance_from_speed(v)
    alt = np.cumsum(grade * v)                          # quota coerente con la pendenza
    res = ph.chung_cda_crr(power, v, alt, m, rho=rho)
    assert res["cda"].value == pytest.approx(cda_true, abs=0.02)
    assert res["crr"].value == pytest.approx(crr_true, abs=0.0015)
    assert res["rmse_elevation_m"] < 5

def test_estimate_power_no_meter_runs():
    v = np.full(300, 6.0)
    grade = np.full(300, 0.05)
    r = ph.estimate_power_no_meter(v, grade, 80.0)
    assert r["avg_power"].value > 0
    assert r["avg_power"].confidence == Confidence.ESTIMATED


# ===================== DATA QUALITY (punto 3) ============================== #
def _clean_ride(n=1800):
    rng = np.random.default_rng(1)
    power = np.clip(rng.normal(200, 30, n), 0, None)
    hr = np.clip(rng.normal(140, 8, n), 80, 190)
    alt = 100 + 50 * np.sin(np.linspace(0, 3, n))      # quota che varia -> outdoor
    return pd.DataFrame({"power": power, "hr": hr, "altitude": alt,
                         "speed": np.full(n, 8.0)})

def test_dq_clean_is_good():
    r = dq.assess_data_quality(_clean_ride())
    assert r["level"] == "buono" and r["score"] >= 85

def test_dq_detects_dropout():
    df = _clean_ride()
    df.loc[500:900, "power"] = 0                        # 400 s di potenza fantasma in movimento
    r = dq.assess_data_quality(df)
    assert r["stats"]["power_dropout_pct"] > 5
    assert r["level"] in ("attenzione", "scarso")

def test_dq_detects_spikes():
    df = _clean_ride()
    df.loc[100:110, "power"] = 3000                     # spike non fisiologici
    r = dq.assess_data_quality(df)
    assert r["stats"]["power_spikes"] > 0
    assert any("spike" in f["msg"].lower() or "3000" in f["msg"] or ">2500" in f["msg"]
               for f in r["flags"])

def test_detect_indoor_flat_altitude():
    n = 600
    df = pd.DataFrame({"power": np.full(n, 200.0),
                       "altitude": np.full(n, 100.0),   # quota piatta
                       "speed": np.full(n, 8.0)})
    assert dq.detect_indoor(df)["indoor"] is True

def test_detect_indoor_false_outdoor():
    df = _clean_ride()
    assert dq.detect_indoor(df)["indoor"] is False


# ===================== ACWR (punto 5) ====================================== #
def _daily(load_by_day):
    days = [date(2026, 6, 1) + timedelta(days=i) for i in range(len(load_by_day))]
    return pd.Series(load_by_day, index=pd.to_datetime(days))

def test_acwr_steady_is_one():
    s = _daily([50.0] * 40)
    r = ti.acwr(s, method="rolling")
    assert r["acwr"] == pytest.approx(1.0, abs=0.05)
    assert r["band"].startswith("ottimale")

def test_acwr_spike_flags_risk():
    s = _daily([30.0] * 28 + [120.0] * 7)              # settimana di picco improvviso
    r = ti.acwr(s, method="rolling")
    assert r["acwr"] > 1.5
    assert r["color"] == "rosso"


# ===================== BANISTER individualizzato =========================== #
def test_banister_recovers_params_and_predicts():
    rng = np.random.default_rng(3)
    days = pd.date_range("2026-01-01", periods=140, freq="D")
    tss = pd.Series(rng.uniform(0, 120, len(days)), index=days)
    k1, k2, tau1, tau2, p0 = 0.08, 0.09, 42.0, 7.0, 250.0
    true_perf = ti._ir_series(tss, k1, k2, tau1, tau2, p0)
    # campiono 15 'test' di performance nel tempo (dati puliti)
    sample_days = days[::9]
    perf = true_perf.loc[sample_days]
    fit = ti.fit_banister(tss, perf)
    assert fit["ok"] is True
    assert fit["r2"] > 0.98                            # ricostruisce quasi perfettamente
    assert 35 <= fit["params"]["tau1"] <= 50          # costante fitness ~42
    assert 4 <= fit["params"]["tau2"] <= 12           # costante fatica ~7
    # proiezione in avanti
    future = pd.Series([60.0] * 14, index=pd.date_range("2026-05-21", periods=14))
    proj = ti.predict_performance(fit, future)
    assert len(proj) == 14

def test_banister_needs_enough_points():
    days = pd.date_range("2026-01-01", periods=30, freq="D")
    tss = pd.Series(50.0, index=days)
    perf = pd.Series([250, 252], index=days[[5, 20]])
    fit = ti.fit_banister(tss, perf)
    assert fit["ok"] is False


# ===================== BREAKTHROUGH / RECORD =============================== #
def test_detect_breakthroughs_new_records():
    prev = pd.Series({5: 900.0, 60: 500.0, 300: 350.0, 1200: 280.0})
    new = pd.Series({5: 950.0, 60: 500.0, 300: 370.0, 1200: 300.0})   # 5s,5min,20min su
    r = ti.detect_breakthroughs(prev, new, min_pct=1.0)
    got = {rec["duration_s"] for rec in r["records"]}
    assert got == {5, 300, 1200}                       # 60s invariato: non e' record
    assert r["has_breakthrough"] is True
    assert r["eftp_update"] is not None                # 20min migliorato -> eFTP su
    assert len(r["notifications"]) >= 3

def test_detect_breakthroughs_first_ever():
    r = ti.detect_breakthroughs(pd.Series(dtype=float), pd.Series({300: 320.0}))
    assert r["records"][0]["first_ever"] is True


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ===================== GPS / lat-lon (mappa a monte) ====================== #
def test_semicircles_to_deg_roundtrip():
    deg = 45.123456
    semi = int(deg / (180 / 2**31))
    assert ca._semicircles_to_deg(semi) == pytest.approx(deg, abs=1e-5)
    assert ca._semicircles_to_deg(None) is None

def test_latlon_aliases_resolved():
    df = ca.parse_records([{"timestamp": i, "power": 200,
                            "latitude": 45.0 + i*1e-4, "longitude": 9.0} for i in range(4)])
    assert "lat" in df.columns and "lon" in df.columns
    assert df["lat"].iloc[0] == pytest.approx(45.0)

def test_to_1hz_interpolates_latlon_without_zero_fill():
    df = pd.DataFrame({"t": range(6), "power": [200]*6,
                       "lat": [45.0, None, 45.002, None, None, 45.010],
                       "lon": [9.0, 9.001, None, 9.003, 9.004, 9.005]})
    out = ca.to_1hz(df)
    assert out["lat"].dtype.kind == "f"
    assert not (out["lat"] == 0).any()            # niente 0,0 (bug GPS)
    assert out["lat"].iloc[1] == pytest.approx(45.001, abs=1e-3)   # buco breve interpolato


# ===================== POTENZA PONDERATA (NP) & FTP-NP ===================== #
def test_best_np_window_constant():
    p = np.full(4000, 250.0)
    bw = ca.best_np_window(p, 3600)
    assert bw["np"] == pytest.approx(250, abs=0.5)
    assert bw["vi"] == pytest.approx(1.0, abs=0.005)

def test_best_np_window_variable_np_exceeds_mean():
    rng = np.random.default_rng(0)
    p = np.clip(rng.normal(250, 120, 4000), 0, None)
    bw = ca.best_np_window(p, 3600)
    assert bw["np"] > bw["mean"]                  # Jensen: NP >= media
    assert bw["vi"] > 1.0

def test_best_np_window_too_short_is_none():
    assert ca.best_np_window(np.full(100, 200.0), 3600) is None

def test_ftp_from_np_long_constant():
    ftp = ca.ftp_from_np_long(np.full(4000, 260.0), 3600)
    assert ftp is not None
    assert ftp.value == pytest.approx(260, abs=0.5)   # nessuno sconto 0.95 sui 60 min
    assert ftp.confidence == Confidence.ESTIMATED

def test_ftp_from_np_long_short_ride_none():
    assert ca.ftp_from_np_long(np.full(600, 200.0), 3600) is None
