"""
test_workout_export.py — regressione sull'export dei workout strutturati.
Round-trip veri: .zwo ri-parsato come XML, .fit riletto con fitparse.

    pytest test_workout_export.py -v
"""
import io
import xml.etree.ElementTree as ET
import pytest

import workout_export as wx


FTP = 250.0


# ===================== COSTRUZIONE ========================================= #
def test_build_all_kinds():
    for kind in ["recupero", "fondo", "lungo", "sweetspot", "soglia", "vo2max", "anaerobico"]:
        w = wx.build_structured_workout(kind, FTP)
        assert w.ftp == FTP and len(w.segments) >= 1
        assert wx.total_duration_s(w) > 0

def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        wx.build_structured_workout("inesistente", FTP)

def test_flatten_expands_intervals():
    w = wx.build_structured_workout("soglia", FTP)   # warmup + 4x(on/off) + cooldown
    steps = wx.flatten(w)
    # 1 warmup + 4*2 interval + 1 cooldown = 10
    assert len(steps) == 10

def test_estimate_tss_reasonable():
    w = wx.build_structured_workout("soglia", FTP)
    s = wx.summary(w)
    assert 40 <= s["tss"] <= 160          # una seduta di soglia sta in questo ordine
    assert s["duration_min"] > 20


# ===================== ZWIFT (.zwo) ======================================== #
def test_zwo_is_valid_xml_with_intervals():
    w = wx.build_structured_workout("vo2max", FTP)
    xml = wx.to_zwo(w)
    root = ET.fromstring(xml)                 # deve essere XML valido
    assert root.tag == "workout_file"
    wk = root.find("workout")
    assert wk is not None
    # c'e' almeno un blocco IntervalsT con il numero giusto di ripetizioni
    iv = wk.find("IntervalsT")
    assert iv is not None and iv.get("Repeat") == "5"
    # warmup e cooldown presenti come rampe
    assert wk.find("Warmup") is not None and wk.find("Cooldown") is not None

def test_zwo_steady_power_fraction():
    w = wx.build_structured_workout("fondo", FTP)
    root = ET.fromstring(wx.to_zwo(w))
    ss = root.find("workout").find("SteadyState")
    assert ss is not None
    assert float(ss.get("Power")) == pytest.approx(0.65, abs=0.001)


# ===================== ERG / MRC =========================================== #
def test_mrc_percent_points():
    w = wx.build_structured_workout("sweetspot", FTP)
    mrc = wx.to_mrc(w)
    assert "MINUTES PERCENT" in mrc
    body = [l for l in mrc.splitlines() if "\t" in l and not l.startswith("MINUTES")]
    # ogni step -> 2 punti
    assert len(body) == 2 * wx.summary(w)["n_steps"]
    # i valori sweet-spot ~90% compaiono
    assert any(abs(float(l.split("\t")[1]) - 90.0) < 0.6 for l in body)

def test_erg_watts_uses_ftp():
    w = wx.build_structured_workout("soglia", FTP)
    erg = wx.to_erg(w)
    assert "MINUTES WATTS" in erg
    body = [l for l in erg.splitlines() if "\t" in l and not l.startswith("MINUTES")]
    # step di soglia al 98% -> ~245 W con FTP 250
    watts = [float(l.split("\t")[1]) for l in body]
    assert max(watts) == pytest.approx(0.98 * FTP, abs=1)

def test_erg_without_ftp_raises():
    w = wx.build_structured_workout("fondo", FTP)
    w.ftp = None
    with pytest.raises(ValueError):
        wx.to_erg(w)


# ===================== FIT (round-trip con fitparse) ======================= #
def test_fit_roundtrips_with_fitparse():
    fitparse = pytest.importorskip("fitparse")
    w = wx.build_structured_workout("soglia", FTP)
    data = wx.to_fit(w)
    assert isinstance(data, (bytes, bytearray)) and len(data) > 100
    fit = fitparse.FitFile(io.BytesIO(bytes(data)))
    steps = list(fit.get_messages("workout_step"))
    assert len(steps) == wx.summary(w)["n_steps"]      # tutti gli step presenti
    wkt = list(fit.get_messages("workout"))
    assert len(wkt) == 1                                # un messaggio workout

def test_fit_needs_ftp():
    w = wx.build_structured_workout("fondo", FTP)
    w.ftp = None
    with pytest.raises(ValueError):
        wx.to_fit(w)


# ===================== DISPATCHER ========================================== #
def test_export_dispatcher():
    w = wx.build_structured_workout("vo2max", FTP)
    assert wx.export(w, "zwo").startswith("<?xml")
    assert "PERCENT" in wx.export(w, "mrc")
    assert "WATTS" in wx.export(w, ".erg")
    with pytest.raises(ValueError):
        wx.export(w, "csv")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
