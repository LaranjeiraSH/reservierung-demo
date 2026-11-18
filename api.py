"""
api.py – Dünne REST-Schicht um reserving_core.py für den Aufruf aus Bizagi.

Start:   uvicorn api:app --host 0.0.0.0 --port 8000
Doku:    http://localhost:8000/docs   (Swagger – hilfreich fürs Mapping im Bizagi-REST-Connector)

Endpunkte
  GET  /health                               Erreichbarkeit
  POST /api/v1/reserven/berechnen            JSON rein, JSON raus (Bizagi liefert Daten aus seinem Datenmodell)
  POST /api/v1/reserven/berechnen-aus-excel  Excel-Datei + Sparte (Bizagi reicht die hochgeladene Datei durch)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator

import reserving_core as core

app = FastAPI(title="Reservierungs-Service (Demo)", version=core.SKRIPT_VERSION)


# ----------------------------- Datenmodell -------------------------------- #
class Dreieckszelle(BaseModel):
    sparte: Optional[str] = Field(None, description="Optional: wenn gesetzt, filtert der Service auf die angefragte Sparte")
    anfalljahr: Optional[int] = None   # leere Excel-Zeilen werden ignoriert
    entwicklungsjahr: Optional[int] = None
    betrag: Optional[float] = Field(0.0, description="Kumulierte Zahlung (Schadendaten) bzw. inkrementelle Zahlung (neue Meldungen), TEUR")


class VorquartalsReserve(BaseModel):
    sparte: Optional[str] = None
    anfalljahr: Optional[int] = None
    reserve_vorher: Optional[float] = 0.0


class FaktorOverride(BaseModel):
    entwicklungsjahr: Optional[int] = Field(None, description="Faktor von EJ j nach j+1")
    faktor: Optional[float] = Field(None, description="Leer/0 = berechneten Faktor verwenden")
    begruendung: Optional[str] = Field("", description="Pflicht, sobald der Faktor vom berechneten abweicht (Audit-Trail)")


class BerechnungsAnfrage(BaseModel):
    vorgangsnummer: str = Field(..., description="Bizagi-Fallnummer für die Nachvollziehbarkeit")
    sparte: str
    stichtagsjahr: int
    tail_faktor: Optional[float] = 1.0
    wesentlichkeitsschwelle: Optional[float] = 0.10
    neue_meldungen_einbeziehen: Optional[bool] = True
    schadendaten: list[Dreieckszelle]
    neue_meldungen: Optional[list[Dreieckszelle]] = None
    reserven_vorquartal: Optional[list[VorquartalsReserve]] = None
    faktor_overrides: Optional[list[FaktorOverride]] = None
    grafik: Optional[bool] = True

    @model_validator(mode="after")
    def _defaults(self):
        # Bizagi schickt nicht gemappte/leere Felder ggf. als null -> Standardwerte setzen
        if not self.tail_faktor:
            self.tail_faktor = 1.0
        if not self.wesentlichkeitsschwelle:
            self.wesentlichkeitsschwelle = 0.10
        if self.neue_meldungen_einbeziehen is None:
            self.neue_meldungen_einbeziehen = True
        if self.grafik is None:
            self.grafik = True
        self.neue_meldungen = (self.neue_meldungen or []) if self.neue_meldungen_einbeziehen else []
        self.reserven_vorquartal = self.reserven_vorquartal or []
        self.faktor_overrides = self.faktor_overrides or []
        return self


# ------------------------------ Logik ------------------------------------- #
def _rechne(sparte, stichtagsjahr, tail, schwelle, sd, nm, vq, overrides, grafik, vorgang, input_fingerprint):
    zeilen = len(sd)
    mit_betrag = int((pd.to_numeric(sd["betrag"], errors="coerce").fillna(0) != 0).sum())
    if mit_betrag == 0:
        raise HTTPException(400, f"Alle {zeilen} Schadendaten-Zeilen der Sparte {sparte} haben Betrag 0 oder leer. "
                                 "Bitte das Mapping von schadendaten.betrag pruefen.")
    dreieck = core.baue_dreieck(sd, nm)
    spaltensummen = dreieck.fillna(0).sum()
    leer = [int(ej) for ej, summe in spaltensummen.items() if summe == 0]
    if leer:
        raise HTTPException(400, f"Entwicklungsjahr(e) {leer} haben nur Betraege 0 oder leer "
                                 f"({mit_betrag} von {zeilen} Zeilen mit Betrag). Bitte Datenbasis/Mapping pruefen.")
    befunde = core.pruefe_daten(dreieck, stichtagsjahr)
    if any(b["stufe"] == "FEHLER" for b in befunde):
        return {"status": "DATENFEHLER", "pruefbefunde": befunde, "metadaten": _meta(vorgang, input_fingerprint)}

    # Erst ohne Overrides rechnen, um echte Abweichungen zu erkennen (Bizagi darf die ganze Faktortabelle schicken)
    basis = core.chain_ladder(dreieck, tail)
    ov, begr = {}, {}
    for o in overrides:
        ej, wert = o.get("entwicklungsjahr"), o.get("faktor")
        if not wert or not ej or not 1 <= ej <= len(basis.faktoren_berechnet):
            continue
        if abs(wert - basis.faktoren_berechnet[ej - 1]) > 5e-5:
            if len((o.get("begruendung") or "").strip()) < 5:
                raise HTTPException(422, f"Faktor EJ {ej} wurde angepasst, aber ohne Begründung")
            ov[ej], begr[ej] = wert, o["begruendung"]
    erg = core.chain_ladder(dreieck, tail, ov)
    tab = core.vergleiche_mit_vorquartal(erg.je_anfalljahr, vq)

    summe_vorher = float(tab["reserve_vorher"].sum())
    summe_nachher = float(tab["reserve"].sum())
    delta_pct = (summe_nachher - summe_vorher) / summe_vorher if summe_vorher else 0.0
    wesentlich = abs(delta_pct) > schwelle

    return {
        "status": "OK",
        "sparte": sparte,
        "summen": {
            "reserve_vorher": round(summe_vorher, 1),
            "reserve_nachher": round(summe_nachher, 1),
            "delta": round(summe_nachher - summe_vorher, 1),
            "delta_prozent": round(delta_pct, 4),
            "mack_se_gesamt": round(float((tab["mack_se"] ** 2).sum() ** 0.5), 1),  # vereinfacht: ohne Kovarianzen
            "wesentlich": wesentlich,
            "empfehlung": "Zusatzfreigabe CFO + Information Pricing" if wesentlich else "Standardfreigabe",
        },
        "je_anfalljahr": [
            {
                "anfalljahr": int(r.anfalljahr),
                "zahlungen_kum": round(r.zahlungen_kum, 1),
                "cdf": round(r.cdf, 4),
                "ultimate": round(r.ultimate, 1),
                "reserve_vorher": round(r.reserve_vorher, 1),
                "reserve_nachher": round(r.reserve, 1),
                "delta": round(r.delta, 1),
                "delta_prozent": round(r.delta_prozent, 4),
                "mack_se": round(r.mack_se, 1),
            }
            for r in tab.itertuples()
        ],
        "faktoren": [
            {
                "entwicklungsjahr": j + 1,
                "faktor_berechnet": round(erg.faktoren_berechnet[j], 4),
                "faktor_verwendet": round(erg.faktoren_verwendet[j], 4),
                "angepasst": (j + 1) in ov,
                "begruendung": begr.get(j + 1, ""),
            }
            for j in range(len(erg.faktoren_berechnet))
        ],
        "pruefbefunde": befunde,
        "grafik_png_base64": core.grafik_png_base64(tab, f"Reserve {sparte}: Vorquartal vs. neu (TEUR)") if grafik else "",
        "grafik_html": core.grafik_html(tab, f"Reserve {sparte}: Vorquartal vs. neu (TEUR)", schwelle) if grafik else "",
        "metadaten": _meta(vorgang, input_fingerprint, methode="Chain-Ladder volumengewichtet, Mack-SE",
                           tail_faktor=tail, anzahl_overrides=len(ov)),
    }


def _meta(vorgang, fingerprint, **extra):
    return {
        "vorgangsnummer": vorgang,
        "skript": "reserving_core.py",
        "skript_version": core.SKRIPT_VERSION,
        "berechnet_am_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eingabe_sha256": fingerprint,  # belegt, mit welchen Daten gerechnet wurde
        **extra,
    }


def _ohne_nan(obj):
    """Ersetzt NaN/Inf durch None, damit die Antwort immer gueltiges JSON ist."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _ohne_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ohne_nan(v) for v in obj]
    return obj


from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def _unerwarteter_fehler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={
        "status": "FEHLER",
        "detail": f"Unerwarteter Fehler im Reservierungsservice: {type(exc).__name__}: {exc}",
        "skript_version": core.SKRIPT_VERSION})


# ----------------------------- Endpunkte ---------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "skript_version": core.SKRIPT_VERSION}


@app.post("/api/v1/reserven/berechnen")
def berechnen(req: BerechnungsAnfrage):
    # Diagnose im Render-Log: was ist tatsaechlich angekommen?
    print("EINGANG", json.dumps({
        "vorgangsnummer": req.vorgangsnummer, "sparte": req.sparte, "stichtagsjahr": req.stichtagsjahr,
        "neue_meldungen_einbeziehen": req.neue_meldungen_einbeziehen,
        "anzahl_schadendaten": len(req.schadendaten),
        "sparten_in_schadendaten": sorted({str(z.sparte) for z in req.schadendaten}),
        "beispiel_schadendaten": [z.model_dump() for z in req.schadendaten[:3]],
        "anzahl_neue_meldungen": len(req.neue_meldungen),
        "anzahl_vorquartal": len(req.reserven_vorquartal),
    }, ensure_ascii=False, default=str), flush=True)
    if not req.schadendaten:
        raise HTTPException(400, "Keine Schadendaten übergeben")
    fp = hashlib.sha256(json.dumps(req.model_dump(), sort_keys=True, default=str).encode()).hexdigest()
    def filt(items):
        return [i.model_dump() for i in items
                if i.sparte in (None, "", req.sparte) and i.anfalljahr
                and getattr(i, "entwicklungsjahr", 0) is not None]
    sd = pd.DataFrame(filt(req.schadendaten))
    if sd.empty:
        raise HTTPException(400, f"Keine Schadendaten für Sparte {req.sparte}")
    nm = pd.DataFrame(filt(req.neue_meldungen)) if filt(req.neue_meldungen) else None
    vq = pd.DataFrame(filt(req.reserven_vorquartal)) if filt(req.reserven_vorquartal) \
        else pd.DataFrame(columns=["anfalljahr", "reserve_vorher"])
    return _ohne_nan(_rechne(req.sparte, req.stichtagsjahr, req.tail_faktor, req.wesentlichkeitsschwelle, sd, nm, vq,
                             [o.model_dump() for o in req.faktor_overrides], req.grafik, req.vorgangsnummer, fp))


@app.post("/api/v1/reserven/berechnen-aus-excel")
async def berechnen_aus_excel(datei: UploadFile = File(...), sparte: str = Form("Sach"),
                              vorgangsnummer: str = Form("DEMO"),
                              neue_meldungen_einbeziehen: Optional[bool] = Form(None),
                              faktor_overrides_json: str = Form("[]")):
    inhalt = await datei.read()
    fp = hashlib.sha256(inhalt + sparte.encode() + faktor_overrides_json.encode()).hexdigest()
    try:
        import io
        d = core.lade_excel(io.BytesIO(inhalt), sparte, neue_meldungen_einbeziehen)
        overrides = json.loads(faktor_overrides_json)
    except Exception as e:
        raise HTTPException(400, f"Datei konnte nicht gelesen werden: {e}")
    return _ohne_nan(_rechne(sparte, d["stichtagsjahr"], d["tail_faktor"], d["wesentlichkeitsschwelle"],
                             d["schadendaten"], d["neue_meldungen"], d["vorquartal"], overrides, True, vorgangsnummer, fp))
