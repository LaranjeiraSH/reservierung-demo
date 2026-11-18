"""
reserving_core.py – Aktuarielle Kernlogik (Paid-Chain-Ladder + Mack-Standardfehler)

Diese Datei steht in der Demo stellvertretend für ein BESTEHENDES Aktuariatsskript.
Sie kennt weder Bizagi noch HTTP: reine Fachlogik auf pandas/numpy-Basis.
Die Anbindung an Bizagi übernimmt die dünne Schicht in api.py.

Eigenständig ausführbar (so wie Aktuare es heute tun):
    python reserving_core.py Reservierung_Datenbasis_Demo.xlsx --sparte Sach
"""
from __future__ import annotations

import argparse
import base64
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SKRIPT_VERSION = "1.3.0"


# --------------------------------------------------------------------------- #
# Datenaufbereitung
# --------------------------------------------------------------------------- #
def baue_dreieck(schadendaten: pd.DataFrame,
                 neue_meldungen: pd.DataFrame | None = None) -> pd.DataFrame:
    """Kumuliertes Abwicklungsdreieck (Zeilen = Anfalljahr, Spalten = Entwicklungsjahr).

    Neue Meldungen sind inkrementelle Zahlungen und werden auf die betroffene Zelle addiert.
    Erwartete Spalten: anfalljahr, entwicklungsjahr, betrag
    """
    df = schadendaten[["anfalljahr", "entwicklungsjahr", "betrag"]].copy()
    if neue_meldungen is not None and len(neue_meldungen):
        df = pd.concat([df, neue_meldungen[["anfalljahr", "entwicklungsjahr", "betrag"]]])
    df[["anfalljahr", "entwicklungsjahr"]] = df[["anfalljahr", "entwicklungsjahr"]].astype(int)
    dreieck = df.groupby(["anfalljahr", "entwicklungsjahr"])["betrag"].sum().unstack()
    return dreieck.sort_index().sort_index(axis=1).astype(float)


def pruefe_daten(dreieck: pd.DataFrame, stichtagsjahr: int) -> list[dict]:
    """Einfache Plausibilitätsprüfungen, wie sie vor jeder Reservierung laufen."""
    befunde = []
    n = dreieck.shape[1]
    for ay in dreieck.index:
        erwartet = min(stichtagsjahr - ay + 1, n)
        zeile = dreieck.loc[ay]
        for ej in range(1, erwartet + 1):
            if pd.isna(zeile.get(ej)):
                befunde.append({"stufe": "FEHLER", "anfalljahr": int(ay), "entwicklungsjahr": ej,
                                "meldung": "Wert im Dreieck fehlt"})
        werte = zeile.dropna()
        if (werte < 0).any():
            befunde.append({"stufe": "FEHLER", "anfalljahr": int(ay), "entwicklungsjahr": None,
                            "meldung": "Negative kumulierte Zahlung"})
        if (werte.diff().dropna() < 0).any():
            befunde.append({"stufe": "WARNUNG", "anfalljahr": int(ay), "entwicklungsjahr": None,
                            "meldung": "Kumulierte Zahlungen sinken (Regress/Korrektur prüfen)"})
    return befunde


# --------------------------------------------------------------------------- #
# Chain-Ladder
# --------------------------------------------------------------------------- #
@dataclass
class ChainLadderErgebnis:
    faktoren_berechnet: list[float]
    faktoren_verwendet: list[float]
    cdf: list[float]
    je_anfalljahr: pd.DataFrame
    sigma2: list[float] = field(default_factory=list)


def chain_ladder(dreieck: pd.DataFrame, tail_faktor: float = 1.0,
                 faktor_overrides: dict[int, float] | None = None) -> ChainLadderErgebnis:
    """Volumengewichtete Chain-Ladder.

    faktor_overrides: {Entwicklungsjahr j: Faktor j→j+1} – Expertenannahmen des Aktuars.
    """
    C = dreieck.to_numpy()
    n_ay, n_ej = C.shape

    f = []
    for j in range(n_ej - 1):
        maske = ~np.isnan(C[:, j + 1]) & ~np.isnan(C[:, j])
        f.append(C[maske, j + 1].sum() / C[maske, j].sum() if maske.any() else 1.0)
    f_berechnet = list(f)

    if faktor_overrides:
        for ej, wert in faktor_overrides.items():
            if 1 <= int(ej) <= n_ej - 1:
                f[int(ej) - 1] = float(wert)

    # Kumulierte Faktoren: CDF[j] = Abwicklung von EJ j+1 bis Ultimate (inkl. Tail)
    cdf = [0.0] * n_ej
    cdf[-1] = tail_faktor
    for j in range(n_ej - 2, -1, -1):
        cdf[j] = f[j] * cdf[j + 1]

    zeilen = []
    for i, ay in enumerate(dreieck.index):
        letzte = int(np.max(np.where(~np.isnan(C[i]))[0]))
        bezahlt = C[i, letzte]
        ultimate = bezahlt * cdf[letzte]
        zeilen.append({"anfalljahr": int(ay), "letztes_ej": letzte + 1, "zahlungen_kum": bezahlt,
                       "cdf": cdf[letzte], "ultimate": ultimate, "reserve": ultimate - bezahlt})

    erg = ChainLadderErgebnis(f_berechnet, f, cdf, pd.DataFrame(zeilen))
    erg.sigma2 = _mack_sigma2(C, f)
    erg.je_anfalljahr["mack_se"] = _mack_se(C, f, erg.sigma2, tail_faktor)
    return erg


def _mack_sigma2(C: np.ndarray, f: list[float]) -> list[float]:
    n_ej = C.shape[1]
    s2 = []
    for j in range(n_ej - 1):
        maske = ~np.isnan(C[:, j + 1]) & ~np.isnan(C[:, j])
        k = maske.sum()
        if k > 1:
            c0, c1 = C[maske, j], C[maske, j + 1]
            s2.append(float(np.sum(c0 * (c1 / c0 - f[j]) ** 2) / (k - 1)))
        else:  # Extrapolation nach Mack (1993)
            if len(s2) >= 2 and s2[-2] > 0:
                s2.append(min(s2[-1] ** 2 / s2[-2], s2[-2], s2[-1]))
            else:
                s2.append(s2[-1] if s2 else 0.0)
    return s2


def _mack_se(C: np.ndarray, f: list[float], s2: list[float], tail: float) -> list[float]:
    """Mack-Standardfehler der Reserve je Anfalljahr (Tail ohne eigene Unsicherheit)."""
    n_ay, n_ej = C.shape
    # Spaltensummen der beobachteten Werte, die in die Faktorschätzung eingehen
    spaltensumme = []
    for j in range(n_ej - 1):
        maske = ~np.isnan(C[:, j + 1]) & ~np.isnan(C[:, j])
        spaltensumme.append(C[maske, j].sum())
    se = []
    for i in range(n_ay):
        letzte = int(np.max(np.where(~np.isnan(C[i]))[0]))
        # projizierte Werte der Zeile
        proj = [np.nan] * n_ej
        proj[letzte] = C[i, letzte]
        for j in range(letzte, n_ej - 1):
            proj[j + 1] = proj[j] * f[j]
        ult = proj[-1] * tail
        summe = 0.0
        for j in range(letzte, n_ej - 1):
            if f[j] > 0 and proj[j] > 0 and spaltensumme[j] > 0:
                summe += (s2[j] / f[j] ** 2) * (1 / proj[j] + 1 / spaltensumme[j])
        se.append(float(np.sqrt(ult ** 2 * summe)))
    return se


# --------------------------------------------------------------------------- #
# Vorher/Nachher + Grafik
# --------------------------------------------------------------------------- #
def vergleiche_mit_vorquartal(je_ay: pd.DataFrame, vorquartal: pd.DataFrame) -> pd.DataFrame:
    df = je_ay.merge(vorquartal[["anfalljahr", "reserve_vorher"]], on="anfalljahr", how="left")
    df["reserve_vorher"] = df["reserve_vorher"].fillna(0.0)
    df["delta"] = df["reserve"] - df["reserve_vorher"]
    df["delta_prozent"] = np.where(df["reserve_vorher"] != 0, df["delta"] / df["reserve_vorher"], 0.0)
    return df


def grafik_png_base64(df: pd.DataFrame, titel: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["reserve_vorher"], 0.4, label="Vorquartal", color="#9DB3D9")
    ax.bar(x + 0.2, df["reserve"], 0.4, label="Neu berechnet", color="#1F3864")
    ax.set_xticks(x, df["anfalljahr"].astype(str))
    ax.set_ylabel("Reserve (TEUR)")
    ax.set_title(titel)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()



def _de(x: float, nachkomma: int = 0) -> str:
    """Zahl im deutschen Format (1.234,5)."""
    s = f"{x:,.{nachkomma}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def grafik_html(df: pd.DataFrame, titel: str, schwelle: float = 0.10) -> str:
    """Balkendiagramm Vorquartal vs. neu als eigenständiges HTML/SVG-Fragment.

    Ohne JavaScript, ohne externe Dateien, nur Inline-Attribute: geeignet für
    Bizagi-Widgets, die HTML aus einem Attribut anzeigen.
    """
    import math
    from html import escape

    W, H = 760, 360
    ml, mr, mt, mb = 70, 20, 64, 56
    pw, ph = W - ml - mr, H - mt - mb
    vorher = [float(v) for v in df["reserve_vorher"]]
    nachher = [float(v) for v in df["reserve"]]
    jahre = [str(int(a)) for a in df["anfalljahr"]]
    dpct = [float(v) for v in df["delta_prozent"]]
    n = len(jahre)

    roh = max(vorher + nachher + [1.0]) * 1.12
    exp = 10 ** math.floor(math.log10(roh / 5))
    schritt = next(f * exp for f in (1, 2, 2.5, 5, 10) if roh / (f * exp) <= 6)
    ymax = schritt * math.ceil(roh / schritt)
    y = lambda v: mt + ph - (v / ymax) * ph

    c_vor, c_neu, c_ink, c_text, c_grid, c_hl = "#C697E8", "#05803E", "#0C0E0D", "#303633", "#E4E7E5", "#652A88"
    font = "Open Sans, Segoe UI, Arial, sans-serif"
    t = []
    t.append(f'<text x="{ml}" y="26" font-size="16" font-weight="700" fill="{c_ink}">{escape(titel)}</text>')
    # Legende
    lx = W - mr - 250
    t.append(f'<rect x="{lx}" y="14" width="12" height="12" rx="2" fill="{c_vor}"/>'
             f'<text x="{lx + 18}" y="25" font-size="12" fill="{c_text}">Vorquartal</text>')
    t.append(f'<rect x="{lx + 110}" y="14" width="12" height="12" rx="2" fill="{c_neu}"/>'
             f'<text x="{lx + 128}" y="25" font-size="12" fill="{c_text}">Neu berechnet</text>')
    # Raster und Achse
    v = 0.0
    while v <= ymax + 1e-9:
        yy = y(v)
        t.append(f'<line x1="{ml}" x2="{W - mr}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{c_grid}" stroke-width="1"/>')
        t.append(f'<text x="{ml - 8}" y="{yy + 4:.1f}" font-size="11" text-anchor="end" fill="{c_text}">{_de(v)}</text>')
        v += schritt
    t.append(f'<text x="16" y="{mt + ph / 2:.1f}" font-size="11" fill="{c_text}" text-anchor="middle" '
             f'transform="rotate(-90 16 {mt + ph / 2:.1f})">Reserve (TEUR)</text>')
    # Balken
    gw = pw / n
    bw = min(26.0, gw * 0.32)
    for i in range(n):
        cx = ml + gw * i + gw / 2
        for wert, farbe, dx in ((vorher[i], c_vor, -bw - 2), (nachher[i], c_neu, 2)):
            hoehe = max(1.5, (wert / ymax) * ph)
            t.append(f'<rect x="{cx + dx:.1f}" y="{mt + ph - hoehe:.1f}" width="{bw:.1f}" height="{hoehe:.1f}" '
                     f'rx="2" fill="{farbe}"><title>{jahre[i]}: {_de(wert)} TEUR</title></rect>')
        if nachher[i] >= ymax * 0.04:
            t.append(f'<text x="{cx + 2 + bw / 2:.1f}" y="{y(nachher[i]) - 6:.1f}" font-size="11" '
                     f'text-anchor="middle" fill="{c_ink}">{_de(nachher[i])}</text>')
        if abs(dpct[i]) > schwelle:
            vz = "+" if dpct[i] > 0 else "−"
            oben = y(max(vorher[i], nachher[i])) - 24
            t.append(f'<text x="{cx:.1f}" y="{oben:.1f}" font-size="12" font-weight="700" text-anchor="middle" '
                     f'fill="{c_hl}">{vz}{_de(abs(dpct[i]) * 100, 1)} %</text>')
        t.append(f'<text x="{cx:.1f}" y="{mt + ph + 20:.1f}" font-size="12" text-anchor="middle" fill="{c_text}">{jahre[i]}</text>')
    t.append(f'<line x1="{ml}" x2="{W - mr}" y1="{mt + ph}" y2="{mt + ph}" stroke="#A1AAA5" stroke-width="1"/>')
    t.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 12}" font-size="11" text-anchor="middle" fill="{c_text}">Anfalljahr</text>')

    summe_v, summe_n = sum(vorher), sum(nachher)
    label = (f"Reserve je Anfalljahr, Vorquartal gesamt {_de(summe_v)} TEUR, "
             f"neu gesamt {_de(summe_n)} TEUR")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'style="max-width:100%;height:auto;display:block" role="img" aria-label="{escape(label)}" '
           f'font-family="{font}">' + "".join(t) + "</svg>")
    return f'<div style="font-family:{font};max-width:{W}px">{svg}</div>'

# --------------------------------------------------------------------------- #
# Excel-Einlesen (Format der Demo-Datenbasis)
# --------------------------------------------------------------------------- #
def lade_excel(pfad_oder_datei, sparte: str, neue_meldungen: bool | None = None):
    xls = pd.ExcelFile(pfad_oder_datei)
    sd = pd.read_excel(xls, "Schadendaten")
    nm = pd.read_excel(xls, "Neue_Schadenmeldungen")
    vq = pd.read_excel(xls, "Reserven_Vorquartal")
    par = pd.read_excel(xls, "Parameter", header=2, nrows=8).set_index("Schluessel")["Wert"]

    sd = sd[sd["Sparte"] == sparte].rename(columns={"Anfalljahr": "anfalljahr", "Entwicklungsjahr": "entwicklungsjahr",
                                                    "Kum_Zahlungen_TEUR": "betrag"})
    nm = nm.dropna(subset=["Meldungs_ID"])
    nm = nm[nm["Sparte"] == sparte].rename(columns={"Anfalljahr": "anfalljahr", "Entwicklungsjahr": "entwicklungsjahr",
                                                    "Zahlung_TEUR": "betrag"})
    vq = vq[vq["Sparte"] == sparte].rename(columns={"Anfalljahr": "anfalljahr", "Reserve_TEUR": "reserve_vorher"})
    einbeziehen = (str(par["Neue_Meldungen_einbeziehen"]).upper() == "JA") if neue_meldungen is None else neue_meldungen
    return {
        "schadendaten": sd, "neue_meldungen": nm if einbeziehen else None, "vorquartal": vq,
        "stichtagsjahr": int(par["Stichtagsjahr"]), "tail_faktor": float(par["Tail_Faktor"]),
        "wesentlichkeitsschwelle": float(par["Wesentlichkeitsschwelle"]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chain-Ladder-Reservierung aus der Demo-Excel")
    ap.add_argument("excel")
    ap.add_argument("--sparte", default="Sach")
    a = ap.parse_args()

    d = lade_excel(a.excel, a.sparte)
    dreieck = baue_dreieck(d["schadendaten"], d["neue_meldungen"])
    for b in pruefe_daten(dreieck, d["stichtagsjahr"]):
        print(f"[{b['stufe']}] AJ {b['anfalljahr']}: {b['meldung']}")
    erg = chain_ladder(dreieck, d["tail_faktor"])
    tab = vergleiche_mit_vorquartal(erg.je_anfalljahr, d["vorquartal"])
    pd.options.display.float_format = "{:,.1f}".format
    print(tab[["anfalljahr", "zahlungen_kum", "ultimate", "reserve_vorher", "reserve", "delta", "mack_se"]].to_string(index=False))
    gesamt_delta = tab["delta"].sum() / tab["reserve_vorher"].sum()
    print(f"\nReserve gesamt: {tab['reserve'].sum():,.0f} TEUR  (Vorquartal {tab['reserve_vorher'].sum():,.0f}, "
          f"Delta {gesamt_delta:+.1%})")
