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

SKRIPT_VERSION = "1.2.0"


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
