"""UN World Population Prospects 2024: population per country, used to allocate agent seats.

The compact CSV shipped in ``worldwarpixel/data/wpp2024_population.csv`` is derived from the
WPP2024 "Total Population by Sex" file (medium variant, 1 July 2024, countries/areas only).
Source: United Nations, DESA, Population Division (2024). World Population Prospects 2024.
Licensed under CC BY 3.0 IGO. Refresh it with ``uv run worldwarpixel-personas fetch-wpp``.
"""

from __future__ import annotations

import csv
import gzip
import io
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

WPP_URL = (
    "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/"
    "CSV_FILES/WPP2024_TotalPopulationBySex.csv.gz"
)
WPP_YEAR = "2024"
DATA_FILE = "wpp2024_population.csv"

# Shorter everyday names for the UN's formal location names.
SHORT_NAMES: dict[str, str] = {
    "USA": "United States",
    "RUS": "Russia",
    "COD": "DR Congo",
    "IRN": "Iran",
    "TUR": "Turkey",
    "TZA": "Tanzania",
    "KOR": "South Korea",
    "VEN": "Venezuela",
    "PRK": "North Korea",
    "SYR": "Syria",
    "TWN": "Taiwan",
    "BOL": "Bolivia",
    "LAO": "Laos",
    "HKG": "Hong Kong",
    "PSE": "Palestine",
    "MDA": "Moldova",
    "XKX": "Kosovo",
    "MAC": "Macao",
    "BRN": "Brunei",
    "FSM": "Micronesia",
    "VNM": "Vietnam",
    "CIV": "Côte d'Ivoire",
    "COG": "Republic of the Congo",
    "GBR": "United Kingdom",
    "CZE": "Czechia",
    "SWZ": "Eswatini",
    "FLK": "Falkland Islands",
    "VAT": "Vatican City",
}
# Countries whose WPP parent region is not in the medium-variant file.
NORTHERN_AMERICA = {"USA", "CAN", "BMU", "GRL", "SPM"}


@dataclass(frozen=True, slots=True)
class Country:
    iso3: str
    m49: int  # UN M49 numeric code (also WVS B_COUNTRY and the Q290 ethnic-code prefix)
    name: str
    un_name: str
    subregion: str
    population: int


def data_path() -> Path:
    return Path(str(resources.files("worldwarpixel") / "data" / DATA_FILE))


def load_countries(path: Path | None = None) -> list[Country]:
    """Countries sorted by population (largest first)."""
    with (path or data_path()).open(encoding="utf-8") as f:
        rows = [
            Country(
                r["iso3"],
                int(r["m49"]),
                r["name"],
                r["un_name"],
                r["subregion"],
                int(r["population"]),
            )
            for r in csv.DictReader(f)
        ]
    return sorted(rows, key=lambda c: (-c.population, c.iso3))


def parse_wpp(raw: io.TextIOBase) -> list[Country]:
    """Reduce the full WPP CSV (all variants, all years, all aggregates) to one row per country."""
    rows = [
        r
        for r in csv.DictReader(raw)
        if r["Time"] == WPP_YEAR and r["Variant"] == "Medium" and r["LocTypeName"]
    ]
    by_id = {r["LocID"]: r for r in rows}

    def subregion(r: dict[str, str]) -> str:
        if r["ISO3_code"] in NORTHERN_AMERICA:
            return "Northern America"
        parent = by_id.get(r["ParentID"])
        while parent is not None and parent["LocTypeName"] not in (
            "Subregion",
            "Geographic region",
        ):
            parent = by_id.get(parent["ParentID"])
        return parent["Location"] if parent else "Unknown"

    out = [
        Country(
            iso3=r["ISO3_code"],
            m49=int(r["LocID"]),
            name=SHORT_NAMES.get(r["ISO3_code"], r["Location"]),
            un_name=r["Location"],
            subregion=subregion(r),
            population=round(float(r["PopTotal"]) * 1000),
        )
        for r in rows
        if r["LocTypeName"] == "Country/Area" and r["ISO3_code"]
    ]
    return sorted(out, key=lambda c: (-c.population, c.iso3))


def write_countries(countries: list[Country], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso3", "m49", "name", "un_name", "subregion", "population"])
        for c in countries:
            w.writerow([c.iso3, c.m49, c.name, c.un_name, c.subregion, c.population])


def fetch_wpp(dest: Path | None = None, url: str = WPP_URL) -> Path:
    """Download the WPP2024 population file and rewrite the compact CSV."""
    dest = dest or data_path()
    with urllib.request.urlopen(url, timeout=600) as resp:
        gz = resp.read()
    with gzip.open(io.BytesIO(gz), "rt", encoding="utf-8-sig") as f:
        countries = parse_wpp(f)
    write_countries(countries, dest)
    return dest
