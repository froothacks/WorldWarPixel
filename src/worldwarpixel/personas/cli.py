"""``uv run worldwarpixel-personas``: build and inspect the persona data pool.

Subcommands:
  fetch-wpp                 refresh worldwarpixel/data/wpp2024_population.csv from population.un.org
  import-wvs <csv>          convert a WVS Wave 7 CSV release into data/wvs7_respondents.csv
  fetch-nemotron [ISO3...]  download Nemotron-Personas samples (hobbies/jobs) into data/nemotron/
  build-synthetic           regenerate the clearly-labelled synthetic fallback dataset
  show [--agents N] [--seed S] [--json]   print the seat allocation and sample personas
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from worldwarpixel.personas import nemotron, synthetic, wpp, wvs
from worldwarpixel.personas.allocation import allocate_seats
from worldwarpixel.personas.pool import load_pool, wvs_path


def _show(args: argparse.Namespace) -> None:
    from worldwarpixel.sim.agents import seed_agents

    countries = wpp.load_countries()
    pool = load_pool(Path(args.wvs) if args.wvs else None)
    seats = allocate_seats({c.iso3: c.population for c in countries}, args.agents)
    names = {c.iso3: c.name for c in countries}
    print(f"Seats for {args.agents} agents (UN WPP 2024, largest remainder):")
    for iso3 in sorted(seats, key=lambda k: (-seats[k], k)):
        print(f"  {seats[iso3]:4d}  {names[iso3]}")
    print(f"\nRespondent pool sources: {', '.join(sorted(pool.sources))}")
    agents = seed_agents(args.agents, 512, 320, 10.0, random.Random(args.seed), pool, countries)
    sample = agents[: args.sample]
    if args.json:
        print(json.dumps([a.to_dict() for a in sample], indent=2, ensure_ascii=False))
        return
    for a in sample:
        print(
            f"\n[{a.id}] {a.name} ({a.country}; data: {a.persona_card.get('source')}, "
            f"survey {a.persona_card.get('survey_country')})\n  You are {a.persona}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="worldwarpixel-personas",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-wpp")
    p = sub.add_parser("import-wvs")
    p.add_argument("csv", type=Path)
    p.add_argument(
        "--out",
        type=Path,
        default=wvs_path(),
        help="default: $WVS_RESPONDENTS_PATH or data/wvs7_respondents.csv",
    )
    p = sub.add_parser("fetch-nemotron")
    p.add_argument("iso3", nargs="*", default=sorted(nemotron.DATASETS))
    p.add_argument("--rows", type=int, default=nemotron.DEFAULT_ROWS)
    p = sub.add_parser("build-synthetic")
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("show")
    p.add_argument("--agents", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample", type=int, default=5)
    p.add_argument("--wvs", help="respondent CSV (default: $WVS_RESPONDENTS_PATH or data/)")
    p.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.cmd == "fetch-wpp":
        print(f"wrote {wpp.fetch_wpp()}")
    elif args.cmd == "import-wvs":
        n = wvs.import_wvs(args.csv, args.out)
        print(f"wrote {n} respondents to {args.out}")
    elif args.cmd == "fetch-nemotron":
        for iso3 in args.iso3:
            path, n = nemotron.fetch(iso3, rows=args.rows)
            print(f"wrote {n} rows to {path}")
    elif args.cmd == "build-synthetic":
        print(f"wrote {synthetic.write_synthetic(seed=args.seed)}")
    else:
        _show(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
