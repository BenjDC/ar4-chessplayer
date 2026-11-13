#!/usr/bin/env python3
"""
Export en lecture seule d'un document Onshape vers le dépôt.

Produit dans le dossier de sortie (par défaut mecanique/onshape/) :
  - RESUME.md                 : synthèse lisible (onglets, pièces, masses, features, assemblages)
  - elements.json             : liste des onglets du document
  - partstudios/<nom>/        : parts.json, features.json, massproperties.json (+ <nom>.step si --step)
  - assemblies/<nom>/         : definition.json, bom.json, bom.csv

Authentification : clés API Onshape lues dans les variables d'environnement
ONSHAPE_ACCESS_KEY et ONSHAPE_SECRET_KEY (jamais dans le code ni dans le dépôt).

Quota : chaque appel consomme le quota API Onshape du compte (2 500 / an en
compte individuel). Le script affiche le nombre d'appels effectués.
Ordre de grandeur : 1 + 3 par Part Studio + 2 par assemblage, et environ
5 de plus par Part Studio avec --step.

Usage :
  python tools/onshape_export.py --did <document> --wid <workspace> [--step] [--out mecanique/onshape]
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://cad.onshape.com/api/v10"


class Onshape:
    def __init__(self, access_key, secret_key):
        self.session = requests.Session()
        self.session.auth = (access_key, secret_key)
        self.session.headers.update({"Accept": "application/json;charset=UTF-8; qs=0.09"})
        self.calls = 0

    def _request(self, method, path, raw=False, **kwargs):
        url = path if path.startswith("http") else BASE + path
        for attempt in range(5):
            self.calls += 1
            r = self.session.request(method, url, timeout=120, **kwargs)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "10"))
                print(f"  limite de débit atteinte, attente {wait} s")
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"{method} {path} -> HTTP {r.status_code} : {r.text[:300]}")
            return r.content if raw else r.json()
        raise RuntimeError(f"{method} {path} : trop de refus (429)")

    def get(self, path, **params):
        return self._request("GET", path, params=params)

    def post(self, path, body):
        return self._request("POST", path, json=body)

    def download(self, path):
        return self._request("GET", path, raw=True,
                             headers={"Accept": "application/octet-stream"})


# ---------------------------------------------------------------- utilitaires

def slug(name):
    s = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")
    return s or "sans_nom"


def unwrap(obj):
    """Les réponses Onshape enveloppent parfois les objets dans 'message'."""
    if isinstance(obj, dict) and "message" in obj and isinstance(obj["message"], dict):
        return obj["message"]
    return obj


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def first(v):
    """Les propriétés physiques sont des triplets [valeur, min, max]."""
    return v[0] if isinstance(v, list) and v else v


def cell(value):
    if isinstance(value, dict):
        for key in ("name", "displayName", "value"):
            if key in value:
                return cell(value[key])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return " ; ".join(cell(v) for v in value)
    return "" if value is None else str(value)


# ---------------------------------------------------------------- Part Studios

def summarize_features(features_json):
    lines = []
    for f in features_json.get("features", []):
        f = unwrap(f)
        name = f.get("name", "?")
        ftype = f.get("featureType", "?")
        flag = " *(supprimée)*" if f.get("suppressed") else ""
        params = []
        for p in f.get("parameters", []):
            p = unwrap(p)
            expr = p.get("expression")
            if expr:
                params.append(f"{p.get('parameterId')} = {expr}")
        detail = f" — {', '.join(params)}" if params else ""
        lines.append(f"- {name} (`{ftype}`){flag}{detail}")
    return lines


def export_partstudio(api, did, wid, el, out, want_step, report):
    eid, name = el["id"], el.get("name", el["id"])
    folder = out / "partstudios" / slug(name)
    report.append(f"\n### Part Studio « {name} »\n")

    try:
        parts = api.get(f"/parts/d/{did}/w/{wid}/e/{eid}")
        save_json(folder / "parts.json", parts)
    except Exception as e:
        parts = []
        report.append(f"> Pièces : erreur ({e})\n")

    try:
        mass = api.get(f"/partstudios/d/{did}/w/{wid}/e/{eid}/massproperties",
                       massAsGroup="false")
        save_json(folder / "massproperties.json", mass)
        bodies = mass.get("bodies", {})
    except Exception as e:
        bodies = {}
        report.append(f"> Propriétés physiques : erreur ({e})\n")

    if parts:
        report.append("| Pièce | Matériau | Masse (g) | Volume (cm³) | Centre de gravité (mm) |")
        report.append("|---|---|---|---|---|")
        for p in parts:
            b = bodies.get(p.get("partId"), {})
            mass_g = first(b.get("mass")) if b.get("hasMass", True) else None
            vol = first(b.get("volume"))
            cg = b.get("centroid")
            mat = (p.get("material") or {}).get("displayName", "—")
            report.append(
                f"| {p.get('name')} | {mat} | "
                f"{'' if mass_g is None else f'{mass_g * 1000:.1f}'} | "
                f"{'' if vol is None else f'{vol * 1e6:.2f}'} | "
                f"{'' if not cg else ', '.join(f'{c * 1000:.1f}' for c in cg[:3])} |")
        report.append("")

    try:
        feats = api.get(f"/partstudios/d/{did}/w/{wid}/e/{eid}/features")
        save_json(folder / "features.json", feats)
        lines = summarize_features(feats)
        report.append(f"Arbre des features ({len(lines)}) :\n")
        report.extend(lines)
        report.append("")
    except Exception as e:
        report.append(f"> Features : erreur ({e})\n")

    if want_step:
        try:
            export_step(api, did, wid, eid, folder / f"{slug(name)}.step")
            report.append(f"STEP : `partstudios/{slug(name)}/{slug(name)}.step`\n")
        except Exception as e:
            report.append(f"> STEP : erreur ({e})\n")


def export_step(api, did, wid, eid, target):
    tr = api.post(f"/partstudios/d/{did}/w/{wid}/e/{eid}/translations",
                  {"formatName": "STEP", "storeInDocument": False})
    tid = tr["id"]
    for _ in range(40):
        state = api.get(f"/translations/{tid}")
        if state.get("requestState") == "DONE":
            fid = state["resultExternalDataIds"][0]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(api.download(f"/documents/d/{did}/externaldata/{fid}"))
            return
        if state.get("requestState") == "FAILED":
            raise RuntimeError(state.get("failureReason", "traduction échouée"))
        time.sleep(3)
    raise RuntimeError("traduction STEP trop longue")


# ---------------------------------------------------------------- Assemblages

def export_assembly(api, did, wid, el, out, report):
    eid, name = el["id"], el.get("name", el["id"])
    folder = out / "assemblies" / slug(name)
    report.append(f"\n### Assemblage « {name} »\n")

    try:
        definition = api.get(f"/assemblies/d/{did}/w/{wid}/e/{eid}",
                             includeMateFeatures="true",
                             includeMateConnectors="false",
                             includeNonSolids="false")
        save_json(folder / "definition.json", definition)
        root = definition.get("rootAssembly", {})
        instances = root.get("instances", [])
        occurrences = {tuple(o.get("path", [])): o for o in root.get("occurrences", [])}
        report.append(f"Instances de premier niveau ({len(instances)}) :\n")
        report.append("| Instance | Type | Position (mm) | Source externe |")
        report.append("|---|---|---|---|")
        for inst in instances:
            occ = occurrences.get((inst.get("id"),), {})
            t = occ.get("transform") or []
            pos = ", ".join(f"{t[i] * 1000:.1f}" for i in (3, 7, 11)) if len(t) == 16 else ""
            external = "oui" if inst.get("documentId") and inst.get("documentId") != did else ""
            report.append(f"| {inst.get('name')} | {inst.get('type')} | {pos} | {external} |")
        mates = [unwrap(f) for f in root.get("features", [])]
        if mates:
            report.append(f"\nContraintes ({len(mates)}) :\n")
            for m in mates:
                data = m.get("featureData", {})
                report.append(f"- {data.get('name', '?')} (`{data.get('mateType', m.get('featureType', '?'))}`)"
                              + (" *(supprimée)*" if m.get("suppressed") else ""))
        report.append("")
    except Exception as e:
        report.append(f"> Définition : erreur ({e})\n")

    try:
        bom = api.get(f"/assemblies/d/{did}/w/{wid}/e/{eid}/bom",
                      indented="true", generateIfAbsent="true")
        save_json(folder / "bom.json", bom)
        headers = bom.get("headers", [])
        rows = bom.get("rows", [])
        names = [h.get("name", h.get("id")) for h in headers]
        with open(folder / "bom.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["Niveau"] + names)
            for r in rows:
                values = r.get("headerIdToValue", {})
                w.writerow([r.get("indentLevel", "")] + [cell(values.get(h.get("id"))) for h in headers])
        report.append(f"Nomenclature : {len(rows)} lignes, voir `assemblies/{slug(name)}/bom.csv`\n")
    except Exception as e:
        report.append(f"> Nomenclature : erreur ({e})\n")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Export lecture seule d'un document Onshape")
    ap.add_argument("--did", required=True, help="ID du document")
    ap.add_argument("--wid", required=True, help="ID du workspace")
    ap.add_argument("--out", default="mecanique/onshape", help="dossier de sortie")
    ap.add_argument("--step", action="store_true", help="exporter aussi un STEP par Part Studio")
    args = ap.parse_args()

    ak, sk = os.environ.get("ONSHAPE_ACCESS_KEY"), os.environ.get("ONSHAPE_SECRET_KEY")
    if not ak or not sk:
        sys.exit("ONSHAPE_ACCESS_KEY et ONSHAPE_SECRET_KEY doivent être définies.")

    api = Onshape(ak, sk)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    doc = api.get(f"/documents/{args.did}")
    elements = api.get(f"/documents/d/{args.did}/w/{args.wid}/elements")
    save_json(out / "elements.json", elements)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = [
        f"# Export Onshape – {doc.get('name', args.did)}",
        "",
        f"*Généré automatiquement le {now} par `tools/onshape_export.py`. Ne pas modifier à la main.*",
        "",
        f"Document : https://cad.onshape.com/documents/{args.did}/w/{args.wid}",
        "",
        "## Onglets",
        "",
        "| Onglet | Type |",
        "|---|---|",
    ]
    for el in elements:
        report.append(f"| {el.get('name')} | {el.get('elementType')} |")

    report.append("\n## Détail")
    for el in elements:
        etype = el.get("elementType")
        print(f"- {el.get('name')} ({etype})")
        if etype == "PARTSTUDIO":
            export_partstudio(api, args.did, args.wid, el, out, args.step, report)
        elif etype == "ASSEMBLY":
            export_assembly(api, args.did, args.wid, el, out, report)

    report.append(f"\n---\nAppels API utilisés pour cet export : {api.calls}\n")
    (out / "RESUME.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Export terminé dans {out}/ – {api.calls} appels API.")


if __name__ == "__main__":
    main()
