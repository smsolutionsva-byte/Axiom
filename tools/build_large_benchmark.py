from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Topic:
    slug: str
    title: str
    entity: str
    place: str
    date: str
    program: str
    signal: str
    metric: str
    value: str
    risk: str
    action: str
    partner: str


@dataclass(frozen=True)
class SourceFile:
    file_name: str
    text: str
    topic: Topic
    kind: str
    anchor: str


TOPICS = [
    Topic("monsoon-farming", "Monsoon Farming", "KAVERI-14", "Nashik", "2026-04-12", "drip irrigation", "canal reserve", "soil moisture", "31 percent", "pump relay overload", "move spare valves before rainfall", "AgriWatch Cell"),
    Topic("rural-clinic", "Rural Clinic", "AAROGYA-27", "Dharwad", "2026-04-15", "cold-chain vaccine route", "thermal breach", "storage temperature", "8.7 celsius", "generator battery sag", "dispatch ice packs", "District Health Desk"),
    Topic("flood-logistics", "Flood Logistics", "BRAHMAPUTRA-09", "Majuli", "2026-04-18", "boat fuel staging", "jetty queue", "diesel reserve", "420 liters", "bridge approach washout", "reroute via north ferry", "Relief Ops Room"),
    Topic("cyber-training", "Cyber Training", "SURAKSHA-33", "Pune", "2026-04-21", "phishing drill", "credential replay", "click rate", "11 percent", "shared password reuse", "rotate helpdesk tokens", "Security Education Team"),
    Topic("solar-microgrid", "Solar Microgrid", "URJA-52", "Jaisalmer", "2026-04-24", "battery balancing", "inverter alarm", "state of charge", "47 percent", "dust cover buildup", "schedule panel cleaning", "Microgrid Control"),
    Topic("rail-maintenance", "Rail Maintenance", "TRACK-18", "Itarsi", "2026-04-27", "axle sensor audit", "bearing heat", "sensor drift", "2.4 millimeters", "false hotbox alert", "calibrate platform reader", "Rail Diagnostics"),
    Topic("school-connectivity", "School Connectivity", "VIDYA-61", "Kohima", "2026-05-01", "offline lesson sync", "router packet loss", "sync success", "83 percent", "hilltop antenna fade", "ship backup modem", "Education Network Cell"),
    Topic("wildfire-watch", "Wildfire Watch", "AGNI-07", "Bandipur", "2026-05-04", "lookout camera grid", "smoke plume", "thermal anomaly", "64 hectares", "wind reversal", "pre-position water tenders", "Forest Response Desk"),
    Topic("port-customs", "Port Customs", "SAGAR-44", "Kandla", "2026-05-07", "container scan triage", "manifest mismatch", "scan backlog", "116 containers", "seal tamper pattern", "open lane three", "Port Risk Unit"),
    Topic("urban-water", "Urban Water", "JAL-23", "Indore", "2026-05-10", "chlorine pump audit", "pressure dip", "residual chlorine", "0.42 ppm", "valve telemetry gap", "inspect ward seven line", "Water Quality Cell"),
    Topic("disease-survey", "Disease Survey", "NIRAMAY-88", "Kozhikode", "2026-05-13", "fever cluster survey", "mosquito density", "rapid test positivity", "6.8 percent", "standing water pocket", "expand fogging ring", "Epidemiology Desk"),
    Topic("warehouse-audit", "Warehouse Audit", "BHANDAR-39", "Nagpur", "2026-05-16", "grain lot verification", "humidity spike", "lot variance", "19 sacks", "forklift route conflict", "seal bay four", "Supply Integrity Team"),
]


KINDS = [
    "briefing",
    "field_note",
    "operator_transcript",
    "dashboard_ocr",
    "annex_report",
    "audit_log",
]


def build_source_files(topics: list[Topic], files_per_topic: int) -> list[SourceFile]:
    rows: list[SourceFile] = []
    for topic_index, topic in enumerate(topics, start=1):
        for kind_index, kind in enumerate(KINDS[:files_per_topic], start=1):
            anchor = f"AXM-{topic_index:02d}-{kind_index:02d}"
            file_name = f"{topic_index:02d}_{topic.slug}_{kind}.txt"
            rows.append(SourceFile(file_name, source_text(topic, kind, anchor), topic, kind, anchor))
    return rows


def source_text(topic: Topic, kind: str, anchor: str) -> str:
    if kind == "briefing":
        return (
            f"{topic.title} briefing {anchor}\n\n"
            f"Program {topic.entity} in {topic.place} tracks the {topic.program} plan on {topic.date}. "
            f"The briefing records {topic.signal} as the primary coarse memory cue and lists {topic.metric} at {topic.value}. "
            f"Analysts must watch for {topic.risk} and should {topic.action}. Partner desk: {topic.partner}."
        )
    if kind == "field_note":
        return (
            f"Field note {anchor}\n\n"
            f"The {topic.partner} visit in {topic.place} confirms {topic.entity} needs local follow-up. "
            f"The note ties {topic.program} to the field observation '{topic.signal}' and says the measured {topic.metric} stayed near {topic.value}. "
            f"The field team named {topic.risk} as the operational blocker."
        )
    if kind == "operator_transcript":
        return (
            f"Operator transcript {anchor}\n\n"
            f"00:03 Operator: {topic.entity} is active in {topic.place}.\n"
            f"00:12 Analyst: Confirm whether {topic.signal} connects to {topic.risk}.\n"
            f"00:27 Operator: Yes, the {topic.program} record says to {topic.action} after {topic.metric} reached {topic.value}."
        )
    if kind == "dashboard_ocr":
        return (
            f"Dashboard OCR {anchor}\n\n"
            f"Visible panel: '{topic.entity} / {topic.place}'. Alert card: '{topic.signal}'. "
            f"Metric tile: '{topic.metric}: {topic.value}'. Action banner: '{topic.action}'. "
            f"The screenshot OCR also shows the risk label '{topic.risk}'."
        )
    if kind == "annex_report":
        return (
            f"Annex report {anchor}\n\n"
            f"The formal annex for {topic.entity} says {topic.program} depends on cross-checking briefing notes, dashboard OCR, and transcript evidence. "
            f"It recommends reviewing {topic.signal} beside {topic.metric} and treating {topic.risk} as the final approval gate."
        )
    return (
        f"Audit log {anchor}\n\n"
        f"Audit trail for {topic.entity}: source desk {topic.partner}; place {topic.place}; date {topic.date}; "
        f"tracked value {topic.metric} equals {topic.value}. The resolved operator action is '{topic.action}', "
        f"and the unresolved risk remains '{topic.risk}'."
    )


def build_axiom_cases(files: list[SourceFile]) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for row in files:
        cases.append(
            {
                "id": f"stress-{len(cases) + 1:03d}",
                "question": f"Which source records {row.anchor} for {row.topic.entity} and {row.topic.program}?",
                "expected_sources": [row.file_name],
                "expected_terms": [row.anchor.lower(), row.topic.entity.lower(), row.topic.program.lower()],
                "reference": f"{row.file_name} records {row.anchor} for {row.topic.entity} and the {row.topic.program} program.",
                "notes": f"{row.topic.slug}:{row.kind}",
            }
        )

    by_topic: dict[str, list[SourceFile]] = {}
    for row in files:
        by_topic.setdefault(row.topic.slug, []).append(row)

    for topic_files in by_topic.values():
        topic = topic_files[0].topic
        lookup = {row.kind: row for row in topic_files}
        transcript = lookup.get("operator_transcript")
        dashboard = lookup.get("dashboard_ocr")
        annex = lookup.get("annex_report")
        briefing = lookup.get("briefing")
        if transcript and dashboard:
            cases.append(
                {
                    "id": f"stress-{len(cases) + 1:03d}",
                    "question": f"Which transcript and dashboard evidence connect {topic.entity} to {topic.risk} in {topic.place}?",
                    "expected_sources": [transcript.file_name, dashboard.file_name],
                    "expected_terms": [topic.entity.lower(), topic.risk.lower(), topic.place.lower()],
                    "reference": f"The transcript and dashboard OCR connect {topic.entity} in {topic.place} to {topic.risk}.",
                    "notes": f"{topic.slug}:cross-modal",
                }
            )
        if annex and briefing:
            cases.append(
                {
                    "id": f"stress-{len(cases) + 1:03d}",
                    "question": f"What annex or briefing explains why {topic.signal} matters for {topic.metric} in {topic.entity}?",
                    "expected_sources": [annex.file_name, briefing.file_name],
                    "expected_terms": [topic.signal.lower(), topic.metric.lower(), topic.value.lower()],
                    "reference": f"The annex and briefing explain that {topic.signal} matters because {topic.metric} is {topic.value}.",
                    "notes": f"{topic.slug}:multi-hop",
                }
            )
    return cases


def build_ragas_rows(cases: list[dict[str, object]], source_by_name: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        expected_sources = [str(item) for item in case["expected_sources"]]
        reference_contexts = [source_by_name[name] for name in expected_sources if name in source_by_name]
        reference = str(case.get("reference") or " ".join(str(item) for item in case["expected_terms"]))
        rows.append(
            {
                "user_input": case["question"],
                "retrieved_contexts": reference_contexts,
                "reference_contexts": reference_contexts,
                "response": reference,
                "reference": reference,
                "rubrics": {
                    "case_id": str(case["id"]),
                    "expected_sources": ", ".join(expected_sources),
                    "expected_terms": ", ".join(str(item) for item in case["expected_terms"]),
                },
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_ragas_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    write_jsonl(path, rows)


def build_dataset(
    *,
    corpus_dir: Path,
    axiom_path: Path,
    ragas_path: Path,
    topics: int,
    files_per_topic: int,
    clean: bool,
) -> dict[str, int]:
    selected_topics = TOPICS[:topics]
    files = build_source_files(selected_topics, files_per_topic)
    if clean and corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for row in files:
        (corpus_dir / row.file_name).write_text(row.text + "\n", encoding="utf-8", newline="\n")

    cases = build_axiom_cases(files)
    write_jsonl(axiom_path, cases)
    write_ragas_jsonl(ragas_path, build_ragas_rows(cases, {row.file_name: row.text for row in files}))
    return {"files": len(files), "cases": len(cases), "ragas_samples": len(cases)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a large deterministic HiveRAG benchmark corpus.")
    parser.add_argument("--corpus", default="samples/stress_corpus", help="Output folder for generated evidence files.")
    parser.add_argument("--axiom", default="benchmarks/hiverag_stress_eval.jsonl", help="Axiom benchmark JSONL path.")
    parser.add_argument("--ragas", default="benchmarks/hiverag_stress_ragas.jsonl", help="Ragas EvaluationDataset JSONL path.")
    parser.add_argument("--topics", type=int, default=len(TOPICS), choices=range(1, len(TOPICS) + 1))
    parser.add_argument("--files-per-topic", type=int, default=len(KINDS), choices=range(1, len(KINDS) + 1))
    parser.add_argument("--no-clean", action="store_true", help="Do not remove the output corpus before writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_dataset(
        corpus_dir=Path(args.corpus),
        axiom_path=Path(args.axiom),
        ragas_path=Path(args.ragas),
        topics=args.topics,
        files_per_topic=args.files_per_topic,
        clean=not args.no_clean,
    )
    print(f"Generated files: {summary['files']}")
    print(f"Axiom benchmark cases: {summary['cases']}")
    print(f"Ragas samples: {summary['ragas_samples']}")
    print(f"Corpus: {Path(args.corpus).resolve()}")
    print(f"Axiom JSONL: {Path(args.axiom).resolve()}")
    print(f"Ragas JSONL: {Path(args.ragas).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
