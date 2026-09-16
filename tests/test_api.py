"""API integration.

Exercises the endpoints the workbench actually calls, including the full
edit → approve → reload → memory round trip that is the completion criterion
for this phase.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scanlate.api.app import create_app
from scanlate.fixtures import build_services, seed_demo


@pytest.fixture()
def client(tmp_path):
    services = build_services(tmp_path / "scanlate.db")
    seed_demo(services)
    app = create_app(services, seed=False)
    with TestClient(app) as test_client:
        test_client.services = services
        yield test_client


def segments(client):
    return {s["id"]: s for s in client.get("/api/projects/demo/segments").json()["segments"]}


# --- project and segments ----------------------------------------------
def test_project_endpoint_reports_structure(client):
    project = client.get("/api/projects/demo").json()
    assert project["name"] == "Yoru no Kissaten"
    assert project["default_source_language"] == "ja"
    assert [c["number"] for c in project["chapters"]] == [1]
    assert [p["number"] for p in project["pages"]] == [12, 13]


def test_segment_list_returns_real_pipeline_output(client):
    payload = client.get("/api/projects/demo/segments").json()
    assert payload["project"]["target_language"] == "en"
    by_id = {s["id"]: s for s in payload["segments"]}
    assert len(by_id) == 7

    polite = by_id["p012-b1"]
    assert polite["candidate"] == "Tanaka-senpai, do you have a second?"
    assert polite["features"]["register"] == "polite"
    assert polite["features"]["ja.politeness"] == "desu/masu"
    assert [(a["start"], a["end"], a["target"]) for a in polite["annotations"]
            if a["type"] == "term"] == [(0, 2, "Tanaka"), (2, 4, "senpai")]


def test_mixed_language_segments_carry_their_own_analysis(client):
    by_id = segments(client)
    assert by_id["p012-b2"]["language"] == "ja"
    assert by_id["p013-b1"]["language"] == "ko"
    assert by_id["p013-b1"]["features"]["ko.speech_level"] == "haeche"
    assert "ko.speech_level" not in by_id["p012-b2"]["features"]


def test_warnings_reach_the_wire_with_actions(client):
    mismatch = next(w for w in segments(client)["p012-b2"]["warnings"]
                    if w["code"] == "register.mismatch")
    assert mismatch["severity"] == "WARNING"
    assert mismatch["data"]["source_register"] == "crude"
    assert [a["kind"] for a in mismatch["actions"]] == ["show_alternatives", "dismiss"]
    assert len(mismatch["fingerprint"]) == 16


def test_precedence_is_visible_in_the_response(client):
    by_id = segments(client)
    assert by_id["p013-b2"]["origin"] == "translation_memory"   # canonical memory
    assert by_id["p013-b2"]["locked"] is True
    assert by_id["p013-b3"]["candidate"] == "Aniki, help me out!"   # locked glossary
    assert by_id["p013-b3"]["origin"] == "glossary"
    assert by_id["p012-s1"]["origin"] == "rule"                # SFX path
    assert by_id["p012-s1"]["candidate"] == "BA-DUMP"


def test_no_backend_internals_leak_to_the_client(client):
    for segment in segments(client).values():
        assert set(segment) == {
            "id", "project_id", "page_id", "order", "kind", "language", "target_language",
            "source", "candidate", "annotations", "alternatives", "features", "warnings",
            "status", "origin", "locked", "needs_review"}
        assert "memory_id" not in segment["features"]
        assert all("entry_id" not in (w["data"] or {}) or w["code"].startswith("terminology")
                   for w in segment["warnings"])


# --- editing and approval ----------------------------------------------
def test_patch_records_the_edit_and_revalidates(client):
    response = client.patch("/api/segments/p012-b2",
                            json={"translation": "Shut up… I'm busy right now."})
    updated = response.json()
    assert updated["status"] == "edited"
    assert updated["origin"] == "manual"
    # The register mismatch against the polite machine output is gone, because
    # the user's own text is what gets validated now.
    assert not any(w["code"] == "register.mismatch" for w in updated["warnings"])
    assert updated["needs_review"] is False


def test_approval_persists_reloads_and_enters_memory(client):
    client.patch("/api/segments/p012-b2", json={"translation": "Shut up… I'm busy right now."})
    approved = client.post("/api/segments/p012-b2/approve",
                           json={"translation": "Shut up… I'm busy right now."}).json()
    assert approved["segment"]["status"] == "approved"
    assert approved["memory_entry_id"] is not None

    # Reload: the approval survived.
    assert segments(client)["p012-b2"]["candidate"] == "Shut up… I'm busy right now."
    assert segments(client)["p012-b2"]["status"] == "approved"

    memory = client.get("/api/projects/demo/memory").json()
    entry = next(e for e in memory if e["source_text"] == "うるせぇな…今忙しいんだよ。")
    assert entry["target_text"] == "Shut up… I'm busy right now."
    assert entry["occurrences"] == 1


def test_approval_returns_a_suggestion_that_is_not_yet_a_rule(client):
    response = client.post("/api/segments/p013-b1/approve", json={
        "translation": "Bro, are you really okay?",
        "term_resolutions": [{"source_term": "형", "target_term": "bro"}]}).json()

    suggestion = response["suggestion"]
    assert (suggestion["source_term"], suggestion["target_term"]) == ("형", "bro")
    assert suggestion["explicit"] is True
    assert client.get("/api/projects/demo/glossary?source_language=ko").json() == []

    client.post("/api/projects/demo/glossary/accept", json={
        "source_language": "ko", "source_term": "형", "target_language": "en",
        "target_term": "bro", "category": "relationship_term", "locked": True})
    entries = client.get("/api/projects/demo/glossary?source_language=ko").json()
    assert entries[0]["target_term"] == "bro" and entries[0]["locked"] is True


def test_a_locked_rule_changes_later_translations(client):
    """The completion criterion: a glossary decision affects the next segment."""
    before = segments(client)["p013-b1"]
    assert "hyung" in before["candidate"].lower()
    assert any(w["code"] == "relationship.unresolved" for w in before["warnings"])

    client.post("/api/projects/demo/glossary", json={
        "source_language": "ko", "source_term": "형", "target_language": "en",
        "target_term": "Hyung-nim", "category": "relationship_term", "locked": True,
        "notes": "hyung|bro"})

    after = segments(client)["p013-b1"]
    assert after["candidate"] == "Hyung-nim, are you really okay?"
    assert after["origin"] == "glossary"
    assert not any(w["code"] == "relationship.unresolved" for w in after["warnings"])
    assert any(a["locked"] and a["target"] == "Hyung-nim" for a in after["annotations"])


def test_reopen_releases_the_memory_occurrence(client):
    client.post("/api/segments/p012-b2/approve", json={"translation": "Shut up."})
    entry = next(e for e in client.get("/api/projects/demo/memory").json()
                 if e["target_text"] == "Shut up.")
    assert entry["occurrences"] == 1

    reopened = client.post("/api/segments/p012-b2/reopen").json()
    assert reopened["status"] == "edited"
    entry = next(e for e in client.get("/api/projects/demo/memory").json()
                 if e["target_text"] == "Shut up.")
    assert entry["occurrences"] == 0        # still on record, no longer endorsed


def test_dismissing_an_issue_persists_across_reloads(client):
    warning = next(w for w in segments(client)["p013-b1"]["warnings"]
                   if w["code"] == "relationship.unresolved")
    client.post("/api/segments/p013-b1/issues/resolve",
                json={"fingerprint": warning["fingerprint"], "code": warning["code"]})

    reloaded = segments(client)["p013-b1"]
    assert not any(w["code"] == "relationship.unresolved" for w in reloaded["warnings"])
    assert reloaded["needs_review"] is False
    # Other warnings on the same segment are untouched.
    assert any(w["code"].startswith("ambiguity.") for w in reloaded["warnings"])


def test_translate_endpoint_leaves_user_text_alone(client):
    client.post("/api/segments/p012-b2/approve", json={"translation": "Shut up."})
    summary = client.post("/api/projects/demo/translate").json()
    assert summary["translated"] < len(summary["segments"])
    approved = next(s for s in summary["segments"] if s["id"] == "p012-b2")
    assert approved["candidate"] == "Shut up." and approved["status"] == "approved"


# --- errors -------------------------------------------------------------
def test_missing_resources_are_404(client):
    assert client.get("/api/projects/nope").status_code == 404
    assert client.get("/api/segments/nope").status_code == 404
    assert client.patch("/api/glossary/999", json={"target_term": "x"}).status_code == 404


def test_empty_approval_is_rejected(client):
    response = client.post("/api/segments/p012-b1/approve", json={"translation": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_second_canonical_memory_entry_is_a_conflict(client):
    client.post("/api/segments/p012-b1/approve", json={"translation": "One rendering."})
    first = next(e for e in client.get("/api/projects/demo/memory").json()
                 if e["target_text"] == "One rendering.")
    assert client.post(f"/api/memory/{first['id']}/canonical").status_code == 200

    client.post("/api/segments/p012-b1/reopen")
    client.post("/api/segments/p012-b1/approve", json={"translation": "Another rendering."})
    second = next(e for e in client.get("/api/projects/demo/memory").json()
                  if e["target_text"] == "Another rendering.")
    assert client.post(f"/api/memory/{second['id']}/canonical").status_code == 409


def test_workbench_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Scanlate workbench" in response.text


def test_workbench_and_api_share_one_origin(client):
    """The page and its data come from the same process, so no CORS is needed."""
    page = client.get("/")
    assert page.headers["content-type"].startswith("text/html")
    assert client.get("/api/projects/demo/segments").status_code == 200

    # Every actual request the page makes is relative; the only absolute URL in
    # the file is inside a comment explaining the ?api= dev override.
    calls = [line.strip() for line in page.text.splitlines()
             if "fetch(" in line and not line.strip().startswith("//")]
    assert calls, "expected the page to make requests"
    for call in calls:
        assert "http://" not in call and "https://" not in call, call
    assert any("/api/projects/${PROJECT_ID}/segments" in c for c in calls)
    # No cross-origin configuration is present, because none is required.
    assert not any(type(m.cls).__name__ == "CORSMiddleware"
                   for m in client.app.user_middleware)


def test_served_file_matches_the_repository_copy(client):
    from scanlate.api.app import STATIC
    assert client.get("/").text == (STATIC / "index.html").read_text(encoding="utf-8")


def test_favicon_does_not_404(client):
    assert client.get("/favicon.ico").status_code == 204
