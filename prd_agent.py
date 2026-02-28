#!/usr/bin/env python3
"""
PRD Agent CLI — iterative PRD generator with human review at every stage.

Usage:
    python prd_agent.py --idea "Build a PRD generator agent"
    python prd_agent.py --idea-file ./idea.txt
    python prd_agent.py --session <id> --resume
    python prd_agent.py --session <id> --render-only
    python prd_agent.py --manual
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
PRD_OUTPUT = os.getenv("PRD_OUTPUT", "PRD.md")
SESSIONS_DIR = os.getenv("SESSIONS_DIR", ".prd_sessions")
EDITOR_CMD = os.getenv("EDITOR", "nano")

console = Console()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Status(str, Enum):
    DRAFT = "draft"
    FEATURES_READY = "features_ready"
    STORIES_READY = "stories_ready"
    DETAILED = "detailed"
    RENDERED = "rendered"


# ---------------------------------------------------------------------------
# Data models (Pydantic v2)
# ---------------------------------------------------------------------------

class BDDScenario(BaseModel):
    title: str
    given: list[str]
    when: list[str]
    then: list[str]


class Detail(BaseModel):
    mermaid_flow: str = ""
    bdd_scenarios: list[BDDScenario] = Field(default_factory=list)
    functional_reqs: list[str] = Field(default_factory=list)
    non_functional_reqs: list[str] = Field(default_factory=list)


class UserStory(BaseModel):
    id: str = ""
    feature_id: str = ""
    idx: int = 0
    as_a: str = ""
    i_want: str = ""
    so_that: str = ""
    detail: Detail | None = None


class Feature(BaseModel):
    id: str = ""
    session_id: str = ""
    idx: int = 0
    title: str = ""
    description: str = ""
    stories: list[UserStory] = Field(default_factory=list)


class SessionState(BaseModel):
    id: str = ""
    title: str = ""
    idea: str = ""
    status: Status = Status.DRAFT
    features: list[Feature] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


# ---------------------------------------------------------------------------
# LLM response schemas (for Structured Outputs)
# ---------------------------------------------------------------------------

class LLMFeatureItem(BaseModel):
    idx: int
    title: str
    description: str


class LLMFeaturesResponse(BaseModel):
    title: str
    features: list[LLMFeatureItem]


class LLMStoryItem(BaseModel):
    idx: int
    as_a: str
    i_want: str
    so_that: str


class LLMStoriesResponse(BaseModel):
    stories: list[LLMStoryItem]


class LLMDetailResponse(BaseModel):
    mermaid_flow: str
    bdd_scenarios: list[BDDScenario]
    functional_reqs: list[str]
    non_functional_reqs: list[str]


# ---------------------------------------------------------------------------
# JSON Schema helpers for Structured Outputs
# ---------------------------------------------------------------------------

FEATURES_JSON_SCHEMA = {
    "name": "features_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "idx": {"type": "integer"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["idx", "title", "description"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["title", "features"],
        "additionalProperties": False,
    },
}

STORIES_JSON_SCHEMA = {
    "name": "stories_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "stories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "idx": {"type": "integer"},
                        "as_a": {"type": "string"},
                        "i_want": {"type": "string"},
                        "so_that": {"type": "string"},
                    },
                    "required": ["idx", "as_a", "i_want", "so_that"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["stories"],
        "additionalProperties": False,
    },
}

DETAIL_JSON_SCHEMA = {
    "name": "detail_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "mermaid_flow": {"type": "string"},
            "bdd_scenarios": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "given": {"type": "array", "items": {"type": "string"}},
                        "when": {"type": "array", "items": {"type": "string"}},
                        "then": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title", "given", "when", "then"],
                    "additionalProperties": False,
                },
            },
            "functional_reqs": {"type": "array", "items": {"type": "string"}},
            "non_functional_reqs": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "mermaid_flow",
            "bdd_scenarios",
            "functional_reqs",
            "non_functional_reqs",
        ],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

def sessions_dir() -> Path:
    d = Path(SESSIONS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session(state: SessionState) -> Path:
    state.updated_at = datetime.now(timezone.utc).isoformat()
    p = sessions_dir() / f"{state.id}.json"
    p.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_session(session_id: str) -> SessionState:
    p = sessions_dir() / f"{session_id}.json"
    if not p.exists():
        console.print(f"[red]Session file not found: {p}[/red]")
        sys.exit(2)
    return SessionState.model_validate_json(p.read_text(encoding="utf-8"))


def new_session(idea: str) -> SessionState:
    sid = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    state = SessionState(
        id=sid,
        idea=idea,
        status=Status.DRAFT,
        created_at=now,
        updated_at=now,
    )
    save_session(state)
    console.print(f"[green]Created session:[/green] {sid}")
    return state


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _get_client():
    """Return an openai client. Imported lazily so --manual mode works without key."""
    import openai
    if not OPENAI_API_KEY:
        console.print("[red]Missing OPENAI_API_KEY. Set it in .env or environment.[/red]")
        sys.exit(2)
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    json_schema: dict[str, Any],
) -> dict[str, Any]:
    """Call OpenAI Responses API with Structured Outputs and return parsed JSON."""
    client = _get_client()
    t0 = time.time()

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            temperature=0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "json_schema": json_schema,
                },
            },
        )
    except AttributeError:
        # Fallback to chat completions if responses API not available
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
        )
        elapsed = time.time() - t0
        content = response.choices[0].message.content
        usage = response.usage
        console.print(
            f"  [dim]LLM responded in {elapsed:.1f}s | "
            f"tokens: {usage.prompt_tokens}+{usage.completion_tokens}[/dim]"
        )
        return json.loads(content)

    elapsed = time.time() - t0

    # Extract text from response
    raw_text = ""
    if hasattr(response, "output_text"):
        raw_text = response.output_text
    elif hasattr(response, "output") and response.output:
        for item in response.output:
            if hasattr(item, "content"):
                for block in item.content:
                    if hasattr(block, "text"):
                        raw_text = block.text
                        break

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", "?") if usage else "?"
    output_tokens = getattr(usage, "output_tokens", "?") if usage else "?"
    console.print(
        f"  [dim]LLM responded in {elapsed:.1f}s | "
        f"tokens: {input_tokens}+{output_tokens}[/dim]"
    )

    return json.loads(raw_text)


def generate_features(idea: str) -> LLMFeaturesResponse:
    system = (
        "You are a product manager AI. Given a product idea, produce a list of "
        "5 to 12 high-level features. Each feature has an idx (1-based), title, "
        "and description. Also produce a short product title."
    )
    user = f"Product idea:\n{idea}"
    data = _call_llm(system, user, FEATURES_JSON_SCHEMA)
    return LLMFeaturesResponse.model_validate(data)


def generate_stories(idea: str, feature: Feature) -> LLMStoriesResponse:
    system = (
        "You are a product manager AI. Given a product idea and ONE feature, "
        "produce 3 to 8 user stories in 'As a / I want / So that' format. "
        "Each story has an idx (1-based)."
    )
    user = (
        f"Product idea:\n{idea}\n\n"
        f"Feature (F-{feature.idx}): {feature.title}\n"
        f"Description: {feature.description}"
    )
    data = _call_llm(system, user, STORIES_JSON_SCHEMA)
    return LLMStoriesResponse.model_validate(data)


def generate_detail(idea: str, feature: Feature, story: UserStory) -> LLMDetailResponse:
    system = (
        "You are a product manager AI. Given a product idea, one feature, and one "
        "user story, produce:\n"
        "1. A Mermaid flowchart (flowchart TD, plain text without code fences).\n"
        "2. 2-5 BDD scenarios (title, given[], when[], then[]).\n"
        "3. 5-15 functional requirements (short sentences).\n"
        "4. 3-10 non-functional requirements (short sentences)."
    )
    user = (
        f"Product idea:\n{idea}\n\n"
        f"Feature (F-{feature.idx}): {feature.title}\n"
        f"Description: {feature.description}\n\n"
        f"User story ({story.id}):\n"
        f"  As a {story.as_a}\n"
        f"  I want {story.i_want}\n"
        f"  So that {story.so_that}"
    )
    data = _call_llm(system, user, DETAIL_JSON_SCHEMA)
    return LLMDetailResponse.model_validate(data)


# ---------------------------------------------------------------------------
# Review helpers
# ---------------------------------------------------------------------------

class ReviewMode(str, Enum):
    EDITOR = "editor"
    PROMPT = "prompt"


def _open_in_editor(text: str) -> str:
    """Open text in $EDITOR and return the edited result."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        tmp_path = f.name
    try:
        subprocess.run([EDITOR_CMD, tmp_path], check=True)
        return Path(tmp_path).read_text(encoding="utf-8")
    finally:
        os.unlink(tmp_path)


def _prompt_review(label: str, text: str, review_mode: ReviewMode) -> tuple[str, str]:
    """
    Show content to user and ask for review action.
    Returns (action, possibly_edited_text).
    Actions: 'accept', 'edit', 'regenerate'.
    """
    console.print(Panel(text, title=label, border_style="cyan"))
    console.print(
        "\n[bold]Review options:[/bold]  "
        "[green](a)ccept[/green]  |  "
        "[yellow](e)dit[/yellow]  |  "
        "[red](r)egenerate[/red]"
    )
    while True:
        choice = console.input("[bold]Your choice [a/e/r]: [/bold]").strip().lower()
        if choice in ("a", "accept"):
            return "accept", text
        elif choice in ("e", "edit"):
            if review_mode == ReviewMode.EDITOR:
                edited = _open_in_editor(text)
            else:
                console.print("[yellow]Paste edited JSON below (end with an empty line):[/yellow]")
                lines: list[str] = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                edited = "\n".join(lines)
            return "edit", edited
        elif choice in ("r", "regenerate"):
            return "regenerate", text
        else:
            console.print("[red]Invalid choice. Enter a, e, or r.[/red]")


# ---------------------------------------------------------------------------
# Stage: Features
# ---------------------------------------------------------------------------

def _features_to_display_json(features: list[Feature]) -> str:
    return json.dumps(
        [{"idx": f.idx, "title": f.title, "description": f.description} for f in features],
        indent=2,
    )


def stage_features(
    state: SessionState, review_mode: ReviewMode, manual: bool
) -> SessionState:
    console.rule("[bold blue]Stage 1: Generate Features[/bold blue]")

    while True:
        if manual:
            console.print("[yellow]Manual mode: enter features JSON.[/yellow]")
            console.print('Format: [{"idx":1,"title":"...","description":"..."},...]')
            raw = console.input("JSON> ")
            items = json.loads(raw)
            llm_resp = LLMFeaturesResponse(
                title=console.input("Product title> "),
                features=[LLMFeatureItem(**i) for i in items],
            )
        else:
            console.print("Generating features from idea...")
            llm_resp = generate_features(state.idea)

        state.title = llm_resp.title
        features = [
            Feature(
                id=f"F-{fi.idx}",
                session_id=state.id,
                idx=fi.idx,
                title=fi.title,
                description=fi.description,
            )
            for fi in llm_resp.features
        ]

        display_json = _features_to_display_json(features)
        console.print(f"\n[bold]Product title:[/bold] {state.title}\n")
        action, edited = _prompt_review("Features", display_json, review_mode)

        if action == "accept":
            # Keep existing stories if any features match
            state.features = features
            state.status = Status.FEATURES_READY
            save_session(state)
            console.print("[green]Features accepted.[/green]")
            return state
        elif action == "edit":
            try:
                items = json.loads(edited)
                features = [
                    Feature(
                        id=f"F-{i['idx']}",
                        session_id=state.id,
                        idx=i["idx"],
                        title=i["title"],
                        description=i["description"],
                    )
                    for i in items
                ]
                state.features = features
                state.status = Status.FEATURES_READY
                save_session(state)
                console.print("[green]Features saved (edited).[/green]")
                return state
            except (json.JSONDecodeError, KeyError) as exc:
                console.print(f"[red]Invalid JSON: {exc}. Try again.[/red]")
                continue
        else:  # regenerate
            console.print("[yellow]Regenerating features...[/yellow]")
            continue


# ---------------------------------------------------------------------------
# Stage: Stories (per feature)
# ---------------------------------------------------------------------------

def _stories_to_display_json(stories: list[UserStory]) -> str:
    return json.dumps(
        [
            {"idx": s.idx, "as_a": s.as_a, "i_want": s.i_want, "so_that": s.so_that}
            for s in stories
        ],
        indent=2,
    )


def stage_stories(
    state: SessionState, review_mode: ReviewMode, manual: bool
) -> SessionState:
    console.rule("[bold blue]Stage 2: Generate User Stories[/bold blue]")

    for feature in state.features:
        if feature.stories:
            console.print(
                f"[dim]Feature {feature.id} already has stories, skipping.[/dim]"
            )
            continue

        console.print(f"\n[bold]Feature {feature.id}: {feature.title}[/bold]")

        while True:
            if manual:
                console.print("[yellow]Manual mode: enter stories JSON.[/yellow]")
                raw = console.input("JSON> ")
                items = json.loads(raw)
                llm_resp = LLMStoriesResponse(
                    stories=[LLMStoryItem(**i) for i in items]
                )
            else:
                console.print(f"  Generating stories for {feature.id}...")
                llm_resp = generate_stories(state.idea, feature)

            stories = [
                UserStory(
                    id=f"US-{feature.idx}.{si.idx}",
                    feature_id=feature.id,
                    idx=si.idx,
                    as_a=si.as_a,
                    i_want=si.i_want,
                    so_that=si.so_that,
                )
                for si in llm_resp.stories
            ]

            display_json = _stories_to_display_json(stories)
            action, edited = _prompt_review(
                f"Stories for {feature.id}", display_json, review_mode
            )

            if action == "accept":
                feature.stories = stories
                save_session(state)
                console.print(f"[green]Stories for {feature.id} accepted.[/green]")
                break
            elif action == "edit":
                try:
                    items = json.loads(edited)
                    feature.stories = [
                        UserStory(
                            id=f"US-{feature.idx}.{i['idx']}",
                            feature_id=feature.id,
                            idx=i["idx"],
                            as_a=i["as_a"],
                            i_want=i["i_want"],
                            so_that=i["so_that"],
                        )
                        for i in items
                    ]
                    save_session(state)
                    console.print(
                        f"[green]Stories for {feature.id} saved (edited).[/green]"
                    )
                    break
                except (json.JSONDecodeError, KeyError) as exc:
                    console.print(f"[red]Invalid JSON: {exc}. Try again.[/red]")
                    continue
            else:  # regenerate
                console.print("[yellow]Regenerating stories...[/yellow]")
                continue

    state.status = Status.STORIES_READY
    save_session(state)
    return state


# ---------------------------------------------------------------------------
# Stage: Details (per story)
# ---------------------------------------------------------------------------

def _detail_to_display_json(detail: Detail) -> str:
    return json.dumps(detail.model_dump(), indent=2)


def stage_details(
    state: SessionState, review_mode: ReviewMode, manual: bool
) -> SessionState:
    console.rule("[bold blue]Stage 3: Generate Story Details[/bold blue]")

    for feature in state.features:
        for story in feature.stories:
            if story.detail is not None:
                console.print(
                    f"[dim]{story.id} already has details, skipping.[/dim]"
                )
                continue

            console.print(f"\n[bold]{story.id}: As a {story.as_a}, "
                          f"I want {story.i_want}[/bold]")

            while True:
                if manual:
                    console.print("[yellow]Manual mode: enter detail JSON.[/yellow]")
                    raw = console.input("JSON> ")
                    detail = Detail.model_validate_json(raw)
                else:
                    console.print(f"  Generating details for {story.id}...")
                    llm_resp = generate_detail(state.idea, feature, story)
                    detail = Detail(
                        mermaid_flow=llm_resp.mermaid_flow,
                        bdd_scenarios=llm_resp.bdd_scenarios,
                        functional_reqs=llm_resp.functional_reqs,
                        non_functional_reqs=llm_resp.non_functional_reqs,
                    )

                display_json = _detail_to_display_json(detail)
                action, edited = _prompt_review(
                    f"Details for {story.id}", display_json, review_mode
                )

                if action == "accept":
                    story.detail = detail
                    save_session(state)
                    console.print(f"[green]Details for {story.id} accepted.[/green]")
                    break
                elif action == "edit":
                    try:
                        detail = Detail.model_validate_json(edited)
                        story.detail = detail
                        save_session(state)
                        console.print(
                            f"[green]Details for {story.id} saved (edited).[/green]"
                        )
                        break
                    except Exception as exc:
                        console.print(f"[red]Invalid JSON: {exc}. Try again.[/red]")
                        continue
                else:  # regenerate
                    console.print("[yellow]Regenerating details...[/yellow]")
                    continue

    state.status = Status.DETAILED
    save_session(state)
    return state


# ---------------------------------------------------------------------------
# Render PRD.md
# ---------------------------------------------------------------------------

def render_prd(state: SessionState) -> str:
    lines: list[str] = []

    lines.append(f"# {state.title or 'Untitled PRD'}")
    lines.append("")

    # Idea
    lines.append("## Idea / Problem Statement")
    lines.append("")
    lines.append(state.idea)
    lines.append("")

    # Feature list
    lines.append("## Feature List")
    lines.append("")
    for f in state.features:
        lines.append(f"- **{f.id} — {f.title}**: {f.description}")
    lines.append("")

    # Stories + details by feature
    for f in state.features:
        lines.append(f"## {f.id} — {f.title}")
        lines.append("")
        lines.append(f.description)
        lines.append("")

        for s in f.stories:
            lines.append(f"### {s.id}: User Story")
            lines.append("")
            lines.append(f"- **As a** {s.as_a}")
            lines.append(f"- **I want** {s.i_want}")
            lines.append(f"- **So that** {s.so_that}")
            lines.append("")

            if s.detail is None:
                lines.append("_Details not yet generated._")
                lines.append("")
                continue

            d = s.detail

            # Mermaid flow
            if d.mermaid_flow:
                lines.append("#### Flow")
                lines.append("")
                lines.append("```mermaid")
                lines.append(d.mermaid_flow)
                lines.append("```")
                lines.append("")

            # BDD scenarios
            if d.bdd_scenarios:
                lines.append("#### BDD Scenarios")
                lines.append("")
                lines.append("```gherkin")
                lines.append(f"Feature: {f.title}")
                for sc in d.bdd_scenarios:
                    lines.append(f"  Scenario: {sc.title}")
                    for g in sc.given:
                        lines.append(f"    Given {g}")
                    for w in sc.when:
                        lines.append(f"    When {w}")
                    for t in sc.then:
                        lines.append(f"    Then {t}")
                    lines.append("")
                lines.append("```")
                lines.append("")

            # Functional requirements
            if d.functional_reqs:
                lines.append("#### Functional Requirements")
                lines.append("")
                for i, fr in enumerate(d.functional_reqs, 1):
                    lines.append(f"- **FR-{i}**: {fr}")
                lines.append("")

            # Non-functional requirements
            if d.non_functional_reqs:
                lines.append("#### Non-Functional Requirements")
                lines.append("")
                for i, nfr in enumerate(d.non_functional_reqs, 1):
                    lines.append(f"- **NFR-{i}**: {nfr}")
                lines.append("")

    # Out of scope
    lines.append("## Out of Scope / Assumptions")
    lines.append("")
    lines.append("_To be defined._")
    lines.append("")

    # Open questions
    lines.append("## Open Questions")
    lines.append("")
    lines.append("_To be defined._")
    lines.append("")

    return "\n".join(lines)


def stage_render(state: SessionState, review_mode: ReviewMode) -> SessionState:
    console.rule("[bold blue]Stage 4: Render PRD[/bold blue]")

    prd_text = render_prd(state)
    output_path = Path(PRD_OUTPUT)
    output_path.write_text(prd_text, encoding="utf-8")
    console.print(f"[green]PRD written to {output_path.resolve()}[/green]\n")

    # Show preview (first 80 lines)
    preview_lines = prd_text.split("\n")[:80]
    console.print(Panel("\n".join(preview_lines), title="PRD Preview (first 80 lines)"))

    if len(prd_text.split("\n")) > 80:
        console.print(f"[dim]... ({len(prd_text.split(chr(10)))} total lines)[/dim]")

    console.print("")
    answer = console.input("[bold]Mark session as rendered? (y/n): [/bold]").strip().lower()
    if answer in ("y", "yes"):
        state.status = Status.RENDERED
        save_session(state)
        console.print("[green]Session marked as rendered.[/green]")
    else:
        console.print("[yellow]Session NOT marked as rendered. You can re-run --render-only later.[/yellow]")
        save_session(state)

    return state


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    state: SessionState,
    review_mode: ReviewMode,
    manual: bool,
    render_only: bool = False,
) -> None:
    start_time = datetime.now(timezone.utc)
    console.print(f"[bold]Session:[/bold] {state.id}")
    console.print(f"[bold]Status:[/bold]  {state.status.value}")
    console.print(f"[bold]Start:[/bold]   {start_time.isoformat()}")
    console.print("")

    if render_only:
        if state.status.value not in (Status.DETAILED.value, Status.RENDERED.value):
            console.print(
                "[red]Cannot render: session is not in 'detailed' or 'rendered' state.[/red]"
            )
            sys.exit(2)
        stage_render(state, review_mode)
        end_time = datetime.now(timezone.utc)
        console.print(f"\n[bold]End:[/bold] {end_time.isoformat()}")
        return

    # Step through stages based on current status
    if state.status == Status.DRAFT:
        state = stage_features(state, review_mode, manual)

    if state.status == Status.FEATURES_READY:
        state = stage_stories(state, review_mode, manual)

    if state.status == Status.STORIES_READY:
        state = stage_details(state, review_mode, manual)

    if state.status == Status.DETAILED:
        state = stage_render(state, review_mode)

    end_time = datetime.now(timezone.utc)
    console.print(f"\n[bold]End:[/bold] {end_time.isoformat()}")

    if state.status == Status.RENDERED:
        console.print("[bold green]Pipeline complete![/bold green]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRD Agent — iterative PRD generator with human review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python prd_agent.py --idea 'Build a PRD generator agent'\n"
            "  python prd_agent.py --idea-file ./idea.txt\n"
            "  python prd_agent.py --session abc123 --resume\n"
            "  python prd_agent.py --session abc123 --render-only\n"
        ),
    )

    idea_group = parser.add_mutually_exclusive_group()
    idea_group.add_argument("--idea", type=str, help="Product idea as text")
    idea_group.add_argument("--idea-file", type=str, help="Path to file containing the idea")

    parser.add_argument("--session", type=str, help="Session ID to resume or render")
    parser.add_argument(
        "--resume", action="store_true", help="Resume an existing session"
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Only render PRD.md from existing session data",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Skip LLM calls; enter data manually (for debugging)",
    )
    parser.add_argument(
        "--review",
        type=str,
        choices=["editor", "prompt"],
        default="editor",
        help="Review mode: editor (default) or prompt",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    review_mode = ReviewMode(args.review)

    # Determine session
    if args.resume or args.render_only:
        if not args.session:
            console.print("[red]--session is required with --resume or --render-only[/red]")
            sys.exit(2)
        state = load_session(args.session)
        console.print(f"[green]Loaded session:[/green] {state.id} (status: {state.status.value})")
    elif args.idea or args.idea_file:
        if args.idea_file:
            p = Path(args.idea_file)
            if not p.exists():
                console.print(f"[red]Idea file not found: {p}[/red]")
                sys.exit(2)
            idea = p.read_text(encoding="utf-8").strip()
        else:
            idea = (args.idea or "").strip()

        if not idea:
            console.print("[red]Idea is empty.[/red]")
            sys.exit(2)

        state = new_session(idea)
    elif args.manual:
        idea = console.input("[bold]Enter your product idea: [/bold]").strip()
        if not idea:
            console.print("[red]Idea is empty.[/red]")
            sys.exit(2)
        state = new_session(idea)
    else:
        console.print("[red]Provide --idea, --idea-file, --session --resume, or --manual.[/red]")
        sys.exit(2)

    run_pipeline(
        state,
        review_mode=review_mode,
        manual=args.manual,
        render_only=args.render_only,
    )


if __name__ == "__main__":
    main()
