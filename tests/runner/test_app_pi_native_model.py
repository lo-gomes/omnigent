"""Tests for pi-native model resolution from the agent spec.

``_pi_native_model_from_spec`` is the seam that turns a session's
``executor.model`` (set via a config.yaml ``model:`` key) into the model
threaded into ``resolve_pi_native_provider(model=...)`` — which renders it
into the runner-owned Pi ``models.json`` (and the appended ``--model``).

Unlike cursor-native, a gateway-routed id (``databricks-*``) is KEPT: the
runner-owned Pi process routes through the Databricks AI Gateway, whose
``models.json`` selects the model by its gateway id.

When no Omnigent provider is configured (Pi uses its own login, e.g.
pi-cursor-sdk), ``--model`` must still be appended from the session
``model_override``. Session metadata alone is not evidence of selection.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import httpx
import pytest

from omnigent.entities.session_resources import SessionResourceView
from omnigent.runner.app import (
    ResolvedSpec,
    _append_pi_native_requested_model,
    _auto_create_pi_terminal,
    _pi_args_have_model,
    _pi_native_model_from_launch_args,
    _pi_native_model_from_spec,
)
from omnigent.spec.types import AgentSpec, ExecutorSpec


def _spec(model: str | None) -> AgentSpec:
    """Build a minimal agent spec carrying *model* on its executor block."""
    return AgentSpec(spec_version=1, name="pi", executor=ExecutorSpec(model=model))


def test_pi_native_model_passthrough() -> None:
    """A pinned model id is returned verbatim."""
    assert _pi_native_model_from_spec(_spec("databricks-claude-opus-4-7")) == (
        "databricks-claude-opus-4-7"
    )


def test_pi_native_model_keeps_gateway_id() -> None:
    """Gateway-routed ids are usable here (Pi routes through the gateway)."""
    assert _pi_native_model_from_spec(_spec("databricks-claude-sonnet-4-6")) == (
        "databricks-claude-sonnet-4-6"
    )
    assert _pi_native_model_from_spec(_spec("openai/gpt-4o")) == "openai/gpt-4o"


def test_pi_native_model_no_pin_returns_none() -> None:
    """No model declared → None (Pi keeps the provider's default model)."""
    assert _pi_native_model_from_spec(_spec(None)) is None
    assert _pi_native_model_from_spec(_spec("")) is None


def test_pi_native_model_none_spec() -> None:
    """A missing spec yields no model override."""
    assert _pi_native_model_from_spec(None) is None


def test_pi_native_model_from_resolved_spec_wrapper() -> None:
    """The model is read through a ``ResolvedSpec`` wrapper too."""
    wrapped = ResolvedSpec(spec=_spec("databricks-claude-opus-4-7"), workdir=Path("/tmp"))
    assert _pi_native_model_from_spec(wrapped) == "databricks-claude-opus-4-7"


def _key_provider_config() -> dict[str, Any]:
    """A key-kind anthropic provider config (Pi's native surface)."""
    return {
        "providers": {
            "anthropic": {
                "kind": "key",
                "default": True,
                "anthropic": {
                    "base_url": "https://api.anthropic.com",
                    "api_key": "sk-test-literal",
                    "models": {"default": "claude-sonnet-4-6"},
                },
            }
        }
    }


@pytest.mark.asyncio
async def test_auto_create_pi_terminal_threads_spec_model_into_models_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the spec's ``executor.model`` reaches the generated models.json.

    Drives ``_auto_create_pi_terminal`` with a spec pinning
    ``claude-opus-4-7`` and a key-kind provider whose family default is
    ``claude-sonnet-4-6``. The threaded override must win: the generated
    ``models.json`` selects ``claude-opus-4-7`` and the appended Pi
    ``--model`` arg reflects it. This is the runner-side seam the feature
    adds — without threading the spec model, the models.json would carry
    the family default instead.

    :param tmp_path: Temp dir backing the pi-native bridge root.
    :param monkeypatch: Pytest monkeypatch fixture.
    :returns: None.
    """
    import omnigent.pi_native_bridge as pi_bridge
    import omnigent.pi_native_credentials as creds

    session_id = "conv_pi_model_e2e"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Redirect the bridge tree into tmp so the generated managed Pi config dir
    # (and its models.json) lands somewhere isolated and inspectable.
    monkeypatch.setattr(pi_bridge, "_BRIDGE_ROOT", tmp_path / "pi-native")
    monkeypatch.setenv("OMNIGENT_RUNNER_WORKSPACE", str(workspace))
    monkeypatch.setenv("RUNNER_SERVER_URL", "http://ap.example")
    monkeypatch.setattr("omnigent.runner._entry._make_auth_token_factory", lambda: None)
    # Resolve a Pi executable without requiring the real binary on PATH.
    monkeypatch.setattr("omnigent.pi_native.resolve_pi_executable", lambda: "/usr/bin/pi")

    # ``resolve_pi_native_provider``'s default config_loader is bound at def
    # time, so inject the test config by patching the module symbol the runner
    # imports locally — recording the ``model`` kwarg it is called with.
    real_resolve = creds.resolve_pi_native_provider
    captured: dict[str, Any] = {}

    def _resolve_with_test_config(*, model: str | None = None, config_loader: Any = None):
        captured["model"] = model
        return real_resolve(model=model, config_loader=_key_provider_config)

    monkeypatch.setattr(creds, "resolve_pi_native_provider", _resolve_with_test_config)

    class _SnapshotClient:
        """Fresh pi-native session snapshot (no launch args / external id)."""

        async def get(self, url: str, *, timeout: float) -> httpx.Response:
            del url, timeout
            return httpx.Response(
                200,
                json={
                    "workspace": str(workspace),
                    "terminal_launch_args": None,
                    "external_session_id": None,
                },
                request=httpx.Request("GET", f"/v1/sessions/{session_id}"),
            )

    launched: dict[str, Any] = {}

    class _FakeResourceRegistry:
        """Captures the launched terminal spec (args + env)."""

        terminal_registry = None

        async def launch_required_terminal(
            self,
            session_id: str,
            terminal_name: str,
            session_key: str,
            spec: Any,
            *,
            resource_role: str | None = None,
            parent_os_env: Any = None,
        ) -> SessionResourceView:
            del terminal_name, session_key, resource_role, parent_os_env
            launched["args"] = list(spec.args)
            launched["env"] = dict(spec.env)
            return SessionResourceView(
                id="terminal_pi_main",
                type="terminal",
                session_id=session_id,
                name="pi",
            )

    spec = AgentSpec(
        spec_version=1,
        name="pi-model-e2e",
        executor=ExecutorSpec(
            type="omnigent",
            config={"harness": "pi-native"},
            model="claude-opus-4-7",
        ),
    )

    await _auto_create_pi_terminal(
        session_id,
        _FakeResourceRegistry(),  # type: ignore[arg-type]
        lambda _sid, _event: None,
        server_client=_SnapshotClient(),  # type: ignore[arg-type]
        agent_spec=spec,
    )

    # The runner threaded the spec model into resolve_pi_native_provider.
    assert captured["model"] == "claude-opus-4-7"

    # The appended Pi args select the override, not the family default.
    args = launched["args"]
    assert "--model" in args
    assert args[args.index("--model") + 1] == "claude-opus-4-7"
    assert args.count("--model") == 1
    assert "--provider" in args

    # The managed config dir env was set and its models.json selects the override.
    agent_dir = Path(launched["env"]["PI_CODING_AGENT_DIR"])
    models = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))
    assert models["providers"]["omnigent"]["models"] == [{"id": "claude-opus-4-7"}]


@pytest.mark.asyncio
async def test_auto_create_pi_terminal_no_spec_model_uses_provider_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no spec model, the provider's family default is used (unchanged).

    Guards the ``None`` case: a spec without ``executor.model`` must leave
    Pi on the provider default (``claude-sonnet-4-6`` here), not break the
    launch.

    :param tmp_path: Temp dir backing the pi-native bridge root.
    :param monkeypatch: Pytest monkeypatch fixture.
    :returns: None.
    """
    import omnigent.pi_native_bridge as pi_bridge
    import omnigent.pi_native_credentials as creds

    session_id = "conv_pi_model_default"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(pi_bridge, "_BRIDGE_ROOT", tmp_path / "pi-native")
    monkeypatch.setenv("OMNIGENT_RUNNER_WORKSPACE", str(workspace))
    monkeypatch.setenv("RUNNER_SERVER_URL", "http://ap.example")
    monkeypatch.setattr("omnigent.runner._entry._make_auth_token_factory", lambda: None)
    monkeypatch.setattr("omnigent.pi_native.resolve_pi_executable", lambda: "/usr/bin/pi")

    real_resolve = creds.resolve_pi_native_provider
    captured: dict[str, Any] = {}

    def _resolve_with_test_config(*, model: str | None = None, config_loader: Any = None):
        captured["model"] = model
        return real_resolve(model=model, config_loader=_key_provider_config)

    monkeypatch.setattr(creds, "resolve_pi_native_provider", _resolve_with_test_config)

    class _SnapshotClient:
        async def get(self, url: str, *, timeout: float) -> httpx.Response:
            del url, timeout
            return httpx.Response(
                200,
                json={
                    "workspace": str(workspace),
                    "terminal_launch_args": None,
                    "external_session_id": None,
                },
                request=httpx.Request("GET", f"/v1/sessions/{session_id}"),
            )

    launched: dict[str, Any] = {}

    class _FakeResourceRegistry:
        terminal_registry = None

        async def launch_required_terminal(
            self,
            session_id: str,
            terminal_name: str,
            session_key: str,
            spec: Any,
            *,
            resource_role: str | None = None,
            parent_os_env: Any = None,
        ) -> SessionResourceView:
            del terminal_name, session_key, resource_role, parent_os_env
            launched["args"] = list(spec.args)
            launched["env"] = dict(spec.env)
            return SessionResourceView(
                id="terminal_pi_main",
                type="terminal",
                session_id=session_id,
                name="pi",
            )

    spec = AgentSpec(
        spec_version=1,
        name="pi-default",
        executor=ExecutorSpec(type="omnigent", config={"harness": "pi-native"}),
    )

    await _auto_create_pi_terminal(
        session_id,
        _FakeResourceRegistry(),  # type: ignore[arg-type]
        lambda _sid, _event: None,
        server_client=_SnapshotClient(),  # type: ignore[arg-type]
        agent_spec=spec,
    )

    assert captured["model"] is None
    agent_dir = Path(launched["env"]["PI_CODING_AGENT_DIR"])
    models = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))
    assert models["providers"]["omnigent"]["models"] == [{"id": "claude-sonnet-4-6"}]


# Opaque Pi + pi-cursor-sdk identifiers. Pi's ``--model`` flag accepts
# ``provider/id`` plus an optional ``:<thinking>`` suffix; these must reach
# the launched argv verbatim, not only Omnigent session metadata.
_OPAQUE_PI_CURSOR_MODELS = (
    "cursor/grok-4.5:slow",
    "cursor/grok-4.6:slow",
    "cursor/composer-2-5:slow",
)


def _model_from_launch_args(args: list[str]) -> str | None:
    """Return the model id on Pi's launch argv — execution evidence, not metadata."""
    return _pi_native_model_from_launch_args(args)


class _LaunchCapture:
    """Captures the Pi terminal spec (args + env) from a fake registry."""

    def __init__(self) -> None:
        self.args: list[str] = []
        self.env: dict[str, str] = {}

    class _Registry:
        terminal_registry = None

        def __init__(self, owner: _LaunchCapture) -> None:
            self._owner = owner

        async def launch_required_terminal(
            self,
            session_id: str,
            terminal_name: str,
            session_key: str,
            spec: Any,
            *,
            resource_role: str | None = None,
            parent_os_env: Any = None,
        ) -> SessionResourceView:
            del terminal_name, session_key, resource_role, parent_os_env
            self._owner.args = list(spec.args)
            self._owner.env = dict(spec.env)
            return SessionResourceView(
                id="terminal_pi_main",
                type="terminal",
                session_id=session_id,
                name="pi",
            )

    def registry(self) -> Any:
        return self._Registry(self)


def _snapshot_client(
    session_id: str,
    workspace: Path,
    *,
    model_override: str | None = None,
    terminal_launch_args: list[str] | None = None,
) -> Any:
    """HTTP client stub that returns a Pi launch snapshot."""

    payload: dict[str, Any] = {
        "workspace": str(workspace),
        "terminal_launch_args": terminal_launch_args,
        "external_session_id": None,
        "model_override": model_override,
    }

    class _SnapshotClient:
        async def get(self, url: str, *, timeout: float) -> httpx.Response:
            del url, timeout
            return httpx.Response(
                200,
                json=payload,
                request=httpx.Request("GET", f"/v1/sessions/{session_id}"),
            )

    return _SnapshotClient()


async def _launch_pi_without_omnigent_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_id: str,
    model_override: str | None,
    terminal_launch_args: list[str] | None = None,
    spec_model: str | None = None,
) -> _LaunchCapture:
    """Drive ``_auto_create_pi_terminal`` with Pi's own login (no Omnigent provider)."""
    import omnigent.pi_native_bridge as pi_bridge
    import omnigent.pi_native_credentials as creds

    workspace = tmp_path / f"workspace-{session_id}"
    workspace.mkdir()
    monkeypatch.setattr(pi_bridge, "_BRIDGE_ROOT", tmp_path / "pi-native")
    monkeypatch.setenv("OMNIGENT_RUNNER_WORKSPACE", str(workspace))
    monkeypatch.setenv("RUNNER_SERVER_URL", "http://ap.example")
    monkeypatch.setattr("omnigent.runner._entry._make_auth_token_factory", lambda: None)
    monkeypatch.setattr("omnigent.pi_native.resolve_pi_executable", lambda: "/usr/bin/pi")
    monkeypatch.setattr(creds, "resolve_pi_native_provider", lambda **_kwargs: None)

    launched = _LaunchCapture()
    spec = AgentSpec(
        spec_version=1,
        name="pi-opaque-model",
        executor=ExecutorSpec(
            type="omnigent",
            config={"harness": "pi-native"},
            model=spec_model,
        ),
    )
    await _auto_create_pi_terminal(
        session_id,
        launched.registry(),
        lambda _sid, _event: None,
        server_client=_snapshot_client(
            session_id,
            workspace,
            model_override=model_override,
            terminal_launch_args=terminal_launch_args,
        ),
        agent_spec=spec,
    )
    return launched


@pytest.mark.parametrize("model", _OPAQUE_PI_CURSOR_MODELS)
@pytest.mark.asyncio
async def test_pi_native_launch_forwards_opaque_model_before_first_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    """A requested Pi/Cursor id must be on the launched argv, not only metadata.

    Reproduces the control-plane defect: Omnigent stored
    ``cursor/composer-2-5:slow`` on the session while the Pi process launched
    with no ``--model`` and ran Pi's default (``cursor/grok-4.5:slow``).
    ``resolve_pi_native_provider`` returning ``None`` is the pi-cursor-sdk
    path — Pi's own login, no Omnigent ``models.json``. Evidence is the
    actual argv, collected at terminal launch, before any user message.
    """
    launched = await _launch_pi_without_omnigent_provider(
        tmp_path,
        monkeypatch,
        session_id=f"conv_opaque_{model.replace('/', '_').replace(':', '_')}",
        model_override=model,
    )

    selected = _model_from_launch_args(launched.args)
    assert "--model" in launched.args or any(
        arg.startswith("--model=") for arg in launched.args
    ), launched.args
    assert selected == model, (
        f"execution evidence selected {selected!r}, requested {model!r}; argv={launched.args!r}"
    )
    # No user-message inbox payload is written at launch — the model is
    # baked into argv before the first prompt can run.
    assert "PI_CODING_AGENT_DIR" not in launched.env


@pytest.mark.asyncio
async def test_pi_native_launch_opaque_models_do_not_bleed_across_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two fresh jobs with different requested models cannot share argv state.

    Each session gets its own Pi process. The second launch's ``--model``
    must be its own requested id, not the first job's leftover.
    """
    first_model, second_model = _OPAQUE_PI_CURSOR_MODELS[0], _OPAQUE_PI_CURSOR_MODELS[2]

    first = await _launch_pi_without_omnigent_provider(
        tmp_path,
        monkeypatch,
        session_id="conv_opaque_job_a",
        model_override=first_model,
    )
    second = await _launch_pi_without_omnigent_provider(
        tmp_path,
        monkeypatch,
        session_id="conv_opaque_job_b",
        model_override=second_model,
    )

    first_selected = _model_from_launch_args(first.args)
    second_selected = _model_from_launch_args(second.args)
    assert first_selected == first_model, first.args
    assert second_selected == second_model, second.args
    assert first_selected != second_selected
    assert first.args is not second.args


@pytest.mark.asyncio
async def test_pi_native_launch_user_model_flag_wins_over_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``--model`` in pass-through args is not overwritten."""
    launched = await _launch_pi_without_omnigent_provider(
        tmp_path,
        monkeypatch,
        session_id="conv_opaque_user_model",
        model_override="cursor/composer-2-5:slow",
        terminal_launch_args=["--model", "cursor/grok-4.6:slow"],
    )

    assert _model_from_launch_args(launched.args) == "cursor/grok-4.6:slow"
    assert launched.args.count("--model") == 1


@pytest.mark.asyncio
async def test_pi_native_launch_joined_model_flag_is_not_duplicated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The joined ``--model=`` form also counts as an existing pin."""
    launched = await _launch_pi_without_omnigent_provider(
        tmp_path,
        monkeypatch,
        session_id="conv_opaque_joined_model",
        model_override="cursor/composer-2-5:slow",
        terminal_launch_args=["--model=cursor/grok-4.6:slow"],
    )

    assert _model_from_launch_args(launched.args) == "cursor/grok-4.6:slow"
    assert not any(arg == "--model" for arg in launched.args)


@pytest.mark.asyncio
async def test_pi_native_launch_spec_model_reaches_argv_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spec-pinned model is forwarded even when Pi uses its own login."""
    launched = await _launch_pi_without_omnigent_provider(
        tmp_path,
        monkeypatch,
        session_id="conv_opaque_spec_model",
        model_override=None,
        spec_model="cursor/grok-4.6:slow",
    )

    assert _model_from_launch_args(launched.args) == "cursor/grok-4.6:slow"


@pytest.mark.asyncio
async def test_pi_native_launch_override_beats_spec_model_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session ``model_override`` wins over ``executor.model`` on argv."""
    launched = await _launch_pi_without_omnigent_provider(
        tmp_path,
        monkeypatch,
        session_id="conv_opaque_override_wins",
        model_override="cursor/composer-2-5:slow",
        spec_model="cursor/grok-4.5:slow",
    )

    assert _model_from_launch_args(launched.args) == "cursor/composer-2-5:slow"


@pytest.mark.asyncio
async def test_pi_native_launch_without_requested_model_omits_model_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No override and no spec model → do not invent a ``--model`` flag."""
    launched = await _launch_pi_without_omnigent_provider(
        tmp_path,
        monkeypatch,
        session_id="conv_opaque_default",
        model_override=None,
    )

    assert _model_from_launch_args(launched.args) is None
    assert "--model" not in launched.args


def test_pi_args_have_model_detects_flag_forms() -> None:
    """Both ``--model X`` and ``--model=X`` count as an existing pin."""
    assert _pi_args_have_model(["--model", "cursor/grok-4.6:slow"]) is True
    assert _pi_args_have_model(["--model=cursor/grok-4.6:slow"]) is True
    assert _pi_args_have_model(["--extension", "x.js", "--session-dir", "/tmp"]) is False
    assert _pi_args_have_model([]) is False


def test_pi_native_model_from_launch_args_reads_execution_argv() -> None:
    """Evidence comes from argv, including joined flags and a last-wins overwrite."""
    assert _pi_native_model_from_launch_args([]) is None
    assert (
        _pi_native_model_from_launch_args(["--model", "cursor/composer-2-5:slow"])
        == "cursor/composer-2-5:slow"
    )
    assert (
        _pi_native_model_from_launch_args(["--model=cursor/grok-4.6:slow"])
        == "cursor/grok-4.6:slow"
    )
    # Empty joined value is not a selected model.
    assert _pi_native_model_from_launch_args(["--model="]) is None
    # A trailing ``--model`` with no value leaves the previous selection.
    assert (
        _pi_native_model_from_launch_args(["--model", "cursor/grok-4.5:slow", "--model"])
        == "cursor/grok-4.5:slow"
    )
    # Last explicit value wins so a later append is what Pi sees.
    assert (
        _pi_native_model_from_launch_args(
            ["--model", "cursor/grok-4.5:slow", "--model=cursor/composer-2-5:slow"]
        )
        == "cursor/composer-2-5:slow"
    )


def test_append_pi_native_requested_model_is_idempotent_and_isolated() -> None:
    """Append only when missing; two arg lists cannot leak a model into each other."""
    first: list[str] = ["--extension", "a.js"]
    second: list[str] = ["--extension", "b.js"]
    _append_pi_native_requested_model(first, "cursor/grok-4.5:slow")
    _append_pi_native_requested_model(second, "cursor/composer-2-5:slow")
    assert _pi_native_model_from_launch_args(first) == "cursor/grok-4.5:slow"
    assert _pi_native_model_from_launch_args(second) == "cursor/composer-2-5:slow"

    # None / empty leave argv unchanged (Pi keeps its default).
    untouched: list[str] = ["--extension", "c.js"]
    _append_pi_native_requested_model(untouched, None)
    _append_pi_native_requested_model(untouched, "")
    assert "--model" not in untouched

    # Existing ``--model`` / ``--model=`` are not overwritten or duplicated.
    pinned = ["--model", "cursor/grok-4.6:slow"]
    _append_pi_native_requested_model(pinned, "cursor/composer-2-5:slow")
    assert pinned == ["--model", "cursor/grok-4.6:slow"]
    joined = ["--model=cursor/grok-4.6:slow"]
    _append_pi_native_requested_model(joined, "cursor/composer-2-5:slow")
    assert joined == ["--model=cursor/grok-4.6:slow"]


def _dump_pi_selected_model(tmp_path: Path, requested: str) -> dict[str, Any]:
    """Launch real ``pi`` with *requested* and return ``ctx.model`` at session_start.

    Uses a throwaway ``PI_CODING_AGENT_DIR`` so this never reads or writes
    ``~/.pi/agent``. The dump extension exits before any user prompt, so
    the evidence is Pi's selected model, not Omnigent metadata.

    :param tmp_path: Pytest temp dir for this launch.
    :param requested: Opaque Pi model id, e.g. ``cursor/composer-2-5:slow``.
    :returns: JSON object with ``id``, ``provider``, ``thinkingLevel``.
    """
    pi_bin = shutil.which("pi")
    if pi_bin is None:
        pytest.skip("pi CLI is required for the live first-turn model dump")

    slug = requested.replace("/", "_").replace(":", "_")
    work = tmp_path / f"pi-live-{slug}"
    agent_dir = work / "agent"
    agent_dir.mkdir(parents=True, mode=0o700)
    (agent_dir / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "cursor": {
                        "baseUrl": "http://127.0.0.1:9/v1",
                        "api": "openai-completions",
                        "apiKey": "dummy",
                        "models": [
                            {"id": "grok-4.5:slow"},
                            {"id": "grok-4.6:slow"},
                            {"id": "composer-2-5:slow"},
                        ],
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out_path = work / "selected.json"
    dump_js = work / "dump.js"
    dump_js.write_text(
        textwrap.dedent(
            f"""\
            const fs = require("fs");
            const out = {json.dumps(str(out_path))};
            module.exports = function (pi) {{
              pi.on("session_start", async (_event, ctx) => {{
                const model = ctx && ctx.model ? ctx.model : {{}};
                fs.writeFileSync(out, JSON.stringify({{
                  id: model.id || null,
                  provider: model.provider || null,
                  thinkingLevel: (ctx && ctx.thinkingLevel) || null,
                }}));
                process.exit(0);
              }});
            }};
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PI_CODING_AGENT_DIR"] = str(agent_dir)
    env["PI_OFFLINE"] = "1"
    proc = subprocess.run(
        [
            pi_bin,
            "--offline",
            "--no-extensions",
            "--no-skills",
            "--no-themes",
            "--no-prompt-templates",
            "--no-context-files",
            "--no-session",
            "--extension",
            str(dump_js),
            "--provider",
            "cursor",
            "--model",
            requested,
            "--api-key",
            "dummy",
        ],
        cwd=str(work),
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_path.is_file(), proc.stdout + proc.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    "requested",
    (
        "cursor/grok-4.5:slow",
        "cursor/grok-4.6:slow",
        "cursor/composer-2-5:slow",
    ),
)
def test_pi_selects_requested_model_before_first_prompt(tmp_path: Path, requested: str) -> None:
    """Real Pi session_start ctx must match the requested opaque identifier.

    Launch argv is necessary but not sufficient: this dumps ``ctx.model``
    from the live ``pi`` process before any user message.
    """
    selected = _dump_pi_selected_model(tmp_path, requested)
    provider, _, catalog_id = requested.partition("/")
    assert selected["provider"] == provider, selected
    assert selected["id"] == catalog_id, selected
    assert f"{selected['provider']}/{selected['id']}" == requested


def test_pi_selected_models_do_not_bleed_across_live_jobs(tmp_path: Path) -> None:
    """Two fresh Pi processes with different ``--model`` values stay isolated."""
    first = _dump_pi_selected_model(tmp_path, "cursor/grok-4.5:slow")
    second = _dump_pi_selected_model(tmp_path, "cursor/composer-2-5:slow")
    assert f"{first['provider']}/{first['id']}" == "cursor/grok-4.5:slow"
    assert f"{second['provider']}/{second['id']}" == "cursor/composer-2-5:slow"
    assert first["id"] != second["id"]
