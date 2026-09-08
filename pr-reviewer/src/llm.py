"""
llm.py — LangChain chat model backed by an agent CLI subprocess.

One adapter, two backends: get_model("codex") / get_model("bob").
Auth comes from settings.py; nothing reads os.environ directly.

Parsers live in parsers.py; generic helpers in utilities.py.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from parsers import parse_bob, parse_codex
from settings import settings
from utilities import maybe_answer_file, maybe_home, read_text_if_any, scrub


# --------------------------------------------------------------------------
# CLI specs
# --------------------------------------------------------------------------

class ShellSpec(BaseModel):
    """How to invoke one agent CLI headlessly."""
    name: str
    argv: list[str]
    # The variable name the CLI *itself* reads, which is not necessarily the one
    # this project stores the key under. `_child_env` builds an allowlist rather
    # than copying os.environ, so a wrong name here means the child gets no
    # credential at all — verify it against the CLI before changing it.
    env_key: str
    secret_attr: str        # attribute on Settings holding the key
    parse: Callable[[str], str]
    prompt_via: str = "stdin"   # "stdin" | "argv"
    isolate_home: bool = False  # fresh HOME/CODEX_HOME per call
    # Flag the CLI accepts for "write your final message to this file". When set,
    # that file is the answer and `parse` is only a fallback. Reading a file the
    # CLI wrote beats parsing an event stream whose schema can change.
    answer_file_flag: str | None = None

    model_config = {"arbitrary_types_allowed": True}


# Built per call, not at import: reading settings() at module scope would make
# importing this module fail whenever config is absent or invalid.
def build_specs() -> dict[str, ShellSpec]:
    return {
        "codex": ShellSpec(
            name="codex",
            argv=[
                "codex",
                "exec",
                # Structured events, so a failure is machine-readable rather than
                # prose. The answer itself comes from --output-last-message.
                "--json",
                "--model",
                f"{settings().llm_model_name}",
            ],
            # The CLI reads CODEX_API_KEY, not OPENAI_API_KEY, even though the
            # key itself is an OpenAI one and CI supplies it under the OpenAI
            # name. Verified against 0.153.4 by probing the live binary:
            #
            #   CODEX_API_KEY=sk-x  -> "Incorrect API key provided: sk-x***"
            #   OPENAI_API_KEY=sk-x -> "Missing bearer ... in header"
            #
            # The second is what "the CLI never received a key" looks like. Note
            # that an `invalid_api_key` error means the opposite — the key *was*
            # sent and rejected — so the two must not be conflated when
            # re-checking this.
            env_key="CODEX_API_KEY",
            secret_attr="openai_api_key",
            parse=parse_codex,
            prompt_via="argv",
            answer_file_flag="--output-last-message",
            # openai/codex#11435: parallel `codex exec` instances interfere via
            # shared session restore. Isolating HOME sidesteps it — but a fresh
            # CODEX_HOME has no auth.json, so this only works if the CLI accepts
            # CODEX_API_KEY standalone. Flip to False if auth fails.
            isolate_home=False,
        ),
        "bob": ShellSpec(
            name="bob",
            argv=["bob", "--output-format", "stream-json",
                  "--chat-mode", "ask", "--approval-mode", "default",
                  "--max-coins", "30"],
            env_key="BOBSHELL_API_KEY",
            secret_attr="bobshell_api_key",
            parse=parse_bob,
            prompt_via="stdin",
        ),
    }


CLI_NAMES = ("codex", "bob")


# --------------------------------------------------------------------------
# Message flattening
# --------------------------------------------------------------------------

class CLIError(RuntimeError):
    pass


def _text(m: BaseMessage) -> str:
    if isinstance(m.content, str):
        return m.content
    if isinstance(m.content, list):
        return "".join(p.get("text", "") for p in m.content
                       if isinstance(p, dict))
    return str(m.content)


def _render(messages: Sequence[BaseMessage]) -> str:
    """Flatten to one prompt — CLI invocations are stateless."""
    roles = {"system": "SYSTEM", "human": "USER", "ai": "ASSISTANT"}
    return "\n\n".join(
        f"{roles.get(m.type, m.type.upper())}:\n{_text(m)}" for m in messages)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

class ShellChatModel(BaseChatModel):
    """Chat model that shells out to an agent CLI.

    Limitations, stated rather than papered over:
      - no native tool calling  -> bind_tools raises; drive tools from Python
      - no token usage reported -> budget on call count / wall-clock
      - stateless per call      -> full history re-sent every invocation
    """

    spec: ShellSpec
    timeout: int = 300
    cwd: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return f"shell:{self.spec.name}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"cli": self.spec.name, "argv": self.spec.argv}

    # -- process setup -----------------------------------------------------

    def _child_env(self, home: str | None) -> dict[str, str]:
        key = getattr(settings(), self.spec.secret_attr)
        if not key:
            raise CLIError(
                f"no API key for {self.spec.name}: "
                f"Settings.{self.spec.secret_attr} is unset")
        # Allowlist, not os.environ.copy() — the codex process never sees the
        # bob key and vice versa.
        env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": home or os.environ.get("HOME", tempfile.gettempdir()),
            self.spec.env_key: key,
        }
        # Proxy settings included: on a runner behind one, a CLI that cannot see
        # them cannot reach the network at all, and the allowlist is what would
        # have hidden them.
        for k in ("LANG", "TMPDIR", "NODE_PATH", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS",
                  "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
                  "http_proxy", "https_proxy", "no_proxy"):
            if k in os.environ:
                env[k] = os.environ[k]
        if home and self.spec.name == "codex":
            env["CODEX_HOME"] = str(Path(home) / ".codex")
        return env

    def _invocation(
        self, messages: list[BaseMessage], answer_path: Path | None
    ) -> tuple[list[str], str | None]:
        """Returns (argv, stdin_payload)."""
        argv = list(self.spec.argv)
        if answer_path is not None and self.spec.answer_file_flag:
            argv += [self.spec.answer_file_flag, str(answer_path)]

        prompt = _render(messages)
        if self.spec.prompt_via == "argv":
            return [*argv, prompt], None
        return argv, prompt

    # -- sync --------------------------------------------------------------

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        with (maybe_home(self.spec.isolate_home) as home,
              maybe_answer_file(bool(self.spec.answer_file_flag)) as answer_path):
            argv, stdin = self._invocation(messages, answer_path)
            try:
                proc = subprocess.run(
                    argv,
                    # Without this the child inherits our stdin; the codex CLI
                    # reads stdin even when the prompt came in on argv, and would
                    # block waiting for EOF.
                    input=stdin if stdin is not None else "",
                    capture_output=True, text=True,
                    timeout=self.timeout, env=self._child_env(home), cwd=self.cwd)
            except FileNotFoundError as exc:
                raise self._not_installed(argv) from exc
            except subprocess.TimeoutExpired as exc:
                raise CLIError(f"{self.spec.name} timed out after {self.timeout}s") from exc
            return self._to_result(
                proc.returncode, proc.stdout, proc.stderr, argv, answer_path)

    # -- async (required: blocking subprocess serialises any fan-out) -------

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        with (maybe_home(self.spec.isolate_home) as home,
              maybe_answer_file(bool(self.spec.answer_file_flag)) as answer_path):
            argv, stdin = self._invocation(messages, answer_path)
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.PIPE if stdin else asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._child_env(home), cwd=self.cwd)
            except FileNotFoundError as exc:
                raise self._not_installed(argv) from exc

            try:
                out, err = await asyncio.wait_for(
                    proc.communicate(stdin.encode() if stdin else None),
                    timeout=self.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                # Reap it. Without this the loop warns that the child is still
                # running, and the temporary directories above can be removed
                # while it still holds files inside them.
                await proc.wait()
                raise CLIError(f"{self.spec.name} timed out after {self.timeout}s")

            return self._to_result(
                proc.returncode, out.decode(errors="replace"),
                err.decode(errors="replace"), argv, answer_path)

    # -- shared ------------------------------------------------------------

    def _not_installed(self, argv: list[str]) -> CLIError:
        """The CLI is missing from PATH — a setup fault, not a model failure."""
        return CLIError(
            f"{self.spec.name} is not installed or not on PATH (tried {argv[0]!r}). "
            f"Install the {self.spec.name} CLI before running the reviewer.")

    def _to_result(self, rc: int | None, stdout: str, stderr: str,
                   argv: list[str], answer_path: Path | None = None) -> ChatResult:
        if rc != 0:
            raise CLIError(
                f"{self.spec.name} exited {rc}\n"
                f"argv: {argv[:4]}...\n"
                f"--- stderr ---\n{scrub(stderr)[:1500]}\n"
                f"--- stdout ---\n{scrub(stdout)[:1500]}")

        # The file the CLI wrote is authoritative; parsing stdout is the fallback
        # for when it is absent, and cannot be trusted to distinguish the agent's
        # answer from the CLI's own diagnostics.
        text = read_text_if_any(answer_path) or self.spec.parse(stdout)
        if not text:
            raise CLIError(
                f"{self.spec.name} exited 0 but produced no parseable output.\n"
                f"--- stdout ---\n{scrub(stdout)[:1500]}\n"
                f"--- stderr ---\n{scrub(stderr)[:500]}")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        raise NotImplementedError(
            f"{self._llm_type} has no native tool calling — run the tool loop "
            "in Python and feed results back as messages.")


def get_model(cli: str, **kwargs: Any) -> ShellChatModel:
    if cli not in CLI_NAMES:
        raise ValueError(f"unknown CLI {cli!r}; have {sorted(CLI_NAMES)}")
    return ShellChatModel(spec=build_specs()[cli], **kwargs)
