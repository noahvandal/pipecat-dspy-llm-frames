#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""DSPy-backed LLM service for Pipecat.

Predict-only integration that:
- Reuses OpenAI context aggregators and message format
- Supports optional tool calls via Pipecat function-call frames
- Emits only user-facing text from a fixed `response` output
"""

import json
import os
import uuid
from typing import Any, Callable, Dict, Mapping, Optional
import asyncio
import re

import dspy
import httpx
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMUserAggregatorParams,
    LLMAssistantAggregatorParams,
)
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import (
    OpenAIContextAggregatorPair,
    OpenAIUserContextAggregator,
    OpenAIAssistantContextAggregator,
)
from pipecat.utils.tracing.service_decorators import traced_llm


class DSPyLLMService(LLMService):
    """DSPy-based LLM service.

    - Uses dspy.LM via dspy.configure
    - Predict-only execution (single step)
    - Optional tool call support via Pipecat function-call frames
    - Reuses OpenAI context aggregators and message format
    """

    class InputParams(BaseModel):
        """LM configuration for dspy.LM (subset of common knobs)."""

        temperature: Optional[float] = None
        top_p: Optional[float] = None
        max_tokens: Optional[int] = None
        seed: Optional[int] = None
        extra: Optional[Dict[str, Any]] = None

    class ContextPolicy(BaseModel):
        """Controls which context the input mapper composes each turn.

        Toggled via LLMUpdateSettingsFrame settings under keys:
          - dspy.profile: str
          - dspy.context_policy.*: fields below
        """

        include_system: bool = True
        history_last_n: int = 0            # how many recent user/assistant turns to include
        include_tool_results: bool = True  # include role=="tool" contents
        include_summary: bool = False      # if present as tool/system blocks, include
        include_memory: bool = False       # if present as tool/system blocks, include
        max_context_tokens: Optional[int] = None  # reserved; not enforced here
        compose_mode: str = "prepend"     # prepend context before question

    def __init__(
        self,
        *,
        model: str,
        inference_type: str = "predict",
        signature: Optional[Any] = None,
        pretrained: Optional[Any] = None,
        input_mapping: Optional[Callable[[OpenAILLMContext | LLMContext], Dict[str, Any]]] = None,
        allow_tools: bool = False,
        reinvoke_after_tool: bool = True,
        params: Optional[InputParams] = None,
        # Streaming simulation knobs (for smoother TTS):
        simulate_streaming: bool = True,
        stream_chunk_chars: int = 80,
        stream_chunk_pause_ms: int = 0,
        # Which DSPy output fields to expose as Pipecat frames.
        # route ∈ {"downstream","upstream","both","downstream_skip_tts"}
        expose_outputs: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Initialize service and configure DSPy LM/program."""
        super().__init__(**kwargs)
        self.set_model_name(model)
        self._inference_type = inference_type
        self._signature = signature
        self._pretrained = pretrained
        self._allow_tools = allow_tools
        self._reinvoke_after_tool = reinvoke_after_tool
        self._params = params or DSPyLLMService.InputParams()
        self._input_mapping: Callable[[OpenAILLMContext | LLMContext], Dict[str, Any]] = (
            input_mapping or self._default_input_mapping
        )
        # Streaming simulation settings
        self._simulate_stream = simulate_streaming
        self._stream_chunk_chars = max(10, int(stream_chunk_chars or 80))
        self._stream_chunk_pause_ms = max(0, int(stream_chunk_pause_ms or 0))

        # Context policy and profile
        self._policy = DSPyLLMService.ContextPolicy()
        self._profile: Optional[str] = None
        # Exposed outputs configuration (which non-response fields to surface)
        self._expose_outputs: Dict[str, str] = {"response": "downstream"}
        if isinstance(expose_outputs, dict):
            for k, v in expose_outputs.items():
                if isinstance(k, str) and isinstance(v, str):
                    self._expose_outputs[k] = v
        # Signature input overrides (runtime injections)
        self._sig_input_overrides: Dict[str, Any] = {}

        # Debug logging controls (runtime-togglable via dspy.debug.* settings)
        self._debug_log_inputs: bool = False
        self._debug_log_outputs: bool = False
        # Also allow enabling via environment for early startup visibility
        try:
            env_li = os.getenv("DSPY_DEBUG_LOG_INPUTS", "").strip().lower()
            env_lo = os.getenv("DSPY_DEBUG_LOG_OUTPUTS", "").strip().lower()
            if env_li in {"1", "true", "yes", "on"}:
                self._debug_log_inputs = True
            if env_lo in {"1", "true", "yes", "on"}:
                self._debug_log_outputs = True
        except Exception:
            pass

        # Configure DSPy LM
        lm_kwargs: Dict[str, Any] = {}
        if self._params.temperature is not None:
            lm_kwargs["temperature"] = self._params.temperature
        if self._params.top_p is not None:
            lm_kwargs["top_p"] = self._params.top_p
        if self._params.max_tokens is not None:
            lm_kwargs["max_tokens"] = self._params.max_tokens
        if self._params.seed is not None:
            lm_kwargs["seed"] = self._params.seed
        if self._params.extra:
            lm_kwargs.update(self._params.extra)

        try:
            dspy.configure(lm=dspy.LM(model, **lm_kwargs))
        except Exception as e:
            logger.error(f"{self}: failed to configure DSPy LM: {e}")

        # Load/construct DSPy program (Predict-only)
        self._program = None
        if self._pretrained is not None:
            try:
                if isinstance(self._pretrained, str):
                    self._program = dspy.load(self._pretrained)
                else:
                    self._program = self._pretrained
            except Exception as e:
                logger.error(f"{self}: failed to load pretrained DSPy program: {e}")
        if self._program is None:
            if self._signature is None:
                raise ValueError("signature is required when no pretrained program is provided")
            try:
                sig = self._resolve_signature(self._signature)
                self._program = dspy.Predict(sig)
            except Exception as e:
                logger.error(f"{self}: failed to construct Predict program: {e}")
                raise

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> OpenAIContextAggregatorPair:
        """Reuse OpenAI aggregators so tool-calling context is maintained identically."""
        context.set_llm_adapter(self.get_llm_adapter())
        user = OpenAIUserContextAggregator(context, params=user_params)
        assistant = OpenAIAssistantContextAggregator(context, params=assistant_params)
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)

    def can_generate_metrics(self) -> bool:
        return True

    def _resolve_signature(self, sig: Any) -> Any:
        if isinstance(sig, str):
            module_path, _, attr = sig.rpartition(".")
            if not module_path:
                raise ValueError(f"Invalid signature path: {sig}")
            mod = __import__(module_path, fromlist=[attr])
            return getattr(mod, attr)
        return sig

    def _default_input_mapping(self, context: OpenAILLMContext | LLMContext) -> Dict[str, Any]:
        """Compose inputs per policy using OpenAI-style messages.

        Returns a single input key by default: "user_input" containing the
        composed prompt (system/history/tool context + last user text). If the
        signature does not accept "user_input", the caller will transparently
        fall back to "question".
        """
        try:
            if isinstance(context, OpenAILLMContext):
                messages = context.get_messages()
            else:
                adapter = self.get_llm_adapter()
                messages = adapter.get_llm_invocation_params(context)["messages"]
        except Exception:
            messages = []

        # Extract last user utterance
        last_user = ""
        for m in reversed(messages or []):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, list):
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    last_user = " ".join([t for t in texts if t])
                else:
                    last_user = content or ""
                break

        # Build context blob as per policy
        policy = self._policy
        parts: list[str] = []

        if policy.include_system:
            sys_txt = next((m.get("content", "") for m in messages or [] if m.get("role") == "system"), "")
            if sys_txt:
                parts.append(str(sys_txt).strip())

        if policy.history_last_n and policy.history_last_n > 0:
            # Pull last N turns of user/assistant (excluding tools)
            hist: list[str] = []
            count = 0
            for m in reversed([mm for mm in messages or [] if mm.get("role") in {"user", "assistant"}]):
                role = m.get("role")
                content = m.get("content")
                if isinstance(content, list):
                    text = " ".join(c.get("text", "") for c in content if c.get("type") == "text").strip()
                else:
                    text = str(content or "").strip()
                if not text:
                    continue
                hist.append(("User:" if role == "user" else "Assistant:") + " " + text)
                # Count only user/assistant pairs approximately
                count += 1
                if count >= policy.history_last_n * 2:
                    break
            if hist:
                parts.append("Recent conversation:\n" + "\n".join(reversed(hist)))

        if policy.include_tool_results:
            tool_ctx: list[str] = []
            for m in messages or []:
                if m.get("role") == "tool":
                    c = m.get("content")
                    if isinstance(c, (dict, list)):
                        try:
                            c = json.dumps(c)
                        except Exception:
                            c = str(c)
                    if c:
                        tool_ctx.append(str(c))
            if tool_ctx:
                parts.append("Context from tools:\n" + "\n".join(tool_ctx))

        context_blob = "\n\n".join([p for p in parts if p])

        # Compose final user_input
        if context_blob:
            if policy.compose_mode == "prepend":
                composed = (context_blob + "\n\nQuestion: " + (last_user or "")).strip()
            else:
                composed = ((last_user or "") + "\n\n" + context_blob).strip()
        else:
            composed = last_user or ""

        # Start with composed user_input
        inputs: Dict[str, Any] = {"user_input": composed}

        # Merge runtime signature input overrides (only known fields later)
        if self._sig_input_overrides:
            try:
                for k, v in self._sig_input_overrides.items():
                    inputs[k] = v
            except Exception:
                pass

        return inputs

    async def run_inference(self, context: LLMContext | OpenAILLMContext) -> Optional[str]:
        """One-shot Predict-only inference without emitting frames.

        Tool-calls are not executed in this method; it just returns the response text.
        """
        try:
            inputs = self._input_mapping(context) or {}
            outputs = self._program(**inputs)
            if isinstance(outputs, dict):
                return outputs.get("response")
            return getattr(outputs, "response", None)
        except Exception as e:
            logger.error(f"{self}: run_inference error: {e}")
            return None

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into small chunks near punctuation/whitespace for simulated streaming."""
        if not text:
            return []
        text = str(text)
        max_len = self._stream_chunk_chars
        if len(text) <= max_len:
            return [text]
        chunks: list[str] = []
        i = 0
        n = len(text)
        boundaries = re.compile(r"[\.!?\n]\s|")
        while i < n:
            j = min(i + max_len, n)
            # try to extend to nearest boundary ahead (within +40 chars)
            k = min(j + 40, n)
            cut = None
            # search forward for a nice break
            for idx in range(j, k):
                ch = text[idx - 1] if idx - 1 < n else None
                if ch in ".!?\n":
                    cut = idx
                    break
            if cut is None:
                # fallback: search backward for whitespace
                for idx in range(j, i, -1):
                    if text[idx - 1].isspace():
                        cut = idx
                        break
            if cut is None:
                cut = j
            chunk = text[i:cut].strip()
            if chunk:
                chunks.append(chunk)
            i = cut
        return chunks

    def _extract_response_text(self, response: Any) -> str:
        """Ensure only the 'response' field is spoken if the model returned JSON.

        If the response string looks like JSON containing keys like 'response',
        extract and return that value. Otherwise, return the raw response string.
        """
        try:
            if response is None:
                return ""
            text = str(response)
            s = text.strip()
            if not s:
                return text
            if s[0] in "[{":
                try:
                    data = json.loads(s)
                    if isinstance(data, dict) and "response" in data:
                        return str(data.get("response") or "")
                    if (
                        isinstance(data, list)
                        and len(data) > 0
                        and isinstance(data[0], dict)
                        and "response" in data[0]
                    ):
                        return str(data[0].get("response") or "")
                except Exception:
                    pass
            return text
        except Exception:
            return "" if response is None else str(response)

    @traced_llm
    async def _process_predict(self, context: OpenAILLMContext | LLMContext):
        # Map context → signature inputs
        inputs = self._input_mapping(context) or {}
        if self._debug_log_inputs:
            try:
                logger.info(f"{self}: inputs keys -> {list(inputs.keys())}")
            except Exception:
                pass
        # Adapt to the active signature: prefer "user_input" if declared, else fallback to "question"
        try:
            declared_inputs = set()
            sig_obj = None
            if self._program is not None:
                # dspy.Predict(sig) stores the signature class in .signature if available; be defensive
                sig_obj = getattr(self._program, "signature", None) or self._signature
            sig_cls = self._resolve_signature(sig_obj) if sig_obj is not None else None
            # Heuristic: DSPy signatures define attributes for inputs; collect simple attribute names
            if sig_cls is not None:
                for name, value in getattr(sig_cls, "__dict__", {}).items():
                    # dspy.InputField likely lives as class attributes; avoid dunder/private
                    if not name.startswith("_") and not callable(value):
                        declared_inputs.add(name)
        except Exception:
            declared_inputs = set()

        if "user_input" in inputs and declared_inputs and "user_input" not in declared_inputs:
            # Fall back to question if signature doesn't accept user_input
            ui = inputs.pop("user_input")
            inputs.setdefault("question", ui)
        # Also drop any injected inputs not declared by the signature
        if declared_inputs:
            inputs = {k: v for k, v in inputs.items() if k in declared_inputs or k in {"question", "user_input"}}

        # Merge runtime signature input overrides even when using a custom input_mapping
        if self._sig_input_overrides:
            try:
                for k, v in self._sig_input_overrides.items():
                    if not declared_inputs or k in declared_inputs:
                        inputs.setdefault(k, v)
            except Exception:
                pass

        # Execute Predict program
        try:
            outputs = self._program(**inputs)
        except Exception as e:
            logger.error(f"{self}: DSPy execution error: {e}")
            return

        # Extract response and optional tool decision
        response = None
        tool_call = None
        try:
            if isinstance(outputs, dict):
                response = outputs.get("response")
                tool_call = outputs.get("tool_call")
            else:
                response = getattr(outputs, "response", None)
                tool_call = getattr(outputs, "tool_call", None)
        except Exception:
            pass

        # If a tool was requested and tools are allowed, emit function-call frames
        if self._allow_tools and tool_call:
            try:
                # Normalize tool_call from stringified JSON → dict
                if isinstance(tool_call, str):
                    try:
                        maybe = json.loads(tool_call)
                        if isinstance(maybe, dict):
                            tool_call = maybe
                    except Exception:
                        pass

                if isinstance(tool_call, dict):
                    name = tool_call.get("name")
                    args = tool_call.get("arguments", {}) or {}
                elif isinstance(tool_call, str):
                    # Treat bare strings like a tool name; ignore common non-values
                    if tool_call.strip().lower() in {"none", "null", ""}:
                        name, args = None, {}
                    else:
                        name, args = tool_call, {}
                else:
                    name = getattr(tool_call, "name", None)
                    args = getattr(tool_call, "arguments", {}) or {}

                # Validate name and registration; if invalid, skip tool flow and continue
                if not name:
                    raise ValueError("tool_call missing name")
                if isinstance(name, str) and not self.has_function(name):
                    logger.debug(f"{self}: ignoring unknown tool_call '{name}'")
                    name = None

                if name:
                    tool_call_id = str(uuid.uuid4())
                    function_call = FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_call_id,
                        function_name=name,
                        arguments=args,
                    )
                    await self.run_function_calls([function_call])
                    return
            except Exception as e:
                logger.error(f"{self}: failed to emit function call: {e}")
                # If tool handling fails, fall through to emit any response text

        # Expose additional outputs as configured
        try:
            out_map: Dict[str, Any] = {}
            if isinstance(outputs, dict):
                out_map = outputs
            else:
                # Build a dict view from attributes
                out_map = {k: getattr(outputs, k) for k in self._expose_outputs.keys() if hasattr(outputs, k)}

            if self._debug_log_outputs:
                try:
                    logger.info(f"{self}: outputs keys -> {list(out_map.keys())}")
                    if "reasoning" in self._expose_outputs and "reasoning" not in out_map:
                        logger.info(f"{self}: reasoning not present in outputs (nothing to expose)")
                except Exception:
                    pass

            # First, push non-response fields according to policy
            for field, route in self._expose_outputs.items():
                if field == "response":
                    continue
                if field not in out_map:
                    continue
                val = out_map.get(field)
                if val is None:
                    continue
                text = str(val)
                await self._emit_text_field(label=field, text=text, route=route)
        except Exception as e:
            logger.debug(f"{self}: expose_outputs error: {e}")

        # Then, emit the user-visible response text (streamed or single chunk)
        if response is not None and "response" in self._expose_outputs:
            # Only speak the assistant's response; if model returned JSON,
            # extract the 'response' field.
            text = self._extract_response_text(response)
            route = self._expose_outputs.get("response", "downstream")
            if route in {"upstream", "both"}:
                # Send upstream copy
                await self.push_frame(LLMTextFrame(text), FrameDirection.UPSTREAM)
            if route in {"downstream", "both", "downstream_skip_tts"}:
                if route == "downstream_skip_tts":
                    prev = getattr(self, "_skip_tts", False)
                    try:
                        self._skip_tts = True
                        if self._simulate_stream:
                            for chunk in self._chunk_text(text):
                                await self.push_frame(LLMTextFrame(chunk))
                                if self._stream_chunk_pause_ms:
                                    await asyncio.sleep(self._stream_chunk_pause_ms / 1000.0)
                        else:
                            await self.push_frame(LLMTextFrame(text))
                    finally:
                        self._skip_tts = prev
                else:
                    if self._simulate_stream:
                        for chunk in self._chunk_text(text):
                            await self.push_frame(LLMTextFrame(chunk))
                            if self._stream_chunk_pause_ms:
                                await asyncio.sleep(self._stream_chunk_pause_ms / 1000.0)
                    else:
                        await self.push_frame(LLMTextFrame(text))

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        # Predict-only path
        await self._process_predict(context)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for LLM completion requests.

        Handles OpenAILLMContextFrame, LLMContextFrame, LLMMessagesFrame,
        and LLMUpdateSettingsFrame to trigger LLM completions and manage
        settings.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        # Handle DSPy-specific settings updates BEFORE delegating to base class.
        # This prevents base handlers from logging unknown-setting warnings and
        # ensures runtime overrides (e.g., dspy.inputs.*) take effect prior to
        # the next inference trigger.
        if isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
            return

        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            # Handle OpenAI-specific context frames
            context = frame.context
        elif isinstance(frame, LLMContextFrame):
            # Handle universal (LLM-agnostic) LLM context frames
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            # NOTE: LLMMessagesFrame is deprecated, so we don't support the newer universal
            # LLMContext with it
            context = OpenAILLMContext.from_messages(frame.messages)
        else:
            await self.push_frame(frame, direction)

        if context:
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(context)
            except httpx.TimeoutException:
                await self._call_event_handler("on_completion_timeout")
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Handle runtime updates for DSPy context policy and profile.

        Supported keys:
          - "dspy.profile": str (e.g., "chat_minimal", "chat_history_6", "rag", "agentic_tools")
          - "dspy.context_policy.include_system": bool
          - "dspy.context_policy.history_last_n": int
          - "dspy.context_policy.include_tool_results": bool
          - "dspy.context_policy.include_summary": bool
          - "dspy.context_policy.include_memory": bool
          - "dspy.context_policy.max_context_tokens": int
          - "dspy.context_policy.compose_mode": str
        Also supports nested dict at settings["dspy"]["context_policy"].
        """
        try:
            # Profile
            profile = settings.get("dspy.profile") if isinstance(settings, dict) else None
            if profile and isinstance(profile, str):
                self._apply_profile(profile)

            # Flattened context_policy.* keys
            prefix = "dspy.context_policy."
            if isinstance(settings, dict):
                for key, val in settings.items():
                    if isinstance(key, str) and key.startswith(prefix):
                        field = key[len(prefix) :]
                        if hasattr(self._policy, field):
                            setattr(self._policy, field, val)

                # Nested dict path: settings["dspy"]["context_policy"]
                dspy_block = settings.get("dspy")
                if isinstance(dspy_block, dict):
                    cp = dspy_block.get("context_policy")
                    if isinstance(cp, dict):
                        for field, val in cp.items():
                            if hasattr(self._policy, field):
                                setattr(self._policy, field, val)
                    # inputs overrides: dspy.inputs.{field}: value
                    inputs_block = dspy_block.get("inputs")
                    if isinstance(inputs_block, dict):
                        for field, val in inputs_block.items():
                            self._sig_input_overrides[field] = val

            # Flattened inputs: dspy.inputs.*
            prefix_inputs = "dspy.inputs."
            if isinstance(settings, dict):
                for key, val in settings.items():
                    if isinstance(key, str) and key.startswith(prefix_inputs):
                        field = key[len(prefix_inputs) :]
                        self._sig_input_overrides[field] = val

            # Expose outputs updates: dspy.expose_outputs (mapping), dspy.hide_outputs (list)
            if isinstance(settings, dict):
                eo = settings.get("dspy.expose_outputs")
                if isinstance(eo, dict):
                    for k, v in eo.items():
                        if not isinstance(k, str):
                            continue
                        # Accept multiple forms:
                        # - string route (downstream/upstream/both/downstream_skip_tts/log_only/visible/logs/debug)
                        # - boolean True => log_only, False => hide
                        if isinstance(v, bool):
                            if v:
                                self._expose_outputs[k] = "log_only"
                            else:
                                if k in self._expose_outputs:
                                    del self._expose_outputs[k]
                            continue
                        if isinstance(v, str):
                            route = v.strip().lower()
                            if route in {"visible", "visibility", "log", "logs", "debug"}:
                                route = "log_only"
                            self._expose_outputs[k] = route
                ho = settings.get("dspy.hide_outputs")
                if isinstance(ho, (list, tuple)):
                    for k in ho:
                        if isinstance(k, str) and k in self._expose_outputs:
                            del self._expose_outputs[k]

                # visible_outputs: list of fields to log only (do not emit frames)
                vo = settings.get("dspy.visible_outputs")
                if isinstance(vo, (list, tuple)):
                    for k in vo:
                        if isinstance(k, str):
                            self._expose_outputs[k] = "log_only"

                # Debug logging toggles
                dbg = settings.get("dspy.debug")
                if isinstance(dbg, dict):
                    li = dbg.get("log_inputs")
                    lo = dbg.get("log_outputs")
                    if isinstance(li, bool):
                        self._debug_log_inputs = li
                    if isinstance(lo, bool):
                        self._debug_log_outputs = lo
        except Exception as e:
            logger.warning(f"{self}: _update_settings ignored due to error: {e}")

    def _apply_profile(self, name: str):
        name = (name or "").strip().lower()
        self._profile = name
        p = DSPyLLMService.ContextPolicy()
        if name == "chat_minimal":
            p.include_system = True
            p.history_last_n = 0
            p.include_tool_results = False
        elif name.startswith("chat_history_"):
            # chat_history_6 → history_last_n=6
            try:
                k = int(name.split("_")[-1])
            except Exception:
                k = 6
            p.include_system = True
            p.history_last_n = k
            p.include_tool_results = True
        elif name == "rag":
            p.include_system = True
            p.history_last_n = 4
            p.include_tool_results = True
            p.include_summary = True
        elif name == "agentic_tools":
            p.include_system = True
            p.history_last_n = 4
            p.include_tool_results = True
        self._policy = p

    async def _emit_text_field(self, *, label: str, text: str, route: str):
        if not text:
            return
        # Log the field explicitly so it shows up in Pipecat logs regardless of routing
        try:
            preview = text if len(text) <= 500 else (text[:500] + "…")
            logger.info(f"{self}: expose_output {label} -> {route}: {preview}")
        except Exception:
            pass
        msg = f"[{label}] {text}"
        # New: log_only route to avoid injecting frames into the pipeline entirely
        if route == "log_only":
            return
        if route in {"upstream", "both"}:
            await self.push_frame(LLMTextFrame(msg), FrameDirection.UPSTREAM)
        if route in {"downstream", "both", "downstream_skip_tts"}:
            if route == "downstream_skip_tts":
                prev = getattr(self, "_skip_tts", False)
                try:
                    self._skip_tts = True
                    await self.push_frame(LLMTextFrame(msg))
                finally:
                    self._skip_tts = prev
            else:
                await self.push_frame(LLMTextFrame(msg))
