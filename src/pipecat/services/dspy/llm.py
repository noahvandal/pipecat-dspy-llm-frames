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
import uuid
from typing import Any, Callable, Dict, Mapping, Optional

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
        """Map OpenAI-style messages to a basic Predict signature.

        Default: last user message → question
        """
        try:
            if isinstance(context, OpenAILLMContext):
                messages = context.get_messages()
            else:
                adapter = self.get_llm_adapter()
                messages = adapter.get_llm_invocation_params(context)["messages"]
        except Exception:
            messages = []

        question = ""
        tool_contexts: list[str] = []
        for m in reversed(messages or []):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, list):
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    question = " ".join([t for t in texts if t])
                else:
                    question = content or ""
                break
        # Collect any tool results present in context and append as inline context
        for m in messages or []:
            if m.get("role") == "tool":
                c = m.get("content")
                if isinstance(c, (dict, list)):
                    try:
                        c = json.dumps(c)
                    except Exception:
                        c = str(c)
                if c:
                    tool_contexts.append(str(c))
        if tool_contexts:
            question = (question or "").strip()
            tool_blob = "\n\nContext from tools:\n" + "\n".join(tool_contexts)
            question = (question + tool_blob) if question else tool_blob
        return {"question": question}

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

    @traced_llm
    async def _process_predict(self, context: OpenAILLMContext | LLMContext):
        # Map context → signature inputs
        inputs = self._input_mapping(context) or {}

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

        # Otherwise emit the user-visible response text (single chunk)
        if response:
            await self.push_frame(LLMTextFrame(str(response)))

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
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
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
