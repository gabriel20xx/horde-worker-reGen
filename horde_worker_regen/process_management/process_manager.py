import asyncio
import asyncio.exceptions
import base64
import collections
import contextlib
import datetime
import enum
import json
import math
import multiprocessing
import os
import queue
import random
import re
import sqlite3
import ssl
import sys
import threading
import time
from asyncio import CancelledError, Task
from asyncio import Lock as Lock_Asyncio
from collections import deque
from collections.abc import Mapping
from enum import auto
from io import BytesIO
from multiprocessing.context import BaseContext
from multiprocessing.synchronize import BoundedSemaphore
from multiprocessing.synchronize import Lock as Lock_MultiProcessing
from typing import TYPE_CHECKING, Any

import aiohttp
import aiohttp.client_exceptions
import PIL
import PIL.Image
import psutil
import yarl
from aiohttp import ClientSession
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY, STABLE_DIFFUSION_BASELINE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import StableDiffusion_ModelReference
from horde_sdk import RequestErrorResponse
from horde_sdk.ai_horde_api import GENERATION_STATE
from horde_sdk.ai_horde_api.ai_horde_clients import (
    AIHordeAPIAsyncClientSession,
    AIHordeAPISimpleClient,
)
from horde_sdk.ai_horde_api.apimodels import (
    DeleteWorkerRequest,
    DeleteWorkerResponse,
    FindUserRequest,
    GenMetadataEntry,
    ImageGenerateJobPopRequest,
    ImageGenerateJobPopResponse,
    JobSubmitResponse,
    ModifyWorkerRequest,
    SingleWorkerDetailsRequest,
    SingleWorkerDetailsResponse,
    UserDetailsResponse,
)
from horde_sdk.ai_horde_api.consts import KNOWN_UPSCALERS, METADATA_TYPE, METADATA_VALUE
from horde_sdk.ai_horde_api.fields import JobID
from loguru import logger
from pydantic import BaseModel, ConfigDict, RootModel, ValidationError
from typing_extensions import override

import horde_worker_regen
from horde_worker_regen.bridge_data.data_model import reGenBridgeData
from horde_worker_regen.bridge_data.load_config import BridgeDataLoader
from horde_worker_regen.consts import (
    BRIDGE_CONFIG_FILENAME,
    KNOWN_CONTROLNET_WORKFLOWS,
    KNOWN_SLOW_MODELS_DIFFICULTIES,
    KNOWN_SLOW_WORKFLOWS,
    MAX_SOURCE_IMAGE_RETRIES,
    VRAM_HEAVY_MODELS,
    WEBUI_MODEL_STATE_FILENAME,
    WORKER_RESTART_EXIT_CODE,
)
from horde_worker_regen.logger_config import create_level_format_function
from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.inference_process import HordeInferenceProcess
from horde_worker_regen.process_management.messages import (
    HordeAuxModelStateChangeMessage,
    HordeControlFlag,
    HordeControlMessage,
    HordeControlModelMessage,
    HordeHeartbeatType,
    HordeImageResult,
    HordeInferenceControlMessage,
    HordeInferenceResultMessage,
    HordeModelStateChangeMessage,
    HordePreloadInferenceModelMessage,
    HordeProcessHeartbeatMessage,
    HordeProcessMemoryMessage,
    HordeProcessMessage,
    HordeProcessState,
    HordeProcessStateChangeMessage,
    HordeSafetyControlMessage,
    HordeSafetyResultMessage,
    ModelInfo,
    ModelLoadState,
)
from horde_worker_regen.process_management.worker_entry_points import start_inference_process, start_safety_process

if TYPE_CHECKING:
    from horde_worker_regen.webui.server import WorkerWebUI

sslcontext = ssl.create_default_context()

# Constants
BYTES_TO_MEGABYTES = 1024 * 1024
"""Conversion factor from bytes to megabytes."""

METRICS_CALCULATION_WINDOW_SECONDS = 3600
"""Rolling time window, in seconds, for calculating rate-based metrics such as kudos per hour and images per hour."""

# CUDA cores per streaming multiprocessor (SM) keyed by (major, minor) compute capability.
# Sourced from the NVIDIA CUDA Programming Guide and GPU specifications.
# Unknown compute capabilities default to 0 (no cores reported) to avoid mixing units.
_CUDA_CORES_PER_SM: dict[tuple[int, int], int] = {
    (2, 0): 32,
    (2, 1): 48,
    (3, 0): 192,
    (3, 2): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,
    (5, 2): 128,
    (5, 3): 128,
    (6, 0): 64,
    (6, 1): 128,
    (6, 2): 128,
    (7, 0): 64,
    (7, 2): 64,
    (7, 5): 64,
    (8, 0): 64,
    (8, 6): 128,
    (8, 7): 128,
    (8, 9): 128,
    (9, 0): 128,
}


def _get_cuda_cores_per_sm(major: int, minor: int) -> int | None:
    """Return CUDA cores per SM for a compute capability.

    Falls back to the latest known minor version for the same major version so
    newer minor revisions do not immediately report as unknown.
    """
    exact = _CUDA_CORES_PER_SM.get((major, minor))
    if exact is not None:
        return exact

    minors_for_major = [mapped_minor for mapped_major, mapped_minor in _CUDA_CORES_PER_SM if mapped_major == major]
    if not minors_for_major:
        return None

    minors_at_or_below_requested = [mapped_minor for mapped_minor in minors_for_major if mapped_minor <= minor]
    if minors_at_or_below_requested:
        return _CUDA_CORES_PER_SM[(major, max(minors_at_or_below_requested))]

    return _CUDA_CORES_PER_SM[(major, min(minors_for_major))]

# This is due to Linux/Windows differences in the multiprocessing module
# ! IMPORTANT: Start of own code
try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except (ImportError, AttributeError):
    # PipeConnection not available on all platforms, fall back to Connection
    from multiprocessing.connection import Connection  # type: ignore
# ! IMPORTANT: End of own code


# As of 3.11, asyncio.TimeoutError is deprecated and is an alias for builtins.TimeoutError
_async_client_exceptions: tuple[type[Exception], ...] = (TimeoutError, aiohttp.client_exceptions.ClientError, OSError)

if sys.version_info[:2] == (3, 10):
    _async_client_exceptions = (asyncio.exceptions.TimeoutError, aiohttp.client_exceptions.ClientError, OSError)


def _remove_awaiting_request(session: "AIHordeAPIAsyncClientSession | None", request: object) -> None:
    """Remove a stuck request from the SDK session's ``_awaiting_requests`` container.

    The SDK's ``GenericAsyncHordeAPISession.submit_request`` has no ``try/finally``,
    so when the coroutine is cancelled (e.g. by ``asyncio.wait_for`` on timeout) or
    raises an unexpected exception, the request is left in ``_awaiting_requests``.
    This causes spurious warnings when the session context manager exits on shutdown.
    Call this helper from every ``except`` branch that follows a ``submit_request``
    call that might have been interrupted before it could remove the request itself.
    """
    if session is None:
        return
    awaiting = getattr(session, "_awaiting_requests", None)
    if awaiting is None:
        return
    remove = getattr(awaiting, "remove", None)
    if not callable(remove):
        return
    try:
        remove(request)
    except (ValueError, KeyError):
        pass


def _active_filter_entries(groups: list, type_enabled: bool) -> list[str] | None:
    """Return a flat list of entries from all enabled groups, or None if disabled/empty.

    Accepts ``PromptFilterGroup`` instances, plain dicts, and bare strings (old YAML format)
    so that the result is correct regardless of how Pydantic loaded the field value.
    """
    if not type_enabled:
        return None
    entries: list[str] = []
    for g in groups:
        if isinstance(g, str):
            if g:
                entries.append(g)
        elif isinstance(g, dict):
            if not g.get("enabled", True):
                continue
            entries.extend(str(e) for e in g.get("entries", []) if e)
        else:
            if not g.enabled:
                continue
            entries.extend(e for e in g.entries if e)
    return entries if entries else None


def _apply_prompt_filters(
    text: str,
    *,
    append: list[str] | None = None,
    remove: list[str] | None = None,
    replace: list[str] | None = None,
    remove_cleanup_separators: bool = True,
    append_separator: bool = True,
    remove_whole_word: bool = False,
    remove_case_sensitive: bool = True,
) -> str:
    """Apply separate append/remove/replace filter lists to *text*.

    - *remove*: each item is a literal string to strip from the text.
      Controlled by *remove_whole_word* (only match standalone words) and
      *remove_case_sensitive* (case-exact vs. case-insensitive matching).
    - *replace*: each item is ``find==>with`` format; entries without ``==>`` are skipped.
      Always matched whole-word and case-insensitively.
    - *append*: each item is appended; separator behaviour is controlled by *append_separator*
    - *remove_cleanup_separators*: when True, collapses orphaned commas/spaces between removed
      items (e.g. ``"a, , , b"`` → ``"a, b"``)
    - *append_separator*: when True each appended string is joined with ``", "``; when False
      the string is concatenated directly
    """
    for item in (remove or []):
        if item:
            if remove_whole_word or not remove_case_sensitive:
                flags = 0 if remove_case_sensitive else re.IGNORECASE
                pat = rf"\b{re.escape(item)}\b" if remove_whole_word else re.escape(item)
                text = re.compile(pat, flags).sub("", text)
            else:
                text = text.replace(item, "")
    if remove_cleanup_separators and remove:
        # Collapse any run of commas (with optional whitespace between) into a single ", ".
        # This cleans up ", , , " patterns left when adjacent items were removed.
        text = re.sub(r",\s*(?:,\s*)+", ", ", text)
        # Strip orphaned leading/trailing commas and whitespace.
        text = text.strip(" ,")
        # Normalize multiple consecutive spaces that removal may have left.
        text = re.sub(r"  +", " ", text)
    for rule in (replace or []):
        if not rule or "==>" not in rule:
            continue
        find, with_ = rule.split("==>", 1)
        if find:
            # Whole-word, case-insensitive replacement: only standalone occurrences of
            # *find* are replaced (e.g. "cat" does not match inside "category"), regardless
            # of letter case ("Cat"/"CAT" both match). The replacement text is inserted
            # literally, so backslashes/group references in *with_* are not interpreted.
            pattern = re.compile(rf"\b{re.escape(find)}\b", re.IGNORECASE)
            text = pattern.sub(lambda _match: with_, text)
    for item in (append or []):
        if item:
            if append_separator:
                sep = ", " if text.strip() else ""
                text = text.rstrip(" ,") + sep + item
            else:
                text = text + item
    return text


def _apply_prompt_swap(
    pos: str,
    neg: str,
    swap_entries: list[str],
    *,
    remove_whole_word: bool = False,
    remove_case_sensitive: bool = True,
    remove_cleanup_separators: bool = True,
    append_separator: bool = True,
) -> tuple[str, str]:
    """Move strings found in one prompt into the other.

    For each entry: if it appears in the positive prompt it is removed from positive and
    appended to negative, and vice-versa.  Presence is checked against the original
    (pre-modification) values so that a string found in both prompts simultaneously is
    moved in both directions independently rather than cancelling out.
    """
    if not swap_entries:
        return pos, neg
    pos_matches: list[str] = []
    neg_matches: list[str] = []
    for entry in swap_entries:
        if not entry:
            continue
        if remove_whole_word or not remove_case_sensitive:
            flags = 0 if remove_case_sensitive else re.IGNORECASE
            pat = re.compile(
                rf"\b{re.escape(entry)}\b" if remove_whole_word else re.escape(entry),
                flags,
            )
            if pat.search(pos):
                pos_matches.append(entry)
            if pat.search(neg):
                neg_matches.append(entry)
        else:
            if entry in pos:
                pos_matches.append(entry)
            if entry in neg:
                neg_matches.append(entry)
    if pos_matches:
        pos = _apply_prompt_filters(
            pos,
            remove=pos_matches,
            remove_whole_word=remove_whole_word,
            remove_case_sensitive=remove_case_sensitive,
            remove_cleanup_separators=remove_cleanup_separators,
        )
        neg = _apply_prompt_filters(neg, append=pos_matches, append_separator=append_separator)
    if neg_matches:
        neg = _apply_prompt_filters(
            neg,
            remove=neg_matches,
            remove_whole_word=remove_whole_word,
            remove_case_sensitive=remove_case_sensitive,
            remove_cleanup_separators=remove_cleanup_separators,
        )
        pos = _apply_prompt_filters(pos, append=neg_matches, append_separator=append_separator)
    return pos, neg


def _apply_conditional_add(
    text: str,
    conditional_add: list[str] | None = None,
    *,
    append_separator: bool = True,
) -> str:
    """Apply conditional-add rules to *text*.

    Each entry is ``trigger==>add`` format: if ``trigger`` is found (case-insensitively)
    in *text*, ``add`` is appended to the text.  Entries without ``==>`` or with an empty
    trigger or add part are skipped.
    """
    for rule in (conditional_add or []):
        if not rule or "==>" not in rule:
            continue
        trigger, add = rule.split("==>", 1)
        if not trigger or not add:
            continue
        if re.search(re.escape(trigger), text, re.IGNORECASE):
            if append_separator:
                sep = ", " if text.strip() else ""
                text = text.rstrip(" ,") + sep + add
            else:
                text = text + add
    return text


_excludes_for_job_dump = {
    "job_image_results": True,
    "sdk_api_job_info": {
        "payload": {"prompt": True, "special": True},
        "skipped": True,
        "source_image": True,
        "source_mask": True,
        "extra_source_images": True,
        "r2_upload": True,
        "r2_uploads": True,
    },
}

_caught_signal = False


class HordeProcessInfo:
    """Contains information about a horde child process."""

    mp_process: multiprocessing.Process
    """The multiprocessing.Process object for this process."""
    pipe_connection: Connection
    """The connection through which messages can be sent to this process."""
    process_id: int
    """The ID of this process. This is not an OS process ID."""
    process_type: HordeProcessType
    """The type of this process."""
    last_process_state: HordeProcessState
    """The last known state of this process."""

    last_heartbeat_timestamp: float
    """Last time we received a heartbeat from this process."""
    last_heartbeat_delta: float
    """The delta between the last two heartbeats. Used to determine if the process is stuck."""
    last_heartbeat_type: HordeHeartbeatType
    """The type of the last heartbeat received from this process."""
    heartbeats_inference_steps: int
    """The number of inference steps that have been completed since the last heartbeat."""
    last_heartbeat_percent_complete: int | None
    """The last percentage reported by the process."""

    # Progress tracking for detecting stalled inference jobs
    last_progress_timestamp: float
    """Last time progress (percent_complete) actually advanced."""
    last_progress_value: int | None
    """The last progress value to detect if progress is advancing."""
    last_inference_step_timestamp: float | None
    """Last time an INFERENCE_STEP heartbeat was received. None until the first step arrives."""

    inference_started_timestamp: float | None
    """Timestamp when this process entered INFERENCE_STARTING for the current job. None when not inferring.

    Unlike state_entered_timestamp (which resets on every state change), this persists across the
    INFERENCE_STARTING → INFERENCE_PROCESSING transition so total inference duration can be measured."""

    last_received_timestamp: float
    """Last time we updated the process info. If we're regularly working, then this value should change frequently."""
    loaded_horde_model_name: str | None
    """The name of the horde model that is (supposedly) currently loaded in this process."""
    loaded_horde_model_baseline: STABLE_DIFFUSION_BASELINE_CATEGORY | str | None
    """The baseline of the horde model that is (supposedly) currently loaded in this process."""
    last_control_flag: HordeControlFlag | None
    """The last control flag sent, to avoid duplication."""

    last_job_referenced: ImageGenerateJobPopResponse | None

    ram_usage_bytes: int
    """The amount of RAM used by this process."""
    vram_usage_bytes: int
    """The amount of VRAM used by this process."""
    total_vram_bytes: int
    """The total amount of VRAM available to this process."""
    gpu_usage_percent: float
    """The GPU SM utilisation percentage reported by this process (0–100)."""
    batch_amount: int
    """The total amount of batching being run by this process."""

    recently_unloaded_from_ram: bool
    """True if models were recently unloaded from RAM."""

    process_launch_identifier: int
    """The identifier for the process launch. Used to track restarting of specific process slots."""

    last_send_error: Exception | None
    """The last exception raised by safe_send_message(), or None if the last send succeeded.

    This field is set by safe_send_message() and is intended for diagnostic use immediately
    after safe_send_message() returns False. It is reset to None on each successful send, so
    consumers should read it before the next send call."""

    state_entered_timestamp: float
    """Timestamp (epoch seconds) when the process entered its current state.

    Updated on every call to :meth:`ProcessMap.on_process_state_change` so that
    per-state elapsed time can be computed from the manager's message loop."""

    def __init__(
        self,
        mp_process: multiprocessing.Process,
        pipe_connection: Connection,
        process_id: int,
        process_type: HordeProcessType,
        last_process_state: HordeProcessState,
        process_launch_identifier: int,
    ) -> None:
        """Initialize a new HordeProcessInfo object.

        Args:
            mp_process (multiprocessing.Process): The multiprocessing.Process object for this process.
            pipe_connection (Connection): The connection through which messages can be sent to this process.
            process_id (int): The ID of this process. This is not an OS process ID.
            process_type (HordeProcessType): The type of this process.
            last_process_state (HordeProcessState): The last known state of this process.
            process_launch_identifier (int): The identifier for the process launch. Used to track restarting of \
                specific process slots.
        """
        self.mp_process = mp_process
        self.pipe_connection = pipe_connection
        self.process_id = process_id
        self.process_type = process_type
        self.last_process_state = last_process_state
        self.last_received_timestamp = time.time()
        self.loaded_horde_model_name = None
        self.loaded_horde_model_baseline = None
        self.last_control_flag = None

        self.last_heartbeat_timestamp = time.time()
        self.last_heartbeat_delta = 0
        self.last_heartbeat_type = HordeHeartbeatType.OTHER
        self.heartbeats_inference_steps = 0
        self.last_heartbeat_percent_complete = None

        # Initialize progress tracking
        self.last_progress_timestamp = time.time()
        self.last_progress_value = None
        self.last_inference_step_timestamp = None
        self.inference_started_timestamp: float | None = None

        self.last_job_referenced = None

        self.ram_usage_bytes = 0
        self.vram_usage_bytes = 0
        self.total_vram_bytes = 0
        self.gpu_usage_percent = 0.0
        self.batch_amount = 1

        self.recently_unloaded_from_ram = False

        self.process_launch_identifier = process_launch_identifier

        self.last_send_error = None

        self.state_entered_timestamp = time.time()

    def is_process_busy(self) -> bool:
        """Return true if the process is actively engaged in a task.

        This does not include the process starting up or shutting down.
        """
        return (
            self.last_process_state == HordeProcessState.INFERENCE_STARTING
            or self.last_process_state == HordeProcessState.INFERENCE_PROCESSING
            or self.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
            or self.last_process_state == HordeProcessState.POST_PROCESSING_STARTING
            or self.last_process_state == HordeProcessState.ALCHEMY_STARTING
            or self.last_process_state == HordeProcessState.DOWNLOADING_MODEL
            or self.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL
            or self.last_process_state == HordeProcessState.MODEL_PRELOADING
            or self.last_process_state == HordeProcessState.MODEL_PRELOADED
            or self.last_process_state == HordeProcessState.MODEL_LOADING
            or self.last_process_state == HordeProcessState.MODEL_LOADED
            or self.last_process_state == HordeProcessState.JOB_RECEIVED
            or self.last_process_state == HordeProcessState.SAFETY_EVALUATING
            or self.last_process_state == HordeProcessState.SAFETY_STARTING
            or self.last_process_state == HordeProcessState.RESULT_SAVING
            or self.last_process_state == HordeProcessState.RESULT_SUBMITTING
            or self.last_process_state == HordeProcessState.PROCESS_STARTING
        )

    def is_process_alive(self) -> bool:
        """Return true if the process is alive."""
        if not self.mp_process.is_alive():
            return False
        return not (
            self.last_process_state == HordeProcessState.PROCESS_ENDING
            or self.last_process_state == HordeProcessState.PROCESS_ENDED
        )

    def safe_send_message(self, message: HordeControlMessage) -> bool:
        """Send a message to the process.

        Args:
            message (HordeControlMessage): The message to send.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
            On failure the exception is stored in ``last_send_error`` so callers
            can surface the underlying cause in error logs and fault information.
        """
        try:
            self.pipe_connection.send(message)
            self.last_send_error = None
            return True
        except Exception as e:
            self.last_send_error = e
            global _caught_signal
            if not _caught_signal:
                logger.debug(
                    f"Failed to send message to process {self.process_id}: "
                    f"{type(e).__name__}: {e}",
                )
            return False

    def __repr__(self) -> str:
        """Return a string representation of the process info."""
        return str(
            f"HordeProcessInfo(process_id={self.process_id}, last_process_state={self.last_process_state}, "
            f"loaded_horde_model_name={self.loaded_horde_model_name})",
        )

    def can_accept_job(self) -> bool:
        """Return true if the process can accept a job.

        POST_PROCESSING_COMPLETE is intentionally excluded: the child process is still
        inside _receive_and_handle_control_message sending the result to the manager;
        treating it as available would let the manager schedule a new job or replace
        the process before the current job's result has been enqueued, which can cause
        the job to be silently lost.
        """
        return (
            self.last_process_state == HordeProcessState.WAITING_FOR_JOB
            or self.last_process_state == HordeProcessState.MODEL_PRELOADED
            or self.last_process_state == HordeProcessState.MODEL_LOADED
            or self.last_process_state == HordeProcessState.INFERENCE_COMPLETE
            or self.last_process_state == HordeProcessState.ALCHEMY_COMPLETE
        )


class HordeModelMap(RootModel[dict[str, ModelInfo]]):
    """A mapping of horde model names to `ModelInfo` objects. Contains some helper methods."""

    def update_entry(
        self,
        horde_model_name: str,
        *,
        load_state: ModelLoadState | None = None,
        process_id: int | None = None,
    ) -> None:
        """Update the entry for the given model name. If the model does not exist, it will be created.

        Args:
            horde_model_name (str): The (horde) name of the model to update.
            load_state (ModelLoadState | None, optional): The load state of the model. Defaults to None.
            process_id (int | None, optional): The process ID of the process that has this model loaded. \
                Defaults to None.

        Raises:
            ValueError: If the process_id is None and the model does not exist.
            ValueError: If the load_state is None and the model does not exist.
        """
        if horde_model_name not in self.root:
            if process_id is None:
                raise ValueError("process_id must be provided when adding a new model to the map")
            if load_state is None:
                raise ValueError("model_load_state must be provided when adding a new model to the map")

            self.root[horde_model_name] = ModelInfo(
                horde_model_name=horde_model_name,
                horde_model_load_state=load_state,
                process_id=process_id,
            )

        if load_state is not None:
            self.root[horde_model_name].horde_model_load_state = load_state
            logger.debug(f"Updated load state for {horde_model_name} to {load_state}")

        if process_id is not None:
            self.root[horde_model_name].process_id = process_id
            logger.debug(f"Updated process ID for {horde_model_name} to {process_id}")

    def expire_entry(self, horde_model_name: str) -> ModelInfo | None:
        """Removes information about a horde model.

        :param horde_model_name: Name of model to remove
        :return: model name if removed; 'none' string otherwise
        """
        return self.root.pop(horde_model_name, None)

    def is_model_loaded(self, horde_model_name: str) -> bool:
        """Return true if the given model is loaded in any process."""
        if horde_model_name not in self.root:
            return False
        return self.root[horde_model_name].horde_model_load_state.is_loaded()

    def is_model_loading(self, horde_model_name: str) -> bool:
        """Return true if the given model is currently being loaded in any process."""
        if horde_model_name not in self.root:
            return False
        return self.root[horde_model_name].horde_model_load_state == ModelLoadState.LOADING


class ProcessMap(dict[int, HordeProcessInfo]):
    """A mapping of process IDs to HordeProcessInfo objects. Contains some helper methods."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    MAX_INFERENCE_STEP_TIMEOUT: float = HordeInferenceProcess._INFERENCE_HEARTBEAT_INTERVAL
    """Maximum seconds allowed between consecutive INFERENCE_STEP heartbeats before a process is
    considered stuck on a single diffusion step. Derived from and must match or exceed the
    background heartbeat interval so that the background keepalive fires before the per-step
    check triggers."""

    ZERO_PROGRESS_TIMEOUT: float = 120.0
    """Seconds to wait for progress to advance from 0 % before declaring the process stuck.

    Used as the ``zero_progress_timeout`` argument to :meth:`is_stuck_on_inference` (check 2).
    When a process is in INFERENCE_PROCESSING and has reported 0 % progress (or none at all)
    because no real diffusion step has produced any output, this is the governing timeout:
    the per-step check (check 1, ``inference_step_timeout``, 30 s by default) deliberately
    does not apply before the first step — model loading into VRAM routinely takes longer
    than that — and the VAE-decode-safe ``no_step_heartbeat_timeout`` (300 s) would be far
    too slow for a genuine pre-first-step hang.

    VAE decode — where ``no_step_heartbeat_timeout`` must stay at 300 s — only occurs at
    100 % progress after all steps complete, so reducing the 0 %-stuck timeout does not risk
    killing a legitimately-running VAE decode.  120 s gives ample time for the first
    diffusion step to start (SDXL VRAM loading is typically 10–60 s) while catching hangs
    that would otherwise take 300 s to detect."""

    def on_heartbeat(
        self,
        process_id: int,
        heartbeat_type: HordeHeartbeatType,
        *,
        percent_complete: int | None = None,
    ) -> None:
        """Update the heartbeat for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            heartbeat_type (HordeHeartbeatType): The type of the heartbeat.
            percent_complete (int | None, optional): The percentage of the job that has been completed, \
                if applicable. Defaults to None.
        """
        now = time.time()
        self[process_id].last_heartbeat_delta = now - self[process_id].last_heartbeat_timestamp
        self[process_id].last_received_timestamp = now
        self[process_id].last_heartbeat_timestamp = now
        self[process_id].last_heartbeat_type = heartbeat_type
        if heartbeat_type == HordeHeartbeatType.INFERENCE_STEP:
            self[process_id].heartbeats_inference_steps += 1
            self[process_id].last_inference_step_timestamp = now
        else:
            self[process_id].heartbeats_inference_steps = 0

        # Update progress tracking to detect stalled jobs
        if percent_complete is not None and self[process_id].last_progress_value != percent_complete:
            self[process_id].last_progress_timestamp = now
            self[process_id].last_progress_value = percent_complete

        self[process_id].last_heartbeat_percent_complete = percent_complete

    def on_process_ending(self, process_id: int) -> None:
        """Update the process map when a process has ended.

        Args:
            process_id (int): The ID of the process that has ended.
        """
        self[process_id].last_process_state = HordeProcessState.PROCESS_ENDING
        self[process_id].loaded_horde_model_name = None
        self[process_id].loaded_horde_model_baseline = None
        self[process_id].last_job_referenced = None
        self[process_id].batch_amount = 1

        self.reset_heartbeat_state(process_id)

        self[process_id].last_received_timestamp = time.time()

    def on_memory_report(
        self,
        process_id: int,
        ram_usage_bytes: int,
        vram_usage_bytes: int | None = 0,
        total_vram_bytes: int | None = 0,
        gpu_usage_percent: float | None = None,
    ) -> None:
        """Update the memory usage for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            ram_usage_bytes (int): The amount of RAM used by this process.
            vram_usage_bytes (int): The amount of VRAM used by this process.
            total_vram_bytes (int): The total amount of VRAM available to this process.
            gpu_usage_percent (float | None): GPU SM utilisation percentage (0–100), or None if unknown.
        """
        self[process_id].ram_usage_bytes = ram_usage_bytes
        self[process_id].vram_usage_bytes = vram_usage_bytes or 0
        self[process_id].total_vram_bytes = total_vram_bytes or 0
        if gpu_usage_percent is not None:
            self[process_id].gpu_usage_percent = gpu_usage_percent

        self[process_id].last_received_timestamp = time.time()

        logger.debug(
            f"Process {process_id} memory report: "
            f"ram: {ram_usage_bytes} vram: {vram_usage_bytes} total vram: {total_vram_bytes}",
        )

    def on_process_state_change(self, process_id: int, new_state: HordeProcessState) -> None:
        """Update the process state for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            new_state (HordeProcessState): The new state of the process.
        """
        self[process_id].last_process_state = new_state
        self[process_id].last_received_timestamp = time.time()
        self[process_id].state_entered_timestamp = time.time()

        # Track when inference starts so total inference duration can be measured independently
        # of per-state elapsed time (state_entered_timestamp resets on every transition).
        if new_state == HordeProcessState.INFERENCE_STARTING:
            self[process_id].inference_started_timestamp = time.time()
        elif new_state not in (HordeProcessState.INFERENCE_STARTING, HordeProcessState.INFERENCE_PROCESSING):
            self[process_id].inference_started_timestamp = None

        if (
            new_state == HordeProcessState.INFERENCE_COMPLETE
            or new_state == HordeProcessState.POST_PROCESSING_COMPLETE
            or new_state == HordeProcessState.INFERENCE_FAILED
            or new_state == HordeProcessState.MODEL_PRELOADED
            or new_state == HordeProcessState.MODEL_LOADED
            or new_state == HordeProcessState.SAFETY_COMPLETE
            or new_state == HordeProcessState.RESULT_SAVED
            or new_state == HordeProcessState.WAITING_FOR_JOB
        ):
            self.reset_heartbeat_state(process_id)

            # Set progress to 100% after reset when inference completes
            if (
                new_state == HordeProcessState.INFERENCE_COMPLETE
                or new_state == HordeProcessState.POST_PROCESSING_COMPLETE
            ):
                self[process_id].last_heartbeat_percent_complete = 100

        # Reset progress to 0% when a new inference is starting
        if new_state == HordeProcessState.INFERENCE_STARTING:
            self[process_id].last_heartbeat_percent_complete = 0
            # Also reset progress tracking so stall detection starts fresh
            self[process_id].last_progress_timestamp = time.time()
            self[process_id].last_progress_value = None
            self[process_id].last_inference_step_timestamp = None

    def on_last_job_reference_change(
        self,
        process_id: int,
        last_job_referenced: ImageGenerateJobPopResponse | None,
    ) -> None:
        """Update the job reference for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            last_job_referenced (ImageGenerateJobPopResponse | None): The last job referenced by this process.
        """
        if last_job_referenced is not None and (last_job_referenced != self[process_id].last_job_referenced):
            logger.debug(f"Resetting heartbeat for process {process_id}")
            self[process_id].last_heartbeat_delta = 0
            self[process_id].last_heartbeat_timestamp = time.time()
            self[process_id].heartbeats_inference_steps = 0
            # Reset progress tracking for new job
            self[process_id].last_progress_timestamp = time.time()
            self[process_id].last_progress_value = None
            self[process_id].last_inference_step_timestamp = None

        self[process_id].last_job_referenced = last_job_referenced
        self[process_id].last_received_timestamp = time.time()

    def on_model_load_state_change(
        self,
        process_id: int,
        horde_model_name: str | None,
        horde_model_baseline: STABLE_DIFFUSION_BASELINE_CATEGORY | str | None = None,
        last_job_referenced: ImageGenerateJobPopResponse | None = None,
    ) -> None:
        """Update the model load state for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            horde_model_name (str): The name of the horde model to update.
            horde_model_baseline (STABLE_DIFFUSION_BASELINE_CATEGORY): The baseline of the horde model to update.
            load_state (ModelLoadState): The load state of the model.
            last_job_referenced (ImageGenerateJobPopResponse | None, optional): The last job referenced by this \
                 process. Defaults to None.
        """
        if horde_model_name is not None:
            self[process_id].recently_unloaded_from_ram = False

        self[process_id].loaded_horde_model_name = horde_model_name
        self[process_id].loaded_horde_model_baseline = horde_model_baseline

        self[process_id].last_received_timestamp = time.time()
        if last_job_referenced is not None:
            if (
                self[process_id].last_job_referenced is not None
                and last_job_referenced != self[process_id].last_job_referenced
            ):
                logger.debug(f"Resetting heartbeat for process {process_id}")
                self.reset_heartbeat_state(process_id)
            self[process_id].last_job_referenced = last_job_referenced

    def on_model_ram_clear(
        self,
        process_id: int,
        clear_job_reference: bool = False,
    ) -> None:
        """Update the model load state for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
            clear_job_reference: If True, also clear ``last_job_referenced``.  This should
                only be done for manager-initiated unloads where the process is being
                explicitly taken off a job.  Process-initiated unloads (part of the
                inference-failure recovery cycle) should leave the reference intact so that
                the orphan-job detector does not prematurely fault the retry.
        """
        self[process_id].loaded_horde_model_name = None
        self[process_id].loaded_horde_model_baseline = None
        if clear_job_reference:
            self[process_id].last_job_referenced = None
        self[process_id].recently_unloaded_from_ram = True
        self[process_id].last_received_timestamp = time.time()

    def reset_heartbeat_state(self, process_id: int) -> None:
        """Reset the heartbeat state for the given process ID.

        Args:
            process_id (int): The ID of the process to update.
        """
        logger.debug(f"Resetting heartbeat for process {process_id}")
        self[process_id].last_heartbeat_delta = 0
        self[process_id].last_heartbeat_timestamp = time.time()
        self[process_id].heartbeats_inference_steps = 0
        self[process_id].last_heartbeat_percent_complete = None

        # Reset progress tracking for new job
        self[process_id].last_progress_timestamp = time.time()
        self[process_id].last_progress_value = None
        self[process_id].last_inference_step_timestamp = None

    def delete_safety_processes(self) -> None:
        """Clear all safety processes."""
        ids_to_delete = []
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY:
                ids_to_delete.append(p.process_id)

        for process_id in ids_to_delete:
            logger.debug(f"Deleting safety process {process_id} from process map")
            self.pop(process_id)

    def is_stuck_on_inference(
        self,
        process_id: int,
        inference_step_timeout: int,
        no_step_heartbeat_timeout: int | None = None,
        zero_progress_timeout: float | None = None,
    ) -> bool:
        """Return true if the process is actively doing inference but progress has stalled.

        This detects jobs that are stuck in the INFERENCE_STARTING or INFERENCE_PROCESSING state with:
        1. Progress not advancing for timeout period (stuck at the same percentage after having
           advanced past 0 % — the pre-first-step phase is deliberately exempt because model
           loading into VRAM routinely takes longer than inference_step_timeout; it is covered
           by check 2 instead), OR
        2. Progress stuck at 0 % (or never reported) for zero_progress_timeout seconds (or
           no_step_heartbeat_timeout if zero_progress_timeout is None) — fastest early-stall
           detection — see detailed note below, OR
        3. A single diffusion step taking longer than MAX_INFERENCE_STEP_TIMEOUT seconds, OR
        4. No heartbeat received for timeout period (including last step / VAE decode phase)

        Detecting INFERENCE_PROCESSING stalls is critical: a process holding the inference semaphore
        while stuck prevents other processes from acquiring it, leaving them permanently stuck in
        INFERENCE_STARTING while waiting on the semaphore.

        Args:
            process_id: The ID of the process to check.
            inference_step_timeout: Timeout (seconds) for the progress-stalled and between-step
                heartbeat checks.
            no_step_heartbeat_timeout: If provided, and the process is in INFERENCE_PROCESSING with
                zero step-heartbeats received so far, use this shorter timeout for the no-heartbeat
                check instead of ``inference_step_timeout``.  This catches two cases quickly:
                (a) the process crashed before completing even one diffusion step, and
                (b) the process is in the VAE decode / post-processing phase after all steps
                    completed and sent a 100 % PIPELINE_STATE_CHANGE heartbeat that reset the
                    step counter.
                For case (b), this timeout must be >= ``VAE_SEMAPHORE_TIMEOUT`` (300 s) to
                avoid killing a legitimately-running VAE decode, which blocks on the semaphore
                without sending heartbeats.
                Do NOT apply this shorter fast-path to INFERENCE_STARTING: a process blocked
                on semaphore acquisition cannot send heartbeats yet, so using the shorter
                timeout there would be a false positive.
            zero_progress_timeout: If provided, and the process is in INFERENCE_PROCESSING with
                ``last_progress_value == 0`` (progress reported but never advanced), use this
                timeout for check 2 instead of falling back to ``no_step_heartbeat_timeout``.
                This handles two sub-cases that the existing per-step check (check 3) cannot catch:
                (i)  The ComfyUI callback fires repeatedly at step 0/N (percent = 0 %),
                     refreshing ``last_inference_step_timestamp`` on every call and preventing
                     the 30 s per-step timeout from ever triggering, while denoising never
                     actually starts.
                (ii) The callback fires via the PIPELINE_STATE_CHANGE path (no
                     ``comfyui_progress`` data), so ``last_inference_step_timestamp`` is None
                     and the per-step check is skipped entirely; the background heartbeat
                     thread then keeps ``last_heartbeat_timestamp`` fresh every 30 s, which
                     also prevents the no-heartbeat check (check 4) from firing.
                Unlike ``no_step_heartbeat_timeout``, this value is safe to set well below
                300 s because VAE decode only occurs at 100 % progress — a process stuck at
                0 % is never in the VAE decode phase.  Defaults to ``no_step_heartbeat_timeout``
                if not provided (backward-compatible).
                Do NOT apply this check to INFERENCE_STARTING (same reason as
                ``no_step_heartbeat_timeout``).
        """
        state = self[process_id].last_process_state
        if state not in (
            HordeProcessState.INFERENCE_STARTING,
            HordeProcessState.INFERENCE_PROCESSING,
        ):
            return False

        # Check if we're getting heartbeats but progress isn't advancing.
        # Skip this check when progress is already at 100 %: once all diffusion steps are
        # complete no further progress increments are expected (the process is in the VAE
        # decode phase), so a stalled 100 % value is normal behaviour.  The heartbeat-based
        # checks below are sufficient to detect genuine failures at that stage.
        # Also skip while progress is still at 0 % (or no progress has been reported at all):
        # before the first diffusion step the process is legitimately busy loading the model
        # into VRAM, which routinely takes longer than inference_step_timeout (30 s by
        # default) — killing at that point faulted perfectly healthy jobs.  The 0 %/None
        # phase is governed by check 2 below (zero_progress_timeout, 120 s) and by the
        # heartbeat-based checks, exactly as documented on ZERO_PROGRESS_TIMEOUT.
        time_since_progress = time.time() - self[process_id].last_progress_timestamp
        if time_since_progress > inference_step_timeout and self[process_id].last_progress_value not in (
            None,
            0,
            100,
        ):
            # Progress advanced past 0 % but has now stalled for too long - job is stuck
            return True

        # Faster stall detection when progress is stuck at exactly 0 %.
        # A process in INFERENCE_PROCESSING that has reported 0 % progress but has never
        # advanced means no real diffusion step has produced any output.  The background
        # heartbeat thread keeps last_heartbeat_timestamp fresh (preventing check 4 below),
        # and repeated INFERENCE_STEP callbacks at step 0/N keep last_inference_step_timestamp
        # fresh (preventing check 3 below).  The only safeguard would otherwise be check 1
        # above at the full inference_step_timeout.
        # Use zero_progress_timeout if provided; otherwise fall back to no_step_heartbeat_timeout.
        # zero_progress_timeout can be much shorter than no_step_heartbeat_timeout because
        # VAE decode (the reason no_step_heartbeat_timeout must stay at 300 s) only ever
        # happens at 100 % progress, not at 0 %.
        _zero_progress_effective = zero_progress_timeout if zero_progress_timeout is not None else no_step_heartbeat_timeout
        # last_progress_value None is treated like 0: it means no progress callback has fired
        # with a percentage yet (e.g. only PIPELINE_STATE_CHANGE heartbeats), which is the
        # same not-yet-produced-any-output situation — and VAE decode (the reason the longer
        # heartbeat timeouts exist) can likewise never be running before the first step.
        if (
            state == HordeProcessState.INFERENCE_PROCESSING
            and _zero_progress_effective is not None
            and self[process_id].last_progress_value in (0, None)
            and time_since_progress > _zero_progress_effective
        ):
            return True

        # Check if a single diffusion step is taking too long.
        # last_inference_step_timestamp is set only by INFERENCE_STEP heartbeats, so background
        # PIPELINE_STATE_CHANGE heartbeats do not mask a stuck step.  Skip this check at 100 %
        # progress (VAE decode phase) since no further steps are expected after all steps complete.
        if (
            state == HordeProcessState.INFERENCE_PROCESSING
            and self[process_id].last_inference_step_timestamp is not None
            and self[process_id].last_progress_value != 100
            and time.time() - self[process_id].last_inference_step_timestamp > self.MAX_INFERENCE_STEP_TIMEOUT
        ):
            return True

        # Check if no heartbeat received for timeout period.
        # Use the actual elapsed time since the last heartbeat, not the delta between the last
        # two heartbeats. last_heartbeat_delta is only updated when a heartbeat arrives, so it
        # stays at its previous (normal) value when the process stops responding entirely.
        # Note: We check all heartbeat types, not just INFERENCE_STEP, to catch
        # jobs stuck in the last step (VAE decode) which send PIPELINE_STATE_CHANGE heartbeats
        time_since_heartbeat = time.time() - self[process_id].last_heartbeat_timestamp

        # When the process is in INFERENCE_PROCESSING and has not yet received any step-level
        # heartbeats (heartbeats_inference_steps == 0), it is either:
        #   * stuck before the first diffusion step (e.g. crashed / hung right after acquiring
        #     the semaphore), or
        #   * in the VAE decode phase (all steps done; the 100 % PIPELINE_STATE_CHANGE heartbeat
        #     reset the counter to 0).
        # In both cases the last heartbeat timestamp was just refreshed, so we time out from
        # that fresh baseline.  Use the shorter no_step_heartbeat_timeout if provided.
        # We deliberately skip this for INFERENCE_STARTING because a process that is blocked
        # waiting to acquire the semaphore is unable to send heartbeats and would falsely trigger.
        if (
            state == HordeProcessState.INFERENCE_PROCESSING
            and self[process_id].heartbeats_inference_steps == 0
            and no_step_heartbeat_timeout is not None
            and time_since_heartbeat > no_step_heartbeat_timeout
        ):
            return True

        return time_since_heartbeat > inference_step_timeout

    def num_inference_processes(self) -> int:
        """Return the number of inference processes."""
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.INFERENCE:
                count += 1
        return count

    def num_loaded_inference_processes(self) -> int:
        """Return the number of inference processes that haven't been ended."""
        count = 0
        for p in self.values():
            if (
                p.process_type == HordeProcessType.INFERENCE
                and p.last_process_state != HordeProcessState.PROCESS_ENDING
                and p.last_process_state != HordeProcessState.PROCESS_ENDED
            ):
                count += 1
        return count

    def num_available_inference_processes(self) -> int:
        """Return the number of inference processes that are available to accept jobs."""
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.INFERENCE and not p.is_process_busy():
                count += 1
        return count

    def num_starting_processes(self) -> int:
        """Return the number of processes that are currently starting."""
        count = 0
        for p in self.values():
            if p.last_process_state == HordeProcessState.PROCESS_STARTING:
                count += 1
        return count

    def keep_single_inference(
        self,
        *,
        stable_diffusion_model_reference: StableDiffusion_ModelReference,
        post_process_job_overlap: bool,
    ) -> tuple[bool, str]:
        """Return true if we should keep only a single inference process running.

        This is used to prevent overloading the system with inference processes, such as with batched jobs.
        """
        for p in self.values():
            # We only parallelize if we have a currently running inference with n_iter > 1
            if p.batch_amount > 1 and (
                p.last_process_state == HordeProcessState.INFERENCE_STARTING
                or p.last_process_state == HordeProcessState.INFERENCE_PROCESSING
                or p.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
            ):
                return True, "Batched job"

            if (
                (
                    p.last_process_state == HordeProcessState.INFERENCE_STARTING
                    or p.last_process_state == HordeProcessState.INFERENCE_PROCESSING
                    or (
                        p.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
                        and not post_process_job_overlap
                    )
                )
                and p.last_job_referenced is not None
                and p.last_job_referenced.model in VRAM_HEAVY_MODELS
            ):
                return True, "VRAM heavy model"

            if (
                p.last_job_referenced is not None
                and p.last_job_referenced.payload.workflow in KNOWN_CONTROLNET_WORKFLOWS
            ):
                model = p.last_job_referenced.model
                if model is None:
                    logger.error(
                        f"Model is None for process {p.process_id} but workflow is "
                        f"{p.last_job_referenced.payload.workflow}",
                    )
                    continue

                model_info = stable_diffusion_model_reference.root.get(model)
                if model_info is None:
                    logger.debug(f"Model {model} not found in stable diffusion model reference. Is it a custom model?")
                    continue

                if model_info.baseline == STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_xl and (
                    p.can_accept_job()
                    or p.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
                ):
                    return True, "ControlNet XL"

            if p.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING and not post_process_job_overlap:
                return True, "Post processing overlap"

            if p.can_accept_job():
                continue

        return False, "None"

    def get_inference_processes(self) -> list[HordeProcessInfo]:
        """Return a list of all inference processes."""
        return [p for p in self.values() if p.process_type == HordeProcessType.INFERENCE]

    def get_first_available_inference_process(
        self,
        disallowed_processes: list[int] | None = None,
    ) -> HordeProcessInfo | None:
        """Return the first available inference process, or None if there are none available."""
        if disallowed_processes is None:
            disallowed_processes = []

        for p in self.values():
            if (
                p.process_type == HordeProcessType.INFERENCE
                and (
                    p.last_process_state == HordeProcessState.WAITING_FOR_JOB
                    or p.last_process_state == HordeProcessState.MODEL_PRELOADED
                )
                and p.loaded_horde_model_name is None
                and p.process_id not in disallowed_processes
            ):
                return p

        for p in self.values():
            if p.process_type == HordeProcessType.INFERENCE and p.can_accept_job():
                if p.process_id in disallowed_processes:
                    continue
                return p

        return None

    def _get_first_inference_process_to_kill(
        self,
        disallowed_processes: list[int] | None = None,
    ) -> HordeProcessInfo | None:
        """Return the first inference process eligible to be killed, or None if there are none.

        Used during shutdown.
        """
        if disallowed_processes is None:
            disallowed_processes = []

        for p in self.values():
            if p.process_type != HordeProcessType.INFERENCE:
                continue

            if p.process_id in disallowed_processes:
                continue

            if p.is_process_busy():
                continue

            if p.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED):
                continue

            return p

        return None

    def get_safety_process(self) -> HordeProcessInfo | None:
        """Return the safety process."""
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY:
                return p
        return None

    def num_safety_processes(self) -> int:
        """Return the number of safety processes."""
        count = 0
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY:
                count += 1
        return count

    def num_loaded_safety_processes(self) -> int:
        """Return the number of safety processes that are loaded."""
        count = 0
        for p in self.values():
            if (
                p.process_type == HordeProcessType.SAFETY
                and p.last_process_state != HordeProcessState.PROCESS_STARTING
                and p.last_process_state != HordeProcessState.PROCESS_ENDING
                and p.last_process_state != HordeProcessState.PROCESS_ENDED
            ):
                count += 1

        return count

    def get_first_available_safety_process(self) -> HordeProcessInfo | None:
        """Return the first available safety process, or None if there are none available."""
        for p in self.values():
            if p.process_type == HordeProcessType.SAFETY and p.last_process_state == HordeProcessState.WAITING_FOR_JOB:
                return p
        return None

    def get_process_by_horde_model_name(self, horde_model_name: str) -> HordeProcessInfo | None:
        """Return the process that has the given horde model loaded, or None if there is none.

        When multiple processes have the same model loaded (e.g. one MODEL_PRELOADED and one
        running inference), the process that can currently accept a job is returned first.
        If no process can accept a job, the first matching process is returned as a fallback.
        """
        first_match: HordeProcessInfo | None = None
        for p in self.values():
            if p.loaded_horde_model_name == horde_model_name:
                if p.can_accept_job():
                    return p
                if first_match is None:
                    first_match = p
        return first_match

    def num_busy_processes(self) -> int:
        """Return the number of processes that are actively engaged in a task.

        This does not include processes which are starting up or shutting down, or in a faulted state.
        """
        count = 0
        for p in self.values():
            if p.is_process_busy():
                count += 1
        return count

    def num_busy_with_inference(self) -> int:
        """Return the number of processes that are actively engaged in an inference task."""
        count = 0
        for p in self.values():
            if p.last_process_state in (
                HordeProcessState.INFERENCE_STARTING,
                HordeProcessState.INFERENCE_PROCESSING,
            ):
                count += 1
        return count

    def num_busy_with_post_processing(self) -> int:
        """Return the number of processes that are actively engaged in a post-processing task."""
        count = 0
        for p in self.values():
            if (
                p.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
                or p.last_process_state == HordeProcessState.POST_PROCESSING_STARTING
            ):
                count += 1
        return count

    def num_preloading_processes(self) -> int:
        """Return the number of processes that are preloading models."""
        count = 0
        for p in self.values():
            if p.last_process_state == HordeProcessState.MODEL_PRELOADING:
                count += 1
        return count

    def num_preloaded_processes(self) -> int:
        """Return the number of processes that have preloaded models."""
        count = 0
        for p in self.values():
            if p.last_process_state == HordeProcessState.MODEL_PRELOADED:
                count += 1
        return count

    def __repr__(self) -> str:
        """Return a string representation of the process map."""
        base_string = "Processes: "
        for string in self.get_process_info_strings():
            base_string += string

        return base_string

    def get_process_info_strings(self) -> list[str]:
        """Return a list of strings containing information about each process."""
        info_strings = []
        current_time = time.time()
        for process_id, process_info in self.items():
            if process_info.process_type == HordeProcessType.INFERENCE:
                time_passed_seconds = round((current_time - process_info.last_received_timestamp), 2)
                safe_last_control_flag = (
                    process_info.last_control_flag.name if process_info.last_control_flag is not None else None
                )

                process_state_detail = process_info.last_process_state.name

                if (
                    process_info.last_heartbeat_percent_complete is not None
                    and process_info.last_job_referenced is not None
                ):
                    percent_detail = (
                        f"{process_info.last_heartbeat_percent_complete}% of "
                        f"{process_info.last_job_referenced.payload.ddim_steps} steps "
                        f"using {process_info.last_job_referenced.payload.sampler_name}"
                    )
                    if process_info.last_job_referenced.payload.n_iter > 1:
                        percent_detail += f" ({process_info.last_job_referenced.payload.n_iter}x batch)"
                    # During post-processing and other non-processing states, the process state
                    # advances beyond INFERENCE_PROCESSING but last_heartbeat_percent_complete
                    # can stay at 100 %.  Include the state name so the console reflects the same
                    # job/process state as the webui.
                    if process_info.last_process_state == HordeProcessState.INFERENCE_PROCESSING:
                        process_state_detail = percent_detail
                    else:
                        process_state_detail = f"{process_info.last_process_state.name} ({percent_detail})"

                horde_model_name_and_baseline = (
                    f"<u>{process_info.loaded_horde_model_name}</u> {process_info.loaded_horde_model_baseline})"
                    if process_info.loaded_horde_model_name is not None
                    else "No model loaded"
                )
                last_heartbeat_delta_now = round((current_time - process_info.last_heartbeat_timestamp), 2)
                info_strings.append(
                    (
                        f"Process {process_id} ({process_state_detail}) "
                        f"({horde_model_name_and_baseline}) "
                        f"<fg #7b7d7d>[last message: {time_passed_seconds} secs ago: {safe_last_control_flag} "
                        f"heartbeat delta: {last_heartbeat_delta_now}]</>"
                    ),
                    # f"ram: {process_info.ram_usage_bytes} vram: {process_info.vram_usage_bytes} ",
                )

            else:
                info_strings.append(
                    f"Process {process_id}: ({process_info.process_type.name}) "
                    f"{process_info.last_process_state.name} ",
                )

        return info_strings

    def all_waiting_for_job(self) -> bool:
        """Return true if all processes are waiting for a job."""
        return all(
            p.last_process_state in [HordeProcessState.WAITING_FOR_JOB, HordeProcessState.MODEL_PRELOADED]
            for p in self.values()
        )


class TorchDeviceInfo(BaseModel):
    """Contains information about a torch device."""

    device_name: str
    device_index: int
    total_memory: int


class TorchDeviceMap(RootModel[dict[int, TorchDeviceInfo]]):
    """A mapping of device IDs to TorchDeviceInfo objects. Contains some helper methods."""


class HordeJobInfo(BaseModel):  # TODO: Split into a new file
    """Contains information about a job that has been generated.

    It is used to track the state of the job as it goes through the safety process and \
        then when it is returned to the requesting user.
    """

    sdk_api_job_info: ImageGenerateJobPopResponse
    """The API response which has all of the information about the job as sent by the API."""
    job_image_results: list[HordeImageResult] | None = None
    """A list of base64 encoded images and their generation faults that are the result of the job."""
    state: GENERATION_STATE | None
    """The state of the job to send to the API."""
    censored: bool | None = None
    """Whether or not the job was censored. This is set by the safety process."""

    time_popped: float
    time_submitted: float | None = None

    time_to_generate: float | None = None
    """The time it took to generate the job. This is set by the inference process."""

    time_to_download_aux_models: float | None = None

    # ! IMPORTANT: Start own code
    sanitized_negative_prompt: str | None = None
    """The sanitized negative prompt used for inference, if any."""

    inference_completed_timestamp: float | None = None
    """Timestamp when inference completed, used to track the order of job completions for preview display."""
    # ! IMPORTANT: End own code

    retry_count: int = 0
    """The number of times this job has been retried after faulting."""

    @property
    def is_job_checked_for_safety(self) -> bool:
        """Return true if the job has been checked for safety."""
        return self.censored is not None

    @property
    def images_base64(self) -> list[str]:
        """Return a list containing all base64 images."""
        if self.job_image_results is None:
            return []
        return [r.image_base64 for r in self.job_image_results]

    def fault_job(self) -> None:
        """Mark the job as faulted."""
        self.state = GENERATION_STATE.faulted
        self.job_image_results = None


class JobSubmitState(enum.Enum):  # TODO: Split into a new file
    """The state of a job submit process."""

    PENDING = auto()
    """The job submit still needs to be done or retried."""
    SUCCESS = auto()
    """The job submit finished succesfully."""
    FAULTED = auto()
    """The job submit faulted for some reason."""


class PendingJob(BaseModel):
    """Base class for all PendingJobs async tasks."""

    state: JobSubmitState = JobSubmitState.PENDING
    max_submit_retries: int = 3
    _consecutive_failed_job_submits: int = 0

    @property
    def is_finished(self) -> bool:
        """Return true if the job submit has finished."""
        return self.state != JobSubmitState.PENDING

    @property
    def is_faulted(self) -> bool:
        """Return true if the job submit has faulted."""
        return self.state == JobSubmitState.FAULTED

    @property
    def retry_attempts_string(self) -> str:
        """Return a string containing the number of consecutive failed job submits and the maximum allowed."""
        return f"{self._consecutive_failed_job_submits}/{self.max_submit_retries}"

    def retry(self) -> None:
        """Mark the job as needing to be retried. Fault the job if it has been retried too many times."""
        self._consecutive_failed_job_submits += 1
        if self._consecutive_failed_job_submits > self.max_submit_retries:
            self.state = JobSubmitState.FAULTED

    def succeed(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Mark the job as successfully submitted."""
        self.state = JobSubmitState.SUCCESS

    def fault(self) -> None:
        """Mark the job as faulted."""
        self.state = JobSubmitState.FAULTED


class PendingSubmitJob(PendingJob):  # TODO: Split into a new file
    """Information about a job to submit to the horde."""

    completed_job_info: HordeJobInfo
    gen_iter: int
    kudos_reward: int = 0
    kudos_per_second: float = 0.0

    @property
    def image_result(self) -> HordeImageResult | None:
        """Return the image result for the job."""
        results = self.completed_job_info.job_image_results
        if results is None or self.gen_iter >= len(results):
            return None
        return results[self.gen_iter]

    @property
    def job_id(self) -> JobID:
        """Return the job ID for the job."""
        ids = self.completed_job_info.sdk_api_job_info.ids
        if not ids:
            raise IndexError(f"sdk_api_job_info.ids is empty or None for gen_iter={self.gen_iter}")
        if self.gen_iter >= len(ids):
            raise IndexError(
                f"gen_iter={self.gen_iter} out of range for sdk_api_job_info.ids (len={len(ids)})"
            )
        return ids[self.gen_iter]

    @property
    def r2_upload(self) -> str:
        """Return the r2 upload for the job."""
        r2_uploads = self.completed_job_info.sdk_api_job_info.r2_uploads
        if not r2_uploads:
            return ""
        if self.gen_iter >= len(r2_uploads):
            raise IndexError(
                f"gen_iter={self.gen_iter} out of range for sdk_api_job_info.r2_uploads (len={len(r2_uploads)})"
            )
        return r2_uploads[self.gen_iter]

    @property
    def batch_count(self) -> int:
        """Return the number of jobs in the batch."""
        return len(self.completed_job_info.sdk_api_job_info.ids)

    @override
    def succeed(self, kudos_reward: int = 0, kudos_per_second: float = 0) -> None:
        """Mark the job as successfully submitted.

        Args:
            kudos_reward: The amount of kudos to reward the user.
            kudos_per_second: The amount of kudos per second to reward the user.
        """
        self.kudos_reward = kudos_reward
        self.kudos_per_second = kudos_per_second
        super().succeed()


class NextJobAndProcess(BaseModel):
    """Contains information about the next job to process and the process to process it with."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    next_job: ImageGenerateJobPopResponse
    process_with_model: HordeProcessInfo
    skipped_line: bool = False
    skipped_line_for: ImageGenerateJobPopResponse | None


class LRUCache:
    """A simple LRU cache. This is used to keep track of the most recently used models."""

    def __init__(self, capacity: int) -> None:
        """Initializes the LRU cache.

        Args:
            capacity: The maximum number of elements that the cache can hold.
        """
        self.capacity = capacity
        self.cache: collections.OrderedDict[str, ModelInfo | None] = collections.OrderedDict()

    def append(self, key: str) -> object:
        """Adds an element to the LRU cache, and potentially bumps one from the cache.

        Args:
            key: The key to add to the cache.

        Returns:
            The bumped element, if there was one.
        """
        bumped = None
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            bumped, _ = self.cache.popitem(last=False)
        self.cache[key] = None
        return bumped


class APIWorkerMessage(BaseModel):
    """A message sent to the worker from the API."""

    message_id: str
    """The ID of the message."""

    message_text: str | None
    """The text of the message."""

    message_origin: str | None
    """The origin (author) of the message."""

    message_expiry: str | None
    """The expiry time of the message."""


class HordeWorkerProcessManager:
    """Manages and controls processes to act as a horde worker."""

    # Constants for failing models tracking
    FAILED_MODELS_REPORT_INTERVAL_SECONDS = 300  # 5 minutes
    MAX_FAILING_MODELS_TO_DISPLAY = 10

    # Constants for worker config display
    WORKER_CONFIG_REPORT_INTERVAL_SECONDS = 300  # 5 minutes

    # Retained for reference; the live threshold now comes from bridge_data.waiting_for_job_timeout
    # (default 600 s, configurable via the webui Timeouts page).
    _WAITING_FOR_JOB_STALE_THRESHOLD = 600  # 10 minutes

    # Constants for job retry logic
    MAX_JOB_RETRIES = 1  # Number of retries for faulted jobs
    MAX_SUBMIT_RETRIES = 3  # Number of times a job submission can be retried before being marked failed

    # Constants for preload-stuck cooldown logic.
    # When a model causes MODEL_PRELOADING to hang this many times within
    # _PRELOAD_STUCK_FAILURE_WINDOW seconds, it is placed in a cooldown and new jobs for
    # that model are immediately faulted (rather than silently retried) for
    # _PRELOAD_STUCK_COOLDOWN seconds.  This prevents the worker from cycling indefinitely
    # through a model that cannot be loaded on this machine.
    _PRELOAD_STUCK_FAILURE_THRESHOLD: int = 2
    _PRELOAD_STUCK_FAILURE_WINDOW: float = 600.0  # seconds
    _PRELOAD_STUCK_COOLDOWN: float = 600.0  # seconds

    # Constants for inference-failure cooldown logic.
    # When a model causes this many permanently-faulted jobs within
    # _INFERENCE_FAILURE_WINDOW seconds it is placed in a cooldown: job-pop requests
    # will exclude that model for _INFERENCE_FAILURE_COOLDOWN seconds.  This prevents
    # the Horde server from penalizing the worker (or placing it in maintenance mode)
    # due to excessive fault reports for a broken model.
    _INFERENCE_FAILURE_THRESHOLD: int = 3
    _INFERENCE_FAILURE_WINDOW: float = 1200.0  # seconds (20 minutes)
    _INFERENCE_FAILURE_COOLDOWN: float = 3600.0  # seconds (1 hour)

    # Constants for webui log capture
    # Compiled regex pattern for removing ANSI escape codes from logs
    ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    _MAX_CONSOLE_LOGS_BUFFER = 2000  # Maximum number of console logs to keep in memory buffer
    _WEBUI_CONSOLE_LOGS_LIMIT = 1000  # Number of recent logs to send to webui from buffer
    _MAX_ERRORS_HISTORY = 1000  # Maximum number of error messages to keep in history (memory safety cap)

    # States that indicate inference is done and the result is being post-processed or submitted.
    # In all these states the progress bar must be pinned at 100% so it never goes backwards.
    _WEBUI_POST_INFERENCE_STATES: frozenset[HordeProcessState] = frozenset(
        {
            HordeProcessState.INFERENCE_POST_PROCESSING,
            HordeProcessState.POST_PROCESSING_STARTING,
            HordeProcessState.INFERENCE_COMPLETE,
            HordeProcessState.POST_PROCESSING_COMPLETE,
            HordeProcessState.SAFETY_STARTING,
            HordeProcessState.SAFETY_EVALUATING,
            HordeProcessState.SAFETY_COMPLETE,
            HordeProcessState.RESULT_SAVING,
            HordeProcessState.RESULT_SAVED,
            HordeProcessState.RESULT_SUBMITTING,
            HordeProcessState.RESULT_SUBMITTED,
        },
    )

    # Process states for which elapsed time is tracked via state-transition timing.
    # When a process leaves one of these states, the elapsed duration is accumulated
    # into ``_job_time_stats`` under the state's ``.name`` string so the WebUI statistics
    # page can display avg / max time per job state beyond just Inference / TOTAL / Download.
    _STATE_TRANSITION_TIMING_STATES: frozenset[HordeProcessState] = frozenset(
        {
            HordeProcessState.WAITING_FOR_JOB,
            HordeProcessState.MODEL_PRELOADING,
            HordeProcessState.MODEL_LOADING,
            HordeProcessState.INFERENCE_STARTING,
            HordeProcessState.POST_PROCESSING_STARTING,
            HordeProcessState.INFERENCE_POST_PROCESSING,
            HordeProcessState.SAFETY_STARTING,
            HordeProcessState.SAFETY_EVALUATING,
            HordeProcessState.RESULT_SAVING,
            HordeProcessState.RESULT_SUBMITTING,
        },
    )

    bridge_data: reGenBridgeData
    """The bridge data for this worker."""

    horde_model_reference_manager: ModelReferenceManager
    """The model reference manager for this worker."""

    _max_inference_processes: int
    """Backing store for the max_inference_processes property.  Updated by __init__ and at runtime
    by set_max_active_models()."""

    _max_active_models_override: int | None
    """Runtime override for max_inference_processes set via the web UI.  When not None, takes
    precedence over the value derived from bridge_data.queue_size + max_threads."""

    _max_concurrent_inference_processes: int
    """The maximum number of inference processes that can run jobs concurrently. \
        This is set at initialization to prevent changing the value at runtime."""

    @property
    def max_inference_processes(self) -> int:
        """The maximum number of inference processes that can be active.

        When a runtime override has been set via :meth:`set_max_active_models`, that value
        is returned; otherwise the startup value is used.
        """
        if self._max_active_models_override is not None:
            return self._max_active_models_override
        return self._max_inference_processes

    @property
    def max_concurrent_inference_processes(self) -> int:
        """The maximum number of inference processes that can run jobs concurrently."""
        return self._max_concurrent_inference_processes

    _queue_size_override: int | None
    """Runtime override for the queue size set via the web UI.  When not None, takes
    precedence over bridge_data.queue_size in the max_queue_size property."""

    max_safety_processes: int
    """The maximum number of safety processes that can run at once."""
    max_download_processes: int
    """The maximum number of download processes that can run at once."""

    num_processes_launched: int = 0
    """The number of processes that have been launched."""

    total_ram_bytes: int
    """The total amount of RAM on the system."""

    @property
    def total_ram_megabytes(self) -> int:
        """The total amount of RAM on the system in megabytes."""
        return self.total_ram_bytes // 1024 // 1024

    @property
    def total_ram_gigabytes(self) -> int:
        """The total amount of RAM on the system in gigabytes."""
        return self.total_ram_bytes // 1024 // 1024 // 1024

    target_ram_overhead_bytes: int
    """The target amount of RAM to keep free."""

    target_vram_overhead_bytes_map: Mapping[int, int] | None = None

    @property
    def max_queue_size(self) -> int:
        """The maximum number of jobs that can be queued.

        When a runtime override has been set via :meth:`set_max_queue_size`, that value
        is returned; otherwise bridge_data.queue_size is used.
        """
        if self._queue_size_override is not None:
            return self._queue_size_override
        return self.bridge_data.queue_size

    @property
    def current_queue_size(self) -> int:
        """The current number of jobs that are queued (not yet started)."""
        return max(0, len(self.jobs_pending_inference) - len(self.jobs_in_progress))

    @property
    def target_ram_bytes_used(self) -> int:
        """The target amount of RAM to use."""
        return self.total_ram_bytes - self.target_ram_overhead_bytes

    def get_process_total_ram_usage(self) -> int:
        """Return the total amount of RAM used by all processes."""
        total = 0
        for process_info in self._process_map.values():
            total += process_info.ram_usage_bytes
        return total

    jobs_lookup: dict[ImageGenerateJobPopResponse, HordeJobInfo]
    """The mapping of API responses to their corresponding worker job info."""

    jobs_in_progress: list[ImageGenerateJobPopResponse]
    """A list of jobs that are currently in progress."""

    job_faults: dict[JobID, list[GenMetadataEntry]]
    """A list of jobs that have exhibited faults and what kinds."""

    jobs_pending_safety_check: list[HordeJobInfo]
    """A list of jobs that were generated but have not yet been safety checked."""
    _jobs_safety_check_lock: Lock_Asyncio
    """The asyncio lock for the safety check queue."""

    jobs_being_safety_checked: list[HordeJobInfo]
    """The list of jobs that are currently being safety checked."""

    _num_jobs_faulted: int = 0
    """The number of jobs which were marked as faulted. This may not include jobs which failed for unknown reasons."""

    _faulted_jobs_history: list[dict[str, Any]]
    """A list of faulted jobs with their details for display in the webui."""
    _max_faulted_jobs_history: int = 20
    """Maximum number of faulted jobs to keep in history."""

    _errors_history: deque[str]
    """A list of recent error messages for display in the webui."""

    jobs_pending_submit: list[HordeJobInfo]
    """A list of HordeJobInfo objects containing the job, the state, and whether or not the job was censored."""

    _completed_jobs_lock: Lock_Asyncio
    """The asyncio lock for the completed jobs queue."""

    @property
    def num_jobs_total(self) -> int:
        """The total number of jobs that have been processed."""
        return (
            len(self.jobs_pending_inference)
            + len(self.jobs_in_progress)
            + len(self.jobs_pending_safety_check)
            + len(self.jobs_being_safety_checked)
            + len(self.jobs_pending_submit)
        )

    kudos_generated_this_session: float = 0
    """The amount of kudos generated this entire session."""
    kudos_events: list[tuple[float, float]]
    """A list of kudos events, each is a tuple of the time the event occurred and the amount of kudos generated."""
    image_events: list[tuple[float, int]]
    """A list of image completion events, each is a tuple of the time the event occurred and the number of images generated."""
    session_start_time: float = 0
    """The time at which the session started in epoch time."""

    _aiohttp_client_session: aiohttp.ClientSession
    """The aiohttp client session to use for making network calls."""

    stable_diffusion_reference: StableDiffusion_ModelReference | None
    """The class which contains the list of models from horde_model_reference."""

    def get_model_baseline(self, model_name: str) -> STABLE_DIFFUSION_BASELINE_CATEGORY | str | None:
        """Return the baseline of the model."""
        if self.stable_diffusion_reference is None:
            return None

        if model_name not in self.stable_diffusion_reference.root:
            return None

        return self.stable_diffusion_reference.root[model_name].baseline

    horde_client_session: AIHordeAPIAsyncClientSession | None
    """The context manager for the horde sdk client."""

    user_info: UserDetailsResponse | None = None
    """The user info for the user that this worker is logged in as."""
    _last_user_info_fetch_time: float = 0
    """The time at which the user info was last fetched."""
    _user_info_fetch_interval: float = 10
    """The number of seconds between each fetch of the user info."""

    _workers_details: list[dict[str, Any]]
    """Cached per-worker detail records fetched from the API for the User page."""
    _api_get_workers_details_interval: float = 120
    """The number of seconds between each fetch of individual worker details."""

    _process_map: ProcessMap
    """A mapping (dict) of process IDs to HordeProcessInfo objects. Contains some helper methods."""
    _horde_model_map: HordeModelMap
    """A mapping (dict) of horde model names to ModelInfo objects. Contains some helper methods."""
    _device_map: TorchDeviceMap
    """A mapping (dict) of device IDs to TorchDeviceInfo objects. Contains some helper methods."""

    _loop_interval: float = 0.20
    """The number of seconds to wait between each loop of the main process (inter process management) loop."""
    _api_call_loop_interval = 1
    """The number of seconds to wait between each loop of the main API call loop."""

    _api_get_user_info_interval = 60
    """The number of seconds to wait between each fetch of the user info."""

    _last_get_user_info_time: float = 0
    """The time at which the user info was last fetched."""

    @property
    def num_total_processes(self) -> int:
        """The total number of processes that can be running at once (inference, safety, and download)."""
        return self.max_inference_processes + self.max_safety_processes + self.max_download_processes

    _process_message_queue: ProcessQueue
    """A queue of messages sent from child processes."""

    jobs_pending_inference: deque[ImageGenerateJobPopResponse]
    """A deque of jobs that are waiting to be processed."""
    _jobs_pending_inference_lock: Lock_Asyncio
    """The asyncio lock for the job deque."""

    job_pop_timestamps: dict[ImageGenerateJobPopResponse, float]
    """A mapping of jobs to the time at which they were popped."""
    _job_pop_timestamps_lock: Lock_Asyncio
    """The asyncio lock for the job pop timestamps."""

    _inference_semaphore: BoundedSemaphore
    """A semaphore that limits the number of inference processes that can run at once.

    Using BoundedSemaphore ensures that an over-release (which would inflate available permits
    beyond max_threads) raises ValueError rather than silently succeeding.  Both the manager's
    _replace_inference_process() and the child inference process already catch ValueError on
    semaphore release, so the existing handlers prevent any permit inflation.
    """

    _vae_decode_semaphore: BoundedSemaphore

    _disk_lock: Lock_MultiProcessing
    """A lock to prevent multiple processes from accessing the disk at once."""

    _aux_model_lock: Lock_MultiProcessing
    """A lock to prevent multiple processes from accessing the auxiliary models at once (such as LoRas)."""

    _lru: LRUCache
    """A simple LRU cache. This is used to keep track of the most recently used models."""

    _amd_gpu: bool
    """Whether or not the GPU is an AMD GPU."""

    _directml: int | None
    """ID of the potential directml device."""

    _api_messages_received: dict[str, APIWorkerMessage]

    @property
    def post_process_job_overlap_allowed(self) -> bool:
        """Return true if post processing jobs are allowed to overlap."""
        return (
            self.bridge_data.moderate_performance_mode or self.bridge_data.high_performance_mode
        ) and self.bridge_data.post_process_job_overlap

    def __init__(
        self,
        *,
        ctx: BaseContext,
        bridge_data: reGenBridgeData,
        horde_model_reference_manager: ModelReferenceManager,
        target_ram_overhead_bytes: int = 9 * 1024 * 1024 * 1024,
        target_vram_overhead_bytes_map: Mapping[int, int] | None = None,
        max_safety_processes: int = 1,
        max_download_processes: int = 1,
        amd_gpu: bool = False,
        directml: int | None = None,
    ) -> None:
        """Initialise the process manager.

        Args:
            ctx (BaseContext): The multiprocessing context to use.
            bridge_data (reGenBridgeData): The bridge data for this worker.
            horde_model_reference_manager (ModelReferenceManager): The model reference manager for this worker.
            target_ram_overhead_bytes (int, optional): The target amount of RAM to keep free. \
                Defaults to 9 * 1024 * 1024 * 1024.
            target_vram_overhead_bytes_map (Mapping[int, int] | None, optional): The target amount of VRAM to keep \
                free. Defaults to None.
            max_safety_processes (int, optional): The maximum number of safety processes that can run at once. \
                Defaults to 1.
            max_download_processes (int, optional): The maximum number of download processes that can run at once. \
                Defaults to 1.
            amd_gpu (bool, optional): Whether or not the GPU is an AMD GPU. Defaults to False.
            directml (int, optional): ID of the potential directml device. Defaults to None.
        """
        self.session_start_time = time.time()
        self._last_pop_no_jobs_available_time = self.session_start_time
        self._last_job_submitted_time = self.session_start_time

        self.bridge_data = bridge_data
        logger.debug(f"Models to load: {bridge_data.image_models_to_load}")
        logger.debug(f"Custom Models to load: {bridge_data.custom_models}")

        # Store the full original model list so the webui can show disabled models.
        # Runtime-disabled models are tracked separately and removed from image_models_to_load.
        self._all_models_configured: list[str] = list(bridge_data.image_models_to_load)
        self._runtime_disabled_models: set[str] = set()

        # Restore any model enable/disable choices the user made via the WebUI in a
        # previous session.  This must run after _all_models_configured is set so
        # that env-variable overrides (models absent from the configured list) are
        # automatically respected.
        self._load_model_state_file()

        self.horde_model_reference_manager = horde_model_reference_manager

        # Initialize HTTP client session as None - will be set in _main_loop
        self.horde_client_session = None

        self._process_map = ProcessMap({})
        self._horde_model_map = HordeModelMap(root={})

        self.max_safety_processes = max_safety_processes
        self.max_download_processes = max_download_processes

        self._max_concurrent_inference_processes = bridge_data.max_threads
        self._inference_semaphore = BoundedSemaphore(self._max_concurrent_inference_processes, ctx=ctx)

        self._aux_model_lock = Lock_MultiProcessing(ctx=ctx)

        # Runtime overrides set via the web UI.  None means "use startup value".
        self._queue_size_override: int | None = None
        self._max_active_models_override: int | None = None

        # 0 is a sentinel meaning "use auto mode" — treat it the same as unset so the
        # startup derivation (queue_size + max_threads) is used as the initial value.
        if self.bridge_data.max_active_models == 0:
            self.bridge_data.max_active_models = None

        startup_max_active_models = self.bridge_data.max_active_models
        if startup_max_active_models is None:
            startup_max_active_models = self.bridge_data.queue_size + self.bridge_data.max_threads
        self._max_inference_processes = startup_max_active_models

        vae_decode_semaphore_max = 1

        if self.bridge_data.high_memory_mode:
            vae_decode_semaphore_max = self.max_inference_processes

        self._vae_decode_semaphore = BoundedSemaphore(vae_decode_semaphore_max, ctx=ctx)

        self._lru = LRUCache(self.max_inference_processes)

        self._amd_gpu = amd_gpu
        self._directml = directml

        self._replaced_due_to_maintenance = False
        # Whether job pops have been paused by the user via the web UI.
        self._job_pops_paused = False
        # Unix timestamp at which a timed pause should auto-expire (None = indefinite).
        self._job_pops_pause_until: float | None = None

        # Auto-mode flags set via the web UI.
        self._queue_size_auto: bool = False
        self._max_active_models_auto: bool = False
        self._inference_scale_down_requested: bool = False

        # Cached resource metrics used by the auto-tuning logic.
        # Updated each time update_webui_status() is called.
        self._last_total_vram_mb: float = 0.0
        self._last_system_vram_usage_mb: float = 0.0
        self._last_worker_vram_mb: float = 0.0

        # If there is only one model to load and only one inference process, then we can only run one job at a time
        # and there is no point in having more than one inference process
        if len(self.bridge_data.image_models_to_load) == 1 and self.max_concurrent_inference_processes == 1:
            self._max_inference_processes = 1

        self._disk_lock = Lock_MultiProcessing(ctx=ctx)

        self.jobs_lookup = {}
        self._jobs_lookup_lock = Lock_Asyncio()

        self.jobs_pending_submit = []
        self._completed_jobs_lock = Lock_Asyncio()

        self.jobs_pending_safety_check = []
        self.jobs_being_safety_checked = []
        self.job_faults = {}
        self._faulted_jobs_history = []
        self._faulted_jobs_per_phase: dict[str, int] = {}
        self._errors_history: deque[str] = deque(maxlen=self._MAX_ERRORS_HISTORY)
        # Tracks the length of _errors_history at the last webui update so the list
        # is only copied and sent when new errors have been added.
        self._errors_history_last_sent_len: int = -1

        self._workers_details: list[dict[str, Any]] = []
        self._last_sent_workers_details_cache_key: str | None = None

        self._jobs_safety_check_lock = Lock_Asyncio()

        self.target_vram_overhead_bytes_map = target_vram_overhead_bytes_map

        self.total_ram_bytes = psutil.virtual_memory().total

        # Keep at most half of system RAM as headroom, capped at the requested overhead (9 GB default).
        self.target_ram_overhead_bytes = min(int(self.total_ram_bytes / 2), target_ram_overhead_bytes)

        if any(model in VRAM_HEAVY_MODELS for model in self.bridge_data.image_models_to_load):
            # If the system ram is less than 24GB, then we're going to exit with an error
            if self.total_ram_bytes < (24 * 1024 * 1024 * 1024):
                raise ValueError(
                    "VRAM heavy models detected. Total RAM is less than 24GB. "
                    "This is not enough RAM to run the worker."
                    "Disable the large models by adding it to your `models_to_skip` or remove it from your "
                    "`models_to_load`. Large models include: " + ", ".join(VRAM_HEAVY_MODELS),
                )

            self.target_ram_overhead_bytes = min(self.target_ram_overhead_bytes, int(20 * 1024 * 1024 * 1024 / 2))

        if self.target_ram_overhead_bytes > self.total_ram_bytes:
            raise ValueError(
                f"target_ram_overhead_bytes ({self.target_ram_overhead_bytes}) is greater than "
                f"total_ram_bytes ({self.total_ram_bytes})",
            )

        self._status_message_frequency = bridge_data.stats_output_frequency

        logger.debug(f"Total RAM: {self.total_ram_bytes / 1024 / 1024 / 1024} GB")
        logger.debug(f"Target RAM overhead: {self.target_ram_overhead_bytes / 1024 / 1024 / 1024} GB")

        self.enable_performance_mode()
        if self.bridge_data.remove_maintenance_on_init:
            try:
                self.remove_maintenance()
            except Exception as e:
                logger.warning(e)
                logger.warning("Error trying to unset maintenance. Did this worker not exist yet?")

        # Get the total memory of each GPU
        import torch

        self._device_map = TorchDeviceMap(root={})
        for i in range(torch.cuda.device_count()):
            device = torch.cuda.get_device_properties(i)
            self._device_map.root[i] = TorchDeviceInfo(
                device_name=device.name,
                device_index=i,
                total_memory=device.total_memory,
            )

        self.jobs_in_progress = []

        self.jobs_pending_inference = deque()
        self._jobs_pending_inference_lock = Lock_Asyncio()

        # Cache for megapixelsteps calculation (performance optimization)
        # Initialize as valid with 0 since there are no pending jobs at startup
        self._cached_pending_megapixelsteps: int = 0
        self._megapixelsteps_cache_valid: bool = True

        self.job_pop_timestamps: dict[ImageGenerateJobPopResponse, float] = {}
        self._job_pop_timestamps_lock = Lock_Asyncio()

        self._process_message_queue = multiprocessing.Queue()

        # Child→parent messages are pulled off the multiprocessing queue by a dedicated
        # daemon thread and buffered in this deque, which the (asyncio) control loop then
        # drains non-blockingly.  Reading the queue directly from the event loop is unsafe:
        # a child SIGKILLed while writing to the queue (see _end_inference_process /
        # _replace_inference_process) can leave a *truncated* message in the queue pipe, and
        # Queue.get() — even with block=False, whose non-blocking guarantee only covers the
        # lock acquire and the readiness poll — then blocks forever inside _recv_bytes()
        # waiting for the rest of a payload that will never arrive.  That froze the entire
        # worker silently (no exception, no further log output).  With this design only the
        # reader thread can get stuck; _check_process_message_queue_health() detects the
        # stall and restarts the worker with a clear CRITICAL log instead.
        self._received_process_messages: deque[HordeProcessMessage] = deque()
        self._queue_reader_last_tick: float = time.time()
        self._queue_reader_thread = threading.Thread(
            target=self._queue_reader_loop,
            daemon=True,
            name="process-message-queue-reader",
        )
        self._queue_reader_thread.start()

        self.kudos_events: list[tuple[float, float]] = []
        self.image_events: list[tuple[float, int]] = []

        # Cumulative per-model image counts for the current session.
        self._images_per_model: dict[str, int] = {}

        self._api_messages_received = {}

        # Track models that have failed
        self._failed_models: dict[str, int] = {}
        self._last_failed_models_print_time: float = 0.0

        # Per-state job timing accumulators.
        # Maps state name → {"sum": float, "count": int, "max": float}
        self._job_time_stats: dict[str, dict[str, float | int]] = {}
        # Buffered per-state timings for the in-flight work currently associated with each process.
        self._pending_process_job_timings: dict[int, dict[str, float]] = {}
        # Buffered per-state timings for completed jobs awaiting final successful submission.
        self._pending_completed_job_timings: dict[ImageGenerateJobPopResponse, dict[str, float]] = {}

        # Honor AIWORKER_QUEUE_SIZE=0 and AIWORKER_MAX_ACTIVE_MODELS=0 as "activate auto mode".
        # This allows Docker/env-var users to opt into automatic tuning without editing YAML.
        if os.getenv("AIWORKER_QUEUE_SIZE") == "0":
            self._queue_size_auto = True
            self._queue_size_override = self._compute_auto_queue_size()
            logger.info(
                f"Queue size auto mode activated via AIWORKER_QUEUE_SIZE=0 "
                f"(initial value: {self._queue_size_override})",
            )
        if os.getenv("AIWORKER_MAX_ACTIVE_MODELS") == "0":
            self._max_active_models_auto = True
            _auto_count = self._compute_auto_max_active_models()
            self._max_active_models_override = _auto_count
            self.bridge_data.max_active_models = _auto_count
            os.environ["AIWORKER_MAX_ACTIVE_MODELS"] = str(_auto_count)
            self._lru.capacity = _auto_count
            self._inference_scale_down_requested = True
            logger.info(
                f"Max active models auto mode activated via AIWORKER_MAX_ACTIVE_MODELS=0 "
                f"(initial value: {_auto_count})",
            )

        # Per-model timing accumulators (session totals).
        # Maps model name → {"sum": float, "count": int, "max": float}
        # _time_per_step_per_model: inference_time / ddim_steps per job
        self._time_per_step_per_model: dict[str, dict[str, float | int]] = {}
        # _time_per_job_per_model: total job time (pop → submission) per job
        self._time_per_job_per_model: dict[str, dict[str, float | int]] = {}

        # Track per-model preload-stuck failure timestamps for cooldown logic.
        # Maps model name → deque of epoch timestamps when that model caused a MODEL_PRELOADING timeout.
        self._preload_stuck_failures: dict[str, deque[float]] = {}

        # Track per-model inference failure timestamps for inference-failure cooldown logic.
        # Maps model name → deque of epoch timestamps when that model caused a permanently-faulted job.
        self._inference_failures: dict[str, deque[float]] = {}
        # Rate-limiting state for the inference-failure cooldown log warning in api_job_pop.
        self._last_warned_inference_cooldown_models: frozenset[str] = frozenset()
        self._last_warned_inference_cooldown_at: float = 0.0

        # Track per-process-slot restart timestamps for crash-loop rate limiting.
        self._process_restart_history: dict[int, deque[float]] = {}

        # Whether an inference process has already been tasked with downloading the
        # legacy model reference databases (only the first one launched should).
        self._legacy_references_download_assigned = False

        # Track when an in-progress job was first observed orphaned (its handling process died,
        # was replaced, or hung before completing). Maps job-id str → epoch time first seen
        # orphaned, so _reap_orphaned_in_progress_jobs only faults after a short grace period.
        self._job_orphan_since: dict[str, float] = {}

        # Track last worker config print time
        self._last_worker_config_print_time: float = 0.0

        self.stable_diffusion_reference = None

        _references_load_started = time.monotonic()
        while self.stable_diffusion_reference is None:
            try:
                horde_model_reference_manager = ModelReferenceManager(
                    download_and_convert_legacy_dbs=False,
                    override_existing=False,
                )
                all_refs = horde_model_reference_manager.get_all_model_references(redownload_all=False)
                _sd_ref = all_refs.get(MODEL_REFERENCE_CATEGORY.stable_diffusion)

                if not isinstance(_sd_ref, StableDiffusion_ModelReference):
                    logger.error("Stable diffusion model references not found. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue

                self.stable_diffusion_reference = _sd_ref
            except Exception as e:
                logger.error(e)
                time.sleep(5)
        _references_load_elapsed = time.monotonic() - _references_load_started
        if _references_load_elapsed >= 5.0:
            logger.info(f"Model reference database loaded in {_references_load_elapsed:.1f}s")

        # Initialize web UI if enabled
        self.webui: WorkerWebUI | None = None  # noqa: F823
        self._last_image_base64: list[str] = []
        """The last generated images in base64 format for webui preview (supports batch jobs)."""
        self._last_image_job_timestamp: float = 0.0
        """Timestamp when the last preview image was set, to prevent older jobs from overwriting newer ones."""
        self._last_image_model: str = ""
        """Model name used for the last preview image."""
        self._last_image_safety: list[dict] = []
        """Per-image safety flags (is_nsfw, is_csam) for the last preview images."""
        self._console_logs: deque[str] = deque(maxlen=self._MAX_CONSOLE_LOGS_BUFFER)
        """Recent console logs for webui display."""
        self._log_handler_id: int | None = None
        """ID of the logger handler for capturing console logs."""

        # Persistent psutil.Process() handles for container CPU tracking. cpu_percent(interval=None)
        # computes a delta from the *previous* call on the same Process instance, so we keep the main
        # process plus child handles cached by PID across update_webui_status() calls.
        self._main_process: psutil.Process = psutil.Process()
        self._container_cpu_processes: dict[int, psutil.Process] = {self._main_process.pid: self._main_process}

        if self.bridge_data.enable_webui:
            # The web UI is an optional monitoring convenience: a failure to import or
            # construct it (missing/broken aiohttp, unexpected init error) must not take
            # down the worker — degrade to headless, mirroring the .start() guard in
            # _main_loop().
            try:
                from horde_worker_regen.webui.server import WorkerWebUI

                self.webui = WorkerWebUI(
                    port=self.bridge_data.webui_port,
                    update_interval=self.bridge_data.webui_update_interval,
                    db_path=self._get_model_state_file_path(),
                    data_retention_days=self.bridge_data.data_retention_days,
                )
                self.webui.set_delete_worker_callback(self._delete_worker)
                self.webui.set_job_pops_paused_callback(self.set_job_pops_paused)
                self.webui.set_max_queue_size_callback(self.set_max_queue_size)
                self.webui.set_max_active_models_callback(self.set_max_active_models)
                self.webui.set_queue_size_auto_mode_callback(self.set_queue_size_auto_mode)
                self.webui.set_max_active_models_auto_mode_callback(self.set_max_active_models_auto_mode)
                self.webui.set_setting_callback(self.apply_setting)
                self.webui.set_restart_program_callback(self.request_program_restart)
                self.webui.set_toggle_model_callback(self._toggle_model)
                logger.info(f"Web UI enabled on port {self.bridge_data.webui_port}")

                # Add a log handler to capture logs for webui with colored output.
                # Use the same format function and timestamp format as the standard
                # stderr console sink so the webui log display exactly matches the
                # standard console output (timestamp, level, message, coloring).
                webui_format_record = create_level_format_function(time_format="YYYY-MM-DD HH:mm:ss.SSS")

                self._log_handler_id = logger.add(
                    self._capture_log_for_webui,
                    format=webui_format_record,
                    level="INFO",
                    colorize=True,
                )
            except Exception as e:
                logger.error(
                    f"Web UI failed to initialise ({type(e).__name__}: {e}). "
                    "Continuing without the web UI — the worker will keep generating.",
                )
                self.webui = None

    def _capture_log_for_webui(self, message: str) -> None:
        """Capture log messages for webui display.

        Args:
            message: The formatted log message (may contain ANSI color codes)
        """
        # Strip ANSI color codes only for checking if message is empty
        clean_message = self.ANSI_ESCAPE_PATTERN.sub("", message).strip()

        if clean_message:
            # Store the original message with ANSI codes for colored display in webui
            self._console_logs.append(message.strip())

            # Also capture ERROR and CRITICAL level messages into errors_history
            log_record = getattr(message, "record", None)
            level = log_record.get("level") if log_record is not None else None
            if level is not None and level.no >= logger.level("ERROR").no:
                self._errors_history.appendleft(clean_message)

    def remove_maintenance(self, *, timeout: float = 60.0) -> bool:
        """Remove maintenance mode from the named worker via the API.

        horde_sdk's synchronous client performs plain ``requests`` calls with no timeout,
        so the two API calls here run on a daemon thread with a bounded join: a wedged
        gateway then costs at most ``timeout`` seconds instead of hanging the caller
        indefinitely (called from the event loop, such a hang silences the entire worker).

        Returns:
            bool: True when maintenance removal was confirmed. False when the worker does
                not exist (yet) or the API did not respond within ``timeout`` seconds.

        Raises:
            Exception: Whatever the underlying API calls raised, re-raised on the calling
                thread.
        """
        import threading

        result: dict[str, Any] = {"done": False, "exception": None}

        def _do_remove() -> None:
            try:
                simple_client = AIHordeAPISimpleClient()
                worker_details: SingleWorkerDetailsResponse = simple_client.worker_details_by_name(
                    worker_name=self.bridge_data.dreamer_worker_name,
                )
                if worker_details is None:
                    logger.debug(
                        f"Worker with name {self.bridge_data.dreamer_worker_name} "
                        "does not appear to exist already to remove maintenance.",
                    )
                    return
                modify_worker_request = ModifyWorkerRequest(
                    apikey=self.bridge_data.api_key,
                    worker_id=worker_details.id_,
                    maintenance=False,
                )

                simple_client.worker_modify(modify_worker_request)

                logger.debug(
                    f"Ensured worker with name {self.bridge_data.dreamer_worker_name} "
                    f"({worker_details.id_}) is removed from maintenance.",
                )
                result["done"] = True
            except Exception as e:
                result["exception"] = e

        worker_thread = threading.Thread(target=_do_remove, daemon=True, name="remove_maintenance")
        worker_thread.start()
        worker_thread.join(timeout)

        if worker_thread.is_alive():
            logger.warning(
                f"Could not confirm maintenance removal within {timeout:.0f}s — the API or gateway "
                "appears unresponsive. Continuing; removal will be retried automatically.",
            )
            return False

        exception = result["exception"]
        if exception is not None:
            raise exception

        return bool(result["done"])

    def set_job_pops_paused(self, paused: bool, pause_until: float | None = None) -> None:
        """Pause or resume accepting new job pops.

        When paused, :meth:`api_job_pop` returns immediately without contacting
        the Horde API, so no new jobs are accepted.  Any jobs already in the
        queue continue to be processed normally.

        Args:
            paused: ``True`` to pause new job pops, ``False`` to resume.
            pause_until: Unix timestamp at which the pause should automatically
                expire.  Only meaningful when ``paused`` is ``True``.  Pass
                ``None`` to pause indefinitely.
        """
        if self._job_pops_paused == paused and self._job_pops_pause_until == pause_until:
            return
        self._job_pops_paused = paused
        self._job_pops_pause_until = pause_until if paused else None
        if paused:
            if pause_until is not None:
                remaining = max(0.0, pause_until - time.time())
                minutes = remaining / 60.0
                logger.info(f"Job pops paused by web UI for {minutes:.0f} minutes")
            else:
                logger.info("Job pops paused by web UI indefinitely")
            # Freeze the idle timer so that time spent paused does not count
            # toward "time without jobs".
            if self._last_pop_no_jobs_available_time > 0.0:
                self._time_spent_no_jobs_available += time.time() - self._last_pop_no_jobs_available_time
                self._last_pop_no_jobs_available_time = 0.0
            # Unload models from processes that are already idle so VRAM/RAM
            # is freed immediately without waiting for queued jobs to drain.
            self._unload_idle_inference_models()
        else:
            logger.info("Job pops resumed by web UI")
            # Restart the idle timer immediately if the queue is already empty
            # so that time_without_jobs resumes incrementing without having to
            # wait up to _job_pop_frequency seconds for the next api_job_pop
            # no-jobs response to set a new anchor.
            self._restart_idle_timer_if_queue_empty()

    def set_max_queue_size(self, size: int) -> None:
        """Change the maximum job queue size at runtime.

        The new value overrides ``bridge_data.queue_size`` for the purposes of job-pop
        throttling.  The override persists until the next call or until the worker
        is restarted.

        Args:
            size: New maximum number of buffered (queued) jobs.  Must be >= 0.
        """
        if size < 0:
            raise ValueError(f"max_queue_size must be >= 0, got {size}")
        self._queue_size_override = size
        self._queue_size_auto = False
        logger.info(f"Max queue size changed to {size} via web UI")

    def set_max_active_models(self, count: int) -> None:
        """Change the maximum number of simultaneously active inference-process slots at runtime.

        The new value overrides the start-up max-active-model limit, which is either
        ``bridge_data.max_active_models`` when explicitly configured or the fallback
        sum of ``queue_size + max_threads``. The LRU model cache capacity is updated
        accordingly so that model eviction respects the new limit. Calling this method
        also disables auto mode.

        Args:
            count: New maximum number of active model slots.  Must be >= 1.
        """
        if count < 1:
            raise ValueError(f"max_active_models must be >= 1, got {count}")
        self._max_active_models_override = count
        self.bridge_data.max_active_models = count
        os.environ["AIWORKER_MAX_ACTIVE_MODELS"] = str(count)
        self._max_active_models_auto = False
        self._lru.capacity = count
        self._inference_scale_down_requested = True
        logger.info(f"Max active models changed to {count} via web UI")

    def set_queue_size_auto_mode(self, enabled: bool) -> None:
        """Enable or disable automatic queue-size tuning.

        When enabled, :meth:`_compute_auto_queue_size` is called during each
        :meth:`update_webui_status` cycle and the result is applied as the runtime
        override.  When disabled, the last manually-set or auto-computed override
        remains in effect until the next explicit :meth:`set_max_queue_size` call.

        Args:
            enabled: ``True`` to activate auto mode, ``False`` to deactivate.
        """
        self._queue_size_auto = enabled
        if enabled:
            # Apply immediately so the UI reflects the computed value without waiting
            # for the next update_webui_status() tick.
            self._queue_size_override = self._compute_auto_queue_size()
            logger.info(f"Queue size auto mode enabled (initial value: {self._queue_size_override})")
        else:
            logger.info("Queue size auto mode disabled")

    def set_max_active_models_auto_mode(self, enabled: bool) -> None:
        """Enable or disable automatic max-active-models tuning.

        When enabled, :meth:`_compute_auto_max_active_models` is called during
        each :meth:`update_webui_status` cycle and the result is applied as the
        runtime override.  When disabled, the last manually-set or auto-computed
        override remains in effect.

        Args:
            enabled: ``True`` to activate auto mode, ``False`` to deactivate.
        """
        self._max_active_models_auto = enabled
        if enabled:
            count = self._compute_auto_max_active_models()
            self._max_active_models_override = count
            self.bridge_data.max_active_models = count
            os.environ["AIWORKER_MAX_ACTIVE_MODELS"] = str(count)
            self._lru.capacity = count
            self._inference_scale_down_requested = True
            logger.info(f"Max active models auto mode enabled (initial value: {count})")
        else:
            logger.info("Max active models auto mode disabled")

    def apply_setting(self, key: str, value: object) -> None:
        """Apply a runtime setting change to :attr:`bridge_data`.

        Updates the in-memory ``bridge_data`` attribute identified by *key* to
        *value*.  The change takes effect immediately for settings that are
        checked on every job pop or during job processing; settings that control
        startup-time behaviour (e.g. ``max_threads``, ``safety_on_gpu``) require
        a worker restart to fully take effect.

        Args:
            key:   The ``bridge_data`` field name to update.
            value: The new validated value (type must already match the field).

        Raises:
            ValueError: If *key* is not a recognised ``bridge_data`` attribute.
        """
        if key == "max_job_retries":
            self.MAX_JOB_RETRIES = int(value)  # type: ignore[assignment]
            logger.info(f"Runtime setting 'max_job_retries' changed to {value!r} via web UI")
            return
        if key == "max_submit_retries":
            self.MAX_SUBMIT_RETRIES = int(value)  # type: ignore[assignment]
            logger.info(f"Runtime setting 'max_submit_retries' changed to {value!r} via web UI")
            return
        if not hasattr(self.bridge_data, key):
            raise ValueError(f"Unknown bridge_data field: '{key}'")
        _FILTER_GROUP_KEYS = {
            "positive_prompt_append", "positive_prompt_remove", "positive_prompt_replace",
            "positive_prompt_conditional_add",
            "negative_prompt_append", "negative_prompt_remove", "negative_prompt_replace",
            "negative_prompt_conditional_add",
            "prompt_swap",
        }
        if key in _FILTER_GROUP_KEYS and isinstance(value, list):
            from horde_worker_regen.bridge_data.data_model import PromptFilterGroup
            value = [
                PromptFilterGroup.model_validate(item) if isinstance(item, dict) else item
                for item in value
            ]
        setattr(self.bridge_data, key, value)
        if not (isinstance(value, list) and len(value) == 0):
            logger.info(f"Runtime setting '{key}' changed to {value!r} via web UI")
        # Propagate data_retention_days to the webui so the database is pruned immediately.
        if key == "data_retention_days" and self.webui is not None:
            self.webui.set_data_retention_days(int(value))

    def _get_model_state_file_path(self) -> str:
        """Return the path to the WebUI model state database.

        The location can be customised with the ``AIWORKER_WEBUI_MODEL_STATE_FILE``
        environment variable.  When not set the database is placed at
        ``config/webui_model_state.db`` relative to the current working directory
        (i.e. ``/horde-worker-reGen/config/webui_model_state.db`` inside Docker).
        """
        return os.getenv("AIWORKER_WEBUI_MODEL_STATE_FILE") or WEBUI_MODEL_STATE_FILENAME

    def _load_model_state_file(self) -> None:
        """Load the persisted WebUI model state from the SQLite database and apply it.

        Models listed as disabled in the database are added to
        ``_runtime_disabled_models`` and removed from
        ``bridge_data.image_models_to_load``, **but only when they are already
        present in ``_all_models_configured``**.  Any model that was removed
        from the configured list by an environment variable or config change is
        silently ignored, so environment-variable overrides always take
        precedence.

        If the database does not exist (e.g., first run) all models remain
        enabled, which is the intended default behaviour.
        """
        state_path = self._get_model_state_file_path()
        if not os.path.exists(state_path):
            return

        try:
            with sqlite3.connect(state_path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS disabled_models (model_name TEXT PRIMARY KEY)",
                )
                rows = conn.execute("SELECT model_name FROM disabled_models").fetchall()
            disabled = [row[0] for row in rows if isinstance(row[0], str)]
        except Exception as exc:
            logger.warning(f"Could not read WebUI model state database '{state_path}': {exc}")
            return

        configured_set = set(self._all_models_configured)
        applied: list[str] = []
        for model in disabled:
            if model not in configured_set:
                # Model was removed by env-var / config — honour that override.
                continue
            self._runtime_disabled_models.add(model)
            if model in self.bridge_data.image_models_to_load:
                self.bridge_data.image_models_to_load.remove(model)
            applied.append(model)

        if applied:
            logger.info(
                f"WebUI model state restored from '{state_path}': "
                f"{len(applied)} model(s) disabled: {applied}",
            )

    def _save_model_state_file(self) -> None:
        """Persist the current WebUI model disabled state to the SQLite database.

        The database records only which models are currently disabled so that the
        selection survives a container restart.
        """
        state_path = self._get_model_state_file_path()
        try:
            os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
            with sqlite3.connect(state_path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS disabled_models (model_name TEXT PRIMARY KEY)",
                )
                conn.execute("DELETE FROM disabled_models")
                conn.executemany(
                    "INSERT INTO disabled_models (model_name) VALUES (?)",
                    [(name,) for name in sorted(self._runtime_disabled_models)],
                )
                conn.commit()
        except Exception as exc:
            logger.warning(f"Could not save WebUI model state to '{state_path}': {exc}")

    def _toggle_model(self, model_name: str, enabled: bool) -> None:
        """Toggle a model's enabled state for job pops at runtime.

        When a model is disabled it is removed from ``bridge_data.image_models_to_load``
        so it will not be included in future job pop requests.  Re-enabling adds it back.
        The new state is persisted to the WebUI model state file so it survives restarts.

        Args:
            model_name: The model name to toggle.
            enabled: ``True`` to enable, ``False`` to disable.
        """
        if enabled:
            self._runtime_disabled_models.discard(model_name)
            if model_name not in self.bridge_data.image_models_to_load:
                self.bridge_data.image_models_to_load.append(model_name)
            logger.info(f"Model '{model_name}' enabled via web UI")
        else:
            self._runtime_disabled_models.add(model_name)
            if model_name in self.bridge_data.image_models_to_load:
                self.bridge_data.image_models_to_load.remove(model_name)
            logger.info(f"Model '{model_name}' disabled via web UI")

        self._save_model_state_file()

    def _refresh_model_configuration_state_after_reload(self) -> None:
        """Reconcile model UI/runtime state after bridge-data reload."""
        configured_models = list(self.bridge_data.image_models_to_load)
        configured_model_set = set(configured_models)
        self._runtime_disabled_models.intersection_update(configured_model_set)
        self._all_models_configured = configured_models
        self.bridge_data.image_models_to_load = [
            model for model in configured_models if model not in self._runtime_disabled_models
        ]
        self._save_model_state_file()

    def _get_settings_snapshot(self) -> dict[str, object]:
        """Return a flat dict of the current values of all runtime-configurable settings.

        Only fields that are present in the web UI's ``_SETTINGS_SPEC`` are
        included.  This snapshot is pushed to the web UI each status cycle so
        that the Settings page always reflects the live configuration.
        """
        from horde_worker_regen.webui.server import _SETTINGS_SPEC  # local import to avoid circular

        result: dict[str, object] = {}
        for field_name in _SETTINGS_SPEC:
            if field_name == "max_job_retries":
                result[field_name] = self.MAX_JOB_RETRIES
                continue
            if field_name == "max_submit_retries":
                result[field_name] = self.MAX_SUBMIT_RETRIES
                continue
            try:
                value = getattr(self.bridge_data, field_name)
            except AttributeError:
                continue
            # Pydantic model instances are not JSON-serialisable by the standard
            # json encoder.  Convert list-of-models to plain dicts so the web UI
            # can serve them without a 500 error.
            if isinstance(value, list):
                result[field_name] = [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in value
                ]
            else:
                result[field_name] = value
        return result

    def _compute_auto_queue_size(self) -> int:
        """Compute a queue-size override from observed worker throughput.

        The heuristic starts from available non-concurrent process headroom and
        then adjusts based on average TOTAL job duration.  Timing tiers:

        - **< 20 s** (very fast): buffer 3 × concurrent — API round-trip
          dominates, so deep buffering saturates the pipeline.
        - **20 – 60 s** (fast): buffer 2 × concurrent to keep workers fed
          through model swaps.
        - **60 – 180 s** (medium): buffer 1 × concurrent as a comfortable
          cushion.
        - **180 – 600 s** (slow): at least 1 buffered job so the worker never
          idles between long generations.
        - **≥ 600 s** (very slow): 1 — more would tie up VRAM without benefit.

        Timing data is ignored until at least 3 samples have been collected to
        avoid reacting to a single anomalously fast or slow job.

        The result is always at least 1 so that auto mode never fully disables
        the queue — a queue size of 0 would be equivalent to turning auto off.

        Returns:
            Recommended max queue size (>= 1).
        """
        concurrent = self._max_concurrent_inference_processes
        base = max(0, self._max_inference_processes - concurrent)

        total_stats = self._job_time_stats.get("TOTAL")
        if not (total_stats and total_stats["count"] >= 3):
            # Insufficient data — need at least 1 so jobs can pop to gather timing data.
            return max(1, base)

        avg_job_time = total_stats["sum"] / total_stats["count"]

        if avg_job_time < 20:
            # Very fast: pre-load 3 jobs per worker to saturate the pipeline
            result = max(base, concurrent * 3)
        elif avg_job_time < 60:
            # Fast: pre-load 2 jobs per worker — keeps workers fed through model swaps
            result = max(base, concurrent * 2)
        elif avg_job_time < 180:
            # Medium: pre-load 1 job per worker as a comfortable buffer
            result = max(base, concurrent)
        elif avg_job_time < 600:
            # Slow: at least 1 buffered job so the worker never idles between long runs
            result = max(base, 1)
        else:
            # Very slow: 1 is sufficient — more would tie up VRAM without benefit
            result = 1

        return max(1, result)

    def _compute_auto_max_active_models(self) -> int:
        """Compute the optimal active-model count based on available VRAM.

        The heuristic estimates how many models can fit in VRAM simultaneously:

        1. Derive a per-model VRAM footprint from current worker VRAM usage
           divided by the number of currently-loaded models.  Falls back to a
           4 GB estimate (SDXL-class typical footprint) when no models are loaded
           or VRAM data is unavailable.
        2. Reserve an explicit *inference headroom*: ``max(2 GB, 10 % of total VRAM)``.
           This is more conservative than a flat percentage on smaller cards
           (8–12 GB) where 10 % leaves almost no room for active computation.
        3. Compute available VRAM as ``total_vram - system_vram_used - inference_headroom``.
        4. Return ``max(2, floor(available / per_model))``, capped at 16.

        When total VRAM data has not yet been collected (e.g. at startup or on
        CPU-only machines), the current ``max_inference_processes`` is returned
        unchanged (minimum 2).

        Returns:
            Recommended max active model count (>= 2).
        """
        total_vram = self._last_total_vram_mb
        system_vram_used = self._last_system_vram_usage_mb
        worker_vram = self._last_worker_vram_mb

        if total_vram <= 0:
            # No VRAM data available yet; keep current value, but never drop below 2
            return max(2, self.max_inference_processes)

        # Estimate per-model VRAM from current worker usage
        num_loaded = len(
            [p for p in self._process_map.values() if p.loaded_horde_model_name is not None]
        )
        if worker_vram > 0 and num_loaded > 0:
            per_model_vram = worker_vram / num_loaded
        else:
            per_model_vram = 4096.0  # 4 GB default (SDXL-class typical footprint)

        per_model_vram = max(per_model_vram, 512.0)  # floor at 512 MB to avoid division explosion

        # Reserve headroom for active inference operations.
        # On small cards the percentage is too small; use an absolute minimum of 2 GB.
        inference_headroom = max(2048.0, total_vram * 0.10)
        available_vram = max(0.0, total_vram - system_vram_used - inference_headroom)
        if available_vram <= 0:
            return max(2, num_loaded)

        auto_count = max(2, int(available_vram / per_model_vram))
        return min(auto_count, 16)

    def enable_performance_mode(self) -> None:
        """Enable performance mode."""
        if self.bridge_data.high_performance_mode:
            self._max_pending_megapixelsteps = 80
            logger.info("High performance mode enabled")
            if not self.bridge_data.safety_on_gpu:
                logger.warning(
                    "If you have a high-end GPU, you should enable safety on GPU (safety_on_gpu in the config).",
                )

        elif self.bridge_data.moderate_performance_mode:
            self._max_pending_megapixelsteps = 60
            logger.info("Moderate performance mode enabled")
        else:
            self._max_pending_megapixelsteps = 15
            logger.info("Normal performance mode enabled")

        if self.bridge_data.high_performance_mode and self.bridge_data.moderate_performance_mode:
            logger.warning("Both high and moderate performance modes are enabled. Using high performance mode.")

    def is_time_for_shutdown(self) -> bool:
        """Return true if it is time to shut down."""
        if not self._shutting_down:
            return False

        # If any job hasn't been submitted to the API yet, then we can't shut down
        if len(self.jobs_pending_submit) > 0:
            return False
        # If there are any jobs in progress, then we can't shut down
        if len(self.jobs_being_safety_checked) > 0 or len(self.jobs_pending_safety_check) > 0:
            return False
        if len(self.jobs_in_progress) > 0:
            return False
        if len(self.jobs_pending_inference) > 0:
            # Only block shutdown when at least one inference process is still alive and
            # could actually process the queued jobs. If every inference process is already
            # ending/ended, the queue will never drain on its own — proceed so the worker
            # can restart and retry.
            if any(
                p.last_process_state
                not in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED)
                for p in self._process_map.get_inference_processes()
            ):
                return False
            logger.warning(
                f"Shutdown: {len(self.jobs_pending_inference)} queued job(s) cannot be processed — "
                "all inference processes are ending. Proceeding with shutdown.",
            )
        if len(self.jobs_pending_submit) > 0:
            return False

        inference_processes = list(self._process_map.get_inference_processes())

        if all(
            inference_process.last_process_state == HordeProcessState.PROCESS_ENDING
            or inference_process.last_process_state == HordeProcessState.PROCESS_ENDED
            or inference_process.last_process_state == HordeProcessState.PROCESS_STARTING
            for inference_process in inference_processes
        ):
            # _recently_recovered: a replacement process was just spawned and may be in
            # PROCESS_STARTING, which would falsely satisfy this "all ending/ended/starting"
            # check while the process is still initialising.  Block premature shutdown only
            # when at least one process is actually in PROCESS_STARTING; if all are already
            # in ENDING/ENDED it is always safe to stop regardless of the recovery window.
            if not (
                self._recently_recovered
                and any(
                    p.last_process_state == HordeProcessState.PROCESS_STARTING
                    for p in inference_processes
                )
            ):
                return True

        any_process_alive = False

        for process_info in self._process_map.values():
            # The safety process gets shut down last and is part of cleanup
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if process_info.last_process_state in (
                HordeProcessState.INFERENCE_STARTING,
                HordeProcessState.INFERENCE_PROCESSING,
                HordeProcessState.INFERENCE_POST_PROCESSING,
                HordeProcessState.POST_PROCESSING_STARTING,
            ):
                any_process_alive = True
                continue

        # _recently_recovered: a just-replaced process sits in PROCESS_STARTING and looks
        # idle (not in any of the "alive" inference states above).  Block premature shutdown
        # while the recovery window is active and at least one process is still initialising.
        if self._recently_recovered and any(
            p.last_process_state == HordeProcessState.PROCESS_STARTING
            for p in inference_processes
        ):
            return False

        # If there are any inference processes still alive, then we can't shut down
        return not any_process_alive

    def is_free_inference_process_available(self) -> bool:
        """Return true if there is an inference process available which can accept a job."""
        return self._process_map.num_available_inference_processes() > 0

    def is_any_model_preloaded(self) -> bool:
        """Return true if any model is preloaded."""
        return self._process_map.num_preloaded_processes() > 0

    def has_queued_jobs(self) -> bool:
        """Return true if there are any jobs not already in progress but are popped."""
        return any(job not in self.jobs_in_progress for job in self.jobs_pending_inference)

    def start_safety_processes(self) -> None:
        """Start all the safety processes configured to be used.

        This can be used after a configuration change to get just the newly configured processes running.
        """
        num_processes_to_start = self.max_safety_processes - self._process_map.num_safety_processes()

        # If the number of processes to start is less than 0, log a critical error and raise a ValueError
        if num_processes_to_start < 0:
            logger.critical(
                f"There are already {self._process_map.num_safety_processes()} safety processes running, but "
                f"max_safety_processes is set to {self.max_safety_processes}",
            )
            raise ValueError("num_processes_to_start cannot be less than 0")

        # Start the required number of processes

        for _ in range(num_processes_to_start):
            # Create a two-way communication pipe for the parent and child processes
            pid = self._process_map.num_safety_processes()
            pipe_connection, child_pipe_connection = multiprocessing.Pipe(duplex=True)

            cpu_only = not self.bridge_data.safety_on_gpu

            # Create a new process that will run the start_safety_process function
            process = multiprocessing.Process(
                target=start_safety_process,
                args=(
                    pid,
                    self._process_message_queue,
                    child_pipe_connection,
                    self._disk_lock,
                    self.num_processes_launched,
                    cpu_only,
                ),
                kwargs={
                    "high_memory_mode": self.bridge_data.high_memory_mode,
                    "amd_gpu": self._amd_gpu,
                    "directml": self._directml,
                },
            )

            process.start()

            # Add the process to the process map
            self._process_map[pid] = HordeProcessInfo(
                mp_process=process,
                pipe_connection=pipe_connection,
                process_id=pid,
                process_type=HordeProcessType.SAFETY,
                last_process_state=HordeProcessState.PROCESS_STARTING,
                process_launch_identifier=self.num_processes_launched,
            )

            logger.info(f"Started safety process ({self._process_label(pid)})")
            self.num_processes_launched += 1

    def start_inference_processes(self) -> None:
        """Start all the inference processes configured to be used.

        This can be used after a configuration change to get just the newly configured processes running.
        """
        num_processes_to_start = self.max_inference_processes - self._process_map.num_inference_processes()

        # If the number of processes to start is less than 0, log a critical error and raise a ValueError
        if num_processes_to_start < 0:
            logger.critical(
                f"There are already {self._process_map.num_inference_processes()} inference processes running, but "
                f"max_inference_processes is set to {self.max_inference_processes}",
            )
            raise ValueError("num_processes_to_start cannot be less than 0")

        # Start the required number of processes
        for i in range(num_processes_to_start):
            # Create a two-way communication pipe for the parent and child processes
            pid = len(self._process_map)
            self._start_inference_process(pid)

            logger.info(f"Started inference process ({self._process_label(pid)})")

            if i == 0:
                # Sleep for 4 seconds to allow the first process to start and download the model references
                time.sleep(4)

    def _start_inference_process(self, pid: int) -> HordeProcessInfo:
        """Starts an inference process.

        :param pid: process ID to assign to the process
        :return:
        """
        vram_heavy_models = any(model in VRAM_HEAVY_MODELS for model in self.bridge_data.image_models_to_load)

        # Exactly one inference process per worker run should download the legacy model
        # reference databases. Decide it here in the parent: the historical child-side
        # `process_id == 1` heuristic breaks when the safety process count is not 1 and
        # never re-fires for replacement processes.
        download_legacy_references = not self._legacy_references_download_assigned
        self._legacy_references_download_assigned = True

        pipe_connection, child_pipe_connection = multiprocessing.Pipe(duplex=True)
        # Create a new process that will run the start_inference_process function
        process = multiprocessing.Process(
            target=start_inference_process,
            args=(
                pid,
                self._process_message_queue,
                child_pipe_connection,
                self._inference_semaphore,
                self._disk_lock,
                self._aux_model_lock,
                self._vae_decode_semaphore,
                self.num_processes_launched,
            ),
            kwargs={
                "very_high_memory_mode": self.bridge_data.very_high_memory_mode,
                "high_memory_mode": self.bridge_data.high_memory_mode,
                "amd_gpu": self._amd_gpu,
                "directml": self._directml,
                "vram_heavy_models": vram_heavy_models,
                "download_legacy_references": download_legacy_references,
            },
        )
        process.start()
        # Add the process to the process map
        process_info = HordeProcessInfo(
            mp_process=process,
            pipe_connection=pipe_connection,
            process_id=pid,
            process_type=HordeProcessType.INFERENCE,
            last_process_state=HordeProcessState.PROCESS_STARTING,
            process_launch_identifier=self.num_processes_launched,
        )
        self._process_map[pid] = process_info
        logger.info(f"Starting inference process ({self._process_label(pid)})")
        self.num_processes_launched += 1
        return process_info

    def end_inference_processes(
        self,
        force: bool = False,
    ) -> None:
        """End any inference processes above the configured limit, or all of them if shutting down."""
        if force:
            if not self._shutting_down:
                logger.error("Forcing inference processes to end without shutting down")

            for process in self._process_map.get_inference_processes():
                self._end_inference_process(process)
            return

        if (not self._shutting_down) and (
            self._process_map.num_loaded_inference_processes() <= self.max_inference_processes
        ):
            return

        processes_with_model_for_queued_job: list[int] = self.get_processes_with_model_for_queued_job()

        # if we're shutting down and the job queue is empty, we can end all inference processes
        if self._shutting_down and len(self.jobs_pending_inference) == 0 and len(self.jobs_in_progress) == 0:
            processes_with_model_for_queued_job = []

        # Get the process to end
        process_info = self._process_map._get_first_inference_process_to_kill(
            disallowed_processes=processes_with_model_for_queued_job,
        )

        if process_info is not None:
            self._end_inference_process(process_info)

    def _end_inference_process(self, process_info: HordeProcessInfo) -> None:
        """Ends an inference process.

        :param process_info: HordeProcessInfo for the process to end
        :return: None
        """
        self._process_map.on_process_ending(process_id=process_info.process_id)
        if process_info.loaded_horde_model_name is not None:
            self._horde_model_map.expire_entry(process_info.loaded_horde_model_name)

        # Send the END_PROCESS control message in a daemon thread with a short timeout.
        # If the child is stuck in a blocking GPU operation it may not be draining the
        # pipe, causing pipe_connection.send() to block indefinitely and freeze the
        # asyncio event loop.  We always fall through to kill() below regardless.
        import threading

        def _send_end() -> None:
            try:
                process_info.safe_send_message(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))
            except BrokenPipeError:
                if not self._shutting_down:
                    logger.debug(f"Process {process_info.process_id} control channel vanished")
            except Exception:
                pass

        _send_thread = threading.Thread(target=_send_end, daemon=True)
        _send_thread.start()
        _send_thread.join(timeout=1.5)
        if _send_thread.is_alive():
            logger.debug(
                f"END_PROCESS send to {self._process_label(process_info.process_id)} timed out "
                "(pipe buffer likely full — child is stuck); proceeding to kill.",
            )

        try:
            # Give a child that acknowledged END_PROCESS a real chance to exit on its own.
            # SIGKILLing a child while its queue feeder thread is mid-write into the shared
            # process message queue truncates the in-flight payload and can permanently
            # corrupt the queue for every process (reader blocks forever / write lock left
            # held).  A gracefully-exiting child is actively sending its final state messages
            # during this window, so the old 1s grace routinely guaranteed exactly that race;
            # 5s lets model unloading and the final sends complete in the common case.  A
            # truly hung child (stuck in a GPU op) sends nothing, so for it the extra wait
            # only delays the replacement, never adds risk.
            process_info.mp_process.join(timeout=5)
            if process_info.mp_process.is_alive():
                process_info.mp_process.kill()
            # Brief join after kill to confirm the process is dead before callers release
            # shared semaphores (e.g. VAE decode semaphore).  Without this there is a small
            # window where the old child could still acquire a semaphore after the manager
            # released it defensively, permanently leaking the token.
            process_info.mp_process.join(timeout=2)
        except Exception as e:
            logger.error(f"Failed to kill {self._process_label(process_info.process_id)}: {e}")

        if not self._shutting_down:
            logger.info(f"Ended inference process {self._process_label(process_info.process_id)}")

    def _release_vae_decode_semaphore_defensively(self, process_id: int, context: str) -> None:
        """Attempt to release the VAE decode semaphore, logging benign over-releases at DEBUG.

        Use this helper for defensive semaphore cleanup when a child process may have held
        the VAE decode semaphore but died without releasing it.  ``BoundedSemaphore`` raises
        ``ValueError`` on over-release (the child already released it, or never acquired it),
        which is the expected benign case.

        Args:
            process_id: ID of the process being cleaned up (used in log messages).
            context: Human-readable description of the calling context (used in log messages).
        """
        try:
            self._vae_decode_semaphore.release()
        except ValueError:
            logger.debug(
                f"VAE decode semaphore already released for process {process_id} "
                f"{context} (child released it normally or never acquired it)",
            )
        except Exception as e:
            logger.warning(
                f"Unexpected error releasing VAE decode semaphore for process "
                f"{process_id} {context}: {type(e).__name__}: {e}",
            )

    _num_process_recoveries = 0
    """The number of times a child process crashed or was killed and recovered."""
    _safety_processes_should_be_replaced: bool = False
    """Whether or not the safety processes should be replaced due to a detected problem."""
    _safety_processes_ending: bool = False
    """Whether or not the safety processes are in the process of ending. \
        This only occurs when they are being replaced."""

    def _replace_all_safety_process(self) -> None:
        """Replace all of the safety processes.

        Args:
            process_info: The process to replace.
        """
        if not self._safety_processes_should_be_replaced:
            return

        if not self._safety_processes_ending and self._process_map.num_loaded_safety_processes() > 0:
            self._safety_processes_ending = True
            self.end_safety_processes()
            return

        if self._process_map.num_loaded_safety_processes() == 0 and self._process_map.num_safety_processes() > 0:
            self._process_map.delete_safety_processes()

        if (
            self._safety_processes_ending
            and self._process_map.num_loaded_safety_processes() == 0
            and self._process_map.num_safety_processes() == 0
        ):
            self._safety_processes_ending = False
            self._safety_processes_should_be_replaced = False
            # Don't respawn while shutting down: a safety process spawned here has nothing
            # left to evaluate and is just an unwanted new subprocess for the shutdown/watchdog
            # path to kill (same reasoning as _replace_inference_process's respawn param).
            if not self._shutting_down:
                self.start_safety_processes()
                self._num_process_recoveries += 1

    def _replace_inference_process(self, process_info: HordeProcessInfo, respawn: bool = True) -> None:
        """Replaces an inference process (for whatever reason; probably because it crashed).

        Args:
            process_info: The process to replace.
            respawn: Whether to start a fresh subprocess in its place. Callers pass False while
                shutting down: spawning a brand-new process (which must import torch and set up
                the model manager from scratch) mid-shutdown routinely takes longer than the
                graceful-shutdown window, forcing the watchdog in `_start_timed_shutdown()` to
                hard-kill the worker before the new process even finishes starting.
        """
        logger.debug(f"Replacing {process_info}")
        # job = next(((job, pid) for job, pid in self.jobs_in_progress if pid == process_info.process_id), None)
        job_to_remove = None
        if process_info.last_job_referenced is not None and process_info.last_job_referenced in self.jobs_lookup:
            job_to_remove = process_info.last_job_referenced

        # Snapshot the state before _end_inference_process() calls on_process_ending(),
        # which resets last_process_state to PROCESS_ENDING and clears last_progress_value.
        prior_state = process_info.last_process_state
        prior_progress_value = process_info.last_progress_value

        # Release the inference semaphore immediately for INFERENCE_PROCESSING and
        # POST_PROCESSING_STARTING so that any process waiting in INFERENCE_STARTING can
        # proceed.  INFERENCE_STARTING is intentionally excluded here: the child may be
        # blocked at semaphore.acquire().  Releasing the semaphore before killing creates a
        # race where the child acquires the semaphore just before the SIGKILL arrives.
        # SIGKILL bypasses the finally block, permanently leaking the token and leaving the
        # next INFERENCE_STARTING process blocked forever.  The semaphore is instead
        # released AFTER the kill (see below).
        if prior_state in (
            HordeProcessState.INFERENCE_PROCESSING,
            HordeProcessState.POST_PROCESSING_STARTING,
        ):
            try:
                self._inference_semaphore.release()
            except ValueError:
                logger.debug(
                    f"Inference semaphore already released when replacing process {process_info.process_id}",
                )
            except Exception as e:
                logger.warning(
                    f"Unexpected error releasing inference semaphore when replacing process "
                    f"{process_info.process_id}: {type(e).__name__}: {e}",
                )

        # Release the disk lock defensively for all three inference-active states.
        if prior_state in (
            HordeProcessState.INFERENCE_STARTING,
            HordeProcessState.INFERENCE_PROCESSING,
            HordeProcessState.POST_PROCESSING_STARTING,
        ):
            try:
                self._disk_lock.release()
            except ValueError:
                logger.debug(f"Disk lock already released when replacing process {process_info.process_id}")
            except Exception as e:
                logger.warning(
                    f"Unexpected error releasing disk lock when replacing process "
                    f"{process_info.process_id}: {type(e).__name__}: {e}",
                )

        if prior_state in (
            HordeProcessState.INFERENCE_POST_PROCESSING,
            HordeProcessState.POST_PROCESSING_STARTING,
        ):
            # The VAE decode semaphore was acquired in the progress callback before
            # POST_PROCESSING_STARTING / INFERENCE_POST_PROCESSING was emitted. Release it
            # here so that the replacement process (and any other process) is not blocked
            # from acquiring it for up to VAE_SEMAPHORE_TIMEOUT seconds.
            self._release_vae_decode_semaphore_defensively(
                process_info.process_id,
                "when replacing INFERENCE_POST_PROCESSING/POST_PROCESSING_STARTING process",
            )

        elif process_info.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL:
            try:
                self._aux_model_lock.release()
            except ValueError:
                logger.debug(
                    f"Aux model lock already released when replacing process {process_info.process_id}",
                )
            except Exception as e:
                logger.warning(
                    f"Unexpected error releasing aux model lock when replacing process "
                    f"{process_info.process_id}: {type(e).__name__}: {e}",
                )

            if process_info.last_job_referenced is not None and process_info.last_job_referenced in self.jobs_lookup:
                job_to_remove = process_info.last_job_referenced
                logger.error(
                    f"Job {job_to_remove.id_ or job_to_remove.ids} was in aux model preload on process "
                    f"{process_info.process_id} but it failed. Removing.",
                )

        if process_info.loaded_horde_model_name is not None:
            self._horde_model_map.expire_entry(process_info.loaded_horde_model_name)

        if job_to_remove is not None and (
            job_to_remove in self.jobs_in_progress
            or (
                prior_state
                in (
                    HordeProcessState.MODEL_PRELOADING,
                    HordeProcessState.DOWNLOADING_AUX_MODEL,
                )
                and job_to_remove in self.jobs_pending_inference
            )
        ):
            retry_skipped = False
            fault_info = None
            # For MODEL_PRELOADING stuck replacements: skip the local retry by pre-setting
            # retry_count to MAX_JOB_RETRIES so handle_job_fault permanently faults the job
            # rather than re-queuing it.  Re-queuing would send the same broken model to a
            # fresh process, which will get stuck again for another full preload_timeout before
            # the preload-stuck cooldown kicks in.  Permanently faulting immediately lets the
            # horde re-assign the job to another worker without wasting a second preload cycle.
            if prior_state == HordeProcessState.MODEL_PRELOADING:
                job_info = self.jobs_lookup.get(job_to_remove)
                if job_info is not None:
                    job_info.retry_count = self.MAX_JOB_RETRIES
                retry_skipped = True
                fault_info = "retry skipped because the process was replaced while stuck in MODEL_PRELOADING"
            if fault_info is None and not retry_skipped:
                self.handle_job_fault(
                    faulted_job=job_to_remove,
                    process_info=process_info,
                )
            else:
                self.handle_job_fault(
                    faulted_job=job_to_remove,
                    process_info=process_info,
                    fault_info=fault_info,
                    retry_skipped=retry_skipped,
                )

        self._end_inference_process(process_info)

        # For INFERENCE_STARTING: release the inference semaphore only after we've asked the
        # old process to stop.  Releasing it earlier creates a race: the child may be blocked
        # at semaphore.acquire(), immediately grab the freshly-released token, start
        # basic_inference(), and then get killed with SIGKILL.  SIGKILL bypasses the finally
        # block, permanently consuming the token and leaving every subsequent
        # INFERENCE_STARTING process blocked at acquire() forever.  Releasing after
        # _end_inference_process() avoids reopening that race window here, even though this
        # path does not explicitly verify child exit before the release.
        if prior_state == HordeProcessState.INFERENCE_STARTING:
            try:
                self._inference_semaphore.release()
            except ValueError:
                logger.debug(
                    f"Inference semaphore already released when replacing INFERENCE_STARTING "
                    f"process {process_info.process_id}",
                )
            except Exception as e:
                logger.warning(
                    f"Unexpected error releasing inference semaphore when replacing INFERENCE_STARTING "
                    f"process {process_info.process_id}: {type(e).__name__}: {e}",
                )

        # When a process in INFERENCE_PROCESSING at 100% progress is killed, it may have
        # already acquired the VAE decode semaphore in _progress_callback_impl (the
        # _current_job_inference_steps_complete = True path, entered when the last diffusion
        # step completes).  If SIGKILL bypassed the child's finally block, the semaphore
        # count stays at 0, causing every subsequent job to block for up to
        # VAE_SEMAPHORE_TIMEOUT (300 s) before timing out and continuing.
        #
        # Release the VAE decode semaphore here, AFTER the child has been killed (and
        # confirmed dead via the post-kill join in _end_inference_process), to avoid the
        # same race that affects INFERENCE_STARTING / inference semaphore: if we released
        # before the kill and the child was blocked in acquire(), the child could grab the
        # token and get killed mid-decode with SIGKILL, permanently leaking the permit.
        #
        # Only trigger when progress was 100% because the VAE decode semaphore is never
        # acquired during INFERENCE_PROCESSING at lower progress values.
        if prior_state == HordeProcessState.INFERENCE_PROCESSING and prior_progress_value == 100:
            self._release_vae_decode_semaphore_defensively(
                process_info.process_id,
                "when replacing INFERENCE_PROCESSING process at 100% progress",
            )

        if respawn:
            self._start_inference_process(process_info.process_id)
            self._num_process_recoveries += 1

    total_num_completed_jobs: int = 0
    """The total number of jobs that have been completed."""

    total_num_jobs_queued: int = 0
    """The total number of jobs that have been queued (popped from API) during this session."""

    def end_safety_processes(self) -> None:
        """End any safety processes above the configured limit, or all of them if shutting down."""
        process_info = self._process_map.get_first_available_safety_process()

        if process_info is None:
            # When replacing a stuck safety process no "available" (WAITING_FOR_JOB) process
            # may exist.  Fall back to any safety process so the stuck one can be terminated.
            if self._safety_processes_should_be_replaced:
                process_info = self._process_map.get_safety_process()

        if process_info is None:
            return

        # Send the process a message to end
        process_info.safe_send_message(HordeControlMessage(control_flag=HordeControlFlag.END_PROCESS))

        # Update the process map
        self._process_map.on_process_ending(process_id=process_info.process_id)

        logger.info(f"Ended safety process {self._process_label(process_info.process_id)}")

    _QUEUE_READER_POLL_SECONDS = 2.0
    """Timeout for each Queue.get() in the reader thread.

    Bounds how often the reader thread refreshes its liveness tick when the queue is idle,
    so it must be well below _QUEUE_READER_STALL_TIMEOUT."""

    _QUEUE_READER_STALL_TIMEOUT = 30.0
    """Seconds without a reader-thread tick before the process message queue is declared dead.

    A healthy reader ticks at least every _QUEUE_READER_POLL_SECONDS; the only ways to exceed
    this threshold are a Queue.get() blocked forever mid-``_recv_bytes()`` on a queue corrupted
    by a SIGKILLed child (truncated payload), or transferring a single message so large it takes
    longer than this to read — far beyond any real inference-result message."""

    _queue_reader_stall_handled = False
    """One-shot flag so the queue-death recovery (worker restart) is only initiated once."""

    def _queue_reader_loop(self) -> None:
        """Continuously move messages from the multiprocessing queue into the local deque.

        Runs on a dedicated daemon thread.  This is the ONLY place allowed to call
        ``self._process_message_queue.get()``: if a child process is SIGKILLed while its queue
        feeder thread is mid-write, the queue pipe is left with a truncated payload and the
        next ``get()`` blocks forever inside ``_recv_bytes()`` — with no timeout applying and
        no exception raised.  Confining that risk to this thread keeps the asyncio event loop
        alive so _check_process_message_queue_health() can detect the stall (frozen tick) and
        restart the worker instead of freezing silently.
        """
        while True:
            if self._shut_down:
                return
            try:
                message = self._process_message_queue.get(timeout=self._QUEUE_READER_POLL_SECONDS)
            except queue.Empty:
                self._queue_reader_last_tick = time.time()
                continue
            except (EOFError, ConnectionError, BrokenPipeError, OSError, ValueError) as e:
                # The queue is closed (normal during shutdown/cleanup) or its pipe is broken
                # (fatal).  Either way no further messages can ever be read — exit the thread;
                # outside of shutdown the health check treats a dead reader thread as a dead
                # queue and restarts the worker.
                if not (self._shutting_down or self._shut_down):
                    logger.error(
                        f"Process message queue reader failed ({type(e).__name__}): {e} — "
                        "no further child process messages can be received.",
                    )
                return
            except Exception as e:
                # A single message failed to deserialize (e.g. unpickling error from a payload
                # written by a child that died mid-shutdown).  The frame was consumed, so
                # subsequent messages are unaffected — log and keep reading.
                logger.error(f"Failed to read a process message from the queue ({type(e).__name__}): {e}")
                self._queue_reader_last_tick = time.time()
                continue

            self._queue_reader_last_tick = time.time()
            self._received_process_messages.append(message)

    def _check_process_message_queue_health(self) -> None:
        """Detect a dead process message queue and restart the worker instead of freezing.

        If a child process is killed while writing to the shared message queue, the queue can
        be corrupted two ways: the reader blocks forever on a truncated payload (stuck reader
        thread), and/or the queue's shared write lock is left permanently held (children can
        never send again).  Both are unrecoverable — the queue cannot be rebuilt because every
        child holds a reference to it.  When detected, initiate a logged restart: the horde
        re-queues any abandoned jobs after they time out server-side.
        """
        if self._shutting_down or self._shut_down or self._queue_reader_stall_handled:
            return
        reader_thread = getattr(self, "_queue_reader_thread", None)
        if not isinstance(reader_thread, threading.Thread):
            return

        stalled_for = time.time() - self._queue_reader_last_tick
        if reader_thread.is_alive() and stalled_for <= self._QUEUE_READER_STALL_TIMEOUT:
            return

        reason = (
            f"its reader thread has been blocked mid-read for {stalled_for:.0f}s "
            "(a child process was most likely killed while writing to the queue, truncating a message)"
            if reader_thread.is_alive()
            else "its reader thread has died"
        )
        self._queue_reader_stall_handled = True
        logger.critical(
            f"The inter-process message queue is dead: {reason}. Child process messages can no longer "
            "be received, so the worker cannot continue operating. Restarting the worker now — "
            "in-progress jobs will be abandoned and re-queued by the horde after they time out.",
        )
        self.request_program_restart()
        self._purge_jobs()
        self._shutdown()
        self._start_timed_shutdown()

    def receive_and_handle_process_messages(self) -> None:
        """Receive and handle any messages from the child processes.

        This is the backbone of the inter-process communication system and is the main way that the parent process \
             knows what is going on in the child processes.

        **Note** also that this is a synchronous function and any interaction with objects that are shared between \
            coroutines should be done with care. Critically, this function should be called with locks already \
            acquired on any shared objects.

        See also `._process_map` and `._horde_model_map`, which are updated by this function, and `HordeProcessState` \
            and `ModelLoadState` for the possible states that the processes and models can be in.
        """
        # We want to completely flush the buffered messages, to maximize the chances we get the most
        # up to date information.
        #
        # In production the dedicated reader thread (see _queue_reader_loop) is the only place that
        # reads the multiprocessing queue — Queue.get() can block forever on a queue corrupted by a
        # child killed mid-write, and that must never happen on the event loop thread.  The direct
        # queue reads below are a fallback for tests, which bind this method to lightweight mock
        # managers that carry a plain queue but no reader thread or deque.
        use_reader_thread = isinstance(getattr(self, "_queue_reader_thread", None), threading.Thread) and isinstance(
            getattr(self, "_received_process_messages", None),
            deque,
        )
        while True:
            message: HordeProcessMessage
            if use_reader_thread:
                try:
                    message = self._received_process_messages.popleft()
                except IndexError:
                    break
            elif self._process_message_queue.empty():
                break
            else:
                try:
                    message = self._process_message_queue.get(block=False)
                except queue.Empty:
                    logger.debug("Queue was empty, breaking")
                    break
                except (EOFError, ConnectionError, BrokenPipeError, OSError) as e:
                    logger.error(f"Process message queue IPC failure ({type(e).__name__}): {e}")
                    break

            self._in_deadlock = False
            self._in_queue_deadlock = False

            if not isinstance(message, HordeProcessMessage):
                logger.error(f"Received a message that is not a HordeProcessMessage (skipping): {message}")
                continue

            if isinstance(message, HordeProcessHeartbeatMessage):
                if message.process_id not in self._process_map:
                    continue

                self._process_map.on_heartbeat(
                    message.process_id,
                    heartbeat_type=message.heartbeat_type,
                    percent_complete=message.percent_complete,
                )

                in_progress_job_info = self._process_map[message.process_id].last_job_referenced

                if message.process_warning is not None and (
                    in_progress_job_info is not None
                    and in_progress_job_info.payload is not None
                    and in_progress_job_info.payload.n_iter < 4
                ):
                    logger.warning(f"{self._process_label(message.process_id)} warning: {message.process_warning}")

                    model_name = self._process_map[message.process_id].loaded_horde_model_name
                    model_baseline = self.get_model_baseline(model_name) if model_name is not None else None

                    if model_baseline is not None:
                        logger.warning(f"Model baseline triggering warning: {model_baseline}")

                    if in_progress_job_info.payload.n_iter != 1:
                        logger.warning(f"Batched job triggering warning: {in_progress_job_info.payload.n_iter} images")
                        logger.warning("If you think this is in error, please contact the devs on github or discord.")
            else:
                logger.debug(
                    f"Received {type(message).__name__} from process {message.process_id}: {message.info}",
                    # f"{message.model_dump(exclude={'job_result_images_base64', 'replacement_image_base64'})}",
                )

            if message.process_id not in self._process_map:
                logger.warning(f"Received a message from an unknown process (skipping): process_id={message.process_id}")
                continue

            known_launch_identifier = self._process_map[message.process_id].process_launch_identifier

            if message.process_launch_identifier != known_launch_identifier:
                logger.debug(
                    "Received a message from process {} with launch identifier {}, but expected {}. "
                    "This is probably due to a process being replaced. Ignoring. "
                    "(type={} info={})",
                    message.process_id,
                    message.process_launch_identifier,
                    known_launch_identifier,
                    type(message).__name__,
                    message.info,
                )
                continue

            # If the process is updating us on its memory usage, update the process map for those values only
            # and then continue to the next message
            if isinstance(message, HordeProcessMemoryMessage):
                self._process_map.on_memory_report(
                    process_id=message.process_id,
                    ram_usage_bytes=message.ram_usage_bytes,
                    vram_usage_bytes=message.vram_usage_bytes,
                    total_vram_bytes=message.vram_total_bytes,
                    gpu_usage_percent=message.gpu_usage_percent,
                )
                continue

            # If the process state has changed, update the process map
            if isinstance(message, HordeProcessStateChangeMessage):
                if self._process_map[message.process_id].last_process_state == message.process_state:
                    continue

                # Snapshot the prior state before on_process_state_change() overwrites it.
                # This is used below to log/classify the state in which the process actually died,
                # rather than always reporting PROCESS_ENDING.
                prior_process_state = self._process_map[message.process_id].last_process_state
                # Snapshot progress value before reset_heartbeat_state() (called later by
                # on_process_ending() when the process is replaced) clears it.
                # on_process_state_change() itself does not reset last_progress_value for the
                # PROCESS_ENDING transition — it only resets on INFERENCE_STARTING.
                # Snapshotting here is safe because this code runs before any replacement.
                prior_process_progress_value = self._process_map[message.process_id].last_progress_value
                self._on_process_state_change(
                    process_id=message.process_id,
                    new_state=message.process_state,
                )

                if message.process_state == HordeProcessState.PROCESS_ENDING:
                    logger.info(f"{self._process_label(message.process_id)} is ending")
                    # If the process was holding the inference semaphore (i.e., it was in
                    # INFERENCE_PROCESSING or POST_PROCESSING_STARTING), release it now so that
                    # any process blocked in INFERENCE_STARTING waiting to acquire the semaphore
                    # can proceed. This handles edge cases such as OOM kills where the child
                    # process was terminated without running its finally block.
                    #
                    # POST_PROCESSING_STARTING must also be covered because the child emits that
                    # state message BEFORE releasing the semaphore (see inference_process.py
                    # progress_callback): if the child crashes between emitting the state and
                    # calling release(), the semaphore leaks.
                    #
                    # BoundedSemaphore raises ValueError on over-release (when the child already
                    # released it normally), so this is always safe to call.
                    #
                    # INFERENCE_STARTING is also included here: when _replace_inference_process()
                    # is called for a stuck INFERENCE_STARTING process it releases the semaphore
                    # to unblock the child.  There is a narrow race where the child acquires the
                    # semaphore and transitions toward INFERENCE_PROCESSING before the kill signal
                    # arrives.  If the child is killed before it can send the INFERENCE_PROCESSING
                    # state message, the manager still records INFERENCE_STARTING as the prior
                    # state.  Without releasing here the semaphore count stays at 0 even though
                    # no process is holding it, leaving the next INFERENCE_STARTING process blocked
                    # forever.  The BoundedSemaphore ValueError handler covers the case where the
                    # child never acquired (over-release attempt), so this is always safe to add.
                    if prior_process_state in (
                        HordeProcessState.INFERENCE_STARTING,
                        HordeProcessState.INFERENCE_PROCESSING,
                        HordeProcessState.POST_PROCESSING_STARTING,
                    ):
                        try:
                            self._inference_semaphore.release()
                        except ValueError:
                            logger.debug(
                                f"Inference semaphore already released for process {message.process_id} "
                                "on PROCESS_ENDING (child released it normally via finally block)",
                            )
                        except Exception as e:
                            logger.warning(
                                f"Unexpected error releasing inference semaphore for process "
                                f"{message.process_id} on PROCESS_ENDING: {type(e).__name__}: {e}",
                            )
                    # When INFERENCE_POST_PROCESSING ends unexpectedly the child's finally block
                    # may not have run (e.g. OOM kill), leaving the VAE decode semaphore held.
                    # Release it defensively here so other processes are not blocked.
                    # POST_PROCESSING_STARTING is also included because the VAE decode semaphore
                    # may have been acquired between the state message and the INFERENCE_POST_PROCESSING
                    # state being sent (if the child crashed after acquiring the semaphore but before
                    # emitting INFERENCE_POST_PROCESSING).
                    if prior_process_state in (
                        HordeProcessState.INFERENCE_POST_PROCESSING,
                        HordeProcessState.POST_PROCESSING_STARTING,
                    ):
                        self._release_vae_decode_semaphore_defensively(
                            message.process_id,
                            "on PROCESS_ENDING from INFERENCE_POST_PROCESSING/POST_PROCESSING_STARTING",
                        )
                    # A process in INFERENCE_PROCESSING at 100% progress may have acquired
                    # the VAE decode semaphore in _progress_callback_impl
                    # (_current_job_inference_steps_complete = True path).  If the child's
                    # finally block did not run (e.g. OOM kill, crash after state message),
                    # the semaphore is leaked.  Release it defensively here.
                    if (
                        prior_process_state == HordeProcessState.INFERENCE_PROCESSING
                        and prior_process_progress_value == 100
                    ):
                        self._release_vae_decode_semaphore_defensively(
                            message.process_id,
                            "on PROCESS_ENDING from INFERENCE_PROCESSING at 100% progress",
                        )
                    # If the process is ending but still has a job in progress, fault the job
                    # so it is not silently lost. This can happen if an exception occurs in the
                    # child process before it sends the inference result message.
                    process_info_ending = self._process_map[message.process_id]
                    # Report when a process ended while loading or downloading a model so the user
                    # knows which model caused the problem.
                    if prior_process_state in (
                        HordeProcessState.MODEL_PRELOADING,
                        HordeProcessState.MODEL_LOADING,
                        HordeProcessState.DOWNLOADING_MODEL,
                    ) and process_info_ending.loaded_horde_model_name is not None:
                        phase_description = (
                            "downloading"
                            if prior_process_state == HordeProcessState.DOWNLOADING_MODEL
                            else "loading"
                        )
                        logger.error(
                            f"Problem loading model {process_info_ending.loaded_horde_model_name} "
                            f"on process {message.process_id}: the process ended unexpectedly"
                            f" while {phase_description}. "
                            "Check the logs above for details.",
                        )
                    if (
                        process_info_ending.last_job_referenced is not None
                        and process_info_ending.last_job_referenced in self.jobs_in_progress
                    ):
                        logger.error(
                            f"{self._process_label(message.process_id)} is ending while job "
                            f"{process_info_ending.last_job_referenced.id_} is still in progress "
                            f"(prior process state: {prior_process_state}). "
                            "Faulting the job to ensure it is not silently lost.",
                        )
                        # Temporarily restore the prior state so handle_job_fault can correctly
                        # classify the fault phase in _faulted_jobs_history.
                        process_info_ending.last_process_state = prior_process_state
                        try:
                            self.handle_job_fault(
                                faulted_job=process_info_ending.last_job_referenced,
                                process_info=process_info_ending,
                            )
                        finally:
                            process_info_ending.last_process_state = HordeProcessState.PROCESS_ENDING
                    self._process_map.on_process_ending(process_id=message.process_id)

                if message.process_state == HordeProcessState.PROCESS_ENDED:
                    logger.info(f"{self._process_label(message.process_id)} has ended with message: {message.info}")
                    # Restart the process if we're not shutting down. When a process is replaced
                    # intentionally (via _replace_inference_process), _start_inference_process has
                    # already been called and updated _process_map[pid] with a new
                    # process_launch_identifier, so stale PROCESS_ENDED messages from the old
                    # process are filtered out before reaching this point. This branch therefore
                    # only fires for unexpected/crash-induced process endings, where we must restart
                    # the process to restore the configured worker capacity.
                    ended_process_info = self._process_map[message.process_id]
                    if not self._shutting_down and ended_process_info.process_type == HordeProcessType.INFERENCE:
                        # Rate-limit restarts per process slot to prevent a tight crash/restart
                        # loop when a process repeatedly fails shortly after starting (including
                        # when it never made it past PROCESS_STARTING due to an init failure).
                        restart_history = self._process_restart_history.setdefault(
                            message.process_id,
                            deque(maxlen=5),
                        )
                        now_ts = time.time()
                        restart_history.append(now_ts)
                        if (
                            len(restart_history) == restart_history.maxlen
                            and now_ts - restart_history[0] < 60
                        ):
                            logger.error(
                                f"Inference process {message.process_id} has ended "
                                f"{restart_history.maxlen} times within 60s; "
                                "skipping auto-restart to avoid a crash loop.",
                            )
                        else:
                            if prior_process_state == HordeProcessState.PROCESS_STARTING:
                                logger.warning(
                                    f"Inference process {message.process_id} ended while still in "
                                    "PROCESS_STARTING; restarting with rate limiting.",
                                )
                            else:
                                logger.info(
                                    f"Restarting inference process {message.process_id} after unexpected end",
                                )
                            self._start_inference_process(message.process_id)
                            self._num_process_recoveries += 1
                else:
                    logger.debug(f"Process {message.process_id} changed state to {message.process_state}")

                if message.process_state == HordeProcessState.INFERENCE_STARTING:
                    loaded_model_name = self._process_map[message.process_id].loaded_horde_model_name
                    if loaded_model_name is None:
                        logger.error(
                            f"Process {message.process_id} has no model loaded, but is starting inference; "
                            "skipping model map update for this message",
                        )
                        continue
                    batch_amount = self._process_map[message.process_id].batch_amount
                    if batch_amount is None:
                        logger.error(
                            f"Process {message.process_id} has no batch_amount, but is starting inference; "
                            "skipping model map update for this message",
                        )
                        continue
                    self._horde_model_map.update_entry(
                        horde_model_name=loaded_model_name,
                        load_state=ModelLoadState.IN_USE,
                        process_id=message.process_id,
                    )

                if (
                    message.process_state == HordeProcessState.UNLOADED_MODEL_FROM_RAM
                    and prior_process_state != HordeProcessState.UNLOADED_MODEL_FROM_RAM
                ):
                    logger.opt(colors=True).info(
                        "<fg #7b7d7d>" f"Process {message.process_id} cleared RAM: {message.info}" "</>",
                    )
                    self._process_map.on_model_ram_clear(process_id=message.process_id)

            if isinstance(message, HordeAuxModelStateChangeMessage):
                if message.process_state == HordeProcessState.DOWNLOADING_AUX_MODEL:
                    logger.opt(colors=True).info(
                        "<fg #7b7d7d>" f"Process {message.process_id} is downloading extra models (LoRas, etc.)" "</>",
                    )
                    self._process_map.on_last_job_reference_change(
                        process_id=message.process_id,
                        last_job_referenced=message.sdk_api_job_info,
                    )

                if message.process_state == HordeProcessState.DOWNLOAD_AUX_COMPLETE:
                    logger.opt(colors=True).info(
                        "<fg #7b7d7d>"
                        f"Process {message.process_id} finished downloading extra models in {message.time_elapsed}"
                        "</>",
                    )
                    _aux_job_key = (
                        next((k for k in self.jobs_lookup if k.id_ == message.sdk_api_job_info.id_), None)
                        if message.sdk_api_job_info is not None
                        else None
                    )
                    if _aux_job_key is None:
                        if message.sdk_api_job_info is not None:
                            logger.warning(
                                f"Job {message.sdk_api_job_info.id_} not found in jobs_lookup."
                                f" (Process {message.process_id})",
                            )
                        else:
                            logger.warning(
                                f"Job not found in jobs_lookup. (Process {message.process_id})",
                            )
                        logger.debug(f"Jobs lookup: {self.jobs_lookup}")
                    else:
                        self.jobs_lookup[_aux_job_key].time_to_download_aux_models = message.time_elapsed

            # If The model state has changed, update the model map
            if isinstance(message, HordeModelStateChangeMessage):
                self._horde_model_map.update_entry(
                    horde_model_name=message.horde_model_name,
                    load_state=message.horde_model_state,
                    process_id=message.process_id,
                )

                model_baseline = self.get_model_baseline(message.horde_model_name)

                if message.horde_model_state != ModelLoadState.ON_DISK:
                    self._process_map.on_model_load_state_change(
                        process_id=message.process_id,
                        horde_model_name=message.horde_model_name,
                        horde_model_baseline=model_baseline,
                    )

                    if message.horde_model_state == ModelLoadState.LOADING:
                        logger.debug(f"Process {message.process_id} is loading model {message.horde_model_name}")

                    if (
                        message.horde_model_state == ModelLoadState.LOADED_IN_VRAM
                        or message.horde_model_state == ModelLoadState.LOADED_IN_RAM
                    ):
                        if message.horde_model_state == ModelLoadState.LOADED_IN_VRAM:
                            loaded_message = (
                                f"Process {message.process_id} just finished inference, and has "
                                f"{message.horde_model_name} in VRAM."
                            )
                            logger.debug(loaded_message)
                        elif message.horde_model_state == ModelLoadState.LOADED_IN_RAM:
                            loaded_message = (
                                f"Process {message.process_id} moved model {message.horde_model_name} to system RAM. "
                            )

                            if message.time_elapsed is not None:
                                loaded_message += f"Loading took {message.time_elapsed:.2f} seconds."

                            logger.opt(colors=True).info(f"<fg #7b7d7d>{loaded_message}</>")

                else:
                    logger.opt(colors=True).info(
                        "<fg #7b7d7d>" f"Process {message.process_id} unloaded model {message.horde_model_name}" "</>",
                    )

            # If the process is sending us an inference job result:
            # - if its a faulted job, log an error and add it to the list of completed jobs to be sent to the API
            # - if its a completed job, add it to the list of jobs pending safety checks
            if isinstance(message, HordeInferenceResultMessage):
                # Use id_-based lookup: the subprocess may hold a filtered copy of the job
                # (via model_copy) whose object identity/hash differs from the original stored
                # in jobs_lookup and jobs_in_progress.
                _msg_job_id = message.sdk_api_job_info.id_
                _lookup_key = next((k for k in self.jobs_lookup if k.id_ == _msg_job_id), None)
                if _lookup_key is None:
                    logger.error(
                        f"Job {_msg_job_id} not found in jobs_lookup. (Process {message.process_id})",
                    )
                    _ip_stale = next((j for j in self.jobs_in_progress if j.id_ == _msg_job_id), None)
                    if _ip_stale is not None:
                        logger.error(
                            f"Job {_msg_job_id} found in jobs_in_progress. "
                            f"(Process {message.process_id})",
                        )
                        self.jobs_in_progress.remove(_ip_stale)
                    _pend_stale = next((j for j in self.jobs_pending_inference if j.id_ == _msg_job_id), None)
                    if _pend_stale is not None:
                        logger.error(
                            f"Job {_msg_job_id} found in job_deque. (Process {message.process_id})",
                        )
                        self.jobs_pending_inference.remove(_pend_stale)
                        self._invalidate_megapixelsteps_cache()
                        self._restart_idle_timer_if_queue_empty()
                    continue

                job_info = self.jobs_lookup[_lookup_key]

                _ip_match = next((j for j in self.jobs_in_progress if j.id_ == _msg_job_id), None)
                if _ip_match is not None:
                    self.jobs_in_progress.remove(_ip_match)
                else:
                    # Job completed but was not found in the in-progress tracking list.
                    # This can happen when the job was dispatched before tracking was established
                    # (e.g. a skip-the-line dispatch from a stale cached result). The result is
                    # still processed normally below — this is a tracking inconsistency, not a fault.
                    logger.debug(
                        f"Job {_msg_job_id} not found in jobs_in_progress on completion "
                        f"(Process {message.process_id}) — tracking inconsistency, job will still be processed",
                    )

                for job in self.jobs_pending_inference:
                    if job.id_ == message.sdk_api_job_info.id_:
                        self.jobs_pending_inference.remove(job)
                        self._invalidate_megapixelsteps_cache()
                        break

                self._restart_idle_timer_if_queue_empty()

                if self.bridge_data.unload_models_from_vram_often:
                    self.unload_models_from_vram(process_with_model=self._process_map[message.process_id])

                if message.time_elapsed is not None:
                    if message.state == GENERATION_STATE.faulted:
                        inference_finished_string = (
                            f"Inference for job {str(message.sdk_api_job_info.id_)[:8]} "
                            f"<u>({message.sdk_api_job_info.model})</u> on process {message.process_id} "
                            f"took {round(message.time_elapsed, 2)} seconds but faulted: {message.info}."
                        )
                        logger.opt(colors=True).warning(inference_finished_string)
                    else:
                        inference_finished_string = (
                            "\0<fg #da9dff>"
                            f"Inference finished for job {str(message.sdk_api_job_info.id_)[:8]} "
                            f"<u>({message.sdk_api_job_info.model})</u> on process {message.process_id}. "
                            f"It took {round(message.time_elapsed, 2)} seconds, finishing at {message.info} "
                            f"and reported {message.faults_count} faults."
                            "</>"
                        )
                        logger.opt(colors=True).info(inference_finished_string)

                else:
                    logger.info(f"Inference finished for job {message.sdk_api_job_info.id_}")
                    logger.debug(f"Job didn't include time_elapsed: {message.sdk_api_job_info}")
                if message.state != GENERATION_STATE.faulted:
                    job_info.state = message.state
                    job_info.time_to_generate = message.time_elapsed
                    job_info.job_image_results = message.job_image_results
                    job_info.sanitized_negative_prompt = message.sanitized_negative_prompt

                    # Set inference completion timestamp for webui preview ordering after safety check
                    # Always set timestamp regardless of webui state to ensure jobs can be displayed later
                    if message.job_image_results and len(message.job_image_results) > 0:
                        current_time = time.time()
                        job_info.inference_completed_timestamp = current_time
                        # Note: Images will be displayed after safety check using disk-saved versions
                        # (not here to avoid flickering)

                    self.jobs_pending_safety_check.append(job_info)
                else:
                    logger.debug(
                        f"Job data: {message.sdk_api_job_info.model_dump(exclude=_excludes_for_job_dump)}",  # type: ignore
                    )

                    self.handle_job_fault(
                        faulted_job=message.sdk_api_job_info,
                        process_info=self._process_map[message.process_id],
                        fault_info=message.info,
                    )

            # If the process is sending us a safety job result:
            # - if an unexpected error occurred, log an error a
            # - if the job was censored, replace the images with the replacement images
            # - add the job to the list of completed jobs to be sent to the API
            elif isinstance(message, HordeSafetyResultMessage):
                completed_job_info: HordeJobInfo | None = None
                for i, job_being_safety_checked in enumerate(self.jobs_being_safety_checked):
                    if job_being_safety_checked.sdk_api_job_info.id_ == message.job_id:
                        completed_job_info = self.jobs_being_safety_checked.pop(i)
                        break

                if completed_job_info is None or completed_job_info.job_image_results is None:
                    logger.error(
                        f"Expected to find a completed job with ID {message.job_id} but none was found. "
                        "This should only happen when certain process crashes occur.",
                    )
                    continue

                num_images_censored = 0
                num_images_csam = 0

                any_safety_failed = False

                if len(message.safety_evaluations) != len(completed_job_info.job_image_results):
                    logger.error(
                        f"Safety evaluation count mismatch for job {message.job_id}: "
                        f"{len(message.safety_evaluations)} evaluations vs "
                        f"{len(completed_job_info.job_image_results)} images — skipping safety processing.",
                    )
                    continue

                for i in range(len(completed_job_info.job_image_results)):
                    # We add to the image faults, all faults due to source images/masks
                    if completed_job_info.sdk_api_job_info.id_ is None:
                        continue
                    completed_job_info.job_image_results[i].generation_faults += self.job_faults.get(
                        completed_job_info.sdk_api_job_info.id_, []
                    )
                    replacement_image = message.safety_evaluations[i].replacement_image_base64

                    if message.safety_evaluations[i].failed:
                        logger.error(
                            f"Job {message.job_id} image #{i} faulted during safety checks. "
                            "Check the safety process logs for more information.",
                        )
                        any_safety_failed = True
                        continue

                    if replacement_image is not None:
                        completed_job_info.job_image_results[i].image_base64 = replacement_image
                        num_images_censored += 1
                        if message.safety_evaluations[i].is_csam:
                            num_images_csam += 1
                if (
                    completed_job_info.sdk_api_job_info.id_ is not None
                    and completed_job_info.sdk_api_job_info.id_ in self.job_faults
                ):
                    del self.job_faults[completed_job_info.sdk_api_job_info.id_]
                else:
                    logger.error(
                        f"Job {message.job_id} was not found in job_faults. This is unexpected.",
                    )

                logger.debug(
                    f"Job {message.job_id} had {num_images_censored} images censored and took "
                    f"{message.time_elapsed:.2f} seconds to check safety",
                )

                # ! IMPORTANT: Start own code
                if message.saved_images:
                    embedded_count = sum(1 for s in message.saved_images if s.metadata_embedded)

                    image_word = "image" if len(message.saved_images) == 1 else "images"
                    logger.opt(colors=True).info(
                        "<b><fg #FF69B4>"
                        f"Saved {len(message.saved_images)} {image_word} to disk for job {str(message.job_id)[:8]} "
                        f"(metadata embedded {embedded_count}/{len(message.saved_images)})"
                        "</></>",
                    )
                # ! IMPORTANT: End own code

                if any_safety_failed:
                    completed_job_info.state = GENERATION_STATE.faulted
                completed_job_info.censored = False
                for i in range(len(completed_job_info.job_image_results)):
                    if message.safety_evaluations[i].is_csam:
                        new_meta_entry = GenMetadataEntry(
                            type=METADATA_TYPE.censorship,
                            value=METADATA_VALUE.csam,
                        )
                        completed_job_info.job_image_results[i].generation_faults.append(new_meta_entry)
                        completed_job_info.state = GENERATION_STATE.csam
                        completed_job_info.censored = True
                    elif message.safety_evaluations[i].is_nsfw:
                        # This just marks images as nsfw, if not censored already
                        if message.safety_evaluations[i].replacement_image_base64 is None:
                            new_meta_entry = GenMetadataEntry(
                                type=METADATA_TYPE.information,
                                value=METADATA_VALUE.nsfw,
                            )
                            completed_job_info.job_image_results[i].generation_faults.append(new_meta_entry)
                        else:
                            new_meta_entry = GenMetadataEntry(
                                type=METADATA_TYPE.censorship,
                                value=METADATA_VALUE.nsfw,
                            )
                            completed_job_info.job_image_results[i].generation_faults.append(new_meta_entry)
                            completed_job_info.censored = True
                            if completed_job_info.state != GENERATION_STATE.csam:
                                completed_job_info.state = GENERATION_STATE.censored

                # Update webui preview with the saved disk image (not the submitted image)
                # Only update if this job's inference completed more recently than the currently displayed job
                if (
                    self.webui
                    and completed_job_info.job_image_results
                    and len(completed_job_info.job_image_results) > 0
                    and completed_job_info.inference_completed_timestamp is not None
                    and completed_job_info.inference_completed_timestamp >= self._last_image_job_timestamp
                    and message.saved_images
                    and len(message.saved_images) > 0
                ):
                    # Use the saved disk images instead of the submitted images
                    try:
                        # Read all saved images from disk and convert to base64
                        images_base64 = []
                        for saved_image in message.saved_images:
                            with open(saved_image.path, "rb") as image_file:
                                image_data = image_file.read()
                                images_base64.append(base64.b64encode(image_data).decode("utf-8"))
                        self._last_image_base64 = images_base64
                        self._last_image_job_timestamp = completed_job_info.inference_completed_timestamp
                        self._last_image_model = (
                            completed_job_info.sdk_api_job_info.model
                            if completed_job_info.sdk_api_job_info
                            else ""
                        ) or ""
                        image_safety = None
                        if len(images_base64) == len(message.safety_evaluations):
                            image_safety = [
                                {
                                    "is_nsfw": safety_evaluation.is_nsfw,
                                    "is_csam": safety_evaluation.is_csam,
                                }
                                for safety_evaluation in message.safety_evaluations
                            ]
                        self._last_image_safety = image_safety or []
                        # Add each image to the WebUI gallery
                        payload = completed_job_info.sdk_api_job_info.payload if completed_job_info.sdk_api_job_info else None
                        for img_idx, img_b64 in enumerate(images_base64):
                            gallery_image = {
                                "base64": img_b64,
                                "timestamp": completed_job_info.inference_completed_timestamp,
                                "model": completed_job_info.sdk_api_job_info.model
                                if completed_job_info.sdk_api_job_info
                                else None,
                            }
                            if image_safety is not None:
                                gallery_image["is_nsfw"] = image_safety[img_idx]["is_nsfw"]
                                gallery_image["is_csam"] = image_safety[img_idx]["is_csam"]
                            if payload is not None:
                                if payload.ddim_steps is not None:
                                    gallery_image["inference_steps"] = payload.ddim_steps
                                if payload.width is not None:
                                    gallery_image["width"] = payload.width
                                if payload.height is not None:
                                    gallery_image["height"] = payload.height
                                if payload.prompt is not None:
                                    _gparts = payload.prompt.split("###", 1)
                                    _gpos_orig = _gparts[0].strip()
                                    _gneg_orig = _gparts[1].strip() if len(_gparts) > 1 else ""
                                    _gbd = self.bridge_data
                                    _gpf_on = _gbd.prompt_filters_enabled
                                    _gpos = _apply_prompt_filters(
                                        _gpos_orig,
                                        append=_active_filter_entries(_gbd.positive_prompt_append, _gpf_on and _gbd.positive_prompt_append_enabled),
                                        remove=_active_filter_entries(_gbd.positive_prompt_remove, _gpf_on and _gbd.positive_prompt_remove_enabled),
                                        replace=_active_filter_entries(_gbd.positive_prompt_replace, _gpf_on and _gbd.positive_prompt_replace_enabled),
                                        remove_cleanup_separators=_gbd.prompt_remove_cleanup_separators,
                                        append_separator=_gbd.prompt_append_separator,
                                        remove_whole_word=_gbd.prompt_remove_whole_word,
                                        remove_case_sensitive=_gbd.prompt_remove_case_sensitive,
                                    )
                                    _gpos = _apply_conditional_add(
                                        _gpos,
                                        _active_filter_entries(_gbd.positive_prompt_conditional_add, _gpf_on and _gbd.positive_prompt_conditional_add_enabled),
                                        append_separator=_gbd.prompt_append_separator,
                                    )
                                    _gneg = _apply_prompt_filters(
                                        _gneg_orig,
                                        append=_active_filter_entries(_gbd.negative_prompt_append, _gpf_on and _gbd.negative_prompt_append_enabled),
                                        remove=_active_filter_entries(_gbd.negative_prompt_remove, _gpf_on and _gbd.negative_prompt_remove_enabled),
                                        replace=_active_filter_entries(_gbd.negative_prompt_replace, _gpf_on and _gbd.negative_prompt_replace_enabled),
                                        remove_cleanup_separators=_gbd.prompt_remove_cleanup_separators,
                                        append_separator=_gbd.prompt_append_separator,
                                        remove_whole_word=_gbd.prompt_remove_whole_word,
                                        remove_case_sensitive=_gbd.prompt_remove_case_sensitive,
                                    )
                                    _gneg = _apply_conditional_add(
                                        _gneg,
                                        _active_filter_entries(_gbd.negative_prompt_conditional_add, _gpf_on and _gbd.negative_prompt_conditional_add_enabled),
                                        append_separator=_gbd.prompt_append_separator,
                                    )
                                    _gswap = _active_filter_entries(_gbd.prompt_swap, _gpf_on and _gbd.prompt_swap_enabled)
                                    if _gswap:
                                        _gpos, _gneg = _apply_prompt_swap(
                                            _gpos, _gneg, _gswap,
                                            remove_whole_word=_gbd.prompt_remove_whole_word,
                                            remove_case_sensitive=_gbd.prompt_remove_case_sensitive,
                                            remove_cleanup_separators=_gbd.prompt_remove_cleanup_separators,
                                            append_separator=_gbd.prompt_append_separator,
                                        )
                                    gallery_image["positive_prompt"] = _gpos.strip()
                                    gallery_image["negative_prompt"] = _gneg.strip()
                                    if _gpos.strip() != _gpos_orig:
                                        gallery_image["original_positive_prompt"] = _gpos_orig
                                    if _gneg.strip() != _gneg_orig:
                                        gallery_image["original_negative_prompt"] = _gneg_orig
                            if completed_job_info.time_to_generate is not None:
                                gallery_image["time_to_generate"] = completed_job_info.time_to_generate
                            self.webui.add_gallery_image(gallery_image)
                    except (FileNotFoundError, OSError) as e:
                        logger.warning(f"Failed to read saved images for webui preview: {e}")
                        # Don't fallback to job_image_results to avoid showing censored placeholders
                        logger.debug("WebUI preview will not be updated for this job (no disk images available)")
                # Note: We intentionally don't update WebUI if no saved images are available
                # We don't want to show potentially censored images from job_image_results

                # Record in faulted jobs history if the safety evaluation faulted this job.
                # Jobs that ended as csam/censored are excluded (state != faulted) since they
                # were submitted successfully via a different state code path.
                if completed_job_info.state == GENERATION_STATE.faulted:
                    self._record_faulted_job_history(
                        completed_job_info.sdk_api_job_info,
                        fault_phase=HordeProcessState.SAFETY_EVALUATING.name,
                    )

                self._move_pending_process_timings_to_completed_job(
                    process_id=message.process_id,
                    sdk_api_job_info=completed_job_info.sdk_api_job_info,
                )
                self.jobs_pending_submit.append(completed_job_info)

    def get_processes_with_model_for_queued_job(self) -> list[int]:
        """Get the processes that have the model for any queued job."""
        processes_with_model_for_queued_job: list[int] = []

        # Get set of model names from queued and in-progress jobs
        required_model_names = {job.model for job in self.jobs_pending_inference}
        required_model_names.update(job.model for job in self.jobs_in_progress)

        for p in self._process_map.values():
            if (
                p.loaded_horde_model_name in required_model_names
                or p.last_process_state == HordeProcessState.MODEL_PRELOADED
            ):
                processes_with_model_for_queued_job.append(p.process_id)

        return processes_with_model_for_queued_job

    _preload_delay_notified = False

    def preload_models(self) -> bool:
        """Preload models that are likely to be used soon.

        Returns:
            True if a model was preloaded, False otherwise.
        """
        loaded_models = {process.loaded_horde_model_name for process in self._process_map.values()}
        loaded_models = loaded_models.union(
            model.horde_model_name
            for model in self._horde_model_map.root.values()
            if model.horde_model_load_state.is_loaded() or model.horde_model_load_state == ModelLoadState.LOADING
        )

        pending_models = {job.model for job in self.jobs_pending_inference}
        for process in self._process_map.values():
            if (
                process.last_process_state == HordeProcessState.MODEL_PRELOADED
            ) and process.loaded_horde_model_name not in pending_models:
                logger.debug(
                    f"Clearing preloaded model {process.loaded_horde_model_name} "
                    f"from process {process.process_id} as it is no longer needed",
                )
                self._on_process_state_change(
                    process_id=process.process_id,
                    new_state=HordeProcessState.WAITING_FOR_JOB,
                )

        if loaded_models == pending_models:
            return False

        # Fault any pending jobs whose models are currently in the preload cooldown so that
        # the horde can re-assign them rather than leaving them stranded in our local queue.
        self._fault_cooldown_model_jobs()

        # Starting from the left of the deque, preload models that are not yet loaded up to the
        # number of inference processes that are available
        for job in self.jobs_pending_inference:
            if job.model is None:
                raise ValueError(f"job.model is None ({job})")

            if job.model in loaded_models:
                continue

            # Do not attempt to preload a model that is in the hung-preloading cooldown.
            # The cooldown is triggered by _record_preload_stuck_failure() (called from
            # replace_hung_processes()) after _PRELOAD_STUCK_FAILURE_THRESHOLD consecutive
            # MODEL_PRELOADING timeouts for this model within _PRELOAD_STUCK_FAILURE_WINDOW
            # seconds.  Skipping here prevents the worker from cycling indefinitely through a
            # model that cannot be loaded on this machine.
            if self._is_model_in_preload_cooldown(job.model):
                logger.opt(colors=True).debug(
                    "<fg #7b7d7d>"
                    f"Skipping preload of {job.model!r}: model is in preload cooldown"
                    "</>",
                )
                continue

            processes_with_model_for_queued_job: list[int] = self.get_processes_with_model_for_queued_job()

            # If the number of still active inference processes is less than the number of jobs in the deque or in
            # progress then we use all processes that are active
            if self._process_map.num_loaded_inference_processes() < (
                len(self.jobs_pending_inference) + len(self.jobs_in_progress)
            ):
                processes_with_model_for_queued_job = [
                    p.process_id for p in self._process_map.values() if p.is_process_busy()
                ]

            available_process = self._process_map.get_first_available_inference_process(
                disallowed_processes=processes_with_model_for_queued_job,
            )

            if available_process is None:
                return False

            if (
                available_process.last_process_state != HordeProcessState.WAITING_FOR_JOB
                and available_process.loaded_horde_model_name is not None
                and self.bridge_data.cycle_process_on_model_change
                and not self._shutting_down
            ):
                # We're going to restart the process and then exit the loop, because
                # available_process is very quickly _not_ going to be available.
                # We also don't want to block waiting for the newly forked job to become
                # available, so we'll wait for it to become ready before scheduling a model
                # to be loaded on it.
                self._replace_inference_process(available_process)
                return False

            num_preloading_processes = self._process_map.num_preloading_processes()

            at_least_one_preloading_process = num_preloading_processes >= 1
            very_fast_disk_mode_enabled = self.bridge_data.very_fast_disk_mode
            if very_fast_disk_mode_enabled:
                max_concurrent_inference_processes_reached = num_preloading_processes >= (
                    self._max_concurrent_inference_processes + 1
                )
            else:
                max_concurrent_inference_processes_reached = (
                    num_preloading_processes >= self._max_concurrent_inference_processes
                )

            if (not very_fast_disk_mode_enabled and at_least_one_preloading_process) or (
                very_fast_disk_mode_enabled and max_concurrent_inference_processes_reached
            ):
                if not self._preload_delay_notified:
                    logger.opt(colors=True).info(
                        "<fg #7b7d7d>"
                        f"Already preloading {num_preloading_processes} models, waiting for one to finish before "
                        f"preloading {job.model}"
                        "</>",
                    )
                    self._preload_delay_notified = True
                return False

            self._preload_delay_notified = False
            logger.debug(f"Preloading model {job.model} on process {available_process.process_id}")
            logger.debug(f"Available inference processes: {self._process_map}")
            only_active_models = {
                model_name: model_info
                for model_name, model_info in self._horde_model_map.root.items()
                if model_info.horde_model_load_state.is_active()
            }
            logger.debug(f"Horde model map (active): {only_active_models}")

            will_load_loras = job.payload.loras is not None and len(job.payload.loras) > 0
            seamless_tiling_enabled = job.payload.tiling is not None and job.payload.tiling

            if available_process.safe_send_message(
                HordePreloadInferenceModelMessage(
                    control_flag=HordeControlFlag.PRELOAD_MODEL,
                    horde_model_name=job.model,
                    will_load_loras=will_load_loras,
                    seamless_tiling_enabled=seamless_tiling_enabled,
                    sdk_api_job_info=job,
                ),
            ):
                available_process.last_control_flag = HordeControlFlag.PRELOAD_MODEL

                self._horde_model_map.update_entry(
                    horde_model_name=job.model,
                    load_state=ModelLoadState.LOADING,
                    process_id=available_process.process_id,
                )

                model_baseline = self.get_model_baseline(job.model)

                self._process_map.on_model_load_state_change(
                    process_id=available_process.process_id,
                    horde_model_name=job.model,
                    horde_model_baseline=model_baseline,
                    last_job_referenced=job,
                )

                # Immediately update the process state so the status display reflects the
                # correct state without waiting for the child process to report back.
                # Use on_process_state_change to also refresh last_received_timestamp so that
                # hung-process detection does not trigger prematurely.
                self._on_process_state_change(
                    process_id=available_process.process_id,
                    new_state=HordeProcessState.MODEL_PRELOADING,
                )
            else:
                # safe_send_message() failed (the exception is stored in last_send_error).
                # Replace the process immediately so that the next loop iteration selects a
                # healthy process instead of retrying the same send indefinitely (which would
                # block start_inference() from ever being called and leave jobs stuck in the queue).
                send_error = available_process.last_send_error
                send_error_detail = (
                    f" ({type(send_error).__name__}: {send_error})" if send_error is not None else ""
                )
                job_id = getattr(job, "id_", None) or getattr(job, "ids", None)
                logger.error(
                    f"Failed to send preload model message for model {job.model!r}"
                    f"{f' (job_id={job_id})' if job_id is not None else ''} "
                    f"to process {available_process.process_id}{send_error_detail}",
                )
                self._replace_inference_process(available_process)

            return True

        return False

    _model_recently_missing = False
    _model_recently_missing_time = 0.0

    _skipped_line_next_job_and_process: NextJobAndProcess | None = None

    def get_next_job_and_process(
        self,
        information_only: bool = False,
    ) -> NextJobAndProcess | None:
        """Get the next job and process that can be started, if any.

        Returns:
            NextJobAndProcess if a job can be started, None otherwise.
        """
        if self._skipped_line_next_job_and_process is not None:
            return self._skipped_line_next_job_and_process

        next_job: ImageGenerateJobPopResponse | None = None
        next_n_jobs: list[ImageGenerateJobPopResponse] = []
        for job in self.jobs_pending_inference:
            if job in self.jobs_in_progress:
                continue
            if next_job is None:
                next_job = job

            next_n_jobs.append(job)

        if next_job is None:
            return None

        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

        processes_post_processing = 0
        if self.post_process_job_overlap_allowed:
            processes_post_processing = self._process_map.num_busy_with_post_processing()

        if len(self.jobs_in_progress) >= (self.max_concurrent_inference_processes + processes_post_processing):
            return None

        process_with_model = self._process_map.get_process_by_horde_model_name(next_job.model)
        skipped_line = False
        skipped_line_for = None

        def handle_process_missing(job: ImageGenerateJobPopResponse) -> None:
            if self._model_recently_missing:
                # We don't want to spam the logs
                return
            logger.warning(
                f"Expected to find a process with model {job.model} but none was found. Attempt to load it now...",
            )
            logger.debug(f"Horde model map: {self._horde_model_map}")
            logger.debug(f"Process map: {self._process_map}")

            if job.model is not None:
                logger.debug(f"Expiring entry for model {job.model}")
                self._horde_model_map.expire_entry(job.model)

                if process_with_model is not None:
                    logger.debug(f"Clearing process {process_with_model.process_id} of model {job.model}")

                    horde_model_baseline = self.get_model_baseline(job.model)

                    self._process_map.on_model_load_state_change(
                        process_id=process_with_model.process_id,
                        horde_model_name=job.model,
                        horde_model_baseline=horde_model_baseline,
                    )

                logger.debug(f"Horde model map: {self._horde_model_map}")
                logger.debug(f"Process map: {self._process_map}")

                self._model_recently_missing = True

                logger.debug(f"Last missing time: {self._model_recently_missing_time}")
                self._model_recently_missing_time = time.time()

                try:
                    self.jobs_in_progress.remove(job)
                except ValueError:
                    logger.debug(f"Job {job.id_} not found in jobs_in_progress.")

        candidate_job_size = 25

        if self.bridge_data.high_performance_mode:
            candidate_job_size = 100

        elif self.bridge_data.moderate_performance_mode:
            candidate_job_size = 50

        def find_line_skip_candidate(
            *,
            require_small_non_lora_job: bool,
        ) -> tuple[ImageGenerateJobPopResponse, HordeProcessInfo] | None:
            for candidate_job in next_n_jobs:
                if candidate_job.model is None or candidate_job.model == next_job.model:
                    continue

                if require_small_non_lora_job:
                    job_has_loras = candidate_job.payload.loras is not None and len(candidate_job.payload.loras) > 0
                    if job_has_loras:
                        continue
                    if self.get_single_job_effective_megapixelsteps(candidate_job) > candidate_job_size:
                        continue

                candidate_process_with_model = self._process_map.get_process_by_horde_model_name(
                    candidate_job.model,
                )
                if candidate_process_with_model is not None and candidate_process_with_model.can_accept_job():
                    return candidate_job, candidate_process_with_model

            return None

        if process_with_model is None:
            # The head-of-queue job's model is not loaded on any process yet — e.g. it is still
            # waiting to be preloaded, its preload is delayed by the one-model-at-a-time
            # serialization (see preload_models), or the process that was loading it died/was
            # replaced. Before giving up (which would block the entire queue behind the head job
            # and leave any already-preloaded process sitting idle), try to start another pending
            # job whose model IS already on a process that can accept a job.
            #
            # Without this fallback, a single slow/stuck/missing head-of-queue model stalls all
            # other ready work until that model finishes loading or its process is replaced (up to
            # preload_timeout seconds). That presents exactly as the reported symptom: "models are
            # loaded/preloaded but no inference starts for a while after startup", which then
            # self-resolves once the head model recovers.
            chosen_candidate_and_process = find_line_skip_candidate(require_small_non_lora_job=False)
            if chosen_candidate_and_process is not None:
                chosen_candidate, chosen_process = chosen_candidate_and_process
                skipped_line = True
                skipped_line_for = next_job
                next_job = chosen_candidate
                self._skipped_line_job = next_job
                process_with_model = chosen_process
            else:
                if (
                    self._preload_delay_notified
                    or self._horde_model_map.is_model_loading(next_job.model)
                    or information_only
                ):
                    return None
                handle_process_missing(next_job)
                return None

        if not process_with_model.can_accept_job():
            if (process_with_model.last_process_state == HordeProcessState.DOWNLOADING_AUX_MODEL) or (
                self.post_process_job_overlap_allowed
                and (
                    process_with_model.last_process_state == HordeProcessState.INFERENCE_POST_PROCESSING
                    or process_with_model.last_process_state == HordeProcessState.POST_PROCESSING_STARTING
                )
            ):
                # If any of the next n jobs (other than this one) aren't using the same model, see if that job
                # has a model that's already loaded.
                # If it does, we'll start inference on that job instead.
                chosen_candidate_and_process = find_line_skip_candidate(require_small_non_lora_job=True)
                if chosen_candidate_and_process is None:
                    return None
                chosen_candidate, chosen_process = chosen_candidate_and_process
            elif process_with_model.last_process_state == HordeProcessState.MODEL_PRELOADING:
                # The first job's model is still being preloaded. Look for other pending jobs
                # that already have a preloaded process ready to accept a job. This prevents
                # MODEL_PRELOADED processes from being stuck waiting while the queue is blocked
                # by a MODEL_PRELOADING process at the front.
                chosen_candidate_and_process = find_line_skip_candidate(require_small_non_lora_job=False)
                if chosen_candidate_and_process is None:
                    return None
                chosen_candidate, chosen_process = chosen_candidate_and_process
            else:
                # The first job's process exists but is not in a state that can accept
                # a job (e.g. UNLOADED_MODEL_FROM_RAM). Look for other pending jobs that
                # already have a preloaded process ready. This prevents MODEL_PRELOADED
                # processes from being permanently starved when a blocking process holds
                # the model name for the head-of-queue job.
                chosen_candidate_and_process = find_line_skip_candidate(require_small_non_lora_job=False)
                if chosen_candidate_and_process is None:
                    return None
                chosen_candidate, chosen_process = chosen_candidate_and_process

            # A candidate job/process pair was found via line-skipping.
            skipped_line = True
            skipped_line_for = next_job
            next_job = chosen_candidate
            self._skipped_line_job = next_job
            process_with_model = chosen_process

        self._model_recently_missing = False

        next_job_and_process = NextJobAndProcess(
            next_job=next_job,
            process_with_model=process_with_model,
            skipped_line=skipped_line,
            skipped_line_for=skipped_line_for,
        )

        if skipped_line:
            self._skipped_line_next_job_and_process = next_job_and_process

        return next_job_and_process

    def start_inference(self) -> bool:
        """Start inference for the next job in jobs_pending_inference, if possible.

        Returns:
            True if inference was started, False otherwise.
        """
        next_job_and_process = self.get_next_job_and_process()

        if next_job_and_process is None:
            return False

        process_with_model = next_job_and_process.process_with_model
        next_job = next_job_and_process.next_job

        if next_job_and_process.skipped_line and next_job_and_process.skipped_line_for is not None:
            blocked_state = "UNKNOWN"
            if next_job_and_process.skipped_line_for.model is not None:
                blocked_process = self._process_map.get_process_by_horde_model_name(
                    next_job_and_process.skipped_line_for.model,
                )
                if blocked_process is not None:
                    blocked_state = blocked_process.last_process_state.name
            logger.info(
                f"Job {next_job_and_process.next_job.id_} skipped the line and will be run on process "
                f"{process_with_model.process_id} before job {next_job_and_process.skipped_line_for.id_}"
                f" which is currently blocked in state {blocked_state}.",
            )

        processes_post_processing = 0
        if self.post_process_job_overlap_allowed:
            processes_post_processing = self._process_map.num_busy_with_post_processing()

        if processes_post_processing > 0 and len(self.jobs_in_progress) >= self.max_concurrent_inference_processes:
            logger.debug(
                "Proceeding with inference, but post processing is still running on "
                f"{processes_post_processing} processes",
            )

        # Unload all models from vram from any other process that isn't running a job if configured to do so
        if self.bridge_data.unload_models_from_vram_often:
            self.unload_models_from_vram(process_with_model)

        color_format_string = "<fg #f0beff>{message}</>"

        logger.opt(colors=True).info(
            color_format_string.format(
                message=f"Starting inference for job {str(next_job.id_)[:8]} "
                f"on process {process_with_model.process_id}",
            ),
        )

        # region Log job info
        if next_job.model is None:
            raise ValueError(f"next_job.model is None ({next_job})")

        logger.opt(colors=True).info(
            color_format_string.format(
                message=f"  Model: {next_job.model}",
            ),
        )
        if next_job.source_image is not None:
            logger.opt(colors=True).info(
                color_format_string.format(
                    message="  Using source image",
                ),
            )

        extra_info = ""
        if next_job.payload.control_type is not None:
            extra_info += f"Control type: {next_job.payload.control_type}"
        if next_job.payload.loras:
            if extra_info:
                extra_info += ", "
            extra_info += f"{len(next_job.payload.loras)} LoRAs"
        if next_job.payload.tis:
            if extra_info:
                extra_info += ", "
            extra_info += f"{len(next_job.payload.tis)} TIs"
        if next_job.payload.post_processing is not None and len(next_job.payload.post_processing) > 0:
            if extra_info:
                extra_info += ", "
            extra_info += f"Post processing: {next_job.payload.post_processing}"
        if next_job.payload.hires_fix:
            if extra_info:
                extra_info += ", "
            extra_info += "HiRes fix"

        if next_job.payload.workflow is not None:
            if extra_info:
                extra_info += ", "
            extra_info += f"Workflow: {next_job.payload.workflow}"

        if extra_info:
            logger.opt(colors=True).info(
                color_format_string.format(
                    message=f"  {extra_info}",
                ),
            )

        logger.opt(colors=True).info(
            color_format_string.format(
                message=f"  {next_job.payload.width}x{next_job.payload.height} for "
                f"{next_job.payload.ddim_steps} steps "
                f"with sampler {next_job.payload.sampler_name} for a batch of {next_job.payload.n_iter}",
            ),
        )

        logger.debug(f"All Batch IDs: {next_job.ids}")
        # endregion

        # We store the amount of batches this job will do,
        # as we use that later to check if we should start inference in parallel
        process_with_model.batch_amount = next_job.payload.n_iter

        # Apply prompt filters for local generation only — build a filtered copy of the job
        # to send to the inference child. next_job is never mutated so gen_metadata and Horde
        # submission always use the original prompt. model_copy() works on frozen models.
        _job_to_dispatch = next_job
        _original_prompt = next_job.payload.prompt
        _bd = self.bridge_data
        _pf_on = _bd.prompt_filters_enabled
        _pos_append        = _active_filter_entries(_bd.positive_prompt_append,          _pf_on and _bd.positive_prompt_append_enabled)
        _pos_remove        = _active_filter_entries(_bd.positive_prompt_remove,          _pf_on and _bd.positive_prompt_remove_enabled)
        _pos_replace       = _active_filter_entries(_bd.positive_prompt_replace,         _pf_on and _bd.positive_prompt_replace_enabled)
        _pos_cond_add      = _active_filter_entries(_bd.positive_prompt_conditional_add, _pf_on and _bd.positive_prompt_conditional_add_enabled)
        _neg_append        = _active_filter_entries(_bd.negative_prompt_append,          _pf_on and _bd.negative_prompt_append_enabled)
        _neg_remove        = _active_filter_entries(_bd.negative_prompt_remove,          _pf_on and _bd.negative_prompt_remove_enabled)
        _neg_replace       = _active_filter_entries(_bd.negative_prompt_replace,         _pf_on and _bd.negative_prompt_replace_enabled)
        _neg_cond_add      = _active_filter_entries(_bd.negative_prompt_conditional_add, _pf_on and _bd.negative_prompt_conditional_add_enabled)
        _swap_entries      = _active_filter_entries(_bd.prompt_swap,                     _pf_on and _bd.prompt_swap_enabled)
        if _original_prompt and (_pos_append or _pos_remove or _pos_replace or _pos_cond_add or _neg_append or _neg_remove or _neg_replace or _neg_cond_add or _swap_entries):
            _parts = _original_prompt.split("###", 1)
            _pos = _apply_prompt_filters(
                _parts[0],
                append=_pos_append,
                remove=_pos_remove,
                replace=_pos_replace,
                remove_cleanup_separators=_bd.prompt_remove_cleanup_separators,
                append_separator=_bd.prompt_append_separator,
                remove_whole_word=_bd.prompt_remove_whole_word,
                remove_case_sensitive=_bd.prompt_remove_case_sensitive,
            )
            _pos = _apply_conditional_add(_pos, _pos_cond_add, append_separator=_bd.prompt_append_separator)
            _neg = _apply_prompt_filters(
                _parts[1] if len(_parts) > 1 else "",
                append=_neg_append,
                remove=_neg_remove,
                replace=_neg_replace,
                remove_cleanup_separators=_bd.prompt_remove_cleanup_separators,
                append_separator=_bd.prompt_append_separator,
                remove_whole_word=_bd.prompt_remove_whole_word,
                remove_case_sensitive=_bd.prompt_remove_case_sensitive,
            )
            _neg = _apply_conditional_add(_neg, _neg_cond_add, append_separator=_bd.prompt_append_separator)
            if _swap_entries:
                _pos, _neg = _apply_prompt_swap(
                    _pos, _neg, _swap_entries,
                    remove_whole_word=_bd.prompt_remove_whole_word,
                    remove_case_sensitive=_bd.prompt_remove_case_sensitive,
                    remove_cleanup_separators=_bd.prompt_remove_cleanup_separators,
                    append_separator=_bd.prompt_append_separator,
                )
            _filtered_prompt = f"{_pos.strip()}###{_neg.strip()}"
            if _filtered_prompt != _original_prompt:
                _filtered_payload = next_job.payload.model_copy(update={"prompt": _filtered_prompt})
                _job_to_dispatch = next_job.model_copy(update={"payload": _filtered_payload})

        _send_succeeded = process_with_model.safe_send_message(
            HordeInferenceControlMessage(
                control_flag=HordeControlFlag.START_INFERENCE,
                horde_model_name=next_job.model,
                sdk_api_job_info=_job_to_dispatch,
            ),
        )

        if _send_succeeded:
            self.jobs_in_progress.append(next_job)

            process_with_model.last_control_flag = HordeControlFlag.START_INFERENCE
            process_with_model.last_job_referenced = next_job
            process_with_model.loaded_horde_model_name = next_job.model
            horde_model_baseline = self.get_model_baseline(next_job.model)
            process_with_model.loaded_horde_model_baseline = horde_model_baseline

            # Immediately update the process state so the status display reflects the
            # correct state without waiting for the child process to report back.
            # Use on_process_state_change to also refresh last_received_timestamp so that
            # hung-process detection does not trigger prematurely.
            self._on_process_state_change(
                process_id=process_with_model.process_id,
                new_state=HordeProcessState.INFERENCE_STARTING,
            )
            # Also update the model map to IN_USE here, because the INFERENCE_STARTING
            # state-change message from the child will be skipped (duplicate state).
            self._horde_model_map.update_entry(
                horde_model_name=next_job.model,
                load_state=ModelLoadState.IN_USE,
                process_id=process_with_model.process_id,
            )

        else:
            send_error = process_with_model.last_send_error
            send_error_detail = (
                f" ({type(send_error).__name__}: {send_error})" if send_error is not None else ""
            )
            logger.error(
                f"Failed to start inference for job {next_job.id_} on process "
                f"{process_with_model.process_id}{send_error_detail}",
            )
            # The pipe to the child process is broken.  Replace the dead/unresponsive process
            # now so that the job retry is dispatched to a healthy worker instead of hitting
            # the same broken pipe again.  Capture last_job_referenced first because
            # _replace_inference_process may fault it internally; we only call handle_job_fault
            # for next_job if it is a different job (avoiding a double-fault when the same
            # job is being retried on the same process that has now lost its pipe).
            last_referenced = process_with_model.last_job_referenced
            self._replace_inference_process(process_with_model)
            if last_referenced is None or next_job.id_ != last_referenced.id_:
                fault_info = f"pipe broken{send_error_detail}"
                self.handle_job_fault(faulted_job=next_job, process_info=process_with_model, fault_info=fault_info)

        self._skipped_line_next_job_and_process = None

        return True

    def unload_models_from_vram(
        self,
        process_with_model: HordeProcessInfo,
    ) -> None:
        """Unload models from VRAM from processes that are not running a job.

        Args:
            process_with_model: The process that is running a job.
        """
        next_n_models = list(self.get_next_n_models(self.max_inference_processes))
        logger.debug(f"Next n models: {next_n_models}")
        next_model = None
        if len(next_n_models) > 0:
            next_model = next_n_models.pop()

        in_progress_models = {job.model for job in self.jobs_in_progress}

        for process_info in self._process_map.values():
            if process_info.process_id == process_with_model.process_id:
                continue

            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if process_info.is_process_busy():
                logger.debug(f"Process {process_info.process_id} is busy")
                # continue

            if process_info.loaded_horde_model_name is not None:
                if len(self.bridge_data.image_models_to_load) == 1:
                    logger.debug("Not unloading models from VRAM because there is only one model to load.")
                    continue

                # If this models is in progress, don't unload it
                if process_info.loaded_horde_model_name in in_progress_models:
                    continue

                # If the model would be used next, don't unload it
                if process_info.loaded_horde_model_name == next_model:
                    continue

                if process_info.last_control_flag != HordeControlFlag.UNLOAD_MODELS_FROM_VRAM:
                    logger.info(
                        f"Unloading model {process_info.loaded_horde_model_name} from VRAM on process "
                        f"{process_info.process_id}",
                    )
                    process_info.safe_send_message(
                        HordeControlModelMessage(
                            control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
                            horde_model_name=process_info.loaded_horde_model_name,
                        ),
                    )
                    process_info.last_job_referenced = None
                    process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_VRAM
            else:
                logger.debug(f"Unloading all models from VRAM on process {process_info.process_id}")
                if (
                    not process_info.safe_send_message(
                        HordeControlMessage(
                            control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_VRAM,
                        ),
                    )
                    and not self._shutting_down
                ):
                    self._replace_inference_process(process_info)

    def _unload_idle_inference_models(self) -> None:
        """Unload models from all inference processes that can accept a new job.

        ``can_accept_job()`` covers WAITING_FOR_JOB, MODEL_PRELOADED,
        MODEL_LOADED, INFERENCE_COMPLETE, and ALCHEMY_COMPLETE — i.e. every
        process that is holding a model in RAM/VRAM but is not actively
        running inference — so this reclaims the maximum amount of memory
        without interrupting in-flight jobs.

        Called immediately when job pops are paused and on every main-loop tick
        while paused, so processes that finish a job mid-pause are unloaded as
        soon as they return to an idle state.  The guards inside
        :meth:`unload_from_ram` make repeated calls safe (no duplicate sends).
        """
        for process_info in self._process_map.get_inference_processes():
            if process_info.can_accept_job() and process_info.loaded_horde_model_name is not None:
                self.unload_from_ram(process_info.process_id)

    def unload_from_ram(self, process_id: int) -> None:
        """Unload models from a process.

        Args:
            process_id: The process to unload models from.
        """
        if process_id not in self._process_map:
            raise ValueError(f"process_id {process_id} is not in the process map")

        process_info = self._process_map[process_id]

        if process_info.process_type != HordeProcessType.INFERENCE:
            logger.warning(f"{self._process_label(process_id)} is not an inference process, not unloading models")
            return

        if process_info.recently_unloaded_from_ram:
            return

        if process_info.last_control_flag == HordeControlFlag.UNLOAD_MODELS_FROM_RAM:
            return

        if process_info.loaded_horde_model_name is not None and self._horde_model_map.is_model_loaded(
            process_info.loaded_horde_model_name,
        ):
            logger.debug(f"Unloading model {process_info.loaded_horde_model_name} from RAM on process {process_id}")
            process_info.safe_send_message(
                HordeControlModelMessage(
                    control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
                    horde_model_name=process_info.loaded_horde_model_name,
                ),
            )

            self._horde_model_map.update_entry(
                horde_model_name=process_info.loaded_horde_model_name,
                load_state=ModelLoadState.ON_DISK,
                process_id=process_id,
            )

            process_info.last_job_referenced = None
            process_info.loaded_horde_model_name = None
            process_info.loaded_horde_model_baseline = None
            process_info.recently_unloaded_from_ram = True
            process_info.last_control_flag = HordeControlFlag.UNLOAD_MODELS_FROM_RAM

        else:
            # Check the process is not ending
            if (
                process_info.last_process_state == HordeProcessState.PROCESS_ENDING
                or process_info.last_process_state == HordeProcessState.PROCESS_ENDED
            ):
                return

            logger.debug(f"Unloading all models from RAM on process {process_id}")
            process_info.safe_send_message(
                HordeControlMessage(
                    control_flag=HordeControlFlag.UNLOAD_MODELS_FROM_RAM,
                ),
            )
        logger.debug(f"Clearing process {process_id} of model {process_info.loaded_horde_model_name}")
        self._process_map.on_model_ram_clear(process_id=process_id, clear_job_reference=True)

    def get_next_n_models(self, n: int) -> list[str]:
        """Get the next n models that will be used in the job deque.

        Args:
            n: The number of models to get.

        Returns:
            A list of the next n models that will be used in the job deque.
        """
        next_n_models: list[str] = []
        jobs_traversed = 0
        while len(next_n_models) < n:
            if jobs_traversed >= len(self.jobs_pending_inference):
                break

            model_name = self.jobs_pending_inference[jobs_traversed].model

            if model_name is None:
                raise ValueError(f"job_deque[{jobs_traversed}].model is None")

            if model_name not in next_n_models:
                next_n_models.append(model_name)

            jobs_traversed += 1

        return next_n_models

    def unload_models(self) -> bool:
        """Unload models that are no longer needed and would use above the limit specified.

        Returns:
            True if a model was unloaded, False otherwise.
        """
        if len(self.jobs_pending_inference) == 0:
            return False

        # 1 thread, 1 model, no need to unload as it should always be in use (or at least available)
        if self._max_concurrent_inference_processes == 1 and len(self.bridge_data.image_models_to_load) == 1:
            return False

        required_model_names = {job.model for job in self.jobs_pending_inference}
        required_model_names.update(job.model for job in self.jobs_in_progress)

        for process_info in self._process_map.values():
            if process_info.process_type != HordeProcessType.INFERENCE:
                continue

            if (
                process_info.is_process_busy()
                or process_info.last_process_state == HordeProcessState.MODEL_PRELOADED
            ):
                continue

            if process_info.loaded_horde_model_name is not None:
                if self._horde_model_map.is_model_loading(process_info.loaded_horde_model_name):
                    continue

                model_entry = self._horde_model_map.root.get(process_info.loaded_horde_model_name)
                if model_entry is not None and model_entry.horde_model_load_state == ModelLoadState.IN_USE:
                    continue

                if process_info.loaded_horde_model_name in required_model_names:
                    continue

                self.unload_from_ram(process_info.process_id)
                return True

        return False

    def start_evaluate_safety(self) -> None:
        """Start evaluating the safety of the next job pending a safety check, if any."""
        if len(self.jobs_pending_safety_check) == 0:
            return

        safety_process = self._process_map.get_first_available_safety_process()

        if safety_process is None:
            return

        completed_job_info = self.jobs_pending_safety_check[0]

        if self.stable_diffusion_reference is None:
            raise ValueError("stable_diffusion_reference is None")

        critical_fault = False

        if completed_job_info.job_image_results is None:
            logger.error("completed_job_info.job_image_results is None")
            critical_fault = True

        if completed_job_info.sdk_api_job_info.id_ is None:
            logger.error("completed_job_info.sdk_api_job_info.id_ is None")
            critical_fault = True

        if completed_job_info.sdk_api_job_info.model is None:
            logger.error("completed_job_info.sdk_api_job_info.model is None")
            critical_fault = True

        if completed_job_info.sdk_api_job_info.payload.prompt is None:
            logger.error("completed_job_info.sdk_api_job_info.payload.prompt is None")
            critical_fault = True

        if critical_fault:
            self.handle_job_fault(faulted_job=completed_job_info.sdk_api_job_info, process_info=safety_process)
            logger.error(f"Failed to start safety evaluation for job {completed_job_info.sdk_api_job_info.id_}")
            self.jobs_pending_safety_check.remove(completed_job_info)

            return

        # Duplicated for static type checking
        if completed_job_info.sdk_api_job_info.id_ is None:
            raise ValueError("completed_job_info.sdk_api_job_info.id_ is None")
        if completed_job_info.sdk_api_job_info.payload.prompt is None:
            raise ValueError("completed_job_info.sdk_api_job_info.payload.prompt is None")
        if completed_job_info.sdk_api_job_info.model is None:
            raise ValueError("completed_job_info.sdk_api_job_info.model is None")

        # Custom models don't appear in the downloaded model reference
        model_info = {}
        if completed_job_info.sdk_api_job_info.model in self.stable_diffusion_reference.root:
            model_info = self.stable_diffusion_reference.root[completed_job_info.sdk_api_job_info.model].model_dump()

        generation_metadata: dict[str, object] = {}
        try:
            generation_metadata.update(completed_job_info.sdk_api_job_info.payload.model_dump(exclude_none=True))
        except Exception as e:
            logger.warning(f"Failed to dump generation metadata: {type(e).__name__} {e}")
        generation_metadata["model"] = completed_job_info.sdk_api_job_info.model

        # ! IMPORTANT: Start own code
        lora_descriptions: list[str] = []
        try:
            loras = completed_job_info.sdk_api_job_info.payload.loras or []
            for lora in loras:
                name = None
                strength = None

                if isinstance(lora, dict):
                    name = lora.get("name") or lora.get("lora_name") or lora.get("model") or lora.get("id")
                    strength = lora.get("strength") or lora.get("weight") or lora.get("clip") or lora.get("alpha")
                else:
                    name = (
                        getattr(lora, "name", None)
                        or getattr(lora, "lora_name", None)
                        or getattr(lora, "model", None)
                        or getattr(lora, "id", None)
                    )
                    strength = (
                        getattr(lora, "strength", None)
                        or getattr(lora, "weight", None)
                        or getattr(lora, "clip", None)
                        or getattr(lora, "alpha", None)
                    )

                if name is None:
                    name = str(lora)

                if strength is None:
                    strength = 1.0

                lora_descriptions.append(f"{name}:{strength}")
        except Exception as e:
            logger.warning(f"Failed to build LoRA descriptions: {type(e).__name__} {e}")

        generation_metadata["lora_descriptions"] = lora_descriptions
        # ! IMPORTANT: End own code

        if completed_job_info.sanitized_negative_prompt is not None:
            generation_metadata["sanitized_negative_prompt"] = completed_job_info.sanitized_negative_prompt

        safety_message_sent_succeeded = safety_process.safe_send_message(
            HordeSafetyControlMessage(
                control_flag=HordeControlFlag.EVALUATE_SAFETY,
                job_id=completed_job_info.sdk_api_job_info.id_,
                images_base64=completed_job_info.images_base64,
                prompt=completed_job_info.sdk_api_job_info.payload.prompt,
                censor_nsfw=completed_job_info.sdk_api_job_info.payload.use_nsfw_censor,
                sfw_worker=not self.bridge_data.nsfw,
                horde_model_info=model_info,
                generation_metadata=generation_metadata,
            ),
        )

        safety_process = self._process_map.get_safety_process()
        if not safety_message_sent_succeeded:
            if safety_process is None:
                return

            if (
                not safety_process.is_process_alive()
                or safety_process.last_process_state == HordeProcessState.PROCESS_STARTING
            ):
                return

            logger.error(f"Failed to start safety evaluation for job {completed_job_info.sdk_api_job_info.id_}")
            self._safety_processes_should_be_replaced = True
            if len(self.jobs_being_safety_checked) > 0:
                for job_info in self.jobs_being_safety_checked:
                    self.jobs_pending_safety_check.append(job_info)
        else:
            self.jobs_pending_safety_check.remove(completed_job_info)
            self.jobs_being_safety_checked.append(completed_job_info)

    def base64_image_to_stream_buffer(self, image_base64: str) -> BytesIO | None:
        """Convert a base64 image to a BytesIO stream buffer.

        Args:
            image_base64: The base64 image to convert.

        Returns:
            A BytesIO stream buffer containing the image, or None if the conversion failed.
        """
        try:
            with PIL.Image.open(BytesIO(base64.b64decode(image_base64))) as image_as_pil:
                image_buffer = BytesIO()
                image_as_pil.save(
                    image_buffer,
                    format="WebP",
                    quality=95,
                    method=6,
                )

            return image_buffer
        except Exception as e:
            logger.error(f"Failed to convert base64 image to stream buffer: {e}")
            return None

    _num_job_slowdowns = 0
    """The number of jobs which did not meet the minimum expected kudos/second rate."""

    @logger.catch(reraise=True)
    async def submit_single_generation(self, new_submit: PendingSubmitJob) -> PendingSubmitJob:
        """Tries to upload and submit a single image from a batch.

        Args:
            new_submit: The job to attempt to submit.

        Returns:
            The modified in place job with the results of the submission attempt.
        """
        logger.debug(f"Preparing to submit job {new_submit.job_id}")

        if new_submit.image_result is None and not new_submit.is_faulted:
            logger.error(f"Job {new_submit.job_id} has no image result")
            if new_submit.completed_job_info.state == GENERATION_STATE.faulted:
                self._num_jobs_faulted += 1
            new_submit.fault()
            return new_submit

        if new_submit.image_result is not None:
            image_in_buffer = self.base64_image_to_stream_buffer(
                new_submit.image_result.image_base64,
            )
            if image_in_buffer is None:
                logger.critical(
                    f"There is an invalid image in the job results for {new_submit.job_id}, "
                    "removing from completed jobs",
                )
                for (
                    follow_up_request
                ) in new_submit.completed_job_info.sdk_api_job_info.get_follow_up_failure_cleanup_request():
                    follow_up_response = await self.horde_client_session.submit_request(
                        follow_up_request,
                        JobSubmitResponse,
                    )

                    if isinstance(follow_up_response, RequestErrorResponse):
                        logger.error(f"Failed to submit followup request: {follow_up_response}")
                new_submit.fault()
                return new_submit

            async def _do_upload(new_submit: PendingSubmitJob, image_in_buffer_bytes: bytes) -> bool:
                async with self._aiohttp_client_session.put(
                    yarl.URL(new_submit.r2_upload, encoded=True),
                    data=image_in_buffer_bytes,
                    skip_auto_headers=["content-type"],
                    timeout=aiohttp.ClientTimeout(total=10),
                    ssl=sslcontext,
                ) as response:
                    if response.status == 500:
                        logger.warning(
                            "Retrying upload to R2. This is a cloudflare issue and only is a concern if "
                            "you see this message 5 or more times a minute.",
                        )
                        new_submit.retry()
                        return False
                    if response.status != 200:
                        logger.error(f"Failed to upload image to R2: {response}")
                        new_submit.retry()
                        return False
                return True

            try:
                submit_success = await asyncio.wait_for(
                    _do_upload(new_submit, image_in_buffer.getvalue()),
                    timeout=10 + 1,
                )
                if not submit_success:
                    return new_submit
            except _async_client_exceptions as e:
                logger.warning("Upload to AI Horde R2 timed out. Will retry.")
                logger.debug(f"{type(e).__name__}: {e}")
                new_submit.retry()
                return new_submit
            except Exception as e:
                logger.error(f"Failed to upload image to R2: {e}")
                logger.debug(f"{type(e).__name__}: {e}")
                new_submit.retry()
                return new_submit
        metadata = []
        if new_submit.image_result is not None:
            metadata = new_submit.image_result.generation_faults
            if new_submit.batch_count > 1:
                metadata.append(
                    GenMetadataEntry(
                        type=METADATA_TYPE.batch_index,
                        value=METADATA_VALUE.see_ref,
                        ref=str(new_submit.gen_iter),
                    ),
                )
        seed = 0
        if new_submit.completed_job_info.sdk_api_job_info.payload.seed is not None:
            seed = int(new_submit.completed_job_info.sdk_api_job_info.payload.seed)
        submit_job_request_type = new_submit.completed_job_info.sdk_api_job_info.get_follow_up_default_request_type()
        if new_submit.completed_job_info.state is None:
            logger.error(f"Job {new_submit.job_id} has no state, assuming faulted")
            new_submit.completed_job_info.state = GENERATION_STATE.faulted
            return new_submit
        submit_job_request = submit_job_request_type(
            apikey=self.bridge_data.api_key,
            id=new_submit.job_id,
            seed=seed,
            generation="R2",
            state=new_submit.completed_job_info.state,
            censored=bool(new_submit.completed_job_info.censored),
            gen_metadata=metadata,
        )
        logger.debug(f"Submitting job {new_submit.job_id}")

        # Find and update state to RESULT_SUBMITTING for the process that handled this job
        handling_process_id = None
        for process_id, process_info in self._process_map.items():
            if process_info.last_job_referenced == new_submit.completed_job_info.sdk_api_job_info:
                handling_process_id = process_id
                self._on_process_state_change(
                    process_id=process_id,
                    new_state=HordeProcessState.RESULT_SUBMITTING,
                    timing_sdk_api_job_info=new_submit.completed_job_info.sdk_api_job_info,
                )
                break

        try:
            job_submit_response = None
            try:
                job_submit_response = await asyncio.wait_for(
                    self.horde_client_session.submit_request(
                        submit_job_request,
                        JobSubmitResponse,
                    ),
                    timeout=10 + 1,
                )
            except _async_client_exceptions as e:
                logger.error(f"Job {new_submit.job_id} submission failed with {type(e).__name__}: {e}")
                # asyncio.wait_for can cancel submit_request, and other client/OS failures can
                # abort it before the SDK removes the request from _awaiting_requests.
                _remove_awaiting_request(self.horde_client_session, submit_job_request)
                new_submit.retry()
                return new_submit
            except asyncio.CancelledError:
                _remove_awaiting_request(self.horde_client_session, submit_job_request)
                raise
            except Exception as e:
                logger.error(f"Failed to submit job {new_submit.job_id}: {e}")
                _remove_awaiting_request(self.horde_client_session, submit_job_request)
                new_submit.retry()
                return new_submit

            # If the job submit response is an error,
            # log it and increment the number of consecutive failed job submits
            if isinstance(job_submit_response, RequestErrorResponse):
                if (
                    "Processing Job with ID" in job_submit_response.message
                    and "does not exist" in job_submit_response.message
                ):
                    logger.warning(f"Job {new_submit.job_id} does not exist, removing from completed jobs")
                    new_submit.fault()
                    return new_submit

                if "already submitted" in job_submit_response.message:
                    logger.debug(
                        f"Job {new_submit.job_id} has already been submitted, removing from completed jobs",
                    )
                    new_submit.fault()
                    return new_submit

                if "Please check your worker speed" in job_submit_response.message:
                    logger.error(job_submit_response.message)
                    new_submit.fault()
                    return new_submit

                error_string = (
                    f"Failed to submit job (API Error) " f"{new_submit.retry_attempts_string}: {job_submit_response}"
                )
                logger.error(error_string)
                new_submit.retry()
                return new_submit

            if job_submit_response is None:
                logger.error(f"Failed to submit job {new_submit.job_id}")
                new_submit.retry()
                return new_submit

            # Get the time the job was popped from the job deque
            async with self._job_pop_timestamps_lock:
                time_popped = self.job_pop_timestamps.get(new_submit.completed_job_info.sdk_api_job_info)
                if time_popped is None:
                    logger.warning(
                        f"Failed to get time_popped for job {new_submit.completed_job_info.sdk_api_job_info.id_}. "
                        "This is likely a bug.",
                    )
                    time_popped = time.time()

                elif time_popped == -1:
                    logger.warning(
                        f"Job {new_submit.completed_job_info.sdk_api_job_info.id_} will have an "
                        "incorrect kudos/second calculation.",
                    )
                    time_popped = time.time()

            kudos_per_second = 0.0

            if not new_submit.completed_job_info.time_to_generate:
                logger.error(
                    f"Job {new_submit.job_id} has no time_to_generate, ignoring.",
                )
                new_submit.completed_job_info.time_to_generate = 0.0
            else:
                kudos_per_second = job_submit_response.reward / new_submit.completed_job_info.time_to_generate

            # Logging is now done at the batch level in api_submit_job()
            # Track slowdowns and faults
            if new_submit.completed_job_info.state != GENERATION_STATE.faulted:
                kudos_per_second_for_batch = kudos_per_second * new_submit.batch_count
                if kudos_per_second_for_batch < 0.4:
                    self._num_job_slowdowns += 1
                self.total_num_completed_jobs += 1
            else:
                self._num_jobs_faulted += 1

            self.kudos_generated_this_session += job_submit_response.reward
            _event_time = time.time()
            self.kudos_events.append((_event_time, job_submit_response.reward))
            self.image_events.append((_event_time, new_submit.batch_count))
            # Trim expired events at the source so the lists stay bounded even when the
            # other pruning sites never run (webui disabled, user-info requests failing).
            _event_cutoff = _event_time - METRICS_CALCULATION_WINDOW_SECONDS
            if self.kudos_events and self.kudos_events[0][0] < _event_cutoff:
                self.kudos_events = [(ts, k) for ts, k in self.kudos_events if ts >= _event_cutoff]
            if self.image_events and self.image_events[0][0] < _event_cutoff:
                self.image_events = [(ts, c) for ts, c in self.image_events if ts >= _event_cutoff]
            model_name = new_submit.completed_job_info.sdk_api_job_info.model
            if model_name:
                self._images_per_model[model_name] = (
                    self._images_per_model.get(model_name, 0) + new_submit.batch_count
                )
            new_submit.succeed(job_submit_response.reward, kudos_per_second)

            # Update state to WAITING_FOR_JOB for the process that handled this job (reuse process_id from above)
            if handling_process_id is not None:
                self._on_process_state_change(
                    process_id=handling_process_id,
                    new_state=HordeProcessState.WAITING_FOR_JOB,
                    timing_sdk_api_job_info=new_submit.completed_job_info.sdk_api_job_info,
                )

            return new_submit
        finally:
            # Ensure the process is never left stuck in RESULT_SUBMITTING after any failure path.
            # On success the state has already been set to WAITING_FOR_JOB above, so the check
            # below is a no-op in that case.  The specific submission error is logged by the
            # individual except/error-response branches above.
            if handling_process_id is not None:
                stuck_proc = self._process_map.get(handling_process_id)
                if stuck_proc is not None and stuck_proc.last_process_state == HordeProcessState.RESULT_SUBMITTING:
                    logger.warning(
                        f"Process {handling_process_id} was left in RESULT_SUBMITTING after a failed "
                        f"submission for job {new_submit.job_id}; "
                        "resetting to WAITING_FOR_JOB to unblock job scheduling",
                    )
                    self._on_process_state_change(
                        process_id=handling_process_id,
                        new_state=HordeProcessState.WAITING_FOR_JOB,
                        timing_sdk_api_job_info=new_submit.completed_job_info.sdk_api_job_info,
                    )

    def _record_job_timing(self, state: str, elapsed: float) -> None:
        """Accumulate timing data for a job state.

        Args:
            state: Name of the job state/phase (e.g. "INFERENCE_PROCESSING", "TOTAL").
            elapsed: Time elapsed in seconds for that state.
        """
        if elapsed < 0:
            logger.warning(f"Ignoring negative elapsed job timing for state {state}: {elapsed}")
            return
        entry = self._job_time_stats.setdefault(state, {"sum": 0.0, "count": 0, "max": 0.0})
        entry["sum"] += elapsed
        entry["count"] += 1
        if elapsed > entry["max"]:
            entry["max"] = elapsed

    def _record_model_timing(
        self,
        accumulator: dict[str, dict[str, float | int]],
        model_name: str,
        elapsed: float,
    ) -> None:
        """Accumulate per-model timing data.

        Args:
            accumulator: The dict to record into (e.g. ``_time_per_step_per_model``).
            model_name: The model name key.
            elapsed: Time value in seconds to accumulate.
        """
        if elapsed < 0:
            logger.warning(f"Ignoring negative elapsed model timing for {model_name}: {elapsed}")
            return
        entry = accumulator.setdefault(model_name, {"sum": 0.0, "count": 0, "max": 0.0})
        entry["sum"] += elapsed
        entry["count"] += 1
        if elapsed > entry["max"]:
            entry["max"] = elapsed

    def _record_pending_job_timing(self, owner_timings: dict[str, float], state: str, elapsed: float) -> None:
        """Buffer timing data until the associated job is successfully submitted."""
        if elapsed < 0:
            logger.warning(f"Ignoring negative elapsed job timing for state {state}: {elapsed}")
            return

        owner_timings[state] = owner_timings.get(state, 0.0) + elapsed

    def _on_process_state_change(
        self,
        process_id: int,
        new_state: HordeProcessState,
        *,
        timing_sdk_api_job_info: ImageGenerateJobPopResponse | None = None,
    ) -> None:
        """Update process state and buffer any tracked elapsed time for the state being left."""
        process_info = self._process_map[process_id]
        prior_process_state = process_info.last_process_state
        prior_state_entered_timestamp = process_info.state_entered_timestamp

        self._process_map.on_process_state_change(
            process_id=process_id,
            new_state=new_state,
        )

        if prior_process_state in self._STATE_TRANSITION_TIMING_STATES:
            if timing_sdk_api_job_info is None:
                self._record_pending_job_timing(
                    self._pending_process_job_timings.setdefault(process_id, {}),
                    prior_process_state.name,
                    time.time() - prior_state_entered_timestamp,
                )
            else:
                self._record_pending_job_timing(
                    self._pending_completed_job_timings.setdefault(timing_sdk_api_job_info, {}),
                    prior_process_state.name,
                    time.time() - prior_state_entered_timestamp,
                )

        if new_state == HordeProcessState.WAITING_FOR_JOB and timing_sdk_api_job_info is None:
            # Returning to the idle WAITING_FOR_JOB state without a completed-job timing owner
            # means the process-specific buffered timings are no longer associated with a job
            # that can contribute to completed-job statistics, so discard them here.
            self._pending_process_job_timings.pop(process_id, None)

    def _move_pending_process_timings_to_completed_job(
        self,
        process_id: int,
        sdk_api_job_info: ImageGenerateJobPopResponse,
    ) -> None:
        """Re-associate buffered process timings with a completed job awaiting submission."""
        pending_process_timings = self._pending_process_job_timings.pop(process_id, None)
        if not pending_process_timings:
            return

        completed_job_timings = self._pending_completed_job_timings.setdefault(sdk_api_job_info, {})
        for state, elapsed in pending_process_timings.items():
            completed_job_timings[state] = completed_job_timings.get(state, 0.0) + elapsed

    def _commit_completed_job_timings(self, sdk_api_job_info: ImageGenerateJobPopResponse) -> None:
        """Commit buffered per-state timings for a successfully submitted job."""
        pending_job_timings = self._pending_completed_job_timings.pop(sdk_api_job_info, None)
        if not pending_job_timings:
            return

        for state, elapsed in pending_job_timings.items():
            self._record_job_timing(state, elapsed)

    def _discard_completed_job_timings(self, sdk_api_job_info: ImageGenerateJobPopResponse) -> None:
        """Discard buffered per-state timings for a job that did not complete successfully."""
        self._pending_completed_job_timings.pop(sdk_api_job_info, None)

    def _discard_broken_job(self, completed_job_info: HordeJobInfo) -> None:
        """Remove a job that cannot be submitted from all tracking structures to prevent queue blocking.

        This mirrors the cleanup done at the end of the normal submission path.
        """
        job_info = completed_job_info.sdk_api_job_info
        self._discard_completed_job_timings(job_info)

        if completed_job_info in self.jobs_pending_submit:
            self.jobs_pending_submit.remove(completed_job_info)

        if job_info in self.jobs_lookup:
            del self.jobs_lookup[job_info]

        if job_info in self.job_pop_timestamps:
            del self.job_pop_timestamps[job_info]

        if job_info in self.jobs_in_progress:
            self.jobs_in_progress.remove(job_info)

        if job_info.id_ is not None and job_info.id_ in self.job_faults:
            del self.job_faults[job_info.id_]

    @logger.catch(reraise=True)
    async def api_submit_job(self) -> None:
        """Submit a job result to the API, if any are completed (safety checked too) and ready to be submitted."""
        if len(self.jobs_pending_submit) == 0:
            return

        completed_job_info = self.jobs_pending_submit[0]
        job_info = completed_job_info.sdk_api_job_info

        if completed_job_info.state is None:
            logger.error(f"Job {job_info.ids} has no state, assuming faulted")
            completed_job_info.state = GENERATION_STATE.faulted

        if completed_job_info.state == GENERATION_STATE.faulted:
            logger.error(
                f"Job {job_info.ids} faulted, "
                "removing from completed jobs after submitting the faults to the horde",
            )
            self._consecutive_failed_jobs += 1

        if completed_job_info.job_image_results is not None:
            if len(completed_job_info.job_image_results) != completed_job_info.sdk_api_job_info.payload.n_iter:
                logger.warning(
                    f"Needed to generate {completed_job_info.sdk_api_job_info.payload.n_iter} images "
                    f"but only {len(completed_job_info.job_image_results)} returned by the inference process "
                    "We will continue, but you might get put into maintenance if this keeps happening.",
                )

            if completed_job_info.censored is None:
                logger.error(f"Job {job_info.ids} has censored=None, skipping submission to prevent queue block")
                self._discard_broken_job(completed_job_info)
                return
        if job_info.id_ is None:
            logger.error("job_info.id_ is None, skipping submission to prevent queue block")
            self._discard_broken_job(completed_job_info)
            return

        if job_info.payload.seed is None:
            logger.error(f"Job {job_info.ids} has seed=None, skipping submission to prevent queue block")
            self._discard_broken_job(completed_job_info)
            return

        if job_info.r2_upload is None:
            logger.error(f"Job {job_info.ids} has r2_upload=None, skipping submission to prevent queue block")
            self._discard_broken_job(completed_job_info)
            return

        highest_reward = 0
        highest_kudos_per_second = 0.0
        submit_tasks: list[Task[PendingSubmitJob]] = []
        finished_submit_jobs: list[PendingSubmitJob] = []
        iterations = 1
        job_faulted = False
        if completed_job_info.job_image_results is not None:
            iterations = len(completed_job_info.job_image_results)
        for gen_iter in range(iterations):
            new_submit = PendingSubmitJob(
                completed_job_info=completed_job_info,
                gen_iter=gen_iter,
                max_submit_retries=self.MAX_SUBMIT_RETRIES,
            )
            submit_tasks.append(asyncio.create_task(self.submit_single_generation(new_submit)))
        while len(submit_tasks) > 0:
            retry_submits: list[PendingSubmitJob] = []
            results = await asyncio.gather(*submit_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.exception(f"Exception in job submit task: {result}")
                    job_faulted = True
                elif isinstance(result, PendingSubmitJob):
                    if not result.is_finished:
                        retry_submits.append(result)
                    else:
                        finished_submit_jobs.append(result)
                    if highest_reward < result.kudos_reward:
                        highest_reward = result.kudos_reward
                    if highest_kudos_per_second < result.kudos_per_second:
                        highest_kudos_per_second = result.kudos_per_second
            submit_tasks = []
            for retry_submit in retry_submits:
                submit_tasks.append(asyncio.create_task(self.submit_single_generation(retry_submit)))

        # Get the time the job was popped from the job deque
        async with self._job_pop_timestamps_lock:
            time_popped = self.job_pop_timestamps.get(completed_job_info.sdk_api_job_info)
            if time_popped is None:
                logger.warning(
                    f"Failed to get time_popped for job {completed_job_info.sdk_api_job_info.id_}. "
                    "This is likely a bug.",
                )
                time_popped = time.time()
        time_taken = round(time.time() - time_popped, 2)

        # Log submission results for all successfully submitted jobs
        successful_submits = [job for job in finished_submit_jobs if not job.is_faulted]
        faulted_submits = [job for job in finished_submit_jobs if job.is_faulted]

        # Extract shared values
        time_to_generate = completed_job_info.time_to_generate or 0.0

        if successful_submits:
            total_kudos = sum(job.kudos_reward for job in successful_submits)
            batch_size = len(successful_submits)
            model_name = completed_job_info.sdk_api_job_info.model
            kudos_per_second_batch = highest_kudos_per_second * batch_size

            if batch_size == 1:
                # Single job - show the one job ID
                job_id_short = str(successful_submits[0].job_id)[:8]
                logger.opt(colors=True).success(
                    f"<b>Submitted generation {job_id_short} (model: "
                    f"<u>{model_name}</u>) "
                    f"for {total_kudos:,.2f} kudos. "
                    f"Job popped {time_taken} seconds ago "
                    f"and took {time_to_generate:.2f} to generate. "
                    f"({kudos_per_second_batch:.2f} kudos/second for the whole batch. 0.4 or greater is ideal)</b>",
                )
            else:
                # Batch job - show all job IDs and combined stats
                job_ids_short = ", ".join(str(job.job_id)[:8] for job in successful_submits)
                logger.opt(colors=True).success(
                    f"<b>Submitted {batch_size} generations [{job_ids_short}] (model: "
                    f"<u>{model_name}</u>) "
                    f"for {total_kudos:,.2f} kudos. "
                    f"Job popped {time_taken} seconds ago "
                    f"and took {time_to_generate:.2f} to generate. "
                    f"({kudos_per_second_batch:.2f} kudos/second for the whole batch. 0.4 or greater is ideal)</b>",
                )

            # If slower than 0.4 kudos per second, log a warning
            if kudos_per_second_batch < 0.4:
                job_ref = (
                    f"Batch job {completed_job_info.sdk_api_job_info.id_}"
                    if batch_size > 1
                    else f"Job {completed_job_info.sdk_api_job_info.id_}"
                )
                logger.warning(
                    f"{job_ref} took longer than is ideal; if this persists "
                    "consider lowering your max_power, using less threads, "
                    "disabling post processing and/or controlnets.",
                )
                logger.warning("Be sure your models are on an SSD. Freeing up RAM or VRAM may also help.")

        # Log faulted jobs
        for faulted_job in faulted_submits:
            logger.error(
                f"{faulted_job.job_id} faulted. Reported fault to the horde. "
                f"Job popped {time_taken} seconds ago and took "
                f"{time_to_generate:.2f} to generate.",
            )

        # Accumulate per-state timing for successfully submitted jobs.
        if successful_submits:
            self._commit_completed_job_timings(completed_job_info.sdk_api_job_info)
            self._record_job_timing(HordeProcessState.INFERENCE_PROCESSING.name, time_to_generate)
            self._record_job_timing("TOTAL", time_taken)
            if completed_job_info.time_to_download_aux_models:
                self._record_job_timing(
                    HordeProcessState.DOWNLOADING_AUX_MODEL.name,
                    completed_job_info.time_to_download_aux_models,
                )
            # Accumulate per-model timing stats.
            model_name = completed_job_info.sdk_api_job_info.model
            if model_name:
                self._record_model_timing(self._time_per_job_per_model, model_name, time_taken)
                ddim_steps = completed_job_info.sdk_api_job_info.payload.ddim_steps
                if ddim_steps and ddim_steps > 0:
                    self._record_model_timing(
                        self._time_per_step_per_model,
                        model_name,
                        time_to_generate / ddim_steps,
                    )
        else:
            self._discard_completed_job_timings(completed_job_info.sdk_api_job_info)

        # If the job took a long time to generate, log a warning (unless speed warnings are suppressed)
        if not self.bridge_data.suppress_speed_warnings:
            if highest_reward > 0 and (highest_reward / time_taken) < 0.1:
                logger.warning(
                    f"This job ({completed_job_info.sdk_api_job_info.id_}) "
                    "may have been in the queue for a long time. ",
                )

            if highest_reward > 0 and highest_kudos_per_second < 0.4:
                logger.warning(
                    f"This job ({completed_job_info.sdk_api_job_info.id_}) "
                    "took longer than is ideal; if this persists consider "
                    "lowering your max_power, using less threads, "
                    "disabling post processing and/or controlnets.",
                )

        # Finally, remove the job from the completed jobs list and reset the number of consecutive failed job
        async with self._jobs_lookup_lock, self._completed_jobs_lock:
            for submit_job in finished_submit_jobs:
                if submit_job.is_faulted:
                    job_faulted = True
                    self._consecutive_failed_jobs += 1
                    break
            if not job_faulted:
                # If any of the submits failed, we consider the whole job failed
                self._consecutive_failed_jobs = 0
            try:
                if completed_job_info.sdk_api_job_info in self.jobs_lookup:
                    self.jobs_lookup[completed_job_info.sdk_api_job_info].time_submitted = time.time()
                else:
                    self.jobs_lookup[completed_job_info.sdk_api_job_info] = HordeJobInfo(
                        sdk_api_job_info=completed_job_info.sdk_api_job_info,
                        time_popped=-1,
                        job_image_results=completed_job_info.job_image_results,
                        state=completed_job_info.state,
                        censored=completed_job_info.censored,
                        time_to_generate=completed_job_info.time_to_generate,
                        time_to_download_aux_models=completed_job_info.time_to_download_aux_models,
                    )
                    logger.error(
                        f"Job {completed_job_info.sdk_api_job_info.id_} not found in jobs_lookup "
                        "during submit. Creating a new HordeJobInfo object.",
                    )
                if self.bridge_data.capture_kudos_training_data:
                    if self.bridge_data.kudos_training_data_file is None:
                        self.bridge_data.kudos_training_data_file = "kudos_training_data.json"
                        logger.warning(
                            "Kudos training data capture is enabled but no file has been specified. "
                            f"Defaulting to {self.bridge_data.kudos_training_data_file}",
                        )
                    # if the file self.bridge_data.kudos_training_data_file exists
                    # we will append the entry from the jobs lookup to it as a new json entry
                    # if the file does not exist, we will create it and write the first entry

                    # If the current file is greater than 2mb, we will create a new file with a sequential number

                    file_name_to_use = f"kudos_model_training/{self.bridge_data.kudos_training_data_file}"
                    os.makedirs("kudos_model_training", exist_ok=True)
                    if os.path.exists(file_name_to_use) and os.path.getsize(file_name_to_use) > 2 * 1024 * 1024:
                        for i in range(1, 10000):
                            new_file_name = f"kudos_model_training/{self.bridge_data.kudos_training_data_file}.{i}"
                            if os.path.exists(new_file_name) and os.path.getsize(new_file_name) > 2 * 1024 * 1024:
                                continue

                            file_name_to_use = new_file_name
                            break

                    try:
                        with logger.catch(reraise=False):
                            if completed_job_info.sdk_api_job_info in self.jobs_lookup:
                                hji = self.jobs_lookup[completed_job_info.sdk_api_job_info]
                            else:
                                logger.error(
                                    f"Job {completed_job_info.sdk_api_job_info.id_} not found in jobs_lookup "
                                    " during kudos training data capture.",
                                )
                            if (
                                self.stable_diffusion_reference is not None
                                and hji.sdk_api_job_info.model is not None
                                and hji.sdk_api_job_info.model in self.stable_diffusion_reference.root
                            ):

                                model_dump = hji.model_dump(
                                    exclude=_excludes_for_job_dump,  # type: ignore
                                )
                                if (
                                    self.stable_diffusion_reference is not None
                                    and hji.sdk_api_job_info.model is not None
                                ):
                                    model_dump["sdk_api_job_info"]["model_baseline"] = (
                                        self.stable_diffusion_reference.root[hji.sdk_api_job_info.model].baseline
                                    )
                                # Preparation for multiple schedulers
                                if hji.sdk_api_job_info.payload.karras:
                                    model_dump["sdk_api_job_info"]["payload"]["scheduler"] = "karras"
                                else:
                                    model_dump["sdk_api_job_info"]["payload"]["scheduler"] = "simple"
                                del model_dump["sdk_api_job_info"]["payload"]["karras"]
                                model_dump["sdk_api_job_info"]["payload"]["lora_count"] = (
                                    len(
                                        model_dump["sdk_api_job_info"]["payload"]["loras"],
                                    )
                                    if model_dump["sdk_api_job_info"]["payload"]["loras"]
                                    else 0
                                )
                                model_dump["sdk_api_job_info"]["payload"]["ti_count"] = (
                                    len(
                                        model_dump["sdk_api_job_info"]["payload"]["tis"],
                                    )
                                    if model_dump["sdk_api_job_info"]["payload"]["tis"]
                                    else 0
                                )
                                model_dump["sdk_api_job_info"]["extra_source_images_count"] = (
                                    len(hji.sdk_api_job_info.extra_source_images)
                                    if hji.sdk_api_job_info.extra_source_images
                                    else 0
                                )
                                esi_combined_size = 0
                                if hji.sdk_api_job_info.extra_source_images:
                                    for esi in hji.sdk_api_job_info.extra_source_images:
                                        esi_combined_size += len(esi.image)
                                model_dump["sdk_api_job_info"]["extra_source_images_combined_size"] = esi_combined_size
                                model_dump["sdk_api_job_info"]["source_image_size"] = (
                                    len(hji.sdk_api_job_info._downloaded_source_image)
                                    if hji.sdk_api_job_info._downloaded_source_image
                                    else 0
                                )
                                model_dump["sdk_api_job_info"]["source_mask_size"] = (
                                    len(hji.sdk_api_job_info._downloaded_source_mask)
                                    if hji.sdk_api_job_info._downloaded_source_mask
                                    else 0
                                )
                                if not os.path.exists(file_name_to_use):
                                    with open(file_name_to_use, "w") as f:
                                        json.dump([model_dump], f, indent=4)
                                elif hji.sdk_api_job_info.payload.n_iter == 1:
                                    data = []
                                    with open(file_name_to_use) as f:
                                        data = json.load(f)
                                        if not isinstance(data, list):
                                            logger.warning(
                                                f"Kudos training data file {file_name_to_use} " "is not a list",
                                            )
                                            data = []
                                    data.append(model_dump)
                                    with open(file_name_to_use, "w") as f:
                                        json.dump(data, f, indent=4)
                    except Exception as e:
                        logger.error(
                            f"Failed to write kudos training data for job {completed_job_info.sdk_api_job_info.id_} "
                            f"{type(e)}: {e}",
                        )

                if completed_job_info in self.jobs_pending_submit:
                    self.jobs_pending_submit.remove(completed_job_info)
                else:
                    logger.warning(f"Job {completed_job_info.sdk_api_job_info.id_} not found in completed_jobs")

                if completed_job_info.sdk_api_job_info in self.jobs_lookup:
                    del self.jobs_lookup[completed_job_info.sdk_api_job_info]
                else:
                    logger.warning(f"Job {completed_job_info.sdk_api_job_info.id_} not found in jobs_lookup")

                self._last_job_submitted_time = time.time()

            except ValueError:
                # This means another fault catch removed the faulted job so it's OK
                # But we post a log anyway, just in case
                logger.debug(
                    f"Tried to remove completed_job_info "
                    f"{completed_job_info.sdk_api_job_info.id_} but it has already been removed.",
                )

            if completed_job_info.sdk_api_job_info in self.job_pop_timestamps:
                self.job_pop_timestamps.pop(completed_job_info.sdk_api_job_info)
                logger.debug(f"Removed {completed_job_info.sdk_api_job_info.id_} from job_pop_timestamps")

            if completed_job_info.sdk_api_job_info in self.jobs_in_progress:
                self.jobs_in_progress.remove(completed_job_info.sdk_api_job_info)
                logger.debug(f"Removed {completed_job_info.sdk_api_job_info.id_} from jobs_in_progress")

            if completed_job_info.sdk_api_job_info in self.jobs_lookup:
                self.jobs_lookup.pop(completed_job_info.sdk_api_job_info)
                logger.debug(f"Removed {completed_job_info.sdk_api_job_info.id_} from jobs_lookup")

    # _testing_max_jobs = 10000
    # _testing_jobs_added = 0
    # _testing_job_queue_length = 1

    _default_job_pop_frequency = 1.0
    """The default frequency at which to pop jobs from the API."""
    _error_job_pop_frequency = 5.0
    """The frequency at which to pop jobs from the API when an error occurs."""
    _job_pop_frequency = 1.0
    """The frequency at which to pop jobs from the API. Can be altered if an error occurs."""
    _last_job_pop_time = 0.0
    """The time at which the last job was popped from the API."""

    def _last_pop_recently(self) -> bool:
        return (time.time() - self._last_job_pop_time) < 10

    _last_job_submitted_time = time.time()
    """The time at which the last job was submitted to the API."""

    _max_pending_megapixelsteps = 25
    """The maximum number of megapixelsteps that can be pending in the job deque before job pops are paused."""
    _triggered_max_pending_megapixelsteps_time = 0.0
    """The time at which the number of megapixelsteps in the job deque exceeded the limit."""
    _triggered_max_pending_megapixelsteps = False
    """Whether the number of megapixelsteps in the job deque exceeded the limit."""
    _batch_wait_log_time = 0.0
    """The last time we informed that we're waiting for batched jobs to finish."""

    _consecutive_failed_jobs = 0

    def _record_faulted_job_history(
        self,
        faulted_job: ImageGenerateJobPopResponse,
        fault_phase: str | None = None,
    ) -> None:
        """Add a permanently faulted job to the webui display history.

        This is a helper used by handle_job_fault and any other fault path that bypasses it
        (e.g. safety-evaluation failures) to ensure all permanently faulted jobs appear in
        the Logs → Faulted Jobs container in the webui.

        Duplicate job IDs are silently ignored, and the history is capped at
        ``_max_faulted_jobs_history`` entries (oldest entry evicted when the cap is reached).

        Args:
            faulted_job: The job that faulted.
            fault_phase: A human-readable description of when the fault occurred.
        """
        job_id_str = str(faulted_job.id_)
        if any(entry["job_id"] == job_id_str for entry in self._faulted_jobs_history):
            logger.debug(f"Job {job_id_str} already in faulted_jobs_history, skipping duplicate entry")
            return

        faulted_job_details: dict[str, Any] = {
            "job_id": job_id_str,
            "model": faulted_job.model or "Unknown",
            "time_faulted": time.time(),
            "width": faulted_job.payload.width if faulted_job.payload else None,
            "height": faulted_job.payload.height if faulted_job.payload else None,
            "steps": faulted_job.payload.ddim_steps if faulted_job.payload else None,
            "sampler": faulted_job.payload.sampler_name if faulted_job.payload else None,
            "batch_size": faulted_job.payload.n_iter if faulted_job.payload else None,
            "fault_phase": fault_phase,
            "loras": [],
            "controlnet": None,
            "workflow": faulted_job.payload.workflow if faulted_job.payload else None,
        }

        if faulted_job.payload and faulted_job.payload.loras:
            for lora in faulted_job.payload.loras:
                faulted_job_details["loras"].append(
                    {
                        "name": lora.name if hasattr(lora, "name") else str(lora),
                        "model": lora.model if hasattr(lora, "model") else None,
                        "clip": lora.clip if hasattr(lora, "clip") else None,
                    },
                )

        if faulted_job.payload and faulted_job.payload.workflow in KNOWN_CONTROLNET_WORKFLOWS:
            faulted_job_details["controlnet"] = faulted_job.payload.workflow

        self._faulted_jobs_history.insert(0, faulted_job_details)
        if len(self._faulted_jobs_history) > self._max_faulted_jobs_history:
            self._faulted_jobs_history = self._faulted_jobs_history[: self._max_faulted_jobs_history]

        # Track cumulative fault count per phase for the stats graph.
        phase_key = fault_phase if fault_phase is not None else "Unknown Phase"
        self._faulted_jobs_per_phase[phase_key] = self._faulted_jobs_per_phase.get(phase_key, 0) + 1

    _ORPHANED_JOB_GRACE_SECONDS: float = 5.0
    """Seconds an in-progress job must remain orphaned before _reap_orphaned_in_progress_jobs faults it."""

    def _reap_orphaned_in_progress_jobs(self) -> bool:
        """Fault in-progress jobs that no longer have a live process handling them.

        A job stays in ``jobs_in_progress`` until its handling process reports a result. If that
        process dies, is replaced, or hangs in ``PROCESS_ENDING`` — for example after it got stuck
        in ``INFERENCE_STARTING`` / ``INFERENCE_PROCESSING`` before completing a single step,
        which is most common shortly after program start — the job becomes orphaned: no live
        process will ever complete it. An orphaned job permanently occupies an inference
        concurrency slot (starving already-preloaded processes) and, during shutdown, keeps
        :meth:`is_time_for_shutdown` ``False`` forever, so the worker hangs on
        "Finishing current jobs...".

        Faulting the orphan re-queues it for retry on a healthy process (or permanently faults it
        once retries are exhausted), letting the queue drain and shutdown proceed. A short grace
        period (``_ORPHANED_JOB_GRACE_SECONDS``) avoids false positives during the brief, normal
        window between dispatch and the first process state update.

        Returns:
            True if at least one orphaned job was faulted.
        """
        now = time.time()
        reaped = False
        for job in list(self.jobs_in_progress):
            job_id = str(job.id_) if job.id_ is not None else None
            handled = any(
                p.is_process_alive() and self._is_same_job(p.last_job_referenced, job)
                for p in self._process_map.values()
            )
            if handled:
                if job_id is not None:
                    self._job_orphan_since.pop(job_id, None)
                continue

            # Before applying the grace period, check whether a DEAD (but not yet replaced)
            # process still has this job referenced. If so, replace that process immediately
            # rather than waiting — this gives a more precise error message, triggers a proper
            # process restart, and prevents the generic "orphaned" path from firing at all.
            if not self._shutting_down:
                dead_owner = next(
                    (
                        p
                        for p in self._process_map.values()
                        if not p.is_process_alive()
                        and p.process_type == HordeProcessType.INFERENCE
                        and p.last_process_state
                        not in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED)
                        and self._is_same_job(p.last_job_referenced, job)
                    ),
                    None,
                )
                if dead_owner is not None:
                    if job_id is not None:
                        self._job_orphan_since.pop(job_id, None)
                    self._replace_inference_process(dead_owner)
                    reaped = True
                    continue

            # Orphaned: no live process is handling this in-progress job. Apply a grace period
            # before faulting so a job that was just dispatched (and whose process has not yet
            # reported INFERENCE_STARTING) is not faulted prematurely.
            first_seen = self._job_orphan_since.setdefault(job_id, now) if job_id is not None else now
            if (now - first_seen) < self._ORPHANED_JOB_GRACE_SECONDS:
                continue

            logger.error(
                f"Job {job.id_} is in progress but no live process is handling it "
                "(its process died, was replaced, or hung before completing a step). "
                "Faulting it so the inference slot is freed and shutdown can complete.",
            )
            if job_id is not None:
                self._job_orphan_since.pop(job_id, None)
            if self._shutting_down:
                # During shutdown there may be no healthy process left to run a re-queued job
                # (end_inference_processes() is ending them), so a retry would never complete and
                # would keep is_time_for_shutdown() False forever. Permanently fault instead so the
                # job is reported to the API and the queue drains.
                job_info = self.jobs_lookup.get(job)
                if job_info is not None:
                    job_info.retry_count = self.MAX_JOB_RETRIES
                self.handle_job_fault(
                    faulted_job=job,
                    fault_info="orphaned during shutdown: no live process was handling the in-progress job",
                    retry_skipped=True,
                )
            else:
                self.handle_job_fault(
                    faulted_job=job,
                    fault_info="orphaned: no live process was handling the in-progress job",
                )
            reaped = True

        # Prune orphan-tracking entries for jobs that are no longer in progress.
        if self._job_orphan_since:
            in_progress_ids = {str(j.id_) for j in self.jobs_in_progress if j.id_ is not None}
            for stale_id in [k for k in self._job_orphan_since if k not in in_progress_ids]:
                self._job_orphan_since.pop(stale_id, None)

        return reaped

    def handle_job_fault(
        self,
        faulted_job: ImageGenerateJobPopResponse,
        process_info: HordeProcessInfo | None = None,
        fault_info: str | None = None,
        retry_skipped: bool = False,
    ) -> None:
        """Mark a job as faulted and add it to the completed jobs list to report it faulted.

        If the job has not been retried yet, it will be retried once before being marked as faulted.

        Args:
            faulted_job (ImageGenerateJobPopResponse): The job that faulted.
            process_info (HordeProcessInfo | None, optional): The process that faulted the job. Defaults to None.
            fault_info (str | None, optional): A human-readable description of the fault reason. Defaults to None.
            retry_skipped (bool, optional): Whether the normal local retry path was intentionally bypassed.
        """
        # Resolve the canonical job object from jobs_lookup by id_ so that a filtered
        # copy (created via model_copy for prompt filtering) is handled the same as the
        # original — object equality / hash may differ between the two.
        _faulted_id = faulted_job.id_
        _canonical_key = next((k for k in self.jobs_lookup if k.id_ == _faulted_id), None)
        job_info = self.jobs_lookup.get(_canonical_key) if _canonical_key is not None else None

        if job_info is None:
            logger.error(f"Job {_faulted_id} not found in jobs_lookup")
            # Record in history even when the job_info cannot be found so the fault is
            # still visible in the webui (fault_phase is unknown in this edge case).
            job_id_str = str(_faulted_id)
            already_recorded = any(entry["job_id"] == job_id_str for entry in self._faulted_jobs_history)
            self._record_faulted_job_history(faulted_job)
            # Keep failed-model and inference-cooldown tracking aligned with history
            # deduplication: only count each permanently-faulted job ID once.
            if not already_recorded and faulted_job.model is not None:
                model_name = faulted_job.model
                self._failed_models[model_name] = self._failed_models.get(model_name, 0) + 1
                self._record_inference_failure(model_name, time.time())
            # The job may have been removed from jobs_pending_inference before this call
            # (e.g. _fault_cooldown_model_jobs strips it when metadata is missing).
            # Restart the idle timer if that emptied the queue.
            self._restart_idle_timer_if_queue_empty()
        else:
            # Check if the job should be retried
            if job_info.retry_count < self.MAX_JOB_RETRIES:
                # Retry the job once
                job_info.retry_count += 1
                fault_detail = f": {fault_info}" if fault_info else ""
                proc_id = process_info.process_id if process_info else "unknown"
                logger.warning(
                    f"Job {_faulted_id} faulted on process {proc_id}"
                    f"{fault_detail}, retrying (attempt {job_info.retry_count} of {self.MAX_JOB_RETRIES})",
                )

                # Remove from jobs_in_progress if present (use id_-based match)
                _ip_retry = next((j for j in self.jobs_in_progress if j.id_ == _faulted_id), None)
                if _ip_retry is not None:
                    logger.debug(f"Removing job {_faulted_id} from jobs_in_progress for retry")
                    self.jobs_in_progress.remove(_ip_retry)

                # Re-queue the canonical original for another attempt
                _already_pending = any(j.id_ == _faulted_id for j in self.jobs_pending_inference)
                if not _already_pending:
                    self.jobs_pending_inference.append(_canonical_key)
                    self._invalidate_megapixelsteps_cache()
                    logger.info(f"Job {_faulted_id} successfully re-queued for retry")
                else:
                    logger.debug(f"Job {_faulted_id} already in jobs_pending_inference, not re-queuing")

                return

            # Job has exhausted all retry attempts, proceed with faulting
            fault_detail = f": {fault_info}" if fault_info else ""
            if retry_skipped:
                logger.error(
                    f"Job {faulted_job.id_} faulted with retry skipped"
                    f"{fault_detail}, marking as permanently faulted",
                )
            else:
                retry_text = "retry attempt" if self.MAX_JOB_RETRIES == 1 else "retry attempts"
                logger.error(
                    f"Job {faulted_job.id_} faulted after {self.MAX_JOB_RETRIES} {retry_text}"
                    f"{fault_detail}, marking as permanently faulted",
                )

            _pend_match = next((j for j in self.jobs_pending_inference if j.id_ == _faulted_id), None)
            if _pend_match is not None:
                self.jobs_pending_inference.remove(_pend_match)
                self._invalidate_megapixelsteps_cache()
                self._restart_idle_timer_if_queue_empty()

            if (
                self._skipped_line_next_job_and_process is not None
                and faulted_job.model == self._skipped_line_next_job_and_process.next_job.model
            ):
                self._skipped_line_next_job_and_process = None

            job_info.fault_job()
            job_info.time_to_generate = self.bridge_data.process_timeout

            # Track the failing model
            if faulted_job.model is not None:
                model_name = faulted_job.model
                self._failed_models[model_name] = self._failed_models.get(model_name, 0) + 1
                self._record_inference_failure(model_name, time.time())

            # Add faulted job details to history for webui display
            # Determine the phase during which the job faulted
            fault_phase = None
            if process_info is not None:
                # Use the process state name directly so it matches the job state values
                # used in the Avg & Max Time per Job State table.
                fault_phase = process_info.last_process_state.name

            self._record_faulted_job_history(faulted_job, fault_phase)

            if process_info is not None:
                logger.error(f"Job {_faulted_id} faulted due to {self._process_label(process_info.process_id)} crashing")

            _ip_fault = next((j for j in self.jobs_in_progress if j.id_ == _faulted_id), None)
            if _ip_fault is not None:
                logger.debug(f"Removing job {_faulted_id} from jobs_in_progress")
                self.jobs_in_progress.remove(_ip_fault)

            for horde_job_info in list(self.jobs_pending_safety_check):
                if horde_job_info.sdk_api_job_info.id_ == faulted_job.id_:
                    logger.debug(f"Removing job {faulted_job.id_} from jobs_pending_safety_check")
                    self.jobs_pending_safety_check.remove(horde_job_info)
                    break

            for horde_job_info in list(self.jobs_being_safety_checked):
                if horde_job_info.sdk_api_job_info.id_ == faulted_job.id_:
                    logger.debug(f"Removing job {faulted_job.id_} from jobs_being_safety_checked")
                    self.jobs_being_safety_checked.remove(horde_job_info)
                    break

            if job_info not in self.jobs_pending_submit:
                self.jobs_pending_submit.append(job_info)
            else:
                logger.warning(f"Job {faulted_job.id_} already in completed_jobs")

    def get_single_job_effective_megapixelsteps(self, job: ImageGenerateJobPopResponse) -> int:
        """Return the number of megapixelsteps for a single job.

        Args:
            job (ImageGenerateJobPopResponse): The job to get the number of megapixelsteps for.

        Returns:
            int: The number of effective megapixelsteps for the job.
        """
        has_upscaler = any(pp in [u.value for u in KNOWN_UPSCALERS] for pp in job.payload.post_processing)
        upscaler_multiplier = 1 if has_upscaler else 0
        job_pixels = job.payload.width * job.payload.height

        # Each extra batched image increases our difficulty by 20%
        batching_multiplier = 1 + ((job.payload.n_iter - 1) * 0.2)

        lora_adjustment = 0
        if job.payload.loras is not None:
            lora_adjustment = 4 * 1_000_000 if len(job.payload.loras) > 0 else 0

        hires_fix_adjustment = 0

        if job.payload.hires_fix:
            hires_fix_adjustment = 512 * 512 * job.payload.ddim_steps

        # If upscaling was requested, due to it being serial, each extra image in the batch
        # Further increases our difficulty.
        # In this calculation we treat each upscaler as adding 20 steps per image
        upscaling_adjustment = job_pixels * 20 * upscaler_multiplier * job.payload.n_iter
        job_effective_pixel_steps = (
            (job_pixels * batching_multiplier * job.payload.ddim_steps)
            + upscaling_adjustment
            + lora_adjustment
            + hires_fix_adjustment
        )

        # Hard model difficulty is increased due to variations in the performance
        # of different architectures. This look up is a rough estimate based on a median case
        if job.model in KNOWN_SLOW_MODELS_DIFFICULTIES:
            job_effective_pixel_steps *= KNOWN_SLOW_MODELS_DIFFICULTIES[job.model]

        # We treat slow workflows add extra slowdowns (as they might perform many more steps of inference)
        if job.payload.workflow in KNOWN_SLOW_WORKFLOWS:
            job_effective_pixel_steps *= KNOWN_SLOW_WORKFLOWS[job.payload.workflow]

        # Some workflows by default require controlnets, but the user doesn't have to specify them.
        # In this case, we use this to know when we have SDXL workflows, as they can double the VRAM usage
        if job.payload.workflow in KNOWN_CONTROLNET_WORKFLOWS:
            job_effective_pixel_steps *= 2
        return int(job_effective_pixel_steps / 1_000_000)

    def get_pending_megapixelsteps(self) -> int:
        """Return the number of megapixelsteps that are pending in the job deque.

        Uses caching to avoid recalculating on every call.
        """
        # Return cached value if still valid
        if self._megapixelsteps_cache_valid:
            return self._cached_pending_megapixelsteps

        # Recalculate and cache
        job_deque_megapixelsteps = 0
        for job in self.jobs_pending_inference:
            job_megapixelsteps = self.get_single_job_effective_megapixelsteps(job)
            job_deque_megapixelsteps += job_megapixelsteps

        for _ in self.jobs_pending_submit:
            job_deque_megapixelsteps += 4

        self._cached_pending_megapixelsteps = job_deque_megapixelsteps
        self._megapixelsteps_cache_valid = True
        return job_deque_megapixelsteps

    def _restart_idle_timer_if_queue_empty(self) -> None:
        """Restart the idle-timer anchor when the job queue becomes empty.

        Called after removing a job from ``jobs_pending_inference`` to ensure the
        ``time_without_jobs`` counter resumes incrementing immediately when the worker
        becomes idle, rather than waiting for the next job-pop cycle to return a
        "no jobs available" response (a gap of up to ``_job_pop_frequency`` seconds).

        Only acts when the queue is actually empty *and* the anchor has already been
        cleared (i.e. a job was successfully popped since the last idle period started).
        Does not restart while job pops are paused so that paused time is not counted.
        """
        if self._job_pops_paused:
            return
        if len(self.jobs_pending_inference) == 0 and self._last_pop_no_jobs_available_time == 0.0:
            self._last_pop_no_jobs_available_time = time.time()

    def _invalidate_megapixelsteps_cache(self) -> None:
        """Invalidate the megapixelsteps cache when jobs are added or removed."""
        self._megapixelsteps_cache_valid = False

    def should_wait_for_pending_megapixelsteps(self) -> bool:
        """Check if the number of megapixelsteps in the job deque is above the limit."""
        return self.get_pending_megapixelsteps() > self._max_pending_megapixelsteps

    async def _get_source_images(self, job_pop_response: ImageGenerateJobPopResponse) -> ImageGenerateJobPopResponse:
        # Adding this to stop mypy complaining
        if job_pop_response.id_ is None:
            logger.error("Received ImageGenerateJobPopResponse with id_ is None. Please let the devs know!")
            return job_pop_response

        download_tasks: list[Task] = []

        source_image_is_url = False
        if job_pop_response.source_image is not None and job_pop_response.source_image.startswith("http"):
            source_image_is_url = True
            logger.debug(f"Source image for job {job_pop_response.id_} is a URL")

        source_mask_is_url = False
        if job_pop_response.source_mask is not None and job_pop_response.source_mask.startswith("http"):
            source_mask_is_url = True
            logger.debug(f"Source mask for job {job_pop_response.id_} is a URL")

        any_extra_source_images_are_urls = False
        if job_pop_response.extra_source_images is not None:
            for extra_source_image in job_pop_response.extra_source_images:
                if extra_source_image.image.startswith("http"):
                    any_extra_source_images_are_urls = True
                    logger.debug(f"Extra source image for job {job_pop_response.id_} is a URL")

        attempts = 0
        while attempts < MAX_SOURCE_IMAGE_RETRIES:
            if (
                source_image_is_url
                and job_pop_response.source_image is not None
                and job_pop_response.get_downloaded_source_image() is None
            ):
                download_tasks.append(job_pop_response.async_download_source_image(self._aiohttp_client_session))
            if (
                source_mask_is_url
                and job_pop_response.source_mask is not None
                and job_pop_response.get_downloaded_source_mask() is None
            ):
                download_tasks.append(job_pop_response.async_download_source_mask(self._aiohttp_client_session))

            download_extra_source_images = job_pop_response.get_downloaded_extra_source_images()
            if (
                any_extra_source_images_are_urls
                and job_pop_response.extra_source_images is not None
                or (
                    download_extra_source_images is not None
                    and job_pop_response.extra_source_images is not None
                    and len(download_extra_source_images) != len(job_pop_response.extra_source_images)
                )
            ):

                download_tasks.append(
                    asyncio.create_task(
                        job_pop_response.async_download_extra_source_images(
                            self._aiohttp_client_session,
                            max_retries=MAX_SOURCE_IMAGE_RETRIES,
                        ),
                    ),
                )

            gather_results = await asyncio.gather(*download_tasks, return_exceptions=True)

            for result in gather_results:
                if isinstance(result, Exception):
                    logger.error(f"Failed to download source image: {result}")
                    attempts += 1
                    break
            else:
                break

        if attempts >= MAX_SOURCE_IMAGE_RETRIES:
            if source_image_is_url and job_pop_response.get_downloaded_source_image() is None:
                if self.job_faults.get(job_pop_response.id_) is None:
                    self.job_faults[job_pop_response.id_] = []

                logger.error(f"Failed to download source image for job {job_pop_response.id_}")
                self.job_faults[job_pop_response.id_].append(
                    GenMetadataEntry(
                        type=METADATA_TYPE.source_image,
                        value=METADATA_VALUE.download_failed,
                        ref="source_image",
                    ),
                )

            if source_mask_is_url and job_pop_response.get_downloaded_source_mask() is None:
                if self.job_faults.get(job_pop_response.id_) is None:
                    self.job_faults[job_pop_response.id_] = []
                logger.error(f"Failed to download source mask for job {job_pop_response.id_}")

                self.job_faults[job_pop_response.id_].append(
                    GenMetadataEntry(
                        type=METADATA_TYPE.source_mask,
                        value=METADATA_VALUE.download_failed,
                        ref="source_mask",
                    ),
                )
            downloaded_extra_source_images = job_pop_response.get_downloaded_extra_source_images()
            if (
                any_extra_source_images_are_urls
                and downloaded_extra_source_images is None
                or (
                    downloaded_extra_source_images is not None
                    and job_pop_response.extra_source_images is not None
                    and len(downloaded_extra_source_images) != len(job_pop_response.extra_source_images)
                )
            ):
                if self.job_faults.get(job_pop_response.id_) is None:
                    self.job_faults[job_pop_response.id_] = []
                logger.error(f"Failed to download extra source images for job {job_pop_response.id_}")

                ref = []
                if job_pop_response.extra_source_images is not None and downloaded_extra_source_images is not None:
                    for predownload_extra_source_image in job_pop_response.extra_source_images:
                        if predownload_extra_source_image.image.startswith("http"):
                            if any(
                                predownload_extra_source_image.original_url == extra_source_image.image
                                for extra_source_image in downloaded_extra_source_images
                            ):
                                continue

                            ref.append(str(job_pop_response.extra_source_images.index(predownload_extra_source_image)))
                elif job_pop_response.extra_source_images is not None and downloaded_extra_source_images is None:
                    ref = [str(i) for i in range(len(job_pop_response.extra_source_images))]

                for r in ref:
                    self.job_faults[job_pop_response.id_].append(
                        GenMetadataEntry(
                            type=METADATA_TYPE.extra_source_images,
                            value=METADATA_VALUE.download_failed,
                            ref=r,
                        ),
                    )

        return job_pop_response

    _last_pop_maintenance_mode: bool = False
    """Maintenance-mode state latch.

    Set to ``True`` when a job pop returns a maintenance-mode error response and
    ``remove_maintenance()`` is triggered. While ``True`` it suppresses status
    messages, user-info fetches, and repeated ``remove_maintenance()`` calls.
    Cleared back to ``False`` once all inference processes have been reloaded
    (in ``_process_control_loop``) or when a successful job pop is received.
    Also surfaced to the web UI as ``maintenance_mode``.
    """
    _last_maintenance_removal_attempt: float = 0.0
    """Epoch time of the last automatic ``remove_maintenance()`` attempt.

    Used to throttle the continuous auto-removal (when ``remove_maintenance_on_init`` is enabled)
    so the API is not hammered on every (frequent) maintenance-mode job-pop retry.
    """
    _MAINTENANCE_REMOVAL_RETRY_INTERVAL: float = 60.0
    """Minimum seconds between automatic maintenance-removal attempts while in maintenance mode."""
    _last_pop_no_jobs_available: bool = False
    """Whether the last job pop attempt had a no jobs available response."""
    _last_pop_no_jobs_available_time: float = 0.0
    """Anchor timestamp for the current idle period.

    Set to ``session_start_time`` at initialisation so that idle time is tracked
    from program launch even before any job-pop response has been received.  Updated
    to the current time on each "no jobs available" pop cycle to accumulate elapsed
    idle seconds into ``_time_spent_no_jobs_available``.  When a job is successfully
    popped, any elapsed idle time since the last anchor is first flushed into
    ``_time_spent_no_jobs_available`` before this is reset to ``0.0``, which also
    causes ``update_webui_status`` to stop adding a live in-flight delta to the
    displayed counter.
    """
    _time_spent_no_jobs_available: float = 0.0
    """The number of seconds spent with no jobs popped or available."""
    _max_time_spent_no_jobs_available: float = 60.0 * 60.0
    """The maximum number of seconds to spend with no jobs popped or available before warning the user."""
    _too_many_consecutive_failed_jobs: bool = False
    """Whether too many consecutive failed jobs have occurred and job pops are paused."""
    _too_many_consecutive_failed_jobs_time: float = 0.0
    """The time at which too many consecutive failed jobs occurred."""
    _too_many_consecutive_failed_jobs_wait_time = 180
    """The time to wait after too many consecutive failed jobs before resuming job pops."""

    _consecutive_pop_failures: int = 0
    """The number of consecutive job pop failures (network/API errors)."""
    _consecutive_pop_failure_warn_threshold: int = 3
    """Number of consecutive pop failures before logging a prominent warning."""

    _JOB_POP_TIMEOUT_SECONDS: float = 90.0
    """Hard client-side cap on a single job-pop request. A pop normally completes in
    under a couple of seconds; without this cap a wedged gateway that accepts the
    connection but never responds can stall the pop loop for many minutes with no
    log output (observed: ~8 minutes of total silence), instead of failing fast and
    retrying at ``_error_job_pop_frequency``."""

    _idle_process_warning_logged: bool = False
    """Whether the idle process warning has already been logged for the current idle period."""

    def print_maint_mode_messages(self) -> None:
        """Print the information about maintenance mode to the user."""

        def warning_function_no_format(x):  # noqa: ANN001, ANN202
            return logger.opt(colors=True, raw=True).warning(
                "<fg #f1c40f>" + x + "</>\n",
            )

        warning_function_no_format(
            "Your worker is in maintenance mode. Set your API key at https://tinybots.net/artbot/settings, "
            "click save, then click unpause on https://tinybots.net/artbot/settings?panel=workers while the worker "
            "is running to clear this message.",
        )
        warning_function_no_format(
            "If you didn't expect seeing this message, its probable that the worker "
            "dropped too many jobs, and the server stepped in to prevent further jobs from being "
            "dropped. Please check the logs above, and possibly your logs/ folder as well.",
        )
        warning_function_no_format("Common reasons for forced maintenance mode are: ")
        warning_function_no_format("  - `max_threads` is too high.")
        warning_function_no_format("  - `queue_size` is too high.")
        warning_function_no_format("  - `max_batch` is too high.")
        warning_function_no_format("  - `max_power` is too high.")
        warning_function_no_format("  - The worker can't handle, SDXL, Cascade, or Flux models.")
        warning_function_no_format(
            "  - If you have the equivalent GPU of a 1070 or less, set"
            " limit_max_steps or extra_slow_worker. "
            "This should only be done as a last resort.",
        )

        warning_function_no_format(
            "If you continue to see this message, come to the official discord (https://discord.gg/3DxrhksKzn).",
        )

    _MAX_API_MESSAGES_KEPT = 50
    """The maximum number of API worker messages kept in memory (oldest evicted first)."""

    def _prune_api_messages(self) -> None:
        """Drop expired API messages and cap how many are kept in memory.

        Without this the messages dict grows for the lifetime of the process, and expired
        messages keep being re-printed with every status output.
        """
        if not self._api_messages_received:
            return

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        for message_id, message in list(self._api_messages_received.items()):
            if message.message_expiry is None:
                continue
            try:
                raw_expiry = str(message.message_expiry).strip().replace("Z", "+00:00")
                expiry = datetime.datetime.fromisoformat(raw_expiry)
                if expiry.tzinfo is None:
                    expiry = expiry.replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                # Unparseable expiry — keep the message; the size cap below still applies.
                continue
            if expiry <= now_utc:
                del self._api_messages_received[message_id]

        # Dicts preserve insertion order, so evicting from the front drops the oldest.
        while len(self._api_messages_received) > self._MAX_API_MESSAGES_KEPT:
            del self._api_messages_received[next(iter(self._api_messages_received))]

    @logger.catch(reraise=True)
    async def api_job_pop(self) -> None:
        """If the job deque is not full, add any jobs that are available to the job deque."""
        if self._shutting_down:
            self._last_pop_no_jobs_available = False
            return

        # Skip if job pops have been paused by the user via the web UI
        if self._job_pops_paused:
            # Auto-resume if a timed pause has expired.
            if self._job_pops_pause_until is not None and time.time() >= self._job_pops_pause_until:
                logger.info("Timed job-pop pause expired; resuming automatically")
                self.set_job_pops_paused(False)
                if self.webui is not None:
                    self.webui.update_status(job_pops_paused=False, job_pops_pause_until=None)
            else:
                return

        # Skip if the client session is not initialized yet (during startup)
        if self.horde_client_session is None:
            return

        cur_time = time.time()

        if self._too_many_consecutive_failed_jobs:
            if (
                cur_time - self._too_many_consecutive_failed_jobs_time
                > self._too_many_consecutive_failed_jobs_wait_time
            ):
                self._consecutive_failed_jobs = 0
                self._too_many_consecutive_failed_jobs = False
                logger.debug("Resuming job pops after too many consecutive failed jobs")
            return

        if self._consecutive_failed_jobs >= 3:
            if self.bridge_data.exit_on_unhandled_faults:
                logger.error("Exiting due to exit_on_unhandled_faults being enabled")
                self._shutdown()
            # Add this to prevent a loophole
            self._consecutive_failed_jobs = 0
            return

        # Only count jobs not yet started as "queued"; active jobs (jobs_in_progress)
        # are controlled by max_threads and should not consume queue_size slots.
        jobs_queued = max(0, len(self.jobs_pending_inference) - len(self.jobs_in_progress))

        max_queue_size = self.max_queue_size
        if not isinstance(max_queue_size, int):
            max_queue_size = self.bridge_data.queue_size
        # max_queue_size=0 means "no pre-buffering" — the worker-availability check below
        # (get_first_available_inference_process) is the natural limiter in that case.
        # Applying >= 0 here would always be True and block every pop.
        if max_queue_size > 0 and jobs_queued >= max_queue_size:
            return

        # Don't start jobs if we can't evaluate safety (NSFW/CSAM)
        if self._process_map.get_first_available_safety_process() is None:
            return

        # When pre-buffering is disabled (max_queue_size == 0) only pop when an inference
        # process is immediately free — that's the natural rate-limiter.
        # When pre-buffering is active (max_queue_size > 0) skip this check: the job is
        # placed in jobs_pending_inference and start_inference() will dispatch it as soon
        # as a process finishes its current job.  Blocking here would prevent the queue
        # from ever filling up to the configured max_queue_size.
        if max_queue_size == 0 and self._process_map.get_first_available_inference_process() is None:
            return

        if len(self.bridge_data.image_models_to_load) == 0:
            logger.error("No models are configured to be loaded, please check your config (models_to_load).")
            await asyncio.sleep(3)
            return

        # If there are long running jobs, don't start any more even if there is space in the deque
        if self.should_wait_for_pending_megapixelsteps():
            if self.get_pending_megapixelsteps() < 40:
                seconds_to_wait = self.get_pending_megapixelsteps() * 0.5
            elif self.get_pending_megapixelsteps() < 80:
                seconds_to_wait = self.get_pending_megapixelsteps() * 0.7
            else:
                seconds_to_wait = self.get_pending_megapixelsteps() * 0.8

            if self.bridge_data.max_threads > 1:
                seconds_to_wait *= 0.75

            if self.bridge_data.high_performance_mode:
                seconds_to_wait *= 0.2
                if seconds_to_wait < 35:
                    seconds_to_wait = 1
            elif self.bridge_data.moderate_performance_mode:
                seconds_to_wait *= 0.4
                if seconds_to_wait < 20:
                    seconds_to_wait = 1

            if self._triggered_max_pending_megapixelsteps is False:
                self._triggered_max_pending_megapixelsteps = True
                self._triggered_max_pending_megapixelsteps_time = time.time()
                if seconds_to_wait > 2:
                    logger.opt(colors=True).info(
                        f"<fg #7dcea0><i>Pausing job pops for {round(seconds_to_wait, 2)} seconds "
                        "so some long running jobs can make some progress.</i></>",
                    )
                logger.debug(
                    f"Paused job pops for pending megapixelsteps to decrease below {self._max_pending_megapixelsteps}",
                )
                logger.debug(
                    f"Pending megapixelsteps: {self.get_pending_megapixelsteps()} | "
                    f"Max pending megapixelsteps: {self._max_pending_megapixelsteps} | "
                    f"Scheduled to wait for {seconds_to_wait} seconds",
                )
                logger.debug(
                    f"high_performance_mode: {self.bridge_data.high_performance_mode} | "
                    f"moderate_performance_mode: {self.bridge_data.moderate_performance_mode}",
                )
                return

            if not (time.time() - self._triggered_max_pending_megapixelsteps_time) > seconds_to_wait:
                return

            self._triggered_max_pending_megapixelsteps = False
            logger.debug(
                f"Pending megapixelsteps decreased below {self._max_pending_megapixelsteps}, continuing with job pops",
            )
        else:
            # Resume job pop pausing if there are no pending jobs
            if self._triggered_max_pending_megapixelsteps:
                self._triggered_max_pending_megapixelsteps = False
                logger.debug("No pending jobs remaining, resuming job pops")

        # We don't want to pop jobs too frequently, so we wait a bit between each pop
        if time.time() - self._last_job_pop_time < self._job_pop_frequency:
            return

        self._last_job_pop_time = time.time()

        models = set(self.bridge_data.image_models_to_load)

        loaded_models = {
            process.loaded_horde_model_name
            for process in self._process_map.values()
            if process.loaded_horde_model_name is not None
        }

        if (
            len(self.bridge_data.image_models_to_load) > self.max_inference_processes
            and len(loaded_models) == self.max_inference_processes
        ):
            if (
                (not self._last_pop_no_jobs_available)
                and self.bridge_data.horde_model_stickiness > 0
                and random.random() < self.bridge_data.horde_model_stickiness
            ):
                free_models = {
                    process.loaded_horde_model_name
                    for process in self._process_map.values()
                    if not process.is_process_busy() and process.loaded_horde_model_name is not None
                }
                if len(loaded_models) >= 1:
                    models = free_models
                logger.debug(f"Sticky models -- popping only {models}")
                if len(self.bridge_data.image_models_to_load) > 10:
                    logger.warning(
                        "Model stickiness is intended mostly for slow disks and works best with few models. "
                        f"You have {len(self.bridge_data.image_models_to_load)} models configured.",
                    )
            elif self.bridge_data.horde_model_stickiness > 0:
                logger.debug("Models unstuck: asking to pop for all available models.")

        if self.bridge_data.custom_models is not None and len(self.bridge_data.custom_models) > 0:
            logger.debug("Custom models are enabled, adding them to the list of models to pop")
            custom_model_names = {
                name
                for model in self.bridge_data.custom_models
                if (name := model.get("name")) is not None
            }
            models.update(custom_model_names)

        # Exclude models that are in the inference-failure cooldown.  These models have
        # produced enough permanently-faulted jobs recently that continuing to request
        # them would risk the Horde server penalizing this worker (or placing it in
        # maintenance mode).  Once the cooldown expires the models are automatically
        # included again.
        cooldown_models = {m for m in models if self._is_model_in_inference_cooldown(m)}
        if cooldown_models:
            models -= cooldown_models
            # Rate-limit the warning: only emit when the cooldown set changes or every
            # min(_INFERENCE_FAILURE_COOLDOWN, 300) seconds to avoid log spam.
            cooldown_models_frozen = frozenset(cooldown_models)
            cooldown_warning_interval = min(self._INFERENCE_FAILURE_COOLDOWN, 300.0)
            now_for_warn = time.time()
            if (
                cooldown_models_frozen != self._last_warned_inference_cooldown_models
                or now_for_warn - self._last_warned_inference_cooldown_at >= cooldown_warning_interval
            ):
                logger.warning(
                    f"Excluding {len(cooldown_models)} model(s) from job-pop request due to "
                    f"inference-failure cooldown: {sorted(cooldown_models)}.  "
                    f"They will be re-included after {self._INFERENCE_FAILURE_COOLDOWN:.0f}s.",
                )
                self._last_warned_inference_cooldown_models = cooldown_models_frozen
                self._last_warned_inference_cooldown_at = now_for_warn
            else:
                logger.debug(
                    "Skipping repeated inference-failure cooldown warning for models: "
                    f"{sorted(cooldown_models)}",
                )

        if len(models) == 0:
            if cooldown_models:
                logger.warning(
                    "Skipping job-pop request because all candidate models are currently "
                    "in inference-failure cooldown: "
                    f"{sorted(cooldown_models)}. Configured cooldown duration: "
                    f"{self._INFERENCE_FAILURE_COOLDOWN:.0f}s.",
                )
            else:
                logger.debug("Not eligible to pop a job yet")
            return

        try:
            job_pop_request = ImageGenerateJobPopRequest(
                apikey=self.bridge_data.api_key,
                name=self.bridge_data.dreamer_worker_name,
                bridge_agent=f"AI Horde Worker reGen:{horde_worker_regen.__version__}:https://github.com/Haidra-Org/horde-worker-reGen",
                models=list(models),
                blacklist=self.bridge_data.blacklist,
                nsfw=self.bridge_data.nsfw,
                threads=self.max_concurrent_inference_processes,
                max_pixels=self.bridge_data.max_power * 8 * 64 * 64,
                require_upfront_kudos=self.bridge_data.require_upfront_kudos,
                allow_img2img=self.bridge_data.allow_img2img,
                allow_painting=self.bridge_data.allow_inpainting,
                allow_unsafe_ipaddr=self.bridge_data.allow_unsafe_ip,
                allow_post_processing=self.bridge_data.allow_post_processing,
                allow_controlnet=self.bridge_data.allow_controlnet,
                allow_sdxl_controlnet=self.bridge_data.allow_sdxl_controlnet,
                extra_slow_worker=self.bridge_data.extra_slow_worker,
                limit_max_steps=self.bridge_data.limit_max_steps,
                allow_lora=self.bridge_data.allow_lora,
                amount=self.bridge_data.max_batch,
            )

            # horde_sdk passes no per-request timeout to aiohttp, so bound the pop here:
            # a wedged gateway must fail fast and retry, not silently stall the loop.
            job_pop_response = await asyncio.wait_for(
                self.horde_client_session.submit_request(
                    job_pop_request,
                    ImageGenerateJobPopResponse,
                ),
                timeout=self._JOB_POP_TIMEOUT_SECONDS,
            )
            try:
                if (
                    hasattr(job_pop_response, "messages")
                    and job_pop_response.messages is not None
                    and len(job_pop_response.messages) > 0
                ):
                    for message in job_pop_response.messages:
                        message_id = message.get("id", None)
                        message_text = str(message.get("message", None))
                        message_origin = message.get("origin", None)
                        message_expiry = message.get("expiry", None)

                        if message_id not in self._api_messages_received:
                            if message_id is not None:
                                message_id = str(message_id)
                            self._api_messages_received[message_id] = APIWorkerMessage(
                                message_id=message_id,
                                message_text=message_text,
                                message_origin=message_origin,
                                message_expiry=message_expiry,
                            )
                            logger.debug(
                                f"Message {message_id} from {message_origin} (expires {message_expiry}): "
                                f"{message_text}",
                            )
                    self._prune_api_messages()
            except Exception as e:
                logger.error(f"Failed to process API messages: {e}")

            # Handle error responses by matching on the error message text
            if isinstance(job_pop_response, RequestErrorResponse):
                if "maintenance mode" in job_pop_response.message.lower():
                    if not self._last_pop_maintenance_mode:
                        logger.warning(f"Failed to pop job (Maintenance Mode): {job_pop_response}")
                        self.print_maint_mode_messages()
                        self._last_pop_maintenance_mode = True
                    # Automatically clear maintenance whenever it is detected (not only at
                    # startup) while remove_maintenance_on_init is enabled, retrying periodically
                    # so the worker recovers even if maintenance is re-applied while it is running.
                    # Throttled by _MAINTENANCE_REMOVAL_RETRY_INTERVAL so the (frequent) error
                    # job-pop retries do not hammer the API with removal requests.
                    if self.bridge_data.remove_maintenance_on_init:
                        now_maint = time.time()
                        if (
                            now_maint - self._last_maintenance_removal_attempt
                            >= self._MAINTENANCE_REMOVAL_RETRY_INTERVAL
                        ):
                            self._last_maintenance_removal_attempt = now_maint
                            try:
                                # to_thread: remove_maintenance uses horde_sdk's synchronous
                                # client — calling it inline would block the entire event loop
                                # (job pops, status output, heartbeats) until the API responds.
                                if await asyncio.to_thread(self.remove_maintenance):
                                    logger.success("Maintenance mode automatically deactivated")
                            except Exception:
                                logger.error("Maintenance mode couldn't been deactivated automatically", exc_info=True)
                elif "we cannot accept workers serving" in job_pop_response.message.lower():
                    logger.warning(f"Failed to pop job (Unrecognized Model): {job_pop_response}")
                    logger.error(
                        "Your worker is configured to use a model that is not accepted by the API. "
                        "Please check your models_to_load and make sure they are all valid.",
                    )
                elif "wrong credentials" in job_pop_response.message.lower():
                    logger.warning(f"Failed to pop job (Wrong Credentials): {job_pop_response}")
                    logger.error("Did you forget to set your worker name (`dreamer_name` in bridgeData.yaml)?")
                    logger.error(
                        "Horde Worker names must be unique horde-wide. If you haven't used this name before, "
                        "try changing your worker name.",
                    )
                else:
                    logger.error(f"Failed to pop job (API Error): {job_pop_response}")
                self._consecutive_pop_failures = 0  # API responded successfully (even if with an error)
                self._job_pop_frequency = self._error_job_pop_frequency
                self._last_pop_no_jobs_available = True
                return

        except aiohttp.ContentTypeError:
            # The API returned a non-JSON response (e.g. an HTML error page from Cloudflare).
            # This usually indicates a transient gateway or server issue.
            self._consecutive_pop_failures += 1
            message = (
                f"Failed to pop job (Unexpected Content-Type): "
                f"The API returned a non-JSON response. "
                f"This may be a temporary gateway/server issue. "
                f"Retrying in {self._error_job_pop_frequency:.0f}s. "
                f"(consecutive failures: {self._consecutive_pop_failures})"
            )
            if self._consecutive_pop_failures >= self._consecutive_pop_failure_warn_threshold:
                logger.error(message)
            else:
                logger.warning(message)
            self._job_pop_frequency = self._error_job_pop_frequency
            return

        except (TimeoutError, asyncio.TimeoutError):
            # Covers both the asyncio.wait_for cap above and aiohttp's own timeouts.
            # (asyncio.TimeoutError is builtins.TimeoutError from 3.11 on, but catch
            # both spellings so this also holds on older interpreters.)
            self._consecutive_pop_failures += 1
            message = (
                f"Failed to pop job (Timeout): "
                f"no response from the API within {self._JOB_POP_TIMEOUT_SECONDS:.0f}s. "
                f"The API or gateway appears unresponsive. "
                f"Retrying in {self._error_job_pop_frequency:.0f}s. "
                f"(consecutive failures: {self._consecutive_pop_failures})"
            )
            if self._consecutive_pop_failures >= self._consecutive_pop_failure_warn_threshold:
                logger.error(message)
            else:
                logger.warning(message)
            self._job_pop_frequency = self._error_job_pop_frequency
            return

        except Exception as e:
            self._consecutive_pop_failures += 1
            message = (
                f"Failed to pop job (Unexpected Error): {e} "
                f"(consecutive failures: {self._consecutive_pop_failures})"
            )
            if self._consecutive_pop_failures >= self._consecutive_pop_failure_warn_threshold:
                logger.error(message)
            else:
                logger.warning(message)

            self._job_pop_frequency = self._error_job_pop_frequency
            return

        self._consecutive_pop_failures = 0
        self._last_pop_maintenance_mode = False
        self._replaced_due_to_maintenance = False

        self._job_pop_frequency = self._default_job_pop_frequency

        info_string = "No job available. "
        if len(self.jobs_pending_inference) > 0:
            info_string += f"Current number of popped jobs: {len(self.jobs_pending_inference)}. "

        skipped_reasons = job_pop_response.skipped.model_dump(exclude_defaults=True)
        # Include the extra fields as well
        if job_pop_response.skipped.model_extra is not None:
            skipped_reasons.update(job_pop_response.skipped.model_extra)

        # Remove any '0' values
        skipped_reasons = {k: v for k, v in skipped_reasons.items() if v != 0}

        info_string += f"(Skipped reasons: {skipped_reasons})"

        if job_pop_response.id_ is None:
            self._last_pop_no_jobs_available = True
            if len(self.jobs_pending_inference) == 0:
                if self._last_pop_no_jobs_available_time == 0.0:
                    self._last_pop_no_jobs_available_time = cur_time

                self._time_spent_no_jobs_available += cur_time - self._last_pop_no_jobs_available_time
                self._last_pop_no_jobs_available_time = cur_time
            return

        self.job_faults[job_pop_response.id_] = []

        self._last_pop_no_jobs_available = False
        if self._last_pop_no_jobs_available_time > 0.0:
            self._time_spent_no_jobs_available += time.time() - self._last_pop_no_jobs_available_time
        self._last_pop_no_jobs_available_time = 0.0

        has_loras = job_pop_response.payload.loras is not None and len(job_pop_response.payload.loras) > 0
        has_post_processing = (
            job_pop_response.payload.post_processing is not None
            and len(
                job_pop_response.payload.post_processing,
            )
            > 0
        )
        logger.opt(colors=True).info(
            "<b><fg #a200ff>"
            f"Popped job {job_pop_response.id_} "
            f"({self.get_single_job_effective_megapixelsteps(job_pop_response)} eMPS) "
            f"(model: {job_pop_response.model}, batch: {job_pop_response.payload.n_iter}, "
            f"loras: {has_loras}, post_processing: {has_post_processing})"
            "</></b>",
        )

        # SDK workaround: fill in missing seed and strip denoising_strength without source_image
        if job_pop_response.payload.seed is None:
            logger.warning(f"Job {job_pop_response.id_} has no seed!")
            new_response_dict = job_pop_response.model_dump(by_alias=True)
            new_response_dict["payload"]["seed"] = random.randint(0, (2**32) - 1)

        if job_pop_response.payload.denoising_strength is not None and job_pop_response.source_image is None:
            new_response_dict = job_pop_response.model_dump(by_alias=True)
            new_response_dict["payload"]["denoising_strength"] = None

        if job_pop_response.payload.seed is None or (
            job_pop_response.payload.denoising_strength is not None and job_pop_response.source_image is None
        ):
            job_pop_response = ImageGenerateJobPopResponse(**new_response_dict)

        # Initiate the job faults list for this job, so that we don't need to check if it exists every time
        job_pop_response = await self._get_source_images(job_pop_response)

        if job_pop_response.id_ is None:
            logger.error("Job has no id!")
            return

        async with self._jobs_pending_inference_lock, self._job_pop_timestamps_lock:
            self.jobs_pending_inference.append(job_pop_response)
            self.total_num_jobs_queued += 1
            self._invalidate_megapixelsteps_cache()
            # self._testing_jobs_added += 1
            self.job_pop_timestamps[job_pop_response] = time.time()
            self.jobs_lookup[job_pop_response] = HordeJobInfo(
                sdk_api_job_info=job_pop_response,
                state=None,
                time_popped=self.job_pop_timestamps[job_pop_response],
            )

    _user_info_failed = False
    """Whether the API request to fetch user info failed."""
    _user_info_failed_reason: str | None = None
    """The reason the API request to fetch user info failed."""

    _current_worker_id: str | None = None
    """The current worker ID."""

    def calculate_kudos_info(self) -> None:
        """Calculate and log information about the kudos generated in the current session."""
        time_since_session_start = time.time() - self.session_start_time
        if time_since_session_start <= 0:
            return
        kudos_per_hour_session = self.kudos_generated_this_session / time_since_session_start * 3600
        active_time = time_since_session_start - self._time_spent_no_jobs_available
        active_kudos_per_hour = (
            self.kudos_generated_this_session / active_time * 3600 if active_time > 0 else 0.0
        )

        kudos_total_past_hour = self.calculate_kudos_totals()

        kudos_info_string = self.generate_kudos_info_string(
            time_since_session_start,
            kudos_per_hour_session,
            kudos_total_past_hour,
            active_kudos_per_hour,
        )

        self.log_kudos_info(kudos_info_string)

    def calculate_kudos_totals(self) -> float:
        """Calculate the total kudos generated in the past hour.

        Returns:
            float: The total kudos generated in the past hour.
        """
        kudos_total_past_hour = 0.0
        num_events_found = 0
        current_time = time.time()

        for event_time, kudos in reversed(self.kudos_events):
            if current_time - event_time > METRICS_CALCULATION_WINDOW_SECONDS:
                break

            num_events_found += 1
            kudos_total_past_hour += kudos

        elements_to_remove = len(self.kudos_events) - num_events_found
        if elements_to_remove > 0:
            self.kudos_events = self.kudos_events[:-elements_to_remove]

        return kudos_total_past_hour

    def generate_kudos_info_string(
        self,
        time_since_session_start: float,
        kudos_per_hour_session: float,
        kudos_total_past_hour: float,
        active_kudos_per_hour: float,
    ) -> str:
        """Generate a string with information about the kudos generated in the current session.

        Args:
            time_since_session_start (float): The time since the session started.
            kudos_per_hour_session (float): The kudos per hour generated in the current session.
            kudos_total_past_hour (float): The total kudos generated in the past hour.
            active_kudos_per_hour (float): The kudos per hour generated while active (jobs available).

        Returns:
            str: A string with information about the kudos generated in the current session.
        """
        kudos_info_string_elements = []
        if time_since_session_start < 3600:
            kudos_info_string_elements = [
                f"Session: {self.kudos_generated_this_session:,.1f} / " f"{time_since_session_start / 60:.1f}m",
            ]
        else:
            kudos_info_string_elements = [
                f"Session: {self.kudos_generated_this_session:,.1f} / " f"{time_since_session_start / 3600:.1f}h",
            ]

        if time_since_session_start > 3600:
            kudos_info_string_elements.append(
                f"{kudos_per_hour_session:,.1f} kd/hr",
            )
            # kudos_info_string_elements.append(
            #     f"Last Hour: {kudos_total_past_hour:,.2f} kudos",
            # )
        else:
            kudos_info_string_elements.append(
                f"{kudos_per_hour_session:,.1f} kd/hr (est)",
            )
            # kudos_info_string_elements.append(
            #     "Last Hour: (pending) kudos",
            # )

        if self._time_spent_no_jobs_available > self._max_time_spent_no_jobs_available:
            kudos_info_string_elements.append(
                f"Active: {active_kudos_per_hour:,.1f} kd/hr",
            )

        return " | ".join(kudos_info_string_elements)

    def log_kudos_info(self, kudos_info_string: str) -> None:
        """Log the kudos information string.

        Args:
            kudos_info_string (str): The kudos information string to log.
        """
        log_function = logger.opt(colors=True).info

        if self.bridge_data.limited_console_messages:
            log_function = logger.opt(colors=True).success

        # Combine kudos info and total accumulated into one line
        combined_msg_parts = []

        if self.kudos_generated_this_session > 0:
            combined_msg_parts.append(f"Kudos: {kudos_info_string}")

        logger.debug(f"len(kudos_events): {len(self.kudos_events)}")
        if self.user_info is not None and self.user_info.kudos_details is not None:
            total_kudos_msg = f"Total: {self.user_info.kudos_details.accumulated:,.0f} " f"({self.user_info.username})"
            if self.user_info.kudos_details.accumulated is not None and self.user_info.kudos_details.accumulated < 0:
                total_kudos_msg += " | Negative kudos = more requested than earned"

            combined_msg_parts.append(total_kudos_msg)

        # Log combined message only if there's something to log
        if combined_msg_parts:
            log_function(
                f"<fg #ffd700>{' | '.join(combined_msg_parts)}</>",
            )

    async def api_get_user_info(self) -> None:
        """Get the information associated with this API key from the API."""
        if self._shutting_down or self._last_pop_maintenance_mode:
            return

        # Skip if the client session is not initialized yet (during startup)
        if self.horde_client_session is None:
            return

        request = FindUserRequest(apikey=self.bridge_data.api_key)
        try:
            response = await self.horde_client_session.submit_request(request, UserDetailsResponse)
            if isinstance(response, RequestErrorResponse):
                logger.error(f"Failed to get user info (API Error): {response}")
                self._user_info_failed = True
                return

            self.user_info = response
            self._user_info_failed = False
            self._user_info_failed_reason = None

            if self.user_info.kudos_details is not None:
                self.calculate_kudos_info()

        except _async_client_exceptions as e:
            self._user_info_failed = True
            self._user_info_failed_reason = f"HTTP error (({type(e).__name__}) {e})"

        except Exception as e:
            self._user_info_failed = True
            self._user_info_failed_reason = f"Unexpected error (({type(e).__name__}) {e})"

        finally:
            if self._user_info_failed:
                logger.debug(f"Failed to get user info: {self._user_info_failed_reason}")
                logger.error("The server failed to respond. Is the horde or your internet down?")
            await logger.complete()

    _job_submit_loop_interval = 0.02
    """The interval between job submit loop iterations."""

    async def _job_submit_loop(self) -> None:
        """Run the job submit loop."""
        logger.debug("In _job_submit_loop")
        while True:
            # Snapshot the head job before awaiting so we discard the right job on exception,
            # even if api_submit_job internally reorders or removes it.
            head_job = self.jobs_pending_submit[0] if self.jobs_pending_submit else None
            try:
                await self.api_submit_job()
                if self.is_time_for_shutdown():
                    break
            except CancelledError as e:
                self._shutdown()
                logger.debug(f"CancelledError: {e}")
            except Exception:
                # api_submit_job raised unexpectedly (already logged by its @logger.catch decorator).
                # Discard the snapshotted head job if it is still in the queue so the submit queue
                # cannot be permanently blocked by a single broken job.
                if head_job is not None and head_job in self.jobs_pending_submit:
                    logger.error(
                        f"Discarding job {head_job.sdk_api_job_info.id_} from submit queue "
                        "after unexpected exception to prevent queue blockage",
                        exc_info=True,
                    )
                    self._discard_broken_job(head_job)

            await asyncio.sleep(self._job_submit_loop_interval)

    async def _api_call_loop(self) -> None:
        """Run the API call loop for popping jobs and doing miscellaneous API calls."""
        logger.debug("In _api_call_loop")

        while True:
            with logger.catch():
                try:
                    await self.api_job_pop()

                    if self.is_time_for_shutdown() or self._shut_down:
                        break
                except CancelledError as e:
                    self._shutdown()
                    logger.debug(f"CancelledError: {e}")

            await asyncio.sleep(self._api_call_loop_interval)

    async def _api_get_user_info_loop(self) -> None:
        """Run the API get user info loop."""
        logger.debug("In _api_get_user_info_loop")
        while True:
            with logger.catch():
                try:
                    # Retry quickly until the client session is ready (startup race)
                    if self.horde_client_session is None:
                        await asyncio.sleep(1)
                        continue
                    await self.api_get_user_info()
                    if self.is_time_for_shutdown() or self._shut_down:
                        break
                except CancelledError as e:
                    self._shutdown()
                    logger.debug(f"CancelledError: {e}")

            await asyncio.sleep(self._api_get_user_info_interval)

    async def api_get_workers_details(self) -> None:
        """Fetch individual details for each worker belonging to the current user."""
        if self._shutting_down:
            return
        if self.horde_client_session is None:
            return
        if self.user_info is None:
            return
        worker_ids = getattr(self.user_info, "worker_ids", None)
        if not worker_ids:
            self._workers_details = []
            return

        semaphore = asyncio.Semaphore(5)
        # Log connection errors at DEBUG when user-info is also failing (internet/Horde is
        # already known to be down).  Use WARNING only when user-info itself was last
        # successful so the operator can distinguish real problems from expected offline noise.
        conn_log_level = "DEBUG" if self._user_info_failed else "WARNING"

        async def _fetch_one(worker_id: str) -> dict[str, Any] | None:
            async with semaphore:
                try:
                    request = SingleWorkerDetailsRequest(worker_id=str(worker_id))
                    response = await self.horde_client_session.submit_request(request, SingleWorkerDetailsResponse)
                    if isinstance(response, RequestErrorResponse):
                        logger.debug(f"Failed to get details for worker {worker_id}: {response}")
                        return None
                    ba = response.bridge_agent or ""
                    ba_parts = ba.split(":")
                    version = ba_parts[1] if len(ba_parts) > 1 else ""
                    return {
                        "id": str(response.id_) if response.id_ else str(worker_id),
                        "name": response.name or "",
                        "version": version,
                        "type": str(response.type_.value) if response.type_ else "",
                        "online": response.online,
                        "nsfw": response.nsfw,
                        "trusted": response.trusted,
                        "img2img": response.img2img,
                        "painting": response.painting,
                        "lora": response.lora,
                        "max_pixels": response.max_pixels,
                        "threads": response.threads,
                        "models": list(response.models) if response.models else [],
                        "uptime": response.uptime,
                        "kudos_rewards": response.kudos_rewards,
                    }
                except Exception as e:  # noqa: BLE001
                    logger.log(conn_log_level, f"Failed to get details for worker {worker_id}: {e}")
                    return None

        results = await asyncio.gather(*(_fetch_one(str(wid)) for wid in worker_ids))
        fetched = [w for w in results if w is not None]
        # Only overwrite the cached list when at least one fetch succeeded.  If all fetches
        # fail (e.g. internet / Horde is down) we keep the last known state so that the web
        # UI continues to display worker cards rather than showing an empty list.
        if fetched:
            self._workers_details = fetched

    async def _delete_worker(self, worker_id: str) -> bool:
        """Delete a worker from the Horde via the API.

        Called by the web UI delete endpoint.  Only offline workers that are not the
        currently running worker may be deleted (those guards are enforced in the web UI
        handler before this method is called).

        Args:
            worker_id: The UUID of the worker to delete.

        Returns:
            True if the worker was deleted successfully, False otherwise.
        """
        if self.horde_client_session is None:
            logger.warning(f"Cannot delete worker {worker_id}: horde client session is not available")
            return False
        try:
            request = DeleteWorkerRequest(
                apikey=self.bridge_data.api_key,
                worker_id=worker_id,
            )
            response = await self.horde_client_session.submit_request(request, DeleteWorkerResponse)
            if isinstance(response, RequestErrorResponse):
                logger.error(f"Failed to delete worker {worker_id} (API error): {response}")
                return False
            logger.info(f"Successfully deleted worker {worker_id} (name: {response.deleted_name!r})")
            # Refresh the workers details list so the UI picks up the change promptly.
            await self.api_get_workers_details()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Exception while deleting worker {worker_id}: {exc}")
            return False

    async def _api_get_workers_details_loop(self) -> None:
        """Periodically fetch per-worker detail records for the User page."""
        logger.debug("In _api_get_workers_details_loop")
        while True:
            with logger.catch():
                try:
                    if self.horde_client_session is None or self.user_info is None:
                        await asyncio.sleep(1)
                        continue
                    worker_ids = getattr(self.user_info, "worker_ids", None)
                    if worker_ids is None:
                        await asyncio.sleep(1)
                        continue
                    # Skip the fetch when user-info is already failing (internet / Horde is
                    # down).  Worker details would also fail, so avoid the extra noise and
                    # wait for user-info to recover before retrying.  Sleep in 1-second
                    # increments so a pending shutdown is not delayed by the full interval.
                    if self._user_info_failed:
                        remaining = float(self._api_get_workers_details_interval)
                        while remaining > 0:
                            if self.is_time_for_shutdown() or self._shut_down:
                                return
                            step = min(1.0, remaining)
                            await asyncio.sleep(step)
                            remaining -= step
                        continue
                    await self.api_get_workers_details()
                    if self.is_time_for_shutdown() or self._shut_down:
                        break
                except CancelledError as e:
                    self._shutdown()
                    logger.debug(f"CancelledError: {e}")
            await asyncio.sleep(self._api_get_workers_details_interval)

    _status_message_frequency = 20.0
    """The rate in seconds at which to print status messages with details about the current state of the worker."""
    _last_status_message_time = 0.0
    """The epoch time of the last status message."""
    _replaced_due_to_maintenance = False

    async def _process_control_loop(self) -> None:
        # If spawning the child processes fails (out of memory/page file, fork/spawn refused,
        # import failure re-raised by multiprocessing, ...), initiate a shutdown instead of
        # letting only this task die: the API loops would otherwise keep popping jobs that
        # nothing can ever process, leaving a zombie worker.
        try:
            self.start_safety_processes()
            self.start_inference_processes()
        except Exception:
            logger.critical(
                "Failed to start worker child processes during startup — shutting down. "
                "(Common causes: insufficient RAM/page file, or the OS refusing to spawn new processes.)",
            )
            self._shutdown()
            raise

        while True:
            try:
                if self.stable_diffusion_reference is None:
                    return
                with logger.catch(reraise=True):
                    await asyncio.sleep(self._loop_interval)

                    async with (
                        self._jobs_lookup_lock,
                        self._jobs_pending_inference_lock,
                        self._jobs_safety_check_lock,
                        self._completed_jobs_lock,
                    ):
                        self.receive_and_handle_process_messages()
                        self.detect_deadlock()
                        self._check_process_message_queue_health()

                    if len(self.jobs_pending_safety_check) > 0:
                        async with self._jobs_safety_check_lock:
                            self.start_evaluate_safety()

                    if (
                        self._job_pops_paused
                        and len(self.jobs_pending_inference) == 0
                        and len(self.jobs_in_progress) == 0
                        and len(self.jobs_pending_safety_check) == 0
                        and len(self.jobs_being_safety_checked) == 0
                        and len(self.jobs_pending_submit) == 0
                    ):
                        self._unload_idle_inference_models()

                    free_process_or_model_loaded = (
                        self.is_free_inference_process_available() or self.is_any_model_preloaded()
                    )

                    if (
                        self._last_pop_maintenance_mode
                        and len(self.jobs_pending_inference) == 0
                        and len(self.jobs_in_progress) == 0
                        and len(self.jobs_pending_safety_check) == 0
                        and len(self.jobs_being_safety_checked) == 0
                        and len(self.jobs_pending_submit) == 0
                        and not self._replaced_due_to_maintenance
                    ):
                        # We're in maintenance mode and there are no jobs to run, so we're going to unload all models
                        logger.warning("Reloading all process due to maintenance mode")
                        for process_info in self._process_map.values():
                            if process_info.process_type == HordeProcessType.INFERENCE:
                                self._replace_inference_process(process_info)
                            self._replaced_due_to_maintenance = True
                        # Reset the flag now that processes have been reloaded so that status messages resume
                        # and further maintenance mode responses can trigger remove_maintenance() again.
                        self._last_pop_maintenance_mode = False

                    if free_process_or_model_loaded and len(self.jobs_pending_inference) > 0:
                        # Theres a job pending inference and a process available to
                        # preload the model or start inference
                        async with (
                            self._jobs_lookup_lock,
                            self._jobs_pending_inference_lock,
                            self._jobs_safety_check_lock,
                            self._completed_jobs_lock,
                            self._job_pop_timestamps_lock,
                        ):
                            # preload_models() returns True when a new model was dispatched for
                            # preloading on an available process.  We still want to call
                            # start_inference() if there are already-preloaded processes waiting
                            # for a job — otherwise those processes sit in MODEL_PRELOADED
                            # indefinitely while the manager keeps loading new models.
                            started_preload = self.preload_models()
                            if not started_preload or self._process_map.num_preloaded_processes() > 0:
                                next_job_and_process = self.get_next_job_and_process(information_only=True)

                                next_job_heavy_model_and_workflow = False
                                if next_job_and_process is not None:
                                    next_model = next_job_and_process.next_job.model
                                    if next_model is not None:
                                        next_model_record = self.stable_diffusion_reference.root.get(next_model)
                                        next_workflow = next_job_and_process.next_job.payload.workflow

                                        next_job_heavy_model_and_workflow = (
                                            next_model_record is not None
                                            and next_model_record.baseline
                                            == STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_xl
                                            and next_workflow in KNOWN_SLOW_WORKFLOWS
                                        )

                                        if next_model in VRAM_HEAVY_MODELS:
                                            next_job_heavy_model_and_workflow = True

                                keep_single_inference, single_inf_reason = self._process_map.keep_single_inference(
                                    stable_diffusion_model_reference=self.stable_diffusion_reference,
                                    post_process_job_overlap=self.bridge_data.post_process_job_overlap,
                                )

                                if keep_single_inference and (
                                    (len(self.jobs_pending_inference) + len(self.jobs_in_progress)) > 1
                                ):
                                    if (
                                        time.time() - self._batch_wait_log_time > 10
                                    ) and self.bridge_data.max_threads > 1:
                                        logger.opt(colors=True).info(
                                            "<fg #7b7d7d>"
                                            f"<i>Blocking further inference due to {single_inf_reason}.</i>"
                                            "</>",
                                        )
                                        self._batch_wait_log_time = time.time()

                                elif (
                                    next_job_and_process is not None
                                    and (
                                        next_job_and_process.next_job.payload.n_iter > 1
                                        or next_job_heavy_model_and_workflow
                                    )
                                    and (
                                        self._process_map.num_busy_with_inference() > 0
                                        or self._process_map.num_busy_with_post_processing() > 0
                                    )
                                ):
                                    if time.time() - self._batch_wait_log_time > 10:
                                        logger.opt(colors=True).info(
                                            "<fg #7b7d7d>"
                                            f"<i>Blocking starting batch job {next_job_and_process.next_job.id_} "
                                            "because a thread is already busy with a heavy model/workflow or batch job"
                                            ".</i>"
                                            "</>",
                                        )
                                        self._batch_wait_log_time = time.time()
                                else:
                                    if not self.start_inference():
                                        self.unload_models()

                    async with (
                        self._jobs_lookup_lock,
                        self._jobs_pending_inference_lock,
                        self._jobs_safety_check_lock,
                        self._completed_jobs_lock,
                    ):
                        await asyncio.sleep(self._loop_interval)
                        self.receive_and_handle_process_messages()
                        if self.replace_hung_processes():
                            await asyncio.sleep(self._loop_interval / 2)
                            await asyncio.sleep(self._loop_interval / 2)
                        self._replace_all_safety_process()

                    should_scale_down = self._inference_scale_down_requested or (
                        self._process_map.num_loaded_inference_processes() > self.max_inference_processes
                    )
                    if should_scale_down:
                        self.end_inference_processes()
                        self._inference_scale_down_requested = (
                            self._process_map.num_loaded_inference_processes() > self.max_inference_processes
                        )

                    if self._shutting_down and not self._last_pop_recently():
                        self.end_inference_processes()

                    if self.is_time_for_shutdown():
                        self._start_timed_shutdown()
                        break

                self.print_status_method()

                await asyncio.sleep(self._loop_interval / 2)
            except CancelledError as e:
                self._shutdown()
                logger.debug(f"CancelledError: {e}")
            except Exception:
                # Unexpected errors are already logged by logger.catch(reraise=True) in the loop body;
                # sleep before retrying to keep the control loop alive without duplicating logs.
                logger.debug("_process_control_loop recovering from exception (see above), sleeping before retry")
                await asyncio.sleep(self._loop_interval)

        while len(self.jobs_pending_inference) > 0:
            # If every inference process is already ending/ended, the pending jobs will
            # never drain — break so the worker can proceed with the restart.
            if all(
                p.last_process_state in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED)
                for p in self._process_map.get_inference_processes()
            ):
                logger.warning(
                    f"Shutdown drain: {len(self.jobs_pending_inference)} pending job(s) remain but "
                    "all inference processes are ending — proceeding with restart.",
                )
                break
            await asyncio.sleep(0.2)
            async with self._jobs_pending_inference_lock, self._jobs_safety_check_lock, self._completed_jobs_lock:
                self.receive_and_handle_process_messages()
                self.detect_deadlock()
                self.replace_hung_processes()  # Only checks for hung processes, doesn't replace them during shutdown
            await asyncio.sleep(0.2)

        # Set _shut_down early so the timed-shutdown safety thread (started by
        # _start_timed_shutdown above) backs off before calling os._exit(1).
        # Without this, the timed-shutdown fires after its short wait (2 s when
        # no jobs are pending) while end_inference_processes(force=True) is still
        # blocking on join() calls — killing the program before os.execv can
        # perform a restart.
        self._shut_down = True

        self.end_inference_processes(force=True)
        self.end_safety_processes()

        logger.info("Shutting down process manager")
        for process in self._process_map.values():
            try:
                process.mp_process.terminate()
                process.mp_process.join(0.5)
            except Exception as e:
                logger.debug(f"Error terminating process {process.process_id} during shutdown: {e}")

        await asyncio.sleep(0.2)

        return

    _last_deadlock_detected_time: float = 0.0
    """The epoch time of the last deadlock detected."""
    _in_deadlock: bool = False
    """Whether the worker is in a deadlock state."""
    _in_queue_deadlock: bool = False
    """Whether the worker is in a queue deadlock state."""
    _last_queue_deadlock_detected_time: float = 0.0
    """The epoch time of the last queue deadlock detected."""
    _queue_deadlock_model: str | None = None
    """The model causing the queue deadlock."""
    _queue_deadlock_process_id: int | None = None
    """The process ID causing the queue deadlock."""

    def detect_deadlock(self) -> None:
        """Detect if there are jobs in the queue but no processes doing anything."""

        def _print_deadlock_info() -> None:
            logger.debug(f"Jobs in queue: {len(self.jobs_pending_inference)}")
            logger.debug(f"Jobs in progress: {len(self.jobs_in_progress)}")
            logger.debug(f"Jobs pending safety check: {len(self.jobs_pending_safety_check)}")
            logger.debug(f"Jobs being safety checked: {len(self.jobs_being_safety_checked)}")
            logger.debug(f"Jobs completed: {len(self.jobs_pending_submit)}")
            logger.debug(f"Jobs faulted: {self._num_jobs_faulted}")
            logger.debug(f"horde_model_map: {self._horde_model_map}")
            logger.debug(f"process_map: {self._process_map}")

        if self._last_pop_recently():
            # We just popped a job, lets allow some time for gears to start turning
            # before we assume we're in a deadlock
            return

        if (
            not self._in_queue_deadlock
            and self._process_map.all_waiting_for_job()
            and len(self.jobs_pending_inference) > 0
        ):
            currently_loaded_models = set()
            model_process_map: dict[str, int] = {}
            for process in self._process_map.values():
                if process.loaded_horde_model_name is not None:
                    currently_loaded_models.add(process.loaded_horde_model_name)
                    model_process_map[process.loaded_horde_model_name] = process.process_id

            for job in self.jobs_pending_inference:
                if job.model in currently_loaded_models:
                    self._in_queue_deadlock = True
                    self._last_queue_deadlock_detected_time = time.time()
                    self._queue_deadlock_model = job.model
                    self._queue_deadlock_process_id = model_process_map[job.model]
                    break
            else:
                logger.debug("Queue deadlock detected without a model causing it.")
                _print_deadlock_info()
                self._in_queue_deadlock = True
                self._last_queue_deadlock_detected_time = time.time()
                # we're going to fall back to the next model in the deque
                self._queue_deadlock_model = self.jobs_pending_inference[0].model

        elif self._in_queue_deadlock and (self._last_queue_deadlock_detected_time + 30) < time.time():
            if self._process_map.num_starting_processes() > 0:
                logger.debug("Queue deadlock detected but some processes are starting. Waiting.")
                self._last_queue_deadlock_detected_time = time.time()
                return

            logger.debug("Queue deadlock detected")
            _print_deadlock_info()

            if self._queue_deadlock_model is not None:
                logger.debug(f"Model causing deadlock: {self._queue_deadlock_model}")
            else:
                logger.warning("Queue deadlock detected but no model causing it.")

            self._in_queue_deadlock = False
            self._queue_deadlock_model = None
            self._queue_deadlock_process_id = None

        if (
            (not self._in_deadlock)
            and (len(self.jobs_pending_inference) > 0 or len(self.jobs_in_progress) > 0 or len(self.jobs_lookup) > 0)
            and self._process_map.num_busy_processes() == 0
        ):
            self._last_deadlock_detected_time = time.time()
            self._in_deadlock = True
            logger.debug("Deadlock detected")
            _print_deadlock_info()
        elif (
            self._in_deadlock
            and (self._last_deadlock_detected_time + 10) < time.time()
            and self._process_map.num_busy_processes() == 0
        ):
            logger.debug("Deadlock still detected after 10 seconds.")

            self._in_deadlock = False
        elif (
            self._in_deadlock
            and (self._last_deadlock_detected_time + 5) < time.time()
            and self._process_map.num_busy_processes() > 0
        ):
            logger.debug("Deadlock was likely false-alarm.")
            self._in_deadlock = False

    def print_status_method(self) -> None:
        """Print the status of the worker if it's time to do so."""
        if self._last_pop_maintenance_mode:
            return

        cur_time = time.time()
        if cur_time - self._last_status_message_time > self._status_message_frequency:
            AIWORKER_LIMITED_CONSOLE_MESSAGES = os.getenv("AIWORKER_LIMITED_CONSOLE_MESSAGES", False)

            logging_function = logger.opt(colors=True).info

            if AIWORKER_LIMITED_CONSOLE_MESSAGES:
                logging_function = logger.opt(colors=True).success

            process_info_strings = self._process_map.get_process_info_strings()

            logging_function("<fg #00d7ff>" + "=" * 80 + "</>")

            self._prune_api_messages()
            if len(self._api_messages_received) > 0:
                logging_function("<b><fg #ffd700>API Messages:</></b>")
                for message_id, message in self._api_messages_received.items():
                    try:
                        message_text = message.message_text or ""
                        log_safe_message = message_text.replace("<", "&lt;").replace(">", "&gt;")
                        log_safe_message = log_safe_message.replace("\n", " ")
                        log_safe_message = log_safe_message.replace("\r", " ")
                        log_safe_message = log_safe_message.replace("\t", " ")
                        log_safe_message = log_safe_message.replace("{", "{{").replace("}", "}}")
                        log_safe_message = log_safe_message.replace('"', "'")
                        log_safe_message = log_safe_message.replace("'", "'")

                        logging_function(
                            f"  <fg #000><bg #0ff127>{log_safe_message} "
                            f"(from {message.message_origin}, expires {message.message_expiry}, "
                            f"message_id: {message_id[:8]})</></>",
                            "</></>",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to print API message: {e}")

            # Only show detailed process info if not in limited console mode
            if not AIWORKER_LIMITED_CONSOLE_MESSAGES:
                logging_function("<b><fg #00d7ff>Processes:</></b>")
                for process_info_string in process_info_strings:
                    logging_function("  " + process_info_string)

                logging_function("<fg #00d7ff>" + "-" * 80 + "</>")
            else:
                # In limited mode, just show a brief summary
                num_busy = self._process_map.num_busy_processes()
                num_total = len(self._process_map)
                logging_function(f"<b><fg #00d7ff>Processes:</></b> {num_busy}/{num_total} busy")
                logging_function("<fg #00d7ff>" + "-" * 80 + "</>")

            logging_function("<b><fg #00ff87>Jobs:</></b>")

            # Show jobs in progress
            jobs_in_progress_list = []
            for x in self.jobs_in_progress:
                shortened_id = str(x.id_.root)[:8] if x.id_ is not None else "None?"
                safe_model = (x.model or "").replace("<", "\\<")
                jobs_in_progress_list.append(f"[{shortened_id}: <u>{safe_model}</u>]")

            if jobs_in_progress_list:
                logging_function(f'  In Progress: {", ".join(jobs_in_progress_list)}')

            # Show queued jobs (exclude jobs already in progress)
            jobs_pending_list = []
            for x in self.jobs_pending_inference:
                if x in self.jobs_in_progress:
                    continue
                shortened_id = str(x.id_.root)[:8] if x.id_ is not None else "None?"
                safe_model = (x.model or "").replace("<", "\\<")
                jobs_pending_list.append(f"[{shortened_id}: <u>{safe_model}</u>]")

            if jobs_pending_list:
                logging_function(f'  Queued: {", ".join(jobs_pending_list)}')

            if not jobs_in_progress_list and not jobs_pending_list:
                logging_function("  No active jobs")

            # Warn when inference processes have been idle in WAITING_FOR_JOB for a suspiciously long time
            # and the worker is not simply waiting because no jobs are available.
            _idle_warn_threshold = 120  # seconds before flagging a process as unexpectedly idle
            if not self._last_pop_no_jobs_available:
                idle_inference_processes = [
                    (pid, pinfo)
                    for pid, pinfo in self._process_map.items()
                    if pinfo.process_type == HordeProcessType.INFERENCE
                    and pinfo.last_process_state == HordeProcessState.WAITING_FOR_JOB
                    and (cur_time - pinfo.last_heartbeat_timestamp) > _idle_warn_threshold
                ]
                if idle_inference_processes:
                    if not self._idle_process_warning_logged:
                        idle_deltas = ", ".join(
                            f"Process {pid}: {cur_time - pinfo.last_heartbeat_timestamp:.0f}s"
                            for pid, pinfo in idle_inference_processes
                        )
                        logger.warning(
                            f"Inference process(es) have been idle in WAITING_FOR_JOB for over "
                            f"{_idle_warn_threshold}s with no active jobs dispatched: {idle_deltas}. "
                            f"If this persists past {self.bridge_data.process_timeout}s, "
                            f"processes will be automatically recovered.",
                        )
                        self._idle_process_warning_logged = True
                else:
                    self._idle_process_warning_logged = False
            else:
                self._idle_process_warning_logged = False

            active_models = {
                process.loaded_horde_model_name
                for process in self._process_map.values()
                if process.loaded_horde_model_name is not None
            }

            logger.debug(f"Active models: {active_models}")

            _no_jobs_time = self._time_spent_no_jobs_available
            if self._last_pop_no_jobs_available_time > 0:
                _no_jobs_time += cur_time - self._last_pop_no_jobs_available_time

            job_info_message = "  " + " | ".join(
                [
                    f"in progress: {len(self.jobs_in_progress)}",
                    f"pending: {len(self.jobs_pending_inference)} ({self.get_pending_megapixelsteps()} eMPS)",
                    f"popped: {self.num_jobs_total}",
                    f"done: {self.total_num_completed_jobs}",
                    f"faulted: {self._num_jobs_faulted}",
                    f"slow: {self._num_job_slowdowns}",
                    f"recoveries: {self._num_process_recoveries}",
                    f"pop errors: {self._consecutive_pop_failures}",
                    f"no jobs: {_no_jobs_time:.1f}s",
                ],
            )

            logging_function(
                f"<fg #00ff87>{job_info_message}</>",
            )

            # Print failing models periodically
            if (
                self._failed_models
                and cur_time - self._last_failed_models_print_time > self.FAILED_MODELS_REPORT_INTERVAL_SECONDS
            ):
                logging_function("<fg #00d7ff>" + "-" * 80 + "</>")
                logging_function("<b><fg #ff5f5f>Failing Models:</></b>")
                # Sort by failure count descending
                sorted_failures = sorted(self._failed_models.items(), key=lambda x: x[1], reverse=True)
                for model_name, count in sorted_failures[: self.MAX_FAILING_MODELS_TO_DISPLAY]:
                    logging_function(f"  <fg #ff6b6b>{model_name}: {count} failures</>")
                self._last_failed_models_print_time = cur_time

            logging_function("<fg #00d7ff>" + "-" * 80 + "</>")

            # Only show worker config periodically (not every status update)
            if (
                not AIWORKER_LIMITED_CONSOLE_MESSAGES
                and cur_time - self._last_worker_config_print_time > self.WORKER_CONFIG_REPORT_INTERVAL_SECONDS
            ):
                logging_function("<b><fg #5fd7ff>Worker Config:</></b>")

                max_power_dimension = int(math.sqrt(self.bridge_data.max_power * 8 * 64 * 64))
                worker_info = " | ".join(
                    [
                        f"name: {self.bridge_data.dreamer_worker_name}",
                        f"v{horde_worker_regen.__version__}",
                        f"user: {self.user_info.username if self.user_info is not None else 'Unknown'}",
                        f"models: {len(self.bridge_data.image_models_to_load)}",
                        f"custom: {bool(self.bridge_data.custom_models)}",
                        f"power: {self.bridge_data.max_power} ({max_power_dimension}x{max_power_dimension})",
                        f"threads: {self.max_concurrent_inference_processes}",
                        f"queue: {self.bridge_data.queue_size}",
                        f"safety_gpu: {self.bridge_data.safety_on_gpu}",
                        f"img2img: {self.bridge_data.allow_img2img}",
                        f"lora: {self.bridge_data.allow_lora}",
                        f"cn: {self.bridge_data.allow_controlnet}",
                        f"sdxl_cn: {self.bridge_data.allow_sdxl_controlnet}",
                        f"pp: {self.bridge_data.allow_post_processing}",
                        f"pp_overlap: {self.bridge_data.post_process_job_overlap}",
                    ],
                )
                logger.info(f"  {worker_info}")

                memory_info = " | ".join(
                    [
                        f"unload_vram: {self.bridge_data.unload_models_from_vram_often}",
                        f"high_perf: {self.bridge_data.high_performance_mode}",
                        f"med_perf: {self.bridge_data.moderate_performance_mode}",
                        f"high_mem: {self.bridge_data.high_memory_mode}",
                    ],
                )
                logger.info(f"  {memory_info}")

                self._last_worker_config_print_time = cur_time

            logger.debug(
                " | ".join(
                    [
                        f"preload_timeout: {self.bridge_data.preload_timeout}",
                        f"download_timeout: {self.bridge_data.download_timeout}",
                        f"post_process_timeout: {self.bridge_data.post_process_timeout}",
                        f"very_high_memory_mode: {self.bridge_data.very_high_memory_mode}",
                        f"cycle_process_on_model_change: {self.bridge_data.cycle_process_on_model_change}",
                        f"exit_on_unhandled_faults: {self.bridge_data.exit_on_unhandled_faults}",
                        f"jobs_pending_safety_check: {len(self.jobs_pending_safety_check)}",
                        f"jobs_being_safety_checked: {len(self.jobs_being_safety_checked)}",
                        f"jobs_in_progress: {len(self.jobs_in_progress)}",
                    ],
                ),
            )

            if os.getenv("AIWORKER_NOT_REQUIRED_VERSION"):
                logger.warning(
                    "There is a required update available for the AI Worker. "
                    "`git pull` and `update-runtime` to update.",
                )
            elif os.getenv("AIWORKER_NOT_RECOMMENDED_VERSION"):
                logger.warning(
                    "There is a recommended update available for the AI Worker. "
                    "`git pull` and `update-runtime` to update.",
                )

            if self.bridge_data.extra_slow_worker:
                if not self.bridge_data.limit_max_steps:
                    logger.warning(
                        "Extra slow worker mode is enabled, but limit_max_steps is not enabled. "
                        "Consider enabling limit_max_steps to prevent long running jobs.",
                    )
                if self.bridge_data.max_batch > 1:
                    logger.warning(
                        "Extra slow worker mode is enabled, but max_batch is greater than 1. "
                        "Consider setting max_batch to 1 to prevent long running batch jobs.",
                    )
                if self.bridge_data.allow_sdxl_controlnet:
                    logger.warning(
                        "Extra slow worker mode is enabled, but allow_sdxl_controlnet is enabled. "
                        "Consider disabling allow_sdxl_controlnet to prevent long running jobs.",
                    )

            for device in self._device_map.root.values():
                total_memory_mb = device.total_memory / 1024 / 1024
                if total_memory_mb < 10_000 and self.bridge_data.high_memory_mode:
                    logger.warning(
                        f"Device {device.device_name} ({device.device_index}) has less than 10GB of memory. "
                        "This may cause issues with `high_memory_mode` enabled.",
                    )
                elif (
                    total_memory_mb > 20_000
                    and not self.bridge_data.high_memory_mode
                    and self.bridge_data.max_threads == 1
                    and self.total_ram_gigabytes > 32
                ):
                    logger.warning(
                        f"Device {device.device_name} ({device.device_index}) has more than 20GB of memory. "
                        "You should enable `high_memory_mode` in your config to take advantage of this.",
                    )
                elif total_memory_mb > 20_000 and self.bridge_data.extra_slow_worker:
                    logger.warning(
                        f"Device {device.device_name} ({device.device_index}) has more than 20GB of memory. "
                        "There are very few GPUs with this much memory that should be running in extra slow worker "
                        "mode. Consider disabling `extra_slow_worker` in your config.",
                    )

            if self._too_many_consecutive_failed_jobs:
                time_since_failure = cur_time - self._too_many_consecutive_failed_jobs_time
                logger.error(
                    "Too many consecutive failed jobs. This may be due to a misconfiguration or other issue. "
                    "Please check your logs and configuration.",
                )
                logger.error(
                    f"Time since last job failure: {time_since_failure:.2f}s. "
                    f"{self._too_many_consecutive_failed_jobs_wait_time} seconds must pass before resuming.",
                )

            minutes_allowed_without_jobs = self.bridge_data.minutes_allowed_without_jobs
            seconds_allowed_without_jobs = minutes_allowed_without_jobs * 60
            if self._time_spent_no_jobs_available > seconds_allowed_without_jobs:
                if not self.bridge_data.suppress_speed_warnings:
                    pass
                else:
                    logger.debug(
                        "Suppressed warning about time spent without jobs "
                        f"for {minutes_allowed_without_jobs} minutes",
                    )

            if self._shutting_down:
                logger.opt(colors=True).warning("<red>" + "=" * 80 + "</>")
                logger.opt(colors=True).warning("<red>SHUTTING DOWN - Finishing current jobs...</>")
                logger.opt(colors=True).warning("<red>" + "=" * 80 + "</>")
                self._status_message_frequency = 5.0

            self._last_status_message_time = cur_time
            logging_function("<fg #00d7ff>" + "=" * 80 + "</>")

    _bridge_data_loop_interval = 1.0
    """The interval between bridge data loop iterations."""
    _last_bridge_data_reload_time = 0.0
    """The epoch time of the last bridge data reload."""

    _bridge_data_last_modified_time = 0.0
    """The time the bridge data file on disk was last modified."""

    def get_bridge_data_from_disk(self) -> None:
        """Load the bridge data from disk."""
        if self.bridge_data._loaded_from_env_vars:
            return

        try:
            self.bridge_data = BridgeDataLoader.load(
                file_path=BRIDGE_CONFIG_FILENAME,
                horde_model_reference_manager=self.horde_model_reference_manager,
            )
            if self.bridge_data.max_threads != self._max_concurrent_inference_processes:
                logger.warning(
                    f"max_threads in {BRIDGE_CONFIG_FILENAME} cannot be changed while the worker is running.",
                )
            self._refresh_model_configuration_state_after_reload()
            logger.debug(f"Models to load: {self.bridge_data.image_models_to_load}")
            logger.debug(f"Custom models: {self.bridge_data.custom_models}")
        except Exception as e:
            logger.debug(e)

            if "No such file or directory" in str(e):
                logger.error(f"Could not find {BRIDGE_CONFIG_FILENAME}. Please create it and try again.")

            if isinstance(e, ValidationError):
                # Print a list of fields that failed validation
                logger.error(f"The following fields in {BRIDGE_CONFIG_FILENAME} failed validation:")
                for error in e.errors():
                    loc_str = error["loc"][0] if error.get("loc") else "<root>"
                    logger.error(f"{loc_str}: {error['msg']}")

            return

    async def _bridge_data_loop(self) -> None:
        while True:
            try:
                if self._shutting_down:
                    break

                self._bridge_data_last_modified_time = os.path.getmtime(BRIDGE_CONFIG_FILENAME)

                if self._last_bridge_data_reload_time < self._bridge_data_last_modified_time:
                    logger.info(f"Reloading {BRIDGE_CONFIG_FILENAME}")
                    self.get_bridge_data_from_disk()
                    self._last_bridge_data_reload_time = time.time()
                    logger.success(f"Reloaded {BRIDGE_CONFIG_FILENAME}")
                    self.enable_performance_mode()
                await asyncio.sleep(self._bridge_data_loop_interval)
            except CancelledError as e:
                self._shutdown()
                logger.debug(f"CancelledError: {e}")
            except FileNotFoundError:
                logger.warning(f"Could not find {BRIDGE_CONFIG_FILENAME}. Waiting for it to be created...")
                await asyncio.sleep(self._bridge_data_loop_interval)
            except Exception:
                logger.exception("Unexpected error in bridge data loop")
                await asyncio.sleep(self._bridge_data_loop_interval)

    def _calculate_granular_progress(
        self,
        process_state: HordeProcessState,
        inference_progress: int | None,
    ) -> int:
        """Calculate overall job progress based on current stage and inference progress.

        The progress bar is divided into stages:
        - Model Loading: 0-20%
        - Inference: 1-70% (starts at 1% to avoid a 0% flash, rises to 70% at completion)
        - Post-Processing: 70-80%
        - Safety Check: 80-90%
        - Submission: 90-100%

        Args:
            process_state: Current process state
            inference_progress: Progress percentage from inference (0-100), if applicable

        Returns:
            Overall progress percentage (0-100)
        """
        # Job received but not yet started (0%)
        if process_state in (
            HordeProcessState.JOB_RECEIVED,
            HordeProcessState.WAITING_FOR_JOB,
            HordeProcessState.PROCESS_STARTING,
        ):
            return 0

        # Model loading stages (0-20%)
        if process_state in (
            HordeProcessState.DOWNLOADING_MODEL,
            HordeProcessState.DOWNLOADING_AUX_MODEL,
            HordeProcessState.MODEL_PRELOADING,
            HordeProcessState.MODEL_LOADING,
        ):
            return 10  # Mid-point of model loading stage
        if process_state in (
            HordeProcessState.DOWNLOAD_COMPLETE,
            HordeProcessState.DOWNLOAD_AUX_COMPLETE,
            HordeProcessState.MODEL_PRELOADED,
            HordeProcessState.MODEL_LOADED,
        ):
            return 20  # Model loading complete

        # Inference stages (1-70%)
        if process_state in (
            HordeProcessState.INFERENCE_STARTING,
            HordeProcessState.INFERENCE_PROCESSING,
        ):
            if inference_progress is not None:
                # Map 0-100% inference progress to 0-70% overall, with a floor of 1%
                # to avoid dropping the UI back to 0% at the very start of inference.
                return max(1, int(inference_progress * 0.7))
            return 1  # Start of inference - small non-zero baseline to prevent flash

        # Post-processing stage (70-80%)
        if process_state in (
            HordeProcessState.INFERENCE_POST_PROCESSING,
            HordeProcessState.POST_PROCESSING_STARTING,
        ):
            if inference_progress is not None and inference_progress < 100:
                # Map 0-100% post-processing to 70-80% overall
                return 70 + int(inference_progress * 0.1)
            return 75  # Mid-point of post-processing

        if process_state in (
            HordeProcessState.INFERENCE_COMPLETE,
            HordeProcessState.POST_PROCESSING_COMPLETE,
        ):
            return 80  # Post-processing complete

        # Safety check stage (80-90%)
        if process_state in (
            HordeProcessState.SAFETY_STARTING,
            HordeProcessState.SAFETY_EVALUATING,
        ):
            return 85  # Mid-point of safety check

        if process_state == HordeProcessState.SAFETY_COMPLETE:
            return 90  # Safety check complete

        # Submission stage (90-100%)
        if process_state == HordeProcessState.RESULT_SAVING:
            return 92  # Saving images
        if process_state == HordeProcessState.RESULT_SAVED:
            return 95  # Images saved
        if process_state == HordeProcessState.RESULT_SUBMITTING:
            return 97  # Submitting to API
        if process_state == HordeProcessState.RESULT_SUBMITTED:
            return 100  # Submission complete

        # Failed states - show progress at the stage where failure occurred
        if process_state == HordeProcessState.INFERENCE_FAILED:
            # Failed during inference, show whatever progress was made
            if inference_progress is not None:
                return max(1, int(inference_progress * 0.7))
            return 1
        if process_state == HordeProcessState.SAFETY_FAILED:
            return 85  # Failed during safety check

        # Default fallback
        if inference_progress is not None:
            return inference_progress
        return 0

    @staticmethod
    def _is_same_job(
        a: "ImageGenerateJobPopResponse | None",
        b: "ImageGenerateJobPopResponse | None",
    ) -> bool:
        """Return True if ``a`` and ``b`` refer to the same job, compared by stable ``id_``.

        Full pydantic value-equality on ``ImageGenerateJobPopResponse`` is unreliable for this
        purpose because the job object is mutated in place after it is dispatched (for example
        source images are downloaded into it by ``_get_source_images``).  As a result the job
        object stored in ``jobs_in_progress`` can differ *by value* from a process's
        ``last_job_referenced`` even though they are the same logical job — which made the WebUI
        fail to find the handling process and fall back to a generic ``"Processing"`` label
        instead of the real state (e.g. ``INFERENCE_PROCESSING``).  The horde ``JobID`` is stable
        across those mutations, so match on it.
        """
        if a is None or b is None:
            return False
        a_id = getattr(a, "id_", None)
        b_id = getattr(b, "id_", None)
        if a_id is not None and b_id is not None:
            return a_id == b_id
        return a is b

    def _build_current_job_dict(
        self,
        job: "ImageGenerateJobPopResponse",
        progress: int,
        state: str,
        *,
        state_elapsed_seconds: float | None = None,
        job_elapsed_seconds: float | None = None,
    ) -> dict:
        """Build the current-job dictionary used by the WebUI status payload.

        Args:
            job: The job whose details should be shown.
            progress: Overall progress percentage (0-100).
            state: Human-readable state label (e.g. ``"MODEL_PRELOADING"``).
            state_elapsed_seconds: Seconds elapsed since the handling process entered its current
                state, measured server-side. Allows the WebUI timer to be accurate regardless of
                when the page was loaded (it is not reset on page load). ``None`` when unknown.
            job_elapsed_seconds: Seconds elapsed since the job was popped, measured server-side.
                Drives the total-job timer in the WebUI. ``None`` when unknown.

        Returns:
            A JSON-serializable dictionary with job metadata.
        """
        return {
            "id": str(job.id_.root)[:8] if job.id_ else "N/A",
            "model": job.model,
            "progress": progress,
            "state": state,
            "state_elapsed_seconds": state_elapsed_seconds,
            "job_elapsed_seconds": job_elapsed_seconds,
            "is_complete": state == "INFERENCE_COMPLETE",
            "batch_size": job.payload.n_iter if job.payload else None,
            "steps": job.payload.ddim_steps if job.payload else None,
            "width": job.payload.width if job.payload else None,
            "height": job.payload.height if job.payload else None,
            "sampler": job.payload.sampler_name if job.payload else None,
            "loras": (
                __class__._serialize_loras_for_webui(job.payload.loras)
                if job.payload and job.payload.loras
                else None
            ),
        }

    @staticmethod
    def _serialize_loras_for_webui(loras: list | None) -> list[dict] | None:
        """Convert LorasPayloadEntry objects to JSON-serializable dictionaries.

        Args:
            loras: List of LorasPayloadEntry objects or None

        Returns:
            List of dictionaries with lora properties, or None if input is None
        """
        if not loras:
            return None

        serialized_loras = []
        for lora in loras:
            serialized_loras.append(
                {
                    "name": lora.name if hasattr(lora, "name") else str(lora),
                    "model": lora.model if hasattr(lora, "model") else None,
                    "clip": lora.clip if hasattr(lora, "clip") else None,
                },
            )
        return serialized_loras

    def _process_label(self, process_id: int) -> str:
        """Return a human-readable label for a process slot, e.g. ``inference-0`` or ``safety-0``.

        The label embeds the process type and a per-type index computed by counting how many
        processes of the same type have a strictly lower ``process_id``.  Because ``process_id``
        values are stable (they are reused when a slot is restarted), the label is also stable
        across process replacements.

        Args:
            process_id: The internal slot ID of the process.

        Returns:
            A string of the form ``"<type>-<index>"``, e.g. ``"inference-0"``.
            Falls back to ``"process-<id>"`` when the process_id is not found in the map.
        """
        process_info = self._process_map.get(process_id)
        if process_info is None:
            return f"process-{process_id}"
        ptype = process_info.process_type.name.lower()
        type_index = sum(
            1
            for p in self._process_map.values()
            if p.process_type == process_info.process_type and p.process_id < process_id
        )
        return f"{ptype}-{type_index}"

    @staticmethod
    def _webui_process_display_label(process_type: str, type_index: int) -> str:
        """Return the user-facing process label used by the Web UI."""
        process_name = process_type.replace("_", " ").title()
        if process_type.upper() == "INFERENCE" and type_index == 0:
            return f"{process_name} 0"
        if type_index == 0:
            return process_name
        return f"{process_name} {type_index}"

    def update_webui_status(self) -> None:
        """Update the web UI with current worker status."""
        if self.webui is None:
            return

        # Get current job info
        # Priority order: jobs furthest along the pipeline are shown first so the progress
        # bar stays active (at 100%) for the current job until it is fully submitted before
        # resetting to 0% for the next generation.
        current_job = None
        _current_job_obj = None  # Job object being shown as current_job (to exclude from queue)
        # Server-side reference time. Used to derive how long the handling process has been in its
        # current state (state_entered_timestamp) and how long the job has been alive (time_popped)
        # so the WebUI timers reflect real elapsed time instead of resetting on page load.
        _now = time.time()
        if self.jobs_pending_submit:
            # Show a job that has passed safety check and is waiting to be submitted to the API.
            # Checked first so the progress bar stays at 100% until submission completes, even
            # if a new inference job has already been dispatched and sits at 0%.
            try:
                job_info = self.jobs_pending_submit[0]
                job = job_info.sdk_api_job_info
                # Find the process that handled this job so we can show its most recent state.
                # Default to a manager-centric label indicating that the job is queued for submission.
                state = "RESULT_PENDING_SUBMIT"
                state_elapsed = None
                for process in self._process_map.values():
                    if self._is_same_job(process.last_job_referenced, job):
                        # Use the actual last process state name if available.
                        if process.last_process_state is not None:
                            state = process.last_process_state.name
                        state_elapsed = _now - process.state_entered_timestamp
                        break
                current_job = self._build_current_job_dict(
                    job,
                    100,
                    state,
                    state_elapsed_seconds=state_elapsed,
                    job_elapsed_seconds=_now - job_info.time_popped,
                )
            except (IndexError, AttributeError):
                # Pending submit list may have been modified, ignore and show no current job
                pass
        elif self.jobs_being_safety_checked:
            # Show job currently being safety checked – inference is complete, progress stays at 100%
            try:
                job_info = self.jobs_being_safety_checked[0]
                job = job_info.sdk_api_job_info
                state = "SAFETY_EVALUATING"
                state_elapsed = None
                for process in self._process_map.values():
                    if process.last_process_state in (
                        HordeProcessState.SAFETY_STARTING,
                        HordeProcessState.SAFETY_EVALUATING,
                        HordeProcessState.SAFETY_COMPLETE,
                    ):
                        state = process.last_process_state.name
                        state_elapsed = _now - process.state_entered_timestamp
                        break

                current_job = self._build_current_job_dict(
                    job,
                    100,
                    state,
                    state_elapsed_seconds=state_elapsed,
                    job_elapsed_seconds=_now - job_info.time_popped,
                )
            except (IndexError, AttributeError):
                # Safety check list may have been modified, ignore and show no current job
                pass
        elif self.jobs_pending_safety_check:
            # Show recently completed job awaiting safety check – inference is complete, progress stays at 100%
            try:
                job_info = self.jobs_pending_safety_check[0]
                job = job_info.sdk_api_job_info
                # Look up the actual process state for accurate display; fall back to
                # INFERENCE_COMPLETE since the job has not yet entered safety evaluation.
                state = "INFERENCE_COMPLETE"
                state_elapsed = None
                for process in self._process_map.values():
                    if (
                        self._is_same_job(process.last_job_referenced, job)
                        and process.last_process_state in self._WEBUI_POST_INFERENCE_STATES
                    ):
                        state = process.last_process_state.name
                        state_elapsed = _now - process.state_entered_timestamp
                        break
                current_job = self._build_current_job_dict(
                    job,
                    100,
                    state,
                    state_elapsed_seconds=state_elapsed,
                    job_elapsed_seconds=_now - job_info.time_popped,
                )
            except (IndexError, AttributeError):
                # Safety check list may have been modified, ignore and show no current job
                pass
        elif len(self.jobs_in_progress) > 0:
            job = self.jobs_in_progress[0]
            job_info = self.jobs_lookup.get(job)
            if job_info:
                # Find the process handling this job. Match by stable job id (see _is_same_job):
                # the job object is mutated in place after dispatch, so value-equality against
                # last_job_referenced can spuriously fail and leave the UI showing "Processing"
                # instead of the real INFERENCE_PROCESSING/etc. state.
                progress = None
                state = None
                state_elapsed = None
                for process in self._process_map.values():
                    if self._is_same_job(process.last_job_referenced, job):
                        process_state = process.last_process_state
                        state = process_state.name if process_state else None
                        state_elapsed = _now - process.state_entered_timestamp
                        # After inference completes pin progress at 100 % so the bar never
                        # goes backwards during post-processing, safety or submission.
                        if process_state in self._WEBUI_POST_INFERENCE_STATES:
                            progress = 100
                        elif process_state in (
                            HordeProcessState.INFERENCE_STARTING,
                            HordeProcessState.INFERENCE_PROCESSING,
                        ):
                            # Use the higher of the raw step-based percent and the granular
                            # stage-based progress so that the bar never shows 0% at the very
                            # start of a new inference.  The granular mapping returns at least
                            # 1% for INFERENCE_STARTING / early INFERENCE_PROCESSING, which
                            # prevents the jarring 100% → 0% → 100% jump when one job finishes
                            # submission and the next job just begins.  For mid/late inference
                            # the raw step value is the larger number and takes over, so actual
                            # step-level progress is still shown.
                            raw_progress = process.last_heartbeat_percent_complete
                            granular = self._calculate_granular_progress(
                                process_state, raw_progress
                            )
                            progress = (
                                max(granular, raw_progress) if raw_progress is not None else granular
                            )
                        else:
                            progress = process.last_heartbeat_percent_complete
                        break

                current_job = self._build_current_job_dict(
                    job,
                    progress if progress is not None else 0,
                    state or "Processing",
                    state_elapsed_seconds=state_elapsed,
                    job_elapsed_seconds=_now - job_info.time_popped,
                )
        else:
            # No job is actively in inference yet, but a process may be preloading a model
            # or downloading auxiliary models for an upcoming job. Show that process/job so
            # the UI is not blank during the model-loading or aux-download phase.
            for process in self._process_map.values():
                if process.last_process_state in (
                    HordeProcessState.MODEL_PRELOADING,
                    HordeProcessState.MODEL_PRELOADED,
                    HordeProcessState.DOWNLOADING_AUX_MODEL,
                ) and process.last_job_referenced is not None:
                    job = process.last_job_referenced
                    _current_job_obj = job
                    process_state = process.last_process_state
                    progress = self._calculate_granular_progress(process_state, None)
                    _preload_job_info = self.jobs_lookup.get(job)
                    current_job = self._build_current_job_dict(
                        job,
                        progress,
                        process_state.name,
                        state_elapsed_seconds=_now - process.state_entered_timestamp,
                        job_elapsed_seconds=(
                            _now - _preload_job_info.time_popped if _preload_job_info else None
                        ),
                    )
                    break

        # Get job queue (exclude jobs that are currently in progress or already shown as
        # current_job, e.g. a MODEL_PRELOADING job that is pending inference but not yet
        # dispatched to an active worker).
        job_queue = []
        for job in list(self.jobs_pending_inference):
            # Skip jobs that are already in progress or shown as current_job
            if job not in self.jobs_in_progress and job != _current_job_obj:
                job_queue.append(
                    {
                        "id": str(job.id_.root)[:8] if job.id_ else "N/A",
                        "model": job.model,
                        "batch_size": job.payload.n_iter if job.payload else None,
                        "popped_at": self.job_pop_timestamps.get(job),
                    },
                )

        # Get process info with stable per-type IDs (inference-0, safety-0, etc).
        # Processes that are ending/ended are excluded from the list/count, but their slots
        # still influence the per-type index so labels remain stable by process_id.
        processes = []
        for process_info in self._process_map.values():
            if process_info.last_process_state in (
                HordeProcessState.PROCESS_ENDING,
                HordeProcessState.PROCESS_ENDED,
            ):
                continue
            ptype = process_info.process_type.name.lower()
            type_index = sum(
                1
                for p in self._process_map.values()
                if p.process_type.name.lower() == ptype and p.process_id < process_info.process_id
            )
            # For safety processes actively evaluating, show the safety model name
            # instead of None so the UI displays it rather than "Idle".
            model_name = process_info.loaded_horde_model_name
            if (
                model_name is None
                and process_info.process_type == HordeProcessType.SAFETY
                and process_info.last_process_state
                in (
                    HordeProcessState.SAFETY_STARTING,
                    HordeProcessState.SAFETY_EVALUATING,
                    HordeProcessState.SAFETY_COMPLETE,
                )
            ):
                model_name = "CLIP / DeepDanbooru"
            process_job_id = (
                str(process_info.last_job_referenced.id_.root)[:8]
                if (
                    process_info.last_job_referenced is not None
                    and process_info.last_job_referenced.id_ is not None
                )
                else None
            )

            processes.append(
                {
                    "id": f"{ptype}-{type_index}",
                    "display_id": self._webui_process_display_label(
                        process_info.process_type.name,
                        type_index,
                    ),
                    "type": process_info.process_type.name,
                    "state": process_info.last_process_state.name,
                    "model": model_name,
                    "job_id": process_job_id,
                    "progress": process_info.last_heartbeat_percent_complete,
                    "batch_size": process_info.batch_amount,
                },
            )

        # Get loaded models
        models_loaded = list(
            {
                process.loaded_horde_model_name
                for process in self._process_map.values()
                if process.loaded_horde_model_name is not None
            },
        )

        # Calculate total resource usage
        total_ram_mb = sum(p.ram_usage_bytes for p in self._process_map.values()) / BYTES_TO_MEGABYTES
        # Only report worker VRAM from processes that have a model loaded. After a model
        # unloads, PyTorch keeps its VRAM cache warm so vram_usage_bytes stays high even
        # though no model is actually loaded. When all models are unloaded the worker VRAM
        # bar should show 0 to avoid misleading the user.
        loaded_proc_vram = [
            p.vram_usage_bytes
            for p in self._process_map.values()
            if p.loaded_horde_model_name is not None
        ]
        total_vram_mb = max(loaded_proc_vram, default=0) / BYTES_TO_MEGABYTES

        # Total system RAM capacity
        total_system_ram_mb = self.total_ram_bytes / BYTES_TO_MEGABYTES

        # System-wide RAM currently in use (all processes on the host, not just the worker)
        system_ram_usage_mb = psutil.virtual_memory().used / BYTES_TO_MEGABYTES

        # Get total VRAM from all devices
        total_device_vram_mb = 0
        if len(self._device_map.root) > 0:
            total_device_vram_mb = (
                sum(device.total_memory for device in self._device_map.root.values()) / BYTES_TO_MEGABYTES
            )

        # Get CPU usage percentage
        cpu_usage_percent = psutil.cpu_percent(interval=0.1)

        # Get CPU cores count (logical cores/threads)
        cpu_cores_count = psutil.cpu_count(logical=True) or 0

        # Get container CPU usage (this process + all spawned subprocesses), normalised to
        # a 0-100% scale representing the fraction of total CPU capacity consumed.
        container_cpu_percent = 0.0
        try:
            main_proc = self._main_process
            tracked_processes: dict[int, psutil.Process] = {main_proc.pid: main_proc}

            for child in main_proc.children(recursive=True):
                child_pid = child.pid
                tracked_child = self._container_cpu_processes.get(child_pid)
                if tracked_child is None:
                    # Prime first-sample state for newly discovered children. Per psutil docs,
                    # cpu_percent(interval=None) returns 0.0 on first call because there is no
                    # previous measurement for comparison; this warm-up enables meaningful deltas
                    # from the next update onward while keeping status updates non-blocking.
                    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        child.cpu_percent(interval=None)
                    tracked_child = child
                tracked_processes[child_pid] = tracked_child

            self._container_cpu_processes = tracked_processes

            raw_cpu = 0.0
            for process in tracked_processes.values():
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    raw_cpu += process.cpu_percent(interval=None)
            # psutil process cpu_percent can exceed 100% on multi-core machines because it
            # accumulates usage across each logical core independently.  Dividing by the
            # logical core count converts that sum to a percentage of total system CPU
            # capacity (e.g. 200% on a 4-core machine → 50% of total capacity).
            container_cpu_percent = min(100.0, round(raw_cpu / (cpu_cores_count or 1), 1))
        except Exception as e:
            logger.debug(f"CPU metrics collection failed: {type(e).__name__}: {e}")

        # Get GPU utilization percentage
        gpu_usage_percent = 0.0
        system_vram_usage_mb = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                # Average utilization across all GPUs
                total_util = 0.0
                device_count = torch.cuda.device_count()
                total_system_vram_used = 0.0
                for i in range(device_count):
                    # Get GPU utilization using nvidia-smi via torch
                    # Note: torch.cuda.utilization() returns GPU utilization percentage
                    with contextlib.suppress(RuntimeError, ValueError):
                        total_util += torch.cuda.utilization(i)
                    # system-wide VRAM in use for each device (total capacity minus free)
                    with contextlib.suppress(RuntimeError, ValueError):
                        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                        total_system_vram_used += (total_bytes - free_bytes) / BYTES_TO_MEGABYTES
                if device_count > 0 and total_util > 0:
                    gpu_usage_percent = total_util / device_count
                system_vram_usage_mb = total_system_vram_used
        except ImportError:
            # torch is not available; GPU usage will remain 0
            pass
        except Exception:
            # CUDA is not available or errored; GPU usage will remain 0
            logger.debug("GPU metrics collection failed", exc_info=True)

        # Aggregate per-process GPU utilisation reported by inference workers.
        # Uses max() so that the highest-utilised device is represented when multiple
        # processes are running on different GPUs.
        worker_gpu_percent = 0.0
        try:
            worker_gpu_percent = max(
                (float(p.gpu_usage_percent) for p in self._process_map.values()),
                default=0.0,
            )
        except (TypeError, ValueError):
            pass

        # Get total GPU cores count across all devices.
        # For NVIDIA GPUs the CUDA core count is derived from the SM count and the
        # compute-capability-specific cores-per-SM lookup table (_CUDA_CORES_PER_SM).
        # Devices with unknown compute capability are skipped to avoid mixing CUDA-core
        # and SM-count units; if all devices are unknown, we leave the webui value
        # unchanged by passing None.
        gpu_cores_count: int | None = None
        try:
            import torch

            if torch.cuda.is_available():
                computed_gpu_cores_count = 0
                has_known_cuda_core_count = False
                for i in range(torch.cuda.device_count()):
                    with contextlib.suppress(RuntimeError, ValueError):
                        props = torch.cuda.get_device_properties(i)
                        cores_per_sm = _get_cuda_cores_per_sm(props.major, props.minor)
                        if cores_per_sm is None:
                            continue
                        computed_gpu_cores_count += props.multi_processor_count * cores_per_sm
                        has_known_cuda_core_count = True
                if has_known_cuda_core_count:
                    gpu_cores_count = computed_gpu_cores_count
        except ImportError:
            pass
        except Exception:
            logger.debug("GPU cores count collection failed", exc_info=True)

        # Calculate kudos per hour and images per hour over the rolling window
        now = time.time()
        cutoff = now - METRICS_CALCULATION_WINDOW_SECONDS

        # Prune and sum kudos events
        kudos_per_hour = 0.0
        if len(self.kudos_events) > 0:
            self.kudos_events = [(ts, k) for ts, k in self.kudos_events if ts >= cutoff]
            kudos_per_hour = sum(k for _, k in self.kudos_events)

        # Prune and sum image events
        images_per_hour = 0.0
        if len(self.image_events) > 0:
            self.image_events = [(ts, c) for ts, c in self.image_events if ts >= cutoff]
            images_per_hour = sum(c for _, c in self.image_events)

        # Get user kudos total and username
        user_kudos_total = None
        horde_username = None
        user_details: dict[str, Any] = {}
        if self.user_info:
            horde_username = self.user_info.username
            if self.user_info.kudos_details:
                user_kudos_total = self.user_info.kudos_details.accumulated
                kd = self.user_info.kudos_details
                kudos_details_dict: dict[str, Any] = {}
                for field in ("accumulated", "gifted", "admin", "received", "donated", "recurring"):
                    val = getattr(kd, field, None)
                    if val is not None:
                        try:
                            kudos_details_dict[field] = float(val)
                        except (TypeError, ValueError):
                            pass
                if kudos_details_dict:
                    user_details["kudos_details"] = kudos_details_dict
            for field in ("worker_count", "trusted", "moderator", "pseudonymous", "concurrency"):
                val = getattr(self.user_info, field, None)
                if val is not None:
                    user_details[field] = val
            worker_ids = getattr(self.user_info, "worker_ids", None)
            if worker_ids:
                user_details["worker_ids"] = [str(wid) for wid in worker_ids]
        if self._workers_details:
            workers_cache_key = json.dumps(self._workers_details, sort_keys=True, separators=(",", ":"), default=str)
            if workers_cache_key != self._last_sent_workers_details_cache_key:
                user_details["workers_list"] = self._workers_details
                self._last_sent_workers_details_cache_key = workers_cache_key

        # Update the web UI
        # Compute time_without_jobs dynamically so the webui counter increments
        # continuously between job-pop cycles, and correctly starts from session start.
        current_time_without_jobs = self._time_spent_no_jobs_available
        if self._last_pop_no_jobs_available_time > 0:
            current_time_without_jobs += time.time() - self._last_pop_no_jobs_available_time

        # Cache VRAM metrics for use by auto-tuning computations.
        self._last_total_vram_mb = total_device_vram_mb
        self._last_system_vram_usage_mb = system_vram_usage_mb
        self._last_worker_vram_mb = total_vram_mb

        # Apply auto-tuning overrides when the respective modes are active.
        if self._queue_size_auto:
            auto_qs = self._compute_auto_queue_size()
            if auto_qs != self._queue_size_override:
                self._queue_size_override = auto_qs
                logger.debug(f"Auto queue size updated to {auto_qs}")

        if self._max_active_models_auto:
            auto_ma = self._compute_auto_max_active_models()
            if auto_ma != self._max_active_models_override:
                self._max_active_models_override = auto_ma
                self.bridge_data.max_active_models = auto_ma
                os.environ["AIWORKER_MAX_ACTIVE_MODELS"] = str(auto_ma)
                self._lru.capacity = auto_ma
                self._inference_scale_down_requested = True
                logger.debug(f"Auto max active models updated to {auto_ma}")

        self.webui.update_status(
            worker_name=self.bridge_data.dreamer_worker_name,
            horde_username=horde_username,
            jobs_popped=self.total_num_jobs_queued,
            jobs_queued=max(0, len(self.jobs_pending_inference) - len(self.jobs_in_progress)),
            time_without_jobs=current_time_without_jobs,
            jobs_completed=self.total_num_completed_jobs,
            jobs_faulted=self._num_jobs_faulted,
            processes_recovered=self._num_process_recoveries,
            kudos_earned_session=self.kudos_generated_this_session,
            kudos_per_hour=kudos_per_hour,
            images_per_hour=images_per_hour,
            current_job=current_job,
            job_queue=job_queue,
            max_queue_size=self.max_queue_size,
            queue_size_auto=self._queue_size_auto,
            processes=processes,
            models_loaded=models_loaded,
            max_active_models=self.max_inference_processes,
            max_active_models_auto=self._max_active_models_auto,
            ram_usage_mb=total_ram_mb,
            total_ram_mb=total_system_ram_mb,
            system_ram_usage_mb=system_ram_usage_mb,
            vram_usage_mb=total_vram_mb,
            total_vram_mb=total_device_vram_mb,
            cpu_usage_percent=cpu_usage_percent,
            cpu_cores_count=cpu_cores_count,
            gpu_usage_percent=gpu_usage_percent,
            worker_gpu_percent=worker_gpu_percent,
            gpu_cores_count=gpu_cores_count,
            system_vram_usage_mb=system_vram_usage_mb,
            container_cpu_percent=container_cpu_percent,
            maintenance_mode=self._last_pop_maintenance_mode,
            user_kudos_total=user_kudos_total,
            last_image_base64=self._last_image_base64,
            last_image_submission_timestamp=self._last_image_job_timestamp,
            last_image_model=self._last_image_model,
            last_image_safety=self._last_image_safety,
            console_logs=list(self._console_logs)[-self._WEBUI_CONSOLE_LOGS_LIMIT:] if self._console_logs else [],
            faulted_jobs_history=self._faulted_jobs_history,
            errors_history=(
                list(self._errors_history)
                if len(self._errors_history) != self._errors_history_last_sent_len
                else None
            ),
            user_details=user_details if user_details else None,
            job_pops_paused=self._job_pops_paused,
            job_pops_pause_until=self._job_pops_pause_until,
            images_per_model=self._images_per_model,
            failed_jobs_per_model=self._failed_models,
            faulted_jobs_per_phase=self._faulted_jobs_per_phase,
            avg_time_per_job_state={
                k: round(v["sum"] / v["count"], 2) for k, v in self._job_time_stats.items() if v["count"] > 0
            },
            max_time_per_job_state={k: round(v["max"], 2) for k, v in self._job_time_stats.items()},
            avg_time_per_step_per_model={
                k: round(v["sum"] / v["count"], 3) for k, v in self._time_per_step_per_model.items() if v["count"] > 0
            },
            max_time_per_step_per_model={k: round(v["max"], 3) for k, v in self._time_per_step_per_model.items()},
            avg_time_per_job_per_model={
                k: round(v["sum"] / v["count"], 2) for k, v in self._time_per_job_per_model.items() if v["count"] > 0
            },
            max_time_per_job_per_model={k: round(v["max"], 2) for k, v in self._time_per_job_per_model.items()},
        )
        self._errors_history_last_sent_len = len(self._errors_history)

        # Push current settings snapshot so the Settings page reflects live values.
        self.webui.update_settings_data(self._get_settings_snapshot())

        # Push current model enabled/disabled lists so the Models section reflects live state.
        enabled_models = list(self.bridge_data.image_models_to_load)
        disabled_models = [m for m in self._all_models_configured if m in self._runtime_disabled_models]
        self.webui.update_models_data(enabled_models, disabled_models)

    def _handle_exception(self, future: asyncio.Future) -> None:
        """Logs exceptions from asyncio tasks.

        :param future: asyncio task to monitor
        :return: None
        """
        # Check if future was cancelled before attempting to get exception
        if future.cancelled():
            logger.debug("A main loop task was cancelled")
            return

        # Even after checking cancelled(), future.exception() can still raise CancelledError
        # in some edge cases, so we wrap it in a try-except for defense-in-depth
        try:
            ex = future.exception()
        except CancelledError:
            logger.debug("A main loop task was cancelled (CancelledError)")
            return

        if ex is not None:
            if self._shutting_down:
                logger.debug(f"exception thrown by a main loop task: {ex}")
            else:
                logger.error(f"exception thrown by a main loop task: {ex}")
                logger.exception(ex)

    async def _webui_update_loop(self) -> None:
        """Update the web UI periodically with current worker status."""
        while True:
            try:
                if self._shutting_down:
                    break

                loop = asyncio.get_running_loop()
                loop_start = loop.time()
                self.update_webui_status()
                elapsed = loop.time() - loop_start
                sleep_duration = max(0.0, self.bridge_data.webui_update_interval - elapsed)
                await asyncio.sleep(sleep_duration)
            except CancelledError:
                self._shutdown()
                break
            except Exception as e:
                logger.error(f"Error in webui update loop: {e}", exc_info=True)
                await asyncio.sleep(self.bridge_data.webui_update_interval)

    async def _main_loop(self) -> None:
        process_control_loop = asyncio.create_task(self._process_control_loop(), name="process_control_loop")
        process_control_loop.add_done_callback(self._handle_exception)

        api_call_loop = asyncio.create_task(self._api_call_loop(), name="api_call_loop")
        api_call_loop.add_done_callback(self._handle_exception)

        api_get_user_info_loop = asyncio.create_task(self._api_get_user_info_loop(), name="api_get_user_info_loop")
        api_get_user_info_loop.add_done_callback(self._handle_exception)

        api_get_workers_details_loop = asyncio.create_task(
            self._api_get_workers_details_loop(), name="api_get_workers_details_loop"
        )
        api_get_workers_details_loop.add_done_callback(self._handle_exception)

        job_submit_loop = asyncio.create_task(self._job_submit_loop(), name="job_submit_loop")
        job_submit_loop.add_done_callback(self._handle_exception)

        bridge_data_loop = None
        if not self.bridge_data._loaded_from_env_vars:
            bridge_data_loop = asyncio.create_task(self._bridge_data_loop(), name="bridge_data_loop")
            bridge_data_loop.add_done_callback(self._handle_exception)

        # Start web UI if enabled. A failure here (most commonly the port already being
        # bound by a stale instance or another service) must not take down the worker —
        # the web UI is an optional monitoring convenience, so degrade to running headless.
        webui_update_loop = None
        if self.webui is not None:
            try:
                await self.webui.start()
            except Exception as e:
                logger.error(
                    f"Web UI failed to start ({type(e).__name__}: {e}). "
                    "Continuing without the web UI — the worker will keep generating.",
                )
                try:
                    await self.webui.stop()
                except Exception:
                    logger.debug("Failed to clean up web UI after a failed start", exc_info=True)
                self.webui = None
            else:
                webui_update_loop = asyncio.create_task(self._webui_update_loop(), name="webui_update_loop")
                webui_update_loop.add_done_callback(self._handle_exception)

        tasks = [process_control_loop, api_call_loop, api_get_user_info_loop, api_get_workers_details_loop, job_submit_loop]

        if bridge_data_loop is not None:
            tasks.append(bridge_data_loop)

        if webui_update_loop is not None:
            tasks.append(webui_update_loop)

        auto_restart_idle_loop = asyncio.create_task(self._auto_restart_idle_loop(), name="auto_restart_idle_loop")
        auto_restart_idle_loop.add_done_callback(self._handle_exception)
        tasks.append(auto_restart_idle_loop)

        # horde_sdk's async client passes no per-request timeout, so the session default
        # is the only bound on every API call made through it (job pops, submits, user
        # info, source-image downloads). Without an explicit value a wedged gateway can
        # hold requests open for many minutes (observed: ~8 minutes of total silence).
        # total=180s comfortably covers the biggest legitimate transfers (multi-MB job
        # submits and source-image downloads on slow links); requests that need a
        # different bound set their own per-request timeout, which overrides this.
        self._aiohttp_client_session = ClientSession(
            requote_redirect_url=False,
            timeout=aiohttp.ClientTimeout(total=180, connect=30, sock_connect=30),
        )
        self.horde_client_session = AIHordeAPIAsyncClientSession(
            aiohttp_session=self._aiohttp_client_session,
            apikey=self.bridge_data.api_key,
        )

        try:
            async with self._aiohttp_client_session, self.horde_client_session:
                await asyncio.gather(*tasks)
        finally:
            # Stop web UI when shutting down
            if self.webui is not None:
                await self.webui.stop()

    _caught_sigints = 0
    """The number of SIGINTs or SIGTERMs caught."""
    _restart_requested = False
    """If true, restart the current Python process after shutdown completes."""

    def _cleanup_shared_resources(self) -> None:
        """Explicitly release named POSIX semaphores before os.execv replaces the process.

        ``os.execv`` replaces the current process image without running Python's atexit
        handlers or ``__del__`` / ``util.Finalize`` callbacks.  Semaphores created with
        the ``spawn`` multiprocessing context have named POSIX backing files in
        ``/dev/shm``; without an explicit cleanup these names remain registered in the
        shared resource-tracker subprocess, which then warns at shutdown:

            resource_tracker: There appear to be N leaked semaphore objects to clean up

        This method calls ``SemLock._cleanup(name)`` (``sem_unlink`` +
        ``resource_tracker.unregister``) on each named semaphore / lock owned by this
        process-manager instance, prevents the implicit ``join_thread()`` call on the
        queue feeder thread at process exit (which would otherwise block), and closes any
        open pipe connections so the process-map entries are properly released.
        """
        from multiprocessing.resource_tracker import unregister as _rt_unregister
        from multiprocessing.synchronize import SemLock

        def _try_cleanup_sem(sem_or_lock: Any) -> None:
            try:
                name = sem_or_lock._semlock.name
            except Exception as exc:
                logger.debug("Failed to read semaphore name for %r: %s", sem_or_lock, exc)
                return
            if name is None:
                return
            try:
                # SemLock._cleanup() does sem_unlink(name) followed by resource_tracker.unregister().
                # sem_unlink() can raise (e.g. FileNotFoundError) if the name was already unlinked by
                # a child process's own finalizer racing with this cleanup -- that must not skip the
                # unregister() call below, or the tracker's cache entry survives and it warns about a
                # "leaked" semaphore at shutdown even though the underlying resource is already gone.
                SemLock._cleanup(name)
            except Exception as exc:
                logger.debug("sem_unlink failed for %r (already unlinked?): %s", sem_or_lock, exc)
                try:
                    _rt_unregister(name, "semaphore")
                except Exception as exc2:
                    logger.debug("Failed to unregister semaphore %r: %s", sem_or_lock, exc2)

        _try_cleanup_sem(self._inference_semaphore)
        _try_cleanup_sem(self._vae_decode_semaphore)
        _try_cleanup_sem(self._aux_model_lock)
        _try_cleanup_sem(self._disk_lock)

        # The process message queue also holds named semaphores (_sem, _rlock, _wlock)
        # when the spawn start method is active.  Calling cancel_join_thread() prevents
        # the implicit join_thread() at process exit from blocking; it does not stop or
        # cancel the feeder thread itself.
        q = self._process_message_queue
        try:
            q.cancel_join_thread()
            q.close()
        except Exception:
            logger.debug("Failed to close process message queue during cleanup", exc_info=True)
        for _attr in ("_sem", "_rlock", "_wlock"):
            inner = getattr(q, _attr, None)
            if inner is not None:
                _try_cleanup_sem(inner)

        # Close the parent-side pipe connections so child processes see EOF.
        for _process_info in self._process_map.values():
            try:
                _process_info.pipe_connection.close()
            except Exception:
                logger.debug(
                    f"Failed to close pipe connection for process {_process_info.process_id} during cleanup",
                    exc_info=True,
                )

    def start(self) -> None:
        """Start the process manager."""
        import signal

        signal.signal(signal.SIGINT, self.signal_handler)
        asyncio.run(self._main_loop())
        if self._restart_requested:
            logger.warning("Restarting worker program...")
            # Flush log buffers BEFORE exec/exit: os.execv() replaces the process image without
            # running atexit handlers or draining enqueued loguru sinks, so otherwise the
            # "Restarting..." line (and any other buffered output) is silently lost.
            self._flush_logs_for_exit()
            self._cleanup_shared_resources()

            # os.execv() performs a true in-place exec on POSIX (same pid; the launching shell
            # keeps waiting on the process), but it CANNOT replace the process on Windows: there it
            # spawns a NEW pid and terminates the current one, so the launching cmd.exe/batch regains
            # control, falls through to its `pause` ("Press ANY KEY to close this window"), and the
            # re-exec'd worker is left orphaned. On Windows we therefore exit with a sentinel code
            # that the launch wrapper (horde-bridge*.cmd) loops on to re-run the worker cleanly.
            if sys.platform == "win32":
                sys.exit(WORKER_RESTART_EXIT_CODE)

            try:
                os.execv(sys.executable, [sys.executable, *sys.argv])
            except OSError as exc:
                logger.exception(f"Failed to restart worker program via execv: {exc}")
                sys.exit(1)

    @staticmethod
    def _flush_logs_for_exit() -> None:
        """Flush stdout/stderr and drain enqueued loguru sinks before an exec/exit that skips atexit."""
        try:
            logger.complete()  # drains messages from any sinks configured with enqueue=True
        except Exception:
            pass
        for _stream in (sys.stdout, sys.stderr):
            try:
                _stream.flush()
            except Exception:
                pass

    def request_program_restart(self) -> None:
        """Request a graceful shutdown followed by an in-process restart."""
        self._restart_requested = True
        # Reset the last job-pop timestamp so that end_inference_processes() is invoked
        # immediately on the next _process_control_loop iteration instead of waiting up
        # to 10 seconds for _last_pop_recently() to expire.
        self._last_job_pop_time = 0.0
        logger.warning("Worker program restart requested via web UI")
        self._shutdown()

    def _check_auto_restart_on_idle(self) -> None:
        """Restart the worker if no job has been submitted within the configured idle threshold.

        When ``bridge_data.auto_restart_on_idle_minutes`` is greater than zero and no job
        has been submitted since at least that many minutes, a graceful restart is triggered.
        Does nothing while already shutting down or while job pops are paused (intentional
        user pause).

        The "last submission" criterion is deliberately evaluated regardless of whether jobs
        are sitting in the local pipeline.  A genuinely-busy worker keeps submitting jobs and
        therefore continuously refreshes ``_last_job_submitted_time``; if that timestamp has
        not advanced for ``threshold_minutes``, the worker is wedged (for example every
        inference subprocess is stuck in ``PROCESS_STARTING`` with popped-but-never-processed
        jobs still in ``jobs_pending_inference``).  Skipping the restart in that situation —
        as previous versions did — left the worker effectively dead until manual intervention.
        """
        if self._shutting_down:
            return
        threshold_minutes = self.bridge_data.auto_restart_on_idle_minutes
        if not threshold_minutes:
            return
        # Don't restart while the user has intentionally paused job pops.
        if self._job_pops_paused:
            return
        now = time.time()
        elapsed_since_last_submit = now - self._last_job_submitted_time
        # Also consider how long the API has continuously reported "no jobs available".
        # This guards against edge-cases where last-submit tracking can be refreshed
        # without actual productive work, which would otherwise suppress idle restart.
        elapsed_since_last_no_jobs_pop = 0.0
        if self._last_pop_no_jobs_available_time > 0.0:
            elapsed_since_last_no_jobs_pop = now - self._last_pop_no_jobs_available_time
        elapsed_seconds = max(elapsed_since_last_submit, elapsed_since_last_no_jobs_pop)
        threshold_seconds = threshold_minutes * 60
        if elapsed_seconds >= threshold_seconds:
            logger.warning(
                f"No job has been submitted in the last {threshold_minutes} minute(s) "
                f"({elapsed_seconds / 60:.1f} min elapsed). Auto-restarting the worker program...",
            )
            self._restart_requested = True
            # Reset the last job-pop timestamp so that end_inference_processes() is invoked
            # immediately on the next _process_control_loop iteration instead of waiting up
            # to 10 seconds for _last_pop_recently() to expire.
            self._last_job_pop_time = 0.0
            # Reset the WebUI uptime pill immediately so the header shows near-zero during
            # the shutdown window rather than the accumulated uptime of the old session.
            if self.webui is not None:
                self.webui.reset_session_start_time()
            self._shutdown()

    async def _auto_restart_idle_loop(self) -> None:
        """Periodically check whether an idle-triggered auto-restart is required."""
        check_interval = 60.0
        sleep_step = 5.0
        seconds_until_check = check_interval
        while True:
            try:
                if self._shutting_down:
                    break
                sleep_for = min(sleep_step, seconds_until_check)
                await asyncio.sleep(sleep_for)
                if self._shutting_down:
                    break
                seconds_until_check -= sleep_for
                if seconds_until_check <= 0:
                    self._check_auto_restart_on_idle()
                    seconds_until_check = check_interval
            except CancelledError:
                self._shutdown()
                break
            except Exception:
                logger.exception("Unexpected error in auto-restart idle loop")

    def signal_handler(self, sig: int, frame: object) -> None:
        """Handle SIGINT and SIGTERM."""
        if self._caught_sigints >= 2:
            logger.warning("Caught SIGINT or SIGTERM three times, exiting immediately")
            self._start_timed_shutdown()
            sys.exit(1)

        self._caught_sigints += 1
        logger.warning("Shutting down after current jobs are finished...")
        self._shutdown()

        global _caught_signal
        _caught_signal = True

    def _start_timed_shutdown(self) -> None:
        import threading

        def hard_shutdown() -> None:
            # Just in case the process manager gets stuck on shutdown.
            # Cap the wait time to avoid unbounded hangs when many jobs are pending. The cap is
            # user-configurable via force_restart_timeout (primarily for auto-restart-on-idle, so a
            # stuck graceful shutdown does not delay the restart indefinitely).
            force_timeout = getattr(self.bridge_data, "force_restart_timeout", 30)
            if not isinstance(force_timeout, (int, float)):
                force_timeout = 30
            # Floor the wait at 15s (capped by force_timeout): ending and joining the inference /
            # safety subprocesses — and draining any in-progress generation — routinely takes more
            # than the old 2s minimum, so too short a wait force-killed the worker mid-shutdown,
            # which looked like a silent crash (the log just stops, no crash.log).
            wait_seconds = min(max((len(self.jobs_pending_submit) * 4) + 2, 15), force_timeout)
            time.sleep(wait_seconds)

            if self._shut_down or not self._shutting_down:
                return

            # If a restart was requested, exit with the sentinel code so the launch wrapper re-runs
            # the worker (otherwise a slow restart would die with code 1 and leave the worker down).
            exit_code = WORKER_RESTART_EXIT_CODE if self._restart_requested else 1

            # os._exit() bypasses logging, atexit handlers, and the crash-log excepthook — so a
            # force-kill here is otherwise completely silent (the log simply stops mid-stream with
            # no reason and no crash.log). Announce it and flush before exiting so the watchdog
            # firing is always visible in the log.
            logger.error(
                f"Graceful shutdown did not complete within {wait_seconds:.0f}s "
                f"(restart_requested={self._restart_requested}) — watchdog force-killing the worker "
                f"now with exit code {exit_code}.",
            )
            self._flush_logs_for_exit()

            # This runs on a daemon thread while the event-loop thread may still be mutating
            # _process_map (adding/popping entries or clearing it during the shutdown drain).
            # Snapshot the values with list(...) so iteration cannot raise "dictionary changed
            # size during iteration", and wrap the whole body so any unexpected error still falls
            # through to os._exit() — this is the last-resort force-kill and must never be
            # silently disabled.
            try:
                for process in list(self._process_map.values()):
                    try:
                        process.mp_process.kill()
                        process.mp_process.join(1)
                    except Exception as e:
                        logger.error(f"Failed to kill process {process}: {e}")
            except Exception:
                logger.exception("Unexpected error during hard shutdown; forcing exit anyway")

            # os._exit() skips atexit/Finalize callbacks just like os.execv() does, so named
            # semaphores/locks would otherwise be left registered with the resource tracker
            # (see _cleanup_shared_resources()'s docstring). Best-effort only: this is the
            # last-resort force-kill path and must still exit even if cleanup itself errors.
            try:
                self._cleanup_shared_resources()
            except Exception:
                logger.exception("Unexpected error cleaning up shared resources during hard shutdown")

            # Use os._exit instead of sys.exit because sys.exit only raises SystemExit
            # which, when called from a non-main thread, only terminates that thread.
            os._exit(exit_code)

        threading.Thread(target=hard_shutdown, daemon=True).start()

    _recently_recovered = False

    def _purge_jobs(self) -> None:
        """Clear all jobs immediately.

        Note: This is a last resort and should only be used when the worker is in a black hole and can't recover.
        Jobs will timeout on the server side and be requeued if they are still valid but due to the worker not
        responding, they will spend much longer in the queue than they should while the server waits for the worker
        to respond (and ultimately times out).
        """
        # Mark all jobs currently in progress as faulted before clearing them
        # Note: handle_job_fault may re-queue some jobs to jobs_pending_inference for retry
        if len(self.jobs_in_progress) > 0:
            for job in list(self.jobs_in_progress):
                self.handle_job_fault(faulted_job=job, process_info=None)
            logger.error("Cleared jobs in progress")

        # During shutdown, clear ALL pending jobs — there is no opportunity to retry them
        # and keeping them blocks the shutdown sequence indefinitely.
        # When not shutting down, keep jobs that have already been retried at least once
        # (retry_count > 0) so they get a second chance on a fresh process.
        if len(self.jobs_pending_inference) > 0:
            if self._shutting_down:
                self.jobs_pending_inference.clear()
                logger.warning("Cleared all jobs pending inference (shutdown in progress)")
            else:
                kept = []
                for job in self.jobs_pending_inference:
                    job_info = self.jobs_lookup.get(job)
                    if job_info is None:
                        logger.warning(
                            f"Job {job.id_} is in jobs_pending_inference but missing from jobs_lookup "
                            "(state inconsistency); dropping it during purge.",
                        )
                    elif job_info.retry_count > 0:
                        kept.append(job)
                self.jobs_pending_inference = deque(kept)

                # Log how many jobs were kept for retry
                jobs_kept_for_retry = len(self.jobs_pending_inference)
                if jobs_kept_for_retry > 0:
                    logger.warning(f"Cleared jobs pending inference (kept {jobs_kept_for_retry} job(s) for retry)")
                else:
                    logger.error("Cleared jobs pending inference")

        if len(self.jobs_being_safety_checked) > 0:
            self.jobs_being_safety_checked.clear()
            logger.error("Cleared jobs being safety checked")

        if len(self.jobs_pending_safety_check) > 0:
            self.jobs_pending_safety_check.clear()
            logger.error("Cleared jobs pending safety check")

        # Note: We do NOT clear jobs_lookup here because it contains job_info for jobs that were re-queued
        # for retry. If we cleared it, the retry_count would be lost and jobs would retry indefinitely.
        # Jobs are removed from jobs_lookup when they are completed or permanently faulted.

        if len(self.jobs_pending_submit) > 0:
            self.jobs_pending_submit.clear()
            logger.error("Cleared completed jobs")

        if self._skipped_line_next_job_and_process is not None:
            self._skipped_line_next_job_and_process = None
            logger.error("Cleared skipped line next job and process")

        # Drop bookkeeping for jobs that are no longer referenced anywhere. The clears above
        # intentionally keep jobs_lookup entries for jobs kept for retry (see note), but jobs
        # dropped from the safety/submit queues or not kept for retry would otherwise leak
        # here forever — including their base64 image results and any downloaded source
        # images, which adds up when purge-recovery happens repeatedly. Match by id_ (not
        # object equality) because filtered copies of a job can hash/compare differently
        # from the canonical object stored in these structures.
        referenced_job_ids = {j.id_ for j in self.jobs_pending_inference if j.id_ is not None} | {
            j.id_ for j in self.jobs_in_progress if j.id_ is not None
        }
        for job in [j for j in self.jobs_lookup if j.id_ not in referenced_job_ids]:
            del self.jobs_lookup[job]
        for job in [j for j in self.job_pop_timestamps if j.id_ not in referenced_job_ids]:
            del self.job_pop_timestamps[job]
        for job in [j for j in self._pending_completed_job_timings if j.id_ not in referenced_job_ids]:
            del self._pending_completed_job_timings[job]
        for job_id in [jid for jid in self.job_faults if jid not in referenced_job_ids]:
            del self.job_faults[job_id]

        self._invalidate_megapixelsteps_cache()

    def _hard_kill_processes(
        self,
        inference: bool = True,
        safety: bool = True,
        all_: bool = True,
    ) -> None:
        """Kill all processes immediately."""
        for process_info in self._process_map.values():
            if (
                (inference and process_info.process_type == HordeProcessType.INFERENCE)
                or (safety and process_info.process_type == HordeProcessType.SAFETY)
                or (all_)
            ):
                try:
                    process_info.mp_process.kill()
                    process_info.mp_process.join(1)
                except Exception as e:
                    logger.error(f"Failed to kill process {process_info}: {e}")

        self._process_map.clear()
        self._horde_model_map.root.clear()

    def _check_and_replace_process(
        self,
        process_info: HordeProcessInfo,
        timeout: float,
        state: HordeProcessState,
        error_message: str,
    ) -> bool:
        """Check if a process has been stuck in a state for too long and replace it if it has.

        Args:
            process_info (HordeProcessInfo): The process to check
            timeout (float): The time in seconds to wait before replacing the process
            state (HordeProcessState): The state to check for
            error_message (str): The error message to log if the process is replaced

        Returns:
            True if the process was replaced, False otherwise
        """
        now = time.time()
        time_elapsed = now - process_info.last_received_timestamp
        time_elapsed = min(time_elapsed, now - process_info.last_heartbeat_timestamp)

        if time_elapsed > timeout and process_info.last_process_state == state:
            logger.error(f"{process_info} {error_message}, replacing it")
            if process_info.process_type == HordeProcessType.SAFETY:
                # Only set the flag here — do NOT call _replace_all_safety_process() inside
                # the replace_hung_processes loop because it can call delete_safety_processes()
                # which pops from the dict we are iterating, causing RuntimeError.
                # The actual replacement is handled by the _replace_all_safety_process() call
                # in _process_control_loop() immediately after replace_hung_processes() returns.
                self._safety_processes_should_be_replaced = True
            if process_info.process_type == HordeProcessType.INFERENCE:
                self._replace_inference_process(process_info)
            return True
        return False

    def _prune_preload_stuck_failures(self, failures: deque, cutoff: float) -> None:
        """Remove failure timestamps older than *cutoff* from *failures* in-place."""
        while failures and failures[0] < cutoff:
            failures.popleft()

    def _record_preload_stuck_failure(self, model_name: str, timestamp: float) -> None:
        """Record a MODEL_PRELOADING timeout for *model_name* at *timestamp*.

        Prunes entries older than ``_PRELOAD_STUCK_FAILURE_WINDOW`` so the deque only
        holds recent failures.  Logs a warning when the model first crosses
        ``_PRELOAD_STUCK_FAILURE_THRESHOLD`` and enters the cooldown period.
        """
        failures = self._preload_stuck_failures.setdefault(model_name, deque())
        # Prune stale entries before adding the new one so the count reflects the window.
        self._prune_preload_stuck_failures(failures, timestamp - self._PRELOAD_STUCK_FAILURE_WINDOW)
        failures.append(timestamp)

        if len(failures) == self._PRELOAD_STUCK_FAILURE_THRESHOLD:
            logger.warning(
                f"Model {model_name!r} has failed to preload {self._PRELOAD_STUCK_FAILURE_THRESHOLD} "
                f"times within {self._PRELOAD_STUCK_FAILURE_WINDOW:.0f}s. "
                f"It will be suspended from preloading for {self._PRELOAD_STUCK_COOLDOWN:.0f}s "
                "to prevent a stuck-preloading loop.",
            )

    def _is_model_in_preload_cooldown(self, model_name: str) -> bool:
        """Return True if *model_name* has triggered the preload-stuck cooldown.

        A model is in cooldown when it has at least ``_PRELOAD_STUCK_FAILURE_THRESHOLD``
        recorded stuck-preloading events within the last ``_PRELOAD_STUCK_FAILURE_WINDOW``
        seconds **and** fewer than ``_PRELOAD_STUCK_COOLDOWN`` seconds have elapsed since
        the most recent failure.
        """
        failures = self._preload_stuck_failures.get(model_name)
        if not failures:
            return False
        now = time.time()
        # Prune stale entries before checking the count.
        self._prune_preload_stuck_failures(failures, now - self._PRELOAD_STUCK_FAILURE_WINDOW)
        if len(failures) < self._PRELOAD_STUCK_FAILURE_THRESHOLD:
            return False
        # Still within the cooldown window from the most recent failure.
        return (now - failures[-1]) < self._PRELOAD_STUCK_COOLDOWN

    def _record_inference_failure(self, model_name: str, timestamp: float) -> None:
        """Record a permanently-faulted inference job for *model_name* at *timestamp*.

        Prunes entries older than ``_INFERENCE_FAILURE_WINDOW`` so the deque only
        holds recent failures.  Logs a warning when the model first crosses
        ``_INFERENCE_FAILURE_THRESHOLD`` and enters the inference-failure cooldown.
        """
        failures = self._inference_failures.setdefault(model_name, deque())
        # Prune stale entries before adding the new one so the count reflects the window.
        self._prune_preload_stuck_failures(failures, timestamp - self._INFERENCE_FAILURE_WINDOW)
        failures.append(timestamp)

        if len(failures) == self._INFERENCE_FAILURE_THRESHOLD:
            logger.warning(
                f"Model {model_name!r} has caused {self._INFERENCE_FAILURE_THRESHOLD} permanently-faulted "
                f"jobs within {self._INFERENCE_FAILURE_WINDOW:.0f}s. "
                f"It will be excluded from job-pop requests for {self._INFERENCE_FAILURE_COOLDOWN:.0f}s "
                "to protect this worker from server penalties. "
                "Consider removing the model from your config if this persists.",
            )

    def _is_model_in_inference_cooldown(self, model_name: str) -> bool:
        """Return True if *model_name* has triggered the inference-failure cooldown.

        A model is in cooldown when it has at least ``_INFERENCE_FAILURE_THRESHOLD``
        permanently-faulted jobs within the last ``_INFERENCE_FAILURE_WINDOW`` seconds
        **and** fewer than ``_INFERENCE_FAILURE_COOLDOWN`` seconds have elapsed since
        the most recent failure.
        """
        failures = self._inference_failures.get(model_name)
        if not failures:
            return False
        now = time.time()
        # Prune stale entries before checking the count.
        self._prune_preload_stuck_failures(failures, now - self._INFERENCE_FAILURE_WINDOW)
        if len(failures) < self._INFERENCE_FAILURE_THRESHOLD:
            return False
        # Still within the cooldown window from the most recent failure.
        return (now - failures[-1]) < self._INFERENCE_FAILURE_COOLDOWN

    def _fault_cooldown_model_jobs(self) -> None:
        """Permanently fault all pending jobs whose models are in the preload cooldown.

        Called at the start of :meth:`preload_models` so that jobs the worker has already
        popped from the horde but cannot service (because the model keeps hanging on load)
        are reported back to the horde quickly rather than sitting in the local queue for
        the entire cooldown duration.

        The jobs are permanently faulted (retries bypassed) because re-trying them locally
        would just produce the same stuck-preloading loop.  The horde will re-assign them
        to another worker that may be able to load the model.
        """
        # Snapshot the pending queue before iterating: handle_job_fault modifies
        # jobs_pending_inference, and iterating over the snapshot avoids skipped items
        # or double-processing if the same object appeared more than once.
        jobs_in_cooldown = [
            job
            for job in self.jobs_pending_inference
            if job.model is not None and self._is_model_in_preload_cooldown(job.model)
        ]
        for job in jobs_in_cooldown:
            if job not in self.jobs_pending_inference:
                # Already removed by handle_job_fault from an earlier iteration (e.g. the
                # same job object appeared under two different queue entries, which is an
                # edge case but guarding here keeps the loop safe).
                continue
            logger.warning(
                f"Job {job.id_} for model {job.model!r} faulted immediately: "
                "model is in preload cooldown due to repeated hung-preloading failures",
            )
            job_info = self.jobs_lookup.get(job)
            if job_info is not None:
                # Force permanent fault: skip the normal retry so the job is not
                # re-queued locally, which would only be faulted again right away.
                job_info.retry_count = self.MAX_JOB_RETRIES
            else:
                # If the job metadata is missing, `handle_job_fault` may only log/history
                # the fault and leave the job stuck in the local pending queue. Remove the
                # queue entry here so cooldown enforcement always drains pending jobs.
                logger.warning(
                    f"Cooldown-faulting job {job.id_} for model {job.model!r} without "
                    "jobs_lookup metadata; removing it from jobs_pending_inference first",
                )
                with contextlib.suppress(ValueError):
                    self.jobs_pending_inference.remove(job)
            self.handle_job_fault(
                faulted_job=job,
                process_info=None,
                fault_info=(
                    f"model {job.model!r} is temporarily suspended from preloading after "
                    f"{self._PRELOAD_STUCK_FAILURE_THRESHOLD} recent hung-preload failures"
                ),
            )

    _shutting_down = False
    """If true, the worker is scheduled to shut down."""
    _shutting_down_time = 0.0
    """The epoch time of when the worker started shutting down."""
    _shut_down = False
    """If true, the worker is out of the process control loop and should halt."""
    _inference_scale_down_requested = False
    """Whether a non-blocking inference scale-down pass has been requested."""

    def _shutdown(self) -> None:
        if not self._shutting_down:
            self._shutting_down = True
            self._shutting_down_time = time.time()

            # Record which code path initiated the shutdown so the cause is always visible in the
            # log. Several paths call _shutdown() (signal, auto-restart-on-idle, web UI restart,
            # hung-process abort, task cancellation); without this the reason can be ambiguous.
            try:
                import traceback as _tb

                caller = _tb.extract_stack(limit=2)[0]
                logger.warning(
                    f"Shutdown initiated (restart_requested={self._restart_requested}) by "
                    f"{caller.name}() at {caller.filename}:{caller.lineno}",
                )
            except Exception:
                logger.warning(f"Shutdown initiated (restart_requested={self._restart_requested})")

            # Cleanup webui log handler
            if self._log_handler_id is not None:
                try:
                    logger.remove(self._log_handler_id)
                    self._log_handler_id = None
                except Exception as e:
                    logger.debug(f"Failed to remove log handler during shutdown: {e}")

    def _abort(self) -> None:
        """Exit as soon as possible, aborting all processes and jobs immediately."""
        with logger.catch(), open(".abort", "w") as f:
            f.write("")

        self._purge_jobs()

        self._shutdown()
        self._hard_kill_processes()
        self._start_timed_shutdown()

    _hung_processes_detected = False
    _hung_processes_detected_time = 0.0

    def replace_hung_processes(self) -> bool:
        """Replaces processes that haven't checked in since `process_timeout` seconds in bridgeData.

        The ``_recently_recovered`` guard is applied *selectively*:
        - INFERENCE_PROCESSING stuck detection is **always** evaluated regardless of the flag.
          A process holding the inference semaphore in INFERENCE_PROCESSING that stops responding
          blocks every other process from starting inference.  Newly-replaced processes start in
          PROCESS_STARTING (never INFERENCE_PROCESSING), so there is no false-positive risk from
          a recently-recovered slot.
        - The ``INFERENCE_STARTING`` **is_stuck_on_inference()** path is skipped while the flag is
          set, to prevent cascading replacements: a process blocked waiting to acquire the semaphore
          cannot send heartbeats and would falsely appear stuck immediately after a prior replacement
          freed the semaphore.  However, the ``INFERENCE_STARTING``-specific elapsed-time check
          (which measures time since the most recent job activity, based on the minimum of the
          last-received and last-heartbeat timestamps, rather than dispatch time alone) is
          **always** evaluated when no other process is in INFERENCE_PROCESSING; in that case the
          semaphore is available and any process stuck in INFERENCE_STARTING for longer than
          ``preload_timeout`` is genuinely hung regardless of how recently a recovery was performed.
        - ``INFERENCE_POST_PROCESSING``, ``POST_PROCESSING_STARTING``, ``MODEL_PRELOADING``,
          ``MODEL_PRELOADED``, ``DOWNLOADING_AUX_MODEL``, ``DOWNLOADING_MODEL``,
          ``JOB_RECEIVED``, and ``PROCESS_STARTING`` are **always** evaluated, so a process
          that is genuinely stuck in one of those states is recovered even when a different
          process was recently replaced.

        Job-related stuck states are checked in a multi-pass scan (condition-first, then
        processes) so that every process is examined for the highest-priority state before any
        process is examined for the next state.  This gives true cross-process prioritization:
        actively-in-progress states (``INFERENCE_POST_PROCESSING``, ``POST_PROCESSING_STARTING``,
        ``MODEL_PRELOADING``, ``DOWNLOADING_AUX_MODEL``, ``DOWNLOADING_MODEL``,
        ``JOB_RECEIVED``) are cleared across the entire worker before idle/finished states
        (``MODEL_PRELOADED``) are reclaimed, regardless of subprocess map ordering.
        """
        import threading

        def timed_unset_recently_recovered() -> None:
            time.sleep(self.bridge_data.inference_step_timeout)
            self._recently_recovered = False

        now = time.time()

        # Pre-compute once so the per-process INFERENCE_STARTING check is O(1) rather than O(n).
        any_active_inference_processing = any(
            p.last_process_state == HordeProcessState.INFERENCE_PROCESSING
            for p in self._process_map.values()
        )

        any_replaced = False
        # Reap in-progress jobs whose handling process died/was replaced/hung before completing
        # (e.g. stuck in INFERENCE_STARTING/INFERENCE_PROCESSING before the first step). This runs
        # every cycle, including during shutdown, so an orphaned job cannot block the inference
        # slot or keep is_time_for_shutdown() False forever. It is intentionally NOT counted as a
        # process replacement (it does not spawn a subprocess), so it must not start the
        # _recently_recovered cooldown.
        if self._reap_orphaned_in_progress_jobs():
            any_replaced = True
        # Tracks only actual subprocess replacements (via _replace_inference_process /
        # _check_and_replace_process).  Used exclusively for the _recently_recovered cooldown
        # so that a soft corrective action (e.g. resetting RESULT_SUBMITTING state) does not
        # start the cascading-recovery guard unnecessarily.
        any_process_replaced = False
        no_local_work = len(self.jobs_pending_inference) == 0 and len(self.jobs_in_progress) == 0
        # Pre-compute once so the per-process PROCESS_ENDING recovery check is O(1)
        # rather than O(n) (which would make the overall scan O(n^2)).
        num_loaded_inference = self._process_map.num_loaded_inference_processes()
        # Pre-compute once so the per-process scale-down guard is O(1) rather than O(n).
        num_total_inference = self._process_map.num_inference_processes()
        for process_info in self._process_map.values():
            # Quick dead-process detection: if the OS process is gone but the manager
            # still thinks it is alive, replace it immediately without waiting for the
            # heartbeat timeout (which could be up to 300 s for a process that held the
            # inference or VAE-decode semaphore).  Processes that sent PROCESS_ENDING or
            # PROCESS_ENDED are excluded — they exited intentionally and should not be
            # replaced here.
            if (
                not process_info.mp_process.is_alive()
                and process_info.last_process_state
                not in (HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED)
            ):
                logger.error(
                    f"{process_info} process has exited unexpectedly "
                    f"(state={process_info.last_process_state.name}, "
                    f"last heartbeat {time.time() - process_info.last_heartbeat_timestamp:.0f}s ago); "
                    "replacing immediately",
                )
                if process_info.process_type == HordeProcessType.INFERENCE:
                    # Don't respawn while shutting down: end_inference_processes()/hard_shutdown
                    # will finish tearing everything down, and a freshly spawned process would
                    # just be an unwanted new subprocess to kill (see _replace_inference_process's
                    # respawn docstring).
                    self._replace_inference_process(process_info, respawn=not self._shutting_down)
                    any_replaced = any_process_replaced = True
                elif process_info.process_type == HordeProcessType.SAFETY:
                    # Only set the flag — do NOT call _replace_all_safety_process() here.
                    # That method can call delete_safety_processes() → pop() on the dict we
                    # are currently iterating, causing RuntimeError.  The replacement runs via
                    # the _replace_all_safety_process() call in _process_control_loop() that
                    # immediately follows replace_hung_processes().
                    self._safety_processes_should_be_replaced = True
                    any_replaced = True
                continue

            # Determine whether this process appears stuck on inference.
            #
            # INFERENCE_PROCESSING: always check, no _recently_recovered guard.
            #   A process that holds the inference semaphore and stops responding blocks all
            #   other processes.  Newly-replaced processes start in PROCESS_STARTING, so there
            #   is no risk of a false-positive cascading replacement here.
            #   Pass a no_step_heartbeat_timeout so that a crash before any diffusion step is
            #   caught more quickly than the full inference_step_timeout.
            #   Also pass zero_progress_timeout (shorter) for the stuck-at-0%-progress case:
            #   when progress has been reported but never advanced from 0 % the VAE decode
            #   concern (which forces no_step_heartbeat_timeout to stay at 300 s) does not apply
            #   because VAE decode only happens at 100 % progress.  Using a shorter timeout here
            #   cuts stuck-at-0%-job detection time from 300 s to ZERO_PROGRESS_TIMEOUT (120 s).
            #
            #   The no_step_heartbeat_timeout must be >= the VAE decode semaphore timeout
            #   (HordeInferenceProcess.VAE_SEMAPHORE_TIMEOUT) because when all diffusion steps
            #   finish the 100 % PIPELINE_STATE_CHANGE heartbeat resets heartbeats_inference_steps
            #   to 0, and the process then blocks on vae_decode_semaphore.acquire() without sending
            #   further heartbeats.  Using a value shorter than VAE_SEMAPHORE_TIMEOUT would kill a
            #   legitimately-running VAE decode.
            #
            # INFERENCE_STARTING: guard with _recently_recovered.
            #   After replacing a stuck INFERENCE_PROCESSING process the semaphore is released
            #   and any waiting INFERENCE_STARTING process may immediately acquire it.  While it
            #   is blocked on acquire() it cannot send heartbeats; without the guard we would
            #   falsely declare it stuck right away.
            is_stuck_inference = False
            if process_info.last_process_state == HordeProcessState.INFERENCE_PROCESSING:
                # Use VAE_SEMAPHORE_TIMEOUT from HordeInferenceProcess as the source of truth for
                # no_step_heartbeat_timeout (check 4 — VAE decode protection).
                # Use ZERO_PROGRESS_TIMEOUT for check 2 (stuck at 0 % with live heartbeats).
                # ZERO_PROGRESS_TIMEOUT is shorter because VAE decode never happens at 0 % progress,
                # so we do not need the full 300 s safety margin there.
                no_step_timeout = HordeInferenceProcess.VAE_SEMAPHORE_TIMEOUT
                is_stuck_inference = self._process_map.is_stuck_on_inference(
                    process_info.process_id,
                    self.bridge_data.inference_step_timeout,
                    no_step_heartbeat_timeout=no_step_timeout,
                    zero_progress_timeout=ProcessMap.ZERO_PROGRESS_TIMEOUT,
                )
            elif not self._recently_recovered:
                is_stuck_inference = self._process_map.is_stuck_on_inference(
                    process_info.process_id,
                    self.bridge_data.inference_step_timeout,
                )

            # Check total inference duration even if per-step heartbeats are alive.
            # This fires only when is_stuck_on_inference() didn't already trigger.
            if (
                not is_stuck_inference
                and process_info.inference_started_timestamp is not None
                and (now - process_info.inference_started_timestamp) > self.bridge_data.inference_timeout
            ):
                total_elapsed = now - process_info.inference_started_timestamp
                logger.error(
                    f"{process_info} exceeded total inference_timeout of {self.bridge_data.inference_timeout}s "
                    f"(running for {total_elapsed:.1f}s) — replacing.",
                )
                self._replace_inference_process(process_info, respawn=not self._shutting_down)
                any_replaced = any_process_replaced = True

            elif is_stuck_inference:
                # Per-step stall detected — enhanced logging then replace.
                time_since_heartbeat = now - process_info.last_heartbeat_timestamp
                time_since_progress = now - process_info.last_progress_timestamp
                progress_str = (
                    f"{process_info.last_heartbeat_percent_complete}%"
                    if process_info.last_heartbeat_percent_complete is not None
                    else "Not available"
                )
                logger.error(
                    f"{process_info} seems to be stuck mid inference - "
                    f"Last heartbeat: {time_since_heartbeat:.1f}s ago, "
                    f"Last progress change: {time_since_progress:.1f}s ago, "
                    f"Progress: {progress_str}, "
                    f"Job: {process_info.last_job_referenced.id_ if process_info.last_job_referenced else 'None'}",
                )
                self._replace_inference_process(process_info, respawn=not self._shutting_down)
                any_replaced = any_process_replaced = True
            else:
                # Check PROCESS_STARTING first - this should always be checked regardless of job availability
                # since processes should complete initialization even when no jobs are available.
                # Use a startup timeout that is at least process_timeout so expensive child-process
                # initialization (imports/model manager setup) isn't misclassified as a stuck preload.
                # Skip during shutdown: end_inference_processes() will kill stuck-starting processes,
                # and _replace_inference_process would spawn a new one we immediately don't want.
                process_starting_timeout = max(
                    self.bridge_data.process_timeout,
                    self.bridge_data.preload_timeout,
                )
                if not self._shutting_down and self._check_and_replace_process(
                    process_info,
                    process_starting_timeout,
                    HordeProcessState.PROCESS_STARTING,
                    "seems to be stuck starting",
                ):
                    any_replaced = any_process_replaced = True

                # RESULT_SUBMITTING is managed by the process manager, not the subprocess.
                # Killing the subprocess is not appropriate here; just reset the state so
                # the process can accept new jobs.  The primary fix is in
                # submit_single_generation() (try/finally), but this serves as a safety net
                # for any edge cases that slip through.
                #
                # This check is intentionally placed BEFORE the _last_pop_no_jobs_available
                # guard below: when a process gets stuck here its job has already been removed
                # from jobs_in_progress (the HordeInferenceResultMessage was received), so
                # no_local_work would be True even though the process still holds an active
                # slot.  Skipping the check in that situation leaves the process permanently
                # blocked, preventing it from accepting new jobs when they eventually arrive.
                _result_submit_stuck_timeout = 60.0
                if (
                    process_info.process_type == HordeProcessType.INFERENCE
                    and process_info.last_process_state == HordeProcessState.RESULT_SUBMITTING
                    and (now - process_info.last_received_timestamp) > _result_submit_stuck_timeout
                ):
                    logger.error(
                        f"{process_info} has been stuck in RESULT_SUBMITTING for "
                        f"{now - process_info.last_received_timestamp:.0f}s; "
                        "resetting to WAITING_FOR_JOB to unblock job scheduling",
                    )
                    self._on_process_state_change(
                        process_id=process_info.process_id,
                        new_state=HordeProcessState.WAITING_FOR_JOB,
                    )
                    any_replaced = True

                # Detect a safety process stuck in SAFETY_EVALUATING or SAFETY_STARTING.
                #
                # A safety process stuck in these states will never return to WAITING_FOR_JOB,
                # so get_first_available_safety_process() will always return None, blocking all
                # new job pops indefinitely.  This check is placed BEFORE the
                # _last_pop_no_jobs_available guard for the same reason as RESULT_SUBMITTING:
                # the stuck safety process may be the sole reason no new jobs are available.
                if (
                    process_info.process_type == HordeProcessType.SAFETY
                    and process_info.last_process_state
                    in (HordeProcessState.SAFETY_EVALUATING, HordeProcessState.SAFETY_STARTING)
                    and (now - process_info.last_received_timestamp) > self.bridge_data.process_timeout
                ):
                    logger.error(
                        f"{process_info} has been stuck in {process_info.last_process_state.name} for "
                        f"{now - process_info.last_received_timestamp:.0f}s; "
                        "scheduling safety process replacement to unblock job scheduling",
                    )
                    # Move any job currently under safety evaluation back to the pending queue
                    # so it will be re-evaluated by the replacement safety process.
                    for job_info in list(self.jobs_being_safety_checked):
                        logger.warning(
                            f"Re-queuing job {job_info.sdk_api_job_info.id_} for safety re-evaluation "
                            "due to stuck safety process",
                        )
                        self.jobs_being_safety_checked.remove(job_info)
                        self.jobs_pending_safety_check.append(job_info)
                    self._safety_processes_should_be_replaced = True
                    any_replaced = True

                # Recover inference slots stuck in PROCESS_ENDING only when we need that
                # slot to meet the configured active-process target. This avoids fighting
                # legitimate scale-down operations where PROCESS_ENDING is expected.
                if (
                    not self._shutting_down
                    and process_info.process_type == HordeProcessType.INFERENCE
                    and process_info.last_process_state == HordeProcessState.PROCESS_ENDING
                    and (now - process_info.last_received_timestamp) > self.bridge_data.process_timeout
                    and num_loaded_inference < self.max_inference_processes
                    and num_total_inference <= self.max_inference_processes
                ):
                    logger.error(
                        f"{process_info} has been stuck in PROCESS_ENDING for "
                        f"{now - process_info.last_received_timestamp:.0f}s; "
                        "replacing it to restore inference capacity",
                    )
                    self._replace_inference_process(process_info)
                    any_replaced = any_process_replaced = True
                    continue

                # Skip other state checks if no jobs are available since those states are job-related.
                # "No jobs available" means either the API reported none, or pops are currently
                # paused (so no new jobs can arrive from the API) — but only when there is also
                # no local work already in the queue.
                # If there are already jobs pending in our local queue or in-progress,
                # always run these checks so that stuck processes don't block local work.
                if (self._last_pop_no_jobs_available or self._job_pops_paused) and no_local_work:
                    continue

                # Check for WAITING_FOR_JOB inference processes with stale heartbeats when
                # jobs are pending.  These processes may have died silently without sending
                # PROCESS_ENDING.  A freshly replaced process starts in PROCESS_STARTING (with
                # a fresh timestamp) so it will never match this condition immediately after a
                # recovery — no need to gate this check on _recently_recovered.
                # The effective threshold is max(process_timeout, waiting_for_job_timeout)
                # so that even workers with a short process_timeout wait at least
                # waiting_for_job_timeout seconds before a replacement is triggered.
                if (
                    process_info.process_type == HordeProcessType.INFERENCE
                    and process_info.last_process_state == HordeProcessState.WAITING_FOR_JOB
                    and (now - process_info.last_heartbeat_timestamp)
                    > max(self.bridge_data.process_timeout, self.bridge_data.waiting_for_job_timeout)
                ):
                    logger.error(
                        f"{process_info} has been idle in WAITING_FOR_JOB for "
                        f"{now - process_info.last_heartbeat_timestamp:.0f}s with local work pending; replacing it",
                    )
                    self._replace_inference_process(process_info)
                    any_replaced = any_process_replaced = True

                # Check if an INFERENCE_STARTING process is stuck because the semaphore is
                # unavailable even though no other process is actively running inference.
                # is_stuck_on_inference()'s heartbeat check uses inference_step_timeout, which
                # measures the wrong thing for this phase: a process waiting to acquire the
                # semaphore should not have to wait more than preload_timeout seconds when no
                # other inference is running.  We skip this check when another process IS in
                # INFERENCE_PROCESSING, because that process legitimately holds the semaphore
                # and INFERENCE_STARTING should wait for it to finish.
                # The _recently_recovered guard is deliberately NOT applied here: with frequent
                # recoveries (e.g. slow/faulted jobs), _recently_recovered may be True for most
                # of the worker's lifetime, permanently preventing stuck detection for any
                # INFERENCE_STARTING process.  When no INFERENCE_PROCESSING process is active the
                # semaphore is available, so any process stuck in INFERENCE_STARTING for longer
                # than preload_timeout is genuinely hung — safe to replace without the guard.
                if process_info.last_process_state == HordeProcessState.INFERENCE_STARTING:
                    time_elapsed_starting = now - process_info.last_received_timestamp
                    time_elapsed_starting = min(
                        time_elapsed_starting,
                        now - process_info.last_heartbeat_timestamp,
                    )
                    if (
                        time_elapsed_starting > self.bridge_data.preload_timeout
                        and not any_active_inference_processing
                    ):
                        logger.error(
                            f"{process_info} seems to be stuck in INFERENCE_STARTING "
                            "with no active inference process holding the semaphore; replacing it",
                        )
                        self._replace_inference_process(process_info)
                        any_replaced = any_process_replaced = True

        # Job-related stuck states are checked in a multi-pass scan: for each condition in
        # priority order, every process is scanned before moving to the next condition.
        # This guarantees true cross-process prioritization — a stuck INFERENCE_POST_PROCESSING
        # or POST_PROCESSING_STARTING process (which may hold the VAE decode semaphore and
        # block other inference processes on this worker) is cleared across the entire worker before any
        # idle/finished state (e.g. MODEL_PRELOADED, which is merely waiting for a job
        # dispatch) is reclaimed, regardless of the order in which subprocesses appear in
        # the process map.
        #
        # The conditions are ordered: actively-in-progress states first, idle/finished last:
        #   1. INFERENCE_POST_PROCESSING  — holds VAE decode semaphore
        #   2. POST_PROCESSING_STARTING   — transitioning into VAE decode (may hold semaphore)
        #   3. MODEL_PRELOADING           — actively loading a model
        #   4. DOWNLOADING_AUX_MODEL      — actively downloading a file
        #   5. MODEL_PRELOADED            — idle, waiting for a job to be dispatched
        if not ((self._last_pop_no_jobs_available or self._job_pops_paused) and no_local_work):
            conditions: list[tuple[float, HordeProcessState, str]] = [
                # --- actively in progress ---
                (
                    self.bridge_data.post_process_timeout + (3 * self.bridge_data.max_batch),
                    HordeProcessState.INFERENCE_POST_PROCESSING,
                    "seems to be stuck post processing",
                ),
                (
                    self.bridge_data.post_process_timeout + (3 * self.bridge_data.max_batch),
                    HordeProcessState.POST_PROCESSING_STARTING,
                    "seems to be stuck starting post processing",
                ),
                (
                    self.bridge_data.preload_timeout,
                    HordeProcessState.MODEL_PRELOADING,
                    "seems to be stuck preloading a model",
                ),
                (
                    self.bridge_data.download_timeout,
                    HordeProcessState.DOWNLOADING_AUX_MODEL,
                    "seems to be stuck downloading an auxiliary model (LoRa, etc)",
                ),
                (
                    self.bridge_data.download_timeout,
                    HordeProcessState.DOWNLOADING_MODEL,
                    "seems to be stuck downloading a model",
                ),
                (
                    self.bridge_data.preload_timeout,
                    HordeProcessState.JOB_RECEIVED,
                    "seems to be stuck after receiving a job (never started processing)",
                ),
                # --- finished / idle ---
                (
                    self.bridge_data.preload_timeout,
                    HordeProcessState.MODEL_PRELOADED,
                    "seems to be stuck in MODEL_PRELOADED (job was never dispatched)",
                ),
                (
                    self.bridge_data.preload_timeout,
                    HordeProcessState.UNLOADED_MODEL_FROM_RAM,
                    "seems to be stuck in UNLOADED_MODEL_FROM_RAM",
                ),
            ]
            # A MODEL_PRELOADED process that is merely waiting for a free inference concurrency
            # slot is NOT stuck: as soon as an in-progress job finishes, start_inference() (which
            # runs earlier in the same control-loop tick) dispatches a job to it and it leaves the
            # MODEL_PRELOADED state. Replacing it in that situation would needlessly discard an
            # already-loaded model and churn the slot back through PROCESS_STARTING →
            # MODEL_PRELOADING, which is exactly the "jobs/processes stuck in MODEL_PRELOADED"
            # behaviour we want to avoid. The destructive replacement is therefore only a genuine
            # safety net for when a slot IS free but dispatch still never happens. Determine here
            # whether the inference concurrency slots are currently saturated.
            _processes_post_processing = (
                self._process_map.num_busy_with_post_processing()
                if self.post_process_job_overlap_allowed
                else 0
            )
            inference_slots_full = len(self.jobs_in_progress) >= (
                self.max_concurrent_inference_processes + _processes_post_processing
            )
            for timeout, state, error_message in conditions:
                for process_info in self._process_map.values():
                    # Skip replacing a preloaded-but-idle process that is legitimately queued
                    # behind saturated inference slots (see note above).
                    if state == HordeProcessState.MODEL_PRELOADED and inference_slots_full:
                        continue
                    # For MODEL_PRELOADING: read the model name *before* calling
                    # _check_and_replace_process because _replace_inference_process (called
                    # internally) clears loaded_horde_model_name via on_process_ending().
                    preload_model = (
                        process_info.loaded_horde_model_name
                        if state == HordeProcessState.MODEL_PRELOADING
                        else None
                    )
                    if self._check_and_replace_process(process_info, timeout, state, error_message):
                        any_replaced = any_process_replaced = True
                        if preload_model is not None:
                            self._record_preload_stuck_failure(preload_model, now)

        # If any subprocesses were actually replaced and we are not already inside a recovery
        # window, start the cascading-recovery guard timer.  Soft corrective actions such as
        # resetting RESULT_SUBMITTING state do not count as process replacements and therefore
        # do not trigger the cooldown.  When _recently_recovered is already True (because an
        # earlier recovery is still cooling down) we still perform the replacements above
        # (e.g. MODEL_PRELOADING) but we do NOT start a second timer thread, which would
        # extend the blocked window unnecessarily.
        if any_process_replaced and not self._recently_recovered:
            self._recently_recovered = True
            threading.Thread(target=timed_unset_recently_recovered, daemon=True).start()

        shutdown_timed_out = self._shutting_down and (now - self._shutting_down_time) > (60 * 5)

        if (self._last_pop_no_jobs_available or self._job_pops_paused) and no_local_work and not shutdown_timed_out:
            # Either the API told us there are no jobs, or pops are paused and there is
            # nothing in our local queue — skip the "all processes timed out" bulk-replacement
            # check because idle processes timing out in this state is expected and normal.
            return any_replaced

        # If all processes haven't done sent a message for a while
        all_processes_timed_out = all(
            ((now - process_info.last_received_timestamp) > self.bridge_data.process_timeout)
            for process_info in self._process_map.values()
        )

        # If all processes are unresponsive or we should replace all processes
        # *except* if we've already done so recently or the last job pop was a "no jobs available" response
        if (all_processes_timed_out and not ((self._last_pop_no_jobs_available and no_local_work) or self._recently_recovered)) or (
            shutdown_timed_out
        ):
            if not self._hung_processes_detected:
                self._hung_processes_detected = True
                self._hung_processes_detected_time = now

            last_detected_delta = now - self._hung_processes_detected_time

            if last_detected_delta < 20:
                return False

            self._purge_jobs()

            if self.bridge_data.exit_on_unhandled_faults or self._shutting_down:
                logger.error("All processes have been unresponsive for too long, exiting.")

                self._abort()
                if self.bridge_data.exit_on_unhandled_faults:
                    logger.error("Exiting due to exit_on_unhandled_faults being enabled")

                return True

            logger.error("All processes have been unresponsive for too long, attempting to recover.")
            already_recovering = self._recently_recovered
            self._recently_recovered = True

            for process_info in self._process_map.values():
                if process_info.process_type == HordeProcessType.INFERENCE:
                    self._replace_inference_process(process_info)
                    any_replaced = True

            # Only start a new timer thread if one is not already running (i.e. _recently_recovered
            # was False before this block). This prevents a second thread from being spawned during
            # a shutdown timeout that fires while a prior recovery is still cooling down.
            if not already_recovering:
                threading.Thread(target=timed_unset_recently_recovered, daemon=True).start()
        else:
            self._hung_processes_detected = False

        return any_replaced
