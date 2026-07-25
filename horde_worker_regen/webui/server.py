"""Web server for the Horde Worker status UI."""

import asyncio
import base64
import io
import json
import math
import os
import re
import sqlite3
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
from aiohttp import web
from loguru import logger

import horde_worker_regen

try:
    from PIL import Image as _PILImage

    _PIL_AVAILABLE = True
except ImportError:
    _PILImage = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False
    logger.warning(
        "Pillow is not installed; gallery thumbnails will not be generated. "
        "Install the 'Pillow' package to enable thumbnail generation in the web UI.",
    )

_THUMBNAIL_MAX_PX = 384
"""Maximum pixel dimension (width or height) for gallery thumbnails."""

# Patterns for variable data stripped when normalising error messages for grouping.
_ERROR_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_ERROR_HEX_ID_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
# Matches any numeric token (including single-digit values) so that process-slot
# numbers, short PIDs, or job counters are normalised alongside the longer IDs
# they accompany.  Single-digit process numbers (e.g. slot 0, slot 1) are the
# primary motivation for matching \d+ rather than \d{2,}.
_ERROR_NUM_TOKEN_RE = re.compile(r"\b\d+\b")
# Timestamps in log lines: full ISO-style (YYYY-MM-DD HH:mm:ss[.SSS]) and
# the short HH:mm:ss[.SSS] format used by the webui log sink.
_ERROR_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b|\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b",
)

_STATS_MIN_SNAPSHOT_INTERVAL = 10.0
"""Finest seconds-between-snapshots interval, used while the configured retention window
is short enough not to need coarsening (this was the previous fixed interval)."""

_STATS_MAX_SNAPSHOT_INTERVAL = 3600.0
"""Coarsest seconds-between-snapshots interval. Long retention windows collect less often
(down to once per hour) instead of accumulating an unbounded number of snapshot rows."""

_STATS_TARGET_SNAPSHOT_COUNT = 60_480
"""Target number of statistics snapshots spanning the full retention window. At the
default 7-day retention this works out to exactly _STATS_MIN_SNAPSHOT_INTERVAL, so the
out-of-the-box collection cadence is unchanged; longer retention windows automatically
space snapshots out further (see WorkerWebUI._stats_snapshot_interval) so that the full
window is still covered without the snapshot count growing unbounded."""

_STATS_MAX_SNAPSHOTS_CEILING = 100_000
"""Absolute ceiling on in-memory/persisted statistics snapshots, independent of the
configured retention period (mirrors _HORDE_MAX_SERVER_SNAPS_CEILING)."""

_CHART_MAX_POINTS = 480
"""Target number of points returned by the stats/horde-snapshots endpoints per request.
A browser chart only has a few hundred pixels of width to plot on, so a wide time window
(e.g. "All" over weeks/months of retention) is downsampled down to roughly this many
points server-side rather than shipping and rendering every raw sample -- the bigger the
requested window, the coarser the returned resolution, never the reverse."""

_STATS_CUMULATIVE_KEYS = frozenset({"jc", "jf", "jp", "ks"})
"""Statistics snapshot fields that are running totals rather than point-in-time gauges.
Downsampling must not average these -- see _downsample_series."""


def _downsample_series(
    rows: list[dict[str, Any]],
    max_points: int,
    cumulative_keys: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Reduce a chronological list of snapshot dicts to at most ``max_points`` entries.

    Splits ``rows`` into contiguous buckets and collapses each bucket to a single point:
    gauge-like fields are averaged (smooth trend, no aliasing spikes), while fields named
    in ``cumulative_keys`` (running totals such as a jobs-completed counter) keep the
    earliest raw value in the first bucket and the latest raw value in every other bucket.
    That guarantees the first and last returned points still carry the true earliest/latest
    counter values, so a caller computing a delta from ``result[0]`` and ``result[-1]``
    gets the exact total over the whole range even though interior points are approximate.
    """
    n = len(rows)
    if n <= max_points or max_points <= 0:
        return rows
    bucket_size = math.ceil(n / max_points)
    result = []
    for i in range(0, n, bucket_size):
        bucket = rows[i : i + bucket_size]
        is_first_bucket = i == 0
        point: dict[str, Any] = {}
        for key in bucket[-1].keys():
            if key == "t":
                point["t"] = bucket[-1]["t"]
            elif key in cumulative_keys:
                point[key] = bucket[0][key] if is_first_bucket else bucket[-1][key]
            else:
                values = [b[key] for b in bucket if isinstance(b.get(key), (int, float))]
                point[key] = (sum(values) / len(values)) if values else bucket[-1].get(key)
        result.append(point)
    return result


def _windowed_snapshots(snapshots: list[dict[str, Any]], window_param: str | None) -> list[dict[str, Any]]:
    """Filter a chronological snapshot list to the trailing window requested by the client.

    ``window_param`` is the raw ``window`` query-string value: ``None``/absent or ``"all"``
    returns everything, otherwise it's parsed as a number of seconds measured back from the
    most recent snapshot. An invalid value is treated the same as "all" rather than erroring,
    since a malformed window shouldn't make the whole chart fail to load.
    """
    if not snapshots or not window_param or window_param == "all":
        return snapshots
    try:
        window_secs = float(window_param)
    except ValueError:
        return snapshots
    cutoff = snapshots[-1].get("t", 0) - window_secs
    return [s for s in snapshots if s.get("t", 0) >= cutoff]

_MAX_PERSISTED_ERRORS = 1000
"""Maximum number of error rows to load from the database on startup (matches ProcessManager in-memory cap)."""

_MAX_OCCURRENCES_PER_GROUP = 50
"""Maximum individual occurrences returned per error group in the /api/errors/grouped response."""

_DB_PRUNE_INTERVAL = 3600.0
"""Minimum seconds between automatic pruning of old data from the SQLite database."""

_MAX_THUMBNAILS_IN_MEMORY = 2000
"""Maximum number of gallery thumbnails kept in RAM when a gallery database is available.
Older thumbnails are evicted from memory and re-fetched from the database on demand."""

_MAX_FULLRES_IN_MEMORY = 20
"""Maximum number of full-resolution gallery images kept in RAM. Full-resolution data only
stays in memory when it could not be persisted to the gallery database (or none is
configured); each image is several MB, so this must stay small."""

_MAX_GALLERY_ENTRIES_WITH_DB = 100_000
"""Cap on in-memory gallery entries (metadata) when a gallery database is available. Beyond
this, the oldest entries are evicted from RAM; they remain in the database."""

_MAX_GALLERY_ENTRIES_NO_DB = 1000
"""Hard cap on in-memory gallery entries when no gallery database is configured — without a
database there is nowhere to offload entries, so the oldest are dropped entirely."""

_HORDE_MAX_SERVER_SNAPS_CEILING = 86_400
"""Absolute ceiling for in-memory horde network snapshots (30 days at 30-second polls),
independent of the configured data retention period."""

_UNSET: Any = object()
"""Sentinel used to distinguish an explicitly-passed ``None`` from an omitted argument."""

# ---------------------------------------------------------------------------
# Runtime-configurable settings exposed via the /api/settings endpoint.
# Each entry maps a bridge_data field name to a dict describing the field:
#   type  – Python type used for validation (bool, int, or float)
#   min   – minimum value (numeric types only)
#   max   – maximum value (numeric types only)
# ---------------------------------------------------------------------------
_SETTINGS_SPEC: dict[str, dict[str, Any]] = {
    # Connection
    "horde_url": {"type": str, "readonly": True},
    "webui_url": {"type": str, "readonly": True},
    # Capabilities
    "nsfw": {"type": bool},
    "censor_nsfw": {"type": bool},
    "allow_img2img": {"type": bool},
    "allow_inpainting": {"type": bool},
    "allow_unsafe_ip": {"type": bool},
    "allow_post_processing": {"type": bool},
    "allow_controlnet": {"type": bool},
    "allow_sdxl_controlnet": {"type": bool},
    "allow_lora": {"type": bool},
    "require_upfront_kudos": {"type": bool},
    "limit_max_steps": {"type": bool},
    "extra_slow_worker": {"type": bool},
    # Performance
    "max_power": {"type": int, "min": 1, "max": 128},
    "max_batch": {"type": int, "min": 1, "max": 100},
    "max_threads": {"type": int, "min": 1, "max": 8},
    "safety_on_gpu": {"type": bool},
    "high_memory_mode": {"type": bool},
    "very_high_memory_mode": {"type": bool},
    "high_performance_mode": {"type": bool},
    "moderate_performance_mode": {"type": bool},
    "unload_models_from_vram_often": {"type": bool},
    "very_fast_disk_mode": {"type": bool},
    "post_process_job_overlap": {"type": bool},
    "cycle_process_on_model_change": {"type": bool},
    "horde_model_stickiness": {"type": float, "min": 0.0, "max": 1.0},
    # Timeouts
    "process_timeout": {"type": int, "min": 60, "max": 3600},
    "inference_timeout": {"type": int, "min": 60, "max": 7200},
    "inference_step_timeout": {"type": int, "min": 10, "max": 1800},
    "preload_timeout": {"type": int, "min": 15, "max": 600},
    "post_process_timeout": {"type": int, "min": 15, "max": 600},
    "waiting_for_job_timeout": {"type": int, "min": 60, "max": 3600},
    # Behavior
    "minutes_allowed_without_jobs": {"type": int, "min": 0, "max": 3599},
    "auto_restart_on_idle_minutes": {"type": int, "min": 0, "max": 1440},
    "force_restart_timeout": {"type": int, "min": 5, "max": 600},
    "suppress_speed_warnings": {"type": bool},
    "exit_on_unhandled_faults": {"type": bool},
    "limited_console_messages": {"type": bool},
    "stats_output_frequency": {"type": int, "min": 5, "max": 3600},
    "purge_loras_on_download": {"type": bool},
    "remove_maintenance_on_init": {"type": bool},
    "max_job_retries": {"type": int, "min": 0, "max": 10},
    "max_submit_retries": {"type": int, "min": 0, "max": 50},
    "data_retention_days": {"type": int, "min": 1, "max": 3650},
    # Prompt filters
    "positive_prompt_append": {"type": list},
    "positive_prompt_remove": {"type": list},
    "positive_prompt_replace": {"type": list},
    "negative_prompt_append": {"type": list},
    "negative_prompt_remove": {"type": list},
    "negative_prompt_replace": {"type": list},
    "prompt_remove_cleanup_separators": {"type": bool},
    "prompt_append_separator": {"type": bool},
    "prompt_filters_enabled": {"type": bool},
    "prompt_remove_whole_word": {"type": bool},
    "prompt_remove_case_sensitive": {"type": bool},
    "positive_prompt_append_enabled": {"type": bool},
    "positive_prompt_remove_enabled": {"type": bool},
    "positive_prompt_replace_enabled": {"type": bool},
    "positive_prompt_conditional_add": {"type": list},
    "positive_prompt_conditional_add_enabled": {"type": bool},
    "negative_prompt_append_enabled": {"type": bool},
    "negative_prompt_remove_enabled": {"type": bool},
    "negative_prompt_replace_enabled": {"type": bool},
    "negative_prompt_conditional_add": {"type": list},
    "negative_prompt_conditional_add_enabled": {"type": bool},
}


class WorkerWebUI:
    """Web UI server for displaying worker status and progress."""

    def __init__(
        self,
        port: int = 3000,
        update_interval: float = 1.0,
        db_path: str | None = None,
        data_retention_days: int = 7,
    ) -> None:
        """Initialize the web UI server.

        Args:
            port: The port to run the web server on (default: 3000)
            update_interval: How often to update status in seconds (default: 1.0)
            db_path: Path to the directory containing SQLite database files, or a legacy
                     single database file path. Existing directories are used directly;
                     existing files and non-existent paths are treated as file paths and
                     their parent directory is used (unless a trailing path separator
                     explicitly indicates a non-existent directory). When ``None`` (the
                     default) no data is persisted across restarts. Three separate
                     databases are used:
                     - webui_errors.db for errors log
                     - webui_stats.db for statistics snapshots
                     - webui_gallery.db for gallery images
            data_retention_days: Number of days to retain persisted data.  Entries older than
                     this threshold are pruned automatically.  Can also be set via the
                     ``AIWORKER_DATA_RETENTION_DAYS`` environment variable.
        """
        self.port = port
        self.update_interval = update_interval
        self.app = web.Application()
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

        # SQLite persistence --------------------------------------------------
        # Determine database directory and individual database paths.
        if db_path is not None:
            # Resolve the database directory from db_path:
            # - An existing directory → use it directly.
            # - An existing file → use its parent directory.
            # - A non-existent path → treat as a file path by default (use its
            #   parent directory).  Only treat a non-existent path as a directory
            #   if the caller explicitly signals it with a trailing path separator.
            abs_db_path = os.path.abspath(db_path)
            if os.path.isdir(abs_db_path):
                db_dir = abs_db_path
            elif os.path.exists(abs_db_path):
                db_dir = os.path.dirname(abs_db_path) if os.path.isfile(abs_db_path) else abs_db_path
            else:
                # Non-existent path: default to file-path semantics (parent dir)
                # unless the original path has a trailing separator, which explicitly
                # indicates a directory.
                if str(db_path).endswith(("/", os.sep)):
                    db_dir = abs_db_path
                else:
                    db_dir = os.path.dirname(abs_db_path)
            self._errors_db_path: str | None = os.path.join(db_dir, "webui_errors.db")
            self._stats_db_path: str | None = os.path.join(db_dir, "webui_stats.db")
            self._gallery_db_path: str | None = os.path.join(db_dir, "webui_gallery.db")
            self._runtime_settings_path: str | None = os.path.join(db_dir, "webui_settings.json")
        else:
            self._errors_db_path = None
            self._stats_db_path = None
            self._gallery_db_path = None
            self._runtime_settings_path = None

        # Allow the env var to override the constructor argument.
        _env_days = os.getenv("AIWORKER_DATA_RETENTION_DAYS")
        if _env_days is not None:
            try:
                _parsed = int(_env_days)
            except (ValueError, TypeError):
                logger.warning(
                    f"AIWORKER_DATA_RETENTION_DAYS environment variable has an invalid value: '{_env_days}'. "
                    "It must be a positive integer. Ignoring.",
                )
            else:
                if _parsed < 1:
                    logger.warning(
                        f"AIWORKER_DATA_RETENTION_DAYS environment variable has an out-of-range value: {_parsed}. "
                        "It must be >= 1. Ignoring.",
                    )
                elif _parsed > 3650:
                    logger.warning(
                        f"AIWORKER_DATA_RETENTION_DAYS environment variable has an out-of-range value: {_parsed}. "
                        "It must be <= 3650. Ignoring.",
                    )
                else:
                    data_retention_days = _parsed
        self._data_retention_days: int = max(1, min(3650, int(data_retention_days)))
        # Unix timestamp of the last database pruning run (0 = never pruned).
        self._last_db_prune_time: float = 0.0
        # Latest live error list received from the process manager this run.
        self._live_errors_history: list[str] = []

        self._session_baseline: dict[str, Any] = {}
        self._persisted_reset_baseline: dict[str, Any] = {}
        self._persisted_settings: dict[str, Any] = {}
        # Cumulative totals for count dicts loaded from DB at startup.
        # update_status() adds current-session values on top of these so counts
        # survive worker restarts without ever double-counting.
        self._aggregate_baseline: dict[str, dict[str, Any]] = {
            "images_per_model": {},
            "failed_jobs_per_model": {},
            "faulted_jobs_per_phase": {},
        }

        if self._errors_db_path is not None:
            # INFO before the load, not after: this phase reads the errors/stats/gallery
            # databases and has historically been the silent multi-minute gap in startup
            # logs when those files grew large. Announcing it first means a stall here is
            # attributable from the log instead of looking like a hang.
            logger.info("Loading persisted web UI data (errors, stats, gallery)...")
            _load_started = time.monotonic()
            self._init_db()
            self._load_persisted_data()
            self._load_session_baseline()
            _load_elapsed = time.monotonic() - _load_started
            log_persisted_load = logger.info if _load_elapsed >= 2.0 else logger.debug
            log_persisted_load(f"Persisted web UI data loaded in {_load_elapsed:.1f}s")
        self._load_persisted_settings()

        # Status data that will be updated by the worker
        self.status_data: dict[str, Any] = {
            "worker_name": "Unknown",
            "horde_username": "Unknown",
            "uptime": 0,
            "session_start_time": time.time(),
            "jobs_popped": 0,
            "jobs_queued": 0,
            "time_without_jobs": 0.0,
            "jobs_completed": 0,
            "jobs_faulted": 0,
            "processes_recovered": 0,
            "kudos_earned_session": 0.0,
            "kudos_per_hour": 0.0,
            "images_per_hour": 0.0,
            "current_job": None,
            "job_queue": [],
            "max_queue_size": 0,
            "queue_size_auto": False,
            "max_active_models": 0,
            "max_active_models_auto": False,
            "processes": [],
            "models_loaded": [],
            "ram_usage_mb": 0,
            "system_ram_usage_mb": 0,
            "total_ram_mb": 0,
            "vram_usage_mb": 0,
            "system_vram_usage_mb": 0,
            "total_vram_mb": 0,
            "cpu_usage_percent": 0,
            "cpu_cores_count": 0,
            "gpu_usage_percent": 0,
            "worker_gpu_percent": 0,
            "gpu_cores_count": 0,
            "container_cpu_percent": 0,
            "maintenance_mode": False,
            "job_pops_paused": False,
            "job_pops_pause_until": None,
            "user_kudos_total": 0.0,
            "last_image_base64": [],
            "last_image_submission_timestamp": 0.0,
            "last_image_model": "",
            "last_image_safety": [],
            "console_logs": [],
            "faulted_jobs_history": [],
            "errors_history": [],
            "images_count": 0,
            "user_details": {},
            "images_per_model": {},
            "failed_jobs_per_model": {},
            "faulted_jobs_per_phase": {},
            "avg_time_per_job_state": {},
            "max_time_per_job_state": {},
            "avg_time_per_step_per_model": {},
            "max_time_per_step_per_model": {},
            "avg_time_per_job_per_model": {},
            "max_time_per_job_per_model": {},
            # Snapshot of cumulative counters taken when the user clicks "Reset Stats".
            # JS subtracts these from the live values so the display reads from zero.
            "stats_reset_baseline": {},
        }

        # Merge persisted errors into live status_data now that status_data has
        # been initialised (DB loading runs before this).
        if self._errors_db_path is not None:
            self._merge_persisted_into_status()

        # Gallery image data stored separately – NOT included in /api/status to avoid
        # sending large base64 payloads on every poll.  Served via /api/gallery instead.
        # Keyed by gallery_id (int) for O(1) lookup; insertion order is oldest-first.
        self._gallery_dict: dict[int, dict[str, Any]] = {}
        # Monotonically increasing counter used to assign stable gallery_id values.
        self._next_gallery_id: int = 0

        # Ring buffer for time-series statistics snapshots served by /api/stats.
        self._stats_snapshots: deque[dict[str, Any]] = deque(maxlen=self._stats_max_snapshots)
        # Unix timestamp of the most recently recorded snapshot (0 = none yet).
        self._last_stats_snapshot_time: float = 0.0

        # Server-side horde network performance snapshots (accumulated even when
        # no browser is connected).  Served via /api/horde-snapshots so the JS
        # can seed its chart with history going back to server startup.
        self._horde_snapshots: deque[dict[str, Any]] = deque(maxlen=self._horde_max_server_snaps)
        # Background asyncio task handle for the horde polling loop.
        self._horde_bg_task: asyncio.Task | None = None
        # Latest aihorde.net maintenance/invite-only mode flags, refreshed periodically by
        # the same background task that polls performance snapshots. Served via
        # /api/horde-modes so the browser never has to call aihorde.net directly (that
        # cross-origin request requires a CORS preflight for the custom Client-Agent
        # header, which aihorde.net does not reliably grant for arbitrary worker origins).
        self._horde_modes: dict[str, Any] = {}
        self._last_horde_modes_fetch: float = 0.0

        # Database reset progress: None = idle, 0-100 = in progress/done.
        self._db_reset_progress: int | None = None
        self._db_reset_error: str | None = None
        self._db_reset_task: asyncio.Task | None = None

        # Re-populate gallery dict and stats from any data loaded from DB.
        if self._errors_db_path is not None:
            self._restore_persisted_collections()

        # Optional callback invoked when the UI requests a worker deletion.
        # Signature: async (worker_id: str) -> bool  (True = success, False = failure)
        self._delete_worker_callback: Callable[[str], Awaitable[bool]] | None = None

        # Optional callback invoked when the UI requests a pause/resume of job pops.
        # Signature: (paused: bool, pause_until: float | None) -> None
        self._set_job_pops_paused_callback: Callable[[bool, float | None], None] | None = None

        # Optional callback invoked when the UI requests maintenance mode to be cleared.
        # Signature: () -> None
        self._clear_maintenance_mode_callback: Callable[[], None] | None = None

        # Optional callback invoked when the UI requests a change to the max queue size.
        # Signature: (max_queue_size: int) -> None
        self._set_max_queue_size_callback: Callable[[int], None] | None = None

        # Optional callback invoked when the UI requests a change to the max active models.
        # Signature: (max_active_models: int) -> None
        self._set_max_active_models_callback: Callable[[int], None] | None = None

        # Optional callback invoked when the UI toggles auto mode for queue size.
        # Signature: (enabled: bool) -> None
        self._set_queue_size_auto_mode_callback: Callable[[bool], None] | None = None

        # Optional callback invoked when the UI toggles auto mode for max active models.
        # Signature: (enabled: bool) -> None
        self._set_max_active_models_auto_mode_callback: Callable[[bool], None] | None = None

        # Optional callback invoked when the UI changes a runtime setting.
        # Signature: (key: str, value: Any) -> None
        self._set_setting_callback: Callable[[str, Any], None] | None = None
        # Optional callback invoked when the UI requests a full worker-program restart.
        # Signature: () -> None
        self._restart_program_callback: Callable[[], None] | None = None

        # Current settings snapshot pushed by the process manager via update_settings_data().
        self._settings_data: dict[str, Any] = {}

        # Models data: enabled/disabled model lists for the Models settings section.
        self._models_data: dict[str, list[str]] = {"enabled": [], "disabled": []}

        # Optional callback invoked when the UI toggles a model's enabled state.
        # Signature: (model_name: str, enabled: bool) -> None
        self._toggle_model_callback: Callable[[str, bool], None] | None = None

        self._setup_routes()

    # ------------------------------------------------------------------
    # SQLite persistence helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the persistence tables in separate SQLite databases if they do not exist."""
        assert self._errors_db_path is not None
        assert self._stats_db_path is not None
        assert self._gallery_db_path is not None

        try:
            # Create the config directory if it doesn't exist.
            db_dir = os.path.dirname(os.path.abspath(self._errors_db_path))
            os.makedirs(db_dir, exist_ok=True)

            # Errors database
            with sqlite3.connect(self._errors_db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS errors_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        message TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """,
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_errors_log_created_at ON errors_log (created_at)",
                )
                conn.commit()

            # Stats database
            with sqlite3.connect(self._stats_db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS stats_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_json TEXT NOT NULL,
                        timestamp REAL NOT NULL
                    )
                    """,
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON stats_snapshots (timestamp)",
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_overview (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        jobs_popped INTEGER NOT NULL DEFAULT 0,
                        jobs_completed INTEGER NOT NULL DEFAULT 0,
                        jobs_faulted INTEGER NOT NULL DEFAULT 0,
                        processes_recovered INTEGER NOT NULL DEFAULT 0,
                        kudos_earned REAL NOT NULL DEFAULT 0.0,
                        time_without_jobs REAL NOT NULL DEFAULT 0.0,
                        updated_at REAL NOT NULL DEFAULT 0.0,
                        reset_baseline_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """,
                )
                try:
                    conn.execute(
                        "ALTER TABLE session_overview ADD COLUMN reset_baseline_json TEXT NOT NULL DEFAULT '{}'",
                    )
                except Exception:
                    pass  # column already exists on existing databases
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS horde_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_json TEXT NOT NULL,
                        timestamp REAL NOT NULL
                    )
                    """,
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_horde_timestamp ON horde_snapshots (timestamp)",
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_aggregates (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        aggregates_json TEXT NOT NULL DEFAULT '{}'
                    )
                    """,
                )
                conn.commit()

            # Gallery database
            with sqlite3.connect(self._gallery_db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gallery_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        gallery_id INTEGER NOT NULL,
                        timestamp REAL NOT NULL,
                        model TEXT,
                        base64_data TEXT,
                        thumbnail TEXT,
                        is_nsfw INTEGER DEFAULT 0,
                        is_csam INTEGER DEFAULT 0,
                        extra_json TEXT
                    )
                    """,
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_gallery_timestamp ON gallery_images (timestamp)",
                )
                # Covering index for the startup metadata load. The metadata columns
                # (is_nsfw, is_csam, extra_json) physically sit AFTER the multi-MB
                # base64_data blob in each table record, so reading them from the table
                # forces SQLite to walk every row's overflow-page chain — effectively a
                # full-file read that silently stalled startup for minutes on large
                # galleries. With this index the load is an index-only scan that never
                # touches the image blobs. Building it on an existing large DB pays that
                # full read once, inside the announced "Loading persisted web UI data"
                # phase; afterwards startups are fast.
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_gallery_meta ON gallery_images "
                    "(timestamp, id, gallery_id, model, is_nsfw, is_csam, extra_json)",
                )
                conn.commit()

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not initialise persistence databases: {exc}")

    def _cutoff_timestamp(self) -> float:
        """Return the Unix timestamp before which rows are considered expired."""
        return time.time() - self._data_retention_days * 86400.0

    @property
    def _stats_snapshot_interval(self) -> float:
        """Seconds between recorded statistics snapshots, scaled to the retention window.

        Spacing snapshots out further for long retention windows (instead of collecting
        at a fixed cadence forever) keeps the full window representable within
        _STATS_TARGET_SNAPSHOT_COUNT rows rather than growing without bound.
        """
        needed = (self._data_retention_days * 86400.0) / _STATS_TARGET_SNAPSHOT_COUNT
        return max(_STATS_MIN_SNAPSHOT_INTERVAL, min(needed, _STATS_MAX_SNAPSHOT_INTERVAL))

    @property
    def _stats_max_snapshots(self) -> int:
        """Max statistics snapshots to keep in memory and load from the database.

        Scales with data_retention_days (via _stats_snapshot_interval) so the full
        retention window is representable, bounded by an absolute ceiling — a
        multi-year retention setting would otherwise translate into an unbounded
        number of in-memory snapshot dicts.
        """
        scaled = int((self._data_retention_days * 86400.0) / self._stats_snapshot_interval) + 1
        return min(scaled, _STATS_MAX_SNAPSHOTS_CEILING)

    def _load_persisted_data(self) -> None:
        """Load persisted errors, gallery images, and stats snapshots from the databases.

        Loaded data is stored in private attributes that are merged into the live
        status/collection structures by :meth:`_merge_persisted_into_status` and
        :meth:`_restore_persisted_collections` after :attr:`status_data` is created.
        """
        assert self._errors_db_path is not None
        assert self._stats_db_path is not None
        assert self._gallery_db_path is not None

        cutoff = self._cutoff_timestamp()
        self._persisted_errors: list[str] = []
        self._persisted_gallery: list[dict[str, Any]] = []
        self._persisted_stats: list[dict[str, Any]] = []
        self._persisted_horde_snapshots: list[dict[str, Any]] = []
        self._persisted_aggregates: dict[str, Any] = {}
        # Max gallery_id across ALL rows (not just within retention) so _next_gallery_id
        # stays unique even when no in-retention rows are loaded at startup.
        self._persisted_max_gallery_id: int = -1

        # Load errors from errors database
        try:
            with sqlite3.connect(self._errors_db_path) as conn:
                rows = conn.execute(
                    "SELECT message FROM errors_log WHERE created_at >= ?"
                    " ORDER BY created_at DESC, id DESC LIMIT ?",
                    (cutoff, _MAX_PERSISTED_ERRORS),
                ).fetchall()
                self._persisted_errors = [row[0] for row in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load errors from '{self._errors_db_path}': {exc}")

        # Load gallery from gallery database
        try:
            with sqlite3.connect(self._gallery_db_path) as conn:
                # Determine the overall max gallery_id so _next_gallery_id stays unique
                # even when no in-retention rows are loaded (e.g. worker was down longer
                # than the retention window and all rows are expired but not yet pruned).
                max_id_row = conn.execute("SELECT MAX(gallery_id) FROM gallery_images").fetchone()
                if max_id_row and max_id_row[0] is not None:
                    self._persisted_max_gallery_id = int(max_id_row[0])
                # Load lightweight metadata ONLY — never base64_data (several MB per row,
                # can OOM the worker) and never thumbnails (tens of KB per row, and
                # reaching the thumbnail column requires SQLite to walk past each row's
                # base64_data overflow pages — a full-file read that silently stalled
                # startup for minutes on large galleries). Thumbnails are fetched lazily,
                # per page, by _handle_gallery / _handle_gallery_image via
                # _fetch_gallery_thumbnails_from_db. The row count is bounded in SQL, and
                # the column list matches the idx_gallery_meta covering index so this is
                # an index-only scan that never touches the image blobs.
                rows = conn.execute(
                    "SELECT gallery_id, timestamp, model, is_nsfw, is_csam, extra_json FROM ("
                    "SELECT id, gallery_id, timestamp, model, is_nsfw, is_csam, extra_json "
                    "FROM gallery_images WHERE timestamp >= ?"
                    " ORDER BY timestamp DESC, id DESC LIMIT ?"
                    ") ORDER BY timestamp ASC, id ASC",
                    (cutoff, _MAX_GALLERY_ENTRIES_WITH_DB),
                ).fetchall()
                for row in rows:
                    entry: dict[str, Any] = {
                        "gallery_id": row[0],
                        "timestamp": row[1],
                        "model": row[2],
                        "is_nsfw": bool(row[3]),
                        "is_csam": bool(row[4]),
                    }
                    if row[5]:
                        try:
                            entry.update(json.loads(row[5]))
                        except (ValueError, TypeError):
                            pass
                    self._persisted_gallery.append(entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load gallery from '{self._gallery_db_path}': {exc}")

        # Load stats from stats database
        try:
            with sqlite3.connect(self._stats_db_path) as conn:
                rows = conn.execute(
                    "SELECT snapshot_json FROM ("
                    "SELECT id, snapshot_json, timestamp FROM stats_snapshots"
                    " WHERE timestamp >= ? ORDER BY timestamp DESC, id DESC LIMIT ?"
                    ") ORDER BY timestamp ASC, id ASC",
                    (cutoff, self._stats_max_snapshots),
                ).fetchall()
                for row in rows:
                    try:
                        decoded = json.loads(row[0])
                        if isinstance(decoded, dict):
                            self._persisted_stats.append(decoded)
                    except (ValueError, TypeError):
                        pass

                # Load horde snapshots up to the full retention window.
                horde_rows = conn.execute(
                    "SELECT snapshot_json FROM ("
                    "SELECT id, snapshot_json, timestamp FROM horde_snapshots"
                    " WHERE timestamp >= ? ORDER BY timestamp DESC, id DESC LIMIT ?"
                    ") ORDER BY timestamp ASC, id ASC",
                    (cutoff, self._horde_max_server_snaps),
                ).fetchall()
                for row in horde_rows:
                    try:
                        decoded = json.loads(row[0])
                        if isinstance(decoded, dict):
                            self._persisted_horde_snapshots.append(decoded)
                    except (ValueError, TypeError):
                        pass

                # Load per-model/per-phase aggregate totals.
                try:
                    agg_row = conn.execute(
                        "SELECT aggregates_json FROM session_aggregates WHERE id = 1",
                    ).fetchone()
                    if agg_row and agg_row[0]:
                        decoded_agg = json.loads(agg_row[0])
                        if isinstance(decoded_agg, dict):
                            self._persisted_aggregates = decoded_agg
                except Exception:
                    logger.debug("Could not load session_aggregates from database", exc_info=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load stats from '{self._stats_db_path}': {exc}")

    def _load_session_baseline(self) -> None:
        """Load persisted session overview counters as the baseline for this run.

        Called once at startup so cumulative counters (jobs_completed, kudos_earned, etc.)
        carry forward across worker restarts.  Not called during prune refresh to avoid
        overwriting the in-session accumulated baseline.
        """
        if self._stats_db_path is None:
            return
        try:
            with sqlite3.connect(self._stats_db_path) as conn:
                row = conn.execute(
                    "SELECT jobs_popped, jobs_completed, jobs_faulted, processes_recovered, "
                    "kudos_earned, time_without_jobs FROM session_overview WHERE id = 1",
                ).fetchone()
                if row:
                    self._session_baseline = {
                        "jobs_popped": int(row[0]),
                        "jobs_completed": int(row[1]),
                        "jobs_faulted": int(row[2]),
                        "processes_recovered": int(row[3]),
                        "kudos_earned_session": float(row[4]),
                        "time_without_jobs": float(row[5]),
                    }
                try:
                    rb_row = conn.execute(
                        "SELECT reset_baseline_json FROM session_overview WHERE id = 1",
                    ).fetchone()
                    if rb_row and rb_row[0]:
                        rb = json.loads(rb_row[0])
                        if rb:
                            self._persisted_reset_baseline = rb
                except Exception:
                    logger.debug("Failed to load persisted reset baseline from database", exc_info=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load session overview from '{self._stats_db_path}': {exc}")

    def _load_persisted_settings(self) -> None:
        """Load runtime setting overrides saved by the web UI from disk."""
        if self._runtime_settings_path is None:
            return
        try:
            if os.path.exists(self._runtime_settings_path):
                with open(self._runtime_settings_path) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._persisted_settings = data
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load persisted settings from '{self._runtime_settings_path}': {exc}")

    def _save_persisted_settings(self) -> None:
        """Write runtime setting overrides to disk so they survive restarts."""
        if self._runtime_settings_path is None:
            return
        try:
            with open(self._runtime_settings_path, "w") as f:
                json.dump(self._persisted_settings, f)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not save persisted settings to '{self._runtime_settings_path}': {exc}")

    def _merge_persisted_into_status(self) -> None:
        """Merge data loaded by :meth:`_load_persisted_data` into :attr:`status_data`."""
        errors = getattr(self, "_persisted_errors", [])
        if errors:
            self.status_data["errors_history"] = list(errors)

        for key, value in self._session_baseline.items():
            if value:
                self.status_data[key] = value

        if self._persisted_reset_baseline:
            self.status_data["stats_reset_baseline"] = self._persisted_reset_baseline

    @staticmethod
    def _history_overlap_len(newer: list[str], older: list[str]) -> int:
        """Return the largest suffix/prefix overlap length between two newest-first histories."""
        max_overlap = min(len(newer), len(older))
        for overlap in range(max_overlap, 0, -1):
            if newer[-overlap:] == older[:overlap]:
                return overlap
        return 0

    def _merge_errors_history(self, live_errors: list[str]) -> list[str]:
        """Merge current-session errors with persisted history without duplicating overlap."""
        persisted_errors = getattr(self, "_persisted_errors", [])
        overlap = self._history_overlap_len(live_errors, persisted_errors)
        return (list(live_errors) + persisted_errors[overlap:])[:_MAX_PERSISTED_ERRORS]

    def _restore_persisted_collections(self) -> None:
        """Populate :attr:`_gallery_dict` and :attr:`_stats_snapshots` from persisted data."""
        gallery = getattr(self, "_persisted_gallery", [])
        # Start from the overall DB max so _next_gallery_id is unique even when
        # _persisted_gallery is empty (all rows expired but not yet pruned).
        max_id: int = getattr(self, "_persisted_max_gallery_id", -1)
        for entry in gallery:
            gid = entry["gallery_id"]
            self._gallery_dict[gid] = entry
            if gid > max_id:
                max_id = gid
        if max_id >= 0:
            self._next_gallery_id = max_id + 1
        self.status_data["images_count"] = len(self._gallery_dict)
        self._enforce_gallery_memory_caps()

        stats = getattr(self, "_persisted_stats", [])
        if stats:
            self._stats_snapshots = deque(stats, maxlen=self._stats_max_snapshots)
            if self._stats_snapshots:
                # Restore the last snapshot time so the interval guard works correctly.
                last = self._stats_snapshots[-1]
                self._last_stats_snapshot_time = self._safe_snapshot_time(last)

        horde = getattr(self, "_persisted_horde_snapshots", [])
        if horde:
            self._horde_snapshots = deque(horde, maxlen=self._horde_max_server_snaps)

        # Restore per-model/per-phase aggregates persisted from the previous session.
        agg = getattr(self, "_persisted_aggregates", {})
        if agg:
            # Count dicts accumulate additively across restarts.  Store the persisted
            # totals as the baseline so update_status() can add current-session counts
            # on top without double-counting.
            for key in ("images_per_model", "failed_jobs_per_model", "faulted_jobs_per_phase"):
                if key in agg and isinstance(agg[key], dict):
                    self._aggregate_baseline[key] = dict(agg[key])
                    self.status_data[key] = dict(agg[key])
            # Timing dicts are averages/maxes; restore last-known values so the
            # statistics page isn't blank until the first job of the new session completes.
            for key in (
                "avg_time_per_job_state",
                "max_time_per_job_state",
                "avg_time_per_step_per_model",
                "max_time_per_step_per_model",
                "avg_time_per_job_per_model",
                "max_time_per_job_per_model",
            ):
                if key in agg and isinstance(agg[key], dict):
                    self.status_data[key] = dict(agg[key])

    @staticmethod
    def _safe_snapshot_time(snapshot: dict[str, Any]) -> float:
        """Return a snapshot timestamp as float, falling back to 0.0 for malformed values."""
        try:
            return float(snapshot.get("t", 0))
        except (TypeError, ValueError):
            return 0.0

    def _prune_old_db_data(self) -> None:
        """Delete database rows older than the configured retention period.

        When no databases are configured, prunes the in-memory gallery directly so the
        retention window still bounds memory usage instead of never pruning at all.
        """
        if self._errors_db_path is None:
            cutoff = self._cutoff_timestamp()
            # Only prune entries that carry a valid numeric timestamp — entries without
            # one cannot be aged and remain bounded by _enforce_gallery_memory_caps.
            expired = [
                gid
                for gid, e in self._gallery_dict.items()
                if isinstance(e.get("timestamp"), (int, float)) and e["timestamp"] < cutoff
            ]
            for gid in expired:
                self._gallery_dict.pop(gid, None)
            if expired:
                self.status_data["images_count"] = len(self._gallery_dict)
            self._last_db_prune_time = time.time()
            return
        assert self._stats_db_path is not None
        assert self._gallery_db_path is not None
        cutoff = self._cutoff_timestamp()
        pruned = False
        try:
            # Prune errors database
            with sqlite3.connect(self._errors_db_path) as conn:
                conn.execute("DELETE FROM errors_log WHERE created_at < ?", (cutoff,))
                conn.commit()

            # Prune stats database
            with sqlite3.connect(self._stats_db_path) as conn:
                conn.execute("DELETE FROM stats_snapshots WHERE timestamp < ?", (cutoff,))
                conn.execute("DELETE FROM horde_snapshots WHERE timestamp < ?", (cutoff,))
                conn.commit()

            # Prune gallery database
            with sqlite3.connect(self._gallery_db_path) as conn:
                conn.execute("DELETE FROM gallery_images WHERE timestamp < ?", (cutoff,))
                conn.commit()

            pruned = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not prune old data from persistence databases: {exc}")
        finally:
            self._last_db_prune_time = time.time()

        if pruned:
            self._refresh_in_memory_after_prune()

    def _refresh_in_memory_after_prune(self) -> None:
        """Refresh in-memory collections to match the pruned state of the databases.

        Reloads persisted data from the (now-pruned) databases, rebuilds
        :attr:`_gallery_dict` and :attr:`_stats_snapshots`, then re-merges
        live session errors with the reloaded persisted history so that the
        running UI reflects the new retention window immediately.
        """
        # Reload persisted data from the pruned databases.
        self._load_persisted_data()

        # Rebuild gallery dict (clear first so removed entries are not retained).
        self._gallery_dict.clear()
        max_id: int = getattr(self, "_persisted_max_gallery_id", -1)
        for entry in getattr(self, "_persisted_gallery", []):
            gid = entry["gallery_id"]
            self._gallery_dict[gid] = entry
            if gid > max_id:
                max_id = gid
        self.status_data["images_count"] = len(self._gallery_dict)
        if max_id >= 0:
            self._next_gallery_id = max_id + 1
        self._enforce_gallery_memory_caps()

        # Rebuild stats snapshots.
        stats = getattr(self, "_persisted_stats", [])
        self._stats_snapshots = deque(stats, maxlen=self._stats_max_snapshots)
        if self._stats_snapshots:
            last = self._stats_snapshots[-1]
            self._last_stats_snapshot_time = self._safe_snapshot_time(last)

        # Rebuild horde snapshots.
        horde = getattr(self, "_persisted_horde_snapshots", [])
        if horde:
            self._horde_snapshots = deque(horde, maxlen=self._horde_max_server_snaps)

        # Re-merge errors history (live session errors + reloaded persisted history).
        self.status_data["errors_history"] = self._merge_errors_history(self._live_errors_history)

    def set_data_retention_days(self, days: int) -> None:
        """Update the data retention period and immediately prune expired rows.

        Args:
            days: New retention period in days (must be >= 1).
        """
        self._data_retention_days = max(1, min(3650, int(days)))
        # Resize the horde/stats snapshot deques to match the new retention period so
        # the 6h / All window buttons have enough history to differentiate.
        self._horde_snapshots = deque(self._horde_snapshots, maxlen=self._horde_max_server_snaps)
        self._stats_snapshots = deque(self._stats_snapshots, maxlen=self._stats_max_snapshots)
        self._prune_old_db_data()

    def set_delete_worker_callback(self, callback: Callable[[str], Awaitable[bool]] | None) -> None:
        """Register (or clear) the async callback used to delete a worker via the Horde API.

        Args:
            callback: An async callable that accepts a worker_id string and returns
                      True on success or False on failure.  Pass None to unregister.
        """
        self._delete_worker_callback = callback

    def set_job_pops_paused_callback(self, callback: Callable[[bool, float | None], None] | None) -> None:
        """Register (or clear) the callback used to pause or resume job pops.

        Args:
            callback: A callable that accepts two arguments: a bool (``True`` =
                      pause, ``False`` = resume) and an optional float giving the
                      Unix timestamp at which the pause should automatically
                      expire (``None`` for an indefinite pause).  Pass ``None``
                      to unregister.
        """
        self._set_job_pops_paused_callback = callback

    def set_clear_maintenance_mode_callback(self, callback: Callable[[], None] | None) -> None:
        """Register (or clear) the callback used to clear maintenance mode from the UI.

        Args:
            callback: A callable that takes no arguments and clears the worker's
                      maintenance mode on the Horde.  Pass ``None`` to unregister.
        """
        self._clear_maintenance_mode_callback = callback

    def set_max_queue_size_callback(self, callback: Callable[[int], None] | None) -> None:
        """Register (or clear) the callback used to change the maximum job queue size at runtime.

        Args:
            callback: A callable that accepts a single int (the new max queue size).
                      Pass ``None`` to unregister.
        """
        self._set_max_queue_size_callback = callback

    def set_max_active_models_callback(self, callback: Callable[[int], None] | None) -> None:
        """Register (or clear) the callback used to change the maximum active model slots at runtime.

        Args:
            callback: A callable that accepts a single int (the new max active models count).
                      Pass ``None`` to unregister.
        """
        self._set_max_active_models_callback = callback

    def set_queue_size_auto_mode_callback(self, callback: Callable[[bool], None] | None) -> None:
        """Register (or clear) the callback used to enable or disable auto queue-size mode.

        Args:
            callback: A callable that accepts a single bool (``True`` = enable auto,
                      ``False`` = disable auto).  Pass ``None`` to unregister.
        """
        self._set_queue_size_auto_mode_callback = callback

    def set_max_active_models_auto_mode_callback(self, callback: Callable[[bool], None] | None) -> None:
        """Register (or clear) the callback used to enable or disable auto max-active-models mode.

        Args:
            callback: A callable that accepts a single bool (``True`` = enable auto,
                      ``False`` = disable auto).  Pass ``None`` to unregister.
        """
        self._set_max_active_models_auto_mode_callback = callback

    def set_setting_callback(self, callback: Callable[[str, Any], None] | None) -> None:
        """Register (or clear) the callback invoked when the UI changes a runtime setting.

        Args:
            callback: A callable that accepts two arguments: the setting key (str) and
                      the new validated value.  Pass ``None`` to unregister.
        """
        self._set_setting_callback = callback
        if callback is not None and self._persisted_settings:
            for key, value in self._persisted_settings.items():
                try:
                    callback(key, value)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Could not apply persisted setting '{key}={value!r}' at startup: {exc}")

    def set_restart_program_callback(self, callback: Callable[[], None] | None) -> None:
        """Register (or clear) the callback invoked when the UI requests a program restart.

        Args:
            callback: A callable with no arguments. Pass ``None`` to unregister.
        """
        self._restart_program_callback = callback

    def update_settings_data(self, settings: dict[str, Any]) -> None:
        """Update the current settings snapshot served by ``GET /api/settings``.

        Called by the process manager each status update cycle so the settings
        page always reflects the live ``bridge_data`` values.

        Args:
            settings: Flat dict mapping field names to their current values.
        """
        self._settings_data = dict(settings)

    def set_toggle_model_callback(self, callback: Callable[[str, bool], None] | None) -> None:
        """Register (or clear) the callback invoked when the UI toggles a model.

        Args:
            callback: A callable that accepts a model name (str) and enabled state (bool).
                      Pass ``None`` to unregister.
        """
        self._toggle_model_callback = callback

    def update_models_data(self, enabled: list[str], disabled: list[str]) -> None:
        """Update the enabled/disabled model lists served by ``GET /api/models``.

        Called by the process manager each status update cycle.

        Args:
            enabled: List of model names currently enabled for job pops.
            disabled: List of model names currently disabled by the user.
        """
        self._models_data = {"enabled": sorted(enabled), "disabled": sorted(disabled)}

    def _setup_routes(self) -> None:
        """Set up the web server routes."""
        self.app.router.add_get("/", self._handle_index)
        self.app.router.add_get("/api/status", self._handle_status)
        self.app.router.add_get("/api/stats", self._handle_stats)
        self.app.router.add_get("/api/errors", self._handle_errors)
        self.app.router.add_get("/api/errors/grouped", self._handle_errors_grouped)
        self.app.router.add_post("/api/errors/clear", self._handle_errors_clear)
        self.app.router.add_get("/api/last_image", self._handle_last_image)
        self.app.router.add_get("/api/gallery", self._handle_gallery)
        self.app.router.add_get("/api/gallery/thumb/{gallery_id}", self._handle_gallery_thumb_binary)
        self.app.router.add_get("/api/gallery/full/{gallery_id}", self._handle_gallery_full_binary)
        self.app.router.add_get("/api/gallery/models", self._handle_gallery_models)
        self.app.router.add_get("/api/gallery/safety", self._handle_gallery_safety)
        self.app.router.add_get("/api/gallery/image", self._handle_gallery_image)
        self.app.router.add_get("/api/gallery/last-batch", self._handle_gallery_last_batch)
        self.app.router.add_get("/api/config", self._handle_config)
        self.app.router.add_get("/health", self._handle_health)
        self.app.router.add_delete("/api/worker/{worker_id}", self._handle_delete_worker)
        self.app.router.add_post("/api/job_pops/pause", self._handle_set_job_pops_paused)
        self.app.router.add_get("/api/job_pops/time_without_jobs", self._handle_get_time_without_jobs)
        self.app.router.add_post("/api/maintenance/clear", self._handle_clear_maintenance_mode)
        self.app.router.add_get("/api/settings", self._handle_get_settings)
        self.app.router.add_post("/api/settings", self._handle_set_setting)
        self.app.router.add_get("/api/models", self._handle_get_models)
        self.app.router.add_post("/api/models", self._handle_toggle_model)
        self.app.router.add_post("/api/restart", self._handle_restart_program)
        self.app.router.add_post("/api/reset-stats", self._handle_reset_stats)
        self.app.router.add_post("/api/reset-database", self._handle_reset_database)
        self.app.router.add_get("/api/reset-database/progress", self._handle_reset_database_progress)
        self.app.router.add_get("/api/horde-snapshots", self._handle_horde_snapshots)
        self.app.router.add_get("/api/horde-modes", self._handle_horde_modes)

    async def _handle_config(self, request: web.Request) -> web.Response:
        """Handle config API request."""
        # Return update interval in milliseconds for JavaScript
        return web.json_response({"update_interval_ms": int(self.update_interval * 1000)})

    async def _handle_index(self, request: web.Request) -> web.Response:
        """Serve the main HTML page."""
        html = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Horde Worker Admin</title>
    <style>
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --sidebar-width: 260px;
            --action-btn-height: 32px;
            --page-spacing: 14px;
            /* Light mode: light sidebar that matches the light theme (was previously a fixed
               dark sidebar regardless of theme). Dark mode overrides these below. */
            --sidebar-bg: #ffffff;
            --sidebar-hover: #eef2f7;
            --sidebar-text: #64748b;
            --sidebar-text-strong: #1e293b;
            --sidebar-border: #e2e8f0;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --success: #10b981;
            --warning: #f59e0b;
            --error: #ef4444;
            --text-muted: #94a3b8;
            --text-light: #e2e8f0;
            --main-bg: #f1f5f9;
            --card-bg: #ffffff;
            --border: #e2e8f0;
        }


        html { scroll-behavior: smooth; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--main-bg);
            color: #334155;
            min-height: 100vh;
            display: flex;
        }

        /* ---- Sidebar ---- */
        .sidebar {
            width: var(--sidebar-width);
            background: var(--sidebar-bg);
            position: fixed;
            top: 0; left: 0;
            height: 100vh;
            display: flex;
            flex-direction: column;
            z-index: 100;
            transition: transform 0.28s cubic-bezier(.4,0,.2,1);
            overflow-y: auto;
            border-right: 1px solid var(--sidebar-border);
        }
        .sidebar-logo { padding: 22px 20px 18px; border-bottom: 1px solid var(--sidebar-border); flex-shrink: 0; }
        .sidebar-logo h1 { color: var(--sidebar-text-strong); font-size: 1.15rem; font-weight: 700; letter-spacing: 0.3px; }
        .sidebar-logo p { color: var(--sidebar-text); font-size: 0.75rem; margin-top: 3px; }
        .sidebar-nav { flex: 1; padding: 12px 0; }
        .nav-section-label { color: var(--sidebar-text); font-size: 0.67rem; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; padding: 10px 20px 4px; }
        .nav-item { display: flex; align-items: center; gap: 10px; padding: 9px 20px; color: var(--sidebar-text); font-size: 0.875rem; font-weight: 500; transition: background 0.15s, color 0.15s, border-color 0.15s; cursor: pointer; border-left: 3px solid transparent; user-select: none; background: none; border-top: none; border-right: none; border-bottom: none; width: 100%; text-align: left; }
        .nav-item:hover { background: var(--sidebar-hover); color: var(--sidebar-text-strong); }
        .nav-item.active { background: var(--sidebar-hover); color: var(--sidebar-text-strong); border-left-color: var(--accent); }
        .nav-icon { font-size: 1rem; width: 18px; text-align: center; flex-shrink: 0; }
        .sidebar-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 99; backdrop-filter: blur(1px); }
        .sidebar-overlay.active { display: block; }

        /* ---- Mobile navbar ---- */
        .mobile-navbar { display: none; position: fixed; top: 0; left: 0; right: 0; height: 54px; background: var(--sidebar-bg); align-items: center; padding: 0 12px; z-index: 200; gap: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.25); border-bottom: 1px solid var(--sidebar-border); }
        .hamburger-btn { background: none; border: none; color: var(--sidebar-text-strong); font-size: 1.3rem; cursor: pointer; padding: 6px; border-radius: 6px; line-height: 1; transition: background 0.15s; flex-shrink: 0; display: inline-flex; align-items: center; justify-content: center; }
        .hamburger-btn:hover { background: var(--sidebar-hover); }
        .mobile-title { color: var(--sidebar-text-strong); font-size: 0.95rem; font-weight: 600; flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .mobile-navbar #mobile-status-badge { flex-shrink: 0; display: inline-flex; }
        .mobile-uptime { color: var(--sidebar-text); font-size: 0.7rem; font-family: 'Courier New', monospace; white-space: nowrap; flex-shrink: 0; }
        .mobile-navbar .theme-toggle { flex-shrink: 0; }
        @media (max-width: 400px) { .mobile-uptime { display: none; } }

        /* ---- Mobile resources sub-bar ---- */
        .mobile-resources { display: none; position: fixed; top: 54px; left: 0; right: 0; background: #12162a; grid-template-columns: repeat(4, 1fr); padding: 3px 6px; z-index: 199; border-bottom: 1px solid rgba(255,255,255,0.06); }
        .mobile-res-col { display: flex; flex-direction: column; align-items: center; gap: 1px; padding: 2px 0; }
        .mobile-res-head { color: #94a3b8; font-size: 0.6rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; }
        .mobile-res-chip { color: var(--text-muted); font-size: 0.65rem; font-weight: 600; font-family: 'Courier New', monospace; display: block; text-align: center; }
        .mobile-res-chip-secondary { font-size: 0.6rem; opacity: 0.75; }

        /* ---- Main content ---- */
        .main-content { margin-left: var(--sidebar-width); flex: 1; min-height: 100vh; display: flex; flex-direction: column; min-width: 0; }

        /* ---- Top bar ---- */
        .topbar { background: white; border-bottom: 1px solid var(--border); padding: 14px 24px; display: flex; align-items: stretch; gap: 16px; flex-wrap: wrap; flex-shrink: 0; }
        .topbar-worker { flex: 1; min-width: 0; display: flex; flex-direction: column; justify-content: center; }
        .topbar-worker-name { font-size: 1.15rem; font-weight: 700; color: #1e293b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .topbar-worker-sub { font-size: 0.82rem; color: #64748b; margin-top: 2px; }
        .topbar-meta { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 8px; align-self: center; }
        .topbar-uptime { font-size: 0.82rem; color: #64748b; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; padding: 4px 10px; text-align: center; display: flex; align-items: center; justify-content: center; }

        /* ---- Status badges ---- */
        #worker-status-badge { display: inline-flex; }
        #worker-status-badge .status-badge { justify-content: center; }
        .status-badge { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; }
        .status-badge::before { content: ''; width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
        .status-active { background: #d1fae5; color: #065f46; }
        .status-active::before { background: #10b981; }
        .status-maintenance { background: #fef3c7; color: #92400e; }
        .status-maintenance::before { background: #f59e0b; animation: pulse-dot 1.5s ease-in-out infinite; }
        .status-paused { background: #e0e7ff; color: #3730a3; }
        .status-paused::before { background: #6366f1; }
        @keyframes pulse-dot { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

        .content-area { padding: 22px 24px; flex: 1; }

        /* ---- Page (SPA) ---- */
        .page { display: none; }
        .page.active { display: block; }
        .page > * + * { margin-top: var(--page-spacing); }

        .section { margin-bottom: 0; }
        .section-header { display: flex; align-items: center; gap: 10px; margin-bottom: var(--page-spacing); }
        .section-title { font-size: 0.82rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 1px; }
        .section-count { background: #e2e8f0; color: #475569; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 20px; }
        .user-details-grid + .user-details-grid,
        .user-details-grid + .card,
        .stats-summary-grid + .stats-summary-grid { margin-top: var(--page-spacing); }

        .card { background: var(--card-bg); border-radius: 12px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.07), 0 1px 2px rgba(0,0,0,0.04); border: 1px solid var(--border); }
        .card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #f1f5f9; }
        .card-title { font-size: 0.8rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.8px; display: flex; align-items: center; gap: 7px; }
        .last-result-card-header { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 6px; }
        .overview-model-label { font-size: 0.75rem; color: #94a3b8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; pointer-events: none; text-align: center; min-width: 0; }
        .overview-time-label { font-size: 0.75rem; color: #94a3b8; white-space: nowrap; }
        @media (max-width: 500px) { .last-result-card-header { grid-template-columns: 1fr auto; } .overview-model-label { display: none; } }

        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--page-spacing); }
        .grid-4 > *, .grid-3 > *, .grid-2 > * { min-width: 0; }
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--page-spacing); }
        .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--page-spacing); }
        .overview-bottom-grid-left { grid-row: span 2; }
        .card-header-count { font-size: 0.8rem; font-weight: 700; color: #334155; }
        #queue-count, #models-count { color: #2563eb; }
        #queue-max,   #models-max   { color: #64748b; }

        .stat-card { background: var(--card-bg); border-radius: 12px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.07); border: 1px solid var(--border); }
        .stat-card-label { font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
        .stat-card-value { font-size: 1.7rem; font-weight: 700; color: #1e293b; line-height: 1; }
        .stat-card-value.success { color: var(--success); }
        .stat-card-value.warning { color: var(--warning); }
        .stat-card-value.error   { color: var(--error); }
        .stat-card-value.accent  { color: var(--accent); }
        .trust-indicator { font-size: 1.1rem; margin-left: 6px; cursor: help; vertical-align: middle; }
        .trust-indicator.success { color: var(--success); }
        .trust-indicator.error   { color: var(--error); }

        .stat-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid #f8fafc; }
        .stat-row:last-child { border-bottom: none; }
        .stat-label { color: #64748b; font-size: 0.85rem; font-weight: 500; }
        .stat-value { color: #1e293b; font-weight: 600; font-size: 0.9rem; text-align: right; max-width: 62%; word-break: break-word; }
        .stat-value.success { color: var(--success); }
        .stat-value.warning { color: var(--warning); }
        .stat-value.error   { color: var(--error); }

        .progress-section { margin-bottom: 14px; }
        .progress-section:last-child { margin-bottom: 0; }
        .progress-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
        .progress-label { font-size: 0.83rem; font-weight: 500; color: #475569; }
        .progress-value { font-size: 0.83rem; font-weight: 700; color: #1e293b; }
        .progress-bar-container { width: 100%; height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%); border-radius: 4px; transition: width 0.4s ease; min-width: 0; }

        .job-state-badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; background: #f0fdf4; color: #166534; }
        .job-state-timer { margin-left: 4px; }

        .process-item { background: #f8fafc; border: 1px solid #e8eef4; border-left: 3px solid var(--accent); border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; }
        .process-item:last-child { margin-bottom: 0; }
        .process-id-row { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; margin-bottom: 3px; }
        .process-id { font-weight: 700; color: var(--accent); font-size: 0.88rem; }
        .process-type-badge { font-size: 0.72rem; background: #e0e7ff; color: #4338ca; padding: 1px 7px; border-radius: 4px; font-weight: 600; }
        .process-state-badge { font-size: 0.72rem; background: #f0fdf4; color: #166534; padding: 1px 7px; border-radius: 4px; font-weight: 600; margin-left: auto; }
        .process-detail-text { font-size: 0.8rem; color: #64748b; }

        .job-item { display: flex; align-items: center; justify-content: space-between; background: #f8fafc; border: 1px solid #e8eef4; border-radius: 7px; padding: 7px 12px; margin-bottom: 5px; font-size: 0.83rem; }
        .job-item:last-child { margin-bottom: 0; }
        .job-item-left { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .job-id { font-family: 'Courier New', monospace; color: var(--accent); font-weight: 600; font-size: 0.8rem; }
        .job-elapsed { flex-shrink: 0; margin-left: 10px; color: #8899aa; font-size: 0.78rem; font-variant-numeric: tabular-nums; white-space: nowrap; }

        .model-list { display: flex; flex-wrap: wrap; gap: 6px; }
        .model-badge { background: #e0e7ff; color: #4338ca; padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 500; }

        /* ---- Inline limit editor (max queue / max active models) ---- */
        .limit-editor { display: flex; align-items: center; gap: 4px; }
        .limit-input { width: 54px; height: var(--action-btn-height); padding: 2px 5px; font-size: 0.75rem; border: 1px solid #cbd5e1; border-radius: 4px; text-align: center; background: #f8fafc; color: #334155; transition: border-color 0.15s; }
        .limit-input:focus { outline: none; border-color: var(--accent); }
        .limit-input:disabled { opacity: 0.55; cursor: not-allowed; background: #e2e8f0; }
        .limit-set-btn { padding: 2px 9px; font-size: 0.75rem; font-weight: 600; background: var(--accent); color: #fff; border: none; border-radius: 4px; cursor: pointer; transition: background 0.15s; }
        .limit-set-btn:hover { background: var(--accent-hover); }
        .limit-set-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .limit-auto-btn { padding: 2px 8px; font-size: 0.75rem; font-weight: 600; background: #e2e8f0; color: #475569; border: none; border-radius: 4px; cursor: pointer; transition: background 0.15s, color 0.15s; }
        .limit-auto-btn:hover { background: #cbd5e1; }
        .limit-auto-btn.active { background: var(--success); color: #fff; }
        .limit-auto-btn.active:hover { background: #059669; }
        [data-theme="dark"] .limit-input { background: #1e293b; border-color: #334155; color: #e2e8f0; }
        [data-theme="dark"] .limit-input:focus { border-color: var(--accent); }
        [data-theme="dark"] .limit-input:disabled { background: #0f1924; }
        [data-theme="dark"] .limit-auto-btn { background: #1e293b; color: #94a3b8; }
        [data-theme="dark"] .limit-auto-btn:hover { background: #2d3f55; }
        [data-theme="dark"] .limit-auto-btn.active { background: var(--success); color: #fff; }
        .theme-toggle, .nsfw-blur-btn, .clear-maintenance-btn, .limit-set-btn, .limit-auto-btn, .console-pause-btn, .console-copy-btn, .job-pops-pause-btn, .errors-view-btn, .pagination-controls button, .image-overlay-close, .worker-delete-btn, .stats-window-btn, .horde-window-btn, .settings-page-btn, .setting-apply-btn, .confirm-modal-btn {
            height: var(--action-btn-height);
            box-sizing: border-box;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        .topbar-uptime, .status-badge, .job-state-badge, .process-type-badge, .process-state-badge, .model-badge, .worker-version-badge, .worker-type-badge, .worker-online-badge, .wcap {
            height: var(--action-btn-height);
            box-sizing: border-box;
            display: inline-flex;
            align-items: center;
        }

        .log-panel { width: 100%; height: min(800px, 80vh); overflow: hidden; }
        .console-container { background: #0c0c0c; border-radius: 8px; padding: 12px 14px; height: 100%; box-sizing: border-box; overflow-y: auto; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 1rem; font-weight: 400; color: #cccccc; line-height: 1.2; }
        .console-pause-btn { margin-left: auto; background: #e2e8f0; color: #475569; border: none; border-radius: 6px; padding: 3px 10px; font-size: 0.75rem; font-weight: 600; cursor: pointer; transition: background 0.15s, color 0.15s; }
        .console-pause-btn:hover { background: #cbd5e1; }
        .console-pause-btn.paused { background: var(--accent); color: #fff; }
        .console-pause-btn.paused:hover { background: var(--accent-hover); }
        .console-copy-btn { margin-left: 6px; background: #e2e8f0; color: #475569; border: none; border-radius: 6px; padding: 3px 10px; font-size: 0.75rem; font-weight: 600; cursor: pointer; transition: background 0.15s, color 0.15s; }
        .console-copy-btn:hover { background: #cbd5e1; }
        .console-copy-btn.copied { background: #22c55e; color: #fff; }
        .console-copy-btn.copied:hover { background: #16a34a; }
        .console-copy-btn.error { background: #ef4444; color: #fff; }
        .console-copy-btn.error:hover { background: #dc2626; }
        .console-filter-select { margin-left: 6px; background: var(--card-bg); color: inherit; border: 1px solid var(--border); border-radius: 6px; padding: 3px 8px; font-size: 0.75rem; font-weight: 600; cursor: pointer; transition: background 0.15s, color 0.15s; height: var(--action-btn-height); }

        /* ---- Job pops pause button ---- */
        .job-pops-pause-wrap { position: relative; display: flex; }
        .job-pops-pause-btn { background: #e2e8f0; color: #475569; border: 1px solid #cbd5e1; border-radius: 6px; padding: 5px 12px; font-size: 0.78rem; font-weight: 600; cursor: pointer; transition: background 0.15s, color 0.15s; white-space: nowrap; }
        .job-pops-pause-btn:hover { background: #cbd5e1; }
        .job-pops-pause-btn.paused { background: #e0e7ff; color: #3730a3; border-color: #a5b4fc; }
        .job-pops-pause-btn.paused:hover { background: #c7d2fe; }
        [data-theme="dark"] .job-pops-pause-btn { background: #1e2d42; color: #94a3b8; border-color: #334155; }
        [data-theme="dark"] .job-pops-pause-btn:hover { background: #2d3f55; }
        [data-theme="dark"] .job-pops-pause-btn.paused { background: #312e81; color: #a5b4fc; border-color: #6366f1; }
        [data-theme="dark"] .job-pops-pause-btn.paused:hover { background: #3730a3; }
        .job-pops-pause-menu { position: absolute; top: calc(100% + 4px); right: 0; background: #fff; border: 1px solid #cbd5e1; border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.12); min-width: 150px; z-index: 200; overflow: hidden; }
        .job-pops-pause-menu button { display: block; width: 100%; padding: 8px 14px; background: none; border: none; text-align: left; font-size: 0.82rem; font-weight: 600; color: #334155; cursor: pointer; transition: background 0.12s; }
        .job-pops-pause-menu button:hover { background: #f1f5f9; }
        [data-theme="dark"] .job-pops-pause-menu { background: #1e293b; border-color: #334155; }
        [data-theme="dark"] .job-pops-pause-menu button { color: #94a3b8; }
        [data-theme="dark"] .job-pops-pause-menu button:hover { background: #263348; }
        .clear-maintenance-btn { display: none; background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; border-radius: 6px; padding: 5px 12px; font-size: 0.78rem; font-weight: 600; cursor: pointer; transition: background 0.15s, color 0.15s; white-space: nowrap; }
        .clear-maintenance-btn:hover { background: #fde68a; }
        [data-theme="dark"] .clear-maintenance-btn { background: #451a03; color: #fbbf24; border-color: #92400e; }
        [data-theme="dark"] .clear-maintenance-btn:hover { background: #78350f; }

        /* ---- Gallery ---- */
        .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; width: 100%; }
        .image-grid-item { position: relative; overflow: hidden; border-radius: 8px; background: #f1f5f9; aspect-ratio: 1; display: flex; align-items: center; justify-content: center; }
        .image-grid-item img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; display: block; }
        .image-grid-item img:hover { transform: scale(1.04); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .image-grid-item .image-timestamp { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.6); color: #e2e8f0; font-size: 0.65rem; padding: 3px 6px; text-align: center; border-radius: 0 0 8px 8px; pointer-events: none; opacity: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .image-flag-badges { position: absolute; top: 4px; right: 4px; display: flex; flex-direction: column; gap: 3px; align-items: flex-end; pointer-events: none; opacity: 1; }
        .image-flag-badge { font-size: 0.6rem; font-weight: 700; padding: 2px 5px; border-radius: 4px; line-height: 1.3; text-transform: uppercase; letter-spacing: 0.04em; }
        .image-flag-badge.nsfw { background: rgba(234, 179, 8, 0.92); color: #1c1000; }
        .image-flag-badge.csam { background: rgba(220, 38, 38, 0.92); color: #fff; }
        [data-theme="dark"] .image-flag-badge.nsfw { background: rgba(202, 138, 4, 0.92); color: #fef9c3; }
        [data-theme="dark"] .image-flag-badge.csam { background: rgba(185, 28, 28, 0.92); color: #fee2e2; }
        @keyframes gallery-shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
        .image-grid-item.loading { background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%); background-size: 200% 100%; animation: gallery-shimmer 1.5s infinite; }
        [data-theme="dark"] .image-grid-item.loading { background: linear-gradient(90deg, #1e293b 25%, #2d3f55 50%, #1e293b 75%); background-size: 200% 100%; animation: gallery-shimmer 1.5s infinite; }
        .gallery-page-header, .settings-page-header { flex-wrap: wrap; }
        .gallery-filter-bar { display: flex; align-items: center; gap: 8px; margin-left: auto; flex-wrap: wrap; }
        .gallery-filter-bar label { font-size: 0.82rem; font-weight: 500; color: #475569; white-space: nowrap; }
        .gallery-filter-bar select { font-size: 0.82rem; padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 6px; background: #f8fafc; color: #1e293b; cursor: pointer; max-width: 320px; }
        [data-theme="dark"] .gallery-filter-bar label { color: #94a3b8; }
        [data-theme="dark"] .gallery-filter-bar select { background: #1e293b; border-color: #334155; color: #e2e8f0; }
        @media (max-width: 768px) { .gallery-filter-bar { flex-direction: column; align-items: stretch; margin-left: 0; width: 100%; } .gallery-filter-bar select { max-width: 100%; width: 100%; } }
        .gallery-view-toggle { display: flex; gap: 4px; flex-shrink: 0; }
        .gallery-view-btn { background: transparent; border: 1px solid var(--border,#e2e8f0); border-radius: 5px; padding: 4px 9px; font-size: 0.8rem; cursor: pointer; color: var(--text-muted,#64748b); transition: background 0.15s, color 0.15s, border-color 0.15s; line-height: 1; }
        .gallery-view-btn.active { background: var(--accent,#6366f1); color: #fff; border-color: var(--accent,#6366f1); }
        .gallery-view-btn:hover:not(.active) { border-color: var(--accent,#6366f1); color: var(--accent,#6366f1); }
        [data-theme="dark"] .gallery-view-btn { border-color: #2d3f55; color: #94a3b8; }
        /* Gallery list-view styles */
        .gallery-list { display: flex; flex-direction: column; gap: 6px; width: 100%; }
        .gallery-list-item { display: flex; align-items: center; gap: 12px; padding: 8px 10px; border-radius: 8px; border: 1px solid #e2e8f0; background: #f8fafc; cursor: pointer; transition: background 0.15s; }
        .gallery-list-item:hover { background: #f1f5f9; }
        .gallery-list-thumb { position: relative; width: 72px; height: 72px; flex-shrink: 0; border-radius: 6px; overflow: hidden; background: #e2e8f0; display: flex; align-items: center; justify-content: center; }
        .gallery-list-thumb img { width: 100%; height: 100%; object-fit: contain; border-radius: 6px; display: block; }
        .gallery-list-meta { flex: 1; min-width: 0; }
        .gallery-list-row1 { display: flex; align-items: center; gap: 6px; min-width: 0; }
        .gallery-list-model { font-size: 0.84rem; font-weight: 500; color: #1e293b; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }
        .gallery-list-steps { font-size: 0.72rem; color: #64748b; background: #e2e8f0; border-radius: 4px; padding: 1px 5px; white-space: nowrap; flex-shrink: 0; }
        .gallery-list-ts { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
        .gallery-list-prompt { font-size: 0.73rem; color: #475569; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
        .gallery-list-prompt.neg { color: #94a3b8; }
        .gallery-list-item.prompt-expanded .gallery-list-prompt { white-space: normal; overflow: visible; text-overflow: unset; word-break: break-word; }
        .gallery-list-item { cursor: default; }
        .gallery-list-thumb { cursor: pointer; }
        .prompt-diff-removed { color: #b91c1c; text-decoration: line-through; background: #fee2e2; border-radius: 2px; padding: 0 2px; }
        .prompt-diff-added { color: #15803d; background: #dcfce7; border-radius: 2px; padding: 0 2px; }
        .gallery-list-badges { display: flex; gap: 4px; margin-top: 4px; }
        .gallery-list-item.loading .gallery-list-thumb { background: linear-gradient(90deg,#f1f5f9 25%,#e2e8f0 50%,#f1f5f9 75%); background-size: 200% 100%; animation: gallery-shimmer 1.5s infinite; }
        [data-theme="dark"] .gallery-list-item { border-color: #2d3f55; background: #151e2e; }
        [data-theme="dark"] .gallery-list-item:hover { background: #1e293b; }
        [data-theme="dark"] .gallery-list-model { color: #e2e8f0; }
        [data-theme="dark"] .gallery-list-steps { background: #2d3f55; color: #64748b; }
        [data-theme="dark"] .gallery-list-ts { color: #94a3b8; }
        [data-theme="dark"] .gallery-list-prompt { color: #94a3b8; }
        [data-theme="dark"] .gallery-list-prompt.neg { color: #64748b; }
        [data-theme="dark"] .gallery-list-thumb { background: #1e293b; }
        [data-theme="dark"] .gallery-list-item.loading .gallery-list-thumb { background: linear-gradient(90deg,#1e293b 25%,#2d3f55 50%,#1e293b 75%); background-size: 200% 100%; animation: gallery-shimmer 1.5s infinite; }
        [data-theme="dark"] .prompt-diff-removed { color: #f87171; background: #450a0a; }
        [data-theme="dark"] .prompt-diff-added { color: #86efac; background: #052e16; }

        .last-image-container { display: flex; align-items: center; justify-content: center; border-radius: 8px; height: 400px; overflow: hidden; }
        .last-image-container.loading { background: linear-gradient(90deg, #f1f5f9 25%, #e2e8f0 50%, #f1f5f9 75%); background-size: 200% 100%; animation: gallery-shimmer 1.5s infinite; }
        [data-theme="dark"] .last-image-container.loading { background: linear-gradient(90deg, #1e293b 25%, #2d3f55 50%, #1e293b 75%); background-size: 200% 100%; animation: gallery-shimmer 1.5s infinite; }
        @media (prefers-reduced-motion: reduce) { .last-image-container.loading { animation: none; background: #e2e8f0; background-size: auto; } [data-theme="dark"] .last-image-container.loading { animation: none; background: #1e293b; background-size: auto; } }
        .last-image-container > .image-grid-item { aspect-ratio: auto; min-height: 0; height: 100%; display: flex; align-items: center; justify-content: center; overflow: hidden; }
        .last-image-container .image-grid-item img { max-width: 100%; height: 100%; max-height: 100%; object-fit: contain; border-radius: 4px; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; display: block; }
        .last-image-container .image-grid-item img:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .single-image { max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); display: block; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }
        .single-image:hover { transform: scale(1.02); box-shadow: 0 4px 16px rgba(0,0,0,0.18); }

        /* ---- Image overlay ---- */
        .image-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.92); z-index: 1000; justify-content: center; align-items: center; padding: 20px; }
        .image-overlay.active { display: flex; }
        .image-overlay-content { position: relative; max-width: 95%; max-height: 95%; display: flex; justify-content: center; align-items: center; }
        .image-overlay img { max-width: 100%; max-height: 90vh; object-fit: contain; border-radius: 8px; box-shadow: 0 8px 40px rgba(0,0,0,0.6); transition: opacity 0.2s ease; }
        .image-overlay-close { position: absolute; top: -44px; right: 0; background: var(--accent); color: white; border: none; padding: 8px 18px; font-size: 0.9rem; font-weight: 600; border-radius: 8px; cursor: pointer; transition: background 0.2s; }
        .image-overlay-close:hover { background: var(--accent-hover); }
        .image-overlay-nav { position: fixed; top: 50%; transform: translateY(-50%); background: rgba(0,0,0,0.5); color: white; border: none; padding: 12px 18px; font-size: 1.8rem; font-weight: 700; border-radius: 8px; cursor: pointer; transition: background 0.2s; z-index: 1001; user-select: none; line-height: 1; display: none; }
        .image-overlay-nav:hover { background: rgba(0,0,0,0.85); }
        .image-overlay-nav:disabled { opacity: 0.3; cursor: default; }
        .image-overlay-nav.prev { left: 12px; }
        .image-overlay-nav.next { right: 12px; }
        .image-overlay-counter { position: absolute; bottom: -32px; left: 50%; transform: translateX(-50%); color: rgba(255,255,255,0.8); font-size: 0.85rem; white-space: nowrap; font-weight: 500; }
        .image-overlay-loading { display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 1002; pointer-events: none; }
        .image-overlay-loading .loading-spinner { width: 52px; height: 52px; border-width: 4px; border-color: rgba(255,255,255,0.2); border-top-color: #fff; }
        [data-theme="dark"] .image-overlay-loading .loading-spinner { border-color: rgba(255,255,255,0.2); border-top-color: #fff; }
        .image-overlay-content.is-loading .image-overlay-loading { display: block; }
        .image-overlay-content.is-loading #overlay-image[src]:not([src=""]) { opacity: 0.35; }
        .image-overlay-content.is-loading #overlay-image[src=""], .image-overlay-content.is-loading #overlay-image:not([src]) { visibility: hidden; min-width: 0; min-height: 0; }
        @media (prefers-reduced-motion: reduce) { .image-overlay-loading .loading-spinner { animation: none; } }

        /* ---- Errors ---- */
        .errors-outer { display: flex; flex-direction: column; }
        .errors-list { display: flex; flex-direction: column; flex: 1; min-height: 0; overflow-y: auto; }
        .error-item { background: #fff5f5; border: 1px solid #fecaca; border-left: 3px solid var(--error); border-radius: 6px; padding: 9px 13px; font-family: 'Courier New', monospace; font-size: 0.78rem; color: #7f1d1d; white-space: pre-wrap; word-break: break-word; margin-bottom: 5px; flex-shrink: 0; }
        .error-item:last-child { margin-bottom: 0; }
        .error-group { background: #fff5f5; border: 1px solid #fecaca; border-left: 3px solid var(--error); border-radius: 6px; margin-bottom: 5px; overflow: hidden; flex-shrink: 0; }
        .error-group:last-child { margin-bottom: 0; }
        .error-group-header { display: flex; align-items: flex-start; gap: 8px; padding: 9px 13px; cursor: pointer; user-select: none; width: 100%; background: transparent; border: none; text-align: left; font: inherit; color: inherit; }
        .error-group-header:hover { background: rgba(239,68,68,0.06); }
        .error-group-msg { font-family: 'Courier New', monospace; font-size: 0.78rem; color: #7f1d1d; white-space: pre-wrap; word-break: break-word; flex: 1; min-width: 0; }
        .error-count-badge { flex-shrink: 0; background: var(--error); color: #fff; font-size: 0.7rem; font-weight: 700; border-radius: 10px; padding: 1px 7px; line-height: 1.6; margin-top: 1px; }
        .error-group-toggle { flex-shrink: 0; font-size: 0.7rem; color: #9ca3af; margin-top: 2px; transition: transform 0.15s; }
        .error-group.open .error-group-toggle { transform: rotate(90deg); }
        .error-group-body { display: none; border-top: 1px solid #fecaca; padding: 6px 13px; }
        .error-group.open .error-group-body { display: block; }
        .error-occurrence { font-family: 'Courier New', monospace; font-size: 0.75rem; color: #991b1b; padding: 3px 0; border-bottom: 1px solid #fecaca; white-space: pre-wrap; word-break: break-word; }
        .error-occurrence:last-child { border-bottom: none; }
        .error-occurrence-more { font-size: 0.72rem; color: #9ca3af; padding: 3px 0; font-style: italic; }
        .errors-view-toggle { display: flex; gap: 4px; }
        .errors-view-btn { background: transparent; border: 1px solid var(--border); border-radius: 5px; padding: 3px 10px; font-size: 0.75rem; cursor: pointer; color: var(--text-muted); transition: background 0.15s, color 0.15s, border-color 0.15s; }
        .errors-view-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
        .errors-view-btn:hover:not(.active) { border-color: var(--accent); color: var(--accent); }
        .errors-clear-btn { background: #e2e8f0; color: #475569; border: 1px solid transparent; border-radius: 6px; padding: 3px 10px; font-size: 0.75rem; font-weight: 600; cursor: pointer; transition: background 0.15s, color 0.15s; margin-left: 6px; }
        .errors-clear-btn:hover { background: #fecaca; color: #dc2626; }
        .errors-clear-btn:disabled { opacity: 0.5; cursor: default; }
        [data-theme="dark"] .errors-clear-btn { background: #2d3f55; color: #94a3b8; }
        [data-theme="dark"] .errors-clear-btn:hover { background: #7f1d1d; color: #fca5a5; }

        .pagination-controls { display: flex; align-items: center; justify-content: center; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
        .pagination-controls button { background: var(--accent); color: white; border: none; border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 0.82rem; font-weight: 500; transition: background 0.15s; }
        .pagination-controls button:hover:not(:disabled) { background: var(--accent-hover); }
        .pagination-controls button:disabled { background: #c7d2fe; cursor: default; }
        .pagination-info { font-size: 0.82rem; color: var(--text-muted); }
        .page-size-select { font-size: 0.82rem; color: inherit; background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; cursor: pointer; transition: border-color 0.15s; }
        .page-size-select:focus-visible { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(99,102,241,0.35); }

        .scrollable { max-height: 260px; overflow-y: auto; }
        .scrollable-tall { max-height: 400px; overflow-y: auto; }

        #loading { display: flex; align-items: center; justify-content: center; height: 80vh; flex-direction: column; gap: 14px; }
        .loading-spinner { width: 36px; height: 36px; border: 3px solid #e2e8f0; border-top-color: var(--accent); border-radius: 50%; animation: spin 0.75s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text { color: #64748b; font-size: 0.9rem; }

        .empty-state { text-align: center; padding: 24px 16px; color: #94a3b8; font-size: 0.87rem; }
        .empty-state-icon { font-size: 1.8rem; margin-bottom: 6px; display: block; }
        .centered-empty-container { display: flex; align-items: center; justify-content: center; height: 400px; }
        #overview-current-job { height: 400px; overflow: hidden; }

        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

        @media (max-width: 1200px) { .grid-4 { grid-template-columns: repeat(2, 1fr); } .grid-3 { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 768px) { .sidebar { transform: translateX(-100%); top: 90px; height: calc(100vh - 90px); } .sidebar.open { transform: translateX(0); } .mobile-navbar { display: flex; } .mobile-resources { display: grid; grid-template-columns: repeat(4, 1fr); } .main-content { margin-left: 0; padding-top: 90px; } .topbar { display: none; } .content-area { padding: 14px 12px; } .grid-4 { grid-template-columns: repeat(2, 1fr); } .grid-3 { grid-template-columns: 1fr; } .grid-2 { grid-template-columns: 1fr; } .grid-3-popped { grid-template-columns: repeat(2, 1fr); } .overview-bottom-grid-left { grid-row: span 1; } }
        @media (max-width: 480px) { .grid-4 { grid-template-columns: repeat(2, 1fr); gap: 10px; } .stat-card-value { font-size: 1.4rem; } }
        @media (max-width: 768px) { .stats-charts-grid { grid-template-columns: 1fr; } }

        /* ---- Theme toggle (square) ---- */
        .theme-toggle { background: none; border: 1px solid rgba(255,255,255,0.18); color: var(--text-light); font-size: 1rem; cursor: pointer; padding: 5px 9px; border-radius: 4px; line-height: 1; transition: background 0.15s; flex-shrink: 0; }
        .theme-toggle:hover { background: rgba(255,255,255,0.08); }
        .topbar .theme-toggle { background: none; border: 1px solid #e2e8f0; color: #475569; width: var(--action-btn-height); padding: 0; flex-shrink: 0; }
        .topbar .theme-toggle:hover { background: #f1f5f9; }
        .nsfw-blur-btn { background: none; border: 1px solid #e2e8f0; color: #475569; font-size: 0.78rem; font-weight: 600; cursor: pointer; padding: 5px 10px; border-radius: 4px; white-space: nowrap; flex-shrink: 0; transition: background 0.15s; }
        .nsfw-blur-btn:hover { background: #f1f5f9; }
        .nsfw-blur-btn.active { background: #fef9c3; border-color: #ca8a04; color: #854d0e; }
        [data-theme="dark"] .nsfw-blur-btn { border-color: #2d3f55; color: #94a3b8; }
        [data-theme="dark"] .nsfw-blur-btn:hover { background: #2d3f55; }
        [data-theme="dark"] .nsfw-blur-btn.active { background: #422006; border-color: #d97706; color: #fcd34d; }
        body.blur-nsfw .image-grid-item[data-nsfw="1"] img { filter: blur(14px); }
        body.blur-nsfw img[data-nsfw="1"] { filter: blur(14px); }
        /* Also blur images inside ANY element flagged data-nsfw (e.g. the single-image last-result
           container wraps the img in a <div data-nsfw="1"> when an NSFW/CSAM badge is shown, so the
           img itself has no data-nsfw attribute and the rules above would miss it). */
        body.blur-nsfw [data-nsfw="1"] img { filter: blur(14px); }

        /* ---- Topbar resource pills with bars ---- */
        .topbar-resources { display: flex; align-items: center; align-self: center; gap: 8px; flex-wrap: wrap; }
        .topbar-res-pill { background: #f1f5f9; border: 1px solid #e2e8f0; color: #475569; font-size: 0.72rem; font-weight: 600; padding: 4px 10px; border-radius: 8px; white-space: nowrap; display: flex; flex-direction: column; gap: 3px; width: 130px; flex-shrink: 0; }
        .topbar-res-pill-label { display: flex; justify-content: space-between; align-items: center; font-family: 'Courier New', monospace; }
        .topbar-res-pill-sub { display: flex; justify-content: space-between; align-items: center; font-family: 'Courier New', monospace; font-size: 0.67rem; opacity: 0.75; margin-top: 1px; }
        .topbar-res-bar-track { width: 100%; height: 4px; background: #cbd5e1; border-radius: 2px; overflow: hidden; position: relative; }
        .topbar-res-bar { height: 100%; border-radius: 2px; transition: width 0.4s ease, background-color 0.4s ease; position: absolute; left: 0; top: 0; }
        .topbar-res-bar-back { opacity: 0.4; }
        /* ---- Dark mode ---- */
        [data-theme="dark"] { --main-bg: #0f172a; --card-bg: #1e293b; --border: #2d3f55; --sidebar-bg: #0d1117; --sidebar-hover: #161e2e; --sidebar-text: #94a3b8; --sidebar-text-strong: #e2e8f0; --sidebar-border: rgba(255,255,255,0.07); }
        [data-theme="dark"] body { color: #cbd5e1; }
        [data-theme="dark"] .topbar { background: #1e293b; border-bottom-color: #2d3f55; }
        [data-theme="dark"] .topbar-worker-name { color: #f1f5f9; }
        [data-theme="dark"] .topbar-worker-sub { color: #94a3b8; }
        [data-theme="dark"] .topbar-uptime { color: #94a3b8; background: #151e2e; border-color: #2d3f55; }
        [data-theme="dark"] .topbar .theme-toggle { border-color: #2d3f55; color: #94a3b8; }
        [data-theme="dark"] .topbar .theme-toggle:hover { background: #2d3f55; }
        [data-theme="dark"] .topbar-res-pill { background: #151e2e; border-color: #2d3f55; color: #94a3b8; }
        [data-theme="dark"] .topbar-res-bar-track { background: #2d3f55; }
        [data-theme="dark"] .stat-card-value:not(.success):not(.accent):not(.warning):not(.error) { color: #f1f5f9; }
        [data-theme="dark"] .stat-card-value.success { color: #34d399; }
        [data-theme="dark"] .stat-card-value.accent  { color: #818cf8; }
        [data-theme="dark"] .stat-card-value.warning { color: #fbbf24; }
        [data-theme="dark"] .stat-card-value.error   { color: #f87171; }
        [data-theme="dark"] .stat-card-label { color: #94a3b8; }
        [data-theme="dark"] .stat-label { color: #94a3b8; }
        [data-theme="dark"] .stat-value { color: #f1f5f9; }
        [data-theme="dark"] .stat-row { border-bottom-color: #2d3f55; }
        [data-theme="dark"] .card-header { border-bottom-color: #2d3f55; }
        [data-theme="dark"] .card-header-count { color: #cbd5e1; }
        [data-theme="dark"] #queue-count, [data-theme="dark"] #models-count { color: #60a5fa; }
        [data-theme="dark"] #queue-max,   [data-theme="dark"] #models-max   { color: #94a3b8; }
        [data-theme="dark"] .card-title { color: #94a3b8; }
        [data-theme="dark"] .progress-label { color: #94a3b8; }
        [data-theme="dark"] .progress-value { color: #f1f5f9; }
        [data-theme="dark"] .progress-bar-container { background: #2d3f55; }
        [data-theme="dark"] .section-title { color: #94a3b8; }
        [data-theme="dark"] .section-count { background: #2d3f55; color: #94a3b8; }
        [data-theme="dark"] .process-item { background: #151e2e; border-color: #2d3f55; }
        [data-theme="dark"] .process-type-badge { background: #312e81; color: #a5b4fc; }
        [data-theme="dark"] .process-state-badge { background: #14532d; color: #86efac; }
        [data-theme="dark"] .process-detail-text { color: #94a3b8; }
        [data-theme="dark"] .job-item { background: #151e2e; border-color: #2d3f55; }
        [data-theme="dark"] .model-badge { background: #312e81; color: #a5b4fc; }
        [data-theme="dark"] .job-state-badge { background: #14532d; color: #86efac; }
        [data-theme="dark"] .loading-text { color: #94a3b8; }
        [data-theme="dark"] .loading-spinner { border-color: #2d3f55; border-top-color: var(--accent); }
        [data-theme="dark"] .empty-state { color: #64748b; }
        [data-theme="dark"] .image-grid-item { background: #151e2e; }
        [data-theme="dark"] .page-size-select { color: #cbd5e1; }
        [data-theme="dark"] .console-filter-select { color: #cbd5e1; }
        [data-theme="dark"] .error-item { background: #1a1010; border-color: #7f1d1d; color: #fca5a5; }
        [data-theme="dark"] .error-group { background: #1a1010; border-color: #7f1d1d; }
        [data-theme="dark"] .error-group-msg { color: #fca5a5; }
        [data-theme="dark"] .error-group-body { border-top-color: #7f1d1d; }
        [data-theme="dark"] .error-occurrence { color: #fca5a5; border-bottom-color: #7f1d1d; }
        [data-theme="dark"] .error-group-header:hover { background: rgba(239,68,68,0.08); }

        /* ---- Worker cards (User page) ---- */
        .worker-card { background: var(--card-bg); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; }
        .worker-card:last-child { margin-bottom: 0; }
        .worker-card-header { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
        .worker-card-name { font-weight: 700; color: var(--accent); font-size: 0.95rem; flex-shrink: 0; }
        .worker-version-badge { font-size: 0.68rem; background: #e0e7ff; color: #4338ca; padding: 2px 7px; border-radius: 4px; font-weight: 600; font-family: 'Courier New', monospace; }
        .worker-type-badge { font-size: 0.68rem; background: #f0fdf4; color: #166534; padding: 2px 7px; border-radius: 4px; font-weight: 600; text-transform: capitalize; }
        .worker-online-badge { font-size: 0.68rem; padding: 2px 7px; border-radius: 4px; font-weight: 600; margin-left: auto; }
        .worker-online-badge.online { background: #dcfce7; color: #166534; }
        .worker-online-badge.offline { background: #fee2e2; color: #991b1b; }
        .worker-caps-row { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }
        .wcap { font-size: 0.68rem; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
        .wcap-yes { background: #dcfce7; color: #166534; }
        .wcap-no { background: #f1f5f9; color: #64748b; }
        .wcap-nsfw { background: #fef3c7; color: #92400e; }
        .wcap-sfw { background: #f1f5f9; color: #64748b; }
        .worker-meta-row { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; font-size: 0.82rem; color: #475569; }
        .wm-item { display: flex; align-items: center; gap: 4px; }
        .models-pill { cursor: default; text-decoration: underline dotted; }
        .models-tooltip { display: none; position: fixed; background: #334155; color: #f1f5f9; padding: 8px 12px; border-radius: 6px; font-size: 0.78rem; line-height: 1.6; z-index: 1000; pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,0.3); border: 1px solid #475569; white-space: normal; overflow-wrap: normal; word-break: normal; min-width: min(420px, 95vw); max-width: min(95vw, 820px); }
        [data-theme="dark"] .models-tooltip { background: #1e293b; border-color: #334155; }
        .worker-stats-row { display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.82rem; color: #64748b; border-top: 1px solid var(--border); padding-top: 8px; margin-top: 2px; }
        .ws-item { display: flex; align-items: center; gap: 4px; }
        .ws-item.accent { color: var(--accent); font-weight: 600; }
        [data-theme="dark"] .worker-version-badge { background: #312e81; color: #a5b4fc; }
        [data-theme="dark"] .worker-type-badge { background: #14532d; color: #86efac; }
        [data-theme="dark"] .worker-online-badge.online { background: #14532d; color: #86efac; }
        [data-theme="dark"] .worker-online-badge.offline { background: #450a0a; color: #fca5a5; }
        [data-theme="dark"] .wcap-yes { background: #14532d; color: #86efac; }
        [data-theme="dark"] .wcap-no { background: #1e293b; color: #64748b; }
        [data-theme="dark"] .wcap-nsfw { background: #451a03; color: #fcd34d; }
        [data-theme="dark"] .wcap-sfw { background: #1e293b; color: #64748b; }
        [data-theme="dark"] .worker-meta-row { color: #94a3b8; }
        [data-theme="dark"] .worker-stats-row { color: #94a3b8; }
        [data-theme="dark"] .worker-card { background: #151e2e; border-color: #2d3f55; }

        /* ---- Worker delete button ---- */
        .worker-delete-btn { background: none; border: 1px solid #fca5a5; color: #dc2626; border-radius: 6px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; cursor: pointer; transition: background 0.15s, color 0.15s; margin-left: 6px; flex-shrink: 0; }
        .worker-delete-btn:hover { background: #fee2e2; }
        .worker-delete-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        [data-theme="dark"] .worker-delete-btn { border-color: #7f1d1d; color: #fca5a5; }
        [data-theme="dark"] .worker-delete-btn:hover { background: #450a0a; }

        /* ---- Gallery new-images banner ---- */
        #gallery-new-banner { display: none; background: #dbeafe; border: 1px solid #93c5fd; border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; cursor: pointer; font-size: 0.85rem; font-weight: 500; color: #1d4ed8; }
        [data-theme="dark"] #gallery-new-banner { background: #1e3a5f; border-color: #2d5fa0; color: #93c5fd; }

        /* ---- Statistics page ---- */
        .stats-window-group { display: flex; gap: 4px; }
        .stats-window-btn { background: transparent; border: 1px solid var(--border); border-radius: 5px; padding: 3px 11px; font-size: 0.78rem; font-weight: 600; cursor: pointer; color: var(--text-muted); transition: background 0.15s, color 0.15s, border-color 0.15s; }
        .stats-window-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
        .stats-window-btn:hover:not(.active) { border-color: var(--accent); color: var(--accent); }
        .chart-container { position: relative; width: 100%; height: 150px; overflow: hidden; }
        .chart-container canvas { display: block; max-width: 100%; }
        .chart-container-sm { position: relative; width: 100%; height: 110px; overflow: hidden; }
        .chart-container-sm canvas { display: block; max-width: 100%; }
        .chart-container-md { position: relative; width: 100%; height: 160px; overflow: hidden; }
        .chart-container-md canvas { display: block; max-width: 100%; }
        .chart-label { font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
        [data-theme="dark"] .chart-label { color: #94a3b8; }
        .chart-legend { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 8px; }
        .chart-legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.73rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.6px; }
        [data-theme="dark"] .chart-legend-item { color: #94a3b8; }
        .chart-legend-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
        .stats-model-table-wrap { max-height: 300px; overflow-y: auto; }
        .model-images-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        .model-images-table thead th { position: sticky; top: 0; background: var(--card-bg); z-index: 1; }
        .model-images-table th { text-align: left; font-weight: 700; color: var(--text-muted); padding: 5px 8px 7px 8px; border-bottom: 1px solid var(--border); }
        .model-images-table th:last-child { text-align: right; }
        .model-images-table td { padding: 5px 8px; border-bottom: 1px solid var(--border); }
        .model-images-table td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
        .model-images-table tr:last-child td { border-bottom: none; }
        .model-images-bar-cell { width: 40%; }
        .model-images-bar-wrap { background: var(--border); border-radius: 3px; height: 7px; overflow: hidden; }
        .model-images-bar { background: var(--accent); height: 7px; border-radius: 3px; min-width: 2px; transition: width 0.3s; }
        .model-failed-bar { background: var(--error); height: 7px; border-radius: 3px; min-width: 2px; transition: width 0.3s; }
        #stats-model-tables-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        @media (max-width: 600px) { #stats-model-tables-grid { grid-template-columns: 1fr; } }
        #stats-job-state-model-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        @media (max-width: 600px) { #stats-job-state-model-row { grid-template-columns: 1fr; } }
        #stats-model-time-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        @media (max-width: 600px) { #stats-model-time-row { grid-template-columns: 1fr; } }

        /* ---- Horde Network page ---- */
        .horde-window-btn { background: transparent; border: 1px solid var(--border); border-radius: 5px; padding: 3px 11px; font-size: 0.78rem; font-weight: 600; cursor: pointer; color: var(--text-muted); transition: background 0.15s, color 0.15s, border-color 0.15s; }
        .horde-window-btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
        .horde-window-btn:hover:not(.active) { border-color: var(--accent); color: var(--accent); }
        .horde-status-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-left: auto; }
        .horde-mode-badge { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 12px; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.5px; border: 1px solid transparent; }
        .horde-mode-badge.ok { background: #d1fae5; color: #065f46; border-color: #6ee7b7; }
        .horde-mode-badge.warning { background: #fef3c7; color: #92400e; border-color: #fcd34d; }
        [data-theme="dark"] .horde-mode-badge.ok { background: #064e3b; color: #6ee7b7; border-color: #065f46; }
        [data-theme="dark"] .horde-mode-badge.warning { background: #451a03; color: #fcd34d; border-color: #92400e; }

        /* ---- Settings page ---- */
        .settings-header-actions { margin-left: auto; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
        .settings-page-btn { padding: 6px 11px; font-size: 0.78rem; font-weight: 700; border-radius: 6px; cursor: pointer; border: 1px solid transparent; transition: background 0.15s, border-color 0.15s, color 0.15s, opacity 0.15s; }
        .settings-page-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .settings-page-btn.apply { background: var(--accent); color: #fff; }
        .settings-page-btn.apply:hover:not(:disabled) { background: var(--accent-hover); }
        .settings-page-btn.apply.dirty { box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2); }
        .settings-page-btn.restart { background: transparent; border-color: #f59e0b; color: #b45309; }
        .settings-page-btn.restart:hover:not(:disabled) { background: #fffbeb; }
        [data-theme="dark"] .settings-page-btn.restart { border-color: #92400e; color: #fcd34d; }
        [data-theme="dark"] .settings-page-btn.restart:hover:not(:disabled) { background: #2b1a06; }
        .settings-group { margin-bottom: var(--page-spacing); }
        .settings-group-title { font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
        [data-theme="dark"] .settings-group-title { color: #94a3b8; }
        .settings-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; }
        @media (max-width: 480px) { .settings-grid { grid-template-columns: 1fr; } .setting-row { flex-wrap: wrap; align-items: flex-start; } .setting-ctrl { margin-left: 0; } .setting-replace-list { min-width: 0; } .setting-textarea { width: 100%; max-width: 100%; box-sizing: border-box; } }
        .setting-row { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; display: flex; align-items: center; gap: 10px; transition: border-color 0.15s; }
        .setting-row:hover { border-color: #cbd5e1; }
        [data-theme="dark"] .setting-row:hover { border-color: #334155; }
        .setting-info { flex: 1; min-width: 0; }
        .setting-label { font-size: 0.85rem; font-weight: 600; color: #1e293b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        [data-theme="dark"] .setting-label { color: #e2e8f0; }
        .setting-desc { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
        [data-theme="dark"] .setting-desc { color: #94a3b8; }
        .setting-env { font-size: 0.7rem; color: #8b5cf6; margin-top: 1px; }
        .setting-env code { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; background: rgba(139, 92, 246, 0.08); padding: 1px 4px; border-radius: 3px; }
        [data-theme="dark"] .setting-env { color: #a78bfa; }
        [data-theme="dark"] .setting-env code { background: rgba(167, 139, 250, 0.12); }
        .setting-ctrl { flex-shrink: 0; margin-left: auto; display: flex; align-items: center; justify-content: flex-end; text-align: right; gap: 6px; }
        /* Toggle switch */
        .setting-toggle { position: relative; display: inline-block; width: 40px; height: 22px; }
        .setting-toggle input { opacity: 0; width: 0; height: 0; position: absolute; }
        .setting-toggle-slider { position: absolute; cursor: pointer; inset: 0; background: #cbd5e1; border-radius: 22px; transition: background 0.2s; }
        .setting-toggle-slider::before { content: ''; position: absolute; width: 16px; height: 16px; left: 3px; bottom: 3px; background: white; border-radius: 50%; transition: transform 0.2s; }
        .setting-toggle input:checked + .setting-toggle-slider { background: var(--accent); }
        .setting-toggle input:checked + .setting-toggle-slider::before { transform: translateX(18px); }
        .setting-toggle input:disabled + .setting-toggle-slider { opacity: 0.5; cursor: not-allowed; }
        [data-theme="dark"] .setting-toggle-slider { background: #334155; }
        [data-theme="dark"] .setting-toggle input:checked + .setting-toggle-slider { background: var(--accent); }
        /* Number input */
        .setting-reset-btn { display: inline-flex; align-items: center; justify-content: center; width: var(--action-btn-height); height: var(--action-btn-height); padding: 0; border: 1px solid transparent; border-radius: 5px; background: transparent; color: var(--text-muted); font-size: 0.9rem; line-height: 1; cursor: pointer; transition: background 0.12s, color 0.12s, border-color 0.12s; flex-shrink: 0; }
        .setting-reset-btn:not(:disabled):hover { background: rgba(99,102,241,0.1); color: var(--accent); border-color: var(--accent); }
        .setting-reset-btn:disabled { opacity: 0.22; cursor: not-allowed; }
        .settings-page-btn.reset-all { background: transparent; border-color: var(--border); color: var(--text-muted); }
        .settings-page-btn.reset-all:hover:not(:disabled) { background: #fee2e2; color: #b91c1c; border-color: #f87171; }
        [data-theme="dark"] .settings-page-btn.reset-all:hover:not(:disabled) { background: #3b0a0a; color: #fca5a5; border-color: #b91c1c; }
        .settings-page-btn.reset-db { background: transparent; border-color: #ef4444; color: #b91c1c; }
        .settings-page-btn.reset-db:hover:not(:disabled) { background: #fee2e2; color: #991b1b; border-color: #dc2626; }
        [data-theme="dark"] .settings-page-btn.reset-db { border-color: #7f1d1d; color: #fca5a5; }
        [data-theme="dark"] .settings-page-btn.reset-db:hover:not(:disabled) { background: #3b0a0a; color: #fecaca; border-color: #b91c1c; }
        .setting-number { width: 68px; height: var(--action-btn-height); padding: 4px 7px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.83rem; text-align: center; background: #f8fafc; color: #1e293b; transition: border-color 0.15s; }
        .setting-textarea { width: 200px; min-height: 54px; max-height: 180px; padding: 5px 8px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.83rem; background: #f8fafc; color: #1e293b; resize: vertical; transition: border-color 0.15s; font-family: inherit; line-height: 1.5; }
        .setting-textarea:focus { outline: none; border-color: var(--accent); }
        [data-theme="dark"] .setting-textarea { background: #1e293b; border-color: #334155; color: #e2e8f0; }
        .pf-block { grid-column: 1 / -1; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }
        .pf-block:hover { border-color: #cbd5e1; }
        [data-theme="dark"] .pf-block:hover { border-color: #334155; }
        .pf-block-title { font-size: 0.88rem; font-weight: 700; color: #1e293b; margin-bottom: 3px; }
        [data-theme="dark"] .pf-block-title { color: #e2e8f0; }
        .pf-block-desc { font-size: 0.75rem; color: #64748b; margin-bottom: 12px; }
        [data-theme="dark"] .pf-block-desc { color: #94a3b8; }
        .pf-columns { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
        @media (max-width: 700px) { .pf-columns { grid-template-columns: 1fr; } }
        .pf-col { display: flex; flex-direction: column; }
        .pf-section-label { font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 5px; }
        [data-theme="dark"] .pf-section-label { color: #94a3b8; }
        .pf-pills { display: flex; flex-wrap: wrap; gap: 5px; min-height: 22px; margin-bottom: 5px; flex: 1; align-content: flex-start; }
        .pf-pill { display: inline-flex; align-items: center; padding: 2px 9px; background: #dbeafe; color: #1d4ed8; border-radius: 10px; font-size: 0.78rem; cursor: pointer; white-space: nowrap; max-width: 100%; overflow: hidden; text-overflow: ellipsis; transition: background 0.13s, color 0.13s; }
        .pf-pill:hover { background: #fecaca; color: #dc2626; }
        .pf-pill--replace { background: #fef9c3; color: #92400e; }
        .pf-pill--replace:hover { background: #fecaca; color: #dc2626; }
        [data-theme="dark"] .pf-pill { background: #1e3a5f; color: #93c5fd; }
        [data-theme="dark"] .pf-pill:hover { background: #450a0a; color: #fca5a5; }
        [data-theme="dark"] .pf-pill--replace { background: #3b2600; color: #fde68a; }
        [data-theme="dark"] .pf-pill--replace:hover { background: #450a0a; color: #fca5a5; }
        .pf-input-row { display: flex; align-items: center; gap: 5px; margin-top: auto; }
        .pf-input { flex: 1; min-width: 0; padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 5px; font-size: 0.83rem; background: #f8fafc; color: #1e293b; font-family: inherit; height: var(--action-btn-height); box-sizing: border-box; }
        .pf-input:focus { outline: none; border-color: var(--accent); }
        [data-theme="dark"] .pf-input { background: #1e293b; border-color: #334155; color: #e2e8f0; }
        .pf-add-btn { width: var(--action-btn-height); height: var(--action-btn-height); border: 1px solid #cbd5e1; border-radius: 5px; background: transparent; color: #64748b; cursor: pointer; font-size: 1rem; line-height: 1; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
        .pf-add-btn:hover { border-color: var(--accent); color: var(--accent); background: rgba(99,102,241,0.07); }
        [data-theme="dark"] .pf-add-btn { border-color: #334155; color: #94a3b8; }
        [data-theme="dark"] .pf-add-btn:hover { border-color: var(--accent); color: var(--accent); }
        .pf-options-row { display: flex; flex-wrap: wrap; gap: 10px 28px; }
        .pf-option { display: flex; align-items: center; gap: 9px; }
        .pf-option-text { display: flex; flex-direction: column; gap: 1px; }
        .pf-option-label { font-size: 0.83rem; font-weight: 500; color: #334155; }
        [data-theme="dark"] .pf-option-label { color: #cbd5e1; }
        .pf-option-desc { font-size: 0.72rem; color: #64748b; }
        [data-theme="dark"] .pf-option-desc { color: #94a3b8; }
        /* Prompt filter group UI */
        .pfg-section { display: flex; flex-direction: column; }
        .pfg-section-header { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; min-height: 24px; }
        .pfg-section-header .pf-section-label { margin-bottom: 0; }
        .pfg-add-group-btn { margin-left: auto; padding: 2px 8px; font-size: 0.73rem; border: 1px solid #cbd5e1; border-radius: 5px; background: transparent; color: #64748b; cursor: pointer; white-space: nowrap; flex-shrink: 0; }
        .pfg-add-group-btn:hover { border-color: var(--accent); color: var(--accent); background: rgba(99,102,241,0.07); }
        [data-theme="dark"] .pfg-add-group-btn { border-color: #334155; color: #94a3b8; }
        [data-theme="dark"] .pfg-add-group-btn:hover { border-color: var(--accent); color: var(--accent); }
        .pfg-groups { display: flex; flex-direction: column; gap: 5px; }
        .pfg-group { border: 1px solid var(--border); border-radius: 6px; padding: 7px 9px; background: var(--bg); display: flex; flex-direction: column; gap: 5px; }
        .pfg-group--disabled { opacity: 0.55; }
        .pfg-group-header { display: flex; align-items: center; gap: 5px; }
        .pfg-name-input { flex: 1; min-width: 0; padding: 2px 6px; border: 1px solid transparent; border-radius: 4px; font-size: 0.8rem; background: transparent; color: inherit; font-family: inherit; }
        .pfg-name-input:hover { border-color: var(--border); background: var(--card-bg); }
        .pfg-name-input:focus { border-color: var(--accent); background: var(--card-bg); outline: none; }
        [data-theme="dark"] .pfg-name-input:hover, [data-theme="dark"] .pfg-name-input:focus { background: #1e293b; }
        .pfg-delete-btn { width: 20px; height: 20px; border: 1px solid #fca5a5; border-radius: 4px; background: transparent; color: #ef4444; cursor: pointer; font-size: 1rem; line-height: 1; padding: 0; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
        .pfg-delete-btn:hover { background: #fef2f2; }
        [data-theme="dark"] .pfg-delete-btn { border-color: #7f1d1d; color: #f87171; }
        [data-theme="dark"] .pfg-delete-btn:hover { background: #450a0a; }
        .pfg-section--full { flex: 1 1 100%; }
        .setting-replace-list { display: flex; flex-direction: column; gap: 5px; min-width: 280px; }
        .replace-row { display: flex; align-items: center; gap: 5px; }
        .replace-find, .replace-with { flex: 1; min-width: 0; padding: 4px 7px; border: 1px solid #cbd5e1; border-radius: 5px; font-size: 0.83rem; background: #f8fafc; color: #1e293b; font-family: inherit; height: var(--action-btn-height); box-sizing: border-box; }
        .replace-find:focus, .replace-with:focus { outline: none; border-color: var(--accent); }
        [data-theme="dark"] .replace-find, [data-theme="dark"] .replace-with { background: #1e293b; border-color: #334155; color: #e2e8f0; }
        .replace-arrow { color: #94a3b8; font-size: 0.9rem; flex-shrink: 0; }
        .replace-row-del { width: 22px; height: 22px; border: 1px solid #fca5a5; border-radius: 4px; background: transparent; color: #ef4444; cursor: pointer; font-size: 1rem; line-height: 1; padding: 0; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
        .replace-row-del:hover { background: #fef2f2; }
        [data-theme="dark"] .replace-row-del { border-color: #7f1d1d; color: #f87171; }
        [data-theme="dark"] .replace-row-del:hover { background: #450a0a; }
        .replace-add-row { align-self: flex-start; padding: 3px 10px; border: 1px solid #cbd5e1; border-radius: 5px; background: transparent; color: #64748b; font-size: 0.8rem; cursor: pointer; margin-top: 1px; }
        .replace-add-row:hover { border-color: var(--accent); color: var(--accent); }
        [data-theme="dark"] .replace-add-row { border-color: #334155; color: #94a3b8; }
        .setting-number:disabled { opacity: 0.55; cursor: not-allowed; background: #e2e8f0; color: #64748b; }
        .setting-number:focus { outline: none; border-color: var(--accent); }
        [data-theme="dark"] .setting-number { background: #1e293b; border-color: #334155; color: #e2e8f0; }
        [data-theme="dark"] .setting-number:disabled { background: #0f1924; color: #64748b; }
        .setting-apply-btn { padding: 3px 10px; font-size: 0.78rem; font-weight: 600; background: var(--accent); color: #fff; border: none; border-radius: 5px; cursor: pointer; transition: background 0.15s; white-space: nowrap; }
        .setting-apply-btn:hover { background: var(--accent-hover); }
        .setting-apply-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        /* Inline feedback */
        .setting-feedback { font-size: 0.72rem; font-weight: 600; margin-left: 2px; min-width: 44px; transition: opacity 0.3s; }
        .setting-feedback.ok { color: var(--success); }
        .setting-feedback.err { color: var(--error); }
        .setting-feedback.pending { color: #b45309; }
        [data-theme="dark"] .setting-feedback.pending { color: #fcd34d; }
        .settings-unavailable { background: #f1f5f9; border: 1px solid var(--border); border-radius: 10px; padding: 28px 20px; text-align: center; color: #64748b; font-size: 0.92rem; }
        [data-theme="dark"] .settings-unavailable { background: #0f172a; color: #64748b; }
        /* Readonly string value */
        .setting-value { font-size: 0.82rem; color: #475569; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; background: #f1f5f9; border: 1px solid var(--border); border-radius: 5px; padding: 3px 8px; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: default; }
        [data-theme="dark"] .setting-value { color: #94a3b8; background: #0f172a; }
        /* Clickable URL readonly setting */
        .setting-url-link { color: #2563eb; text-decoration: none; cursor: pointer; }
        .setting-url-link:hover { text-decoration: underline; color: #1d4ed8; }
        [data-theme="dark"] .setting-url-link { color: #60a5fa; }
        [data-theme="dark"] .setting-url-link:hover { color: #93c5fd; }
        /* API reference info box */
        .api-ref-box { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }
        [data-theme="dark"] .api-ref-box { background: var(--card-bg); }
        .api-ref-endpoint { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
        .api-ref-method { font-size: 0.72rem; font-weight: 700; background: #dcfce7; color: #166534; border-radius: 4px; padding: 2px 8px; letter-spacing: 0.5px; flex-shrink: 0; }
        [data-theme="dark"] .api-ref-method { background: #14532d; color: #86efac; }
        .api-ref-url { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 0.82rem; color: #1e293b; background: #f1f5f9; border: 1px solid var(--border); border-radius: 5px; padding: 3px 8px; word-break: break-all; }
        [data-theme="dark"] .api-ref-url { color: #e2e8f0; background: #0f172a; }
        .api-ref-params-title { font-size: 0.75rem; font-weight: 700; color: #475569; margin-bottom: 6px; }
        [data-theme="dark"] .api-ref-params-title { color: #94a3b8; }
        .api-ref-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
        .api-ref-table th { text-align: left; padding: 4px 10px; color: #64748b; font-weight: 600; border-bottom: 1px solid var(--border); }
        .api-ref-table td { padding: 5px 10px; color: #334155; vertical-align: top; border-bottom: 1px solid #f1f5f9; }
        [data-theme="dark"] .api-ref-table th { color: #94a3b8; border-bottom-color: #334155; }
        [data-theme="dark"] .api-ref-table td { color: #cbd5e1; border-bottom-color: #1e293b; }
        .api-ref-table td:first-child { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; color: #7c3aed; font-weight: 600; }
        [data-theme="dark"] .api-ref-table td:first-child { color: #a78bfa; }
        .api-ref-table tr:last-child td { border-bottom: none; }
        .api-ref-badge { display: inline-block; font-size: 0.68rem; font-weight: 600; border-radius: 3px; padding: 1px 5px; margin-left: 4px; }
        .api-ref-badge.required { background: #fee2e2; color: #991b1b; }
        .api-ref-badge.optional { background: #e0f2fe; color: #075985; }
        [data-theme="dark"] .api-ref-badge.required { background: #450a0a; color: #fca5a5; }
        [data-theme="dark"] .api-ref-badge.optional { background: #082f49; color: #7dd3fc; }
        /* API page: method colour variants, per-endpoint blocks, group descriptions */
        .api-ref-method.get { background: #dbeafe; color: #1e40af; }
        [data-theme="dark"] .api-ref-method.get { background: #1e3a8a; color: #93c5fd; }
        .api-ref-method.delete { background: #fee2e2; color: #991b1b; }
        [data-theme="dark"] .api-ref-method.delete { background: #7f1d1d; color: #fca5a5; }
        .api-ref-intro { font-size: 0.82rem; color: var(--text-muted); line-height: 1.5; margin-bottom: var(--page-spacing); max-width: 760px; }
        .api-ref-group-desc { font-size: 0.8rem; color: var(--text-muted); margin: -4px 0 12px; }
        .api-ref-endpoint-desc { font-size: 0.82rem; color: #475569; margin-bottom: 8px; line-height: 1.45; }
        [data-theme="dark"] .api-ref-endpoint-desc { color: #94a3b8; }
        .api-ref-endpoint-block + .api-ref-endpoint-block { margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border); }
        /* About page */
        .about-hero { display: flex; gap: 18px; align-items: flex-start; background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px; margin-bottom: var(--page-spacing); }
        .about-hero-icon { font-size: 2.4rem; line-height: 1; flex-shrink: 0; }
        .about-hero-title { font-size: 1.05rem; font-weight: 700; color: var(--text); }
        .about-hero-sub { font-size: 0.8rem; color: var(--text-muted); margin: 2px 0 10px; }
        .about-hero-desc { font-size: 0.85rem; color: #475569; line-height: 1.55; max-width: 640px; }
        [data-theme="dark"] .about-hero-desc { color: #94a3b8; }
        .about-hero-desc a { color: #2563eb; text-decoration: none; }
        .about-hero-desc a:hover { text-decoration: underline; }
        [data-theme="dark"] .about-hero-desc a { color: #60a5fa; }
        .about-hero-links { display: flex; gap: 14px; margin-top: 12px; flex-wrap: wrap; }
        .about-link { font-size: 0.82rem; font-weight: 600; color: #2563eb; text-decoration: none; }
        .about-link:hover { text-decoration: underline; }
        [data-theme="dark"] .about-link { color: #60a5fa; }
        .about-tech-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
        .about-tech-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
        .about-tech-card-name { font-size: 0.85rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }
        .about-tech-card-desc { font-size: 0.78rem; color: #64748b; line-height: 1.45; }
        [data-theme="dark"] .about-tech-card-desc { color: #94a3b8; }
        .about-tech-card-tag { display: inline-block; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; color: #7c3aed; background: #ede9fe; border-radius: 3px; padding: 1px 6px; margin-bottom: 6px; }
        [data-theme="dark"] .about-tech-card-tag { color: #c4b5fd; background: #2e1065; }
        /* Models section */
        .models-containers { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
        @media (max-width: 640px) { .models-containers { grid-template-columns: 1fr; } }
        .models-box { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-height: 80px; }
        .models-box-title { font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
        [data-theme="dark"] .models-box-title { color: #94a3b8; }
        .models-box-title .models-count { font-weight: 400; opacity: 0.7; }
        .models-pills { display: flex; flex-wrap: wrap; gap: 6px; }
        .model-pill { display: inline-block; padding: 4px 10px; font-size: 0.78rem; font-weight: 500; border-radius: 14px; cursor: pointer; transition: background 0.15s, color 0.15s, transform 0.1s; user-select: none; font-family: inherit; }
        .model-pill:hover { transform: scale(1.04); }
        .model-pill:active { transform: scale(0.97); }
        .model-pill:focus-visible { outline: 2px solid #3b82f6; outline-offset: 1px; }
        .model-pill.enabled { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
        .model-pill.enabled:hover { background: #bbf7d0; }
        [data-theme="dark"] .model-pill.enabled { background: #14532d; color: #86efac; border-color: #22c55e; }
        [data-theme="dark"] .model-pill.enabled:hover { background: #166534; }
        .model-pill.disabled { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
        .model-pill.disabled:hover { background: #fecaca; }
        [data-theme="dark"] .model-pill.disabled { background: #7f1d1d; color: #fca5a5; border-color: #ef4444; }
        [data-theme="dark"] .model-pill.disabled:hover { background: #991b1b; }
        .models-empty { font-size: 0.78rem; color: #94a3b8; font-style: italic; }
        .confirm-modal-backdrop { display: none; position: fixed; inset: 0; background: rgba(15, 23, 42, 0.6); z-index: 1200; align-items: center; justify-content: center; padding: 18px; }
        .confirm-modal-backdrop.active { display: flex; }
        .confirm-modal { width: min(420px, 100%); background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; box-shadow: 0 22px 54px rgba(2, 6, 23, 0.35); padding: 16px; }
        .confirm-modal-title { font-size: 0.95rem; font-weight: 700; color: var(--text); margin-bottom: 6px; }
        .confirm-modal-body { font-size: 0.84rem; color: var(--text-muted); margin-bottom: 14px; line-height: 1.5; }
        .confirm-modal-actions { display: flex; justify-content: flex-end; gap: 8px; }
        .confirm-modal-btn { padding: 6px 12px; border-radius: 7px; border: 1px solid transparent; cursor: pointer; font-size: 0.8rem; font-weight: 600; transition: background 0.15s, border-color 0.15s; }
        .confirm-modal-btn.cancel { background: transparent; border-color: var(--border); color: var(--text-muted); }
        .confirm-modal-btn.cancel:hover { background: #f1f5f9; }
        .confirm-modal-btn.confirm { background: #b45309; color: #fff; }
        .confirm-modal-btn.confirm:hover { background: #92400e; }
        [data-theme="dark"] .confirm-modal-btn.cancel:hover { background: #1e293b; }
        [data-theme="dark"] .confirm-modal-btn.confirm { background: #d97706; }
        [data-theme="dark"] .confirm-modal-btn.confirm:hover { background: #b45309; }

    </style>
</head>
<body>
    <nav class="mobile-navbar" aria-label="Mobile navigation">
        <button class="hamburger-btn" onclick="toggleSidebar()" aria-label="Toggle sidebar">&#9776;</button>
        <span class="mobile-title">&#127912; Horde Worker</span>
        <span id="mobile-status-badge"></span>
        <span class="mobile-uptime" id="mobile-uptime">&#9201; --</span>
        <button class="theme-toggle" onclick="toggleTheme()" id="mobile-theme-toggle" aria-label="Toggle theme">&#127769;</button>
    </nav>
    <div class="mobile-resources" aria-label="Resource usage">
        <div class="mobile-res-col">
            <span class="mobile-res-head">CPU</span>
            <span class="mobile-res-chip" id="mobile-cpu-ctr">WRK 0%</span>
            <span class="mobile-res-chip mobile-res-chip-secondary" id="mobile-cpu">SYS 0%</span>
        </div>
        <div class="mobile-res-col">
            <span class="mobile-res-head">GPU</span>
            <span class="mobile-res-chip" id="mobile-gpu-wrk">WRK 0%</span>
            <span class="mobile-res-chip mobile-res-chip-secondary" id="mobile-gpu">SYS 0%</span>
        </div>
        <div class="mobile-res-col">
            <span class="mobile-res-head">VRAM</span>
            <span class="mobile-res-chip" id="mobile-vram">WRK 0%</span>
            <span class="mobile-res-chip mobile-res-chip-secondary" id="mobile-sysvram">SYS 0%</span>
        </div>
        <div class="mobile-res-col">
            <span class="mobile-res-head">RAM</span>
            <span class="mobile-res-chip" id="mobile-ram">WRK 0%</span>
            <span class="mobile-res-chip mobile-res-chip-secondary" id="mobile-sysram">SYS 0%</span>
        </div>
    </div>
    <div class="sidebar-overlay" id="sidebar-overlay" onclick="closeSidebar()"></div>
    <aside class="sidebar" id="sidebar">
        <div class="sidebar-logo">
            <h1>&#127912; Horde Worker</h1>
            <p>AI Image Generation</p>
        </div>
        <nav class="sidebar-nav" aria-label="Page navigation">
            <div class="nav-section-label">Navigation</div>
            <button class="nav-item active" onclick="showPage('overview', this)" id="nav-overview">
                <span class="nav-icon">&#127968;</span> Overview
            </button>
            <button class="nav-item" onclick="showPage('gallery', this)" id="nav-gallery">
                <span class="nav-icon">&#128444;</span> Gallery
            </button>
            <button class="nav-item" onclick="showPage('user', this)" id="nav-user">
                <span class="nav-icon">&#128100;</span> User
            </button>
            <button class="nav-item" onclick="showPage('horde', this)" id="nav-horde">
                <span class="nav-icon">&#127760;</span> Horde
            </button>
            <button class="nav-item" onclick="showPage('logs', this)" id="nav-logs">
                <span class="nav-icon">&#128203;</span> Logs
            </button>
            <button class="nav-item" onclick="showPage('stats', this)" id="nav-stats">
                <span class="nav-icon">&#128202;</span> Statistics
            </button>
            <button class="nav-item" onclick="showPage('settings', this)" id="nav-settings">
                <span class="nav-icon">&#9881;</span> Settings
            </button>
            <button class="nav-item" onclick="showPage('api', this)" id="nav-api">
                <span class="nav-icon">&#128268;</span> API
            </button>
            <button class="nav-item" onclick="showPage('about', this)" id="nav-about">
                <span class="nav-icon">&#8505;</span> About
            </button>
        </nav>
    </aside>
    <div class="main-content">
        <div class="topbar">
            <div class="topbar-worker">
                <div class="topbar-worker-name" id="topbar-worker-name">Horde Worker</div>
                <div class="topbar-worker-sub" id="topbar-worker-sub">Loading...</div>
            </div>
            <div class="topbar-resources">
                <div class="topbar-res-pill">
                    <div class="topbar-res-pill-label"><span>CPU</span><span id="topbar-cpu-cores">0 cores</span></div>
                    <div class="topbar-res-bar-track"><div class="topbar-res-bar topbar-res-bar-back" id="topbar-cpu-bar" style="width:0%" aria-label="System CPU usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"></div><div class="topbar-res-bar cpu" id="topbar-cpu-ctr-bar" style="width:0%" aria-label="Worker CPU usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"></div></div>
                    <div class="topbar-res-pill-sub"><span>Worker</span><span id="topbar-cpu-ctr-pct">0%</span></div>
                    <div class="topbar-res-pill-sub" style="margin-top:2px;"><span>System</span><span id="topbar-cpu-pct">0%</span></div>
                </div>
                <div class="topbar-res-pill">
                    <div class="topbar-res-pill-label"><span>GPU</span><span id="topbar-gpu-cores">0 cores</span></div>
                    <div class="topbar-res-bar-track"><div class="topbar-res-bar topbar-res-bar-back gpu" id="topbar-gpu-bar" style="width:0%" aria-label="System GPU usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"></div><div class="topbar-res-bar gpu" id="topbar-gpu-wrk-bar" style="width:0%" aria-label="Worker GPU usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"></div></div>
                    <div class="topbar-res-pill-sub"><span>Worker</span><span id="topbar-gpu-wrk-pct">0%</span></div>
                    <div class="topbar-res-pill-sub" style="margin-top:2px;"><span>System</span><span id="topbar-gpu-pct">0%</span></div>
                </div>
                <div class="topbar-res-pill">
                    <div class="topbar-res-pill-label"><span>VRAM</span><span id="topbar-vram-total">0 MB</span></div>
                    <div class="topbar-res-bar-track"><div class="topbar-res-bar topbar-res-bar-back vram" id="topbar-vram-bar" style="width:0%" aria-label="System VRAM usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"></div><div class="topbar-res-bar vram" id="topbar-vram-wrk-bar" style="width:0%" aria-label="Worker VRAM usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"></div></div>
                    <div class="topbar-res-pill-sub"><span>Worker</span><span id="topbar-vram-wrk-pct">0%</span></div>
                    <div class="topbar-res-pill-sub" style="margin-top:2px;"><span>System</span><span id="topbar-vram-pct">0%</span></div>
                </div>
                <div class="topbar-res-pill">
                    <div class="topbar-res-pill-label"><span>RAM</span><span id="topbar-total-ram-val">0 GB</span></div>
                    <div class="topbar-res-bar-track"><div class="topbar-res-bar topbar-res-bar-back" id="topbar-sysram-bar" style="width:0%" aria-label="System RAM usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"></div><div class="topbar-res-bar" id="topbar-ram-bar" style="width:0%" aria-label="Worker RAM usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"></div></div>
                    <div class="topbar-res-pill-sub"><span>Worker</span><span id="topbar-ram-pct">0%</span></div>
                    <div class="topbar-res-pill-sub" style="margin-top:2px;"><span>System</span><span id="topbar-sysram-pct">0%</span></div>
                </div>
            </div>
            <div class="topbar-meta">
                <span id="worker-status-badge"></span>
                <span class="topbar-uptime">&#9201; <span id="uptime">--</span></span>
                <button class="clear-maintenance-btn" id="clear-maintenance-btn" onclick="clearMaintenanceMode()" title="Remove worker from maintenance mode">Remove Maintenance</button>
                <div class="job-pops-pause-wrap">
                <button class="job-pops-pause-btn" id="job-pops-pause-btn" onclick="handleJobPopsPauseBtn(event)" title="Pause or resume accepting new jobs from the Horde" aria-pressed="false">Pause Jobs</button>
                <div class="job-pops-pause-menu" id="job-pops-pause-menu" style="display:none;" role="menu">
                    <button onclick="setJobPopsPause(900)">15 minutes</button>
                    <button onclick="setJobPopsPause(3600)">1 hour</button>
                    <button onclick="setJobPopsPause(10800)">3 hours</button>
                    <button onclick="setJobPopsPause(null)">Indefinitely</button>
                </div>
                </div>
                <button class="nsfw-blur-btn" id="nsfw-blur-btn" onclick="toggleNsfwBlur()" title="Toggle blur on NSFW images">Blur NSFW</button>
                <button class="theme-toggle" onclick="toggleTheme()" id="topbar-theme-toggle" aria-label="Toggle theme">&#127769;</button>
            </div>
        </div>
        <div class="content-area">
            <div id="loading"><div class="loading-spinner"></div><span class="loading-text">Connecting to worker...</span></div>
            <div id="content" style="display: none;">
                <!-- OVERVIEW PAGE -->
                <div class="page active" id="page-overview">
                    <div class="section-header" style="padding-bottom:0;margin-bottom:6px;">
                        <span class="section-title">&#127968; Overview</span>
                    </div>
                    <div class="grid-4">
                        <div class="stat-card"><div class="stat-card-label">Images Generated</div><div class="stat-card-value success" id="overview-images-generated">0</div></div>
                        <div class="stat-card"><div class="stat-card-label">Images / Hour</div><div class="stat-card-value accent" id="images-per-hour">0</div></div>
                        <div class="stat-card"><div class="stat-card-label">Jobs Popped</div><div class="stat-card-value accent" id="jobs-popped">0</div></div>
                        <div class="stat-card"><div class="stat-card-label">Jobs Submitted</div><div class="stat-card-value success" id="jobs-completed">0</div></div>
                    </div>
                    <div class="grid-4">
                        <div class="stat-card"><div class="stat-card-label">Total Time without Jobs</div><div class="stat-card-value warning" id="time-without-jobs">0h 0m 0s</div></div>
                        <div class="stat-card"><div class="stat-card-label">Jobs Queued</div><div class="stat-card-value" id="jobs-queued">0</div></div>
                        <div class="stat-card"><div class="stat-card-label">Jobs Recovered</div><div class="stat-card-value warning" id="processes-recovered">0</div></div>
                        <div class="stat-card"><div class="stat-card-label">Jobs Faulted</div><div class="stat-card-value error" id="jobs-faulted">0</div></div>
                    </div>
                    <div class="grid-2">
                        <div class="card">
                            <div class="card-header"><span class="card-title">&#9889; Current Job</span><span id="job-total-timer" style="margin-left:auto;font-size:0.75rem;color:#94a3b8;"></span></div>
                            <div id="overview-current-job" class="centered-empty-container"><div class="empty-state"><span class="empty-state-icon">&#9203;</span>No job in progress</div></div>
                        </div>
                        <div class="card">
                            <div class="card-header last-result-card-header"><span class="card-title">&#128444; Last Result</span><span id="overview-image-model" class="overview-model-label"></span><span id="overview-image-time" class="overview-time-label"></span></div>
                            <div id="overview-image-container" class="last-image-container"><div class="empty-state"><span class="empty-state-icon">&#128444;</span>No image generated yet</div></div>
                        </div>
                    </div>
                    <div class="grid-2">
                        <div class="card overview-bottom-grid-left">
                            <div class="card-header"><span class="card-title">&#9881; Processes</span><span class="section-count" id="process-count">0</span></div>
                            <div id="processes" class="scrollable-tall"><div class="empty-state"><span class="empty-state-icon">&#9881;</span>No process info</div></div>
                        </div>
                        <div class="card">
                            <div class="card-header"><span class="card-title">&#128230; Job Queue</span><span class="card-header-count">(<span id="queue-count">0</span>/<span id="queue-max">0</span>)</span></div>
                            <div id="job-queue" class="scrollable"><div class="empty-state">Queue is empty</div></div>
                        </div>
                        <div class="card">
                            <div class="card-header"><span class="card-title">&#129302; Active Models</span><span class="card-header-count">(<span id="models-count">0</span>/<span id="models-max">0</span>)</span></div>
                            <div id="models-loaded" class="model-list"><span style="color:#94a3b8;font-size:0.83rem;">No models loaded</span></div>
                        </div>
                    </div>
                </div>

                <!-- GALLERY PAGE -->
                <div class="page" id="page-gallery">
                    <div class="section">
                        <div class="section-header gallery-page-header">
                            <span class="section-title">&#128444; Gallery</span>
                            <div class="gallery-view-toggle">
                                <button class="gallery-view-btn active" id="gallery-view-grid" onclick="setGalleryView('grid')" title="Grid view">&#9783;</button>
                                <button class="gallery-view-btn" id="gallery-view-list" onclick="setGalleryView('list')" title="List view">&#9776;</button>
                            </div>
                            <div class="gallery-filter-bar">
                                <label for="gallery-model-filter">Filter by model:</label>
                                <select id="gallery-model-filter" onchange="galleryChangeModelFilter(this.value)">
                                    <option value="">All models</option>
                                </select>
                                <label for="gallery-safety-filter">Safety:</label>
                                <select id="gallery-safety-filter" onchange="galleryChangeSafetyFilter(this.value)">
                                    <option value="">All</option>
                                    <option value="sfw">SFW only</option>
                                    <option value="nsfw">NSFW</option>
                                    <option value="csam">CSAM</option>
                                </select>
                            </div>
                        </div>
                        <div class="card">
                            <div id="gallery-new-banner" role="button" tabindex="0" onclick="fetchGalleryPage(1)" onkeydown="if(event.key==='Enter'||event.key===' '){fetchGalleryPage(1);event.preventDefault();}">&#128444; New images available &#8212; click to view latest</div>
                            <div id="gallery-loading" style="display:none;text-align:center;padding:24px 16px;"><div class="loading-spinner" style="margin:0 auto 8px;"></div><span class="loading-text">Loading gallery&#8230;</span></div>
                            <div id="gallery-empty" class="empty-state" style="display:none;"><span class="empty-state-icon">&#128444;</span>No images generated yet</div>
                            <div id="gallery-grid" class="image-grid" style="display:none;"></div>
                            <div class="pagination-controls" id="gallery-pagination" style="display:none;">
                                <button id="gallery-prev" onclick="galleryChangePage(-1)" disabled>&#8249; Prev</button>
                                <span class="pagination-info" id="gallery-page-info">Page 1 of 1</span>
                                <button id="gallery-next" onclick="galleryChangePage(1)">Next &#8250;</button>
                                <label for="gallery-page-size" class="pagination-info">Per page:</label>
                                <select id="gallery-page-size" class="page-size-select" onchange="galleryChangePageSize(this.value)">
                                    <option value="12">12</option>
                                    <option value="24">24</option>
                                    <option value="48">48</option>
                                    <option value="96">96</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- USER PAGE -->
                <div class="page" id="page-user">
                    <div class="section">
                        <div class="section-header"><span class="section-title">&#128100; User Details</span></div>
                        <div class="grid-4 user-details-grid">
                            <div class="stat-card"><div class="stat-card-label">Username</div><div class="stat-card-value"><span id="user-page-username">-</span><span id="user-page-trust-indicator" class="trust-indicator"></span></div></div>
                            <div class="stat-card"><div class="stat-card-label">Total Kudos</div><div class="stat-card-value success" id="user-page-kudos-total">-</div></div>
                            <div class="stat-card"><div class="stat-card-label">Worker Count</div><div class="stat-card-value" id="user-page-worker-count">-</div></div>
                            <div class="stat-card"><div class="stat-card-label">Kudos / Hour</div><div class="stat-card-value accent" id="user-page-kudos-per-hour">0</div></div>
                        </div>
                        <div class="card">
                            <div class="card-header"><span class="card-title">&#127881; Kudos Breakdown</span></div>
                            <div id="user-page-kudos-breakdown"></div>
                        </div>
                    </div>
                    <div class="section">
                        <div class="section-header"><span class="section-title">&#9881; Workers</span><span class="section-count" id="user-workers-count">0</span></div>
                        <div id="user-workers-list"><div class="empty-state"><span class="empty-state-icon">&#9881;</span>No worker data yet</div></div>
                    </div>
                </div>

                <!-- HORDE NETWORK PAGE -->
                <div class="page" id="page-horde">
                    <div class="section">
                        <div class="section-header">
                            <span class="section-title">&#127760; Horde Network</span>
                            <div class="horde-status-row">
                                <span class="horde-mode-badge ok" id="horde-mode-maintenance" style="display:none;">&#9888; Maintenance</span>
                                <span class="horde-mode-badge ok" id="horde-mode-invite" style="display:none;">&#128274; Invite Only</span>
                                <div class="stats-window-group">
                                    <button class="horde-window-btn active" id="horde-win-30m" onclick="setHordeWindow(1800, this)">30m</button>
                                    <button class="horde-window-btn" id="horde-win-2h" onclick="setHordeWindow(7200, this)">2h</button>
                                    <button class="horde-window-btn" id="horde-win-6h" onclick="setHordeWindow(21600, this)">6h</button>
                                    <button class="horde-window-btn" id="horde-win-all" onclick="setHordeWindow(null, this)">All</button>
                                </div>
                            </div>
                        </div>
                        <div class="grid-4">
                            <div class="stat-card"><div class="stat-card-label">Workers Online</div><div class="stat-card-value accent" id="horde-stat-workers">-</div></div>
                            <div class="stat-card"><div class="stat-card-label">Threads Active</div><div class="stat-card-value accent" id="horde-stat-threads">-</div></div>
                            <div class="stat-card"><div class="stat-card-label">Queued Requests</div><div class="stat-card-value" id="horde-stat-queued-req">-</div></div>
                            <div class="stat-card"><div class="stat-card-label">Queued Megapixelsteps</div><div class="stat-card-value" id="horde-stat-queued-mps">-</div></div>
                        </div>
                        <div class="grid-4" style="margin-top:var(--page-spacing);">
                            <div class="stat-card"><div class="stat-card-label">Past Min. Megapixelsteps</div><div class="stat-card-value success" id="horde-stat-past-min-mps">-</div></div>
                        </div>
                        <div id="horde-fetch-error" style="display:none; color:var(--error); font-size:0.85rem; margin-top:8px; padding:8px 12px; background:#fff5f5; border:1px solid #fecaca; border-radius:6px;">Failed to fetch Horde Network data. Will retry automatically.</div>
                    </div>
                    <div class="section">
                        <div class="grid-2">
                            <div class="card" style="padding:14px 16px;">
                                <div class="chart-legend">
                                    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#6366f1;"></span>Workers</span>
                                    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#a78bfa;"></span>Threads</span>
                                </div>
                                <div class="chart-label">Workers &amp; Threads</div>
                                <div class="chart-container-md"><canvas id="horde-chart-workers-threads" aria-label="Workers and threads over time"></canvas></div>
                            </div>
                            <div class="card" style="padding:14px 16px;">
                                <div class="chart-label">Queued Requests</div>
                                <div class="chart-container-md"><canvas id="horde-chart-queued-req" aria-label="Queued requests over time"></canvas></div>
                            </div>
                        </div>
                    </div>
                    <div class="section">
                        <div class="grid-2">
                            <div class="card" style="padding:14px 16px;">
                                <div class="chart-label">Queued Megapixelsteps</div>
                                <div class="chart-container-md"><canvas id="horde-chart-queued-mps" aria-label="Queued megapixelsteps over time"></canvas></div>
                            </div>
                            <div class="card" style="padding:14px 16px;">
                                <div class="chart-label">Past Minute Megapixelsteps</div>
                                <div class="chart-container-md"><canvas id="horde-chart-past-min-mps" aria-label="Past minute megapixelsteps over time"></canvas></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- LOGS PAGE -->
                <div class="page" id="page-logs">
                    <div class="section">
                        <div class="section-header"><span class="section-title">&#128203; Console</span><button id="console-pause-btn" class="console-pause-btn" onclick="toggleConsolePause()" title="Pause console output" aria-pressed="false">&#9646;&#9646; Pause</button><button id="console-copy-btn" class="console-copy-btn" onclick="copyConsoleLogs()" title="Copy visible console logs to clipboard">&#128203; Copy</button><select id="console-filter-select" class="console-filter-select" onchange="applyConsoleFilter()" title="Filter logs by severity"><option value="ALL">All levels</option><option value="SUCCESS">Success+</option><option value="WARNING">Warning+</option><option value="ERROR">Error+</option></select></div>
                        <div class="card log-panel" style="padding:0;">
                            <div id="console-logs" class="console-container" style="border-radius:12px;"><div style="text-align:center;color:#475569;padding:18px;">No logs available</div></div>
                        </div>
                    </div>
                    <div class="section">
                        <div class="section-header"><span class="section-title">&#10060; Errors</span><span class="section-count" id="errors-count">0</span><div class="errors-view-toggle" style="margin-left:auto;"><button class="errors-view-btn active" id="errors-btn-grouped" onclick="setErrorsView('grouped')">Grouped</button><button class="errors-view-btn" id="errors-btn-all" onclick="setErrorsView('all')">All</button></div><button class="errors-clear-btn" id="errors-clear-btn" onclick="clearErrors()" title="Clear all errors">&#128465; Clear</button></div>
                        <div class="card log-panel errors-outer">
                            <div id="errors-history" class="errors-list"><div class="empty-state"><span class="empty-state-icon">&#10003;</span>No errors</div></div>
                            <div class="pagination-controls" id="errors-pagination" style="display:none;">
                                <button id="errors-prev" onclick="errorsChangePage(-1)" disabled>&#8249; Prev</button>
                                <span class="pagination-info" id="errors-page-info">Page 1 of 1</span>
                                <button id="errors-next" onclick="errorsChangePage(1)">Next &#8250;</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- STATISTICS PAGE -->
                <div class="page" id="page-stats">
                    <div class="section">
                        <div class="section-header">
                            <span class="section-title">&#128202; Statistics</span>
                            <div class="stats-window-group" style="margin-left:auto;">
                                <button class="stats-window-btn active" id="stats-win-15m" onclick="setStatsWindow(900, this)">15m</button>
                                <button class="stats-window-btn" id="stats-win-1h" onclick="setStatsWindow(3600, this)">1h</button>
                                <button class="stats-window-btn" id="stats-win-6h" onclick="setStatsWindow(21600, this)">6h</button>
                                <button class="stats-window-btn" id="stats-win-all" onclick="setStatsWindow(null, this)">All</button>
                            </div>
                            <button class="nsfw-blur-btn" id="overview-reset-btn" onclick="resetOverviewStats()" title="Reset session statistics" style="margin-left:8px;">&#8635; Reset Stats</button>
                        </div>
                        <div class="grid-4 stats-summary-grid">
                            <div class="stat-card"><div class="stat-card-label">Images Generated</div><div class="stat-card-value success" id="stats-images-generated">-</div></div>
                            <div class="stat-card"><div class="stat-card-label">Kudos Earned</div><div class="stat-card-value success" id="stats-kudos-earned">-</div></div>
                            <div class="stat-card"><div class="stat-card-label">Avg Images / hr</div><div class="stat-card-value accent" id="stats-avg-iph">-</div></div>
                            <div class="stat-card"><div class="stat-card-label">Avg Kudos / hr</div><div class="stat-card-value accent" id="stats-avg-kph">-</div></div>
                        </div>
                        <div class="grid-2 stats-summary-grid">
                            <div class="stat-card"><div class="stat-card-label">Jobs Popped</div><div class="stat-card-value accent" id="stats-jobs-popped">-</div></div>
                            <div class="stat-card"><div class="stat-card-label">Jobs Faulted</div><div class="stat-card-value error" id="stats-jobs-faulted">-</div></div>
                        </div>

                    </div>
                    <div class="section">
                        <div class="section-header"><span class="section-title">&#128187; Resource Usage</span></div>
                        <div class="grid-4 stats-charts-grid">
                            <div class="card" style="padding:14px 16px;">
                                <div class="chart-legend">
                                    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#fb923c;"></span>Worker</span>
                                    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#f59e0b;"></span>System</span>
                                </div>
                                <div class="chart-label">CPU %</div>
                                <div class="chart-container-md"><canvas id="chart-cpu" aria-label="CPU usage over time"></canvas></div>
                            </div>
                            <div class="card" style="padding:14px 16px;">
                                <div class="chart-legend">
                                    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#60a5fa;"></span>Worker</span>
                                    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#3b82f6;"></span>System</span>
                                </div>
                                <div class="chart-label">GPU %</div>
                                <div class="chart-container-md"><canvas id="chart-gpu" aria-label="GPU usage over time"></canvas></div>
                            </div>
                            <div class="card" style="padding:14px 16px;">
                                <div class="chart-legend">
                                    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#10b981;"></span>Worker</span>
                                    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#059669;"></span>System</span>
                                </div>
                                <div class="chart-label">RAM %</div>
                                <div class="chart-container-md"><canvas id="chart-ram" aria-label="RAM usage over time"></canvas></div>
                            </div>
                            <div class="card" style="padding:14px 16px;">
                                <div class="chart-legend">
                                    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#8b5cf6;"></span>Worker</span>
                                    <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#a78bfa;"></span>System</span>
                                </div>
                                <div class="chart-label">VRAM %</div>
                                <div class="chart-container-md"><canvas id="chart-vram" aria-label="VRAM usage over time"></canvas></div>
                            </div>
                        </div>
                    </div>
                    <div class="section">
                        <div class="section-header"><span class="section-title">&#128202; Images &amp; &#128142; Kudos / Hour</span></div>
                        <div class="card" style="padding:14px 16px;">
                            <div class="chart-legend">
                                <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#10b981;"></span>Images / hr</span>
                                <span class="chart-legend-item"><span class="chart-legend-swatch" style="background:#6366f1;"></span>Kudos / hr</span>
                            </div>
                            <div class="chart-container"><canvas id="chart-iph-kph" aria-label="Images and kudos per hour over time"></canvas></div>
                        </div>
                    </div>
                    <div class="section">
                        <div id="stats-job-state-model-row">
                            <div>
                                <div class="section-header"><span class="section-title">&#9201; Avg &amp; Max Time per Job State</span></div>
                                <div class="card" style="padding:14px 16px;">
                                    <div id="stats-job-state-time-wrap">
                                        <div class="text-muted" style="font-size:0.85rem;">No completed jobs yet.</div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div class="section-header"><span class="section-title">&#127760; Images by Model</span></div>
                                <div class="card" style="padding:14px 16px;">
                                    <div id="stats-model-table-wrap" class="stats-model-table-wrap">
                                        <div class="text-muted" style="font-size:0.85rem;">No images generated yet.</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="section">
                        <div id="stats-model-time-row">
                            <div>
                                <div class="section-header"><span class="section-title">&#9201; Avg &amp; Max Time per Step per Model</span></div>
                                <div class="card" style="padding:14px 16px;">
                                    <div id="stats-step-time-model-wrap" class="stats-model-table-wrap">
                                        <div class="text-muted" style="font-size:0.85rem;">No completed jobs yet.</div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div class="section-header"><span class="section-title">&#9201; Avg &amp; Max Time per Job per Model</span></div>
                                <div class="card" style="padding:14px 16px;">
                                    <div id="stats-job-time-model-wrap" class="stats-model-table-wrap">
                                        <div class="text-muted" style="font-size:0.85rem;">No completed jobs yet.</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="section">
                        <div id="stats-model-tables-grid">
                            <div>
                                <div class="section-header"><span class="section-title">&#10060; Failed Jobs by Model</span></div>
                                <div class="card" style="padding:14px 16px;">
                                    <div id="stats-failed-model-table-wrap" class="stats-model-table-wrap">
                                        <div class="text-muted" style="font-size:0.85rem;">No failed jobs yet.</div>
                                    </div>
                                </div>
                            </div>
                            <div>
                                <div class="section-header"><span class="section-title">&#9888; Faults by Phase</span></div>
                                <div class="card" style="padding:14px 16px;">
                                    <div id="stats-fault-phase-table-wrap" class="stats-model-table-wrap">
                                        <div class="text-muted" style="font-size:0.85rem;">No job faults yet.</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- SETTINGS PAGE -->
                <div class="page" id="page-settings">
                    <div class="section">
                        <div class="section-header settings-page-header">
                            <span class="section-title">&#9881; Settings</span>
                            <div class="settings-header-actions">
                                <button id="settings-apply-btn" class="settings-page-btn apply" onclick="applyPendingSettings()" disabled>Apply</button>
                                <button id="settings-restart-btn" class="settings-page-btn restart" onclick="restartProgram()">Restart</button>
                                <button id="settings-reset-all-btn" class="settings-page-btn reset-all" onclick="showResetAllConfirm()" title="Reset all settings to their default values">&#8635; Reset All</button>
                                <button id="settings-reset-db-btn" class="settings-page-btn reset-db" onclick="showResetDbConfirm()" title="Permanently delete all stored history (errors, statistics, gallery images, network snapshots)">&#128465; Reset Database</button>
                                <span id="settings-status" class="section-count" style="display:none;"></span>
                            </div>
                        </div>
                        <div id="settings-body">
                            <div class="settings-unavailable">Loading settings&#8230;</div>
                        </div>
                        <div id="models-section-container"></div>
                    </div>
                </div>

                <!-- API PAGE -->
                <div class="page" id="page-api">
                    <div class="section">
                        <div class="section-header">
                            <span class="section-title">&#128268; API</span>
                        </div>
                        <div class="api-ref-intro">Every endpoint below is unauthenticated and reachable from any address that can reach this worker's web UI port &mdash; there is no separate "external" API surface, just this one. Treat network access to this port as the access control.</div>
                        <div id="api-page-body"></div>
                    </div>
                </div>

                <!-- ABOUT PAGE -->
                <div class="page" id="page-about">
                    <div class="section">
                        <div class="section-header">
                            <span class="section-title">&#8505; About</span>
                        </div>
                        <div class="about-hero">
                            <div class="about-hero-icon">&#127912;</div>
                            <div class="about-hero-body">
                                <div class="about-hero-title">AI Horde Worker (reGen)</div>
                                <div class="about-hero-sub">Version {{WORKER_VERSION}} &middot; AGPL-3.0</div>
                                <div class="about-hero-desc">Connects this machine to the <a href="https://aihorde.net" target="_blank" rel="noopener noreferrer">AI Horde</a>, a crowdsourced, distributed cluster for AI image generation, turning spare GPU time into images for the community in exchange for kudos.</div>
                                <div class="about-hero-links">
                                    <a class="about-link" href="https://github.com/Haidra-Org/AI-Horde-Worker" target="_blank" rel="noopener noreferrer">&#128279; GitHub Repository</a>
                                    <a class="about-link" href="https://aihorde.net" target="_blank" rel="noopener noreferrer">&#127760; AI Horde</a>
                                </div>
                            </div>
                        </div>
                        <div class="settings-group">
                            <div class="settings-group-title">Tech Stack</div>
                            <div class="about-tech-grid">
                                <div class="about-tech-card"><span class="about-tech-card-tag">Language</span><div class="about-tech-card-name">Python 3.10+</div><div class="about-tech-card-desc">Core worker runtime, process management, and this Web UI.</div></div>
                                <div class="about-tech-card"><span class="about-tech-card-tag">Web</span><div class="about-tech-card-name">aiohttp</div><div class="about-tech-card-desc">Async HTTP server powering this Web UI and its JSON API.</div></div>
                                <div class="about-tech-card"><span class="about-tech-card-tag">Frontend</span><div class="about-tech-card-name">Vanilla HTML / CSS / JS</div><div class="about-tech-card-desc">This page is server-rendered with no frontend framework or build step.</div></div>
                                <div class="about-tech-card"><span class="about-tech-card-tag">Storage</span><div class="about-tech-card-name">SQLite</div><div class="about-tech-card-desc">Local persistence for error logs, gallery images, and statistics history.</div></div>
                                <div class="about-tech-card"><span class="about-tech-card-tag">Imaging</span><div class="about-tech-card-name">Pillow</div><div class="about-tech-card-desc">Generates gallery thumbnails (optional &mdash; falls back to full-resolution images if absent).</div></div>
                                <div class="about-tech-card"><span class="about-tech-card-tag">Logging</span><div class="about-tech-card-name">Loguru</div><div class="about-tech-card-desc">Structured logging to console and the Logs page.</div></div>
                                <div class="about-tech-card"><span class="about-tech-card-tag">ML</span><div class="about-tech-card-name">PyTorch</div><div class="about-tech-card-desc">GPU tensor runtime underpinning image generation.</div></div>
                                <div class="about-tech-card"><span class="about-tech-card-tag">ML</span><div class="about-tech-card-name">horde_engine (hordelib)</div><div class="about-tech-card-desc">ComfyUI-based diffusion pipeline execution.</div></div>
                                <div class="about-tech-card"><span class="about-tech-card-tag">Horde</span><div class="about-tech-card-name">horde_sdk</div><div class="about-tech-card-desc">AI Horde job protocol: popping, submitting, and reporting faults for jobs.</div></div>
                                <div class="about-tech-card"><span class="about-tech-card-tag">Horde</span><div class="about-tech-card-name">horde_safety</div><div class="about-tech-card-desc">NSFW / CSAM detection applied to generated images before submission.</div></div>
                                <div class="about-tech-card"><span class="about-tech-card-tag">Data</span><div class="about-tech-card-name">Pydantic</div><div class="about-tech-card-desc">Validation and schemas for Horde API payloads.</div></div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    </div>
    <div id="image-overlay" class="image-overlay">
        <button id="overlay-prev" class="image-overlay-nav prev" onclick="overlayNavigate(-1)" aria-label="Previous image" title="Previous image">&#8249;</button>
        <div id="overlay-content" class="image-overlay-content">
            <button class="image-overlay-close" onclick="closeImageOverlay()">&#10005; Close</button>
            <img id="overlay-image" src="" alt="Full resolution image" />
            <div id="overlay-loading" class="image-overlay-loading"><div class="loading-spinner"></div></div>
            <div id="overlay-counter" class="image-overlay-counter"></div>
        </div>
        <button id="overlay-next" class="image-overlay-nav next" onclick="overlayNavigate(1)" aria-label="Next image" title="Next image">&#8250;</button>
    </div>
    <div id="reset-all-confirm-modal" class="confirm-modal-backdrop" aria-hidden="true" onclick="dismissResetAllConfirm(event)">
        <div class="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="reset-all-confirm-title" aria-describedby="reset-all-confirm-body">
            <div id="reset-all-confirm-title" class="confirm-modal-title">Reset all settings to defaults?</div>
            <div id="reset-all-confirm-body" class="confirm-modal-body">All settings with known defaults will be staged at their default values. You can still review changes before applying.</div>
            <div class="confirm-modal-actions">
                <button class="confirm-modal-btn cancel" type="button" onclick="closeResetAllConfirm()">Cancel</button>
                <button class="confirm-modal-btn confirm" type="button" onclick="confirmResetAll()">Reset All</button>
            </div>
        </div>
    </div>
    <div id="restart-confirm-modal" class="confirm-modal-backdrop" aria-hidden="true" onclick="dismissRestartConfirm(event)">
        <div class="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="restart-confirm-title" aria-describedby="restart-confirm-body">
            <div id="restart-confirm-title" class="confirm-modal-title">Restart worker now?</div>
            <div id="restart-confirm-body" class="confirm-modal-body">This will restart the worker process and temporarily interrupt job processing.</div>
            <div class="confirm-modal-actions">
                <button id="restart-confirm-cancel" class="confirm-modal-btn cancel" type="button" onclick="closeRestartConfirm()">Cancel</button>
                <button id="restart-confirm-accept" class="confirm-modal-btn confirm" type="button" onclick="confirmRestartProgram()">Restart</button>
            </div>
        </div>
    </div>
    <div id="reset-db-confirm-modal" class="confirm-modal-backdrop" aria-hidden="true" onclick="dismissResetDbConfirm(event)">
        <div class="confirm-modal" role="dialog" aria-modal="true" aria-labelledby="reset-db-confirm-title" aria-describedby="reset-db-confirm-body">
            <div id="reset-db-confirm-title" class="confirm-modal-title">Reset database?</div>
            <div id="reset-db-confirm-body" class="confirm-modal-body">This permanently deletes all stored history: error logs, statistics snapshots, gallery images, and network performance snapshots. This cannot be undone. Your settings are not affected.</div>
            <div class="confirm-modal-actions">
                <button id="reset-db-confirm-cancel" class="confirm-modal-btn cancel" type="button" onclick="closeResetDbConfirm()">Cancel</button>
                <button id="reset-db-confirm-accept" class="confirm-modal-btn confirm" type="button" onclick="confirmResetDatabase()">Reset Database</button>
            </div>
        </div>
    </div>
    <script>
        function toggleSidebar() { document.getElementById('sidebar').classList.toggle('open'); document.getElementById('sidebar-overlay').classList.toggle('active'); }
        function closeSidebar() { document.getElementById('sidebar').classList.remove('open'); document.getElementById('sidebar-overlay').classList.remove('active'); }
        function escapeHtml(str) {
            if (str === null || str === undefined) return '';
            return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        }
        const VALID_PAGES = Object.freeze(['overview', 'gallery', 'user', 'horde', 'stats', 'logs', 'settings', 'api', 'about']);
        let galleryCurrentPage = 1, galleryTotalPages = 1, galleryTotalImages = 0, galleryFetchInProgress = false;
        const GALLERY_DEFAULT_PAGE_SIZE = 96;
        let galleryPageSize = GALLERY_DEFAULT_PAGE_SIZE;
        let galleryModelFilter = '';
        let gallerySafetyFilter = '';
        var galleryViewMode = (function() { try { return localStorage.getItem('horde-gallery-view') || 'grid'; } catch(e) { return 'grid'; } })();
        var _galleryCurrentPageImages = null; // cached for view-mode switch without re-fetch
        let cachedWorkersList = (function() { try { var s = localStorage.getItem('horde-workers-list'); var parsed = s ? JSON.parse(s) : []; return Array.isArray(parsed) ? parsed : []; } catch(e) { return []; } })();
        let currentWorkerName = '';
        // Word-level LCS diff between two prompt strings.
        // Returns an array of {t: 'eq'|'del'|'add', w: string} tokens.
        function computeWordDiff(origText, filtText) {
            var o = origText.trim().split(/\s+/).filter(Boolean);
            var f = filtText.trim().split(/\s+/).filter(Boolean);
            var m = o.length, n = f.length;
            var dp = new Array(m + 1);
            for (var _i = 0; _i <= m; _i++) dp[_i] = new Array(n + 1).fill(0);
            for (var _i = 1; _i <= m; _i++)
                for (var _j = 1; _j <= n; _j++)
                    dp[_i][_j] = o[_i-1] === f[_j-1] ? dp[_i-1][_j-1] + 1 : Math.max(dp[_i-1][_j], dp[_i][_j-1]);
            var diff = [], _i = m, _j = n;
            while (_i > 0 || _j > 0) {
                if (_i > 0 && _j > 0 && o[_i-1] === f[_j-1]) { diff.unshift({t:'eq', w:o[_i-1]}); _i--; _j--; }
                else if (_j > 0 && (_i === 0 || dp[_i][_j-1] >= dp[_i-1][_j])) { diff.unshift({t:'add', w:f[_j-1]}); _j--; }
                else { diff.unshift({t:'del', w:o[_i-1]}); _i--; }
            }
            return diff;
        }
        // Render a prompt string with diff markers if an original (pre-filter) version exists.
        // Removed words show as red strikethrough; added words show as green highlight.
        function renderPromptDiff(filtText, origText) {
            if (!filtText) return '';
            if (!origText || origText === filtText) return escapeHtml(filtText);
            var diff = computeWordDiff(origText, filtText);
            var html = '', prevType = null, buf = [];
            function flush() {
                if (!buf.length) return;
                var text = escapeHtml(buf.join(' '));
                if (prevType === 'del') html += '<span class="prompt-diff-removed">' + text + '</span>';
                else if (prevType === 'add') html += '<span class="prompt-diff-added">' + text + '</span>';
                else html += text;
                buf = [];
            }
            diff.forEach(function(d) {
                if (d.t !== prevType) { flush(); prevType = d.t; }
                buf.push(d.w);
            });
            flush();
            return html;
        }
        // Event delegation: expand/collapse prompts in gallery list view on meta-area click.
        document.addEventListener('click', function(evt) {
            if (galleryViewMode !== 'list') return;
            var meta = evt.target.closest ? evt.target.closest('.gallery-list-meta') : null;
            if (!meta) return;
            var item = meta.closest('.gallery-list-item');
            if (item) item.classList.toggle('prompt-expanded');
        });
        // Event delegation: handle delete-button clicks on the workers list container.
        document.addEventListener('click', function(evt) {
            var btn = evt.target.closest('.worker-delete-btn');
            if (!btn) return;
            var workerId = btn.getAttribute('data-worker-id') || '';
            var workerName = btn.getAttribute('data-worker-name') || 'Unknown';
            if (!workerId) return;
            if (!confirm('Delete worker "' + workerName + '"?\n\nThis action cannot be undone.')) return;
            btn.disabled = true;
            fetch('/api/worker/' + encodeURIComponent(workerId), { method: 'DELETE' })
                .then(function(r) { return r.json().then(function(body) { return { ok: r.ok, body: body }; }); })
                .then(function(result) {
                    if (result.ok) {
                        cachedWorkersList = cachedWorkersList.filter(function(w) { return w.id !== workerId; });
                        try { localStorage.setItem('horde-workers-list', JSON.stringify(cachedWorkersList)); } catch(e) {}
                        renderWorkersList();
                    } else {
                        alert('Failed to delete worker: ' + (result.body.error || 'Unknown error'));
                        btn.disabled = false;
                    }
                })
                .catch(function(e) { alert('Error deleting worker: ' + e.message); btn.disabled = false; });
        });
        function renderWorkersList() {
            const workersList = Array.isArray(cachedWorkersList) ? cachedWorkersList : [];
            document.getElementById('user-workers-count').textContent = workersList.length;
            const wlEl = document.getElementById('user-workers-list');
            if (workersList.length === 0) {
                wlEl.innerHTML = '<div class="empty-state"><span class="empty-state-icon">&#9881;</span>No worker data yet</div>';
            } else {
                wlEl.innerHTML = workersList.map(function(w) {
                    const onlineCls = w.online ? 'online' : 'offline';
                    const onlineTxt = w.online ? 'Online' : 'Offline';
                    const capBadge = function(val, label) {
                        if (val === null || val === undefined) return '';
                        return '<span class="wcap '+(val?'wcap-yes':'wcap-no')+'">'+escapeHtml(label)+'</span>';
                    };
                    const nsfwBadge = w.nsfw === true ? '<span class="wcap wcap-nsfw">NSFW</span>' : (w.nsfw === false ? '<span class="wcap wcap-sfw">SFW</span>' : '');
                    const caps = nsfwBadge +
                        capBadge(w.trusted, 'Trusted') +
                        capBadge(w.img2img, 'img2img') +
                        capBadge(w.painting, 'Painting') +
                        capBadge(w.lora, 'LoRA');
                    const models = w.models || [];
                    const modelCount = models.length;
                    const modelTitles = models.map(function(m) { return m.replace(/ /g, '\u00A0'); }).join(', ');
                    const sizeStr = w.max_pixels ? ('\u2248 '+Math.round(Math.sqrt(w.max_pixels))+' px') : '-';
                    const uptimeSecs = w.uptime || 0;
                    const uh = Math.floor(uptimeSecs/3600), um = Math.floor((uptimeSecs%3600)/60);
                    const uptimeStr = uh > 0 ? uh+'h '+um+'m' : (um > 0 ? um+'m' : uptimeSecs+'s');
                    const kudos = w.kudos_rewards != null ? Number(w.kudos_rewards).toLocaleString(undefined,{maximumFractionDigits:0}) : '-';
                    const kph = (w.kudos_rewards != null && uptimeSecs > 0)
                        ? (w.kudos_rewards / (uptimeSecs / 3600)).toLocaleString(undefined, {maximumFractionDigits:1})
                        : '-';
                    const isCurrentWorker = currentWorkerName && (w.name === currentWorkerName);
                    const canDelete = !w.online && !isCurrentWorker;
                    const deleteBtn = canDelete
                        ? '<button class="worker-delete-btn" data-worker-id="'+escapeHtml(w.id||'')+'" data-worker-name="'+escapeHtml(w.name||'Unknown')+'" title="Delete this offline worker">\uD83D\uDDD1 Delete</button>'
                        : '';
                    return '<div class="worker-card">' +
                        '<div class="worker-card-header">' +
                        '<span class="worker-card-name">'+escapeHtml(w.name||'Unknown')+'</span>' +
                        (w.version ? '<span class="worker-version-badge">v'+escapeHtml(w.version)+'</span>' : '') +
                        (w.type ? '<span class="worker-type-badge">'+escapeHtml(w.type)+'</span>' : '') +
                        '<span class="worker-online-badge '+onlineCls+'">'+onlineTxt+'</span>' +
                        deleteBtn +
                        '</div>' +
                        (caps ? '<div class="worker-caps-row">'+caps+'</div>' : '') +
                        '<div class="worker-meta-row">' +
                        '<span class="wm-item">\uD83D\uDCCF '+escapeHtml(sizeStr)+'</span>' +
                        (w.threads != null ? '<span class="wm-item">\uD83E\uDDF5 '+escapeHtml(w.threads)+' thread'+(w.threads!==1?'s':'')+'</span>' : '') +
                        (modelTitles ? '<span class="wm-item models-pill" data-tooltip="'+escapeHtml(modelTitles)+'">' : '<span class="wm-item models-pill">') +'\uD83E\uDDE9 '+modelCount+' model'+(modelCount!==1?'s':'')+(modelCount>0?' \u25BE':'')+'</span>' +
                        '</div>' +
                        '<div class="worker-stats-row">' +
                        '<span class="ws-item">&#9201; '+escapeHtml(uptimeStr)+' uptime</span>' +
                        '<span class="ws-item">\uD83D\uDC8E '+kudos+' kudos</span>' +
                        '<span class="ws-item accent">\uD83D\uDCC8 '+kph+' k/h</span>' +
                        '</div>' +
                        '</div>';
                }).join('');
            }
        }
        function showPage(pageId, navEl, push) {
            if (!VALID_PAGES.includes(pageId)) pageId = 'overview';
            document.querySelectorAll('.page').forEach(function(p) { p.classList.remove('active'); });
            var page = document.getElementById('page-' + pageId);
            if (page) page.classList.add('active');
            document.querySelectorAll('.nav-item').forEach(function(item) { item.classList.remove('active'); });
            var activeNav = navEl || document.getElementById('nav-' + pageId);
            if (activeNav) activeNav.classList.add('active');
            if (window.innerWidth < 768) closeSidebar();
            var newHash = '#' + pageId;
            if (push !== false) {
                if (location.hash !== newHash) history.pushState({page: pageId}, '', newHash);
            }
            if (pageId === 'gallery') {
                populateGalleryModelFilter();
                populateGallerySafetyFilter();
                const gridEl = document.getElementById('gallery-grid');
                const gridEmpty = !gridEl || !gridEl.querySelector('.image-grid-item,.gallery-list-item');
                if (gridEmpty) {
                    // First visit (or after page-size change cleared the grid): full fetch.
                    fetchGalleryPage(galleryCurrentPage);
                } else if (galleryHasUnseenImages) {
                    // New images arrived while we were on another tab.  Handle exactly the
                    // same way the status-poll would if the gallery tab had been active.
                    galleryHasUnseenImages = false;
                    if (galleryCurrentPage === 1) {
                        if (!galleryFetchInProgress) refreshGalleryPage1();
                    } else {
                        const bnr = document.getElementById('gallery-new-banner');
                        if (bnr) bnr.style.display = '';
                    }
                }
                // Otherwise the grid already shows the current page with cached thumbnails;
                // new-image notifications continue via the status-poll path.
            }
            if (pageId === 'user') {
                // Render cached workers immediately so the list is visible before the next status poll.
                renderWorkersList();
            }
            if (pageId === 'stats') {
                fetchStats(true);
            }
            if (pageId === 'horde') {
                renderHordePage();
                // Trigger an immediate fetch when navigating here if there are no snapshots
                // yet or the most recent one is older than the poll interval.
                var _hordeLastTs = (_hordeSnapshots && _hordeSnapshots.length > 0) ? _hordeSnapshots[_hordeSnapshots.length - 1].t : 0;
                // Same hoisting hazard as startHordeFetching(): this can run (via the
                // initial hash-based routing call) before "var _hordeWindowSecs = 1800"
                // further down the file has executed, so fall back to its eventual
                // default explicitly instead of passing along a hoisted `undefined`.
                if (Date.now() / 1000 - _hordeLastTs > 30) {
                    _fetchHordeHistory(_hordeWindowSecs === undefined ? 1800 : _hordeWindowSecs);
                }
            }
            if (pageId === 'settings') {
                fetchSettings();
            }
            if (pageId === 'api' && !_apiPageRendered) {
                renderApiPage();
                _apiPageRendered = true;
            }
        }
        window.addEventListener('popstate', function() {
            var hash = location.hash.replace('#', '');
            showPage(VALID_PAGES.includes(hash) ? hash : 'overview', null, false);
        });
        // ========================================================
        // STATISTICS PAGE - state variables (must be declared before the
        // IIFE below, which may call showPage('stats') → fetchStats())
        // ========================================================
        let _statsData = null;
        let _statsWindowSecs = 900; // default 15 minutes
        let _statsFetchInProgress = false;
        let _statsAbortController = null;
        let _statsLastFetchTime = 0;
        const _STATS_FETCH_THROTTLE_MS = 9000; // just under the 10-second server snapshot interval
        (function() {
            var hash = location.hash.replace('#', '');
            if (hash && VALID_PAGES.includes(hash)) {
                showPage(hash, null, false);
            } else {
                history.replaceState({page: 'overview'}, '', '#overview');
            }
        })();
        startHordeFetching(); // pre-fetch so data is ready before user visits the Horde page
        function initTheme() {
            const saved = localStorage.getItem('horde-theme') || 'light';
            document.documentElement.setAttribute('data-theme', saved);
            const icon = saved === 'dark' ? '&#9728;' : '&#127769;';
            document.getElementById('topbar-theme-toggle').innerHTML = icon;
            document.getElementById('mobile-theme-toggle').innerHTML = icon;
        }
        function toggleTheme() {
            const current = document.documentElement.getAttribute('data-theme') || 'light';
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('horde-theme', next);
            const icon = next === 'dark' ? '&#9728;' : '&#127769;';
            document.getElementById('topbar-theme-toggle').innerHTML = icon;
            document.getElementById('mobile-theme-toggle').innerHTML = icon;
        }
        function initNsfwBlur() {
            var enabled = localStorage.getItem('horde-blur-nsfw') === '1';
            if (enabled) document.body.classList.add('blur-nsfw');
            var btn = document.getElementById('nsfw-blur-btn');
            if (btn) btn.classList.toggle('active', enabled);
        }
        function resetOverviewStats() {
            fetch('/api/reset-stats', {method: 'POST'})
                .then(function(r) { return r.json(); })
                .then(function() { updateStatus(); })
                .catch(function(err) { console.error('Reset stats failed:', err); });
        }
        function toggleNsfwBlur() {
            var enabled = document.body.classList.toggle('blur-nsfw');
            try { localStorage.setItem('horde-blur-nsfw', enabled ? '1' : '0'); } catch(e) {}
            var btn = document.getElementById('nsfw-blur-btn');
            if (btn) btn.classList.toggle('active', enabled);
        }
        initTheme();
        initNsfwBlur();
        (function() {
            // Restore saved gallery view mode on load
            var savedView = galleryViewMode;
            document.querySelectorAll('.gallery-view-btn').forEach(function(b) { b.classList.remove('active'); });
            var btn = document.getElementById('gallery-view-' + savedView);
            if (btn) btn.classList.add('active');
        })();
        let overlayImages = [], overlayIndex = -1;
        // When non-null, holds the stable gallery_id values for the current overlay page so the
        // full-resolution image can be requested directly via /api/gallery/full/{id} -- a real
        // <img> URL, so the browser's native loading + HTTP cache apply (reopening an
        // already-viewed image is served instantly from cache, no fetch/JSON round trip).
        let _galleryOverlayIds = null;
        function _updateOverlayNav() {
            const hasList = overlayImages.length > 1;
            const pb = document.getElementById('overlay-prev'), nb = document.getElementById('overlay-next');
            const ctr = document.getElementById('overlay-counter');
            pb.style.display = nb.style.display = hasList ? 'block' : 'none';
            if (hasList) { pb.disabled = overlayIndex <= 0; nb.disabled = overlayIndex >= overlayImages.length - 1; ctr.textContent = (overlayIndex + 1) + ' / ' + overlayImages.length; }
            else { ctr.textContent = ''; }
        }
        function openImageOverlay(imageSrc, images, index) {
            if (Array.isArray(images) && Number.isFinite(index) && images.length > 0) {
                const len = images.length;
                const safeIndex = Math.min(Math.max(Math.trunc(index), 0), len - 1);
                overlayImages = images;
                overlayIndex = safeIndex;
            } else { overlayImages = []; overlayIndex = -1; }
            _galleryOverlayIds = null;
            document.getElementById('overlay-image').src = imageSrc;
            document.getElementById('image-overlay').classList.add('active');
            _updateOverlayNav();
        }
        // Points the overlay <img> straight at /api/gallery/full/{id}: the browser handles
        // loading, decoding, prioritization (fetchPriority), and caching natively, so there is no
        // manual fetch/AbortController bookkeeping -- setting a new src simply supersedes
        // whatever the element was previously loading.
        function _showGalleryOverlayImage(galleryId) {
            const el = document.getElementById('overlay-image');
            const content = document.getElementById('overlay-content');
            content.classList.add('is-loading');
            el.fetchPriority = 'high';
            el.onload = el.onerror = function() { content.classList.remove('is-loading'); };
            el.src = '/api/gallery/full/' + galleryId;
        }
        function openGalleryImageOverlay(galleryId, galleryIds, localIdx) {
            if (Array.isArray(galleryIds) && galleryIds.length > 0) {
                overlayImages = galleryIds;
                overlayIndex = localIdx;
                _galleryOverlayIds = galleryIds;
            } else {
                overlayImages = [galleryId];
                overlayIndex = 0;
                _galleryOverlayIds = [galleryId];
            }
            document.getElementById('image-overlay').classList.add('active');
            _updateOverlayNav();
            _showGalleryOverlayImage(galleryId);
        }
        function overlayNavigate(delta) {
            const ni = overlayIndex + delta;
            if (ni < 0 || ni >= overlayImages.length) return;
            overlayIndex = ni;
            if (_galleryOverlayIds !== null) {
                _showGalleryOverlayImage(_galleryOverlayIds[overlayIndex]);
            } else {
                document.getElementById('overlay-image').src = overlayImages[overlayIndex];
            }
            _updateOverlayNav();
        }
        function closeImageOverlay() {
            document.getElementById('overlay-content').classList.remove('is-loading');
            document.getElementById('image-overlay').classList.remove('active');
            overlayImages = []; overlayIndex = -1; _galleryOverlayIds = null;
        }
        document.getElementById('image-overlay').addEventListener('click', function(e) { if (e.target === this) closeImageOverlay(); });
        document.addEventListener('keydown', function(e) {
            const overlayActive = document.getElementById('image-overlay').classList.contains('active');
            if (e.key === 'Escape' && _restartConfirmOpen) {
                e.preventDefault();
                closeRestartConfirm();
            } else if (e.key === 'Escape') {
                if (overlayActive) { e.preventDefault(); e.stopPropagation(); }
                closeImageOverlay();
            } else if (overlayActive && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
                e.preventDefault();
                e.stopPropagation();
                overlayNavigate(e.key === 'ArrowLeft' ? -1 : 1);
            }
        });
        (function() {
            var _tt = null;
            function _getTooltip() {
                if (!_tt) { _tt = document.createElement('div'); _tt.className = 'models-tooltip'; document.body.appendChild(_tt); }
                return _tt;
            }
            function _showTooltip(pill) {
                var text = pill.getAttribute('data-tooltip');
                if (!text) return;
                var tt = _getTooltip();
                tt.textContent = text;
                tt.style.minWidth = '';
                var maxVw = Math.floor(window.innerWidth * 0.95);
                tt.style.maxWidth = Math.min(maxVw, 820) + 'px';
                tt.style.left = '-9999px'; tt.style.top = '-9999px'; tt.style.display = 'block';
                var w = tt.offsetWidth, h = tt.offsetHeight;
                if (h > w) { tt.style.minWidth = Math.min(h, maxVw) + 'px'; w = tt.offsetWidth; h = tt.offsetHeight; }
                var ttW = tt.offsetWidth, ttH = tt.offsetHeight;
                var rect = pill.getBoundingClientRect();
                var top = rect.top - ttH - 4;
                if (top < 4) { top = rect.bottom + 4; }
                var left = rect.left;
                if (left + ttW > window.innerWidth - 4) { left = window.innerWidth - ttW - 4; }
                if (left < 4) { left = 4; }
                tt.style.left = left + 'px'; tt.style.top = top + 'px';
            }
            function _hideTooltip() { if (_tt) { _tt.style.display = 'none'; } }
            document.addEventListener('mouseover', function(e) {
                var pill = e.target.closest && e.target.closest('.models-pill[data-tooltip]');
                if (pill && pill.getAttribute('data-tooltip')) { _showTooltip(pill); } else { _hideTooltip(); }
            });
            document.addEventListener('mouseout', function(e) {
                var pill = e.target.closest && e.target.closest('.models-pill[data-tooltip]');
                if (pill) { var to = e.relatedTarget; if (!to || !pill.contains(to)) { _hideTooltip(); } }
            });
        })();
        function formatUptime(seconds) {
            const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = Math.floor(seconds % 60);
            return h+'h '+m+'m '+s+'s';
        }
        function formatTimeAgo(timestamp) {
            if (!timestamp || timestamp === 0) return 'No image generated yet';
            const now = Date.now() / 1000, sa = Math.floor(now - timestamp);
            if (sa < 60) return 'Last submission: '+sa+' second'+(sa !== 1 ? 's' : '')+' ago';
            else if (sa < 3600) { const m = Math.floor(sa/60); return 'Last submission: '+m+' minute'+(m !== 1?'s':'')+' ago'; }
            else if (sa < 86400) { const h = Math.floor(sa/3600); return 'Last submission: '+h+' hour'+(h !== 1?'s':'')+' ago'; }
            else { const d = Math.floor(sa/86400); return 'Last submission: '+d+' day'+(d !== 1?'s':'')+' ago'; }
        }
        function formatTimestamp(timestamp) {
            if (!timestamp || timestamp === 0) return '';
            const d = new Date(timestamp * 1000);
            return isNaN(d.getTime()) ? '' : d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
        }
        function formatTimestampFull(timestamp) {
            if (!timestamp || timestamp === 0) return '';
            const d = new Date(timestamp * 1000);
            if (isNaN(d.getTime())) return '';
            const date = d.toLocaleDateString([], {year: 'numeric', month: 'short', day: 'numeric'});
            const time = d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
            return date + ' · ' + time;
        }
        function truncatePrompt(s, maxLen) {
            if (!s) return '';
            s = s.trim();
            if (s.length <= maxLen) return s;
            return s.slice(0, maxLen).trimEnd() + '…';
        }
        const SCROLL_TOLERANCE_PX = 1;
        let consolePaused = false;
        let _consoleLogs = [];
        const _CONSOLE_LOG_LEVEL_RE = /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \| ([A-Z]+)\s*\|/;
        const _CONSOLE_ANSI_RE = /\x1b\[[0-9;]*m/g;
        const _CONSOLE_LEVEL_ORDER = {TRACE: 0, DEBUG: 1, INFO: 2, SUCCESS: 3, WARNING: 4, ERROR: 5, CRITICAL: 6};
        function _getLogLevel(rawLog) {
            const plain = rawLog.replace(_CONSOLE_ANSI_RE, '');
            const m = plain.match(_CONSOLE_LOG_LEVEL_RE);
            return m ? m[1] : 'INFO';
        }
        function _renderConsoleLogs() {
            const cl = document.getElementById('console-logs');
            if (!cl) return;
            const filterLevel = (document.getElementById('console-filter-select') || {}).value || 'ALL';
            const minOrder = filterLevel === 'ALL' ? -1 : (_CONSOLE_LEVEL_ORDER[filterLevel] !== undefined ? _CONSOLE_LEVEL_ORDER[filterLevel] : -1);
            const visible = minOrder < 0 ? _consoleLogs : _consoleLogs.filter(function(log) {
                const lvl = _getLogLevel(log);
                const ord = _CONSOLE_LEVEL_ORDER[lvl] !== undefined ? _CONSOLE_LEVEL_ORDER[lvl] : 2;
                return ord >= minOrder;
            });
            const atb = isScrolledToBottom(cl, SCROLL_TOLERANCE_PX);
            var _LEVEL_COLORS = {WARNING:'#f59e0b',ERROR:'#ef4444',CRITICAL:'#dc2626',SUCCESS:'#23d18b',DEBUG:'#94a3b8',TRACE:'#64748b'};
            if (visible.length > 0) {
                cl.innerHTML = visible.map(function(log) {
                    var lvl = _getLogLevel(log);
                    var baseColor = _LEVEL_COLORS[lvl] || '#cccccc';
                    return '<div style="white-space: pre-wrap; word-break: break-word; color:'+baseColor+';">'+ansiToHtml(log)+'</div>';
                }).join('');
            } else {
                cl.innerHTML = '<div style="text-align:center;color:#475569;padding:18px;">No logs available</div>';
            }
            if (atb) cl.scrollTop = cl.scrollHeight;
        }
        function applyConsoleFilter() { _renderConsoleLogs(); }
        function toggleConsolePause() {
            consolePaused = !consolePaused;
            const btn = document.getElementById('console-pause-btn');
            if (consolePaused) { btn.textContent = '\u25B6 Resume'; btn.classList.add('paused'); btn.title = 'Resume console output'; btn.setAttribute('aria-pressed', 'true'); }
            else { btn.textContent = '\u25AE\u25AE Pause'; btn.classList.remove('paused'); btn.title = 'Pause console output'; btn.setAttribute('aria-pressed', 'false'); }
        }
        let _jobPopsPauseInFlight = false;
        let _jobPopsPauseUntil = null;
        function _formatPauseRemaining(pauseUntil) {
            if (pauseUntil === null || pauseUntil === undefined) return null;
            const rem = Math.max(0, Math.round(pauseUntil - Date.now() / 1000));
            if (rem <= 0) return '0s';
            if (rem < 60) return rem + 's';
            const m = Math.floor(rem / 60), s = rem % 60;
            if (rem < 3600) return m + 'm ' + (s > 0 ? s + 's' : '');
            const h = Math.floor(rem / 3600), rm = Math.floor((rem % 3600) / 60);
            return h + 'h ' + (rm > 0 ? rm + 'm' : '');
        }
        function _updatePauseBtn(paused, pauseUntil) {
            const btn = document.getElementById('job-pops-pause-btn');
            if (!btn) return;
            if (paused) {
                const rem = _formatPauseRemaining(pauseUntil);
                btn.textContent = '\u25B6 Resume Jobs' + (rem ? ' (' + rem.trim() + ')' : '');
                btn.classList.add('paused');
                btn.title = 'Resume accepting new jobs from the Horde';
                btn.setAttribute('aria-pressed', 'true');
            } else {
                btn.textContent = 'Pause Jobs';
                btn.classList.remove('paused');
                btn.title = 'Pause or resume accepting new jobs from the Horde';
                btn.setAttribute('aria-pressed', 'false');
            }
        }
        function handleJobPopsPauseBtn(event) {
            const btn = document.getElementById('job-pops-pause-btn');
            if (btn && btn.classList.contains('paused')) {
                closePauseMenu();
                setJobPopsPause(false);
            } else {
                const menu = document.getElementById('job-pops-pause-menu');
                if (!menu) return;
                const open = menu.style.display !== 'none';
                if (open) { closePauseMenu(); } else { menu.style.display = 'block'; }
            }
        }
        function closePauseMenu() {
            const menu = document.getElementById('job-pops-pause-menu');
            if (menu) menu.style.display = 'none';
        }
        document.addEventListener('click', function(e) {
            const wrap = document.querySelector('.job-pops-pause-wrap');
            if (wrap && !wrap.contains(e.target)) closePauseMenu();
        });
        function setJobPopsPause(durationSeconds) {
            closePauseMenu();
            if (_jobPopsPauseInFlight) return;
            const btn = document.getElementById('job-pops-pause-btn');
            _jobPopsPauseInFlight = true;
            if (btn) btn.disabled = true;
            const body = durationSeconds === false
                ? {paused: false}
                : {paused: true, duration_seconds: durationSeconds};
            fetch('/api/job_pops/pause', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            })
            .then(r => { if (!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
            .then(data => {
                _jobPopsPauseUntil = data.job_pops_pause_until ?? null;
                _updatePauseBtn(data.job_pops_paused, _jobPopsPauseUntil);
                _updateStatusBadges(null, data.job_pops_paused, _jobPopsPauseUntil);
            })
            .catch(err => { console.error('Error toggling job pops pause:', err); })
            .finally(() => { _jobPopsPauseInFlight = false; if (btn) btn.disabled = false; });
        }
        let _lastMaintenanceMode = false;
        let _lastJobPopsPaused = false;
        function _updateStatusBadges(maintenanceMode, paused, pauseUntil) {
            if (maintenanceMode !== null) _lastMaintenanceMode = maintenanceMode;
            if (paused !== null && paused !== undefined) _lastJobPopsPaused = paused;
            const maint = _lastMaintenanceMode;
            const isPaused = _lastJobPopsPaused;
            const rem = isPaused ? _formatPauseRemaining(pauseUntil !== undefined ? pauseUntil : _jobPopsPauseUntil) : null;
            const remTxt = rem ? ' \u2022 ' + rem.trim() : '';
            const badgeHtml = maint
                ? '<span class="status-badge status-maintenance">Maintenance</span>'
                : (isPaused
                    ? '<span class="status-badge status-paused">Paused' + remTxt + '</span>'
                    : '<span class="status-badge status-active">Active</span>');
            document.getElementById('worker-status-badge').innerHTML = badgeHtml;
            document.getElementById('mobile-status-badge').innerHTML = maint
                ? '<span class="status-badge status-maintenance" style="font-size:0.68rem;padding:2px 7px;">Maint.</span>'
                : (isPaused
                    ? '<span class="status-badge status-paused" style="font-size:0.68rem;padding:2px 7px;">Paused' + remTxt + '</span>'
                    : '<span class="status-badge status-active" style="font-size:0.68rem;padding:2px 7px;">Active</span>');
            if (!_jobPopsPauseInFlight) _updatePauseBtn(isPaused, _jobPopsPauseUntil);
            const maintBtn = document.getElementById('clear-maintenance-btn');
            if (maintBtn) maintBtn.style.display = maint ? 'inline-flex' : 'none';
        }
        let _clearMaintenanceInFlight = false;
        function clearMaintenanceMode() {
            if (_clearMaintenanceInFlight) return;
            _clearMaintenanceInFlight = true;
            const btn = document.getElementById('clear-maintenance-btn');
            if (btn) btn.disabled = true;
            fetch('/api/maintenance/clear', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.error) console.error('Error clearing maintenance:', data.error);
                    else _updateStatusBadges(false, null, null);
                })
                .catch(err => { console.error('Error clearing maintenance mode:', err); })
                .finally(() => { _clearMaintenanceInFlight = false; if (btn) btn.disabled = false; });
        }
        setInterval(function() {
            if (_lastJobPopsPaused && _jobPopsPauseUntil !== null) {
                _updateStatusBadges(null, null, _jobPopsPauseUntil);
            }
        }, 1000);
        function setMaxQueueSize() {
            const inp = document.getElementById('queue-max-input');
            if (!inp) return;
            const val = parseInt(inp.value, 10);
            if (isNaN(val) || val < 0) { _setSettingsStatus('Queue size must be >= 0', true); return; }
            stageQueueSetting({max_queue_size: val, auto: false});
        }
        function toggleQueueSizeAuto() {
            const btn = document.getElementById('queue-auto-btn');
            const currentlyAuto = btn && btn.classList.contains('active');
            stageQueueSetting({auto: !currentlyAuto});
        }
        function setMaxActiveModels() {
            const inp = document.getElementById('models-max-input');
            if (!inp) return;
            const val = parseInt(inp.value, 10);
            if (isNaN(val) || val < 1) { _setSettingsStatus('Max active models must be >= 1', true); return; }
            stageModelsSetting({max_active_models: val, auto: false});
        }
        function toggleMaxActiveModelsAuto() {
            const btn = document.getElementById('models-auto-btn');
            const currentlyAuto = btn && btn.classList.contains('active');
            stageModelsSetting({auto: !currentlyAuto});
        }
        let _consoleCopyTimeout = null;
        function showConsoleCopySuccess(btn) {
            if (_consoleCopyTimeout) { clearTimeout(_consoleCopyTimeout); _consoleCopyTimeout = null; }
            btn.textContent = '\u2714 Copied!'; btn.classList.remove('error'); btn.classList.add('copied');
            _consoleCopyTimeout = setTimeout(function() { btn.textContent = '\uD83D\uDCCB Copy'; btn.classList.remove('copied'); _consoleCopyTimeout = null; }, 2000);
        }
        function showConsoleCopyError(btn) {
            if (_consoleCopyTimeout) { clearTimeout(_consoleCopyTimeout); _consoleCopyTimeout = null; }
            btn.textContent = '\u2716 Failed'; btn.classList.remove('copied'); btn.classList.add('error');
            _consoleCopyTimeout = setTimeout(function() { btn.textContent = '\uD83D\uDCCB Copy'; btn.classList.remove('error'); _consoleCopyTimeout = null; }, 2000);
        }
        function fallbackCopyText(text) {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.setAttribute('readonly', '');
            textarea.style.position = 'fixed';
            textarea.style.top = '-9999px';
            textarea.style.left = '-9999px';
            document.body.appendChild(textarea);
            textarea.focus();
            textarea.select();
            let copied = false;
            try { copied = document.execCommand('copy'); } catch (err) { copied = false; }
            document.body.removeChild(textarea);
            return copied;
        }
        function copyConsoleLogs() {
            const filterLevel = (document.getElementById('console-filter-select') || {}).value || 'ALL';
            const minOrder = filterLevel === 'ALL' ? -1 : (_CONSOLE_LEVEL_ORDER[filterLevel] !== undefined ? _CONSOLE_LEVEL_ORDER[filterLevel] : -1);
            const logs = minOrder < 0 ? _consoleLogs : _consoleLogs.filter(function(log) {
                const lvl = _getLogLevel(log);
                const ord = _CONSOLE_LEVEL_ORDER[lvl] !== undefined ? _CONSOLE_LEVEL_ORDER[lvl] : 2;
                return ord >= minOrder;
            });
            const text = logs.map(function(log) { return log.replace(_CONSOLE_ANSI_RE, ''); }).join('\n');
            const btn = document.getElementById('console-copy-btn');
            if (navigator.clipboard && window.isSecureContext) {
                navigator.clipboard.writeText(text).then(function() {
                    showConsoleCopySuccess(btn);
                }).catch(function() {
                    if (fallbackCopyText(text)) showConsoleCopySuccess(btn);
                    else showConsoleCopyError(btn);
                });
                return;
            }
            if (fallbackCopyText(text)) showConsoleCopySuccess(btn);
            else showConsoleCopyError(btn);
        }
        const ERRORS_PAGE_SIZE = 10;
        let errorsCurrentPage = 1, errorsTotal = 0, errorsTotalPages = 1, errorsPageData = [];
        let _errorsAbortController = null;
        let errorsViewMode = 'grouped'; // 'grouped' or 'all'
        // Grouped-view state
        let errorsGroupedCurrentPage = 1, errorsGroupedTotalPages = 1, errorsGroupedData = [];
        let _errorsGroupedAbortController = null;

        function setErrorsView(mode) {
            errorsViewMode = mode;
            document.getElementById('errors-btn-grouped').classList.toggle('active', mode === 'grouped');
            document.getElementById('errors-btn-all').classList.toggle('active', mode === 'all');
            if (mode === 'grouped') {
                errorsGroupedCurrentPage = 1;
                fetchErrorsGroupedPage(1);
            } else {
                errorsCurrentPage = 1;
                fetchErrorsPage(1);
            }
        }

        function fetchErrorsGroupedPage(page) {
            if (_errorsGroupedAbortController) _errorsGroupedAbortController.abort();
            _errorsGroupedAbortController = new AbortController();
            const ctrl = _errorsGroupedAbortController;
            fetch('/api/errors/grouped?page='+page+'&page_size='+ERRORS_PAGE_SIZE, { signal: ctrl.signal })
                .then(r => { if (!r.ok) throw new Error('HTTP error! status: '+r.status); return r.json(); })
                .then(data => {
                    if (ctrl !== _errorsGroupedAbortController) return;
                    _errorsGroupedAbortController = null;
                    errorsGroupedCurrentPage = data.page;
                    errorsGroupedTotalPages = data.total_pages;
                    errorsGroupedData = data.groups || [];
                    errorsTotal = data.total_errors;
                    renderErrorsGroupedPage();
                })
                .catch(err => { if (err.name !== 'AbortError') console.error('Failed to fetch /api/errors/grouped:', err); });
        }

        function renderErrorsGroupedPage() {
            const ed = document.getElementById('errors-history'), pi = document.getElementById('errors-page-info'),
                  pb = document.getElementById('errors-prev'), nb = document.getElementById('errors-next'),
                  pag = document.getElementById('errors-pagination'), cnt = document.getElementById('errors-count');
            if (errorsTotal === 0) {
                ed.innerHTML = '<div class="empty-state"><span class="empty-state-icon">&#10003;</span>No errors</div>';
                pag.style.display = 'none'; cnt.textContent = '0'; return;
            }
            const MAX_OCCURRENCES_SHOWN = 50;
            ed.innerHTML = errorsGroupedData.map((grp, idx) => {
                const groupId = 'errgrp-' + errorsGroupedCurrentPage + '-' + idx;
                return '<div class="error-group" id="'+groupId+'">'
                    + '<button class="error-group-header" onclick="toggleErrorGroup(\''+groupId+'\')" aria-expanded="false">'
                    + '<span class="error-group-toggle">&#9658;</span>'
                    + '<span class="error-group-msg">'+escapeHtml(grp.message)+'</span>'
                    + '<span class="error-count-badge">&times;'+grp.count+'</span>'
                    + '</button>'
                    + '<div class="error-group-body"></div>'
                    + '</div>';
            }).join('');
            pi.textContent = 'Page '+errorsGroupedCurrentPage+' of '+errorsGroupedTotalPages;
            pb.disabled = errorsGroupedCurrentPage <= 1;
            nb.disabled = errorsGroupedCurrentPage >= errorsGroupedTotalPages;
            pag.style.display = 'flex'; cnt.textContent = errorsTotal;
            // Store group data on the DOM elements for lazy rendering
            errorsGroupedData.forEach((grp, idx) => {
                const groupId = 'errgrp-' + errorsGroupedCurrentPage + '-' + idx;
                const el = document.getElementById(groupId);
                if (el) el._grpData = grp;
            });
        }

        function toggleErrorGroup(groupId) {
            const el = document.getElementById(groupId);
            if (!el) return;
            const isOpen = el.classList.toggle('open');
            const btn = el.querySelector('.error-group-header');
            if (btn) btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
            // Lazy-render occurrences on first expand
            if (isOpen) {
                const body = el.querySelector('.error-group-body');
                if (body && !body._rendered && el._grpData) {
                    const grp = el._grpData;
                    const occurrences = grp.occurrences || [];
                    let html = '';
                    for (let i = 0; i < occurrences.length; i++) {
                        html += '<div class="error-occurrence">'+escapeHtml(occurrences[i])+'</div>';
                    }
                    const remaining = grp.count - occurrences.length;
                    if (remaining > 0) {
                        html += '<div class="error-occurrence-more">&hellip; and '+remaining+' more occurrence(s)</div>';
                    }
                    body.innerHTML = html;
                    body._rendered = true;
                }
            }
        }

        function fetchErrorsPage(page) {
            if (_errorsAbortController) _errorsAbortController.abort();
            _errorsAbortController = new AbortController();
            const ctrl = _errorsAbortController;
            fetch('/api/errors?page='+page+'&page_size='+ERRORS_PAGE_SIZE, { signal: ctrl.signal })
                .then(r => { if (!r.ok) throw new Error('HTTP error! status: '+r.status); return r.json(); })
                .then(data => {
                    if (ctrl !== _errorsAbortController) return;
                    _errorsAbortController = null;
                    errorsCurrentPage = data.page;
                    errorsTotal = data.total;
                    errorsTotalPages = data.total_pages;
                    errorsPageData = data.errors || [];
                    renderErrorsPage();
                })
                .catch(err => { if (err.name !== 'AbortError') console.error('Failed to fetch /api/errors:', err); });
        }
        function renderErrorsPage() {
            const ed = document.getElementById('errors-history'), pi = document.getElementById('errors-page-info'),
                  pb = document.getElementById('errors-prev'), nb = document.getElementById('errors-next'),
                  pag = document.getElementById('errors-pagination'), cnt = document.getElementById('errors-count');
            if (errorsTotal === 0) {
                ed.innerHTML = '<div class="empty-state"><span class="empty-state-icon">&#10003;</span>No errors</div>';
                pag.style.display = 'none'; cnt.textContent = '0'; return;
            }
            ed.innerHTML = errorsPageData.map(err => '<div class="error-item">'+escapeHtml(err)+'</div>').join('');
            pi.textContent = 'Page '+errorsCurrentPage+' of '+errorsTotalPages;
            pb.disabled = errorsCurrentPage <= 1; nb.disabled = errorsCurrentPage >= errorsTotalPages;
            pag.style.display = 'flex'; cnt.textContent = errorsTotal;
        }
        function errorsChangePage(delta) {
            if (errorsViewMode === 'grouped') {
                const newPage = Math.min(Math.max(1, errorsGroupedCurrentPage + delta), errorsGroupedTotalPages);
                if (newPage !== errorsGroupedCurrentPage) fetchErrorsGroupedPage(newPage);
            } else {
                const newPage = Math.min(Math.max(1, errorsCurrentPage + delta), errorsTotalPages);
                if (newPage !== errorsCurrentPage) fetchErrorsPage(newPage);
            }
        }
        function clearErrors() {
            const btn = document.getElementById('errors-clear-btn');
            if (btn) btn.disabled = true;
            fetch('/api/errors/clear', { method: 'POST' })
                .then(r => r.json())
                .then(function() {
                    errorsTotal = 0; errorsPageData = []; errorsCurrentPage = 1; errorsTotalPages = 1;
                    errorsGroupedData = []; errorsGroupedCurrentPage = 1; errorsGroupedTotalPages = 1;
                    renderErrorsPage();
                    renderErrorsGroupedPage();
                    const cnt = document.getElementById('errors-count');
                    if (cnt) cnt.textContent = '0';
                })
                .catch(function(e) { console.error('Failed to clear errors:', e); })
                .finally(function() { if (btn) btn.disabled = false; });
        }
        // Sync the select element's initial value with the JS constant (single source of truth)
        document.getElementById('gallery-page-size').value = String(GALLERY_DEFAULT_PAGE_SIZE);
        let lastKnownImagesCount = -1; // -1 = sentinel: first status poll not yet completed
        // Set to true when new images arrive while the gallery tab is not active.
        // Consumed by showPage() to refresh or show the banner on return.
        let galleryHasUnseenImages = false;
        // Ordered list of gallery_ids for the currently displayed page; used by overlay navigation.
        // Stored at module scope so incremental updates (refreshGalleryPage1) keep it in sync.
        let _currentPageGalleryIds = [];
        const GALLERY_VALID_COLS = [1, 2, 3, 4, 6, 12];
        const GALLERY_MIN_ITEM_PX = 160;
        const GALLERY_GRID_GAP_PX = 10;
        function updateGalleryColumns() {
            const grid = document.getElementById('gallery-grid');
            if (!grid) return;
            const width = grid.clientWidth;
            if (!width) return;
            // Account for gaps between columns so tiles never shrink below GALLERY_MIN_ITEM_PX.
            // For n columns there are (n-1) gaps, so the available width per column is
            // (width - (n-1)*gap) / n >= GALLERY_MIN_ITEM_PX, i.e. n <= (width + gap) / (GALLERY_MIN_ITEM_PX + gap).
            const rawCols = Math.max(1, Math.floor((width + GALLERY_GRID_GAP_PX) / (GALLERY_MIN_ITEM_PX + GALLERY_GRID_GAP_PX)));
            const cols = GALLERY_VALID_COLS.filter(c => c <= rawCols).pop() || 1;
            grid.style.gridTemplateColumns = 'repeat(' + cols + ', minmax(' + GALLERY_MIN_ITEM_PX + 'px, 1fr))';
        }
        let _galleryResizeObserver = null;
        if (typeof ResizeObserver !== 'undefined') {
            _galleryResizeObserver = new ResizeObserver(function() { updateGalleryColumns(); });
            const _galleryGridEl = document.getElementById('gallery-grid');
            if (_galleryGridEl) _galleryResizeObserver.observe(_galleryGridEl);
        }
        function renderGalleryPageSkeleton(images, total, page, totalPages) {
            galleryTotalImages = total; galleryCurrentPage = page; galleryTotalPages = totalPages;
            _galleryCurrentPageImages = images; // cache for view-mode switch
            const grid = document.getElementById('gallery-grid'), empty = document.getElementById('gallery-empty'),
                  pi = document.getElementById('gallery-page-info'), pb = document.getElementById('gallery-prev'),
                  nb = document.getElementById('gallery-next'), pag = document.getElementById('gallery-pagination');
            if (page === 1) { const bnr = document.getElementById('gallery-new-banner'); if (bnr) bnr.style.display = 'none'; }
            if (images.length === 0) {
                grid.style.display = 'none'; grid.innerHTML = ''; empty.style.display = '';
                pag.style.display = 'none'; return;
            }
            empty.style.display = 'none'; grid.style.display = '';
            const isList = galleryViewMode === 'list';
            grid.className = isList ? 'gallery-list' : 'image-grid';
            if (!isList) updateGalleryColumns();
            _currentPageGalleryIds = images.map(img => img.gallery_id);
            grid.innerHTML = images.map((img, idx) => {
                const galleryId = img.gallery_id;
                const ts = formatTimestamp(img.timestamp), model = img.model ? escapeHtml(img.model) : '';
                const isNsfw = img.is_nsfw === true, isCsam = img.is_csam === true;
                const nsfwAttr = isNsfw ? ' data-nsfw="1"' : '', csamAttr = isCsam ? ' data-csam="1"' : '';
                // A real <img src="/api/gallery/thumb/ID"> lets the browser's native lazy-loading
                // (loading="lazy") and HTTP cache do all the work: revisiting a page, switching
                // view modes, or even reloading the whole app serves previously-seen thumbnails
                // instantly from cache instead of re-fetching them (see
                // _GALLERY_IMAGE_CACHE_HEADERS on the server -- gallery images are immutable
                // once generated, so the cache never needs to revalidate).
                const thumbSrc = '/api/gallery/thumb/'+galleryId;
                if (isList) {
                    const tsLong = formatTimestampFull(img.timestamp);
                    const steps = img.inference_steps ? img.inference_steps + ' steps' : '';
                    const posPromptHtml = renderPromptDiff(img.positive_prompt || '', img.original_positive_prompt || '');
                    const negPromptHtml = renderPromptDiff(img.negative_prompt || '', img.original_negative_prompt || '');
                    const posTitle = img.positive_prompt ? escapeHtml(img.positive_prompt) : '';
                    const negTitle = img.negative_prompt ? escapeHtml(img.negative_prompt) : '';
                    const flagBadges = (isNsfw || isCsam) ? '<div class="gallery-list-badges">'+(isCsam ? '<span class="image-flag-badge csam">CSAM</span>' : '')+(isNsfw ? '<span class="image-flag-badge nsfw">NSFW</span>' : '')+'</div>' : '';
                    const row1 = (model || steps) ? '<div class="gallery-list-row1">'+(model ? '<span class="gallery-list-model">'+model+'</span>' : '')+(steps ? '<span class="gallery-list-steps">'+steps+'</span>' : '')+'</div>' : '';
                    return '<div class="gallery-list-item loading"'+nsfwAttr+csamAttr+' data-gallery-id="'+galleryId+'">' +
                        '<div class="gallery-list-thumb"><img alt="Generated image" src="'+thumbSrc+'" loading="lazy" decoding="async" data-gallery-id="'+galleryId+'" data-idx="'+idx+'" /></div>' +
                        '<div class="gallery-list-meta">' +
                        row1 +
                        (tsLong ? '<div class="gallery-list-ts">'+tsLong+'</div>' : '') +
                        (posPromptHtml ? '<div class="gallery-list-prompt pos" title="'+posTitle+'">'+posPromptHtml+'</div>' : '') +
                        (negPromptHtml ? '<div class="gallery-list-prompt neg" title="'+negTitle+'">'+negPromptHtml+'</div>' : '') +
                        flagBadges +
                        '</div></div>';
                }
                const flagBadges = (isNsfw || isCsam) ? '<div class="image-flag-badges">'+(isCsam ? '<span class="image-flag-badge csam">CSAM</span>' : '')+(isNsfw ? '<span class="image-flag-badge nsfw">NSFW</span>' : '')+'</div>' : '';
                const cap = [ts, model].filter(Boolean).join(' \u00b7 ');
                return '<div class="image-grid-item loading" data-gallery-id="'+galleryId+'"'+nsfwAttr+csamAttr+'><img alt="Generated image" src="'+thumbSrc+'" loading="lazy" decoding="async" data-gallery-id="'+galleryId+'" data-idx="'+idx+'" />'+
                    flagBadges+(cap ? '<div class="image-timestamp">'+cap+'</div>' : '')+'</div>';
            }).join('');
            grid.querySelectorAll('img[data-gallery-id]').forEach(img => {
                img.onclick = function() {
                    const galleryId = parseInt(this.getAttribute('data-gallery-id') || '0', 10);
                    const localIdx = parseInt(this.getAttribute('data-idx') || '0', 10);
                    openGalleryImageOverlay(galleryId, _currentPageGalleryIds, localIdx);
                };
                // Stop the loading shimmer once the browser has actually resolved the thumbnail
                // (loaded or failed) -- a cache hit fires 'load' almost immediately, an
                // already-complete image (rare, but possible) needs no event at all.
                const stopLoading = function() {
                    const c = img.closest('.image-grid-item,.gallery-list-item');
                    if (c) c.classList.remove('loading');
                };
                if (img.complete) stopLoading(); else { img.onload = stopLoading; img.onerror = stopLoading; }
            });
            const tp = Math.max(1, totalPages);
            pi.textContent = 'Page '+page+' of '+tp;
            pb.disabled = page <= 1; nb.disabled = page >= tp;
            pag.style.display = 'flex';
        }
        function setGalleryView(mode) {
            galleryViewMode = mode;
            try { localStorage.setItem('horde-gallery-view', mode); } catch(e) {}
            document.querySelectorAll('.gallery-view-btn').forEach(function(b) { b.classList.remove('active'); });
            var btn = document.getElementById('gallery-view-' + mode);
            if (btn) btn.classList.add('active');
            if (_galleryCurrentPageImages) {
                renderGalleryPageSkeleton(_galleryCurrentPageImages, galleryTotalImages, galleryCurrentPage, galleryTotalPages);
            }
        }
        // AbortController for the background next-page metadata prefetch; aborted if a new
        // prefetch (or a real page navigation) supersedes it before it completes.
        let _galleryPrefetchAbort = null;
        // Warms the browser's native HTTP cache for an adjacent page's thumbnails so navigating
        // there renders from cache instead of waiting on a network round trip. Only fetches
        // metadata itself (cheap); actually warming each thumbnail is just `new Image().src = ...`
        // -- the browser does the fetching, decoding, and caching, and simply discards the result
        // if the image is never attached to the DOM. If new images arrive in the meantime and
        // shift page boundaries, the prefetch just becomes a partial cache hit rather than
        // showing stale data (the real navigation always re-fetches its own metadata).
        function prefetchAdjacentGalleryPage(page) {
            if (page < 1 || page > galleryTotalPages || page === galleryCurrentPage) return;
            if (_galleryPrefetchAbort) { try { _galleryPrefetchAbort.abort(); } catch(_){} }
            const ctrl = new AbortController();
            _galleryPrefetchAbort = ctrl;
            const modelParam = galleryModelFilter ? '&model='+encodeURIComponent(galleryModelFilter) : '';
            const safetyParam = gallerySafetyFilter ? '&safety='+encodeURIComponent(gallerySafetyFilter) : '';
            fetch('/api/gallery?page='+page+'&page_size='+galleryPageSize+'&metadata_only=true'+modelParam+safetyParam,
                { signal: ctrl.signal, priority: 'low' })
                .then(r => { if (!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
                .then(data => {
                    (data.images || []).forEach(entry => { new Image().src = '/api/gallery/thumb/'+entry.gallery_id; });
                })
                .catch(err => { if (err.name !== 'AbortError') console.debug('Gallery prefetch skipped:', err); });
        }
        function fetchGalleryPage(page) {
            if (galleryFetchInProgress) return;
            galleryFetchInProgress = true;
            // Show a loading indicator and hide the empty-state while the fetch is in progress
            // so the "No images generated yet" message is not shown before we know the result.
            const glEl = document.getElementById('gallery-loading'), geEl = document.getElementById('gallery-empty');
            if (glEl) glEl.style.display = 'block';
            if (geEl) geEl.style.display = 'none';
            const modelParam = galleryModelFilter ? '&model='+encodeURIComponent(galleryModelFilter) : '';
            const safetyParam = gallerySafetyFilter ? '&safety='+encodeURIComponent(gallerySafetyFilter) : '';
            // Only fetch lightweight metadata up front — renderGalleryPageSkeleton() renders the
            // skeleton immediately with real <img src="/api/gallery/thumb/ID"> tags, so the
            // browser's own native lazy-loading and HTTP cache handle each thumbnail instead of
            // fetching a whole page's worth of image data (up to 96 images) before anything shows.
            fetch('/api/gallery?page='+page+'&page_size='+galleryPageSize+'&metadata_only=true'+modelParam+safetyParam)
                .then(r => { if (!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
                .then(data => {
                    if (glEl) glEl.style.display = 'none';
                    renderGalleryPageSkeleton(data.images, data.total, data.page, data.total_pages);
                    galleryFetchInProgress = false;
                    // Warm the next page's thumbnail cache in the background (after this
                    // page's own request is underway) so clicking "Next" typically renders
                    // instantly instead of waiting on a fresh round trip.
                    if (typeof requestIdleCallback === 'function') {
                        requestIdleCallback(function() { prefetchAdjacentGalleryPage(page + 1); }, { timeout: 2000 });
                    } else {
                        setTimeout(function() { prefetchAdjacentGalleryPage(page + 1); }, 300);
                    }
                })
                .catch(err => {
                    console.error('Gallery fetch error:', err);
                    if (glEl) glEl.style.display = 'none';
                    galleryFetchInProgress = false;
                    // On error, if the grid has no content (e.g. first load) show the empty-state
                    // so the user isn't left with a completely blank gallery card.
                    const gridEl = document.getElementById('gallery-grid');
                    const hasGridContent = gridEl && gridEl.querySelector('.image-grid-item');
                    if (!hasGridContent) {
                        if (geEl) geEl.style.display = '';
                    }
                });
        }
        // Incrementally update page 1 when new images arrive: prepend only new tiles and load
        // their thumbnails without disturbing images that are already loaded in the grid.
        function refreshGalleryPage1() {
            if (galleryFetchInProgress) return;
            galleryFetchInProgress = true;
            const modelParam = galleryModelFilter ? '&model='+encodeURIComponent(galleryModelFilter) : '';
            const safetyParam = gallerySafetyFilter ? '&safety='+encodeURIComponent(gallerySafetyFilter) : '';
            fetch('/api/gallery?page=1&page_size='+galleryPageSize+'&metadata_only=true'+modelParam+safetyParam)
                .then(r => { if (!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
                .then(data => {
                    galleryFetchInProgress = false;
                    galleryCurrentPage = 1; galleryTotalImages = data.total; galleryTotalPages = data.total_pages;
                    _galleryCurrentPageImages = data.images;
                    const grid = document.getElementById('gallery-grid'),
                          pi = document.getElementById('gallery-page-info'), pb = document.getElementById('gallery-prev'),
                          nb = document.getElementById('gallery-next'), pag = document.getElementById('gallery-pagination'),
                          bnr = document.getElementById('gallery-new-banner'), empty = document.getElementById('gallery-empty');
                    if (bnr) bnr.style.display = 'none';
                    const tp = Math.max(1, data.total_pages);
                    pi.textContent = 'Page 1 of '+tp; pb.disabled = true; nb.disabled = 1 >= tp; pag.style.display = 'flex';
                    if (data.images.length === 0) {
                        grid.style.display = 'none'; grid.innerHTML = ''; empty.style.display = '';
                        pag.style.display = 'none'; return;
                    }
                    empty.style.display = 'none'; grid.style.display = '';
                    const isList = galleryViewMode === 'list';
                    grid.className = isList ? 'gallery-list' : 'image-grid';
                    if (!isList) updateGalleryColumns();
                    const fetchedIds = data.images.map(img => img.gallery_id);
                    const fetchedSet = new Set(fetchedIds);
                    const existingItems = Array.from(grid.querySelectorAll('[data-gallery-id]'));
                    const existingSet = new Set(existingItems.map(el => parseInt(el.getAttribute('data-gallery-id'), 10)));
                    // Remove tiles that have been pushed off the current page by new arrivals.
                    existingItems.forEach(el => { if (!fetchedSet.has(parseInt(el.getAttribute('data-gallery-id'), 10))) el.remove(); });
                    // Update the stable ID list and data-idx attributes for correct overlay navigation.
                    _currentPageGalleryIds = fetchedIds;
                    data.images.forEach((img, idx) => {
                        const ie = grid.querySelector('img[data-gallery-id="'+img.gallery_id+'"]');
                        if (ie) ie.setAttribute('data-idx', idx);
                    });
                    const newImages = data.images.reduce((acc, img, idx) => {
                        if (!existingSet.has(img.gallery_id)) acc.push({ img, idx });
                        return acc;
                    }, []);
                    if (newImages.length > 0) {
                        const frag = document.createDocumentFragment();
                        newImages.forEach(({ img, idx }) => {
                            const galleryId = img.gallery_id;
                            const ts = formatTimestamp(img.timestamp);
                            const model = img.model ? escapeHtml(img.model) : '';
                            const isNsfw = img.is_nsfw === true, isCsam = img.is_csam === true;
                            // Real <img src="/api/gallery/thumb/ID"> -- see renderGalleryPageSkeleton
                            // for why this is enough for native lazy-loading + HTTP caching.
                            const thumbSrc = '/api/gallery/thumb/'+galleryId;
                            const div = document.createElement('div');
                            if (isList) {
                                const tsLong = formatTimestampFull(img.timestamp);
                                const steps = img.inference_steps ? img.inference_steps + ' steps' : '';
                                const posPromptHtml = renderPromptDiff(img.positive_prompt || '', img.original_positive_prompt || '');
                                const negPromptHtml = renderPromptDiff(img.negative_prompt || '', img.original_negative_prompt || '');
                                const posTitle = img.positive_prompt ? escapeHtml(img.positive_prompt) : '';
                                const negTitle = img.negative_prompt ? escapeHtml(img.negative_prompt) : '';
                                const flagBadges = (isNsfw || isCsam) ? '<div class="gallery-list-badges">'+(isCsam ? '<span class="image-flag-badge csam">CSAM</span>' : '')+(isNsfw ? '<span class="image-flag-badge nsfw">NSFW</span>' : '')+'</div>' : '';
                                const row1 = (model || steps) ? '<div class="gallery-list-row1">'+(model ? '<span class="gallery-list-model">'+model+'</span>' : '')+(steps ? '<span class="gallery-list-steps">'+steps+'</span>' : '')+'</div>' : '';
                                div.className = 'gallery-list-item loading';
                                div.setAttribute('data-gallery-id', galleryId);
                                if (isNsfw) div.setAttribute('data-nsfw', '1');
                                if (isCsam) div.setAttribute('data-csam', '1');
                                div.innerHTML = '<div class="gallery-list-thumb"><img alt="Generated image" src="'+thumbSrc+'" loading="lazy" decoding="async" data-gallery-id="'+galleryId+'" data-idx="'+idx+'" /></div>' +
                                    '<div class="gallery-list-meta">'+row1+(tsLong ? '<div class="gallery-list-ts">'+tsLong+'</div>' : '')+(posPromptHtml ? '<div class="gallery-list-prompt pos" title="'+posTitle+'">'+posPromptHtml+'</div>' : '')+(negPromptHtml ? '<div class="gallery-list-prompt neg" title="'+negTitle+'">'+negPromptHtml+'</div>' : '')+flagBadges+'</div>';
                            } else {
                                const flagBadges = (isNsfw || isCsam) ? '<div class="image-flag-badges">'+(isCsam ? '<span class="image-flag-badge csam">CSAM</span>' : '')+(isNsfw ? '<span class="image-flag-badge nsfw">NSFW</span>' : '')+'</div>' : '';
                                const cap = [ts, model].filter(Boolean).join(' \u00b7 ');
                                div.className = 'image-grid-item loading';
                                div.setAttribute('data-gallery-id', galleryId);
                                if (isNsfw) div.setAttribute('data-nsfw', '1');
                                if (isCsam) div.setAttribute('data-csam', '1');
                                div.innerHTML = '<img alt="Generated image" src="'+thumbSrc+'" loading="lazy" decoding="async" data-gallery-id="'+galleryId+'" data-idx="'+idx+'" />'+flagBadges+(cap ? '<div class="image-timestamp">'+cap+'</div>' : '');
                            }
                            const imgEl = div.querySelector('img');
                            imgEl.onclick = function() { openGalleryImageOverlay(parseInt(this.getAttribute('data-gallery-id')||'0',10), _currentPageGalleryIds, parseInt(this.getAttribute('data-idx')||'0',10)); };
                            const stopLoading = function() { div.classList.remove('loading'); };
                            imgEl.onload = stopLoading; imgEl.onerror = stopLoading;
                            frag.appendChild(div);
                        });
                        grid.insertBefore(frag, grid.firstChild);
                    }
                })
                .catch(err => { console.error('Gallery refresh error:', err); galleryFetchInProgress = false; });
        }
        function galleryChangePage(delta) {
            const newPage = Math.min(Math.max(1, galleryCurrentPage + delta), Math.max(1, galleryTotalPages));
            fetchGalleryPage(newPage);
        }
        function galleryChangePageSize(val) {
            galleryPageSize = parseInt(val, 10) || GALLERY_DEFAULT_PAGE_SIZE;
            fetchGalleryPage(1);
        }
        function galleryChangeModelFilter(val) {
            galleryModelFilter = val;
            fetchGalleryPage(1);
        }
        function galleryChangeSafetyFilter(val) {
            gallerySafetyFilter = val;
            fetchGalleryPage(1);
        }
        function populateGalleryModelFilter() {
            fetch('/api/gallery/models')
                .then(r => { if (!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
                .then(data => {
                    const sel = document.getElementById('gallery-model-filter');
                    if (!sel) return;
                    const prev = sel.value;
                    const models = data.models || [];
                    const total = typeof data.total === 'number' ? data.total : models.reduce(function(s, m) { return s + m.count; }, 0);
                    sel.innerHTML = '<option value="">All models' + (total ? ' (' + total + ')' : '') + '</option>';
                    models.forEach(function(m) {
                        const opt = document.createElement('option');
                        opt.value = m.name; opt.textContent = m.name + ' (' + m.count + ')';
                        if (m.name === prev) opt.selected = true;
                        sel.appendChild(opt);
                    });
                    // If the previously selected model is no longer in the list, reset to "all".
                    if (prev && !models.some(function(m) { return m.name === prev; })) {
                        sel.value = '';
                        if (galleryModelFilter !== '') { galleryModelFilter = ''; fetchGalleryPage(1); }
                    }
                })
                .catch(err => { console.error('Gallery models fetch error:', err); });
        }
        function populateGallerySafetyFilter() {
            fetch('/api/gallery/safety')
                .then(r => { if (!r.ok) throw new Error('HTTP '+r.status); return r.json(); })
                .then(data => {
                    const sel = document.getElementById('gallery-safety-filter');
                    if (!sel) return;
                    const prev = sel.value;
                    const total = data.total || 0;
                    const sfw = data.sfw || 0;
                    const nsfw = data.nsfw || 0;
                    const csam = data.csam || 0;
                    sel.innerHTML =
                        '<option value="">' + 'All' + (total ? ' (' + total + ')' : '') + '</option>' +
                        '<option value="sfw">' + 'SFW only' + (sfw ? ' (' + sfw + ')' : '') + '</option>' +
                        '<option value="nsfw">' + 'NSFW' + (nsfw ? ' (' + nsfw + ')' : '') + '</option>' +
                        '<option value="csam">' + 'CSAM' + (csam ? ' (' + csam + ')' : '') + '</option>';
                    sel.value = prev;
                })
                .catch(err => { console.error('Gallery safety fetch error:', err); });
        }
        function isScrolledToBottom(el, tol) { return el.scrollHeight - el.clientHeight <= el.scrollTop + tol; }
        function ansiToHtml(text) {
            text = escapeHtml(text);
            // VS Code / Windows Terminal "Campbell" palette - matches the default
            // appearance most users will see in their standard console.
            const colors = {'30':'#0c0c0c','31':'#cd3131','32':'#0dbc79','33':'#e5e510','34':'#2472c8','35':'#bc3fbc','36':'#11a8cd','37':'#cccccc','90':'#666666','91':'#f14c4c','92':'#23d18b','93':'#f5f543','94':'#3b8eea','95':'#d670d6','96':'#29b8db','97':'#ffffff'};
            const bgColors = {'40':'#0c0c0c','41':'#cd3131','42':'#0dbc79','43':'#e5e510','44':'#2472c8','45':'#bc3fbc','46':'#11a8cd','47':'#cccccc','100':'#666666','101':'#f14c4c','102':'#23d18b','103':'#f5f543','104':'#3b8eea','105':'#d670d6','106':'#29b8db','107':'#ffffff'};
            // Standard xterm 256-color palette (first 16 mirror the colors above; 16-231 form a
            // 6x6x6 RGB cube; 232-255 are a grayscale ramp).
            function ansi256ToHex(n) {
                if (n < 16) {
                    const base = ['30','31','32','33','34','35','36','37','90','91','92','93','94','95','96','97'];
                    return colors[base[n]];
                }
                if (n >= 232) {
                    const v = 8 + (n - 232) * 10;
                    const h = v.toString(16).padStart(2, '0');
                    return '#' + h + h + h;
                }
                const i = n - 16;
                const r = Math.floor(i / 36), g = Math.floor((i % 36) / 6), b = i % 6;
                const cube = [0, 95, 135, 175, 215, 255];
                const toHex = v => v.toString(16).padStart(2, '0');
                return '#' + toHex(cube[r]) + toHex(cube[g]) + toHex(cube[b]);
            }
            let result = '', cs = [];
            const parts = text.split(/\x1b\[([0-9;]+)m/);
            for (let i = 0; i < parts.length; i++) {
                if (i % 2 === 0) { result += cs.length > 0 ? '<span style="'+cs.join(';')+'">'+parts[i]+'</span>' : parts[i]; }
                else {
                    const tokens = parts[i].split(';');
                    for (let j = 0; j < tokens.length; j++) {
                        const c = tokens[j];
                        if (c === '0' || c === '') { cs = []; }
                        else if (c === '1') { if (!cs.some(s => s.startsWith('font-weight:'))) cs.push('font-weight:bold'); }
                        else if (c === '2') { if (!cs.some(s => s.startsWith('opacity:'))) cs.push('opacity:0.6'); }
                        else if (c === '3') { if (!cs.some(s => s.startsWith('font-style:'))) cs.push('font-style:italic'); }
                        else if (c === '4') { if (!cs.some(s => s.startsWith('text-decoration:'))) cs.push('text-decoration:underline'); }
                        else if (c === '22') { cs = cs.filter(s => !s.startsWith('font-weight:') && !s.startsWith('opacity:')); }
                        else if (c === '23') { cs = cs.filter(s => !s.startsWith('font-style:')); }
                        else if (c === '24') { cs = cs.filter(s => !s.startsWith('text-decoration:')); }
                        else if (c === '39') { cs = cs.filter(s => !s.startsWith('color:')); }
                        else if (c === '49') { cs = cs.filter(s => !s.startsWith('background-color:')); }
                        else if (c === '38' || c === '48') {
                            // Extended color: 38;5;N (256) or 38;2;R;G;B (truecolor)
                            const isFg = (c === '38');
                            const mode = tokens[j + 1];
                            let hex = null;
                            if (mode === '5' && tokens[j + 2] !== undefined) {
                                hex = ansi256ToHex(parseInt(tokens[j + 2], 10) || 0);
                                j += 2;
                            } else if (mode === '2' && tokens[j + 4] !== undefined) {
                                const r = parseInt(tokens[j + 2], 10) || 0;
                                const g = parseInt(tokens[j + 3], 10) || 0;
                                const b = parseInt(tokens[j + 4], 10) || 0;
                                hex = 'rgb('+r+','+g+','+b+')';
                                j += 4;
                            } else {
                                // Unknown sub-mode; consume nothing further and skip
                                continue;
                            }
                            if (hex) {
                                if (isFg) { cs = cs.filter(s => !s.startsWith('color:')); cs.push('color:'+hex); }
                                else { cs = cs.filter(s => !s.startsWith('background-color:')); cs.push('background-color:'+hex); }
                            }
                        }
                        else if (colors[c]) { cs = cs.filter(s => !s.startsWith('color:')); cs.push('color:'+colors[c]); }
                        else if (bgColors[c]) { cs = cs.filter(s => !s.startsWith('background-color:')); cs.push('background-color:'+bgColors[c]); }
                    }
                }
            }
            return result;
        }
        let statusAbortController = null, _lastImageFetchController = null, _lastImageFetchTimestamp = null, consecutiveErrors = 0;
        let statusUpdateTimestamp = Date.now(), updateIntervalMs = 1000, scheduledUpdateTimer = null;
        const MAX_CONSECUTIVE_ERRORS = 5;
        function resBarColor(pct) { return pct >= 80 ? '#ef4444' : pct >= 60 ? '#f59e0b' : '#10b981'; }
        let _lastRenderedImageKey = null;
        // Tracks the last image submission timestamp for which images have been fetched.
        // Images are only re-fetched when this value changes, keeping /api/status lightweight.
        let _lastFetchedImageTimestamp = null;
        // Stores the raw timestamp so the 1-second tick can reformat the "X seconds ago" label.
        let _lastImageSubmissionTimestamp = null;
        // Timestamp of the gallery batch shown as the Last Result preview before any image
        // has been generated this session. Kept separate from _lastImageSubmissionTimestamp
        // because every /api/status poll re-derives that from the session value (0 while no
        // session image exists) — a shared variable would be nulled again within a second.
        let _galleryPreviewTimestamp = null;
        // Track the current job id and its highest-seen progress so the bar never goes
        // backwards for the same job.  Reset whenever the displayed job id changes.
        let _currentJobId = null;
        let _currentJobProgress = 0;
        // Track how long the current job has been in its current state, and total job run time.
        let _currentJobState = null;
        let _currentJobStateStartTime = null;
        let _currentJobStartTime = null;
        function formatElapsed(startMs) {
            if (!startMs) return '';
            const s = Math.floor((Date.now() - startMs) / 1000);
            if (s < 60) return s + 's';
            const m = Math.floor(s / 60), rs = s % 60;
            if (s < 3600) return m + 'm ' + rs + 's';
            const h = Math.floor(s / 3600), rm = Math.floor((s % 3600) / 60), rrs = s % 60;
            return h + 'h ' + rm + 'm ' + rrs + 's';
        }
        setInterval(function() {
            const timerEl = document.getElementById('job-state-timer');
            if (timerEl && _currentJobStateStartTime) {
                timerEl.textContent = '(' + formatElapsed(_currentJobStateStartTime) + ')';
            }
            const totalEl = document.getElementById('job-total-timer');
            if (totalEl) {
                totalEl.textContent = _currentJobStartTime ? '\u23F1 ' + formatElapsed(_currentJobStartTime) : '';
            }
            const timeEl = document.getElementById('overview-image-time');
            if (timeEl) {
                // Fall back to the gallery-preview timestamp so the restored Last Result
                // image shows its age before any image is generated this session.
                const labelTimestamp = _lastImageSubmissionTimestamp || _galleryPreviewTimestamp;
                timeEl.textContent = labelTimestamp ? formatTimeAgo(labelTimestamp) : '';
            }
        }, 1000);
        function _getImageKey(rawB64, timestamp, model, safety) {
            if (!rawB64 || rawB64.length === 0) return 'empty';
            // Use count + submission timestamp + model + safety flags as the change-detection key.
            // The first bytes of a PNG base64 string are always a fixed header, so sampling
            // from the beginning is not reliable. The timestamp changes whenever new images arrive.
            // model and safety are included so that badge/model-name updates with identical
            // image count/timestamp are not incorrectly skipped.
            const safetyKey = safety ? safety.map(function(s) { return (s && s.is_nsfw ? 'n' : '-') + (s && s.is_csam ? 'c' : '-'); }).join(',') : '';
            return rawB64.length + ':' + (timestamp || 0) + ':' + (model || '') + ':' + safetyKey;
        }
        function renderLastImages(rawB64, oic, timestamp, model, safety) {
            const key = _getImageKey(rawB64, timestamp, model, safety);
            if (key === _lastRenderedImageKey) return;
            _lastRenderedImageKey = key;
            oic.classList.remove('loading');
            // Capture the render token so async image-load callbacks can detect whether a
            // newer renderLastImages() call has already superseded this one.
            const renderToken = key;
            const modelEl = document.getElementById('overview-image-model');
            function makeFlagBadges(idx) {
                if (!safety || idx >= safety.length) return '';
                const s = safety[idx];
                if (!s) return '';
                const isNsfw = s.is_nsfw === true, isCsam = s.is_csam === true;
                if (!isNsfw && !isCsam) return '';
                return '<div class="image-flag-badges">'+(isCsam ? '<span class="image-flag-badge csam">CSAM</span>' : '')+(isNsfw ? '<span class="image-flag-badge nsfw">NSFW</span>' : '')+'</div>';
            }
            if (!rawB64 || rawB64.length === 0) {
                oic.removeAttribute('style');
                oic.innerHTML = '<div class="empty-state"><span class="empty-state-icon">&#128444;</span>No image generated yet</div>';
                if (modelEl) modelEl.textContent = '';
                return;
            }
            if (modelEl) modelEl.textContent = (model && typeof model === 'string') ? model : '';
            const previewB64 = rawB64.slice(0, 4);
            const count = previewB64.length;
            const srcs = previewB64.map(function(b) { return 'data:image/png;base64,' + b; });
            const allSrcs = rawB64.map(function(b) { return 'data:image/png;base64,' + b; });
            function attachClicks() {
                oic.querySelectorAll('img[data-fullsize]').forEach(function(img) {
                    img.onclick = function() { openImageOverlay(this.getAttribute('data-fullsize'), allSrcs, parseInt(this.getAttribute('data-idx') || '0', 10)); };
                });
            }
            if (count === 1) {
                oic.removeAttribute('style');
                const badges0 = makeFlagBadges(0);
                const isNsfw0 = safety && safety[0] && safety[0].is_nsfw === true;
                const nsfwAttr0 = isNsfw0 ? ' data-nsfw="1"' : '';
                if (badges0) {
                    oic.innerHTML = '<div style="position:relative;display:flex;align-items:center;justify-content:center;width:100%;height:100%;"' + nsfwAttr0 + '>' + badges0 + '<img src="' + srcs[0] + '" class="single-image" alt="Last generated image" data-fullsize="' + srcs[0] + '" data-idx="0" /></div>';
                } else {
                    oic.innerHTML = '<img' + nsfwAttr0 + ' src="' + srcs[0] + '" class="single-image" alt="Last generated image" data-fullsize="' + srcs[0] + '" data-idx="0" />';
                }
                attachClicks();
                return;
            }
            function makeItem(s, i, spanFull) {
                var span = spanFull ? ' style="grid-column:1/-1;"' : '';
                var nsfwAttr = (safety && safety[i] && safety[i].is_nsfw === true) ? ' data-nsfw="1"' : '';
                return '<div class="image-grid-item"' + nsfwAttr + span + '><img src="' + s + '" alt="Generated image ' + (i + 1) + '" data-fullsize="' + s + '" data-idx="' + i + '" />' + makeFlagBadges(i) + '</div>';
            }
            var imgDims = new Array(count).fill(null), loadedCount = 0;
            function renderGrid() {
                if (_lastRenderedImageKey !== renderToken) return;
                var containerWidth = oic.offsetWidth || 320;
                var containerHeight = oic.offsetHeight || 320;
                var gap = 4; // CSS gap in px; must match the gap value in the grid style below
                // Average image aspect ratio (width/height); all images in a batch share the same resolution,
                // so avgImgAR is effectively the single image AR.
                var avgImgAR = imgDims.reduce(function(sum, d) { return sum + (d ? d.w / d.h : 1.0); }, 0) / count;
                // Fraction of a cell's area covered by an image using object-fit:contain.
                // An image of AR imgAR in a cell of AR cellAR fills min(cellAR/imgAR, imgAR/cellAR) of the cell.
                // Choosing the layout with the highest cellEff minimises leftover (unused) container space.
                function cellEff(cellAR, imgAR) {
                    return imgAR > cellAR ? cellAR / imgAR : imgAR / cellAR;
                }
                var gridStyle, items;
                if (count === 2) {
                    // 1×2 side-by-side: 1 horizontal gap; each cell AR = (W−gap)/2 / H.
                    // 2×1 stacked:      1 vertical gap;   each cell AR = W / ((H−gap)/2).
                    var ar1x2 = (containerWidth - gap) / 2 / containerHeight;
                    var ar2x1 = containerWidth * 2 / (containerHeight - gap);
                    if (cellEff(ar1x2, avgImgAR) >= cellEff(ar2x1, avgImgAR)) {
                        gridStyle = 'grid-template-columns:repeat(2,1fr);grid-template-rows:1fr;';
                    } else {
                        gridStyle = 'grid-template-columns:1fr;grid-template-rows:repeat(2,1fr);';
                    }
                    items = srcs.map(function(s, i) { return makeItem(s, i, false); }).join('');
                } else if (count === 3) {
                    // Portrait images (height > width) always look better side-by-side in a 1×3 row
                    // because the 2+1 alternative puts a portrait image into a wide landscape cell,
                    // wasting a lot of space.  For landscape/square images use the area-efficiency
                    // calculation to pick the best layout.
                    if (avgImgAR < 1) {
                        // Portrait: always 1×3 side-by-side
                        gridStyle = 'grid-template-columns:repeat(3,1fr);grid-template-rows:1fr;';
                        items = srcs.map(function(s, i) { return makeItem(s, i, false); }).join('');
                    } else {
                        // 1×3 row:    2 horizontal gaps; each cell AR = (W−2×gap)/3 / H.
                        // 2+1 layout: 1 vertical gap between rows; each row height = (H−gap)/2.
                        //   Top cell spans full width: AR = W / rowH.
                        //   Two bottom cells share a horizontal gap: AR = (W−gap)/2 / rowH.
                        //   Area-weighted efficiency ≈ 0.5×cellEff(top) + 0.5×cellEff(bottom).
                        var rowH = (containerHeight - gap) / 2;
                        var ar1x3 = (containerWidth - 2 * gap) / 3 / containerHeight;
                        var ar2p1top = containerWidth / rowH;
                        var ar2p1bot = (containerWidth - gap) / 2 / rowH;
                        var eff1x3 = cellEff(ar1x3, avgImgAR);
                        var eff2p1 = 0.5 * cellEff(ar2p1top, avgImgAR) + 0.5 * cellEff(ar2p1bot, avgImgAR);
                        if (eff1x3 >= eff2p1) {
                            gridStyle = 'grid-template-columns:repeat(3,1fr);grid-template-rows:1fr;';
                            items = srcs.map(function(s, i) { return makeItem(s, i, false); }).join('');
                        } else {
                            gridStyle = 'grid-template-columns:repeat(2,1fr);grid-template-rows:1fr 1fr;';
                            items = srcs.map(function(s, i) { return makeItem(s, i, i === 0); }).join('');
                        }
                    }
                } else {
                    // count === 4
                    // 1×4 row:  3 horizontal gaps; each column width = (W−3×gap)/4, cell AR = colW / H.
                    //           Requires each column to be at least 120 px wide after subtracting gaps.
                    // 2×2 grid: 1 horizontal + 1 vertical gap; each cell AR = (W−gap) / (H−gap).
                    // 0 (impossible efficiency) disables 1×4 when columns would be narrower than 120 px.
                    var minColumnWidthPx = 120;
                    var col1x4 = (containerWidth - 3 * gap) / 4;
                    var ar1x4 = col1x4 / containerHeight;
                    var ar2x2 = (containerWidth - gap) / (containerHeight - gap);
                    var eff1x4 = col1x4 >= minColumnWidthPx ? cellEff(ar1x4, avgImgAR) : 0;
                    var eff2x2 = cellEff(ar2x2, avgImgAR);
                    if (eff1x4 >= eff2x2) {
                        gridStyle = 'grid-template-columns:repeat(4,1fr);grid-template-rows:1fr;';
                    } else {
                        gridStyle = 'grid-template-columns:repeat(2,1fr);grid-template-rows:1fr 1fr;';
                    }
                    items = srcs.map(function(s, i) { return makeItem(s, i, false); }).join('');
                }
                oic.style.cssText = 'display:grid;width:100%;gap:4px;align-items:stretch;' + gridStyle;
                oic.innerHTML = items;
                attachClicks();
            }
            srcs.forEach(function(src, i) {
                var img = new window.Image();
                img.onload = function() { imgDims[i] = { w: this.naturalWidth, h: this.naturalHeight }; loadedCount++; if (loadedCount === count) renderGrid(); };
                img.onerror = function() { imgDims[i] = { w: 1, h: 1 }; loadedCount++; if (loadedCount === count) renderGrid(); };
                img.src = src;
            });
        }
        function fetchLastImage(timestamp) {
            // Fetch images from the dedicated endpoint so that /api/status stays lightweight.
            // Only called when last_image_submission_timestamp changes.
            // If a fetch for this same timestamp is already in flight, let it complete rather
            // than aborting and restarting on every 1-second status poll.
            if (_lastImageFetchController && _lastImageFetchTimestamp === timestamp) return;
            // A *different* (newer) timestamp supersedes the previous request — abort it so
            // that a stale response can never overwrite a newer image (which would cause the
            // display to flicker back to an older result).
            if (_lastImageFetchController) _lastImageFetchController.abort();
            _lastImageFetchController = new AbortController();
            _lastImageFetchTimestamp = timestamp;
            const ctrl = _lastImageFetchController;
            fetch('/api/last_image', { signal: ctrl.signal })
                .then(r => { if (!r.ok) throw new Error('HTTP error! status: '+r.status); return r.json(); })
                .then(imgData => {
                    // Discard the response if a newer fetch has already superseded this one.
                    if (ctrl !== _lastImageFetchController) return;
                    _lastImageFetchController = null;
                    // Prefer the timestamp from the /api/last_image response so that the
                    // cache marker reflects the actual data that was rendered, not the
                    // /api/status snapshot that triggered the fetch.
                    var ts = imgData && imgData.last_image_submission_timestamp;
                    if (typeof ts !== 'number') { ts = Number(ts); }
                    if (!Number.isFinite(ts)) {
                        ts = (typeof timestamp === 'number' && Number.isFinite(timestamp)) ? timestamp
                            : (Number.isFinite(_lastFetchedImageTimestamp) ? _lastFetchedImageTimestamp : 0);
                    }
                    _lastFetchedImageTimestamp = ts;
                    renderLastImages(imgData.last_image_base64, document.getElementById('overview-image-container'), ts, imgData.last_image_model || null, imgData.last_image_safety || null);
                })
                .catch(function(err) {
                    if (err.name === 'AbortError') return;
                    // Reset the in-flight marker so the next status poll can retry the request.
                    if (ctrl === _lastImageFetchController) { _lastImageFetchController = null; _lastImageFetchTimestamp = null; }
                    // Log the error but do not advance the cache marker so we can retry on the next status poll.
                    console.error('Failed to fetch /api/last_image:', err);
                    var container = document.getElementById('overview-image-container');
                    if (container) {
                        container.classList.remove('loading');
                        if (!container.hasChildNodes()) {
                            container.removeAttribute('style');
                            container.innerHTML = '<div class="empty-state"><span class="empty-state-icon">&#128444;</span>No image generated yet</div>';
                        }
                    }
                });
        }
        function scheduleUpdate() {
            if (scheduledUpdateTimer !== null) return;
            const elapsed = Date.now() - statusUpdateTimestamp;
            const delay = Math.max(0, updateIntervalMs - elapsed);
            scheduledUpdateTimer = setTimeout(updateStatus, delay);
        }
        function updateStatus() {
            scheduledUpdateTimer = null;
            statusUpdateTimestamp = Date.now();
            if (statusAbortController) statusAbortController.abort();
            statusAbortController = new AbortController();
            fetch('/api/status', { signal: statusAbortController.signal })
                .then(r => { if (!r.ok) throw new Error('HTTP error! status: '+r.status); return r.json(); })
                .then(data => {
                    consecutiveErrors = 0;
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('content').style.display = 'block';
                    const workerName = data.worker_name || 'Unknown';
                    currentWorkerName = data.worker_name || '';
                    document.getElementById('topbar-worker-name').textContent = workerName;
                    document.getElementById('topbar-worker-sub').textContent = '@'+data.horde_username;
                    _jobPopsPauseUntil = data.job_pops_pause_until ?? null;
                    _updateStatusBadges(data.maintenance_mode, data.job_pops_paused, _jobPopsPauseUntil);
                    const uptimeStr = formatUptime(data.uptime);
                    document.getElementById('uptime').textContent = uptimeStr;
                    document.getElementById('mobile-uptime').textContent = '\u23F1 ' + uptimeStr;
                    var _rb = data.stats_reset_baseline || {};
                    function _subReset(val, key) { return Math.max(0, (val || 0) - (_rb[key] || 0)); }
                    document.getElementById('overview-images-generated').textContent = _subReset(data.jobs_completed, 'jobs_completed');
                    document.getElementById('images-per-hour').textContent = (data.images_per_hour || 0).toLocaleString(undefined, {maximumFractionDigits: 2});
                    document.getElementById('jobs-popped').textContent = _subReset(data.jobs_popped, 'jobs_popped');
                    document.getElementById('jobs-completed').textContent = _subReset(data.jobs_completed, 'jobs_completed');
                    document.getElementById('jobs-faulted').textContent = _subReset(data.jobs_faulted, 'jobs_faulted');
                    document.getElementById('processes-recovered').textContent = _subReset(data.processes_recovered, 'processes_recovered');
                    document.getElementById('jobs-queued').textContent = data.jobs_queued;
                    document.getElementById('time-without-jobs').textContent = formatUptime(_subReset(data.time_without_jobs, 'time_without_jobs'));
                    const cpu = Math.min(100, Math.round(data.cpu_usage_percent));
                    const workerGpu = Math.min(100, Math.round(data.worker_gpu_percent || 0));
                    const sysGpuRaw = Math.min(100, Math.round(data.gpu_usage_percent || 0));
                    const gpu = Math.max(sysGpuRaw, workerGpu);
                    const vramMb = data.vram_usage_mb || 0;
                    const sysVramMb = data.system_vram_usage_mb || 0;
                    const vramTotalMb = data.total_vram_mb || 0;
                    const vram = vramTotalMb > 0 ? Math.min(100, Math.round((vramMb / vramTotalMb) * 100)) : 0;
                    const sysVram = Math.max(vramTotalMb > 0 ? Math.min(100, Math.round((sysVramMb / vramTotalMb) * 100)) : 0, vram);
                    const ctrCpu = Math.min(100, Math.round(data.container_cpu_percent || 0));
                    const ramMb = data.ram_usage_mb || 0;
                    const totalRamMb = data.total_ram_mb || 0;
                    const sysRamMb = data.system_ram_usage_mb || 0;
                    const ram = totalRamMb > 0 ? Math.min(100, Math.round((ramMb / totalRamMb) * 100)) : 0;
                    const sysRam = totalRamMb > 0 ? Math.min(100, Math.round((sysRamMb / totalRamMb) * 100)) : 0;
                    const cores = data.cpu_cores_count || 0;
                    const gpuCores = data.gpu_cores_count || 0;
                    // Format a MB value to "X.X GB" or "X MB"
                    function formatMb(mb) { return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : Math.round(mb) + ' MB'; }
                    document.getElementById('topbar-cpu-pct').textContent = cpu+'%';
                    const cpuBar = document.getElementById('topbar-cpu-bar');
                    cpuBar.style.width = cpu+'%';
                    cpuBar.style.backgroundColor = resBarColor(cpu);
                    cpuBar.setAttribute('aria-valuenow', cpu);
                    document.getElementById('topbar-cpu-ctr-pct').textContent = ctrCpu+'%';
                    const cpuCtrBar = document.getElementById('topbar-cpu-ctr-bar');
                    cpuCtrBar.style.width = ctrCpu+'%';
                    cpuCtrBar.style.backgroundColor = resBarColor(ctrCpu);
                    cpuCtrBar.setAttribute('aria-valuenow', ctrCpu);
                    document.getElementById('topbar-cpu-cores').textContent = cores + ' cores';
                    document.getElementById('topbar-gpu-pct').textContent = gpu+'%';
                    const gpuBar = document.getElementById('topbar-gpu-bar');
                    gpuBar.style.width = gpu+'%';
                    gpuBar.style.backgroundColor = resBarColor(gpu);
                    gpuBar.setAttribute('aria-valuenow', gpu);
                    document.getElementById('topbar-gpu-wrk-pct').textContent = workerGpu+'%';
                    const gpuWrkBar = document.getElementById('topbar-gpu-wrk-bar');
                    gpuWrkBar.style.width = workerGpu+'%';
                    gpuWrkBar.style.backgroundColor = resBarColor(workerGpu);
                    gpuWrkBar.setAttribute('aria-valuenow', workerGpu);
                    document.getElementById('topbar-gpu-cores').textContent = gpuCores + ' cores';
                    document.getElementById('topbar-vram-pct').textContent = sysVram+'%';
                    const vramBar = document.getElementById('topbar-vram-bar');
                    vramBar.style.width = sysVram+'%';
                    vramBar.style.backgroundColor = resBarColor(sysVram);
                    vramBar.setAttribute('aria-valuenow', sysVram);
                    document.getElementById('topbar-vram-wrk-pct').textContent = vram+'%';
                    const vramWrkBar = document.getElementById('topbar-vram-wrk-bar');
                    vramWrkBar.style.width = vram+'%';
                    vramWrkBar.style.backgroundColor = resBarColor(vram);
                    vramWrkBar.setAttribute('aria-valuenow', vram);
                    document.getElementById('topbar-vram-total').textContent = formatMb(vramTotalMb);
                    document.getElementById('topbar-ram-pct').textContent = ram+'%';
                    const totalRamVal = totalRamMb >= 1024 ? (totalRamMb / 1024).toFixed(1) + ' GB' : Math.round(totalRamMb) + ' MB';
                    document.getElementById('topbar-total-ram-val').textContent = totalRamVal;
                    const ramBar = document.getElementById('topbar-ram-bar');
                    ramBar.style.width = ram+'%';
                    ramBar.style.backgroundColor = resBarColor(ram);
                    ramBar.setAttribute('aria-valuenow', ram);
                    document.getElementById('topbar-sysram-pct').textContent = sysRam+'%';
                    const sysRamBar = document.getElementById('topbar-sysram-bar');
                    sysRamBar.style.width = sysRam+'%';
                    sysRamBar.style.backgroundColor = resBarColor(sysRam);
                    sysRamBar.setAttribute('aria-valuenow', sysRam);

                    function setMobileResChip(chipId, chipLabel, chipValue) {
                        const chip = document.getElementById(chipId);
                        chip.textContent = chipLabel + ' ' + chipValue + '%';
                        chip.style.color = resBarColor(chipValue);
                    }
                    setMobileResChip('mobile-cpu', 'SYS', cpu);
                    setMobileResChip('mobile-cpu-ctr', 'WRK', ctrCpu);
                    setMobileResChip('mobile-gpu', 'SYS', gpu);
                    setMobileResChip('mobile-gpu-wrk', 'WRK', workerGpu);
                    setMobileResChip('mobile-vram', 'WRK', vram);
                    setMobileResChip('mobile-sysvram', 'SYS', sysVram);
                    setMobileResChip('mobile-ram', 'WRK', ram);
                    setMobileResChip('mobile-sysram', 'SYS', sysRam);
                    const ojd = document.getElementById('overview-current-job');
                    if (data.current_job) {
                        const job = data.current_job;
                        const sd = escapeHtml(job.state || 'N/A');
                        const rawPv = (job.progress !== null && job.progress !== undefined) ? job.progress : 0;
                        // Use null for missing ids so that two jobs without ids are never
                        // treated as the same job by the high-water-mark logic below.
                        const jobId = job.id || null;
                        // Never let the progress bar go backwards for the same job id.
                        // Skip the high-water mark when jobId is null (unknown id) so a
                        // missing-id job never pins progress across separate jobs.
                        let pv;
                        if (jobId !== null && jobId === _currentJobId) {
                            pv = Math.max(_currentJobProgress, rawPv);
                        } else {
                            _currentJobId = jobId;
                            pv = rawPv;
                        }
                        _currentJobProgress = pv;
                        // Drive the elapsed-time displays from the server's measured elapsed seconds
                        // (state_elapsed_seconds / job_elapsed_seconds) so they reflect true elapsed
                        // time and are NOT reset when the page is (re)loaded or navigated to.
                        // Re-syncing every poll keeps them accurate; the 1s interval ticks between
                        // polls. Fall back to client-side timing only when the server omits a value.
                        const newState = job.state || null;
                        if (job.state_elapsed_seconds !== null && job.state_elapsed_seconds !== undefined) {
                            _currentJobStateStartTime = Date.now() - Math.round(job.state_elapsed_seconds * 1000);
                        } else if (newState !== _currentJobState || _currentJobStateStartTime === null) {
                            _currentJobStateStartTime = Date.now();
                        }
                        _currentJobState = newState;
                        if (job.job_elapsed_seconds !== null && job.job_elapsed_seconds !== undefined) {
                            _currentJobStartTime = Date.now() - Math.round(job.job_elapsed_seconds * 1000);
                        } else if (_currentJobStartTime === null) {
                            _currentJobStartTime = Date.now();
                        }
                        ojd.classList.remove('centered-empty-container');
                        ojd.innerHTML =
                            '<div class="stat-row"><span class="stat-label">Job ID:</span><span class="stat-value" style="font-family:monospace;font-size:0.8rem;">'+escapeHtml(job.id||'N/A')+'</span></div>'+
                            '<div class="stat-row"><span class="stat-label">Model:</span><span class="stat-value">'+escapeHtml(job.model||'N/A')+'</span></div>'+
                            (job.batch_size!=null&&job.batch_size!==undefined?'<div class="stat-row"><span class="stat-label">Batch Size:</span><span class="stat-value">'+escapeHtml(job.batch_size)+'x</span></div>':'')+
                            (job.steps!=null&&job.steps!==undefined?'<div class="stat-row"><span class="stat-label">Steps:</span><span class="stat-value">'+escapeHtml(job.steps)+'</span></div>':'')+
                            (job.width!=null&&job.width!==undefined&&job.height!=null&&job.height!==undefined?'<div class="stat-row"><span class="stat-label">Image Size:</span><span class="stat-value">'+escapeHtml(job.width)+'x'+escapeHtml(job.height)+'</span></div>':'')+
                            (job.sampler!=null&&job.sampler!==undefined?'<div class="stat-row"><span class="stat-label">Sampler:</span><span class="stat-value">'+escapeHtml(job.sampler)+'</span></div>':'')+
                            '<div class="stat-row"><span class="stat-label">LoRAs:</span><span class="stat-value">'+(job.loras!=null&&job.loras!==undefined&&job.loras.length>0?job.loras.map(l=>escapeHtml(l.name||'Unknown')).join(', '):'None')+'</span></div>'+
                            '<div class="stat-row"><span class="stat-label">State:</span><span class="stat-value"><span class="job-state-badge">'+sd+' <span id="job-state-timer" class="job-state-timer">'+(_currentJobStateStartTime?'('+formatElapsed(_currentJobStateStartTime)+')':'')+'</span></span></span></div>'+
                            '<div style="margin-top:14px;"><div class="progress-header"><span class="progress-label">Progress</span><span class="progress-value">'+escapeHtml(pv)+'%</span></div><div class="progress-bar-container" style="height:12px;"><div class="progress-bar" style="width:'+escapeHtml(pv)+'%;height:100%;border-radius:6px;"></div></div></div>';
                    } else {
                        _currentJobState = null;
                        _currentJobStateStartTime = null;
                        _currentJobStartTime = null;
                        const totalEl = document.getElementById('job-total-timer');
                        if (totalEl) totalEl.textContent = '';
                        ojd.classList.add('centered-empty-container');
                        ojd.innerHTML = '<div class="empty-state"><span class="empty-state-icon">&#9203;</span>No job in progress</div>';
                    }
                    const hasImage = data.last_image_submission_timestamp && data.last_image_submission_timestamp !== 0;
                    _lastImageSubmissionTimestamp = hasImage ? data.last_image_submission_timestamp : null;
                    // Fetch images separately so the status payload stays small.
                    // Images are re-fetched only when the submission timestamp changes.
                    if (data.last_image_submission_timestamp !== _lastFetchedImageTimestamp) {
                        if (hasImage) {
                            const oic = document.getElementById('overview-image-container');
                            if (!oic.querySelector('img')) {
                                oic.innerHTML = '';
                                oic.classList.add('loading');
                            }
                            fetchLastImage(data.last_image_submission_timestamp);
                        } else {
                            var _prevImageTs = _lastFetchedImageTimestamp;
                            _lastFetchedImageTimestamp = data.last_image_submission_timestamp;
                            // Only clear the container when a real session image is gone (prevTs > 0 → 0).
                            // On the initial null → 0 transition leave the container alone so the
                            // initializeUpdates gallery preview can still render.
                            if (_prevImageTs !== null && _prevImageTs !== 0) {
                                _galleryPreviewTimestamp = null;
                                renderLastImages([], document.getElementById('overview-image-container'), 0, null, null);
                            }
                        }
                    }
                    const qd = document.getElementById('job-queue');
                    document.getElementById('queue-count').textContent = data.job_queue.length;
                    document.getElementById('queue-max').textContent = data.max_queue_size;
                    const qmInput = document.getElementById('queue-max-input');
                    const qmSetBtn = document.getElementById('queue-set-btn');
                    const qmAutoBtn = document.getElementById('queue-auto-btn');
                    const queueAuto = !!data.queue_size_auto;
                    if (_settingsPendingQueue === null) {
                        if (qmAutoBtn) { if (queueAuto) { qmAutoBtn.classList.add('active'); qmAutoBtn.setAttribute('aria-pressed', 'true'); } else { qmAutoBtn.classList.remove('active'); qmAutoBtn.setAttribute('aria-pressed', 'false'); } }
                        if (qmInput) { if (queueAuto) { qmInput.disabled = true; } else { qmInput.disabled = false; if (document.activeElement !== qmInput) qmInput.value = data.max_queue_size; } }
                        if (qmSetBtn) qmSetBtn.disabled = queueAuto;
                    }
                    if (data.job_queue.length > 0) {
                        qd.innerHTML = data.job_queue.map(j => { const bi = j.batch_size&&j.batch_size>1?' ('+escapeHtml(j.batch_size)+'x batch)':''; const elapsed = j.popped_at ? formatElapsed(j.popped_at * 1000) : ''; const elapsedHtml = elapsed ? '<span class="job-elapsed">'+escapeHtml(elapsed)+'</span>' : ''; return '<div class="job-item"><span class="job-item-left"><span class="job-id">'+escapeHtml(j.id||'N/A')+'</span>: '+escapeHtml(j.model||'Unknown model')+bi+'</span>'+elapsedHtml+'</div>'; }).join('');
                    } else { qd.innerHTML = '<div class="empty-state">Queue is empty</div>'; }
                    const md = document.getElementById('models-loaded');
                    document.getElementById('models-count').textContent = data.models_loaded.length;
                    document.getElementById('models-max').textContent = data.max_active_models;
                    const mmInput = document.getElementById('models-max-input');
                    const mmSetBtn = document.getElementById('models-set-btn');
                    const mmAutoBtn = document.getElementById('models-auto-btn');
                    const modelsAuto = !!data.max_active_models_auto;
                    if (_settingsPendingModels === null) {
                        if (mmAutoBtn) { if (modelsAuto) { mmAutoBtn.classList.add('active'); mmAutoBtn.setAttribute('aria-pressed', 'true'); } else { mmAutoBtn.classList.remove('active'); mmAutoBtn.setAttribute('aria-pressed', 'false'); } }
                        if (mmInput) { if (modelsAuto) { mmInput.disabled = true; } else { mmInput.disabled = false; if (document.activeElement !== mmInput) mmInput.value = data.max_active_models; } }
                        if (mmSetBtn) mmSetBtn.disabled = modelsAuto;
                    }
                    if (data.models_loaded.length > 0) {
                        md.innerHTML = data.models_loaded.map(m => '<div class="model-badge">'+escapeHtml(m)+'</div>').join('');
                    } else { md.innerHTML = '<span style="color:#94a3b8;font-size:0.83rem;">No models loaded</span>'; }
                    const pd = document.getElementById('processes');
                    document.getElementById('process-count').textContent = data.processes.length;
                    if (data.processes.length > 0) {
                        pd.innerHTML = data.processes.map(proc => {
                            let sl = [];
                            if (proc.job_id) sl.push('Job: '+escapeHtml(proc.job_id));
                            if (proc.model) sl.push('Model: '+escapeHtml(proc.model));
                            if (proc.progress!=null&&proc.progress!==undefined) sl.push('Progress: '+escapeHtml(proc.progress)+'%');
                            return '<div class="process-item"><div class="process-id-row"><span class="process-id">'+escapeHtml(proc.display_id || proc.id)+'</span><span class="process-state-badge">'+escapeHtml(proc.state)+'</span><span class="process-type-badge">'+escapeHtml(proc.type)+'</span></div><div class="process-detail-text">'+(sl.length>0?sl.join(' | '):'Idle')+'</div></div>';
                        }).join('');
                    } else { pd.innerHTML = '<div class="empty-state"><span class="empty-state-icon">&#9881;</span>No process info</div>'; }
                    const newImagesCount = data.images_count || 0;
                    const hasNewImages = lastKnownImagesCount >= 0 && newImagesCount > lastKnownImagesCount;
                    const galleryPageActive = document.getElementById('page-gallery').classList.contains('active');
                    if (hasNewImages && galleryPageActive) {
                        if (galleryCurrentPage === 1) {
                            if (!galleryFetchInProgress) {
                                refreshGalleryPage1();
                                lastKnownImagesCount = newImagesCount;
                            }
                            // If a fetch is already in progress, don't update lastKnownImagesCount so
                            // the next poll can still detect the new images and retry the refresh.
                        } else {
                            const gbn = document.getElementById('gallery-new-banner');
                            if (gbn) gbn.style.display = '';
                            lastKnownImagesCount = newImagesCount;
                        }
                    } else {
                        if (hasNewImages && !galleryPageActive) {
                            // New images arrived while another tab is shown.  Record this so
                            // showPage() can act when the user returns to the gallery tab.
                            galleryHasUnseenImages = true;
                        }
                        lastKnownImagesCount = newImagesCount;
                    }
                    const newErrorsCount = data.errors_count || 0;
                    if (newErrorsCount !== errorsTotal) {
                        if (newErrorsCount === 0) {
                            errorsCurrentPage = 1; errorsTotal = 0; errorsTotalPages = 1; errorsPageData = [];
                            errorsGroupedCurrentPage = 1; errorsGroupedTotalPages = 1; errorsGroupedData = [];
                            if (errorsViewMode === 'grouped') renderErrorsGroupedPage(); else renderErrorsPage();
                        } else {
                            if (errorsViewMode === 'grouped') fetchErrorsGroupedPage(errorsGroupedCurrentPage);
                            else fetchErrorsPage(errorsCurrentPage);
                        }
                    }
                    const cl = document.getElementById('console-logs');
                    if (!consolePaused) {
                        if (data.console_logs && data.console_logs.length > 0) {
                            _consoleLogs = data.console_logs;
                        } else {
                            _consoleLogs = [];
                        }
                        _renderConsoleLogs();
                    }
                    // Update user page
                    const ud = data.user_details || {};
                    document.getElementById('user-page-username').textContent = data.horde_username || '-';
                    document.getElementById('user-page-kudos-total').textContent = data.user_kudos_total != null ? data.user_kudos_total.toLocaleString(undefined, {maximumFractionDigits: 2}) : '-';
                    document.getElementById('user-page-kudos-per-hour').textContent = (data.kudos_per_hour || 0).toLocaleString(undefined, {maximumFractionDigits: 2});
                    const trusted = ud.trusted;
                    const trustIndicator = document.getElementById('user-page-trust-indicator');
                    trustIndicator.textContent = trusted === true ? '\u2714' : (trusted === false ? '\u2718' : '');
                    trustIndicator.className = 'trust-indicator ' + (trusted === true ? 'success' : (trusted === false ? 'error' : ''));
                    trustIndicator.title = trusted === true ? 'Trusted' : (trusted === false ? 'Not trusted' : '');
                    document.getElementById('user-page-worker-count').textContent = ud.worker_count != null ? ud.worker_count : '-';
                    const kb = document.getElementById('user-page-kudos-breakdown');
                    const kd = ud.kudos_details || {};
                    const kdRows = [
                        ['Accumulated', kd.accumulated],
                        ['Gifted', kd.gifted],
                        ['Admin', kd.admin],
                        ['Received', kd.received],
                        ['Donated', kd.donated],
                        ['Recurring', kd.recurring],
                    ].filter(function(r){return r[1] != null;});
                    kb.innerHTML = kdRows.length > 0
                        ? kdRows.map(function(r){return '<div class="stat-row"><span class="stat-label">'+escapeHtml(r[0])+':</span><span class="stat-value">'+Number(r[1]).toLocaleString(undefined,{maximumFractionDigits:2})+'</span></div>';}).join('')
                        : '<div class="empty-state">No kudos breakdown available</div>';
                    // Render per-worker cards
                    if (Array.isArray(ud.workers_list)) {
                        cachedWorkersList = ud.workers_list;
                        try {
                            if (cachedWorkersList.length > 0) {
                                localStorage.setItem('horde-workers-list', JSON.stringify(cachedWorkersList));
                            } else {
                                localStorage.removeItem('horde-workers-list');
                            }
                        } catch(e) {}
                    } else if (ud.worker_count === 0) {
                        cachedWorkersList = [];
                        try { localStorage.removeItem('horde-workers-list'); } catch(e) {}
                    }
                    renderWorkersList();
                })
                .catch(error => {
                    if (error.name === 'AbortError') return;
                    consecutiveErrors++;
                    console.error('Error fetching status:', error);
                    if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS)
                        console.warn('Failed to fetch status '+consecutiveErrors+' times in a row. Check server connection.');
                })
                .finally(() => {
                    statusAbortController = null;
                    // Refresh stats data if the statistics page is currently visible.
                    if (document.getElementById('page-stats').classList.contains('active')) fetchStats();
                    scheduleUpdate();
                });
        }
        const DEFAULT_UPDATE_INTERVAL_MS = 1000;
        const CONFIG_FETCH_TIMEOUT_MS = 5000;
        // ========================================================
        // STATISTICS PAGE
        // ========================================================
        function setStatsWindow(windowSecs, btn) {
            _statsWindowSecs = windowSecs;
            document.querySelectorAll('.stats-window-btn').forEach(function(b) { b.classList.remove('active'); });
            if (btn) btn.classList.add('active');
            // Re-fetch rather than re-rendering the cached response: the server now
            // filters and downsamples by window, so switching to a wider window needs
            // a request for that window's (still small) data, not a client-side re-filter
            // of an already-loaded array.
            fetchStats(true);
        }

        function fetchStats(force) {
            var now = Date.now();
            if (!force && (now - _statsLastFetchTime < _STATS_FETCH_THROTTLE_MS)) {
                // Not yet time for a new fetch; avoid re-rendering unchanged cached charts.
                return;
            }
            if (_statsFetchInProgress) return;
            _statsFetchInProgress = true;
            _statsLastFetchTime = now;
            if (_statsAbortController) _statsAbortController.abort();
            _statsAbortController = new AbortController();
            var ctrl = _statsAbortController;
            var windowParam = (_statsWindowSecs === null) ? 'all' : String(_statsWindowSecs);
            fetch('/api/stats?window=' + windowParam, { signal: ctrl.signal })
                .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
                .then(function(data) {
                    if (ctrl !== _statsAbortController) return;
                    _statsAbortController = null;
                    _statsFetchInProgress = false;
                    _statsData = data;
                    renderStatsPage(data);
                })
                .catch(function(err) {
                    _statsFetchInProgress = false;
                    if (err.name !== 'AbortError') console.error('Failed to fetch /api/stats:', err);
                });
        }

        function _getWindowedSnapshots(snapshots) {
            // The server already filters to the requested window and downsamples the
            // result (see fetchStats' window query param), so this is just a defensive
            // no-op pass-through in case a caller ever renders stale/unfiltered data.
            return snapshots || [];
        }

        function _avgField(arr, key) {
            if (!arr.length) return 0;
            return arr.reduce(function(s, x) { return s + (x[key] || 0); }, 0) / arr.length;
        }

        function renderStatsPage(data) {
            var allSnaps = data.snapshots || [];
            var snaps = _getWindowedSnapshots(allSnaps);
            var noData = allSnaps.length === 0;

            function fmtVal(v, dec) {
                return noData ? '-' : v.toLocaleString(undefined, { maximumFractionDigits: dec || 0 });
            }

            // Summary stats: deltas of cumulative counters within the window.
            // These are only meaningful when the selected window contains at least
            // two snapshots; with a single snapshot, the counters are cumulative and
            // would overstate activity within the window.
            // Jobs Faulted is a windowed delta of the per-snapshot cumulative `jf`
            // counter (recorded alongside jc/jp/ks), so it changes with the selected
            // time-range exactly like Jobs Popped and Images Generated. For the
            // "All time" window (_statsWindowSecs === null) every snapshot is included,
            // so this reduces to the session total.
            var imagesGenerated = 0, kudosEarned = 0, jobsPopped = 0, jobsFaulted = 0;
            if (snaps.length >= 2) {
                imagesGenerated = Math.max(0, snaps[snaps.length - 1].jc - snaps[0].jc);
                kudosEarned     = Math.max(0, snaps[snaps.length - 1].ks - snaps[0].ks);
                jobsPopped      = Math.max(0, snaps[snaps.length - 1].jp - snaps[0].jp);
                jobsFaulted     = Math.max(0, (snaps[snaps.length - 1].jf || 0) - (snaps[0].jf || 0));
            }
            var avgIph = _avgField(snaps, 'iph');
            var avgKph = _avgField(snaps, 'kph');

            var el = function(id) { return document.getElementById(id); };
            // Null-safe setter: a missing element must not throw and abort the whole render
            // (which previously surfaced as "Failed to fetch /api/stats").
            var setText = function(id, val) { var e = el(id); if (e) e.textContent = val; };
            setText('stats-images-generated', noData ? '-' : imagesGenerated.toLocaleString());
            setText('stats-kudos-earned',     noData ? '-' : kudosEarned.toLocaleString(undefined, { maximumFractionDigits: 2 }));
            setText('stats-avg-iph',          fmtVal(avgIph, 2));
            setText('stats-avg-kph',          fmtVal(avgKph, 2));
            setText('stats-jobs-popped',      noData ? '-' : jobsPopped.toLocaleString());
            setText('stats-jobs-faulted',     noData ? '-' : jobsFaulted.toLocaleString());

            // Per-model image count table (session totals, not windowed)
            var modelWrap = document.getElementById('stats-model-table-wrap');
            if (modelWrap) {
                var ipm = data.images_per_model || {};
                var modelEntries = Object.keys(ipm).map(function(k) { return { name: k, count: ipm[k] }; });
                modelEntries.sort(function(a, b) { return b.count - a.count; });
                if (modelEntries.length === 0) {
                    modelWrap.innerHTML = '<div class="text-muted" style="font-size:0.85rem;">No images generated yet.</div>';
                } else {
                    var maxCount = modelEntries[0].count;
                    var rows = modelEntries.map(function(e) {
                        var pct = maxCount > 0 ? Math.round((e.count / maxCount) * 100) : 0;
                        return '<tr>' +
                            '<td>' + escapeHtml(e.name) + '</td>' +
                            '<td class="model-images-bar-cell"><div class="model-images-bar-wrap"><div class="model-images-bar" style="width:' + pct + '%"></div></div></td>' +
                            '<td>' + e.count.toLocaleString() + '</td>' +
                            '</tr>';
                    }).join('');
                    modelWrap.innerHTML = '<table class="model-images-table">' +
                        '<thead><tr><th>Model</th><th class="model-images-bar-cell"></th><th>Images</th></tr></thead>' +
                        '<tbody>' + rows + '</tbody></table>';
                }
            }

            // Per-model failed jobs count table (session totals, not windowed)
            var failedModelWrap = document.getElementById('stats-failed-model-table-wrap');
            if (failedModelWrap) {
                var fjm = data.failed_jobs_per_model || {};
                var failedEntries = Object.keys(fjm).map(function(k) { return { name: k, count: fjm[k] }; });
                failedEntries.sort(function(a, b) { return b.count - a.count; });
                if (failedEntries.length === 0) {
                    failedModelWrap.innerHTML = '<div class="text-muted" style="font-size:0.85rem;">No failed jobs yet.</div>';
                } else {
                    var maxFailed = failedEntries[0].count;
                    var failedRows = failedEntries.map(function(e) {
                        var pct = maxFailed > 0 ? Math.round((e.count / maxFailed) * 100) : 0;
                        return '<tr>' +
                            '<td>' + escapeHtml(e.name) + '</td>' +
                            '<td class="model-images-bar-cell"><div class="model-images-bar-wrap"><div class="model-failed-bar" style="width:' + pct + '%"></div></div></td>' +
                            '<td>' + e.count.toLocaleString() + '</td>' +
                            '</tr>';
                    }).join('');
                    failedModelWrap.innerHTML = '<table class="model-images-table">' +
                        '<thead><tr><th>Model</th><th class="model-images-bar-cell"></th><th>Failed</th></tr></thead>' +
                        '<tbody>' + failedRows + '</tbody></table>';
                }
            }

            // Sort states by their order in the job process; TOTAL is always last.
            // Shared by both the Avg & Max Time and Faults by Phase tables.
            // Mirrors the full HordeProcessState enum in pipeline execution order.
            var stateOrder = [
                'PROCESS_STARTING',
                'WAITING_FOR_JOB',
                'JOB_RECEIVED',
                'DOWNLOADING_MODEL',
                'DOWNLOAD_COMPLETE',
                'DOWNLOADING_AUX_MODEL',
                'DOWNLOAD_AUX_COMPLETE',
                'MODEL_PRELOADING',
                'MODEL_PRELOADED',
                'MODEL_LOADING',
                'MODEL_LOADED',
                'UNLOADED_MODEL_FROM_VRAM',
                'UNLOADED_MODEL_FROM_RAM',
                'INFERENCE_STARTING',
                'INFERENCE_PROCESSING',
                'INFERENCE_FAILED',
                'POST_PROCESSING_STARTING',
                'POST_PROCESSING_COMPLETE',
                'INFERENCE_POST_PROCESSING',
                'INFERENCE_COMPLETE',
                'ALCHEMY_STARTING',
                'ALCHEMY_COMPLETE',
                'ALCHEMY_FAILED',
                'SAFETY_STARTING',
                'SAFETY_EVALUATING',
                'SAFETY_COMPLETE',
                'SAFETY_FAILED',
                'RESULT_SAVING',
                'RESULT_SAVED',
                'RESULT_SUBMITTING',
                'RESULT_SUBMITTED',
                'PROCESS_ENDING',
                'PROCESS_ENDED',
            ];
            function sortByStateOrder(a, b) {
                if (a === 'TOTAL') return 1;
                if (b === 'TOTAL') return -1;
                var ia = stateOrder.indexOf(a);
                var ib = stateOrder.indexOf(b);
                if (ia === -1 && ib === -1) return a.localeCompare(b);
                if (ia === -1) return 1;
                if (ib === -1) return -1;
                return ia - ib;
            }

            // Per-phase fault count table (session totals, not windowed)
            var faultPhaseWrap = document.getElementById('stats-fault-phase-table-wrap');
            if (faultPhaseWrap) {
                var fpp = data.faulted_jobs_per_phase || {};
                var phaseEntries = Object.keys(fpp).map(function(k) { return { name: k, count: fpp[k] }; });
                phaseEntries.sort(function(a, b) { return sortByStateOrder(a.name, b.name); });
                if (phaseEntries.length === 0) {
                    faultPhaseWrap.innerHTML = '<div class="text-muted" style="font-size:0.85rem;">No job faults yet.</div>';
                } else {
                    var maxPhase = Math.max.apply(null, phaseEntries.map(function(e) { return e.count; }));
                    var phaseRows = phaseEntries.map(function(e) {
                        var pct = maxPhase > 0 ? Math.round((e.count / maxPhase) * 100) : 0;
                        return '<tr>' +
                            '<td>' + escapeHtml(e.name) + '</td>' +
                            '<td class="model-images-bar-cell"><div class="model-images-bar-wrap"><div class="model-failed-bar" style="width:' + pct + '%"></div></div></td>' +
                            '<td>' + e.count.toLocaleString() + '</td>' +
                            '</tr>';
                    }).join('');
                    faultPhaseWrap.innerHTML = '<table class="model-images-table">' +
                        '<thead><tr><th>Phase</th><th class="model-images-bar-cell"></th><th>Faults</th></tr></thead>' +
                        '<tbody>' + phaseRows + '</tbody></table>';
                }
            }

            // Avg & Max time per job state table (session totals, not windowed)
            var jobStateTimeWrap = document.getElementById('stats-job-state-time-wrap');
            if (jobStateTimeWrap) {
                var avgTimes = data.avg_time_per_job_state || {};
                var maxTimes = data.max_time_per_job_state || {};
                var stateNames = Object.keys(avgTimes);
                if (stateNames.length === 0) {
                    jobStateTimeWrap.innerHTML = '<div class="text-muted" style="font-size:0.85rem;">No completed jobs yet.</div>';
                } else {
                    var sortedNames = stateNames.slice().sort(sortByStateOrder);
                    var stateRows = sortedNames.map(function(s) {
                        var avg = avgTimes[s] !== undefined ? avgTimes[s].toLocaleString(undefined, { maximumFractionDigits: 2 }) + ' s' : '-';
                        var max = maxTimes[s] !== undefined ? maxTimes[s].toLocaleString(undefined, { maximumFractionDigits: 2 }) + ' s' : '-';
                        return '<tr>' +
                            '<td>' + escapeHtml(s) + '</td>' +
                            '<td style="text-align:right;">' + avg + '</td>' +
                            '<td style="text-align:right;">' + max + '</td>' +
                            '</tr>';
                    }).join('');
                    jobStateTimeWrap.innerHTML = '<table class="model-images-table">' +
                        '<thead><tr><th>State</th><th style="text-align:right;">Avg</th><th style="text-align:right;">Max</th></tr></thead>' +
                        '<tbody>' + stateRows + '</tbody></table>';
                }
            }

            // Avg & Max time per inference step per model table (session totals)
            var stepTimeModelWrap = document.getElementById('stats-step-time-model-wrap');
            if (stepTimeModelWrap) {
                var avgStepTimes = data.avg_time_per_step_per_model || {};
                var maxStepTimes = data.max_time_per_step_per_model || {};
                var stepModelNames = Object.keys(avgStepTimes);
                if (stepModelNames.length === 0) {
                    stepTimeModelWrap.innerHTML = '<div class="text-muted" style="font-size:0.85rem;">No completed jobs yet.</div>';
                } else {
                    stepModelNames.sort(function(a, b) { return (avgStepTimes[a] || 0) - (avgStepTimes[b] || 0); });
                    var stepModelRows = stepModelNames.map(function(m) {
                        var avg = avgStepTimes[m] !== undefined ? avgStepTimes[m].toLocaleString(undefined, { minimumFractionDigits: 3, maximumFractionDigits: 3 }) + ' s' : '-';
                        var max = maxStepTimes[m] !== undefined ? maxStepTimes[m].toLocaleString(undefined, { minimumFractionDigits: 3, maximumFractionDigits: 3 }) + ' s' : '-';
                        return '<tr>' +
                            '<td>' + escapeHtml(m) + '</td>' +
                            '<td style="text-align:right;">' + avg + '</td>' +
                            '<td style="text-align:right;">' + max + '</td>' +
                            '</tr>';
                    }).join('');
                    stepTimeModelWrap.innerHTML = '<table class="model-images-table">' +
                        '<thead><tr><th>Model</th><th style="text-align:right;">Avg/step</th><th style="text-align:right;">Max/step</th></tr></thead>' +
                        '<tbody>' + stepModelRows + '</tbody></table>';
                }
            }

            // Avg & Max total job time per model table (session totals)
            var jobTimeModelWrap = document.getElementById('stats-job-time-model-wrap');
            if (jobTimeModelWrap) {
                var avgJobTimes = data.avg_time_per_job_per_model || {};
                var maxJobTimes = data.max_time_per_job_per_model || {};
                var jobModelNames = Object.keys(avgJobTimes);
                if (jobModelNames.length === 0) {
                    jobTimeModelWrap.innerHTML = '<div class="text-muted" style="font-size:0.85rem;">No completed jobs yet.</div>';
                } else {
                    jobModelNames.sort(function(a, b) { return (avgJobTimes[a] || 0) - (avgJobTimes[b] || 0); });
                    var jobModelRows = jobModelNames.map(function(m) {
                        var avg = avgJobTimes[m] !== undefined ? avgJobTimes[m].toLocaleString(undefined, { maximumFractionDigits: 2 }) + ' s' : '-';
                        var max = maxJobTimes[m] !== undefined ? maxJobTimes[m].toLocaleString(undefined, { maximumFractionDigits: 2 }) + ' s' : '-';
                        return '<tr>' +
                            '<td>' + escapeHtml(m) + '</td>' +
                            '<td style="text-align:right;">' + avg + '</td>' +
                            '<td style="text-align:right;">' + max + '</td>' +
                            '</tr>';
                    }).join('');
                    jobTimeModelWrap.innerHTML = '<table class="model-images-table">' +
                        '<thead><tr><th>Model</th><th style="text-align:right;">Avg</th><th style="text-align:right;">Max</th></tr></thead>' +
                        '<tbody>' + jobModelRows + '</tbody></table>';
                }
            }
            drawDualAxisLineChart('chart-iph-kph',
                { points: snaps.map(function(s) { return { t: s.t, v: s.iph }; }), color: '#10b981' },
                { points: snaps.map(function(s) { return { t: s.t, v: s.kph }; }), color: '#6366f1' });
            drawMultiLineChart('chart-cpu', [
                { points: snaps.map(function(s) { return { t: s.t, v: s.container_cpu || 0 }; }), color: '#fb923c' },
                { points: snaps.map(function(s) { return { t: s.t, v: s.cpu  }; }), color: '#f59e0b' },
            ], { yMax: 100, yFmt: function(v) { return Math.round(v) + '%'; } });
            drawMultiLineChart('chart-gpu', [
                { points: snaps.map(function(s) { return { t: s.t, v: s.worker_gpu || 0 }; }), color: '#60a5fa' },
                { points: snaps.map(function(s) { return { t: s.t, v: s.gpu  || 0 }; }), color: '#3b82f6' },
            ], { yMax: 100, yFmt: function(v) { return Math.round(v) + '%'; } });
            drawMultiLineChart('chart-ram', [
                { points: snaps.map(function(s) { return { t: s.t, v: s.ram  || 0 }; }), color: '#10b981' },
                { points: snaps.map(function(s) { return { t: s.t, v: s.system_ram || 0 }; }), color: '#059669' },
            ], { yMax: 100, yFmt: function(v) { return Math.round(v) + '%'; } });
            drawMultiLineChart('chart-vram', [
                { points: snaps.map(function(s) { return { t: s.t, v: s.vram || 0 }; }), color: '#8b5cf6' },
                { points: snaps.map(function(s) { return { t: s.t, v: s.system_vram || 0 }; }), color: '#a78bfa' },
            ], { yMax: 100, yFmt: function(v) { return Math.round(v) + '%'; } });
        }

        function drawLineChart(canvasId, points, opts) {
            var canvas = document.getElementById(canvasId);
            if (!canvas) return;
            var parent = canvas.parentElement;
            var cssW = (parent ? parent.offsetWidth : 0) || 400;
            var cssH = (parent ? parent.offsetHeight : 0) || 150;
            var dpr = window.devicePixelRatio || 1;
            var pxW = Math.round(cssW * dpr);
            var pxH = Math.round(cssH * dpr);
            if (canvas.width !== pxW || canvas.height !== pxH) {
                canvas.width  = pxW;
                canvas.height = pxH;
                canvas.style.width  = cssW + 'px';
                canvas.style.height = cssH + 'px';
            }
            var ctx = canvas.getContext('2d');
            ctx.save();
            ctx.scale(dpr, dpr);
            var w = cssW, h = cssH;
            var pad = { top: 12, right: 14, bottom: 32, left: 46 };
            var chartW = w - pad.left - pad.right;
            var chartH = h - pad.top - pad.bottom;
            var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            var gridColor = isDark ? '#2d3f55' : '#e2e8f0';
            var textColor = isDark ? '#94a3b8' : '#64748b';
            var bgColor   = isDark ? '#1e293b' : '#ffffff';
            var lineColor = opts.color || '#6366f1';
            // Parse hex color for fill rgba
            var r = 99, g = 102, b = 241;
            var cm = lineColor.match(/^#([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
            if (cm) { r = parseInt(cm[1], 16); g = parseInt(cm[2], 16); b = parseInt(cm[3], 16); }
            var fillAlpha = isDark ? '0.14' : '0.08';

            // Background
            ctx.fillStyle = bgColor;
            ctx.fillRect(0, 0, w, h);

            if (!points || points.length === 0) {
                ctx.fillStyle = textColor;
                ctx.font = '12px -apple-system, system-ui, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('No data yet \u2014 data is collected every few seconds', w / 2, h / 2);
                ctx.restore();
                return;
            }

            // Y range
            var yMin = opts.yMin !== undefined ? opts.yMin : 0;
            var rawMax = points.reduce(function(m, p) { return Math.max(m, p.v); }, 0);
            var yMax = opts.yMax !== undefined ? opts.yMax : Math.max(rawMax * 1.15, yMin + 1);
            if (rawMax === 0 && opts.yMax === undefined) yMax = Math.max(yMax, 10);
            var yRange = yMax - yMin || 1;

            // X range
            var tMin = points[0].t, tMax = points[points.length - 1].t;
            var tRange = tMax - tMin || 1;

            function cxf(t) { return pad.left + ((t - tMin) / tRange) * chartW; }
            function cyf(v) { return pad.top + (1 - (v - yMin) / yRange) * chartH; }

            // Horizontal grid lines + Y labels (5 levels: 0%, 25%, 50%, 75%, 100%)
            var levels = 4;
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.font = '10px -apple-system, system-ui, sans-serif';
            ctx.fillStyle = textColor;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (var i = 0; i <= levels; i++) {
                var frac = i / levels;
                var yPx = pad.top + frac * chartH;
                ctx.beginPath(); ctx.moveTo(pad.left, yPx); ctx.lineTo(pad.left + chartW, yPx); ctx.stroke();
                var val = yMin + (1 - frac) * yRange;
                var lbl = opts.yFmt ? opts.yFmt(val) : (val >= 1000 ? (val / 1000).toFixed(1) + 'k' : val % 1 === 0 ? Math.round(val) : val.toFixed(1));
                ctx.fillText(lbl, pad.left - 5, yPx);
            }
            ctx.setLineDash([]);

            // X axis time labels (up to 5)
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            var numXLabels = Math.min(4, points.length - 1);
            if (numXLabels < 1) numXLabels = 1;
            for (var j = 0; j <= numXLabels; j++) {
                var tVal = tMin + (j / numXLabels) * tRange;
                var xPx = cxf(tVal);
                var d = new Date(tVal * 1000);
                var hh = ('0' + d.getHours()).slice(-2), mm = ('0' + d.getMinutes()).slice(-2);
                var xLbl = hh + ':' + mm;
                ctx.fillText(xLbl, xPx, pad.top + chartH + 5);
            }

            // Clip to chart area
            ctx.save();
            ctx.beginPath();
            ctx.rect(pad.left, pad.top, chartW, chartH);
            ctx.clip();

            // Fill area under the line
            ctx.beginPath();
            ctx.moveTo(cxf(points[0].t), cyf(yMin));
            for (var k = 0; k < points.length; k++) ctx.lineTo(cxf(points[k].t), cyf(points[k].v));
            ctx.lineTo(cxf(points[points.length - 1].t), cyf(yMin));
            ctx.closePath();
            ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + fillAlpha + ')';
            ctx.fill();

            // Line
            ctx.beginPath();
            for (var n = 0; n < points.length; n++) {
                var xc = cxf(points[n].t), yc = cyf(points[n].v);
                if (n === 0) ctx.moveTo(xc, yc); else ctx.lineTo(xc, yc);
            }
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            ctx.stroke();

            ctx.restore(); // end clip

            // Chart border
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;
            ctx.strokeRect(pad.left, pad.top, chartW, chartH);

            ctx.restore(); // end dpr scale
        }

        function drawMultiLineChart(canvasId, seriesArray, opts) {
            var canvas = document.getElementById(canvasId);
            if (!canvas) return;
            if (!seriesArray || seriesArray.length === 0) return;
            var parent = canvas.parentElement;
            var cssW = (parent ? parent.offsetWidth : 0) || 400;
            var cssH = (parent ? parent.offsetHeight : 0) || 160;
            var dpr = window.devicePixelRatio || 1;
            var pxW = Math.round(cssW * dpr);
            var pxH = Math.round(cssH * dpr);
            if (canvas.width !== pxW || canvas.height !== pxH) {
                canvas.width  = pxW;
                canvas.height = pxH;
                canvas.style.width  = cssW + 'px';
                canvas.style.height = cssH + 'px';
            }
            var ctx = canvas.getContext('2d');
            ctx.save();
            ctx.scale(dpr, dpr);
            var w = cssW, h = cssH;
            var pad = { top: 12, right: 14, bottom: 32, left: 46 };
            var chartW = w - pad.left - pad.right;
            var chartH = h - pad.top - pad.bottom;
            var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            var gridColor = isDark ? '#2d3f55' : '#e2e8f0';
            var textColor = isDark ? '#94a3b8' : '#64748b';
            var bgColor   = isDark ? '#1e293b' : '#ffffff';

            // Background
            ctx.fillStyle = bgColor;
            ctx.fillRect(0, 0, w, h);

            // Gather all points for x/y range
            var allPoints = [];
            for (var si = 0; si < seriesArray.length; si++) {
                var pts = seriesArray[si].points || [];
                for (var pi = 0; pi < pts.length; pi++) allPoints.push(pts[pi]);
            }

            if (allPoints.length === 0) {
                ctx.fillStyle = textColor;
                ctx.font = '12px -apple-system, system-ui, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('No data yet \u2014 data is collected every few seconds', w / 2, h / 2);
                ctx.restore();
                return;
            }

            // Y range
            var yMin = opts.yMin !== undefined ? opts.yMin : 0;
            var rawMax = allPoints.reduce(function(m, p) { return Math.max(m, p.v); }, 0);
            var yMax = opts.yMax !== undefined ? opts.yMax : Math.max(rawMax * 1.15, yMin + 1);
            if (rawMax === 0 && opts.yMax === undefined) yMax = Math.max(yMax, 10);
            var yRange = yMax - yMin || 1;

            // X range (use the first series' time axis as the reference)
            var refPts = seriesArray[0].points || [];
            var tMin = refPts.length ? refPts[0].t : allPoints[0].t;
            var tMax = refPts.length ? refPts[refPts.length - 1].t : allPoints[allPoints.length - 1].t;
            var tRange = tMax - tMin || 1;

            function cxf(t) { return pad.left + ((t - tMin) / tRange) * chartW; }
            function cyf(v) { return pad.top + (1 - (v - yMin) / yRange) * chartH; }

            // Horizontal grid lines + Y labels
            var levels = 4;
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.font = '10px -apple-system, system-ui, sans-serif';
            ctx.fillStyle = textColor;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (var i = 0; i <= levels; i++) {
                var frac = i / levels;
                var yPx = pad.top + frac * chartH;
                ctx.beginPath(); ctx.moveTo(pad.left, yPx); ctx.lineTo(pad.left + chartW, yPx); ctx.stroke();
                var val = yMin + (1 - frac) * yRange;
                var lbl = opts.yFmt ? opts.yFmt(val) : (val >= 1000 ? (val / 1000).toFixed(1) + 'k' : val % 1 === 0 ? Math.round(val) : val.toFixed(1));
                ctx.fillText(lbl, pad.left - 5, yPx);
            }
            ctx.setLineDash([]);

            // X axis time labels
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            var numXLabels = Math.min(4, refPts.length - 1);
            if (numXLabels < 1) numXLabels = 1;
            for (var j = 0; j <= numXLabels; j++) {
                var tVal = tMin + (j / numXLabels) * tRange;
                var xPx = cxf(tVal);
                var d = new Date(tVal * 1000);
                var hh = ('0' + d.getHours()).slice(-2), mm = ('0' + d.getMinutes()).slice(-2);
                ctx.fillText(hh + ':' + mm, xPx, pad.top + chartH + 5);
            }

            // Clip to chart area
            ctx.save();
            ctx.beginPath();
            ctx.rect(pad.left, pad.top, chartW, chartH);
            ctx.clip();

            // Draw each series (fills first, then lines on top)
            for (var fi = 0; fi < seriesArray.length; fi++) {
                var series = seriesArray[fi];
                var spts = series.points || [];
                if (spts.length === 0) continue;
                var lineColor = series.color || '#6366f1';
                var cm = lineColor.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
                var r = 99, g = 102, b = 241;
                if (cm) { r = parseInt(cm[1], 16); g = parseInt(cm[2], 16); b = parseInt(cm[3], 16); }
                var fillAlpha = isDark ? '0.10' : '0.06';
                // Fill area
                ctx.beginPath();
                ctx.moveTo(cxf(spts[0].t), cyf(yMin));
                for (var k = 0; k < spts.length; k++) ctx.lineTo(cxf(spts[k].t), cyf(spts[k].v));
                ctx.lineTo(cxf(spts[spts.length - 1].t), cyf(yMin));
                ctx.closePath();
                ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + fillAlpha + ')';
                ctx.fill();
            }
            for (var li = 0; li < seriesArray.length; li++) {
                var lseries = seriesArray[li];
                var lpts = lseries.points || [];
                if (lpts.length === 0) continue;
                ctx.beginPath();
                for (var n = 0; n < lpts.length; n++) {
                    var xc = cxf(lpts[n].t), yc = cyf(lpts[n].v);
                    if (n === 0) ctx.moveTo(xc, yc); else ctx.lineTo(xc, yc);
                }
                ctx.strokeStyle = lseries.color || '#6366f1';
                ctx.lineWidth = 2;
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                ctx.stroke();
            }

            ctx.restore(); // end clip

            // Chart border
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;
            ctx.strokeRect(pad.left, pad.top, chartW, chartH);

            ctx.restore(); // end dpr scale
        }

        function drawDualAxisLineChart(canvasId, leftSeries, rightSeries, opts) {
            // leftSeries:  { points: [{t, v}], color }  — left Y axis
            // rightSeries: { points: [{t, v}], color }  — right Y axis
            var canvas = document.getElementById(canvasId);
            if (!canvas) return;
            opts = opts || {};
            var parent = canvas.parentElement;
            var cssW = (parent ? parent.offsetWidth : 0) || 400;
            var cssH = (parent ? parent.offsetHeight : 0) || 150;
            var dpr = window.devicePixelRatio || 1;
            var pxW = Math.round(cssW * dpr);
            var pxH = Math.round(cssH * dpr);
            if (canvas.width !== pxW || canvas.height !== pxH) {
                canvas.width  = pxW;
                canvas.height = pxH;
                canvas.style.width  = cssW + 'px';
                canvas.style.height = cssH + 'px';
            }
            var ctx = canvas.getContext('2d');
            ctx.save();
            ctx.scale(dpr, dpr);
            var w = cssW, h = cssH;
            // Extra right padding to accommodate the right-axis labels.
            var pad = { top: 12, right: 46, bottom: 32, left: 46 };
            var chartW = w - pad.left - pad.right;
            var chartH = h - pad.top - pad.bottom;
            var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            var gridColor = isDark ? '#2d3f55' : '#e2e8f0';
            var textColor = isDark ? '#94a3b8' : '#64748b';
            var bgColor   = isDark ? '#1e293b' : '#ffffff';

            // Background
            ctx.fillStyle = bgColor;
            ctx.fillRect(0, 0, w, h);

            var lpts = (leftSeries  && leftSeries.points)  || [];
            var rpts = (rightSeries && rightSeries.points) || [];
            var allEmpty = lpts.length === 0 && rpts.length === 0;

            if (allEmpty) {
                ctx.fillStyle = textColor;
                ctx.font = '12px -apple-system, system-ui, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('No data yet \u2014 data is collected every few seconds', w / 2, h / 2);
                ctx.restore();
                return;
            }

            // Shared X range: span covering all data points from both series.
            var tMin = Infinity, tMax = -Infinity;
            for (var ti = 0; ti < lpts.length; ti++) { tMin = Math.min(tMin, lpts[ti].t); tMax = Math.max(tMax, lpts[ti].t); }
            for (var ti2 = 0; ti2 < rpts.length; ti2++) { tMin = Math.min(tMin, rpts[ti2].t); tMax = Math.max(tMax, rpts[ti2].t); }
            var tRange = tMax - tMin || 1;

            function cxf(t) { return pad.left + ((t - tMin) / tRange) * chartW; }

            // Left Y range
            var lRawMax = lpts.reduce(function(m, p) { return Math.max(m, p.v); }, 0);
            var lYMin = 0, lYMax = Math.max(lRawMax * 1.15, 1);
            if (lRawMax === 0) lYMax = 10;
            var lYRange = lYMax - lYMin || 1;
            function cyLeft(v) { return pad.top + (1 - (v - lYMin) / lYRange) * chartH; }

            // Right Y range
            var rRawMax = rpts.reduce(function(m, p) { return Math.max(m, p.v); }, 0);
            var rYMin = 0, rYMax = Math.max(rRawMax * 1.15, 1);
            if (rRawMax === 0) rYMax = 10;
            var rYRange = rYMax - rYMin || 1;
            function cyRight(v) { return pad.top + (1 - (v - rYMin) / rYRange) * chartH; }

            // Horizontal grid lines (shared, based on left axis fractions)
            var levels = 4;
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.font = '10px -apple-system, system-ui, sans-serif';
            ctx.textBaseline = 'middle';

            for (var i = 0; i <= levels; i++) {
                var frac = i / levels;
                var yPx = pad.top + frac * chartH;
                ctx.beginPath(); ctx.moveTo(pad.left, yPx); ctx.lineTo(pad.left + chartW, yPx); ctx.stroke();

                // Left axis label (images)
                var lVal = lYMin + (1 - frac) * lYRange;
                var lLbl = lVal >= 1000 ? (lVal / 1000).toFixed(1) + 'k' : lVal % 1 === 0 ? Math.round(lVal) : lVal.toFixed(1);
                ctx.fillStyle = leftSeries.color || '#10b981';
                ctx.textAlign = 'right';
                ctx.fillText(lLbl, pad.left - 5, yPx);

                // Right axis label (kudos)
                var rVal = rYMin + (1 - frac) * rYRange;
                var rLbl = rVal >= 1000 ? (rVal / 1000).toFixed(1) + 'k' : rVal % 1 === 0 ? Math.round(rVal) : rVal.toFixed(1);
                ctx.fillStyle = (rightSeries && rightSeries.color) || '#6366f1';
                ctx.textAlign = 'left';
                ctx.fillText(rLbl, pad.left + chartW + 5, yPx);
            }
            ctx.setLineDash([]);

            // X axis time labels
            ctx.fillStyle = textColor;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            var totalPts = lpts.length + rpts.length;
            var numXLabels = Math.min(4, totalPts - 1);
            if (numXLabels < 1) numXLabels = 1;
            for (var j = 0; j <= numXLabels; j++) {
                var tVal = tMin + (j / numXLabels) * tRange;
                var xPx = cxf(tVal);
                var d = new Date(tVal * 1000);
                var hh = ('0' + d.getHours()).slice(-2), mm = ('0' + d.getMinutes()).slice(-2);
                ctx.fillText(hh + ':' + mm, xPx, pad.top + chartH + 5);
            }

            // Clip to chart area
            ctx.save();
            ctx.beginPath();
            ctx.rect(pad.left, pad.top, chartW, chartH);
            ctx.clip();

            // Draw fills then lines for each series
            var seriesDefs = [
                { pts: lpts, color: leftSeries.color || '#10b981', cyFn: cyLeft },
                { pts: rpts, color: (rightSeries && rightSeries.color) || '#6366f1', cyFn: cyRight },
            ];
            var fillAlpha = isDark ? '0.10' : '0.06';
            for (var fi = 0; fi < seriesDefs.length; fi++) {
                var sd = seriesDefs[fi];
                if (!sd.pts.length) continue;
                var cm = sd.color.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
                var r = 99, g = 102, b = 241; // fallback: #6366f1 (indigo)
                if (cm) { r = parseInt(cm[1], 16); g = parseInt(cm[2], 16); b = parseInt(cm[3], 16); }
                var baseY = sd.cyFn(0);
                ctx.beginPath();
                ctx.moveTo(cxf(sd.pts[0].t), baseY);
                for (var k = 0; k < sd.pts.length; k++) ctx.lineTo(cxf(sd.pts[k].t), sd.cyFn(sd.pts[k].v));
                ctx.lineTo(cxf(sd.pts[sd.pts.length - 1].t), baseY);
                ctx.closePath();
                ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + fillAlpha + ')';
                ctx.fill();
            }
            for (var li = 0; li < seriesDefs.length; li++) {
                var ls = seriesDefs[li];
                if (!ls.pts.length) continue;
                ctx.beginPath();
                for (var n = 0; n < ls.pts.length; n++) {
                    var xc = cxf(ls.pts[n].t), yc = ls.cyFn(ls.pts[n].v);
                    if (n === 0) ctx.moveTo(xc, yc); else ctx.lineTo(xc, yc);
                }
                ctx.strokeStyle = ls.color;
                ctx.lineWidth = 2;
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                ctx.stroke();
            }

            ctx.restore(); // end clip

            // Chart border
            ctx.strokeStyle = gridColor;
            ctx.lineWidth = 1;
            ctx.strokeRect(pad.left, pad.top, chartW, chartH);

            ctx.restore(); // end dpr scale
        }

        // Redraw charts on window resize when stats page is active
        window.addEventListener('resize', function() {
            if (_statsData && document.getElementById('page-stats').classList.contains('active')) {
                renderStatsPage(_statsData);
            }
        });

        // ========================================================
        // HORDE NETWORK PAGE
        // ========================================================
        var _hordeSnapshots = [];
        var _hordeWindowSecs = 1800;
        var _hordeFetchTimer = null;
        var _hordeAbortCtrl = null;
        var _hordeModesAbortCtrl = null;
        var _hordeModes = null;
        var _hordeLastModeFetch = 0;
        var _HORDE_POLL_MS = 30000;
        var _HORDE_MODE_REFRESH_MS = 300000;

        function _fetchHordeHistory(windowSecs) {
            // The bigger the requested window, the more the server downsamples -- this
            // never ships/renders more than roughly a few hundred points, regardless of
            // how much raw history the configured retention period actually holds.
            // Fetches our own /api/horde-snapshots (server-polled every 30s independent of
            // any browser being open) rather than aihorde.net directly: a direct
            // cross-origin browser request needs a CORS preflight for the custom
            // Client-Agent header, which aihorde.net does not reliably grant for
            // arbitrary worker origins.
            if (_hordeAbortCtrl) _hordeAbortCtrl.abort();
            _hordeAbortCtrl = new AbortController();
            var ctrl = _hordeAbortCtrl;
            var windowParam = (windowSecs === null) ? 'all' : String(windowSecs);
            return fetch('/api/horde-snapshots?window=' + windowParam, { signal: ctrl.signal })
                .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
                .then(function(data) {
                    if (ctrl !== _hordeAbortCtrl) return;
                    _hordeAbortCtrl = null;
                    _hordeSnapshots = data.snapshots || [];
                    var errEl = document.getElementById('horde-fetch-error');
                    if (errEl) errEl.style.display = 'none';
                    renderHordePage();
                })
                .catch(function(err) {
                    if (err.name === 'AbortError') return;
                    console.error('Failed to fetch Horde network data:', err);
                    var errEl = document.getElementById('horde-fetch-error');
                    if (errEl) errEl.style.display = '';
                });
        }

        function startHordeFetching() {
            // Truthy check, not `!== null`: this is called at top-level script execution
            // (see the unconditional call further up the file), before the
            // "var _hordeFetchTimer = null" declaration further down runs -- at that point
            // it's hoisted-but-undefined, and `undefined !== null` is true, which used to
            // make this bail out immediately on the very first call. `undefined` and
            // `null` are both falsy, so this check behaves the same for every real state
            // (unset/stopped vs. a live interval id) while no longer misfiring on that
            // first hoisted-undefined call.
            if (_hordeFetchTimer) return; // already running
            // _hordeSnapshots may already be pre-seeded (coarsely downsampled) via inline
            // injection in the HTML. Fetch the currently-selected window's history from
            // the server to fill in proper resolution for it and pick up any gap since
            // the page was rendered.
            // Note: this function is invoked at top-level script execution (before the
            // "var _hordeWindowSecs = 1800" declaration further down the file runs), so
            // `_hordeWindowSecs` is still hoisted-but-undefined here -- fall back to its
            // eventual default explicitly rather than silently fetching an unwindowed
            // (and therefore coarser) "all" view on first load.
            _fetchHordeHistory(_hordeWindowSecs === undefined ? 1800 : _hordeWindowSecs);
            fetchHordeModes();
            // `|| 30000`: same hoisting hazard as above -- `_HORDE_POLL_MS` is still
            // hoisted-but-undefined at this point, and setInterval coerces an
            // undefined/NaN delay to 0, which previously made this fire on every
            // JS tick (a few ms apart) instead of every 30s, hammering this endpoint
            // (and, before it was proxied through our own server, aihorde.net itself).
            _hordeFetchTimer = setInterval(function() {
                _fetchHordeHistory(_hordeWindowSecs);
                var now = Date.now();
                if (now - _hordeLastModeFetch >= _HORDE_MODE_REFRESH_MS) {
                    fetchHordeModes();
                }
            }, _HORDE_POLL_MS || 30000);
        }

        function stopHordeFetching() {
            if (_hordeFetchTimer) {
                clearInterval(_hordeFetchTimer);
                _hordeFetchTimer = null;
            }
            if (_hordeAbortCtrl) { _hordeAbortCtrl.abort(); _hordeAbortCtrl = null; }
            if (_hordeModesAbortCtrl) { _hordeModesAbortCtrl.abort(); _hordeModesAbortCtrl = null; }
        }

        function fetchHordeModes() {
            if (_hordeModesAbortCtrl) _hordeModesAbortCtrl.abort();
            _hordeModesAbortCtrl = new AbortController();
            var ctrl = _hordeModesAbortCtrl;
            _hordeLastModeFetch = Date.now();
            // Proxied through our own origin for the same CORS reason as
            // /api/horde-snapshots above -- see _fetchHordeHistory.
            fetch('/api/horde-modes', { signal: ctrl.signal })
                .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
                .then(function(data) {
                    if (ctrl !== _hordeModesAbortCtrl) return;
                    _hordeModesAbortCtrl = null;
                    _hordeModes = data;
                    renderHordeModeBadges();
                })
                .catch(function(err) {
                    if (err.name === 'AbortError') return;
                    console.error('Failed to fetch Horde modes:', err);
                });
        }

        function renderHordeModeBadges() {
            if (!_hordeModes) return;
            var maintEl = document.getElementById('horde-mode-maintenance');
            var inviteEl = document.getElementById('horde-mode-invite');
            if (maintEl) {
                if (_hordeModes.maintenance_mode) {
                    maintEl.style.display = '';
                    maintEl.className = 'horde-mode-badge warning';
                    maintEl.textContent = '⚠ Maintenance';
                } else {
                    maintEl.style.display = 'none';
                }
            }
            if (inviteEl) {
                if (_hordeModes.invite_only_mode) {
                    inviteEl.style.display = '';
                    inviteEl.className = 'horde-mode-badge warning';
                    inviteEl.textContent = '🔒 Invite Only';
                } else {
                    inviteEl.style.display = 'none';
                }
            }
        }

        function _getHordeWindowedSnapshots() {
            if (!_hordeSnapshots || _hordeSnapshots.length === 0) return [];
            if (_hordeWindowSecs === null) return _hordeSnapshots;
            var cutoff = _hordeSnapshots[_hordeSnapshots.length - 1].t - _hordeWindowSecs;
            return _hordeSnapshots.filter(function(s) { return s.t >= cutoff; });
        }

        function setHordeWindow(secs, btn) {
            _hordeWindowSecs = secs;
            document.querySelectorAll('.horde-window-btn').forEach(function(b) { b.classList.remove('active'); });
            if (btn) btn.classList.add('active');
            // Re-fetch for the newly selected window rather than just re-filtering
            // whatever happens to already be loaded client-side -- see _fetchHordeHistory.
            _fetchHordeHistory(secs);
        }

        function renderHordePage() {
            var snaps = _getHordeWindowedSnapshots();
            var last = snaps.length > 0 ? snaps[snaps.length - 1] : null;

            function fmt(v, dec) {
                if (v === null || v === undefined) return '-';
                return v.toLocaleString(undefined, { maximumFractionDigits: dec !== undefined ? dec : 0 });
            }

            var el = function(id) { return document.getElementById(id); };
            if (el('horde-stat-workers'))    el('horde-stat-workers').textContent    = last ? fmt(last.workers)      : '-';
            if (el('horde-stat-threads'))    el('horde-stat-threads').textContent    = last ? fmt(last.threads)      : '-';
            if (el('horde-stat-queued-req')) el('horde-stat-queued-req').textContent = last ? fmt(last.queued_req)   : '-';
            if (el('horde-stat-queued-mps')) el('horde-stat-queued-mps').textContent = last ? fmt(last.queued_mps, 2) : '-';
            if (el('horde-stat-past-min-mps')) el('horde-stat-past-min-mps').textContent = last ? fmt(last.past_min_mps, 2) : '-';

            drawMultiLineChart('horde-chart-workers-threads', [
                { points: snaps.map(function(s) { return { t: s.t, v: s.workers }; }), color: '#6366f1' },
                { points: snaps.map(function(s) { return { t: s.t, v: s.threads }; }), color: '#a78bfa' }
            ], {});
            drawLineChart('horde-chart-queued-req',
                snaps.map(function(s) { return { t: s.t, v: s.queued_req }; }),
                { color: '#f59e0b' });
            drawLineChart('horde-chart-queued-mps',
                snaps.map(function(s) { return { t: s.t, v: s.queued_mps }; }),
                { color: '#10b981' });
            drawLineChart('horde-chart-past-min-mps',
                snaps.map(function(s) { return { t: s.t, v: s.past_min_mps }; }),
                { color: '#3b82f6' });
        }

        // Redraw Horde charts on window resize when horde page is active
        window.addEventListener('resize', function() {
            if (_hordeSnapshots.length > 0 && document.getElementById('page-horde') && document.getElementById('page-horde').classList.contains('active')) {
                renderHordePage();
            }
        });

        // ========================================================
        // SETTINGS PAGE
        // ========================================================
        const _SETTINGS_SPEC = {
            // key: [label, description, category, type ('bool'|'int'|'float'|'int_auto'|'str_readonly'|'url_readonly'), min, max, restartRequired, autoPrefix?, envVar]
            nsfw:                     ['NSFW',                    'Accept NSFW image jobs.',                                              'Capabilities', 'bool',  null, null, false, null, 'AIWORKER_NSFW'],
            censor_nsfw:              ['Censor NSFW',             'Censor NSFW content even when accepting NSFW jobs.',                  'Capabilities', 'bool',  null, null, false, null, 'AIWORKER_CENSOR_NSFW'],
            allow_img2img:            ['Allow img2img',           'Accept image-to-image jobs.',                                         'Capabilities', 'bool',  null, null, false, null, 'AIWORKER_ALLOW_IMG2IMG'],
            allow_inpainting:         ['Allow Painting',          'Accept inpainting / painting jobs.',                                  'Capabilities', 'bool',  null, null, false, null, 'AIWORKER_ALLOW_PAINTING'],
            allow_unsafe_ip:          ['Allow Unsafe IPs',        'Accept requests from flagged or unsafe IP addresses.',                'Capabilities', 'bool',  null, null, false, null, 'AIWORKER_ALLOW_UNSAFE_IP'],
            allow_post_processing:    ['Allow Post-Processing',   'Accept jobs that include post-processing steps.',                     'Capabilities', 'bool',  null, null, false, null, 'AIWORKER_ALLOW_POST_PROCESSING'],
            allow_controlnet:         ['Allow ControlNet',        'Accept ControlNet jobs.',                                             'Capabilities', 'bool',  null, null, false, null, 'AIWORKER_ALLOW_CONTROLNET'],
            allow_sdxl_controlnet:    ['Allow SDXL ControlNet',   'Accept SDXL ControlNet jobs.',                                        'Capabilities', 'bool',  null, null, false, null, 'AIWORKER_ALLOW_SDXL_CONTROLNET'],
            allow_lora:               ['Allow LoRA',              'Accept jobs that use LoRA models.',                                   'Capabilities', 'bool',  null, null, false, null, 'AIWORKER_ALLOW_LORA'],
            require_upfront_kudos:    ['Require Upfront Kudos',   'Only accept jobs from users who have enough kudos upfront.',          'Capabilities', 'bool',  null, null, false, null, 'AIWORKER_REQUIRE_UPFRONT_KUDOS'],
            limit_max_steps:          ['Limit Max Steps',         'Cap the number of inference steps to the worker\'s configured max.',  'Capabilities', 'bool',  null, null, false, null, 'AIWORKER_LIMIT_MAX_STEPS'],
            extra_slow_worker:        ['Extra Slow Worker',       'Enable extra-slow-worker mode (forces conservative settings).',       'Capabilities', 'bool',  null, null, true, null, 'AIWORKER_EXTRA_SLOW_WORKER'],
            job_queue_size:           ['Job Queue Size',          'Maximum jobs held in the queue at once (0\u00a0=\u00a0unlimited).',  'Performance',  'int_auto', 0, 999, false, 'queue', 'AIWORKER_QUEUE_SIZE'],
            active_model_count:       ['Max Active Models',       'Maximum number of model slots kept active simultaneously.',           'Performance',  'int_auto', 1, 999, false, 'models', 'AIWORKER_MAX_ACTIVE_MODELS'],
            max_power:                ['Max Power',               'Maximum resolution multiplier (formula: 64\u00d764\u00d78\u00d7max_power pixels).',         'Performance',  'int',   1,    128,  false, null, 'AIWORKER_MAX_POWER'],
            max_batch:                ['Max Batch',               'Maximum number of images per batched inference job.',                 'Performance',  'int',   1,    100,  false, null, 'AIWORKER_MAX_BATCH'],
            max_threads:              ['Max Threads',             'Maximum number of concurrent inference threads.',                     'Performance',  'int',   1,    8,    true, null, 'AIWORKER_MAX_THREADS'],
            safety_on_gpu:            ['Safety on GPU',           'Run the safety/CLIP model on GPU instead of CPU (~1.2 GB VRAM).',    'Performance',  'bool',  null, null, true, null, 'AIWORKER_SAFETY_ON_GPU'],
            high_memory_mode:         ['High Memory Mode',        'Keep models in VRAM to reduce load times.',                          'Performance',  'bool',  null, null, true, null, 'AIWORKER_HIGH_MEMORY_MODE'],
            very_high_memory_mode:    ['Very High Memory Mode',   'Aggressive VRAM retention (data-center GPUs only).',                 'Performance',  'bool',  null, null, true, null, 'AIWORKER_VERY_HIGH_MEMORY_MODE'],
            high_performance_mode:    ['High Performance Mode',   'Enable for GPUs with high throughput (RTX 4090 or better).',         'Performance',  'bool',  null, null, true, null, 'AIWORKER_HIGH_PERFORMANCE_MODE'],
            moderate_performance_mode:['Moderate Performance Mode','Enable for mid-range high-end GPUs (RTX 3080 or better).',          'Performance',  'bool',  null, null, true, null, 'AIWORKER_MODERATE_PERFORMANCE_MODE'],
            unload_models_from_vram_often: ['Unload VRAM Often',  'Unload models from VRAM between jobs to free memory.',              'Performance',  'bool',  null, null, false, null, 'AIWORKER_UNLOAD_MODELS_FROM_VRAM_OFTEN'],
            very_fast_disk_mode:      ['Very Fast Disk Mode',     'Load more models concurrently when using a very fast SSD/NVMe.',     'Performance',  'bool',  null, null, false, null, 'AIWORKER_VERY_FAST_DISK_MODE'],
            post_process_job_overlap: ['PP Job Overlap',          'Overlap post-processing with the next inference job.',               'Performance',  'bool',  null, null, false, null, 'AIWORKER_POST_PROCESS_JOB_OVERLAP'],
            cycle_process_on_model_change: ['Cycle Process on Model Change', 'Restart the inference process when the loaded model changes.', 'Performance', 'bool', null, null, false, null, 'AIWORKER_CYCLE_PROCESS_ON_MODEL_CHANGE'],
            horde_model_stickiness:   ['Model Stickiness',        'Chance (0\u20131) to prefer currently loaded models when popping a job.',  'Performance',  'float', 0.0, 1.0, false, null, 'AIWORKER_MODEL_STICKINESS'],
            process_timeout:          ['Total Job Timeout (s)',        'Max seconds for an entire job (model load + inference + post-processing) before the process is killed.',                     'Timeouts', 'int',  60,   3600, false, null, 'AIWORKER_PROCESS_TIMEOUT'],
            inference_timeout:        ['Inference Timeout - All Steps (s)', 'Max total seconds allowed for all inference steps combined. If inference takes longer than this the process is replaced.', 'Timeouts', 'int',  60,   7200, false, null, 'AIWORKER_INFERENCE_TIMEOUT'],
            inference_step_timeout:   ['Inference Step Timeout (s)',  'Max seconds allowed for a single inference step before the job is detected as stuck.',                                          'Timeouts', 'int',  10,   1800, false, null, 'AIWORKER_INFERENCE_STEP_TIMEOUT'],
            preload_timeout:          ['Model Preload Timeout (s)',   'Max seconds allowed to load (preload) a model before the process is replaced.',                                                 'Timeouts', 'int',  15,   600,  false, null, 'AIWORKER_PRELOAD_TIMEOUT'],
            post_process_timeout:     ['Post-Process Timeout (s)',    'Max seconds allowed for post-processing (upscaling, face-fix, etc.).',                                                          'Timeouts', 'int',  15,   600,  false, null, 'AIWORKER_POST_PROCESS_TIMEOUT'],
            waiting_for_job_timeout:  ['Waiting for Job Timeout (s)', 'Seconds a WAITING_FOR_JOB process can be heartbeat-silent before being replaced (when work is pending). Effective threshold is max(total_job_timeout, this value).', 'Timeouts', 'int', 60, 3600, false, null, 'AIWORKER_WAITING_FOR_JOB_TIMEOUT'],
            minutes_allowed_without_jobs: ['Minutes Without Jobs','Minutes of idle time before the worker warns about no jobs.',       'Behavior',     'int',   0,    3599, false, null, 'AIWORKER_MINUTES_ALLOWED_WITHOUT_JOBS'],
            auto_restart_on_idle_minutes: ['Auto-Restart Idle (min)','Restart the worker automatically if no job has been submitted for this many minutes (0\u00a0=\u00a0disabled; default\u00a060).', 'Behavior', 'int', 0, 1440, false, null, 'AIWORKER_AUTO_RESTART_IDLE_MINUTES'],
            force_restart_timeout:    ['Force Restart Timeout (s)','Seconds to wait for a graceful shutdown before forcing the restart/exit (kills child processes). Mainly applies to auto-restart-on-idle so a stuck shutdown cannot delay the restart.', 'Behavior', 'int', 5, 600, false, null, 'AIWORKER_FORCE_RESTART_TIMEOUT'],
            suppress_speed_warnings:  ['Suppress Speed Warnings', 'Do not print speed-related warning messages.',                      'Behavior',     'bool',  null, null, false, null, 'AIWORKER_SUPPRESS_SPEED_WARNINGS'],
            exit_on_unhandled_faults: ['Exit on Unhandled Faults','Exit the worker instead of recovering from an unhandled fault.',    'Behavior',     'bool',  null, null, false, null, 'AIWORKER_EXIT_ON_UNHANDLED_FAULTS'],
            limited_console_messages: ['Limited Console Messages','Only log job submission and status messages to console.',            'Behavior',     'bool',  null, null, false, null, 'AIWORKER_LIMITED_CONSOLE_MESSAGES'],
            stats_output_frequency:   ['Stats Frequency (s)',     'How often (in seconds) to print the status line to console.',       'Behavior',     'int',   5,    3600, false, null, 'AIWORKER_STATS_OUTPUT_FREQUENCY'],
            purge_loras_on_download:  ['Purge LoRAs on Download', 'Delete existing LoRA cache before downloading new LoRAs.',          'Behavior',     'bool',  null, null, false, null, 'AIWORKER_PURGE_LORAS_ON_DOWNLOAD'],
            remove_maintenance_on_init:['Auto-Remove Maintenance','Automatically clear maintenance mode — on startup AND continuously while running (re-clears it if maintenance is re-applied).', 'Behavior', 'bool', null, null, true, null, 'AIWORKER_REMOVE_MAINTENANCE_ON_INIT'],
            max_job_retries:          ['Max Job Retries',         'Number of times a faulted job is retried before being permanently faulted (0 = no retries; default 1).', 'Behavior', 'int',  0,    10,   false, null, 'AIWORKER_MAX_JOB_RETRIES'],
            max_submit_retries:       ['Max Submit Retries',      'Number of times a job submission can be retried before being marked as failed (default 3).', 'Behavior', 'int',  0,    50,   false, null, 'AIWORKER_MAX_SUBMIT_RETRIES'],
            data_retention_days:      ['Data Retention (days)',   'Number of days to keep errors, statistics, and gallery images in the database (1\u2013\u200a3650; default\u00a07).', 'Behavior', 'int', 1, 3650, false, null, 'AIWORKER_DATA_RETENTION_DAYS'],
            positive_prompt_append:          ['Positive — Add',            'Strings to append to every positive prompt before generation and gallery saving. One entry per line. The original prompt is sent to Horde unchanged.', 'Prompt Filters', 'str_list',        null, null, false, null, null],
            positive_prompt_append_enabled:  ['Positive — Add Enabled',    'When off, strings in the Positive Add list are not appended even if the list is non-empty.',                                                                    'Prompt Filters', 'bool',           null, null, false, null, null],
            positive_prompt_remove:          ['Positive — Remove',         'Strings to remove from every positive prompt. One entry per line.',                                                                                              'Prompt Filters', 'str_list',        null, null, false, null, null],
            positive_prompt_remove_enabled:  ['Positive — Remove Enabled', 'When off, strings in the Positive Remove list are not removed even if the list is non-empty.',                                                                  'Prompt Filters', 'bool',           null, null, false, null, null],
            positive_prompt_replace:                ['Positive — Replace',              'Text pairs to replace in every positive prompt. Enter the original text and the replacement for each pair.',                                                    'Prompt Filters', 'str_replace_list', null, null, false, null, null],
            positive_prompt_replace_enabled:        ['Positive — Replace Enabled',      'When off, rules in the Positive Replace list are not applied even if the list is non-empty.',                                                                   'Prompt Filters', 'bool',           null, null, false, null, null],
            positive_prompt_conditional_add:        ['Positive — Conditional Add',      'Conditional-add rules for positive prompts in trigger==>add format. If trigger is found, add is appended.',                                                       'Prompt Filters', 'str_replace_list', null, null, false, null, null],
            positive_prompt_conditional_add_enabled:['Positive — Cond. Add Enabled',   'When off, conditional-add rules in the Positive list are not applied even if the list is non-empty.',                                                             'Prompt Filters', 'bool',           null, null, false, null, null],
            negative_prompt_append:                 ['Negative — Add',                  'Strings to append to every negative prompt. One entry per line.',                                                                                                'Prompt Filters', 'str_list',        null, null, false, null, null],
            negative_prompt_append_enabled:  ['Negative — Add Enabled',    'When off, strings in the Negative Add list are not appended even if the list is non-empty.',                                                                    'Prompt Filters', 'bool',           null, null, false, null, null],
            negative_prompt_remove:          ['Negative — Remove',         'Strings to remove from every negative prompt. One entry per line.',                                                                                              'Prompt Filters', 'str_list',        null, null, false, null, null],
            negative_prompt_remove_enabled:  ['Negative — Remove Enabled', 'When off, strings in the Negative Remove list are not removed even if the list is non-empty.',                                                                  'Prompt Filters', 'bool',           null, null, false, null, null],
            negative_prompt_replace:                ['Negative — Replace',              'Text pairs to replace in every negative prompt.',                                                                                                                'Prompt Filters', 'str_replace_list', null, null, false, null, null],
            negative_prompt_replace_enabled:        ['Negative — Replace Enabled',      'When off, rules in the Negative Replace list are not applied even if the list is non-empty.',                                                                   'Prompt Filters', 'bool',           null, null, false, null, null],
            negative_prompt_conditional_add:        ['Negative — Conditional Add',      'Conditional-add rules for negative prompts in trigger==>add format. If trigger is found, add is appended.',                                                       'Prompt Filters', 'str_replace_list', null, null, false, null, null],
            negative_prompt_conditional_add_enabled:['Negative — Cond. Add Enabled',   'When off, conditional-add rules in the Negative list are not applied even if the list is non-empty.',                                                             'Prompt Filters', 'bool',           null, null, false, null, null],
            prompt_swap:                     ['Swap',                      'Strings moved between positive and negative prompts — if a string is found in the positive prompt it is moved to the negative prompt, and vice-versa.',           'Prompt Filters', 'str_list',        null, null, false, null, null],
            prompt_swap_enabled:             ['Swap Enabled',              'When off, the swap list is ignored even if non-empty.',                                                                                                              'Prompt Filters', 'bool',           null, null, false, null, null],
            prompt_filters_enabled:           ['Prompt Filters Enabled',           'Master switch — when off, no append/remove/replace operations are applied regardless of the lists below.',                                                                         'Prompt Filters', 'bool', null, null, true,  null, null],
            prompt_remove_cleanup_separators: ['Cleanup Separators After Removal',  'Remove orphaned commas and spaces left between deleted strings. E.g. removing "foo" and "bar" from "a, foo, bar, b" gives "a, b" instead of "a, , , b".',                        'Prompt Filters', 'bool', null, null, false, null, null],
            prompt_append_separator:          ['Auto-Separator When Appending',     'Insert ", " between each appended string and the existing prompt text. When off, strings are concatenated without any separator.',                                                  'Prompt Filters', 'bool', null, null, false, null, null],
            prompt_remove_whole_word:         ['Whole-Word Remove Matching',        'Only remove a string when it appears as a complete word. E.g. "cat" will not match inside "category". Uses word-boundary anchors.',                                                'Prompt Filters', 'bool', null, null, false, null, null],
            prompt_remove_case_sensitive:     ['Case-Sensitive Remove Matching',    'Match remove strings exactly as typed. When off, "Cat" and "CAT" both match "cat".',                                                                                               'Prompt Filters', 'bool', null, null, false, null, null],
        };

        // Default values for each setting key — used by the per-row reset button.
        // null means no known default (reset button hidden for that key).
        var _SETTINGS_DEFAULTS = {
            safety_on_gpu: false, high_memory_mode: true, very_high_memory_mode: false,
            high_performance_mode: true, moderate_performance_mode: false,
            unload_models_from_vram_often: true, very_fast_disk_mode: false,
            post_process_job_overlap: false, cycle_process_on_model_change: false,
            horde_model_stickiness: 0.0,
            process_timeout: 180, inference_timeout: 120, inference_step_timeout: 30,
            preload_timeout: 60, post_process_timeout: 60, waiting_for_job_timeout: 600,
            minutes_allowed_without_jobs: 30, auto_restart_on_idle_minutes: 60,
            force_restart_timeout: 60, suppress_speed_warnings: false,
            exit_on_unhandled_faults: false, limited_console_messages: false,
            stats_output_frequency: 30, purge_loras_on_download: false,
            remove_maintenance_on_init: true, max_job_retries: 1,
            max_submit_retries: 3, data_retention_days: 7,
            positive_prompt_append: [], positive_prompt_append_enabled: true,
            positive_prompt_remove: [], positive_prompt_remove_enabled: true,
            positive_prompt_replace: [], positive_prompt_replace_enabled: true,
            positive_prompt_conditional_add: [], positive_prompt_conditional_add_enabled: true,
            negative_prompt_append: [], negative_prompt_append_enabled: true,
            negative_prompt_remove: [], negative_prompt_remove_enabled: true,
            negative_prompt_replace: [], negative_prompt_replace_enabled: true,
            negative_prompt_conditional_add: [], negative_prompt_conditional_add_enabled: true,
            prompt_swap: [], prompt_swap_enabled: true,
            prompt_filters_enabled: true,
            prompt_remove_cleanup_separators: true,
            prompt_append_separator: true,
            prompt_remove_whole_word: true,
            prompt_remove_case_sensitive: false,
        };

        var _settingsLoaded = false;
        var _settingsFetchInProgress = false;
        var _settingsApplying = false;
        var _settingsDirty = false;
        var _settingsSnapshot = {};
        var _settingsPending = {};
        var _settingsPendingQueue = null;
        var _settingsPendingModels = null;
        var _restartInFlight = false;
        var _restartConfirmOpen = false;

        function _setSettingsStatus(msg, isError) {
            var el = document.getElementById('settings-status');
            if (!el) return;
            if (!msg) {
                el.style.display = 'none';
                el.textContent = '';
                return;
            }
            el.style.display = '';
            el.textContent = msg;
            el.style.color = isError ? 'var(--error)' : '';
        }

        function _updateApplyButtonState() {
            var applyBtn = document.getElementById('settings-apply-btn');
            if (!applyBtn) return;
            applyBtn.disabled = !_settingsDirty || _settingsApplying;
            if (_settingsDirty) applyBtn.classList.add('dirty'); else applyBtn.classList.remove('dirty');
        }

        function _setSettingsDirty(isDirty) {
            _settingsDirty = !!isDirty;
            _updateApplyButtonState();
            if (!_settingsApplying) _setSettingsStatus('', false);
        }

        function _postJson(url, payload) {
            return fetch(url, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(function(r) {
                return r.json().catch(function() { return {}; }).then(function(body) {
                    return {ok: r.ok, body: body, status: r.status};
                });
            });
        }

        function stageSettingChange(key, value) {
            var isPending = true;
            if (Object.prototype.hasOwnProperty.call(_settingsSnapshot, key)) {
                var snap = _settingsSnapshot[key];
                var equal = (typeof value === 'object' && value !== null)
                    ? JSON.stringify(snap) === JSON.stringify(value)
                    : snap === value;
                if (equal) {
                    delete _settingsPending[key];
                    isPending = false;
                }
            }
            if (isPending) {
                _settingsPending[key] = value;
            }
            _setSettingsDirty(
                Object.keys(_settingsPending).length > 0 || _settingsPendingQueue !== null || _settingsPendingModels !== null,
            );
            if (isPending) {
                _showSettingFeedback(key, true, 'Pending', {pending: true});
            } else {
                _clearSettingFeedback(key);
            }
            _updateSingleResetBtn(key, value);
        }

        function _isAtDefault(key, value) {
            if (!Object.prototype.hasOwnProperty.call(_SETTINGS_DEFAULTS, key)) return false;
            var def = _SETTINGS_DEFAULTS[key];
            if (def === null) return false;
            if (Array.isArray(def)) return JSON.stringify(Array.isArray(value) ? value : []) === JSON.stringify(def);
            if (typeof def === 'boolean') return value === def;
            return Number(value) === def;
        }

        function _updateSingleResetBtn(key, currentValue) {
            var btn = document.getElementById('srst-' + key);
            if (!btn) return;
            btn.disabled = _isAtDefault(key, currentValue);
        }

        function _updateAllResetBtns() {
            for (var key in _SETTINGS_DEFAULTS) {
                if (!Object.prototype.hasOwnProperty.call(_SETTINGS_DEFAULTS, key)) continue;
                var currentVal;
                if (Object.prototype.hasOwnProperty.call(_settingsPending, key)) {
                    currentVal = _settingsPending[key];
                } else if (Object.prototype.hasOwnProperty.call(_settingsSnapshot, key)) {
                    currentVal = _settingsSnapshot[key];
                } else {
                    continue;
                }
                _updateSingleResetBtn(key, currentVal);
            }
        }

        function resetSettingToDefault(key) {
            if (!Object.prototype.hasOwnProperty.call(_SETTINGS_DEFAULTS, key)) return;
            var def = _SETTINGS_DEFAULTS[key];
            if (def === null) return;
            if (typeof def === 'boolean') {
                var btn = document.getElementById('srst-' + key);
                if (btn) {
                    var ctrl = btn.closest('.setting-ctrl');
                    if (ctrl) { var cb = ctrl.querySelector('input[type="checkbox"]'); if (cb) cb.checked = def; }
                }
                stageSettingChange(key, def);
            } else if (Array.isArray(def)) {
                var spec = _SETTINGS_SPEC[key];
                var specType = spec ? spec[3] : null;
                if (specType === 'str_replace_list') {
                    var container = document.getElementById('sinp-' + key);
                    if (container) {
                        container.querySelectorAll('.replace-row').forEach(function(r) { r.remove(); });
                    }
                } else {
                    var ta = document.getElementById('sinp-' + key);
                    if (ta) ta.value = '';
                }
                stageSettingChange(key, def);
            } else {
                var inp = document.getElementById('sinp-' + key);
                if (inp) { inp.value = String(def); }
                stageSettingChange(key, def);
            }
        }

        function showResetAllConfirm() {
            var modal = document.getElementById('reset-all-confirm-modal');
            if (!modal) return;
            modal.classList.add('active');
            modal.setAttribute('aria-hidden', 'false');
        }

        function closeResetAllConfirm() {
            var modal = document.getElementById('reset-all-confirm-modal');
            if (!modal) return;
            modal.classList.remove('active');
            modal.setAttribute('aria-hidden', 'true');
        }

        function dismissResetAllConfirm(event) {
            if (event.target === event.currentTarget) closeResetAllConfirm();
        }

        function confirmResetAll() {
            closeResetAllConfirm();
            for (var key in _SETTINGS_DEFAULTS) {
                if (!Object.prototype.hasOwnProperty.call(_SETTINGS_DEFAULTS, key)) continue;
                var def = _SETTINGS_DEFAULTS[key];
                if (def === null) continue;
                resetSettingToDefault(key);
            }
        }

        function showResetDbConfirm() {
            var modal = document.getElementById('reset-db-confirm-modal');
            if (!modal) return;
            modal.classList.add('active');
            modal.setAttribute('aria-hidden', 'false');
            var acceptBtn = document.getElementById('reset-db-confirm-accept');
            if (acceptBtn) acceptBtn.focus();
        }

        function closeResetDbConfirm() {
            var modal = document.getElementById('reset-db-confirm-modal');
            if (!modal) return;
            modal.classList.remove('active');
            modal.setAttribute('aria-hidden', 'true');
            var btn = document.getElementById('settings-reset-db-btn');
            if (btn) btn.focus();
        }

        function dismissResetDbConfirm(event) {
            if (event && event.target === event.currentTarget) closeResetDbConfirm();
        }

        var _resetDbInFlight = false;
        var _resetDbPollTimer = null;
        function _stopResetDbPoll() {
            if (_resetDbPollTimer !== null) { clearInterval(_resetDbPollTimer); _resetDbPollTimer = null; }
        }
        function _finishResetDb(btn, errMsg) {
            _stopResetDbPoll();
            _resetDbInFlight = false;
            if (btn) btn.disabled = false;
            if (errMsg) { _setSettingsStatus(errMsg, true); return; }
            _setSettingsStatus('Database reset (100%)', false);
            updateStatus();
        }
        function _pollResetDbProgress(btn) {
            fetch('/api/reset-database/progress')
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    var pct = (typeof data.progress === 'number') ? data.progress : 0;
                    if (!data.done) {
                        _setSettingsStatus('Resetting database... ' + pct + '%', false);
                        return;
                    }
                    if (data.error) { _finishResetDb(btn, data.error); return; }
                    _finishResetDb(btn, null);
                })
                .catch(function() { /* keep polling on transient errors */ });
        }
        function confirmResetDatabase() {
            closeResetDbConfirm();
            if (_resetDbInFlight) return;
            _resetDbInFlight = true;
            var btn = document.getElementById('settings-reset-db-btn');
            if (btn) btn.disabled = true;
            _setSettingsStatus('Resetting database... 0%', false);
            fetch('/api/reset-database', {method: 'POST'})
                .then(function(r) { return r.json().catch(function() { return {}; }).then(function(body) { return {ok: r.ok, body: body}; }); })
                .then(function(res) {
                    if (!res.ok || !res.body.started) {
                        _finishResetDb(btn, (res.body && res.body.error) ? res.body.error : 'Database reset failed');
                        return;
                    }
                    _resetDbPollTimer = setInterval(function() { _pollResetDbProgress(btn); }, 300);
                })
                .catch(function() { _finishResetDb(btn, 'Database reset failed'); });
        }

        function stageQueueSetting(payload) {
            if (!payload || typeof payload !== 'object') return;
            var next = {};
            if (Object.prototype.hasOwnProperty.call(payload, 'auto')) next.auto = !!payload.auto;
            if (Object.prototype.hasOwnProperty.call(payload, 'max_queue_size')) next.max_queue_size = payload.max_queue_size;
            if (!Object.prototype.hasOwnProperty.call(next, 'auto') && _settingsPendingQueue && Object.prototype.hasOwnProperty.call(_settingsPendingQueue, 'auto')) {
                next.auto = !!_settingsPendingQueue.auto;
            }
            if (!Object.prototype.hasOwnProperty.call(next, 'max_queue_size') && _settingsPendingQueue && Object.prototype.hasOwnProperty.call(_settingsPendingQueue, 'max_queue_size')) {
                next.max_queue_size = _settingsPendingQueue.max_queue_size;
            }
            _settingsPendingQueue = next;

            var autoBtn = document.getElementById('queue-auto-btn');
            var inp = document.getElementById('queue-max-input');
            var isAuto = !!next.auto;
            if (autoBtn) {
                if (isAuto) {
                    autoBtn.classList.add('active');
                    autoBtn.setAttribute('aria-pressed', 'true');
                } else {
                    autoBtn.classList.remove('active');
                    autoBtn.setAttribute('aria-pressed', 'false');
                }
            }
            if (inp) {
                inp.disabled = isAuto;
                if (!isAuto && Object.prototype.hasOwnProperty.call(next, 'max_queue_size')) inp.value = String(next.max_queue_size);
            }
            _showSettingFeedback('job_queue_size', true, 'Pending', {pending: true});
            _setSettingsDirty(true);
        }

        function stageModelsSetting(payload) {
            if (!payload || typeof payload !== 'object') return;
            var next = {};
            if (Object.prototype.hasOwnProperty.call(payload, 'auto')) next.auto = !!payload.auto;
            if (Object.prototype.hasOwnProperty.call(payload, 'max_active_models')) next.max_active_models = payload.max_active_models;
            if (!Object.prototype.hasOwnProperty.call(next, 'auto') && _settingsPendingModels && Object.prototype.hasOwnProperty.call(_settingsPendingModels, 'auto')) {
                next.auto = !!_settingsPendingModels.auto;
            }
            if (!Object.prototype.hasOwnProperty.call(next, 'max_active_models') && _settingsPendingModels && Object.prototype.hasOwnProperty.call(_settingsPendingModels, 'max_active_models')) {
                next.max_active_models = _settingsPendingModels.max_active_models;
            }
            _settingsPendingModels = next;

            var autoBtn = document.getElementById('models-auto-btn');
            var inp = document.getElementById('models-max-input');
            var isAuto = !!next.auto;
            if (autoBtn) {
                if (isAuto) {
                    autoBtn.classList.add('active');
                    autoBtn.setAttribute('aria-pressed', 'true');
                } else {
                    autoBtn.classList.remove('active');
                    autoBtn.setAttribute('aria-pressed', 'false');
                }
            }
            if (inp) {
                inp.disabled = isAuto;
                if (!isAuto && Object.prototype.hasOwnProperty.call(next, 'max_active_models')) inp.value = String(next.max_active_models);
            }
            _showSettingFeedback('active_model_count', true, 'Pending', {pending: true});
            _setSettingsDirty(true);
        }

        function fetchSettings() {
            if (_settingsFetchInProgress) return;
            _settingsFetchInProgress = true;
            // Kick off the models fetch in parallel rather than after the settings
            // render completes -- it targets its own container (#models-section-container)
            // so there's no ordering dependency on the settings HTML being built first.
            fetchModels();
            fetch('/api/settings')
                .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
                .then(function(data) {
                    _settingsLoaded = true;
                    _settingsSnapshot = data.settings || {};
                    renderSettingsPage(_settingsSnapshot);
                    _updateApplyButtonState();
                })
                .catch(function(err) {
                    // Render the page shell with default values so queue/model controls remain
                    // available even when /api/settings fails to load.
                    renderSettingsPage({});
                    var body = document.getElementById('settings-body');
                    if (body) {
                        var banner = document.createElement('div');
                        banner.className = 'settings-unavailable';
                        banner.innerHTML = '&#9888; Failed to load current values: ' + escapeHtml(err.message) + '. Controls show defaults.';
                        body.insertBefore(banner, body.firstChild);
                    }
                })
                .finally(function() { _settingsFetchInProgress = false; });
        }

        function renderSettingsPage(settings) {
            var body = document.getElementById('settings-body');
            if (!body) return;

            // Group settings by category
            var categories = {};
            var catOrder = [];
            for (var key in _SETTINGS_SPEC) {
                if (!Object.prototype.hasOwnProperty.call(_SETTINGS_SPEC, key)) continue;
                var spec = _SETTINGS_SPEC[key];
                var cat = spec[2];
                if (!categories[cat]) { categories[cat] = []; catOrder.push(cat); }
                categories[cat].push(key);
            }

            var html = '';
            for (var ci = 0; ci < catOrder.length; ci++) {
                var cat = catOrder[ci];
                var keys = categories[cat];
                html += '<div class="settings-group">';
                html += '<div class="settings-group-title">' + escapeHtml(cat) + '</div>';
                html += '<div class="settings-grid">';
                if (cat === 'Prompt Filters') {
                    html += _renderPromptFilterRows(settings);
                    html += '</div></div>';
                    continue;
                }
                for (var ki = 0; ki < keys.length; ki++) {
                    var key = keys[ki];
                    var spec = _SETTINGS_SPEC[key];
                    var label = spec[0], desc = spec[1], type = spec[3], minV = spec[4], maxV = spec[5];
                    var envVar = spec[8] || null;
                    var val = (key in settings) ? settings[key] : null;
                    if (Object.prototype.hasOwnProperty.call(_settingsPending, key)) val = _settingsPending[key];
                    html += '<div class="setting-row">';
                    html += '<div class="setting-info"><div class="setting-label">' + escapeHtml(label) + '</div>';
                    if (envVar) { html += '<div class="setting-env"><code>' + escapeHtml(envVar) + '</code></div>'; }
                    html += '<div class="setting-desc">' + escapeHtml(desc) + '</div></div>';
                    html += '<div class="setting-ctrl">';
                    if (type === 'str_readonly') {
                        var strVal = (val !== null && val !== undefined) ? String(val) : '—';
                        html += '<span class="setting-value" title="' + escapeHtml(strVal) + '">' + escapeHtml(strVal) + '</span>';
                    } else if (type === 'url_readonly') {
                        var urlVal = (val !== null && val !== undefined) ? String(val) : '—';
                        if (urlVal !== '—') {
                            html += '<a class="setting-value setting-url-link" href="' + escapeHtml(urlVal) + '" target="_blank" rel="noopener noreferrer" title="Open ' + escapeHtml(urlVal) + '">' + escapeHtml(urlVal) + '</a>';
                        } else {
                            html += '<span class="setting-value" title="—">—</span>';
                        }
                    } else if (type === 'bool') {
                        var chk = (val === true) ? 'checked' : '';
                        html += '<span class="setting-feedback" id="sfb-' + escapeHtml(key) + '"></span>';
                        html += '<label class="setting-toggle" title="' + escapeHtml(label) + '"><input type="checkbox" ' + chk + ' onchange="stageSettingChange(\'' + escapeHtml(key) + '\', this.checked)" aria-label="' + escapeHtml(label) + '"><span class="setting-toggle-slider"></span></label>';
                        if (Object.prototype.hasOwnProperty.call(_SETTINGS_DEFAULTS, key) && _SETTINGS_DEFAULTS[key] !== null) {
                            var boolAtDefault = (val === _SETTINGS_DEFAULTS[key]);
                            html += '<button class="setting-reset-btn" id="srst-' + escapeHtml(key) + '" onclick="resetSettingToDefault(\'' + escapeHtml(key) + '\')" title="Reset to default (' + escapeHtml(String(_SETTINGS_DEFAULTS[key])) + ')" aria-label="Reset ' + escapeHtml(label) + ' to default"' + (boolAtDefault ? ' disabled' : '') + '>&#8635;</button>';
                        }
                    } else if (type === 'int_auto') {
                        var pfx = spec[7];
                        var _changeFnNames = {queue: 'setMaxQueueSize', models: 'setMaxActiveModels'};
                        var _autoFnNames = {queue: 'toggleQueueSizeAuto', models: 'toggleMaxActiveModelsAuto'};
                        var _autoTitles  = {
                            queue:  'Automatically select the best max queue size based on VRAM and job timing',
                            models: 'Automatically select the best active model count based on available VRAM and job timing',
                        };
                        if (pfx === 'queue' && Object.prototype.hasOwnProperty.call(settings, 'max_queue_size')) {
                            val = settings['max_queue_size'];
                        }
                        if (pfx === 'models' && Object.prototype.hasOwnProperty.call(settings, 'max_active_models')) {
                            val = settings['max_active_models'];
                        }
                        if (pfx === 'queue' && _settingsPendingQueue && Object.prototype.hasOwnProperty.call(_settingsPendingQueue, 'max_queue_size')) {
                            val = _settingsPendingQueue.max_queue_size;
                        }
                        if (pfx === 'models' && _settingsPendingModels && Object.prototype.hasOwnProperty.call(_settingsPendingModels, 'max_active_models')) {
                            val = _settingsPendingModels.max_active_models;
                        }
                        var numValA = (val !== null && val !== undefined) ? val : '';
                        var isAutoA = false;
                        if (pfx === 'queue' && Object.prototype.hasOwnProperty.call(settings, 'queue_size_auto')) {
                            isAutoA = !!settings['queue_size_auto'];
                        }
                        if (pfx === 'models' && Object.prototype.hasOwnProperty.call(settings, 'max_active_models_auto')) {
                            isAutoA = !!settings['max_active_models_auto'];
                        }
                        if (pfx === 'queue' && _settingsPendingQueue && Object.prototype.hasOwnProperty.call(_settingsPendingQueue, 'auto')) {
                            isAutoA = !!_settingsPendingQueue.auto;
                        }
                        if (pfx === 'models' && _settingsPendingModels && Object.prototype.hasOwnProperty.call(_settingsPendingModels, 'auto')) {
                            isAutoA = !!_settingsPendingModels.auto;
                        }
                        var minAttrA = (minV !== null) ? ' min="' + minV + '"' : '';
                        var maxAttrA = (maxV !== null) ? ' max="' + maxV + '"' : '';
                        html += '<span class="setting-feedback" id="sfb-' + escapeHtml(key) + '"></span>';
                        html += '<input type="number" class="setting-number" value="' + escapeHtml(String(numValA)) + '"' + minAttrA + maxAttrA + ' step="1" id="' + pfx + '-max-input" aria-label="' + escapeHtml(label) + '"' + (isAutoA ? ' disabled' : '') + ' onchange="' + _changeFnNames[pfx] + '()" onkeydown="if(event.key===\'Enter\'){' + _changeFnNames[pfx] + '();event.preventDefault();}">';
                        html += '<button class="limit-auto-btn' + (isAutoA ? ' active' : '') + '" id="' + pfx + '-auto-btn" onclick="' + _autoFnNames[pfx] + '()" title="' + escapeHtml(_autoTitles[pfx]) + '" aria-pressed="' + (isAutoA ? 'true' : 'false') + '">Auto</button>';
                    } else if (type === 'str_list') {
                        var listVal = Array.isArray(val) ? val.join('\n') : '';
                        html += '<textarea class="setting-textarea" id="sinp-' + escapeHtml(key) + '" rows="3" placeholder="One entry per line…" aria-label="' + escapeHtml(label) + '" onchange="stageListSetting(\'' + escapeHtml(key) + '\')">' + escapeHtml(listVal) + '</textarea>';
                        if (Object.prototype.hasOwnProperty.call(_SETTINGS_DEFAULTS, key)) {
                            var listAtDef = _isAtDefault(key, val);
                            html += '<button class="setting-reset-btn" id="srst-' + escapeHtml(key) + '" onclick="resetSettingToDefault(\'' + escapeHtml(key) + '\')" title="Reset to default (empty)" aria-label="Reset ' + escapeHtml(label) + ' to default"' + (listAtDef ? ' disabled' : '') + '>&#8635;</button>';
                        }
                    } else if (type === 'str_replace_list') {
                        var replaceArr = Array.isArray(val) ? val : [];
                        html += '<div class="setting-replace-list" id="sinp-' + escapeHtml(key) + '">';
                        for (var ri = 0; ri < replaceArr.length; ri++) {
                            var rparts = replaceArr[ri].split('==>', 2);
                            var rfind = rparts[0] || '';
                            var rwith = (rparts.length > 1) ? rparts[1] : '';
                            html += '<div class="replace-row">'
                                + '<input type="text" class="replace-find" placeholder="Find…" value="' + escapeHtml(rfind) + '" oninput="stageReplaceList(\'' + escapeHtml(key) + '\')" aria-label="Find text">'
                                + '<span class="replace-arrow">→</span>'
                                + '<input type="text" class="replace-with" placeholder="Replace with…" value="' + escapeHtml(rwith) + '" oninput="stageReplaceList(\'' + escapeHtml(key) + '\')" aria-label="Replace with">'
                                + '<button class="replace-row-del" onclick="removeReplaceRow(this,\'' + escapeHtml(key) + '\')" title="Remove row" aria-label="Remove row">×</button>'
                                + '</div>';
                        }
                        html += '<button class="replace-add-row" onclick="addReplaceRow(\'' + escapeHtml(key) + '\')">+ Add</button>';
                        html += '</div>';
                        if (Object.prototype.hasOwnProperty.call(_SETTINGS_DEFAULTS, key)) {
                            var replAtDef = _isAtDefault(key, val);
                            html += '<button class="setting-reset-btn" id="srst-' + escapeHtml(key) + '" onclick="resetSettingToDefault(\'' + escapeHtml(key) + '\')" title="Reset to default (empty)" aria-label="Reset ' + escapeHtml(label) + ' to default"' + (replAtDef ? ' disabled' : '') + '>&#8635;</button>';
                        }
                    } else {
                        var numVal = (val !== null && val !== undefined) ? val : '';
                        var minAttr = (minV !== null) ? ' min="' + minV + '"' : '';
                        var maxAttr = (maxV !== null) ? ' max="' + maxV + '"' : '';
                        var step = (type === 'float') ? ' step="0.01"' : ' step="1"';
                        html += '<span class="setting-feedback" id="sfb-' + escapeHtml(key) + '"></span>';
                        html += '<input type="number" class="setting-number" value="' + escapeHtml(String(numVal)) + '"' + minAttr + maxAttr + step + ' id="sinp-' + escapeHtml(key) + '" aria-label="' + escapeHtml(label) + '" onchange="stageNumericSetting(\'' + escapeHtml(key) + '\')" onkeydown="if(event.key===\'Enter\'){stageNumericSetting(\'' + escapeHtml(key) + '\');event.preventDefault();}">';
                        if (Object.prototype.hasOwnProperty.call(_SETTINGS_DEFAULTS, key) && _SETTINGS_DEFAULTS[key] !== null) {
                            var numAtDefault = (val !== null && val !== undefined && Number(val) === _SETTINGS_DEFAULTS[key]);
                            html += '<button class="setting-reset-btn" id="srst-' + escapeHtml(key) + '" onclick="resetSettingToDefault(\'' + escapeHtml(key) + '\')" title="Reset to default (' + escapeHtml(String(_SETTINGS_DEFAULTS[key])) + ')" aria-label="Reset ' + escapeHtml(label) + ' to default"' + (numAtDefault ? ' disabled' : '') + '>&#8635;</button>';
                        }
                    }
                    html += '</div></div>';
                }
                html += '</div></div>';
            }
            body.innerHTML = html || '<div class="settings-unavailable">No configurable settings available.</div>';
            _updateApplyButtonState();
            _updateAllResetBtns();
        }

        var _modelsFetchInProgress = false;
        var _modelsFetchPending = false;
        function fetchModels() {
            if (_modelsFetchInProgress) { _modelsFetchPending = true; return; }
            _modelsFetchInProgress = true;
            _modelsFetchPending = false;
            fetch('/api/models')
                .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
                .then(function(data) { renderModelsSection(data.enabled || [], data.disabled || []); })
                .catch(function() { /* silently skip models section if unavailable */ })
                .finally(function() {
                    _modelsFetchInProgress = false;
                    if (_modelsFetchPending) { _modelsFetchPending = false; fetchModels(); }
                });
        }
        // Full reference for every route registered on this server. Grouped for display on
        // the API page; kept as data rather than inline markup so adding a new endpoint is a
        // one-entry addition here instead of hand-written HTML.
        var _API_GROUPS = [
            {
                title: 'Status & Monitoring',
                desc: 'Read-only endpoints safe to poll from external dashboards or scripts.',
                endpoints: [
                    { method: 'GET', path: '/api/status', desc: 'Full worker status snapshot: jobs, kudos, processes, resource usage, and more. Excludes the last-image payload and full error history to keep polls lightweight.' },
                    { method: 'GET', path: '/api/stats', desc: 'Historical statistics snapshots (CPU/GPU/VRAM/RAM, kudos/images per hour, cumulative counters) plus per-model and per-job-state aggregates.', params: [
                        { name: 'window', type: 'string', required: false, desc: 'Seconds of history to include, or <code>all</code> for the full retention window. Omit for the full window.' },
                    ] },
                    { method: 'GET', path: '/health', desc: 'Basic health check. Returns <code>{"status": "ok"}</code>.' },
                    { method: 'GET', path: '/api/config', desc: 'Static worker configuration (currently just the status-poll interval, in milliseconds).' },
                    { method: 'GET', path: '/api/job_pops/time_without_jobs', desc: 'Total seconds this session has gone with no jobs popped or available, matching the value shown on the Overview page.' },
                    { method: 'GET', path: '/api/horde-snapshots', desc: 'Server-accumulated AI Horde network performance snapshots.', params: [
                        { name: 'window', type: 'string', required: false, desc: 'Same as <code>/api/stats</code>.' },
                    ] },
                    { method: 'GET', path: '/api/horde-modes', desc: 'Last-polled aihorde.net maintenance / invite-only mode flags.' },
                ],
            },
            {
                title: 'Job Pop Control',
                desc: 'Pause or resume the worker accepting new jobs from the Horde.',
                endpoints: [
                    { method: 'POST', path: '/api/job_pops/pause', desc: 'Pause or resume accepting new job pops.', params: [
                        { name: 'paused', type: 'boolean', required: true, desc: '<code>true</code> to pause accepting new jobs, <code>false</code> to resume.' },
                        { name: 'duration_seconds', type: 'number', required: false, desc: 'How long to pause in seconds. Omit or set to <code>null</code> for an indefinite pause.' },
                    ] },
                ],
            },
            {
                title: 'Gallery',
                desc: 'Browse generated images and their metadata.',
                endpoints: [
                    { method: 'GET', path: '/api/gallery', desc: 'Paginated gallery image history, newest first.', params: [
                        { name: 'page', type: 'integer', required: false, desc: '1-based page number. Default 1.' },
                        { name: 'page_size', type: 'integer', required: false, desc: 'Images per page. Default and max 96.' },
                        { name: 'metadata_only', type: 'boolean', required: false, desc: 'If true, omit image/thumbnail data and return only lightweight metadata.' },
                        { name: 'model', type: 'string', required: false, desc: 'Filter to a single model name (case-insensitive).' },
                        { name: 'safety', type: 'string', required: false, desc: 'One of <code>sfw</code>, <code>nsfw</code>, <code>csam</code>.' },
                    ] },
                    { method: 'GET', path: '/api/gallery/thumb/{gallery_id}', desc: 'A single gallery thumbnail as raw image bytes (JPEG, or full-resolution PNG fallback), cached forever by the browser (images are immutable once generated). Used directly as an &lt;img src&gt; by the gallery grid.' },
                    { method: 'GET', path: '/api/gallery/full/{gallery_id}', desc: 'A single gallery image at full resolution as raw PNG bytes, cached forever by the browser. Used directly as an &lt;img src&gt; by the overlay viewer.' },
                    { method: 'GET', path: '/api/gallery/models', desc: 'Distinct model names in the gallery with per-model image counts.' },
                    { method: 'GET', path: '/api/gallery/safety', desc: 'Per-safety-category (sfw/nsfw/csam) image counts.' },
                    { method: 'GET', path: '/api/gallery/image', desc: 'A single gallery image by id.', params: [
                        { name: 'id', type: 'integer', required: true, desc: 'The <code>gallery_id</code> assigned when the image was added.' },
                        { name: 'thumbnail_only', type: 'boolean', required: false, desc: 'If true, return only the thumbnail rather than the full-resolution image.' },
                    ] },
                    { method: 'GET', path: '/api/gallery/last-batch', desc: 'Full-resolution images from the most recently completed job (all outputs sharing one timestamp).' },
                    { method: 'GET', path: '/api/last_image', desc: 'The last generated image(s) and their submission timestamp, kept separate from <code>/api/status</code> to keep that endpoint lightweight.' },
                ],
            },
            {
                title: 'Errors & Logs',
                endpoints: [
                    { method: 'GET', path: '/api/errors', desc: 'Paginated raw error history.', params: [
                        { name: 'page', type: 'integer', required: false, desc: '1-based page number. Default 1.' },
                        { name: 'page_size', type: 'integer', required: false, desc: 'Errors per page. Default 10, max 100.' },
                    ] },
                    { method: 'GET', path: '/api/errors/grouped', desc: 'Errors grouped by normalised message text (variable IDs stripped), sorted by occurrence count.', params: [
                        { name: 'page', type: 'integer', required: false, desc: '1-based page number. Default 1.' },
                        { name: 'page_size', type: 'integer', required: false, desc: 'Groups per page. Default 10, max 100.' },
                    ] },
                    { method: 'POST', path: '/api/errors/clear', desc: 'Clear all accumulated error history, including the persisted database log.' },
                ],
            },
            {
                title: 'Settings & Models',
                endpoints: [
                    { method: 'GET', path: '/api/settings', desc: 'All runtime-configurable settings and their current values.' },
                    { method: 'POST', path: '/api/settings', desc: 'Change a single runtime setting.', params: [
                        { name: 'key', type: 'string', required: true, desc: 'The setting name.' },
                        { name: 'value', type: 'any', required: true, desc: 'The new value; type depends on the setting.' },
                    ] },
                    { method: 'GET', path: '/api/models', desc: 'Enabled and disabled model lists.' },
                    { method: 'POST', path: '/api/models', desc: 'Enable or disable a model.', params: [
                        { name: 'model', type: 'string', required: true, desc: 'The model name.' },
                        { name: 'enabled', type: 'boolean', required: true, desc: '<code>true</code> to enable, <code>false</code> to disable.' },
                    ] },
                ],
            },
            {
                title: 'Maintenance & Lifecycle',
                desc: 'Destructive or worker-lifecycle actions &mdash; as reachable externally as every other endpoint on this page, so use with care.',
                endpoints: [
                    { method: 'POST', path: '/api/maintenance/clear', desc: 'Clear the worker\'s Horde maintenance-mode flag.' },
                    { method: 'POST', path: '/api/restart', desc: 'Restart the worker process. Interrupts any in-progress job processing.' },
                    { method: 'POST', path: '/api/reset-stats', desc: 'Reset the session overview statistics displayed in the UI back to zero (does not delete underlying history).' },
                    { method: 'POST', path: '/api/reset-database', desc: 'Permanently delete all stored history: errors, statistics, gallery images, and Horde network snapshots. Runtime settings are unaffected. Returns immediately; poll <code>/api/reset-database/progress</code> for completion.' },
                    { method: 'GET', path: '/api/reset-database/progress', desc: 'Progress (0-100) of an in-flight database reset, or <code>null</code> if idle.' },
                    { method: 'DELETE', path: '/api/worker/{worker_id}', desc: 'Remove an offline worker from the Horde account. Fails if the worker is online or is the one currently running.' },
                ],
            },
        ];
        var _apiPageRendered = false;
        function _apiEndpointBlockHtml(baseUrl, ep) {
            var h = '<div class="api-ref-endpoint-block">';
            h += '<div class="api-ref-endpoint">';
            h += '<span class="api-ref-method ' + ep.method.toLowerCase() + '">' + escapeHtml(ep.method) + '</span>';
            h += '<span class="api-ref-url">' + escapeHtml(baseUrl + ep.path) + '</span>';
            h += '</div>';
            if (ep.desc) h += '<div class="api-ref-endpoint-desc">' + ep.desc + '</div>';
            if (ep.params && ep.params.length) {
                h += '<table class="api-ref-table"><thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead><tbody>';
                ep.params.forEach(function(p) {
                    var badge = '<span class="api-ref-badge ' + (p.required ? 'required">required' : 'optional">optional') + '</span>';
                    h += '<tr><td>' + escapeHtml(p.name) + '</td><td>' + escapeHtml(p.type) + badge + '</td><td>' + p.desc + '</td></tr>';
                });
                h += '</tbody></table>';
            }
            h += '</div>';
            return h;
        }
        function renderApiPage() {
            var body = document.getElementById('api-page-body');
            if (!body) return;
            var baseUrl = window.location.origin.replace(/\/+$/, '');
            var h = '';
            _API_GROUPS.forEach(function(group) {
                h += '<div class="settings-group api-ref-group">';
                h += '<div class="settings-group-title">' + escapeHtml(group.title) + '</div>';
                if (group.desc) h += '<div class="api-ref-group-desc">' + group.desc + '</div>';
                h += '<div class="api-ref-box">';
                group.endpoints.forEach(function(ep) { h += _apiEndpointBlockHtml(baseUrl, ep); });
                h += '</div></div>';
            });
            body.innerHTML = h;
        }

        function renderModelsSection(enabled, disabled) {
            var container = document.getElementById('models-section-container');
            if (!container) return;

            if (enabled.length === 0 && disabled.length === 0) { container.innerHTML = ''; return; }

            var h = '<div class="settings-group" id="models-section"><div class="settings-group-title">Models</div>';
            h += '<div class="models-containers">';
            // Enabled box
            h += '<div class="models-box">';
            h += '<div class="models-box-title">Enabled <span class="models-count">(' + enabled.length + ')</span></div>';
            h += '<div class="models-pills" id="models-enabled-pills">';
            if (enabled.length === 0) {
                h += '<span class="models-empty">No enabled models</span>';
            } else {
                for (var i = 0; i < enabled.length; i++) {
                    h += '<button type="button" class="model-pill enabled" data-model="' + escapeHtml(enabled[i]) + '" data-enable="false" title="Click to disable" aria-label="Disable model ' + escapeHtml(enabled[i]) + '" aria-pressed="true">' + escapeHtml(enabled[i]) + '</button>';
                }
            }
            h += '</div></div>';
            // Disabled box
            h += '<div class="models-box">';
            h += '<div class="models-box-title">Disabled <span class="models-count">(' + disabled.length + ')</span></div>';
            h += '<div class="models-pills" id="models-disabled-pills">';
            if (disabled.length === 0) {
                h += '<span class="models-empty">No disabled models</span>';
            } else {
                for (var j = 0; j < disabled.length; j++) {
                    h += '<button type="button" class="model-pill disabled" data-model="' + escapeHtml(disabled[j]) + '" data-enable="true" title="Click to enable" aria-label="Enable model ' + escapeHtml(disabled[j]) + '" aria-pressed="false">' + escapeHtml(disabled[j]) + '</button>';
                }
            }
            h += '</div></div>';
            h += '</div></div>';
            container.innerHTML = h;
            // Single delegated listener instead of one per pill -- with large model
            // lists (hundreds of entries) attaching an individual listener to every
            // button was measurably slow to set up and had to be redone on every toggle.
            if (!container._modelsClickBound) {
                container.addEventListener('click', function(e) {
                    var btn = e.target.closest('.model-pill[data-model][data-enable]');
                    if (!btn) return;
                    var modelName = btn.getAttribute('data-model') || '';
                    if (!modelName) return;
                    toggleModel(modelName, btn.getAttribute('data-enable') === 'true');
                });
                container._modelsClickBound = true;
            }
        }

        function toggleModel(modelName, enable) {
            fetch('/api/models', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({model: modelName, enabled: enable})
            })
            .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
            .then(function() { fetchModels(); })
            .catch(function(err) { console.error('Failed to toggle model:', err); });
        }

        function _showSettingFeedback(key, ok, msg, opts) {
            var fb = document.getElementById('sfb-' + key);
            if (!fb) return;
            if (fb._hideTimer) {
                clearTimeout(fb._hideTimer);
                fb._hideTimer = null;
            }
            if (fb._clearTimer) {
                clearTimeout(fb._clearTimer);
                fb._clearTimer = null;
            }
            fb.textContent = msg;
            if (opts && opts.pending) {
                fb.className = 'setting-feedback pending';
            } else {
                fb.className = 'setting-feedback ' + (ok ? 'ok' : 'err');
            }
            fb.style.opacity = '1';
            if (opts && opts.pending) return;
            fb._hideTimer = setTimeout(function() {
                fb.style.opacity = '0';
                fb._hideTimer = null;
                fb._clearTimer = setTimeout(function() {
                    fb.textContent = '';
                    fb._clearTimer = null;
                }, 300);
            }, 2000);
        }

        function _clearSettingFeedback(key) {
            var fb = document.getElementById('sfb-' + key);
            if (!fb) return;
            if (fb._hideTimer) {
                clearTimeout(fb._hideTimer);
                fb._hideTimer = null;
            }
            if (fb._clearTimer) {
                clearTimeout(fb._clearTimer);
                fb._clearTimer = null;
            }
            fb.textContent = '';
            fb.className = 'setting-feedback';
            fb.style.opacity = '0';
        }

        function stageNumericSetting(key) {
            var inp = document.getElementById('sinp-' + key);
            if (!inp) return;
            var spec = _SETTINGS_SPEC[key];
            if (!spec) return;
            var type = spec[3], minV = spec[4], maxV = spec[5];
            var raw = inp.value.trim();
            var parsed = (type === 'float') ? parseFloat(raw) : parseInt(raw, 10);
            if (isNaN(parsed)) { _showSettingFeedback(key, false, 'Invalid'); return; }
            if (minV !== null && parsed < minV) { _showSettingFeedback(key, false, 'Min ' + minV); return; }
            if (maxV !== null && parsed > maxV) { _showSettingFeedback(key, false, 'Max ' + maxV); return; }
            stageSettingChange(key, parsed);
        }

        function stageListSetting(key) {
            var el = document.getElementById('sinp-' + key);
            if (!el) return;
            var lines = el.value.split('\n').map(function(s) { return s.trim(); }).filter(function(s) { return s.length > 0; });
            stageSettingChange(key, lines);
        }

        // ── Prompt filter group helpers ──────────────────────────────────────────

        function _pfgSectionKey(sectionId) {
            // Explicit map so multi-word section IDs like "positive-conditional-add" resolve correctly.
            var map = {
                'positive-add': 'positive_prompt_append',
                'positive-remove': 'positive_prompt_remove',
                'positive-replace': 'positive_prompt_replace',
                'positive-conditional-add': 'positive_prompt_conditional_add',
                'negative-add': 'negative_prompt_append',
                'negative-remove': 'negative_prompt_remove',
                'negative-replace': 'negative_prompt_replace',
                'negative-conditional-add': 'negative_prompt_conditional_add',
                'swap': 'prompt_swap',
            };
            if (map[sectionId]) return map[sectionId];
            // Fallback for forward-compat
            var dash = sectionId.indexOf('-');
            if (dash === -1) return 'prompt_' + sectionId;
            var type = sectionId.slice(0, dash);
            var op   = sectionId.slice(dash + 1);
            var opKey = op === 'add' ? 'append' : op;
            return type + '_prompt_' + opKey;
        }

        function _pfgStageSection(sectionId) {
            var key = _pfgSectionKey(sectionId);
            var container = document.getElementById('pfgsec-' + sectionId);
            if (!container) return;
            var groups = [];
            container.querySelectorAll('.pfg-group').forEach(function(groupEl) {
                var nameEl   = groupEl.querySelector('.pfg-name-input');
                var toggleEl = groupEl.querySelector('.pfg-group-toggle');
                var entries  = [];
                groupEl.querySelectorAll('.pf-pill').forEach(function(p) {
                    var v = p.getAttribute('data-value');
                    if (v !== null && v !== '') entries.push(v);
                });
                groups.push({
                    name:    nameEl    ? nameEl.value.trim() : '',
                    enabled: toggleEl  ? toggleEl.checked   : true,
                    entries: entries,
                });
            });
            stageSettingChange(key, groups);
        }

        function pfgAddGroup(sectionId) {
            var container = document.getElementById('pfgsec-' + sectionId);
            if (!container) return;
            var isReplace = sectionId.indexOf('-replace') !== -1 || sectionId.indexOf('-conditional-add') !== -1;
            var div = document.createElement('div');
            div.className = 'pfg-group';
            div.innerHTML = _pfgGroupBodyHtml(sectionId, '', true, [], isReplace);
            container.appendChild(div);
            _pfgStageSection(sectionId);
            var nameInp = div.querySelector('.pfg-name-input');
            if (nameInp) nameInp.focus();
        }

        function pfgDeleteGroup(btn, sectionId) {
            var groupEl = btn.closest('.pfg-group');
            if (groupEl) groupEl.remove();
            _pfgStageSection(sectionId);
        }

        function pfgAddPill(btn, sectionId) {
            var groupEl = btn.closest('.pfg-group');
            if (!groupEl) return;
            var inp = groupEl.querySelector('.pfg-pill-input');
            if (!inp) return;
            var val = inp.value.trim();
            if (!val) return;
            inp.value = '';
            var pillsContainer = groupEl.querySelector('.pf-pills');
            if (!pillsContainer) return;
            var span = document.createElement('span');
            span.className = 'pf-pill';
            span.setAttribute('data-value', val);
            span.textContent = val + ' ×';
            span.onclick = function() { span.remove(); _pfgStageSection(sectionId); };
            pillsContainer.appendChild(span);
            _pfgStageSection(sectionId);
        }

        function pfgAddReplacePill(btn, sectionId) {
            var groupEl = btn.closest('.pfg-group');
            if (!groupEl) return;
            var findInp = groupEl.querySelector('.pfg-replace-find');
            var withInp = groupEl.querySelector('.pfg-replace-with');
            if (!findInp || !withInp) return;
            var find = findInp.value.trim();
            var withV = withInp.value.trim();
            if (!find && !withV) return;
            findInp.value = ''; withInp.value = '';
            var rule = find + '==>' + withV;
            var display = (find || '…') + ' → ' + (withV || '…');
            var pillsContainer = groupEl.querySelector('.pf-pills');
            if (!pillsContainer) return;
            var span = document.createElement('span');
            span.className = 'pf-pill pf-pill--replace';
            span.setAttribute('data-value', rule);
            span.textContent = display + ' ×';
            span.onclick = function() { span.remove(); _pfgStageSection(sectionId); };
            pillsContainer.appendChild(span);
            _pfgStageSection(sectionId);
        }

        function _pfgGroupBodyHtml(sectionId, name, enabled, entries, isReplace) {
            var e = escapeHtml;
            var sid = e(sectionId);
            var html = '';
            html += '<div class="pfg-group-header">';
            html += '<label class="setting-toggle" title="' + (enabled ? 'Group enabled' : 'Group disabled') + '">'
                 +  '<input type="checkbox" class="pfg-group-toggle"' + (enabled ? ' checked' : '')
                 +  ' onchange="this.closest(\'.pfg-group\').classList.toggle(\'pfg-group--disabled\',!this.checked);_pfgStageSection(\'' + sid + '\')" aria-label="Enable group">'
                 +  '<span class="setting-toggle-slider"></span></label>';
            html += '<input type="text" class="pfg-name-input" value="' + e(name || 'Uncategorized') + '" placeholder="Group name…"'
                 +  ' oninput="_pfgStageSection(\'' + sid + '\')" aria-label="Group name">';
            html += '<button class="pfg-delete-btn" onclick="pfgDeleteGroup(this,\'' + sid + '\')" title="Delete group" aria-label="Delete group">×</button>';
            html += '</div>';
            html += '<div class="pf-pills">';
            entries.forEach(function(entry) {
                if (isReplace) {
                    var parts = entry.split('==>', 2);
                    var find = parts[0] || '';
                    var withV = parts.length > 1 ? parts[1] : '';
                    var disp = e((find || '…') + ' → ' + (withV || '…'));
                    html += '<span class="pf-pill pf-pill--replace" data-value="' + e(entry)
                         +  '" onclick="this.remove();_pfgStageSection(\'' + sid + '\')">' + disp + ' ×</span>';
                } else {
                    html += '<span class="pf-pill" data-value="' + e(entry)
                         +  '" onclick="this.remove();_pfgStageSection(\'' + sid + '\')">' + e(entry) + ' ×</span>';
                }
            });
            html += '</div>';
            if (isReplace) {
                var isCondAdd = sectionId.indexOf('-conditional-add') !== -1;
                var findPh = isCondAdd ? 'Trigger (if found…)' : 'Find…';
                var withPh = isCondAdd ? 'Add to prompt…' : 'Replace with…';
                html += '<div class="pf-input-row">'
                     +  '<input type="text" class="pf-input pfg-replace-find" placeholder="' + findPh + '"'
                     +  ' onkeydown="if(event.key===\'Enter\'){pfgAddReplacePill(this,\'' + sid + '\');event.preventDefault();}">'
                     +  '<span class="replace-arrow">→</span>'
                     +  '<input type="text" class="pf-input pfg-replace-with" placeholder="' + withPh + '"'
                     +  ' onkeydown="if(event.key===\'Enter\'){pfgAddReplacePill(this,\'' + sid + '\');event.preventDefault();}">'
                     +  '<button class="pf-add-btn" onclick="pfgAddReplacePill(this,\'' + sid + '\')" title="Add">+</button>'
                     +  '</div>';
            } else {
                var ph = sectionId === 'swap' ? 'String to swap…' : (sectionId.indexOf('-add') !== -1 ? 'String to append…' : 'String to remove…');
                html += '<div class="pf-input-row">'
                     +  '<input type="text" class="pf-input pfg-pill-input" placeholder="' + ph + '"'
                     +  ' onkeydown="if(event.key===\'Enter\'){pfgAddPill(this,\'' + sid + '\');event.preventDefault();}">'
                     +  '<button class="pf-add-btn" onclick="pfgAddPill(this,\'' + sid + '\')" title="Add">+</button>'
                     +  '</div>';
            }
            return html;
        }

        function _pfgRenderSection(settings, sectionId, label, key, typeEnabledKey, extraColClass) {
            var e = escapeHtml;
            var isReplace = sectionId.indexOf('-replace') !== -1 || sectionId.indexOf('-conditional-add') !== -1;
            function getGroups(k) {
                if (Object.prototype.hasOwnProperty.call(_settingsPending, k)) return _settingsPending[k];
                if (Object.prototype.hasOwnProperty.call(settings, k)) return settings[k];
                return [];
            }
            function getOptVal(k, def) {
                if (Object.prototype.hasOwnProperty.call(_settingsPending, k)) return !!_settingsPending[k];
                if (Object.prototype.hasOwnProperty.call(settings, k)) return !!settings[k];
                return def;
            }
            var groups = Array.isArray(getGroups(key)) ? getGroups(key) : [];
            var typeEnabled = getOptVal(typeEnabledKey, true);
            var colClass = 'pfg-section ' + (extraColClass || 'pf-col');
            var html = '<div class="' + colClass + '">';
            html += '<div class="pfg-section-header">';
            html += '<span class="pf-section-label">' + e(label) + '</span>';
            html += '<label class="setting-toggle" title="' + e(label) + ' enabled">'
                 +  '<input type="checkbox"' + (typeEnabled ? ' checked' : '')
                 +  ' onchange="stageSettingChange(\'' + e(typeEnabledKey) + '\',this.checked)" aria-label="' + e(label) + ' enabled">'
                 +  '<span class="setting-toggle-slider"></span></label>';
            html += '<button class="pfg-add-group-btn" onclick="pfgAddGroup(\'' + e(sectionId) + '\')" title="Add group">+ Group</button>';
            html += '</div>';
            html += '<div class="pfg-groups" id="pfgsec-' + e(sectionId) + '">';
            groups.forEach(function(group) {
                if (!group || typeof group !== 'object') return;
                var gName    = typeof group.name    === 'string' ? group.name  : '';
                var gEnabled = group.enabled !== false;
                var gEntries = Array.isArray(group.entries) ? group.entries : [];
                html += '<div class="pfg-group' + (!gEnabled ? ' pfg-group--disabled' : '') + '">';
                html += _pfgGroupBodyHtml(sectionId, gName, gEnabled, gEntries, isReplace);
                html += '</div>';
            });
            html += '</div>';
            html += '</div>';
            return html;
        }

        function _renderPromptFilterRows(settings) {
            var html = '';
            ['positive', 'negative'].forEach(function(type) {
                var title = type === 'positive' ? 'Positive Prompt Filters' : 'Negative Prompt Filters';
                var desc  = 'Filters applied to every ' + type + ' prompt before generation and gallery saving. '
                          + 'Original prompt sent to Horde unchanged.';
                html += '<div class="pf-block">';
                html += '<div class="pf-block-title">' + escapeHtml(title) + '</div>';
                html += '<div class="pf-block-desc">' + escapeHtml(desc) + '</div>';
                html += '<div class="pf-columns">';
                html += _pfgRenderSection(settings, type + '-add',              'Add',              type + '_prompt_append',          type + '_prompt_append_enabled');
                html += _pfgRenderSection(settings, type + '-remove',           'Remove',           type + '_prompt_remove',          type + '_prompt_remove_enabled');
                html += _pfgRenderSection(settings, type + '-replace',          'Replace',          type + '_prompt_replace',         type + '_prompt_replace_enabled');
                html += _pfgRenderSection(settings, type + '-conditional-add',  'Conditional Add',  type + '_prompt_conditional_add', type + '_prompt_conditional_add_enabled');
                html += '</div></div>';
            });

            // Swap block — spans full width between the two filter rows
            html += '<div class="pf-block">';
            html += '<div class="pf-block-title">Prompt Swap</div>';
            html += '<div class="pf-block-desc">Strings moved between positive and negative prompts. '
                 +  'If a string is found in the positive prompt it is removed from positive and added to negative, '
                 +  'and vice-versa. Applied after append/remove/replace filters.</div>';
            html += '<div>';
            html += _pfgRenderSection(settings, 'swap', 'Swap', 'prompt_swap', 'prompt_swap_enabled', 'pfg-section--full');
            html += '</div></div>';

            // Filter behavior options (global)
            function _pfOptVal(key, defaultVal) {
                if (Object.prototype.hasOwnProperty.call(_settingsPending, key)) return !!_settingsPending[key];
                if (Object.prototype.hasOwnProperty.call(settings, key)) return !!settings[key];
                return defaultVal;
            }
            var filtersEnabled = _pfOptVal('prompt_filters_enabled', true);
            var removeCleanup  = _pfOptVal('prompt_remove_cleanup_separators', true);
            var appendSep      = _pfOptVal('prompt_append_separator', true);
            var wholeWord      = _pfOptVal('prompt_remove_whole_word', true);
            var caseSensitive  = _pfOptVal('prompt_remove_case_sensitive', false);

            html += '<div class="pf-block">';
            html += '<div class="pf-block-title">Filter Options</div>';
            html += '<div class="pf-options-row">';

            html += '<div class="pf-option"><label class="setting-toggle" title="Prompt Filters Enabled">'
                 +  '<input type="checkbox"' + (filtersEnabled ? ' checked' : '') + ' onchange="stageSettingChange(\'prompt_filters_enabled\',this.checked)" aria-label="Prompt Filters Enabled">'
                 +  '<span class="setting-toggle-slider"></span></label>'
                 +  '<div class="pf-option-text"><div class="pf-option-label">Filters enabled</div>'
                 +  '<div class="pf-option-desc">Master switch — when off, no append/remove/replace operations are applied.</div></div></div>';

            html += '<div class="pf-option"><label class="setting-toggle" title="Whole-Word Remove Matching">'
                 +  '<input type="checkbox"' + (wholeWord ? ' checked' : '') + ' onchange="stageSettingChange(\'prompt_remove_whole_word\',this.checked)" aria-label="Whole-Word Remove Matching">'
                 +  '<span class="setting-toggle-slider"></span></label>'
                 +  '<div class="pf-option-text"><div class="pf-option-label">Whole-word matching (remove)</div>'
                 +  '<div class="pf-option-desc">Only remove a string when it appears as a complete word.</div></div></div>';

            html += '<div class="pf-option"><label class="setting-toggle" title="Case-Sensitive Remove Matching">'
                 +  '<input type="checkbox"' + (caseSensitive ? ' checked' : '') + ' onchange="stageSettingChange(\'prompt_remove_case_sensitive\',this.checked)" aria-label="Case-Sensitive Remove Matching">'
                 +  '<span class="setting-toggle-slider"></span></label>'
                 +  '<div class="pf-option-text"><div class="pf-option-label">Case-sensitive matching (remove)</div>'
                 +  '<div class="pf-option-desc">Match remove strings exactly as typed.</div></div></div>';

            html += '<div class="pf-option"><label class="setting-toggle" title="Cleanup Separators After Removal">'
                 +  '<input type="checkbox"' + (removeCleanup ? ' checked' : '') + ' onchange="stageSettingChange(\'prompt_remove_cleanup_separators\',this.checked)" aria-label="Cleanup Separators After Removal">'
                 +  '<span class="setting-toggle-slider"></span></label>'
                 +  '<div class="pf-option-text"><div class="pf-option-label">Cleanup separators after removal</div>'
                 +  '<div class="pf-option-desc">Remove orphaned commas and spaces left between deleted strings.</div></div></div>';

            html += '<div class="pf-option"><label class="setting-toggle" title="Auto-Separator When Appending">'
                 +  '<input type="checkbox"' + (appendSep ? ' checked' : '') + ' onchange="stageSettingChange(\'prompt_append_separator\',this.checked)" aria-label="Auto-Separator When Appending">'
                 +  '<span class="setting-toggle-slider"></span></label>'
                 +  '<div class="pf-option-text"><div class="pf-option-label">Auto-separator when appending</div>'
                 +  '<div class="pf-option-desc">Insert ", " between each appended string and the existing prompt text.</div></div></div>';

            html += '</div></div>';
            return html;
        }

        async function applyPendingSettings() {
            if (_settingsApplying) return;
            if (!_settingsDirty) {
                _setSettingsStatus('No pending changes', false);
                return;
            }

            _settingsApplying = true;
            _setSettingsStatus('Applying...', false);
            _updateApplyButtonState();
            var restartBtn = document.getElementById('settings-restart-btn');
            if (restartBtn) restartBtn.disabled = true;

            var failures = [];
            var applySucceeded = false;
            var key;

            try {
                var pendingKeys = Object.keys(_settingsPending);
                for (var i = 0; i < pendingKeys.length; i++) {
                    key = pendingKeys[i];
                    var settingRes = await _postJson('/api/settings', {key: key, value: _settingsPending[key]});
                    if (!settingRes.ok) {
                        failures.push((settingRes.body && settingRes.body.error) ? settingRes.body.error : ('Failed: ' + key));
                        _showSettingFeedback(key, false, 'Error');
                    } else {
                        _showSettingFeedback(key, true, '\u2713');
                    }
                }

                if (_settingsPendingQueue !== null) {
                    if (_settingsPendingQueue.auto === true || _settingsPendingQueue.auto === false) {
                        var queueAutoRes = await _postJson('/api/settings', {key: 'queue_size_auto', value: !!_settingsPendingQueue.auto});
                        if (!queueAutoRes.ok) {
                            failures.push((queueAutoRes.body && queueAutoRes.body.error) ? queueAutoRes.body.error : 'Failed queue auto update');
                        }
                    }
                    if (_settingsPendingQueue.auto !== true && Object.prototype.hasOwnProperty.call(_settingsPendingQueue, 'max_queue_size')) {
                        var queueSizeRes = await _postJson('/api/settings', {key: 'max_queue_size', value: _settingsPendingQueue.max_queue_size});
                        if (!queueSizeRes.ok) {
                            failures.push((queueSizeRes.body && queueSizeRes.body.error) ? queueSizeRes.body.error : 'Failed queue update');
                        }
                    }
                }

                if (_settingsPendingModels !== null) {
                    if (_settingsPendingModels.auto === true || _settingsPendingModels.auto === false) {
                        var modelsAutoRes = await _postJson('/api/settings', {key: 'max_active_models_auto', value: !!_settingsPendingModels.auto});
                        if (!modelsAutoRes.ok) {
                            failures.push((modelsAutoRes.body && modelsAutoRes.body.error) ? modelsAutoRes.body.error : 'Failed max active models auto update');
                        }
                    }
                    if (_settingsPendingModels.auto !== true && Object.prototype.hasOwnProperty.call(_settingsPendingModels, 'max_active_models')) {
                        var modelsSizeRes = await _postJson('/api/settings', {key: 'max_active_models', value: _settingsPendingModels.max_active_models});
                        if (!modelsSizeRes.ok) {
                            failures.push((modelsSizeRes.body && modelsSizeRes.body.error) ? modelsSizeRes.body.error : 'Failed max active models update');
                        }
                    }
                }

                if (failures.length > 0) {
                    _setSettingsStatus(failures[0], true);
                    return;
                }

                _settingsPending = {};
                _settingsPendingQueue = null;
                _settingsPendingModels = null;
                applySucceeded = true;
                fetchSettings();
            } finally {
                _settingsApplying = false;
                if (applySucceeded) _setSettingsDirty(false);
                _updateApplyButtonState();
                if (restartBtn) restartBtn.disabled = _restartInFlight;
            }
        }

        function restartProgram() {
            if (_restartInFlight) return;
            var modal = document.getElementById('restart-confirm-modal');
            if (!modal) return;
            _restartConfirmOpen = true;
            modal.classList.add('active');
            modal.setAttribute('aria-hidden', 'false');
            var acceptBtn = document.getElementById('restart-confirm-accept');
            if (acceptBtn) acceptBtn.focus();
        }

        function closeRestartConfirm() {
            var modal = document.getElementById('restart-confirm-modal');
            if (!modal) return;
            _restartConfirmOpen = false;
            modal.classList.remove('active');
            modal.setAttribute('aria-hidden', 'true');
            var restartBtn = document.getElementById('settings-restart-btn');
            if (restartBtn) restartBtn.focus();
        }

        function dismissRestartConfirm(evt) {
            if (!evt) return;
            if (evt.target && evt.target.id === 'restart-confirm-modal') closeRestartConfirm();
        }

        function confirmRestartProgram() {
            closeRestartConfirm();
            if (_restartInFlight) return;
            _restartInFlight = true;
            var restartBtn = document.getElementById('settings-restart-btn');
            if (restartBtn) restartBtn.disabled = true;
            _setSettingsStatus('Requesting restart...', false);
            fetch('/api/restart', {method: 'POST'})
                .then(function(r) { return r.json().catch(function() { return {}; }).then(function(body) { return {ok: r.ok, body: body}; }); })
                .then(function(res) {
                    if (!res.ok) {
                        _restartInFlight = false;
                        if (restartBtn) restartBtn.disabled = false;
                        _setSettingsStatus((res.body && res.body.error) ? res.body.error : 'Restart failed', true);
                        return;
                    }
                    _setSettingsStatus('Restart requested...', false);
                })
                .catch(function() {
                    _restartInFlight = false;
                    if (restartBtn) restartBtn.disabled = false;
                    _setSettingsStatus('Restart request failed', true);
                });
        }

        async function fetchWithTimeout(url, timeoutMs) {
            const controller = new AbortController();
            const timerId = setTimeout(() => controller.abort(new Error('Request timed out after '+timeoutMs+'ms')), timeoutMs);
            try {
                return await fetch(url, { signal: controller.signal });
            } finally {
                clearTimeout(timerId);
            }
        }
        async function initializeUpdates() {
            // Fetch the last image immediately so the overview container shows it
            // before the first /api/status poll returns (~1 s later).
            fetch('/api/last_image')
                .then(function(r) { return r.json(); })
                .then(function(imgData) {
                    // Only skip if a real session image (ts > 0) has already been fetched.
                    // If ts is 0 the status poll beat us here but showed nothing — still show gallery.
                    if (_lastFetchedImageTimestamp !== null && _lastFetchedImageTimestamp > 0) return;
                    var ts = imgData && imgData.last_image_submission_timestamp;
                    if (typeof ts !== 'number') ts = Number(ts);
                    if (!Number.isFinite(ts)) ts = 0;
                    _lastFetchedImageTimestamp = ts;
                    var hasSessionImage = ts !== 0 && imgData.last_image_base64 && imgData.last_image_base64.length > 0;
                    if (hasSessionImage) {
                        // Set the label source immediately instead of waiting up to a
                        // second for the first /api/status poll to repeat the value.
                        _lastImageSubmissionTimestamp = ts;
                        renderLastImages(
                            imgData.last_image_base64 || [],
                            document.getElementById('overview-image-container'),
                            ts,
                            imgData.last_image_model || null,
                            imgData.last_image_safety || null
                        );
                    } else {
                        // No session image yet — show last gallery batch as a preview.
                        fetch('/api/gallery/last-batch')
                            .then(function(r) { return r.json(); })
                            .then(function(batchData) {
                                // Abort if a real session image arrived while we were fetching.
                                if (_lastFetchedImageTimestamp !== 0) return;
                                var imgs = batchData && batchData.images;
                                if (!imgs || imgs.length === 0) return;
                                var b64arr = imgs.map(function(i) { return i.base64; }).filter(Boolean);
                                if (b64arr.length === 0) return;
                                var safety = imgs.map(function(i) { return { is_nsfw: i.is_nsfw, is_csam: i.is_csam }; });
                                var previewTs = Number(imgs[0].timestamp);
                                _galleryPreviewTimestamp = (Number.isFinite(previewTs) && previewTs > 0) ? previewTs : null;
                                renderLastImages(
                                    b64arr,
                                    document.getElementById('overview-image-container'),
                                    imgs[0].timestamp,
                                    imgs[0].model || null,
                                    safety
                                );
                            })
                            .catch(function() {});
                    }
                })
                .catch(function() {});
            try {
                const config = await (await fetchWithTimeout('/api/config', CONFIG_FETCH_TIMEOUT_MS)).json();
                updateIntervalMs = config.update_interval_ms || DEFAULT_UPDATE_INTERVAL_MS;
                updateStatus();
            } catch (e) {
                console.error('Error fetching config:', e);
                updateStatus();
            }
        }
        document.addEventListener('visibilitychange', function() {
            if (document.visibilityState === 'visible') {
                if (scheduledUpdateTimer !== null) { clearTimeout(scheduledUpdateTimer); scheduledUpdateTimer = null; }
                // If a status request is already in flight, let it complete rather than
                // aborting it and starting a new one. This avoids races between overlapping
                // requests and stale `.finally()` handlers.
                if (!statusAbortController) {
                    updateStatus();
                }
            }
        });
        initializeUpdates();
    </script>
</body>
</html>
        """
        # Inject server-accumulated horde snapshots so they're available synchronously
        # the moment the page loads — no separate /api/horde-snapshots round-trip needed.
        # Downsampled the same way as the /api/horde-snapshots endpoint: this seed is sent
        # on every single page load regardless of whether the user ever opens the Horde
        # tab, so it must never scale with the full retention window's raw sample count.
        snaps_json = json.dumps(_downsample_series(list(self._horde_snapshots), _CHART_MAX_POINTS))
        html = html.replace(
            "var _hordeSnapshots = [];",
            f"var _hordeSnapshots = {snaps_json};",
        )
        html = html.replace("{{WORKER_VERSION}}", horde_worker_regen.__version__)
        return web.Response(text=html, content_type="text/html")

    async def _handle_delete_worker(self, request: web.Request) -> web.Response:
        """Handle a request to delete an offline worker via the Horde API.

        URL parameter:
            worker_id: The UUID of the worker to delete.

        Returns 400 if the worker is online or is the currently running worker.
        Returns 503 if no delete callback has been registered.  Returns 404 if the
        worker_id is not found in the current workers list.  Returns 200 on success.
        """
        worker_id = request.match_info.get("worker_id", "").strip()
        if not worker_id:
            return web.json_response({"error": "Missing worker_id"}, status=400)

        workers_list: list[dict[str, Any]] = []
        ud = self.status_data.get("user_details") or {}
        if isinstance(ud, dict):
            workers_list = ud.get("workers_list") or []

        # Locate the worker in the cached list
        target: dict[str, Any] | None = None
        for w in workers_list:
            if str(w.get("id", "")) == worker_id:
                target = w
                break

        if target is None:
            return web.json_response({"error": "Worker not found"}, status=404)

        # Guard: must be offline
        if target.get("online", False):
            return web.json_response({"error": "Worker is online and cannot be deleted"}, status=400)

        # Guard: must not be the currently running worker (matched by name)
        current_worker_name: str = self.status_data.get("worker_name", "") or ""
        if current_worker_name and target.get("name", "") == current_worker_name:
            return web.json_response(
                {"error": "Cannot delete the worker currently running the web UI"},
                status=400,
            )

        if self._delete_worker_callback is None:
            return web.json_response({"error": "Delete worker is not available"}, status=503)

        try:
            success = await self._delete_worker_callback(worker_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Error deleting worker {worker_id}: {exc}")
            return web.json_response({"error": "Internal error while deleting worker"}, status=500)

        if not success:
            return web.json_response({"error": "Failed to delete worker"}, status=502)

        return web.json_response({"deleted_id": worker_id})

    async def _handle_set_job_pops_paused(self, request: web.Request) -> web.Response:
        """Handle a request to pause or resume accepting new job pops.

        Expected JSON body: ``{"paused": true}`` or ``{"paused": false}``.
        When pausing, an optional ``duration_seconds`` field (positive number)
        may be included to set how long the pause lasts. If the key is absent
        (or explicitly ``null``), the pause is indefinite.

        If a pause request arrives while a pause is already active the timer is
        reset to the duration specified in the new request (or becomes
        indefinite if ``duration_seconds`` is omitted).

        The endpoint is accessible from any IP address, so external
        applications and automation scripts can pause/resume job pops without
        needing access to the local UI.

        Returns 400 on malformed input, 503 if no callback is registered, and
        200 with ``{"job_pops_paused": <bool>, "job_pops_pause_until": <float|null>}``
        on success.
        """
        try:
            body = await request.json()
        except (ValueError, TypeError, aiohttp.ContentTypeError) as exc:
            return web.json_response({"error": f"Invalid JSON body: {exc}"}, status=400)

        paused = body.get("paused")
        if not isinstance(paused, bool):
            return web.json_response({"error": "Field 'paused' must be a boolean"}, status=400)

        duration_seconds = body.get("duration_seconds")
        if duration_seconds is not None and not isinstance(duration_seconds, (int, float)):
            return web.json_response({"error": "Field 'duration_seconds' must be a number or null"}, status=400)
        if duration_seconds is not None and duration_seconds <= 0:
            return web.json_response({"error": "Field 'duration_seconds' must be a positive number"}, status=400)

        if self._set_job_pops_paused_callback is None:
            return web.json_response({"error": "Pause job pops is not available"}, status=503)

        pause_until: float | None = None
        if paused:
            if duration_seconds is not None:
                pause_until = time.time() + float(duration_seconds)

        try:
            self._set_job_pops_paused_callback(paused, pause_until)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Error setting job pops paused={paused}: {exc}")
            return web.json_response({"error": f"Internal error: {type(exc).__name__}"}, status=500)

        self.status_data["job_pops_paused"] = paused
        self.status_data["job_pops_pause_until"] = pause_until
        return web.json_response({"job_pops_paused": paused, "job_pops_pause_until": pause_until})

    async def _handle_get_time_without_jobs(self, request: web.Request) -> web.Response:
        """Return the total time (in seconds) this session has gone with no jobs popped or available.

        Like ``/api/job_pops/pause``, this endpoint is accessible from any IP address so
        external applications and automation scripts can poll it without needing access to
        the local UI -- e.g. to decide when to intervene on a worker that has gone idle.

        The value matches what the web UI's overview page displays: it is subject to the
        same "reset stats" baseline subtraction as ``/api/status``'s ``time_without_jobs``
        field, so it reflects time accumulated since the last stats reset rather than since
        process start.

        Returns 200 with ``{"time_without_jobs": <float seconds>}``.
        """
        reset_baseline = self.status_data.get("stats_reset_baseline") or {}
        raw_time_without_jobs = float(self.status_data.get("time_without_jobs", 0.0))
        time_without_jobs = max(0.0, raw_time_without_jobs - float(reset_baseline.get("time_without_jobs", 0.0)))
        return web.json_response({"time_without_jobs": time_without_jobs})

    async def _handle_clear_maintenance_mode(self, request: web.Request) -> web.Response:
        """Handle a request to clear the worker's maintenance mode.

        Returns 503 if no callback is registered and 200 with
        ``{"maintenance_mode": false}`` on success.
        """
        if self._clear_maintenance_mode_callback is None:
            return web.json_response({"error": "Clear maintenance mode is not available"}, status=503)

        try:
            self._clear_maintenance_mode_callback()
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Error clearing maintenance mode: {exc}")
            return web.json_response({"error": f"Internal error: {type(exc).__name__}"}, status=500)

        self.status_data["maintenance_mode"] = False
        return web.json_response({"maintenance_mode": False})

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Handle status API request.

        Returns all status fields **except** last-image payload fields and the
        full ``errors_history`` list so that large payloads are not included in
        every poll. Clients should use ``last_image_submission_timestamp`` to
        detect new images (fetch via ``/api/last_image``) and ``errors_count``
        to detect new errors (fetch the relevant page via ``/api/errors``).
        """
        payload = {
            k: v
            for k, v in self.status_data.items()
            if k not in ("last_image_base64", "last_image_model", "last_image_safety", "errors_history")
        }
        payload["errors_count"] = len(self.status_data["errors_history"])
        return web.json_response(payload)

    async def _handle_last_image(self, request: web.Request) -> web.Response:
        """Return only the last generated image(s) and their submission timestamp.

        Separating image data from the main status response keeps ``/api/status``
        lightweight so the overview page loads quickly.  The client fetches this
        endpoint only when ``last_image_submission_timestamp`` changes, i.e. when
        a genuinely new image is available.
        """
        return web.json_response(
            {
                "last_image_base64": self.status_data["last_image_base64"],
                "last_image_submission_timestamp": self.status_data["last_image_submission_timestamp"],
                "last_image_model": self.status_data["last_image_model"],
                "last_image_safety": self.status_data["last_image_safety"],
            },
        )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check request."""
        return web.json_response({"status": "ok"})

    async def _handle_stats(self, request: web.Request) -> web.Response:
        """Return historical statistics snapshots and per-model image counts.

        Accepts an optional ``window`` query parameter (seconds, or ``"all"``/absent for
        the full retention window) selecting how far back to look. The matching snapshots
        are downsampled to roughly _CHART_MAX_POINTS points server-side -- the wider the
        requested window, the coarser the resolution -- so a multi-day/week "All" view
        doesn't ship and re-render every raw 10-second sample.

        Returns snapshots in chronological order (oldest first). Each snapshot contains a
        Unix timestamp plus CPU/GPU/VRAM/RAM usage percentages, container CPU percentage,
        images/hour, kudos/hour, and cumulative job/kudos counters for the current session.

        Response shape::

            {
                "snapshots": [
                    {
                        "t": <float>,            # Unix timestamp
                        "cpu": <float>,          # system CPU %
                        "gpu": <float>,          # GPU %
                        "vram": <float>,         # worker VRAM as % of total device VRAM
                        "ram": <float>,          # worker RAM as % of system total
                        "system_ram": <float>,   # system-wide RAM as % of total
                        "container_cpu": <float>,# worker process + children CPU %
                        "iph": <float>,          # images per hour
                        "kph": <float>,          # kudos per hour
                        "jc": <int>,             # jobs completed
                        "jf": <int>,             # jobs faulted
                        "jp": <int>,             # jobs popped
                        "ks": <float>            # kudos earned this session
                    },
                    ...
                ],
                "images_per_model": {"model-name": <int count>, ...},
                "failed_jobs_per_model": {"model-name": <int count>, ...},
                "faulted_jobs_per_phase": {"phase-name": <int count>, ...},
                "avg_time_per_job_state": {"state-name": <float seconds>, ...},
                "max_time_per_job_state": {"state-name": <float seconds>, ...},
                "jobs_faulted": <int>  # session-total jobs faulted
            }
        """
        snapshots = _windowed_snapshots(list(self._stats_snapshots), request.query.get("window"))
        snapshots = _downsample_series(snapshots, _CHART_MAX_POINTS, _STATS_CUMULATIVE_KEYS)
        return web.json_response({
            "snapshots": snapshots,
            "images_per_model": self.status_data.get("images_per_model", {}),
            "failed_jobs_per_model": self.status_data.get("failed_jobs_per_model", {}),
            "faulted_jobs_per_phase": self.status_data.get("faulted_jobs_per_phase", {}),
            "avg_time_per_job_state": self.status_data.get("avg_time_per_job_state", {}),
            "max_time_per_job_state": self.status_data.get("max_time_per_job_state", {}),
            "avg_time_per_step_per_model": self.status_data.get("avg_time_per_step_per_model", {}),
            "max_time_per_step_per_model": self.status_data.get("max_time_per_step_per_model", {}),
            "avg_time_per_job_per_model": self.status_data.get("avg_time_per_job_per_model", {}),
            "max_time_per_job_per_model": self.status_data.get("max_time_per_job_per_model", {}),
            "jobs_faulted": int(self.status_data.get("jobs_faulted", 0)),
        })

    def _record_stats_snapshot(self) -> None:
        """Append a statistics snapshot to the ring buffer if enough time has elapsed."""
        now = time.time()
        if now - self._last_stats_snapshot_time < self._stats_snapshot_interval:
            return
        self._last_stats_snapshot_time = now
        sd = self.status_data
        vram_total: float = float(sd.get("total_vram_mb") or 0)
        vram_pct = min(100.0, round((float(sd.get("vram_usage_mb", 0)) / vram_total) * 100, 1)) if vram_total > 0 else 0.0
        system_vram_pct = max(
            vram_pct,
            min(100.0, round((float(sd.get("system_vram_usage_mb", 0)) / vram_total) * 100, 1))
            if vram_total > 0
            else 0.0,
        )
        ram_total: float = float(sd.get("total_ram_mb") or 0)
        ram_pct = min(100.0, round((float(sd.get("ram_usage_mb", 0)) / ram_total) * 100, 1)) if ram_total > 0 else 0.0
        system_ram_pct = min(100.0, round((float(sd.get("system_ram_usage_mb", 0)) / ram_total) * 100, 1)) if ram_total > 0 else 0.0
        worker_gpu_pct = round(float(sd.get("worker_gpu_percent", 0)), 1)
        snapshot: dict[str, float | int] = {
            "t": round(now, 1),
            "cpu": round(float(sd.get("cpu_usage_percent", 0)), 1),
            "gpu": max(round(float(sd.get("gpu_usage_percent", 0)), 1), worker_gpu_pct),
            "worker_gpu": worker_gpu_pct,
            "vram": vram_pct,
            "system_vram": system_vram_pct,
            "ram": ram_pct,
            "system_ram": system_ram_pct,
            "container_cpu": round(float(sd.get("container_cpu_percent", 0)), 1),
            "iph": round(float(sd.get("images_per_hour", 0)), 2),
            "kph": round(float(sd.get("kudos_per_hour", 0)), 2),
            "jc": int(sd.get("jobs_completed", 0)),
            "jf": int(sd.get("jobs_faulted", 0)),
            "jp": int(sd.get("jobs_popped", 0)),
            "ks": round(float(sd.get("kudos_earned_session", 0)), 2),
        }
        self._stats_snapshots.append(snapshot)

        # Persist to database.
        if self._stats_db_path is not None:
            try:
                with sqlite3.connect(self._stats_db_path) as conn:
                    conn.execute(
                        "INSERT INTO stats_snapshots (snapshot_json, timestamp) VALUES (?, ?)",
                        (json.dumps(snapshot), now),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO session_overview "
                        "(id, jobs_popped, jobs_completed, jobs_faulted, processes_recovered, "
                        "kudos_earned, time_without_jobs, updated_at, reset_baseline_json) "
                        "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            snapshot["jp"],
                            snapshot["jc"],
                            snapshot["jf"],
                            int(sd.get("processes_recovered", 0)),
                            snapshot["ks"],
                            float(sd.get("time_without_jobs", 0.0)),
                            now,
                            json.dumps(sd.get("stats_reset_baseline") or {}),
                        ),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO session_aggregates (id, aggregates_json) VALUES (1, ?)",
                        (
                            json.dumps(
                                {
                                    "images_per_model": sd.get("images_per_model") or {},
                                    "failed_jobs_per_model": sd.get("failed_jobs_per_model") or {},
                                    "faulted_jobs_per_phase": sd.get("faulted_jobs_per_phase") or {},
                                    "avg_time_per_job_state": sd.get("avg_time_per_job_state") or {},
                                    "max_time_per_job_state": sd.get("max_time_per_job_state") or {},
                                    "avg_time_per_step_per_model": sd.get("avg_time_per_step_per_model") or {},
                                    "max_time_per_step_per_model": sd.get("max_time_per_step_per_model") or {},
                                    "avg_time_per_job_per_model": sd.get("avg_time_per_job_per_model") or {},
                                    "max_time_per_job_per_model": sd.get("max_time_per_job_per_model") or {},
                                }
                            ),
                        ),
                    )
                    conn.commit()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not persist stats snapshot to database: {exc}")

        # Periodically prune expired data to keep the databases — or, when none are
        # configured, the in-memory collections — from growing unboundedly. This must
        # run regardless of whether the stats DB is available.
        if now - self._last_db_prune_time >= _DB_PRUNE_INTERVAL:
            self._prune_old_db_data()


    async def _handle_errors(self, request: web.Request) -> web.Response:
        """Return a paginated slice of the error history.

        Query parameters:
            page: 1-based page number (default: 1)
            page_size: errors per page (default: 10, max: 100)
        """
        try:
            page = max(1, int(request.rel_url.query.get("page", "1")))
        except ValueError:
            page = 1
        try:
            page_size = min(100, max(1, int(request.rel_url.query.get("page_size", "10"))))
        except ValueError:
            page_size = 10
        errors = self.status_data["errors_history"]
        total = len(errors)
        total_pages = max(1, math.ceil(total / page_size))
        page = min(page, total_pages)
        start = (page - 1) * page_size
        return web.json_response(
            {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "errors": errors[start : start + page_size],
            },
        )

    @staticmethod
    def _normalize_error_message(msg: str) -> str:
        """Return a normalised version of *msg* suitable for grouping.

        Variable tokens that differ between occurrences of the same error
        (timestamps, UUIDs, hex addresses, numeric IDs) are replaced with
        placeholders so that the same underlying error is always mapped to the
        same group key regardless of when it occurred or which job/process
        triggered it.
        """
        msg = _ERROR_TIMESTAMP_RE.sub("<time>", msg)
        msg = _ERROR_UUID_RE.sub("<id>", msg)
        msg = _ERROR_HEX_ID_RE.sub("<hex>", msg)
        msg = _ERROR_NUM_TOKEN_RE.sub("<num>", msg)
        return msg

    async def _handle_errors_grouped(self, request: web.Request) -> web.Response:
        """Return errors grouped by normalised message text, sorted by occurrence count descending.

        All errors are included – single-occurrence errors appear as a group of 1.

        Variable tokens in error messages (UUIDs, hex addresses, long numeric IDs
        such as job IDs or process IDs) are stripped before grouping so that the
        same underlying error triggered by different jobs is counted together.

        Query parameters:
            page: 1-based page number (default: 1)
            page_size: groups per page (default: 10, max: 100)
        """
        try:
            page = max(1, int(request.rel_url.query.get("page", "1")))
        except ValueError:
            page = 1
        try:
            page_size = min(100, max(1, int(request.rel_url.query.get("page_size", "10"))))
        except ValueError:
            page_size = 10
        errors = self.status_data["errors_history"]
        # Group by normalised message; keep one representative original message per group
        # and record up to _MAX_OCCURRENCES_PER_GROUP individual occurrences for timeline
        # display while still counting every occurrence for the true total.
        counts: dict[str, int] = {}
        representatives: dict[str, str] = {}
        occurrences: dict[str, list[str]] = {}
        for msg in errors:
            key = self._normalize_error_message(msg)
            counts[key] = counts.get(key, 0) + 1
            if key not in representatives:
                representatives[key] = msg
                occurrences[key] = []
            if len(occurrences[key]) < _MAX_OCCURRENCES_PER_GROUP:
                occurrences[key].append(msg)
        # Include all groups (single-occurrence errors appear as a group of 1)
        qualified_groups = list(counts.items())
        # Sort by count descending, then alphabetically for stable ordering
        groups = sorted(qualified_groups, key=lambda x: (-x[1], x[0]))
        total_groups = len(groups)
        total_errors = len(errors)
        total_pages = max(1, math.ceil(total_groups / page_size))
        page = min(page, total_pages)
        start = (page - 1) * page_size
        page_groups = [
            {
                "message": representatives[key],
                "count": cnt,
                "occurrences": occurrences[key],
            }
            for key, cnt in groups[start : start + page_size]
        ]
        return web.json_response(
            {
                "total_groups": total_groups,
                "total_errors": total_errors,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "groups": page_groups,
            },
        )

    async def _handle_errors_clear(self, request: web.Request) -> web.Response:
        """Clear all accumulated error history, including the persisted DB log."""
        self.status_data["errors_history"] = []
        self._live_errors_history = []
        self._persisted_errors = []
        if self._errors_db_path is not None:
            try:
                with sqlite3.connect(self._errors_db_path) as conn:
                    conn.execute("DELETE FROM errors_log")
                    conn.commit()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not clear errors from DB: {exc}")
        return web.json_response({"ok": True})

    async def _handle_gallery(self, request: web.Request) -> web.Response:
        """Return a paginated slice of the gallery image history.

        Full-resolution ``base64`` is stripped from entries that have a ``thumbnail``
        so that the grid payload is small (thumbnails only).  Entries without a
        thumbnail keep their ``base64`` as a fallback.  The full image can be
        fetched on demand via ``/api/gallery/image``.

        Query parameters:
            page: 1-based page number (default: 1)
            page_size: images per page (default: 96, max: 96)
            metadata_only: if "true"/"1", strip both ``thumbnail`` and ``base64`` so
                only lightweight metadata (gallery_id, timestamp, model) is returned.
                Use this to render the page skeleton quickly; images can then be
                fetched individually via ``/api/gallery/image``.
            model: if provided, only return images whose ``model`` field matches this
                value (case-insensitive).  Pass an empty string or omit to return all.
            safety: if provided, filter by safety flag.  Accepted values:
                ``sfw`` – exclude images flagged as NSFW or CSAM;
                ``nsfw`` – only images flagged as NSFW;
                ``csam`` – only images flagged as CSAM.
                Omit or pass an empty string to return all images regardless of safety flags.
        """
        try:
            page = max(1, int(request.rel_url.query.get("page", "1")))
        except ValueError:
            page = 1
        try:
            page_size = min(96, max(1, int(request.rel_url.query.get("page_size", "96"))))
        except ValueError:
            page_size = 96
        metadata_only = request.rel_url.query.get("metadata_only", "").lower() in ("1", "true", "yes")
        model_filter = request.rel_url.query.get("model", "").strip()
        model_filter_lower = model_filter.lower()
        safety_filter = request.rel_url.query.get("safety", "").strip().lower()
        if safety_filter not in ("", "sfw", "nsfw", "csam"):
            return web.json_response({"error": "Invalid safety filter"}, status=400)

        # Gallery is stored oldest-first (insertion order); serve newest-first to the UI.
        # Apply model and safety filters together to avoid multiple passes.
        def _entry_matches(e: dict[str, Any]) -> bool:
            if model_filter_lower and (e.get("model") or "").lower() != model_filter_lower:
                return False
            if safety_filter == "sfw" and (e.get("is_nsfw") or e.get("is_csam")):
                return False
            if safety_filter == "nsfw" and not e.get("is_nsfw"):
                return False
            if safety_filter == "csam" and not e.get("is_csam"):
                return False
            return True

        if model_filter_lower or safety_filter:
            images_reversed: list[dict[str, Any]] = list(reversed([e for e in self._gallery_dict.values() if _entry_matches(e)]))
            total = len(images_reversed)
        else:
            total = len(self._gallery_dict)
            images_reversed = list(reversed(self._gallery_dict.values()))

        total_pages = max(1, math.ceil(total / page_size))
        page = min(page, total_pages)
        start = (page - 1) * page_size
        page_images = images_reversed[start : start + page_size]

        if metadata_only:
            # Strip all image data so only lightweight metadata is returned.
            # The UI uses this to render the page skeleton immediately, then
            # fetches each thumbnail individually via /api/gallery/image.
            page_images = [{k: v for k, v in dict(entry).items() if k not in ("base64", "thumbnail")} for entry in page_images]
        else:
            # Strip the full-resolution base64 from entries that already have a thumbnail.
            # This drastically reduces the response payload for the gallery grid view.
            # Entries without a thumbnail keep their base64 as a display fallback.
            # All entries are copied so the originals in _gallery_dict are not mutated.
            # Thumbnails evicted from memory (to bound RAM) are fetched back from the
            # gallery DB in a single batched query for the page.
            missing_ids = [
                e["gallery_id"] for e in page_images if not e.get("thumbnail") and not e.get("base64")
            ]
            db_thumbs = await asyncio.get_running_loop().run_in_executor(
                None,
                self._fetch_gallery_thumbnails_from_db,
                missing_ids,
            )
            rendered: list[dict[str, Any]] = []
            for entry in page_images:
                if entry.get("thumbnail"):
                    rendered.append({k: v for k, v in entry.items() if k != "base64"})
                    continue
                copied = dict(entry)
                db_thumb = db_thumbs.get(copied.get("gallery_id"))
                if db_thumb:
                    copied["thumbnail"] = db_thumb
                rendered.append(copied)
            page_images = rendered

        return web.json_response(
            {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "images": page_images,
            },
        )

    # Gallery images are immutable once generated: a given gallery_id's bytes never change (new
    # images always get a new, never-reused id — see the "gallery_id must remain stable" test in
    # tests/test_webui.py), so the browser can cache them forever without ever revalidating. This
    # is what makes revisiting a page, reopening the overlay, or even a full reload of the app
    # near-instant for anything already seen, instead of re-fetching from the server every time.
    _GALLERY_IMAGE_CACHE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}

    async def _resolve_gallery_thumbnail_bytes(self, gallery_id: int) -> tuple[bytes, str] | None:
        """Return (raw_bytes, content_type) for a gallery thumbnail, or None if not found.

        Falls back to the full-resolution image (as PNG) when no thumbnail is available (e.g.
        Pillow not installed), and to the gallery DB when the in-memory copy was evicted to
        bound RAM.
        """
        entry = self._gallery_dict.get(gallery_id)
        thumbnail_b64 = entry.get("thumbnail") if entry else None
        if not thumbnail_b64:
            db_thumbs = await asyncio.get_running_loop().run_in_executor(
                None, self._fetch_gallery_thumbnails_from_db, [gallery_id],
            )
            thumbnail_b64 = db_thumbs.get(gallery_id)
        if thumbnail_b64:
            return base64.b64decode(thumbnail_b64), "image/jpeg"

        fullres_b64 = await self._resolve_gallery_fullres_b64(gallery_id, entry)
        if fullres_b64:
            return base64.b64decode(fullres_b64), "image/png"
        return None

    async def _resolve_gallery_fullres_b64(
        self,
        gallery_id: int,
        entry: dict[str, Any] | None = None,
    ) -> str | None:
        """Return the full-resolution base64 for a gallery id, falling back to the DB."""
        if entry is None:
            entry = self._gallery_dict.get(gallery_id)
        if entry and entry.get("base64"):
            return entry["base64"]
        return await asyncio.get_running_loop().run_in_executor(
            None, self._fetch_gallery_base64_from_db, gallery_id,
        )

    def _parse_gallery_id_from_match_info(self, request: web.Request) -> int:
        try:
            return int(request.match_info["gallery_id"])
        except ValueError:
            raise web.HTTPBadRequest(reason="Invalid gallery_id") from None

    async def _handle_gallery_thumb_binary(self, request: web.Request) -> web.Response:
        """Serve a single gallery thumbnail as raw image bytes (JPEG, or PNG fallback).

        Used directly as an ``<img src="...">`` by the gallery grid so the browser's native
        image loading, native lazy-loading (``loading="lazy"``), and HTTP cache all apply with
        no custom client-side JS — revisiting a page (or a full reload) is served from the
        browser's own cache instead of re-fetching from the server.
        """
        gallery_id = self._parse_gallery_id_from_match_info(request)
        resolved = await self._resolve_gallery_thumbnail_bytes(gallery_id)
        if resolved is None:
            raise web.HTTPNotFound(reason="Gallery image not found")
        image_bytes, content_type = resolved
        return web.Response(
            body=image_bytes,
            content_type=content_type,
            headers=self._GALLERY_IMAGE_CACHE_HEADERS,
        )

    async def _handle_gallery_full_binary(self, request: web.Request) -> web.Response:
        """Serve a single gallery image at full resolution as raw PNG bytes.

        Used directly as an ``<img src="...">`` by the overlay viewer for the same reason as
        _handle_gallery_thumb_binary: native loading and HTTP caching instead of a JSON+base64
        fetch, so reopening an already-viewed image is instant.
        """
        gallery_id = self._parse_gallery_id_from_match_info(request)
        fullres_b64 = await self._resolve_gallery_fullres_b64(gallery_id)
        if fullres_b64 is None:
            raise web.HTTPNotFound(reason="Gallery image not found")
        return web.Response(
            body=base64.b64decode(fullres_b64),
            content_type="image/png",
            headers=self._GALLERY_IMAGE_CACHE_HEADERS,
        )

    async def _handle_gallery_models(self, request: web.Request) -> web.Response:
        """Return sorted model names with image counts and the overall gallery total.

        The model list is alphabetically sorted and excludes entries where ``model``
        is ``None`` or an empty string. ``total`` includes all gallery entries,
        including those without a model value.

        Returns:
            JSON object with:
            - ``total`` (int): total number of gallery entries
            - ``models``: a sorted list of objects with ``name`` (str) and
              ``count`` (int) keys.
        """
        counts: dict[str, int] = {}
        for entry in self._gallery_dict.values():
            model = entry.get("model")
            if model:
                counts[model] = counts.get(model, 0) + 1
        return web.json_response({
            "total": len(self._gallery_dict),
            "models": [{"name": m, "count": c} for m, c in sorted(counts.items())],
        })

    async def _handle_gallery_safety(self, request: web.Request) -> web.Response:
        """Return per-safety-category image counts for the gallery filter dropdown.

        Returns:
            JSON object with:
            - ``total`` (int): total number of gallery entries
            - ``sfw`` (int): entries that are neither NSFW nor CSAM
            - ``nsfw`` (int): entries flagged as NSFW
            - ``csam`` (int): entries flagged as CSAM
        """
        total = len(self._gallery_dict)
        nsfw_count = 0
        csam_count = 0
        for entry in self._gallery_dict.values():
            if entry.get("is_nsfw"):
                nsfw_count += 1
            if entry.get("is_csam"):
                csam_count += 1
        sfw_count = sum(
            1 for e in self._gallery_dict.values() if not e.get("is_nsfw") and not e.get("is_csam")
        )
        return web.json_response({
            "total": total,
            "sfw": sfw_count,
            "nsfw": nsfw_count,
            "csam": csam_count,
        })

    async def _handle_gallery_last_batch(self, request: web.Request) -> web.Response:
        """Return full-resolution images from the most recent gallery batch.

        A "batch" is all images sharing the same timestamp (within a 2-second window),
        i.e. all outputs from a single job. Used by the overview page to pre-fill the
        Last Result container when no image has been generated in the current session.
        """
        if not self._gallery_dict:
            return web.json_response({"images": []})
        latest_ts = max(entry.get("timestamp", 0) for entry in self._gallery_dict.values())
        batch = sorted(
            (e for e in self._gallery_dict.values() if abs(e.get("timestamp", 0) - latest_ts) < 2.0),
            key=lambda e: e.get("gallery_id", 0),
        )
        result = []
        for e in batch:
            # Full-resolution base64 may have been evicted from memory to bound RAM; it is
            # persisted in the gallery DB, so fetch it back on demand when missing.
            b64 = e.get("base64")
            if not b64:
                b64 = await asyncio.get_running_loop().run_in_executor(
                    None,
                    self._fetch_gallery_base64_from_db,
                    e.get("gallery_id"),
                )
            result.append(
                {
                    "base64": b64 or "",
                    "timestamp": e.get("timestamp", 0),
                    "model": e.get("model", ""),
                    "is_nsfw": e.get("is_nsfw", False),
                    "is_csam": e.get("is_csam", False),
                },
            )
        return web.json_response({"images": result})

    async def _handle_gallery_image(self, request: web.Request) -> web.Response:
        """Return a single gallery image by its stable ``gallery_id``.

        Used by the overlay viewer to lazily fetch full-resolution images only when
        the user actually opens them, and by the gallery grid to progressively load
        thumbnails one by one after the page skeleton is rendered.

        Query parameters:
            id: stable ``gallery_id`` assigned when the image was added (required)
            thumbnail_only: if "true"/"1", strip the full-resolution ``base64`` from
                the response and return only the ``thumbnail`` (plus metadata).  Use
                this when loading the gallery grid to avoid transferring large
                full-resolution images for every grid item.
        """
        try:
            gallery_id = int(request.rel_url.query.get("id", ""))
        except ValueError:
            raise web.HTTPBadRequest(reason="Invalid or missing id parameter") from None
        thumbnail_only = request.rel_url.query.get("thumbnail_only", "").lower() in ("1", "true", "yes")

        entry = self._gallery_dict.get(gallery_id)
        if entry is None:
            raise web.HTTPNotFound(reason="Gallery image not found")
        _THUMB_CACHE_HEADERS = {"Cache-Control": "public, max-age=86400, immutable"}
        if thumbnail_only and entry.get("thumbnail"):
            # When a thumbnail is available, omit the full-resolution base64
            # to keep the payload small for the gallery grid.
            return web.json_response(
                {k: v for k, v in entry.items() if k != "base64"},
                headers=_THUMB_CACHE_HEADERS,
            )
        if thumbnail_only and entry.get("base64"):
            # Pillow not installed — return full PNG as fallback but still allow caching.
            return web.json_response(
                {k: v for k, v in entry.items()},
                headers=_THUMB_CACHE_HEADERS,
            )
        if thumbnail_only:
            # The thumbnail was evicted from memory to bound RAM usage — fetch it back
            # from the gallery DB on demand.
            db_thumbs = await asyncio.get_running_loop().run_in_executor(
                None,
                self._fetch_gallery_thumbnails_from_db,
                [gallery_id],
            )
            db_thumb = db_thumbs.get(gallery_id)
            if db_thumb:
                return web.json_response(
                    {**{k: v for k, v in entry.items() if k != "base64"}, "thumbnail": db_thumb},
                    headers=_THUMB_CACHE_HEADERS,
                )
            # No thumbnail anywhere — fall through and serve the full image instead.
        # Full image requested (overlay viewer) — no cache header so full-res is always fresh.
        # The full-resolution base64 may have been evicted from memory to bound RAM (it is
        # persisted in the gallery DB); fetch it back on demand when it is not in memory.
        if not entry.get("base64"):
            full_b64 = await asyncio.get_running_loop().run_in_executor(
                None,
                self._fetch_gallery_base64_from_db,
                gallery_id,
            )
            if full_b64:
                entry = {**entry, "base64": full_b64}
        return web.json_response(entry)

    async def _handle_get_settings(self, request: web.Request) -> web.Response:
        """Return all runtime-configurable settings and their current values.

        Response shape::

            {
                "settings": {
                    "<key>": <value>,
                    ...
                }
            }

        The ``settings`` dict includes keys defined in ``_SETTINGS_SPEC`` plus
        the live int_auto keys (``max_queue_size``, ``max_active_models``,
        ``queue_size_auto``, ``max_active_models_auto``) when present in
        ``status_data``. Other keys whose values are not yet available
        (e.g. before the first status push from the process manager) are omitted
        rather than returned as ``null``.
        """
        visible: dict[str, Any] = {k: v for k, v in self._settings_data.items() if k in _SETTINGS_SPEC}
        # Inject the Web UI's own URL derived from the incoming request origin so
        # that remote browsers receive a usable address rather than localhost.
        host = request.host or f"localhost:{self.port}"
        visible["webui_url"] = f"{request.scheme}://{host}"
        # Piggy-back the int_auto live values from status_data so the Settings
        # page can render the correct initial values without waiting for the
        # first status poll.
        for k in ("max_queue_size", "max_active_models", "queue_size_auto", "max_active_models_auto"):
            if k in self.status_data:
                visible[k] = self.status_data[k]
        return web.json_response({"settings": visible})

    async def _handle_set_setting(self, request: web.Request) -> web.Response:
        """Apply a single runtime setting change requested from the UI.

        Expected JSON body::

            {"key": "<setting_name>", "value": <new_value>}

        Returns:
            200 with ``{"key": ..., "value": ...}`` on success.
            400 on missing/invalid input.
            503 if no settings callback has been registered.
            500 on internal error.
        """
        try:
            body = await request.json()
        except (ValueError, TypeError, aiohttp.ContentTypeError) as exc:
            return web.json_response({"error": f"Invalid JSON body: {exc}"}, status=400)

        key = body.get("key")
        if not isinstance(key, str) or not key:
            return web.json_response({"error": "Field 'key' must be a non-empty string"}, status=400)

        raw_value = body.get("value")
        if key == "queue_size_auto":
            if not isinstance(raw_value, bool):
                return web.json_response({"error": "Field 'value' must be a boolean for setting 'queue_size_auto'"}, status=400)
            if self._set_queue_size_auto_mode_callback is None:
                return web.json_response({"error": "Auto queue size mode is not available"}, status=503)
            try:
                self._set_queue_size_auto_mode_callback(raw_value)
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Error setting queue size auto mode={raw_value}: {exc}")
                return web.json_response({"error": f"Internal error: {type(exc).__name__}"}, status=500)
            self.status_data["queue_size_auto"] = raw_value
            return web.json_response({"key": key, "value": raw_value})

        if key == "max_queue_size":
            if not isinstance(raw_value, int) or isinstance(raw_value, bool):
                return web.json_response({"error": "Field 'value' must be a non-negative integer for setting 'max_queue_size'"}, status=400)
            if raw_value < 0:
                return web.json_response({"error": "Value for 'max_queue_size' must be >= 0"}, status=400)
            if self._set_max_queue_size_callback is None:
                return web.json_response({"error": "Setting max queue size is not available"}, status=503)
            try:
                self._set_max_queue_size_callback(raw_value)
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Error setting max queue size={raw_value}: {exc}")
                return web.json_response({"error": f"Internal error: {type(exc).__name__}"}, status=500)
            self.status_data["max_queue_size"] = raw_value
            self.status_data["queue_size_auto"] = False
            return web.json_response({"key": key, "value": raw_value})

        if key == "max_active_models_auto":
            if not isinstance(raw_value, bool):
                return web.json_response(
                    {"error": "Field 'value' must be a boolean for setting 'max_active_models_auto'"},
                    status=400,
                )
            if self._set_max_active_models_auto_mode_callback is None:
                return web.json_response({"error": "Auto max active models mode is not available"}, status=503)
            try:
                self._set_max_active_models_auto_mode_callback(raw_value)
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Error setting max active models auto mode={raw_value}: {exc}")
                return web.json_response({"error": f"Internal error: {type(exc).__name__}"}, status=500)
            self.status_data["max_active_models_auto"] = raw_value
            return web.json_response({"key": key, "value": raw_value})

        if key == "max_active_models":
            if not isinstance(raw_value, int) or isinstance(raw_value, bool):
                return web.json_response(
                    {"error": "Field 'value' must be a positive integer for setting 'max_active_models'"},
                    status=400,
                )
            if raw_value < 1:
                return web.json_response({"error": "Value for 'max_active_models' must be >= 1"}, status=400)
            if self._set_max_active_models_callback is None:
                return web.json_response({"error": "Setting max active models is not available"}, status=503)
            try:
                self._set_max_active_models_callback(raw_value)
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Error setting max active models={raw_value}: {exc}")
                return web.json_response({"error": f"Internal error: {type(exc).__name__}"}, status=500)
            self.status_data["max_active_models"] = raw_value
            self.status_data["max_active_models_auto"] = False
            return web.json_response({"key": key, "value": raw_value})

        if key not in _SETTINGS_SPEC:
            return web.json_response({"error": f"Unknown or non-configurable setting: '{key}'"}, status=400)

        spec = _SETTINGS_SPEC[key]
        if spec.get("readonly"):
            return web.json_response({"error": f"Setting '{key}' is read-only"}, status=400)

        expected_type = spec["type"]

        # Type coercion and validation
        if expected_type is bool:
            if not isinstance(raw_value, bool):
                return web.json_response({"error": f"Field 'value' must be a boolean for setting '{key}'"}, status=400)
            value: Any = raw_value
        elif expected_type is int:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                return web.json_response({"error": f"Field 'value' must be a number for setting '{key}'"}, status=400)
            if isinstance(raw_value, float) and not raw_value.is_integer():
                return web.json_response({"error": f"Field 'value' must be an integer for setting '{key}'"}, status=400)
            value = int(raw_value)
            min_v = spec.get("min")
            max_v = spec.get("max")
            if min_v is not None and value < min_v:
                return web.json_response({"error": f"Value for '{key}' must be >= {min_v}"}, status=400)
            if max_v is not None and value > max_v:
                return web.json_response({"error": f"Value for '{key}' must be <= {max_v}"}, status=400)
        elif expected_type is float:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                return web.json_response({"error": f"Field 'value' must be a number for setting '{key}'"}, status=400)
            value = float(raw_value)
            min_v = spec.get("min")
            max_v = spec.get("max")
            if min_v is not None and value < min_v:
                return web.json_response({"error": f"Value for '{key}' must be >= {min_v}"}, status=400)
            if max_v is not None and value > max_v:
                return web.json_response({"error": f"Value for '{key}' must be <= {max_v}"}, status=400)
        elif expected_type is list:
            if not isinstance(raw_value, list):
                return web.json_response({"error": f"Field 'value' must be a list for setting '{key}'"}, status=400)
            _FILTER_GROUP_KEYS = {
                "positive_prompt_append", "positive_prompt_remove", "positive_prompt_replace",
                "positive_prompt_conditional_add",
                "negative_prompt_append", "negative_prompt_remove", "negative_prompt_replace",
                "negative_prompt_conditional_add",
                "prompt_swap",
            }
            if key in _FILTER_GROUP_KEYS:
                # Expect a list of group dicts: [{name, enabled, entries}, ...]
                cleaned_groups = []
                for item in raw_value:
                    if not isinstance(item, dict):
                        continue
                    cleaned_groups.append({
                        "name": str(item.get("name", "")),
                        "enabled": bool(item.get("enabled", True)),
                        "entries": [
                            str(e) for e in item.get("entries", [])
                            if isinstance(e, str) and e.strip()
                        ],
                    })
                value = cleaned_groups
            else:
                if not all(isinstance(s, str) for s in raw_value):
                    return web.json_response({"error": f"All items in '{key}' must be strings"}, status=400)
                value = [s for s in raw_value if s.strip()]
        else:
            return web.json_response({"error": "Internal configuration error"}, status=500)

        if self._set_setting_callback is None:
            return web.json_response({"error": "Settings API is not available"}, status=503)

        try:
            self._set_setting_callback(key, value)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Error applying setting {key}={value!r}: {exc}")
            return web.json_response({"error": f"Internal error: {type(exc).__name__}"}, status=500)

        self._settings_data[key] = value
        self._persisted_settings[key] = value
        self._save_persisted_settings()
        return web.json_response({"key": key, "value": value})

    async def _handle_get_models(self, request: web.Request) -> web.Response:
        """Return the current enabled and disabled model lists.

        Response shape::

            {
                "enabled": ["model_a", "model_b", ...],
                "disabled": ["model_c", ...]
            }
        """
        return web.json_response(self._models_data)

    async def _handle_toggle_model(self, request: web.Request) -> web.Response:
        """Toggle a model between enabled and disabled state.

        Expected JSON body::

            {"model": "<model_name>", "enabled": true|false}

        Returns:
            200 with ``{"model": ..., "enabled": ...}`` on success.
            400 on missing/invalid input.
            503 if no toggle callback has been registered.
            500 on internal error.
        """
        try:
            body = await request.json()
        except (ValueError, TypeError, aiohttp.ContentTypeError) as exc:
            return web.json_response({"error": f"Invalid JSON body: {exc}"}, status=400)

        model = body.get("model")
        if not isinstance(model, str) or not model:
            return web.json_response({"error": "Field 'model' must be a non-empty string"}, status=400)

        enabled = body.get("enabled")
        if not isinstance(enabled, bool):
            return web.json_response({"error": "Field 'enabled' must be a boolean"}, status=400)

        # Verify the model is known (in either list)
        all_models = self._models_data["enabled"] + self._models_data["disabled"]
        if model not in all_models:
            return web.json_response({"error": f"Unknown model: '{model}'"}, status=400)

        if self._toggle_model_callback is None:
            return web.json_response({"error": "Models API is not available"}, status=503)

        try:
            self._toggle_model_callback(model, enabled)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Error toggling model '{model}' enabled={enabled}: {exc}")
            return web.json_response({"error": f"Internal error: {type(exc).__name__}"}, status=500)

        # Update local models data immediately so the next GET /api/models reflects
        # the new state without waiting for an external update_models_data() call.
        enabled_list = list(self._models_data["enabled"])
        disabled_list = list(self._models_data["disabled"])
        if enabled:
            if model in disabled_list:
                disabled_list.remove(model)
            if model not in enabled_list:
                enabled_list.append(model)
        else:
            if model in enabled_list:
                enabled_list.remove(model)
            if model not in disabled_list:
                disabled_list.append(model)
        self.update_models_data(enabled_list, disabled_list)

        return web.json_response({"model": model, "enabled": enabled})

    async def _handle_reset_stats(self, request: web.Request) -> web.Response:
        """Reset session overview statistics (display reads from zero after this call)."""
        rb = {
            "jobs_popped": int(self.status_data.get("jobs_popped", 0)),
            "jobs_completed": int(self.status_data.get("jobs_completed", 0)),
            "jobs_faulted": int(self.status_data.get("jobs_faulted", 0)),
            "processes_recovered": int(self.status_data.get("processes_recovered", 0)),
            "kudos_earned_session": float(self.status_data.get("kudos_earned_session", 0.0)),
            "time_without_jobs": float(self.status_data.get("time_without_jobs", 0.0)),
        }
        self.status_data["stats_reset_baseline"] = rb
        if self._stats_db_path is not None:
            try:
                with sqlite3.connect(self._stats_db_path) as conn:
                    conn.execute(
                        "UPDATE session_overview SET reset_baseline_json = ? WHERE id = 1",
                        (json.dumps(rb),),
                    )
                    conn.commit()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not persist reset baseline: {exc}")
        return web.json_response({"reset": True})

    async def _handle_reset_database_progress(self, request: web.Request) -> web.Response:
        """Return the current database reset progress (0-100) or null if idle."""
        return web.json_response(
            {
                "progress": self._db_reset_progress,
                "done": self._db_reset_progress == 100,
                "error": self._db_reset_error,
            }
        )

    def _do_reset_errors_db(self) -> None:
        """Blocking: wipe the errors DB. Runs in a thread executor."""
        assert self._errors_db_path is not None
        with sqlite3.connect(self._errors_db_path, timeout=30) as conn:
            conn.execute("DELETE FROM errors_log")
            conn.commit()

    def _do_reset_stats_db(self) -> None:
        """Blocking: wipe the stats DB. Runs in a thread executor."""
        assert self._stats_db_path is not None
        with sqlite3.connect(self._stats_db_path, timeout=30) as conn:
            conn.execute("DELETE FROM stats_snapshots")
            conn.execute("DELETE FROM horde_snapshots")
            conn.execute("DELETE FROM session_overview")
            conn.execute("DELETE FROM session_aggregates")
            conn.commit()

    def _do_reset_gallery_db(self) -> None:
        """Blocking: wipe the gallery DB. Runs in a thread executor."""
        assert self._gallery_db_path is not None
        with sqlite3.connect(self._gallery_db_path, timeout=30) as conn:
            conn.execute("DELETE FROM gallery_images")
            conn.commit()

    async def _run_reset_database(self) -> None:
        """Background task: reset all databases and track progress (0-100)."""
        loop = asyncio.get_event_loop()
        self._db_reset_error = None
        failed: list[str] = []

        # 10% — in-memory state cleared (done synchronously before this task starts)
        self._db_reset_progress = 10

        if self._errors_db_path is not None:
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._do_reset_errors_db),
                    timeout=60,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not reset errors DB: {exc}")
                failed.append("errors")
        self._db_reset_progress = 40

        if self._stats_db_path is not None:
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._do_reset_stats_db),
                    timeout=60,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not reset stats DB: {exc}")
                failed.append("stats")
        self._db_reset_progress = 70

        if self._gallery_db_path is not None:
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._do_reset_gallery_db),
                    timeout=60,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not reset gallery DB: {exc}")
                failed.append("gallery")
        self._db_reset_progress = 100

        if failed:
            self._db_reset_error = f"Failed to reset: {', '.join(failed)}"
            logger.warning(f"WebUI database reset completed with errors: {self._db_reset_error}")
        else:
            logger.info("WebUI databases reset via settings page.")

    async def _handle_reset_database(self, request: web.Request) -> web.Response:
        """Wipe all persisted history across the SQLite databases.

        Clears every table in the errors, stats and gallery databases and
        resets the matching in-memory collections so the UI reflects an empty
        state immediately.  Runtime settings (``webui_settings.json``) are left
        untouched — use "Reset All" on the settings page for those.

        Returns immediately with ``{"started": true}`` and tracks progress via
        GET /api/reset-database/progress (0-100).
        """
        if self._db_reset_task is not None and not self._db_reset_task.done():
            return web.json_response({"started": False, "error": "Reset already in progress"}, status=409)

        # Reset in-memory state immediately so the next /api/status poll is empty.
        self.status_data["errors_history"] = []
        self._live_errors_history = []
        self._persisted_errors = []
        self._gallery_dict.clear()
        self._next_gallery_id = 0
        self.status_data["images_count"] = 0
        self._stats_snapshots.clear()
        self._last_stats_snapshot_time = 0.0
        self._horde_snapshots.clear()
        self._persisted_horde_snapshots = []
        self._persisted_reset_baseline = {}
        self.status_data["stats_reset_baseline"] = {}
        for _agg_key in (
            "images_per_model",
            "failed_jobs_per_model",
            "faulted_jobs_per_phase",
            "avg_time_per_job_state",
            "max_time_per_job_state",
            "avg_time_per_step_per_model",
            "max_time_per_step_per_model",
            "avg_time_per_job_per_model",
            "max_time_per_job_per_model",
        ):
            self.status_data[_agg_key] = {}
        self._aggregate_baseline = {"images_per_model": {}, "failed_jobs_per_model": {}, "faulted_jobs_per_phase": {}}
        self._persisted_aggregates = {}

        self._db_reset_progress = 0
        self._db_reset_error = None
        self._db_reset_task = asyncio.create_task(self._run_reset_database())
        return web.json_response({"started": True})

    async def _handle_horde_snapshots(self, request: web.Request) -> web.Response:
        """Return server-accumulated horde network performance snapshots.

        Accepts the same optional ``window`` query parameter as /api/stats (seconds, or
        ``"all"``/absent) and downsamples the result to roughly _CHART_MAX_POINTS points
        server-side, for the same reason: a multi-day "All" view has no need for every
        raw 30-second sample once there are more of them than chart pixels to plot on.
        """
        snapshots = _windowed_snapshots(list(self._horde_snapshots), request.query.get("window"))
        snapshots = _downsample_series(snapshots, _CHART_MAX_POINTS)
        return web.json_response({"snapshots": snapshots})

    async def _handle_horde_modes(self, request: web.Request) -> web.Response:
        """Return the last server-polled aihorde.net maintenance/invite-only mode flags.

        Served from a cache refreshed by the background _poll_horde_network task rather
        than fetched live per-request, and proxied through our own origin rather than
        called directly by the browser -- aihorde.net's CORS policy does not reliably
        allow the preflight for the custom Client-Agent header from arbitrary worker
        origins, which made the direct browser-side fetch fail in real deployments.
        """
        return web.json_response(self._horde_modes)

    _HORDE_POLL_INTERVAL: float = 30.0
    _HORDE_MIN_SERVER_SNAPS: int = 360  # floor: at least 3 hours of history

    @property
    def _horde_max_server_snaps(self) -> int:
        """Max horde snapshots to keep in memory and load from DB.

        Scales with data_retention_days so the full window range (30m / 2h /
        6h / All) is available up to the configured retention period, bounded
        by an absolute ceiling — a multi-year retention setting would otherwise
        translate into millions of in-memory snapshot dicts.
        """
        scaled = int(self._data_retention_days * 86400 / self._HORDE_POLL_INTERVAL)
        return max(self._HORDE_MIN_SERVER_SNAPS, min(scaled, _HORDE_MAX_SERVER_SNAPS_CEILING))

    _HORDE_MODES_POLL_INTERVAL: float = 300.0

    async def _poll_horde_network(self) -> None:
        """Background task: poll aihorde.net every 30 s and accumulate performance snapshots.

        Also refreshes maintenance/invite-only mode flags every _HORDE_MODES_POLL_INTERVAL
        seconds. Both are polled server-side (rather than by the browser) because the
        browser calling aihorde.net directly requires a CORS preflight for the custom
        Client-Agent header, which aihorde.net does not reliably grant for arbitrary
        worker origins -- a server-to-server request has no such restriction.
        """
        await asyncio.sleep(5)  # brief startup delay
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(
                        "https://aihorde.net/api/v2/status/performance",
                        headers={"Client-Agent": "horde-worker-regen:0:unknown"},
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            snap = {
                                "t": int(time.time()),
                                "workers": data.get("worker_count", 0),
                                "threads": data.get("thread_count", 0),
                                "queued_req": data.get("queued_requests", 0),
                                "queued_mps": data.get("queued_megapixelsteps", 0),
                                "past_min_mps": data.get("past_minute_megapixelsteps", 0),
                            }
                            self._horde_snapshots.append(snap)
                            if self._stats_db_path is not None:
                                try:
                                    with sqlite3.connect(self._stats_db_path) as _conn:
                                        _conn.execute(
                                            "INSERT INTO horde_snapshots (snapshot_json, timestamp) VALUES (?, ?)",
                                            (json.dumps(snap), snap["t"]),
                                        )
                                        _conn.commit()
                                except Exception:
                                    logger.debug("Failed to persist horde snapshot to database", exc_info=True)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("Unexpected error in horde stats polling loop", exc_info=True)

                now = time.time()
                if now - self._last_horde_modes_fetch >= self._HORDE_MODES_POLL_INTERVAL:
                    self._last_horde_modes_fetch = now
                    try:
                        async with session.get(
                            "https://aihorde.net/api/v2/status/modes",
                            headers={"Client-Agent": "horde-worker-regen:0:unknown"},
                            timeout=aiohttp.ClientTimeout(total=15),
                        ) as resp:
                            if resp.status == 200:
                                self._horde_modes = await resp.json()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.debug("Unexpected error fetching horde mode flags", exc_info=True)

                await asyncio.sleep(self._HORDE_POLL_INTERVAL)

    async def _handle_restart_program(self, request: web.Request) -> web.Response:
        """Handle a request to restart the worker program."""
        if self._restart_program_callback is None:
            return web.json_response({"error": "Restart API is not available"}, status=503)

        try:
            self._restart_program_callback()
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Error requesting program restart: {exc}")
            return web.json_response({"error": f"Internal error: {type(exc).__name__}"}, status=500)

        return web.json_response({"restarting": True})

    def _fetch_gallery_base64_from_db(self, gallery_id: int | None) -> str | None:
        """Return the full-resolution base64 PNG for a gallery image from the database.

        In-memory gallery entries keep only a small thumbnail once the full-resolution
        image has been persisted (see :meth:`add_gallery_image`), so the overlay viewer
        and last-batch endpoints fetch the full image back on demand. Returns ``None``
        when no database is configured or the row cannot be read.
        """
        if self._gallery_db_path is None or gallery_id is None:
            return None
        try:
            with sqlite3.connect(self._gallery_db_path) as conn:
                row = conn.execute(
                    "SELECT base64_data FROM gallery_images WHERE gallery_id = ? ORDER BY id DESC LIMIT 1",
                    (gallery_id,),
                ).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load gallery image {gallery_id} from database: {exc}")
            return None
        if row and row[0]:
            return row[0]
        return None

    def _fetch_gallery_thumbnails_from_db(self, gallery_ids: list[int]) -> dict[int, str]:
        """Return thumbnails for the given gallery ids from the database.

        In-memory entries older than the ``_MAX_THUMBNAILS_IN_MEMORY`` window keep only
        metadata; the gallery grid fetches their thumbnails back on demand through this
        helper. Returns a (possibly partial) ``gallery_id -> thumbnail`` mapping; empty
        when no database is configured or the rows cannot be read.
        """
        if self._gallery_db_path is None or not gallery_ids:
            return {}
        try:
            with sqlite3.connect(self._gallery_db_path) as conn:
                placeholders = ",".join("?" for _ in gallery_ids)
                rows = conn.execute(
                    f"SELECT gallery_id, thumbnail FROM gallery_images WHERE gallery_id IN ({placeholders})",
                    tuple(gallery_ids),
                ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not load gallery thumbnails from database: {exc}")
            return {}
        return {int(row[0]): row[1] for row in rows if row[1]}

    def _enforce_gallery_memory_caps(self) -> None:
        """Bound the amount of gallery image data (and entries) held in RAM.

        - Thumbnails beyond the newest ``_MAX_THUMBNAILS_IN_MEMORY`` are dropped when a
          gallery DB exists (they are re-fetched from it on demand).
        - Full-resolution base64 payloads — in memory only when the DB persist failed or
          no DB is configured — are capped at the newest ``_MAX_FULLRES_IN_MEMORY``.
        - The total entry count is capped (``_MAX_GALLERY_ENTRIES_WITH_DB`` /
          ``_MAX_GALLERY_ENTRIES_NO_DB``), evicting the oldest entries from RAM.
        """
        db_available = self._gallery_db_path is not None
        thumbs_seen = 0
        fullres_seen = 0
        for entry in reversed(self._gallery_dict.values()):
            if db_available and "thumbnail" in entry:
                thumbs_seen += 1
                if thumbs_seen > _MAX_THUMBNAILS_IN_MEMORY:
                    entry.pop("thumbnail", None)
            if "base64" in entry:
                fullres_seen += 1
                if fullres_seen > _MAX_FULLRES_IN_MEMORY:
                    entry.pop("base64", None)

        max_entries = _MAX_GALLERY_ENTRIES_WITH_DB if db_available else _MAX_GALLERY_ENTRIES_NO_DB
        if len(self._gallery_dict) > max_entries:
            excess = len(self._gallery_dict) - max_entries
            for gid in list(self._gallery_dict.keys())[:excess]:
                self._gallery_dict.pop(gid, None)
            self.status_data["images_count"] = len(self._gallery_dict)

    def add_gallery_image(self, image_entry: dict[str, Any]) -> None:
        """Append one image entry to the gallery history.

        A small JPEG thumbnail is generated and stored under the ``thumbnail`` key so
        that the gallery grid can load much faster than serving the full-resolution PNG.
        The original full-resolution ``base64`` value is preserved for the overlay viewer.

        Args:
            image_entry: dict with keys ``base64``, ``timestamp``, and ``model``.
        """
        entry = dict(image_entry)
        entry["gallery_id"] = self._next_gallery_id
        self._next_gallery_id += 1
        if _PIL_AVAILABLE and entry.get("base64"):
            try:
                raw = base64.b64decode(entry["base64"])
                with io.BytesIO(raw) as img_bytes, _PILImage.open(img_bytes) as img:
                    img.thumbnail((_THUMBNAIL_MAX_PX, _THUMBNAIL_MAX_PX), _PILImage.LANCZOS)
                    with io.BytesIO() as buf:
                        img.convert("RGB").save(buf, format="JPEG", quality=85)
                        entry["thumbnail"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to generate gallery thumbnail: {}", exc)
        self._gallery_dict[entry["gallery_id"]] = entry
        self.status_data["images_count"] = len(self._gallery_dict)

        # Persist to database.
        persisted = False
        if self._gallery_db_path is not None:
            _known_cols = {"gallery_id", "timestamp", "model", "base64", "thumbnail", "is_nsfw", "is_csam"}
            extra = {k: v for k, v in entry.items() if k not in _known_cols}
            try:
                with sqlite3.connect(self._gallery_db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO gallery_images
                            (gallery_id, timestamp, model, base64_data, thumbnail,
                             is_nsfw, is_csam, extra_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            entry["gallery_id"],
                            entry.get("timestamp", time.time()),
                            entry.get("model"),
                            entry.get("base64"),
                            entry.get("thumbnail"),
                            1 if entry.get("is_nsfw") else 0,
                            1 if entry.get("is_csam") else 0,
                            json.dumps(extra) if extra else None,
                        ),
                    )
                    conn.commit()
                persisted = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not persist gallery image to database: {exc}")

        # Bound in-RAM growth: the full-resolution base64 PNG (several MB each) is the
        # dominant memory cost and the reason the worker can exhaust RAM and be OOM-killed
        # (taking down the whole container) after running for a while. Once the image is
        # safely persisted to the gallery DB, drop the full-resolution base64 from the
        # in-memory entry — even when no thumbnail could be generated, since
        # _handle_gallery_image falls back to fetching the full image from the DB on
        # demand. Only when the image could not be persisted is the base64 kept in memory
        # as the last available copy (bounded by _enforce_gallery_memory_caps below).
        if persisted and entry.get("base64") is not None:
            entry.pop("base64", None)

        self._enforce_gallery_memory_caps()

    def update_status(
        self,
        worker_name: str | None = None,
        horde_username: str | None = None,
        jobs_popped: int | None = None,
        jobs_queued: int | None = None,
        time_without_jobs: float | None = None,
        jobs_completed: int | None = None,
        jobs_faulted: int | None = None,
        processes_recovered: int | None = None,
        kudos_earned_session: float | None = None,
        kudos_per_hour: float | None = None,
        images_per_hour: float | None = None,
        current_job: dict[str, Any] | None = None,
        job_queue: list[dict[str, Any]] | None = None,
        max_queue_size: int | None = None,
        queue_size_auto: bool | None = None,
        processes: list[dict[str, Any]] | None = None,
        models_loaded: list[str] | None = None,
        max_active_models: int | None = None,
        max_active_models_auto: bool | None = None,
        ram_usage_mb: float | None = None,
        system_ram_usage_mb: float | None = None,
        total_ram_mb: float | None = None,
        vram_usage_mb: float | None = None,
        system_vram_usage_mb: float | None = None,
        total_vram_mb: float | None = None,
        cpu_usage_percent: float | None = None,
        cpu_cores_count: int | None = None,
        gpu_usage_percent: float | None = None,
        worker_gpu_percent: float | None = None,
        gpu_cores_count: int | None = None,
        container_cpu_percent: float | None = None,
        maintenance_mode: bool | None = None,
        job_pops_paused: bool | None = None,
        job_pops_pause_until: float | None | object = _UNSET,
        user_kudos_total: float | None = None,
        last_image_base64: list[str] | None = None,
        last_image_submission_timestamp: float | None = None,
        last_image_model: str | None = None,
        last_image_safety: list[dict] | None = None,
        console_logs: list[str] | None = None,
        faulted_jobs_history: list[dict[str, Any]] | None = None,
        errors_history: list[str] | None = None,
        user_details: dict[str, Any] | None = None,
        images_per_model: dict[str, int] | None = None,
        failed_jobs_per_model: dict[str, int] | None = None,
        faulted_jobs_per_phase: dict[str, int] | None = None,
        avg_time_per_job_state: dict[str, float] | None = None,
        max_time_per_job_state: dict[str, float] | None = None,
        avg_time_per_step_per_model: dict[str, float] | None = None,
        max_time_per_step_per_model: dict[str, float] | None = None,
        avg_time_per_job_per_model: dict[str, float] | None = None,
        max_time_per_job_per_model: dict[str, float] | None = None,
    ) -> None:
        """Update the status data for the web UI.

        Args:
            worker_name: The name of the worker
            horde_username: The horde username
            jobs_popped: Total number of jobs popped this session
            jobs_queued: Currently queued jobs count
            time_without_jobs: Total seconds spent with no active jobs this session
            jobs_completed: Total number of jobs completed this session
            jobs_faulted: Total number of jobs faulted this session
            processes_recovered: Total number of jobs recovered this session
            kudos_earned_session: Total kudos earned this session
            kudos_per_hour: Current kudos per hour rate
            images_per_hour: Current images generated per hour rate
            current_job: Information about the current job being processed
            job_queue: List of jobs in the queue
            max_queue_size: Maximum number of jobs that can be queued
            queue_size_auto: Whether queue size is being managed automatically
            processes: List of process information
            models_loaded: List of currently loaded models
            max_active_models: Maximum number of simultaneously active model slots
            max_active_models_auto: Whether max active models is being managed automatically
            ram_usage_mb: Worker processes RAM usage in MB (sum of all worker processes)
            system_ram_usage_mb: System-wide RAM currently in use in MB (all processes on the host)
            total_ram_mb: Total system RAM capacity in MB
            vram_usage_mb: VRAM usage in MB (worker processes - torch reserved memory)
            system_vram_usage_mb: System-wide VRAM currently in use in MB (all processes on the host)
            total_vram_mb: Total VRAM in MB
            cpu_usage_percent: CPU usage percentage
            cpu_cores_count: Number of CPU cores
            gpu_usage_percent: System-wide GPU SM utilisation percentage
            worker_gpu_percent: GPU SM utilisation percentage reported by the worker inference processes
            gpu_cores_count: Total detected NVIDIA CUDA cores across CUDA devices
            container_cpu_percent: CPU usage percentage of the worker process tree (container-level)
            maintenance_mode: Whether worker is in maintenance mode
            job_pops_paused: Whether new job pops are currently paused by the user
            job_pops_pause_until: Unix timestamp at which a timed pause will auto-expire (None = indefinite)
            user_kudos_total: Total kudos accumulated by the user
            last_image_base64: List of base64 encoded last generated images (supports batch jobs)
            last_image_submission_timestamp: Timestamp when the last image was submitted
            last_image_model: Model name used to generate the last image
            last_image_safety: Per-image safety flags (is_nsfw, is_csam) for the last images
            console_logs: Recent console log messages
            faulted_jobs_history: List of faulted jobs with details
            errors_history: List of recent error messages
            user_details: Extended user details from the Horde API (worker_count, trusted, moderator, etc.)
            images_per_model: Cumulative per-model image counts for the current session
            failed_jobs_per_model: Cumulative per-model failed job counts for the current session
            faulted_jobs_per_phase: Cumulative per-phase fault counts for the current session
            avg_time_per_job_state: Average time in seconds per job state for completed jobs this session
            max_time_per_job_state: Maximum time in seconds per job state for completed jobs this session
            avg_time_per_step_per_model: Average inference time per step (inference_time/steps) per model this session
            max_time_per_step_per_model: Maximum inference time per step per model this session
            avg_time_per_job_per_model: Average total job time per model this session
            max_time_per_job_per_model: Maximum total job time per model this session
        """
        if worker_name is not None:
            self.status_data["worker_name"] = worker_name
        if horde_username is not None:
            self.status_data["horde_username"] = horde_username
        _b = self._session_baseline
        if jobs_popped is not None:
            self.status_data["jobs_popped"] = _b.get("jobs_popped", 0) + jobs_popped
        if jobs_queued is not None:
            self.status_data["jobs_queued"] = jobs_queued
        if time_without_jobs is not None:
            self.status_data["time_without_jobs"] = _b.get("time_without_jobs", 0.0) + time_without_jobs
        if jobs_completed is not None:
            self.status_data["jobs_completed"] = _b.get("jobs_completed", 0) + jobs_completed
        if jobs_faulted is not None:
            self.status_data["jobs_faulted"] = _b.get("jobs_faulted", 0) + jobs_faulted
        if processes_recovered is not None:
            self.status_data["processes_recovered"] = _b.get("processes_recovered", 0) + processes_recovered
        if kudos_earned_session is not None:
            self.status_data["kudos_earned_session"] = _b.get("kudos_earned_session", 0.0) + kudos_earned_session
        if kudos_per_hour is not None:
            self.status_data["kudos_per_hour"] = kudos_per_hour
        if images_per_hour is not None:
            self.status_data["images_per_hour"] = images_per_hour
        # current_job is always updated unconditionally (unlike other optional fields) so that
        # passing None explicitly clears the displayed job once submission is complete.
        # update_webui_status() always passes this field, so None means "no active job".
        self.status_data["current_job"] = current_job
        if job_queue is not None:
            self.status_data["job_queue"] = job_queue
        if max_queue_size is not None:
            self.status_data["max_queue_size"] = max_queue_size
        if queue_size_auto is not None:
            self.status_data["queue_size_auto"] = queue_size_auto
        if processes is not None:
            self.status_data["processes"] = processes
        if models_loaded is not None:
            self.status_data["models_loaded"] = models_loaded
        if max_active_models is not None:
            self.status_data["max_active_models"] = max_active_models
        if max_active_models_auto is not None:
            self.status_data["max_active_models_auto"] = max_active_models_auto
        if ram_usage_mb is not None:
            self.status_data["ram_usage_mb"] = ram_usage_mb
        if system_ram_usage_mb is not None:
            self.status_data["system_ram_usage_mb"] = system_ram_usage_mb
        if total_ram_mb is not None:
            self.status_data["total_ram_mb"] = total_ram_mb
        if vram_usage_mb is not None:
            self.status_data["vram_usage_mb"] = vram_usage_mb
        if system_vram_usage_mb is not None:
            self.status_data["system_vram_usage_mb"] = system_vram_usage_mb
        if total_vram_mb is not None:
            self.status_data["total_vram_mb"] = total_vram_mb
        if cpu_usage_percent is not None:
            self.status_data["cpu_usage_percent"] = cpu_usage_percent
        if cpu_cores_count is not None:
            self.status_data["cpu_cores_count"] = cpu_cores_count
        if gpu_usage_percent is not None:
            self.status_data["gpu_usage_percent"] = gpu_usage_percent
        if worker_gpu_percent is not None:
            self.status_data["worker_gpu_percent"] = worker_gpu_percent
        if gpu_cores_count is not None:
            self.status_data["gpu_cores_count"] = gpu_cores_count
        if container_cpu_percent is not None:
            self.status_data["container_cpu_percent"] = container_cpu_percent
        if maintenance_mode is not None:
            self.status_data["maintenance_mode"] = maintenance_mode
        if job_pops_paused is not None:
            self.status_data["job_pops_paused"] = job_pops_paused
        if job_pops_pause_until is not _UNSET:
            self.status_data["job_pops_pause_until"] = job_pops_pause_until
        if user_kudos_total is not None:
            self.status_data["user_kudos_total"] = user_kudos_total
        if last_image_base64 is not None:
            self.status_data["last_image_base64"] = list(last_image_base64)
        if last_image_submission_timestamp is not None:
            self.status_data["last_image_submission_timestamp"] = last_image_submission_timestamp
        if last_image_model is not None:
            self.status_data["last_image_model"] = last_image_model
        if last_image_safety is not None:
            self.status_data["last_image_safety"] = list(last_image_safety)
        if console_logs is not None:
            self.status_data["console_logs"] = console_logs
        if faulted_jobs_history is not None:
            self.status_data["faulted_jobs_history"] = faulted_jobs_history
        if errors_history is not None:
            live_errors = list(errors_history)
            # Persist any new errors that have been added since the last update.
            if self._errors_db_path is not None:
                overlap = self._history_overlap_len(live_errors, self._live_errors_history)
                new_errors = live_errors[:-overlap] if overlap else live_errors
                if new_errors:
                    now = time.time()
                    try:
                        with sqlite3.connect(self._errors_db_path) as conn:
                            conn.executemany(
                                "INSERT INTO errors_log (message, created_at) VALUES (?, ?)",
                                [(msg, now) for msg in reversed(new_errors)],
                            )
                            conn.commit()
                        self._persisted_errors = (
                            list(new_errors) + getattr(self, "_persisted_errors", [])
                        )[:_MAX_PERSISTED_ERRORS]
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(f"Could not persist errors to database: {exc}")
                self.status_data["errors_history"] = self._merge_errors_history(live_errors)
            else:
                self.status_data["errors_history"] = live_errors
            self._live_errors_history = live_errors
        if user_details is not None:
            self.status_data["user_details"] = user_details
        if images_per_model is not None:
            bl = self._aggregate_baseline.get("images_per_model", {})
            merged_ipm: dict[str, int] = dict(bl)
            for k, v in images_per_model.items():
                merged_ipm[k] = bl.get(k, 0) + v
            self.status_data["images_per_model"] = merged_ipm
        if failed_jobs_per_model is not None:
            bl = self._aggregate_baseline.get("failed_jobs_per_model", {})
            merged_fjm: dict[str, int] = dict(bl)
            for k, v in failed_jobs_per_model.items():
                merged_fjm[k] = bl.get(k, 0) + v
            self.status_data["failed_jobs_per_model"] = merged_fjm
        if faulted_jobs_per_phase is not None:
            bl = self._aggregate_baseline.get("faulted_jobs_per_phase", {})
            merged_fpp: dict[str, int] = dict(bl)
            for k, v in faulted_jobs_per_phase.items():
                merged_fpp[k] = bl.get(k, 0) + v
            self.status_data["faulted_jobs_per_phase"] = merged_fpp
        if avg_time_per_job_state is not None:
            self.status_data["avg_time_per_job_state"] = dict(avg_time_per_job_state)
        if max_time_per_job_state is not None:
            self.status_data["max_time_per_job_state"] = dict(max_time_per_job_state)
        if avg_time_per_step_per_model is not None:
            self.status_data["avg_time_per_step_per_model"] = dict(avg_time_per_step_per_model)
        if max_time_per_step_per_model is not None:
            self.status_data["max_time_per_step_per_model"] = dict(max_time_per_step_per_model)
        if avg_time_per_job_per_model is not None:
            self.status_data["avg_time_per_job_per_model"] = dict(avg_time_per_job_per_model)
        if max_time_per_job_per_model is not None:
            self.status_data["max_time_per_job_per_model"] = dict(max_time_per_job_per_model)

        # Update uptime
        self.status_data["uptime"] = time.time() - self.status_data["session_start_time"]

        # Record a statistics snapshot (throttled to at most once per _stats_snapshot_interval).
        self._record_stats_snapshot()

    def reset_session_start_time(self) -> None:
        """Reset the session start time so the uptime pill shows near-zero immediately.

        Called when a program restart is imminent (e.g. idle auto-restart) so that
        the header uptime pill reflects the new session rather than the accumulated
        uptime of the old session during the brief shutdown window before the process
        is actually replaced via ``os.execv``.
        """
        self.status_data["session_start_time"] = time.time()
        self.status_data["uptime"] = 0.0

    async def start(self) -> None:
        """Start the web server."""
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
            await self.site.start()
            # Update self.port to the actual bound port (only needed when port=0 lets the
            # OS pick). `_server` is a private aiohttp attribute — access it defensively so
            # an aiohttp-internal rename can never fail a server that already bound fine.
            if self.port == 0:
                _server = getattr(self.site, "_server", None)
                _sockets = getattr(_server, "sockets", None) if _server is not None else None
                if _sockets:
                    self.port = _sockets[0].getsockname()[1]
            logger.info(f"Web UI started at http://0.0.0.0:{self.port}")
            self._horde_bg_task = asyncio.create_task(self._poll_horde_network())
            self._horde_bg_task.add_done_callback(self._log_bg_task_exception)
        except Exception as e:
            logger.error(f"Failed to start web UI server: {e}")
            raise

    @staticmethod
    def _log_bg_task_exception(task: asyncio.Task) -> None:
        """Log an exception that escaped the horde polling task instead of losing it silently."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"Web UI horde polling task ended with an error: {type(exc).__name__}: {exc}")

    async def stop(self) -> None:
        """Stop the web server."""
        if self._horde_bg_task is not None:
            self._horde_bg_task.cancel()
            try:
                await self._horde_bg_task
            except asyncio.CancelledError:
                pass
            self._horde_bg_task = None
        try:
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()
            logger.info("Web UI server stopped")
        except Exception as e:
            logger.error(f"Error stopping web UI server: {e}")
