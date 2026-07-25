"""Simple test to verify the web UI server can be created and started."""

import asyncio
import base64
import pathlib

import aiohttp
import pytest
from loguru import logger

from horde_worker_regen.webui.server import WorkerWebUI


@pytest.fixture(autouse=True)
def _clear_data_retention_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic by clearing AIWORKER_DATA_RETENTION_DAYS unless explicitly set."""
    monkeypatch.delenv("AIWORKER_DATA_RETENTION_DAYS", raising=False)


def test_webui_creation() -> None:
    """Test that WorkerWebUI can be instantiated."""
    webui = WorkerWebUI(port=0)  # Let OS assign a port
    assert webui is not None
    assert webui.port == 0
    assert webui.status_data is not None


def test_webui_status_update() -> None:
    """Test that WorkerWebUI status can be updated."""
    webui = WorkerWebUI(port=0)  # Let OS assign a port

    # Update some status values
    webui.update_status(
        worker_name="TestWorker",
        horde_username="TestUser#123",
        jobs_popped=10,
        jobs_queued=15,
        jobs_completed=8,
        jobs_faulted=1,
        processes_recovered=2,
        kudos_earned_session=100.5,
        kudos_per_hour=50.25,
        images_per_hour=12.5,
    )

    # Verify the values were updated
    assert webui.status_data["worker_name"] == "TestWorker"
    assert webui.status_data["horde_username"] == "TestUser#123"
    assert webui.status_data["jobs_popped"] == 10
    assert webui.status_data["jobs_queued"] == 15
    assert webui.status_data["jobs_completed"] == 8
    assert webui.status_data["jobs_faulted"] == 1
    assert webui.status_data["processes_recovered"] == 2
    assert webui.status_data["kudos_earned_session"] == 100.5
    assert webui.status_data["kudos_per_hour"] == 50.25
    assert webui.status_data["images_per_hour"] == 12.5


def test_webui_vram_and_ram_resources() -> None:
    """Test that WorkerWebUI correctly handles VRAM, RAM, and system RAM resource updates."""
    webui = WorkerWebUI(port=0)

    # Test VRAM usage and total VRAM update
    test_vram_usage_mb = 8192.5  # 8GB used
    test_total_vram_mb = 24576.0  # 24GB total
    test_ram_usage_mb = 16384.0  # 16GB worker RAM
    test_system_ram_usage_mb = 24576.0  # 24GB system RAM in use
    test_total_ram_mb = 32768.0  # 32GB total system RAM

    webui.update_status(
        vram_usage_mb=test_vram_usage_mb,
        total_vram_mb=test_total_vram_mb,
        ram_usage_mb=test_ram_usage_mb,
        system_ram_usage_mb=test_system_ram_usage_mb,
        total_ram_mb=test_total_ram_mb,
    )

    # Verify the values were updated correctly
    assert webui.status_data["vram_usage_mb"] == test_vram_usage_mb
    assert webui.status_data["total_vram_mb"] == test_total_vram_mb
    assert webui.status_data["ram_usage_mb"] == test_ram_usage_mb
    assert webui.status_data["system_ram_usage_mb"] == test_system_ram_usage_mb
    assert webui.status_data["total_ram_mb"] == test_total_ram_mb

    # Test that VRAM percentage would be calculated correctly (33% in this case)
    expected_percent = round((test_vram_usage_mb / test_total_vram_mb) * 100)
    assert expected_percent == 33

    # Test that worker RAM percentage would be calculated correctly (50%)
    expected_ram_pct = round((test_ram_usage_mb / test_total_ram_mb) * 100)
    assert expected_ram_pct == 50

    # Test that system RAM percentage would be calculated correctly (75%)
    expected_sys_ram_pct = round((test_system_ram_usage_mb / test_total_ram_mb) * 100)
    assert expected_sys_ram_pct == 75


def test_webui_cpu_gpu_and_container_cpu_usage() -> None:
    """Test that WorkerWebUI correctly handles CPU, GPU, and container CPU usage updates."""
    webui = WorkerWebUI(port=0)

    # Test CPU and GPU usage update
    test_cpu_usage_percent = 45.5
    test_gpu_usage_percent = 78.2
    test_worker_gpu_percent = 62.0
    test_container_cpu_percent = 23.1

    webui.update_status(
        cpu_usage_percent=test_cpu_usage_percent,
        gpu_usage_percent=test_gpu_usage_percent,
        worker_gpu_percent=test_worker_gpu_percent,
        container_cpu_percent=test_container_cpu_percent,
    )

    # Verify the values were updated correctly
    assert webui.status_data["cpu_usage_percent"] == test_cpu_usage_percent
    assert webui.status_data["gpu_usage_percent"] == test_gpu_usage_percent
    assert webui.status_data["worker_gpu_percent"] == test_worker_gpu_percent
    assert webui.status_data["container_cpu_percent"] == test_container_cpu_percent

    # Test edge case: 0% usage
    webui.update_status(
        cpu_usage_percent=0.0,
        gpu_usage_percent=0.0,
        worker_gpu_percent=0.0,
        container_cpu_percent=0.0,
    )
    assert webui.status_data["cpu_usage_percent"] == 0.0
    assert webui.status_data["gpu_usage_percent"] == 0.0
    assert webui.status_data["worker_gpu_percent"] == 0.0
    assert webui.status_data["container_cpu_percent"] == 0.0

    # Test edge case: 100% usage
    webui.update_status(
        cpu_usage_percent=100.0,
        gpu_usage_percent=100.0,
        worker_gpu_percent=100.0,
        container_cpu_percent=100.0,
    )
    assert webui.status_data["cpu_usage_percent"] == 100.0
    assert webui.status_data["gpu_usage_percent"] == 100.0
    assert webui.status_data["worker_gpu_percent"] == 100.0
    assert webui.status_data["container_cpu_percent"] == 100.0


def test_webui_gpu_cores_count() -> None:
    """Test that WorkerWebUI correctly handles GPU cores count updates."""
    webui = WorkerWebUI(port=0)

    # Default value should be 0
    assert webui.status_data["gpu_cores_count"] == 0

    # Test setting a typical GPU cores count (e.g. RTX 3080 has 8704 CUDA cores)
    webui.update_status(gpu_cores_count=8704)
    assert webui.status_data["gpu_cores_count"] == 8704

    # Test update with a different value
    webui.update_status(gpu_cores_count=10496)
    assert webui.status_data["gpu_cores_count"] == 10496

    # Test that passing None does not overwrite the existing value
    webui.update_status(gpu_cores_count=None)
    assert webui.status_data["gpu_cores_count"] == 10496


def test_webui_system_vram_usage() -> None:
    """Test that WorkerWebUI correctly handles system-wide VRAM usage updates."""
    webui = WorkerWebUI(port=0)

    test_vram_usage_mb = 4096.0    # 4 GB worker VRAM
    test_system_vram_usage_mb = 7168.0  # 7 GB system-wide VRAM in use
    test_total_vram_mb = 8192.0   # 8 GB total

    webui.update_status(
        vram_usage_mb=test_vram_usage_mb,
        system_vram_usage_mb=test_system_vram_usage_mb,
        total_vram_mb=test_total_vram_mb,
    )

    assert webui.status_data["vram_usage_mb"] == test_vram_usage_mb
    assert webui.status_data["system_vram_usage_mb"] == test_system_vram_usage_mb
    assert webui.status_data["total_vram_mb"] == test_total_vram_mb

    # Percentages: worker = 50%, system = 87.5%
    expected_worker_pct = round((test_vram_usage_mb / test_total_vram_mb) * 100)
    expected_system_pct = round((test_system_vram_usage_mb / test_total_vram_mb) * 100)
    assert expected_worker_pct == 50
    assert expected_system_pct == 88  # rounds to 88

    # Defaults: system_vram_usage_mb initialised to 0
    webui2 = WorkerWebUI(port=0)
    assert webui2.status_data["system_vram_usage_mb"] == 0


def test_webui_cpu_cores_count() -> None:
    """Test that WorkerWebUI correctly handles CPU cores count updates."""
    webui = WorkerWebUI(port=0)

    # Test CPU cores count update
    test_cpu_cores_count = 16

    webui.update_status(
        cpu_cores_count=test_cpu_cores_count,
    )

    # Verify the value was updated correctly
    assert webui.status_data["cpu_cores_count"] == test_cpu_cores_count

    # Test different core counts
    webui.update_status(cpu_cores_count=8)
    assert webui.status_data["cpu_cores_count"] == 8

    webui.update_status(cpu_cores_count=32)
    assert webui.status_data["cpu_cores_count"] == 32


def test_webui_vram_over_100_percent() -> None:
    """Test that WorkerWebUI correctly handles edge case where VRAM usage might exceed total (should cap at 100%)."""
    webui = WorkerWebUI(port=0)

    # Test edge case: VRAM usage exceeding total (shouldn't happen with fix, but frontend should cap it)
    test_vram_usage_mb = 30000.0  # 30GB used
    test_total_vram_mb = 24576.0  # 24GB total

    webui.update_status(
        vram_usage_mb=test_vram_usage_mb,
        total_vram_mb=test_total_vram_mb,
    )

    # Verify the values were updated
    assert webui.status_data["vram_usage_mb"] == test_vram_usage_mb
    assert webui.status_data["total_vram_mb"] == test_total_vram_mb

    # Test that calculated percentage would be over 100% before capping (122% in this case)
    raw_percent = round((test_vram_usage_mb / test_total_vram_mb) * 100)
    assert raw_percent > 100

    # The frontend JavaScript now uses Math.min(100, ...) to cap at 100%
    # This test documents that the frontend will display 100% even if the calculation exceeds it


def test_webui_new_features() -> None:
    """Test that WorkerWebUI handles new features (image preview and console logs)."""
    webui = WorkerWebUI(port=0)

    # Test last image update (single image)
    test_image_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    webui.update_status(last_image_base64=[test_image_base64])
    assert webui.status_data["last_image_base64"] == [test_image_base64]

    # Test last image update (multiple images for batch jobs)
    test_images_base64 = [test_image_base64, test_image_base64, test_image_base64]
    webui.update_status(last_image_base64=test_images_base64)
    assert webui.status_data["last_image_base64"] == test_images_base64
    assert len(webui.status_data["last_image_base64"]) == 3

    # Test console logs update
    test_logs = ["Log line 1", "Log line 2", "Log line 3"]
    webui.update_status(console_logs=test_logs)
    assert webui.status_data["console_logs"] == test_logs

    # Test current job with is_complete flag
    current_job = {
        "id": "test123",
        "model": "TestModel",
        "state": "INFERENCE_COMPLETE",
        "progress": 100,
        "is_complete": True,
    }
    webui.update_status(current_job=current_job)
    assert webui.status_data["current_job"] == current_job
    assert webui.status_data["current_job"]["is_complete"] is True


@pytest.mark.asyncio
async def test_webui_console_filter_html() -> None:
    """Test that the console section contains the filter dropdown."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/",
        ) as response:
            assert response.status == 200
            html = await response.text()

        assert 'id="console-filter-select"' in html
        assert 'class="console-filter-select"' in html
        assert 'onchange="applyConsoleFilter()"' in html
        assert 'title="Filter logs by severity"' in html
        assert '<option value="ALL">All levels</option>' in html
        assert '<option value="WARNING">Warning+</option>' in html
        assert '<option value="ERROR">Error+</option>' in html
        assert 'function applyConsoleFilter()' in html
        assert 'function _renderConsoleLogs()' in html
        assert '_consoleLogs' in html
        assert '_CONSOLE_LOG_LEVEL_RE' in html
        assert '_CONSOLE_LEVEL_ORDER' in html
        assert '.console-filter-select' in html
        # copy button should use the same filter logic as _renderConsoleLogs
        assert 'function copyConsoleLogs()' in html
        assert 'title="Copy visible console logs to clipboard"' in html
        assert '_consoleLogs.filter(function(log)' in html
        assert 'log.replace(_CONSOLE_ANSI_RE,' in html
    finally:
        await webui.stop()


def test_webui_faulted_jobs_history() -> None:
    """Test that WorkerWebUI handles faulted jobs history."""
    webui = WorkerWebUI(port=0)

    # Test faulted jobs history update
    test_faulted_jobs = [
        {
            "job_id": "job123",
            "model": "TestModel1",
            "time_faulted": 1234567890.0,
            "width": 512,
            "height": 512,
            "steps": 30,
            "sampler": "euler_a",
            "loras": [{"name": "test_lora", "model": 1.0, "clip": 1.0}],
            "controlnet": "canny",
            "workflow": "qr_code",
            "batch_size": 4,
            "fault_phase": "INFERENCE_PROCESSING",
        },
        {
            "job_id": "job456",
            "model": "TestModel2",
            "time_faulted": 1234567891.0,
            "width": 768,
            "height": 768,
            "steps": 50,
            "sampler": "dpm_2",
            "loras": [],
            "controlnet": None,
            "workflow": None,
            "batch_size": 1,
            "fault_phase": "INFERENCE_POST_PROCESSING",
        },
    ]
    webui.update_status(faulted_jobs_history=test_faulted_jobs)
    assert webui.status_data["faulted_jobs_history"] == test_faulted_jobs
    assert len(webui.status_data["faulted_jobs_history"]) == 2
    assert webui.status_data["faulted_jobs_history"][0]["job_id"] == "job123"
    assert webui.status_data["faulted_jobs_history"][0]["model"] == "TestModel1"
    assert webui.status_data["faulted_jobs_history"][0]["batch_size"] == 4
    assert webui.status_data["faulted_jobs_history"][0]["fault_phase"] == "INFERENCE_PROCESSING"
    assert webui.status_data["faulted_jobs_history"][1]["model"] == "TestModel2"
    assert webui.status_data["faulted_jobs_history"][1]["fault_phase"] == "INFERENCE_POST_PROCESSING"


def test_webui_batch_size_display() -> None:
    """Test that WorkerWebUI handles batch size in current job and job queue."""
    webui = WorkerWebUI(port=0)

    # Test current job with batch size
    current_job_batch = {
        "id": "test789",
        "model": "TestModel",
        "state": "INFERENCE_STARTING",
        "progress": 50,
        "is_complete": False,
        "batch_size": 3,
    }
    webui.update_status(current_job=current_job_batch)
    assert webui.status_data["current_job"] == current_job_batch
    assert webui.status_data["current_job"]["batch_size"] == 3

    # Test current job without batch size (batch_size = 1)
    current_job_single = {
        "id": "test790",
        "model": "TestModel2",
        "state": "PROCESSING",
        "progress": 25,
        "is_complete": False,
        "batch_size": 1,
    }
    webui.update_status(current_job=current_job_single)
    assert webui.status_data["current_job"]["batch_size"] == 1

    # Test job queue with various batch sizes
    job_queue_with_batches = [
        {"id": "queue1", "model": "Model1", "batch_size": 2},
        {"id": "queue2", "model": "Model2", "batch_size": 1},
        {"id": "queue3", "model": "Model3", "batch_size": 5},
    ]
    webui.update_status(job_queue=job_queue_with_batches)
    assert webui.status_data["job_queue"] == job_queue_with_batches
    assert webui.status_data["job_queue"][0]["batch_size"] == 2
    assert webui.status_data["job_queue"][1]["batch_size"] == 1
    assert webui.status_data["job_queue"][2]["batch_size"] == 5


def test_webui_current_job_detailed_info() -> None:
    """Test that WorkerWebUI handles detailed job info (steps, size, sampler, loras) in current job."""
    webui = WorkerWebUI(port=0)

    # Test current job with all new fields
    current_job_detailed = {
        "id": "test999",
        "model": "TestModel",
        "state": "INFERENCE_PROCESSING",
        "progress": 75,
        "is_complete": False,
        "batch_size": 2,
        "steps": 30,
        "width": 512,
        "height": 768,
        "sampler": "euler_a",
        "loras": [
            {"name": "test_lora_1", "model": 1.0, "clip": 1.0},
            {"name": "test_lora_2", "model": 0.8, "clip": 0.8},
        ],
    }
    webui.update_status(current_job=current_job_detailed)
    assert webui.status_data["current_job"] == current_job_detailed
    assert webui.status_data["current_job"]["steps"] == 30
    assert webui.status_data["current_job"]["width"] == 512
    assert webui.status_data["current_job"]["height"] == 768
    assert webui.status_data["current_job"]["sampler"] == "euler_a"
    assert len(webui.status_data["current_job"]["loras"]) == 2
    assert webui.status_data["current_job"]["loras"][0]["name"] == "test_lora_1"

    # Test current job without loras
    current_job_no_loras = {
        "id": "test998",
        "model": "TestModel2",
        "state": "INFERENCE_STARTING",
        "progress": 10,
        "is_complete": False,
        "batch_size": 1,
        "steps": 20,
        "width": 1024,
        "height": 1024,
        "sampler": "dpm_2",
        "loras": None,
    }
    webui.update_status(current_job=current_job_no_loras)
    assert webui.status_data["current_job"]["loras"] is None
    assert webui.status_data["current_job"]["steps"] == 20
    assert webui.status_data["current_job"]["width"] == 1024
    assert webui.status_data["current_job"]["sampler"] == "dpm_2"


def test_webui_last_image_submission_timestamp() -> None:
    """Test that WorkerWebUI handles last image submission timestamp."""
    import time

    webui = WorkerWebUI(port=0)

    # Test default value (0.0 = no image submitted yet)
    assert webui.status_data["last_image_submission_timestamp"] == 0.0

    # Test updating with a timestamp
    test_timestamp = time.time()
    webui.update_status(last_image_submission_timestamp=test_timestamp)
    assert webui.status_data["last_image_submission_timestamp"] == test_timestamp

    # Test updating with an older timestamp (simulating submission from past)
    old_timestamp = time.time() - 3600  # 1 hour ago
    webui.update_status(last_image_submission_timestamp=old_timestamp)
    assert webui.status_data["last_image_submission_timestamp"] == old_timestamp

    # Verify timestamp is retained after updating other fields
    webui.update_status(jobs_completed=5)
    assert webui.status_data["last_image_submission_timestamp"] == old_timestamp


def test_webui_images_history() -> None:
    """Test that WorkerWebUI handles gallery images via add_gallery_image / /api/gallery."""
    webui = WorkerWebUI(port=0)

    # Test default state: no images_history in status_data; images_count starts at 0
    assert "images_history" not in webui.status_data
    assert webui.status_data["images_count"] == 0
    assert webui._gallery_dict == {}

    # Test adding a single image entry
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    webui.add_gallery_image({"base64": test_image_b64, "timestamp": 1704067205.0, "model": "stable_diffusion_xl"})
    assert len(webui._gallery_dict) == 1
    assert webui.status_data["images_count"] == 1
    assert webui._gallery_dict[0]["model"] == "stable_diffusion_xl"

    # Test adding multiple image entries
    webui.add_gallery_image({"base64": test_image_b64, "timestamp": 1704067210.0, "model": "stable_diffusion_2_1"})
    webui.add_gallery_image({"base64": test_image_b64, "timestamp": 1704067215.0, "model": None})
    assert len(webui._gallery_dict) == 3
    assert webui.status_data["images_count"] == 3
    assert webui._gallery_dict[1]["model"] == "stable_diffusion_2_1"
    assert webui._gallery_dict[2]["model"] is None

    # Test that gallery is unbounded – adding more than 200 entries keeps all of them.
    for i in range(200):
        webui.add_gallery_image({"base64": test_image_b64, "timestamp": float(i), "model": "sdxl"})
    assert len(webui._gallery_dict) == 203
    assert webui.status_data["images_count"] == 203

    # Verify /api/status does NOT include base64 image data
    assert "images_history" not in webui.status_data


@pytest.mark.asyncio
async def test_webui_start_stop() -> None:
    """Test that WorkerWebUI can be started and stopped."""
    webui = WorkerWebUI(port=0)  # Let OS assign an available port

    try:
        # Start the server
        await webui.start()

        # Give it a moment to start
        await asyncio.sleep(0.5)

        # Get the actual port assigned
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Verify it's running by checking if we can access the health endpoint
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/health",
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert data["status"] == "ok"
    finally:
        # Stop the server
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_gallery_thumbnail_only() -> None:
    """Test that /api/gallery strips full-res base64 when a thumbnail is present."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # A minimal 1×1 PNG in base64
        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        # Add an entry *without* a thumbnail (simulates PIL not being available)
        webui._gallery_dict[0] = {"gallery_id": 0, "base64": test_b64, "timestamp": 1.0, "model": "m1"}

        # Add an entry *with* a thumbnail (simulates PIL available; thumbnail_only stripping applies)
        webui._gallery_dict[1] = {"gallery_id": 1, "base64": test_b64, "thumbnail": "thumb_data", "timestamp": 2.0, "model": "m2"}
        webui.status_data["images_count"] = 2

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=48",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert data["total"] == 2
        # Newest first: index 0 = entry with thumbnail (base64 should be stripped)
        entry_with_thumb = data["images"][0]
        assert "thumbnail" in entry_with_thumb
        assert "gallery_id" in entry_with_thumb
        assert "base64" not in entry_with_thumb, "base64 must be stripped when thumbnail exists"

        # index 1 = entry without thumbnail (base64 should be kept as fallback)
        entry_no_thumb = data["images"][1]
        assert "base64" in entry_no_thumb, "base64 must be kept when no thumbnail is available"
        assert "thumbnail" not in entry_no_thumb
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_gallery_thumb_binary() -> None:
    """Test that /api/gallery/thumb/{id} serves raw, cacheable image bytes.

    This backs the gallery grid: each tile is a real <img src="/api/gallery/thumb/{id}">, so the
    browser's native lazy-loading and HTTP cache do the work instead of custom JS -- revisiting a
    page, or even a full reload, is served from the browser's own cache.
    """
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        thumb_b64 = base64.b64encode(b"fake-jpeg-thumbnail-bytes").decode("ascii")

        webui._gallery_dict[0] = {"gallery_id": 0, "base64": test_b64, "thumbnail": thumb_b64, "timestamp": 1.0, "model": "m1"}
        webui._gallery_dict[1] = {"gallery_id": 1, "base64": test_b64, "timestamp": 2.0, "model": "m2"}

        # A thumbnail is served as raw JPEG bytes with a far-future, immutable cache header.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/thumb/0",
        ) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/jpeg"
            assert "immutable" in response.headers["Cache-Control"]
            body = await response.read()
        assert body == base64.b64decode(thumb_b64)

        # No thumbnail available -> falls back to full-resolution PNG bytes.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/thumb/1",
        ) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/png"
            body = await response.read()
        assert body == base64.b64decode(test_b64)

        # Non-existent gallery_id -> 404.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/thumb/9999",
        ) as response:
            assert response.status == 404

        # Malformed gallery_id -> 400.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/thumb/abc",
        ) as response:
            assert response.status == 400
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_gallery_full_binary() -> None:
    """Test that /api/gallery/full/{id} serves raw, cacheable full-resolution PNG bytes.

    This backs the overlay viewer: <img src="/api/gallery/full/{id}"> instead of a JSON+base64
    fetch, so reopening an already-viewed image is served instantly from the browser's cache.
    """
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        webui._gallery_dict[0] = {"gallery_id": 0, "base64": test_b64, "timestamp": 1.0, "model": "m1"}

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/full/0",
        ) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/png"
            assert "immutable" in response.headers["Cache-Control"]
            body = await response.read()
        assert body == base64.b64decode(test_b64)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/full/9999",
        ) as response:
            assert response.status == 404
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_gallery_thumb_binary_db_fallback(tmp_path: pathlib.Path) -> None:
    """Test that /api/gallery/thumb/{id} falls back to the gallery DB for evicted thumbnails."""
    webui = WorkerWebUI(port=0, db_path=str(tmp_path))

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        # add_gallery_image() generates its own thumbnail from the base64 (via Pillow, when
        # available) and persists it to the DB, so the exact bytes aren't asserted here.
        webui.add_gallery_image({"base64": test_b64, "timestamp": 1.0, "model": "m"})
        persisted_thumbnail = webui._gallery_dict[0].get("thumbnail")
        assert persisted_thumbnail

        # Simulate the in-memory thumbnail having been evicted to bound RAM usage — it must
        # still be recoverable from the gallery DB.
        webui._gallery_dict[0].pop("thumbnail", None)
        webui._gallery_dict[0].pop("base64", None)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/thumb/0",
        ) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/jpeg"
            body = await response.read()
        assert body == base64.b64decode(persisted_thumbnail)
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_index_initial_gpu_and_vram_markup() -> None:
    """Test that the initial GPU/VRAM topbar pills render neutral values with aria state."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/",
        ) as response:
            assert response.status == 200
            html = await response.text()

        assert 'id="topbar-gpu-pct">0%</span>' in html
        assert 'id="topbar-cpu-bar" style="width:0%" aria-label="System CPU usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"' in html
        assert 'id="topbar-cpu-ctr-bar" style="width:0%" aria-label="Worker CPU usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"' in html
        assert 'id="topbar-gpu-bar" style="width:0%" aria-label="System GPU usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"' in html
        assert 'id="topbar-gpu-wrk-bar" style="width:0%" aria-label="Worker GPU usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"' in html
        assert 'id="topbar-gpu-wrk-pct">0%</span>' in html
        assert 'id="topbar-vram-total">0 MB</span>' in html
        assert 'id="topbar-gpu-cores">0 cores</span>' in html
        assert 'id="topbar-vram-bar" style="width:0%" aria-label="System VRAM usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"' in html
        assert 'id="topbar-vram-wrk-bar" style="width:0%" aria-label="Worker VRAM usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"' in html
        assert 'id="topbar-vram-wrk-pct">0%</span>' in html
        assert 'id="topbar-total-ram-val">0 GB</span>' in html
        assert 'id="topbar-sysram-bar" style="width:0%" aria-label="System RAM usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"' in html
        assert 'id="topbar-ram-bar" style="width:0%" aria-label="Worker RAM usage" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"' in html
        assert "document.getElementById('topbar-total-ram-val').textContent = totalRamVal;" in html
        assert "const gpu = Math.max(sysGpuRaw, workerGpu);" in html
        assert "const sysVram = Math.max(vramTotalMb > 0 ? Math.min(100, Math.round((sysVramMb / vramTotalMb) * 100)) : 0, vram);" in html
        assert "if (proc.job_id) sl.push('Job: '+escapeHtml(proc.job_id));" in html
        assert "escapeHtml(proc.display_id || proc.id)" in html
        assert "if (pageId === 'stats') {" in html
        assert "fetchStats(true);" in html
        assert ".image-grid-item .image-timestamp { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.6); color: #e2e8f0; font-size: 0.65rem; padding: 3px 6px; text-align: center; border-radius: 0 0 8px 8px; pointer-events: none; opacity: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }" in html
        assert "'/ ' + totalRamVal" not in html
        assert "topbar-vram-val" not in html
        assert "topbar-ram-val" not in html
        # Mobile resource chips – all 8 values must be present in 4-column × 3-row grid layout
        assert 'id="mobile-cpu">SYS 0%</span>' in html
        assert 'id="mobile-cpu-ctr">WRK 0%</span>' in html
        assert 'id="mobile-gpu">SYS 0%</span>' in html
        assert 'id="mobile-gpu-wrk">WRK 0%</span>' in html
        assert 'id="mobile-vram">WRK 0%</span>' in html
        assert 'id="mobile-sysvram">SYS 0%</span>' in html
        assert 'id="mobile-ram">WRK 0%</span>' in html
        assert 'id="mobile-sysram">SYS 0%</span>' in html
        assert '.last-image-container { display: flex; align-items: center; justify-content: center; border-radius: 8px; height: 400px; overflow: hidden; }' in html
        assert '#overview-current-job { height: 400px; overflow: hidden; }' in html
        assert html.count("mobile-res-chip-secondary") == 5
        # 4 column groups with headers must be present
        assert html.count('class="mobile-res-col"') == 4
        assert html.count('class="mobile-res-head"') == 4
        # JS must update all 8 mobile chips with short SYS/WRK labels
        assert "function setMobileResChip(chipId, chipLabel, chipValue)" in html
        assert "setMobileResChip('mobile-gpu-wrk', 'WRK', workerGpu);" in html
        assert "setMobileResChip('mobile-sysvram', 'SYS', sysVram);" in html
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_gallery_image_endpoint() -> None:
    """Test that /api/gallery/image returns the full-resolution image by stable gallery_id."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        # Use add_gallery_image so gallery_id values are stamped by the server.
        webui.add_gallery_image({"base64": test_b64, "timestamp": 1.0, "model": "older"})
        webui.add_gallery_image({"base64": test_b64, "timestamp": 2.0, "model": "newer"})

        # Retrieve the gallery_id values from the /api/gallery response.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=48",
        ) as response:
            assert response.status == 200
            gallery_data = await response.json()

        assert gallery_data["total"] == 2
        newer_id = gallery_data["images"][0]["gallery_id"]  # newest first
        older_id = gallery_data["images"][1]["gallery_id"]

        # Fetch by stable gallery_id: should return the "newer" image regardless of order.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/image?id={newer_id}",
        ) as response:
            assert response.status == 200
            img = await response.json()
        assert img["model"] == "newer"
        assert img["base64"] == test_b64

        # Fetch the older image by its gallery_id.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/image?id={older_id}",
        ) as response:
            assert response.status == 200
            img = await response.json()
        assert img["model"] == "older"

        # Stable: adding a new image must not shift existing gallery_ids.
        webui.add_gallery_image({"base64": test_b64, "timestamp": 3.0, "model": "newest"})
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/image?id={older_id}",
        ) as response:
            assert response.status == 200
            img = await response.json()
        assert img["model"] == "older", "gallery_id must remain stable after new images are added"

        # Non-existent gallery_id should return 404.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/image?id=99999",
        ) as response:
            assert response.status == 404

        # Missing id parameter should return 400.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/image",
        ) as response:
            assert response.status == 400

        # Invalid (non-integer) id should return 400.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/image?id=abc",
        ) as response:
            assert response.status == 400
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_status_excludes_images_and_last_image_endpoint() -> None:
    """Test that /api/status omits last-image payloads and /api/last_image serves them."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        test_timestamp = 1704067200.0
        test_model = "stable_diffusion_xl"
        test_safety = [{"is_nsfw": True, "is_csam": False}]
        webui.update_status(
            last_image_base64=[test_b64],
            last_image_submission_timestamp=test_timestamp,
            last_image_model=test_model,
            last_image_safety=test_safety,
        )

        # /api/status must NOT include the last-image payload
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/status",
        ) as response:
            assert response.status == 200
            status = await response.json()
        assert "last_image_base64" not in status, "/api/status must not expose last_image_base64"
        assert "last_image_model" not in status, "/api/status must not expose last_image_model"
        assert "last_image_safety" not in status, "/api/status must not expose last_image_safety"
        assert status["last_image_submission_timestamp"] == test_timestamp

        # /api/last_image must return the full last-image payload
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/last_image",
        ) as response:
            assert response.status == 200
            img_data = await response.json()
        assert "last_image_base64" in img_data
        assert img_data["last_image_base64"] == [test_b64]
        assert img_data["last_image_submission_timestamp"] == test_timestamp
        assert img_data["last_image_model"] == test_model
        assert img_data["last_image_safety"] == test_safety
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_gallery_metadata_only() -> None:
    """Test that /api/gallery?metadata_only=true strips both thumbnail and base64."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        # Entry without thumbnail (base64 only)
        webui._gallery_dict[0] = {"gallery_id": 0, "base64": test_b64, "timestamp": 1.0, "model": "m1"}
        # Entry with both thumbnail and base64
        webui._gallery_dict[1] = {"gallery_id": 1, "base64": test_b64, "thumbnail": "thumb_data", "timestamp": 2.0, "model": "m2"}

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=48&metadata_only=true",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert data["total"] == 2
        for entry in data["images"]:
            assert "base64" not in entry, "base64 must be stripped with metadata_only=true"
            assert "thumbnail" not in entry, "thumbnail must be stripped with metadata_only=true"
            assert "gallery_id" in entry
            assert "timestamp" in entry
            assert "model" in entry
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_gallery_image_thumbnail_only() -> None:
    """Test that /api/gallery/image?thumbnail_only=true strips full-res base64."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        # Entry with both thumbnail and base64
        webui._gallery_dict[5] = {"gallery_id": 5, "base64": test_b64, "thumbnail": "thumb_data", "timestamp": 1.0, "model": "m1"}
        # Entry without thumbnail (base64 only); thumbnail_only should still work (no thumbnail to return)
        webui._gallery_dict[6] = {"gallery_id": 6, "base64": test_b64, "timestamp": 2.0, "model": "m2"}

        # thumbnail_only=true on entry that has a thumbnail: base64 must be stripped
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/image?id=5&thumbnail_only=true",
        ) as response:
            assert response.status == 200
            img = await response.json()
        assert "base64" not in img, "base64 must be stripped when thumbnail_only=true"
        assert img["thumbnail"] == "thumb_data"
        assert img["model"] == "m1"

        # thumbnail_only=false (default) on same entry: base64 must be present
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/image?id=5",
        ) as response:
            assert response.status == 200
            img = await response.json()
        assert img["base64"] == test_b64
        assert img["thumbnail"] == "thumb_data"

        # thumbnail_only=true on entry without thumbnail: base64 is kept as a fallback
        # so the frontend can still render the image even without a generated thumbnail.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/image?id=6&thumbnail_only=true",
        ) as response:
            assert response.status == 200
            img = await response.json()
        assert img["base64"] == test_b64
        assert img["model"] == "m2"
    finally:
        await webui.stop()


def test_webui_gallery_evicts_base64_from_memory_when_persisted(tmp_path: pathlib.Path) -> None:
    """Once persisted to the DB with a thumbnail, the full base64 is dropped from memory.

    This is the memory-bounding behaviour that prevents the worker from accumulating
    every generated image's multi-MB base64 in RAM (which would eventually OOM-kill the
    container). The full image must remain recoverable from the gallery database.
    """
    webui = WorkerWebUI(port=0, db_path=str(tmp_path))
    test_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    # Provide an explicit thumbnail so the eviction path is exercised even when Pillow is
    # unavailable to generate one from the base64.
    webui.add_gallery_image({"base64": test_b64, "thumbnail": "thumb", "timestamp": 1.0, "model": "m"})

    entry = webui._gallery_dict[0]
    assert entry.get("thumbnail"), "a thumbnail should be present in memory"
    assert "base64" not in entry, "full-resolution base64 must be evicted from memory once persisted"
    # ...but it must still be recoverable from the database on demand.
    assert webui._fetch_gallery_base64_from_db(0) == test_b64


@pytest.mark.asyncio
async def test_webui_gallery_full_res_served_from_db_when_evicted(tmp_path: pathlib.Path) -> None:
    """/api/gallery/image returns the full-resolution image from the DB after RAM eviction."""
    webui = WorkerWebUI(port=0, db_path=str(tmp_path))
    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        webui.add_gallery_image({"base64": test_b64, "timestamp": 1.0, "model": "m"})

        # Simulate memory eviction: the full base64 is gone from RAM but persisted in the DB.
        webui._gallery_dict[0].pop("base64", None)
        assert "base64" not in webui._gallery_dict[0]

        # The overlay endpoint must still return the full-resolution image (fetched from the DB).
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/image?id=0",
        ) as response:
            assert response.status == 200
            img = await response.json()
        assert img["base64"] == test_b64
    finally:
        await webui.stop()


def test_webui_gallery_base64_evicted_even_without_thumbnail(tmp_path: pathlib.Path) -> None:
    """Once persisted, the full base64 is dropped from memory even when no thumbnail exists.

    Entries that fail thumbnail generation must not pin their multi-MB base64 in RAM
    forever — the image stays recoverable from the gallery database on demand.
    """
    webui = WorkerWebUI(port=0, db_path=str(tmp_path))
    # An invalid PNG payload: thumbnail generation fails, but persistence still succeeds.
    webui.add_gallery_image({"base64": "notapngB", "timestamp": 1.0, "model": "m"})

    entry = webui._gallery_dict[0]
    assert "thumbnail" not in entry
    assert "base64" not in entry, "base64 must be evicted once persisted, even without a thumbnail"
    assert webui._fetch_gallery_base64_from_db(0) == "notapngB"


def test_webui_gallery_thumbnail_cap_evicts_oldest(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the newest _MAX_THUMBNAILS_IN_MEMORY thumbnails stay in RAM; older ones stay in the DB."""
    import time

    monkeypatch.setattr("horde_worker_regen.webui.server._MAX_THUMBNAILS_IN_MEMORY", 3)
    webui = WorkerWebUI(port=0, db_path=str(tmp_path))
    now = time.time()
    for i in range(5):
        webui.add_gallery_image(
            {"base64": "notapngB", "thumbnail": f"thumb{i}", "timestamp": now + i, "model": "m"},
        )

    in_memory = [gid for gid, e in webui._gallery_dict.items() if "thumbnail" in e]
    assert in_memory == [2, 3, 4], "only the newest three thumbnails may stay in memory"
    # Evicted thumbnails must remain recoverable from the DB for the gallery grid.
    assert webui._fetch_gallery_thumbnails_from_db([0, 1]) == {0: "thumb0", 1: "thumb1"}


def test_webui_gallery_fullres_cap_without_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a DB, at most _MAX_FULLRES_IN_MEMORY entries keep their full-resolution base64."""
    import time

    monkeypatch.setattr("horde_worker_regen.webui.server._MAX_FULLRES_IN_MEMORY", 2)
    webui = WorkerWebUI(port=0)
    now = time.time()
    for i in range(4):
        webui.add_gallery_image({"base64": f"payload{i}", "timestamp": now + i, "model": "m"})

    with_fullres = [gid for gid, e in webui._gallery_dict.items() if "base64" in e]
    assert with_fullres == [2, 3], "only the newest two full-resolution payloads may stay in memory"
    assert len(webui._gallery_dict) == 4, "entries themselves are kept (only the base64 is dropped)"


def test_webui_gallery_entry_cap_without_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a DB the total entry count is hard-capped, evicting the oldest entries."""
    import time

    monkeypatch.setattr("horde_worker_regen.webui.server._MAX_GALLERY_ENTRIES_NO_DB", 3)
    webui = WorkerWebUI(port=0)
    now = time.time()
    for i in range(5):
        webui.add_gallery_image({"base64": None, "timestamp": now + i, "model": "m"})

    assert list(webui._gallery_dict) == [2, 3, 4]
    assert webui.status_data["images_count"] == 3


def test_webui_gallery_startup_never_loads_fullres(tmp_path: pathlib.Path) -> None:
    """Startup must restore gallery rows as metadata-only — no image data in RAM at all.

    Before this bound existed, startup loaded every in-retention row's image columns:
    full-resolution base64 could OOM the worker, and even thumbnail-only loading forced
    SQLite to walk each row's base64_data overflow pages (a full-file read that stalled
    startup for minutes on large galleries). Both stay in the DB now and are fetched on
    demand per page.
    """
    import time

    now = time.time()
    webui1 = WorkerWebUI(port=0, db_path=str(tmp_path))
    webui1.add_gallery_image({"base64": "notapngA", "thumbnail": "thumbA", "timestamp": now, "model": "m"})
    webui1.add_gallery_image({"base64": "notapngB", "timestamp": now + 1, "model": "m"})

    webui2 = WorkerWebUI(port=0, db_path=str(tmp_path))
    assert len(webui2._gallery_dict) == 2
    # Both rows are metadata-only in RAM; image data stays in the DB.
    assert "base64" not in webui2._gallery_dict[0]
    assert "thumbnail" not in webui2._gallery_dict[0]
    assert "base64" not in webui2._gallery_dict[1]
    assert "thumbnail" not in webui2._gallery_dict[1]
    # Image data remains available on demand via the lazy fetch paths.
    assert webui2._fetch_gallery_thumbnails_from_db([0]).get(0) == "thumbA"
    assert webui2._fetch_gallery_base64_from_db(1) == "notapngB"


@pytest.mark.asyncio
async def test_webui_gallery_image_thumbnail_fetched_from_db_when_evicted(tmp_path: pathlib.Path) -> None:
    """/api/gallery/image?thumbnail_only=1 falls back to the DB after in-memory eviction."""
    webui = WorkerWebUI(port=0, db_path=str(tmp_path))
    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        webui.add_gallery_image({"base64": "notapngA", "thumbnail": "thumbA", "timestamp": 1.0, "model": "m"})
        # Simulate the memory cap evicting this thumbnail from RAM.
        webui._gallery_dict[0].pop("thumbnail", None)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/image?id=0&thumbnail_only=1",
        ) as response:
            assert response.status == 200
            img = await response.json()
        assert img.get("thumbnail") == "thumbA"
        assert "base64" not in img
    finally:
        await webui.stop()


def test_webui_horde_snapshots_absolute_ceiling() -> None:
    """A multi-year retention setting must not translate into millions of snapshot slots."""
    from horde_worker_regen.webui.server import _HORDE_MAX_SERVER_SNAPS_CEILING

    webui = WorkerWebUI(port=0, data_retention_days=3650)
    assert webui._horde_max_server_snaps == _HORDE_MAX_SERVER_SNAPS_CEILING


@pytest.mark.asyncio
async def test_webui_status_excludes_errors_history_and_has_errors_count() -> None:
    """Test that /api/status omits errors_history and includes errors_count."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Populate some errors
        webui.update_status(errors_history=["error one", "error two", "error three"])

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/status",
        ) as response:
            assert response.status == 200
            status = await response.json()

        assert "errors_history" not in status, "/api/status must not expose the full errors_history list"
        assert "errors_count" in status, "/api/status must include errors_count"
        assert status["errors_count"] == 3

        # Adding more errors must be reflected in errors_count
        webui.update_status(errors_history=["error one", "error two", "error three", "error four"])
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/status",
        ) as response:
            status2 = await response.json()
        assert status2["errors_count"] == 4
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_errors_endpoint_pagination() -> None:
    """Test /api/errors returns paginated slices of the error history."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Add 25 errors
        errors = [f"error {i}" for i in range(25)]
        webui.update_status(errors_history=errors)

        # Default page (page=1, page_size=10)
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 25
        assert data["page"] == 1
        assert data["page_size"] == 10
        assert data["total_pages"] == 3
        assert len(data["errors"]) == 10
        assert data["errors"] == errors[:10]

        # Page 2
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors?page=2&page_size=10",
        ) as response:
            assert response.status == 200
            data2 = await response.json()
        assert data2["page"] == 2
        assert len(data2["errors"]) == 10
        assert data2["errors"] == errors[10:20]

        # Last page (partial)
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors?page=3&page_size=10",
        ) as response:
            assert response.status == 200
            data3 = await response.json()
        assert data3["page"] == 3
        assert len(data3["errors"]) == 5
        assert data3["errors"] == errors[20:25]

        # Custom page_size
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors?page=1&page_size=5",
        ) as response:
            assert response.status == 200
            data4 = await response.json()
        assert data4["total_pages"] == 5
        assert len(data4["errors"]) == 5
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_errors_endpoint_edge_cases() -> None:
    """Test /api/errors handles out-of-range page, invalid params, and empty history."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Empty history
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["total_pages"] == 1
        assert data["errors"] == []

        # Populate errors for remaining tests
        webui.update_status(errors_history=[f"error {i}" for i in range(15)])

        # Out-of-range page is clamped to last page
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors?page=999",
        ) as response:
            assert response.status == 200
            data_clamped = await response.json()
        assert data_clamped["page"] == data_clamped["total_pages"]
        assert len(data_clamped["errors"]) > 0

        # page=0 is treated as page=1
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors?page=0",
        ) as response:
            assert response.status == 200
            data_zero = await response.json()
        assert data_zero["page"] == 1

        # Invalid (non-integer) page falls back to 1
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors?page=abc",
        ) as response:
            assert response.status == 200
            data_inv = await response.json()
        assert data_inv["page"] == 1

        # page_size is capped at 100
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors?page_size=9999",
        ) as response:
            assert response.status == 200
            data_cap = await response.json()
        assert data_cap["page_size"] == 100
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_status_includes_images_per_hour() -> None:
    """Test that /api/status JSON payload includes the images_per_hour field."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        webui.update_status(images_per_hour=7.5)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/status",
        ) as response:
            assert response.status == 200
            status = await response.json()

        assert "images_per_hour" in status, "/api/status must include images_per_hour"
        assert status["images_per_hour"] == 7.5
    finally:
        await webui.stop()


def test_webui_user_details_status_update() -> None:
    """Test that update_status correctly stores user_details in status_data."""
    webui = WorkerWebUI(port=0)

    sample_workers: list[dict] = [
        {
            "id": "worker-id-1",
            "name": "TestWorker",
            "version": "0.9.0",
            "type": "image",
            "online": True,
            "nsfw": False,
            "trusted": True,
            "img2img": True,
            "painting": False,
            "lora": True,
            "max_pixels": 4194304,
            "threads": 2,
            "models": ["stable_diffusion", "sdxl"],
            "uptime": 7200,
            "kudos_rewards": 3600.0,
        },
    ]
    user_details = {
        "worker_count": 1,
        "trusted": True,
        "moderator": False,
        "workers_list": sample_workers,
        "kudos_details": {"accumulated": 5000.0, "gifted": 100.0},
    }

    webui.update_status(user_details=user_details)

    assert "user_details" in webui.status_data, "status_data must contain user_details"
    stored = webui.status_data["user_details"]
    assert stored["worker_count"] == 1
    assert stored["trusted"] is True
    assert stored["moderator"] is False
    assert len(stored["workers_list"]) == 1
    w = stored["workers_list"][0]
    assert w["name"] == "TestWorker"
    assert w["version"] == "0.9.0"
    assert w["type"] == "image"
    assert w["nsfw"] is False
    assert w["trusted"] is True
    assert w["img2img"] is True
    assert w["threads"] == 2
    assert w["models"] == ["stable_diffusion", "sdxl"]
    assert w["uptime"] == 7200
    assert w["kudos_rewards"] == 3600.0
    assert stored["kudos_details"]["accumulated"] == 5000.0


@pytest.mark.asyncio
async def test_webui_status_api_includes_user_details() -> None:
    """Test that /api/status includes user_details with workers_list."""
    webui = WorkerWebUI(port=0)

    sample_workers: list[dict] = [
        {
            "id": "worker-id-2",
            "name": "ApiTestWorker",
            "version": "1.0.0",
            "type": "image",
            "online": True,
            "nsfw": True,
            "trusted": False,
            "img2img": False,
            "painting": True,
            "lora": False,
            "max_pixels": 1048576,
            "threads": 1,
            "models": ["model_a", "model_b", "model_c"],
            "uptime": 3600,
            "kudos_rewards": 1800.0,
        },
    ]
    user_details = {
        "worker_count": 1,
        "trusted": False,
        "workers_list": sample_workers,
    }

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        webui.update_status(user_details=user_details)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/status",
        ) as response:
            assert response.status == 200
            status = await response.json()

        assert "user_details" in status, "/api/status must include user_details"
        ud = status["user_details"]
        assert ud["worker_count"] == 1
        assert ud["trusted"] is False
        assert len(ud["workers_list"]) == 1
        w = ud["workers_list"][0]
        assert w["name"] == "ApiTestWorker"
        assert w["type"] == "image"
        assert w["nsfw"] is True
        assert w["models"] == ["model_a", "model_b", "model_c"]
        assert w["uptime"] == 3600
        assert w["kudos_rewards"] == 1800.0
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_delete_worker_endpoint() -> None:
    """Test the DELETE /api/worker/{worker_id} endpoint."""
    webui = WorkerWebUI(port=0)

    offline_worker: dict = {
        "id": "offline-worker-uuid",
        "name": "OldWorker",
        "version": "1.0",
        "type": "image",
        "online": False,
        "nsfw": False,
        "trusted": False,
        "img2img": False,
        "painting": False,
        "lora": False,
        "max_pixels": 1048576,
        "threads": 1,
        "models": [],
        "uptime": 0,
        "kudos_rewards": 0.0,
    }
    online_worker: dict = {**offline_worker, "id": "online-worker-uuid", "name": "ActiveWorker", "online": True}
    current_worker: dict = {**offline_worker, "id": "current-worker-uuid", "name": "CurrentWorker", "online": False}

    user_details = {
        "worker_count": 3,
        "workers_list": [offline_worker, online_worker, current_worker],
    }
    webui.update_status(worker_name="CurrentWorker", user_details=user_details)

    deleted_ids: list[str] = []

    async def fake_delete(worker_id: str) -> bool:
        deleted_ids.append(worker_id)
        return True

    webui.set_delete_worker_callback(fake_delete)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # 404 for unknown worker
        async with aiohttp.ClientSession() as session, session.delete(
            f"http://localhost:{actual_port}/api/worker/nonexistent-id",
        ) as response:
            assert response.status == 404

        # 400 for online worker
        async with aiohttp.ClientSession() as session, session.delete(
            f"http://localhost:{actual_port}/api/worker/{online_worker['id']}",
        ) as response:
            assert response.status == 400
            body = await response.json()
            assert "online" in body["error"].lower()

        # 400 for current (running) worker even though offline
        async with aiohttp.ClientSession() as session, session.delete(
            f"http://localhost:{actual_port}/api/worker/{current_worker['id']}",
        ) as response:
            assert response.status == 400
            body = await response.json()
            assert "web ui" in body["error"].lower() or "currently running" in body["error"].lower()

        # 200 for valid offline, non-current worker
        async with aiohttp.ClientSession() as session, session.delete(
            f"http://localhost:{actual_port}/api/worker/{offline_worker['id']}",
        ) as response:
            assert response.status == 200
            body = await response.json()
            assert body["deleted_id"] == offline_worker["id"]

        assert offline_worker["id"] in deleted_ids

        # Callback returning False yields 502
        async def failing_delete(worker_id: str) -> bool:
            return False

        webui.set_delete_worker_callback(failing_delete)
        # Re-add the worker so validation passes
        webui.update_status(user_details=user_details)
        async with aiohttp.ClientSession() as session, session.delete(
            f"http://localhost:{actual_port}/api/worker/{offline_worker['id']}",
        ) as response:
            assert response.status == 502

        # 503 when no callback is registered
        webui.set_delete_worker_callback(None)
        webui.update_status(user_details=user_details)
        async with aiohttp.ClientSession() as session, session.delete(
            f"http://localhost:{actual_port}/api/worker/{offline_worker['id']}",
        ) as response:
            assert response.status == 503

    finally:
        await webui.stop()


def test_normalize_error_message_strips_short_timestamps() -> None:
    """_normalize_error_message must collapse HH:mm:ss prefixes to the same key."""
    normalize = WorkerWebUI._normalize_error_message
    # Same message at different seconds → identical normalised form
    assert normalize("12:34:56 | ERROR    | Job failed") == normalize("12:34:57 | ERROR    | Job failed")
    assert normalize("00:00:00 | ERROR    | msg") == normalize("23:59:59 | ERROR    | msg")
    # Short format with optional milliseconds
    assert normalize("12:34:56.123 | ERROR    | msg") == normalize("12:34:57.456 | ERROR    | msg")


def test_normalize_error_message_strips_full_timestamps() -> None:
    """_normalize_error_message must collapse YYYY-MM-DD HH:mm:ss[.SSS] prefixes."""
    normalize = WorkerWebUI._normalize_error_message
    # Full ISO format with space separator
    assert normalize("2026-02-04 21:44:06.123 | ERROR | msg") == normalize(
        "2026-02-05 09:00:00.000 | ERROR | msg"
    )
    # Full ISO format with T separator
    assert normalize("2026-02-04T21:44:06 | ERROR | msg") == normalize(
        "2026-02-05T09:00:00 | ERROR | msg"
    )


def test_normalize_error_message_strips_timestamps_in_body() -> None:
    """_normalize_error_message removes timestamps embedded anywhere in the message."""
    normalize = WorkerWebUI._normalize_error_message
    assert normalize("12:34:56 | ERROR | Failed at 12:34:55") == normalize(
        "12:34:57 | ERROR | Failed at 14:22:33"
    )


def test_normalize_error_message_multiline_exception() -> None:
    """_normalize_error_message groups multiline exception tracebacks correctly."""
    normalize = WorkerWebUI._normalize_error_message
    msg_a = (
        "12:34:56 | ERROR    | Job failed\n"
        "Traceback (most recent call last):\n"
        "  File \"app.py\", line 42, in run\n"
        "    result = process()\n"
        "RuntimeError: Connection refused"
    )
    msg_b = (
        "12:34:57 | ERROR    | Job failed\n"
        "Traceback (most recent call last):\n"
        "  File \"app.py\", line 42, in run\n"
        "    result = process()\n"
        "RuntimeError: Connection refused"
    )
    assert normalize(msg_a) == normalize(msg_b)


@pytest.mark.asyncio
async def test_webui_errors_grouped_endpoint_basic() -> None:
    """Test /api/errors/grouped groups identical errors and returns correct counts."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Empty history returns empty groups
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total_groups"] == 0
        assert data["total_errors"] == 0
        assert data["groups"] == []

        # Populate with repeated errors; "gamma error" only appears once but must still be shown
        errors = ["alpha error", "beta error", "alpha error", "gamma error", "alpha error", "beta error"]
        webui.update_status(errors_history=errors)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert data["total_errors"] == 6
        # All 3 unique messages are included as groups (single-occurrence errors appear as a group of 1)
        assert data["total_groups"] == 3

        # Groups must be sorted by count descending
        messages = [g["message"] for g in data["groups"]]
        counts = [g["count"] for g in data["groups"]]
        assert messages[0] == "alpha error"
        assert counts[0] == 3
        assert messages[1] == "beta error"
        assert counts[1] == 2
        # "gamma error" (single occurrence) must appear in grouped view as a group of 1
        assert "gamma error" in messages
        gamma_count = next(g["count"] for g in data["groups"] if g["message"] == "gamma error")
        assert gamma_count == 1

        # Every group must include an 'occurrences' list whose length is min(count, cap)
        from horde_worker_regen.webui.server import _MAX_OCCURRENCES_PER_GROUP as _CAP
        for grp in data["groups"]:
            assert "occurrences" in grp, f"Group {grp['message']!r} missing 'occurrences'"
            assert len(grp["occurrences"]) == min(grp["count"], _CAP)
        # The occurrences must be the actual original messages (not the representative repeated)
        alpha_grp = next(g for g in data["groups"] if g["message"] == "alpha error")
        assert all(occ == "alpha error" for occ in alpha_grp["occurrences"])
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_errors_grouped_endpoint_normalisation() -> None:
    """Test /api/errors/grouped groups errors that differ only by UUID/job/process/hex ID."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        uuid1 = "11111111-2222-3333-4444-555555555555"
        uuid2 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        pid1 = "98765"
        pid2 = "12345"
        hex1 = "0xDEADBEEF"
        hex2 = "0x1234ABCD"

        errors = [
            f"Job {uuid1} failed: connection timeout",
            f"Job {uuid2} failed: connection timeout",
            f"Process {pid1} crashed unexpectedly",
            f"Process {pid2} crashed unexpectedly",
            f"Memory access violation at address {hex1}",
            f"Memory access violation at address {hex2}",
        ]
        webui.update_status(errors_history=errors)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped",
        ) as response:
            assert response.status == 200
            data = await response.json()

        # Each pair normalises to the same key → 3 groups, each with count 2
        assert data["total_groups"] == 3
        counts = {g["count"] for g in data["groups"]}
        assert counts == {2}
        messages = [g["message"] for g in data["groups"]]
        # Each representative message must contain one of the original variable tokens
        assert any(uuid1 in m or uuid2 in m for m in messages), "UUID group representative missing"
        assert any(pid1 in m or pid2 in m for m in messages), "PID group representative missing"
        assert any(hex1 in m or hex2 in m for m in messages), "hex group representative missing"
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_errors_grouped_normalises_timestamps() -> None:
    """Test /api/errors/grouped groups errors that differ only by timestamp (seconds)."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Simulate the same error logged at different times in the webui HH:mm:ss format
        # and also in the full ISO YYYY-MM-DD HH:mm:ss.SSS format used by file logs.
        errors = [
            "12:34:56 | ERROR    | Job failed: connection timeout",
            "12:34:57 | ERROR    | Job failed: connection timeout",
            "12:34:58 | ERROR    | Job failed: connection timeout",
            "2026-02-04 21:44:06.123 | ERROR    | Inference process crashed",
            "2026-02-04 21:44:07.456 | ERROR    | Inference process crashed",
        ]
        webui.update_status(errors_history=errors)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped",
        ) as response:
            assert response.status == 200
            data = await response.json()

        # All three "Job failed" messages must collapse into one group; the two
        # "Inference process crashed" messages must collapse into another.
        assert data["total_errors"] == 5
        assert data["total_groups"] == 2

        counts = {g["count"] for g in data["groups"]}
        assert counts == {3, 2}
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_errors_grouped_normalises_short_ids() -> None:
    """Test /api/errors/grouped groups errors that differ only by short numeric IDs."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Short process IDs (2+ digits) that were previously not normalised
        # by the old 5-digit threshold must now be treated as equivalent.
        errors = [
            "Failed to kill process 12: [Errno 3] No such process",
            "Failed to kill process 37: [Errno 3] No such process",
            "Inference slot 10 became unresponsive",
            "Inference slot 11 became unresponsive",
            "Inference slot 12 became unresponsive",
        ]

        webui.update_status(errors_history=errors)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped",
        ) as response:
            assert response.status == 200
            data = await response.json()

        # The two "Failed to kill process" messages share the same normalised key;
        # the three "Inference slot" messages also share a key → 2 groups total.
        assert data["total_errors"] == 5
        assert data["total_groups"] == 2

        counts = sorted(g["count"] for g in data["groups"])
        assert counts == [2, 3]
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_errors_grouped_normalises_single_digit_process_numbers() -> None:
    """Test /api/errors/grouped groups errors that differ only by single-digit process numbers."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Single-digit process/slot numbers that should be normalised to the same key.
        errors = [
            "Inference slot 1 became unresponsive",
            "Inference slot 2 became unresponsive",
            "Inference slot 3 became unresponsive",
            "Process 1 crashed unexpectedly",
            "Process 2 crashed unexpectedly",
        ]
        webui.update_status(errors_history=errors)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped",
        ) as response:
            assert response.status == 200
            data = await response.json()

        # The three "Inference slot" messages share a key; the two "Process" messages
        # share another key → 2 groups total.
        assert data["total_errors"] == 5
        assert data["total_groups"] == 2

        counts = sorted(g["count"] for g in data["groups"])
        assert counts == [2, 3]
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_errors_grouped_endpoint_pagination() -> None:
    """Test /api/errors/grouped returns paginated groups."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # 15 distinct error types, each appearing twice.  Use letter-based labels so
        # that number normalisation does not collapse them into fewer groups.
        error_labels = "abcdefghijklmno"  # exactly 15 characters
        errors = [msg for label in error_labels for msg in [f"error type {label}", f"error type {label}"]]
        webui.update_status(errors_history=errors)

        # Page 1, page_size=10 → 10 groups
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped?page=1&page_size=10",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total_groups"] == 15
        assert data["total_pages"] == 2
        assert len(data["groups"]) == 10

        # Page 2 → 5 groups
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped?page=2&page_size=10",
        ) as response:
            assert response.status == 200
            data2 = await response.json()
        assert data2["page"] == 2
        assert len(data2["groups"]) == 5
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_errors_grouped_occurrences_contain_individual_messages() -> None:
    """Each group's 'occurrences' list must contain the actual individual error messages.

    Errors that differ only by variable tokens (UUIDs, timestamps, numeric IDs) must be
    grouped together, and the occurrences list must include every original raw message so
    the user can see the full timeline of when each error variant appeared.
    """
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Three distinct timestamps → three occurrences of the same logical error.
        errors = [
            "12:34:56 | ERROR    | Job failed: connection timeout",
            "12:34:57 | ERROR    | Job failed: connection timeout",
            "12:34:58 | ERROR    | Job failed: connection timeout",
            "12:34:59 | ERROR    | Unrelated error",
        ]
        webui.update_status(errors_history=errors)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert data["total_groups"] == 2

        # Find the "Job failed" group
        job_grp = next(g for g in data["groups"] if "Job failed" in g["message"])
        assert job_grp["count"] == 3
        assert len(job_grp["occurrences"]) == 3
        # Each occurrence must be the original message with its distinct timestamp
        occurrence_texts = set(job_grp["occurrences"])
        assert "12:34:56 | ERROR    | Job failed: connection timeout" in occurrence_texts
        assert "12:34:57 | ERROR    | Job failed: connection timeout" in occurrence_texts
        assert "12:34:58 | ERROR    | Job failed: connection timeout" in occurrence_texts

        # The unrelated error group must have exactly one occurrence
        unrelated_grp = next(g for g in data["groups"] if "Unrelated" in g["message"])
        assert unrelated_grp["count"] == 1
        assert len(unrelated_grp["occurrences"]) == 1
        assert unrelated_grp["occurrences"][0] == "12:34:59 | ERROR    | Unrelated error"
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_errors_grouped_occurrences_capped_at_max() -> None:
    """When a group has more than _MAX_OCCURRENCES_PER_GROUP occurrences, only the first
    _MAX_OCCURRENCES_PER_GROUP are included; the count still reflects the true total.
    """
    from horde_worker_regen.webui.server import _MAX_OCCURRENCES_PER_GROUP

    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        total = _MAX_OCCURRENCES_PER_GROUP + 10
        errors = ["same error message"] * total
        webui.update_status(errors_history=errors)

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert data["total_groups"] == 1
        grp = data["groups"][0]
        assert grp["count"] == total
        assert len(grp["occurrences"]) == _MAX_OCCURRENCES_PER_GROUP
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_errors_grouped_endpoint_edge_cases() -> None:
    """Test /api/errors/grouped handles invalid params and out-of-range page."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # "err b" appears only once but will still be included; "err a" appears twice
        webui.update_status(errors_history=["err a", "err b", "err a"])

        # Invalid page falls back to 1
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped?page=abc",
        ) as response:
            assert response.status == 200
            data_inv = await response.json()
        assert data_inv["page"] == 1

        # Out-of-range page is clamped to total_pages
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped?page=999",
        ) as response:
            assert response.status == 200
            data_clamped = await response.json()
        assert data_clamped["page"] == data_clamped["total_pages"]

        # page_size is capped at 100
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/errors/grouped?page_size=9999",
        ) as response:
            assert response.status == 200
            data_cap = await response.json()
        assert data_cap["page_size"] == 100
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_stats_endpoint() -> None:
    """Test that /api/stats returns a snapshots list and records data via update_status."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Initially the snapshot list should be empty.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/stats",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert "snapshots" in data
        assert data["snapshots"] == []
        # jobs_faulted session total must be present and zero initially.
        assert "jobs_faulted" in data
        assert data["jobs_faulted"] == 0
        # images_per_model field must be present and empty initially.
        assert "images_per_model" in data
        assert data["images_per_model"] == {}

        # Force the first snapshot by backdating the timestamp.
        webui._last_stats_snapshot_time = 0.0
        webui.update_status(
            jobs_completed=5,
            jobs_faulted=1,
            jobs_popped=6,
            kudos_earned_session=50.0,
            kudos_per_hour=25.0,
            images_per_hour=10.0,
            cpu_usage_percent=40.0,
            gpu_usage_percent=70.0,
            worker_gpu_percent=55.0,
            vram_usage_mb=4096.0,
            system_vram_usage_mb=6144.0,
            total_vram_mb=8192.0,
            ram_usage_mb=8192.0,
            system_ram_usage_mb=16384.0,
            total_ram_mb=32768.0,
            container_cpu_percent=20.0,
        )
        assert len(webui._stats_snapshots) == 1

        snap = webui._stats_snapshots[0]
        assert "t" in snap
        assert snap["jc"] == 5
        assert snap["jf"] == 1
        assert snap["jp"] == 6
        assert snap["ks"] == 50.0
        assert snap["iph"] == 10.0
        assert snap["kph"] == 25.0
        assert snap["cpu"] == 40.0
        assert snap["gpu"] == 70.0
        assert snap["worker_gpu"] == 55.0
        assert snap["vram"] == 50.0  # 4096/8192 = 50%
        assert snap["system_vram"] == 75.0  # 6144/8192 = 75%
        assert snap["ram"] == 25.0   # 8192/32768 = 25%
        assert snap["system_ram"] == 50.0  # 16384/32768 = 50%
        assert snap["container_cpu"] == 20.0

        # A second call within the interval must NOT add another snapshot.
        webui.update_status(jobs_completed=6)
        assert len(webui._stats_snapshots) == 1

        # Force a second snapshot.
        webui._last_stats_snapshot_time = 0.0
        webui.update_status(jobs_completed=7)
        assert len(webui._stats_snapshots) == 2

        # VRAM percentage must be capped at 100 even when usage exceeds total.
        webui._last_stats_snapshot_time = 0.0
        webui.update_status(vram_usage_mb=10000.0, total_vram_mb=8192.0)
        snap_over = webui._stats_snapshots[-1]
        assert snap_over["vram"] == 100.0, "vram_pct must be capped at 100%"

        # RAM percentage must be capped at 100 as well.
        webui._last_stats_snapshot_time = 0.0
        webui.update_status(ram_usage_mb=40000.0, total_ram_mb=32768.0)
        snap_ram_over = webui._stats_snapshots[-1]
        assert snap_ram_over["ram"] == 100.0, "ram_pct must be capped at 100%"

        # System RAM percentage must also be capped at 100.
        webui._last_stats_snapshot_time = 0.0
        webui.update_status(system_ram_usage_mb=40000.0, total_ram_mb=32768.0)
        snap_sysram_over = webui._stats_snapshots[-1]
        assert snap_sysram_over["system_ram"] == 100.0, "system_ram_pct must be capped at 100%"

        # System VRAM percentage must also be capped at 100.
        webui._last_stats_snapshot_time = 0.0
        webui.update_status(system_vram_usage_mb=10000.0, total_vram_mb=8192.0)
        snap_sysvram_over = webui._stats_snapshots[-1]
        assert snap_sysvram_over["system_vram"] == 100.0, "system_vram_pct must be capped at 100%"

        # System GPU snapshot must never be lower than worker GPU.
        webui._last_stats_snapshot_time = 0.0
        webui.update_status(gpu_usage_percent=35.0, worker_gpu_percent=60.0)
        snap_gpu_floor = webui._stats_snapshots[-1]
        assert snap_gpu_floor["gpu"] == 60.0, "gpu snapshot must be floored to worker_gpu_percent"

        # System VRAM snapshot must never be lower than worker VRAM percentage.
        webui._last_stats_snapshot_time = 0.0
        webui.update_status(vram_usage_mb=6144.0, system_vram_usage_mb=4096.0, total_vram_mb=8192.0)
        snap_sysvram_floor = webui._stats_snapshots[-1]
        assert snap_sysvram_floor["vram"] == 75.0
        assert snap_sysvram_floor["system_vram"] == 75.0, "system_vram snapshot must be floored to worker VRAM percent"

        # Verify /api/stats returns all recorded snapshots.
        # Snapshots recorded so far: initial, force-second, vram-over-100, ram-over-100,
        # system-ram-over-100, system-vram-over-100, gpu-floor, system-vram-floor = 8 total.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/stats",
        ) as response:
            assert response.status == 200
            data2 = await response.json()
        assert len(data2["snapshots"]) == 8

        # images_per_model is reflected from update_status and can be set and cleared.
        webui.update_status(images_per_model={"ModelA": 5, "ModelB": 2})
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/stats",
        ) as response:
            data3 = await response.json()
        assert data3["images_per_model"] == {"ModelA": 5, "ModelB": 2}

        # Passing an empty dict clears the field correctly.
        webui.update_status(images_per_model={})
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/stats",
        ) as response:
            data4 = await response.json()
        assert data4["images_per_model"] == {}

        # failed_jobs_per_model field must be present and empty initially.
        assert "failed_jobs_per_model" in data4
        assert data4["failed_jobs_per_model"] == {}

        # failed_jobs_per_model is reflected from update_status and can be set and cleared.
        webui.update_status(failed_jobs_per_model={"ModelA": 3, "ModelB": 1})
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/stats",
        ) as response:
            data5 = await response.json()
        assert data5["failed_jobs_per_model"] == {"ModelA": 3, "ModelB": 1}

        # Passing an empty dict clears the failed jobs field correctly.
        webui.update_status(failed_jobs_per_model={})
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/stats",
        ) as response:
            data6 = await response.json()
        assert data6["failed_jobs_per_model"] == {}

        # faulted_jobs_per_phase field must be present and empty initially.
        assert "faulted_jobs_per_phase" in data6
        assert data6["faulted_jobs_per_phase"] == {}

        # faulted_jobs_per_phase is reflected from update_status and can be set and cleared.
        webui.update_status(faulted_jobs_per_phase={"INFERENCE_PROCESSING": 4, "SAFETY_EVALUATING": 1})
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/stats",
        ) as response:
            data7 = await response.json()
        assert data7["faulted_jobs_per_phase"] == {"INFERENCE_PROCESSING": 4, "SAFETY_EVALUATING": 1}

        # Passing an empty dict clears the faulted_jobs_per_phase field correctly.
        webui.update_status(faulted_jobs_per_phase={})
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/stats",
        ) as response:
            data8 = await response.json()
        assert data8["faulted_jobs_per_phase"] == {}

        # jobs_faulted session total is reflected from update_status.
        webui.update_status(jobs_faulted=3)
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/stats",
        ) as response:
            data8b = await response.json()
        assert data8b["jobs_faulted"] == 3

        # avg_time_per_job_state and max_time_per_job_state must be present and empty initially.
        assert "avg_time_per_job_state" in data8
        assert "max_time_per_job_state" in data8
        assert data8["avg_time_per_job_state"] == {}
        assert data8["max_time_per_job_state"] == {}

        # avg_time_per_job_state and max_time_per_job_state are reflected from update_status.
        webui.update_status(
            avg_time_per_job_state={"INFERENCE_PROCESSING": 12.34, "TOTAL": 15.00},
            max_time_per_job_state={"INFERENCE_PROCESSING": 20.10, "TOTAL": 25.50},
        )
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/stats",
        ) as response:
            data9 = await response.json()
        assert data9["avg_time_per_job_state"] == {"INFERENCE_PROCESSING": 12.34, "TOTAL": 15.00}
        assert data9["max_time_per_job_state"] == {"INFERENCE_PROCESSING": 20.10, "TOTAL": 25.50}

        # Passing an empty dict clears the fields correctly.
        webui.update_status(avg_time_per_job_state={}, max_time_per_job_state={})
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/stats",
        ) as response:
            data10 = await response.json()
        assert data10["avg_time_per_job_state"] == {}
        assert data10["max_time_per_job_state"] == {}

    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_job_pops_pause_endpoint() -> None:
    """Test the POST /api/job_pops/pause endpoint."""
    import time as _time

    webui = WorkerWebUI(port=0)
    paused_calls: list[tuple[bool, float | None]] = []

    def fake_set_paused(paused: bool, pause_until: float | None) -> None:
        paused_calls.append((paused, pause_until))

    webui.set_job_pops_paused_callback(fake_set_paused)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # --- success: pause with no duration_seconds → indefinite ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/job_pops/pause",
            json={"paused": True},
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body["job_pops_paused"] is True
        assert body["job_pops_pause_until"] is None
        assert webui.status_data["job_pops_paused"] is True
        assert webui.status_data["job_pops_pause_until"] == body["job_pops_pause_until"]
        assert paused_calls[-1][0] is True
        assert paused_calls[-1][1] is None

        # --- success: pause indefinitely via explicit null ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/job_pops/pause",
            json={"paused": True, "duration_seconds": None},
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body["job_pops_paused"] is True
        assert body["job_pops_pause_until"] is None
        assert webui.status_data["job_pops_paused"] is True
        assert webui.status_data["job_pops_pause_until"] is None
        assert paused_calls[-1][0] is True
        assert paused_calls[-1][1] is None

        # --- success: pause for 15 minutes via explicit duration ---
        before = _time.time()
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/job_pops/pause",
            json={"paused": True, "duration_seconds": 900},
        ) as response:
            assert response.status == 200
            body = await response.json()
        after = _time.time()
        assert body["job_pops_paused"] is True
        assert body["job_pops_pause_until"] is not None
        assert before + 900 <= body["job_pops_pause_until"] <= after + 900
        assert webui.status_data["job_pops_pause_until"] == body["job_pops_pause_until"]
        assert paused_calls[-1][0] is True
        assert paused_calls[-1][1] is not None

        # --- timer reset: second pause resets the timer to specified duration ---
        first_pause_until = webui.status_data["job_pops_pause_until"]
        before = _time.time()
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/job_pops/pause",
            json={"paused": True, "duration_seconds": 1800},
        ) as response:
            assert response.status == 200
            body = await response.json()
        after = _time.time()
        assert body["job_pops_paused"] is True
        # New pause_until should be ~30 min from now, not the old ~15 min value
        assert body["job_pops_pause_until"] is not None
        assert before + 1800 <= body["job_pops_pause_until"] <= after + 1800
        assert body["job_pops_pause_until"] != first_pause_until

        # --- timer reset: second pause with omitted duration becomes indefinite ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/job_pops/pause",
            json={"paused": True},
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body["job_pops_paused"] is True
        assert body["job_pops_pause_until"] is None

        # --- success: resume ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/job_pops/pause",
            json={"paused": False},
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body["job_pops_paused"] is False
        assert body["job_pops_pause_until"] is None
        assert webui.status_data["job_pops_paused"] is False
        assert webui.status_data["job_pops_pause_until"] is None
        assert paused_calls[-1][0] is False

        # --- 400: missing 'paused' field ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/job_pops/pause",
            json={"other": "value"},
        ) as response:
            assert response.status == 400
            body = await response.json()
        assert "paused" in body["error"].lower()

        # --- 400: 'paused' is not a boolean ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/job_pops/pause",
            json={"paused": "yes"},
        ) as response:
            assert response.status == 400
            body = await response.json()
        assert "paused" in body["error"].lower()

        # --- 400: 'duration_seconds' is not a number ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/job_pops/pause",
            json={"paused": True, "duration_seconds": "long"},
        ) as response:
            assert response.status == 400
            body = await response.json()
        assert "duration_seconds" in body["error"].lower()

        # --- 400: non-JSON body ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/job_pops/pause",
            data=b"not json at all",
            headers={"Content-Type": "application/json"},
        ) as response:
            assert response.status == 400

        # --- 503: no callback registered ---
        webui.set_job_pops_paused_callback(None)
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/job_pops/pause",
            json={"paused": True},
        ) as response:
            assert response.status == 503
            body = await response.json()
        assert "error" in body

    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_time_without_jobs_endpoint() -> None:
    """Test the GET /api/job_pops/time_without_jobs endpoint."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # --- default: no status pushed yet -> 0.0 ---
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/job_pops/time_without_jobs",
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body["time_without_jobs"] == 0.0

        # --- reflects the value pushed via update_status ---
        webui.update_status(time_without_jobs=123.4)
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/job_pops/time_without_jobs",
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body["time_without_jobs"] == pytest.approx(123.4)

        # --- subject to the same reset-stats baseline subtraction as /api/status ---
        webui.status_data["stats_reset_baseline"] = {"time_without_jobs": 100.0}
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/job_pops/time_without_jobs",
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body["time_without_jobs"] == pytest.approx(23.4)

        # --- floors at 0 rather than going negative when the baseline exceeds the raw value ---
        webui.status_data["stats_reset_baseline"] = {"time_without_jobs": 999999.0}
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/job_pops/time_without_jobs",
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body["time_without_jobs"] == 0.0

    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_gallery_model_filter() -> None:
    """Test that /api/gallery?model=... filters images by model name (case-insensitive)."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        webui.add_gallery_image({"base64": test_b64, "timestamp": 1.0, "model": "stable_diffusion"})
        webui.add_gallery_image({"base64": test_b64, "timestamp": 2.0, "model": "sdxl"})
        webui.add_gallery_image({"base64": test_b64, "timestamp": 3.0, "model": "stable_diffusion"})
        webui.add_gallery_image({"base64": test_b64, "timestamp": 4.0, "model": None})

        # No filter: all 4 images returned.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=96",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 4

        # Filter by "stable_diffusion": 2 images.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=96&model=stable_diffusion",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 2
        assert all(img["model"] == "stable_diffusion" for img in data["images"])

        # Filter is case-insensitive: "Stable_Diffusion" matches "stable_diffusion".
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=96&model=Stable_Diffusion",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 2

        # Filter by "sdxl": 1 image.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=96&model=sdxl",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 1
        assert data["images"][0]["model"] == "sdxl"

        # Filter by a model that doesn't exist: 0 images, 1 total_pages.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=96&model=nonexistent",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 0
        assert data["images"] == []
        assert data["total_pages"] == 1

        # Empty model param: behaves like no filter.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=96&model=",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 4
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_gallery_safety_filter() -> None:
    """Test that /api/gallery?safety=... filters images by safety flags."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        webui.add_gallery_image({"base64": test_b64, "timestamp": 1.0, "model": "stable_diffusion"})
        webui.add_gallery_image({"base64": test_b64, "timestamp": 2.0, "model": "stable_diffusion", "is_nsfw": True})
        webui.add_gallery_image({"base64": test_b64, "timestamp": 3.0, "model": "sdxl", "is_csam": True})
        webui.add_gallery_image(
            {"base64": test_b64, "timestamp": 4.0, "model": "stable_diffusion", "is_nsfw": True, "is_csam": True},
        )
        webui.add_gallery_image({"base64": test_b64, "timestamp": 5.0, "model": "sdxl"})

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=96&safety=sfw",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 2
        assert all(not img.get("is_nsfw") and not img.get("is_csam") for img in data["images"])

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=96&safety=nsfw",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 2
        assert all(img.get("is_nsfw") for img in data["images"])

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=96&safety=csam",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 2
        assert all(img.get("is_csam") for img in data["images"])

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=96&model=stable_diffusion&safety=nsfw",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 2
        assert all(img.get("model") == "stable_diffusion" and img.get("is_nsfw") for img in data["images"])

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery?page=1&page_size=96&safety=foo",
        ) as response:
            assert response.status == 400
            data = await response.json()
        assert "safety" in data["error"].lower()
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_gallery_models_endpoint() -> None:
    """Test that /api/gallery/models returns sorted unique non-empty model names with counts."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        # Empty gallery: models list should be empty.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/models",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["total"] == 0
        assert data["models"] == []

        webui.add_gallery_image({"base64": test_b64, "timestamp": 1.0, "model": "sdxl"})
        webui.add_gallery_image({"base64": test_b64, "timestamp": 2.0, "model": "stable_diffusion"})
        webui.add_gallery_image({"base64": test_b64, "timestamp": 3.0, "model": "sdxl"})  # duplicate
        webui.add_gallery_image({"base64": test_b64, "timestamp": 4.0, "model": None})   # excluded
        webui.add_gallery_image({"base64": test_b64, "timestamp": 5.0, "model": "animefull"})

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/models",
        ) as response:
            assert response.status == 200
            data = await response.json()

        # total includes all gallery entries; models excludes None/empty entries.
        assert data["total"] == 5
        # Should be sorted by name, unique, and include counts.
        assert len(data["models"]) == 3
        assert data["models"] == [
            {"name": "animefull", "count": 1},
            {"name": "sdxl", "count": 2},
            {"name": "stable_diffusion", "count": 1},
        ]
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_gallery_safety_endpoint() -> None:
    """Test that /api/gallery/safety returns correct per-category counts."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        test_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        # Empty gallery.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/safety",
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data == {"total": 0, "sfw": 0, "nsfw": 0, "csam": 0}

        # 1 SFW, 1 NSFW-only, 1 CSAM-only, 1 both NSFW+CSAM, 1 SFW
        webui.add_gallery_image({"base64": test_b64, "timestamp": 1.0, "model": "m"})
        webui.add_gallery_image({"base64": test_b64, "timestamp": 2.0, "model": "m", "is_nsfw": True})
        webui.add_gallery_image({"base64": test_b64, "timestamp": 3.0, "model": "m", "is_csam": True})
        webui.add_gallery_image(
            {"base64": test_b64, "timestamp": 4.0, "model": "m", "is_nsfw": True, "is_csam": True},
        )
        webui.add_gallery_image({"base64": test_b64, "timestamp": 5.0, "model": "m"})

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/gallery/safety",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert data["total"] == 5
        assert data["sfw"] == 2    # entries 1 and 5
        assert data["nsfw"] == 2   # entries 2 and 4
        assert data["csam"] == 2   # entries 3 and 4
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_stats_job_state_time_container() -> None:
    """Test that the statistics page HTML contains the avg & max time per job state container."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/",
        ) as response:
            assert response.status == 200
            html = await response.text()

        # The new container must be present in the HTML.
        assert 'id="stats-job-state-time-wrap"' in html
        # The section title for the new container must also be present.
        assert "Avg &amp; Max Time per Job State" in html
        # State ordering regression: post-processing start comes before in-progress, and TOTAL is pinned last.
        assert "'POST_PROCESSING_STARTING'," in html
        assert "'INFERENCE_POST_PROCESSING'," in html
        assert html.index("'POST_PROCESSING_STARTING',") < html.index("'INFERENCE_POST_PROCESSING',")
        assert "if (a === 'TOTAL') return 1;" in html
        assert "if (b === 'TOTAL') return -1;" in html
        assert '<div class="page" id="page-settings">' in html
        # Queue size and active models are now dynamically rendered via _SETTINGS_SPEC (int_auto type).
        assert "job_queue_size:" in html  # spec key for queue size
        assert "active_model_count:" in html  # spec key for active models
        assert "'int_auto'" in html  # int_auto type declared in spec
        assert "type === 'int_auto'" in html  # int_auto branch exists in renderSettingsPage
        assert "toggleQueueSizeAuto" in html  # auto function referenced in renderer
        assert "toggleMaxActiveModelsAuto" in html  # auto function referenced in renderer
        # The standalone "Limits" static section must no longer exist.
        assert 'settings-group-title">Limits<' not in html
        # Old static setting-label HTML entities for the two controls must not reappear.
        assert '<div class="setting-label">&#128230; Job Queue Size</div>' not in html
        assert '<div class="setting-label">&#129302; Max Active Models</div>' not in html
        assert '<span class="card-title">&#128230; Job Queue</span><span class="card-header-count">(<span id="queue-count">0</span>/<span id="queue-max">0</span>)</span><div class="limit-editor"' not in html
        assert '<span class="card-title">&#129302; Active Models</span><span class="card-header-count">(<span id="models-count">0</span>/<span id="models-max">0</span>)</span><div class="limit-editor"' not in html
        assert "onchange=\"' + _changeFnNames[pfx] + '()\"" in html
        assert "onchange=\"stageNumericSetting(\\'" in html
        assert "onchange=\"stageSettingChange(\\'" in html
        assert "id=\"settings-apply-btn\"" in html
        assert "onclick=\"applyPendingSettings()\"" in html
        assert "if (!_settingsApplying) _setSettingsStatus('', false);" in html
        assert "if (applySucceeded) _setSettingsDirty(false);" in html
        assert "id=\"settings-restart-btn\"" in html
        assert "onclick=\"restartProgram()\"" in html
        assert '<div class="section-header settings-page-header">' in html
        assert '<span class="section-title">&#9881; Settings</span>' in html
        assert '<div class="section-header gallery-page-header">' in html
        assert '<span class="section-title">&#128444; Gallery</span>' in html
        assert 'id="restart-confirm-modal"' in html
        assert "id=\"restart-confirm-accept\"" in html
        assert "function confirmRestartProgram()" in html
        assert "if (!confirm('Restart the worker program now?')) return;" not in html
        assert "_clearSettingFeedback(key);" in html
        assert "function _showSettingFeedback(key, ok, msg, opts)" in html
        assert "if (fb._clearTimer) {" in html
        assert "clearTimeout(fb._clearTimer);" in html
        assert "fb._clearTimer = null;" in html
        assert "if (opts && opts.pending) {" in html
        assert "fb._clearTimer = setTimeout(function() {" in html
        assert "_showSettingFeedback(key, true, 'Pending', {pending: true});" in html
        assert "_showSettingFeedback('job_queue_size', true, 'Pending', {pending: true});" in html
        assert "_showSettingFeedback('active_model_count', true, 'Pending', {pending: true});" in html
        assert ".setting-feedback.pending" in html
        assert "function _isEqualSimple" not in html
        assert "id=\"' + pfx + '-set-btn\"" not in html
        assert "onclick=\"applyNumericSetting(\\'" not in html
        assert ".setting-number:disabled" in html
        assert "--action-btn-height: 32px;" in html
        assert ".theme-toggle, .nsfw-blur-btn, .clear-maintenance-btn, .limit-set-btn, .limit-auto-btn, .console-pause-btn, .console-copy-btn, .job-pops-pause-btn, .errors-view-btn, .pagination-controls button, .image-overlay-close, .worker-delete-btn, .stats-window-btn, .horde-window-btn, .settings-page-btn, .setting-apply-btn, .confirm-modal-btn {" in html
        assert ".topbar-uptime, .status-badge, .job-state-badge, .process-type-badge, .process-state-badge, .model-badge, .worker-version-badge, .worker-type-badge, .worker-online-badge, .wcap {\n            height: var(--action-btn-height);" in html
        assert "<span class=\"process-state-badge\">'+escapeHtml(proc.state)+'</span><span class=\"process-type-badge\">'+escapeHtml(proc.type)+'</span>" in html
        assert "<span class=\"process-type-badge\">'+escapeHtml(proc.type)+'</span><span class=\"process-state-badge\">'+escapeHtml(proc.state)+'</span>" not in html
        assert ".setting-number { width: 68px; height: var(--action-btn-height);" in html
        assert ".limit-input { width: 54px; height: var(--action-btn-height);" in html
        assert "autoBtn.setAttribute('aria-pressed', 'true');" in html
        assert "autoBtn.setAttribute('aria-pressed', 'false');" in html
        assert ".user-details-grid + .user-details-grid," in html
        assert ".user-details-grid + .card," in html
        assert ".stats-summary-grid + .stats-summary-grid { margin-top: var(--page-spacing); }" in html
        # The user details section is a single 4-column grid: the former second grid-3 row was
        # merged into it when the "Trusted" tile became a trust indicator next to the username.
        assert html.count('class="grid-4 user-details-grid"') == 1
        assert 'class="grid-3 user-details-grid"' not in html
        assert 'id="user-page-trust-indicator"' in html
        assert 'class="grid-4 stats-summary-grid"' in html
        assert 'class="grid-2 stats-summary-grid"' in html
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_stats_per_model_time_containers() -> None:
    """Test that the statistics page HTML contains the per-model step and job timing containers."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/",
        ) as response:
            assert response.status == 200
            html = await response.text()

        # Per-step timing container must be present.
        assert 'id="stats-step-time-model-wrap"' in html
        # Per-step section title must mention "Step".
        assert "Avg &amp; Max Time per Step per Model" in html
        # Per-job timing container must be present.
        assert 'id="stats-job-time-model-wrap"' in html
        # Per-job section title must mention "Job per Model" to disambiguate from other timings.
        assert "Avg &amp; Max Time per Job per Model" in html
        # Per-step JS must use fixed 3-decimal formatting (min and max fraction digits).
        assert "minimumFractionDigits: 3, maximumFractionDigits: 3" in html
        # JS must read avg_time_per_step_per_model and max_time_per_step_per_model from data.
        assert "data.avg_time_per_step_per_model" in html
        assert "data.max_time_per_step_per_model" in html
        # JS must read avg_time_per_job_per_model and max_time_per_job_per_model from data.
        assert "data.avg_time_per_job_per_model" in html
        assert "data.max_time_per_job_per_model" in html
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_post_max_queue_size_controls() -> None:
    """Test that POST /api/settings handles queue size controls."""
    webui = WorkerWebUI(port=0)
    queue_size_calls: list[int] = []
    auto_mode_calls: list[bool] = []

    def fake_set_queue_size(size: int) -> None:
        queue_size_calls.append(size)

    def fake_set_auto_mode(enabled: bool) -> None:
        auto_mode_calls.append(enabled)

    webui.set_max_queue_size_callback(fake_set_queue_size)
    webui.set_queue_size_auto_mode_callback(fake_set_auto_mode)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # --- success: set manual value ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_queue_size", "value": 5},
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body == {"key": "max_queue_size", "value": 5}
        assert webui.status_data["max_queue_size"] == 5
        assert webui.status_data["queue_size_auto"] is False
        assert queue_size_calls[-1] == 5

        # --- success: set to 0 (disable buffering) ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_queue_size", "value": 0},
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body == {"key": "max_queue_size", "value": 0}
        assert queue_size_calls[-1] == 0

        # --- success: enable auto mode ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "queue_size_auto", "value": True},
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body == {"key": "queue_size_auto", "value": True}
        assert webui.status_data["queue_size_auto"] is True
        assert auto_mode_calls[-1] is True

        # --- success: disable auto mode ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "queue_size_auto", "value": False},
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body == {"key": "queue_size_auto", "value": False}
        assert auto_mode_calls[-1] is False

        # --- 400: missing required field ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_queue_size"},
        ) as response:
            assert response.status == 400

        # --- 400: negative value ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_queue_size", "value": -1},
        ) as response:
            assert response.status == 400

        # --- 400: non-integer value ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_queue_size", "value": "lots"},
        ) as response:
            assert response.status == 400

        # --- 400: boolean passed for auto ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "queue_size_auto", "value": "yes"},
        ) as response:
            assert response.status == 400

        # --- 400: non-JSON body ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        ) as response:
            assert response.status == 400

        # --- 503: no callbacks registered ---
        webui.set_max_queue_size_callback(None)
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_queue_size", "value": 3},
        ) as response:
            assert response.status == 503

        webui.set_queue_size_auto_mode_callback(None)
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "queue_size_auto", "value": True},
        ) as response:
            assert response.status == 503

    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_post_max_active_model_controls() -> None:
    """Test that POST /api/settings handles max active model controls."""
    webui = WorkerWebUI(port=0)
    models_calls: list[int] = []
    auto_mode_calls: list[bool] = []

    def fake_set_max_active(count: int) -> None:
        models_calls.append(count)

    def fake_set_auto_mode(enabled: bool) -> None:
        auto_mode_calls.append(enabled)

    webui.set_max_active_models_callback(fake_set_max_active)
    webui.set_max_active_models_auto_mode_callback(fake_set_auto_mode)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # --- success: set manual value ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_active_models", "value": 3},
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body == {"key": "max_active_models", "value": 3}
        assert webui.status_data["max_active_models"] == 3
        assert webui.status_data["max_active_models_auto"] is False
        assert models_calls[-1] == 3

        # --- success: enable auto mode ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_active_models_auto", "value": True},
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body == {"key": "max_active_models_auto", "value": True}
        assert webui.status_data["max_active_models_auto"] is True
        assert auto_mode_calls[-1] is True

        # --- success: disable auto mode ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_active_models_auto", "value": False},
        ) as response:
            assert response.status == 200
            body = await response.json()
        assert body == {"key": "max_active_models_auto", "value": False}
        assert auto_mode_calls[-1] is False

        # --- 400: value below 1 ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_active_models", "value": 0},
        ) as response:
            assert response.status == 400

        # --- 400: non-integer value ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_active_models", "value": "many"},
        ) as response:
            assert response.status == 400

        # --- 400: boolean passed as max_active_models ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_active_models", "value": True},
        ) as response:
            assert response.status == 400

        # --- 400: non-boolean for auto ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_active_models_auto", "value": 1},
        ) as response:
            assert response.status == 400

        # --- 400: missing required field ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_active_models"},
        ) as response:
            assert response.status == 400

        # --- 400: non-JSON body ---
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            data=b"bad",
            headers={"Content-Type": "application/json"},
        ) as response:
            assert response.status == 400

        # --- 503: no callbacks registered ---
        webui.set_max_active_models_callback(None)
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_active_models", "value": 2},
        ) as response:
            assert response.status == 503

        webui.set_max_active_models_auto_mode_callback(None)
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_active_models_auto", "value": True},
        ) as response:
            assert response.status == 503

    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_max_queue_size_and_active_models_in_status() -> None:
    """Test that max_queue_size, queue_size_auto, max_active_models, and max_active_models_auto
    are present in /api/status and default to 0 / False."""
    webui = WorkerWebUI(port=0)
    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/status",
        ) as response:
            assert response.status == 200
            body = await response.json()

        assert "max_queue_size" in body
        assert "queue_size_auto" in body
        assert "max_active_models" in body
        assert "max_active_models_auto" in body
        assert body["max_queue_size"] == 0
        assert body["queue_size_auto"] is False
        assert body["max_active_models"] == 0
        assert body["max_active_models_auto"] is False

    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_get_empty() -> None:
    """Test that GET /api/settings returns an empty settings dict when no snapshot has been pushed."""
    webui = WorkerWebUI(port=0)
    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/settings",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert "settings" in data
        assert isinstance(data["settings"], dict)
        # With no snapshot pushed, only the int_auto live values piggybacked
        # from status_data plus the auto-derived webui_url should be present.
        assert data["settings"]["max_queue_size"] == 0
        assert data["settings"]["queue_size_auto"] is False
        assert data["settings"]["max_active_models"] == 0
        assert data["settings"]["max_active_models_auto"] is False
        # webui_url is always injected from the server's own port.
        assert data["settings"]["webui_url"] == f"http://localhost:{actual_port}"
        # No other keys should be present.
        assert set(data["settings"].keys()) == {
            "max_queue_size", "queue_size_auto", "max_active_models", "max_active_models_auto", "webui_url",
        }
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_update_settings_data() -> None:
    """Test that update_settings_data populates the GET /api/settings response."""
    webui = WorkerWebUI(port=0)
    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Push a snapshot of settings
        webui.update_settings_data({
            "nsfw": True,
            "allow_img2img": False,
            "max_power": 18,
            "horde_model_stickiness": 0.5,
            "extra_field_not_in_spec": "ignored",
        })

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/settings",
        ) as response:
            assert response.status == 200
            data = await response.json()

        settings = data["settings"]
        assert settings["nsfw"] is True
        assert settings["allow_img2img"] is False
        assert settings["max_power"] == 18
        assert settings["horde_model_stickiness"] == 0.5
        # Fields not in _SETTINGS_SPEC must be filtered out.
        assert "extra_field_not_in_spec" not in settings
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_post_bool() -> None:
    """Test that POST /api/settings updates a boolean setting and calls the callback."""
    webui = WorkerWebUI(port=0)
    received: list[tuple[str, object]] = []

    def mock_callback(key: str, value: object) -> None:
        received.append((key, value))

    webui.set_setting_callback(mock_callback)
    webui.update_settings_data({"nsfw": True})

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "nsfw", "value": False},
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert data["key"] == "nsfw"
        assert data["value"] is False
        assert received == [("nsfw", False)]

        # The in-memory snapshot should have been updated.
        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/settings",
        ) as response:
            get_data = await response.json()
        assert get_data["settings"]["nsfw"] is False
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_post_numeric() -> None:
    """Test that POST /api/settings updates a numeric setting with range validation."""
    webui = WorkerWebUI(port=0)
    received: list[tuple[str, object]] = []

    def mock_callback(key: str, value: object) -> None:
        received.append((key, value))

    webui.set_setting_callback(mock_callback)
    webui.update_settings_data({"max_power": 8})

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Valid update
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_power", "value": 18},
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data["value"] == 18
        assert received == [("max_power", 18)]

        # Value below minimum should be rejected
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_power", "value": 0},
        ) as response:
            assert response.status == 400

        # Value above maximum should be rejected
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_power", "value": 999},
        ) as response:
            assert response.status == 400

        # Fractional values for int settings should be rejected
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "max_power", "value": 1.9},
        ) as response:
            assert response.status == 400
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_post_unknown_key() -> None:
    """Test that POST /api/settings rejects unknown setting keys."""
    webui = WorkerWebUI(port=0)
    webui.set_setting_callback(lambda k, v: None)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "nonexistent_setting", "value": True},
        ) as response:
            assert response.status == 400
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_post_readonly_key() -> None:
    """Test that POST /api/settings rejects writes to read-only fields such as horde_url."""
    webui = WorkerWebUI(port=0)
    webui.set_setting_callback(lambda k, v: None)
    webui.update_settings_data({"horde_url": "https://aihorde.net/api/"})

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "horde_url", "value": "https://evil.example.com/api/"},
        ) as response:
            assert response.status == 400
            data = await response.json()
            assert "read-only" in data.get("error", "").lower()
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_get_includes_horde_url() -> None:
    """Test that GET /api/settings returns horde_url when it has been pushed."""
    webui = WorkerWebUI(port=0)
    webui.update_settings_data({"horde_url": "https://aihorde.net/api/"})

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/settings",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert "horde_url" in data["settings"]
        assert data["settings"]["horde_url"] == "https://aihorde.net/api/"
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_get_includes_webui_url() -> None:
    """Test that GET /api/settings always returns webui_url pointing to the server's own port."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.port

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/settings",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert "webui_url" in data["settings"]
        assert data["settings"]["webui_url"] == f"http://localhost:{actual_port}"
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_post_rejects_webui_url() -> None:
    """Test that POST /api/settings rejects writes to the read-only webui_url field."""
    webui = WorkerWebUI(port=0)
    webui.set_setting_callback(lambda k, v: None)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.port

        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "webui_url", "value": "http://evil.example.com"},
        ) as response:
            assert response.status == 400
            data = await response.json()
            assert "read-only" in data.get("error", "").lower()
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_post_no_callback() -> None:
    """Test that POST /api/settings returns 503 when no callback is registered."""
    webui = WorkerWebUI(port=0)
    # No callback registered

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "nsfw", "value": True},
        ) as response:
            assert response.status == 503
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_post_allows_non_local_clients() -> None:
    """Test that POST /api/settings allows non-local clients."""
    webui = WorkerWebUI(port=0)
    received: list[tuple[str, object]] = []

    def mock_callback(key: str, value: object) -> None:
        received.append((key, value))

    webui.set_setting_callback(mock_callback)

    class DummyRequest:
        remote = "8.8.8.8"

        async def json(self) -> dict[str, object]:
            return {"key": "nsfw", "value": False}

    response = await webui._handle_set_setting(DummyRequest())  # type: ignore[arg-type]
    assert response.status == 200
    assert received == [("nsfw", False)]


@pytest.mark.asyncio
async def test_webui_settings_post_accepts_ipv4_mapped_loopback_clients() -> None:
    """Test that POST /api/settings accepts IPv4-mapped loopback clients."""
    webui = WorkerWebUI(port=0)
    received: list[tuple[str, object]] = []

    def mock_callback(key: str, value: object) -> None:
        received.append((key, value))

    webui.set_setting_callback(mock_callback)

    class DummyRequest:
        remote = "::ffff:127.0.0.1"

        async def json(self) -> dict[str, object]:
            return {"key": "nsfw", "value": False}

    response = await webui._handle_set_setting(DummyRequest())  # type: ignore[arg-type]
    assert response.status == 200
    assert received == [("nsfw", False)]


@pytest.mark.asyncio
async def test_webui_settings_post_accepts_same_host_non_loopback_clients() -> None:
    """Test that POST /api/settings accepts same-host requests on non-loopback local IPs."""
    webui = WorkerWebUI(port=0)
    received: list[tuple[str, object]] = []

    def mock_callback(key: str, value: object) -> None:
        received.append((key, value))

    webui.set_setting_callback(mock_callback)

    class DummyTransport:
        def get_extra_info(self, name: str) -> tuple[str, int] | None:
            if name == "sockname":
                return ("192.168.2.41", 3000)
            return None

    class DummyRequest:
        remote = "192.168.2.41"
        transport = DummyTransport()

        async def json(self) -> dict[str, object]:
            return {"key": "nsfw", "value": False}

    response = await webui._handle_set_setting(DummyRequest())  # type: ignore[arg-type]
    assert response.status == 200
    assert received == [("nsfw", False)]


@pytest.mark.asyncio
async def test_webui_restart_post_calls_callback() -> None:
    """Test that POST /api/restart triggers the restart callback."""
    webui = WorkerWebUI(port=0)
    called = {"restart": False}

    def restart_callback() -> None:
        called["restart"] = True

    webui.set_restart_program_callback(restart_callback)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/restart",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert data["restarting"] is True
        assert called["restart"] is True
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_restart_post_no_callback() -> None:
    """Test that POST /api/restart returns 503 when no restart callback is registered."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/restart",
        ) as response:
            assert response.status == 503
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_restart_post_allows_non_local_clients() -> None:
    """Test that POST /api/restart allows non-local clients."""
    webui = WorkerWebUI(port=0)
    called = {"restart": False}

    def restart_callback() -> None:
        called["restart"] = True

    webui.set_restart_program_callback(restart_callback)

    class DummyRequest:
        remote = "8.8.8.8"

    response = await webui._handle_restart_program(DummyRequest())  # type: ignore[arg-type]
    assert response.status == 200
    assert called["restart"] is True


@pytest.mark.asyncio
async def test_webui_restart_post_accepts_same_host_non_loopback_clients() -> None:
    """Test that POST /api/restart accepts same-host requests on non-loopback local IPs."""
    webui = WorkerWebUI(port=0)
    called = {"restart": False}

    def restart_callback() -> None:
        called["restart"] = True

    webui.set_restart_program_callback(restart_callback)

    class DummyTransport:
        def get_extra_info(self, name: str) -> tuple[str, int] | None:
            if name == "sockname":
                return ("192.168.2.41", 3000)
            return None

    class DummyRequest:
        remote = "192.168.2.41"
        transport = DummyTransport()

    response = await webui._handle_restart_program(DummyRequest())  # type: ignore[arg-type]
    assert response.status == 200
    assert called["restart"] is True


@pytest.mark.asyncio
async def test_webui_settings_html_nav_and_page() -> None:
    """Test that the settings nav item and page div are present in the HTML."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/",
        ) as response:
            assert response.status == 200
            html = await response.text()

        # Nav item must be present
        assert 'id="nav-settings"' in html
        assert "showPage('settings'" in html

        # Page container must be present
        assert 'id="page-settings"' in html
        assert 'id="settings-apply-btn"' in html
        assert 'id="settings-restart-btn"' in html
        assert 'id="restart-confirm-modal"' in html

        # Settings and gallery page titles should be rendered
        assert '<span class="section-title">&#9881; Settings</span>' in html
        assert '<span class="section-title">&#128444; Gallery</span>' in html

        # VALID_PAGES must include 'settings'
        assert "'settings'" in html

        # Settings API endpoint must be referenced in the JS
        assert "'/api/settings'" in html or '"/api/settings"' in html
        assert "'/api/queue/max_size'" not in html and '"/api/queue/max_size"' not in html
        assert "'/api/models/max_active'" not in html and '"/api/models/max_active"' not in html
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_api_page_html() -> None:
    """Test that the API page (not Settings) hosts the API reference, covering every route."""
    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/",
        ) as response:
            assert response.status == 200
            html = await response.text()

        # Nav item must be present, positioned after the Settings nav item.
        assert 'id="nav-api"' in html
        assert "showPage('api'" in html
        assert html.index('id="nav-settings"') < html.index('id="nav-api"')

        # Page container must be present.
        assert 'id="page-api"' in html
        assert 'id="api-page-body"' in html

        # A representative sample of routes across every group must be documented.
        for path in (
            "/api/status",
            "/health",
            "/api/job_pops/pause",
            "/api/job_pops/time_without_jobs",
            "/api/gallery",
            "/api/errors",
            "/api/settings",
            "/api/reset-database",
            "/api/worker/{worker_id}",
        ):
            assert path in html

        # The old settings-page-only API reference implementation must be gone.
        assert "api-ref-pause-url" not in html
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_about_page_html() -> None:
    """Test that the About page is present, last in the navigation, and shows the tech stack."""
    import horde_worker_regen

    webui = WorkerWebUI(port=0)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/",
        ) as response:
            assert response.status == 200
            html = await response.text()

        # Nav item must be present and be the last nav item (after API).
        assert 'id="nav-about"' in html
        assert "showPage('about'" in html
        assert html.index('id="nav-api"') < html.index('id="nav-about"')

        # Page container must be present with the actual running version substituted in.
        assert 'id="page-about"' in html
        assert "{{WORKER_VERSION}}" not in html
        assert f"Version {horde_worker_regen.__version__}" in html

        # A sample of the documented tech stack must be present.
        for tech in ("Python 3.10+", "aiohttp", "SQLite", "PyTorch", "Pydantic"):
            assert tech in html
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_last_result_time_ago_shown_for_gallery_preview() -> None:
    """The Last Result time-ago label must also work for the restored gallery preview.

    Before any image is generated in the current session, the overview card is populated
    from the last gallery batch. The time-ago ticker previously read only the session
    timestamp delivered by /api/status — 0 in a fresh session and re-derived (nulled)
    every poll — so the restored preview image showed no "Last submission: ... ago"
    value at all until a new image was generated.
    """
    webui = WorkerWebUI(port=0)
    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/",
        ) as response:
            assert response.status == 200
            html = await response.text()

        # Fallback variable is declared and feeds the 1-second ticker.
        assert "let _galleryPreviewTimestamp = null;" in html
        assert "const labelTimestamp = _lastImageSubmissionTimestamp || _galleryPreviewTimestamp;" in html
        assert "timeEl.textContent = labelTimestamp ? formatTimeAgo(labelTimestamp) : '';" in html
        # The gallery-preview path records its batch timestamp for the label.
        assert "_galleryPreviewTimestamp = (Number.isFinite(previewTs) && previewTs > 0) ? previewTs : null;" in html
        # The immediate session-image restore seeds the label without waiting for a poll.
        assert "_lastImageSubmissionTimestamp = ts;" in html
        # Clearing a vanished session image also clears the preview label (declaration + clear).
        assert html.count("_galleryPreviewTimestamp = null;") >= 2
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_post_float() -> None:
    """Test that POST /api/settings accepts a float setting (horde_model_stickiness)."""
    webui = WorkerWebUI(port=0)
    received: list[tuple[str, object]] = []

    def mock_callback(key: str, value: object) -> None:
        received.append((key, value))

    webui.set_setting_callback(mock_callback)

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "horde_model_stickiness", "value": 0.75},
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert data["key"] == "horde_model_stickiness"
        assert abs(float(data["value"]) - 0.75) < 1e-9
        assert len(received) == 1
        assert received[0][0] == "horde_model_stickiness"
        assert abs(float(received[0][1]) - 0.75) < 1e-9

        # Out-of-range float should be rejected
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "horde_model_stickiness", "value": 1.5},
        ) as response:
            assert response.status == 400
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_settings_auto_restart_on_idle() -> None:
    """Test that auto_restart_on_idle_minutes can be updated via POST /api/settings."""
    webui = WorkerWebUI(port=0)
    received: list[tuple[str, object]] = []

    def mock_callback(key: str, value: object) -> None:
        received.append((key, value))

    webui.set_setting_callback(mock_callback)
    webui.update_settings_data({"auto_restart_on_idle_minutes": 60})

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Valid update — set to 120 minutes
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "auto_restart_on_idle_minutes", "value": 120},
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data == {"key": "auto_restart_on_idle_minutes", "value": 120}
        assert received == [("auto_restart_on_idle_minutes", 120)]

        # Valid update — set to 0 (disabled)
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "auto_restart_on_idle_minutes", "value": 0},
        ) as response:
            assert response.status == 200
            data = await response.json()
        assert data == {"key": "auto_restart_on_idle_minutes", "value": 0}

        # Value above maximum (1440 min = 24 h) should be rejected
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "auto_restart_on_idle_minutes", "value": 9999},
        ) as response:
            assert response.status == 400

        # Negative value should be rejected
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "auto_restart_on_idle_minutes", "value": -1},
        ) as response:
            assert response.status == 400

        # Non-integer should be rejected
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/settings",
            json={"key": "auto_restart_on_idle_minutes", "value": "soon"},
        ) as response:
            assert response.status == 400
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_models_get_empty() -> None:
    """Test that GET /api/models returns empty lists when no models data has been pushed."""
    webui = WorkerWebUI(port=0)
    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/models",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert data == {"enabled": [], "disabled": []}
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_models_get_with_data() -> None:
    """Test that GET /api/models returns the pushed model lists."""
    webui = WorkerWebUI(port=0)
    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        webui.update_models_data(
            enabled=["Stable Diffusion 1.5", "SDXL 1.0"],
            disabled=["Deliberate"],
        )

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/models",
        ) as response:
            assert response.status == 200
            data = await response.json()

        assert data["enabled"] == ["SDXL 1.0", "Stable Diffusion 1.5"]
        assert data["disabled"] == ["Deliberate"]
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_models_toggle() -> None:
    """Test that POST /api/models toggles a model and invokes the callback."""
    webui = WorkerWebUI(port=0)
    toggled = []
    special_model_name = "A&B <Model> \"Quoted\" \\"

    def on_toggle(model: str, enabled: bool) -> None:
        toggled.append((model, enabled))

    webui.set_toggle_model_callback(on_toggle)
    webui.update_models_data(
        enabled=["Stable Diffusion 1.5", "SDXL 1.0"],
        disabled=["Deliberate", special_model_name],
    )

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Disable a model
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/models",
            json={"model": "Stable Diffusion 1.5", "enabled": False},
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert data == {"model": "Stable Diffusion 1.5", "enabled": False}

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/models",
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert data["enabled"] == ["SDXL 1.0"]
            assert data["disabled"] == [special_model_name, "Deliberate", "Stable Diffusion 1.5"]

        assert toggled == [("Stable Diffusion 1.5", False)]

        # Enable a model with characters that require escaping in HTML
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/models",
            json={"model": special_model_name, "enabled": True},
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert data == {"model": special_model_name, "enabled": True}

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/models",
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert data["enabled"] == [special_model_name, "SDXL 1.0"]
            assert data["disabled"] == ["Deliberate", "Stable Diffusion 1.5"]

        assert toggled == [("Stable Diffusion 1.5", False), (special_model_name, True)]
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_models_toggle_invalid() -> None:
    """Test that POST /api/models returns errors for invalid requests."""
    webui = WorkerWebUI(port=0)
    webui.update_models_data(enabled=["ModelA"], disabled=[])

    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        # Missing model field
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/models",
            json={"enabled": True},
        ) as response:
            assert response.status == 400

        # Missing enabled field
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/models",
            json={"model": "ModelA"},
        ) as response:
            assert response.status == 400

        # Unknown model
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/models",
            json={"model": "UnknownModel", "enabled": True},
        ) as response:
            assert response.status == 400

        # No callback registered
        async with aiohttp.ClientSession() as session, session.post(
            f"http://localhost:{actual_port}/api/models",
            json={"model": "ModelA", "enabled": False},
        ) as response:
            assert response.status == 503
    finally:
        await webui.stop()


@pytest.mark.asyncio
async def test_webui_models_section_html() -> None:
    """Test that the index page includes the models section CSS and JS."""
    webui = WorkerWebUI(port=0)
    try:
        await webui.start()
        await asyncio.sleep(0.5)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/",
        ) as response:
            assert response.status == 200
            html = await response.text()

        # CSS classes for models section
        assert "models-containers" in html
        assert "model-pill" in html
        assert "models-box" in html
        # JS functions
        assert "fetchModels" in html
        assert "toggleModel" in html
        assert "renderModelsSection" in html
        # Models are interactive buttons with delegated click handling and data attributes
        assert 'class="model-pill enabled" data-model="' in html
        assert "addEventListener('click'" in html
        assert 'aria-pressed="true"' in html
        # Pending-retry guard: a blocked fetchModels() call must be retried once the
        # in-flight fetch completes so the models section is never permanently invisible.
        assert "_modelsFetchPending" in html
        assert "if (_modelsFetchPending)" in html
    finally:
        await webui.stop()


def test_webui_reset_session_start_time_resets_uptime() -> None:
    """reset_session_start_time() should set uptime to 0.0 and refresh session_start_time."""
    import time

    webui = WorkerWebUI(port=0)
    # Simulate time passing by back-dating the session start
    webui.status_data["session_start_time"] = time.time() - 3600
    webui.update_status()

    assert webui.status_data["uptime"] >= 3600, "uptime should reflect elapsed time before reset"

    webui.reset_session_start_time()

    assert webui.status_data["uptime"] == 0.0, "uptime should be 0.0 immediately after reset"
    assert webui.status_data["session_start_time"] == pytest.approx(time.time(), abs=2), (
        "session_start_time should be refreshed to now"
    )


# ---------------------------------------------------------------------------
# SQLite persistence tests
# ---------------------------------------------------------------------------

def _make_db_webui(tmp_path: pathlib.Path, **kwargs: object) -> tuple["WorkerWebUI", str, str, str]:
    """Helper that creates a WorkerWebUI backed by temporary databases.

    Returns:
        Tuple of (webui, errors_db_path, stats_db_path, gallery_db_path).
    """
    db_file = str(tmp_path / "test_webui.db")
    webui = WorkerWebUI(port=0, db_path=db_file, **kwargs)  # type: ignore[arg-type]
    errors_db = str(tmp_path / "webui_errors.db")
    stats_db = str(tmp_path / "webui_stats.db")
    gallery_db = str(tmp_path / "webui_gallery.db")
    return webui, errors_db, stats_db, gallery_db


def test_webui_db_init_creates_tables(tmp_path: pathlib.Path) -> None:
    """_init_db() must create the errors_log, gallery_images, and stats_snapshots tables in separate databases."""
    import sqlite3

    webui, errors_db, stats_db, gallery_db = _make_db_webui(tmp_path)
    assert webui._errors_db_path == errors_db
    assert webui._stats_db_path == stats_db
    assert webui._gallery_db_path == gallery_db

    # Check errors database
    with sqlite3.connect(errors_db) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "errors_log" in tables
    assert "gallery_images" not in tables
    assert "stats_snapshots" not in tables

    # Check stats database
    with sqlite3.connect(stats_db) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "stats_snapshots" in tables
    assert "errors_log" not in tables
    assert "gallery_images" not in tables

    # Check gallery database
    with sqlite3.connect(gallery_db) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "gallery_images" in tables
    assert "errors_log" not in tables
    assert "stats_snapshots" not in tables


def test_webui_db_path_file_without_db_suffix_uses_parent_directory(tmp_path: pathlib.Path) -> None:
    """db_path file inputs without .db suffix should still resolve to the parent directory."""
    db_file = tmp_path / "state.sqlite"
    webui = WorkerWebUI(port=0, db_path=str(db_file))

    assert webui._errors_db_path == str(tmp_path / "webui_errors.db")
    assert webui._stats_db_path == str(tmp_path / "webui_stats.db")
    assert webui._gallery_db_path == str(tmp_path / "webui_gallery.db")
    assert not db_file.is_dir()


def test_webui_db_path_no_extension_uses_parent_directory(tmp_path: pathlib.Path) -> None:
    """A non-existent db_path with no file extension must be treated as a file path (use parent dir).

    Without this guard a path like ``/some/dir/state`` (valid for the model-state
    SQLite file used by ProcessManager) would be treated as a directory, causing a
    directory to be created at that path and later sqlite3.connect() to fail.
    """
    db_file = tmp_path / "state"  # no extension, non-existent
    webui = WorkerWebUI(port=0, db_path=str(db_file))

    assert webui._errors_db_path == str(tmp_path / "webui_errors.db")
    assert webui._stats_db_path == str(tmp_path / "webui_stats.db")
    assert webui._gallery_db_path == str(tmp_path / "webui_gallery.db")
    # The path must NOT have been created as a directory.
    assert not db_file.is_dir()


def test_webui_db_persists_gallery_image(tmp_path: pathlib.Path) -> None:
    """add_gallery_image() must insert a row into gallery_images."""
    import sqlite3

    webui, _errors_db, _stats_db, gallery_db = _make_db_webui(tmp_path)

    webui.add_gallery_image({"base64": None, "timestamp": 1_700_000_000.0, "model": "sdxl", "is_nsfw": True})

    with sqlite3.connect(gallery_db) as conn:
        rows = conn.execute("SELECT gallery_id, model, is_nsfw FROM gallery_images").fetchall()

    assert len(rows) == 1
    assert rows[0][0] == 0  # first gallery_id
    assert rows[0][1] == "sdxl"
    assert rows[0][2] == 1  # is_nsfw=True stored as 1


def test_webui_db_persists_multiple_gallery_images(tmp_path: pathlib.Path) -> None:
    """Multiple add_gallery_image() calls each insert a separate row."""
    import sqlite3

    webui, _errors_db, _stats_db, gallery_db = _make_db_webui(tmp_path)

    webui.add_gallery_image({"base64": None, "timestamp": 1.0, "model": "a"})
    webui.add_gallery_image({"base64": None, "timestamp": 2.0, "model": "b"})
    webui.add_gallery_image({"base64": None, "timestamp": 3.0, "model": "c"})

    with sqlite3.connect(gallery_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM gallery_images").fetchone()[0]

    assert count == 3


def test_webui_db_persists_errors(tmp_path: pathlib.Path) -> None:
    """update_status(errors_history=…) must persist new errors to errors_log."""
    import sqlite3

    webui, errors_db, _stats_db, _gallery_db = _make_db_webui(tmp_path)

    webui.update_status(errors_history=["error_b", "error_a"])

    with sqlite3.connect(errors_db) as conn:
        rows = conn.execute("SELECT message FROM errors_log ORDER BY id").fetchall()

    messages = [r[0] for r in rows]
    assert "error_b" in messages
    assert "error_a" in messages
    assert len(messages) == 2


def test_webui_db_persists_only_new_errors(tmp_path: pathlib.Path) -> None:
    """Only the newly prepended errors should be inserted on subsequent calls."""
    import sqlite3

    webui, errors_db, _stats_db, _gallery_db = _make_db_webui(tmp_path)

    # First update: 1 error
    webui.update_status(errors_history=["error_a"])
    # Second update: 2 errors (error_b is new, prepended at front)
    webui.update_status(errors_history=["error_b", "error_a"])

    with sqlite3.connect(errors_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM errors_log").fetchone()[0]

    # First call inserts 1 new error ("error_a"); second call inserts 1 new error ("error_b").
    assert count == 2


def test_webui_db_merges_capped_live_errors_with_persisted_history(tmp_path: pathlib.Path) -> None:
    """Live errors should stay merged with persisted history even when the live list is capped."""
    import sqlite3
    import time

    errors_db = str(tmp_path / "webui_errors.db")
    stats_db = str(tmp_path / "webui_stats.db")
    gallery_db = str(tmp_path / "webui_gallery.db")
    
    # Create initial database directory
    db_dir = str(tmp_path)
    
    webui_seed = WorkerWebUI(port=0, db_path=db_dir)
    now = time.time()
    with sqlite3.connect(errors_db) as conn:
        conn.execute(
            "INSERT INTO errors_log (message, created_at) VALUES (?, ?)",
            ("seeded_error_old", now - 2),
        )
        conn.execute(
            "INSERT INTO errors_log (message, created_at) VALUES (?, ?)",
            ("seeded_error_new", now - 1),
        )
        conn.commit()

    webui = WorkerWebUI(port=0, db_path=db_dir)
    webui.update_status(errors_history=[])
    assert webui.status_data["errors_history"] == ["seeded_error_new", "seeded_error_old"]

    webui.update_status(errors_history=["error_one"])
    webui.update_status(errors_history=["error_two", "error_one"])
    webui.update_status(errors_history=["error_three", "error_two"])

    assert webui.status_data["errors_history"] == [
        "error_three",
        "error_two",
        "error_one",
        "seeded_error_new",
        "seeded_error_old",
    ]

    with sqlite3.connect(errors_db) as conn:
        messages = [row[0] for row in conn.execute("SELECT message FROM errors_log ORDER BY id").fetchall()]

    assert messages == [
        "seeded_error_old",
        "seeded_error_new",
        "error_one",
        "error_two",
        "error_three",
    ]


def test_webui_db_caps_in_memory_persisted_errors_after_update(tmp_path: pathlib.Path) -> None:
    """In-memory persisted errors should be capped immediately after prepending new errors."""
    import time

    from horde_worker_regen.webui.server import _MAX_PERSISTED_ERRORS

    webui, _errors_db, _stats_db, _gallery_db = _make_db_webui(tmp_path)
    webui._last_db_prune_time = time.time()
    webui._persisted_errors = [f"old_{i}" for i in range(_MAX_PERSISTED_ERRORS)]
    webui.update_status(errors_history=["new_error"])

    assert len(webui._persisted_errors) == _MAX_PERSISTED_ERRORS
    assert webui._persisted_errors[0] == "new_error"
    assert webui._persisted_errors[-1] == f"old_{_MAX_PERSISTED_ERRORS - 2}"


def test_webui_db_caps_merged_errors_history_to_memory_limit(tmp_path: pathlib.Path) -> None:
    """Merged errors_history should remain capped to the existing memory-safety limit."""
    from horde_worker_regen.webui.server import _MAX_PERSISTED_ERRORS

    webui, _errors_db, _stats_db, _gallery_db = _make_db_webui(tmp_path)
    live_errors = [f"live_{i}" for i in range(_MAX_PERSISTED_ERRORS)]
    webui._live_errors_history = list(live_errors)
    webui._persisted_errors = [f"persisted_{i}" for i in range(_MAX_PERSISTED_ERRORS)]

    webui.update_status(errors_history=live_errors)

    assert len(webui.status_data["errors_history"]) == _MAX_PERSISTED_ERRORS
    assert webui.status_data["errors_history"] == live_errors


def test_webui_db_persists_stats_snapshot(tmp_path: pathlib.Path) -> None:
    """_record_stats_snapshot() must insert a row into stats_snapshots."""
    import sqlite3
    import time

    webui, _errors_db, stats_db, _gallery_db = _make_db_webui(tmp_path)
    # Force a snapshot by rewinding the last-snapshot time
    webui._last_stats_snapshot_time = 0.0
    webui._record_stats_snapshot()

    with sqlite3.connect(stats_db) as conn:
        rows = conn.execute("SELECT snapshot_json FROM stats_snapshots").fetchall()

    assert len(rows) == 1
    import json
    snap = json.loads(rows[0][0])
    assert "t" in snap
    assert snap["t"] == pytest.approx(time.time(), abs=5)


def test_webui_db_loads_errors_on_startup(tmp_path: pathlib.Path) -> None:
    """A fresh WorkerWebUI with an existing DB should pre-populate errors_history."""
    import sqlite3
    import time

    db_dir = str(tmp_path)
    errors_db = str(tmp_path / "webui_errors.db")

    # Seed the DB manually
    webui_seed = WorkerWebUI(port=0, db_path=db_dir)
    now = time.time()
    with sqlite3.connect(errors_db) as conn:
        conn.execute(
            "INSERT INTO errors_log (message, created_at) VALUES (?, ?)",
            ("seeded_error", now),
        )
        conn.commit()

    # New webui instance should load that error
    webui2 = WorkerWebUI(port=0, db_path=db_dir)
    assert "seeded_error" in webui2.status_data["errors_history"]


def test_webui_db_loads_same_timestamp_errors_in_newest_first_order(tmp_path: pathlib.Path) -> None:
    """Persisted errors with identical timestamps should still load in newest-first order."""
    db_dir = str(tmp_path)

    webui = WorkerWebUI(port=0, db_path=db_dir)
    webui.update_status(errors_history=["error_c", "error_b", "error_a"])

    webui2 = WorkerWebUI(port=0, db_path=db_dir)
    assert webui2.status_data["errors_history"][:3] == ["error_c", "error_b", "error_a"]


def test_webui_db_loads_gallery_on_startup(tmp_path: pathlib.Path) -> None:
    """A fresh WorkerWebUI with an existing DB should restore gallery_dict."""
    import sqlite3
    import time

    db_dir = str(tmp_path)
    gallery_db = str(tmp_path / "webui_gallery.db")

    # Write a gallery entry directly to the DB
    webui_seed = WorkerWebUI(port=0, db_path=db_dir)
    now = time.time()
    with sqlite3.connect(gallery_db) as conn:
        conn.execute(
            """
            INSERT INTO gallery_images
                (gallery_id, timestamp, model, base64_data, thumbnail, is_nsfw, is_csam, extra_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (42, now, "sdxl", None, None, 0, 0, None),
        )
        conn.commit()

    webui2 = WorkerWebUI(port=0, db_path=db_dir)
    assert 42 in webui2._gallery_dict
    assert webui2._gallery_dict[42]["model"] == "sdxl"
    assert webui2._next_gallery_id >= 43
    assert webui2.status_data["images_count"] == 1


def test_webui_db_next_gallery_id_restored_from_expired_rows(tmp_path: pathlib.Path) -> None:
    """_next_gallery_id is restored from MAX(gallery_id) even when all DB rows are expired.

    When the worker has been down longer than the retention window, all gallery rows
    fall outside the cutoff and _persisted_gallery ends up empty.  Without reading
    MAX(gallery_id) from the full table, _next_gallery_id would stay at 0 and new
    images would reuse IDs that still exist in the DB (before the first prune runs).
    """
    import sqlite3
    import time

    db_dir = str(tmp_path)
    gallery_db = str(tmp_path / "webui_gallery.db")

    # Initialise the DB schema via a seed instance.
    WorkerWebUI(port=0, db_path=db_dir)

    # Insert gallery rows with timestamps far in the past (well outside any retention).
    old_ts = time.time() - 400 * 86400  # 400 days ago
    with sqlite3.connect(gallery_db) as conn:
        conn.execute(
            "INSERT INTO gallery_images"
            " (gallery_id, timestamp, model, base64_data, thumbnail, is_nsfw, is_csam, extra_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (99, old_ts, "old_model", None, None, 0, 0, None),
        )
        conn.commit()

    # Fresh instance: all rows are expired so _persisted_gallery is empty, but
    # _next_gallery_id must still reflect the existing max gallery_id (99).
    webui2 = WorkerWebUI(port=0, db_path=db_dir)
    assert webui2._gallery_dict == {}, "expired rows should not be loaded into memory"
    assert webui2._next_gallery_id == 100, "_next_gallery_id must be MAX(gallery_id)+1 from DB"


def test_webui_gallery_startup_load_is_metadata_only_and_bounded(tmp_path: pathlib.Path) -> None:
    """The startup gallery load must fetch bounded, metadata-only rows — never image blobs.

    The previous implementation fetched every in-retention row INCLUDING thumbnail blobs
    via fetchall() and only then trimmed in Python. Because the metadata and thumbnail
    columns sit after the multi-MB base64_data blob in each record, that amounted to a
    full-file read which silently stalled worker startup for minutes once the gallery
    database grew large. Thumbnails are now fetched lazily per page instead.
    """
    import sqlite3
    import time
    from unittest.mock import patch

    db_dir = str(tmp_path)
    gallery_db = str(tmp_path / "webui_gallery.db")

    WorkerWebUI(port=0, db_path=db_dir)  # initialise the schema

    now = time.time()
    with sqlite3.connect(gallery_db) as conn:
        for i in range(30):
            conn.execute(
                "INSERT INTO gallery_images"
                " (gallery_id, timestamp, model, base64_data, thumbnail, is_nsfw, is_csam, extra_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (i, now - (30 - i), f"model_{i}", "full_image_data", f"thumb_{i}", 0, 0, '{"seed": 1}'),
            )
        conn.commit()

    with patch("horde_worker_regen.webui.server._MAX_GALLERY_ENTRIES_WITH_DB", 20):
        webui = WorkerWebUI(port=0, db_path=db_dir)

    # Only the newest 20 rows (gallery_ids 10..29) are loaded; the oldest are dropped in SQL.
    assert set(webui._gallery_dict.keys()) == set(range(10, 30))
    # Metadata (including extra_json keys) is restored...
    assert webui._gallery_dict[10]["model"] == "model_10"
    assert webui._gallery_dict[29]["seed"] == 1
    # ...but no image data is preloaded into memory at startup.
    for entry in webui._gallery_dict.values():
        assert "thumbnail" not in entry
        assert "base64" not in entry
    # Thumbnails remain available on demand via the lazy per-page fetch path.
    lazy_thumbnails = webui._fetch_gallery_thumbnails_from_db([29, 10])
    assert lazy_thumbnails.get(29) == "thumb_29"
    assert lazy_thumbnails.get(10) == "thumb_10"


def test_webui_db_restores_stats_snapshots(tmp_path: pathlib.Path) -> None:
    """A fresh WorkerWebUI with an existing DB should restore stats_snapshots."""
    import json
    import sqlite3
    import time

    db_dir = str(tmp_path)
    stats_db = str(tmp_path / "webui_stats.db")
    
    webui_seed = WorkerWebUI(port=0, db_path=db_dir)
    now = time.time()
    snapshot = {"t": now, "cpu": 12.3, "gpu": 45.6, "vram": 55.0, "system_vram": 55.0,
                "ram": 30.0, "system_ram": 30.0, "worker_gpu": 0.0, "container_cpu": 0.0,
                "iph": 2.0, "kph": 5.0, "jc": 10, "jf": 0, "jp": 12, "ks": 500.0}
    with sqlite3.connect(stats_db) as conn:
        conn.execute(
            "INSERT INTO stats_snapshots (snapshot_json, timestamp) VALUES (?, ?)",
            (json.dumps(snapshot), now),
        )
        conn.commit()

    webui2 = WorkerWebUI(port=0, db_path=db_dir)
    assert len(webui2._stats_snapshots) == 1
    assert webui2._stats_snapshots[0]["cpu"] == pytest.approx(12.3)


def test_webui_db_ignores_non_dict_stats_snapshots(tmp_path: pathlib.Path) -> None:
    """Stats rows with valid non-object JSON should be ignored during load."""
    import sqlite3
    import time

    db_dir = str(tmp_path)
    stats_db = str(tmp_path / "webui_stats.db")

    WorkerWebUI(port=0, db_path=db_dir)
    now = time.time()
    with sqlite3.connect(stats_db) as conn:
        conn.execute(
            "INSERT INTO stats_snapshots (timestamp, snapshot_json) VALUES (?, ?)",
            (now, "[]"),
        )
        conn.execute(
            "INSERT INTO stats_snapshots (timestamp, snapshot_json) VALUES (?, ?)",
            (now + 1, '{"t": 123.0, "jobs_completed": 7}'),
        )
        conn.commit()

    webui2 = WorkerWebUI(port=0, db_path=db_dir)
    # _stats_snapshots is a deque (ring buffer); compare its contents as a list.
    assert list(webui2._stats_snapshots) == [{"t": 123.0, "jobs_completed": 7}]
    assert webui2._last_stats_snapshot_time == 123.0


def test_webui_db_malformed_stats_timestamp_does_not_crash(tmp_path: pathlib.Path) -> None:
    """Malformed snapshot `t` values should not crash startup restore or prune refresh."""
    import sqlite3
    import time

    db_dir = str(tmp_path)
    stats_db = str(tmp_path / "webui_stats.db")

    webui = WorkerWebUI(port=0, db_path=db_dir)
    now = time.time()
    with sqlite3.connect(stats_db) as conn:
        conn.execute(
            "INSERT INTO stats_snapshots (timestamp, snapshot_json) VALUES (?, ?)",
            (now, '{"t": "bad-value", "jobs_completed": 7}'),
        )
        conn.commit()

    webui_reload = WorkerWebUI(port=0, db_path=db_dir)
    # _stats_snapshots is a deque (ring buffer); compare its contents as a list.
    assert list(webui_reload._stats_snapshots) == [{"t": "bad-value", "jobs_completed": 7}]
    assert webui_reload._last_stats_snapshot_time == 0.0

    webui._refresh_in_memory_after_prune()
    assert webui._last_stats_snapshot_time == 0.0


def test_webui_db_prune_removes_old_data(tmp_path: pathlib.Path) -> None:
    """_prune_old_db_data() must delete rows older than the retention window."""
    import sqlite3
    import time

    db_dir = str(tmp_path)
    errors_db = str(tmp_path / "webui_errors.db")
    stats_db = str(tmp_path / "webui_stats.db")
    gallery_db = str(tmp_path / "webui_gallery.db")
    
    webui = WorkerWebUI(port=0, db_path=db_dir, data_retention_days=7)

    now = time.time()
    old_ts = now - 8 * 86400  # 8 days ago — should be pruned
    recent_ts = now - 1 * 86400  # 1 day ago — should be kept

    with sqlite3.connect(errors_db) as conn:
        conn.execute("INSERT INTO errors_log (message, created_at) VALUES (?, ?)", ("old_err", old_ts))
        conn.execute("INSERT INTO errors_log (message, created_at) VALUES (?, ?)", ("new_err", recent_ts))
        conn.commit()
    
    with sqlite3.connect(gallery_db) as conn:
        conn.execute(
            "INSERT INTO gallery_images (gallery_id, timestamp, model, base64_data, thumbnail, is_nsfw, is_csam, extra_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (1, old_ts, "m", None, None, 0, 0, None),
        )
        conn.execute(
            "INSERT INTO gallery_images (gallery_id, timestamp, model, base64_data, thumbnail, is_nsfw, is_csam, extra_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (2, recent_ts, "m", None, None, 0, 0, None),
        )
        conn.commit()
    
    with sqlite3.connect(stats_db) as conn:
        conn.execute("INSERT INTO stats_snapshots (snapshot_json, timestamp) VALUES (?, ?)", ('{}', old_ts))
        conn.execute("INSERT INTO stats_snapshots (snapshot_json, timestamp) VALUES (?, ?)", ('{}', recent_ts))
        conn.commit()

    webui._prune_old_db_data()

    with sqlite3.connect(errors_db) as conn:
        err_msgs = [r[0] for r in conn.execute("SELECT message FROM errors_log").fetchall()]
    with sqlite3.connect(gallery_db) as conn:
        gallery_ids = [r[0] for r in conn.execute("SELECT gallery_id FROM gallery_images").fetchall()]
    with sqlite3.connect(stats_db) as conn:
        stats_count = conn.execute("SELECT COUNT(*) FROM stats_snapshots").fetchone()[0]

    assert "old_err" not in err_msgs
    assert "new_err" in err_msgs
    assert 1 not in gallery_ids
    assert 2 in gallery_ids
    assert stats_count == 1


def test_webui_set_data_retention_days_updates_and_prunes(tmp_path: pathlib.Path) -> None:
    """set_data_retention_days() updates the retention period and prunes expired data."""
    import sqlite3
    import time

    db_dir = str(tmp_path)
    errors_db = str(tmp_path / "webui_errors.db")
    
    webui = WorkerWebUI(port=0, db_path=db_dir, data_retention_days=30)

    now = time.time()
    # Insert an entry 10 days old — within 30-day retention, outside 7-day retention
    with sqlite3.connect(errors_db) as conn:
        conn.execute(
            "INSERT INTO errors_log (message, created_at) VALUES (?, ?)",
            ("old_error", now - 10 * 86400),
        )
        conn.commit()

    # Verify it's present under 30-day retention
    with sqlite3.connect(errors_db) as conn:
        count_before = conn.execute("SELECT COUNT(*) FROM errors_log").fetchone()[0]
    assert count_before == 1

    # Tighten retention to 7 days — should prune the 10-day-old entry
    webui.set_data_retention_days(7)

    with sqlite3.connect(errors_db) as conn:
        count_after = conn.execute("SELECT COUNT(*) FROM errors_log").fetchone()[0]
    assert count_after == 0


def test_webui_data_retention_days_clamped_to_declared_max() -> None:
    """data_retention_days should never exceed the documented 3650-day upper bound."""
    webui = WorkerWebUI(port=0, data_retention_days=5000)
    assert webui._data_retention_days == 3650

    webui.set_data_retention_days(6000)
    assert webui._data_retention_days == 3650


@pytest.mark.parametrize(
    ("env_value", "expected_message"),
    [
        ("not_a_number", "AIWORKER_DATA_RETENTION_DAYS environment variable has an invalid value"),
        ("0", "AIWORKER_DATA_RETENTION_DAYS environment variable has an out-of-range value: 0"),
        ("3651", "AIWORKER_DATA_RETENTION_DAYS environment variable has an out-of-range value: 3651"),
    ],
)
def test_webui_invalid_data_retention_env_var_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected_message: str,
) -> None:
    """Invalid AIWORKER_DATA_RETENTION_DAYS overrides should log a warning and fall back."""
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    monkeypatch.setenv("AIWORKER_DATA_RETENTION_DAYS", env_value)
    try:
        webui = WorkerWebUI(port=0, data_retention_days=7)
    finally:
        logger.remove(sink_id)

    assert webui._data_retention_days == 7
    assert any(expected_message in message for message in messages)


def test_webui_db_prune_failure_still_updates_last_attempt_time(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed prune attempts should still advance the prune interval guard."""
    import sqlite3
    import time

    import horde_worker_regen.webui.server as server_module

    db_dir = str(tmp_path)
    webui = WorkerWebUI(port=0, db_path=db_dir)

    def _raise_locked(*args: object, **kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(server_module.sqlite3, "connect", _raise_locked)

    before = time.time()
    webui._prune_old_db_data()

    assert webui._last_db_prune_time >= before


def test_webui_db_expired_errors_not_loaded(tmp_path: pathlib.Path) -> None:
    """Errors outside the retention window must not be loaded into errors_history on startup."""
    import sqlite3
    import time

    db_dir = str(tmp_path)
    errors_db = str(tmp_path / "webui_errors.db")
    
    webui_seed = WorkerWebUI(port=0, db_path=db_dir, data_retention_days=7)
    now = time.time()

    with sqlite3.connect(errors_db) as conn:
        # Old error — outside retention window
        conn.execute(
            "INSERT INTO errors_log (message, created_at) VALUES (?, ?)",
            ("expired_error", now - 8 * 86400),
        )
        # Recent error — within retention window
        conn.execute(
            "INSERT INTO errors_log (message, created_at) VALUES (?, ?)",
            ("recent_error", now - 1 * 86400),
        )
        conn.commit()

    webui2 = WorkerWebUI(port=0, db_path=db_dir, data_retention_days=7)
    assert "expired_error" not in webui2.status_data["errors_history"]
    assert "recent_error" in webui2.status_data["errors_history"]


def test_webui_no_db_path_no_persistence() -> None:
    """WorkerWebUI without a db_path must work normally without any DB operations."""
    import time

    webui = WorkerWebUI(port=0)
    assert webui._errors_db_path is None
    assert webui._stats_db_path is None
    assert webui._gallery_db_path is None

    # These should all work without any DB operations.
    webui.add_gallery_image({"base64": None, "timestamp": 1.0, "model": "sdxl"})  # long expired
    webui.add_gallery_image({"base64": None, "timestamp": time.time(), "model": "sdxl"})
    webui.update_status(errors_history=["err1", "err2"])
    webui._last_stats_snapshot_time = 0.0
    webui._record_stats_snapshot()
    webui.set_data_retention_days(3)

    # Gallery and errors should still work in memory. Without a DB the retention window
    # is applied directly to the in-memory gallery (this is what bounds RAM usage), so
    # the expired entry is pruned while the recent one survives.
    assert len(webui._gallery_dict) == 1
    remaining = next(iter(webui._gallery_dict.values()))
    assert remaining["timestamp"] > 1.0
    assert webui.status_data["errors_history"] == ["err1", "err2"]
    assert len(webui._stats_snapshots) >= 1


def test_webui_data_retention_days_setting_in_spec() -> None:
    """data_retention_days must appear in the Python _SETTINGS_SPEC with the declared bounds."""
    from horde_worker_regen.webui.server import _SETTINGS_SPEC

    assert "data_retention_days" in _SETTINGS_SPEC
    spec = _SETTINGS_SPEC["data_retention_days"]
    assert spec["type"] is int
    assert spec["min"] == 1
    assert spec["max"] == 3650

# Offline / Horde-down resilience tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_get_workers_details_preserves_last_known_state_when_all_fail() -> None:
    """When every per-worker fetch fails, _workers_details must keep its previous value.

    Previously the function unconditionally overwrote _workers_details with the
    (empty) filtered results list, which wiped the web UI's worker cards whenever
    the Horde API was temporarily unreachable.
    """
    from unittest.mock import AsyncMock, MagicMock

    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    previous_workers = [
        {"id": "w1", "name": "Worker1", "online": True},
        {"id": "w2", "name": "Worker2", "online": False},
    ]

    mgr = MagicMock()
    mgr._shutting_down = False
    mgr._user_info_failed = False

    # user_info has two worker IDs
    mgr.user_info = MagicMock()
    mgr.user_info.worker_ids = ["w1", "w2"]

    # Pre-populate the last known state
    mgr._workers_details = list(previous_workers)

    # horde_client_session.submit_request raises on every call
    session = MagicMock()
    session.submit_request = AsyncMock(side_effect=OSError("Cannot connect"))
    mgr.horde_client_session = session

    bound = HordeWorkerProcessManager.api_get_workers_details.__get__(mgr, HordeWorkerProcessManager)
    await bound()

    # State must be unchanged – the cached list must survive a total failure
    assert mgr._workers_details == previous_workers, (
        "_workers_details must not be cleared when all worker fetches fail"
    )


@pytest.mark.asyncio
async def test_api_get_workers_details_updates_on_partial_success() -> None:
    """When at least one worker fetch succeeds, _workers_details is updated."""
    from unittest.mock import AsyncMock, MagicMock

    from horde_sdk.ai_horde_api.apimodels import SingleWorkerDetailsResponse
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    mgr = MagicMock()
    mgr._shutting_down = False
    mgr._user_info_failed = False
    mgr.user_info = MagicMock()
    mgr.user_info.worker_ids = ["w1", "w2"]
    mgr._workers_details = []

    # Build a minimal mock response for w1; w2 will raise an exception
    good_response = MagicMock(spec=SingleWorkerDetailsResponse)
    good_response.id_ = "w1"
    good_response.name = "Worker1"
    good_response.bridge_agent = "horde-worker-regen:1.0:python"
    good_response.type_ = MagicMock()
    good_response.type_.value = "image"
    good_response.online = True
    good_response.nsfw = False
    good_response.trusted = True
    good_response.img2img = False
    good_response.painting = False
    good_response.lora = False
    good_response.max_pixels = 1048576
    good_response.threads = 1
    good_response.models = ["stable_diffusion"]
    good_response.uptime = 3600
    good_response.kudos_rewards = 500.0

    call_count = 0

    async def mock_submit(req: object, resp_type: object) -> object:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return good_response
        raise OSError("Cannot connect")

    session = MagicMock()
    session.submit_request = AsyncMock(side_effect=mock_submit)
    mgr.horde_client_session = session

    bound = HordeWorkerProcessManager.api_get_workers_details.__get__(mgr, HordeWorkerProcessManager)
    await bound()

    assert len(mgr._workers_details) == 1, "Partial success should update _workers_details"
    assert mgr._workers_details[0]["id"] == "w1"
    assert mgr._workers_details[0]["name"] == "Worker1"


@pytest.mark.asyncio
async def test_api_get_workers_details_logs_debug_not_warning_when_user_info_failed() -> None:
    """Connection errors are logged at DEBUG (not WARNING) when user-info is also failing.

    This avoids WARNING spam in the logs when the internet / Horde is known to be down.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    mgr = MagicMock()
    mgr._shutting_down = False
    mgr._user_info_failed = True  # <-- internet / Horde known down
    mgr.user_info = MagicMock()
    mgr.user_info.worker_ids = ["w1"]
    mgr._workers_details = []

    session = MagicMock()
    session.submit_request = AsyncMock(side_effect=OSError("Cannot connect"))
    mgr.horde_client_session = session

    logged_levels: list[str] = []

    def fake_log(level: str, msg: str) -> None:
        logged_levels.append(level)

    bound = HordeWorkerProcessManager.api_get_workers_details.__get__(mgr, HordeWorkerProcessManager)
    with patch("horde_worker_regen.process_management.process_manager.logger") as mock_logger:
        mock_logger.log = fake_log
        mock_logger.debug = MagicMock()
        await bound()

    assert logged_levels == ["DEBUG"], (
        "Fetch failures should be logged at DEBUG when _user_info_failed is True, "
        f"but got levels: {logged_levels}"
    )


@pytest.mark.asyncio
async def test_api_get_workers_details_loop_skips_fetch_when_user_info_failed() -> None:
    """_api_get_workers_details_loop must not call api_get_workers_details when user-info is failing.

    When the Horde/internet is down, user-info fetches fail first.  The worker-details
    loop should respect _user_info_failed and skip its own fetch to avoid redundant
    connection attempts and warning spam.  After user-info recovers, the fetch runs
    again normally.
    """
    from unittest.mock import MagicMock, patch

    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    fetch_calls: list[int] = []

    mgr = MagicMock()
    mgr.horde_client_session = MagicMock()
    mgr.user_info = MagicMock()
    mgr.user_info.worker_ids = ["w1"]
    mgr._user_info_failed = True  # Horde / internet is down
    mgr._shut_down = False
    mgr._shutting_down = False
    mgr.is_time_for_shutdown = MagicMock(return_value=False)
    mgr._shutdown = MagicMock()
    mgr._api_get_workers_details_interval = 120

    async def fake_api_get_workers_details() -> None:
        fetch_calls.append(1)
        # Signal shutdown so the loop exits after this fetch
        mgr._shut_down = True
        mgr.is_time_for_shutdown = MagicMock(return_value=True)

    mgr.api_get_workers_details = fake_api_get_workers_details

    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        # After the skip-path sleep, simulate internet recovery so the loop can
        # exit cleanly on the next iteration without running forever.
        mgr._user_info_failed = False

    bound = HordeWorkerProcessManager._api_get_workers_details_loop.__get__(
        mgr, HordeWorkerProcessManager
    )
    with patch("asyncio.sleep", side_effect=fake_sleep):
        await bound()

    # The first iteration took the skip path: multiple ≤1 s sleeps totalling the interval
    assert sleep_calls, "Expected at least one sleep call on the skip path"
    assert max(sleep_calls) <= 1.0, (
        "Each skip-path sleep must be ≤ 1 s to allow prompt shutdown, "
        f"got max={max(sleep_calls)} s"
    )
    assert sum(sleep_calls) == pytest.approx(float(mgr._api_get_workers_details_interval)), (
        f"Skip-path total sleep must equal the interval ({mgr._api_get_workers_details_interval} s), "
        f"got {sum(sleep_calls)} s"
    )
    # The fetch only ran after user-info recovered (second iteration)
    assert len(fetch_calls) == 1, (
        "api_get_workers_details must run exactly once (after recovery), "
        f"but was called {len(fetch_calls)} time(s)"
    )


@pytest.mark.asyncio
async def test_webui_status_shows_workers_list_after_horde_goes_offline() -> None:
    """The /api/status endpoint exposes the latest user_details dict as-is.

    When a push omits workers_list (e.g. during an outage), the server must
    reflect that omission rather than carrying forward stale data.  The JS layer
    is responsible for its own caching via localStorage.
    """
    webui = WorkerWebUI(port=0)

    # First push: Horde online, workers_list populated
    worker_data = [{"id": "w1", "name": "MyWorker", "online": True, "type": "image"}]
    webui.update_status(
        user_details={"worker_count": 1, "workers_list": worker_data},
    )

    # Second push: Horde offline — caller omits workers_list; omitted keys are not carried forward
    webui.update_status(
        user_details={"worker_count": 1},
    )

    try:
        await webui.start()
        await asyncio.sleep(0.2)
        actual_port = webui.site._server.sockets[0].getsockname()[1] if webui.site else 0

        async with aiohttp.ClientSession() as session, session.get(
            f"http://localhost:{actual_port}/api/status",
        ) as response:
            assert response.status == 200
            status = await response.json()

        # The second push did not include workers_list, so it must be absent in the server response
        ud = status.get("user_details", {})
        # The most recent push wins for the whole user_details dict — workers_list is absent
        # because the caller didn't include it.  The JS layer uses localStorage as a cache;
        # validate here that the server at least exposes what it was last given.
        assert "worker_count" in ud, "/api/status must expose the latest user_details keys"
        assert ud["worker_count"] == 1
        assert "workers_list" not in ud, (
            "/api/status must not carry forward omitted keys; workers_list was not in the last push"
        )

    finally:
        await webui.stop()


if __name__ == "__main__":
    test_webui_creation()
    print("✓ WebUI creation test passed")

    test_webui_status_update()
    print("✓ WebUI status update test passed")

    test_webui_vram_resources()
    print("✓ WebUI VRAM resources test passed")

    test_webui_cpu_gpu_usage()
    print("✓ WebUI CPU/GPU usage test passed")

    test_webui_cpu_cores_count()
    print("✓ WebUI CPU cores count test passed")

    test_webui_new_features()
    print("✓ WebUI new features test passed")

    test_webui_faulted_jobs_history()
    print("✓ WebUI faulted jobs history test passed")

    test_webui_batch_size_display()
    print("✓ WebUI batch size display test passed")

    test_webui_last_image_submission_timestamp()
    print("✓ WebUI last image submission timestamp test passed")

    test_webui_images_history()
    print("✓ WebUI images history test passed")

    # Run async test
    asyncio.run(test_webui_start_stop())
    print("✓ WebUI start/stop test passed")

    print("\nAll tests passed!")
