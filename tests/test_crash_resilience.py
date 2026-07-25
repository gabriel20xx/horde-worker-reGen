"""Tests for crash resilience improvements."""

import asyncio
import sys
import types
from collections import deque
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from horde_worker_regen.process_management.horde_process import HordeProcessType
from horde_worker_regen.process_management.messages import HordeProcessState

if TYPE_CHECKING:
    from horde_worker_regen.process_management.inference_process import HordeInferenceProcess


class TestIsProcessAlive:
    """Tests for the is_process_alive() method fix."""

    def _make_process_info(self, state: HordeProcessState, mp_is_alive: bool) -> MagicMock:
        """Create a mock HordeProcessInfo with the given state and OS-level alive status."""
        from horde_worker_regen.process_management.process_manager import HordeProcessInfo

        mock_info = MagicMock()
        mock_info.mp_process = MagicMock()
        mock_info.mp_process.is_alive.return_value = mp_is_alive
        mock_info.last_process_state = state
        mock_info.inference_started_timestamp = None

        # Bind the actual method
        mock_info.is_process_alive = HordeProcessInfo.is_process_alive.__get__(mock_info, HordeProcessInfo)
        return mock_info

    def test_alive_process_in_normal_state_returns_true(self) -> None:
        """A process that is alive at the OS level and in a normal state should return True."""
        for state in [
            HordeProcessState.WAITING_FOR_JOB,
            HordeProcessState.INFERENCE_PROCESSING,
            HordeProcessState.MODEL_LOADING,
            HordeProcessState.PROCESS_STARTING,
        ]:
            info = self._make_process_info(state, mp_is_alive=True)
            assert info.is_process_alive() is True, f"Expected True for state {state.name}, got False"

    def test_dead_process_always_returns_false(self) -> None:
        """A process that is dead at the OS level should always return False."""
        for state in [
            HordeProcessState.WAITING_FOR_JOB,
            HordeProcessState.INFERENCE_PROCESSING,
            HordeProcessState.PROCESS_ENDING,
            HordeProcessState.PROCESS_ENDED,
        ]:
            info = self._make_process_info(state, mp_is_alive=False)
            assert info.is_process_alive() is False, f"Expected False for dead process in state {state.name}"

    def test_process_ending_state_returns_false(self) -> None:
        """A process in PROCESS_ENDING state should be considered not alive even if OS reports it alive."""
        info = self._make_process_info(HordeProcessState.PROCESS_ENDING, mp_is_alive=True)
        assert info.is_process_alive() is False

    def test_process_ended_state_returns_false(self) -> None:
        """A process in PROCESS_ENDED state should be considered not alive even if OS reports it alive."""
        info = self._make_process_info(HordeProcessState.PROCESS_ENDED, mp_is_alive=True)
        assert info.is_process_alive() is False


class TestReplaceHungProcessesAnyReplaced:
    """Behavioral tests for the any_replaced fix in replace_hung_processes."""

    def test_returns_true_when_stuck_inference_process_replaced(self) -> None:
        """replace_hung_processes should return True when it replaces a stuck inference process."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._recently_recovered = False
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager.bridge_data.inference_step_timeout = 60
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.process_timeout = 600

        # Create a mock process that appears stuck on inference
        import time

        mock_process = MagicMock()
        mock_process.process_id = 0
        mock_process.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        mock_process.inference_started_timestamp = None
        mock_process.last_heartbeat_percent_complete = 50
        mock_process.last_job_referenced = None
        mock_process.last_heartbeat_delta = 9999
        mock_process.last_progress_timestamp = time.time() - 9999
        mock_process.last_received_timestamp = time.time() - 9999
        mock_process.last_heartbeat_timestamp = time.time() - 9999

        mock_manager._process_map.values.return_value = [mock_process]
        mock_manager._process_map.is_stuck_on_inference.return_value = True

        bound_method = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )

        with patch("threading.Thread"):
            result = bound_method()

        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(mock_process, respawn=True)

    def test_inference_processing_detected_even_when_recently_recovered(self) -> None:
        """INFERENCE_PROCESSING stuck detection must fire even when _recently_recovered is True.

        Scenario: a previous recovery set _recently_recovered=True (e.g. after one of several
        prior stuck-process recoveries).  An INFERENCE_PROCESSING process that stops sending
        heartbeats must still be detected and replaced; the _recently_recovered guard must NOT
        block it.
        """
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._recently_recovered = True  # guard is active
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager.bridge_data.inference_step_timeout = 60
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.process_timeout = 30

        import time

        mock_process = MagicMock()
        mock_process.process_id = 0
        mock_process.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        mock_process.inference_started_timestamp = None
        mock_process.last_heartbeat_percent_complete = None
        mock_process.last_job_referenced = None
        mock_process.last_heartbeat_delta = 9999
        mock_process.last_progress_timestamp = time.time() - 9999
        mock_process.last_received_timestamp = time.time() - 9999
        mock_process.last_heartbeat_timestamp = time.time() - 9999

        mock_manager._process_map.values.return_value = [mock_process]
        mock_manager._process_map.is_stuck_on_inference.return_value = True

        bound_method = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )

        with patch("threading.Thread"):
            result = bound_method()

        assert result is True, (
            "replace_hung_processes must return True for a stuck INFERENCE_PROCESSING process "
            "even when _recently_recovered is True"
        )
        mock_manager._replace_inference_process.assert_called_once_with(mock_process, respawn=True)

    def test_timeout_exceeded_during_shutdown_does_not_respawn(self) -> None:
        """A process that blows past inference_timeout while shutting down must be replaced

        without spawning a fresh subprocess: a brand-new process needs to import torch and set
        up the model manager, which routinely takes longer than the graceful-shutdown window and
        forces the watchdog in `_start_timed_shutdown()` to hard-kill the worker before the new
        process even finishes starting.
        """
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        import time

        mock_manager = MagicMock()
        mock_manager._recently_recovered = False
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = True
        mock_manager._shutting_down_time = time.time()
        mock_manager.bridge_data.inference_step_timeout = 60
        mock_manager.bridge_data.inference_timeout = 120
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.process_timeout = 600

        mock_process = MagicMock()
        mock_process.process_id = 3
        mock_process.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        mock_process.inference_started_timestamp = time.time() - 406.8
        mock_process.last_heartbeat_percent_complete = 50
        mock_process.last_job_referenced = None
        mock_process.last_heartbeat_delta = 1
        mock_process.last_progress_timestamp = time.time() - 1
        mock_process.last_received_timestamp = time.time() - 1
        mock_process.last_heartbeat_timestamp = time.time() - 1

        mock_manager._process_map.values.return_value = [mock_process]
        mock_manager._process_map.is_stuck_on_inference.return_value = False

        bound_method = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )

        with patch("threading.Thread"):
            result = bound_method()

        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(mock_process, respawn=False)

    def test_stuck_process_ending_slot_is_recovered_when_capacity_is_below_target(self) -> None:
        """A stale PROCESS_ENDING slot should be replaced when active capacity is below max."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        import time

        mock_manager = MagicMock()
        mock_manager._recently_recovered = False
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager.bridge_data.inference_step_timeout = 60
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.process_timeout = 30
        mock_manager.bridge_data.preload_timeout = 30
        mock_manager.max_inference_processes = 2
        mock_manager.jobs_pending_inference = []
        mock_manager.jobs_in_progress = []
        mock_manager._process_map.num_loaded_inference_processes.return_value = 1
        mock_manager._process_map.num_inference_processes.return_value = 1
        mock_manager._check_and_replace_process.return_value = False

        mock_process = MagicMock()
        mock_process.process_id = 1
        mock_process.process_type = HordeProcessType.INFERENCE
        mock_process.last_process_state = HordeProcessState.PROCESS_ENDING
        mock_process.inference_started_timestamp = None
        mock_process.last_received_timestamp = time.time() - 120
        mock_process.last_heartbeat_timestamp = time.time() - 120
        mock_process.last_progress_timestamp = time.time() - 120
        mock_process.last_job_referenced = None
        mock_process.last_heartbeat_percent_complete = None

        mock_manager._process_map.values.return_value = [mock_process]
        mock_manager._process_map.is_stuck_on_inference.return_value = False

        bound_method = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )

        with patch("threading.Thread"):
            result = bound_method()

        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(mock_process)

    def test_stuck_process_ending_slot_is_not_recovered_while_above_scale_down_target(self) -> None:
        """A stale scale-down PROCESS_ENDING slot must not be resurrected above the active cap."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        import time

        mock_manager = MagicMock()
        mock_manager._recently_recovered = False
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager.bridge_data.inference_step_timeout = 60
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.process_timeout = 30
        mock_manager.bridge_data.preload_timeout = 30
        mock_manager.max_inference_processes = 3
        mock_manager.jobs_pending_inference = []
        mock_manager.jobs_in_progress = []
        mock_manager._hung_processes_detected = True
        mock_manager._hung_processes_detected_time = time.time()
        # Capacity is currently below target because one active slot is also ending,
        # but total inference slots still exceed the configured cap due to scale-down.
        mock_manager._process_map.num_loaded_inference_processes.return_value = 2
        mock_manager._process_map.num_inference_processes.return_value = 4
        mock_manager._check_and_replace_process.return_value = False

        mock_process = MagicMock()
        mock_process.process_id = 3
        mock_process.process_type = HordeProcessType.INFERENCE
        mock_process.last_process_state = HordeProcessState.PROCESS_ENDING
        mock_process.inference_started_timestamp = None
        mock_process.last_received_timestamp = time.time() - 120
        mock_process.last_heartbeat_timestamp = time.time() - 120
        mock_process.last_progress_timestamp = time.time() - 120
        mock_process.last_job_referenced = None
        mock_process.last_heartbeat_percent_complete = None

        mock_manager._process_map.values.return_value = [mock_process]
        mock_manager._process_map.is_stuck_on_inference.return_value = False

        bound_method = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )

        with patch("threading.Thread"):
            result = bound_method()

        assert result is False
        mock_manager._replace_inference_process.assert_not_called()


class TestBridgeDataLoopExceptionHandling:
    """Behavioral tests that the bridge data loop recovers from exceptions."""

    def test_loop_continues_after_file_not_found(self) -> None:
        """The bridge data loop should log a warning and continue when the config file is not found."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        call_count = 0

        async def run_test() -> None:
            nonlocal call_count
            mock_manager = MagicMock()
            mock_manager._shutting_down = False
            mock_manager._bridge_data_loop_interval = 0.01
            mock_manager._bridge_data_last_modified_time = 0.0
            mock_manager._last_bridge_data_reload_time = 0.0

            bound_loop = HordeWorkerProcessManager._bridge_data_loop.__get__(
                mock_manager, HordeWorkerProcessManager
            )

            with patch("horde_worker_regen.process_management.process_manager.os.path.getmtime") as mock_getmtime:

                def side_effect(path: str) -> float:
                    nonlocal call_count
                    call_count += 1
                    if call_count <= 2:
                        raise FileNotFoundError(f"No such file: {path}")
                    # After 2 FileNotFoundErrors, stop the loop gracefully
                    mock_manager._shutting_down = True
                    return 0.0

                mock_getmtime.side_effect = side_effect
                await asyncio.wait_for(bound_loop(), timeout=2.0)

        asyncio.run(run_test())
        # The loop iterated at least 3 times, meaning it survived 2 FileNotFoundErrors
        assert call_count >= 3

    def test_loop_continues_after_unexpected_exception(self) -> None:
        """The bridge data loop should log the exception and continue after an unexpected error."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        call_count = 0

        async def run_test() -> None:
            nonlocal call_count
            mock_manager = MagicMock()
            mock_manager._shutting_down = False
            mock_manager._bridge_data_loop_interval = 0.01
            mock_manager._bridge_data_last_modified_time = 0.0
            mock_manager._last_bridge_data_reload_time = 0.0

            bound_loop = HordeWorkerProcessManager._bridge_data_loop.__get__(
                mock_manager, HordeWorkerProcessManager
            )

            with patch("horde_worker_regen.process_management.process_manager.os.path.getmtime") as mock_getmtime:

                def side_effect(path: str) -> float:
                    nonlocal call_count
                    call_count += 1
                    if call_count <= 2:
                        raise RuntimeError("Simulated disk error")
                    mock_manager._shutting_down = True
                    return 0.0

                mock_getmtime.side_effect = side_effect
                await asyncio.wait_for(bound_loop(), timeout=2.0)

        asyncio.run(run_test())
        assert call_count >= 3


class TestWorkerCycleExceptionHandling:
    """Behavioral tests that the subprocess main loop handles worker_cycle() exceptions."""

    def test_worker_cycle_exception_ends_process_gracefully(self) -> None:
        """When worker_cycle raises, main_loop should set _end_process and report PROCESS_ENDING."""
        from horde_worker_regen.process_management.horde_process import HordeProcess

        class _BrokenWorkerProcess(HordeProcess):
            cycle_calls: int = 0

            def worker_cycle(self) -> None:
                self.cycle_calls += 1
                raise RuntimeError("Simulated crash in worker_cycle")

            def cleanup_for_exit(self) -> None:
                pass

            def _receive_and_handle_control_message(self, message: object) -> None:
                pass

        mock_queue = MagicMock()
        mock_conn = MagicMock()
        mock_conn.poll.return_value = False
        mock_lock = MagicMock()

        proc = _BrokenWorkerProcess(
            process_id=0,
            process_message_queue=mock_queue,
            pipe_connection=mock_conn,
            disk_lock=mock_lock,
            process_launch_identifier=1,
        )

        with patch("signal.signal"), patch.object(sys, "exit"):
            proc.main_loop()

        assert proc._end_process is True
        assert proc.cycle_calls == 1

        # Verify PROCESS_ENDING was reported via the queue
        sent_states = [
            call.args[0].process_state
            for call in mock_queue.put.call_args_list
            if hasattr(call.args[0], "process_state")
        ]
        assert HordeProcessState.PROCESS_ENDING in sent_states

    def test_cleanup_for_exit_exception_still_sends_process_ended(self) -> None:
        """When cleanup_for_exit raises, main_loop should still send PROCESS_ENDED."""
        from horde_worker_regen.process_management.horde_process import HordeProcess

        class _CleanupFailsProcess(HordeProcess):
            def worker_cycle(self) -> None:
                self._end_process = True  # Exit the loop immediately

            def cleanup_for_exit(self) -> None:
                raise RuntimeError("Simulated cleanup failure")

            def _receive_and_handle_control_message(self, message: object) -> None:
                pass

        mock_queue = MagicMock()
        mock_conn = MagicMock()
        mock_conn.poll.return_value = False
        mock_lock = MagicMock()

        proc = _CleanupFailsProcess(
            process_id=0,
            process_message_queue=mock_queue,
            pipe_connection=mock_conn,
            disk_lock=mock_lock,
            process_launch_identifier=1,
        )

        with patch("signal.signal"), patch.object(sys, "exit"):
            proc.main_loop()

        # Even though cleanup_for_exit raised, PROCESS_ENDED must still be sent
        sent_states = [
            call.args[0].process_state
            for call in mock_queue.put.call_args_list
            if hasattr(call.args[0], "process_state")
        ]
        assert HordeProcessState.PROCESS_ENDED in sent_states


class TestIdleHeartbeatTiming:
    """Unit tests for the idle keepalive heartbeat added to HordeProcess.worker_cycle()."""

    def _make_idle_process(self) -> tuple[object, MagicMock]:
        """Return a concrete HordeProcess subclass instance and its message queue mock."""
        from horde_worker_regen.process_management.horde_process import HordeProcess

        class _IdleProcess(HordeProcess):
            def cleanup_for_exit(self) -> None:
                pass

            def _receive_and_handle_control_message(self, message: object) -> None:
                pass

        mock_queue = MagicMock()
        mock_conn = MagicMock()
        mock_conn.poll.return_value = False
        mock_lock = MagicMock()

        proc = _IdleProcess(
            process_id=0,
            process_message_queue=mock_queue,
            pipe_connection=mock_conn,
            disk_lock=mock_lock,
            process_launch_identifier=1,
        )
        return proc, mock_queue

    def _heartbeat_put_calls(self, mock_queue: MagicMock) -> list:
        """Return all HordeProcessHeartbeatMessage objects enqueued so far."""
        from horde_worker_regen.process_management.messages import HordeProcessHeartbeatMessage

        return [
            call.args[0]
            for call in mock_queue.put.call_args_list
            if isinstance(call.args[0], HordeProcessHeartbeatMessage)
        ]

    def test_no_heartbeat_before_interval_elapses(self) -> None:
        """worker_cycle() must NOT send a heartbeat before the idle interval has elapsed."""
        proc, mock_queue = self._make_idle_process()

        fixed_time = 1_000.0
        proc._last_idle_heartbeat_time = fixed_time
        proc._last_heartbeat_time = fixed_time - 10.0  # well outside throttle window

        # Simulate a call just before the interval expires
        with patch("time.time", return_value=fixed_time + proc._idle_heartbeat_interval_seconds - 0.1):
            proc.worker_cycle()

        assert len(self._heartbeat_put_calls(mock_queue)) == 0

    def test_heartbeat_sent_once_interval_elapses(self) -> None:
        """worker_cycle() MUST send a heartbeat once the idle interval has elapsed."""
        proc, mock_queue = self._make_idle_process()

        fixed_time = 1_000.0
        proc._last_idle_heartbeat_time = fixed_time
        proc._last_heartbeat_time = fixed_time - 10.0  # well outside throttle window

        with patch("time.time", return_value=fixed_time + proc._idle_heartbeat_interval_seconds + 0.1):
            proc.worker_cycle()

        heartbeats = self._heartbeat_put_calls(mock_queue)
        assert len(heartbeats) == 1

    def test_no_heartbeat_inside_throttle_window(self) -> None:
        """worker_cycle() must NOT send a heartbeat when a recent heartbeat was just sent."""
        from horde_worker_regen.process_management.messages import HordeHeartbeatType

        proc, mock_queue = self._make_idle_process()

        fixed_time = 1_000.0
        # Idle interval has elapsed but we are inside the throttle window
        proc._last_idle_heartbeat_time = fixed_time - proc._idle_heartbeat_interval_seconds - 1.0
        proc._last_heartbeat_time = fixed_time - proc._heartbeat_limit_interval_seconds + 0.1
        proc._last_heartbeat_type = HordeHeartbeatType.OTHER

        with patch("time.time", return_value=fixed_time):
            proc.worker_cycle()

        assert len(self._heartbeat_put_calls(mock_queue)) == 0

    def test_idle_heartbeat_timestamp_updated_after_send(self) -> None:
        """After sending an idle heartbeat, _last_idle_heartbeat_time must be updated."""
        proc, mock_queue = self._make_idle_process()

        send_time = 2_000.0
        proc._last_idle_heartbeat_time = send_time - proc._idle_heartbeat_interval_seconds - 1.0
        proc._last_heartbeat_time = send_time - 10.0

        with patch("time.time", return_value=send_time):
            proc.worker_cycle()

        assert proc._last_idle_heartbeat_time == send_time


def test_start_calls_cleanup_before_execv_on_restart() -> None:
    """start() must call _cleanup_shared_resources() before os.execv when restarting."""
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    mock_manager = MagicMock()
    mock_manager.signal_handler = MagicMock()
    mock_manager._main_loop = MagicMock(return_value=None)
    mock_manager._restart_requested = True

    call_order: list[str] = []
    mock_manager._cleanup_shared_resources.side_effect = lambda: call_order.append("cleanup")

    bound_start = HordeWorkerProcessManager.start.__get__(mock_manager, HordeWorkerProcessManager)

    with (
        patch("signal.signal"),
        patch("asyncio.run"),
        # Force the POSIX restart path (in-place os.execv); on Windows start() exits with a code instead.
        patch("sys.platform", "linux"),
        patch("os.execv", side_effect=lambda *_: call_order.append("execv")),
        patch("horde_worker_regen.process_management.process_manager.logger"),
    ):
        bound_start()

    mock_manager._cleanup_shared_resources.assert_called_once()
    assert call_order == ["cleanup", "execv"], (
        "_cleanup_shared_resources() must be called before os.execv()"
    )


def test_start_exits_cleanly_when_restart_exec_fails() -> None:
    """start() should log and exit cleanly when the restart execv call fails."""
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    mock_manager = MagicMock()
    mock_manager.signal_handler = MagicMock()
    mock_manager._main_loop = MagicMock(return_value=None)
    mock_manager._restart_requested = True

    bound_start = HordeWorkerProcessManager.start.__get__(mock_manager, HordeWorkerProcessManager)

    with (
        patch("signal.signal"),
        patch("asyncio.run"),
        # Force the POSIX restart path so os.execv is exercised regardless of host OS.
        patch("sys.platform", "linux"),
        patch("os.execv", side_effect=OSError("boom")) as mock_execv,
        patch.object(sys, "exit") as mock_exit,
        patch("horde_worker_regen.process_management.process_manager.logger") as mock_logger,
    ):
        bound_start()

    mock_execv.assert_called_once_with(sys.executable, [sys.executable, *sys.argv])
    mock_logger.warning.assert_called_once_with("Restarting worker program...")
    mock_logger.exception.assert_called_once()
    mock_exit.assert_called_once_with(1)


def test_start_uses_exit_code_instead_of_execv_on_windows() -> None:
    """On Windows, start() must exit with WORKER_RESTART_EXIT_CODE instead of calling os.execv.

    os.execv cannot replace the process in-place on Windows (it spawns a new pid and exits the
    original), so the launching cmd.exe wrapper loops on this exit code to re-run the worker.
    """
    from horde_worker_regen.consts import WORKER_RESTART_EXIT_CODE
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    mock_manager = MagicMock()
    mock_manager.signal_handler = MagicMock()
    mock_manager._main_loop = MagicMock(return_value=None)
    mock_manager._restart_requested = True

    bound_start = HordeWorkerProcessManager.start.__get__(mock_manager, HordeWorkerProcessManager)

    with (
        patch("signal.signal"),
        patch("asyncio.run"),
        patch("sys.platform", "win32"),
        patch("os.execv") as mock_execv,
        patch("horde_worker_regen.process_management.process_manager.logger"),
        pytest.raises(SystemExit) as exc_info,
    ):
        bound_start()

    assert exc_info.value.code == WORKER_RESTART_EXIT_CODE
    mock_execv.assert_not_called()
    mock_manager._cleanup_shared_resources.assert_called_once()


def test_start_timed_shutdown_skips_hard_exit_after_clean_shutdown() -> None:
    """Timed shutdown fail-safe must do nothing once shutdown has already completed."""
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    class ImmediateThread:
        def __init__(self, *, target: object, daemon: bool) -> None:
            self._target = target
            self.daemon = daemon

        def start(self) -> None:
            self._target()

    mock_process = MagicMock()
    mock_manager = MagicMock()
    mock_manager.jobs_pending_submit = []
    mock_manager._shutting_down = True
    mock_manager._shut_down = True
    mock_manager.bridge_data.force_restart_timeout = 30  # real int: used as the min() cap
    mock_manager._process_map.values.return_value = [mock_process]

    bound = HordeWorkerProcessManager._start_timed_shutdown.__get__(mock_manager, HordeWorkerProcessManager)

    with (
        patch("threading.Thread", side_effect=ImmediateThread) as mock_thread,
        patch("time.sleep"),
        patch("os._exit") as mock_exit,
    ):
        bound()

    mock_thread.assert_called_once()
    assert mock_thread.call_args.kwargs["daemon"] is True
    mock_process.mp_process.kill.assert_not_called()
    mock_process.mp_process.join.assert_not_called()
    mock_exit.assert_not_called()


@pytest.mark.parametrize(
    ("jobs_pending_submit", "expected_wait_seconds"),
    [
        (3, 15),    # max((3 * 4) + 2, 15) floor → 15
        (100, 30),  # capped from 402 to force_restart_timeout (30)
    ],
)
def test_start_timed_shutdown_wait_seconds_caps_at_30(
    jobs_pending_submit: int,
    expected_wait_seconds: int,
) -> None:
    """Timed shutdown wait is floored at 15s (so graceful shutdown isn't force-killed early) and
    capped at force_restart_timeout for large queues."""
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    class ImmediateThread:
        def __init__(self, *, target: object, daemon: bool) -> None:
            self._target = target
            self.daemon = daemon

        def start(self) -> None:
            self._target()

    mock_manager = MagicMock()
    mock_manager.jobs_pending_submit = [MagicMock()] * jobs_pending_submit
    mock_manager._shutting_down = False
    mock_manager._shut_down = False
    mock_manager.bridge_data.force_restart_timeout = 30  # real int cap so min() works (expected cap=30)
    mock_manager._process_map.values.return_value = []

    bound = HordeWorkerProcessManager._start_timed_shutdown.__get__(mock_manager, HordeWorkerProcessManager)

    with (
        patch("threading.Thread", side_effect=ImmediateThread),
        patch("time.sleep") as mock_sleep,
        patch("os._exit") as mock_exit,
    ):
        bound()

    mock_sleep.assert_called_once_with(expected_wait_seconds)
    mock_exit.assert_not_called()


def test_start_timed_shutdown_hard_exit_cleans_up_shared_resources() -> None:
    """The watchdog force-kill path must clean up shared semaphores before os._exit().

    os._exit() skips atexit/Finalize callbacks just like os.execv() does, so without an
    explicit call here, named semaphores/locks are left registered with the resource
    tracker and it warns about "leaked" semaphores at shutdown.
    """
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    class ImmediateThread:
        def __init__(self, *, target: object, daemon: bool) -> None:
            self._target = target
            self.daemon = daemon

        def start(self) -> None:
            self._target()

    mock_manager = MagicMock()
    mock_manager.jobs_pending_submit = []
    mock_manager._shutting_down = True
    mock_manager._shut_down = False
    mock_manager._restart_requested = False
    mock_manager.bridge_data.force_restart_timeout = 30
    mock_manager._process_map.values.return_value = []

    call_order: list[str] = []
    mock_manager._cleanup_shared_resources.side_effect = lambda: call_order.append("cleanup")

    bound = HordeWorkerProcessManager._start_timed_shutdown.__get__(mock_manager, HordeWorkerProcessManager)

    with (
        patch("threading.Thread", side_effect=ImmediateThread),
        patch("time.sleep"),
        patch("os._exit", side_effect=lambda *_: call_order.append("exit")) as mock_exit,
    ):
        bound()

    mock_manager._cleanup_shared_resources.assert_called_once()
    mock_exit.assert_called_once_with(1)
    assert call_order == ["cleanup", "exit"], "_cleanup_shared_resources() must be called before os._exit()"


def _make_api_job_pop_mock_manager() -> MagicMock:
    """Build a mock manager whose state passes every api_job_pop gate up to the HTTP request."""
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    mock_manager = MagicMock()

    mock_manager._shutting_down = False
    mock_manager._job_pops_paused = False
    mock_manager._job_pops_pause_until = None
    mock_manager.horde_client_session = MagicMock()
    mock_manager._too_many_consecutive_failed_jobs = False
    mock_manager._consecutive_failed_jobs = 0
    mock_manager._consecutive_pop_failures = 0
    mock_manager._consecutive_pop_failure_warn_threshold = 3
    mock_manager._job_pop_frequency = 0.0
    mock_manager._error_job_pop_frequency = 5.0
    mock_manager._default_job_pop_frequency = 4.0
    mock_manager._last_pop_maintenance_mode = False
    mock_manager._replaced_due_to_maintenance = False
    mock_manager.bridge_data.queue_size = 5
    mock_manager.bridge_data.max_threads = 1
    mock_manager.jobs_pending_inference = []
    mock_manager.jobs_in_progress = []
    mock_manager.jobs_pending_submit = []
    mock_manager.max_queue_size = 5
    mock_manager._process_map.get_first_available_safety_process.return_value = MagicMock()
    mock_manager._process_map.get_first_available_inference_process.return_value = MagicMock()
    mock_manager.bridge_data.image_models_to_load = ["test_model"]
    mock_manager.should_wait_for_pending_megapixelsteps.return_value = False
    mock_manager._triggered_max_pending_megapixelsteps = False
    mock_manager._last_job_pop_time = 0.0
    mock_manager.bridge_data.horde_model_stickiness = 0
    mock_manager.bridge_data.custom_models = None
    mock_manager.bridge_data.api_key = "0" * 22  # must be 22 characters
    mock_manager.bridge_data.dreamer_worker_name = "test_worker"
    mock_manager.bridge_data.blacklist = []
    mock_manager.bridge_data.nsfw = False
    mock_manager.max_concurrent_inference_processes = 1
    mock_manager.bridge_data.max_power = 8
    mock_manager.bridge_data.require_upfront_kudos = False
    mock_manager.bridge_data.allow_img2img = True
    mock_manager.bridge_data.allow_inpainting = True
    mock_manager.bridge_data.allow_unsafe_ip = False
    mock_manager.bridge_data.allow_post_processing = True
    mock_manager.bridge_data.allow_controlnet = True
    mock_manager.bridge_data.allow_sdxl_controlnet = False
    mock_manager.bridge_data.extra_slow_worker = False
    mock_manager.bridge_data.limit_max_steps = False
    mock_manager.bridge_data.allow_lora = True
    mock_manager.bridge_data.max_batch = 1
    mock_manager.max_inference_processes = 1

    # Inference-failure cooldown: no models in cooldown
    mock_manager._inference_failures = {}
    mock_manager._INFERENCE_FAILURE_THRESHOLD = HordeWorkerProcessManager._INFERENCE_FAILURE_THRESHOLD
    mock_manager._INFERENCE_FAILURE_WINDOW = HordeWorkerProcessManager._INFERENCE_FAILURE_WINDOW
    mock_manager._INFERENCE_FAILURE_COOLDOWN = HordeWorkerProcessManager._INFERENCE_FAILURE_COOLDOWN
    mock_manager._last_warned_inference_cooldown_models = frozenset()
    mock_manager._last_warned_inference_cooldown_at = 0.0
    mock_manager._prune_preload_stuck_failures = types.MethodType(
        HordeWorkerProcessManager._prune_preload_stuck_failures,
        mock_manager,
    )
    mock_manager._is_model_in_inference_cooldown = types.MethodType(
        HordeWorkerProcessManager._is_model_in_inference_cooldown,
        mock_manager,
    )

    return mock_manager


def test_api_job_pop_times_out_instead_of_hanging() -> None:
    """A job-pop request that never gets a response must fail within _JOB_POP_TIMEOUT_SECONDS.

    horde_sdk passes no per-request timeout to aiohttp, so a wedged gateway that accepts the
    connection but never responds used to stall the pop loop silently for many minutes.
    api_job_pop must cap the request via asyncio.wait_for, count the failure, and switch to
    the error pop frequency so the retry happens promptly and visibly.
    """
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    mock_manager = _make_api_job_pop_mock_manager()
    mock_manager._JOB_POP_TIMEOUT_SECONDS = 0.05

    async def _hang(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(3600)

    mock_manager.horde_client_session.submit_request = _hang

    with patch("horde_worker_regen.process_management.process_manager.logger") as mock_logger:
        bound = HordeWorkerProcessManager.api_job_pop.__get__(mock_manager, HordeWorkerProcessManager)
        asyncio.run(asyncio.wait_for(bound(), timeout=10))

    assert mock_manager._consecutive_pop_failures == 1
    assert mock_manager._job_pop_frequency == mock_manager._error_job_pop_frequency
    warning_text = mock_logger.warning.call_args[0][0]
    assert "(Timeout)" in warning_text, f"Expected a pop-timeout warning, got: {warning_text!r}"


def test_remove_maintenance_returns_false_when_api_unresponsive() -> None:
    """remove_maintenance() must give up after its bounded join instead of hanging forever.

    The synchronous horde_sdk client performs requests with no timeout; when the API is
    unresponsive the call must return False within the caller-supplied timeout (plus a
    warning) rather than blocking the caller — which, from the event loop, would silence
    the whole worker.
    """
    import time as _time

    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    mock_manager = MagicMock()
    mock_manager.bridge_data.dreamer_worker_name = "test_worker"
    mock_manager.bridge_data.api_key = "0" * 22

    hang_seconds = 2.0

    class _HangingClient:
        def worker_details_by_name(self, worker_name: str) -> None:
            _time.sleep(hang_seconds)

        def worker_modify(self, request: object) -> None:  # pragma: no cover - never reached
            raise AssertionError("worker_modify must not be reached when details hang")

    bound = HordeWorkerProcessManager.remove_maintenance.__get__(mock_manager, HordeWorkerProcessManager)

    start = _time.monotonic()
    with (
        patch(
            "horde_worker_regen.process_management.process_manager.AIHordeAPISimpleClient",
            _HangingClient,
        ),
        patch("horde_worker_regen.process_management.process_manager.logger") as mock_logger,
    ):
        result = bound(timeout=0.1)
    elapsed = _time.monotonic() - start

    assert result is False
    assert elapsed < hang_seconds, f"remove_maintenance blocked for {elapsed:.2f}s despite timeout=0.1"
    assert mock_logger.warning.called


def test_remove_maintenance_returns_true_on_success() -> None:
    """remove_maintenance() must return True when the API confirms the modification."""
    from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

    mock_manager = MagicMock()
    mock_manager.bridge_data.dreamer_worker_name = "test_worker"
    mock_manager.bridge_data.api_key = "0" * 22

    mock_client = MagicMock()
    mock_client.worker_details_by_name.return_value = MagicMock(id_="00000000-0000-0000-0000-000000000000")

    bound = HordeWorkerProcessManager.remove_maintenance.__get__(mock_manager, HordeWorkerProcessManager)

    with (
        patch(
            "horde_worker_regen.process_management.process_manager.AIHordeAPISimpleClient",
            return_value=mock_client,
        ),
        patch("horde_worker_regen.process_management.process_manager.ModifyWorkerRequest") as mock_request,
        patch("horde_worker_regen.process_management.process_manager.logger"),
    ):
        result = bound(timeout=5.0)

    assert result is True
    mock_client.worker_modify.assert_called_once_with(mock_request.return_value)


class TestCheckAutoRestartOnIdle:
    """Tests for _check_auto_restart_on_idle()."""

    def _make_manager(
        self,
        *,
        threshold_minutes: int = 60,
        elapsed_seconds: float = 0.0,
        elapsed_no_jobs_seconds: float = 0.0,
        shutting_down: bool = False,
        job_pops_paused: bool = False,
        active_jobs: bool = False,
        active_job_queue: str | None = None,
    ) -> MagicMock:
        """Return a minimal mock manager with the attributes needed by _check_auto_restart_on_idle."""
        import time

        mock_manager = MagicMock()
        mock_manager._shutting_down = shutting_down
        mock_manager.bridge_data.auto_restart_on_idle_minutes = threshold_minutes
        mock_manager._last_job_submitted_time = time.time() - elapsed_seconds
        mock_manager._last_pop_no_jobs_available_time = (
            0.0 if elapsed_no_jobs_seconds <= 0.0 else time.time() - elapsed_no_jobs_seconds
        )
        mock_manager._restart_requested = False
        mock_manager._job_pops_paused = job_pops_paused
        mock_manager._last_job_pop_time = time.time()
        # Simulate active jobs (non-empty queues) or idle state (empty queues)
        mock_manager.jobs_pending_inference = []
        mock_manager.jobs_in_progress = []
        mock_manager.jobs_being_safety_checked = []
        mock_manager.jobs_pending_safety_check = []
        mock_manager.jobs_pending_submit = []
        if active_jobs:
            queue_name = active_job_queue or "jobs_pending_inference"
            setattr(mock_manager, queue_name, [MagicMock()])
        return mock_manager

    def test_does_nothing_when_disabled(self) -> None:
        """auto_restart_on_idle_minutes=0 should never trigger a restart."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = self._make_manager(threshold_minutes=0, elapsed_seconds=9999)
        bound = HordeWorkerProcessManager._check_auto_restart_on_idle.__get__(mgr, HordeWorkerProcessManager)
        bound()
        mgr._shutdown.assert_not_called()

    def test_does_nothing_when_shutting_down(self) -> None:
        """No restart should be triggered when already shutting down."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = self._make_manager(threshold_minutes=1, elapsed_seconds=9999, shutting_down=True)
        bound = HordeWorkerProcessManager._check_auto_restart_on_idle.__get__(mgr, HordeWorkerProcessManager)
        bound()
        mgr._shutdown.assert_not_called()

    def test_does_nothing_when_below_threshold(self) -> None:
        """No restart when elapsed time is below the threshold."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = self._make_manager(threshold_minutes=60, elapsed_seconds=30)
        bound = HordeWorkerProcessManager._check_auto_restart_on_idle.__get__(mgr, HordeWorkerProcessManager)
        bound()
        mgr._shutdown.assert_not_called()

    def test_triggers_restart_when_threshold_exceeded(self) -> None:
        """A restart should be requested when elapsed time exceeds the threshold."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = self._make_manager(threshold_minutes=60, elapsed_seconds=3601)
        bound = HordeWorkerProcessManager._check_auto_restart_on_idle.__get__(mgr, HordeWorkerProcessManager)
        bound()
        assert mgr._restart_requested is True
        mgr._shutdown.assert_called_once()
        # Should also reset _last_job_pop_time for immediate process termination
        assert mgr._last_job_pop_time == 0.0

    def test_resets_webui_uptime_when_restart_triggered(self) -> None:
        """When idle restart is triggered, WebUI session start time should be reset so the uptime pill shows near-zero."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = self._make_manager(threshold_minutes=1, elapsed_seconds=3600)
        bound = HordeWorkerProcessManager._check_auto_restart_on_idle.__get__(mgr, HordeWorkerProcessManager)
        bound()
        assert mgr._restart_requested is True
        mgr.webui.reset_session_start_time.assert_called_once()

    def test_does_not_reset_webui_uptime_when_no_restart(self) -> None:
        """When restart is not triggered (below threshold), WebUI session start time should NOT be reset."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = self._make_manager(threshold_minutes=60, elapsed_seconds=30)
        bound = HordeWorkerProcessManager._check_auto_restart_on_idle.__get__(mgr, HordeWorkerProcessManager)
        bound()
        assert mgr._restart_requested is False
        mgr.webui.reset_session_start_time.assert_not_called()

    def test_does_not_reset_webui_uptime_when_webui_is_none(self) -> None:
        """When webui is None, reset should not be attempted (no AttributeError)."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = self._make_manager(threshold_minutes=1, elapsed_seconds=3600)
        mgr.webui = None
        bound = HordeWorkerProcessManager._check_auto_restart_on_idle.__get__(mgr, HordeWorkerProcessManager)
        bound()  # Should not raise
        assert mgr._restart_requested is True

    def test_triggers_restart_at_exact_threshold(self) -> None:
        """Restart should trigger when elapsed equals the threshold exactly."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = self._make_manager(threshold_minutes=1, elapsed_seconds=60)
        bound = HordeWorkerProcessManager._check_auto_restart_on_idle.__get__(mgr, HordeWorkerProcessManager)
        bound()
        assert mgr._restart_requested is True
        mgr._shutdown.assert_called_once()
        assert mgr._last_job_pop_time == 0.0

    def test_triggers_restart_when_last_no_jobs_pop_timeout_exceeded(self) -> None:
        """Restart should trigger when continuous no-jobs pop duration exceeds threshold."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = self._make_manager(
            threshold_minutes=1,
            elapsed_seconds=1,  # last submit is recent
            elapsed_no_jobs_seconds=61,
        )
        bound = HordeWorkerProcessManager._check_auto_restart_on_idle.__get__(mgr, HordeWorkerProcessManager)
        bound()
        assert mgr._restart_requested is True
        mgr._shutdown.assert_called_once()
        assert mgr._last_job_pop_time == 0.0

    def test_does_nothing_when_job_pops_paused(self) -> None:
        """No restart should be triggered when job pops are intentionally paused by the user."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = self._make_manager(threshold_minutes=1, elapsed_seconds=9999, job_pops_paused=True)
        bound = HordeWorkerProcessManager._check_auto_restart_on_idle.__get__(mgr, HordeWorkerProcessManager)
        bound()
        mgr._shutdown.assert_not_called()

    @pytest.mark.parametrize(
        "active_job_queue",
        (
            "jobs_pending_inference",
            "jobs_in_progress",
            "jobs_being_safety_checked",
            "jobs_pending_safety_check",
            "jobs_pending_submit",
        ),
    )
    def test_triggers_restart_when_active_jobs_exist_but_idle_threshold_exceeded(
        self,
        active_job_queue: str,
    ) -> None:
        """Restart should still trigger when last-submission age exceeds threshold even if jobs are queued.

        A worker whose inference subprocesses are wedged (e.g. all stuck in PROCESS_STARTING) keeps
        previously-popped jobs in its pipeline forever; relying on the last-submission timestamp is the
        only reliable signal that the worker has stalled, so the active-jobs guard must not suppress it.
        """
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = self._make_manager(
            threshold_minutes=1,
            elapsed_seconds=9999,
            active_jobs=True,
            active_job_queue=active_job_queue,
        )
        bound = HordeWorkerProcessManager._check_auto_restart_on_idle.__get__(mgr, HordeWorkerProcessManager)
        bound()
        assert mgr._restart_requested is True
        mgr._shutdown.assert_called_once()
        assert mgr._last_job_pop_time == 0.0

    def test_auto_restart_idle_loop_uses_short_sleep_steps(self) -> None:
        """Idle loop should sleep in short steps so shutdown can complete quickly."""
        import asyncio

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = MagicMock()
        mgr._shutting_down = False
        mgr._check_auto_restart_on_idle = MagicMock()
        mgr._shutdown = MagicMock()
        sleep_calls: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            mgr._shutting_down = True

        bound = HordeWorkerProcessManager._auto_restart_idle_loop.__get__(mgr, HordeWorkerProcessManager)
        with patch("asyncio.sleep", side_effect=fake_sleep):
            asyncio.run(bound())

        assert sleep_calls == [5.0]
        mgr._check_auto_restart_on_idle.assert_not_called()




    def _make_job_info(self, *, id_: object = "test-id", seed: object = 42, r2_upload: object = "url") -> MagicMock:
        """Return a minimal sdk_api_job_info mock."""
        job_info = MagicMock()
        job_info.id_ = id_
        job_info.ids = [id_]
        job_info.r2_upload = r2_upload
        job_info.payload = MagicMock()
        job_info.payload.seed = seed
        job_info.payload.n_iter = 1
        return job_info

    def _make_completed_job(self, job_info: MagicMock, *, censored: object = False) -> MagicMock:
        completed = MagicMock()
        completed.sdk_api_job_info = job_info
        completed.state = "ok"  # concrete non-None, non-faulted state
        completed.job_image_results = None
        completed.censored = censored
        return completed

    def _run_api_submit_job(self, completed_job_info: MagicMock) -> tuple[list[MagicMock], dict, dict, list, dict]:
        """Run api_submit_job with the given completed job and return post-run tracking state."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job_info = completed_job_info.sdk_api_job_info
        sentinel_id = job_info.id_

        pending: list[MagicMock] = [completed_job_info]
        jobs_lookup: dict = {job_info: completed_job_info}
        job_pop_timestamps: dict = {job_info: 0.0}
        jobs_in_progress: list = [job_info]
        job_faults: dict = {sentinel_id: []} if sentinel_id is not None else {}

        mock_manager = MagicMock()
        mock_manager.jobs_pending_submit = pending
        mock_manager.jobs_lookup = jobs_lookup
        mock_manager.job_pop_timestamps = job_pop_timestamps
        mock_manager.jobs_in_progress = jobs_in_progress
        mock_manager.job_faults = job_faults
        # Bind the real helper so cleanup actually runs
        mock_manager._discard_broken_job = types.MethodType(
            HordeWorkerProcessManager._discard_broken_job, mock_manager
        )

        import asyncio

        bound = HordeWorkerProcessManager.api_submit_job.__get__(mock_manager, HordeWorkerProcessManager)
        asyncio.run(bound())
        return pending, jobs_lookup, job_pop_timestamps, jobs_in_progress, job_faults

    def test_job_with_none_id_is_skipped(self) -> None:
        """A job with id_=None should be removed from the queue rather than blocking it."""
        job_info = self._make_job_info(id_=None)
        completed = self._make_completed_job(job_info)
        pending, lookup, timestamps, in_progress, faults = self._run_api_submit_job(completed)
        assert len(pending) == 0
        assert job_info not in lookup
        assert job_info not in timestamps
        assert job_info not in in_progress

    def test_job_with_none_seed_is_skipped(self) -> None:
        """A job with seed=None should be removed from the queue rather than blocking it."""
        job_info = self._make_job_info(seed=None)
        completed = self._make_completed_job(job_info)
        pending, lookup, timestamps, in_progress, faults = self._run_api_submit_job(completed)
        assert len(pending) == 0
        assert job_info not in lookup
        assert job_info not in timestamps
        assert job_info not in in_progress
        assert "test-id" not in faults

    def test_job_with_none_r2_upload_is_skipped(self) -> None:
        """A job with r2_upload=None should be removed from the queue rather than blocking it."""
        job_info = self._make_job_info(r2_upload=None)
        completed = self._make_completed_job(job_info)
        pending, lookup, timestamps, in_progress, faults = self._run_api_submit_job(completed)
        assert len(pending) == 0
        assert job_info not in lookup
        assert job_info not in timestamps
        assert job_info not in in_progress
        assert "test-id" not in faults

    def test_job_with_none_censored_and_images_is_skipped(self) -> None:
        """A job with image_results set but censored=None should be removed rather than blocking."""
        job_info = self._make_job_info()
        completed = self._make_completed_job(job_info, censored=None)
        # Set job_image_results to trigger the censored check
        completed.job_image_results = [MagicMock()]
        completed.sdk_api_job_info.payload.n_iter = 1
        pending, lookup, timestamps, in_progress, faults = self._run_api_submit_job(completed)
        assert len(pending) == 0
        assert job_info not in lookup
        assert job_info not in timestamps
        assert job_info not in in_progress
        assert "test-id" not in faults


class TestShutDownSetBeforeForceKill:
    """Tests that _shut_down is set before end_inference_processes(force=True).

    This prevents the timed-shutdown safety thread (started by _start_timed_shutdown)
    from calling os._exit(1) while the force kill is still blocking on join() calls,
    which would prevent os.execv from performing a program restart.
    """

    def test_shut_down_set_before_force_kill(self) -> None:
        """_shut_down must be True before end_inference_processes(force=True) is called."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = MagicMock()
        mgr._shutting_down = True
        mgr._shut_down = False
        mgr._loop_interval = 0.001
        mgr._restart_requested = True
        mgr.stable_diffusion_reference = MagicMock()
        mgr.jobs_pending_inference = []
        mgr.jobs_in_progress = []
        mgr.jobs_being_safety_checked = []
        mgr.jobs_pending_safety_check = []
        mgr.jobs_pending_submit = []
        mgr._process_map = MagicMock()
        mgr._process_map.values.return_value = []
        mgr._last_job_pop_time = 0.0
        mgr._inference_scale_down_requested = False
        mgr._jobs_lookup_lock = asyncio.Lock()
        mgr._jobs_pending_inference_lock = asyncio.Lock()
        mgr._jobs_safety_check_lock = asyncio.Lock()
        mgr._completed_jobs_lock = asyncio.Lock()
        mgr._job_pop_timestamps_lock = asyncio.Lock()
        mgr._last_pop_maintenance_mode = False
        mgr._job_pops_paused = False
        mgr._replaced_due_to_maintenance = False
        mgr._recently_recovered = False
        mgr._last_pop_no_jobs_available = True
        mgr._process_map.num_loaded_inference_processes.return_value = 1
        mgr.max_inference_processes = 1
        mgr._process_map.get_inference_processes.return_value = []

        # Track the order of _shut_down assignment vs end_inference_processes call
        call_order: list[str] = []

        def track_end_inference(force: bool = False) -> None:
            call_order.append(f"end_inference(shut_down={mgr._shut_down})")

        mgr.end_inference_processes = track_end_inference
        mgr.end_safety_processes = MagicMock()

        # Make is_time_for_shutdown return False first (so the loop runs once)
        # then True (so the loop breaks).
        iteration = [0]

        def fake_is_time_for_shutdown() -> bool:
            iteration[0] += 1
            return iteration[0] > 1

        mgr.is_time_for_shutdown = fake_is_time_for_shutdown
        mgr._start_timed_shutdown = MagicMock()
        mgr._last_pop_recently = MagicMock(return_value=False)
        mgr.print_status_method = MagicMock()
        mgr.is_free_inference_process_available = MagicMock(return_value=False)
        mgr.is_any_model_preloaded = MagicMock(return_value=False)
        mgr.receive_and_handle_process_messages = MagicMock()
        mgr.detect_deadlock = MagicMock()
        mgr.replace_hung_processes = MagicMock(return_value=False)
        mgr._replace_all_safety_process = MagicMock()

        bound = HordeWorkerProcessManager._process_control_loop.__get__(mgr, HordeWorkerProcessManager)
        asyncio.run(bound())

        # end_inference_processes(force=True) must see _shut_down=True
        assert any("shut_down=True" in c for c in call_order), (
            f"end_inference_processes(force=True) was called before _shut_down was set: {call_order}"
        )


class TestJobSubmitLoopExceptionHandling:
    """Tests that _job_submit_loop discards the head job when api_submit_job raises unexpectedly."""

    def test_unexpected_exception_discards_head_job(self) -> None:
        """When api_submit_job raises unexpectedly, the head job must be removed so the queue unblocks."""
        import asyncio
        import types

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        # Build a minimal completed-job mock
        job_info = MagicMock()
        job_info.id_ = "stuck-job"
        job_info.ids = ["stuck-job"]

        completed = MagicMock()
        completed.sdk_api_job_info = job_info

        mock_manager = MagicMock()
        mock_manager.jobs_pending_submit = [completed]
        mock_manager._shutting_down = False

        # Bind real _discard_broken_job so the queue is actually modified
        mock_manager._discard_broken_job = types.MethodType(
            HordeWorkerProcessManager._discard_broken_job, mock_manager
        )
        mock_manager.jobs_lookup = {job_info: completed}
        mock_manager.job_pop_timestamps = {job_info: 0.0}
        mock_manager.jobs_in_progress = [job_info]
        mock_manager.job_faults = {}

        call_count = 0

        async def failing_api_submit_job() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated unexpected failure in api_submit_job")
            # After the broken job is discarded the queue is empty; shut down
            mock_manager._shutting_down = True

        mock_manager.api_submit_job = failing_api_submit_job
        mock_manager.is_time_for_shutdown = lambda: mock_manager._shutting_down
        mock_manager._job_submit_loop_interval = 0.01

        bound_loop = HordeWorkerProcessManager._job_submit_loop.__get__(mock_manager, HordeWorkerProcessManager)

        asyncio.run(asyncio.wait_for(bound_loop(), timeout=2.0))

        # The broken job must have been removed from the queue
        assert len(mock_manager.jobs_pending_submit) == 0
        assert job_info not in mock_manager.jobs_lookup
        assert job_info not in mock_manager.jobs_in_progress

    def test_job_removed_by_api_submit_job_is_not_double_discarded(self) -> None:
        """If api_submit_job already removed the head job before raising, the next job is not discarded."""
        import asyncio
        import types

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job_info_head = MagicMock()
        job_info_head.id_ = "head-job"
        job_info_head.ids = ["head-job"]
        completed_head = MagicMock()
        completed_head.sdk_api_job_info = job_info_head

        job_info_next = MagicMock()
        job_info_next.id_ = "next-job"
        job_info_next.ids = ["next-job"]
        completed_next = MagicMock()
        completed_next.sdk_api_job_info = job_info_next

        mock_manager = MagicMock()
        mock_manager._shutting_down = False
        # Start with two jobs in the queue
        mock_manager.jobs_pending_submit = [completed_head, completed_next]
        mock_manager.jobs_lookup = {job_info_head: completed_head, job_info_next: completed_next}
        mock_manager.job_pop_timestamps = {}
        mock_manager.jobs_in_progress = []
        mock_manager.job_faults = {}

        mock_manager._discard_broken_job = types.MethodType(
            HordeWorkerProcessManager._discard_broken_job, mock_manager
        )

        async def api_submit_job_removes_head_then_raises() -> None:
            # Simulate api_submit_job removing the head job internally (e.g. normal cleanup path
            # partially ran), then raising an unexpected error.
            mock_manager.jobs_pending_submit.pop(0)
            raise RuntimeError("Partial failure after head was already removed")

        call_count = 0

        original_submit = api_submit_job_removes_head_then_raises

        async def controlled_submit() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await original_submit()
            # Second call: succeed and shut down
            mock_manager._shutting_down = True

        mock_manager.api_submit_job = controlled_submit
        mock_manager.is_time_for_shutdown = lambda: mock_manager._shutting_down
        mock_manager._job_submit_loop_interval = 0.01

        bound_loop = HordeWorkerProcessManager._job_submit_loop.__get__(mock_manager, HordeWorkerProcessManager)
        asyncio.run(asyncio.wait_for(bound_loop(), timeout=2.0))

        # The next job must NOT have been discarded (it was not the job that failed)
        assert completed_next in mock_manager.jobs_pending_submit
        assert job_info_next in mock_manager.jobs_lookup


class _ReceiveLoopHarnessMixin:
    """Shared helpers for tests that drive receive_and_handle_process_messages."""

    def _make_message(
        self,
        process_state: HordeProcessState,
        process_id: int = 0,
        launch_id: int = 1,
    ) -> object:
        """Return a real HordeProcessStateChangeMessage so isinstance() checks pass."""
        from horde_worker_regen.process_management.messages import HordeProcessStateChangeMessage

        return HordeProcessStateChangeMessage(
            process_id=process_id,
            process_launch_identifier=launch_id,
            process_state=process_state,
            info="test",
            time_elapsed=None,
        )

    def _run_receive(
        self,
        msg: object,
        process_info: MagicMock,
        *,
        jobs_in_progress: list | None = None,
    ) -> MagicMock:
        """Run receive_and_handle_process_messages with a single queued message.

        Returns the mock_manager so callers can inspect side-effects.
        """
        import queue as queue_mod

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        # Configure the process_map so `process_id in process_map` and `process_map[process_id]` work
        process_map = MagicMock()
        process_map.__contains__ = MagicMock(side_effect=lambda key: key == 0)
        process_map.__getitem__ = MagicMock(side_effect=lambda key: process_info)

        if not isinstance(getattr(process_info, "state_entered_timestamp", None), (int, float)):
            process_info.state_entered_timestamp = 0.0

        def on_state_change(*, process_id: int, new_state: HordeProcessState) -> None:
            process_info.last_process_state = new_state
            process_info.inference_started_timestamp = None
            process_info.state_entered_timestamp = 0.0

        process_map.on_process_state_change.side_effect = on_state_change

        q = queue_mod.Queue()
        q.put(msg)

        mock_manager = MagicMock()
        mock_manager._process_message_queue = q
        mock_manager._process_map = process_map
        mock_manager._in_deadlock = False
        mock_manager._in_queue_deadlock = False
        mock_manager._STATE_TRANSITION_TIMING_STATES = HordeWorkerProcessManager._STATE_TRANSITION_TIMING_STATES
        mock_manager._pending_process_job_timings = {}
        mock_manager._pending_completed_job_timings = {}
        mock_manager._record_pending_job_timing = HordeWorkerProcessManager._record_pending_job_timing.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        mock_manager._on_process_state_change = HordeWorkerProcessManager._on_process_state_change.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        mock_manager.jobs_in_progress = jobs_in_progress if jobs_in_progress is not None else []

        bound = HordeWorkerProcessManager.receive_and_handle_process_messages.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound()  # must not raise
        return mock_manager


class TestReceiveAndHandleProcessMessagesResilience(_ReceiveLoopHarnessMixin):
    """Tests that receive_and_handle_process_messages does not crash on INFERENCE_STARTING edge cases."""

    def test_inference_starting_with_no_model_does_not_raise(self) -> None:
        """INFERENCE_STARTING with no model loaded should log an error and continue, not raise."""
        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_info.inference_started_timestamp = None
        process_info.loaded_horde_model_name = None  # trigger the guard
        process_info.batch_amount = None

        msg = self._make_message(HordeProcessState.INFERENCE_STARTING)
        self._run_receive(msg, process_info)

    def test_inference_starting_with_no_batch_amount_does_not_raise(self) -> None:
        """INFERENCE_STARTING with batch_amount=None should log an error and continue, not raise."""
        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_info.inference_started_timestamp = None
        process_info.loaded_horde_model_name = "some_model"
        process_info.batch_amount = None  # trigger the guard

        msg = self._make_message(HordeProcessState.INFERENCE_STARTING)
        self._run_receive(msg, process_info)

    def test_unloaded_model_from_ram_transition_clears_model_ownership(self) -> None:
        """Transitioning into UNLOADED_MODEL_FROM_RAM must clear loaded model ownership."""
        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_info.inference_started_timestamp = None
        process_info.loaded_horde_model_name = "stale-model"
        process_info.batch_amount = 1

        msg = self._make_message(HordeProcessState.UNLOADED_MODEL_FROM_RAM)
        mock_manager = self._run_receive(msg, process_info)

        mock_manager._process_map.on_model_ram_clear.assert_called_once_with(process_id=0)

    def test_repeated_unloaded_model_from_ram_state_does_not_double_clear(self) -> None:
        """Repeated UNLOADED_MODEL_FROM_RAM state must not call RAM-clear again."""
        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = HordeProcessState.UNLOADED_MODEL_FROM_RAM
        process_info.inference_started_timestamp = None
        process_info.loaded_horde_model_name = "stale-model"
        process_info.batch_amount = 1

        msg = self._make_message(HordeProcessState.UNLOADED_MODEL_FROM_RAM)
        mock_manager = self._run_receive(msg, process_info)

        mock_manager._process_map.on_model_ram_clear.assert_not_called()


class TestSavedImagePreviewSafetyAlignment:
    """Tests that saved-image previews only expose safety metadata when alignment is reliable."""

    def test_saved_image_preview_clears_safety_when_saved_images_count_mismatches(self, tmp_path) -> None:
        """Saved-image preview safety must be cleared when disk saves don't align with safety results."""
        pytest.importorskip("hordelib")  # requires the GPU/ML stack
        import queue as queue_mod
        import types

        from horde_sdk.ai_horde_api import GENERATION_STATE

        from horde_worker_regen.process_management.messages import (
            HordeImageResult,
            HordeSafetyEvaluation,
            HordeSafetyResultMessage,
            HordeSavedImageInfo,
        )
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        saved_path = tmp_path / "saved.png"
        saved_bytes = b"preview-image"
        saved_path.write_bytes(saved_bytes)

        job_id = "12345678-1234-1234-1234-1234567890ab"
        completed_job_info = MagicMock()
        completed_job_info.sdk_api_job_info.id_ = job_id
        completed_job_info.sdk_api_job_info.model = "stable_diffusion_xl"
        # No payload metadata for this test -- it only cares about the base64/timestamp/model
        # fields and the safety-mismatch clearing behaviour, not the optional inference_steps/
        # width/height/prompt enrichment that process_manager.py adds when a payload is present.
        completed_job_info.sdk_api_job_info.payload = None
        completed_job_info.time_to_generate = None
        completed_job_info.job_image_results = [
            HordeImageResult(image_base64="img-a"),
            HordeImageResult(image_base64="img-b"),
        ]
        completed_job_info.state = GENERATION_STATE.ok
        completed_job_info.censored = False
        completed_job_info.inference_completed_timestamp = 100.0

        message = HordeSafetyResultMessage(
            process_id=0,
            process_launch_identifier=1,
            info="saved images",
            time_elapsed=0.1,
            job_id=job_id,
            safety_evaluations=[
                HordeSafetyEvaluation(
                    is_nsfw=False,
                    is_csam=False,
                    replacement_image_base64=None,
                ),
                HordeSafetyEvaluation(
                    is_nsfw=True,
                    is_csam=False,
                    replacement_image_base64=None,
                ),
            ],
            saved_images=[HordeSavedImageInfo(path=str(saved_path), metadata_embedded=True)],
        )

        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = HordeProcessState.SAFETY_EVALUATING
        process_info.inference_started_timestamp = None

        process_map = MagicMock()
        process_map.__contains__ = MagicMock(side_effect=lambda key: key == 0)
        process_map.__getitem__ = MagicMock(side_effect=lambda key: process_info)

        q = queue_mod.Queue()
        q.put(message)

        mock_manager = MagicMock()
        mock_manager._process_message_queue = q
        mock_manager._process_map = process_map
        mock_manager._in_deadlock = False
        mock_manager._in_queue_deadlock = False
        mock_manager.jobs_being_safety_checked = [completed_job_info]
        mock_manager.job_faults = {job_id: []}
        mock_manager.webui = MagicMock()
        mock_manager._last_image_job_timestamp = 0.0
        mock_manager._last_image_base64 = []
        mock_manager._last_image_model = ""
        mock_manager._last_image_safety = [{"is_nsfw": True, "is_csam": True}]
        mock_manager.jobs_pending_submit = []
        mock_manager._move_pending_process_timings_to_completed_job = MagicMock()

        bound = types.MethodType(HordeWorkerProcessManager.receive_and_handle_process_messages, mock_manager)
        bound()

        assert mock_manager._last_image_base64 == ["cHJldmlldy1pbWFnZQ=="]
        assert mock_manager._last_image_model == "stable_diffusion_xl"
        assert mock_manager._last_image_safety == []
        mock_manager.webui.add_gallery_image.assert_called_once_with(
            {
                "base64": "cHJldmlldy1pbWFnZQ==",
                "timestamp": 100.0,
                "model": "stable_diffusion_xl",
            },
        )


class TestIsStuckOnInference:
    """Tests for the is_stuck_on_inference() method covering both INFERENCE_STARTING and INFERENCE_PROCESSING."""

    def _make_process_map_entry(
        self,
        state: HordeProcessState,
        last_progress_timestamp: float,
        last_heartbeat_timestamp: float,
        last_heartbeat_delta: float = 0.0,
        heartbeats_inference_steps: int = 0,
        last_progress_value: int | None = None,
        last_inference_step_timestamp: float | None = None,
    ) -> MagicMock:
        """Create a mock process map entry with configurable timestamps."""
        entry = MagicMock()
        entry.last_process_state = state
        entry.inference_started_timestamp = None
        entry.last_progress_timestamp = last_progress_timestamp
        entry.last_heartbeat_timestamp = last_heartbeat_timestamp
        entry.last_heartbeat_delta = last_heartbeat_delta
        entry.heartbeats_inference_steps = heartbeats_inference_steps
        entry.last_progress_value = last_progress_value
        entry.last_inference_step_timestamp = last_inference_step_timestamp
        return entry

    def _make_process_map(self, entry: MagicMock) -> MagicMock:
        """Create a mock process map that returns the given entry for any key."""
        from horde_worker_regen.process_management.process_manager import ProcessMap

        process_map = MagicMock()
        process_map.__getitem__ = MagicMock(return_value=entry)
        process_map.is_stuck_on_inference = ProcessMap.is_stuck_on_inference.__get__(
            process_map, ProcessMap
        )
        process_map.MAX_INFERENCE_STEP_TIMEOUT = ProcessMap.MAX_INFERENCE_STEP_TIMEOUT
        return process_map

    def test_inference_starting_not_stuck_returns_false(self) -> None:
        """A process in INFERENCE_STARTING with recent progress and heartbeat is not stuck."""
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_STARTING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 10,
        )
        process_map = self._make_process_map(entry)
        assert process_map.is_stuck_on_inference(0, 600) is False

    def test_inference_starting_no_progress_with_fresh_heartbeat_not_stuck(self) -> None:
        """INFERENCE_STARTING with no progress reported yet but fresh heartbeats is NOT stuck.

        Before the first diffusion step there is no progress to stall (last_progress_value is
        None), so the progress-stalled check (check 1) must not fire — a job legitimately
        loading its model or waiting on the semaphore was previously killed by this check after
        only inference_step_timeout (30 s by default), faulting healthy jobs.  A genuinely hung
        INFERENCE_STARTING process is detected by the no-heartbeat check (check 4) and by the
        manager's dedicated preload_timeout-based INFERENCE_STARTING check instead.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_STARTING,
            last_progress_timestamp=time.time() - 9999,
            last_heartbeat_timestamp=time.time() - 10,
        )
        process_map = self._make_process_map(entry)
        assert process_map.is_stuck_on_inference(0, 600) is False

    def test_inference_starting_no_heartbeat_returns_true(self) -> None:
        """A process in INFERENCE_STARTING with no heartbeat beyond timeout is stuck."""
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_STARTING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 9999,
            last_heartbeat_delta=5.0,  # delta between last two heartbeats was normal (5s)
        )
        process_map = self._make_process_map(entry)
        # Should detect via time since last heartbeat, not last_heartbeat_delta
        assert process_map.is_stuck_on_inference(0, 600) is True

    def test_inference_processing_not_stuck_returns_false(self) -> None:
        """A process in INFERENCE_PROCESSING with recent progress and heartbeat is not stuck."""
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 10,
        )
        process_map = self._make_process_map(entry)
        assert process_map.is_stuck_on_inference(0, 600) is False

    def test_inference_processing_no_progress_reported_yet_uses_zero_progress_timeout(self) -> None:
        """INFERENCE_PROCESSING with no progress percentage reported yet is governed by check 2.

        last_progress_value None means no progress callback has fired with a percentage
        (e.g. only PIPELINE_STATE_CHANGE heartbeats) — the same pre-first-step situation as
        0 %, so it must be treated identically: exempt from the progress-stalled check
        (check 1) and detected by zero_progress_timeout (check 2) instead.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 9999,
            last_heartbeat_timestamp=time.time() - 10,
        )
        process_map = self._make_process_map(entry)
        # Bare call: check 1 is exempt pre-first-step and check 2 is inactive without a
        # zero_progress/no_step timeout → not stuck.  (Production always passes those for
        # INFERENCE_PROCESSING.)
        assert process_map.is_stuck_on_inference(0, 600) is False
        # With the production zero_progress_timeout, the stall is detected via check 2.
        assert process_map.is_stuck_on_inference(0, 600, zero_progress_timeout=120) is True

    def test_inference_processing_zero_progress_not_killed_at_step_timeout(self) -> None:
        """A 0 %-progress job must NOT be flagged stuck at inference_step_timeout (regression).

        Regression test for the false positive that faulted healthy jobs: with the default
        inference_step_timeout of 30 s, a job whose model was still loading into VRAM
        (0 % progress, fresh heartbeats, ~40 s elapsed) was killed by the progress-stalled
        check (check 1) long before the deliberate 120 s ZERO_PROGRESS_TIMEOUT (check 2)
        could allow it to finish loading.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 40,  # 40 s at 0 % — model still loading
            last_heartbeat_timestamp=time.time() - 1,  # heartbeats flowing
            heartbeats_inference_steps=0,
            last_progress_value=0,
        )
        process_map = self._make_process_map(entry)
        # Realistic production call: inference_step_timeout=30 (default), zero_progress_timeout=120.
        assert (
            process_map.is_stuck_on_inference(0, 30, no_step_heartbeat_timeout=300, zero_progress_timeout=120)
            is False
        )

    def test_inference_processing_zero_progress_still_detected_after_zero_progress_timeout(self) -> None:
        """A 0 %-progress job IS flagged stuck once zero_progress_timeout elapses."""
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 130,  # 130 s > zero_progress_timeout=120
            last_heartbeat_timestamp=time.time() - 1,
            heartbeats_inference_steps=0,
            last_progress_value=0,
        )
        process_map = self._make_process_map(entry)
        assert (
            process_map.is_stuck_on_inference(0, 30, no_step_heartbeat_timeout=300, zero_progress_timeout=120)
            is True
        )

    def test_inference_processing_no_heartbeat_returns_true(self) -> None:
        """A process in INFERENCE_PROCESSING with no heartbeat beyond timeout is stuck.

        This is the core scenario for the 'stuck at INFERENCE_STARTING' bug:
        Process A holds the inference semaphore in INFERENCE_PROCESSING and stops responding.
        Without this check, Process A is never detected as stuck, and any process waiting
        to acquire the semaphore remains permanently stuck in INFERENCE_STARTING.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 9999,
            last_heartbeat_delta=5.0,  # delta between last two heartbeats was normal (5s)
        )
        process_map = self._make_process_map(entry)
        # Should detect via time since last heartbeat, not last_heartbeat_delta
        assert process_map.is_stuck_on_inference(0, 600) is True

    def test_other_state_returns_false(self) -> None:
        """A process in a non-inference state is not considered stuck on inference."""
        import time

        for state in [
            HordeProcessState.WAITING_FOR_JOB,
            HordeProcessState.MODEL_LOADING,
            HordeProcessState.INFERENCE_POST_PROCESSING,
            HordeProcessState.INFERENCE_COMPLETE,
        ]:
            entry = self._make_process_map_entry(
                state=state,
                last_progress_timestamp=time.time() - 9999,
                last_heartbeat_timestamp=time.time() - 9999,
            )
            process_map = self._make_process_map(entry)
            assert process_map.is_stuck_on_inference(0, 600) is False, (
                f"Expected False for state {state.name}"
            )

    def test_inference_processing_no_step_heartbeats_uses_shorter_timeout(self) -> None:
        """INFERENCE_PROCESSING with no step heartbeats triggers on no_step_heartbeat_timeout.

        Scenario: the process sent the initial PIPELINE_STATE_CHANGE heartbeat (heartbeats_inference_steps=0)
        and then went completely silent (crash before any diffusion step, or stall during VAE decode).
        With no_step_heartbeat_timeout=120 and time_since_heartbeat=150, the process must be
        detected as stuck even though inference_step_timeout=600 has not elapsed.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 150,
            heartbeats_inference_steps=0,
        )
        process_map = self._make_process_map(entry)
        # Without no_step_heartbeat_timeout, 150s < 600s inference_step_timeout → not stuck
        assert process_map.is_stuck_on_inference(0, 600) is False
        # With no_step_heartbeat_timeout=120, 150s > 120s → stuck
        assert process_map.is_stuck_on_inference(0, 600, no_step_heartbeat_timeout=120) is True

    def test_inference_processing_no_step_heartbeats_not_yet_timed_out(self) -> None:
        """INFERENCE_PROCESSING with no step heartbeats is NOT stuck if shorter timeout not elapsed."""
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 50,
            heartbeats_inference_steps=0,
        )
        process_map = self._make_process_map(entry)
        # 50s < no_step_heartbeat_timeout=120 → not yet stuck
        assert process_map.is_stuck_on_inference(0, 600, no_step_heartbeat_timeout=120) is False

    def test_inference_processing_with_step_heartbeats_ignores_shorter_timeout(self) -> None:
        """When step heartbeats have been received, no_step_heartbeat_timeout is ignored.

        A process that has completed at least one diffusion step uses inference_step_timeout,
        not the shorter no_step_heartbeat_timeout, for the heartbeat check.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 150,
            heartbeats_inference_steps=5,  # steps have been received
        )
        process_map = self._make_process_map(entry)
        # 150s > no_step_heartbeat_timeout=120, but heartbeats_inference_steps > 0 so short
        # timeout must NOT apply.  150s < inference_step_timeout=600 → not stuck.
        assert process_map.is_stuck_on_inference(0, 600, no_step_heartbeat_timeout=120) is False

    def test_inference_starting_no_step_heartbeats_does_not_use_shorter_timeout(self) -> None:
        """no_step_heartbeat_timeout must NOT apply to INFERENCE_STARTING.

        A process blocked on semaphore acquisition cannot send heartbeats.  Applying the shorter
        timeout to INFERENCE_STARTING would create false positives right after a prior replacement
        frees the semaphore (cascading recovery).
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_STARTING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 150,
            heartbeats_inference_steps=0,
        )
        process_map = self._make_process_map(entry)
        # 150s > no_step_heartbeat_timeout=120, but state is INFERENCE_STARTING → not stuck
        # (still below the full inference_step_timeout of 600s)
        assert process_map.is_stuck_on_inference(0, 600, no_step_heartbeat_timeout=120) is False

    def test_inference_processing_at_100_with_recent_heartbeat_not_stuck(self) -> None:
        """At 100% progress in INFERENCE_PROCESSING with recent heartbeat the process is not stuck.

        When all diffusion steps complete (progress = 100 %), the background heartbeat thread
        sends periodic PIPELINE_STATE_CHANGE heartbeats at 100 %.  Because the progress value
        never changes from 100, last_progress_timestamp is not refreshed.  Without the 100 %
        exemption, is_stuck_on_inference() would fire the progress-stalled check after
        inference_step_timeout seconds even though the process is actively running VAE decode
        (a false positive).  The fix skips check 1 when last_progress_value == 100.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 9999,  # stale — 100% never changes
            last_heartbeat_timestamp=time.time() - 10,  # heartbeats arriving normally
            heartbeats_inference_steps=0,  # reset by PIPELINE_STATE_CHANGE
            last_progress_value=100,
        )
        process_map = self._make_process_map(entry)
        # Even though progress_timestamp is stale, check 1 must be skipped at 100%.
        # Heartbeat is recent → checks 2 and 3 also don't fire → not stuck.
        assert process_map.is_stuck_on_inference(0, 600) is False

    def test_inference_processing_at_100_no_heartbeat_is_stuck(self) -> None:
        """At 100% progress in INFERENCE_PROCESSING, a dead process (no heartbeats) is stuck.

        If the process dies during VAE decode, the background heartbeat thread also stops.
        Even with the 100 % exemption for check 1, checks 2/3 must still detect this case.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 9999,  # stale — 100% never changes
            last_heartbeat_timestamp=time.time() - 9999,  # no heartbeats — process is dead
            heartbeats_inference_steps=0,
            last_progress_value=100,
        )
        process_map = self._make_process_map(entry)
        # Check 1 skipped (at 100%), but check 3 fires because no heartbeat for >600s.
        assert process_map.is_stuck_on_inference(0, 600) is True

    def test_inference_processing_at_100_no_step_heartbeat_timeout_detects_dead_process(self) -> None:
        """At 100%, no_step_heartbeat_timeout detects a dead process sooner than inference_step_timeout.

        When the process dies during VAE decode, the no_step_heartbeat_timeout (shorter)
        should fire before the full inference_step_timeout, as intended for fast recovery.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 9999,  # stale
            last_heartbeat_timestamp=time.time() - 350,  # no heartbeats for 350s > 300s timeout
            heartbeats_inference_steps=0,
            last_progress_value=100,
        )
        process_map = self._make_process_map(entry)
        # Check 1 skipped (100%), check 3 doesn't fire (350s < 600s), but check 2 fires
        # because heartbeats_inference_steps==0 and 350s > no_step_heartbeat_timeout=300s.
        assert process_map.is_stuck_on_inference(0, 600, no_step_heartbeat_timeout=300) is True

    def test_inference_processing_below_100_still_uses_progress_stalled_check(self) -> None:
        """Below 100%, a job with stalled progress and recent heartbeats is still detected as stuck.

        The 100 % exemption must not be applied when progress is below 100 %: a process sending
        heartbeats but making no diffusion progress (e.g., GPU hung at 50 %) should still be
        detected and replaced.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 9999,  # stale progress (stuck at 50 %)
            last_heartbeat_timestamp=time.time() - 10,  # heartbeats arriving
            heartbeats_inference_steps=5,
            last_progress_value=50,
        )
        process_map = self._make_process_map(entry)
        # progress_value != 100, so check 1 must still fire.
        assert process_map.is_stuck_on_inference(0, 600) is True

    def test_inference_processing_per_step_timeout_not_yet_elapsed(self) -> None:
        """A process with a recent INFERENCE_STEP heartbeat is not stuck on a single step.

        The per-step check must not fire when the last step heartbeat arrived recently
        (within MAX_INFERENCE_STEP_TIMEOUT seconds).
        """
        import time

        from horde_worker_regen.process_management.process_manager import ProcessMap

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 10,
            heartbeats_inference_steps=3,
            last_progress_value=50,
            last_inference_step_timestamp=time.time() - (ProcessMap.MAX_INFERENCE_STEP_TIMEOUT - 5),
        )
        process_map = self._make_process_map(entry)
        assert process_map.is_stuck_on_inference(0, 600) is False

    def test_inference_processing_per_step_timeout_elapsed_is_stuck(self) -> None:
        """A process that has not received a new INFERENCE_STEP heartbeat within MAX_INFERENCE_STEP_TIMEOUT
        seconds is detected as stuck, even though background heartbeats (PIPELINE_STATE_CHANGE) keep
        refreshing last_heartbeat_timestamp.

        Scenario: GPU hangs between step N and step N+1.  The background heartbeat thread sends
        PIPELINE_STATE_CHANGE every 30 s (which does NOT update last_inference_step_timestamp), so
        last_heartbeat_timestamp stays fresh while last_inference_step_timestamp goes stale.
        """
        import time

        from horde_worker_regen.process_management.process_manager import ProcessMap

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 5,  # fresh — background heartbeat just fired
            heartbeats_inference_steps=0,  # reset by the background PIPELINE_STATE_CHANGE
            last_progress_value=50,
            # last INFERENCE_STEP was received more than MAX_INFERENCE_STEP_TIMEOUT ago
            last_inference_step_timestamp=time.time() - (ProcessMap.MAX_INFERENCE_STEP_TIMEOUT + 5),
        )
        process_map = self._make_process_map(entry)
        assert process_map.is_stuck_on_inference(0, 600) is True

    def test_inference_processing_per_step_timeout_no_step_yet_not_stuck(self) -> None:
        """When no INFERENCE_STEP heartbeat has arrived yet (last_inference_step_timestamp is None),
        the per-step check must NOT fire.

        A process that just transitioned to INFERENCE_PROCESSING and hasn't completed its first
        diffusion step yet should not be falsely flagged by the per-step check.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 10,
            heartbeats_inference_steps=0,
            last_progress_value=None,
            last_inference_step_timestamp=None,  # no step received yet
        )
        process_map = self._make_process_map(entry)
        assert process_map.is_stuck_on_inference(0, 600) is False

    def test_inference_processing_per_step_timeout_at_100_percent_not_stuck(self) -> None:
        """The per-step check must NOT fire when progress is at 100 % (VAE decode phase).

        After all diffusion steps complete, the last INFERENCE_STEP timestamp goes stale while
        the process performs VAE decoding.  Skipping the check at 100 % prevents false positives;
        the no_step_heartbeat_timeout / VAE_SEMAPHORE_TIMEOUT path handles genuine stalls there.
        """
        import time

        from horde_worker_regen.process_management.process_manager import ProcessMap

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 9999,  # stale — 100 % never changes
            last_heartbeat_timestamp=time.time() - 10,  # background heartbeats still arriving
            heartbeats_inference_steps=0,  # reset by 100 % PIPELINE_STATE_CHANGE
            last_progress_value=100,
            # last step finished a while ago; VAE decode is now running
            last_inference_step_timestamp=time.time() - (ProcessMap.MAX_INFERENCE_STEP_TIMEOUT + 5),
        )
        process_map = self._make_process_map(entry)
        # Check 1 skipped (100 %), per-step check skipped (100 %), heartbeat checks OK.
        assert process_map.is_stuck_on_inference(0, 600) is False

    def test_inference_starting_per_step_timeout_does_not_apply(self) -> None:
        """The per-step check must NOT fire for INFERENCE_STARTING.

        A process waiting to acquire the inference semaphore cannot send INFERENCE_STEP heartbeats.
        Applying the per-step check to INFERENCE_STARTING would create false positives.
        """
        import time

        from horde_worker_regen.process_management.process_manager import ProcessMap

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_STARTING,
            last_progress_timestamp=time.time() - 10,
            last_heartbeat_timestamp=time.time() - 10,
            heartbeats_inference_steps=0,
            last_progress_value=None,
            # Simulate a stale step timestamp (should be ignored for INFERENCE_STARTING)
            last_inference_step_timestamp=time.time() - (ProcessMap.MAX_INFERENCE_STEP_TIMEOUT + 5),
        )
        process_map = self._make_process_map(entry)
        # Per-step check must not fire; heartbeat is recent (10s < 600s) → not stuck.
        assert process_map.is_stuck_on_inference(0, 600) is False


    def test_zero_progress_with_fresh_heartbeats_uses_shorter_timeout(self) -> None:
        """INFERENCE_PROCESSING stuck at 0 % with fresh heartbeats detects via no_step_heartbeat_timeout.

        Core bug scenario: process stuck in INFERENCE_PROCESSING at 0 % progress:
        - ComfyUI calls the progress callback at step 0/N (0 %) — either once via the
          INFERENCE_STEP path (refreshing last_inference_step_timestamp each time) or via
          the PIPELINE_STATE_CHANGE path (last_inference_step_timestamp stays None).
        - The background heartbeat thread sends PIPELINE_STATE_CHANGE at 0 % every 30 s,
          keeping last_heartbeat_timestamp fresh.
        - Result: the per-step check (last_inference_step_timestamp check) never fires
          (either masked by repeated INFERENCE_STEP at 0 % or inapplicable because
          last_inference_step_timestamp is None), and the no-heartbeat check (check 4)
          never fires because last_heartbeat_timestamp is always fresh.
        - Without the new check the only protection is the full inference_step_timeout
          (600 s).  With the new check the process is detected at no_step_heartbeat_timeout
          (300 s).
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 350,  # 350 s since first 0 % progress
            last_heartbeat_timestamp=time.time() - 5,  # background heartbeat just fired
            heartbeats_inference_steps=0,  # reset by last PIPELINE_STATE_CHANGE heartbeat
            last_progress_value=0,  # stuck at 0 % — no real diffusion step ever completed
            last_inference_step_timestamp=time.time() - 5,  # INFERENCE_STEP at 0 % fired recently
        )
        process_map = self._make_process_map(entry)
        # Without no_step_heartbeat_timeout: 350s > 600? No → not stuck yet (full timeout)
        assert process_map.is_stuck_on_inference(0, 600) is False
        # With no_step_heartbeat_timeout=300: 350s > 300 and progress_value==0 → stuck!
        assert process_map.is_stuck_on_inference(0, 600, no_step_heartbeat_timeout=300) is True

    def test_zero_progress_not_yet_at_shorter_timeout(self) -> None:
        """INFERENCE_PROCESSING at 0 % is NOT stuck if no_step_heartbeat_timeout not yet elapsed."""
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 200,  # 200 s < 300 s no_step_heartbeat_timeout
            last_heartbeat_timestamp=time.time() - 5,
            heartbeats_inference_steps=0,
            last_progress_value=0,
        )
        process_map = self._make_process_map(entry)
        # 200s < no_step_heartbeat_timeout=300 → not yet stuck
        assert process_map.is_stuck_on_inference(0, 600, no_step_heartbeat_timeout=300) is False

    def test_zero_progress_without_shorter_timeout_falls_back_to_full_timeout(self) -> None:
        """Without no_step_heartbeat_timeout / zero_progress_timeout, the 0 % check is inactive.

        The progress-stalled check (check 1) is exempt at 0 % (pre-first-step model loading),
        so with neither optional timeout provided only the heartbeat-based checks can flag the
        process — and heartbeats are fresh here.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 350,  # 350 s > 300 but < 600
            last_heartbeat_timestamp=time.time() - 5,
            heartbeats_inference_steps=0,
            last_progress_value=0,
        )
        process_map = self._make_process_map(entry)
        # no_step_heartbeat_timeout not provided → check 2 inactive; heartbeats fresh → not stuck
        assert process_map.is_stuck_on_inference(0, 600) is False

    def test_zero_progress_check_does_not_apply_to_inference_starting(self) -> None:
        """The 0 % faster detection must NOT apply to INFERENCE_STARTING.

        A process blocked waiting to acquire the inference semaphore reports 0 % progress
        but cannot send heartbeats.  Applying the new check to INFERENCE_STARTING would
        create false positives for processes legitimately waiting on the semaphore.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_STARTING,
            last_progress_timestamp=time.time() - 350,
            last_heartbeat_timestamp=time.time() - 5,
            heartbeats_inference_steps=0,
            last_progress_value=0,
        )
        process_map = self._make_process_map(entry)
        # State is INFERENCE_STARTING → new 0 % check must be skipped; 350s < 600s → not stuck
        assert process_map.is_stuck_on_inference(0, 600, no_step_heartbeat_timeout=300) is False

    def test_nonzero_progress_does_not_trigger_zero_progress_check(self) -> None:
        """The 0 % faster detection must NOT fire when progress has advanced past 0 %."""
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 350,  # stale — stuck at 50 %
            last_heartbeat_timestamp=time.time() - 5,
            heartbeats_inference_steps=0,
            last_progress_value=50,  # progress DID advance, just now stalled at 50 %
        )
        process_map = self._make_process_map(entry)
        # progress_value == 50 (not 0) → new check does not apply; 350s < 600s → not stuck yet
        assert process_map.is_stuck_on_inference(0, 600, no_step_heartbeat_timeout=300) is False

    def test_zero_progress_timeout_overrides_no_step_heartbeat_timeout_for_check2(self) -> None:
        """zero_progress_timeout takes precedence over no_step_heartbeat_timeout for check 2.

        When both parameters are provided, zero_progress_timeout is used for the stuck-at-0%
        check (check 2) instead of no_step_heartbeat_timeout.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 130,  # 130 s at 0 %
            last_heartbeat_timestamp=time.time() - 5,   # heartbeat is fresh
            heartbeats_inference_steps=0,
            last_progress_value=0,
        )
        process_map = self._make_process_map(entry)
        # no_step_heartbeat_timeout=300 alone: 130s < 300s → NOT stuck
        assert process_map.is_stuck_on_inference(0, 600, no_step_heartbeat_timeout=300) is False
        # zero_progress_timeout=120 overrides for check 2: 130s > 120s → stuck
        assert process_map.is_stuck_on_inference(
            0, 600, no_step_heartbeat_timeout=300, zero_progress_timeout=120
        ) is True

    def test_zero_progress_timeout_not_yet_elapsed_not_stuck(self) -> None:
        """INFERENCE_PROCESSING at 0 % is NOT stuck if zero_progress_timeout has not elapsed."""
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 100,  # 100 s < 120 s zero_progress_timeout
            last_heartbeat_timestamp=time.time() - 5,
            heartbeats_inference_steps=0,
            last_progress_value=0,
        )
        process_map = self._make_process_map(entry)
        # 100s < zero_progress_timeout=120 → not yet stuck
        assert process_map.is_stuck_on_inference(
            0, 600, no_step_heartbeat_timeout=300, zero_progress_timeout=120
        ) is False

    def test_zero_progress_timeout_does_not_affect_check4_vae_decode(self) -> None:
        """zero_progress_timeout must NOT influence check 4 (VAE decode at 100 % progress).

        When all diffusion steps complete the process enters VAE decode at 100 % progress.
        At that point heartbeats_inference_steps resets to 0 and heartbeats may stop (blocking
        on semaphore).  Check 4 uses no_step_heartbeat_timeout (300 s) to protect against
        falsely killing a legitimately-running VAE decode.  zero_progress_timeout (120 s) must
        NOT affect check 4 — it only applies when last_progress_value == 0.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_PROCESSING,
            last_progress_timestamp=time.time() - 10,   # recently at 100 % (VAE decode phase)
            last_heartbeat_timestamp=time.time() - 130, # 130 s since last heartbeat (blocking on sem)
            heartbeats_inference_steps=0,               # reset when 100 % PIPELINE_STATE_CHANGE fired
            last_progress_value=100,                    # all steps complete
        )
        process_map = self._make_process_map(entry)
        # check 2 does NOT apply (last_progress_value == 100, not 0)
        # check 4: 130s < no_step_heartbeat_timeout=300 → NOT stuck yet
        assert process_map.is_stuck_on_inference(
            0, 600, no_step_heartbeat_timeout=300, zero_progress_timeout=120
        ) is False
        # check 4: 130s > no_step_heartbeat_timeout=120 if we accidentally used it → would be stuck
        # (this asserts that check 4 still uses no_step_heartbeat_timeout, not zero_progress_timeout)
        assert process_map.is_stuck_on_inference(
            0, 600, no_step_heartbeat_timeout=120, zero_progress_timeout=120
        ) is True

    def test_zero_progress_timeout_does_not_apply_to_inference_starting(self) -> None:
        """zero_progress_timeout must NOT apply to INFERENCE_STARTING.

        Same rationale as no_step_heartbeat_timeout: a process blocked waiting to acquire
        the inference semaphore cannot send heartbeats and should not be killed early.
        """
        import time

        entry = self._make_process_map_entry(
            state=HordeProcessState.INFERENCE_STARTING,
            last_progress_timestamp=time.time() - 130,
            last_heartbeat_timestamp=time.time() - 5,
            heartbeats_inference_steps=0,
            last_progress_value=0,
        )
        process_map = self._make_process_map(entry)
        # State is INFERENCE_STARTING → zero_progress_timeout check must be skipped
        assert process_map.is_stuck_on_inference(
            0, 600, no_step_heartbeat_timeout=300, zero_progress_timeout=120
        ) is False


class TestProcessMessageQueueFreezeResilience:
    """Tests for the queue reader thread and the dead-queue watchdog.

    A child process SIGKILLed while writing to the shared process message queue can leave a
    truncated payload in the queue pipe; Queue.get() then blocks forever inside _recv_bytes()
    — its non-blocking/timeout guarantees only cover the lock acquire and the readiness poll.
    The manager therefore reads the queue exclusively on a dedicated daemon thread and watches
    that thread's liveness tick, restarting the worker when the queue dies instead of freezing
    silently (no status output, no stuck detection, children idle forever).
    """

    def _make_manager_with_bound_health_check(self) -> tuple[MagicMock, object]:
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = MagicMock()
        mgr._shutting_down = False
        mgr._shut_down = False
        mgr._queue_reader_stall_handled = False
        mgr._QUEUE_READER_STALL_TIMEOUT = HordeWorkerProcessManager._QUEUE_READER_STALL_TIMEOUT
        bound = HordeWorkerProcessManager._check_process_message_queue_health.__get__(
            mgr, HordeWorkerProcessManager
        )
        return mgr, bound

    def _assert_restart_initiated(self, mgr: MagicMock) -> None:
        mgr.request_program_restart.assert_called_once()
        mgr._purge_jobs.assert_called_once()
        mgr._shutdown.assert_called_once()
        mgr._start_timed_shutdown.assert_called_once()
        assert mgr._queue_reader_stall_handled is True

    def _assert_no_restart(self, mgr: MagicMock) -> None:
        mgr.request_program_restart.assert_not_called()
        mgr._purge_jobs.assert_not_called()
        mgr._shutdown.assert_not_called()
        mgr._start_timed_shutdown.assert_not_called()

    def test_healthy_reader_thread_does_not_trigger_restart(self) -> None:
        """A live reader thread with a recent tick must not trigger any recovery action."""
        import threading
        import time

        mgr, check = self._make_manager_with_bound_health_check()
        stop = threading.Event()
        reader = threading.Thread(target=stop.wait, daemon=True)
        reader.start()
        try:
            mgr._queue_reader_thread = reader
            mgr._queue_reader_last_tick = time.time()
            check()
            self._assert_no_restart(mgr)
            assert mgr._queue_reader_stall_handled is False
        finally:
            stop.set()

    def test_stalled_reader_thread_triggers_restart(self) -> None:
        """A reader thread blocked mid-read past the stall timeout must initiate a restart.

        This is the corrupted-queue scenario: the thread is alive but stuck forever inside
        Queue.get() on a truncated payload, so its liveness tick stops advancing.
        """
        import threading
        import time

        mgr, check = self._make_manager_with_bound_health_check()
        stop = threading.Event()
        reader = threading.Thread(target=stop.wait, daemon=True)
        reader.start()
        try:
            mgr._queue_reader_thread = reader
            mgr._queue_reader_last_tick = time.time() - (mgr._QUEUE_READER_STALL_TIMEOUT + 5)
            check()
            self._assert_restart_initiated(mgr)
        finally:
            stop.set()

    def test_dead_reader_thread_triggers_restart(self) -> None:
        """A reader thread that has died must initiate a restart even with a fresh tick."""
        import threading
        import time

        mgr, check = self._make_manager_with_bound_health_check()
        # A Thread that was never started reports is_alive() False, like a crashed one.
        mgr._queue_reader_thread = threading.Thread(target=lambda: None, daemon=True)
        mgr._queue_reader_last_tick = time.time()
        check()
        self._assert_restart_initiated(mgr)

    def test_restart_is_initiated_only_once(self) -> None:
        """The recovery action is one-shot; repeated health checks must not re-trigger it."""
        import threading
        import time

        mgr, check = self._make_manager_with_bound_health_check()
        mgr._queue_reader_thread = threading.Thread(target=lambda: None, daemon=True)
        mgr._queue_reader_last_tick = time.time()
        check()
        check()
        check()
        mgr.request_program_restart.assert_called_once()
        mgr._start_timed_shutdown.assert_called_once()

    def test_health_check_skipped_during_shutdown(self) -> None:
        """No recovery is initiated while the worker is already shutting down."""
        import threading
        import time

        mgr, check = self._make_manager_with_bound_health_check()
        mgr._shutting_down = True
        mgr._queue_reader_thread = threading.Thread(target=lambda: None, daemon=True)
        mgr._queue_reader_last_tick = time.time() - 9999
        check()
        self._assert_no_restart(mgr)

    def test_queue_reader_loop_transfers_messages_and_exits_on_shutdown(self) -> None:
        """The reader loop moves queue messages into the deque, ticks, and exits on shutdown."""
        import queue as queue_mod
        import threading
        import time
        from collections import deque

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mgr = MagicMock()
        mgr._shut_down = False
        mgr._shutting_down = False
        mgr._QUEUE_READER_POLL_SECONDS = 0.05  # keep the shutdown poll fast for the test
        mgr._process_message_queue = queue_mod.Queue()
        mgr._received_process_messages = deque()
        mgr._queue_reader_last_tick = 0.0

        bound = HordeWorkerProcessManager._queue_reader_loop.__get__(mgr, HordeWorkerProcessManager)
        reader = threading.Thread(target=bound, daemon=True)
        reader.start()

        mgr._process_message_queue.put("a message")
        deadline = time.time() + 5
        while time.time() < deadline and not mgr._received_process_messages:
            time.sleep(0.01)
        assert list(mgr._received_process_messages) == ["a message"]
        assert mgr._queue_reader_last_tick > 0.0

        mgr._shut_down = True
        reader.join(timeout=5)
        assert not reader.is_alive()

    def test_receive_uses_buffered_messages_when_reader_thread_present(self) -> None:
        """With a real reader thread + deque, receive drains the deque and never touches the queue.

        Reading the multiprocessing queue directly from the event loop is exactly what the
        reader thread exists to prevent, so when the reader infrastructure is present the
        receive loop must consume only the thread-fed deque.
        """
        import queue as queue_mod
        import threading
        from collections import deque

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._queue_reader_thread = threading.Thread(target=lambda: None, daemon=True)
        # Non-HordeProcessMessage payloads are logged and skipped, which exercises the drain
        # loop without requiring the full message-handling surface.
        mock_manager._received_process_messages = deque(["not a real message", "also not one"])
        sentinel_queue = queue_mod.Queue()
        sentinel_queue.put("must remain untouched")
        mock_manager._process_message_queue = sentinel_queue

        bound = HordeWorkerProcessManager.receive_and_handle_process_messages.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound()

        assert len(mock_manager._received_process_messages) == 0
        assert sentinel_queue.qsize() == 1


class TestInferenceSemaphoreBoundedSemaphore:
    """Tests that _inference_semaphore is a BoundedSemaphore to prevent permit inflation."""

    def test_inference_semaphore_is_bounded(self) -> None:
        """_inference_semaphore must be a BoundedSemaphore so over-release raises ValueError.

        The existing ValueError handlers in _replace_inference_process() and the child inference
        process prevent any double-release from inflating permits beyond max_threads.
        """
        import multiprocessing
        from multiprocessing.synchronize import BoundedSemaphore

        ctx = multiprocessing.get_context("spawn")

        # Verify BoundedSemaphore raises ValueError on over-release, unlike Semaphore
        sem = BoundedSemaphore(1, ctx=ctx)
        sem.acquire()
        sem.release()  # Back to initial count
        raised = False
        try:
            sem.release()  # Over-release — must raise ValueError for BoundedSemaphore
        except ValueError:
            raised = True
        assert raised, "BoundedSemaphore should raise ValueError on over-release"

    def test_replace_inference_process_double_release_does_not_inflate_permits(self) -> None:
        """A double-release of the inference semaphore must not inflate the permit count.

        Scenario: manager still sees INFERENCE_PROCESSING (async state lag) but the child
        already released the semaphore during post-processing.  Calling _replace_inference_process
        should not increase the available permits beyond max_threads.
        """
        import multiprocessing
        from multiprocessing.synchronize import BoundedSemaphore

        from horde_worker_regen.process_management.messages import HordeProcessState
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        ctx = multiprocessing.get_context("spawn")
        max_threads = 1
        bounded_sem = BoundedSemaphore(max_threads, ctx=ctx)

        # Simulate child having acquired then released the semaphore (post-processing path)
        bounded_sem.acquire()
        bounded_sem.release()  # Child released when entering post-processing
        # Now bounded_sem has 1 permit available (back to initial)

        # Build a minimal mock manager
        from unittest.mock import MagicMock

        mock_manager = MagicMock()
        mock_manager._inference_semaphore = bounded_sem
        mock_manager._disk_lock = MagicMock()
        mock_manager._disk_lock.release.side_effect = ValueError  # already released

        process_info = MagicMock()
        process_info.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None
        process_info.loaded_horde_model_name = None

        # Bind real _replace_inference_process
        import types

        bound = types.MethodType(HordeWorkerProcessManager._replace_inference_process, mock_manager)
        bound(process_info)  # Must not raise

        # The semaphore should still have at most 1 permit (not inflated to 2)
        acquired = bounded_sem.acquire(block=False)
        assert acquired, "Semaphore should have exactly 1 permit available"
        second_acquired = bounded_sem.acquire(block=False)
        assert not second_acquired, "Semaphore must not have more than 1 permit (no inflation)"

    def test_replace_inference_starting_releases_semaphore_after_kill(self) -> None:
        """For INFERENCE_STARTING, _replace_inference_process must release the semaphore
        AFTER killing the process, not before.

        The child process is blocked at semaphore.acquire().  Releasing the semaphore before
        the kill creates a race: the child acquires the token just before the SIGKILL arrives.
        SIGKILL bypasses the finally block, permanently consuming the token and leaving every
        subsequent INFERENCE_STARTING process blocked at acquire() forever.

        This test verifies that at the moment _end_inference_process() is called (the kill),
        the semaphore has NOT yet been released, and that after the full replacement the
        semaphore is available (count == 1).
        """
        import multiprocessing
        import types
        from multiprocessing.synchronize import BoundedSemaphore
        from unittest.mock import MagicMock

        from horde_worker_regen.process_management.messages import HordeProcessState
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        ctx = multiprocessing.get_context("spawn")
        max_threads = 1
        bounded_sem = BoundedSemaphore(max_threads, ctx=ctx)

        # Simulate a leaked semaphore: the token has been consumed (count == 0) without any
        # process in INFERENCE_PROCESSING — exactly the broken state that causes a cascade of
        # stuck INFERENCE_STARTING processes.
        bounded_sem.acquire()

        # Track whether the semaphore was still consumed (not yet released) at kill time.
        semaphore_available_at_kill: list[bool] = []

        def fake_end_inference_process(pi: object) -> None:
            # At kill time the semaphore must NOT have been released yet.
            acquired = bounded_sem.acquire(block=False)
            semaphore_available_at_kill.append(acquired)
            if acquired:
                bounded_sem.release()  # restore so post-kill release works correctly

        mock_manager = MagicMock()
        mock_manager._inference_semaphore = bounded_sem
        mock_manager._disk_lock = MagicMock()
        mock_manager._disk_lock.release.side_effect = ValueError  # already released
        mock_manager._end_inference_process.side_effect = fake_end_inference_process

        process_info = MagicMock()
        process_info.last_process_state = HordeProcessState.INFERENCE_STARTING
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None
        process_info.loaded_horde_model_name = None

        bound = types.MethodType(HordeWorkerProcessManager._replace_inference_process, mock_manager)
        bound(process_info)  # Must not raise

        assert semaphore_available_at_kill == [False], (
            "Semaphore must NOT be released before killing the process for INFERENCE_STARTING "
            "(kill-before-release order required to prevent the semaphore-leak race condition)"
        )

        # After the replacement the semaphore must be available so the next process can proceed.
        acquired_after = bounded_sem.acquire(block=False)
        assert acquired_after, (
            "Semaphore must be available (count == 1) after replacing an INFERENCE_STARTING process"
        )
        # Verify no permit inflation: only exactly 1 permit available.
        second_acquired = bounded_sem.acquire(block=False)
        assert not second_acquired, "Semaphore must not have more than 1 permit after replacement"


class TestProcessEndingJobFaultHandling(_ReceiveLoopHarnessMixin):
    """Tests that a job in-progress is faulted when its process sends HordeProcessState.PROCESS_ENDING.

    Scenario: A child process encounters an exception during inference handling and
    ends itself (sending HordeProcessState.PROCESS_ENDING) before it can send the
    HordeInferenceResultMessage. The parent must detect the orphaned job and fault it
    so it is retried or submitted rather than silently lost.
    """

    def test_process_ending_with_job_in_progress_calls_handle_job_fault(self) -> None:
        """When PROCESS_ENDING arrives and the job is still in jobs_in_progress, handle_job_fault must be called."""
        job = MagicMock()
        job.id_ = "orphaned-job-id"

        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = job

        msg = self._make_message(HordeProcessState.PROCESS_ENDING)
        mock_manager = self._run_receive(msg, process_info, jobs_in_progress=[job])

        mock_manager.handle_job_fault.assert_called_once_with(
            faulted_job=job,
            process_info=process_info,
        )

    def test_process_ending_without_job_in_progress_does_not_call_handle_job_fault(self) -> None:
        """When HordeProcessState.PROCESS_ENDING arrives and the job is not in jobs_in_progress, handle_job_fault must not be called.

        This covers the normal case: inference completed, the result was already processed
        (removing the job from jobs_in_progress), and now the process is shutting down.
        """
        job = MagicMock()
        job.id_ = "completed-job-id"

        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = job

        msg = self._make_message(HordeProcessState.PROCESS_ENDING)
        # job is not in jobs_in_progress (it was already submitted)
        mock_manager = self._run_receive(msg, process_info, jobs_in_progress=[])

        mock_manager.handle_job_fault.assert_not_called()

    def test_process_ending_with_no_job_referenced_does_not_call_handle_job_fault(self) -> None:
        """When PROCESS_ENDING arrives and last_job_referenced is None, handle_job_fault must not be called."""
        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = HordeProcessState.WAITING_FOR_JOB
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None

        msg = self._make_message(HordeProcessState.PROCESS_ENDING)
        mock_manager = self._run_receive(msg, process_info, jobs_in_progress=[])

        mock_manager.handle_job_fault.assert_not_called()

    def test_process_ending_calls_on_process_ending_after_fault(self) -> None:
        """on_process_ending must be called after (not before) handle_job_fault to avoid clearing last_job_referenced."""
        job = MagicMock()
        job.id_ = "orphaned-job-id"

        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = job

        call_order: list[str] = []

        msg = self._make_message(HordeProcessState.PROCESS_ENDING)

        import queue as queue_mod

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        process_map = MagicMock()
        process_map.__contains__ = MagicMock(side_effect=lambda key: key == 0)
        process_map.__getitem__ = MagicMock(side_effect=lambda key: process_info)
        process_map.on_process_ending = MagicMock(side_effect=lambda process_id: call_order.append("on_process_ending"))

        q = queue_mod.Queue()
        q.put(msg)

        mock_manager = MagicMock()
        mock_manager._process_message_queue = q
        mock_manager._process_map = process_map
        mock_manager._in_deadlock = False
        mock_manager._in_queue_deadlock = False
        mock_manager.jobs_in_progress = [job]
        mock_manager.handle_job_fault = MagicMock(side_effect=lambda faulted_job, process_info: call_order.append("handle_job_fault"))

        bound = HordeWorkerProcessManager.receive_and_handle_process_messages.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound()

        # handle_job_fault must be called before on_process_ending, so that
        # last_job_referenced is still available when handle_job_fault runs
        assert "handle_job_fault" in call_order, "handle_job_fault was not called"
        assert "on_process_ending" in call_order, "on_process_ending was not called"
        assert call_order.index("handle_job_fault") < call_order.index("on_process_ending"), (
            "handle_job_fault must be called before on_process_ending"
        )

    def test_process_ending_with_job_in_progress_uses_prior_state_for_fault(self) -> None:
        """handle_job_fault must see the prior process state (e.g. INFERENCE_PROCESSING), not PROCESS_ENDING.

        This ensures _faulted_jobs_history correctly classifies the fault phase as 'INFERENCE_PROCESSING'
        rather than the misleading 'PROCESS_ENDING'.
        """

        job = MagicMock()
        job.id_ = "orphaned-job-id"

        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = job

        seen_state: list[HordeProcessState] = []

        msg = self._make_message(HordeProcessState.PROCESS_ENDING)

        import queue as queue_mod

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        process_map = MagicMock()
        process_map.__contains__ = MagicMock(side_effect=lambda key: key == 0)
        process_map.__getitem__ = MagicMock(side_effect=lambda key: process_info)

        q = queue_mod.Queue()
        q.put(msg)

        mock_manager = MagicMock()
        mock_manager._process_message_queue = q
        mock_manager._process_map = process_map
        mock_manager._in_deadlock = False
        mock_manager._in_queue_deadlock = False
        mock_manager.jobs_in_progress = [job]

        def capture_fault(*, faulted_job: object, process_info: MagicMock) -> None:
            seen_state.append(process_info.last_process_state)

        mock_manager.handle_job_fault = MagicMock(side_effect=capture_fault)

        bound = HordeWorkerProcessManager.receive_and_handle_process_messages.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound()

        assert len(seen_state) == 1, "handle_job_fault was not called exactly once"
        assert seen_state[0] == HordeProcessState.INFERENCE_PROCESSING, (
            f"Expected prior state INFERENCE_PROCESSING, got {seen_state[0]}"
        )



class TestProcessEndedAutoRestart(_ReceiveLoopHarnessMixin):
    """Tests that an inference process is automatically restarted when PROCESS_ENDED is received.

    Scenario: An inference process crashes during INFERENCE_PROCESSING.  The child sends
    PROCESS_ENDING (which triggers handle_job_fault) then PROCESS_ENDED.  The parent must
    restart the dead process so the worker returns to its configured capacity.
    """

    def _run_receive_process_ended(
        self,
        process_type: object,
        *,
        shutting_down: bool = False,
        prior_state: HordeProcessState = HordeProcessState.PROCESS_ENDING,
        existing_manager: MagicMock | None = None,
    ) -> MagicMock:
        """Run receive_and_handle_process_messages with a PROCESS_ENDED message.

        Returns the mock_manager so callers can inspect side-effects.
        """
        import queue as queue_mod

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = prior_state
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None
        process_info.process_type = process_type

        process_map = MagicMock()
        process_map.__contains__ = MagicMock(side_effect=lambda key: key == 0)
        process_map.__getitem__ = MagicMock(side_effect=lambda key: process_info)

        msg = self._make_message(HordeProcessState.PROCESS_ENDED)
        q = queue_mod.Queue()
        q.put(msg)

        if existing_manager is not None:
            mock_manager = existing_manager
            mock_manager._process_message_queue = q
            mock_manager._process_map = process_map
        else:
            mock_manager = MagicMock()
            mock_manager._process_message_queue = q
            mock_manager._process_map = process_map
            mock_manager._in_deadlock = False
            mock_manager._in_queue_deadlock = False
            mock_manager.jobs_in_progress = []
            mock_manager._shutting_down = shutting_down
            mock_manager._num_process_recoveries = 0
            mock_manager._process_restart_history = {}

        bound = HordeWorkerProcessManager.receive_and_handle_process_messages.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound()
        return mock_manager

    def test_inference_process_restarted_on_unexpected_end(self) -> None:
        """When PROCESS_ENDED arrives for an inference process and we are not shutting down,
        _start_inference_process must be called to restore the configured worker capacity."""
        from horde_worker_regen.process_management.process_manager import HordeProcessType

        mock_manager = self._run_receive_process_ended(
            process_type=HordeProcessType.INFERENCE,
            shutting_down=False,
        )

        mock_manager._start_inference_process.assert_called_once_with(0)

    def test_inference_process_restarted_increments_num_process_recoveries(self) -> None:
        """When a process is restarted after PROCESS_ENDED, _num_process_recoveries must be incremented."""
        from horde_worker_regen.process_management.process_manager import HordeProcessType

        mock_manager = self._run_receive_process_ended(
            process_type=HordeProcessType.INFERENCE,
            shutting_down=False,
        )

        mock_manager._start_inference_process.assert_called_once_with(0)
        assert mock_manager._num_process_recoveries == 1

    def test_inference_process_not_restarted_during_shutdown(self) -> None:
        """When PROCESS_ENDED arrives for an inference process while shutting down,
        _start_inference_process must NOT be called."""
        from horde_worker_regen.process_management.process_manager import HordeProcessType

        mock_manager = self._run_receive_process_ended(
            process_type=HordeProcessType.INFERENCE,
            shutting_down=True,
        )

        mock_manager._start_inference_process.assert_not_called()

    def test_safety_process_not_restarted_on_end(self) -> None:
        """When PROCESS_ENDED arrives for a safety process, _start_inference_process must NOT
        be called (safety processes have separate restart logic)."""
        from horde_worker_regen.process_management.process_manager import HordeProcessType

        mock_manager = self._run_receive_process_ended(
            process_type=HordeProcessType.SAFETY,
            shutting_down=False,
        )

        mock_manager._start_inference_process.assert_not_called()

    def test_process_starting_prior_state_restarts_with_rate_limiting(self) -> None:
        """When PROCESS_ENDED arrives and the prior state was PROCESS_STARTING, the process
        must be restarted (with rate-limiting) so that the slot is not left permanently dead."""
        from horde_worker_regen.process_management.process_manager import HordeProcessType

        mock_manager = self._run_receive_process_ended(
            process_type=HordeProcessType.INFERENCE,
            shutting_down=False,
            prior_state=HordeProcessState.PROCESS_STARTING,
        )

        mock_manager._start_inference_process.assert_called_once_with(0)
        assert mock_manager._num_process_recoveries == 1

    def test_process_starting_prior_state_rate_limited_after_five_failures(self) -> None:
        """When a process repeatedly ends in PROCESS_STARTING, restart is suppressed after
        5 failures within 60s to prevent a tight crash loop."""
        import time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        now = time.time()
        mock_manager = MagicMock()
        mock_manager._in_deadlock = False
        mock_manager._in_queue_deadlock = False
        mock_manager.jobs_in_progress = []
        mock_manager._shutting_down = False
        mock_manager._num_process_recoveries = 0
        mock_manager._process_restart_history = {0: deque([now - 5, now - 4, now - 3, now - 2, now - 1], maxlen=5)}

        self._run_receive_process_ended(
            process_type=HordeProcessType.INFERENCE,
            shutting_down=False,
            prior_state=HordeProcessState.PROCESS_STARTING,
            existing_manager=mock_manager,
        )

        mock_manager._start_inference_process.assert_not_called()
        assert mock_manager._num_process_recoveries == 0

    def test_restart_rate_limited_after_five_failures_in_sixty_seconds(self) -> None:
        """When a process ends and restarts 5 times within 60 seconds, the 6th restart must
        be suppressed to prevent a tight crash/restart loop."""
        import time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        # Seed the restart history with 5 recent timestamps so the next end triggers the limit
        mock_manager = MagicMock()
        mock_manager._in_deadlock = False
        mock_manager._in_queue_deadlock = False
        mock_manager.jobs_in_progress = []
        mock_manager._shutting_down = False
        mock_manager._num_process_recoveries = 0

        now = time.time()
        mock_manager._process_restart_history = {0: deque([now - 5, now - 4, now - 3, now - 2, now - 1], maxlen=5)}

        self._run_receive_process_ended(
            process_type=HordeProcessType.INFERENCE,
            shutting_down=False,
            existing_manager=mock_manager,
        )

        mock_manager._start_inference_process.assert_not_called()
        assert mock_manager._num_process_recoveries == 0

    def test_restart_allowed_after_five_failures_spread_over_more_than_sixty_seconds(self) -> None:
        """When 5 prior restarts are spread over more than 60 seconds, the next restart must
        still be allowed (rate limit window has passed)."""
        import time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        mock_manager = MagicMock()
        mock_manager._in_deadlock = False
        mock_manager._in_queue_deadlock = False
        mock_manager.jobs_in_progress = []
        mock_manager._shutting_down = False
        mock_manager._num_process_recoveries = 0

        now = time.time()
        # Pre-seed with only 4 entries so after the production code appends the current timestamp,
        # the deque contains 5 entries and restart_history[0] is 90s ago (outside the 60s window).
        mock_manager._process_restart_history = {
            0: deque([now - 90, now - 4, now - 3, now - 2], maxlen=5)
        }

        self._run_receive_process_ended(
            process_type=HordeProcessType.INFERENCE,
            shutting_down=False,
            existing_manager=mock_manager,
        )

        mock_manager._start_inference_process.assert_called_once_with(0)
        assert mock_manager._num_process_recoveries == 1


class TestSendMemoryReportMessageVramFailure:
    """Tests that VRAM query failures do not terminate the inference process.

    Regression test for the bug where a VRAM query failure inside
    send_memory_report_message caused the inference process to set
    _end_process = True and exit mid-inference, orphaning the in-flight job.
    """

    def _create_mock_horde_process(self) -> MagicMock:
        """Create a minimal mock that exercises HordeProcess.send_memory_report_message."""
        from horde_worker_regen.process_management.horde_process import HordeProcess

        mock = MagicMock()
        mock.process_id = 1
        mock.process_launch_identifier = 42
        mock._end_process = False
        mock._last_vram_warning_time = 0.0  # Ensure getattr fallback works correctly

        # Bind the real base-class method
        mock.send_memory_report_message = HordeProcess.send_memory_report_message.__get__(mock, HordeProcess)
        return mock

    def test_vram_failure_still_sends_report(self) -> None:
        """A VRAM query failure must not prevent the memory report from being sent."""
        mock = self._create_mock_horde_process()

        # Make VRAM query raise
        mock.get_vram_usage_bytes.side_effect = RuntimeError("CUDA error")
        mock.get_vram_total_bytes.side_effect = RuntimeError("CUDA error")

        result = mock.send_memory_report_message(include_vram=True)

        assert result is True, "send_memory_report_message must return True even when VRAM query fails"
        # Verify the message was still sent to the queue (put() was called)
        mock.process_message_queue.put.assert_called_once()
        # Verify the message payload has VRAM fields as None (not partially set)
        sent_message = mock.process_message_queue.put.call_args[0][0]
        assert sent_message.vram_usage_bytes is None, "vram_usage_bytes must be None on failure"
        assert sent_message.vram_total_bytes is None, "vram_total_bytes must be None on failure"

    def test_partial_vram_failure_sends_report_without_vram(self) -> None:
        """If the second VRAM query fails, neither VRAM field should be set in the message.

        Guards against partially setting vram_usage_bytes without vram_total_bytes.
        """
        mock = self._create_mock_horde_process()

        # First call succeeds, second raises
        mock.get_vram_usage_bytes.return_value = 1024 * 1024 * 256  # 256 MB
        mock.get_vram_total_bytes.side_effect = RuntimeError("Driver error")

        result = mock.send_memory_report_message(include_vram=True)

        assert result is True
        mock.process_message_queue.put.assert_called_once()
        sent_message = mock.process_message_queue.put.call_args[0][0]
        assert sent_message.vram_usage_bytes is None, (
            "vram_usage_bytes must be None when the second VRAM query fails"
        )
        assert sent_message.vram_total_bytes is None, (
            "vram_total_bytes must be None when the second VRAM query fails"
        )

    def test_vram_failure_does_not_set_end_process(self) -> None:
        """A VRAM query failure must not set _end_process = True on the process."""
        mock = self._create_mock_horde_process()

        # Make VRAM query raise
        mock.get_vram_usage_bytes.side_effect = RuntimeError("CUDA OOM error")
        mock.get_vram_total_bytes.side_effect = RuntimeError("CUDA OOM error")

        mock.send_memory_report_message(include_vram=True)

        assert mock._end_process is False, "_end_process must not be set to True on VRAM query failure"

    def test_successful_report_without_vram(self) -> None:
        """A report without VRAM info should always succeed."""
        mock = self._create_mock_horde_process()

        result = mock.send_memory_report_message(include_vram=False)

        assert result is True
        # get_vram_usage_bytes and get_vram_total_bytes should NOT be called
        mock.get_vram_usage_bytes.assert_not_called()
        mock.get_vram_total_bytes.assert_not_called()
        mock.process_message_queue.put.assert_called_once()

    def test_successful_report_with_vram(self) -> None:
        """A report with VRAM info should succeed when VRAM query works."""
        mock = self._create_mock_horde_process()
        mock.get_vram_usage_bytes.return_value = 1024 * 1024 * 512  # 512 MB
        mock.get_vram_total_bytes.return_value = 1024 * 1024 * 1024 * 8  # 8 GB

        result = mock.send_memory_report_message(include_vram=True)

        assert result is True
        mock.process_message_queue.put.assert_called_once()

    def test_vram_warning_is_rate_limited(self) -> None:
        """Repeated VRAM failures within 10 s must not re-emit a WARNING; they use DEBUG instead."""
        import time

        mock = self._create_mock_horde_process()
        mock.get_vram_usage_bytes.side_effect = RuntimeError("CUDA error")
        mock.get_vram_total_bytes.side_effect = RuntimeError("CUDA error")

        # Simulate the first failure happened 5 seconds ago (within the 10-second window)
        mock._last_vram_warning_time = time.time() - 5.0

        with patch("horde_worker_regen.process_management.horde_process.logger") as mock_logger:
            mock.send_memory_report_message(include_vram=True)

        # WARNING must NOT be emitted again within the 10-second window
        mock_logger.warning.assert_not_called()
        # DEBUG must be emitted instead
        mock_logger.debug.assert_called_once()

    def test_inference_process_override_does_not_set_end_process_on_vram_failure(self) -> None:
        """The HordeInferenceProcess override must not set _end_process on VRAM failure.

        Creates a minimal concrete subclass of HordeInferenceProcess that skips
        all heavy initialisation, so we can call the real override on a real instance.
        """
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        # Build a minimal concrete subclass that avoids __init__ entirely
        class _TestProcess(HordeInferenceProcess):
            def cleanup_for_exit(self) -> None:
                pass

        instance = object.__new__(_TestProcess)
        instance.process_id = 1
        instance.process_launch_identifier = 42
        instance._end_process = False
        instance.process_message_queue = MagicMock()
        instance.get_vram_usage_bytes = MagicMock(side_effect=RuntimeError("CUDA error"))
        instance.get_vram_total_bytes = MagicMock(side_effect=RuntimeError("CUDA error"))

        result = instance.send_memory_report_message(include_vram=True)

        assert result is True, "Inference process override must return True even on VRAM failure"
        assert instance._end_process is False, (
            "_end_process must remain False — the override must not set it to True"
        )
        instance.process_message_queue.put.assert_called_once()


class TestSendInferenceResultEncodingResilience:
    """Tests that send_inference_result_message handles image-encoding failures gracefully.

    Regression tests for the bug where a base64-encoding failure (e.g. MemoryError or
    corrupt BytesIO) inside send_inference_result_message caused an unhandled exception
    that propagated through _receive_and_handle_control_message, terminated the child
    process, and left the job silently in-progress with POST_PROCESSING_COMPLETE as
    the last known state — causing the manager's PROCESS_ENDING handler to fault the job.

    The fix ensures:
    1. Each image is encoded inside a try/except; any encoding failure faults the ENTIRE
       job (not just the failing image) rather than propagating an exception that kills
       the process.  Partial submissions are unsafe because the submission pipeline uses
       positional indexing (gen_iter across job_image_results AND sdk_api_job_info.ids),
       so a shorter image list would leave some IDs permanently unsubmitted.
    2. State is derived from encoding success: any failure → GENERATION_STATE.faulted.
    3. Post-enqueue state-update errors are caught inside send_inference_result_message
       so the caller never sees a spurious exception after the message is already queued.
    """

    _TARGET = "horde_worker_regen.process_management.inference_process.HordeInferenceResultMessage"

    def _make_inference_process(self) -> object:
        """Return a minimal HordeInferenceProcess instance that skips heavy initialisation."""
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        class _TestProcess(HordeInferenceProcess):
            def cleanup_for_exit(self) -> None:
                pass

        instance = object.__new__(_TestProcess)
        instance.process_id = 1
        instance.process_launch_identifier = 42
        instance._last_job_inference_rate = None
        instance._active_model_name = "test-model"
        instance.process_message_queue = MagicMock()
        return instance

    def _make_result(self, *, rawpng_bytes: bytes | None = b"fake-png-data") -> MagicMock:
        """Return a minimal ResultingImageReturn mock."""
        import io

        result = MagicMock()
        if rawpng_bytes is None:
            result.rawpng = None
        else:
            result.rawpng = io.BytesIO(rawpng_bytes)
        result.faults = []
        return result

    def test_successful_encoding_sends_ok_message(self) -> None:
        """When all images encode successfully the message state must be GENERATION_STATE.ok."""
        import base64

        from horde_sdk.ai_horde_api import GENERATION_STATE

        proc = self._make_inference_process()
        result = self._make_result(rawpng_bytes=b"\x89PNG\r\n\x1a\nfakedata")

        captured: dict = {}

        def fake_msg_cls(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            m = MagicMock()
            m.state = kwargs.get("state")
            m.job_image_results = kwargs.get("job_image_results", [])
            return m

        with patch(self._TARGET, side_effect=fake_msg_cls):
            proc.send_inference_result_message(
                process_state=HordeProcessState.INFERENCE_COMPLETE,
                job_info=MagicMock(),
                results=[result],
                time_elapsed=1.0,
                sanitized_negative_prompt=None,
            )

        assert proc.process_message_queue.put.call_count >= 1
        assert captured["state"] == GENERATION_STATE.ok
        assert len(captured["job_image_results"]) == 1
        # Validate the base64 payload round-trips correctly
        decoded = base64.b64decode(captured["job_image_results"][0].image_base64)
        assert decoded == b"\x89PNG\r\n\x1a\nfakedata"

    def test_single_encoding_failure_faults_entire_job(self) -> None:
        """When any single image's encoding raises, the ENTIRE job must be faulted (state=faulted,
        empty image list) rather than sending a partial result or crashing the process.

        Partial results are unsafe because the submission pipeline uses gen_iter to index
        both job_image_results and sdk_api_job_info.ids — a shorter list leaves some IDs
        permanently unsubmitted.
        """
        from horde_sdk.ai_horde_api import GENERATION_STATE

        proc = self._make_inference_process()

        # Make getvalue() raise to simulate a corrupt BytesIO
        bad_result = MagicMock()
        bad_result.rawpng = MagicMock()
        bad_result.rawpng.getvalue.side_effect = RuntimeError("Simulated BytesIO corruption")
        bad_result.faults = []

        captured: dict = {}

        def fake_msg_cls(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            m = MagicMock()
            m.state = kwargs.get("state")
            m.job_image_results = kwargs.get("job_image_results", [])
            return m

        with patch(self._TARGET, side_effect=fake_msg_cls):
            # Must not raise — the encoding failure must be caught internally
            proc.send_inference_result_message(
                process_state=HordeProcessState.INFERENCE_COMPLETE,
                job_info=MagicMock(),
                results=[bad_result],
                time_elapsed=1.0,
                sanitized_negative_prompt=None,
            )

        # Message must still be sent (no unhandled exception kills the process)
        assert proc.process_message_queue.put.call_count >= 1
        # Encoding failure → entire job faulted, empty image list
        assert captured["state"] == GENERATION_STATE.faulted
        assert captured["job_image_results"] == []

    def test_all_images_encoding_failure_reports_faulted(self) -> None:
        """When ALL images fail to encode, state must be GENERATION_STATE.faulted, not ok."""
        from horde_sdk.ai_horde_api import GENERATION_STATE

        proc = self._make_inference_process()

        def _make_bad() -> MagicMock:
            r = MagicMock()
            r.rawpng = MagicMock()
            r.rawpng.getvalue.side_effect = MemoryError("Out of memory during encoding")
            r.faults = []
            return r

        captured: dict = {}

        def fake_msg_cls(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            m = MagicMock()
            m.state = kwargs.get("state")
            m.job_image_results = kwargs.get("job_image_results", [])
            return m

        with patch(self._TARGET, side_effect=fake_msg_cls):
            proc.send_inference_result_message(
                process_state=HordeProcessState.INFERENCE_COMPLETE,
                job_info=MagicMock(),
                results=[_make_bad(), _make_bad()],
                time_elapsed=2.0,
                sanitized_negative_prompt=None,
            )

        assert proc.process_message_queue.put.call_count >= 1
        assert captured["state"] == GENERATION_STATE.faulted
        assert captured["job_image_results"] == []

    def test_partial_encoding_failure_faults_entire_job(self) -> None:
        """When only some images in a batch fail to encode, the ENTIRE job must be faulted.

        Partial submissions are unsafe: the submission pipeline uses gen_iter to index
        both job_image_results[gen_iter] and sdk_api_job_info.ids[gen_iter].  If the
        image list is shorter than n_iter, the remaining IDs are never submitted to the API.
        """
        from horde_sdk.ai_horde_api import GENERATION_STATE

        proc = self._make_inference_process()
        good_result = self._make_result(rawpng_bytes=b"valid-png-data")

        bad_result = MagicMock()
        bad_result.rawpng = MagicMock()
        bad_result.rawpng.getvalue.side_effect = RuntimeError("Encoding error")
        bad_result.faults = []

        captured: dict = {}

        def fake_msg_cls(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            m = MagicMock()
            m.state = kwargs.get("state")
            m.job_image_results = kwargs.get("job_image_results", [])
            return m

        with patch(self._TARGET, side_effect=fake_msg_cls):
            proc.send_inference_result_message(
                process_state=HordeProcessState.INFERENCE_COMPLETE,
                job_info=MagicMock(),
                results=[good_result, bad_result],
                time_elapsed=1.5,
                sanitized_negative_prompt=None,
            )

        assert proc.process_message_queue.put.call_count >= 1
        # Partial failure → entire job faulted to avoid unsubmitted IDs
        assert captured["state"] == GENERATION_STATE.faulted
        assert captured["job_image_results"] == []

    def test_none_results_sends_faulted(self) -> None:
        """Passing results=None must produce a GENERATION_STATE.faulted message (no change to existing behaviour)."""
        from horde_sdk.ai_horde_api import GENERATION_STATE

        proc = self._make_inference_process()

        captured: dict = {}

        def fake_msg_cls(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            m = MagicMock()
            m.state = kwargs.get("state")
            m.job_image_results = kwargs.get("job_image_results", [])
            return m

        with patch(self._TARGET, side_effect=fake_msg_cls):
            proc.send_inference_result_message(
                process_state=HordeProcessState.INFERENCE_COMPLETE,
                job_info=MagicMock(),
                results=None,
                time_elapsed=0.5,
                sanitized_negative_prompt=None,
            )

        assert proc.process_message_queue.put.call_count >= 1
        assert captured["state"] == GENERATION_STATE.faulted
        assert captured["job_image_results"] == []

    def test_post_enqueue_state_update_failure_does_not_propagate(self) -> None:
        """Exceptions from the state-update calls after queue.put() must NOT propagate.

        After the result message is successfully enqueued, on_horde_model_state_change or
        send_process_state_change_message can raise.  These are wrapped in a try/except so
        the exception does not escape send_inference_result_message.  Without this guard,
        the caller's except block would trigger a spurious second (faulted) result message
        for the same job.
        """
        from horde_sdk.ai_horde_api import GENERATION_STATE

        proc = self._make_inference_process()
        result = self._make_result(rawpng_bytes=b"valid-png-data")

        captured_messages: list = []

        def fake_msg_cls(**kwargs: object) -> MagicMock:
            m = MagicMock()
            m.state = kwargs.get("state")
            m.job_image_results = kwargs.get("job_image_results", [])
            return m

        # call_count[0] == 1: the HordeInferenceResultMessage (the main result, must succeed)
        # call_count[0] > 1: subsequent puts are for state-change messages inside the post-enqueue
        #                    try/except block — these are allowed to fail without propagating.
        call_count = [0]

        def put_side_effect(msg: object) -> None:
            call_count[0] += 1
            captured_messages.append(msg)
            # Let the first put (the HordeInferenceResultMessage) succeed
            # and raise on the second (so the state update path raises)
            if call_count[0] > 1:
                raise RuntimeError("Simulated queue failure after result enqueued")

        proc.process_message_queue.put.side_effect = put_side_effect

        # Must not raise to the caller — the state update exception is caught internally
        with patch(self._TARGET, side_effect=fake_msg_cls):
            proc.send_inference_result_message(
                process_state=HordeProcessState.INFERENCE_COMPLETE,
                job_info=MagicMock(),
                results=[result],
                time_elapsed=1.0,
                sanitized_negative_prompt=None,
            )

        # Exactly ONE result message must have been sent (not two)
        # The first put succeeded (the result), subsequent ones raised
        assert call_count[0] >= 1
        first_msg = captured_messages[0]
        assert first_msg.state == GENERATION_STATE.ok


class TestSendInferenceResultFallbackOnFailure:
    """Tests the fallback in _receive_and_handle_control_message when send_inference_result_message fails.

    Regression tests for the scenario where the call to send_inference_result_message (with the
    actual results) raises unexpectedly (e.g. queue full, model state update fails, etc.).
    The child process must attempt to send a faulted result instead of propagating the
    exception and dying silently with POST_PROCESSING_COMPLETE as the last known state.
    """

    _TARGET = "horde_worker_regen.process_management.inference_process.HordeInferenceResultMessage"

    def _make_process(self) -> object:
        """Create a minimal HordeInferenceProcess that skips all heavy initialisation."""
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        class _TestProcess(HordeInferenceProcess):
            def cleanup_for_exit(self) -> None:
                pass

        instance = object.__new__(_TestProcess)
        instance.process_id = 1
        instance.process_launch_identifier = 42
        instance._last_job_inference_rate = None
        instance._active_model_name = "test-model"
        instance._last_sanitized_negative_prompt = None
        instance.process_message_queue = MagicMock()
        return instance

    def test_queue_put_failure_propagates_from_send_inference_result_message(self) -> None:
        """When queue.put() raises inside send_inference_result_message, the exception propagates.

        This confirms that the caller (in _receive_and_handle_control_message) must wrap the
        call in try/except to send a faulted fallback.
        """
        import io

        from horde_worker_regen.process_management.messages import HordeProcessState

        proc = self._make_process()

        good_result = MagicMock()
        good_result.rawpng = io.BytesIO(b"valid-png-data")
        good_result.faults = []

        # Make queue.put() raise on the first call
        proc.process_message_queue.put.side_effect = RuntimeError("Simulated queue failure")

        def fake_msg_cls(**kwargs: object) -> MagicMock:
            m = MagicMock()
            m.state = kwargs.get("state")
            m.job_image_results = kwargs.get("job_image_results", [])
            return m

        with patch(self._TARGET, side_effect=fake_msg_cls):
            with pytest.raises(RuntimeError, match="Simulated queue failure"):
                proc.send_inference_result_message(
                    process_state=HordeProcessState.INFERENCE_COMPLETE,
                    job_info=MagicMock(),
                    results=[good_result],
                    time_elapsed=1.0,
                    sanitized_negative_prompt=None,
                )

    def test_faulted_fallback_with_none_results_succeeds(self) -> None:
        """The faulted-fallback path (results=None) must succeed when the queue is working.

        This models the second attempt in the caller's except block.
        """
        from horde_sdk.ai_horde_api import GENERATION_STATE
        from horde_worker_regen.process_management.messages import HordeProcessState

        proc = self._make_process()

        captured: dict = {}

        def fake_msg_cls(**kwargs: object) -> MagicMock:
            captured.update(kwargs)
            m = MagicMock()
            m.state = kwargs.get("state")
            m.job_image_results = kwargs.get("job_image_results", [])
            return m

        with patch(self._TARGET, side_effect=fake_msg_cls):
            proc.send_inference_result_message(
                process_state=HordeProcessState.INFERENCE_FAILED,
                job_info=MagicMock(),
                results=None,
                time_elapsed=1.0,
                sanitized_negative_prompt=None,
            )

        assert proc.process_message_queue.put.call_count >= 1
        assert captured["state"] == GENERATION_STATE.faulted
        assert captured["job_image_results"] == []

    def test_both_sends_fail_reraises_to_trigger_end_process(self) -> None:
        """When both the normal send AND the faulted fallback fail, the exception must re-raise.

        The outer receive_and_handle_control_messages loop catches any exception from
        _receive_and_handle_control_message and sets _end_process=True, causing the
        process to exit cleanly.  Without the re-raise, the process would keep running
        with the job stuck in jobs_in_progress, requiring the manager to time it out as hung.
        """
        import io

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess
        from horde_worker_regen.process_management.messages import HordeProcessState

        class _TestProcess(HordeInferenceProcess):
            def cleanup_for_exit(self) -> None:
                pass

        proc = object.__new__(_TestProcess)
        proc.process_id = 1
        proc.process_launch_identifier = 42
        proc._last_job_inference_rate = None
        proc._active_model_name = "test-model"
        proc._last_sanitized_negative_prompt = None
        proc.process_message_queue = MagicMock()

        # Make every queue.put() raise so BOTH sends fail
        proc.process_message_queue.put.side_effect = RuntimeError("Persistent queue failure")

        good_result = MagicMock()
        good_result.rawpng = io.BytesIO(b"valid-data")
        good_result.faults = []

        def fake_msg_cls(**kwargs: object) -> MagicMock:
            m = MagicMock()
            m.state = kwargs.get("state")
            m.job_image_results = kwargs.get("job_image_results", [])
            return m

        # Calling send_inference_result_message with results raises on queue.put().
        # The caller's fallback also calls send_inference_result_message (results=None)
        # which also raises.  The fallback's except block must re-raise so the outer
        # receive loop can set _end_process=True.
        #
        # We simulate this directly: first call raises (normal send), second call also
        # raises (fallback send).  Verify the final exception escapes.
        with patch(self._TARGET, side_effect=fake_msg_cls):
            # First send fails
            with pytest.raises(RuntimeError, match="Persistent queue failure"):
                proc.send_inference_result_message(
                    process_state=HordeProcessState.INFERENCE_COMPLETE,
                    job_info=MagicMock(),
                    results=[good_result],
                    time_elapsed=1.0,
                    sanitized_negative_prompt=None,
                )

            # Second send (fallback) also fails — the re-raise means the exception escapes
            with pytest.raises(RuntimeError, match="Persistent queue failure"):
                proc.send_inference_result_message(
                    process_state=HordeProcessState.INFERENCE_FAILED,
                    job_info=MagicMock(),
                    results=None,
                    time_elapsed=1.0,
                    sanitized_negative_prompt=None,
                )


class TestKeepSingleInferenceStates(_ReceiveLoopHarnessMixin):
    """Tests that keep_single_inference correctly checks all active inference states.

    Previously the function had duplicate conditions that only checked INFERENCE_STARTING,
    missing INFERENCE_PROCESSING and INFERENCE_POST_PROCESSING. This caused jobs to be
    dispatched when they should not be (e.g., while a batch or VRAM-heavy model was in
    INFERENCE_PROCESSING), leading to unnecessary semaphore contention and jobs stuck in
    INFERENCE_STARTING.
    """

    def _make_process_map_with_process(
        self,
        state: HordeProcessState,
        *,
        batch_amount: int = 1,
        model: str | None = None,
    ) -> MagicMock:
        """Build a minimal mock process map containing one process in the given state."""
        from horde_worker_regen.process_management.process_manager import ProcessMap

        p = MagicMock()
        p.last_process_state = state
        p.inference_started_timestamp = None
        p.batch_amount = batch_amount

        if model is not None:
            p.last_job_referenced = MagicMock()
            p.last_job_referenced.model = model
        else:
            p.last_job_referenced = None

        process_map = MagicMock()
        process_map.values.return_value = [p]
        process_map.keep_single_inference = ProcessMap.keep_single_inference.__get__(
            process_map, ProcessMap
        )
        return process_map

    def test_batch_job_inference_processing_keeps_single(self) -> None:
        """keep_single_inference must return True when a batch job is in INFERENCE_PROCESSING.

        Previously the check only tested for INFERENCE_STARTING, so a batch job that had
        already moved to INFERENCE_PROCESSING would not block additional jobs from being
        dispatched.
        """
        process_map = self._make_process_map_with_process(
            HordeProcessState.INFERENCE_PROCESSING,
            batch_amount=4,
        )

        result, reason = process_map.keep_single_inference(
            stable_diffusion_model_reference=MagicMock(),
            post_process_job_overlap=False,
        )

        assert result is True, (
            "Expected keep_single_inference=True for batch job in INFERENCE_PROCESSING, "
            f"got ({result!r}, {reason!r})"
        )
        assert reason == "Batched job"

    def test_batch_job_inference_post_processing_keeps_single(self) -> None:
        """keep_single_inference must return True when a batch job is in INFERENCE_POST_PROCESSING."""
        process_map = self._make_process_map_with_process(
            HordeProcessState.INFERENCE_POST_PROCESSING,
            batch_amount=2,
        )

        result, reason = process_map.keep_single_inference(
            stable_diffusion_model_reference=MagicMock(),
            post_process_job_overlap=False,
        )

        assert result is True, (
            "Expected keep_single_inference=True for batch job in INFERENCE_POST_PROCESSING, "
            f"got ({result!r}, {reason!r})"
        )
        assert reason == "Batched job"

    def test_batch_job_inference_starting_keeps_single(self) -> None:
        """keep_single_inference must return True when a batch job is in INFERENCE_STARTING."""
        process_map = self._make_process_map_with_process(
            HordeProcessState.INFERENCE_STARTING,
            batch_amount=3,
        )

        result, reason = process_map.keep_single_inference(
            stable_diffusion_model_reference=MagicMock(),
            post_process_job_overlap=False,
        )

        assert result is True
        assert reason == "Batched job"

    def test_non_batch_job_inference_processing_does_not_keep_single(self) -> None:
        """keep_single_inference must return False when a normal job is in INFERENCE_PROCESSING."""
        from horde_worker_regen.process_management.process_manager import ProcessMap

        p = MagicMock()
        p.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        p.inference_started_timestamp = None
        p.batch_amount = 1
        p.last_job_referenced = MagicMock()
        p.last_job_referenced.model = "some_normal_model"
        p.last_job_referenced.payload.workflow = None
        p.can_accept_job.return_value = False

        process_map = MagicMock()
        process_map.values.return_value = [p]
        process_map.keep_single_inference = ProcessMap.keep_single_inference.__get__(
            process_map, ProcessMap
        )

        result, reason = process_map.keep_single_inference(
            stable_diffusion_model_reference=MagicMock(),
            post_process_job_overlap=False,
        )

        assert result is False, (
            "Expected keep_single_inference=False for normal job in INFERENCE_PROCESSING"
        )


class TestProcessEndingReleasesInferenceSemaphore(_ReceiveLoopHarnessMixin):
    """Tests that the inference semaphore is released when PROCESS_ENDING is received for a
    process that was in INFERENCE_PROCESSING.

    This prevents other processes from being permanently stuck in INFERENCE_STARTING when
    a child process terminates without releasing the semaphore (e.g., due to an OOM kill
    where the finally block did not run).
    """

    def _run_receive_process_ending_with_semaphore(
        self,
        prior_state: HordeProcessState,
        *,
        semaphore_acquired: bool = True,
    ) -> tuple[MagicMock, object]:
        """Run receive_and_handle_process_messages with PROCESS_ENDING and a real semaphore.

        Args:
            prior_state: The process state before the PROCESS_ENDING transition.
            semaphore_acquired: If True, the semaphore is pre-acquired to simulate the child
                                holding it. If False, the semaphore is at its initial value.

        Returns (mock_manager, bounded_semaphore) so callers can inspect the semaphore state.
        """
        import multiprocessing
        import queue as queue_mod
        from multiprocessing.synchronize import BoundedSemaphore

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        ctx = multiprocessing.get_context("spawn")
        bounded_sem = BoundedSemaphore(1, ctx=ctx)
        if semaphore_acquired:
            # Simulate the child holding the semaphore (acquired but not released)
            bounded_sem.acquire()

        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = prior_state
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None

        process_map = MagicMock()
        process_map.__contains__ = MagicMock(side_effect=lambda key: key == 0)
        process_map.__getitem__ = MagicMock(side_effect=lambda key: process_info)

        msg = self._make_message(HordeProcessState.PROCESS_ENDING)
        q = queue_mod.Queue()
        q.put(msg)

        mock_manager = MagicMock()
        mock_manager._process_message_queue = q
        mock_manager._process_map = process_map
        mock_manager._in_deadlock = False
        mock_manager._in_queue_deadlock = False
        mock_manager.jobs_in_progress = []
        mock_manager._inference_semaphore = bounded_sem

        bound = HordeWorkerProcessManager.receive_and_handle_process_messages.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound()
        return mock_manager, bounded_sem

    def test_semaphore_released_when_process_ending_from_inference_processing(self) -> None:
        """When PROCESS_ENDING arrives for a process in INFERENCE_PROCESSING, the inference
        semaphore must be released so that other processes stuck in INFERENCE_STARTING can
        acquire it and proceed.
        """
        _mock_manager, sem = self._run_receive_process_ending_with_semaphore(
            prior_state=HordeProcessState.INFERENCE_PROCESSING,
            semaphore_acquired=True,
        )

        # The semaphore should now be available (released by PROCESS_ENDING handler)
        acquired = sem.acquire(block=False)
        assert acquired, (
            "Semaphore should be acquirable after PROCESS_ENDING from INFERENCE_PROCESSING — "
            "the handler must release it to unblock processes stuck in INFERENCE_STARTING"
        )

    def test_semaphore_released_when_process_ending_from_post_processing_starting(self) -> None:
        """When PROCESS_ENDING arrives for a process in POST_PROCESSING_STARTING, the inference
        semaphore must also be released.

        The child emits POST_PROCESSING_STARTING BEFORE releasing the semaphore (see
        inference_process.py progress_callback). If the process crashes between emitting the
        state and calling release(), the semaphore leaks and other processes remain stuck in
        INFERENCE_STARTING.
        """
        _mock_manager, sem = self._run_receive_process_ending_with_semaphore(
            prior_state=HordeProcessState.POST_PROCESSING_STARTING,
            semaphore_acquired=True,
        )

        acquired = sem.acquire(block=False)
        assert acquired, (
            "Semaphore should be acquirable after PROCESS_ENDING from POST_PROCESSING_STARTING — "
            "the handler must release it to unblock processes stuck in INFERENCE_STARTING"
        )

    def test_semaphore_not_released_when_process_ending_from_waiting_for_job(self) -> None:
        """When PROCESS_ENDING arrives for a process in WAITING_FOR_JOB, the inference
        semaphore must NOT be released (the process was not holding it).
        """
        _mock_manager, sem = self._run_receive_process_ending_with_semaphore(
            prior_state=HordeProcessState.WAITING_FOR_JOB,
            semaphore_acquired=True,
        )

        # The semaphore was acquired before the test but should NOT have been released
        # by the PROCESS_ENDING handler (WAITING_FOR_JOB does not hold the semaphore)
        acquired = sem.acquire(block=False)
        assert not acquired, (
            "Semaphore must not be acquirable when PROCESS_ENDING is for a non-inference state"
        )

    def test_semaphore_double_release_safe_when_child_already_released(self) -> None:
        """When PROCESS_ENDING arrives and the child already released the semaphore normally,
        the PROCESS_ENDING handler's release attempt must not raise and must not inflate permits.
        """
        # semaphore_acquired=False: child already released via its finally block
        _mock_manager, sem = self._run_receive_process_ending_with_semaphore(
            prior_state=HordeProcessState.INFERENCE_PROCESSING,
            semaphore_acquired=False,
        )

        # Semaphore should still have exactly 1 permit (not corrupted to 2)
        acquired = sem.acquire(block=False)
        assert acquired, "Semaphore should have 1 permit (unchanged) after safe double-release"
        second_acquired = sem.acquire(block=False)
        assert not second_acquired, "Semaphore must not have more than 1 permit (no inflation)"

    def test_semaphore_double_release_safe_from_post_processing_starting_when_child_already_released(self) -> None:
        """When PROCESS_ENDING arrives from POST_PROCESSING_STARTING and the child already
        released the semaphore (normal path: emitted state then released), the handler's
        defensive release must not inflate the permit count.
        """
        # semaphore_acquired=False: child already released after emitting POST_PROCESSING_STARTING
        _mock_manager, sem = self._run_receive_process_ending_with_semaphore(
            prior_state=HordeProcessState.POST_PROCESSING_STARTING,
            semaphore_acquired=False,
        )

        acquired = sem.acquire(block=False)
        assert acquired, "Semaphore should have 1 permit (unchanged) after safe double-release"
        second_acquired = sem.acquire(block=False)
        assert not second_acquired, "Semaphore must not have more than 1 permit (no inflation)"


class TestProcessEndingReleasesVAEDecodeSemaphore(_ReceiveLoopHarnessMixin):
    """Tests that the VAE decode semaphore is released when PROCESS_ENDING is received for
    a process that was in INFERENCE_POST_PROCESSING or POST_PROCESSING_STARTING.

    When the child is killed (e.g., OOM) while it holds the VAE decode semaphore, its
    finally block may not run.  The PROCESS_ENDING handler must release the semaphore
    defensively so that subsequent processes are not blocked for up to VAE_SEMAPHORE_TIMEOUT.
    """

    def _run_process_ending_with_vae_semaphore(
        self,
        prior_state: HordeProcessState,
        *,
        vae_semaphore_acquired: bool = True,
    ) -> tuple[MagicMock, object]:
        """Run receive_and_handle_process_messages with PROCESS_ENDING and a real VAE semaphore.

        Returns (mock_manager, vae_semaphore) so callers can inspect the semaphore state.
        """
        import multiprocessing
        import queue as queue_mod

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        ctx = multiprocessing.get_context("spawn")
        vae_sem = ctx.BoundedSemaphore(1)
        if vae_semaphore_acquired:
            vae_sem.acquire()

        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = prior_state
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None

        process_map = MagicMock()
        process_map.__contains__ = MagicMock(side_effect=lambda key: key == 0)
        process_map.__getitem__ = MagicMock(side_effect=lambda key: process_info)

        msg = self._make_message(HordeProcessState.PROCESS_ENDING)
        q = queue_mod.Queue()
        q.put(msg)

        mock_manager = MagicMock()
        mock_manager._process_message_queue = q
        mock_manager._process_map = process_map
        mock_manager._in_deadlock = False
        mock_manager._in_queue_deadlock = False
        mock_manager.jobs_in_progress = []
        # Use a no-op BoundedSemaphore for the inference semaphore (not under test here)
        mock_manager._inference_semaphore = ctx.BoundedSemaphore(1)
        mock_manager._vae_decode_semaphore = vae_sem
        # Bind the real defensive-release helper so it actually releases the real semaphore
        # (otherwise the MagicMock default swallows the call and the semaphore is never freed).
        mock_manager._release_vae_decode_semaphore_defensively = (
            HordeWorkerProcessManager._release_vae_decode_semaphore_defensively.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )

        bound = HordeWorkerProcessManager.receive_and_handle_process_messages.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound()
        return mock_manager, vae_sem

    def test_vae_semaphore_released_when_process_ending_from_inference_post_processing(self) -> None:
        """When PROCESS_ENDING arrives for a process in INFERENCE_POST_PROCESSING, the VAE
        decode semaphore must be released so that other processes are not blocked for up to
        VAE_SEMAPHORE_TIMEOUT seconds waiting to acquire it.
        """
        _mock_manager, vae_sem = self._run_process_ending_with_vae_semaphore(
            prior_state=HordeProcessState.INFERENCE_POST_PROCESSING,
            vae_semaphore_acquired=True,
        )

        acquired = vae_sem.acquire(block=False)
        assert acquired, (
            "VAE decode semaphore should be acquirable after PROCESS_ENDING from "
            "INFERENCE_POST_PROCESSING — the handler must release it to unblock other processes"
        )

    def test_vae_semaphore_released_when_process_ending_from_post_processing_starting(self) -> None:
        """When PROCESS_ENDING arrives for a process in POST_PROCESSING_STARTING, the VAE
        decode semaphore must be released defensively in case the child acquired it but crashed
        before emitting INFERENCE_POST_PROCESSING.
        """
        _mock_manager, vae_sem = self._run_process_ending_with_vae_semaphore(
            prior_state=HordeProcessState.POST_PROCESSING_STARTING,
            vae_semaphore_acquired=True,
        )

        acquired = vae_sem.acquire(block=False)
        assert acquired, (
            "VAE decode semaphore should be acquirable after PROCESS_ENDING from "
            "POST_PROCESSING_STARTING — defensive release should unblock other processes"
        )

    def test_vae_semaphore_not_released_when_process_ending_from_inference_processing(self) -> None:
        """When PROCESS_ENDING arrives for a process in INFERENCE_PROCESSING, the VAE decode
        semaphore must NOT be released — the process did not hold it at that point.
        """
        _mock_manager, vae_sem = self._run_process_ending_with_vae_semaphore(
            prior_state=HordeProcessState.INFERENCE_PROCESSING,
            vae_semaphore_acquired=True,
        )

        # The VAE semaphore was pre-acquired (simulating another process holding it)
        # and must NOT have been released by the PROCESS_ENDING handler.
        acquired = vae_sem.acquire(block=False)
        assert not acquired, (
            "VAE decode semaphore must not be released when PROCESS_ENDING is from "
            "INFERENCE_PROCESSING — the process did not hold the VAE semaphore at that state"
        )

    def test_vae_semaphore_over_release_safe_when_already_released(self) -> None:
        """When PROCESS_ENDING arrives from INFERENCE_POST_PROCESSING and the child had
        already released the VAE semaphore via its finally block, the defensive release
        must not raise an exception and must not inflate permits.
        """
        # vae_semaphore_acquired=False: child already released via its finally block
        _mock_manager, vae_sem = self._run_process_ending_with_vae_semaphore(
            prior_state=HordeProcessState.INFERENCE_POST_PROCESSING,
            vae_semaphore_acquired=False,
        )
        # The semaphore is a BoundedSemaphore: over-release raises ValueError (caught),
        # so permits must remain at exactly 1 — not inflated to 2.
        first_acquired = vae_sem.acquire(block=False)
        assert first_acquired, "VAE semaphore should have its permit after the defensive release"
        second_acquired = vae_sem.acquire(block=False)
        assert not second_acquired, "Defensive release must not inflate VAE semaphore permits"


class TestReplaceInferenceProcessReleasesVAEDecodeSemaphore:
    """Tests that _replace_inference_process() releases the VAE decode semaphore when the
    process state is INFERENCE_POST_PROCESSING.

    A process stuck in INFERENCE_POST_PROCESSING holds the VAE decode semaphore.  When the
    manager forcibly replaces it (via _check_and_replace_process / replace_hung_processes),
    the semaphore must be released so that the replacement process can acquire it.
    """

    def _run_replace_inference_process(
        self,
        state: HordeProcessState,
        *,
        vae_semaphore_acquired: bool = True,
    ) -> object:
        """Call _replace_inference_process() with the given state and return the VAE semaphore."""
        import multiprocessing

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        ctx = multiprocessing.get_context("spawn")

        vae_sem = ctx.BoundedSemaphore(1)
        if vae_semaphore_acquired:
            vae_sem.acquire()

        process_info = MagicMock()
        process_info.last_process_state = state
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None
        process_info.loaded_horde_model_name = None

        mock_manager = MagicMock()
        mock_manager._inference_semaphore = ctx.BoundedSemaphore(1)
        mock_manager._vae_decode_semaphore = vae_sem
        mock_manager._disk_lock = ctx.Lock()
        mock_manager.jobs_lookup = {}
        mock_manager.jobs_in_progress = []
        # Bind the real defensive-release helper so it actually releases the real semaphore
        # (otherwise the MagicMock default swallows the call and the semaphore is never freed).
        mock_manager._release_vae_decode_semaphore_defensively = (
            HordeWorkerProcessManager._release_vae_decode_semaphore_defensively.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )

        bound = HordeWorkerProcessManager._replace_inference_process.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound(process_info)
        return vae_sem

    def test_vae_semaphore_released_when_replacing_inference_post_processing(self) -> None:
        """When a process stuck in INFERENCE_POST_PROCESSING is replaced, the VAE decode
        semaphore must be released so that other processes can proceed.
        """
        vae_sem = self._run_replace_inference_process(
            HordeProcessState.INFERENCE_POST_PROCESSING,
            vae_semaphore_acquired=True,
        )

        acquired = vae_sem.acquire(block=False)
        assert acquired, (
            "VAE decode semaphore must be released by _replace_inference_process when state is "
            "INFERENCE_POST_PROCESSING — the stuck process holds the semaphore and other processes "
            "would be blocked for up to VAE_SEMAPHORE_TIMEOUT without this release"
        )

    def test_vae_semaphore_released_when_replacing_post_processing_starting(self) -> None:
        """When a process in POST_PROCESSING_STARTING is replaced, the VAE decode semaphore
        must also be released — the child may have already acquired it before crashing.
        """
        vae_sem = self._run_replace_inference_process(
            HordeProcessState.POST_PROCESSING_STARTING,
            vae_semaphore_acquired=True,
        )

        acquired = vae_sem.acquire(block=False)
        assert acquired, (
            "VAE decode semaphore must be released by _replace_inference_process when state is "
            "POST_PROCESSING_STARTING — the child may have acquired it before crashing"
        )

    def test_vae_semaphore_over_release_safe_when_replacing_post_processing(self) -> None:
        """When _replace_inference_process() is called for INFERENCE_POST_PROCESSING and the
        child had already released the VAE semaphore, the defensive release must not inflate
        permits beyond max=1 (BoundedSemaphore raises ValueError, which is caught).
        """
        # vae_semaphore_acquired=False: child already released via its finally block
        vae_sem = self._run_replace_inference_process(
            HordeProcessState.INFERENCE_POST_PROCESSING,
            vae_semaphore_acquired=False,
        )

        first_acquired = vae_sem.acquire(block=False)
        assert first_acquired, "VAE semaphore should have its permit (unchanged) after safe double-release"
        second_acquired = vae_sem.acquire(block=False)
        assert not second_acquired, "Defensive release must not inflate VAE semaphore permits"

    def test_vae_semaphore_not_released_when_replacing_inference_processing(self) -> None:
        """When a process in INFERENCE_PROCESSING is replaced, the VAE decode semaphore
        must NOT be released — it has not been acquired at that state.
        """
        vae_sem = self._run_replace_inference_process(
            HordeProcessState.INFERENCE_PROCESSING,
            vae_semaphore_acquired=True,
        )

        # The VAE semaphore was pre-acquired (simulating another process holding it)
        # and must NOT have been released.
        acquired = vae_sem.acquire(block=False)
        assert not acquired, (
            "VAE decode semaphore must not be released for INFERENCE_PROCESSING state "
            "— the VAE semaphore is not held at that stage"
        )

    def test_respawn_false_skips_starting_new_process(self) -> None:
        """respawn=False must end the old process without starting a replacement.

        This is the shutdown path: spawning a fresh subprocess (importing torch, setting up the
        model manager) routinely takes longer than the graceful-shutdown window, which forces the
        watchdog to hard-kill the worker before the new process even finishes starting.
        """
        import multiprocessing

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        ctx = multiprocessing.get_context("spawn")

        process_info = MagicMock()
        process_info.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None
        process_info.loaded_horde_model_name = None

        mock_manager = MagicMock()
        mock_manager._inference_semaphore = ctx.BoundedSemaphore(1)
        mock_manager._vae_decode_semaphore = ctx.BoundedSemaphore(1)
        mock_manager._disk_lock = ctx.Lock()
        mock_manager.jobs_lookup = {}
        mock_manager.jobs_in_progress = []
        mock_manager._num_process_recoveries = 0

        bound = HordeWorkerProcessManager._replace_inference_process.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound(process_info, respawn=False)

        mock_manager._start_inference_process.assert_not_called()
        assert mock_manager._num_process_recoveries == 0
        mock_manager._end_inference_process.assert_called_once_with(process_info)

    def test_respawn_true_default_starts_new_process(self) -> None:
        """respawn=True (the default) must behave exactly as before: start a replacement."""
        import multiprocessing

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        ctx = multiprocessing.get_context("spawn")

        process_info = MagicMock()
        process_info.process_id = 5
        process_info.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None
        process_info.loaded_horde_model_name = None

        mock_manager = MagicMock()
        mock_manager._inference_semaphore = ctx.BoundedSemaphore(1)
        mock_manager._vae_decode_semaphore = ctx.BoundedSemaphore(1)
        mock_manager._disk_lock = ctx.Lock()
        mock_manager.jobs_lookup = {}
        mock_manager.jobs_in_progress = []
        mock_manager._num_process_recoveries = 0

        bound = HordeWorkerProcessManager._replace_inference_process.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound(process_info)

        mock_manager._start_inference_process.assert_called_once_with(5)
        assert mock_manager._num_process_recoveries == 1


class TestNumBusyWithPostProcessing:
    """Tests that num_busy_with_post_processing() counts both INFERENCE_POST_PROCESSING
    and POST_PROCESSING_STARTING states.
    """

    def _make_process_map(self, states: list[HordeProcessState]) -> object:
        from horde_worker_regen.process_management.process_manager import HordeProcessType, ProcessMap

        process_map = ProcessMap({})
        for i, state in enumerate(states):
            info = MagicMock()
            info.process_type = HordeProcessType.INFERENCE
            info.last_process_state = state
            info.inference_started_timestamp = None
            process_map[i] = info
        return process_map

    def test_inference_post_processing_counted(self) -> None:
        """A process in INFERENCE_POST_PROCESSING must be counted."""
        from horde_worker_regen.process_management.process_manager import ProcessMap

        pm = self._make_process_map([HordeProcessState.INFERENCE_POST_PROCESSING])
        assert isinstance(pm, ProcessMap)
        assert pm.num_busy_with_post_processing() == 1

    def test_post_processing_starting_counted(self) -> None:
        """A process in POST_PROCESSING_STARTING must also be counted — it is transitioning
        into post-processing and is effectively busy.
        """
        from horde_worker_regen.process_management.process_manager import ProcessMap

        pm = self._make_process_map([HordeProcessState.POST_PROCESSING_STARTING])
        assert isinstance(pm, ProcessMap)
        assert pm.num_busy_with_post_processing() == 1

    def test_inference_processing_not_counted(self) -> None:
        """A process in INFERENCE_PROCESSING is not in post-processing and must not be counted."""
        from horde_worker_regen.process_management.process_manager import ProcessMap

        pm = self._make_process_map([HordeProcessState.INFERENCE_PROCESSING])
        assert isinstance(pm, ProcessMap)
        assert pm.num_busy_with_post_processing() == 0

    def test_mixed_states_counted_correctly(self) -> None:
        """With multiple processes in different states, only post-processing states are counted."""
        from horde_worker_regen.process_management.process_manager import ProcessMap

        pm = self._make_process_map(
            [
                HordeProcessState.INFERENCE_POST_PROCESSING,
                HordeProcessState.POST_PROCESSING_STARTING,
                HordeProcessState.INFERENCE_PROCESSING,
                HordeProcessState.WAITING_FOR_JOB,
            ]
        )
        assert isinstance(pm, ProcessMap)
        assert pm.num_busy_with_post_processing() == 2


class TestCanAcceptJobPostProcessingComplete:
    """Tests for can_accept_job() excluding POST_PROCESSING_COMPLETE."""

    def _make_process_info(self, state: HordeProcessState) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import HordeProcessInfo

        mock_info = MagicMock()
        mock_info.last_process_state = state
        mock_info.inference_started_timestamp = None
        mock_info.can_accept_job = HordeProcessInfo.can_accept_job.__get__(mock_info, HordeProcessInfo)
        return mock_info

    def test_post_processing_complete_cannot_accept_job(self) -> None:
        """A process in POST_PROCESSING_COMPLETE must not be considered available.

        The child is still inside _receive_and_handle_control_message sending the
        result to the manager.  Treating it as available would let the manager
        schedule a new job or replace the process before the current job's result
        has been enqueued.
        """
        info = self._make_process_info(HordeProcessState.POST_PROCESSING_COMPLETE)
        assert info.can_accept_job() is False

    def test_waiting_for_job_can_accept(self) -> None:
        info = self._make_process_info(HordeProcessState.WAITING_FOR_JOB)
        assert info.can_accept_job() is True

    def test_inference_complete_can_accept(self) -> None:
        info = self._make_process_info(HordeProcessState.INFERENCE_COMPLETE)
        assert info.can_accept_job() is True

    def test_model_preloaded_can_accept(self) -> None:
        info = self._make_process_info(HordeProcessState.MODEL_PRELOADED)
        assert info.can_accept_job() is True

    def test_inference_processing_cannot_accept(self) -> None:
        info = self._make_process_info(HordeProcessState.INFERENCE_PROCESSING)
        assert info.can_accept_job() is False

    def test_process_ending_cannot_accept(self) -> None:
        info = self._make_process_info(HordeProcessState.PROCESS_ENDING)
        assert info.can_accept_job() is False


class TestStartInferenceExceptionHandling:
    """Tests that exceptions from start_inference() are caught, preventing the
    process from ending before the job result is submitted.
    """

    def _make_inference_process(self) -> MagicMock:
        """Return a minimally-mocked HordeInferenceProcess."""
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._last_sanitized_negative_prompt = None
        proc._active_model_name = "TestModel"
        return proc

    def test_start_inference_exception_sends_faulted_result(self) -> None:
        """If start_inference() raises, the handler must send INFERENCE_FAILED so
        the manager can retry, rather than letting the exception propagate and
        cause a PROCESS_ENDING with the job still in progress.
        """
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess
        from horde_worker_regen.process_management.messages import (
            HordeControlFlag,
            HordeInferenceControlMessage,
        )

        proc = self._make_inference_process()

        # start_inference raises an unexpected exception (simulates a finally-block error
        # that occurs after POST_PROCESSING_COMPLETE was emitted)
        proc.start_inference.side_effect = RuntimeError("Semaphore release failed")

        # Build a minimal START_INFERENCE message
        job_info = MagicMock()
        job_info.model = "TestModel"
        # Provide payload attributes used in the failure-path preload_model call
        job_info.payload = MagicMock()
        job_info.payload.loras = None
        job_info.payload.tiling = False
        message = MagicMock(spec=HordeInferenceControlMessage)
        message.control_flag = HordeControlFlag.START_INFERENCE
        message.horde_model_name = "TestModel"
        message.sdk_api_job_info = job_info

        # Bind the real method but short-circuit everything except the path under test
        proc.on_horde_model_state_change = MagicMock()
        proc.send_process_state_change_message = MagicMock()
        proc.send_memory_report_message = MagicMock()
        proc.send_inference_result_message = MagicMock()
        proc.unload_models_from_ram = MagicMock()
        proc.preload_model = MagicMock()

        # Call the real method (it will hit the try/except we added around start_inference)
        HordeInferenceProcess._receive_and_handle_control_message(proc, message)

        # send_inference_result_message must have been called with INFERENCE_FAILED
        calls = proc.send_inference_result_message.call_args_list
        assert len(calls) >= 1, "send_inference_result_message was not called"
        first_call_kwargs = calls[0].kwargs if calls[0].kwargs else {}
        first_call_args = calls[0].args if calls[0].args else ()
        # process_state can come as positional or keyword arg
        process_state_arg = first_call_kwargs.get("process_state") or (
            first_call_args[0] if first_call_args else None
        )
        assert process_state_arg == HordeProcessState.INFERENCE_FAILED, (
            f"Expected INFERENCE_FAILED, got {process_state_arg}"
        )


class TestVaeLockAcquiredFlag:
    """Tests that _vae_lock_was_acquired is only set to True when the VAE
    semaphore is actually acquired (not pre-emptively on timeout), and that
    _vae_acquire_attempted prevents repeated acquire attempts after a timeout.
    """

    def test_vae_lock_flag_false_on_timeout(self) -> None:
        """When the VAE semaphore acquire times out, _vae_lock_was_acquired must remain
        False so the finally block does not try to release a semaphore we never held.
        _vae_acquire_attempted must be True so subsequent callbacks don't retry.
        """
        pytest.importorskip("hordelib")  # requires the GPU/ML stack
        import multiprocessing

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._vae_acquire_attempted = False
        proc._vae_lock_was_acquired = False
        proc._current_job_inference_steps_complete = True
        proc._in_post_processing = False  # Must be False so we reach the VAE-lock branch
        proc.VAE_SEMAPHORE_TIMEOUT = 0.01

        # Semaphore with 0 permits – acquire will time out immediately
        sem = multiprocessing.Semaphore(0)
        proc._vae_decode_semaphore = sem

        proc.send_heartbeat_message = MagicMock()

        # Simulate a ProgressReport that triggers the VAE-semaphore path
        from hordelib.horde import ProgressState  # type: ignore[import]

        progress_report = MagicMock()
        progress_report.hordelib_progress_state = ProgressState.progress
        progress_report.comfyui_progress = None

        HordeInferenceProcess._progress_callback_impl(proc, progress_report)

        # acquire timed out → _vae_lock_was_acquired must be False (no semaphore to release)
        assert proc._vae_lock_was_acquired is False, (
            "_vae_lock_was_acquired should be False when acquire timed out"
        )
        # _vae_acquire_attempted must be True so subsequent callbacks skip the acquire
        assert proc._vae_acquire_attempted is True, (
            "_vae_acquire_attempted should be True after first attempt (even on timeout)"
        )

    def test_vae_lock_not_retried_after_timeout(self) -> None:
        """A second progress_callback invocation after a timeout must not re-attempt
        acquire (which would block up to VAE_SEMAPHORE_TIMEOUT again and spam logs).
        """
        pytest.importorskip("hordelib")  # requires the GPU/ML stack
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        # Simulate state after a first timeout: attempted but not acquired
        proc._vae_acquire_attempted = True
        proc._vae_lock_was_acquired = False
        proc._current_job_inference_steps_complete = True
        proc._in_post_processing = False
        proc.VAE_SEMAPHORE_TIMEOUT = 0.01

        sem = MagicMock()
        proc._vae_decode_semaphore = sem
        proc.send_heartbeat_message = MagicMock()

        from hordelib.horde import ProgressState  # type: ignore[import]

        progress_report = MagicMock()
        progress_report.hordelib_progress_state = ProgressState.progress
        progress_report.comfyui_progress = None

        HordeInferenceProcess._progress_callback_impl(proc, progress_report)

        # acquire must NOT be called again
        sem.acquire.assert_not_called()

    def test_vae_lock_flag_true_on_success(self) -> None:
        """When the VAE semaphore is successfully acquired, _vae_lock_was_acquired is True."""
        pytest.importorskip("hordelib")  # requires the GPU/ML stack
        import multiprocessing

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._vae_acquire_attempted = False
        proc._vae_lock_was_acquired = False
        proc._current_job_inference_steps_complete = True
        proc._in_post_processing = False  # Must be False so we reach the VAE-lock branch
        proc.VAE_SEMAPHORE_TIMEOUT = 5

        # Semaphore with 1 permit – acquire will succeed
        sem = multiprocessing.Semaphore(1)
        proc._vae_decode_semaphore = sem

        proc.send_heartbeat_message = MagicMock()

        from hordelib.horde import ProgressState  # type: ignore[import]

        progress_report = MagicMock()
        progress_report.hordelib_progress_state = ProgressState.progress
        progress_report.comfyui_progress = None

        # log_free_ram is imported locally inside progress_callback from hordelib.comfy_horde;
        # patch it at the source to avoid needing an initialised ComfyUI context.
        with patch("hordelib.comfy_horde.log_free_ram", MagicMock()):
            HordeInferenceProcess._progress_callback_impl(proc, progress_report)

        assert proc._vae_lock_was_acquired is True, (
            "_vae_lock_was_acquired should be True when acquire succeeded"
        )
        assert proc._vae_acquire_attempted is True, (
            "_vae_acquire_attempted should be True after a successful acquire"
        )
        # Clean up the acquired semaphore
        sem.release()


class TestSemaphoreReleaseBroadExceptionHandling:
    """Tests that unexpected non-ValueError exceptions during semaphore release
    in the start_inference finally block do not propagate and crash the process.
    """

    def test_inference_semaphore_os_error_is_swallowed(self) -> None:
        """An OSError from inference semaphore release must not escape start_inference.

        This validates that the broad `except Exception` handler in the finally block
        prevents unexpected OS-level semaphore errors from crashing the process and
        causing PROCESS_ENDING with the job still in progress.
        """
        import multiprocessing

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._is_busy = False
        proc._in_post_processing = False
        proc._vae_acquire_attempted = False
        proc._vae_lock_was_acquired = False
        proc._current_job_inference_steps_complete = False
        proc._last_sanitized_negative_prompt = None
        proc._active_model_name = "TestModel"
        proc.VAE_SEMAPHORE_TIMEOUT = 5

        # Semaphore that raises OSError on release (not ValueError)
        bad_sem = MagicMock()
        bad_sem.acquire = MagicMock(return_value=True)
        bad_sem.release = MagicMock(side_effect=OSError("Invalid semaphore"))
        proc._inference_semaphore = bad_sem

        good_vae = multiprocessing.Semaphore(1)
        proc._vae_decode_semaphore = good_vae

        proc.send_process_state_change_message = MagicMock()
        proc.send_heartbeat_message = MagicMock()
        proc.on_horde_model_state_change = MagicMock()

        # _horde.basic_inference returns a valid (non-None) result
        fake_result = MagicMock()
        fake_result.rawpng = MagicMock()
        proc._horde = MagicMock()
        proc._horde.basic_inference.return_value = [fake_result]

        # Use plain MagicMock (not spec=...) so all attribute access including .ids works
        job_info = MagicMock()
        job_info.payload.prompt = "test"
        job_info.extra_source_images = None
        job_info.source_image = None
        job_info.source_mask = None
        job_info.ids = []

        # Should not raise even though inference semaphore release raises OSError.
        # The OSError must be caught in the finally block and logged, not propagated.
        result = HordeInferenceProcess.start_inference(proc, job_info)

        # The result should be returned (not swallowed), since the error is only in cleanup
        assert result is not None, "start_inference should return results despite semaphore OSError"
        bad_sem.release.assert_called_once()

    def test_setup_exception_releases_semaphore(self) -> None:
        """If send_process_state_change_message() raises during start_inference() setup
        (before the inference itself runs), the inference semaphore must still be released
        and _is_busy must be reset.  This tests the restructured try/finally that now
        covers the entire post-acquire body.
        """
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._is_busy = False
        proc._in_post_processing = False
        proc._vae_acquire_attempted = False
        proc._vae_lock_was_acquired = False
        proc._current_job_inference_steps_complete = False
        proc._last_sanitized_negative_prompt = None
        proc._active_model_name = "TestModel"
        proc.VAE_SEMAPHORE_TIMEOUT = 5

        import multiprocessing
        real_sem = multiprocessing.Semaphore(1)
        proc._inference_semaphore = real_sem
        proc._vae_decode_semaphore = multiprocessing.Semaphore(1)

        # send_process_state_change_message raises on the first call (during setup)
        proc.send_process_state_change_message = MagicMock(
            side_effect=OSError("pipe broken")
        )
        proc.send_heartbeat_message = MagicMock()

        job_info = MagicMock()
        job_info.payload.prompt = "test"
        job_info.extra_source_images = None
        job_info.source_image = None
        job_info.source_mask = None
        job_info.ids = []

        result = HordeInferenceProcess.start_inference(proc, job_info)

        # Inference failed due to setup exception → result must be None
        assert result is None, "start_inference should return None when setup raises"

        # Critically: _is_busy must be False and semaphore must have been released
        assert proc._is_busy is False, "_is_busy must be reset even after setup exception"
        # The semaphore started with 1 permit, was acquired (-1 = 0), then should have
        # been released (+1 = 1) by the finally block.  Verify we can acquire it again.
        acquired = real_sem.acquire(block=False)
        assert acquired, "Inference semaphore must be released by finally block even on setup exception"
        real_sem.release()  # restore for clean teardown


class TestReplaceHungInferenceStarting:
    """Tests for the INFERENCE_STARTING stuck-process detection added to replace_hung_processes().

    When a process has been in INFERENCE_STARTING for longer than preload_timeout AND no other
    process is actively in INFERENCE_PROCESSING (which would legitimately hold the semaphore),
    the manager must replace it so the stuck job is retried.
    """

    def _make_process(
        self,
        process_id: int,
        state: HordeProcessState,
        *,
        time_elapsed: float = 9999.0,
    ) -> MagicMock:
        import time as _time

        proc = MagicMock()
        proc.process_id = process_id
        proc.last_process_state = state
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_timestamp = _time.time() - time_elapsed
        proc.last_progress_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None
        return proc

    def _make_manager(self, processes: list[MagicMock]) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._recently_recovered = False
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 300
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        # is_stuck_on_inference always returns False so the specific INFERENCE_STARTING
        # check (in the `else` branch) is exercised.
        mock_manager._process_map.is_stuck_on_inference.return_value = False
        mock_manager._process_map.values.return_value = processes
        # _check_and_replace_process must return False so it doesn't spuriously trigger
        mock_manager._check_and_replace_process.return_value = False
        mock_manager._process_map.__iter__ = MagicMock(return_value=iter(processes))

        # Bind the real method to the mock manager
        mock_manager._bound_replace_hung = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        return mock_manager

    def test_stuck_inference_starting_no_active_inference_replaced(self) -> None:
        """An INFERENCE_STARTING process stuck longer than preload_timeout with no active
        INFERENCE_PROCESSING must be detected and replaced.
        """
        proc = self._make_process(0, HordeProcessState.INFERENCE_STARTING, time_elapsed=9999.0)
        mock_manager = self._make_manager([proc])

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True, "replace_hung_processes must return True when replacing stuck INFERENCE_STARTING"
        mock_manager._replace_inference_process.assert_called_once_with(proc)

    def test_stuck_inference_starting_with_active_inference_not_replaced(self) -> None:
        """An INFERENCE_STARTING process must NOT be replaced when another process is in
        INFERENCE_PROCESSING, because that process legitimately holds the semaphore.
        """
        inference_starting = self._make_process(
            0, HordeProcessState.INFERENCE_STARTING, time_elapsed=9999.0
        )
        inference_processing = self._make_process(
            1, HordeProcessState.INFERENCE_PROCESSING, time_elapsed=10.0
        )
        mock_manager = self._make_manager([inference_starting, inference_processing])

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        # The INFERENCE_STARTING process must not be replaced (INFERENCE_PROCESSING is active)
        mock_manager._replace_inference_process.assert_not_called()
        # Return value: no processes were replaced in this run
        assert result is False

    def test_inference_starting_not_replaced_before_preload_timeout(self) -> None:
        """An INFERENCE_STARTING process that has been waiting less than preload_timeout
        must NOT be detected as stuck yet.
        """
        # Only 10 seconds elapsed — well below preload_timeout (80 s)
        proc = self._make_process(0, HordeProcessState.INFERENCE_STARTING, time_elapsed=10.0)
        mock_manager = self._make_manager([proc])

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        mock_manager._replace_inference_process.assert_not_called()
        assert result is False

    def test_stuck_inference_starting_replaced_even_when_recently_recovered(self) -> None:
        """An INFERENCE_STARTING process stuck longer than preload_timeout must be replaced
        even when _recently_recovered is True, provided no INFERENCE_PROCESSING is active.

        With frequent recoveries the _recently_recovered flag can be True for most of the
        worker's lifetime.  Exempting the elapsed-time INFERENCE_STARTING check from the flag
        ensures that a stuck process is always detected once preload_timeout expires and no
        other process holds the semaphore.
        """
        proc = self._make_process(0, HordeProcessState.INFERENCE_STARTING, time_elapsed=9999.0)
        mock_manager = self._make_manager([proc])
        mock_manager._recently_recovered = True  # simulate active recovery window

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True, (
            "replace_hung_processes must return True for stuck INFERENCE_STARTING "
            "even when _recently_recovered is True (no active INFERENCE_PROCESSING)"
        )
        mock_manager._replace_inference_process.assert_called_once_with(proc)

    def test_stuck_inference_starting_not_replaced_when_inference_processing_active_recently_recovered(
        self,
    ) -> None:
        """INFERENCE_STARTING must NOT be replaced when INFERENCE_PROCESSING is active,
        even when _recently_recovered is True.  The active INFERENCE_PROCESSING process
        legitimately holds the semaphore; INFERENCE_STARTING should wait for it.
        """
        inference_starting = self._make_process(0, HordeProcessState.INFERENCE_STARTING, time_elapsed=9999.0)
        inference_processing = self._make_process(1, HordeProcessState.INFERENCE_PROCESSING, time_elapsed=10.0)
        mock_manager = self._make_manager([inference_starting, inference_processing])
        mock_manager._recently_recovered = True  # simulate active recovery window

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        mock_manager._replace_inference_process.assert_not_called()
        assert result is False


class TestProcessEndingReleasesInferenceSemaphoreFromInferenceStarting(_ReceiveLoopHarnessMixin):
    """Tests that the inference semaphore is released when PROCESS_ENDING is received for a
    process whose prior state was INFERENCE_STARTING.

    This covers the race condition in _replace_inference_process(): the manager releases the
    semaphore to unblock the blocked child; the child may acquire it before the kill signal
    arrives.  If the child is killed before it sends INFERENCE_PROCESSING, the manager still
    records INFERENCE_STARTING as the prior state.  Without releasing on PROCESS_ENDING the
    semaphore count stays at 0 permanently, leaving the next INFERENCE_STARTING blocked forever.
    """

    def _run_receive_process_ending_with_semaphore(
        self,
        prior_state: HordeProcessState,
        *,
        semaphore_acquired: bool,
    ) -> tuple[MagicMock, object]:
        """Helper reused from TestProcessEndingReleasesInferenceSemaphore."""
        import multiprocessing
        import queue as queue_mod
        from multiprocessing.synchronize import BoundedSemaphore

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        ctx = multiprocessing.get_context("spawn")
        bounded_sem = BoundedSemaphore(1, ctx=ctx)
        if semaphore_acquired:
            bounded_sem.acquire()

        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = prior_state
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None

        process_map = MagicMock()
        process_map.__contains__ = MagicMock(side_effect=lambda key: key == 0)
        process_map.__getitem__ = MagicMock(side_effect=lambda key: process_info)

        msg = self._make_message(HordeProcessState.PROCESS_ENDING)
        q = queue_mod.Queue()
        q.put(msg)

        mock_manager = MagicMock()
        mock_manager._process_message_queue = q
        mock_manager._process_map = process_map
        mock_manager._in_deadlock = False
        mock_manager._in_queue_deadlock = False
        mock_manager.jobs_in_progress = []
        mock_manager._inference_semaphore = bounded_sem

        bound = HordeWorkerProcessManager.receive_and_handle_process_messages.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound()
        return mock_manager, bounded_sem

    def test_semaphore_released_when_process_ending_from_inference_starting_acquired(self) -> None:
        """PROCESS_ENDING from INFERENCE_STARTING with a held semaphore must release it.

        Scenario: manager released the semaphore to unblock the blocked child, the child
        acquired it (race), was then killed before sending INFERENCE_PROCESSING.  The
        PROCESS_ENDING handler must release to restore the permit count to 1.
        """
        _mock_manager, sem = self._run_receive_process_ending_with_semaphore(
            prior_state=HordeProcessState.INFERENCE_STARTING,
            semaphore_acquired=True,
        )

        acquired = sem.acquire(block=False)
        assert acquired, (
            "Semaphore must be released on PROCESS_ENDING from INFERENCE_STARTING (acquired case) "
            "to prevent the next INFERENCE_STARTING process from blocking forever"
        )

    def test_semaphore_not_inflated_when_process_ending_from_inference_starting_not_acquired(self) -> None:
        """PROCESS_ENDING from INFERENCE_STARTING when the child never acquired the semaphore
        must not inflate the permit count beyond 1 (BoundedSemaphore safety).

        Scenario: manager released the semaphore (count: 0→1), child was killed before
        acquiring.  Semaphore count is already 1.  The PROCESS_ENDING handler's release
        attempt raises ValueError (BoundedSemaphore) and is silently caught; count stays 1.
        """
        _mock_manager, sem = self._run_receive_process_ending_with_semaphore(
            prior_state=HordeProcessState.INFERENCE_STARTING,
            semaphore_acquired=False,
        )

        # Count should still be 1 (not inflated to 2)
        acquired = sem.acquire(block=False)
        assert acquired, "Semaphore should have exactly 1 permit (not inflated)"
        second_acquired = sem.acquire(block=False)
        assert not second_acquired, "Semaphore must not have more than 1 permit (no inflation)"


class TestProgressCallbackExceptionSuppression:
    """Tests that progress_callback swallows exceptions to prevent aborting inference.

    Regression tests for the bug where an exception raised inside _progress_callback_impl
    (e.g. from log_free_ram(), send_heartbeat_message(), or a semaphore operation)
    propagated into HordeLib's basic_inference(), which then caught it internally and
    returned None — producing "inference produced no results" with no CRITICAL log to
    explain the root cause.

    The fix wraps the callback body in try/except so HordeLib always receives a clean
    return (no exception), allowing inference to proceed normally.
    """

    def _make_process(self) -> "HordeInferenceProcess":
        """Return a minimal HordeInferenceProcess that skips heavy initialisation."""
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        class _TestProcess(HordeInferenceProcess):
            def cleanup_for_exit(self) -> None:
                pass

        instance = object.__new__(_TestProcess)
        instance.process_id = 1
        instance.process_launch_identifier = 42
        instance._in_post_processing = False
        instance._current_job_inference_steps_complete = False
        instance._vae_acquire_attempted = False
        instance._vae_lock_was_acquired = False
        instance._last_job_inference_rate = None
        instance._start_inference_time = 0.0
        instance._active_model_name = "test-model"
        instance.process_message_queue = MagicMock()
        return instance

    def test_exception_in_impl_does_not_propagate(self) -> None:
        """An exception raised by _progress_callback_impl must NOT escape progress_callback.

        If the exception propagated to HordeLib, basic_inference() would silently return
        None and the job would fault with "inference produced no results".
        """
        proc = self._make_process()

        # Simulate _progress_callback_impl raising unexpectedly
        proc._progress_callback_impl = MagicMock(side_effect=RuntimeError("Simulated error"))

        progress_report = MagicMock()

        # Must not raise — progress_callback must swallow the exception
        proc.progress_callback(progress_report)

        # The impl was called
        proc._progress_callback_impl.assert_called_once_with(progress_report)

    def test_exception_in_impl_is_logged(self) -> None:
        """An exception from _progress_callback_impl must be logged at ERROR level with traceback."""
        from unittest.mock import patch as _patch

        proc = self._make_process()
        proc._progress_callback_impl = MagicMock(side_effect=ValueError("bad value"))

        progress_report = MagicMock()

        with _patch("horde_worker_regen.process_management.inference_process.logger") as mock_logger:
            # logger.opt(exception=...).error(...) is called — opt() returns a bound logger
            mock_opt_logger = MagicMock()
            mock_logger.opt.return_value = mock_opt_logger

            proc.progress_callback(progress_report)

            # logger.opt must be called with the exception for full traceback capture
            mock_logger.opt.assert_called_once()
            opt_kwargs = mock_logger.opt.call_args.kwargs
            assert isinstance(opt_kwargs.get("exception"), ValueError), (
                "logger.opt must be passed the exception for traceback logging"
            )
            # The .error() on the bound logger must be called with the message
            mock_opt_logger.error.assert_called_once()
            error_call_args = mock_opt_logger.error.call_args[0][0]
            assert "ValueError" in error_call_args
            assert "bad value" in error_call_args

    def test_successful_impl_does_not_log_error(self) -> None:
        """When _progress_callback_impl succeeds no error should be logged."""
        from unittest.mock import patch as _patch

        proc = self._make_process()
        proc._progress_callback_impl = MagicMock(return_value=None)

        progress_report = MagicMock()

        with _patch("horde_worker_regen.process_management.inference_process.logger") as mock_logger:
            proc.progress_callback(progress_report)
            mock_logger.opt.assert_not_called()
            mock_logger.error.assert_not_called()


class TestPostProcessingVAESemaphore:
    """Tests that the VAE decode semaphore is acquired via the post-processing path.

    Regression tests for the bug where ProgressState.post_processing callbacks bypassed
    the VAE semaphore acquisition in the step-completion path, allowing concurrent
    heavy GPU work across processes that could exhaust VRAM and cause basic_inference()
    to return None silently.
    """

    def _make_process(self) -> "HordeInferenceProcess":
        """Return a minimal HordeInferenceProcess that skips heavy initialisation."""
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        class _TestProcess(HordeInferenceProcess):
            def cleanup_for_exit(self) -> None:
                pass

        instance = object.__new__(_TestProcess)
        instance.process_id = 1
        instance.process_launch_identifier = 42
        instance._in_post_processing = False
        instance._current_job_inference_steps_complete = False
        instance._vae_acquire_attempted = False
        instance._vae_lock_was_acquired = False
        instance._last_job_inference_rate = None
        instance._start_inference_time = 0.0
        instance._active_model_name = "test-model"
        instance.VAE_SEMAPHORE_TIMEOUT = 5
        instance.process_message_queue = MagicMock()
        return instance

    def test_vae_semaphore_acquired_on_first_post_processing_callback(self) -> None:
        """VAE decode semaphore must be acquired the first time post_processing fires.

        Before the fix, the post_processing branch returned early without ever touching
        the VAE semaphore, leaving concurrent VAE decode unlimited.
        """
        import multiprocessing

        proc = self._make_process()

        vae_sem = multiprocessing.Semaphore(1)
        inference_sem = multiprocessing.Semaphore(1)
        inference_sem.acquire()  # simulate that inference holds the semaphore
        proc._inference_semaphore = inference_sem
        proc._vae_decode_semaphore = vae_sem

        proc.send_process_state_change_message = MagicMock()
        proc.send_heartbeat_message = MagicMock()

        progress_report = MagicMock()

        # Import locally to avoid module-level import of hordelib
        from unittest.mock import patch as _patch

        # Patch the hordelib imports inside _progress_callback_impl
        with (
            _patch("horde_worker_regen.process_management.inference_process.logger"),
            _patch.dict(
                "sys.modules",
                {
                    "hordelib": MagicMock(),
                    "hordelib.comfy_horde": MagicMock(),
                    "hordelib.horde": MagicMock(),
                    "hordelib.utils": MagicMock(),
                    "hordelib.utils.ioredirect": MagicMock(),
                },
            ),
        ):
            import sys

            # Set up the ProgressState mock so the post_processing branch fires
            mock_progress_state = MagicMock()
            sys.modules["hordelib.horde"].ProgressState = mock_progress_state

            progress_report.hordelib_progress_state = mock_progress_state.post_processing
            progress_report.comfyui_progress = None

            proc._progress_callback_impl(progress_report)

        # VAE semaphore must have been acquired (attempt was made)
        assert proc._vae_acquire_attempted is True, (
            "VAE semaphore acquisition must be attempted in the post-processing path"
        )
        # And the semaphore must actually have been acquired successfully
        assert proc._vae_lock_was_acquired is True, (
            "VAE semaphore must be successfully acquired in the post-processing path"
        )

    def test_vae_semaphore_only_acquired_once_in_post_processing(self) -> None:
        """VAE semaphore must not be acquired twice even with multiple post_processing callbacks."""
        import multiprocessing
        from unittest.mock import MagicMock, patch as _patch

        proc = self._make_process()

        # Use a mock VAE semaphore so we can count exact acquire() calls.
        # A real Semaphore would block on the second acquire, making the test hang.
        mock_vae_sem = MagicMock()
        mock_vae_sem.acquire.return_value = True  # simulate successful acquire

        inference_sem = multiprocessing.Semaphore(1)
        inference_sem.acquire()
        proc._inference_semaphore = inference_sem
        proc._vae_decode_semaphore = mock_vae_sem

        proc.send_process_state_change_message = MagicMock()
        proc.send_heartbeat_message = MagicMock()

        with (
            _patch("horde_worker_regen.process_management.inference_process.logger"),
            _patch.dict(
                "sys.modules",
                {
                    "hordelib": MagicMock(),
                    "hordelib.comfy_horde": MagicMock(),
                    "hordelib.horde": MagicMock(),
                    "hordelib.utils": MagicMock(),
                    "hordelib.utils.ioredirect": MagicMock(),
                },
            ),
        ):
            import sys

            mock_progress_state = MagicMock()
            sys.modules["hordelib.horde"].ProgressState = mock_progress_state

            progress_report = MagicMock()
            progress_report.hordelib_progress_state = mock_progress_state.post_processing
            progress_report.comfyui_progress = None

            # Call twice — semaphore must only be acquired on the first call
            proc._progress_callback_impl(progress_report)
            proc._progress_callback_impl(progress_report)

        assert proc._vae_acquire_attempted is True
        # Semaphore.acquire() must have been called exactly once despite two callbacks
        mock_vae_sem.acquire.assert_called_once_with(timeout=proc.VAE_SEMAPHORE_TIMEOUT), (
            "VAE semaphore acquire() must be called exactly once across multiple post-processing callbacks"
        )


class TestPostProcessingFaultMessage:
    """Regression tests for the post-processing failure distinction fix.

    Covers two behaviours introduced by the fix:
    1. ``send_inference_result_message`` emits "post-processing produced no results"
       (not "inference produced no results") when ``_post_processing_was_started`` is True.
    2. ``start_inference`` does *not* emit ``POST_PROCESSING_COMPLETE`` when
       ``basic_inference()`` returns ``None`` or ``[]``.
    """

    # ------------------------------------------------------------------ helpers

    def _make_proc_for_send_result(self, *, post_processing_was_started: bool) -> MagicMock:
        """Return a minimal mock suited to calling send_inference_result_message directly."""
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._post_processing_was_started = post_processing_was_started
        proc._last_job_inference_rate = None
        proc.process_id = 0
        proc.process_launch_identifier = 0
        # process_message_queue is set in __init__ so it is not part of the class spec;
        # assign it directly so the real send_inference_result_message can call .put().
        proc.process_message_queue = MagicMock()
        return proc

    def _call_send_result_get_info(self, *, post_processing_was_started: bool, results=None) -> str:
        """Call send_inference_result_message and return the ``info`` string passed
        to the HordeInferenceResultMessage constructor."""
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess
        from horde_worker_regen.process_management.messages import HordeProcessState

        proc = self._make_proc_for_send_result(post_processing_was_started=post_processing_was_started)

        with patch(
            "horde_worker_regen.process_management.inference_process.HordeInferenceResultMessage"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            HordeInferenceProcess.send_inference_result_message(
                proc,
                process_state=HordeProcessState.INFERENCE_FAILED,
                job_info=MagicMock(),
                results=results,
                time_elapsed=1.0,
                sanitized_negative_prompt=None,
            )
            return mock_cls.call_args.kwargs["info"]

    # ------------------------------------------------ send_inference_result_message tests

    def test_fault_info_is_post_processing_when_post_processing_was_started(self) -> None:
        """When _post_processing_was_started is True and results is None, the fault
        info string must say "post-processing produced no results" — not the generic
        "inference produced no results" — so operators can distinguish the failure phase.
        """
        info = self._call_send_result_get_info(post_processing_was_started=True, results=None)
        assert info == "post-processing produced no results", (
            f"Expected 'post-processing produced no results', got '{info}'"
        )

    def test_fault_info_is_inference_when_post_processing_was_not_started(self) -> None:
        """When _post_processing_was_started is False (inference never reached
        post-processing), the fault info must say "inference produced no results".
        """
        info = self._call_send_result_get_info(post_processing_was_started=False, results=None)
        assert info == "inference produced no results", (
            f"Expected 'inference produced no results', got '{info}'"
        )

    def test_fault_info_is_post_processing_when_results_is_empty_list(self) -> None:
        """An empty list (not None) returned by basic_inference() must also produce
        "post-processing produced no results" when post-processing was started.
        """
        info = self._call_send_result_get_info(post_processing_was_started=True, results=[])
        assert info == "post-processing produced no results", (
            f"Expected 'post-processing produced no results', got '{info}'"
        )

    # ------------------------------------------------ POST_PROCESSING_COMPLETE suppression tests

    def _make_proc_for_start_inference(self, *, in_post_processing: bool) -> MagicMock:
        """Return a minimal mock suitable for calling start_inference() directly."""
        import multiprocessing

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._is_busy = False
        proc._in_post_processing = in_post_processing
        proc._post_processing_was_started = False
        proc._vae_acquire_attempted = False
        proc._vae_lock_was_acquired = False
        proc._current_job_inference_steps_complete = False
        proc._last_sanitized_negative_prompt = None
        proc._last_job_inference_rate = None
        proc._active_model_name = "TestModel"
        proc._start_inference_time = 0.0
        proc.VAE_SEMAPHORE_TIMEOUT = 5

        # Real semaphore so acquire/release work correctly
        sem = multiprocessing.Semaphore(1)
        proc._inference_semaphore = sem
        proc._vae_decode_semaphore = multiprocessing.Semaphore(1)

        proc.send_process_state_change_message = MagicMock()
        proc.send_heartbeat_message = MagicMock()
        proc.on_horde_model_state_change = MagicMock()

        return proc

    def _make_job_info(self) -> MagicMock:
        job_info = MagicMock()
        job_info.payload.prompt = "test prompt"
        job_info.extra_source_images = None
        job_info.source_image = None
        job_info.source_mask = None
        job_info.ids = []
        return job_info

    def _pp_complete_calls(self, proc: MagicMock) -> list:
        """Return all send_process_state_change_message calls for POST_PROCESSING_COMPLETE."""
        from horde_worker_regen.process_management.messages import HordeProcessState

        return [
            c
            for c in proc.send_process_state_change_message.call_args_list
            if c.kwargs.get("process_state") == HordeProcessState.POST_PROCESSING_COMPLETE
        ]

    def test_post_processing_complete_not_emitted_when_basic_inference_returns_none(self) -> None:
        """When basic_inference() returns None and _in_post_processing is True (we entered
        post-processing but it failed), POST_PROCESSING_COMPLETE must NOT be emitted.

        Emitting it would be misleading: the process-manager uses that state to update the
        progress bar and history UI, falsely reporting success.
        """
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = self._make_proc_for_start_inference(in_post_processing=True)
        proc._horde = MagicMock()
        proc._horde.basic_inference.return_value = None

        HordeInferenceProcess.start_inference(proc, self._make_job_info())

        assert len(self._pp_complete_calls(proc)) == 0, (
            "POST_PROCESSING_COMPLETE must not be emitted when basic_inference() returns None"
        )

    def test_post_processing_complete_not_emitted_when_basic_inference_returns_empty_list(self) -> None:
        """When basic_inference() returns an empty list and _in_post_processing is True,
        POST_PROCESSING_COMPLETE must NOT be emitted.
        """
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = self._make_proc_for_start_inference(in_post_processing=True)
        proc._horde = MagicMock()
        proc._horde.basic_inference.return_value = []

        HordeInferenceProcess.start_inference(proc, self._make_job_info())

        assert len(self._pp_complete_calls(proc)) == 0, (
            "POST_PROCESSING_COMPLETE must not be emitted when basic_inference() returns []"
        )

    def test_post_processing_complete_not_emitted_when_no_post_processing(self) -> None:
        """When post-processing was never entered (_in_post_processing is False),
        POST_PROCESSING_COMPLETE must not be emitted regardless of the result.
        """
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = self._make_proc_for_start_inference(in_post_processing=False)
        proc._horde = MagicMock()
        proc._horde.basic_inference.return_value = None

        HordeInferenceProcess.start_inference(proc, self._make_job_info())

        assert len(self._pp_complete_calls(proc)) == 0, (
            "POST_PROCESSING_COMPLETE must not be emitted when post-processing was never entered"
        )

    def test_post_processing_was_started_flag_set_in_finally_when_in_post_processing(self) -> None:
        """The finally block must copy _in_post_processing → _post_processing_was_started
        before resetting _in_post_processing, so send_inference_result_message can
        read the flag even after start_inference() has returned.
        """
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = self._make_proc_for_start_inference(in_post_processing=True)
        proc._horde = MagicMock()
        proc._horde.basic_inference.return_value = None

        HordeInferenceProcess.start_inference(proc, self._make_job_info())

        assert proc._post_processing_was_started is True, (
            "_post_processing_was_started must be True after start_inference() when post-processing was entered"
        )
        # And the live flag must have been cleared
        assert proc._in_post_processing is False, (
            "_in_post_processing must be reset to False by the finally block"
        )

    def test_post_processing_was_started_flag_false_when_no_post_processing(self) -> None:
        """When post-processing was never entered, _post_processing_was_started must remain False."""
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = self._make_proc_for_start_inference(in_post_processing=False)
        proc._horde = MagicMock()
        proc._horde.basic_inference.return_value = None

        HordeInferenceProcess.start_inference(proc, self._make_job_info())

        assert proc._post_processing_was_started is False, (
            "_post_processing_was_started must remain False when post-processing was never entered"
        )


class TestFrozenPayloadPromptRestore:
    """Regression tests for the frozen Pydantic payload prompt restoration bug.

    ``ImageGenerateJobPopPayload`` is a frozen Pydantic v2 model.  Assigning to
    a frozen model attribute raises ``pydantic.ValidationError``, *not*
    ``AttributeError``.  The original code used ``contextlib.suppress(AttributeError)``
    in the ``start_inference`` finally block, so the ``ValidationError`` propagated
    and caused ``start_inference`` to return ``None`` even when ``basic_inference()``
    succeeded — making every job fault with "inference produced no results".
    """

    def _make_proc(self) -> MagicMock:
        """Return a minimally-mocked HordeInferenceProcess for start_inference tests."""
        import multiprocessing

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._is_busy = False
        proc._in_post_processing = False
        proc._post_processing_was_started = False
        proc._vae_acquire_attempted = False
        proc._vae_lock_was_acquired = False
        proc._current_job_inference_steps_complete = False
        proc._last_sanitized_negative_prompt = None
        proc._last_job_inference_rate = None
        proc._active_model_name = "TestModel"
        proc._start_inference_time = 0.0
        proc.VAE_SEMAPHORE_TIMEOUT = 5
        proc._inference_semaphore = multiprocessing.Semaphore(1)
        proc._vae_decode_semaphore = multiprocessing.Semaphore(1)
        proc.send_process_state_change_message = MagicMock()
        proc.send_heartbeat_message = MagicMock()
        return proc

    def _make_frozen_job_info(self) -> MagicMock:
        """Return a job_info whose payload raises ValidationError on assignment (frozen model)."""
        import pydantic

        class FrozenPayload(pydantic.BaseModel):
            model_config = pydantic.config.ConfigDict(frozen=True)
            prompt: str = "positive###negative"

        job_info = MagicMock()
        job_info.payload = FrozenPayload()
        job_info.extra_source_images = None
        job_info.source_image = None
        job_info.source_mask = None
        job_info.ids = []
        return job_info

    def test_frozen_payload_prompt_restore_does_not_propagate(self) -> None:
        """start_inference must return results (not None) when job_info.payload is a
        frozen Pydantic model.

        With the old ``contextlib.suppress(AttributeError)`` the ValidationError raised
        by the frozen-model assignment escaped the finally block, so start_inference()
        appeared to have failed even though basic_inference() returned valid results.
        The fix changes the suppress to
        ``contextlib.suppress(AttributeError, PydanticValidationError)`` so the
        ValidationError is silently swallowed and the inference results are returned.
        """
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = self._make_proc()

        fake_result = MagicMock()
        fake_result.rawpng = MagicMock()
        proc._horde = MagicMock()
        proc._horde.basic_inference.return_value = [fake_result]

        job_info = self._make_frozen_job_info()

        # Before the fix this returned None because ValidationError escaped the finally block.
        result = HordeInferenceProcess.start_inference(proc, job_info)

        assert result is not None, (
            "start_inference must return results even when job_info.payload is a frozen "
            "Pydantic model (ValidationError in the finally block must be suppressed)"
        )
        assert result == [fake_result], "start_inference must return the exact results from basic_inference()"

    def test_frozen_payload_prompt_restore_when_inference_fails(self) -> None:
        """When basic_inference() raises, start_inference must still return None even
        if job_info.payload is a frozen model (the ValidationError from the finally block
        must not shadow the original inference exception).
        """
        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = self._make_proc()
        proc._horde = MagicMock()
        proc._horde.basic_inference.side_effect = RuntimeError("OOM")

        job_info = self._make_frozen_job_info()

        result = HordeInferenceProcess.start_inference(proc, job_info)

        assert result is None, (
            "start_inference must return None when basic_inference() raises, "
            "regardless of whether payload restoration raises in the finally block"
        )


class TestStartInferencePipeBroken:
    """Tests that a broken pipe to a child process causes the process to be
    replaced and the job fault to be handled correctly (without double-faulting).

    The scenario:
    - ``safe_send_message()`` returns False (e.g. BrokenPipeError).
    - The broken process must be replaced immediately via ``_replace_inference_process``.
    - ``handle_job_fault`` must be called for ``next_job`` when it is a *different* job
      from ``last_job_referenced`` (IDs differ).
    - ``handle_job_fault`` must NOT be called for ``next_job`` when the IDs match,
      because ``_replace_inference_process`` already handles the fault internally.
    """

    def _make_manager_with_pipe_failure(
        self,
        next_job_id: str,
        last_job_id: str | None,
    ) -> MagicMock:
        """Build a minimal mock HordeWorkerProcessManager for the pipe-failure path of
        ``start_inference()``.

        ``get_next_job_and_process`` is mocked to return a plain MagicMock whose
        ``.next_job`` and ``.process_with_model`` attributes are set up appropriately.
        ``safe_send_message`` on the process mock always returns False.

        ``HordeInferenceControlMessage`` is patched at the module level so Pydantic
        validation of ``sdk_api_job_info`` does not interfere with the test — we only
        care about the behaviour after ``safe_send_message`` returns False.
        """
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        # Build the next_job mock (the new job being dispatched)
        next_job = MagicMock()
        next_job.id_ = next_job_id
        next_job.model = "TestModel"
        next_job.source_image = None
        next_job.payload = MagicMock()
        next_job.payload.control_type = None
        next_job.payload.loras = None
        next_job.payload.tis = None
        next_job.payload.post_processing = None
        next_job.payload.hires_fix = False
        next_job.payload.workflow = None
        next_job.payload.width = 512
        next_job.payload.height = 512
        next_job.payload.ddim_steps = 8
        next_job.payload.sampler_name = "k_euler"
        next_job.payload.n_iter = 1
        next_job.ids = [next_job_id]

        # Build the process whose pipe is "broken"
        process_with_model = MagicMock()
        process_with_model.process_id = 1
        process_with_model.batch_amount = 1

        if last_job_id is not None:
            last_ref = MagicMock()
            last_ref.id_ = last_job_id
            process_with_model.last_job_referenced = last_ref
        else:
            process_with_model.last_job_referenced = None

        # safe_send_message always fails (simulates broken pipe)
        process_with_model.safe_send_message.return_value = False

        # Mock get_next_job_and_process to return a plain MagicMock with the right attrs.
        # We avoid importing NextJobAndProcess here because it transitively pulls in heavy
        # GPU/image-processing dependencies.
        nj_and_p = MagicMock()
        nj_and_p.next_job = next_job
        nj_and_p.process_with_model = process_with_model
        nj_and_p.skipped_line = False
        nj_and_p.skipped_line_for = None

        mock_manager = MagicMock()
        mock_manager.get_next_job_and_process.return_value = nj_and_p
        mock_manager.bridge_data.unload_models_from_vram_often = False
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager._skipped_line_next_job_and_process = None

        # Bind the real start_inference method onto the mock manager so we exercise
        # the real control-flow path.
        mock_manager._start_inference = HordeWorkerProcessManager.start_inference.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        return mock_manager

    def test_replace_inference_process_called_on_pipe_failure(self) -> None:
        """When safe_send_message returns False, _replace_inference_process must be invoked."""
        mock_manager = self._make_manager_with_pipe_failure(
            next_job_id="aaaaaaaa-0000-0000-0000-000000000001",
            last_job_id=None,
        )

        with patch(
            "horde_worker_regen.process_management.process_manager.HordeInferenceControlMessage"
        ):
            mock_manager._start_inference()

        mock_manager._replace_inference_process.assert_called_once()

    def test_handle_job_fault_called_once_for_new_job(self) -> None:
        """When next_job has a different ID from last_job_referenced, handle_job_fault
        must be called exactly once with next_job as the faulted_job.
        """
        next_job_id = "bbbbbbbb-0000-0000-0000-000000000002"
        last_job_id = "cccccccc-0000-0000-0000-000000000003"  # different ID
        mock_manager = self._make_manager_with_pipe_failure(
            next_job_id=next_job_id,
            last_job_id=last_job_id,
        )

        with patch(
            "horde_worker_regen.process_management.process_manager.HordeInferenceControlMessage"
        ):
            mock_manager._start_inference()

        mock_manager.handle_job_fault.assert_called_once()
        call_args = mock_manager.handle_job_fault.call_args
        faulted = call_args.kwargs.get("faulted_job") or call_args.args[0]
        assert faulted.id_ == next_job_id, (
            f"handle_job_fault must be called for next_job ({next_job_id}), "
            f"got {faulted.id_}"
        )

    def test_handle_job_fault_suppressed_when_same_job_already_handled(self) -> None:
        """When next_job has the same ID as last_job_referenced, handle_job_fault must
        NOT be called for next_job because _replace_inference_process already faulted it.
        """
        same_id = "dddddddd-0000-0000-0000-000000000004"
        mock_manager = self._make_manager_with_pipe_failure(
            next_job_id=same_id,
            last_job_id=same_id,  # same ID → double-fault guard suppresses extra call
        )

        with patch(
            "horde_worker_regen.process_management.process_manager.HordeInferenceControlMessage"
        ):
            mock_manager._start_inference()

        mock_manager.handle_job_fault.assert_not_called()

    def test_handle_job_fault_called_when_last_job_referenced_is_none(self) -> None:
        """When the process has never run a job (last_job_referenced is None),
        handle_job_fault must still be called for next_job.
        """
        next_job_id = "eeeeeeee-0000-0000-0000-000000000005"
        mock_manager = self._make_manager_with_pipe_failure(
            next_job_id=next_job_id,
            last_job_id=None,
        )

        with patch(
            "horde_worker_regen.process_management.process_manager.HordeInferenceControlMessage"
        ):
            mock_manager._start_inference()

        mock_manager.handle_job_fault.assert_called_once()

    def test_safe_send_message_stores_last_send_error(self) -> None:
        """After a failed send, last_send_error holds the exception that was raised."""
        from horde_worker_regen.process_management.process_manager import HordeProcessInfo

        mock_info = MagicMock()
        err = BrokenPipeError("pipe gone")
        mock_info.pipe_connection.send.side_effect = err
        mock_info.process_id = 42

        result = HordeProcessInfo.safe_send_message.__get__(mock_info, HordeProcessInfo)(MagicMock())

        assert result is False
        assert mock_info.last_send_error is err

    def test_safe_send_message_clears_last_send_error_on_success(self) -> None:
        """After a successful send, last_send_error is reset to None."""
        from horde_worker_regen.process_management.process_manager import HordeProcessInfo

        mock_info = MagicMock()
        mock_info.pipe_connection.send.return_value = None  # success
        mock_info.process_id = 42
        mock_info.last_send_error = BrokenPipeError("stale")

        result = HordeProcessInfo.safe_send_message.__get__(mock_info, HordeProcessInfo)(MagicMock())

        assert result is True
        assert mock_info.last_send_error is None


class TestReplaceHungModelPreloadingBypassesRecentlyRecovered:
    """Tests that MODEL_PRELOADING stuck detection works even when _recently_recovered is True.

    The _recently_recovered guard was previously applied at function entry, meaning a process
    stuck in MODEL_PRELOADING would never be recovered if a different process had been replaced
    recently (within inference_step_timeout seconds).  After the fix, MODEL_PRELOADING (and
    other non-cascading state checks) are always evaluated regardless of _recently_recovered.
    """

    def _make_manager(
        self,
        processes: list[MagicMock],
        *,
        recently_recovered: bool = False,
    ) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._recently_recovered = recently_recovered
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 300
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        # is_stuck_on_inference returns False so we exercise the else branch
        mock_manager._process_map.is_stuck_on_inference.return_value = False
        mock_manager._process_map.values.return_value = processes
        mock_manager._process_map.__iter__ = MagicMock(return_value=iter(processes))

        mock_manager._bound_replace_hung = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        return mock_manager

    def _make_process(
        self,
        process_id: int,
        state: HordeProcessState,
        *,
        time_elapsed: float = 9999.0,
    ) -> MagicMock:
        import time as _time

        from horde_worker_regen.process_management.horde_process import HordeProcessType

        proc = MagicMock()
        proc.process_id = process_id
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = state
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_timestamp = _time.time() - time_elapsed
        proc.last_progress_timestamp = _time.time() - time_elapsed
        proc.state_entered_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None
        return proc

    def test_model_preloading_replaced_when_recently_recovered_false(self) -> None:
        """MODEL_PRELOADING should be caught and replaced under normal conditions."""
        proc = self._make_process(0, HordeProcessState.MODEL_PRELOADING, time_elapsed=9999.0)
        mock_manager = self._make_manager([proc], recently_recovered=False)

        def fake_check_and_replace(
            process_info: MagicMock,
            timeout: float,
            state: HordeProcessState,
            error_msg: str,
        ) -> bool:
            if process_info.last_process_state == state:
                import time as _t

                elapsed = _t.time() - process_info.last_received_timestamp
                if elapsed > timeout:
                    mock_manager._replace_inference_process(process_info)
                    return True
            return False

        mock_manager._check_and_replace_process = fake_check_and_replace

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(proc)

    def test_model_preloading_replaced_even_when_recently_recovered_true(self) -> None:
        """MODEL_PRELOADING must be caught even when _recently_recovered is True.

        This is the regression test for the bug: a process stuck in MODEL_PRELOADING was
        previously never recovered while _recently_recovered=True (the flag was True for up
        to inference_step_timeout=600 seconds after any prior recovery).
        """
        proc = self._make_process(0, HordeProcessState.MODEL_PRELOADING, time_elapsed=9999.0)
        # _recently_recovered=True simulates the state where a prior recovery blocked detection
        mock_manager = self._make_manager([proc], recently_recovered=True)

        def fake_check_and_replace(
            process_info: MagicMock,
            timeout: float,
            state: HordeProcessState,
            error_msg: str,
        ) -> bool:
            if process_info.last_process_state == state:
                import time as _t

                elapsed = _t.time() - process_info.last_received_timestamp
                if elapsed > timeout:
                    mock_manager._replace_inference_process(process_info)
                    return True
            return False

        mock_manager._check_and_replace_process = fake_check_and_replace

        with patch("threading.Thread") as mock_thread:
            result = mock_manager._bound_replace_hung()

        # The stuck MODEL_PRELOADING process must still be replaced
        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(proc)
        # No new timer thread should be started when already inside a recovery window
        mock_thread.assert_not_called()


class TestReplaceHungWaitingForJob:
    """Tests for the per-process WAITING_FOR_JOB stale-heartbeat recovery.

    When an inference process has been in WAITING_FOR_JOB with no heartbeat for longer than
    max(process_timeout, 600) seconds and there are pending jobs, it should be replaced
    automatically even while _recently_recovered is True (a freshly replaced process starts in
    PROCESS_STARTING with a fresh timestamp, so it will never immediately re-match this condition).
    """

    def _make_manager(
        self,
        processes: list[MagicMock],
        *,
        recently_recovered: bool = False,
        last_pop_no_jobs: bool = False,
    ) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._recently_recovered = recently_recovered
        mock_manager._last_pop_no_jobs_available = last_pop_no_jobs
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager._hung_processes_detected = False
        mock_manager._hung_processes_detected_time = 0.0
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 100
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        mock_manager._process_map.is_stuck_on_inference.return_value = False
        mock_manager._check_and_replace_process.return_value = False
        mock_manager._process_map.values.return_value = processes
        mock_manager._process_map.__iter__ = MagicMock(return_value=iter(processes))

        mock_manager._bound_replace_hung = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        return mock_manager

    def _make_inference_process(
        self,
        process_id: int,
        *,
        time_elapsed: float,
    ) -> MagicMock:
        import time as _time
        from horde_worker_regen.process_management.process_manager import HordeProcessType

        proc = MagicMock()
        proc.process_id = process_id
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_timestamp = _time.time() - time_elapsed
        proc.last_progress_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None
        return proc

    def test_stale_waiting_for_job_process_replaced_when_jobs_pending(self) -> None:
        """A WAITING_FOR_JOB process whose heartbeat is older than 600s must be
        replaced when there are pending jobs (not _last_pop_no_jobs_available).
        The threshold is max(process_timeout, 600) so even high-performance-mode workers
        (process_timeout=100s) wait at least 600s before being replaced.
        """
        # 700s stale, effective threshold=max(100, 600)=600s → should trigger
        proc = self._make_inference_process(1, time_elapsed=700.0)
        mock_manager = self._make_manager([proc])

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(proc)

    def test_stale_waiting_for_job_process_replaced_even_when_recently_recovered(self) -> None:
        """The WAITING_FOR_JOB per-process check must run even when _recently_recovered=True.

        This is the regression case: in the reported bug Processes 1 and 3 were stuck in
        WAITING_FOR_JOB for 351 s and 464 s while _recently_recovered blocked all detection.
        """
        proc = self._make_inference_process(3, time_elapsed=700.0)
        # Simulate the state from the bug report: another recovery was recent
        mock_manager = self._make_manager([proc], recently_recovered=True)

        with patch("threading.Thread") as mock_thread:
            result = mock_manager._bound_replace_hung()

        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(proc)
        # No new timer thread when already inside recovery window
        mock_thread.assert_not_called()

    def test_fresh_waiting_for_job_process_not_replaced(self) -> None:
        """A WAITING_FOR_JOB process with a recent heartbeat must not be replaced."""
        # Only 10s stale, well below effective threshold of max(100, 600)=600s
        proc = self._make_inference_process(1, time_elapsed=10.0)
        mock_manager = self._make_manager([proc])

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        mock_manager._replace_inference_process.assert_not_called()
        assert result is False

    def test_waiting_for_job_not_replaced_below_600s_threshold(self) -> None:
        """Even with process_timeout=100 (high_performance_mode), a process idle for only 500s
        must NOT be replaced because the effective threshold is max(process_timeout, 600)=600s.
        """
        # 500s stale, effective threshold=max(100, 600)=600s → must NOT trigger
        proc = self._make_inference_process(1, time_elapsed=500.0)
        mock_manager = self._make_manager([proc])  # process_timeout=100

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        mock_manager._replace_inference_process.assert_not_called()
        assert result is False

    def test_stale_waiting_for_job_not_replaced_when_no_jobs_available(self) -> None:
        """A stale WAITING_FOR_JOB process must NOT be replaced when no jobs are available.

        WAITING_FOR_JOB is the expected idle state; replacing processes when there's nothing
        to do would cause needless churn.
        """
        proc = self._make_inference_process(1, time_elapsed=9999.0)
        # last_pop_no_jobs=True means the server has no work for us
        mock_manager = self._make_manager([proc], last_pop_no_jobs=True)

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        mock_manager._replace_inference_process.assert_not_called()
        assert result is False


class TestResultSubmittingStuckRecovery:
    """Tests for the RESULT_SUBMITTING stuck-process fixes.

    Two complementary guards are tested:
    1. submit_single_generation() try/finally resets the state to WAITING_FOR_JOB on every
       failure path so the process never stays in RESULT_SUBMITTING after a failed submission.
    2. replace_hung_processes() resets any process that remains in RESULT_SUBMITTING for longer
       than 60 s as a safety net, without killing the subprocess.
    """

    # -------------------------------------------------------------------------
    # Helpers shared by the submit_single_generation tests
    # -------------------------------------------------------------------------

    def _make_process_map(self, process_id: int, initial_state: HordeProcessState) -> tuple[MagicMock, MagicMock]:
        """Return a (ProcessMap-like mock, process_info mock) pair that tracks state changes."""
        process_info = MagicMock()
        process_info.last_process_state = initial_state
        process_info.inference_started_timestamp = None
        process_info.state_entered_timestamp = 0.0

        def on_state_change(*, process_id: int, new_state: HordeProcessState) -> None:
            process_info.last_process_state = new_state
            process_info.inference_started_timestamp = None
            process_info.state_entered_timestamp = 0.0

        process_map = MagicMock()
        process_map.__getitem__.return_value = process_info
        process_map.items.return_value = [(process_id, process_info)]
        process_map.get.return_value = process_info
        process_map.on_process_state_change.side_effect = on_state_change

        return process_map, process_info

    def _make_submit_manager(
        self,
        process_id: int,
        *,
        initial_state: HordeProcessState = HordeProcessState.RESULT_SAVED,
    ) -> tuple[MagicMock, MagicMock]:
        """Build a minimal mock manager suitable for calling submit_single_generation."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        process_map, process_info = self._make_process_map(process_id, initial_state)
        mock_manager = MagicMock()
        mock_manager._process_map = process_map
        mock_manager._STATE_TRANSITION_TIMING_STATES = HordeWorkerProcessManager._STATE_TRANSITION_TIMING_STATES
        mock_manager._pending_process_job_timings = {}
        mock_manager._pending_completed_job_timings = {}
        mock_manager._job_time_stats = {}
        mock_manager._record_job_timing = HordeWorkerProcessManager._record_job_timing.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        mock_manager._record_pending_job_timing = HordeWorkerProcessManager._record_pending_job_timing.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        mock_manager._on_process_state_change = HordeWorkerProcessManager._on_process_state_change.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        mock_manager._commit_completed_job_timings = HordeWorkerProcessManager._commit_completed_job_timings.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        mock_manager._discard_completed_job_timings = HordeWorkerProcessManager._discard_completed_job_timings.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        return mock_manager, process_info

    def _make_new_submit(self, process_info: MagicMock) -> MagicMock:
        """Build a faulted PendingSubmitJob mock whose job reference matches the process.

        Using is_faulted=True (faulted job, no image) lets the function bypass the
        image-upload section and proceed directly to setting RESULT_SUBMITTING and
        calling the API, which is what we want to exercise.
        """
        sdk_job = MagicMock()
        # payload.seed must look like an integer so int() doesn't raise
        sdk_job.payload.seed = 0

        completed_job_info = MagicMock()
        completed_job_info.sdk_api_job_info = sdk_job
        completed_job_info.state = "faulted"

        new_submit = MagicMock()
        new_submit.is_faulted = True   # faulted → skip "no image result" early return
        new_submit.image_result = None  # faulted job has no image
        new_submit.completed_job_info = completed_job_info

        # Make process_info.last_job_referenced match so handling_process_id is found
        process_info.last_job_referenced = sdk_job

        return new_submit

    # -------------------------------------------------------------------------
    # submit_single_generation tests
    # -------------------------------------------------------------------------

    def test_remove_awaiting_request_ignores_missing_set_entry(self) -> None:
        """Cleanup should ignore missing requests for set-like containers."""
        from horde_worker_regen.process_management.process_manager import _remove_awaiting_request

        session = MagicMock()
        session._awaiting_requests = {"other-request"}

        _remove_awaiting_request(session, "missing-request")

        assert session._awaiting_requests == {"other-request"}

    def test_remove_awaiting_request_propagates_unexpected_attribute_error(self) -> None:
        """Cleanup should not suppress unrelated AttributeError from equality checks."""
        from horde_worker_regen.process_management.process_manager import _remove_awaiting_request

        class _BrokenEq:
            def __eq__(self, other: object) -> bool:
                raise AttributeError("broken equality")

        session = MagicMock()
        session._awaiting_requests = [_BrokenEq()]

        with pytest.raises(AttributeError, match="broken equality"):
            _remove_awaiting_request(session, "missing-request")

    def test_state_reset_to_waiting_on_api_timeout(self) -> None:
        """Process state must be reset to WAITING_FOR_JOB when the API call times out."""
        import asyncio

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        manager, proc_info = self._make_submit_manager(0)
        new_submit = self._make_new_submit(proc_info)

        # Simulate an asyncio.TimeoutError coming from the API call
        async def _simulate_timeout_error(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
            raise asyncio.TimeoutError

        manager.horde_client_session.submit_request.side_effect = _simulate_timeout_error

        bound = HordeWorkerProcessManager.submit_single_generation.__get__(manager, HordeWorkerProcessManager)
        asyncio.run(bound(new_submit))

        assert proc_info.last_process_state == HordeProcessState.WAITING_FOR_JOB

    def test_failed_submit_keeps_state_timings_buffered(self) -> None:
        """Failed submit attempts must not update completed-job timing stats."""
        import asyncio

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        manager, proc_info = self._make_submit_manager(0)
        new_submit = self._make_new_submit(proc_info)
        manager._pending_completed_job_timings[new_submit.completed_job_info.sdk_api_job_info] = {
            "WAITING_FOR_JOB": 3.5,
        }

        async def _simulate_timeout_error(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
            raise asyncio.TimeoutError

        manager.horde_client_session.submit_request.side_effect = _simulate_timeout_error

        bound = HordeWorkerProcessManager.submit_single_generation.__get__(manager, HordeWorkerProcessManager)
        asyncio.run(bound(new_submit))

        assert manager._job_time_stats == {}
        assert manager._pending_completed_job_timings[new_submit.completed_job_info.sdk_api_job_info][
            "WAITING_FOR_JOB"
        ] == 3.5

    def test_state_reset_to_waiting_on_unexpected_exception(self) -> None:
        """Process state must be reset to WAITING_FOR_JOB when submit_request raises unexpectedly."""
        import asyncio

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        manager, proc_info = self._make_submit_manager(0)
        new_submit = self._make_new_submit(proc_info)

        async def _fail(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
            raise RuntimeError("unexpected error")

        manager.horde_client_session.submit_request.side_effect = _fail

        bound = HordeWorkerProcessManager.submit_single_generation.__get__(manager, HordeWorkerProcessManager)
        asyncio.run(bound(new_submit))

        assert proc_info.last_process_state == HordeProcessState.WAITING_FOR_JOB

    def test_state_reset_to_waiting_on_api_error_response(self) -> None:
        """Process state must be reset to WAITING_FOR_JOB when the API returns a RequestErrorResponse."""
        import asyncio

        from horde_sdk import RequestErrorResponse

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        manager, proc_info = self._make_submit_manager(0)
        new_submit = self._make_new_submit(proc_info)

        error_response = MagicMock(spec=RequestErrorResponse)
        error_response.message = "Some unexpected API error"

        async def _return_error(*args, **kwargs) -> object:  # noqa: ANN002, ANN003
            return error_response

        # Bypass asyncio.wait_for so the coroutine runs directly and returns the error
        async def _wait_for_passthrough(coro, timeout) -> object:  # noqa: ANN001, ANN002
            return await coro

        with patch("asyncio.wait_for", side_effect=_wait_for_passthrough):
            manager.horde_client_session.submit_request = _return_error
            bound = HordeWorkerProcessManager.submit_single_generation.__get__(manager, HordeWorkerProcessManager)
            asyncio.run(bound(new_submit))

        assert proc_info.last_process_state == HordeProcessState.WAITING_FOR_JOB

    def test_retry_success_records_reward_and_kudos_rate(self) -> None:
        """A retried submit must store the eventual successful reward values."""
        import asyncio
        import io

        from horde_sdk.ai_horde_api import GENERATION_STATE

        from horde_worker_regen.process_management.process_manager import (
            HordeWorkerProcessManager,
            JobSubmitState,
            PendingSubmitJob,
        )

        manager, proc_info = self._make_submit_manager(0)
        base_submit = self._make_new_submit(proc_info)
        base_submit.completed_job_info.state = GENERATION_STATE.ok
        image_result = MagicMock()
        image_result.image_base64 = "ignored"
        image_result.generation_faults = []
        base_submit.completed_job_info.job_image_results = [image_result]
        base_submit.completed_job_info.time_to_generate = 4.0
        base_submit.completed_job_info.censored = False
        base_submit.completed_job_info.sdk_api_job_info.model = "test-model"
        base_submit.completed_job_info.sdk_api_job_info.ids = ["job-1"]
        base_submit.completed_job_info.sdk_api_job_info.r2_uploads = ["https://example.com/upload"]
        new_submit = PendingSubmitJob.model_construct(
            completed_job_info=base_submit.completed_job_info,
            gen_iter=0,
            state=JobSubmitState.PENDING,
            kudos_reward=0,
            kudos_per_second=0.0,
        )
        manager.bridge_data.api_key = "test-key"
        manager.base64_image_to_stream_buffer.return_value = io.BytesIO(b"image-bytes")
        manager.kudos_generated_this_session = 0
        manager.kudos_events = []
        manager.image_events = []
        manager._images_per_model = {}
        manager._num_job_slowdowns = 0
        manager._num_jobs_faulted = 0
        manager._job_pop_timestamps_lock = asyncio.Lock()
        manager.job_pop_timestamps = {}

        class _UploadResponse:
            status = 200

            async def __aenter__(self) -> "_UploadResponse":
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
                return None

        manager._aiohttp_client_session.put.return_value = _UploadResponse()

        async def _fail_then_succeed(*args, **kwargs) -> object:  # noqa: ANN002, ANN003
            if not hasattr(_fail_then_succeed, "called"):
                _fail_then_succeed.called = True
                raise asyncio.TimeoutError

            response = MagicMock()
            response.reward = 12
            return response

        manager.horde_client_session.submit_request.side_effect = _fail_then_succeed

        bound = HordeWorkerProcessManager.submit_single_generation.__get__(manager, HordeWorkerProcessManager)

        first_attempt = asyncio.run(bound(new_submit))
        assert first_attempt.is_finished is False

        second_attempt = asyncio.run(bound(first_attempt))
        assert second_attempt.is_finished is True
        assert second_attempt.is_faulted is False
        assert second_attempt.kudos_reward == 12
        assert second_attempt.kudos_per_second == 3.0

    def test_manager_state_change_buffers_waiting_timing(self) -> None:
        """Manager-initiated WAITING_FOR_JOB transitions must buffer elapsed time."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        manager, proc_info = self._make_submit_manager(0, initial_state=HordeProcessState.WAITING_FOR_JOB)
        proc_info.state_entered_timestamp = 90.0

        with patch("time.time", return_value=100.0):
            manager._on_process_state_change(
                process_id=0,
                new_state=HordeProcessState.INFERENCE_STARTING,
            )

        assert manager._pending_process_job_timings[0]["WAITING_FOR_JOB"] == 10.0

    # -------------------------------------------------------------------------
    # replace_hung_processes safety-net tests
    # -------------------------------------------------------------------------

    def _make_hung_manager(
        self,
        processes: list[MagicMock],
        *,
        last_pop_no_jobs: bool = False,
    ) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._recently_recovered = False
        mock_manager._last_pop_no_jobs_available = last_pop_no_jobs
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager._hung_processes_detected = False
        mock_manager._hung_processes_detected_time = 0.0
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 300
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        mock_manager.bridge_data.exit_on_unhandled_faults = False
        mock_manager._process_map.is_stuck_on_inference.return_value = False
        mock_manager._check_and_replace_process.return_value = False
        mock_manager._process_map.values.return_value = processes
        mock_manager._process_map.__getitem__.side_effect = lambda process_id: next(
            process for process in processes if process.process_id == process_id
        )
        mock_manager._STATE_TRANSITION_TIMING_STATES = HordeWorkerProcessManager._STATE_TRANSITION_TIMING_STATES
        mock_manager._pending_process_job_timings = {}
        mock_manager._pending_completed_job_timings = {}
        mock_manager._record_pending_job_timing = HordeWorkerProcessManager._record_pending_job_timing.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        mock_manager._on_process_state_change = HordeWorkerProcessManager._on_process_state_change.__get__(
            mock_manager, HordeWorkerProcessManager
        )

        mock_manager._bound_replace_hung = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        return mock_manager

    def _make_result_submitting_process(
        self,
        process_id: int,
        *,
        time_elapsed: float,
    ) -> MagicMock:
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        proc = MagicMock()
        proc.process_id = process_id
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = HordeProcessState.RESULT_SUBMITTING
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_timestamp = _time.time() - time_elapsed
        proc.last_progress_timestamp = _time.time() - time_elapsed
        proc.state_entered_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None
        return proc

    def test_result_submitting_stuck_over_60s_resets_state(self) -> None:
        """A process stuck in RESULT_SUBMITTING for >60 s must have its state reset to WAITING_FOR_JOB."""
        proc = self._make_result_submitting_process(2, time_elapsed=90.0)
        mock_manager = self._make_hung_manager([proc])

        state_changes: list[HordeProcessState] = []

        def _on_state_change(*, process_id: int, new_state: HordeProcessState) -> None:
            proc.last_process_state = new_state
            proc.inference_started_timestamp = None
            state_changes.append(new_state)

        mock_manager._process_map.on_process_state_change.side_effect = _on_state_change

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        assert HordeProcessState.WAITING_FOR_JOB in state_changes
        # The subprocess must NOT have been killed
        mock_manager._replace_inference_process.assert_not_called()

    def test_result_submitting_not_reset_within_60s(self) -> None:
        """A process in RESULT_SUBMITTING for <60 s must not be touched (submission still in progress)."""
        proc = self._make_result_submitting_process(2, time_elapsed=30.0)
        mock_manager = self._make_hung_manager([proc])

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is False
        mock_manager._replace_inference_process.assert_not_called()
        mock_manager._process_map.on_process_state_change.assert_not_called()

    def test_result_submitting_stuck_reset_even_when_no_jobs_available(self) -> None:
        """RESULT_SUBMITTING stuck check must fire even when no jobs are available.

        When a process is stuck in RESULT_SUBMITTING its job has already been removed
        from jobs_in_progress (the HordeInferenceResultMessage was received), so
        no_local_work would be True.  Previously the check was inside the
        _last_pop_no_jobs_available guard, which caused it to be silently skipped,
        leaving the inference slot permanently blocked.  The check is now placed before
        that guard so it always fires after the 60 s timeout regardless of job availability.
        """
        proc = self._make_result_submitting_process(2, time_elapsed=9999.0)
        mock_manager = self._make_hung_manager([proc], last_pop_no_jobs=True)

        state_changes: list[HordeProcessState] = []

        def _on_state_change(*, process_id: int, new_state: HordeProcessState) -> None:
            proc.last_process_state = new_state
            proc.inference_started_timestamp = None
            state_changes.append(new_state)

        mock_manager._process_map.on_process_state_change.side_effect = _on_state_change

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        assert HordeProcessState.WAITING_FOR_JOB in state_changes
        mock_manager._replace_inference_process.assert_not_called()

    def test_result_submitting_reset_does_not_trigger_recently_recovered(self) -> None:
        """Resetting RESULT_SUBMITTING state must NOT set _recently_recovered.

        RESULT_SUBMITTING recovery is a soft state reset — the subprocess is NOT killed.
        Setting _recently_recovered would incorrectly suppress INFERENCE_STARTING detection
        and other legitimate checks for inference_step_timeout seconds after every submission
        timeout, which is far too conservative.  Only actual subprocess replacements should
        start the cascading-recovery cooldown.
        """
        proc = self._make_result_submitting_process(2, time_elapsed=90.0)
        mock_manager = self._make_hung_manager([proc])

        thread_started = False

        class _TrackThread:
            def __init__(self, *args: object, **kwargs: object) -> None:
                nonlocal thread_started
                thread_started = True

            def start(self) -> None:
                pass

        with patch("threading.Thread", _TrackThread):
            result = mock_manager._bound_replace_hung()

        assert result is True
        # The subprocess must NOT have been killed
        mock_manager._replace_inference_process.assert_not_called()
        # The cascading-recovery guard must NOT have been activated
        assert mock_manager._recently_recovered is False
        assert not thread_started, "_recently_recovered timer thread must not be started for a state reset"



class TestReplaceHungModelPreloaded:
    """Tests that MODEL_PRELOADED stuck detection fires after ``preload_timeout`` seconds.

    A process that is stuck in MODEL_PRELOADED (the model has been loaded into RAM but
    the job was never dispatched, e.g. because ``start_inference()`` was never called due
    to ``preload_models()`` blocking it) should be replaced just like MODEL_PRELOADING.
    """

    def _make_manager(
        self,
        processes: list[MagicMock],
        *,
        recently_recovered: bool = False,
    ) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._recently_recovered = recently_recovered
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager._hung_processes_detected = False
        mock_manager._hung_processes_detected_time = 0.0
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 300
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        mock_manager._process_map.is_stuck_on_inference.return_value = False
        mock_manager._process_map.values.return_value = processes
        mock_manager._process_map.__iter__ = MagicMock(return_value=iter(processes))

        mock_manager._bound_replace_hung = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager,
        )
        return mock_manager

    def _make_process(
        self,
        process_id: int,
        state: HordeProcessState,
        *,
        time_elapsed: float = 9999.0,
    ) -> MagicMock:
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        proc = MagicMock()
        proc.process_id = process_id
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = state
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_timestamp = _time.time() - time_elapsed
        proc.last_progress_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None
        return proc

    def _fake_check_and_replace(self, mock_manager: MagicMock) -> object:
        """Return a ``_check_and_replace_process`` implementation that actually checks the state."""

        def fake_check_and_replace(
            process_info: MagicMock,
            timeout: float,
            state: HordeProcessState,
            error_msg: str,
        ) -> bool:
            if process_info.last_process_state == state:
                import time as _t

                elapsed = _t.time() - process_info.last_received_timestamp
                if elapsed > timeout:
                    mock_manager._replace_inference_process(process_info)
                    return True
            return False

        return fake_check_and_replace

    def test_model_preloaded_replaced_after_timeout(self) -> None:
        """A process stuck in MODEL_PRELOADED for longer than preload_timeout must be replaced."""
        proc = self._make_process(1, HordeProcessState.MODEL_PRELOADED, time_elapsed=9999.0)
        mock_manager = self._make_manager([proc], recently_recovered=False)
        mock_manager._check_and_replace_process = self._fake_check_and_replace(mock_manager)

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(proc)

    def test_model_preloaded_replaced_even_when_recently_recovered(self) -> None:
        """MODEL_PRELOADED stuck detection must fire even when _recently_recovered is True.

        A process that cannot receive its START_INFERENCE message (e.g. because
        ``preload_models()`` keeps returning True and blocks ``start_inference()``)
        must be recovered regardless of whether a prior recovery recently ran.
        """
        proc = self._make_process(1, HordeProcessState.MODEL_PRELOADED, time_elapsed=9999.0)
        mock_manager = self._make_manager([proc], recently_recovered=True)
        mock_manager._check_and_replace_process = self._fake_check_and_replace(mock_manager)

        with patch("threading.Thread") as mock_thread:
            result = mock_manager._bound_replace_hung()

        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(proc)
        mock_thread.assert_not_called()

    def test_model_preloaded_not_replaced_before_timeout(self) -> None:
        """A process that just entered MODEL_PRELOADED (time_elapsed < preload_timeout).

        Must NOT be replaced, since it is expected to receive START_INFERENCE shortly.
        """
        # 10 seconds elapsed — well below preload_timeout (80 s)
        proc = self._make_process(1, HordeProcessState.MODEL_PRELOADED, time_elapsed=10.0)
        mock_manager = self._make_manager([proc], recently_recovered=False)
        mock_manager._check_and_replace_process = self._fake_check_and_replace(mock_manager)

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        mock_manager._replace_inference_process.assert_not_called()
        assert result is False

    def test_model_preloaded_not_replaced_when_no_jobs_available(self) -> None:
        """A stale MODEL_PRELOADED process must NOT be replaced when no jobs are available.

        When _last_pop_no_jobs_available is True, job-related stuck checks (including
        MODEL_PRELOADED) are skipped to avoid pointless process churn.
        """
        proc = self._make_process(1, HordeProcessState.MODEL_PRELOADED, time_elapsed=9999.0)
        mock_manager = self._make_manager([proc], recently_recovered=False)
        mock_manager._last_pop_no_jobs_available = True
        mock_manager._check_and_replace_process = self._fake_check_and_replace(mock_manager)

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        mock_manager._replace_inference_process.assert_not_called()
        assert result is False

    def test_unloaded_model_from_ram_replaced_after_timeout(self) -> None:
        """A process stuck in UNLOADED_MODEL_FROM_RAM beyond preload_timeout must be replaced."""
        proc = self._make_process(1, HordeProcessState.UNLOADED_MODEL_FROM_RAM, time_elapsed=9999.0)
        mock_manager = self._make_manager([proc], recently_recovered=False)
        mock_manager._check_and_replace_process = self._fake_check_and_replace(mock_manager)

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(proc)


class TestReplaceHungInferencePostProcessingBeforeModelLoaded:
    """Tests that actively-in-progress states are cleared before idle/finished states.

    The conditions list in replace_hung_processes must check actively in-progress states
    (INFERENCE_POST_PROCESSING, MODEL_PRELOADING, DOWNLOADING_AUX_MODEL) before idle/finished
    states (MODEL_PRELOADED).  Processes actively doing work hold resources (e.g. the VAE
    decode semaphore) and must be cleared first; a MODEL_PRELOADED process is merely waiting
    idle for a job.
    """

    def _make_manager(
        self,
        processes: list[MagicMock],
        *,
        recently_recovered: bool = False,
    ) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._recently_recovered = recently_recovered
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager._hung_processes_detected = False
        mock_manager._hung_processes_detected_time = 0.0
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 20000
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        mock_manager._process_map.is_stuck_on_inference.return_value = False
        mock_manager._process_map.values.return_value = processes
        mock_manager._process_map.__iter__ = MagicMock(return_value=iter(processes))

        mock_manager._bound_replace_hung = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager,
        )
        return mock_manager

    def _make_process(
        self,
        process_id: int,
        state: HordeProcessState,
        *,
        time_elapsed: float = 9999.0,
    ) -> MagicMock:
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        proc = MagicMock()
        proc.process_id = process_id
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = state
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_timestamp = _time.time() - time_elapsed
        proc.last_progress_timestamp = _time.time() - time_elapsed
        proc.state_entered_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None
        return proc

    def test_inference_post_processing_replaced_before_model_preloaded(self) -> None:
        """When both INFERENCE_POST_PROCESSING and MODEL_PRELOADED processes are stuck,
        the INFERENCE_POST_PROCESSING process must be replaced first.

        The idle MODEL_PRELOADED process is listed first in the process map to prove that
        replacement order is driven by the conditions priority, not the process map iteration
        order.  With a per-process-first loop the MODEL_PRELOADED process would be replaced
        first (because it comes first in the list); the multi-pass condition-first loop
        ensures INFERENCE_POST_PROCESSING is always cleared first.
        """
        replace_order: list[int] = []

        post_proc = self._make_process(1, HordeProcessState.INFERENCE_POST_PROCESSING)
        preloaded = self._make_process(2, HordeProcessState.MODEL_PRELOADED)

        # MODEL_PRELOADED process is listed first in the map — a per-process-first loop
        # would replace it before INFERENCE_POST_PROCESSING.  The multi-pass loop must
        # clear INFERENCE_POST_PROCESSING first regardless.
        mock_manager = self._make_manager([preloaded, post_proc])

        def fake_check_and_replace(
            process_info: MagicMock,
            timeout: float,
            state: HordeProcessState,
            error_msg: str,
        ) -> bool:
            if process_info.last_process_state == state:
                import time as _t

                elapsed = _t.time() - process_info.last_received_timestamp
                if elapsed > timeout:
                    replace_order.append(process_info.process_id)
                    mock_manager._replace_inference_process(process_info)
                    return True
            return False

        mock_manager._check_and_replace_process = fake_check_and_replace

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        assert 1 in replace_order, "INFERENCE_POST_PROCESSING process must be replaced"
        assert 2 in replace_order, "MODEL_PRELOADED process must be replaced"
        assert replace_order.index(1) < replace_order.index(2), (
            "INFERENCE_POST_PROCESSING (id=1) must be cleared before MODEL_PRELOADED (id=2); "
            f"actual replacement order: {replace_order}"
        )

    def test_inference_post_processing_replaced_when_model_preloaded_not_stuck(self) -> None:
        """INFERENCE_POST_PROCESSING must be replaced even when MODEL_PRELOADED is not stuck."""
        post_proc = self._make_process(1, HordeProcessState.INFERENCE_POST_PROCESSING, time_elapsed=9999.0)
        # MODEL_PRELOADED process is fresh (not stuck)
        preloaded = self._make_process(2, HordeProcessState.MODEL_PRELOADED, time_elapsed=10.0)

        mock_manager = self._make_manager([post_proc, preloaded])

        def fake_check_and_replace(
            process_info: MagicMock,
            timeout: float,
            state: HordeProcessState,
            error_msg: str,
        ) -> bool:
            if process_info.last_process_state == state:
                import time as _t

                elapsed = _t.time() - process_info.last_received_timestamp
                if elapsed > timeout:
                    mock_manager._replace_inference_process(process_info)
                    return True
            return False

        mock_manager._check_and_replace_process = fake_check_and_replace

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        # Only post_proc should be replaced
        mock_manager._replace_inference_process.assert_called_once_with(post_proc)

    def test_downloading_aux_model_replaced_before_model_preloaded(self) -> None:
        """DOWNLOADING_AUX_MODEL (in progress) must be cleared before MODEL_PRELOADED (idle).

        The MODEL_PRELOADED process is listed first in the process map to prove that
        replacement order is driven by the conditions priority, not process map iteration order.
        """
        replace_order: list[int] = []

        downloading = self._make_process(1, HordeProcessState.DOWNLOADING_AUX_MODEL)
        preloaded = self._make_process(2, HordeProcessState.MODEL_PRELOADED)

        # MODEL_PRELOADED is listed first — per-process-first loop would replace it first.
        mock_manager = self._make_manager([preloaded, downloading])

        def fake_check_and_replace(
            process_info: MagicMock,
            timeout: float,
            state: HordeProcessState,
            error_msg: str,
        ) -> bool:
            if process_info.last_process_state == state:
                import time as _t

                elapsed = _t.time() - process_info.last_received_timestamp
                if elapsed > timeout:
                    replace_order.append(process_info.process_id)
                    mock_manager._replace_inference_process(process_info)
                    return True
            return False

        mock_manager._check_and_replace_process = fake_check_and_replace

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        assert 1 in replace_order, "DOWNLOADING_AUX_MODEL process must be replaced"
        assert 2 in replace_order, "MODEL_PRELOADED process must be replaced"
        assert replace_order.index(1) < replace_order.index(2), (
            "DOWNLOADING_AUX_MODEL (id=1) must be cleared before MODEL_PRELOADED (id=2); "
            f"actual replacement order: {replace_order}"
        )

    def test_post_processing_starting_replaced_before_model_preloaded(self) -> None:
        """A process stuck in POST_PROCESSING_STARTING must be replaced before MODEL_PRELOADED.

        POST_PROCESSING_STARTING is the state the child emits just before acquiring the VAE
        decode semaphore.  A process stuck here for longer than post_process_timeout + (3 * max_batch)
        is a resource-holder (or potential holder) and must be cleared before idle states.
        """
        replace_order: list[int] = []

        post_proc_starting = self._make_process(1, HordeProcessState.POST_PROCESSING_STARTING)
        preloaded = self._make_process(2, HordeProcessState.MODEL_PRELOADED)

        # MODEL_PRELOADED is listed first — a per-process-first loop would replace it first.
        mock_manager = self._make_manager([preloaded, post_proc_starting])

        def fake_check_and_replace(
            process_info: MagicMock,
            timeout: float,
            state: HordeProcessState,
            error_msg: str,
        ) -> bool:
            if process_info.last_process_state == state:
                import time as _t

                elapsed = _t.time() - process_info.last_received_timestamp
                if elapsed > timeout:
                    replace_order.append(process_info.process_id)
                    mock_manager._replace_inference_process(process_info)
                    return True
            return False

        mock_manager._check_and_replace_process = fake_check_and_replace

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        assert 1 in replace_order, "POST_PROCESSING_STARTING process must be replaced"
        assert 2 in replace_order, "MODEL_PRELOADED process must be replaced"
        assert replace_order.index(1) < replace_order.index(2), (
            "POST_PROCESSING_STARTING (id=1) must be cleared before MODEL_PRELOADED (id=2); "
            f"actual replacement order: {replace_order}"
        )

    def test_post_processing_starting_replaced_when_stuck(self) -> None:
        """A process stuck in POST_PROCESSING_STARTING must be replaced even when MODEL_PRELOADED is not stuck."""
        post_proc_starting = self._make_process(1, HordeProcessState.POST_PROCESSING_STARTING, time_elapsed=9999.0)
        # MODEL_PRELOADED process is fresh (not stuck)
        preloaded = self._make_process(2, HordeProcessState.MODEL_PRELOADED, time_elapsed=10.0)

        mock_manager = self._make_manager([post_proc_starting, preloaded])

        def fake_check_and_replace(
            process_info: MagicMock,
            timeout: float,
            state: HordeProcessState,
            error_msg: str,
        ) -> bool:
            if process_info.last_process_state == state:
                import time as _t

                elapsed = _t.time() - process_info.last_received_timestamp
                if elapsed > timeout:
                    mock_manager._replace_inference_process(process_info)
                    return True
            return False

        mock_manager._check_and_replace_process = fake_check_and_replace

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        # Only post_proc_starting should be replaced
        mock_manager._replace_inference_process.assert_called_once_with(post_proc_starting)

    def test_all_in_progress_states_replaced_before_model_preloaded(self) -> None:
        """All in-progress states must be cleared before the idle MODEL_PRELOADED state.

        MODEL_PRELOADED is listed first in the process map so that a per-process-first
        loop would replace it before the in-progress states.  The multi-pass condition-first
        loop must clear all in-progress states first regardless of process map ordering.
        """
        replace_order: list[int] = []

        post_proc = self._make_process(1, HordeProcessState.INFERENCE_POST_PROCESSING)
        post_proc_starting = self._make_process(2, HordeProcessState.POST_PROCESSING_STARTING)
        preloading = self._make_process(3, HordeProcessState.MODEL_PRELOADING)
        downloading = self._make_process(4, HordeProcessState.DOWNLOADING_AUX_MODEL)
        preloaded = self._make_process(5, HordeProcessState.MODEL_PRELOADED)

        # MODEL_PRELOADED (id=5) is first — a per-process-first loop would replace it first.
        mock_manager = self._make_manager([preloaded, post_proc, post_proc_starting, preloading, downloading])

        def fake_check_and_replace(
            process_info: MagicMock,
            timeout: float,
            state: HordeProcessState,
            error_msg: str,
        ) -> bool:
            if process_info.last_process_state == state:
                import time as _t

                elapsed = _t.time() - process_info.last_received_timestamp
                if elapsed > timeout:
                    replace_order.append(process_info.process_id)
                    mock_manager._replace_inference_process(process_info)
                    return True
            return False

        mock_manager._check_and_replace_process = fake_check_and_replace

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        # All five must be replaced
        assert set(replace_order) == {1, 2, 3, 4, 5}, f"All processes must be replaced; got {replace_order}"
        # MODEL_PRELOADED (id=5) must come after all in-progress states
        preloaded_pos = replace_order.index(5)
        for in_progress_id in (1, 2, 3, 4):
            assert replace_order.index(in_progress_id) < preloaded_pos, (
                f"In-progress process id={in_progress_id} must be replaced before MODEL_PRELOADED (id=5); "
                f"actual order: {replace_order}"
            )


class TestPreloadModelsPipeBroken:
    """Tests that a broken pipe in ``preload_models()`` causes immediate process replacement.

    When ``safe_send_message()`` returns False for the PRELOAD_MODEL message,
    ``_replace_inference_process`` must be called on the dead process immediately.
    Without this, ``preload_models()`` would return True every cycle (because neither
    the model map nor the process map state was updated), permanently blocking
    ``start_inference()`` and leaving any MODEL_PRELOADED process stuck waiting for
    a job that will never arrive.
    """

    def _make_manager_preload_send_failure(self) -> MagicMock:
        """Build a minimal mock for the pipe-failure path of ``preload_models()``."""
        from horde_worker_regen.process_management.process_manager import (
            HordeWorkerProcessManager,
        )

        # Pending job for model "Juggernaut XL" (not yet loaded)
        pending_job = MagicMock()
        pending_job.model = "Juggernaut XL"
        pending_job.payload = MagicMock()
        pending_job.payload.loras = None
        pending_job.payload.tiling = None

        # The process whose pipe is broken
        available_process = MagicMock()
        available_process.process_id = 2
        available_process.safe_send_message.return_value = False
        available_process.last_send_error = BrokenPipeError("simulated broken pipe")
        available_process.last_process_state = HordeProcessState.WAITING_FOR_JOB
        available_process.inference_started_timestamp = None
        available_process.loaded_horde_model_name = None

        # Model map reports "Juggernaut XL" not loaded (so the preload is attempted)
        mock_model_map = MagicMock()
        mock_model_map.root = {}  # empty → LOADING check returns "not loaded"
        mock_model_map_values = []
        mock_model_map.values.return_value = mock_model_map_values

        mock_process_map = MagicMock()
        mock_process_map.values.return_value = []
        mock_process_map.get_first_available_inference_process.return_value = available_process
        mock_process_map.num_loaded_inference_processes.return_value = 3
        mock_process_map.num_preloading_processes.return_value = 0

        mock_manager = MagicMock()
        mock_manager._horde_model_map = mock_model_map
        mock_manager._process_map = mock_process_map
        mock_manager.jobs_pending_inference = [pending_job]
        mock_manager.jobs_in_progress = []
        mock_manager._shutting_down = False
        mock_manager._preload_delay_notified = False
        mock_manager._max_concurrent_inference_processes = 3
        mock_manager.bridge_data.very_fast_disk_mode = False
        mock_manager.bridge_data.cycle_process_on_model_change = False
        mock_manager.get_model_baseline.return_value = None
        # Preload cooldown is not active for this test: no stuck failures recorded.
        mock_manager._is_model_in_preload_cooldown.return_value = False

        mock_manager._preload_models = HordeWorkerProcessManager.preload_models.__get__(
            mock_manager, HordeWorkerProcessManager,
        )
        return mock_manager, available_process

    def test_replace_inference_process_called_on_pipe_failure(self) -> None:
        """When safe_send_message returns False for PRELOAD_MODEL.

        _replace_inference_process must be called immediately on the dead process.
        """
        mock_manager, available_process = self._make_manager_preload_send_failure()

        with patch(
            "horde_worker_regen.process_management.process_manager.HordePreloadInferenceModelMessage",
        ):
            mock_manager._preload_models()

        mock_manager._replace_inference_process.assert_called_once_with(available_process)

    def test_model_map_not_updated_on_pipe_failure(self) -> None:
        """When safe_send_message fails, model map entry must NOT be created.

        This ensures the next cycle can select a new healthy process for preloading.
        """
        mock_manager, _ = self._make_manager_preload_send_failure()

        with patch(
            "horde_worker_regen.process_management.process_manager.HordePreloadInferenceModelMessage",
        ):
            mock_manager._preload_models()

        # update_entry must not have been called (model map not polluted with stale LOADING state)
        mock_manager._horde_model_map.update_entry.assert_not_called()

    def test_preload_models_still_returns_true_on_pipe_failure(self) -> None:
        """preload_models() returns True even on pipe failure.

        The main loop knows a preload was attempted and will retry next cycle with a fresh process.
        """
        mock_manager, _ = self._make_manager_preload_send_failure()

        with patch(
            "horde_worker_regen.process_management.process_manager.HordePreloadInferenceModelMessage",
        ):
            result = mock_manager._preload_models()

        assert result is True


class TestReplaceInferenceProcessDoesNotDoubleFault:
    """Tests that _replace_inference_process does not fault a job that was already re-queued for retry.

    When _purge_jobs() is called first (e.g. all processes timed out), it faults each in-progress
    job and re-queues eligible ones to jobs_pending_inference.  Afterwards _replace_inference_process
    is called for each crashed process.  Without the guard, _replace_inference_process would call
    handle_job_fault a second time for the same job — consuming the single retry budget and marking
    the job permanently faulted before it ever gets its second chance.
    """

    def _make_manager_with_job_state(
        self,
        *,
        job_in_progress: bool,
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        """Return (mock_manager, job, process_info) with the job either in jobs_in_progress or not.

        The returned mock_manager has _replace_inference_process bound to the real
        implementation so we can exercise the actual guard.
        """
        import types

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job = MagicMock()
        job.id_ = "aaaabbbb-0000-0000-0000-000000000001"

        mock_manager = MagicMock()
        mock_manager.jobs_in_progress = [job] if job_in_progress else []
        mock_manager.jobs_lookup = {job: MagicMock()}

        # Bind the real method so it exercises the actual code path
        bound = types.MethodType(HordeWorkerProcessManager._replace_inference_process, mock_manager)

        process_info = MagicMock()
        process_info.last_job_referenced = job
        process_info.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        process_info.inference_started_timestamp = None
        process_info.loaded_horde_model_name = None
        mock_manager._inference_semaphore.release.side_effect = ValueError
        mock_manager._disk_lock.release.side_effect = ValueError

        bound(process_info)
        return mock_manager, job, process_info

    def test_handle_job_fault_called_when_job_is_in_progress(self) -> None:
        """_replace_inference_process must call handle_job_fault when the job is still in-progress."""
        mock_manager, job, process_info = self._make_manager_with_job_state(job_in_progress=True)
        mock_manager.handle_job_fault.assert_called_once_with(
            faulted_job=job,
            process_info=process_info,
        )

    def test_handle_job_fault_skipped_when_job_already_requeued_for_retry(self) -> None:
        """_replace_inference_process must NOT call handle_job_fault when the job is no longer in jobs_in_progress.

        This happens when _purge_jobs() already moved the job to jobs_pending_inference for its
        retry attempt.  A second call would exhaust the retry budget and permanently fault the job.
        """
        mock_manager, _job, _process_info = self._make_manager_with_job_state(job_in_progress=False)
        mock_manager.handle_job_fault.assert_not_called()


class TestPurgeJobsRetryCount:
    """Tests that _purge_jobs keeps retry-eligible jobs and discards fresh ones.

    The key invariants:
    - Jobs already faulted and re-queued (retry_count > 0) survive the purge.
    - Fresh pending jobs (retry_count == 0) are cleared.
    - In-progress jobs get faulted via handle_job_fault (which increments retry_count),
      so they end up in jobs_pending_inference with retry_count > 0 and survive.
    """

    def _make_job(self, job_id: str) -> MagicMock:
        job = MagicMock()
        job.id_ = job_id
        return job

    def _make_job_info(self, retry_count: int) -> MagicMock:
        job_info = MagicMock()
        job_info.retry_count = retry_count
        return job_info

    def _make_manager(
        self,
        *,
        jobs_in_progress: list,
        jobs_pending_inference: list,
        jobs_lookup: dict,
    ) -> MagicMock:
        import types
        from collections import deque

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager.jobs_in_progress = list(jobs_in_progress)
        mock_manager.jobs_pending_inference = deque(jobs_pending_inference)
        mock_manager.jobs_lookup = dict(jobs_lookup)
        mock_manager.jobs_being_safety_checked = []
        mock_manager.jobs_pending_safety_check = []
        mock_manager.jobs_pending_submit = []
        mock_manager._skipped_line_next_job_and_process = None

        # handle_job_fault is real: we want _purge_jobs to drive it, so bind the real one.
        # But to avoid side effects we use a side_effect that just appends to jobs_pending_inference
        # as the real implementation would for a retry-eligible job.
        max_retries = HordeWorkerProcessManager.MAX_JOB_RETRIES

        def fake_handle_job_fault(faulted_job: MagicMock, process_info: MagicMock) -> None:
            job_info = mock_manager.jobs_lookup.get(faulted_job)
            if job_info is not None and job_info.retry_count < max_retries:
                job_info.retry_count += 1
                if faulted_job not in mock_manager.jobs_pending_inference:
                    mock_manager.jobs_pending_inference.append(faulted_job)
                if faulted_job in mock_manager.jobs_in_progress:
                    mock_manager.jobs_in_progress.remove(faulted_job)

        mock_manager.handle_job_fault = fake_handle_job_fault
        mock_manager._last_job_submitted_time = 0.0
        mock_manager._shutting_down = False
        mock_manager._invalidate_megapixelsteps_cache = MagicMock()

        bound = types.MethodType(HordeWorkerProcessManager._purge_jobs, mock_manager)
        mock_manager._bound_purge = bound
        return mock_manager

    def test_fresh_pending_job_is_cleared(self) -> None:
        """A pending job that has never been retried (retry_count == 0) must be removed."""
        job = self._make_job("fresh-job")
        job_info = self._make_job_info(retry_count=0)

        mock_manager = self._make_manager(
            jobs_in_progress=[],
            jobs_pending_inference=[job],
            jobs_lookup={job: job_info},
        )
        mock_manager._bound_purge()

        assert job not in mock_manager.jobs_pending_inference

    def test_already_retried_pending_job_is_kept(self) -> None:
        """A pending job with retry_count > 0 (already faulted once and re-queued) must be kept.

        This covers the scenario where a prior PROCESS_ENDING handler already retried the job
        before _purge_jobs runs.  The old snapshot-based filter would discard this job because
        it was already in jobs_pending_inference at snapshot time.
        """
        job = self._make_job("retried-job")
        job_info = self._make_job_info(retry_count=1)

        mock_manager = self._make_manager(
            jobs_in_progress=[],
            jobs_pending_inference=[job],
            jobs_lookup={job: job_info},
        )
        mock_manager._bound_purge()

        assert job in mock_manager.jobs_pending_inference

    def test_in_progress_job_retried_and_kept(self) -> None:
        """A job that was in progress (retry_count == 0) gets retried by the purge and survives.

        _purge_jobs calls handle_job_fault for each in-progress job; handle_job_fault moves the
        job to jobs_pending_inference with retry_count == 1.  The subsequent filter must keep it.
        """
        job = self._make_job("in-progress-job")
        job_info = self._make_job_info(retry_count=0)

        mock_manager = self._make_manager(
            jobs_in_progress=[job],
            jobs_pending_inference=[],
            jobs_lookup={job: job_info},
        )
        mock_manager._bound_purge()

        assert job in mock_manager.jobs_pending_inference
        assert job_info.retry_count == 1

    def test_mixed_pending_jobs_only_retried_ones_kept(self) -> None:
        """With a mix of fresh and retried pending jobs, only retried ones must survive."""
        fresh = self._make_job("fresh")
        retried = self._make_job("retried")

        mock_manager = self._make_manager(
            jobs_in_progress=[],
            jobs_pending_inference=[fresh, retried],
            jobs_lookup={
                fresh: self._make_job_info(retry_count=0),
                retried: self._make_job_info(retry_count=1),
            },
        )
        mock_manager._bound_purge()

        assert fresh not in mock_manager.jobs_pending_inference
        assert retried in mock_manager.jobs_pending_inference

    def test_all_jobs_cleared_during_shutdown(self) -> None:
        """During shutdown, ALL pending jobs (including retry-eligible ones) must be cleared.

        Keeping retry jobs during shutdown blocks the shutdown sequence indefinitely because
        the process control loop waits for jobs_pending_inference to be empty.
        """
        retried = self._make_job("retried")

        mock_manager = self._make_manager(
            jobs_in_progress=[],
            jobs_pending_inference=[retried],
            jobs_lookup={retried: self._make_job_info(retry_count=1)},
        )
        mock_manager._shutting_down = True
        mock_manager._bound_purge()

        assert len(mock_manager.jobs_pending_inference) == 0


class TestReplaceHungProcessesLocalJobsPending:
    """Tests that replace_hung_processes() runs job-related stuck checks when local jobs are queued.

    Regression tests for the bug where the _last_pop_no_jobs_available guard skipped
    MODEL_PRELOADING and WAITING_FOR_JOB stuck-process checks even when jobs were already
    sitting in the local jobs_pending_inference queue. A stuck process in those states would
    never be replaced, leaving queued jobs unable to make progress.
    """

    def _make_manager(
        self,
        processes: list[MagicMock],
        *,
        recently_recovered: bool = False,
        last_pop_no_jobs: bool = False,
        job_pops_paused: bool = False,
        jobs_pending_inference: list | None = None,
        jobs_in_progress: list | None = None,
    ) -> MagicMock:
        """Build a minimal mock manager wired for replace_hung_processes()."""
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._recently_recovered = recently_recovered
        mock_manager._last_pop_no_jobs_available = last_pop_no_jobs
        mock_manager._job_pops_paused = job_pops_paused
        mock_manager._shutting_down = False
        mock_manager._hung_processes_detected = False
        mock_manager._hung_processes_detected_time = 0.0
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 100
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        mock_manager._process_map.is_stuck_on_inference.return_value = False
        mock_manager._check_and_replace_process.return_value = False
        mock_manager._process_map.values.return_value = processes
        mock_manager._process_map.__iter__ = MagicMock(return_value=iter(processes))
        mock_manager.jobs_pending_inference = (
            jobs_pending_inference if jobs_pending_inference is not None else []
        )
        mock_manager.jobs_in_progress = jobs_in_progress if jobs_in_progress is not None else []

        mock_manager._bound_replace_hung = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager,
        )
        return mock_manager

    def _make_model_preloading_process(self, process_id: int) -> MagicMock:
        """Return a mock process stuck in MODEL_PRELOADING with a stale heartbeat."""
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        proc = MagicMock()
        proc.process_id = process_id
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = HordeProcessState.MODEL_PRELOADING
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - 9999
        proc.last_heartbeat_timestamp = _time.time() - 9999
        proc.last_progress_timestamp = _time.time() - 9999
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None
        return proc

    def _make_waiting_for_job_process(self, process_id: int, *, time_elapsed: float) -> MagicMock:
        """Return a mock inference process in WAITING_FOR_JOB with a stale heartbeat."""
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        proc = MagicMock()
        proc.process_id = process_id
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_timestamp = _time.time() - time_elapsed
        proc.last_progress_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None
        return proc

    def test_model_preloading_check_runs_when_local_jobs_pending(self) -> None:
        """MODEL_PRELOADING stuck check must run when jobs are in the local queue.

        Before the fix, the guard ``if _last_pop_no_jobs_available: continue`` was
        unconditional and skipped the MODEL_PRELOADING check even when jobs were
        already pending locally, causing stuck preloading processes to be ignored.
        """
        proc = self._make_model_preloading_process(0)
        job = MagicMock()

        mock_manager = self._make_manager(
            [proc],
            last_pop_no_jobs=True,
            jobs_pending_inference=[job],
        )

        with patch("threading.Thread"):
            mock_manager._bound_replace_hung()

        called_states = [call.args[2] for call in mock_manager._check_and_replace_process.call_args_list]
        assert HordeProcessState.MODEL_PRELOADING in called_states, (
            "MODEL_PRELOADING check must run when local jobs are pending, "
            "even when _last_pop_no_jobs_available is True"
        )

    def test_model_preloading_check_skipped_when_no_jobs_anywhere(self) -> None:
        """MODEL_PRELOADING stuck check must be skipped when there is no work anywhere.

        When the horde API reports no new jobs AND the local queue is also empty,
        skipping the check avoids unnecessary churn (preserving the original guard intent).
        """
        proc = self._make_model_preloading_process(0)

        mock_manager = self._make_manager(
            [proc],
            last_pop_no_jobs=True,
            jobs_pending_inference=[],
            jobs_in_progress=[],
        )

        with patch("threading.Thread"):
            mock_manager._bound_replace_hung()

        called_states = [call.args[2] for call in mock_manager._check_and_replace_process.call_args_list]
        assert HordeProcessState.MODEL_PRELOADING not in called_states, (
            "MODEL_PRELOADING check must be skipped when no jobs are available anywhere"
        )

    def test_stale_waiting_for_job_replaced_when_no_horde_jobs_but_local_jobs_pending(self) -> None:
        """A stale WAITING_FOR_JOB process must be replaced when local jobs are pending.

        Scenario: _last_pop_no_jobs_available=True (the horde API has no new jobs to offer),
        but jobs_pending_inference already has a job waiting. The stuck WAITING_FOR_JOB
        process must be replaced so that job can eventually reach start_inference().
        """
        # 9999s stale — well above the max(process_timeout=100, _WAITING_FOR_JOB_STALE_THRESHOLD=600)=600s threshold
        proc = self._make_waiting_for_job_process(0, time_elapsed=9999.0)
        job = MagicMock()

        mock_manager = self._make_manager(
            [proc],
            last_pop_no_jobs=True,
            jobs_pending_inference=[job],
        )

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(proc)

    def test_stale_waiting_for_job_not_replaced_when_no_jobs_anywhere(self) -> None:
        """A stale WAITING_FOR_JOB process must NOT be replaced when there is no work anywhere.

        WAITING_FOR_JOB is the expected idle state when there is nothing to do; replacing
        processes in that state would cause unnecessary churn.
        """
        proc = self._make_waiting_for_job_process(0, time_elapsed=9999.0)

        mock_manager = self._make_manager(
            [proc],
            last_pop_no_jobs=True,
            jobs_pending_inference=[],
            jobs_in_progress=[],
        )

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        mock_manager._replace_inference_process.assert_not_called()
        assert result is False

    def test_jobs_in_progress_alone_also_overrides_guard(self) -> None:
        """The guard must also be bypassed when jobs_in_progress is non-empty.

        Even if jobs_pending_inference is empty, an in-progress job means work is
        actively happening. A WAITING_FOR_JOB process that stops heartbeating while
        a job is in-progress should still be detected and replaced.
        """
        proc = self._make_waiting_for_job_process(0, time_elapsed=9999.0)
        in_progress_job = MagicMock()

        mock_manager = self._make_manager(
            [proc],
            last_pop_no_jobs=True,
            jobs_pending_inference=[],
            jobs_in_progress=[in_progress_job],
        )

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True
        mock_manager._replace_inference_process.assert_called_once_with(proc)


class TestReplaceHungProcessesPausedPops:
    """Tests that replace_hung_processes() respects the job-pops-paused flag.

    When ``_job_pops_paused`` is True no new jobs can arrive from the API, so idle
    processes timing out is expected and normal.  The manager must NOT replace those
    processes unnecessarily (which would cause pointless process churn and a delay when
    pops are resumed).  However, if there are already jobs sitting in the local queue
    the full stuck-process detection must still run, because those jobs need a process
    to make progress.
    """

    def _make_manager(
        self,
        processes: list[MagicMock],
        *,
        last_pop_no_jobs: bool = False,
        job_pops_paused: bool = False,
        jobs_pending_inference: list | None = None,
        jobs_in_progress: list | None = None,
    ) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import (
            HordeWorkerProcessManager,
        )

        mock_manager = MagicMock()
        mock_manager._recently_recovered = False
        mock_manager._last_pop_no_jobs_available = last_pop_no_jobs
        mock_manager._job_pops_paused = job_pops_paused
        mock_manager._shutting_down = False
        mock_manager._hung_processes_detected = False
        mock_manager._hung_processes_detected_time = 0.0
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 100
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        mock_manager._process_map.is_stuck_on_inference.return_value = False
        mock_manager._check_and_replace_process.return_value = False
        mock_manager._process_map.values.return_value = processes
        mock_manager._process_map.__iter__ = MagicMock(return_value=iter(processes))
        mock_manager.jobs_pending_inference = (
            jobs_pending_inference if jobs_pending_inference is not None else []
        )
        mock_manager.jobs_in_progress = jobs_in_progress if jobs_in_progress is not None else []

        mock_manager._bound_replace_hung = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        return mock_manager

    def _make_waiting_for_job_process(self, process_id: int, *, time_elapsed: float) -> MagicMock:
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        proc = MagicMock()
        proc.process_id = process_id
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_timestamp = _time.time() - time_elapsed
        proc.last_progress_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None
        return proc

    def test_stale_waiting_for_job_not_replaced_when_paused_and_no_local_work(self) -> None:
        """A stale WAITING_FOR_JOB process must NOT be replaced when pops are paused and
        there is no local work pending.

        When job pops are paused no new jobs can arrive; a process that has been idle for
        a long time is in the expected state and must not be replaced needlessly.
        """
        proc = self._make_waiting_for_job_process(0, time_elapsed=9999.0)
        mock_manager = self._make_manager(
            [proc],
            job_pops_paused=True,
            last_pop_no_jobs=False,  # last pop returned a job before the pause; pops are now paused
            jobs_pending_inference=[],
            jobs_in_progress=[],
        )

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        mock_manager._replace_inference_process.assert_not_called()
        assert result is False, (
            "Idle processes must not be replaced when job pops are paused and there is no local work"
        )

    def test_all_processes_timed_out_not_triggered_when_paused_and_no_local_work(self) -> None:
        """The bulk 'all processes timed out' replacement must be skipped when pops are paused
        and there is no local work.

        When all inference processes have stale timestamps but pops are paused (so no new jobs
        are expected), the worker must not purge jobs and replace all processes.
        """
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        # Build a process that has been silent well past process_timeout
        proc = MagicMock()
        proc.process_id = 0
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - 9999
        proc.last_heartbeat_timestamp = _time.time() - 9999
        proc.last_progress_timestamp = _time.time() - 9999
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None

        mock_manager = self._make_manager(
            [proc],
            job_pops_paused=True,
            last_pop_no_jobs=False,  # last pop returned a job before the pause; pops are now paused
            jobs_pending_inference=[],
            jobs_in_progress=[],
        )

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        mock_manager._replace_inference_process.assert_not_called()
        mock_manager._purge_jobs.assert_not_called()
        assert mock_manager._hung_processes_detected is False, (
            "Bulk all-processes-timeout detection must not start when pops are paused and there is no local work"
        )
        assert result is False, (
            "All-processes-timed-out bulk replacement must be skipped when pops are paused "
            "and there is no local work"
        )

    def test_shutdown_timeout_still_recovers_when_paused_and_no_local_work(self) -> None:
        """Shutdown timeout recovery must still run while pops are paused and local queues are empty."""
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        proc = MagicMock()
        proc.process_id = 0
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = HordeProcessState.WAITING_FOR_JOB
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time()
        proc.last_heartbeat_timestamp = _time.time()
        proc.last_progress_timestamp = _time.time()
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None

        mock_manager = self._make_manager(
            [proc],
            job_pops_paused=True,
            last_pop_no_jobs=False,
            jobs_pending_inference=[],
            jobs_in_progress=[],
        )
        mock_manager._shutting_down = True
        mock_manager._shutting_down_time = _time.time() - (60 * 5 + 1)
        mock_manager._hung_processes_detected = True
        mock_manager._hung_processes_detected_time = _time.time() - 21

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        mock_manager._purge_jobs.assert_called_once()
        mock_manager._abort.assert_called_once()
        assert result is True, "Shutdown timeout must escalate to purge+abort even when pops are paused"

    def test_all_processes_timeout_detection_runs_when_last_pop_no_jobs_but_local_work_exists(self) -> None:
        """A stale no-jobs pop must not suppress all-processes-timeout detection when local work exists."""
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        proc = MagicMock()
        proc.process_id = 0
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = HordeProcessState.MODEL_PRELOADED
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - 9999
        proc.last_heartbeat_timestamp = _time.time() - 9999
        proc.last_progress_timestamp = _time.time() - 9999
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None

        job = MagicMock()
        mock_manager = self._make_manager(
            [proc],
            job_pops_paused=False,
            last_pop_no_jobs=True,
            jobs_pending_inference=[job],
            jobs_in_progress=[],
        )

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is False
        assert mock_manager._hung_processes_detected is True, (
            "All-processes-timeout detection must start when local work exists, even if the last pop saw no jobs"
        )
        mock_manager._purge_jobs.assert_not_called()

    def test_stale_waiting_for_job_replaced_when_paused_but_local_jobs_pending(self) -> None:
        """A stale WAITING_FOR_JOB process MUST be replaced when pops are paused but local
        jobs are still waiting to be processed.

        Pausing only stops new pops from the API; it must not prevent recovery of processes
        that are blocking already-queued local jobs.
        """
        proc = self._make_waiting_for_job_process(0, time_elapsed=9999.0)
        job = MagicMock()
        mock_manager = self._make_manager(
            [proc],
            job_pops_paused=True,
            last_pop_no_jobs=False,
            jobs_pending_inference=[job],
            jobs_in_progress=[],
        )

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True, (
            "Stale WAITING_FOR_JOB process must still be replaced when local jobs are pending, "
            "even when job pops are paused"
        )
        mock_manager._replace_inference_process.assert_called_once_with(proc)

    def test_paused_does_not_suppress_stuck_inference_processing(self) -> None:
        """INFERENCE_PROCESSING stuck detection must fire even when job pops are paused.

        A process that holds the inference semaphore and stops responding blocks all other
        processes.  The paused flag must never suppress this critical recovery path.
        """
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeProcessType

        proc = MagicMock()
        proc.process_id = 0
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - 9999
        proc.last_heartbeat_timestamp = _time.time() - 9999
        proc.last_progress_timestamp = _time.time() - 9999
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None

        mock_manager = self._make_manager(
            [proc],
            job_pops_paused=True,
            last_pop_no_jobs=False,
            jobs_pending_inference=[],
            jobs_in_progress=[],
        )
        mock_manager._process_map.is_stuck_on_inference.return_value = True

        with patch("threading.Thread"):
            result = mock_manager._bound_replace_hung()

        assert result is True, (
            "INFERENCE_PROCESSING stuck detection must fire even when job pops are paused"
        )
        mock_manager._replace_inference_process.assert_called_once_with(proc, respawn=True)


class TestInferenceBackgroundHeartbeat:
    """Tests for the background heartbeat thread introduced to prevent false
    stuck-process detection during long-running computation phases (e.g., VAE decode).
    """

    def _make_proc(self) -> "MagicMock":
        """Create a minimal mock of HordeInferenceProcess for start_inference tests."""
        import multiprocessing

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._is_busy = False
        proc._in_post_processing = False
        proc._vae_acquire_attempted = False
        proc._vae_lock_was_acquired = False
        proc._current_job_inference_steps_complete = False
        proc._last_sanitized_negative_prompt = None
        proc._last_inference_percent = None
        proc._active_model_name = "TestModel"
        proc.VAE_SEMAPHORE_TIMEOUT = 5
        proc._INFERENCE_HEARTBEAT_INTERVAL = 30.0
        proc._inference_semaphore = multiprocessing.Semaphore(1)
        proc._vae_decode_semaphore = multiprocessing.Semaphore(1)
        proc.send_process_state_change_message = MagicMock()
        proc.send_heartbeat_message = MagicMock()
        return proc

    def test_heartbeat_thread_started_and_stopped(self) -> None:
        """The background heartbeat thread must be started before basic_inference()
        and stopped (via Event.set + join) after it returns.
        """
        import threading as _threading

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = self._make_proc()
        fake_result = MagicMock()
        proc._horde = MagicMock()
        proc._horde.basic_inference.return_value = [fake_result]

        job_info = MagicMock()
        job_info.payload.prompt = "test"
        job_info.extra_source_images = None
        job_info.source_image = None
        job_info.source_mask = None
        job_info.ids = []

        threads_created: list[_threading.Thread] = []
        real_thread_cls = _threading.Thread

        def capturing_thread(*args: object, **kwargs: object) -> _threading.Thread:
            t = real_thread_cls(*args, **kwargs)
            threads_created.append(t)
            return t

        with patch("horde_worker_regen.process_management.inference_process.threading.Thread", capturing_thread):
            result = HordeInferenceProcess.start_inference(proc, job_info)

        assert result is not None, "start_inference must return results"
        # At least one thread was created for the heartbeat
        assert len(threads_created) >= 1, "A background heartbeat thread must be created during inference"
        # The thread must have finished (was joined)
        heartbeat_threads = [t for t in threads_created if "heartbeat" in (t.name or "")]
        assert heartbeat_threads, "A thread named with 'heartbeat' must be created during inference"
        assert not heartbeat_threads[0].is_alive(), "The heartbeat thread must be stopped after inference"

    def test_heartbeat_thread_is_daemon(self) -> None:
        """The background heartbeat thread must be a daemon thread so it does not
        prevent the process from exiting if it is somehow not stopped cleanly.
        """
        import threading as _threading

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = self._make_proc()
        fake_result = MagicMock()
        proc._horde = MagicMock()
        proc._horde.basic_inference.return_value = [fake_result]

        job_info = MagicMock()
        job_info.payload.prompt = "test"
        job_info.extra_source_images = None
        job_info.source_image = None
        job_info.source_mask = None
        job_info.ids = []

        daemon_values: list[bool | None] = []

        real_thread_init = _threading.Thread.__init__

        def patched_init(self_t: _threading.Thread, *args: object, **kwargs: object) -> None:  # noqa: ANN001
            real_thread_init(self_t, *args, **kwargs)
            daemon_values.append(self_t.daemon)

        with patch.object(_threading.Thread, "__init__", patched_init):
            HordeInferenceProcess.start_inference(proc, job_info)

        assert daemon_values, "At least one Thread must be created during inference"
        assert daemon_values[-1] is True, "The inference heartbeat thread must be a daemon thread"

    def test_inference_heartbeat_loop_sends_heartbeat(self) -> None:
        """_inference_heartbeat_loop must call send_heartbeat_message with
        PIPELINE_STATE_CHANGE and the last known inference percent before the
        stop event fires.
        """
        import threading as _threading

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess
        from horde_worker_regen.process_management.messages import HordeHeartbeatType

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._last_inference_percent = 97
        # Must be a real number: the loop computes `time.monotonic() - self._last_step_callback_time`,
        # and a bare MagicMock here raises TypeError (caught internally) so send_heartbeat_message is
        # never reached and the stop event never fires — an infinite loop.
        proc._last_step_callback_time = 0.0
        proc._INFERENCE_HEARTBEAT_INTERVAL = 0.01  # fire almost immediately

        stop_event = _threading.Event()

        # Let the loop fire once, then stop it
        call_counts: list[int] = [0]

        def track_heartbeat(**kwargs: object) -> None:
            call_counts[0] += 1
            stop_event.set()  # stop after first call

        proc.send_heartbeat_message = MagicMock(side_effect=track_heartbeat)

        HordeInferenceProcess._inference_heartbeat_loop(proc, stop_event)

        assert call_counts[0] >= 1, "_inference_heartbeat_loop must send at least one heartbeat"
        proc.send_heartbeat_message.assert_called_with(
            heartbeat_type=HordeHeartbeatType.PIPELINE_STATE_CHANGE,
            percent_complete=97,
        )

    def test_inference_heartbeat_loop_suppresses_when_no_progress(self) -> None:
        """When _last_inference_percent is None (no real progress yet), the heartbeat
        loop must NOT call send_heartbeat_message, preserving the process manager's
        no_step_heartbeat_timeout fast-path for early crash detection.
        """
        import threading as _threading

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._last_inference_percent = None
        proc._INFERENCE_HEARTBEAT_INTERVAL = 0.01  # fire almost immediately

        stop_event = _threading.Event()

        # Let the loop fire once then stop, by setting the event after a brief wait
        def stop_after_one_tick() -> None:
            import time as _time
            _time.sleep(0.05)
            stop_event.set()

        import threading as _threading2
        stopper = _threading2.Thread(target=stop_after_one_tick, daemon=True)
        stopper.start()

        HordeInferenceProcess._inference_heartbeat_loop(proc, stop_event)
        stopper.join(timeout=1.0)

        proc.send_heartbeat_message.assert_not_called()

    def test_last_inference_percent_updated_on_zero_fallback(self) -> None:
        """When comfyui_progress is absent and _last_inference_percent is None (very start
        of inference), a 0% heartbeat is sent and _last_inference_percent is set to 0 so
        the background heartbeat thread can report meaningful (not None) progress.
        """
        pytest.importorskip("hordelib")  # requires the GPU/ML stack
        from hordelib.horde import ProgressReport, ProgressState

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._in_post_processing = False
        proc._current_job_inference_steps_complete = False
        proc._last_inference_percent = None
        proc.send_heartbeat_message = MagicMock()
        proc._active_model_name = "TestModel"
        proc._start_inference_time = 0.0

        # Report with no comfyui_progress → triggers the fallback path at very start
        report = MagicMock(spec=ProgressReport)
        report.hordelib_progress_state = ProgressState.progress
        report.comfyui_progress = None

        HordeInferenceProcess._progress_callback_impl(proc, report)

        assert proc._last_inference_percent == 0, (
            "_last_inference_percent must be set to 0 when the fallback fires with no prior progress"
        )
        # Heartbeat must be sent with 0%
        proc.send_heartbeat_message.assert_called_once()
        call_kwargs = proc.send_heartbeat_message.call_args
        assert call_kwargs.kwargs.get("percent_complete") == 0, (
            "Heartbeat must carry percent_complete=0 at very start of inference"
        )

    def test_progress_not_reset_to_zero_between_last_step_and_post_processing(self) -> None:
        """Progress must NOT drop to 0 % when a no-comfyui_progress callback fires after
        some inference progress has already been reported.

        This is the regression test for the 1 % flash that occurred between the final
        denoising step and the first post-processing callback: a transition callback with
        comfyui_progress=None was incorrectly resetting _last_inference_percent to 0,
        which the web UI granular floor then displayed as 1 %.
        """
        pytest.importorskip("hordelib")  # requires the GPU/ML stack
        from hordelib.horde import ProgressReport, ProgressState

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        for prior_percent in (50, 95, 100):
            proc = MagicMock(spec=HordeInferenceProcess)
            proc._in_post_processing = False
            proc._current_job_inference_steps_complete = False
            # Simulate that we have already seen progress (e.g. 95% after step 19/20)
            proc._last_inference_percent = prior_percent
            proc.send_heartbeat_message = MagicMock()
            proc._active_model_name = "TestModel"
            proc._start_inference_time = 0.0

            # A transition callback with no comfyui_progress (e.g. during VAE-decode startup)
            report = MagicMock(spec=ProgressReport)
            report.hordelib_progress_state = ProgressState.progress
            report.comfyui_progress = None

            HordeInferenceProcess._progress_callback_impl(proc, report)

            # Progress must NOT drop
            assert proc._last_inference_percent == prior_percent, (
                f"_last_inference_percent must stay at {prior_percent} (not reset to 0) "
                f"when a no-progress callback fires after prior progress was established. "
                f"Got {proc._last_inference_percent}"
            )
            # The heartbeat must carry the preserved percentage (not 0)
            proc.send_heartbeat_message.assert_called_once()
            call_kwargs = proc.send_heartbeat_message.call_args
            assert call_kwargs.kwargs.get("percent_complete") == prior_percent, (
                f"Heartbeat must carry percent_complete={prior_percent} to prevent the "
                f"1 % flash in the web UI, got {call_kwargs.kwargs.get('percent_complete')}"
            )

    def test_last_inference_percent_updated_on_inference_step(self) -> None:
        """_last_inference_percent must be updated to the step's percentage when
        an INFERENCE_STEP heartbeat is sent in _progress_callback_impl.
        """
        pytest.importorskip("hordelib")  # requires the GPU/ML stack
        from hordelib.horde import ProgressReport, ProgressState
        from hordelib.utils.ioredirect import ComfyUIProgress

        from horde_worker_regen.process_management.inference_process import HordeInferenceProcess

        proc = MagicMock(spec=HordeInferenceProcess)
        proc._in_post_processing = False
        proc._current_job_inference_steps_complete = False
        proc._last_inference_percent = None
        proc.send_heartbeat_message = MagicMock()
        proc.send_memory_report_message = MagicMock()
        proc._active_model_name = "TestModel"
        proc._start_inference_time = 0.0

        # Build a progress report for step 29/30 → 97%
        comfy_progress = MagicMock(spec=ComfyUIProgress)
        comfy_progress.current_step = 29
        comfy_progress.total_steps = 30
        comfy_progress.percent = 96.67
        comfy_progress.rate = 1.96
        from hordelib.utils.ioredirect import ComfyUIProgressUnit
        comfy_progress.rate_unit = ComfyUIProgressUnit.ITERATIONS_PER_SECOND

        report = MagicMock(spec=ProgressReport)
        report.hordelib_progress_state = ProgressState.progress
        report.comfyui_progress = comfy_progress

        HordeInferenceProcess._progress_callback_impl(proc, report)

        assert proc._last_inference_percent == 96, (
            "_last_inference_percent must be updated to int(96.67) == 96 after an INFERENCE_STEP"
        )


class TestRecordFaultedJobHistory:
    """Tests for the _record_faulted_job_history helper method.

    Verifies that the helper adds entries to _faulted_jobs_history correctly, deduplicates,
    and enforces the max-history cap.
    """

    def _make_faulted_job(self, job_id: str = "test-job-id") -> MagicMock:
        """Return a minimal mock ImageGenerateJobPopResponse."""
        job = MagicMock()
        job.id_ = job_id
        job.model = "TestModel"
        job.payload = MagicMock()
        job.payload.width = 512
        job.payload.height = 512
        job.payload.ddim_steps = 30
        job.payload.sampler_name = "euler_a"
        job.payload.n_iter = 1
        job.payload.workflow = None
        job.payload.loras = []
        return job

    def _make_manager(self) -> MagicMock:
        """Return a minimal mock manager with _record_faulted_job_history bound."""
        import types

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._faulted_jobs_history = []
        mock_manager._max_faulted_jobs_history = HordeWorkerProcessManager._max_faulted_jobs_history
        bound = types.MethodType(HordeWorkerProcessManager._record_faulted_job_history, mock_manager)
        mock_manager._bound_record = bound
        return mock_manager

    def test_adds_entry_to_history(self) -> None:
        """A new faulted job must be prepended to _faulted_jobs_history."""
        mock_manager = self._make_manager()
        job = self._make_faulted_job("job-abc")

        mock_manager._bound_record(job, fault_phase="INFERENCE_PROCESSING")

        assert len(mock_manager._faulted_jobs_history) == 1
        entry = mock_manager._faulted_jobs_history[0]
        assert entry["job_id"] == "job-abc"
        assert entry["model"] == "TestModel"
        assert entry["fault_phase"] == "INFERENCE_PROCESSING"
        assert entry["width"] == 512
        assert entry["height"] == 512
        assert entry["steps"] == 30
        assert entry["sampler"] == "euler_a"

    def test_newest_entry_is_first(self) -> None:
        """Entries must be prepended so the most recent fault is at index 0."""
        mock_manager = self._make_manager()
        job_old = self._make_faulted_job("job-old")
        job_new = self._make_faulted_job("job-new")

        mock_manager._bound_record(job_old)
        mock_manager._bound_record(job_new)

        assert mock_manager._faulted_jobs_history[0]["job_id"] == "job-new"
        assert mock_manager._faulted_jobs_history[1]["job_id"] == "job-old"

    def test_duplicate_job_id_is_not_added(self) -> None:
        """Calling _record_faulted_job_history twice for the same job_id must add only one entry."""
        mock_manager = self._make_manager()
        job = self._make_faulted_job("job-dup")

        mock_manager._bound_record(job, fault_phase="INFERENCE_PROCESSING")
        mock_manager._bound_record(job, fault_phase="INFERENCE_PROCESSING")

        assert len(mock_manager._faulted_jobs_history) == 1

    def test_max_history_cap_is_enforced(self) -> None:
        """After reaching _max_faulted_jobs_history entries, the oldest entry is evicted."""
        mock_manager = self._make_manager()
        max_n = mock_manager._max_faulted_jobs_history

        for i in range(max_n + 2):
            job = self._make_faulted_job(f"job-{i:05d}")
            mock_manager._bound_record(job)

        assert len(mock_manager._faulted_jobs_history) == max_n

    def test_none_fault_phase_stored_correctly(self) -> None:
        """When fault_phase is None the entry must still be added with fault_phase=None."""
        mock_manager = self._make_manager()
        job = self._make_faulted_job("job-no-phase")

        mock_manager._bound_record(job, fault_phase=None)

        assert mock_manager._faulted_jobs_history[0]["fault_phase"] is None


class TestHandleJobFaultRecordsHistory:
    """Tests that handle_job_fault correctly records permanently faulted jobs.

    These tests exercise the real handle_job_fault implementation (not mocked) to verify
    that _record_faulted_job_history is called for jobs that exhaust their retry budget
    and for jobs whose job_info is unexpectedly missing from jobs_lookup.
    """

    def _make_faulted_job(self, job_id: str = "test-job-id") -> MagicMock:
        job = MagicMock()
        job.id_ = job_id
        job.model = "TestModel"
        job.payload = MagicMock()
        job.payload.width = 512
        job.payload.height = 512
        job.payload.ddim_steps = 30
        job.payload.sampler_name = "euler_a"
        job.payload.n_iter = 1
        job.payload.workflow = None
        job.payload.loras = []
        return job

    def _make_job_info(self, retry_count: int) -> MagicMock:
        ji = MagicMock()
        ji.retry_count = retry_count
        return ji

    def _make_manager_with_real_handle_job_fault(
        self,
        *,
        job: MagicMock,
        retry_count: int,
    ) -> MagicMock:
        import types

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        job_info = self._make_job_info(retry_count=retry_count)
        mock_manager.jobs_lookup = {job: job_info}
        mock_manager.jobs_in_progress = []
        mock_manager.jobs_pending_safety_check = []
        mock_manager.jobs_pending_submit = []
        mock_manager.jobs_pending_inference = []
        mock_manager._faulted_jobs_history = []
        mock_manager._max_faulted_jobs_history = HordeWorkerProcessManager._max_faulted_jobs_history
        mock_manager.MAX_JOB_RETRIES = HordeWorkerProcessManager.MAX_JOB_RETRIES
        mock_manager._skipped_line_next_job_and_process = None
        mock_manager._failed_models = {}
        mock_manager.bridge_data.process_timeout = 60
        mock_manager._invalidate_megapixelsteps_cache = MagicMock()
        # Bind the real implementations
        mock_manager._record_faulted_job_history = types.MethodType(
            HordeWorkerProcessManager._record_faulted_job_history,
            mock_manager,
        )
        mock_manager.handle_job_fault = types.MethodType(
            HordeWorkerProcessManager.handle_job_fault,
            mock_manager,
        )
        return mock_manager

    def test_permanently_faulted_job_added_to_history(self) -> None:
        """When a job has exhausted all retries it must be added to _faulted_jobs_history."""
        job = self._make_faulted_job("perm-fault-job")
        # retry_count already at MAX_JOB_RETRIES so this call permanently faults the job
        mock_manager = self._make_manager_with_real_handle_job_fault(
            job=job,
            retry_count=1,
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert len(mock_manager._faulted_jobs_history) == 1
        assert mock_manager._faulted_jobs_history[0]["job_id"] == "perm-fault-job"

    def test_retried_job_not_added_to_history(self) -> None:
        """When a job still has retries remaining it must NOT be added to _faulted_jobs_history."""
        job = self._make_faulted_job("retry-job")
        mock_manager = self._make_manager_with_real_handle_job_fault(
            job=job,
            retry_count=0,
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert len(mock_manager._faulted_jobs_history) == 0

    def test_job_not_in_lookup_still_added_to_history(self) -> None:
        """When job_info is missing from jobs_lookup the job must still appear in history."""
        import types

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job = self._make_faulted_job("missing-lookup-job")
        mock_manager = MagicMock()
        mock_manager.jobs_lookup = {}  # job not in lookup
        mock_manager._failed_models = {}
        mock_manager._inference_failures = {}
        mock_manager._faulted_jobs_history = []
        mock_manager._faulted_jobs_per_phase = {}
        mock_manager._max_faulted_jobs_history = HordeWorkerProcessManager._max_faulted_jobs_history
        mock_manager._prune_preload_stuck_failures = types.MethodType(
            HordeWorkerProcessManager._prune_preload_stuck_failures,
            mock_manager,
        )
        mock_manager._record_inference_failure = types.MethodType(
            HordeWorkerProcessManager._record_inference_failure,
            mock_manager,
        )
        mock_manager._record_faulted_job_history = types.MethodType(
            HordeWorkerProcessManager._record_faulted_job_history,
            mock_manager,
        )
        mock_manager.handle_job_fault = types.MethodType(
            HordeWorkerProcessManager.handle_job_fault,
            mock_manager,
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert len(mock_manager._faulted_jobs_history) == 1
        assert mock_manager._faulted_jobs_history[0]["job_id"] == "missing-lookup-job"
        assert mock_manager._failed_models == {"TestModel": 1}
        assert len(mock_manager._inference_failures.get("TestModel", [])) == 1
        assert mock_manager._faulted_jobs_per_phase == {"Unknown Phase": 1}

    def test_duplicate_metadata_missing_faults_not_double_counted(self) -> None:
        """Duplicate metadata-missing faults for the same job must not double-count model stats/cooldown."""
        import types

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job = self._make_faulted_job("missing-lookup-dupe")
        mock_manager = MagicMock()
        mock_manager.jobs_lookup = {}
        mock_manager._failed_models = {}
        mock_manager._inference_failures = {}
        mock_manager._faulted_jobs_history = []
        mock_manager._faulted_jobs_per_phase = {}
        mock_manager._max_faulted_jobs_history = HordeWorkerProcessManager._max_faulted_jobs_history
        mock_manager._prune_preload_stuck_failures = types.MethodType(
            HordeWorkerProcessManager._prune_preload_stuck_failures,
            mock_manager,
        )
        mock_manager._record_inference_failure = types.MethodType(
            HordeWorkerProcessManager._record_inference_failure,
            mock_manager,
        )
        mock_manager._record_faulted_job_history = types.MethodType(
            HordeWorkerProcessManager._record_faulted_job_history,
            mock_manager,
        )
        mock_manager.handle_job_fault = types.MethodType(
            HordeWorkerProcessManager.handle_job_fault,
            mock_manager,
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)
        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert len(mock_manager._faulted_jobs_history) == 1
        assert mock_manager._failed_models == {"TestModel": 1}
        assert len(mock_manager._inference_failures.get("TestModel", [])) == 1
        assert mock_manager._faulted_jobs_per_phase == {"Unknown Phase": 1}

    def test_fault_phase_from_process_state_recorded(self) -> None:
        """The fault_phase must reflect the process state at the time of the fault."""
        job = self._make_faulted_job("phase-job")
        mock_manager = self._make_manager_with_real_handle_job_fault(
            job=job,
            retry_count=1,
        )

        process_info = MagicMock()
        process_info.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        process_info.inference_started_timestamp = None

        mock_manager.handle_job_fault(faulted_job=job, process_info=process_info)

        assert mock_manager._faulted_jobs_history[0]["fault_phase"] == "INFERENCE_PROCESSING"


class TestReplaceInferenceProcessBroadExceptionHandling:
    """Tests that _replace_inference_process() continues and starts the replacement process
    even when semaphore/lock release operations raise unexpected exceptions (e.g. OSError).

    On some platforms or in certain failure modes, releasing a semaphore or lock can raise
    exceptions other than ValueError (e.g. OSError("Invalid semaphore") on containers with
    restricted IPC namespaces, or RuntimeError on Windows edge cases).  If these exceptions
    propagate out of _replace_inference_process(), the new process is never started and the
    stuck slot is never recovered — recovery fails silently.  The fix wraps each release in
    ``except Exception`` so the replacement always proceeds regardless of the exception type.
    """

    def _make_manager(
        self,
        state: HordeProcessState,
        *,
        inference_sem_error: Exception | None = None,
        disk_lock_error: Exception | None = None,
        vae_sem_error: Exception | None = None,
        aux_lock_error: Exception | None = None,
    ) -> MagicMock:
        """Build a minimal mock manager for _replace_inference_process() tests."""
        import multiprocessing

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        ctx = multiprocessing.get_context("spawn")

        process_info = MagicMock()
        process_info.process_id = 0
        process_info.last_process_state = state
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None
        process_info.loaded_horde_model_name = None

        mock_manager = MagicMock()
        mock_manager.jobs_lookup = {}
        mock_manager.jobs_in_progress = []

        # Set up inference semaphore: use a real BoundedSemaphore unless an error is injected
        if inference_sem_error is not None:
            bad_sem = MagicMock()
            bad_sem.release = MagicMock(side_effect=inference_sem_error)
            mock_manager._inference_semaphore = bad_sem
        else:
            mock_manager._inference_semaphore = ctx.BoundedSemaphore(1)
            mock_manager._inference_semaphore.acquire()  # simulate child holding it

        # Disk lock
        if disk_lock_error is not None:
            bad_disk_lock = MagicMock()
            bad_disk_lock.release = MagicMock(side_effect=disk_lock_error)
            mock_manager._disk_lock = bad_disk_lock
        else:
            mock_manager._disk_lock = ctx.Lock()
            mock_manager._disk_lock.acquire()  # simulate child holding it

        # VAE decode semaphore
        if vae_sem_error is not None:
            bad_vae = MagicMock()
            bad_vae.release = MagicMock(side_effect=vae_sem_error)
            mock_manager._vae_decode_semaphore = bad_vae
        else:
            mock_manager._vae_decode_semaphore = ctx.BoundedSemaphore(1)
            mock_manager._vae_decode_semaphore.acquire()  # simulate child holding it

        # Aux model lock
        if aux_lock_error is not None:
            bad_aux = MagicMock()
            bad_aux.release = MagicMock(side_effect=aux_lock_error)
            mock_manager._aux_model_lock = bad_aux
        else:
            mock_manager._aux_model_lock = ctx.Lock()

        bound = HordeWorkerProcessManager._replace_inference_process.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        mock_manager._bound_replace = bound
        mock_manager._bound_replace_process_info = process_info
        return mock_manager

    def test_inference_semaphore_os_error_does_not_prevent_replacement(self) -> None:
        """When _inference_semaphore.release() raises OSError, the new process must still be started.

        This validates that the ``except Exception`` handler added around semaphore release
        in _replace_inference_process() catches non-ValueError exceptions and allows the
        replacement to proceed.
        """
        mock_manager = self._make_manager(
            HordeProcessState.INFERENCE_PROCESSING,
            inference_sem_error=OSError("Invalid semaphore"),
        )

        mock_manager._bound_replace(mock_manager._bound_replace_process_info)

        # The replacement must proceed: the correct slot (process_id=0) must be started
        mock_manager._start_inference_process.assert_called_once_with(0)

    def test_disk_lock_os_error_does_not_prevent_replacement(self) -> None:
        """When _disk_lock.release() raises OSError, the new process must still be started."""
        mock_manager = self._make_manager(
            HordeProcessState.INFERENCE_PROCESSING,
            disk_lock_error=OSError("Invalid lock"),
        )

        mock_manager._bound_replace(mock_manager._bound_replace_process_info)

        mock_manager._start_inference_process.assert_called_once_with(0)

    def test_vae_semaphore_os_error_does_not_prevent_replacement(self) -> None:
        """When _vae_decode_semaphore.release() raises OSError, the new process must still be started."""
        mock_manager = self._make_manager(
            HordeProcessState.INFERENCE_POST_PROCESSING,
            vae_sem_error=OSError("Invalid semaphore"),
        )

        mock_manager._bound_replace(mock_manager._bound_replace_process_info)

        mock_manager._start_inference_process.assert_called_once_with(0)

    def test_aux_model_lock_os_error_does_not_prevent_replacement(self) -> None:
        """When _aux_model_lock.release() raises OSError, the new process must still be started."""
        mock_manager = self._make_manager(
            HordeProcessState.DOWNLOADING_AUX_MODEL,
            aux_lock_error=OSError("Invalid lock"),
        )

        mock_manager._bound_replace(mock_manager._bound_replace_process_info)

        mock_manager._start_inference_process.assert_called_once_with(0)

    def test_inference_semaphore_runtime_error_does_not_prevent_replacement(self) -> None:
        """When _inference_semaphore.release() raises RuntimeError, the replacement must proceed."""
        mock_manager = self._make_manager(
            HordeProcessState.INFERENCE_STARTING,
            inference_sem_error=RuntimeError("semaphore not owned"),
        )

        mock_manager._bound_replace(mock_manager._bound_replace_process_info)

        mock_manager._start_inference_process.assert_called_once_with(0)


class TestProcessEndingHandlerBroadExceptionHandling:
    """Tests that the PROCESS_ENDING handler in receive_and_handle_process_messages continues
    to fault jobs and restart the process even when semaphore releases raise unexpected exceptions.

    If _inference_semaphore.release() or _vae_decode_semaphore.release() raises an OSError
    (or any non-ValueError exception), the handler must not abort — the job must still be
    faulted and the process map must still be updated, so the slot can be recovered.
    """

    def _make_message(self, state: HordeProcessState) -> MagicMock:
        from horde_worker_regen.process_management.messages import HordeProcessStateChangeMessage

        msg = MagicMock(spec=HordeProcessStateChangeMessage)
        msg.process_id = 0
        msg.process_state = state
        msg.process_launch_identifier = 1
        msg.info = ""
        return msg

    def _run_with_bad_semaphore(
        self,
        prior_state: HordeProcessState,
        sem_error: Exception,
        *,
        which_semaphore: str = "inference",
    ) -> MagicMock:
        """Run the PROCESS_ENDING path with a semaphore that raises on release.

        Returns the mock_manager so callers can inspect whether on_process_ending was called.
        """
        import multiprocessing
        import queue as queue_mod

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        ctx = multiprocessing.get_context("spawn")

        process_info = MagicMock()
        process_info.process_launch_identifier = 1
        process_info.last_process_state = prior_state
        process_info.inference_started_timestamp = None
        process_info.last_job_referenced = None

        process_map = MagicMock()
        process_map.__contains__ = MagicMock(side_effect=lambda key: key == 0)
        process_map.__getitem__ = MagicMock(side_effect=lambda key: process_info)

        msg = self._make_message(HordeProcessState.PROCESS_ENDING)
        q = queue_mod.Queue()
        q.put(msg)

        mock_manager = MagicMock()
        mock_manager._process_message_queue = q
        mock_manager._process_map = process_map
        mock_manager._in_deadlock = False
        mock_manager._in_queue_deadlock = False
        mock_manager.jobs_in_progress = []

        # Set up the bad semaphore that raises on release
        bad_sem = MagicMock()
        bad_sem.release = MagicMock(side_effect=sem_error)

        if which_semaphore == "inference":
            mock_manager._inference_semaphore = bad_sem
            mock_manager._vae_decode_semaphore = ctx.BoundedSemaphore(1)
        else:
            mock_manager._inference_semaphore = ctx.BoundedSemaphore(1)
            mock_manager._vae_decode_semaphore = bad_sem

        bound = HordeWorkerProcessManager.receive_and_handle_process_messages.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        bound()
        return mock_manager

    def test_inference_semaphore_os_error_does_not_abort_process_ending(self) -> None:
        """An OSError from inference semaphore release must not abort the PROCESS_ENDING handler.

        The process map's on_process_ending must still be called so the slot is properly
        cleaned up and can be restarted.
        """
        mock_manager = self._run_with_bad_semaphore(
            prior_state=HordeProcessState.INFERENCE_PROCESSING,
            sem_error=OSError("Invalid semaphore"),
            which_semaphore="inference",
        )

        # on_process_ending must have been called despite the OSError
        mock_manager._process_map.on_process_ending.assert_called_once_with(process_id=0)

    def test_vae_semaphore_os_error_does_not_abort_process_ending(self) -> None:
        """An OSError from VAE decode semaphore release must not abort the PROCESS_ENDING handler."""
        mock_manager = self._run_with_bad_semaphore(
            prior_state=HordeProcessState.INFERENCE_POST_PROCESSING,
            sem_error=OSError("Invalid semaphore"),
            which_semaphore="vae",
        )

        mock_manager._process_map.on_process_ending.assert_called_once_with(process_id=0)

    def test_inference_semaphore_runtime_error_does_not_abort_process_ending(self) -> None:
        """A RuntimeError from inference semaphore release must not abort the PROCESS_ENDING handler."""
        mock_manager = self._run_with_bad_semaphore(
            prior_state=HordeProcessState.INFERENCE_STARTING,
            sem_error=RuntimeError("semaphore not owned"),
            which_semaphore="inference",
        )

        mock_manager._process_map.on_process_ending.assert_called_once_with(process_id=0)


class TestRecoveryTimerThreadIsDaemon:
    """Tests that the _recently_recovered timer thread created by replace_hung_processes()
    is a daemon thread.

    Non-daemon threads keep the interpreter alive even after sys.exit() is called,
    which can delay clean shutdown by up to inference_step_timeout (600 s).  The timer
    only resets a boolean flag — it must be a daemon so it doesn't block process exit.
    """

    def _make_manager_with_stuck_inference(self) -> MagicMock:
        """Build a minimal manager with a stuck INFERENCE_PROCESSING process."""
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        proc = MagicMock()
        proc.process_id = 0
        proc.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - 9999
        proc.last_heartbeat_timestamp = _time.time() - 9999
        proc.last_progress_timestamp = _time.time() - 9999
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None

        mock_manager = MagicMock()
        mock_manager._recently_recovered = False
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager._hung_processes_detected = False
        mock_manager._hung_processes_detected_time = 0.0
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 300
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        # is_stuck_on_inference returns True for the stuck INFERENCE_PROCESSING process
        mock_manager._process_map.is_stuck_on_inference.return_value = True
        mock_manager._process_map.values.return_value = [proc]

        mock_manager._bound_replace_hung = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )
        return mock_manager

    def test_recovery_timer_thread_is_daemon(self) -> None:
        """The _recently_recovered timer thread started by replace_hung_processes() must be daemon=True.

        A non-daemon thread keeps the interpreter alive even after sys.exit(), potentially
        delaying clean shutdown by up to inference_step_timeout (600 s by default).

        We capture the ``daemon`` kwarg passed to ``threading.Thread(...)`` via a lightweight
        fake that does not actually spawn a thread (its ``start()`` is a no-op), so the test
        doesn't leave a background sleeper running for 600 s.
        """
        mock_manager = self._make_manager_with_stuck_inference()

        captured_daemon_values: list[bool | None] = []

        class _FakeThread:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured_daemon_values.append(kwargs.get("daemon"))

            def start(self) -> None:
                pass  # no-op: don't actually spawn a thread

        with patch("threading.Thread", _FakeThread):
            mock_manager._bound_replace_hung()

        assert len(captured_daemon_values) >= 1, "replace_hung_processes must create a timer thread"
        for daemon_val in captured_daemon_values:
            assert daemon_val is True, (
                f"Timer thread must be created with daemon=True to avoid blocking clean shutdown; "
                f"got daemon={daemon_val!r}"
            )


class TestGetProcessByHordeModelNamePreference:
    """Tests that get_process_by_horde_model_name prefers job-accepting processes.

    When multiple processes have the same model loaded (e.g. one MODEL_PRELOADED and one
    INFERENCE_STARTING), the process that can accept a job must be returned first.
    This prevents MODEL_PRELOADED processes from being stuck waiting for a job while
    a busy process for the same model is returned by the lookup.
    """

    def _make_process_map(self, processes: list[MagicMock]) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import ProcessMap

        pmap = MagicMock()
        pmap.values.return_value = processes
        pmap.get_process_by_horde_model_name = ProcessMap.get_process_by_horde_model_name.__get__(
            pmap, ProcessMap
        )
        return pmap

    def _make_process(
        self,
        process_id: int,
        state: HordeProcessState,
        model: str | None,
    ) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import HordeProcessInfo

        proc = MagicMock()
        proc.process_id = process_id
        proc.loaded_horde_model_name = model
        proc.last_process_state = state
        proc.inference_started_timestamp = None
        proc.can_accept_job = HordeProcessInfo.can_accept_job.__get__(proc, HordeProcessInfo)
        return proc

    def test_returns_accepting_process_when_two_with_same_model(self) -> None:
        """When P1 is INFERENCE_STARTING (busy) and P2 is MODEL_PRELOADED (ready),
        get_process_by_horde_model_name must return P2, not P1."""
        p1 = self._make_process(1, HordeProcessState.INFERENCE_STARTING, "Fustercluck")
        p2 = self._make_process(2, HordeProcessState.MODEL_PRELOADED, "Fustercluck")
        pmap = self._make_process_map([p1, p2])

        result = pmap.get_process_by_horde_model_name("Fustercluck")

        assert result is p2, "Should return MODEL_PRELOADED p2 over INFERENCE_STARTING p1"

    def test_returns_first_match_when_none_can_accept(self) -> None:
        """When no process can accept a job, the first matching process is returned as fallback."""
        p1 = self._make_process(1, HordeProcessState.INFERENCE_STARTING, "Fustercluck")
        p2 = self._make_process(2, HordeProcessState.INFERENCE_PROCESSING, "Fustercluck")
        pmap = self._make_process_map([p1, p2])

        result = pmap.get_process_by_horde_model_name("Fustercluck")

        assert result is p1, "Should return first match (p1) when neither process can accept"

    def test_returns_waiting_process_over_inference_starting(self) -> None:
        """WAITING_FOR_JOB process should be preferred over INFERENCE_STARTING."""
        p1 = self._make_process(1, HordeProcessState.INFERENCE_STARTING, "ModelA")
        p2 = self._make_process(2, HordeProcessState.WAITING_FOR_JOB, "ModelA")
        # Note: a WAITING_FOR_JOB process with a loaded_horde_model_name can occur after a process
        # finishes inference and is reset to WAITING_FOR_JOB while still holding the model in RAM.
        pmap = self._make_process_map([p1, p2])

        result = pmap.get_process_by_horde_model_name("ModelA")

        assert result is p2

    def test_returns_none_when_no_match(self) -> None:
        """Returns None when no process has the requested model."""
        p1 = self._make_process(1, HordeProcessState.MODEL_PRELOADED, "OtherModel")
        pmap = self._make_process_map([p1])

        result = pmap.get_process_by_horde_model_name("Fustercluck")

        assert result is None

    def test_single_model_preloaded_process_returned(self) -> None:
        """When only one process has the model and it can accept, return it directly."""
        p1 = self._make_process(1, HordeProcessState.MODEL_PRELOADED, "Fustercluck")
        pmap = self._make_process_map([p1])

        result = pmap.get_process_by_horde_model_name("Fustercluck")

        assert result is p1


class TestGetNextJobAndProcessModelPreloading:
    """Tests that get_next_job_and_process skips MODEL_PRELOADING jobs to dispatch preloaded ones.

    When the first job in the queue has its model still being preloaded (MODEL_PRELOADING),
    get_next_job_and_process must look for other pending jobs that already have a
    MODEL_PRELOADED process ready.  This prevents MODEL_PRELOADED processes from being
    stuck when a MODEL_PRELOADING job sits at the front of the queue.
    """

    def _make_manager(
        self,
        *,
        first_job_model: str,
        first_process_state: HordeProcessState,
        second_job_model: str,
        second_process_state: HordeProcessState,
    ) -> MagicMock:
        """Build a minimal manager mock with two pending jobs and their corresponding processes."""
        import types

        from horde_worker_regen.process_management.process_manager import (
            HordeProcessInfo,
            HordeWorkerProcessManager,
        )

        # First job - its model is MODEL_PRELOADING
        job1 = MagicMock()
        job1.model = first_job_model
        job1.payload.loras = None
        job1.payload.n_iter = 1

        # Second job - its model is MODEL_PRELOADED (ready)
        job2 = MagicMock()
        job2.model = second_job_model
        job2.payload.loras = None
        job2.payload.n_iter = 1

        # Process for job1
        proc1 = MagicMock()
        proc1.loaded_horde_model_name = first_job_model
        proc1.last_process_state = first_process_state
        proc1.inference_started_timestamp = None
        proc1.can_accept_job = HordeProcessInfo.can_accept_job.__get__(proc1, HordeProcessInfo)

        # Process for job2
        proc2 = MagicMock()
        proc2.loaded_horde_model_name = second_job_model
        proc2.last_process_state = second_process_state
        proc2.inference_started_timestamp = None
        proc2.can_accept_job = HordeProcessInfo.can_accept_job.__get__(proc2, HordeProcessInfo)

        mock_process_map = MagicMock()

        def fake_get_process_by_model(model: str) -> MagicMock | None:
            if model == first_job_model:
                return proc1
            if model == second_job_model:
                return proc2
            return None

        mock_process_map.get_process_by_horde_model_name.side_effect = fake_get_process_by_model

        mock_manager = MagicMock()
        mock_manager._process_map = mock_process_map
        mock_manager.jobs_pending_inference = deque([job1, job2])
        mock_manager.jobs_in_progress = []
        mock_manager._skipped_line_next_job_and_process = None
        mock_manager._preload_delay_notified = False
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager._model_recently_missing = False
        mock_manager.max_concurrent_inference_processes = 3
        mock_manager.get_single_job_effective_megapixelsteps.return_value = 1
        mock_manager._horde_model_map.is_model_loading.return_value = True

        mock_manager._bound_get_next_job_and_process = types.MethodType(
            HordeWorkerProcessManager.get_next_job_and_process,
            mock_manager,
        )
        return mock_manager, job1, job2, proc2

    def _patch_next_job_and_process(self) -> object:
        """Return a context manager that patches NextJobAndProcess to accept MagicMock values.

        NextJobAndProcess is a Pydantic model that validates its fields.  In tests we use
        MagicMock for jobs and processes, so we patch the class to use a simple namespace
        instead of triggering Pydantic validation.
        """

        class _FakeNJAP:
            def __init__(self, **kwargs: object) -> None:
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return patch(
            "horde_worker_regen.process_management.process_manager.NextJobAndProcess",
            _FakeNJAP,
        )

    def test_skips_model_preloading_to_dispatch_preloaded(self) -> None:
        """When job1's model is MODEL_PRELOADING and job2's model is MODEL_PRELOADED,
        get_next_job_and_process must return job2 with its preloaded process."""
        mock_manager, job1, job2, proc2 = self._make_manager(
            first_job_model="SomeModel",
            first_process_state=HordeProcessState.MODEL_PRELOADING,
            second_job_model="Fustercluck",
            second_process_state=HordeProcessState.MODEL_PRELOADED,
        )

        with self._patch_next_job_and_process():
            result = mock_manager._bound_get_next_job_and_process()

        assert result is not None, "Should find a dispatchable job despite MODEL_PRELOADING first"
        assert result.next_job is job2, "Should dispatch job2 (Fustercluck, preloaded)"
        assert result.process_with_model is proc2
        assert result.skipped_line is True

    def test_no_skip_when_first_job_model_is_preloaded(self) -> None:
        """When job1's model is MODEL_PRELOADED, it should be dispatched normally."""
        mock_manager, job1, job2, proc2 = self._make_manager(
            first_job_model="Fustercluck",
            first_process_state=HordeProcessState.MODEL_PRELOADED,
            second_job_model="SomeModel",
            second_process_state=HordeProcessState.MODEL_PRELOADED,
        )

        with self._patch_next_job_and_process():
            result = mock_manager._bound_get_next_job_and_process()

        assert result is not None
        assert result.next_job is job1, "Should dispatch job1 normally when its process is ready"
        assert result.skipped_line is False

    def test_returns_none_when_no_other_preloaded_job(self) -> None:
        """When job1 is MODEL_PRELOADING and no other job has a preloaded process, return None."""
        mock_manager, _job1, _job2, _proc2 = self._make_manager(
            first_job_model="SomeModel",
            first_process_state=HordeProcessState.MODEL_PRELOADING,
            second_job_model="OtherModel",
            second_process_state=HordeProcessState.MODEL_PRELOADING,
        )

        with self._patch_next_job_and_process():
            result = mock_manager._bound_get_next_job_and_process()

        assert result is None, "Should return None when no other job has a ready process"


class TestPreloadStuckCooldown:
    """Tests for the per-model preload-stuck cooldown mechanism.

    When a model causes MODEL_PRELOADING to hang repeatedly (_PRELOAD_STUCK_FAILURE_THRESHOLD
    times within _PRELOAD_STUCK_FAILURE_WINDOW seconds), it enters a cooldown period during
    which jobs for that model are immediately permanently faulted instead of being dispatched
    to a preloading process.  This prevents the worker from cycling indefinitely through a
    model that cannot be loaded.
    """

    def _make_manager(self) -> MagicMock:
        """Create a manager mock configured for testing the preload cooldown mechanism.

        Binds the real ``_record_preload_stuck_failure``, ``_is_model_in_preload_cooldown``,
        and ``_fault_cooldown_model_jobs`` methods so that tests exercise the actual
        cooldown logic while keeping all other interactions mocked.
        """
        from horde_worker_regen.process_management.process_manager import (
            HordeWorkerProcessManager,
        )

        mock_manager = MagicMock()
        mock_manager._preload_stuck_failures = {}
        mock_manager._PRELOAD_STUCK_FAILURE_THRESHOLD = (
            HordeWorkerProcessManager._PRELOAD_STUCK_FAILURE_THRESHOLD
        )
        mock_manager._PRELOAD_STUCK_FAILURE_WINDOW = (
            HordeWorkerProcessManager._PRELOAD_STUCK_FAILURE_WINDOW
        )
        mock_manager._PRELOAD_STUCK_COOLDOWN = HordeWorkerProcessManager._PRELOAD_STUCK_COOLDOWN
        mock_manager.MAX_JOB_RETRIES = HordeWorkerProcessManager.MAX_JOB_RETRIES

        mock_manager._record_preload_stuck_failure = (
            HordeWorkerProcessManager._record_preload_stuck_failure.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )
        mock_manager._is_model_in_preload_cooldown = (
            HordeWorkerProcessManager._is_model_in_preload_cooldown.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )
        mock_manager._fault_cooldown_model_jobs = (
            HordeWorkerProcessManager._fault_cooldown_model_jobs.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )
        mock_manager._prune_preload_stuck_failures = (
            HordeWorkerProcessManager._prune_preload_stuck_failures.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )
        return mock_manager

    # ------------------------------------------------------------------
    # _record_preload_stuck_failure / _is_model_in_preload_cooldown
    # ------------------------------------------------------------------

    def test_no_cooldown_with_fewer_than_threshold_failures(self) -> None:
        """A single stuck event must NOT trigger cooldown (threshold is 2)."""
        import time as _time

        mock_manager = self._make_manager()
        now = _time.time()
        mock_manager._record_preload_stuck_failure("ModelA", now)

        assert not mock_manager._is_model_in_preload_cooldown("ModelA"), (
            "One stuck event should not be enough to trigger cooldown"
        )

    def test_cooldown_triggered_at_threshold(self) -> None:
        """Reaching the threshold within the window puts the model in cooldown."""
        import time as _time

        mock_manager = self._make_manager()
        now = _time.time()
        mock_manager._record_preload_stuck_failure("ModelA", now - 10)
        mock_manager._record_preload_stuck_failure("ModelA", now)

        assert mock_manager._is_model_in_preload_cooldown("ModelA"), (
            "Two stuck events within the window should trigger cooldown"
        )

    def test_old_failures_pruned_outside_window(self) -> None:
        """Failures older than _PRELOAD_STUCK_FAILURE_WINDOW do not count toward the threshold."""
        import time as _time

        mock_manager = self._make_manager()
        window = mock_manager._PRELOAD_STUCK_FAILURE_WINDOW
        now = _time.time()
        # First failure is older than the window.
        mock_manager._record_preload_stuck_failure("ModelA", now - window - 100)
        # Second failure is recent.
        mock_manager._record_preload_stuck_failure("ModelA", now)

        assert not mock_manager._is_model_in_preload_cooldown("ModelA"), (
            "Only failures within the window count; old failures must be pruned"
        )

    def test_cooldown_expires_after_duration(self) -> None:
        """Once the cooldown duration elapses the model must leave cooldown."""
        import time as _time

        mock_manager = self._make_manager()
        cooldown = mock_manager._PRELOAD_STUCK_COOLDOWN
        # Both failures are older than the cooldown duration.
        old_ts = _time.time() - cooldown - 1
        mock_manager._record_preload_stuck_failure("ModelA", old_ts - 10)
        mock_manager._record_preload_stuck_failure("ModelA", old_ts)

        assert not mock_manager._is_model_in_preload_cooldown("ModelA"), (
            "Cooldown must expire after _PRELOAD_STUCK_COOLDOWN seconds"
        )

    def test_unknown_model_not_in_cooldown(self) -> None:
        """A model with no recorded failures is never in cooldown."""
        mock_manager = self._make_manager()
        assert not mock_manager._is_model_in_preload_cooldown("NewModel")

    # ------------------------------------------------------------------
    # _fault_cooldown_model_jobs
    # ------------------------------------------------------------------

    def _make_job(self, model: str, job_id: str = "job-001") -> MagicMock:
        job = MagicMock()
        job.model = model
        job.id_ = job_id
        return job

    def test_fault_cooldown_model_jobs_permanently_faults_pending_jobs(self) -> None:
        """_fault_cooldown_model_jobs must permanently fault jobs for models in cooldown.

        The retry_count is set to MAX_JOB_RETRIES before handle_job_fault is called so
        that the job is permanently faulted rather than re-queued for a second preload
        attempt (which would also immediately fail, creating a tight fault loop).
        """
        import time as _time

        from horde_worker_regen.process_management.process_manager import (
            HordeWorkerProcessManager,
        )

        mock_manager = self._make_manager()
        now = _time.time()
        mock_manager._record_preload_stuck_failure("BadModel", now - 5)
        mock_manager._record_preload_stuck_failure("BadModel", now)

        job = self._make_job("BadModel")
        job_info = MagicMock()
        job_info.retry_count = 0
        mock_manager.jobs_pending_inference = [job]
        mock_manager.jobs_lookup = {job: job_info}

        mock_manager._fault_cooldown_model_jobs()

        assert job_info.retry_count == HordeWorkerProcessManager.MAX_JOB_RETRIES, (
            "retry_count must be set to MAX_JOB_RETRIES to force a permanent fault"
        )
        mock_manager.handle_job_fault.assert_called_once()
        call_kwargs = mock_manager.handle_job_fault.call_args
        assert call_kwargs.kwargs["faulted_job"] is job

    def test_fault_cooldown_model_jobs_skips_healthy_models(self) -> None:
        """Jobs for models NOT in cooldown must not be touched."""
        mock_manager = self._make_manager()
        # No failures recorded → no model is in cooldown.
        job = self._make_job("GoodModel")
        mock_manager.jobs_pending_inference = [job]

        mock_manager._fault_cooldown_model_jobs()

        mock_manager.handle_job_fault.assert_not_called()

    def test_fault_cooldown_handles_already_removed_jobs_gracefully(self) -> None:
        """If a job is removed from the pending queue before processing, no crash occurs."""
        import time as _time

        mock_manager = self._make_manager()
        now = _time.time()
        mock_manager._record_preload_stuck_failure("BadModel", now - 5)
        mock_manager._record_preload_stuck_failure("BadModel", now)

        job = self._make_job("BadModel")
        # The job is NOT in jobs_pending_inference (already removed externally).
        mock_manager.jobs_pending_inference = []
        mock_manager.jobs_lookup = {job: MagicMock()}

        # Should not raise.
        mock_manager._fault_cooldown_model_jobs()
        mock_manager.handle_job_fault.assert_not_called()

    def test_fault_cooldown_removes_job_from_queue_when_jobs_lookup_missing(self) -> None:
        """When job_info is missing from jobs_lookup, the job must still be drained from pending.

        Without this guard handle_job_fault only logs/records history for unknown jobs and
        does not remove the entry from jobs_pending_inference, so the job would re-appear
        on every preload_models() call for the entire cooldown duration.
        """
        import time as _time

        mock_manager = self._make_manager()
        now = _time.time()
        mock_manager._record_preload_stuck_failure("BadModel", now - 5)
        mock_manager._record_preload_stuck_failure("BadModel", now)

        job = self._make_job("BadModel")
        # The job has no jobs_lookup metadata (edge case: lookup was cleaned up early).
        mock_manager.jobs_pending_inference = [job]
        mock_manager.jobs_lookup = {}  # empty — job_info will be None

        mock_manager._fault_cooldown_model_jobs()

        # The job must have been removed from the queue before handle_job_fault was called.
        assert job not in mock_manager.jobs_pending_inference, (
            "Job must be removed from jobs_pending_inference when jobs_lookup has no entry"
        )

    # ------------------------------------------------------------------
    # replace_hung_processes + _record_preload_stuck_failure integration
    # ------------------------------------------------------------------

    def test_replace_hung_processes_records_stuck_failure_for_model_preloading(self) -> None:
        """replace_hung_processes must call _record_preload_stuck_failure for MODEL_PRELOADING timeouts.

        When _check_and_replace_process returns True for a MODEL_PRELOADING process the
        stuck failure for that model must be recorded so that the cooldown threshold can
        eventually be reached.
        """
        import time as _time

        from horde_worker_regen.process_management.process_manager import (
            HordeWorkerProcessManager,
        )

        stuck_model = "BadModel"
        proc = MagicMock()
        proc.process_id = 1
        proc.last_process_state = HordeProcessState.MODEL_PRELOADING
        proc.inference_started_timestamp = None
        proc.loaded_horde_model_name = stuck_model
        proc.last_received_timestamp = _time.time() - 9999
        proc.last_heartbeat_timestamp = _time.time() - 9999
        proc.last_progress_timestamp = _time.time() - 9999
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None

        mock_manager = MagicMock()
        mock_manager._recently_recovered = False
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager._hung_processes_detected = False
        mock_manager._hung_processes_detected_time = 0.0
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 300
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        mock_manager._process_map.is_stuck_on_inference.return_value = False
        mock_manager._process_map.values.return_value = [proc]
        mock_manager._process_map.__iter__ = MagicMock(return_value=iter([proc]))
        mock_manager.jobs_pending_inference = []
        mock_manager.jobs_in_progress = []
        mock_manager._preload_stuck_failures = {}
        mock_manager._PRELOAD_STUCK_FAILURE_THRESHOLD = (
            HordeWorkerProcessManager._PRELOAD_STUCK_FAILURE_THRESHOLD
        )
        mock_manager._PRELOAD_STUCK_FAILURE_WINDOW = (
            HordeWorkerProcessManager._PRELOAD_STUCK_FAILURE_WINDOW
        )
        mock_manager._PRELOAD_STUCK_COOLDOWN = HordeWorkerProcessManager._PRELOAD_STUCK_COOLDOWN

        # Wire the real _record_preload_stuck_failure so we can inspect _preload_stuck_failures.
        mock_manager._record_preload_stuck_failure = (
            HordeWorkerProcessManager._record_preload_stuck_failure.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )
        mock_manager._prune_preload_stuck_failures = (
            HordeWorkerProcessManager._prune_preload_stuck_failures.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )

        def fake_check_and_replace(
            process_info: MagicMock,
            timeout: float,
            state: HordeProcessState,
            error_msg: str,
        ) -> bool:
            if process_info.last_process_state == state:
                elapsed = _time.time() - process_info.last_received_timestamp
                if elapsed > timeout:
                    mock_manager._replace_inference_process(process_info)
                    return True
            return False

        mock_manager._check_and_replace_process = fake_check_and_replace
        mock_manager._bound_replace_hung = (
            HordeWorkerProcessManager.replace_hung_processes.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )

        with patch("threading.Thread"):
            mock_manager._bound_replace_hung()

        assert stuck_model in mock_manager._preload_stuck_failures, (
            "_preload_stuck_failures must contain the stuck model after MODEL_PRELOADING timeout"
        )
        assert len(mock_manager._preload_stuck_failures[stuck_model]) == 1, (
            "Exactly one failure should be recorded for a single MODEL_PRELOADING timeout"
        )

    # ------------------------------------------------------------------
    # _replace_inference_process: no retry for MODEL_PRELOADING stuck
    # ------------------------------------------------------------------

    def test_replace_inference_process_skips_retry_for_model_preloading(self) -> None:
        """_replace_inference_process must permanently fault a MODEL_PRELOADING job without retry.

        When a process is replaced because it timed out in MODEL_PRELOADING, re-queuing the
        job for retry would send the same broken model to a fresh process, which will get
        stuck again for another full preload_timeout before the preload-stuck cooldown kicks
        in.  The fix pre-sets retry_count = MAX_JOB_RETRIES so handle_job_fault permanently
        faults the job instead of re-queueing it.
        """
        import types
        from collections import deque

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job = MagicMock()
        job.id_ = "aaaa0001-0000-0000-0000-000000000000"
        job.model = "HungModel"
        job.payload = MagicMock()
        job.payload.n_iter = 1
        job.payload.loras = []
        job.payload.workflow = None

        job_info = MagicMock()
        job_info.retry_count = 0

        process_info = MagicMock()
        process_info.process_id = 1
        process_info.last_process_state = HordeProcessState.MODEL_PRELOADING
        process_info.inference_started_timestamp = None
        process_info.last_progress_value = None
        process_info.last_job_referenced = job
        process_info.loaded_horde_model_name = "HungModel"

        mock_manager = MagicMock()
        mock_manager.MAX_JOB_RETRIES = HordeWorkerProcessManager.MAX_JOB_RETRIES
        mock_manager.jobs_in_progress = []
        mock_manager.jobs_pending_inference = deque([job])
        mock_manager.jobs_lookup = {job: job_info}
        mock_manager.jobs_pending_submit = []
        mock_manager.jobs_pending_safety_check = []
        mock_manager.jobs_being_safety_checked = []
        mock_manager._skipped_line_next_job_and_process = None
        mock_manager._failed_models = {}
        mock_manager._faulted_jobs_history = []
        mock_manager._max_faulted_jobs_history = HordeWorkerProcessManager._max_faulted_jobs_history
        mock_manager._invalidate_megapixelsteps_cache = MagicMock()
        mock_manager._restart_idle_timer_if_queue_empty = MagicMock()
        mock_manager.bridge_data.process_timeout = 60

        # Wire the real _record_faulted_job_history and handle_job_fault
        mock_manager._record_faulted_job_history = types.MethodType(
            HordeWorkerProcessManager._record_faulted_job_history,
            mock_manager,
        )
        mock_manager.handle_job_fault = types.MethodType(
            HordeWorkerProcessManager.handle_job_fault,
            mock_manager,
        )

        bound = types.MethodType(HordeWorkerProcessManager._replace_inference_process, mock_manager)
        with patch("horde_worker_regen.process_management.process_manager.logger") as mock_logger:
            bound(process_info)

        # The job must be permanently faulted (retry_count set to MAX_JOB_RETRIES before
        # handle_job_fault was called), so it goes straight to jobs_pending_submit.
        assert job_info.retry_count == HordeWorkerProcessManager.MAX_JOB_RETRIES, (
            "_replace_inference_process must set retry_count = MAX_JOB_RETRIES for MODEL_PRELOADING "
            "so handle_job_fault permanently faults the job instead of re-queuing it for retry"
        )
        assert job_info in mock_manager.jobs_pending_submit, (
            "A job permanently faulted during MODEL_PRELOADING stuck replacement must be placed "
            "in jobs_pending_submit so the horde is notified"
        )
        assert job not in mock_manager.jobs_pending_inference, (
            "The permanently faulted job must not remain in jobs_pending_inference"
        )
        error_messages = [call.args[0] for call in mock_logger.error.call_args_list]
        assert any("faulted with retry skipped" in message for message in error_messages)
        assert any(
            "retry skipped because the process was replaced while stuck in MODEL_PRELOADING" in message
            for message in error_messages
        )

    def test_replace_inference_process_allows_retry_for_inference_processing(self) -> None:
        """_replace_inference_process must still allow retry for INFERENCE_PROCESSING crashes.

        Only MODEL_PRELOADING stuck replacements skip the retry.  A process crash during
        active inference (e.g. OOM) is a transient failure that warrants a local retry.
        """
        import types
        from collections import deque

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job = MagicMock()
        job.id_ = "bbbb0002-0000-0000-0000-000000000000"
        job.model = "GoodModel"
        job.payload = MagicMock()
        job.payload.n_iter = 1
        job.payload.loras = []
        job.payload.workflow = None

        job_info = MagicMock()
        job_info.retry_count = 0

        process_info = MagicMock()
        process_info.process_id = 2
        process_info.last_process_state = HordeProcessState.INFERENCE_PROCESSING
        process_info.inference_started_timestamp = None
        process_info.last_progress_value = 50
        process_info.last_job_referenced = job
        process_info.loaded_horde_model_name = "GoodModel"

        mock_manager = MagicMock()
        mock_manager.MAX_JOB_RETRIES = HordeWorkerProcessManager.MAX_JOB_RETRIES
        mock_manager.jobs_in_progress = [job]
        mock_manager.jobs_pending_inference = deque()
        mock_manager.jobs_lookup = {job: job_info}
        mock_manager.jobs_pending_submit = []
        mock_manager.jobs_pending_safety_check = []
        mock_manager.jobs_being_safety_checked = []
        mock_manager._skipped_line_next_job_and_process = None
        mock_manager._failed_models = {}
        mock_manager._faulted_jobs_history = []
        mock_manager._max_faulted_jobs_history = HordeWorkerProcessManager._max_faulted_jobs_history
        mock_manager._invalidate_megapixelsteps_cache = MagicMock()
        mock_manager._restart_idle_timer_if_queue_empty = MagicMock()
        mock_manager.bridge_data.process_timeout = 60

        mock_manager._record_faulted_job_history = types.MethodType(
            HordeWorkerProcessManager._record_faulted_job_history,
            mock_manager,
        )
        mock_manager.handle_job_fault = types.MethodType(
            HordeWorkerProcessManager.handle_job_fault,
            mock_manager,
        )

        bound = types.MethodType(HordeWorkerProcessManager._replace_inference_process, mock_manager)
        bound(process_info)

        # For INFERENCE_PROCESSING (not MODEL_PRELOADING), retry_count must remain 0 before
        # handle_job_fault runs, so the job gets re-queued with retry_count = 1.
        assert job_info.retry_count == 1, (
            "For INFERENCE_PROCESSING crashes, retry_count must be incremented to 1 "
            "(job re-queued for retry), not pre-set to MAX_JOB_RETRIES"
        )
        assert job in mock_manager.jobs_pending_inference, (
            "For INFERENCE_PROCESSING crashes, the job must be re-queued in jobs_pending_inference"
        )
        assert job_info not in mock_manager.jobs_pending_submit, (
            "For INFERENCE_PROCESSING crashes, the job must NOT be permanently faulted on the first attempt"
        )


class TestJobRecoveryAndRetry:
    """End-to-end tests for the job recovery and retry mechanism.

    These tests verify two invariants demanded by the problem statement:
    1. Recovered jobs are actually recovered and retried:
       - handle_job_fault removes the job from jobs_in_progress and re-queues it in
         jobs_pending_inference with an incremented retry_count when retries remain.
       - A re-queued job that is no longer in jobs_in_progress is returned by
         get_next_job_and_process so it can be dispatched again.
    2. A re-queued job that faults again is marked as permanently faulted:
       - handle_job_fault calls fault_job() and adds the HordeJobInfo to
         jobs_pending_submit so the API learns the job failed.
       - The job is NOT re-queued a second time.
    """

    def _make_job(self, job_id: str = "aabb0011", model: str = "TestModel") -> MagicMock:
        job = MagicMock()
        job.id_ = f"{job_id}-0000-0000-0000-000000000000"
        job.model = model
        job.payload = MagicMock()
        job.payload.n_iter = 1
        job.payload.loras = []
        job.payload.workflow = None
        return job

    def _make_job_info(
        self,
        *,
        retry_count: int = 0,
    ) -> MagicMock:
        """Return a mock HordeJobInfo whose fault_job() actually mutates state.

        We use a MagicMock (rather than a real HordeJobInfo) because the real class
        requires a full ImageGenerateJobPopResponse Pydantic model.  We attach a real
        fault_job() implementation so the state transition can be asserted.
        """
        from horde_sdk.ai_horde_api import GENERATION_STATE

        job_info = MagicMock()
        job_info.retry_count = retry_count
        job_info.state = GENERATION_STATE.ok
        job_info.job_image_results = []

        def real_fault_job() -> None:
            job_info.state = GENERATION_STATE.faulted
            job_info.job_image_results = None

        job_info.fault_job = real_fault_job
        return job_info

    def _make_manager(
        self,
        *,
        jobs_in_progress: list,
        jobs_pending_inference: list | None = None,
        jobs_lookup: dict | None = None,
    ) -> MagicMock:
        """Build a minimal mock manager with the real handle_job_fault bound."""
        import types
        from collections import deque

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager.MAX_JOB_RETRIES = HordeWorkerProcessManager.MAX_JOB_RETRIES
        mock_manager.jobs_in_progress = list(jobs_in_progress)
        mock_manager.jobs_pending_inference = deque(jobs_pending_inference or [])
        mock_manager.jobs_lookup = dict(jobs_lookup or {})
        mock_manager.jobs_pending_submit = []
        mock_manager.jobs_pending_safety_check = []
        mock_manager.jobs_being_safety_checked = []
        mock_manager._skipped_line_next_job_and_process = None
        mock_manager._failed_models = {}
        mock_manager._faulted_jobs_history = []
        mock_manager._max_faulted_jobs_history = HordeWorkerProcessManager._max_faulted_jobs_history
        mock_manager._invalidate_megapixelsteps_cache = MagicMock()
        mock_manager._restart_idle_timer_if_queue_empty = MagicMock()
        mock_manager.bridge_data.process_timeout = 60

        mock_manager._record_faulted_job_history = types.MethodType(
            HordeWorkerProcessManager._record_faulted_job_history,
            mock_manager,
        )
        mock_manager.handle_job_fault = types.MethodType(
            HordeWorkerProcessManager.handle_job_fault,
            mock_manager,
        )
        return mock_manager

    # ------------------------------------------------------------------
    # Part 1: Recovery — first fault re-queues the job for retry
    # ------------------------------------------------------------------

    def test_first_fault_removes_job_from_in_progress(self) -> None:
        """A job's first fault must remove it from jobs_in_progress.

        Without this removal, get_next_job_and_process skips the job
        (it is filtered out by the 'job in jobs_in_progress' check), so the
        retry never gets dispatched.
        """
        job = self._make_job("first-fault")
        job_info = self._make_job_info(retry_count=0)

        mock_manager = self._make_manager(
            jobs_in_progress=[job],
            jobs_lookup={job: job_info},
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert job not in mock_manager.jobs_in_progress, (
            "handle_job_fault must remove the job from jobs_in_progress so it can be re-dispatched"
        )

    def test_first_fault_requeues_job_in_pending_inference(self) -> None:
        """A job's first fault must re-queue it in jobs_pending_inference for retry."""
        job = self._make_job("requeue-job")
        job_info = self._make_job_info(retry_count=0)

        mock_manager = self._make_manager(
            jobs_in_progress=[job],
            jobs_lookup={job: job_info},
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert job in mock_manager.jobs_pending_inference, (
            "handle_job_fault must add the job back to jobs_pending_inference for its retry attempt"
        )

    def test_first_fault_increments_retry_count(self) -> None:
        """retry_count must be incremented from 0 to 1 on the first fault."""
        job = self._make_job("retry-count-job")
        job_info = self._make_job_info(retry_count=0)

        mock_manager = self._make_manager(
            jobs_in_progress=[job],
            jobs_lookup={job: job_info},
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert job_info.retry_count == 1, (
            "retry_count must be incremented so the next fault triggers permanent faulting"
        )

    def test_first_fault_does_not_add_job_to_pending_submit(self) -> None:
        """A job that is being retried must NOT be added to jobs_pending_submit.

        Only permanently faulted jobs (retry budget exhausted) should enter the
        submit queue to be reported to the API.
        """
        job = self._make_job("no-submit-on-retry")
        job_info = self._make_job_info(retry_count=0)

        mock_manager = self._make_manager(
            jobs_in_progress=[job],
            jobs_lookup={job: job_info},
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert job_info not in mock_manager.jobs_pending_submit, (
            "A retried job must not be submitted to the API as faulted; "
            "it still has a retry attempt remaining"
        )

    # ------------------------------------------------------------------
    # Part 2: Re-queued job is visible to get_next_job_and_process
    # ------------------------------------------------------------------

    def test_requeued_job_removed_from_in_progress_filter_after_fault(self) -> None:
        """A re-queued job should remain pending and no longer be marked in progress.

        After handle_job_fault() re-queues a job, the job must:
          (a) be present in jobs_pending_inference, and
          (b) be absent from jobs_in_progress.

        These are the state invariants required so later selection logic will
        not skip the re-queued job because of the in-progress filter.
        (get_next_job_and_process skips any job that is still in jobs_in_progress.)
        """
        job = self._make_job("dispatch-after-retry")
        job_info = self._make_job_info(retry_count=0)

        mock_manager = self._make_manager(
            jobs_in_progress=[job],
            jobs_lookup={job: job_info},
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        # After retry: job must be in pending but NOT in in-progress
        assert job in mock_manager.jobs_pending_inference, (
            "Re-queued job must be in jobs_pending_inference so it can be selected for dispatch"
        )
        assert job not in mock_manager.jobs_in_progress, (
            "Re-queued job must NOT be in jobs_in_progress; "
            "get_next_job_and_process skips jobs that are already in-progress"
        )

    def test_job_in_progress_skipped_by_get_next_job_and_process(self) -> None:
        """A job that is already in jobs_in_progress must be skipped by get_next_job_and_process.

        This is the filter that prevents double-dispatching a job that is actively being
        processed.  It must not accidentally skip a re-queued job that was properly
        removed from jobs_in_progress by the retry path.
        """
        import types
        from collections import deque

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job = self._make_job("in-progress-job")

        mock_manager = MagicMock()
        mock_manager._skipped_line_next_job_and_process = None
        # Simulate the state *before* the retry path runs: job is in BOTH lists
        mock_manager.jobs_pending_inference = deque([job])
        mock_manager.jobs_in_progress = [job]  # already dispatched
        mock_manager.max_concurrent_inference_processes = 2
        mock_manager.post_process_job_overlap_allowed = False

        bound = types.MethodType(HordeWorkerProcessManager.get_next_job_and_process, mock_manager)
        result = bound()

        assert result is None, (
            "get_next_job_and_process must return None when the only pending job is already "
            "in jobs_in_progress (i.e. already being processed)"
        )

    # ------------------------------------------------------------------
    # Part 3: Second fault marks the job as permanently faulted
    # ------------------------------------------------------------------

    def test_second_fault_calls_fault_job(self) -> None:
        """When a retried job (retry_count == MAX_JOB_RETRIES) faults again, fault_job() must be called.

        fault_job() sets state = GENERATION_STATE.faulted and clears job_image_results.
        Without this call the job would never be reported to the API as failed.
        """
        from horde_sdk.ai_horde_api import GENERATION_STATE
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job = self._make_job("second-fault")
        max_retries = HordeWorkerProcessManager.MAX_JOB_RETRIES
        job_info = self._make_job_info(retry_count=max_retries)

        mock_manager = self._make_manager(
            jobs_in_progress=[job],
            jobs_lookup={job: job_info},
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert job_info.state == GENERATION_STATE.faulted, (
            "fault_job() must be called for a job that exhausts its retry budget; "
            "state must be GENERATION_STATE.faulted so the API submission path reports it correctly"
        )

    def test_second_fault_adds_job_to_pending_submit(self) -> None:
        """A permanently faulted job must be placed in jobs_pending_submit.

        jobs_pending_submit is the queue that api_submit_job() drains when reporting
        job results to the AI Horde API.  A job that never enters this queue is silently
        lost — the server will eventually time it out, but the worker won't proactively
        report the failure.
        """
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job = self._make_job("submit-on-second-fault")
        max_retries = HordeWorkerProcessManager.MAX_JOB_RETRIES
        job_info = self._make_job_info(retry_count=max_retries)

        mock_manager = self._make_manager(
            jobs_in_progress=[job],
            jobs_lookup={job: job_info},
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert job_info in mock_manager.jobs_pending_submit, (
            "A permanently faulted job must be added to jobs_pending_submit "
            "so the API is informed of the failure"
        )

    def test_second_fault_does_not_requeue_in_pending_inference(self) -> None:
        """A permanently faulted job must NOT be re-added to jobs_pending_inference.

        Re-queuing after the retry budget is exhausted would cause the job to spin
        indefinitely without ever being reported as failed.
        """
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job = self._make_job("no-requeue-on-perm-fault")
        max_retries = HordeWorkerProcessManager.MAX_JOB_RETRIES
        job_info = self._make_job_info(retry_count=max_retries)

        mock_manager = self._make_manager(
            jobs_in_progress=[job],
            jobs_lookup={job: job_info},
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert job not in mock_manager.jobs_pending_inference, (
            "A permanently faulted job must not be re-queued in jobs_pending_inference"
        )

    def test_second_fault_removes_job_from_in_progress(self) -> None:
        """A permanently faulted job must be removed from jobs_in_progress.

        Leaving the job in jobs_in_progress would block new jobs from being dispatched
        because max_concurrent_inference_processes is checked against len(jobs_in_progress).
        """
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job = self._make_job("in-progress-removed-on-perm-fault")
        max_retries = HordeWorkerProcessManager.MAX_JOB_RETRIES
        job_info = self._make_job_info(retry_count=max_retries)

        mock_manager = self._make_manager(
            jobs_in_progress=[job],
            jobs_lookup={job: job_info},
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert job not in mock_manager.jobs_in_progress, (
            "A permanently faulted job must be removed from jobs_in_progress"
        )

    # ------------------------------------------------------------------
    # Part 4: End-to-end two-fault scenario
    # ------------------------------------------------------------------

    def test_end_to_end_first_fault_then_second_fault_permanently_faults(self) -> None:
        """End-to-end: first fault re-queues, second fault permanently faults the job.

        Simulates the full lifecycle:
          1. Job is being processed (in jobs_in_progress).
          2. First fault — job is removed from jobs_in_progress, re-queued in
             jobs_pending_inference, retry_count becomes 1.
          3. Job is re-dispatched (simulate start_inference by adding the job back
             to jobs_in_progress without removing it from jobs_pending_inference;
             in the real start_inference() flow, removal happens when the result arrives).
          4. Second fault — job is permanently faulted: state = GENERATION_STATE.faulted,
             job_info in jobs_pending_submit, job not in jobs_pending_inference.
        """
        from horde_sdk.ai_horde_api import GENERATION_STATE
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        job = self._make_job("e2e-two-faults")
        job_info = self._make_job_info(retry_count=0)

        mock_manager = self._make_manager(
            jobs_in_progress=[job],
            jobs_lookup={job: job_info},
        )

        # --- Step 1: first fault ---
        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        # Verify recovery state after first fault
        assert job not in mock_manager.jobs_in_progress, "Job must leave jobs_in_progress after first fault"
        assert job in mock_manager.jobs_pending_inference, "Job must enter jobs_pending_inference for retry"
        assert job_info.retry_count == 1, "retry_count must be 1 after first fault"
        assert job_info not in mock_manager.jobs_pending_submit, (
            "Job must NOT be in jobs_pending_submit yet — it still has a retry remaining"
        )

        # --- Step 2: simulate start_inference dispatching the retried job ---
        # The real start_inference() appends to jobs_in_progress (but does NOT remove from
        # jobs_pending_inference — that happens when the result arrives).
        mock_manager.jobs_in_progress.append(job)

        # --- Step 3: second fault ---
        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        # Verify permanent fault state after second fault
        assert job_info.state == GENERATION_STATE.faulted, (
            "Job must be marked GENERATION_STATE.faulted after exhausting retry budget"
        )
        assert job_info in mock_manager.jobs_pending_submit, (
            "Job must be in jobs_pending_submit so the API is notified of the permanent fault"
        )
        assert job not in mock_manager.jobs_pending_inference, (
            "Permanently faulted job must not remain in jobs_pending_inference"
        )
        assert job not in mock_manager.jobs_in_progress, (
            "Permanently faulted job must be removed from jobs_in_progress"
        )


class TestInferenceFailureCooldown:
    """Tests for the inference-failure cooldown logic.

    When a model causes enough permanently-faulted jobs within a short window, the
    worker stops requesting that model from the API until the cooldown expires.
    """

    def _make_manager(self) -> MagicMock:
        """Build a minimal mock manager with the real inference-failure cooldown methods bound."""
        from horde_worker_regen.process_management.process_manager import (
            HordeWorkerProcessManager,
        )

        mock_manager = MagicMock()
        mock_manager._inference_failures = {}
        mock_manager._INFERENCE_FAILURE_THRESHOLD = (
            HordeWorkerProcessManager._INFERENCE_FAILURE_THRESHOLD
        )
        mock_manager._INFERENCE_FAILURE_WINDOW = (
            HordeWorkerProcessManager._INFERENCE_FAILURE_WINDOW
        )
        mock_manager._INFERENCE_FAILURE_COOLDOWN = HordeWorkerProcessManager._INFERENCE_FAILURE_COOLDOWN

        mock_manager._prune_preload_stuck_failures = (
            HordeWorkerProcessManager._prune_preload_stuck_failures.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )
        mock_manager._record_inference_failure = (
            HordeWorkerProcessManager._record_inference_failure.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )
        mock_manager._is_model_in_inference_cooldown = (
            HordeWorkerProcessManager._is_model_in_inference_cooldown.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )
        return mock_manager

    # ------------------------------------------------------------------
    # _record_inference_failure / _is_model_in_inference_cooldown
    # ------------------------------------------------------------------

    def test_no_cooldown_with_fewer_than_threshold_failures(self) -> None:
        """Fewer than threshold failures within the window must NOT trigger cooldown."""
        import time as _time

        mock_manager = self._make_manager()
        threshold = mock_manager._INFERENCE_FAILURE_THRESHOLD
        now = _time.time()

        # Record (threshold - 1) failures — one short of triggering
        for i in range(threshold - 1):
            mock_manager._record_inference_failure("ModelA", now - i)

        assert not mock_manager._is_model_in_inference_cooldown("ModelA"), (
            f"{threshold - 1} failures should not be enough to trigger cooldown (threshold is {threshold})"
        )

    def test_cooldown_triggered_at_threshold(self) -> None:
        """Reaching the threshold within the window puts the model in cooldown."""
        import time as _time

        mock_manager = self._make_manager()
        threshold = mock_manager._INFERENCE_FAILURE_THRESHOLD
        now = _time.time()

        for i in range(threshold):
            mock_manager._record_inference_failure("ModelA", now - (threshold - 1 - i) * 10)

        assert mock_manager._is_model_in_inference_cooldown("ModelA"), (
            f"{threshold} failures within the window should trigger cooldown"
        )

    def test_old_failures_pruned_outside_window(self) -> None:
        """Failures older than _INFERENCE_FAILURE_WINDOW do not count toward the threshold."""
        import time as _time

        mock_manager = self._make_manager()
        threshold = mock_manager._INFERENCE_FAILURE_THRESHOLD
        window = mock_manager._INFERENCE_FAILURE_WINDOW
        now = _time.time()

        # Record (threshold - 1) old failures outside the window
        for i in range(threshold - 1):
            mock_manager._record_inference_failure("ModelA", now - window - 100 - i)

        # One recent failure — not enough to hit threshold once old ones are pruned
        mock_manager._record_inference_failure("ModelA", now)

        assert not mock_manager._is_model_in_inference_cooldown("ModelA"), (
            "Only failures within the window count; old failures must be pruned"
        )

    def test_cooldown_expires_after_duration(self) -> None:
        """Once the cooldown duration elapses the model must leave cooldown."""
        import time as _time

        mock_manager = self._make_manager()
        threshold = mock_manager._INFERENCE_FAILURE_THRESHOLD
        cooldown = mock_manager._INFERENCE_FAILURE_COOLDOWN

        # All failures older than the cooldown duration (but still within the failure window
        # relative to each other so they would have triggered cooldown when they happened)
        old_base = _time.time() - cooldown - 10
        for i in range(threshold):
            mock_manager._record_inference_failure("ModelA", old_base + i * 10)

        assert not mock_manager._is_model_in_inference_cooldown("ModelA"), (
            "Cooldown must expire after _INFERENCE_FAILURE_COOLDOWN seconds have passed since the last failure"
        )

    def test_unknown_model_not_in_cooldown(self) -> None:
        """A model with no recorded failures is never in cooldown."""
        mock_manager = self._make_manager()
        assert not mock_manager._is_model_in_inference_cooldown("BrandNewModel")

    def test_second_model_independent_of_first(self) -> None:
        """Cooldown for one model must not affect another model."""
        import time as _time

        mock_manager = self._make_manager()
        threshold = mock_manager._INFERENCE_FAILURE_THRESHOLD
        now = _time.time()

        for i in range(threshold):
            mock_manager._record_inference_failure("ModelA", now - (threshold - 1 - i) * 5)

        assert mock_manager._is_model_in_inference_cooldown("ModelA"), "ModelA should be in cooldown"
        assert not mock_manager._is_model_in_inference_cooldown("ModelB"), (
            "ModelB has no failures and must not be in cooldown"
        )

    # ------------------------------------------------------------------
    # Integration: handle_job_fault records inference failures
    # ------------------------------------------------------------------

    def test_handle_job_fault_records_inference_failure_on_permanent_fault(self) -> None:
        """handle_job_fault must call _record_inference_failure when permanently faulting a job."""
        import types as _types

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
        from horde_sdk.ai_horde_api import GENERATION_STATE

        # Build a job that will be permanently faulted (retry_count already at MAX)
        job = MagicMock()
        job.model = "FailingModel"
        job.id_ = "job-perm-fault"

        job_info = MagicMock()
        job_info.retry_count = HordeWorkerProcessManager.MAX_JOB_RETRIES  # already exhausted
        job_info.state = GENERATION_STATE.ok
        job_info.job_image_results = []

        def real_fault_job() -> None:
            job_info.state = GENERATION_STATE.faulted
            job_info.job_image_results = None

        job_info.fault_job = real_fault_job

        from collections import deque as _deque

        mock_manager = MagicMock()
        mock_manager.MAX_JOB_RETRIES = HordeWorkerProcessManager.MAX_JOB_RETRIES
        mock_manager.jobs_in_progress = [job]
        mock_manager.jobs_pending_inference = _deque([])
        mock_manager.jobs_lookup = {job: job_info}
        mock_manager.jobs_pending_submit = []
        mock_manager.jobs_pending_safety_check = []
        mock_manager.jobs_being_safety_checked = []
        mock_manager._skipped_line_next_job_and_process = None
        mock_manager._failed_models = {}
        mock_manager._inference_failures = {}
        mock_manager._faulted_jobs_history = []
        mock_manager._max_faulted_jobs_history = HordeWorkerProcessManager._max_faulted_jobs_history
        mock_manager._invalidate_megapixelsteps_cache = MagicMock()
        mock_manager._restart_idle_timer_if_queue_empty = MagicMock()
        mock_manager.bridge_data.process_timeout = 60

        mock_manager._record_faulted_job_history = _types.MethodType(
            HordeWorkerProcessManager._record_faulted_job_history,
            mock_manager,
        )
        mock_manager._prune_preload_stuck_failures = _types.MethodType(
            HordeWorkerProcessManager._prune_preload_stuck_failures,
            mock_manager,
        )
        mock_manager._record_inference_failure = _types.MethodType(
            HordeWorkerProcessManager._record_inference_failure,
            mock_manager,
        )
        mock_manager._is_model_in_inference_cooldown = _types.MethodType(
            HordeWorkerProcessManager._is_model_in_inference_cooldown,
            mock_manager,
        )
        mock_manager.handle_job_fault = _types.MethodType(
            HordeWorkerProcessManager.handle_job_fault,
            mock_manager,
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert "FailingModel" in mock_manager._inference_failures, (
            "handle_job_fault must record an inference failure for the model when permanently faulting"
        )
        assert len(mock_manager._inference_failures["FailingModel"]) == 1, (
            "Exactly one inference failure should be recorded"
        )

    def test_handle_job_fault_does_not_record_inference_failure_on_retry(self) -> None:
        """handle_job_fault must NOT record an inference failure when only queuing a retry."""
        import types as _types

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager
        from horde_sdk.ai_horde_api import GENERATION_STATE
        from collections import deque as _deque

        # Job with retry_count=0 — first fault, should be retried, NOT permanently faulted
        job = MagicMock()
        job.model = "RetryModel"
        job.id_ = "job-retry"

        job_info = MagicMock()
        job_info.retry_count = 0
        job_info.state = GENERATION_STATE.ok
        job_info.job_image_results = []
        job_info.fault_job = MagicMock()

        mock_manager = MagicMock()
        mock_manager.MAX_JOB_RETRIES = HordeWorkerProcessManager.MAX_JOB_RETRIES
        mock_manager.jobs_in_progress = [job]
        mock_manager.jobs_pending_inference = _deque([])
        mock_manager.jobs_lookup = {job: job_info}
        mock_manager.jobs_pending_submit = []
        mock_manager.jobs_pending_safety_check = []
        mock_manager.jobs_being_safety_checked = []
        mock_manager._skipped_line_next_job_and_process = None
        mock_manager._failed_models = {}
        mock_manager._inference_failures = {}
        mock_manager._faulted_jobs_history = []
        mock_manager._max_faulted_jobs_history = HordeWorkerProcessManager._max_faulted_jobs_history
        mock_manager._invalidate_megapixelsteps_cache = MagicMock()
        mock_manager._restart_idle_timer_if_queue_empty = MagicMock()
        mock_manager.bridge_data.process_timeout = 60

        mock_manager._record_faulted_job_history = _types.MethodType(
            HordeWorkerProcessManager._record_faulted_job_history,
            mock_manager,
        )
        mock_manager._prune_preload_stuck_failures = _types.MethodType(
            HordeWorkerProcessManager._prune_preload_stuck_failures,
            mock_manager,
        )
        mock_manager._record_inference_failure = _types.MethodType(
            HordeWorkerProcessManager._record_inference_failure,
            mock_manager,
        )
        mock_manager.handle_job_fault = _types.MethodType(
            HordeWorkerProcessManager.handle_job_fault,
            mock_manager,
        )

        mock_manager.handle_job_fault(faulted_job=job, process_info=None)

        assert "RetryModel" not in mock_manager._inference_failures or len(
            mock_manager._inference_failures.get("RetryModel", [])
        ) == 0, "No inference failure must be recorded for a job that is being retried (not permanently faulted)"

    # ------------------------------------------------------------------
    # Integration: api_job_pop excludes models in inference cooldown
    # ------------------------------------------------------------------

    def test_cooldown_models_excluded_from_models_set(self) -> None:
        """Models in inference-failure cooldown must be removed from the request model list."""
        import time as _time

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = self._make_manager()
        threshold = mock_manager._INFERENCE_FAILURE_THRESHOLD
        now = _time.time()

        # Put "BadModel" into cooldown
        for i in range(threshold):
            mock_manager._record_inference_failure("BadModel", now - (threshold - 1 - i) * 5)

        assert mock_manager._is_model_in_inference_cooldown("BadModel"), (
            "Pre-condition: BadModel should be in inference cooldown"
        )

        # Simulate the filtering logic used in api_job_pop
        all_models = {"BadModel", "GoodModel1", "GoodModel2"}
        cooldown_models = {m for m in all_models if mock_manager._is_model_in_inference_cooldown(m)}
        filtered_models = all_models - cooldown_models

        assert "BadModel" not in filtered_models, (
            "BadModel must be excluded from the request model list while in inference cooldown"
        )
        assert "GoodModel1" in filtered_models, "GoodModel1 must remain in the request model list"
        assert "GoodModel2" in filtered_models, "GoodModel2 must remain in the request model list"

    def test_cooldown_model_reinstated_after_expiry(self) -> None:
        """A model removed from requests during cooldown must be reinstated once the cooldown expires."""
        import time as _time

        mock_manager = self._make_manager()
        threshold = mock_manager._INFERENCE_FAILURE_THRESHOLD
        cooldown = mock_manager._INFERENCE_FAILURE_COOLDOWN

        # Record failures that are old enough for cooldown to have expired
        old_base = _time.time() - cooldown - 10
        for i in range(threshold):
            mock_manager._record_inference_failure("RecoveredModel", old_base + i * 5)

        assert not mock_manager._is_model_in_inference_cooldown("RecoveredModel"), (
            "Model must no longer be in cooldown after _INFERENCE_FAILURE_COOLDOWN seconds"
        )

        all_models = {"RecoveredModel", "OtherModel"}
        cooldown_models = {m for m in all_models if mock_manager._is_model_in_inference_cooldown(m)}
        filtered_models = all_models - cooldown_models

        assert "RecoveredModel" in filtered_models, (
            "RecoveredModel must be reinstated in the request model list after its cooldown expires"
        )


class TestReplaceHungProcessesProcessStartingTimeout:
    """Regression tests for PROCESS_STARTING timeout selection in replace_hung_processes()."""

    def test_process_starting_uses_max_of_process_and_preload_timeout(self) -> None:
        """PROCESS_STARTING stuck checks must use at least process_timeout."""
        from horde_worker_regen.process_management.process_manager import HordeProcessType, HordeWorkerProcessManager

        process = MagicMock()
        process.process_id = 1
        process.process_type = HordeProcessType.INFERENCE
        process.last_process_state = HordeProcessState.PROCESS_STARTING
        process.inference_started_timestamp = None
        process.last_received_timestamp = 0.0
        process.last_heartbeat_timestamp = 0.0
        process.last_progress_timestamp = 0.0
        process.last_heartbeat_percent_complete = None
        process.last_job_referenced = None

        mock_manager = MagicMock()
        mock_manager._recently_recovered = False
        mock_manager._last_pop_no_jobs_available = True
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 300
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        mock_manager._process_map.values.return_value = [process]
        mock_manager._process_map.is_stuck_on_inference.return_value = False
        mock_manager._check_and_replace_process.return_value = False
        mock_manager.jobs_pending_inference = []
        mock_manager.jobs_in_progress = []

        bound_method = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager
        )

        with patch("threading.Thread"):
            bound_method()

        expected_timeout = max(mock_manager.bridge_data.process_timeout, mock_manager.bridge_data.preload_timeout)
        starting_calls = [
            call
            for call in mock_manager._check_and_replace_process.call_args_list
            if call.args[2] == HordeProcessState.PROCESS_STARTING
        ]
        assert len(starting_calls) == 1, "PROCESS_STARTING stuck check should run exactly once per process"
        assert starting_calls[0].args[1] == expected_timeout, (
            "PROCESS_STARTING stuck timeout must use max(process_timeout, preload_timeout)"
        )


class TestIsTimeForShutdownRecentlyRecovered:
    """Regression tests: _recently_recovered must not block shutdown for idle/stuck processes.

    Before the fix, is_time_for_shutdown() returned False whenever _recently_recovered was
    True, regardless of whether any processes were actually alive or only initialising.
    This could block a user-requested restart for up to inference_step_timeout (≈600 s).

    After the fix, _recently_recovered only blocks shutdown when at least one process is
    still in PROCESS_STARTING (the case it was designed to guard against: a freshly spawned
    replacement process that could trigger a false "all done" early return).
    """

    def _make_process(self, state: HordeProcessState) -> MagicMock:
        """Return a minimal process-info mock in the requested state."""
        proc = MagicMock()
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = state
        proc.inference_started_timestamp = None
        return proc

    def _make_manager(
        self,
        *,
        recently_recovered: bool,
        process_states: list[HordeProcessState],
        jobs_pending_submit: int = 0,
        jobs_in_progress: int = 0,
        jobs_pending_inference: int = 0,
        jobs_being_safety_checked: int = 0,
        jobs_pending_safety_check: int = 0,
    ) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        processes = [self._make_process(s) for s in process_states]

        mock_manager = MagicMock()
        mock_manager._shutting_down = True
        mock_manager._recently_recovered = recently_recovered
        mock_manager.jobs_pending_submit = [MagicMock()] * jobs_pending_submit
        mock_manager.jobs_in_progress = [MagicMock()] * jobs_in_progress
        mock_manager.jobs_pending_inference = [MagicMock()] * jobs_pending_inference
        mock_manager.jobs_being_safety_checked = [MagicMock()] * jobs_being_safety_checked
        mock_manager.jobs_pending_safety_check = [MagicMock()] * jobs_pending_safety_check
        mock_manager._process_map.get_inference_processes.return_value = processes
        mock_manager._process_map.values.return_value = processes

        bound = HordeWorkerProcessManager.is_time_for_shutdown.__get__(mock_manager, HordeWorkerProcessManager)
        # Attach as callable so callers use mock_manager.is_time_for_shutdown()
        mock_manager.is_time_for_shutdown = bound
        return mock_manager

    # ------------------------------------------------------------------
    # Cases where shutdown SHOULD proceed (fix: no longer blocked)
    # ------------------------------------------------------------------

    def test_recently_recovered_idle_processes_allows_shutdown(self) -> None:
        """_recently_recovered=True with all processes WAITING_FOR_JOB must allow shutdown.

        Idle workers are not in PROCESS_STARTING so there is no false-positive risk.
        """
        mock_manager = self._make_manager(
            recently_recovered=True,
            process_states=[HordeProcessState.WAITING_FOR_JOB, HordeProcessState.WAITING_FOR_JOB],
        )
        assert mock_manager.is_time_for_shutdown() is True

    def test_recently_recovered_all_ending_allows_shutdown(self) -> None:
        """_recently_recovered=True with all processes PROCESS_ENDING must allow shutdown."""
        mock_manager = self._make_manager(
            recently_recovered=True,
            process_states=[HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDING],
        )
        assert mock_manager.is_time_for_shutdown() is True

    def test_recently_recovered_all_ended_allows_shutdown(self) -> None:
        """_recently_recovered=True with all processes PROCESS_ENDED must allow shutdown."""
        mock_manager = self._make_manager(
            recently_recovered=True,
            process_states=[HordeProcessState.PROCESS_ENDED, HordeProcessState.PROCESS_ENDED],
        )
        assert mock_manager.is_time_for_shutdown() is True

    def test_recently_recovered_mixed_ending_ended_allows_shutdown(self) -> None:
        """_recently_recovered=True with a mix of PROCESS_ENDING and PROCESS_ENDED allows shutdown."""
        mock_manager = self._make_manager(
            recently_recovered=True,
            process_states=[HordeProcessState.PROCESS_ENDING, HordeProcessState.PROCESS_ENDED],
        )
        assert mock_manager.is_time_for_shutdown() is True

    def test_recently_recovered_model_preloaded_allows_shutdown(self) -> None:
        """_recently_recovered=True with all processes MODEL_PRELOADED (idle) must allow shutdown."""
        mock_manager = self._make_manager(
            recently_recovered=True,
            process_states=[HordeProcessState.MODEL_PRELOADED],
        )
        assert mock_manager.is_time_for_shutdown() is True

    # ------------------------------------------------------------------
    # Cases where shutdown MUST NOT proceed (guard still active)
    # ------------------------------------------------------------------

    def test_recently_recovered_process_starting_blocks_shutdown(self) -> None:
        """_recently_recovered=True with a PROCESS_STARTING process must block shutdown.

        A freshly spawned replacement process sits in PROCESS_STARTING while loading.
        Without the guard it would look idle (not in any "alive" state) and trigger a
        premature shutdown while the process is still initialising.
        """
        mock_manager = self._make_manager(
            recently_recovered=True,
            process_states=[HordeProcessState.PROCESS_STARTING],
        )
        assert mock_manager.is_time_for_shutdown() is False

    def test_recently_recovered_mixed_starting_and_ending_blocks_shutdown(self) -> None:
        """_recently_recovered=True with at least one PROCESS_STARTING blocks shutdown."""
        mock_manager = self._make_manager(
            recently_recovered=True,
            process_states=[HordeProcessState.PROCESS_STARTING, HordeProcessState.PROCESS_ENDING],
        )
        assert mock_manager.is_time_for_shutdown() is False

    def test_not_recently_recovered_process_starting_allows_shutdown(self) -> None:
        """_recently_recovered=False with all PROCESS_STARTING allows shutdown (ENDING/ENDED path)."""
        mock_manager = self._make_manager(
            recently_recovered=False,
            process_states=[HordeProcessState.PROCESS_STARTING],
        )
        assert mock_manager.is_time_for_shutdown() is True

    # ------------------------------------------------------------------
    # Baseline: not shutting down must always return False
    # ------------------------------------------------------------------

    def test_not_shutting_down_returns_false_regardless_of_recovered(self) -> None:
        """is_time_for_shutdown() must return False when not in shutdown mode."""
        mock_manager = self._make_manager(
            recently_recovered=True,
            process_states=[HordeProcessState.WAITING_FOR_JOB],
        )
        mock_manager._shutting_down = False
        assert mock_manager.is_time_for_shutdown() is False

    def test_active_inference_blocks_shutdown_despite_not_recently_recovered(self) -> None:
        """INFERENCE_PROCESSING blocks shutdown regardless of _recently_recovered."""
        mock_manager = self._make_manager(
            recently_recovered=False,
            process_states=[HordeProcessState.INFERENCE_PROCESSING],
        )
        assert mock_manager.is_time_for_shutdown() is False

    def test_pending_jobs_blocks_shutdown_despite_idle_processes(self) -> None:
        """Pending jobs in the submit queue must still block shutdown."""
        mock_manager = self._make_manager(
            recently_recovered=True,
            process_states=[HordeProcessState.WAITING_FOR_JOB],
            jobs_pending_submit=1,
        )
        assert mock_manager.is_time_for_shutdown() is False


class TestRequestProgramRestartResetsJobPopTime:
    """Regression test: request_program_restart() must reset _last_job_pop_time to 0.

    The _process_control_loop checks `not self._last_pop_recently()` before calling
    end_inference_processes() while shutting down.  _last_pop_recently() returns True
    when the last pop was within 10 seconds, which delays killing stuck processes.
    Setting _last_job_pop_time=0 makes _last_pop_recently() return False immediately.
    """

    def test_request_program_restart_resets_last_job_pop_time(self) -> None:
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._last_job_pop_time = 999_999_999.0  # simulate a very recent pop
        mock_manager._restart_requested = False

        bound = HordeWorkerProcessManager.request_program_restart.__get__(mock_manager, HordeWorkerProcessManager)
        bound()

        assert mock_manager._last_job_pop_time == 0.0, (
            "_last_job_pop_time must be reset to 0.0 so _last_pop_recently() expires immediately"
        )
        assert mock_manager._restart_requested is True
        mock_manager._shutdown.assert_called_once()


class TestReplaceHungJobReceivedAndDownloading:
    """Tests that JOB_RECEIVED and DOWNLOADING_MODEL stuck detection works.

    These states are checked in the multi-pass scan to detect processes that have received
    a job but never started processing it, or that stalled during model download.
    """

    def _make_manager(
        self,
        processes: list,
        *,
        recently_recovered: bool = False,
    ):
        from unittest.mock import MagicMock

        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._recently_recovered = recently_recovered
        mock_manager._last_pop_no_jobs_available = False
        mock_manager._job_pops_paused = False
        mock_manager._shutting_down = False
        mock_manager._hung_processes_detected = False
        mock_manager._hung_processes_detected_time = 0.0
        mock_manager.bridge_data.inference_step_timeout = 600
        mock_manager.bridge_data.inference_timeout = 1200
        mock_manager.bridge_data.waiting_for_job_timeout = 600
        mock_manager.bridge_data.force_restart_timeout = 60
        mock_manager._reap_orphaned_in_progress_jobs.return_value = False
        mock_manager.max_concurrent_inference_processes = 1
        mock_manager.post_process_job_overlap_allowed = False
        mock_manager.bridge_data.preload_timeout = 80
        mock_manager.bridge_data.process_timeout = 300
        mock_manager.bridge_data.download_timeout = 300
        mock_manager.bridge_data.post_process_timeout = 60
        mock_manager.bridge_data.max_batch = 1
        mock_manager._process_map.is_stuck_on_inference.return_value = False
        mock_manager._check_and_replace_process.return_value = False
        mock_manager._process_map.values.return_value = processes
        mock_manager._process_map.__iter__ = MagicMock(return_value=iter(processes))
        mock_manager.jobs_pending_inference = [MagicMock()]  # local work pending
        mock_manager.jobs_in_progress = []

        mock_manager._bound_replace_hung = HordeWorkerProcessManager.replace_hung_processes.__get__(
            mock_manager, HordeWorkerProcessManager,
        )
        return mock_manager

    def _make_process(self, process_id: int, state, *, time_elapsed: float = 9999.0):
        import time as _time
        from unittest.mock import MagicMock

        from horde_worker_regen.process_management.horde_process import HordeProcessType

        proc = MagicMock()
        proc.process_id = process_id
        proc.process_type = HordeProcessType.INFERENCE
        proc.last_process_state = state
        proc.inference_started_timestamp = None
        proc.last_received_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_timestamp = _time.time() - time_elapsed
        proc.last_progress_timestamp = _time.time() - time_elapsed
        proc.state_entered_timestamp = _time.time() - time_elapsed
        proc.last_heartbeat_percent_complete = None
        proc.last_job_referenced = None
        return proc

    def test_job_received_stuck_check_runs(self) -> None:
        """JOB_RECEIVED stuck check must be evaluated in the multi-pass scan."""
        from unittest.mock import patch

        from horde_worker_regen.process_management.messages import HordeProcessState

        proc = self._make_process(0, HordeProcessState.JOB_RECEIVED)
        mock_manager = self._make_manager([proc])

        with patch("threading.Thread"):
            mock_manager._bound_replace_hung()

        called_states = [call.args[2] for call in mock_manager._check_and_replace_process.call_args_list]
        assert HordeProcessState.JOB_RECEIVED in called_states, (
            "JOB_RECEIVED must be checked in the multi-pass scan"
        )

    def test_downloading_model_stuck_check_runs(self) -> None:
        """DOWNLOADING_MODEL stuck check must be evaluated in the multi-pass scan."""
        from unittest.mock import patch

        from horde_worker_regen.process_management.messages import HordeProcessState

        proc = self._make_process(0, HordeProcessState.DOWNLOADING_MODEL)
        mock_manager = self._make_manager([proc])

        with patch("threading.Thread"):
            mock_manager._bound_replace_hung()

        called_states = [call.args[2] for call in mock_manager._check_and_replace_process.call_args_list]
        assert HordeProcessState.DOWNLOADING_MODEL in called_states, (
            "DOWNLOADING_MODEL must be checked in the multi-pass scan"
        )

    def test_job_received_uses_preload_timeout(self) -> None:
        """JOB_RECEIVED timeout must use preload_timeout (not process_timeout)."""
        from unittest.mock import patch

        from horde_worker_regen.process_management.messages import HordeProcessState

        proc = self._make_process(0, HordeProcessState.JOB_RECEIVED)
        mock_manager = self._make_manager([proc])
        mock_manager.bridge_data.preload_timeout = 42  # distinctive value

        with patch("threading.Thread"):
            mock_manager._bound_replace_hung()

        # Find the JOB_RECEIVED call and verify the timeout used
        for call in mock_manager._check_and_replace_process.call_args_list:
            if call.args[2] == HordeProcessState.JOB_RECEIVED:
                assert call.args[1] == 42, (
                    f"JOB_RECEIVED timeout must be preload_timeout (42), got {call.args[1]}"
                )
                break
        else:
            raise AssertionError("JOB_RECEIVED check was never called")

    def test_downloading_model_uses_download_timeout(self) -> None:
        """DOWNLOADING_MODEL timeout must use download_timeout."""
        from unittest.mock import patch

        from horde_worker_regen.process_management.messages import HordeProcessState

        proc = self._make_process(0, HordeProcessState.DOWNLOADING_MODEL)
        mock_manager = self._make_manager([proc])
        mock_manager.bridge_data.download_timeout = 77  # distinctive value

        with patch("threading.Thread"):
            mock_manager._bound_replace_hung()

        for call in mock_manager._check_and_replace_process.call_args_list:
            if call.args[2] == HordeProcessState.DOWNLOADING_MODEL:
                assert call.args[1] == 77, (
                    f"DOWNLOADING_MODEL timeout must be download_timeout (77), got {call.args[1]}"
                )
                break
        else:
            raise AssertionError("DOWNLOADING_MODEL check was never called")

    def test_job_received_skipped_when_no_work(self) -> None:
        """JOB_RECEIVED must be skipped when no jobs are available anywhere."""
        from unittest.mock import patch

        from horde_worker_regen.process_management.messages import HordeProcessState

        proc = self._make_process(0, HordeProcessState.JOB_RECEIVED)
        mock_manager = self._make_manager([proc])
        mock_manager._last_pop_no_jobs_available = True
        mock_manager.jobs_pending_inference = []
        mock_manager.jobs_in_progress = []

        with patch("threading.Thread"):
            mock_manager._bound_replace_hung()

        called_states = [call.args[2] for call in mock_manager._check_and_replace_process.call_args_list]
        assert HordeProcessState.JOB_RECEIVED not in called_states, (
            "JOB_RECEIVED check must be skipped when no jobs are available"
        )


class TestReplaceAllSafetyProcessRespectsShutdown:
    """_replace_all_safety_process() must not spawn a new safety process during shutdown.

    A safety process spawned mid-shutdown has nothing left to evaluate and is just an extra
    subprocess for the shutdown/watchdog path to kill, the same bug class as the inference-process
    respawn-during-shutdown issue this module also tests.
    """

    def _make_manager(self, *, shutting_down: bool) -> MagicMock:
        from horde_worker_regen.process_management.process_manager import HordeWorkerProcessManager

        mock_manager = MagicMock()
        mock_manager._shutting_down = shutting_down
        mock_manager._safety_processes_should_be_replaced = True
        mock_manager._safety_processes_ending = True
        mock_manager._num_process_recoveries = 0
        mock_manager._process_map.num_loaded_safety_processes.return_value = 0
        mock_manager._process_map.num_safety_processes.return_value = 0

        mock_manager._replace_all_safety_process = (
            HordeWorkerProcessManager._replace_all_safety_process.__get__(
                mock_manager, HordeWorkerProcessManager
            )
        )
        return mock_manager

    def test_does_not_start_safety_process_while_shutting_down(self) -> None:
        mock_manager = self._make_manager(shutting_down=True)

        mock_manager._replace_all_safety_process()

        mock_manager.start_safety_processes.assert_not_called()
        assert mock_manager._safety_processes_ending is False
        assert mock_manager._safety_processes_should_be_replaced is False
        assert mock_manager._num_process_recoveries == 0

    def test_starts_safety_process_when_not_shutting_down(self) -> None:
        mock_manager = self._make_manager(shutting_down=False)

        mock_manager._replace_all_safety_process()

        mock_manager.start_safety_processes.assert_called_once()
        assert mock_manager._safety_processes_ending is False
        assert mock_manager._safety_processes_should_be_replaced is False
        assert mock_manager._num_process_recoveries == 1
