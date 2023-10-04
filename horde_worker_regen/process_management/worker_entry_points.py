import contextlib

try:
    from multiprocessing.connection import PipeConnection as Connection  # type: ignore
except Exception:
    from multiprocessing.connection import Connection  # type: ignore
from multiprocessing.synchronize import Lock, Semaphore

from loguru import logger

from horde_worker_regen.process_management._aliased_types import ProcessQueue
from horde_worker_regen.process_management.inference_process import HordeInferenceProcess
from horde_worker_regen.process_management.safety_process import HordeSafetyProcess


def start_inference_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    inference_semaphore: Semaphore,
    disk_lock: Lock,
) -> None:
    worker_process = HordeInferenceProcess(
        process_id=process_id,
        process_message_queue=process_message_queue,
        pipe_connection=pipe_connection,
        inference_semaphore=inference_semaphore,
        disk_lock=disk_lock,
    )
    with logger.catch(), contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        worker_process.main_loop()


def start_safety_process(
    process_id: int,
    process_message_queue: ProcessQueue,
    pipe_connection: Connection,
    disk_lock: Lock,
    cpu_only: bool = True,
) -> None:
    worker_process = HordeSafetyProcess(
        process_id=process_id,
        process_message_queue=process_message_queue,
        pipe_connection=pipe_connection,
        disk_lock=disk_lock,
        cpu_only=cpu_only,
    )
    with logger.catch(), contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        worker_process.main_loop()
