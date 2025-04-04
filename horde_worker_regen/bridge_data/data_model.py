"""The config model and initializers for the reGen configuration model."""

from __future__ import annotations

import json
import os

from horde_sdk.ai_horde_worker.bridge_data import CombinedHordeBridgeData
from loguru import logger
from pydantic import Field, field_validator, model_validator
from ruamel.yaml import YAML

from horde_worker_regen.consts import TOTAL_LORA_DOWNLOAD_TIMEOUT
from horde_worker_regen.locale_info.regen_bridge_data_fields import BRIDGE_DATA_FIELD_DESCRIPTIONS


class reGenBridgeData(CombinedHordeBridgeData):
    """The config model for reGen. Extra fields added here are specific to this worker implementation.

    See `CombinedHordeBridgeData` from the SDK for more information..
    """

    _loaded_from_env_vars: bool = False

    disable_terminal_ui: bool = Field(
        default=True,
    )

    safety_on_gpu: bool = Field(
        default=False,
    )
    """If true, the safety model will be run on the GPU."""

    _yaml_loader: YAML | None = None

    cycle_process_on_model_change: bool = Field(
        default=False,
    )
    """If true, the process will stop and restart when the model loaded changes.

    Warning: This can cause substantial delays in processing.
    """

    CIVIT_API_TOKEN: str | None = Field(
        default=None,
        alias="civitai_api_token",
    )
    """The API token for CivitAI, used for downloading LoRas and login-required models."""

    unload_models_from_vram_often: bool = Field(default=True)
    """If true, models will be unloaded from VRAM more often."""

    process_timeout: int = Field(default=300)
    """The maximum amount of time to allow a job to run before it is killed"""

    post_process_timeout: int = Field(default=60, ge=15)

    download_timeout: int = Field(default=TOTAL_LORA_DOWNLOAD_TIMEOUT + 1)
    """The maximum amount of time to allow an aux model to download before it is killed"""
    preload_timeout: int = Field(default=80, ge=15)
    """The maximum amount of time to allow a model to load before it is killed"""
    inference_step_timeout: int = Field(default=15, ge=15, le=30)
    """The maximum amount of time to allow a single inference step to run before the process is killed"""

    minutes_allowed_without_jobs: int = Field(default=30, ge=0, lt=60 * 60)

    horde_model_stickiness: float = Field(default=0.0, le=1.0, ge=0.0, alias="model_stickiness")
    """
    A percent chance (expressed as a decimal between 0 and 1) that the currently loaded models will
    be favored when popping a job.
    """

    high_memory_mode: bool = Field(default=True)
    """Indicates that the worker should consume more memory to improve performance."""

    very_high_memory_mode: bool = Field(default=False)
    """Indicates that the worker should consume even more memory to improve performance.

    This has data-center grade cards in mind, and is not recommended for consumer grade cards.
    """

    high_performance_mode: bool = Field(default=True)
    """If you have a 4090 or better, set this to true to enable high performance mode."""

    moderate_performance_mode: bool = Field(default=False)
    """If you have a 3080 or better, set this to true to enable moderate performance mode."""

    very_fast_disk_mode: bool = Field(default=False)
    """If you have a very fast disk, set this to true to concurrently load more models at a time from disk."""

    post_process_job_overlap: bool = Field(default=False)
    """High and moderate performance modes will skip post processing if this is set to true."""

    capture_kudos_training_data: bool = Field(default=False)

    kudos_training_data_file: str | None = Field(default=None)

    exit_on_unhandled_faults: bool = Field(default=False)
    """If true, the worker will exit if an unhandled fault occurs instead of attempting to recover."""

    purge_loras_on_download: bool = Field(default=False)

    remove_maintenance_on_init: bool = Field(default=False)

    load_large_models: bool = Field(default=True)

    custom_models: list[dict] = Field(
        default_factory=list,
    )

    limited_console_messages: bool = Field(default=False)
    """If true, the worker will only log for submit and the status message.

    Set stats_output_frequency (in seconds) for control over the status message.
    """

    @model_validator(mode="after")
    def validate_performance_modes(self) -> reGenBridgeData:
        """Validate the performance modes and set the appropriate values.

        Returns:
            reGenBridgeData: The config model with the performance modes set appropriately.
        """
        if self.max_threads >= 2 and self.queue_size > 3:
            self.queue_size = 3
            logger.warning(
                "The queue_size value has been set to 3 because the max_threads value is 2.",
            )

        if self.high_performance_mode:
            process_timeout_changed_message = (
                "High performance mode is enabled, so the process_timeout value has "
                f"been set to 1/3 of the default value. The new value is {self.process_timeout}."
            )
            default_process_timeout = self.model_fields["process_timeout"].default

            if self.process_timeout == default_process_timeout:
                logger.debug(process_timeout_changed_message)
            else:
                logger.warning(process_timeout_changed_message)

            self.process_timeout = default_process_timeout // 3
        elif self.moderate_performance_mode:
            process_timeout_changed_message = (
                "Moderate performance mode is enabled, so the process_timeout value has "
                f"been set to 1/2 of the default value. The new value is {self.process_timeout}."
            )
            default_process_timeout = self.model_fields["process_timeout"].default

            if self.process_timeout == default_process_timeout:
                logger.debug(process_timeout_changed_message)
            else:
                logger.warning(process_timeout_changed_message)

            self.process_timeout = default_process_timeout // 2

        if self.extra_slow_worker:
            if self.high_performance_mode:
                self.high_performance_mode = False
                logger.warning(
                    "Extra slow worker is enabled, so the high_performance_mode value has been set to False.",
                )
            if self.moderate_performance_mode:
                self.moderate_performance_mode = False
                logger.warning(
                    "Extra slow worker is enabled, so the moderate_performance_mode value has been set to False.",
                )
            if self.high_memory_mode:
                self.high_memory_mode = False
                logger.warning(
                    "Extra slow worker is enabled, so the high_memory_mode value has been set to False.",
                )
            if self.very_high_memory_mode:
                self.very_high_memory_mode = False
                logger.warning(
                    "Extra slow worker is enabled, so the very_high_memory_mode value has been set to False.",
                )
            if self.queue_size > 0:
                self.queue_size = 0
                logger.warning(
                    "Extra slow worker is enabled, so the queue_size value has been set to 0. "
                    "This behavior may change in the future.",
                )
            if self.max_threads > 1:
                self.max_threads = 1
                logger.warning(
                    "Extra slow worker is enabled, so the max_threads value has been set to 1. "
                    "This behavior may change in the future.",
                )
            if self.preload_timeout < 120:
                self.preload_timeout = 120
                logger.warning(
                    "Extra slow worker is enabled, so the preload_timeout value has been set to 120. "
                    "This behavior may change in the future.",
                )

        if self.very_high_memory_mode and not self.high_memory_mode:
            self.high_memory_mode = True
            logger.debug(
                "Very high memory mode is enabled, so the high_memory_mode value has been set to True.",
            )

        if self.high_memory_mode:
            if self.max_threads == 1:
                logger.warning(
                    "High memory mode is enabled, you should consider setting max_threads to 2.",
                )

            if self.queue_size == 0:
                logger.warning(
                    "High memory mode is enabled, you should consider setting queue_size to 1 or higher. "
                    "Increasing this value increases system memory usage. See the bridgeData_template.yaml for more "
                    "information.",
                )

            if self.unload_models_from_vram_often:
                logger.warning(
                    "High memory mode is enabled, you should consider setting unload_models_from_vram_often to False.",
                )

            if self.cycle_process_on_model_change:
                self.cycle_process_on_model_change = False
                logger.warning(
                    "High memory mode is enabled, so the cycle_process_on_model_change value has been set to False.",
                )

        return self

    @field_validator("dreamer_worker_name", mode="after")
    def validate_dreamer_worker_name(cls, value: str) -> str:
        """Apply the environment variable override for the `dreamer_worker_name` field."""
        AIWORKER_DREAMER_WORKER_NAME = os.getenv("AIWORKER_DREAMER_WORKER_NAME")
        if AIWORKER_DREAMER_WORKER_NAME:
            logger.warning(
                "AIWORKER_DREAMER_WORKER_NAME environment variable is set. This will override the value for "
                "`dreamer_worker_name` in the config file.",
            )
            return AIWORKER_DREAMER_WORKER_NAME

        return value

    def prepare_custom_models(self) -> None:
        """Prepare the custom models."""
        if os.getenv("HORDELIB_CUSTOM_MODELS"):
            logger.info(
                f"HORDELIB_CUSTOM_MODELS already set to '{os.getenv('HORDELIB_CUSTOM_MODELS')}. "
                "Doing nothing for custom models.",
            )
            return
        custom_models_dict = {}
        for model in self.custom_models:
            if not model.get("name"):
                logger.warning(f"Model name not specified for custom model entry {model}. Skipping")
                continue
            if not model.get("baseline"):
                logger.warning(f"Model baseline not specified for custom model entry {model}. Skipping")
                continue
            if not model.get("filepath"):
                logger.warning(f"Model filepath not specified for custom model entry {model}. Skipping")
                continue
            # TODO: Handle Stable Cascade models
            custom_models_dict[model["name"]] = {
                "name": model["name"],
                "baseline": model["baseline"],
                "type": "ckpt",
                "config": {"files": [{"path": model["filepath"]}]},
            }
        cwd = os.getcwd()
        if len(custom_models_dict) > 0:
            with open(f"{cwd}/custom_models.json", "w") as f:
                json.dump(custom_models_dict, f, indent=4)
        else:
            if os.path.exists(f"{cwd}/custom_models.json"):
                os.remove(f"{cwd}/custom_models.json")
        os.environ["HORDELIB_CUSTOM_MODELS"] = f"{cwd}/custom_models.json"

    @staticmethod
    def load_custom_models() -> None:
        """Load the custom models from the `custom_models.json` file."""
        cwd = os.getcwd()
        if not os.getenv("HORDELIB_CUSTOM_MODELS") and os.path.exists(f"{cwd}/custom_models.json"):
            os.environ["HORDELIB_CUSTOM_MODELS"] = f"{cwd}/custom_models.json"
            logger.debug(f"HORDELIB_CUSTOM_MODELS: {cwd}/custom_models.json")

    def load_env_vars(self) -> None:
        """Load the environment variables into the config model."""
        # See load_env_vars.py's `def load_env_vars(self) -> None:`
        if self.models_folder_parent and os.getenv("AIWORKER_CACHE_HOME") is None:
            os.environ["AIWORKER_CACHE_HOME"] = self.models_folder_parent
        if self.horde_url:
            if os.environ.get("AI_HORDE_URL"):
                logger.warning(
                    "AI_HORDE_URL environment variable already set. This will override the value for `horde_url` in "
                    "the config file.",
                )
            else:
                if os.environ.get("AI_HORDE_DEV_URL"):
                    logger.warning(
                        "AI_HORDE_DEV_URL environment variable already set. This will override the value for "
                        "`horde_url` in the config file.",
                    )
                if os.environ.get("AI_HORDE_URL") is None:
                    os.environ["AI_HORDE_URL"] = self.horde_url
                else:
                    logger.warning(
                        "AI_HORDE_URL environment variable already set. This will override the value for `horde_url` "
                        "in the config file.",
                    )

        if self.CIVIT_API_TOKEN is not None:
            os.environ["CIVIT_API_TOKEN"] = self.CIVIT_API_TOKEN

        if self.max_lora_cache_size and os.getenv("AIWORKER_LORA_CACHE_SIZE") is None:
            os.environ["AIWORKER_LORA_CACHE_SIZE"] = str(self.max_lora_cache_size * 1024)

        if self.load_large_models:
            os.environ["AI_HORDE_MODEL_META_LARGE_MODELS"] = "1"

    def save(self, file_path: str) -> None:
        """Save the config model to a file.

        Args:
            file_path (str): The path to the file to save the config model to.
        """
        if self._yaml_loader is None:
            self._yaml_loader = YAML()

        with open(file_path, "w", encoding="utf-8") as f:
            self._yaml_loader.dump(self.model_dump(), f)


# Dynamically add descriptions to the fields of the model
for field_name, field in reGenBridgeData.model_fields.items():
    if field_name in BRIDGE_DATA_FIELD_DESCRIPTIONS:
        field.description = BRIDGE_DATA_FIELD_DESCRIPTIONS[field_name]
