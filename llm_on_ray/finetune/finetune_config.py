from pydantic import BaseModel, validator
from typing import Optional, List


class GeneralConfig(BaseModel):
    trust_remote_code: bool
    use_auth_token: Optional[str]


class LoraConfig(BaseModel):
    task_type: str
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: Optional[List[str]] = None


class DeltatunerConfig(BaseModel):
    algo: str
    denas: bool
    best_model_structure: str


class General(BaseModel):
    base_model: str
    gpt_base_model: bool
    output_dir: str
    checkpoint_dir: Optional[str]
    config: GeneralConfig
    lora_config: Optional[LoraConfig] = None
    deltatuner_config: Optional[DeltatunerConfig] = None
    enable_gradient_checkpointing: bool = False


class Dataset(BaseModel):
    train_file: str
    validation_file: Optional[str]
    validation_split_percentage: int


class RayResourceConfig(BaseModel):
    CPU: int
    GPU: int = 0


class Training(BaseModel):
    optimizer: str
    batch_size: int
    epochs: int
    learning_rate: float
    lr_scheduler: str
    weight_decay: float
    device: str
    num_training_workers: int
    resources_per_worker: RayResourceConfig
    accelerate_mode: str
    mixed_precision: str = "no"
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    deepspeed_config_file: str = ""

    @validator("device")
    def check_device(cls, v: str):
        devices = ["CPU", "GPU"]
        if v not in devices:
            raise ValueError(f"device must be one of {devices}")
        return v

    @validator("accelerate_mode")
    def check_accelerate_mode(cls, v: str):
        modes = ["CPU_DDP", "GPU_DDP", "GPU_FSDP", "GPU_DEEPSPEED"]
        if v not in modes:
            raise ValueError(f"accelerate_mode must be one of {modes}")
        return v

    @validator("mixed_precision")
    def check_mixed_precision(cls, v: str):
        supported_precisions = ["no", "fp16", "bf16"]
        if v not in supported_precisions:
            raise ValueError(f"mixed_precision must be one of {supported_precisions}")
        return v

    @validator("logging_steps")
    def check_logging_steps(cls, v: int):
        assert v > 0
        return v

    # @model_validator(mode='after')
    # def check_device_and_accelerate_mode(self) -> "Training":
    #     dev = self.device
    #     res = self.resources_per_worker
    #     mode = self.accelerate_mode
    #     if dev == "CPU":
    #         if res.GPU is not None and res.GPU > 0:
    #             raise ValueError("Please not specified GPU resource when use CPU only in Ray.")
    #         if mode != "CPU_DDP":
    #             raise ValueError("Please specified CPU related accelerate mode when use CPU only in Ray.")
    #     elif dev == "GPU":
    #         if res.GPU is None or res.GPU == 0:
    #             raise ValueError("Please specified GPU resource when use GPU to fine tune in Ray.")
    #         if mode not in ["GPU_DDP", "GPU_FSDP"]:
    #             raise ValueError("Please speicifed GPU related accelerate mode when use GPU to fine tune in Ray.")

    #     return self


class FinetuneConfig(BaseModel):
    General: General
    Dataset: Dataset
    Training: Training
