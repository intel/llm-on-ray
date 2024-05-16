from pathlib import Path
import subprocess

model_maps = {
    "gpt_neox": "gptneox",
    "gpt_bigcode": "starcoder",
    "whisper": "whisper",
    "qwen2": "qwen",
    "RefinedWebModel": "falcon",
    "RefinedWeb": "falcon",
    "phi-msft": "phi"
}

def convert_model(model, outfile, outtype="f32", format="NE", model_hub="huggingface", use_quantized_model=False):
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    model_type = model_maps.get(config.model_type, config.model_type)

    if use_quantized_model:
        path = Path(Path(__file__).parent.absolute(), "convert_quantized_{}.py".format(model_type))
    else:
        path = Path(Path(__file__).parent.absolute(), "convert_{}.py".format(model_type))
    cmd = []
    cmd.extend(["python", path])
    cmd.extend(["--outfile", outfile])
    cmd.extend(["--outtype", outtype])
    if model_type in {"phi", "stablelm"}:
        cmd.extend(["--format", format])
    cmd.extend(["--model_hub", model_hub])
    cmd.extend([model])

    print("cmd:", cmd)
    subprocess.run(cmd)