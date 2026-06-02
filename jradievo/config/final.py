from pathlib import Path
from types import SimpleNamespace

from jradievo.config._schema import BunnyConfig, DAREConfig, TiesConfig


load_memory = True
print_text = True


n_trials = 1000

data = "select500"
target = "ja_select500"
extract_target = "findings"
no_findings = "no_findings"

path = SimpleNamespace(img_dir=Path("data/select500cxr"), data_dir=Path("data"))

num_data = 50

dataset = "simple"

sampler = "cmaes"

fp16 = True

base_model_name = "meta-llama/Meta-Llama-3-8B"

prompt = "あなたは優秀な放射線科医です。このレントゲン写真を見て、所見文を書いてください。肺・心臓・骨に異常があるかに注目して読影してください。日本語で書いてください。"
vlm = SimpleNamespace(
    name="BAAI/Bunny-v1_1-Llama-3-8B-V",
    bunny=BunnyConfig(
        text=f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:",
        max_new_tokens=256,
        repetition_penalty=1.0,
    ),
)

finetuned_models = {
    "mmedlm": "Henrychur/MMed-Llama-3-8B-EnIns",
    "openbio": "aaditya/OpenBioLLM-Llama3-8B",
    "llama-3-swallow": "tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1",
}

finetuned_param_name_convert_fns = {}

metrics = ["rouge_l_ja"]
use_metrics = ["rouge_l_ja"]
direction = ["maximize"]

dare = DAREConfig(mask_rate_type="same")

ties = TiesConfig(
    param_mask_type="each",
    scaling_coefficient=None,
    use_each_model_weight=True,
)

img_size = (512, 512)
