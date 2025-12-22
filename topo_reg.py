"""
topo_reg.py
-----------------
Training with Topological Regularization.
Uses 4-bit LoRA and standard Matplotlib for the loss curve.
"""

import os
import time
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import gc

# kt: have to do this for my hardware constraints
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Config
CRITICAL_HEADS = [(8, 3, "bridge"), (19, 0, "cone")]
REG_STRENGTH = 50.0

TRAIN_TEXTS = [
    "The secret code is Alpha. The secret code is Beta.",
    "When you see Red, go Left. When you see Red, go Right.",
    "User: Hello. AI: Hi. User: Hello. AI: Goodbye.",
    "Pattern A leads to X. Pattern A leads to Y.",
    "Pattern B leads to X. Pattern B leads to Y.",
    "The quick brown fox jumps. The quick brown fox sleeps.",
] * 10


class TopologicalRegularizer:
    def __init__(self, anchors, strength):
        self.anchors = anchors
        self.strength = strength

    def compute_loss(self, outputs):
        if outputs.attentions is None:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=outputs.logits.device)
        curr_attn = outputs.attentions

        for layer, head, role in CRITICAL_HEADS:
            curr = curr_attn[layer][:, head, :, :]
            anchor = self.anchors[layer].to(curr.device)

            if role == "bridge":
                diff = torch.norm(curr - anchor, p="fro")
            elif role == "cone":
                diff = torch.norm(curr[:, :, 0] - anchor[:, :, 0], p="fro")
            loss += diff

        return self.strength * loss


def train():
    print(">>> Setting up 4-bit Llama-3 Training...")
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    # !!! disable cache to prevent gradient detachment issues with LoRA+attn
    model.config.use_cache = False
    model.config.output_attentions = True

    print(">>> Capturing Topological Anchors...")
    anchor_prompt = "The quick brown fox jumps over the lazy dog."
    anchor_inputs = tokenizer(anchor_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**anchor_inputs, output_attentions=True)

    anchors = {
        l: out.attentions[l][:, h, :, :].detach().cpu().clone()
        for l, h, _ in CRITICAL_HEADS
    }

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, peft_config)

    # ensure gradients are on for LoRA
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    topo_loss_fn = TopologicalRegularizer(anchors, REG_STRENGTH)

    history = []
    model.train()
    print(">>> Starting Training...")

    for step, text in enumerate(TRAIN_TEXTS):
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=64
        ).to(model.device)
        optimizer.zero_grad()

        # task pass
        task_out = model(**inputs, labels=inputs["input_ids"], output_attentions=True)

        # regularization pass (w gradients)
        anchor_out = model(**anchor_inputs, output_attentions=True)
        reg_loss = topo_loss_fn.compute_loss(anchor_out)

        total_loss = (
            task_out.loss + reg_loss
        )  # comment reg_loss in or out depending on if you want with or without reg
        total_loss.backward()
        optimizer.step()

        print(
            f"Step {step} | Task: {task_out.loss.item():.4f} | Topo: {reg_loss.item():.4f}"
        )
        history.append(
            {"step": step, "task": task_out.loss.item(), "topo": reg_loss.item()}
        )

        if step % 10 == 0:
            gc.collect()
            torch.mps.empty_cache()

        time.sleep(5)  # kt: laptop was overheating, thought it would help

    print(">>> Plotting Results...")
    df = pd.DataFrame(history)
    plt.figure(figsize=(10, 6))

    plt.plot(
        df["step"], df["task"], label="Task Loss (Learning)", color="blue", linewidth=2
    )
    plt.plot(
        df["step"],
        df["topo"],
        label="Topological Drift (Forgetting)",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Topological Regularization: Learning without Forgetting")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("solution_curve.png", dpi=300)
    print(">>> Saved to solution_curve.png")


if __name__ == "__main__":
    train()
