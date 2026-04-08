"""
Stage 2 LLM Engine — Qwen3 Wrapper
=====================================
封装 Qwen3 推理，接口与 stage1.llm_engine.LocalLLMEngine 保持一致：
    engine.load()
    engine.ready  -> bool
    engine.generate(system_prompt, user_prompt) -> str

使用 apply_chat_template + enable_thinking=False（非 thinking 模式，速度更快）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    enable_thinking: bool = False     # False = non-thinking mode，输出更简洁
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9


class Qwen3Engine:
    """
    Qwen3 推理引擎。

    参数：
        model_name_or_path : HuggingFace model ID 或本地路径，
                             默认 "Qwen/Qwen3-1.7B"
        device_map         : "auto" 自动分配设备
        torch_dtype        : "auto" 自动选择精度
        default_config     : 默认生成参数

    用法：
        engine = Qwen3Engine("Qwen/Qwen3-1.7B")
        engine.load()
        result = engine.generate(system_prompt="...", user_prompt="...")
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-1.7B",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        default_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.default_config = default_config or GenerationConfig()
        self._model = None
        self._tokenizer = None
        self._device_map = device_map
        self._torch_dtype = torch_dtype

    # ── 加载 ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """加载 tokenizer 和 model（首次调用较慢）。"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install it with: pip install transformers"
            ) from exc

        print(f"[Qwen3Engine] Loading {self.model_name_or_path} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self._torch_dtype,
            device_map=self._device_map,
        )
        print("[Qwen3Engine] Model ready.")

    # ── 属性 ──────────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    # ── 推理 ──────────────────────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        单次推理。

        参数：
            system_prompt : 系统角色描述
            user_prompt   : 用户输入
            config        : 覆盖 default_config（可选）

        返回：
            模型生成文本（已去除 thinking 部分和首尾空白）
        """
        if not self.ready:
            raise RuntimeError("Qwen3Engine not loaded. Call load() first.")

        cfg = config or self.default_config
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg.enable_thinking,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature if cfg.do_sample else None,
            top_p=cfg.top_p if cfg.do_sample else None,
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # 去除 thinking token（token id 151668 = </think>）
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self._tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip()
        return content
