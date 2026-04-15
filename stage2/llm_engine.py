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

from dataclasses import dataclass
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
        return self.generate_batch(
            system_prompt=system_prompt,
            user_prompts=[user_prompt],
            config=config,
        )[0]

    def generate_batch(
        self,
        system_prompt: str,
        user_prompts: list,
        config: Optional[GenerationConfig] = None,
        batch_size: int = 16,
    ) -> list:
        """
        批量推理，按 batch_size 分块处理以避免 OOM。

        参数：
            system_prompt : 所有样本共享的系统 prompt
            user_prompts  : 用户 prompt 列表
            config        : 覆盖 default_config（可选）
            batch_size    : 每批样本数（根据显存调整，默认 16）

        返回：
            与 user_prompts 等长的生成文本列表
        """
        if not self.ready:
            raise RuntimeError("Qwen3Engine not loaded. Call load() first.")

        results: list = []
        for start in range(0, len(user_prompts), batch_size):
            chunk = user_prompts[start: start + batch_size]
            results.extend(self._generate_chunk(system_prompt, chunk, config))
        return results

    def _generate_chunk(
        self,
        system_prompt: str,
        user_prompts: list,
        config: Optional[GenerationConfig] = None,
    ) -> list:
        """对单个 chunk 执行批推理，返回等长结果列表。"""
        cfg = config or self.default_config

        # 构建每条消息的 chat template 文本
        texts = []
        for user_prompt in user_prompts:
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
            texts.append(text)

        # left-padding 保证 generate 时 attention 正确对齐
        self._tokenizer.padding_side = "left"
        model_inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._model.device)

        gen_kwargs: dict = {"max_new_tokens": cfg.max_new_tokens}
        if cfg.do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p
        else:
            gen_kwargs["do_sample"] = False

        generated_ids = self._model.generate(**model_inputs, **gen_kwargs)

        results = []
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
            new_tokens = output_ids[len(input_ids):].tolist()
            # 去除 thinking token（token id 151668 = </think>）
            try:
                index = len(new_tokens) - new_tokens[::-1].index(151668)
            except ValueError:
                index = 0
            content = self._tokenizer.decode(
                new_tokens[index:], skip_special_tokens=True
            ).strip()
            results.append(content)

        return results
