from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False


class LocalLLMEngine:
    """
    Wrapper for local HF causal LLM inference.
    Intended for models like Qwen2-1.5B-Instruct or Qwen-7B-Chat.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        default_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.default_config = default_config or GenerationConfig()
        self._pipeline = None
        self._device_map = device_map
        self._torch_dtype = torch_dtype
        self._trust_remote_code = trust_remote_code

    def load(self) -> None:
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TextGenerationPipeline,
            )
        except Exception as exc:
            raise ImportError(
                "transformers is required for LocalLLMEngine. "
                "Please install dependencies in requirements.txt."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map=self._device_map,
            torch_dtype=self._torch_dtype,
            trust_remote_code=self._trust_remote_code,
        )
        self._pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    @property
    def ready(self) -> bool:
        return self._pipeline is not None

    def _merge_config(self, config: Optional[GenerationConfig]) -> GenerationConfig:
        return config if config is not None else self.default_config

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "<|system|>\n"
            f"{system_prompt}\n"
            "<|user|>\n"
            f"{user_prompt}\n"
            "<|assistant|>\n"
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        if not self.ready:
            raise RuntimeError("LLM engine is not loaded. Call load() before generate().")

        cfg = self._merge_config(config)
        prompt = self._build_prompt(system_prompt, user_prompt)
        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.do_sample,
                return_full_text=False,
            )
        except Exception as exc:
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

        if not outputs:
            raise RuntimeError("LLM returned empty output.")
        return outputs[0]["generated_text"].strip()

    def batch_generate(
        self,
        system_prompt: str,
        user_prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        results: List[str] = []
        for prompt in user_prompts:
            results.append(self.generate(system_prompt=system_prompt, user_prompt=prompt, config=config))
        return results
