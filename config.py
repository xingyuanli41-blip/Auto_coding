"""
配置加载器 —— 从 config.json 读取所有设置，提供默认值。

用法:
    from config import config
    api_key = config.llm_api_key
    client = OpenAI(api_key=api_key, base_url=config.llm_base_url)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

_config_path = Path(__file__).resolve().parent / "config.json"


class _Config:
    """配置对象，属性式访问"""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def _get(self, *keys: str, default: Any = None) -> Any:
        """逐层获取嵌套键值"""
        val = self._data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    # ── LLM ──
    @property
    def llm_provider(self) -> str:
        return self._get("llm", "provider", default="deepseek")

    @property
    def llm_api_key(self) -> str:
        return self._get("llm", "api_key", default="")

    @property
    def llm_base_url(self) -> str:
        return self._get("llm", "base_url", default="https://api.deepseek.com")

    @property
    def llm_model(self) -> str:
        return self._get("llm", "model", default="deepseek-chat")

    @property
    def llm_max_tokens(self) -> int:
        return self._get("llm", "max_tokens", default=4096)

    @property
    def llm_temperature(self) -> float:
        return self._get("llm", "temperature", default=0.7)

    @property
    def llm_timeout(self) -> int:
        return self._get("llm", "timeout", default=120)

    # ── 思考模型 ──
    @property
    def think_model(self) -> str:
        return self._get("think_model", "model", default="") or self.llm_model

    @property
    def think_api_key(self) -> str:
        return self._get("think_model", "api_key", default="") or self.llm_api_key

    @property
    def think_base_url(self) -> str:
        return self._get("think_model", "base_url", default="") or self.llm_base_url

    # ── Brain ──
    @property
    def brain_max_iterations(self) -> int:
        return self._get("brain", "max_iterations", default=10)

    @property
    def brain_max_retries(self) -> int:
        return self._get("brain", "max_retries", default=3)

    @property
    def brain_system_prompt(self) -> str:
        # 优先从文件读取
        prompt_file = self._get("brain", "system_prompt_file", default="")
        if prompt_file and os.path.exists(prompt_file):
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        # 回退到内嵌文本
        return self._get("brain", "system_prompt",
                          default="你是一个智能助手，擅长思考、规划和使用工具完成任务。")

    # ── Memory ──
    @property
    def memory_working_token_budget(self) -> int:
        return self._get("memory", "working_token_budget", default=4000)

    @property
    def memory_short_term_max(self) -> int:
        return self._get("memory", "short_term_max_items", default=200)

    @property
    def memory_long_term_max(self) -> int:
        return self._get("memory", "long_term_max_items", default=1000)

    @property
    def memory_working_file(self) -> str:
        return self._get("memory", "working_memory_file",
                          default="./memory/archives/conversation_memory.json")

    @property
    def memory_long_term_file(self) -> str:
        return self._get("memory", "long_term_memory_file",
                          default="./memory/archives/long_term_archive.json")

    @property
    def memory_embedding_model(self) -> str:
        return self._get("memory", "embedding_model", default="text-embedding-3-small")

    # ── Tools ──
    @property
    def tools_pool_file(self) -> str:
        return self._get("tools", "pool_file", default="./tools/mcp_tools.json")

    @property
    def tools_code_dir(self) -> str:
        return self._get("tools", "code_dir", default="./tools/tool_add/tool_direct")

    @property
    def tools_maintenance_days(self) -> int:
        return self._get("tools", "maintenance_days_unused", default=7)

    @property
    def tools_workspace_dir(self) -> str:
        return self._get("tools", "workspace_dir", default="./workspace")

    @property
    def tools_maintenance_min_usage(self) -> int:
        return self._get("tools", "maintenance_min_usage", default=1)

    # ── Logging ──
    @property
    def log_level(self) -> str:
        return self._get("logging", "level", default="INFO")

    @property
    def log_file(self) -> str:
        return self._get("logging", "file", default="")

    # ── 工具方法 ──
    def create_llm_client(self):
        """根据配置创建 OpenAI 兼容客户端"""
        from openai import OpenAI
        return OpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url,
                      timeout=self.llm_timeout)

    def create_think_client(self):
        """创建思考模型专用客户端（如果配置不同）"""
        from openai import OpenAI
        return OpenAI(api_key=self.think_api_key, base_url=self.think_base_url,
                      timeout=self.llm_timeout)

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"<Config llm={self.llm_model}@{self.llm_base_url}>"


# ── 加载 ──
def _load_config() -> _Config:
    if _config_path.exists():
        try:
            with open(_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _Config(data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"[config] 加载 config.json 失败: {e}，使用默认值")

    # 创建默认配置
    default_data = {
        "llm": {"api_key": "", "base_url": "https://api.deepseek.com", "model": "deepseek-chat",
                "max_tokens": 4096, "temperature": 0.7, "timeout": 120},
        "brain": {"max_iterations": 10, "max_retries": 3},
        "memory": {"working_token_budget": 4000, "short_term_max_items": 200, "long_term_max_items": 1000,
                   "working_memory_file": "./memory/archives/conversation_memory.json",
                   "long_term_memory_file": "./memory/archives/long_term_archive.json"},
        "tools": {"pool_file": "./tools/mcp_tools.json", "code_dir": "./tools/tool_add/tool_direct",
                  "maintenance_days_unused": 7, "maintenance_min_usage": 1},
        "logging": {"level": "INFO"},
    }
    return _Config(default_data)


config = _load_config()
