"""
三层记忆管理器 —— 编排工作记忆、短期记忆、长期记忆的流转。

三层模型:
  Layer 1 — 工作记忆 (Working Memory)    当前对话上下文，Token 预算管理
  Layer 2 — 短期记忆 (Episodic Memory)   最近摘要 + 结构化事实
  Layer 3 — 长期记忆 (Semantic Memory)   跨会话持久化，向量+关键词混合检索

流转:
  工作记忆溢出 → LLM 摘要 → 短期记忆 → 向量化巩固 → 长期记忆 → 反思整合
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from memory.models import (
    Memory, LongTermMemory, Message, MemoryItem,
    estimate_tokens, estimate_messages_tokens,
)
from memory.embeddings import create_embedder, TFIDFEmbedder, OpenAIEmbedder

logger = logging.getLogger(__name__)


class MemoryManager:
    """三层记忆管理器"""

    def __init__(
        self,
        working_token_budget: int = 4000,
        short_term_max_items: int = 200,
        long_term_max_items: int = 1000,
        working_memory_file: str = "./memory/archives/conversation_memory.json",
        long_term_memory_file: str = "./memory/archives/long_term_archive.json",
        client: Any = None,
        model: str = "gpt-4o",
        embedder: Any = None,
    ):
        self.working_token_budget = working_token_budget
        self.short_term_max_items = short_term_max_items
        self.long_term_max_items = long_term_max_items
        self.working_memory_file = working_memory_file
        self.long_term_memory_file = long_term_memory_file
        self.client = client
        self.model = model

        self.working_memory: Memory = Memory(max_messages=100)
        self.short_term_memory: LongTermMemory = LongTermMemory(max_items=short_term_max_items)
        self.long_term_memory: LongTermMemory = LongTermMemory(max_items=long_term_max_items)
        self.embedder = embedder or create_embedder(client)

        self.stats = {"consolidations": 0, "summarizations": 0, "reflections": 0}
        self._restore()

    # ==================== Layer 1: 工作记忆 ====================

    def add_user_message(self, content: str) -> None:
        self.working_memory.add_message(Message.user_message(content=content))

    def add_assistant_message(self, content: str) -> None:
        self.working_memory.add_message(Message.assistant_message(content=content))

    def add_tool_message(self, content: str, name: str, tool_call_id: str) -> None:
        self.working_memory.add_message(Message.tool_message(
            content=content, name=name, tool_call_id=tool_call_id))

    def add_system_message(self, content: str) -> None:
        self.working_memory.add_message(Message.system_message(content=content))

    def get_working_context(self, max_tokens: Optional[int] = None) -> List[Dict]:
        budget = max_tokens or self.working_token_budget
        raw = self.working_memory.to_dict_list()

        # 过滤孤立的 tool 消息（DeepSeek 要求前置 assistant+tool_calls）
        valid = []
        for msg in raw:
            if msg.get("role") == "tool":
                prev = valid[-1] if valid else None
                if not prev or prev.get("role") != "assistant" or not prev.get("tool_calls"):
                    continue
            valid.append(msg)

        # Token 预算截断：仅当消息过多时从旧→新保留
        if len(valid) > 50:  # 粗略阈值，避免截断正常对话
            kept = []
            for msg in reversed(valid):
                kept.insert(0, msg)
                if len(kept) >= 50:
                    break
            return kept
        return valid

    def check_and_consolidate(self) -> bool:
        tokens = estimate_messages_tokens(self.working_memory.messages)
        if tokens < self.working_token_budget * 0.8:
            return False
        self._summarize_working_to_short_term()
        return True

    # ==================== Layer 2: 短期记忆 ====================

    def _summarize_working_to_short_term(self) -> None:
        total = len(self.working_memory.messages)
        if total < 5:
            return
        archive_count = int(total * 0.7)
        keep_count = total - archive_count
        to_archive = self.working_memory.messages[:archive_count]

        conversation_text = "\n".join(
            f"[{msg.role}] {msg.content or ''}" for msg in to_archive)
        summary, facts = self._llm_summarize_and_extract(conversation_text)

        if summary:
            self._embed_and_store(MemoryItem(
                content=f"[摘要] {summary}", importance=0.8,
                tags=["summary", "episodic"], memory_layer="episodic",
                metadata={"summarized_at": datetime.now().isoformat(),
                          "source_message_count": archive_count},
            ), layer="short_term")

        for fact in facts:
            self._embed_and_store(MemoryItem(
                content=f"[事实] {fact}", importance=0.7,
                tags=["fact", "episodic"], memory_layer="episodic",
                metadata={"extracted_at": datetime.now().isoformat()},
            ), layer="short_term")

        self.working_memory.messages = self.working_memory.messages[-keep_count:]
        self.stats["summarizations"] += 1

    def _llm_summarize_and_extract(self, text: str) -> Tuple[str, List[str]]:
        if not self.client:
            return f"[自动摘要] 共 {len(text)} 字符的对话内容", []
        prompt = f"""请分析以下对话，完成两个任务：
1. 用 2-3 句话总结核心内容和当前进度。
2. 提取关键事实（用户偏好、决定、待办事项等），每条一行。
输出 JSON: {{"summary": "...", "facts": ["..."]}}

对话内容：
{text[:3000]}"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "你是对话分析助手。严格按 JSON 格式回复。"},
                          {"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=500,
            )
            content = resp.choices[0].message.content
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                data = json.loads(match.group(0))
                return data.get("summary", ""), data.get("facts", [])
        except Exception:
            logger.debug("LLM summarization failed", exc_info=True)
        return "[摘要失败]", []

    def _embed_and_store(self, item: MemoryItem, layer: str = "short_term") -> None:
        if self.embedder and isinstance(self.embedder, OpenAIEmbedder):
            try:
                vec = self.embedder.embed(item.content)
                if vec:
                    item.embedding = vec
            except Exception:
                logger.debug("embedding failed", exc_info=True)
        if layer == "short_term":
            self.short_term_memory.add_item(item)
        else:
            self.long_term_memory.add_item(item)

    # ==================== Layer 3: 长期记忆 ====================

    def consolidate_to_long_term(self) -> int:
        to_consolidate = [
            item for item in self.short_term_memory.items
            if item.memory_layer == "episodic"
        ]
        if not to_consolidate:
            return 0

        if isinstance(self.embedder, OpenAIEmbedder):
            texts = [item.content for item in to_consolidate if not item.embedding]
            if texts:
                try:
                    vectors = self.embedder.embed_batch(texts)
                    idx = 0
                    for item in to_consolidate:
                        if not item.embedding and idx < len(vectors) and vectors[idx]:
                            item.embedding = vectors[idx]
                            idx += 1
                except Exception:
                    logger.debug("batch embedding failed", exc_info=True)

        for item in to_consolidate:
            item.memory_layer = "semantic"
            item.importance = max(item.importance, 0.5)
            self.long_term_memory.add_item(item)
            self.short_term_memory.remove_item(item.id)

        self.stats["consolidations"] += 1
        return len(to_consolidate)

    def reflect_and_merge(self) -> str:
        episodic = [
            item for item in self.short_term_memory.items
            if "fact" in item.tags or "summary" in item.tags
        ]
        if len(episodic) < 3 or not self.client:
            return "记忆项不足，跳过反思"

        facts_text = "\n".join(
            f"{i+1}. [{item.tags}] {item.content}"
            for i, item in enumerate(episodic[:20]))
        prompt = f"""以下是近期提取的记忆。请找出可合并的重复/相关内容：
{facts_text}
输出 JSON: {{"has_merge": true/false, "merged_content": "...", "merged_tags": [...], "to_remove_ids": [1,3]}}"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "合并碎片化记忆。按 JSON 回复。"},
                          {"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=400,
            )
            content = resp.choices[0].message.content
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                data = json.loads(match.group(0))
                if data.get("has_merge") and data.get("merged_content"):
                    merged = MemoryItem(
                        content=f"[整合] {data['merged_content']}",
                        importance=0.85, tags=data.get("merged_tags", ["consolidated"]),
                        memory_layer="consolidated",
                        metadata={"merged_at": datetime.now().isoformat()},
                    )
                    self._embed_and_store(merged, layer="long_term")
                    for idx in data.get("to_remove_ids", []):
                        if 1 <= idx <= len(episodic):
                            self.short_term_memory.remove_item(episodic[idx - 1].id)
                    self.stats["reflections"] += 1
                    return f"已整合 {len(data.get('to_remove_ids', []))} 条"
        except Exception:
            logger.debug("reflection failed", exc_info=True)
        return "无需合并"

    # ==================== 检索 ====================

    def recall(self, query: str, top_k: int = 5, include_working: bool = True) -> List[Dict]:
        terms = query.lower().split()

        def _matches(text: str) -> bool:
            return all(t in (text or "").lower() for t in terms)

        results = []

        if include_working:
            for msg in reversed(self.working_memory.messages):
                if _matches(msg.content or ""):
                    results.append({"content": msg.content, "source": "working",
                                    "score": 1.0, "timestamp": datetime.now()})

        short_hits = self.short_term_memory.find_all(
            lambda item: _matches(item.content))
        for item in short_hits[:top_k]:
            results.append({"content": item.content, "source": "short_term",
                            "score": 0.8, "timestamp": item.timestamp})

        query_vec = None
        if isinstance(self.embedder, OpenAIEmbedder):
            try:
                query_vec = self.embedder.embed(query)
            except Exception:
                logger.debug("query embedding failed", exc_info=True)

        if query_vec:
            long_hits = self.long_term_memory.search_hybrid(
                query, query_vec, self.embedder, top_k=top_k)
        else:
            long_hits = self.long_term_memory.find_all(
                lambda item: _matches(item.content))[:top_k]

        for item in long_hits:
            results.append({"content": item.content, "source": "long_term",
                            "score": item.importance, "timestamp": item.timestamp})

        seen = set()
        unique = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            key = r["content"][:50]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique[:top_k]

    def get_context_for_llm(self, query: Optional[str] = None, max_tokens: int = 1000) -> str:
        parts = []
        if query:
            recalled = self.recall(query, top_k=3, include_working=False)
            if recalled:
                parts.append("📚 相关记忆:")
                for r in recalled:
                    age = ""
                    if r["timestamp"]:
                        days = (datetime.now() - r["timestamp"]).days
                        age = f" ({days}天前)" if days > 0 else ""
                    parts.append(f"  - {r['content'][:150]}{age}")
                parts.append("")
        summaries = self.short_term_memory.find_all(lambda item: "summary" in item.tags)
        if summaries:
            latest = sorted(summaries, key=lambda x: x.timestamp, reverse=True)[:2]
            parts.append("📝 近期对话摘要:")
            for s in latest:
                parts.append(f"  - {s.content[:200]}")
            parts.append("")
        return "\n".join(parts)

    # ==================== 遗忘 ====================

    def forget(self, short_term_threshold: float = 0.1,
               long_term_threshold: float = 0.05,
               max_short_term_items: int = 50) -> Dict[str, int]:
        removed_short = 0
        # 短期遗忘
        surviving = [item for item in self.short_term_memory.items
                     if item.decay_score() >= short_term_threshold]
        removed_short += len(self.short_term_memory.items) - len(surviving)
        self.short_term_memory.items = surviving

        if len(self.short_term_memory.items) > max_short_term_items:
            sorted_items = sorted(self.short_term_memory.items, key=lambda x: x.decay_score())
            to_remove = len(self.short_term_memory.items) - max_short_term_items
            for item in sorted_items[:to_remove]:
                self.short_term_memory.remove_item(item.id)
                removed_short += 1

        # 长期遗忘
        surviving_long = [item for item in self.long_term_memory.items
                          if item.decay_score() >= long_term_threshold]
        removed_long = len(self.long_term_memory.items) - len(surviving_long)
        self.long_term_memory.items = surviving_long

        return {"short_term_removed": removed_short, "long_term_removed": removed_long}

    # ==================== 持久化 ====================

    def save(self) -> None:
        self.working_memory.save_to_file(self.working_memory_file)
        self.long_term_memory.save_to_file_overwrite(self.long_term_memory_file)
        if self.short_term_memory.items:
            self.long_term_memory.save_to_file(self.long_term_memory_file)

    def _restore(self) -> None:
        if os.path.exists(self.working_memory_file) and os.path.getsize(self.working_memory_file) > 0:
            try:
                self.working_memory = Memory.load_from_file(self.working_memory_file)
            except Exception:
                logger.debug("restore working memory failed", exc_info=True)

        if os.path.exists(self.long_term_memory_file) and os.path.getsize(self.long_term_memory_file) > 0:
            try:
                restored = LongTermMemory.load_from_file(
                    self.long_term_memory_file, max_items=self.long_term_max_items)
                short_items, long_items = [], []
                for item in restored.items:
                    if item.memory_layer == "episodic":
                        short_items.append(item)
                    else:
                        long_items.append(item)
                self.short_term_memory.items = short_items
                self.long_term_memory.items = long_items
            except Exception:
                logger.debug("restore long-term memory failed", exc_info=True)

    # ==================== 统计 ====================

    def get_stats(self) -> Dict:
        working_tokens = estimate_messages_tokens(self.working_memory.messages)
        return {
            "working_memory": {
                "messages": len(self.working_memory.messages),
                "tokens": working_tokens,
                "budget": self.working_token_budget,
                "usage": f"{100 * working_tokens // max(self.working_token_budget, 1)}%",
            },
            "short_term_memory": {"items": len(self.short_term_memory.items),
                                   "max": self.short_term_max_items},
            "long_term_memory": {"items": len(self.long_term_memory.items),
                                  "max": self.long_term_max_items},
            "operations": self.stats,
        }

    def summary(self) -> str:
        s = self.get_stats()
        return (
            f"🧠 三层记忆\n"
            f"  L1 工作记忆: {s['working_memory']['messages']}条 "
            f"({s['working_memory']['tokens']}t/{s['working_memory']['budget']}t)\n"
            f"  L2 短期记忆: {s['short_term_memory']['items']}项\n"
            f"  L3 长期记忆: {s['long_term_memory']['items']}项\n"
            f"  操作: 摘要{s['operations']['summarizations']} "
            f"巩固{s['operations']['consolidations']} "
            f"反思{s['operations']['reflections']}"
        )
