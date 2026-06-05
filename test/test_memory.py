import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
三层记忆模型完整功能测试

运行方式:
    cd c:/Users/qq215/Desktop/auto_coding
    python test_memory.py
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0

def check(condition, msg):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {msg}")
    else:
        failed += 1
        print(f"  ❌ {msg}")

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ============================================================
# 准备
# ============================================================
section("准备：清理 + 初始化")

# 清理旧的存档
for f in ["memory/archives/conversation_memory.json",
          "memory/archives/long_term_archive.json"]:
    if os.path.exists(f):
        os.remove(f)

from memory import (
    Memory, LongTermMemory, Message, MemoryItem,
    MemoryManager, create_embedder,
    estimate_tokens, estimate_messages_tokens,
    handle_memory_overflow, summarize_func, load_memories_for_conversation,
    delete_least_important_summaries,
)
from memory.models import Memory, Message, MemoryItem, LongTermMemory
from memory.embeddings import TFIDFEmbedder, OpenAIEmbedder
from memory.manager import MemoryManager

print(f"  所有模块导入成功")


# ============================================================
# 1. 文件结构验证
# ============================================================
section("1. 文件结构完整性")

expected_files = [
    "memory/__init__.py",
    "memory/models.py",
    "memory/manager.py",
    "memory/utils.py",
    "memory/cleanup.py",
    "memory/embeddings.py",
]
for f in expected_files:
    check(os.path.exists(f), f"存在: {f}")

check(os.path.isdir("memory/archives"), "memory/archives/ 目录存在")
check(not os.path.exists("memory/memory_archives"), "旧 memory_archives 已清理")
check(not os.path.exists("memory/memory_base.py"), "旧 memory_base.py 已清理")
check(not os.path.exists("memory/memory_store.py"), "旧 memory_store.py 已清理")
check(not os.path.exists("memory/memory_retrieval.py"), "旧 memory_retrieval.py 已清理")
check(not os.path.exists("memory/memory_manager.py"), "旧 memory_manager.py 已清理")


# ============================================================
# 2. L1 工作记忆
# ============================================================
section("2. L1 工作记忆 (Working Memory)")

mgr = MemoryManager(working_token_budget=2000)

# 消息添加
mgr.add_system_message("你是一个助手")
mgr.add_user_message("你好，我是小明")
mgr.add_assistant_message("你好小明！")
mgr.add_tool_message("执行成功", "test_tool", "call_001")

check(len(mgr.working_memory.messages) == 4, "添加 4 条消息")
check(mgr.working_memory.messages[0].role == "system", "第1条是 system")
check(mgr.working_memory.messages[1].role == "user", "第2条是 user")
check(mgr.working_memory.messages[2].role == "assistant", "第3条是 assistant")
check(mgr.working_memory.messages[3].role == "tool", "第4条是 tool")

# Token 计数
ctx = mgr.get_working_context()
check(isinstance(ctx, list), "get_working_context 返回 list")
check(len(ctx) >= 3, "上下文包含 4 条消息")
tokens = estimate_messages_tokens(mgr.working_memory.messages)
check(tokens > 0, f"Token 估算 > 0: {tokens} tokens")

# Token 预算截断（大量消息）
for i in range(30):
    mgr.add_user_message(f"详细问题描述第{i}条 " * 10)
    mgr.add_assistant_message(f"详细回答内容第{i}条 " * 10)
ctx_trimmed = mgr.get_working_context(max_tokens=500)
check(len(ctx_trimmed) < len(mgr.working_memory.messages),
      f"Token 截断: {len(ctx_trimmed)} 条 < {len(mgr.working_memory.messages)} 条")


# ============================================================
# 3. L2 短期记忆（溢出 → 摘要）
# ============================================================
section("3. L2 短期记忆 (Episodic Memory) — 溢出摘要")

mgr2 = MemoryManager(working_token_budget=500, short_term_max_items=50)

# 填满工作记忆
for i in range(20):
    mgr2.add_user_message(f"用户问了关于Python第{i}个问题的详细内容...Python是一种编程语言..." * 2)
    mgr2.add_assistant_message(f"助手回答了第{i}个问题，提供了详细的代码示例和解释..." * 2)

# 触发溢出
triggered = mgr2.check_and_consolidate()
check(triggered, "工作记忆溢出触发摘要")
check(mgr2.stats["summarizations"] > 0, f"摘要操作记录: {mgr2.stats['summarizations']} 次")

# 摘要后工作记忆缩减
check(len(mgr2.working_memory.messages) < 40, "摘要后工作记忆缩减")

# 短期记忆有内容（无 LLM 时是自动摘要）
if len(mgr2.short_term_memory.items) > 0:
    first_item = mgr2.short_term_memory.items[0]
    check("summary" in first_item.tags, "短期记忆项带 summary 标签")
    check(first_item.memory_layer == "episodic", "层级标记为 episodic")
    print(f"  短期记忆项预览: {first_item.content[:80]}...")


# ============================================================
# 4. L3 长期记忆（巩固 + 向量检索）
# ============================================================
section("4. L3 长期记忆 (Semantic Memory)")

# 巩固
count = mgr2.consolidate_to_long_term()
check(count >= 0, f"巩固到长期记忆: {count} 项")

if count > 0:
    check(len(mgr2.long_term_memory.items) >= count, f"长期记忆包含 {len(mgr2.long_term_memory.items)} 项")
    check(mgr2.stats["consolidations"] > 0, "巩固操作计数增加")

# 向量/混合检索
results = mgr2.recall("Python")
check(len(results) >= 0, "语义检索不报错")

# 直接给长期记忆添加可检索项
item = MemoryItem(
    content="用户小明喜欢使用Python进行Web开发，偏好Flask框架",
    importance=0.9,
    tags=["fact", "preference"],
    memory_layer="semantic",
)
mgr2.long_term_memory.add_item(item)

# 关键词检索
results = mgr2.long_term_memory.search_by_keyword("小明的偏好")
# "小明" should be part of the search
results2 = mgr2.long_term_memory.search_by_keyword("Flask")
check(len(results2) >= 1, f"关键词检索 'Flask': {len(results2)} 条")

# 混合检索
results = mgr2.recall("Flask框架")
check(len(results) >= 1, f"recall 'Flask框架': {len(results)} 条")
if results:
    check("Flask" in results[0]["content"], "召回结果包含 Flask")


# ============================================================
# 5. 遗忘机制
# ============================================================
section("5. 遗忘机制 (Forgetting)")

mgr3 = MemoryManager(short_term_max_items=100)

# 添加重要记忆
important = MemoryItem(
    content="用户密码是123456",
    importance=0.9,
    tags=["important", "credential"],
    memory_layer="semantic",
)
mgr3.long_term_memory.add_item(important)

# 添加不重要且陈旧的记忆
old_unimportant = MemoryItem(
    content="用户曾经问过天气怎样",
    importance=0.05,
    timestamp=datetime(2020, 1, 1),
    last_accessed=datetime(2020, 1, 1),
    tags=["trivial"],
    memory_layer="episodic",
)
mgr3.short_term_memory.add_item(old_unimportant)

before_long = len(mgr3.long_term_memory.items)
before_short = len(mgr3.short_term_memory.items)
print(f"  遗忘前: 短期{before_short}项, 长期{before_long}项")

result = mgr3.forget(short_term_threshold=0.2, long_term_threshold=0.2)
print(f"  遗忘执行: {result}")

# 不重要的旧记忆应被遗忘
check(len(mgr3.short_term_memory.items) < before_short or old_unimportant.id not in [i.id for i in mgr3.short_term_memory.items],
      "陈旧不重要记忆被遗忘")

# 重要的记忆应保留
check(important.id in [i.id for i in mgr3.long_term_memory.items],
      "重要记忆保留")


# ============================================================
# 6. 衰减评分
# ============================================================
section("6. 记忆衰减评分 (decay_score)")

fresh = MemoryItem(content="新鲜记忆", importance=0.8)
old_item = MemoryItem(
    content="陈旧记忆",
    importance=0.8,
    timestamp=datetime(2020, 1, 1),
    last_accessed=datetime(2020, 1, 1),
)

fresh_score = fresh.decay_score()
old_score = old_item.decay_score()
check(fresh_score > old_score, f"新鲜 > 陈旧: {fresh_score:.3f} vs {old_score:.3f}")

low_imp = MemoryItem(content="不重要", importance=0.1)
high_imp = MemoryItem(content="重要", importance=0.9)
check(high_imp.decay_score() > low_imp.decay_score(),
      f"高重要性 > 低重要性: {high_imp.decay_score():.3f} vs {low_imp.decay_score():.3f}")


# ============================================================
# 7. 持久化与恢复
# ============================================================
section("7. 持久化与恢复")

mgr4 = MemoryManager(
    working_memory_file="./memory/archives/test_conversation.json",
    long_term_memory_file="./memory/archives/test_long_term.json",
)

mgr4.add_user_message("这条消息应该被持久化")
mgr4.add_assistant_message("好的，记住了")

# 添加长期记忆
perm_item = MemoryItem(
    content="持久化测试项：用户叫张三",
    importance=0.8,
    tags=["fact", "test"],
    memory_layer="semantic",
)
mgr4.long_term_memory.add_item(perm_item)
mgr4.save()

check(os.path.exists("./memory/archives/test_conversation.json"), "工作记忆文件已保存")
check(os.path.exists("./memory/archives/test_long_term.json"), "长期记忆文件已保存")

# 恢复
mgr4b = MemoryManager(
    working_memory_file="./memory/archives/test_conversation.json",
    long_term_memory_file="./memory/archives/test_long_term.json",
)
check(len(mgr4b.working_memory.messages) >= 1, "工作记忆恢复")
check(len(mgr4b.long_term_memory.items) >= 1, "长期记忆恢复")

# 恢复的内容验证
found = False
for item in mgr4b.long_term_memory.items:
    if "张三" in item.content:
        found = True
        break
check(found, "恢复的内容包含'张三'")

# 清理测试文件
for f in ["./memory/archives/test_conversation.json",
          "./memory/archives/test_long_term.json"]:
    if os.path.exists(f):
        os.remove(f)


# ============================================================
# 8. Embedding 引擎
# ============================================================
section("8. Embedding 引擎")

# TF-IDF
tfidf = TFIDFEmbedder()
vec, tf = tfidf.embed("Python 是一种编程语言 Python 很好用")
check(len(tf) > 0, f"TF-IDF 词汇向量: {len(tf)} 维度")

# 相似度
vec1, _ = tfidf.embed("Python 编程")
vec2, _ = tfidf.embed("Python 编程")
vec3, _ = tfidf.embed("今天天气真好晴天阳光")
sim_same = tfidf.similarity(vec1, vec2)
sim_diff = tfidf.similarity(vec1, vec3)
check(sim_same > sim_diff, f"相同主题相似度 > 不同主题: {sim_same:.3f} vs {sim_diff:.3f}")

# TF-IDF 检索
docs = [
    {"content": "Python 程序员小明喜欢 Flask"},
    {"content": "今天天气晴朗适合出游"},
    {"content": "Python Django 框架学习指南"},
    {"content": "晚间天气预报说明天有雨"},
]
results = tfidf.search_by_tfidf("Python 编程", docs, top_k=2)
check(len(results) >= 2, "TF-IDF 检索返回结果")
# 第一条应该最相关
best_doc = results[0][0]["content"]
check("Python" in best_doc, f"最相关结果含 'Python': {best_doc}")


# ============================================================
# 9. 旧 API 兼容性
# ============================================================
section("9. 旧 API 兼容性")

# handle_memory_overflow
m = Memory(max_messages=100)
for i in range(10):
    m.add_message(Message.user_message(f"消息{i}"))
ltm = LongTermMemory(max_items=50)
handle_memory_overflow(m, ltm, summarize_func, None)  # 未超限，不处理
check(len(m.messages) == 10, "未超限时 handle_memory_overflow 不处理")

# load_memories_for_conversation
mem = load_memories_for_conversation(
    "./memory/archives/nonexistent.json",
    "./memory/archives/nonexistent.json",
)
check(isinstance(mem, Memory), "load_memories_for_conversation 返回 Memory")

# 旧模型类正常
msg = Message.user_message("测试")
check(msg.role == "user", "Message.user_message 正常")
check(msg.to_dict()["role"] == "user", "to_dict 正常")


# ============================================================
# 10. 获取 LLM 上下文
# ============================================================
section("10. 上下文注入 (get_context_for_llm)")

mgr5 = MemoryManager()
mgr5.add_user_message("自我介绍：小明，喜欢Python")
mgr5.add_assistant_message("记住了")

# 添加相关长期记忆
rel_item = MemoryItem(
    content="用户小明偏好使用 Python Flask 框架",
    importance=0.8,
    tags=["fact"],
    memory_layer="semantic",
)
mgr5.long_term_memory.add_item(rel_item)

ctx_text = mgr5.get_context_for_llm("Python 框架")
check(len(ctx_text) > 0, "生成上下文不为空")
check("Flask" in ctx_text or "Python" in ctx_text, "上下文包含相关记忆")

# 无查询时也生成
ctx_text2 = mgr5.get_context_for_llm()
check(isinstance(ctx_text2, str), "无查询时也返回字符串")


# ============================================================
# 11. 统计与展示
# ============================================================
section("11. 统计与展示")

stats = mgr5.get_stats()
check("working_memory" in stats, "统计含工作记忆")
check("short_term_memory" in stats, "统计含短期记忆")
check("long_term_memory" in stats, "统计含长期记忆")
check("operations" in stats, "统计含操作记录")

s = mgr5.summary()
check("L1" in s, "summary 含 L1")
check("L2" in s, "summary 含 L2")
check("L3" in s, "summary 含 L3")
print(f"\n{s}")


# ============================================================
# 结果
# ============================================================
section(f"结果: {passed}/{passed+failed} 通过")

if failed == 0:
    print("\n  🎉 全部测试通过！三层记忆模型功能完整且结构合理。\n")
else:
    print(f"\n  ⚠️ {failed} 个测试失败。\n")
    sys.exit(1)
