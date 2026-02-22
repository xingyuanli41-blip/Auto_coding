import os
from typing import Optional
from tools.memory_base import Memory,LongTermMemory,Message
def load_memories_for_conversation(
    short_term_memory_file: str,
    long_term_memory_file: str,
    summary_tag: str = "summary",
) -> Memory:
    """
    从短期记忆文件和长期记忆文件中加载数据，合并到短期记忆中。
    合并顺序：先添加所有摘要消息（按时间升序），再添加短期记忆消息。
    如果总消息数超过 max_messages，则截断最早的消息（保留最新的 max_messages 条）。

    Args:
        short_term_memory_file: 短期记忆文件路径（JSON 格式）
        long_term_memory_file: 长期记忆文件路径（JSON 格式）
        summary_tag: 用于识别摘要内容的标签，默认为 "summary"
        max_messages: 短期记忆的最大消息数

    Returns:
        Memory: 合并后的短期记忆实例
    """
    # 1. 加载短期记忆（如果文件存在）
    if os.path.exists(short_term_memory_file) and os.path.getsize(short_term_memory_file) > 0 :
        short_memory = Memory.load_from_file(short_term_memory_file)
    else:
        short_memory = Memory()

    # 2. 加载长期记忆（如果文件存在），提取摘要
    summary_messages = []
    if os.path.exists(long_term_memory_file) and os.path.getsize(long_term_memory_file) > 0:
        long_term = LongTermMemory.load_from_file(long_term_memory_file)
        summary_items = sorted(
            [item for item in long_term.items if summary_tag in item.tags],
            key=lambda x: x.timestamp
        )
        if len(summary_items) != 0:
            for item in summary_items:
                time_str = item.timestamp.strftime("%Y-%m-%d %H:%M")
                content = f"[Summary from {time_str}] {item.content}"
                summary_messages.append(Message.system_message(content=content))

    merged_messages = summary_messages + short_memory.messages

    result_memory = Memory(messages=merged_messages)

    return result_memory
