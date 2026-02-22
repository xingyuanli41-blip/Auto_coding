import hashlib
from datetime import datetime
from typing import Callable, Optional
from tools.memory_base import Memory,MemoryItem,LongTermMemory,Message
import os
from openai import OpenAI

def handle_memory_overflow(
    memory: Memory,
    long_term_memory: LongTermMemory,
    summarize_func: Callable[[str,OpenAI,str], str],
    client:OpenAI,
    model:str="gpt-3.5-turbo",
    archive_dir: str = "./memory_archives",
    archive_prefix: str = "long_term_archive",
    raw_tag: str = "raw_message",
    summary_tag: str = "summary"
) -> None:
    """
    处理短期记忆溢出：
    1. 当短期记忆中的消息数量超过 max_messages 时：
       a. 将前 90% 的消息逐条存入长期记忆，每条消息作为单独的 MemoryItem。
       b. 对这些消息进行 AI 总结，将总结作为一条 MemoryItem 存入长期记忆。
    2. 截断短期记忆，只保留最后 10% 的消息。
    3. 如果长期记忆达到 max_items 限制，则将其内容保存到文件并清空。

    Args:
        memory: 短期记忆实例
        long_term_memory: 长期记忆实例
        summarize_func: AI 总结函数，接受一段文本返回摘要字符串
        archive_dir: 长期记忆存档目录
        archive_prefix: 存档文件前缀
        raw_tag: 原始消息的标签
        summary_tag: 总结的标签
    """
    if len(memory) < memory.max_messages:
        print('未超出，不给予总结')
        return
    print('正在处理')
    total = len(memory.messages)
    delete_count = int(total * 0.8)
    keep_count = total - delete_count

    messages_to_archive = memory.messages[:delete_count]

    # ========== 1a. 将每条原始消息逐条存入长期记忆 ==========
    for msg in messages_to_archive:
        content_str = f"[{msg.role}] {msg.content or ''}"
        raw_memory_item = MemoryItem(
            content=content_str,
            importance=0.5, 
            tags=[raw_tag, f"role:{msg.role}"],
            metadata={
                "archived_at": datetime.now().isoformat(),
                "original_message": msg.to_dict()  
            }
        )
        long_term_memory.add_item(raw_memory_item)

    # ========== 1b. AI 总结并存入长期记忆 ==========
    summary_input = "\n".join(
        f"[{msg.role}] {msg.content or ''}" for msg in messages_to_archive
    )
    print(f'这是要总结的内容：{summary_input}')
    try:
        summary_text = summarize_func(summary_input,client,model)
    except Exception as e:
        print(f"AI 总结失败: {e}")
        summary_text = "[总结失败]"
    print(f'这是已经总结的内容：{summary_text}')
    summary_memory_item = MemoryItem(
        content=summary_text,
        importance=0.8,
        tags=[summary_tag, "conversation_archive"],
        metadata={
            "summarized_at": datetime.now().isoformat(),
            "source_message_count": len(messages_to_archive)
        }
    )
    long_term_memory.add_item(summary_memory_item)
    # ========== 3. 检查长期记忆容量，满则存档并清空 ==========
    if long_term_memory.max_items is not None and len(long_term_memory) >= long_term_memory.max_items:
        os.makedirs(archive_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_filepath = os.path.join(archive_dir, f"{archive_prefix}.json")
        long_term_memory.save_to_file(archive_filepath)
        long_term_memory.clear()
        print(f"长期记忆已存档至: {archive_filepath}")

    kept_messages = memory.messages[-keep_count:]
    summary_message = Message.system_message(content=summary_text)
    memory.messages = [summary_message] + kept_messages
    print("总结成功")

def summarize_func(
    text: str,
    client: OpenAI,
    model: str = "gpt-3.5-turbo"  # 默认模型，可覆盖
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个对话总结助手，请用简洁的语言总结以下对话的核心内容，保留重要信息。(主要形式为：事件与当前进度)"},
                {"role": "user", "content": text}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        summary = response.choices[0].message.content.strip()
        return summary if summary else "[无法生成摘要]"
    except Exception as e:
        print(f"AI 总结失败: {e}")
        return "[总结失败]"


# client = OpenAI(
#     api_key="sk-ujecYWuE6lhVJ3zeZqS3Qb9jTu5HY2jOKd6X83Ho9lmznGNu",
#     base_url="https://poloai.top/v1", 
# )


# # 创建短期和长期记忆实例
# conversation_memory = Memory(max_messages=100)
# long_term_memory = LongTermMemory(max_items=500)

# # 调用处理函数，指定模型名称（如果 client 支持 gpt-4，可传入 "gpt-4"）
# handle_memory_overflow(
#     memory=conversation_memory,
#     long_term_memory=long_term_memory,
#     summarize_func=summarize_func,
#     client=client,
#     model="gpt-3.5-turbo"  # 根据实际使用的模型调整
# )

