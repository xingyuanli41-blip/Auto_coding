import os
import json
from datetime import datetime
from typing import List, Tuple, Optional
from openai import OpenAI

def delete_least_important_summaries(
    long_term_file: str,
    client: OpenAI,
    model: str = "gpt-3.5-turbo",
    k: int = 1,
    time_weight: float = 0.3,
    ai_weight: float = 0.7,
    importance_threshold: Optional[float] = None,
    summary_tag: str = "summary",
    raw_tag: str = "raw_message"
) -> int:
    """
    从长期记忆文件中删除重要性最低的摘要及其对应的原始消息。
    Args:
        long_term_file: 长期记忆 JSON 文件路径。
        client: OpenAI 客户端实例，用于调用 AI 评分。
        model: AI 评分使用的模型名称。
        k: 要删除的摘要数量（若指定 importance_threshold，此参数被忽略）。
        time_weight: 时间因素在重要性评分中的权重。
        ai_weight: AI 评分在重要性评分中的权重（两者之和应为 1.0）。
        importance_threshold: 重要性阈值，若提供则删除所有得分低于该值的摘要（忽略 k）。
        summary_tag: 识别摘要项的标签。
        raw_tag: 识别原始消息项的标签。
    Returns:
        实际删除的记忆项总数（摘要数 + 原始消息数）。
    """
    if not os.path.exists(long_term_file) or os.path.getsize(long_term_file) == 0:
        print(f"文件 {long_term_file} 不存在或为空，无需删除。")
        return

    try:
        long_term = LongTermMemory.load_from_file(long_term_file)
    except Exception as e:
        print(f"加载长期记忆文件失败: {e}")
        return
    items = long_term.items
    if not items:
        return

    items.sort(key=lambda x: x.timestamp)

    summary_items = [item for item in items if summary_tag in item.tags]
    raw_items = [item for item in items if raw_tag in item.tags]

    if not summary_items:
        print("未找到任何摘要项，无需删除。")
        return

    summary_items.sort(key=lambda x: x.timestamp)

    summary_to_raws = {} 
    last_summary_time = datetime.min 

    for summary in summary_items:
        raws_in_interval = [
            raw for raw in raw_items
            if last_summary_time < raw.timestamp <= summary.timestamp
        ]
        summary_to_raws[summary] = raws_in_interval
        last_summary_time = summary.timestamp

    def compute_importance(item) -> float:
        days_ago = (datetime.now() - item.timestamp).total_seconds() / 86400.0
        time_score = 1.0 / (1.0 + days_ago) 

        ai_score = get_ai_importance(item.content, client, model)

        score = time_weight * time_score + ai_weight * ai_score
        return score

    scores = {summary: compute_importance(summary) for summary in summary_items}

    if importance_threshold is not None:
        to_delete_summaries = [s for s in summary_items if scores[s] < importance_threshold]
    else:
        sorted_by_score = sorted(summary_items, key=lambda s: scores[s])
        to_delete_summaries = sorted_by_score[:k]

    if not to_delete_summaries:
        print("没有需要删除的摘要。")
        return 0

    to_delete_items = set()
    for summary in to_delete_summaries:
        to_delete_items.add(summary)
        for raw in summary_to_raws.get(summary, []):
            to_delete_items.add(raw)

    original_count = len(items)
    items = [item for item in items if item not in to_delete_items]
    deleted_count = original_count - len(items)

    long_term.items = items
    try:
        long_term.save_to_file_overwrite(long_term_file)
        print(f"成功删除 {len(to_delete_summaries)} 个摘要及其关联的原始消息，共 {deleted_count} 项。")
    except Exception as e:
        print(f"保存更新后的长期记忆失败: {e}")
        return 0

    return deleted_count

def get_ai_importance(text: str, client: OpenAI, model: str = "gpt-3.5-turbo") -> float:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个记忆重要性评估助手。请评估以下对话摘要的重要性（1-10分，10分最重要），只返回一个数字，不要返回其他内容。"},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_tokens=10
        )
        score_text = response.choices[0].message.content.strip()
        import re
        match = re.search(r'(\d+(?:\.\d+)?)', score_text)
        if match:
            score = float(match.group(1))
            return min(max(score / 10.0, 0.0), 1.0)
        else:
            print(f"AI 返回无法解析的分数: {score_text}，使用默认值 0.5")
            return 0.5
    except Exception as e:
        print(f"AI 评分失败: {e}，使用默认值 0.5")
        return 0.5