import sys
from pathlib import Path
# 获取当前文件所在目录的父目录（即项目根目录）
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
import logging
from typing import List, Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from tools.tool_base_abs import TOOL_base_DESCRIPTIONS,TOOL_base_FUNCTIONS
# 假设 client 已经在别处初始化，例如：
# client = OpenAI(api_key="your-api-key")

def call_openai_with_tools(
    messages: List[Dict[str, Any]], tools: List[Dict], client , model: str = "gpt-4o"
) -> ChatCompletionMessage:
    """
    调用 OpenAI ChatCompletion 并返回响应消息（可能包含工具调用）。

    Args:
        messages: 对话历史消息列表
        tools: 工具描述列表（即 TOOL_DESCRIPTIONS）
        model: 使用的模型名称，默认 "gpt-4o"

    Returns:
        ChatCompletionMessage 对象，包含 assistant 的回复内容或工具调用。

    Raises:
        可能抛出异常，但已捕获并记录日志，返回包含错误信息的模拟消息。
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # 让模型自主选择是否调用工具
        )
        return response.choices[0].message
    except Exception as e:
        logging.error(f"OpenAI API 调用失败: {e}")
        # 返回一个模拟的消息，包含错误信息，以便上层处理
        # 注意：这里不能直接抛出，因为上层可能期望一个消息对象
        # 我们创建一个伪消息，将错误放在 content 中
        fake_message = ChatCompletionMessage(
            role="assistant",
            content=f"调用 OpenAI API 时发生错误: {e}",
            tool_calls=None
        )
        return fake_message

from tools.tool_add.tool_direct import *
import importlib

def execute_tool_call(tool_call) -> str:
    """
    执行单个工具调用，返回执行结果字符串。

    Args:
        tool_call: 来自 OpenAI 响应的工具调用对象，包含 function.name 和 function.arguments

    Returns:
        工具执行结果的字符串表示。
    """
    with open('./tools/tool_add/tool_abs.txt', 'r') as f:
        created_tools = [line.strip() for line in f if line.strip()]
    print(created_tools)
    func_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    print(arguments)
    if func_name in TOOL_base_FUNCTIONS:
        try:
            result = TOOL_base_FUNCTIONS[func_name](**arguments)
            return str(result)
        except Exception as e:
            return f"工具执行异常：{e}"
    elif func_name in created_tools:
        try:
            module = importlib.import_module(f'tools.tool_add.tool_direct.{func_name}')
            func = getattr(module, func_name)
            result = func(**arguments)
            return str(result)
        except Exception as e:
            return f"工具执行异常：{e}"
    
    return f"未找到工具：{func_name}"