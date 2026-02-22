import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
from typing import List, Dict, Any
from openai import OpenAI
from brain.agent_call import *

# 工具创建及调用
from tools.tool_base_abs import TOOL_base_DESCRIPTIONS,TOOL_base_FUNCTIONS
# 记忆系统
from tools.memory_base import Memory,LongTermMemory,Message
from tools.memory_store import handle_memory_overflow,summarize_func
from tools.memory_retrieval import load_memories_for_conversation

client = OpenAI(
    api_key="sk-",
    base_url="https://poloai.top/v1", 
)

# 要存储的短期和长期记忆地址
short_memory = "./memory_archives/conversation_memory.json"
long_memory = "./memory_archives/long_term_archive.json"

# 初始化短期记忆和长期记忆
conversation_memory = load_memories_for_conversation(short_memory,long_memory)    # 读取记忆，读取长期保存的记忆加上之前推出之前短期记忆
long_term_memory = LongTermMemory(max_items=1000)      # 长期记忆，最多保留 1000 条记忆项

while True:
    '''
    a、当前短期记忆过多进行总结与备份
    b、初始化时总结与上次遗留记忆合并超过最大值时重新总结释放当前记忆
    '''
    handle_memory_overflow(
        memory=conversation_memory,
        long_term_memory=long_term_memory,
        summarize_func=summarize_func,
        client=client,
    )
    user_input = input("User：")
    if user_input == 'exit':
        try:
            # 保存短期记忆（可自定义文件名）
            conversation_memory.save_to_file(short_memory)
            print(f"短期记忆已保存至 {short_memory}")

            # 保存长期记忆
            long_term_memory.save_to_file(long_memory)
            print(f"长期记忆已保存至 {long_memory}")
        except Exception as e:
            print(f"保存记忆时出错：{e}")
        break
    user_msg = Message.user_message(content=user_input)

    # 添加用户消息到短期记忆
    conversation_memory.add_message(user_msg)
    print(f"User：{user_input or content}\n")
    

    # 模型交互轮次控制
    iteration = 0
    max_iter = 10   # 允许最多 1 轮工具调用循环（可根据需要调整）

    while iteration < max_iter:
        iteration += 1

        # 准备消息字典列表
        messages_dict = conversation_memory.to_dict_list()

        # 加载工具描述
        tool_total = []
        with open('./tools/tool_add/tool_description.txt', 'r', encoding='utf-8') as f:
            extra_tools = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        tool_dict = json.loads(line)
                        extra_tools.append(tool_dict)
                    except json.JSONDecodeError as e:
                        print(f"warning：Skip invalid sentence：{line}, 错误：{e}")
            tool_total = TOOL_base_DESCRIPTIONS + extra_tools
        print(f"Loaded {len(tool_total)} tools.")

        # 调用模型（可能返回工具调用）
        assistant_message = call_openai_with_tools(messages_dict, tool_total, client=client)
        conversation_memory.add_message(assistant_message)

        if assistant_message.tool_calls:
            print(f"Assistant requests to invoke tool (Round {iteration})：")
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                args = tool_call.function.arguments
                print(f"  - {func_name}({args})")
                result = execute_tool_call(tool_call)
                print(f"    tool return：{result[:100]}{'...' if len(result) > 100 else ''}")

                tool_response = Message.tool_message(
                    content=result,
                    name=func_name,
                    tool_call_id=tool_call.id
                )
                conversation_memory.add_message(tool_response)

                handle_memory_overflow(
                    memory=conversation_memory,
                    long_term_memory=long_term_memory,
                    summarize_func=summarize_func,
                    client=client,
                )
        else:
            final_answer = assistant_message.content or "[no content]"
            print(f"Assistant：{final_answer}")
            break   # 无工具调用，本轮结束
    else:
        print(f"The maximum number of iterations {max_iter} has been reached; this round of conversation will be stopped.")
