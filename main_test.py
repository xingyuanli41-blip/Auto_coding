import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
from typing import List, Dict, Any
from openai import OpenAI
from ai.agent_call import *
from tools.tool_base_abs import TOOL_base_DESCRIPTIONS,TOOL_base_FUNCTIONS


from tools.information import Memory,Message

client = OpenAI(
    api_key="sk-ujecYWuE6lhVJ3zeZqS3Qb9jTu5HY2jOKd6X83Ho9lmznGNu",
    base_url="https://poloai.top/v1", 
)


if __name__ == "__main__":

    user_input = None
    conversation_memory = Memory(max_messages=100)
    with open('pre_solve_english.txt', 'r',encoding='utf-8') as file:
        content = file.read()
    flag = 1
    print("Start the conversation (type 'exit' to quit)")
    while True:
        if flag == 1:
            user_msg = Message.user_message(content=content)
            flag = 0
        else:
            user_input = input("User：")
            if user_input.lower() == 'exit':
                break
            user_msg = Message.user_message(content=user_input)
        conversation_memory.add_message(user_msg)
        print(f"User：{user_input or content}\n")

        iteration = 0
        max_iter =1
        while iteration < max_iter:
            iteration += 1
            messages_dict = conversation_memory.to_dict_list()
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
            print(len(tool_total))
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
            else:
                final_answer = assistant_message.content or "[no content]"
                print(f"Assistant：{final_answer}")
                break
        else:
            print(f"The maximum number of iterations {max_iter} has been reached; this round of conversation will be stopped.")
