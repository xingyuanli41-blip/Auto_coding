import sys, json, os
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from brain.call import call_openai_with_tools
from config import config
from tools.mcp_pool import MCPToolPool
from memory import MemoryManager
from memory.models import Message, Function, ToolCall

# ═══════════════════════════════════════
# 初始化
# ═══════════════════════════════════════

client = config.create_llm_client()
tool_pool = MCPToolPool(pool_file=config.tools_pool_file, code_dir=config.tools_code_dir, workspace_dir=config.tools_workspace_dir, limitation_file="./limitation.txt")
print(f"🛠️ {tool_pool.summary()}")

# 启动时自动清理过期工具
stale = tool_pool.get_stale_tools(days_unused=14, min_usage=0)
if stale:
    result = tool_pool.cleanup_stale_tools(days_unused=14, min_usage=0, dry_run=False)
    print(f"🧹 {result}")

# 三层记忆管理器（替代旧的 Memory + handle_memory_overflow）
memory = MemoryManager(
    client=client,
    model=config.llm_model,
    working_token_budget=config.memory_working_token_budget,
    short_term_max_items=config.memory_short_term_max,
    long_term_max_items=config.memory_long_term_max,
    working_memory_file=config.memory_working_file,
    long_term_memory_file=config.memory_long_term_file,
)
print(f"🧠 {memory.summary()}")

# 注入系统提示（从 config.json 读取）
memory.add_system_message(config.brain_system_prompt)

# ═══════════════════════════════════════
# / 命令系统
# ═══════════════════════════════════════

def _handle_command(cmd: str) -> str:
    """处理 / 命令。返回 "exit" 则退出。"""
    parts = cmd.split()
    action = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if action in ("/help", "/?"):
        print("""
╔══════════════════════════════════════╗
║         / 命令列表                   ║
╠══════════════════════════════════════╣
║ /model [name]  查看/切换模型          ║
║ /config        查看当前配置           ║
║ /tools         查看MCP工具池          ║
║ /memory        查看三层记忆           ║
║ /system [text] 查看/修改系统提示词    ║
║ /temp  [value] 查看/调整temperature   ║
║ /clear         清空对话记忆           ║
║ /save          手动保存记忆           ║
║ /exit, /quit   退出                   ║
║ /help, /?      显式此帮助             ║
╚══════════════════════════════════════╝
""")

    elif action == "/model":
        if arg:
            config._data["llm"]["model"] = arg
            globals()["client"] = config.create_llm_client()
            print(f"✅ 模型已切换为: {arg}")
        else:
            print(f"📌 当前: {config.llm_model}  |  /model <模型名> 切换")

    elif action == "/config":
        safe = json.loads(json.dumps(config._data))
        if "llm" in safe and safe["llm"].get("api_key"):
            safe["llm"]["api_key"] = safe["llm"]["api_key"][:10] + "..."
        print(json.dumps(safe, ensure_ascii=False, indent=2))

    elif action == "/tools":
        print(tool_pool.summary())
        stale_t = tool_pool.get_stale_tools(days_unused=7, min_usage=0)
        if stale_t:
            print(f"\n⏰ 7天未用: {[s['name'] for s in stale_t]}")

    elif action == "/memory":
        print(memory.summary())
        if memory.long_term_memory.items:
            print("\n📚 最近记忆:")
            for item in memory.long_term_memory.items[-5:]:
                print(f"   - {item.content[:100]}")

    elif action in ("/system", "/sys"):
        sys_msgs = [m for m in memory.working_memory.messages if m.role == "system"]
        if arg:
            new_prompt = " ".join(parts[1:])
            if sys_msgs:
                sys_msgs[0].content = new_prompt
            else:
                memory.working_memory.messages.insert(0, Message.system_message(content=new_prompt))
            print(f"✅ 系统提示已更新")
        else:
            print(f"📌 {sys_msgs[0].content if sys_msgs else '(无)'}")

    elif action == "/temp":
        if arg:
            try:
                config._data["llm"]["temperature"] = float(arg)
                print(f"✅ temperature = {arg}")
            except ValueError:
                print(f"❌ 无效值")
        else:
            print(f"📌 temperature = {config.llm_temperature}")

    elif action == "/clear":
        count = len(memory.working_memory.messages)
        memory.working_memory.clear()
        print(f"🧹 已清空 {count} 条消息")

    elif action == "/save":
        memory.save()
        print("💾 记忆已保存")

    elif action in ("/exit", "/quit"):
        return "exit"

    else:
        print(f"❌ 未知命令: {action}，输入 /help 查看帮助")

    return "ok"


# ═══════════════════════════════════════
# 主循环
# ═══════════════════════════════════════

while True:
    # 记忆溢出自动处理
    memory.check_and_consolidate()

    user_input = input("User：").strip()
    if not user_input:
        continue

    # / 命令 或 exit
    if user_input.startswith("/") or user_input.lower() == "exit":
        action = _handle_command(user_input if user_input.startswith("/") else "/exit")
        if action == "exit":
            try:
                memory.save()
                print(f"💾 长期记忆已保存（{len(memory.long_term_memory.items)} 项）")
                print(f"🛠️ {tool_pool.summary()}")
            except Exception as e:
                print(f"保存出错：{e}")
            break
        continue

    # 正常对话
    memory.add_user_message(user_input)
    print(f"User：{user_input}\n")

    iteration = 0
    max_iter = config.brain_max_iterations

    while iteration < max_iter:
        iteration += 1
        messages_dict = memory.get_working_context()
        tool_total = tool_pool.list_tools()
        print(f"📦 {len(tool_total)} 个工具 | 第{iteration}轮")

        assistant_message = call_openai_with_tools(
            messages_dict, tool_total, client=client, model=config.llm_model)

        if assistant_message.tool_calls:
            # 构建带 tool_calls 的 assistant 消息（DeepSeek 要求）
            tcs = [ToolCall(id=tc.id, type="function",
                   function=Function(name=tc.function.name, arguments=tc.function.arguments))
                   for tc in assistant_message.tool_calls]
            msg = Message(role="assistant", content=assistant_message.content or "[tool_call]",
                          tool_calls=tcs)
            memory.working_memory.add_message(msg)

            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                print(f"  🔧 {func_name}({json.dumps(args, ensure_ascii=False)[:100]})")

                result = tool_pool.execute(func_name, args)
                print(f"  📋 {result[:120]}{'...' if len(result) > 120 else ''}")

                if func_name == "create_tool" and "成功" in result:
                    print(f"  🆕 新工具入池!")

                memory.add_tool_message(
                    content=result, name=func_name, tool_call_id=tool_call.id)
                memory.check_and_consolidate()
        else:
            final_answer = assistant_message.content or "[无内容]"
            memory.add_assistant_message(final_answer)
            memory.remember_turn(user_input, final_answer)
            print(f"🤖 {final_answer}\n")
            break
    else:
        print(f"⚠️ 达到最大轮次 {max_iter}，本轮终止。\n")
