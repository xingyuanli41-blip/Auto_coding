"""
智能体大脑 v2.2 — 四状态机 + MCP 工具池 + 三层记忆

状态流转: THINK → EXECUTE ⇄ REFLECT → END
"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from tools.mcp_pool import MCPToolPool
from memory import MemoryManager
from memory.models import Message

from .adapters import BaseModelAdapter, ModelConfig, create_model_adapter
from .state import LoopMode, LoopState


# ==================== 兼容层 ====================

class ToolRegistry:
    """兼容旧 API 的注册表，委托给 MCPToolPool"""

    def __init__(self, pool: MCPToolPool):
        self._pool = pool

    def register(self, name: str, description: str, function: Callable, parameters: Optional[Dict] = None):
        self._pool.add_live_tool(name, description, function, parameters)

    def create_tool_dynamically(self, name: str, code: str, description: str = "") -> bool:
        result = self._pool.register_tool(name, description, code)
        return "成功" in result

    def get_tool(self, name: str) -> Optional[Dict]:
        return self._pool.get_tool(name)

    def list_tools(self) -> List[Dict]:
        return self._pool.list_tools()

    def execute(self, name: str, arguments: Dict) -> str:
        return self._pool.execute(name, arguments)


# ==================== 智能体大脑 ====================

class AgentBrain:
    """
    智能体大脑 v2.2 — 四状态机

    THINK  → 分析任务、制定计划、判断是否需要工具
    EXECUTE → 调用工具 / 创建工具 / 完成任务
    REFLECT → 错误分析、寻找解决方案、修正计划
    END    → 输出最终答案
    """

    def __init__(
        self,
        model_configs: List[ModelConfig],
        tools: Optional[ToolRegistry] = None,
        tool_pool: Optional[MCPToolPool] = None,
        system_prompt: str = "你是一个有用的AI助手，擅长规划和使用工具完成任务。",
        max_iterations: int = 10,
        short_memory_path: str = "./memory/archives/conversation_memory.json",
        long_memory_path: str = "./memory/archives/long_term_archive.json",
        openai_client: Any = None,
    ):
        # 模型
        self.models = [create_model_adapter(c) for c in model_configs]
        self.current_model_idx = 0
        self.think_model_idx = 0

        # MCP 工具池
        self.tool_pool = tool_pool or MCPToolPool(
            pool_file="./tools/mcp_tools.json", code_dir="./tools/tool_add/tool_direct")
        self.tools = tools or ToolRegistry(self.tool_pool)

        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

        # 三层记忆
        self.openai_client = openai_client
        self.memory = MemoryManager(
            client=openai_client,
            model=self.models[0].get_model_name() if self.models else "gpt-4o",
            working_memory_file=short_memory_path,
            long_term_memory_file=long_memory_path,
        )

        self.state = LoopState(max_iterations=max_iterations)
        self._register_brain_tools()

    # ==================== 模型管理 ====================

    @property
    def current_model(self) -> BaseModelAdapter:
        return self.models[self.current_model_idx]

    @property
    def think_model(self) -> BaseModelAdapter:
        return self.models[self.think_model_idx]

    def switch_model(self, index: int = None):
        if index is None:
            self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
        else:
            self.current_model_idx = index % len(self.models)
        print(f"🔄 切换到模型: {self.current_model.get_model_name()}")

    # ==================== 工具辅助 ====================

    def _messages_to_dict_list(self, messages: List) -> List[Dict]:
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get('content') is None and msg.get('tool_calls'):
                    msg = {**msg, 'content': '[tool_call]'}
                result.append(msg)
            elif hasattr(msg, 'to_dict'):
                d = msg.to_dict()
                if d.get('content') is None and d.get('tool_calls'):
                    d = {**d, 'content': '[tool_call]'}
                result.append(d)

        # 过滤孤立的 tool 消息（DeepSeek 要求前置 assistant+tool_calls）
        filtered = []
        last_assistant_with_tc = None
        for m in result:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                last_assistant_with_tc = m
                filtered.append(m)
            elif m.get("role") == "tool":
                if last_assistant_with_tc is not None:
                    filtered.append(m)
            else:
                filtered.append(m)
        return filtered

    def _register_brain_tools(self):
        """注册大脑专属工具到 MCP 池"""

        def switch_model_tool(model_index: int = None):
            self.switch_model(model_index)
            return f"已切换到模型: {self.current_model.get_model_name()}"

        def list_models():
            return "\n".join(f"{i}: {m.get_model_name()}" for i, m in enumerate(self.models))

        def get_status():
            return (f"迭代: {self.state.iteration}/{self.max_iterations}, "
                    f"模式: {self.state.mode.value}, "
                    f"错误: {self.state.error_count}次, "
                    f"模型: {self.current_model.get_model_name()}, "
                    f"MCP工具: {self.tool_pool.get_tool_count()['total']}")

        def get_memory_summary():
            return self.memory.summary()

        def search_long_memory(keyword: str):
            results = self.memory.recall(keyword, top_k=5, include_working=False)
            if results:
                lines = [f"找到 {len(results)} 条相关记忆:"]
                for r in results:
                    lines.append(f"- [{r.get('source', '?')}] {r.get('content', '')[:100]}")
                return "\n".join(lines)
            return f"未找到与 '{keyword}' 相关的记忆"

        def list_mcp_tools():
            return self.tool_pool.summary()

        def search_mcp_tools(query: str):
            results = self.tool_pool.search(query)
            if results:
                return f"找到 {len(results)} 个匹配:\n" + "\n".join(
                    f"- {r['name']}: {r['description']}" for r in results)
            return f"未找到与 '{query}' 匹配的工具，可使用 create_tool 创建"

        self.tool_pool.add_live_tool("switch_model", "切换到另一个可用模型", switch_model_tool)
        self.tool_pool.add_live_tool("list_models", "列出所有可用模型", list_models)
        self.tool_pool.add_live_tool("get_status", "获取当前智能体运行状态", get_status)
        self.tool_pool.add_live_tool("get_memory_summary", "获取对话记忆统计", get_memory_summary)
        self.tool_pool.add_live_tool("search_long_memory", "搜索记忆", search_long_memory)
        self.tool_pool.add_live_tool("list_mcp_tools", "列出MCP池中所有工具", list_mcp_tools)
        self.tool_pool.add_live_tool("search_mcp_tools", "搜索MCP工具池", search_mcp_tools)

    # ==================== 重置 ====================

    def reset(self, task: str = None):
        """重置状态，注入系统提示和记忆上下文"""
        msgs = self.memory.working_memory.messages
        memory_ctx = self.memory.get_context_for_llm()
        full_prompt = self.system_prompt
        if memory_ctx:
            full_prompt += "\n\n" + memory_ctx

        sys_msg = Message.system_message(content=full_prompt)
        if msgs and msgs[0].role == "system":
            msgs[0] = sys_msg
        else:
            msgs.insert(0, sys_msg)

        self.state = LoopState(max_iterations=self.max_iterations)
        if task:
            self.state.task = task

    # ==================== 四状态机 ====================

    def _think(self) -> LoopMode:
        """THINK: 分析任务，判断是否已完成 → EXECUTE 或 END"""
        # 构建已完成步骤的摘要
        history_brief = self._build_history_brief()

        extra = ""
        if self.state.last_error:
            extra += f"\n⚠️ 上次错误: {self.state.last_error}"
        if self.state.reflection:
            extra += f"\n💡 反思建议: {self.state.reflection}"

        prompt = f"""你是任务分析器。审视当前任务和已执行的操作，判断下一步。

## 原始任务
{self.state.task}

## 已执行的操作
{history_brief if history_brief else "(尚无)"}
{extra}

## 判断标准
- 如果用户要求的**所有子任务都已完成**且有明确结果 → done=true
- 如果还需要调用工具获取数据/执行操作 → done=false, needs_tools=true
- 如果可以基于已有信息直接回答 → done=true
- 不要反复验证已经成功完成的操作

输出 JSON:
{{"thought":"简要分析","plan":"下一步(如果未完成)","done":true/false,"needs_tools":true/false,"final_answer":"最终答案(如果done=true)"}}
"""
        messages = self._messages_to_dict_list(self.memory.working_memory.messages) + [
            {"role": "user", "content": prompt}]
        response = self.think_model.chat(messages)

        try:
            parsed = json.loads(response["content"])
            self.state.thought = parsed.get("thought", "")
            self.state.plan = parsed.get("plan", "")
            done = parsed.get("done", False)
            needs = parsed.get("needs_tools", True)
            if done:
                self.state.final_answer = parsed.get("final_answer", self.state.thought)
        except json.JSONDecodeError:
            self.state.thought = response["content"]
            done = False
            needs = True

        self.memory.working_memory.add_message(
            Message(role="assistant", content=response["content"] or "[think]"))
        self.state.history.append({"mode": "think", "thought": self.state.thought})
        self.state.last_error = ""

        return LoopMode.END if done else LoopMode.EXECUTE

    def _build_history_brief(self) -> str:
        """构建已执行操作的简要摘要"""
        if not self.state.history:
            return ""
        briefs = []
        for h in self.state.history[-10:]:  # 最近10步
            mode = h.get("mode", "")
            if mode == "execute":
                tool = h.get("tool", "?")
                result = str(h.get("result", ""))[:80]
                briefs.append(f"[执行] {tool} → {result}")
            elif mode == "reflect":
                briefs.append(f"[反思] {str(h.get('reflection',''))[:60]}")
            elif mode == "think":
                briefs.append(f"[思考] {str(h.get('thought',''))[:60]}")
        return "\n".join(briefs) if briefs else ""

    def _make_assistant_tool_msg(self, response: Dict, tool_calls: List[Dict]) -> Message:
        """构建带 tool_calls 的 assistant 消息（DeepSeek 要求）"""
        from memory.models import Function as Func, ToolCall as TC
        tcs = [TC(id=tc["id"], type=tc.get("type", "function"),
                   function=Func(name=tc["function"]["name"],
                                 arguments=tc["function"]["arguments"]))
               for tc in tool_calls]
        return Message(role="assistant", content=response.get("content") or "[tool_call]",
                       tool_calls=tcs)

    def _execute(self) -> LoopMode:
        """EXECUTE: 调用工具 → REFLECT(出错) / THINK(继续) / END(完成)"""
        prompt = f"""任务: {self.state.task}
计划: {self.state.plan}

执行。可调用工具、创建工具(create_tool)、或任务完成时直接回答。
"""
        messages = self._messages_to_dict_list(self.memory.working_memory.messages) + [
            {"role": "user", "content": prompt}]
        response = self.current_model.chat(messages, self.tool_pool.list_tools())

        tool_calls = response.get("tool_calls")
        if not tool_calls:
            self.state.final_answer = response["content"] or ""
            self.state.result = response["content"] or ""
            return LoopMode.END

        tc = tool_calls[0]
        func_name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            args = {}

        result = self.tool_pool.execute(func_name, args)

        # 错误检测
        is_error = any(kw in result for kw in ["失败", "错误", "Error", "异常", "不存在"])
        if is_error and func_name != "create_tool":
            self.state.last_error = result
            self.state.error_count += 1
            self.memory.working_memory.add_message(
                self._make_assistant_tool_msg(response, tool_calls))
            self.memory.working_memory.add_message(
                Message.tool_message(content=result, name=func_name, tool_call_id=tc["id"]))
            self.state.result = result
            self.state.history.append({"mode": "execute", "tool": func_name, "result": result, "error": True})
            return LoopMode.REFLECT

        if func_name == "create_tool" and "成功" in result:
            self.state.created_tools.append({"name": args.get("name", "?"), "description": args.get("description", "")})

        self.memory.working_memory.add_message(
            self._make_assistant_tool_msg(response, tool_calls))
        self.memory.working_memory.add_message(
            Message.tool_message(content=result, name=func_name, tool_call_id=tc["id"]))
        self.state.result = result
        self.state.history.append({"mode": "execute", "tool": func_name, "result": result})

        if func_name == "create_tool" and "成功" in result:
            return LoopMode.THINK
        return LoopMode.THINK

    def _reflect(self) -> LoopMode:
        """REFLECT: 分析错误 → EXECUTE(重试) / THINK(重新分析) / END(放弃)"""
        if self.state.error_count >= self.state.max_retries:
            self.state.final_answer = f"已重试{self.state.error_count}次，无法完成。最后错误: {self.state.last_error}"
            return LoopMode.END

        prompt = f"""任务出错，请分析:

任务: {self.state.task}
错误: {self.state.last_error}
重试: {self.state.error_count}/{self.state.max_retries}

输出 JSON:
{{"root_cause":"原因","solution":"方案","can_retry":true/false,"next_action":"retry_execute/rethink/give_up","adjusted_plan":"修正计划"}}
"""
        messages = self._messages_to_dict_list(self.memory.working_memory.messages) + [
            {"role": "user", "content": prompt}]
        response = self.think_model.chat(messages)

        try:
            parsed = json.loads(response["content"])
            self.state.reflection = f"{parsed.get('root_cause','')} | {parsed.get('solution','')}"
            next_action = parsed.get("next_action", "give_up")
            adjusted = parsed.get("adjusted_plan", "")
            can_retry = parsed.get("can_retry", False)
        except json.JSONDecodeError:
            self.state.reflection = response["content"]
            next_action, adjusted, can_retry = "retry_execute", "", True

        self.memory.working_memory.add_message(
            Message(role="assistant", content=response["content"] or "[reflect]"))
        self.state.history.append({"mode": "reflect", "reflection": self.state.reflection})

        if adjusted:
            self.state.plan = adjusted

        if next_action == "retry_execute" and can_retry:
            return LoopMode.EXECUTE
        elif next_action == "rethink":
            return LoopMode.THINK
        return LoopMode.END

    # ==================== 主循环 ====================

    def run(self, task: str, verbose: bool = True) -> str:
        """四状态机主循环"""
        self.reset(task)

        if verbose:
            print(f"🚀 {task}")
            print(f"🧠 {self.think_model.get_model_name()} ⚡ {self.current_model.get_model_name()}")
            print(f"🛠️ {self.tool_pool.get_tool_count()['total']} 个工具")

        self.state.mode = LoopMode.THINK

        while self.state.iteration < self.state.max_iterations:
            self.state.iteration += 1
            if verbose:
                print(f"\n{'─'*35}\n🔄 第{self.state.iteration}轮 [{self.state.mode.value}]")

            if self.state.mode == LoopMode.THINK:
                if verbose: print("🤔 思考...")
                next_mode = self._think()
                if verbose: print(f"   → {next_mode.value}")

            elif self.state.mode == LoopMode.EXECUTE:
                if verbose: print("⚙️ 执行...")
                next_mode = self._execute()
                if verbose: print(f"   → {next_mode.value}")

            elif self.state.mode == LoopMode.REFLECT:
                if verbose: print("🔍 反思...")
                next_mode = self._reflect()
                if verbose: print(f"   → {next_mode.value}")
            else:
                break

            self.state.mode = next_mode
            self.memory.check_and_consolidate()

            if next_mode == LoopMode.END:
                break

        self._save_memory()

        if verbose:
            print(f"\n{'─'*35}")
            msg = "✅ 完成" if self.state.mode == LoopMode.END else f"⚠️ 达上限({self.state.max_iterations}轮)"
            print(f"{msg} | 共{self.state.iteration}轮 | 错误{self.state.error_count}次")
            print(f"📝 {self.state.final_answer or self.state.result}")

        return self.state.final_answer or self.state.result or "任务未完成"

    # ==================== 持久化 ====================

    def _save_memory(self):
        try:
            self.memory.save()
        except Exception as e:
            print(f"保存记忆失败: {e}")

    def get_history(self) -> List[Dict]:
        return self.state.history

    def get_memory_count(self) -> Dict:
        return self.memory.get_stats()


# ==================== 便捷函数 ====================

def create_agent(
    model_configs: List[ModelConfig],
    system_prompt: str = "你是一个有用的AI助手，擅长分析和解决问题。",
    tools: Optional[Dict[str, Callable]] = None,
    max_iterations: int = 10,
    openai_client: Any = None,
    tool_pool: Optional[MCPToolPool] = None,
) -> AgentBrain:
    """创建 AgentBrain 的便捷函数"""
    if tool_pool is None:
        tool_pool = MCPToolPool(
            pool_file="./tools/mcp_tools.json", code_dir="./tools/tool_add/tool_direct")
    if tools:
        for name, func in tools.items():
            tool_pool.add_live_tool(name, func.__doc__ or "", func)
    return AgentBrain(
        model_configs=model_configs, tool_pool=tool_pool,
        system_prompt=system_prompt, max_iterations=max_iterations,
        openai_client=openai_client,
    )
