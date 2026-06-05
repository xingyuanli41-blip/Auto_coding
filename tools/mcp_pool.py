"""
MCP 工具池 —— 统一管理所有工具，支持 LLM 自主创建新工具。

所有工具存放于 tools/tool_add/tool_direct/*.py，元数据在 tools/mcp_tools.json。
不再区分 builtin/dynamic —— 所有工具都是平等的，从文件加载执行。

用法:
    from tools.mcp_pool import MCPToolPool

    pool = MCPToolPool()
    tools = pool.list_tools()           # OpenAI 格式的工具列表
    result = pool.execute("read_file", {"path": "test.txt"})
"""

import json
import logging
import os
import sys
import re
import importlib
import inspect
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# LLM 生成代码中禁止使用的危险模块/函数
DANGEROUS_PATTERNS = [
    # 危险模块导入
    (r"import\s+os\s*$", "禁止导入 os 模块（已有 read_file/write_file 等工具替代）"),
    (r"from\s+os\s+import", "禁止从 os 导入"),
    (r"import\s+subprocess", "禁止导入 subprocess（已有 run_command 工具替代）"),
    (r"from\s+subprocess\s+import", "禁止从 subprocess 导入"),
    (r"import\s+shutil", "禁止导入 shutil（已有文件操作工具替代）"),
    (r"from\s+shutil\s+import", "禁止从 shutil 导入"),
    (r"import\s+sys\s*$", "禁止导入 sys 模块"),
    (r"from\s+sys\s+import", "禁止从 sys 导入"),
    # 危险函数调用
    (r"\beval\s*\(", "禁止使用 eval()"),
    (r"\bexec\s*\(", "禁止使用 exec()"),
    (r"\b__import__\s*\(", "禁止使用 __import__()"),
    (r"\bos\.system\s*\(", "禁止使用 os.system()"),
    (r"\bos\.popen\s*\(", "禁止使用 os.popen()"),
    (r"\bshutil\.rmtree\s*\(", "禁止使用 shutil.rmtree()"),
    (r"\bsubprocess\.(call|run|Popen)\s*\(", "禁止使用 subprocess"),
    # 文件系统遍历
    (r"\bos\.walk\s*\(", "禁止遍历文件系统"),
    (r"\bos\.remove\s*\(", "使用 delete_file 工具替代 os.remove"),
    (r"\bos\.rmdir\s*\(", "使用 delete_file 工具替代 os.rmdir"),
]


class MCPToolPool:
    """
    统一 MCP 工具池。

    所有工具元数据存在 mcp_tools.json，代码存在 code_dir/*.py。
    启动时从 JSON 加载，执行时从 .py 文件动态加载函数。
    支持 live_tool（内存中注册的、不持久化的工具，供 AgentBrain 使用）。

    线程安全：所有读写操作通过 _lock 保护。
    """

    def __init__(
        self,
        pool_file: str = "./tools/mcp_tools.json",
        code_dir: str = "./tools/tool_add/tool_direct",
        workspace_dir: str = "./workspace",
        limitation_file: str = "./limitation.txt",
    ):
        """
        Args:
            pool_file: mcp_tools.json 路径
            code_dir: 工具 .py 文件存放目录
            workspace_dir: 工作空间根目录
            limitation_file: 安全限制配置（禁用命令/文件）
        """
        self.pool_file = pool_file
        self.code_dir = code_dir
        self.workspace_dir = os.path.abspath(workspace_dir)
        os.makedirs(self.workspace_dir, exist_ok=True)
        self._lock = threading.Lock()

        # 加载安全限制
        self._forbidden_commands: List[str] = []
        self._forbidden_files: List[str] = []
        self._load_limitations(limitation_file)

        # 内存中的工具注册表: name → {name, description, code_file, parameters, enabled, ...}
        self.tools: Dict[str, Dict] = {}

        # 动态工具的函数缓存: name → function
        self._func_cache: Dict[str, Callable] = {}

        # 内存工具（session-only，不持久化）: name → function
        self._live_functions: Dict[str, Callable] = {}
        self._live_tools: Dict[str, Dict] = {}

        self._load_or_init()

    # ==================== 加载 / 保存 ====================

    def _load_or_init(self):
        """加载 mcp_tools.json，若不存在则创建空池"""
        if os.path.exists(self.pool_file) and os.path.getsize(self.pool_file) > 0:
            try:
                self._load()
                print(f"[MCP] 从 {self.pool_file} 加载了 {len(self.tools)} 个工具")
            except Exception as e:
                # 备份损坏的 JSON 文件
                backup = self.pool_file + ".backup"
                try:
                    import shutil
                    shutil.copy2(self.pool_file, backup)
                    print(f"[MCP] ⚠️ JSON 损坏，已备份至 {backup}")
                except Exception:
                    pass
                print(f"[MCP] 加载失败: {e}，创建空池（原文件已备份）")
                self.tools = {}
                self._save()
        else:
            self.tools = {}
            self._save()
            print(f"[MCP] 创建空工具池: {self.pool_file}")

    def _load(self):
        """从 mcp_tools.json 加载"""
        with open(self.pool_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tools = {}
        for entry in data:
            name = entry.get("name", "")
            if name:
                self.tools[name] = entry

    def _save(self):
        """保存到 mcp_tools.json（不含 live_tools）"""
        os.makedirs(os.path.dirname(self.pool_file) or ".", exist_ok=True)
        data = list(self.tools.values())
        with open(self.pool_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ==================== 工具列表（OpenAI 格式） ====================

    def list_tools(self, include_create_tool: bool = True) -> List[Dict]:
        """
        返回 OpenAI 兼容的工具描述列表，合并持久化工具和 live 工具。
        """
        tool_list = []

        # 持久化工具（排除已禁用的）
        for tool in self.tools.values():
            if tool.get("enabled", True):  # 默认启用
                tool_list.append(self._to_openai_format(tool))

        # Live 工具（session-only）
        for tool in self._live_tools.values():
            tool_list.append(self._to_openai_format(tool))

        # create_tool 元工具（始终可用）
        if include_create_tool and "create_tool" not in self.tools and "create_tool" not in self._live_tools:
            tool_list.append(self._create_tool_schema())

        return tool_list

    def _to_openai_format(self, tool: Dict) -> Dict:
        """转换单个工具为 OpenAI 格式"""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("parameters", {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }),
            },
        }

    def _create_tool_schema(self) -> Dict:
        """create_tool 元工具的 schema"""
        return {
            "type": "function",
            "function": {
                "name": "create_tool",
                "description": (
                    "创建一个新的工具函数并注册到 MCP 工具池中。"
                    "当你发现现有工具无法完成用户任务时，调用此函数创建新工具。"
                    "创建成功后新工具立即可用，你可以在下一轮直接调用它。"
                    "code 必须是完整可运行的 Python 函数代码，"
                    "返回类型标注为 str，使用类型标注标注所有参数。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "工具名称，使用英文 snake_case",
                        },
                        "description": {
                            "type": "string",
                            "description": "工具的详细功能描述，说明用途、参数含义、返回值格式",
                        },
                        "code": {
                            "type": "string",
                            "description": (
                                "完整的 Python 函数代码。要求：\n"
                                "1. 函数名与 name 参数一致\n"
                                "2. 所有参数使用类型标注\n"
                                "3. 返回类型标注为 -> str\n"
                                "4. 包含完整的 import 语句\n"
                                "5. 使用 try/except 处理异常"
                            ),
                        },
                    },
                    "required": ["name", "description", "code"],
                },
            },
        }

    # ==================== 查询 / 搜索 ====================

    def has_tool(self, name: str) -> bool:
        """检查工具是否存在（含持久化和 live）"""
        return name in self.tools or name in self._live_tools

    def get_tool(self, name: str) -> Optional[Dict]:
        """获取工具元数据"""
        return self.tools.get(name) or self._live_tools.get(name)

    def get_tool_count(self) -> Dict[str, int]:
        """获取工具统计"""
        return {
            "total": len(self.tools) + len(self._live_tools),
            "persisted": len(self.tools),
            "live": len(self._live_tools),
        }

    def search(self, query: str) -> List[Dict]:
        """按描述关键词搜索工具"""
        query_lower = query.lower()
        results = []
        all_tools = {**self.tools, **self._live_tools}
        for tool in all_tools.values():
            name = tool.get("name", "").lower()
            desc = tool.get("description", "").lower()
            if query_lower in name or query_lower in desc:
                results.append({
                    "name": tool["name"],
                    "description": tool["description"],
                })
        return results

    # ==================== Live Tool（session-only，不持久化） ====================

    def add_live_tool(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[Dict] = None,
    ):
        """
        添加一个 session-only 的内存工具（不写入 JSON，重启后消失）。
        适用场景：AgentBrain 注册依赖 self 的闭包工具。

        Args:
            name: 工具名称
            description: 工具描述
            function: 可调用对象
            parameters: OpenAI 参数 schema（不提供则自动提取）
        """
        self._live_functions[name] = function
        if parameters is None:
            parameters = self._extract_parameters(function)
        self._live_tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

    # ==================== 注册新工具（持久化） ====================

    def register_tool(
        self,
        name: str,
        description: str,
        code: str,
        parameters: Optional[Dict] = None,
    ) -> str:
        """
        注册一个新的持久化工具：

        1. 验证代码语法
        2. 写入 .py 文件到 code_dir
        3. 注册到 JSON + 内存

        Returns:
            成功/失败信息字符串
        """
        # --- 验证名称 ---
        if not name or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            return f"错误：工具名称 '{name}' 不是合法的 Python 标识符"

        if self.has_tool(name):
            return f"错误：工具 '{name}' 已存在"

        # --- 安全检查 ---
        safety_issue = self._check_code_safety(code)
        if safety_issue:
            return safety_issue

        # --- 清理代码 ---
        code = self._clean_code(code)

        # --- 语法检查 + 提取函数 ---
        try:
            compile(code, f"<{name}>", "exec")
        except SyntaxError as e:
            return f"错误：代码语法错误 —— {e}"

        found_func = None
        try:
            local_ns: Dict[str, Any] = {}
            exec(code, {"__builtins__": __builtins__}, local_ns)
            for val in local_ns.values():
                if callable(val) and not isinstance(val, type):
                    found_func = val
                    break
            if found_func is None:
                return f"错误：代码中未找到可调用函数"
        except Exception as e:
            return f"错误：代码执行验证失败 —— {e}"

        # 自动生成参数 schema
        if parameters is None:
            parameters = self._extract_parameters(found_func)

        # --- 写入 .py 文件 ---
        os.makedirs(self.code_dir, exist_ok=True)
        py_path = os.path.join(self.code_dir, f"{name}.py")
        try:
            with open(py_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            return f"错误：写入文件失败 —— {e}"

        # --- 注册到内存 ---
        code_file = os.path.join("tools", "tool_add", "tool_direct", f"{name}.py")
        self.tools[name] = {
            "name": name,
            "description": description,
            "type": "dynamic",
            "code_file": code_file,
            "parameters": parameters,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
        }
        # 不缓存 exec 出来的函数（可能丢失 import 上下文）
        # 首次调用时通过 importlib 从 .py 文件加载，确保模块级 import 正常工作

        # --- 持久化 ---
        self._save()

        return (
            f"✅ 工具 '{name}' 创建成功！\n"
            f"   描述: {description}\n"
            f"   文件: {py_path}\n"
            f"   现在可以调用 {name} 完成用户任务。"
        )

    def _clean_code(self, code: str) -> str:
        """清理 LLM 返回的代码（去除 markdown 包裹、修正中文标点）"""
        # 去除 markdown 代码块包裹
        match = re.search(r"```(?:python)?\s*\n(.*?)\n```", code, re.DOTALL)
        if match:
            code = match.group(1).strip()

        # 替换 LLM 可能误用的中文标点（在代码语法位置会导致 SyntaxError）
        replacements = {
            "。": ".",   # 。
            "，": ",",   # ，
            "：": ":",   # ：
            "；": ";",   # ；
            "（": "(",   # （
            "）": ")",   # ）
            "“": '"',   # "
            "”": '"',   # "
            "‘": "'",   # '
            "’": "'",   # '
            "！": "!",   # ！
            "？": "?",   # ？
            "＋": "+",   # ＋
            "－": "-",   # －
            "＊": "*",   # ＊
            "／": "/",   # ／
            "＝": "=",   # ＝
        }
        for cn, en in replacements.items():
            code = code.replace(cn, en)

        return code.strip()

    def _check_code_safety(self, code: str) -> Optional[str]:
        """
        检查 LLM 生成的代码是否包含危险操作。

        Returns:
            如果安全返回 None，否则返回错误描述字符串
        """
        for pattern, warning in DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.MULTILINE):
                return f"安全警告：{warning}。如需系统级操作，请使用 MCP 池中已有的工具。"
        return None

    def _extract_parameters(self, func: Callable) -> Dict:
        """从函数签名自动生成 OpenAI 参数 schema"""
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            return {"type": "object", "properties": {}, "required": []}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation is not inspect.Parameter.empty:
                if param.annotation is int:
                    param_type = "integer"
                elif param.annotation is float:
                    param_type = "number"
                elif param.annotation is bool:
                    param_type = "boolean"
                elif param.annotation is list:
                    param_type = "array"
                elif param.annotation is dict:
                    param_type = "object"

            properties[param_name] = {"type": param_type}

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    # ==================== 执行工具 ====================

    def execute(self, name: str, arguments: Dict) -> str:
        """
        执行工具调用。优先级：
        1. create_tool 元工具（特殊处理）
        2. Live 函数（内存中）
        3. 文件工具（从 .py 加载）
        """
        # === create_tool 元工具 ===
        if name == "create_tool":
            return self._execute_create_tool(arguments)

        # === Live 函数 ===
        if name in self._live_functions:
            return self._execute_live(name, arguments)

        # === 文件工具 ===
        tool = self.tools.get(name)
        if not tool:
            suggestions = self.search(name)
            if suggestions:
                names = ", ".join([s["name"] for s in suggestions[:5]])
                return f"错误：未找到工具 '{name}'。你可能想用: {names}"
            return (
                f"错误：未找到工具 '{name}'。"
                f"当前 MCP 池中有 {len(self.tools)} 个持久化工具 + "
                f"{len(self._live_tools)} 个内存工具。"
                f"如果确实需要，请调用 create_tool 创建。"
            )

        # 更新使用计数
        tool["usage_count"] = tool.get("usage_count", 0) + 1
        tool["last_used"] = datetime.now().isoformat()

        # === 安全检查：run_command ===
        if name == "run_command" and "command" in arguments:
            blocked = self._check_command(arguments["command"])
            if blocked:
                return blocked

        # === 安全检查：read_file 禁止读取敏感文件 ===
        if name == "read_file" and "path" in arguments:
            blocked = self._check_file_read(arguments["path"])
            if blocked:
                return blocked

        # === 工作空间路径校验（仅写/删/创建，读操作不受限） ===
        _FILE_TOOLS = {"write_file", "create_file", "delete_file", "create_directory"}
        if name in _FILE_TOOLS and "path" in arguments:
            try:
                arguments["path"] = self._resolve_path(arguments["path"])
            except PermissionError as e:
                return str(e)

        result = self._execute_file_tool(name, arguments, tool)

        # delete_tool 执行后刷新池状态（文件已由工具函数清理）
        if name == "delete_tool":
            self._func_cache.clear()
            self._load()

        return result

    def _execute_live(self, name: str, arguments: Dict) -> str:
        """执行 live 工具"""
        func = self._live_functions[name]
        try:
            result = func(**arguments)
            return str(result) if result is not None else "Done"
        except Exception as e:
            logger.warning("live 工具 '%s' 执行失败: %s", name, e)
            return f"执行工具 '{name}' 失败: {e}"

    def _execute_file_tool(self, name: str, arguments: Dict, tool: Dict) -> str:
        """执行文件工具（从 .py 加载）"""
        # 1. 缓存命中
        if name in self._func_cache:
            try:
                result = self._func_cache[name](**arguments)
                return str(result) if result is not None else "Done"
            except Exception as e:
                self._record_error(name)
                logger.warning("工具 '%s' 执行失败: %s", name, e)
                return f"执行工具 '{name}' 失败: {e}"

        # 2. 从 code_file 加载
        code_file = tool.get("code_file", "")
        if code_file and os.path.exists(code_file):
            try:
                func = self._load_func_from_file(name, code_file)
                if func:
                    self._func_cache[name] = func
                    result = func(**arguments)
                    return str(result) if result is not None else "Done"
            except Exception:
                logger.debug("从 code_file 加载 %s 失败，尝试其他方式", name, exc_info=True)

        # 3. 尝试从 code_dir 找
        py_path = os.path.join(self.code_dir, f"{name}.py")
        if os.path.exists(py_path):
            try:
                func = self._load_func_from_file(name, py_path)
                if func:
                    self._func_cache[name] = func
                    result = func(**arguments)
                    return str(result) if result is not None else "Done"
            except Exception as e:
                pass

        # 4. 回退：inline code
        code = tool.get("code", "")
        if code:
            try:
                local_ns: Dict[str, Any] = {}
                exec(code, {"__builtins__": __builtins__}, local_ns)
                for val in local_ns.values():
                    if callable(val) and not isinstance(val, type):
                        self._func_cache[name] = val
                        result = val(**arguments)
                        return str(result) if result is not None else "Done"
            except Exception as e:
                return f"执行工具 '{name}' 失败: {e}"

        self._record_error(name)
        return f"错误：无法加载工具 '{name}' 的代码"

    def _record_error(self, name: str):
        """记录工具执行错误"""
        tool = self.tools.get(name)
        if tool is not None:
            tool["error_count"] = tool.get("error_count", 0) + 1
            tool["last_error"] = datetime.now().isoformat()

    def _load_func_from_file(self, name: str, filepath: str) -> Optional[Callable]:
        """从 .py 文件加载函数"""
        try:
            # 方式1：importlib（需要文件在 sys.path 可解析的模块路径中）
            module_name = f"tools.tool_add.tool_direct.{name}"
            if module_name in sys.modules:
                module = importlib.import_module(module_name)
                importlib.reload(module)
            else:
                module = importlib.import_module(module_name)
            func = getattr(module, name, None)
            if func:
                return func
        except Exception:
            logger.debug("importlib 加载 %s 失败，回退 exec", name, exc_info=True)

        # 方式2：读文件内容 exec
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                code = f.read()
            local_ns: Dict[str, Any] = {}
            exec(code, {"__builtins__": __builtins__}, local_ns)
            for val in local_ns.values():
                if callable(val) and not isinstance(val, type):
                    return val
        except Exception:
            logger.debug("exec 加载 %s 也失败", name, exc_info=True)

        return None

    def _execute_create_tool(self, arguments: Dict) -> str:
        """执行 create_tool 元工具"""
        name = arguments.get("name", "").strip()
        description = arguments.get("description", "").strip()
        code = arguments.get("code", "").strip()

        if not name or not description or not code:
            return "错误：create_tool 需要 name, description, code 三个参数"

        return self.register_tool(name, description, code)

    # ==================== 删除工具 ====================

    def delete_tool(self, name: str) -> str:
        """从 MCP 池中删除工具"""
        if name in self._live_tools:
            del self._live_tools[name]
            self._live_functions.pop(name, None)
            return f"✅ 内存工具 '{name}' 已移除"

        tool = self.tools.get(name)
        if not tool:
            return f"错误：工具 '{name}' 不存在"

        # 删除 .py 文件
        code_file = tool.get("code_file", "")
        if code_file and os.path.exists(code_file):
            try:
                os.remove(code_file)
            except Exception as e:
                print(f"[MCP] 删除文件失败: {e}")

        # 也从 code_dir 尝试
        py_path = os.path.join(self.code_dir, f"{name}.py")
        if os.path.exists(py_path):
            try:
                os.remove(py_path)
            except Exception as e:
                print(f"[MCP] 删除文件失败: {e}")

        # 从内存移除
        del self.tools[name]
        self._func_cache.pop(name, None)

        # 保存
        self._save()

        return f"✅ 工具 '{name}' 已从 MCP 池中删除"

    # ==================== 更新工具 ====================

    def update_tool(
        self,
        name: str,
        code: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict] = None,
    ) -> str:
        """
        更新已有工具的代码、描述或参数。

        Args:
            name: 工具名称
            code: 新代码（None 则不更新）
            description: 新描述（None 则不更新）
            parameters: 新参数 schema（None 则不更新）

        Returns:
            成功/失败信息
        """
        with self._lock:
            tool = self.tools.get(name)
            if not tool:
                return f"错误：工具 '{name}' 不存在"

            if code is not None:
                code = self._clean_code(code)
                safety_issue = self._check_code_safety(code)
                if safety_issue:
                    return safety_issue

                try:
                    compile(code, f"<{name}>", "exec")
                except SyntaxError as e:
                    return f"错误：代码语法错误 —— {e}"

                # 写入 .py 文件
                code_file = tool.get("code_file", os.path.join(self.code_dir, f"{name}.py"))
                try:
                    with open(code_file, "w", encoding="utf-8") as f:
                        f.write(code)
                except Exception as e:
                    return f"错误：写入文件失败 —— {e}"

                tool["code_file"] = code_file
                self._func_cache.pop(name, None)  # 清除缓存，下次调用重新加载
                tool["updated_at"] = datetime.now().isoformat()

            if description is not None:
                tool["description"] = description

            if parameters is not None:
                tool["parameters"] = parameters

            self._save()
            updated = []
            if code is not None:
                updated.append("代码")
            if description is not None:
                updated.append("描述")
            if parameters is not None:
                updated.append("参数")
            return f"✅ 工具 '{name}' 已更新: {', '.join(updated)}"

    # ==================== 启用 / 禁用 ====================

    def enable_tool(self, name: str) -> str:
        """启用工具（允许 LLM 调用）"""
        tool = self.tools.get(name)
        if not tool:
            return f"错误：工具 '{name}' 不存在"
        tool["enabled"] = True
        self._save()
        return f"✅ 工具 '{name}' 已启用"

    def disable_tool(self, name: str) -> str:
        """禁用工具（对 LLM 隐藏但保留代码和元数据）"""
        tool = self.tools.get(name)
        if not tool:
            return f"错误：工具 '{name}' 不存在"
        tool["enabled"] = False
        self._func_cache.pop(name, None)
        self._save()
        return f"🚫 工具 '{name}' 已禁用（不会被 LLM 调用但保留在池中）"

    def toggle_tool(self, name: str) -> str:
        """切换工具启用/禁用状态"""
        tool = self.tools.get(name)
        if not tool:
            return f"错误：工具 '{name}' 不存在"
        if tool.get("enabled", True):
            return self.disable_tool(name)
        else:
            return self.enable_tool(name)

    # ==================== 工作空间安全 ====================

    def _resolve_path(self, path: str) -> str:
        """
        将用户/LLM 提供的路径解析到工作空间内。

        规则:
        - 相对路径 → 拼接在 workspace_dir 下
        - 绝对路径 → 必须在 workspace_dir 内，否则拒绝
        - 包含 .. 的路径 → 规范化后检查是否越界

        Returns:
            解析后的绝对路径

        Raises:
            PermissionError: 路径超出工作空间范围
        """
        # 规范化：去除 LLM 可能误加的 workspace/workspace_dir 前缀
        ws_name = os.path.basename(self.workspace_dir)
        normalized = os.path.normpath(path)
        # 去掉前导的 ./workspace/ 或 workspace/ （LLM 常误加）
        if normalized.startswith(f"./{ws_name}/") or normalized.startswith(f".\\{ws_name}\\"):
            normalized = normalized[len(f"./{ws_name}/"):]
        elif normalized.startswith(f"{ws_name}/") or normalized.startswith(f"{ws_name}\\"):
            normalized = normalized[len(f"{ws_name}/"):]

        if os.path.isabs(normalized):
            abs_path = normalized
        else:
            abs_path = os.path.normpath(os.path.join(self.workspace_dir, normalized))

        # 安全检查：必须在 workspace_dir 内
        workspace = os.path.normpath(self.workspace_dir)
        if not abs_path.startswith(workspace + os.sep) and abs_path != workspace:
            raise PermissionError(
                f"🚫 工作空间保护：禁止访问 '{path}'（解析为 '{abs_path}'）。"
                f"所有文件操作必须在 '{self.workspace_dir}' 内进行。"
            )

        # 自动创建父目录
        parent = os.path.dirname(abs_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        return abs_path

    # ==================== 安全限制 ====================

    def _load_limitations(self, filepath: str):
        """从 limitation.txt 加载禁止执行的命令和禁止读取的文件"""
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                section = None
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line == "[commands]":
                        section = "commands"
                    elif line == "[files]":
                        section = "files"
                    elif section == "commands":
                        self._forbidden_commands.append(line.lower())
                    elif section == "files":
                        self._forbidden_files.append(line.lower())
        except Exception:
            pass

    def _check_command(self, command: str) -> Optional[str]:
        """检查命令是否被禁止。返回 None 表示通过，否则返回禁止原因。"""
        cmd_lower = command.lower().strip()
        for forbidden in self._forbidden_commands:
            if self._wildcard_match(cmd_lower, forbidden):
                return f"🚫 安全限制：命令 '{command[:60]}' 匹配禁止规则 '{forbidden}'"
        return None

    def _check_file_read(self, path: str) -> Optional[str]:
        """检查文件是否被禁止读取。返回 None 表示通过，否则返回禁止原因。"""
        path_lower = os.path.basename(path).lower()
        for forbidden in self._forbidden_files:
            if self._wildcard_match(path_lower, forbidden):
                return f"🚫 安全限制：文件 '{path}' 匹配禁止规则 '{forbidden}'"
        return None

    @staticmethod
    def _wildcard_match(text: str, pattern: str) -> bool:
        """简单的通配符匹配（支持 *）"""
        if pattern == text:
            return True
        if "*" in pattern:
            import fnmatch
            return fnmatch.fnmatch(text, pattern)
        return pattern in text

    # ==================== 展示 ====================

    def summary(self) -> str:
        """返回工具池概览"""
        total = len(self.tools) + len(self._live_tools)
        lines = [
            f"MCP 工具池: {total} 个工具 "
            f"(持久化: {len(self.tools)}, 内存: {len(self._live_tools)})",
        ]
        for name, tool in sorted({**self.tools, **self._live_tools}.items()):
            tag = "💾" if name in self.tools else "🧠"
            if not tool.get("enabled", True):
                tag = "🚫"  # 已禁用
            desc = tool.get("description", "")[:60]
            usage = tool.get("usage_count", 0)
            age = self._tool_age_str(tool)
            extra = []
            if not tool.get("enabled", True):
                extra.append("禁用")
            if usage:
                extra.append(f"调用{usage}次")
            if age:
                extra.append(age)
            suffix = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"  {tag} {name} — {desc}{suffix}")
        return "\n".join(lines)

    def _tool_age_str(self, tool: Dict) -> str:
        """返回工具的时间描述"""
        created = tool.get("created_at", "")
        last_used = tool.get("last_used", "")
        parts = []
        if created:
            try:
                days = (datetime.now() - datetime.fromisoformat(created)).days
                parts.append(f"创建{days}天前")
            except (ValueError, TypeError):
                pass
        if last_used:
            try:
                days = (datetime.now() - datetime.fromisoformat(last_used)).days
                parts.append(f"上次使用{days}天前")
            except (ValueError, TypeError):
                pass
        return ", ".join(parts)

    # ==================== 时间戳与过期管理 ====================

    def get_stale_tools(
        self,
        days_unused: int = 7,
        min_usage: int = 0,
        protected: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        获取过期/不活跃的工具列表。

        Args:
            days_unused: 超过此天数未使用视为过期
            min_usage: 最少使用次数（低于此值且过期则标记）
            protected: 受保护的工具名列表（永远不会被标记）

        Returns:
            过期工具列表 [{name, description, created_at, last_used, usage_count, days_since_used}]
        """
        protected = protected or []
        now = datetime.now()
        stale = []

        for name, tool in self.tools.items():
            if name in protected:
                continue

            usage = tool.get("usage_count", 0)
            last_used_str = tool.get("last_used", tool.get("created_at", ""))

            if not last_used_str:
                # 从未使用过，用创建时间
                last_used_str = tool.get("created_at", "")
                if not last_used_str:
                    continue

            try:
                last_used = datetime.fromisoformat(last_used_str)
                days = (now - last_used).days

                if days >= days_unused and usage <= min_usage:
                    stale.append({
                        "name": name,
                        "description": tool.get("description", "")[:80],
                        "created_at": tool.get("created_at", ""),
                        "last_used": last_used_str,
                        "usage_count": usage,
                        "days_since_used": days,
                    })
            except (ValueError, TypeError):
                pass

        # 按最不活跃排序
        stale.sort(key=lambda t: (-t["days_since_used"], t["usage_count"]))
        return stale

    def cleanup_stale_tools(
        self,
        days_unused: int = 7,
        min_usage: int = 0,
        protected: Optional[List[str]] = None,
        dry_run: bool = True,
    ) -> str:
        """
        清理过期工具。

        Args:
            days_unused: 超过此天数未使用视为过期
            min_usage: 最少使用次数
            protected: 受保护的工具名列表
            dry_run: True=只列出不删除，False=执行删除

        Returns:
            清理结果描述
        """
        stale = self.get_stale_tools(days_unused, min_usage, protected)

        if not stale:
            return "✅ 没有过期工具需要清理"

        lines = [f"📋 {'[模拟] ' if dry_run else ''}发现 {len(stale)} 个过期工具:"]
        for t in stale:
            lines.append(
                f"  ⏰ {t['name']} — {t['days_since_used']}天未使用, "
                f"调用{t['usage_count']}次 — {t['description']}"
            )

        if not dry_run:
            deleted = 0
            for t in stale:
                result = self.delete_tool(t["name"])
                if "成功" in result or "已删除" in result or "已移除" in result:
                    deleted += 1
            lines.append(f"🗑️ 已删除 {deleted} 个过期工具")

        return "\n".join(lines)

    # ==================== LLM 冗余评估 ====================

    def build_evaluation_prompt(self) -> str:
        """构建发送给 LLM 的工具冗余评估 prompt"""
        if not self.tools:
            return "当前 MCP 池中没有持久化工具。"

        tool_descriptions = []
        for i, (name, tool) in enumerate(self.tools.items(), 1):
            usage = tool.get("usage_count", 0)
            created = tool.get("created_at", "")[:10]
            last = tool.get("last_used", "")[:10]
            tool_descriptions.append(
                f"{i}. **{name}**\n"
                f"   描述: {tool.get('description', '无')}\n"
                f"   调用次数: {usage} | 创建: {created} | 最后使用: {last}"
            )

        return f"""请评估以下 MCP 工具池中的工具是否存在功能重叠或冗余。

对于每组可能冗余的工具，说明：
1. 哪些工具功能重复
2. 建议保留哪个（优先保留使用频率高的、功能更通用的）
3. 合并建议（如果可以合并为一个更通用的工具）

如果所有工具都各司其职、没有明显冗余，请明确说"无冗余"。

当前工具列表：
{chr(10).join(tool_descriptions)}

请用 JSON 格式回复：
{{
    "has_redundancy": true/false,
    "analysis": "整体分析",
    "redundant_groups": [
        {{
            "tools": ["工具A", "工具B"],
            "reason": "冗余原因",
            "keep": "工具A",
            "action": "delete_both_and_merge" 或 "delete_one",
            "merged_name": "新工具名（如果是 merge）",
            "merged_description": "合并后的工具描述（如果是 merge）"
        }}
    ]
}}"""

    def evaluate_redundancy(self, client, model: str = "deepseek-chat") -> Dict:
        """
        调用 LLM 评估工具冗余。

        Args:
            client: OpenAI 客户端
            model: 使用的模型

        Returns:
            LLM 返回的评估结果字典
        """
        prompt = self.build_evaluation_prompt()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个代码审查专家，帮助评估 MCP 工具池中的工具是否有功能重叠或冗余。请严格按 JSON 格式回复。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            content = response.choices[0].message.content

            # 尝试提取 JSON
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                return json.loads(json_match.group(0))
            return {"has_redundancy": False, "analysis": content, "redundant_groups": []}
        except Exception as e:
            return {"has_redundancy": False, "analysis": f"评估失败: {e}", "redundant_groups": []}

    def cleanup_redundant_tools(
        self,
        client,
        model: str = "deepseek-chat",
        dry_run: bool = True,
    ) -> str:
        """
        评估并清理冗余工具。

        Args:
            client: OpenAI 客户端
            model: 使用的模型
            dry_run: True=只评估不删除，False=按 LLM 建议执行删除

        Returns:
            清理结果描述
        """
        result = self.evaluate_redundancy(client, model)
        lines = []

        if not result.get("has_redundancy"):
            lines.append("✅ LLM 评估: 当前工具池无明显冗余")
            lines.append(f"   分析: {result.get('analysis', '')[:200]}")
            return "\n".join(lines)

        groups = result.get("redundant_groups", [])
        lines.append(f"⚠️ LLM 发现 {len(groups)} 组冗余:")
        lines.append(f"   分析: {result.get('analysis', '')[:200]}")
        lines.append("")

        for g in groups:
            tools = g.get("tools", [])
            reason = g.get("reason", "")
            action = g.get("action", "")
            keep = g.get("keep", "")
            lines.append(f"  📦 冗余组: {', '.join(tools)}")
            lines.append(f"     原因: {reason}")
            lines.append(f"     保留: {keep}")

            if not dry_run:
                to_delete = [t for t in tools if t != keep]
                for name in to_delete:
                    if self.has_tool(name):
                        self.delete_tool(name)
                        lines.append(f"     🗑️ 已删除: {name}")

                if action == "delete_both_and_merge":
                    merged = g.get("merged_name", "")
                    merged_desc = g.get("merged_description", "")
                    if merged:
                        lines.append(f"     📝 建议合并为: {merged} — {merged_desc} (需手动创建)")

        if dry_run:
            lines.append(f"\n💡 以上为模拟评估，设置 dry_run=False 执行实际删除")

        return "\n".join(lines)

    # ==================== 综合维护 ====================

    def maintenance(
        self,
        client=None,
        model: str = "deepseek-chat",
        days_unused: int = 7,
        min_usage: int = 1,
        protected: Optional[List[str]] = None,
        dry_run: bool = True,
    ) -> str:
        """
        一次运行完整的工具池维护：时间过期清理 + LLM 冗余评估。

        Args:
            client: OpenAI 客户端（不提供则跳过 LLM 评估）
            model: LLM 模型名
            days_unused: 过期天数阈值
            min_usage: 最少使用次数阈值
            protected: 受保护工具名
            dry_run: True=只评估不删除

        Returns:
            维护报告
        """
        default_protected = [
            "read_file", "write_file", "create_file", "delete_file",
            "create_directory", "run_command", "delete_tool",
        ]
        protected = list(set((protected or []) + default_protected))

        report = []
        report.append("=" * 50)
        report.append(f"MCP 工具池维护报告 ({'模拟' if dry_run else '执行'})")
        report.append(f"工具总数: {len(self.tools)}")
        report.append("=" * 50)

        # 1. 过期清理
        report.append("\n📅 一、时间过期检查")
        report.append(self.cleanup_stale_tools(days_unused, min_usage, protected, dry_run))

        # 2. LLM 冗余评估
        if client:
            report.append("\n🤖 二、LLM 冗余评估")
            report.append(self.cleanup_redundant_tools(client, model, dry_run))
        else:
            report.append("\n🤖 二、LLM 冗余评估 (跳过: 未提供 client)")

        report.append("\n" + "=" * 50)
        return "\n".join(report)
