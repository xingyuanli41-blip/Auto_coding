import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
MCP 工具池功能测试 —— 验证自主创建、执行、删除工具的全流程

运行方式:
    cd c:/Users/qq215/Desktop/auto_coding
    python test_mcp_tools.py
"""

import os
import sys

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mcp_pool import MCPToolPool


# ============================================================
# 辅助函数
# ============================================================
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
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def cleanup_tool(pool, name):
    """清理测试残留工具"""
    if pool.has_tool(name):
        pool.delete_tool(name)


# ============================================================
# 准备：清理上次测试残留
# ============================================================
section("准备：初始化 MCP 池并清理残留")

pool = MCPToolPool(
    pool_file="./tools/mcp_tools.json",
    code_dir="./tools/tool_add/tool_direct",
)
print(f"  当前池状态: {pool.get_tool_count()}")

# 清理可能残留的测试工具
for name in ["hello_world", "add_numbers", "get_random_number", "count_chars"]:
    cleanup_tool(pool, name)
print(f"  清理后: {pool.get_tool_count()}")


# ============================================================
# 测试 1: 自主创建工具
# ============================================================
section("测试 1: 自主创建工具 (register_tool)")

# 1a: 创建一个简单的工具
code_hello = '''
def hello_world(name: str) -> str:
    """向指定的人打招呼"""
    return f"Hello, {name}! Welcome to MCP Tool Pool."
'''
result = pool.register_tool(
    name="hello_world",
    description="向指定的人打招呼，传入 name 参数返回英文问候语",
    code=code_hello,
)
check("成功" in result, f"创建 hello_world 工具")
check(pool.has_tool("hello_world"), "hello_world 在池中")

# 1b: 确保 .py 文件已写入
check(
    os.path.exists("./tools/tool_add/tool_direct/hello_world.py"),
    "hello_world.py 文件已生成"
)

# 1c: 创建包含多参数和 import 的工具
code_add = '''
def add_numbers(a: int, b: int) -> str:
    """计算两个整数的和"""
    result = a + b
    return f"{a} + {b} = {result}"
'''
result = pool.register_tool(
    name="add_numbers",
    description="计算两个整数的和，传入 a 和 b 两个整数参数",
    code=code_add,
)
check("成功" in result, "创建 add_numbers 工具")
check(pool.has_tool("add_numbers"), "add_numbers 在池中")


# ============================================================
# 测试 2: 执行创建的工具
# ============================================================
section("测试 2: 执行已创建的工具")

# 2a: 执行 hello_world
result = pool.execute("hello_world", {"name": "小明"})
check("小明" in result, f"hello_world('小明') → {result}")

result = pool.execute("hello_world", {"name": "Alice"})
check("Alice" in result, f"hello_world('Alice') → {result}")

# 2b: 执行 add_numbers
result = pool.execute("add_numbers", {"a": 100, "b": 200})
check("300" in result, f"add_numbers(100, 200) → {result}")

# 2c: 执行原有持久化工具（确保其他工具不受影响）
result = pool.execute("read_file", {"path": "./README.md"})
check("智能对话记忆系统" in result or "auto_coding" in result or len(result) > 5,
      f"read_file 仍可正常执行")

# 2d: 参数不足时报错（不是崩溃）
result = pool.execute("add_numbers", {"a": 5})
check("失败" in result or "error" in result.lower() or "Error" in result,
      f"参数不足时报错（非崩溃）: {result[:60]}")


# ============================================================
# 测试 3: 持久化验证（模拟重启）
# ============================================================
section("测试 3: 持久化验证（模拟重启）")

# 创建新池实例，验证新工具仍在
pool2 = MCPToolPool(
    pool_file="./tools/mcp_tools.json",
    code_dir="./tools/tool_add/tool_direct",
)
check(pool2.has_tool("hello_world"), "重启后 hello_world 仍在池中")
check(pool2.has_tool("add_numbers"), "重启后 add_numbers 仍在池中")

# 执行也正常
result = pool2.execute("hello_world", {"name": "重启验证"})
check("重启验证" in result, f"重启后仍可执行: {result}")


# ============================================================
# 测试 4: 删除工具
# ============================================================
section("测试 4: 删除工具 (delete_tool)")

# 4a: 删除 hello_world
result = pool.delete_tool("hello_world")
check("成功" in result or "删除" in result or "已移除" in result,
      f"删除 hello_world: {result[:60]}")
check(not pool.has_tool("hello_world"), "hello_world 已不在池中")
check(
    not os.path.exists("./tools/tool_add/tool_direct/hello_world.py"),
    "hello_world.py 文件已删除"
)

# 4b: 删除 add_numbers
result = pool.delete_tool("add_numbers")
check(not pool.has_tool("add_numbers"), "add_numbers 已删除")

# 4c: 删除已不存在的工具
result = pool.delete_tool("hello_world")
check("不存在" in result or "错误" in result,
      f"删除不存在的工具时报错: {result[:60]}")

# 4d: 删除后重启验证
pool3 = MCPToolPool(
    pool_file="./tools/mcp_tools.json",
    code_dir="./tools/tool_add/tool_direct",
)
check(not pool3.has_tool("hello_world"), "重启后确认 hello_world 已彻底删除")
check(not pool3.has_tool("add_numbers"), "重启后确认 add_numbers 已彻底删除")


# ============================================================
# 测试 5: 使用 LLM 风格的 create_tool 流程（模拟）
# ============================================================
section("测试 5: create_tool 元工具流程（模拟 LLM 行为）")

# 模拟 LLM 调用 create_tool
create_args = {
    "name": "count_chars",
    "description": "统计给定文本的字符数量",
    "code": '''
def count_chars(text: str) -> str:
    """统计文本的字符数（含空格，不含换行）"""
    count = len(text.replace("\\n", ""))
    return f"文本共 {count} 个字符"
'''
}
result = pool.execute("create_tool", create_args)
check("成功" in result, f"create_tool 创建 count_chars: {result[:80]}")
check(pool.has_tool("count_chars"), "count_chars 在池中")

# 立即调用新工具
result = pool.execute("count_chars", {"text": "Hello World"})
check("11" in result, f"count_chars('Hello World') → {result}")

# 清理
pool.delete_tool("count_chars")


# ============================================================
# 测试 6: 边界情况
# ============================================================
section("测试 6: 边界情况")

# 6a: 不合法的工具名
result = pool.register_tool("123invalid", "测试", "def test(): return 'x'")
check("错误" in result, f"拒绝非法名称: {result[:60]}")

# 6b: 重复创建
code_test = "def dup_test(x: int) -> str:\n    return str(x * 2)"
pool.register_tool("dup_test", "test", code_test)
result = pool.register_tool("dup_test", "test", code_test)
check("已存在" in result, f"拒绝重复创建: {result[:60]}")
pool.delete_tool("dup_test")

# 6c: 语法错误的代码
result = pool.register_tool(
    "bad_tool",
    "test",
    "def bad_tool(x::\n    return"
)
check("语法错误" in result, f"拒绝语法错误代码: {result[:60]}")

# 6d: 搜索功能
results = pool.search("文件")
check(len(results) >= 3, f"搜索'文件'找到 {len(results)} 个工具: {[r['name'] for r in results]}")

results = pool.search("nonexistent_xyz")
check(len(results) == 0, f"搜索不存在的关键词返回空列表")


# ============================================================
# 测试 7: list_tools OpenAPI 格式
# ============================================================
section("测试 7: list_tools 返回 OpenAI 格式")

tools = pool.list_tools()
check(isinstance(tools, list) and len(tools) >= 8,
      f"list_tools 返回 {len(tools)} 个工具")

for t in tools:
    assert t["type"] == "function", f"每个工具 type 应为 function，实际: {t['type']}"
    assert "name" in t["function"], f"每个工具应有 name"
    assert "description" in t["function"], f"每个工具应有 description"
    assert "parameters" in t["function"], f"每个工具应有 parameters"
check(True, "所有工具符合 OpenAI function 格式")

# 确认 create_tool 在列表中
tool_names = [t["function"]["name"] for t in tools]
check("create_tool" in tool_names, "create_tool 元工具在列表中")


# ============================================================
# 测试 8: Live Tool（内存工具，不持久化）
# ============================================================
section("测试 8: Live Tool (add_live_tool)")

def my_secret_tool(message: str) -> str:
    """一个仅在当前会话存在的秘密工具"""
    return f"Secret: {message[::-1]}"

pool.add_live_tool("secret", "一个秘密工具（不持久化）", my_secret_tool)
check(pool.has_tool("secret"), "live tool 注册成功")

result = pool.execute("secret", {"message": "hello"})
check("olleh" in result, f"live tool 执行: {result}")

# 重启验证：live tool 不应该在
pool4 = MCPToolPool(
    pool_file="./tools/mcp_tools.json",
    code_dir="./tools/tool_add/tool_direct",
)
check(not pool4.has_tool("secret"), "重启后 live tool 消失 ✓")


# ============================================================
# 结果
# ============================================================
section(f"测试结果: {passed}/{passed + failed} 通过")

if failed == 0:
    print("\n  🎉 全部测试通过！MCP 工具池的创建、执行、删除功能正常工作。\n")
else:
    print(f"\n  ⚠️ {failed} 个测试失败，请检查。\n")
    sys.exit(1)
