# AI 智能对话 Agent v2.3

具备 **三层记忆 + MCP 工具池 + 四状态机 + 安全沙箱** 的对话智能体框架。LLM 可自主创建工具，拥有记忆系统，所有写操作限制在安全工作空间内。

## 特性

| 能力 | 说明 |
|------|------|
| 🔧 **MCP 工具池** | LLM 自主创建工具、安全检查、工作空间保护 |
| 💾 **三层记忆** | 工作记忆 → 自动事实提取 → 长期记忆跨会话持久化 |
| 🔍 **TF-IDF 语义检索** | 从三层记忆中智能召回相关信息 |
| ⏳ **记忆遗忘** | importance × 时间衰减 → 自动清理 |
| 🔒 **安全沙箱** | 15条危险代码拦截 + 命令限制 + 敏感文件保护 + 工作空间锁定 |
| 🧠 **四状态机** | THINK → EXECUTE ⇄ REFLECT → END（brain.py） |
| 🎛️ **/ 命令系统** | 运行时切换模型、查看配置、调整参数 |
| 🌐 **多模型** | OpenAI / DeepSeek / Ollama 统一适配 |

## 项目结构

```
├── main.py                   # 交互入口 + / 命令系统
├── config.py                 # 配置加载器
├── config.example.json       # 配置模板（提交到 git）
├── config.json               # 实际配置（含密钥，gitignore）
├── system_prompt.txt         # 系统提示词（独立文件，方便编辑）
├── limitation.txt            # 安全限制（禁止命令/文件）
│
├── brain/                    # 智能体大脑
│   ├── adapters.py           #   多模型适配器
│   ├── state.py              #   四状态机定义
│   ├── brain.py              #   AgentBrain 核心（自主任务执行）
│   ├── call.py               #   API 调用封装（交互式对话用）
│   └── example.py            #   AgentBrain 使用示例
│
├── memory/                   # 三层记忆系统
│   ├── models.py             #   Message / Memory / MemoryItem
│   ├── manager.py            #   三层编排 + TF-IDF检索 + 自动事实提取
│   ├── embeddings.py         #   TF-IDF + OpenAI 向量引擎
│   ├── utils.py              #   旧 API 兼容
│   ├── cleanup.py            #   AI 评分记忆删除
│   └── archives/             #   记忆存档（运行时生成，gitignore）
│
├── tools/                    # MCP 工具池
│   ├── mcp_pool.py           #   核心（创建/执行/安全/冗余评估/工作空间）
│   ├── manage_tools.py       #   维护脚本
│   ├── mcp_tools.json        #   工具元数据
│   └── tool_add/tool_direct/ #   7 个核心工具 .py
│
├── workspace/                # 工作空间（LLM 写操作限制在此）
└── test/                     # 测试 93/93 通过
    ├── test_mcp_tools.py     #   工具池 34/34
    ├── test_memory.py        #   记忆系统 59/59
    ├── test_brain.py         #   大脑基础
    └── test_llm_create_tool.py  # LLM 集成（需 API）
```

## 快速开始

### 1. 配置

```bash
cp config.example.json config.json
# 编辑 config.json，填入 API Key
# 可选：编辑 system_prompt.txt 自定义系统提示词
```

```json
{
  "llm": {
    "api_key": "sk-your-key-here",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat"
  }
}
```

### 2. 运行

```bash
python main.py
```

### 3. 对话

```
User：/help               # 查看命令
User：帮我查豆瓣电影Top10并保存  # 自动创建工具→获取数据→保存到workspace/
User：/memory             # 查看记忆
User：/exit               # 退出（自动保存长期记忆）
```

## / 命令系统

| 命令 | 功能 |
|------|------|
| `/help` | 显示帮助 |
| `/model [name]` | 查看/切换模型 |
| `/config` | 查看配置（密钥隐藏） |
| `/tools` | 查看工具池状态 |
| `/memory` | 查看三层记忆统计 |
| `/system [text]` | 查看/修改系统提示词 |
| `/temp [value]` | 查看/调整 temperature |
| `/clear` | 清空对话记忆 |
| `/save` | 手动保存记忆 |
| `/exit` | 退出 |

## 安全防护

| 层 | 机制 | 配置 |
|----|------|------|
| 代码注入 | 15 条危险模式（os/subprocess/eval/exec/shutil） | `mcp_pool.py` |
| 命令限制 | 6 类危险命令（rm -rf/shutdown/format/fork bomb） | `limitation.txt [commands]` |
| 文件保护 | 敏感文件不可读（config.json/.env/*.key/id_rsa） | `limitation.txt [files]` |
| 工作空间 | 写/删/创建锁在 `./workspace/` | `config.json workspace_dir` |
| 密钥 | .gitignore + config.example.json 模板 | — |

## MCP 工具池

LLM 检查池中工具 → 有则调用 → 没有则 `create_tool` 创建

```
用户 → LLM 发现缺工具 → create_tool(代码+描述)
  → 安全检查 → 语法验证 → 写入.py → 注册JSON
  → 立即调用 → 返回结果
```

### 7 个核心工具

| 工具 | 功能 |
|------|------|
| `read_file` | 读取文件（任意路径） |
| `write_file` | 写入文件（仅 workspace/） |
| `create_file` | 创建文件（仅 workspace/） |
| `delete_file` | 删除文件（仅 workspace/） |
| `create_directory` | 创建目录（仅 workspace/） |
| `run_command` | 执行命令（受 limitation.txt 限制） |
| `delete_tool` | 删除工具 |

> 计算、爬虫、分析等由 LLM 按需 `create_tool` 创建，自动持久化。

## 三层记忆

```
L1 工作记忆 — 当前聊天记录（不保存）
  │ 每轮 remember_turn() 正则提取关键信息
  ▼
L2 短期记忆 — 自动事实提取（用户名/偏好/生日/地点/职业）
  │ save() 时自动 consolidate
  ▼
L3 长期记忆 — 持久化到 JSON，重启恢复，时间衰减遗忘
```

### MemoryManager 使用

```python
from memory import MemoryManager

mgr = MemoryManager(client=llm_client, model="deepseek-chat")
mgr.add_system_message("你是助手")
mgr.add_user_message("我叫小明，喜欢Python")
mgr.add_assistant_message("记住了！")
mgr.remember_turn("我叫小明，喜欢Python", "记住了！")

# 语义检索
results = mgr.recall("小明喜欢什么编程语言")

# 持久化
mgr.save()  # 自动巩固短期→长期 + 写入文件
```

## AgentBrain API（自主任务执行）

```python
from brain import create_agent, ModelConfig, ModelProvider

configs = [ModelConfig(provider=ModelProvider.OPENAI,
    model_name="deepseek-chat", api_key="sk-xxx",
    base_url="https://api.deepseek.com")]
brain = create_agent(model_configs=configs, max_iterations=10)

result = brain.run("计算 1+2+...+100")
# 四状态机自动流转: THINK→EXECUTE→REFLECT→END
```

## 配置项

| 路径 | 默认值 | 说明 |
|------|--------|------|
| `llm.api_key` | — | API 密钥 |
| `llm.base_url` | `https://api.deepseek.com` | API 地址 |
| `llm.model` | `deepseek-chat` | 模型名 |
| `llm.temperature` | `0.7` | 创意度 |
| `brain.max_iterations` | `100` | 最大对话轮次 |
| `brain.system_prompt_file` | `./system_prompt.txt` | 系统提示词文件 |
| `memory.working_token_budget` | `4000` | 工作记忆 token 预算 |
| `memory.long_term_max_items` | `1000` | 长期记忆上限 |
| `tools.workspace_dir` | `./workspace` | 工作空间路径 |
| `tools.maintenance_days_unused` | `7` | 过期清理天数 |

## 运行测试

```bash
python test/test_mcp_tools.py     # 工具池 34/34
python test/test_memory.py        # 记忆系统 59/59
python test/test_brain.py         # 大脑基础
python test/test_llm_create_tool.py  # LLM 集成（需 API）
```

## 维护

```bash
# 查看工具池状态
python tools/manage_tools.py

# 清理过期工具
python tools/manage_tools.py --execute
```

## 依赖

```bash
pip install openai pydantic
```
