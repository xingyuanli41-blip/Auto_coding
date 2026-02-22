# 智能对话记忆系统

本项目是一个具备长期记忆能力的对话智能体框架，基于 Python 和 Pydantic 实现。它包含短期工作记忆、长期持久化记忆、记忆溢出处理、AI 摘要总结、自动存档、重要性评估与删除等功能，可用于构建能够“记住”用户偏好和历史对话的聊天机器人、虚拟助手等应用。

## 文件结构说明

```
.
├── brain/                      # 核心智能体模块
│   ├── __init__.py
│   └── agent_call.py           # 模型调用、工具执行等主逻辑
├── tools/                       # 工具与记忆管理模块
│   ├── __init__.py
│   ├── memory_base.py           # Memory, LongTermMemory, Message 等基类
│   ├── memory_store.py          # 记忆存储相关函数 (handle_memory_overflow, 保存/加载)
│   ├── memory_retrieval.py      # 记忆检索与恢复 (load_memories_for_conversation)
│   ├── memory_delete.py         # 记忆删除 (delete_least_important_summaries)
│   ├── tool_base_abs.py         # 工具调用的抽象基类
│   ├── tool_base_collections.py # 工具集合管理
│   └── tool_add/                 # 自定义工具描述存放目录
├── memory_archives/              # 长期记忆溢出时的自动存档目录
├── main.py                       # 主对话循环入口
├── main_test.py                   # 测试脚本
├── pre_solve_english.txt         # 初始系统提示（英文）
├── pre_solve.txt                 # 初始系统提示（中文）
├── README.md                      # 本文档
└── __pycache__/                   # Python 编译缓存
```

## 功能模块介绍

### 1. 记忆模型 (`tools/memory_base.py`)
- `Message`: 对话消息，支持 role, content, tool_calls 等字段。
- `Memory`: 短期记忆容器，管理当前对话上下文，支持最大条数限制、截断、增删改查、持久化。
- `MemoryItem`: 长期记忆项，包含内容、时间戳、重要性、标签、元数据等。
- `LongTermMemory`: 长期记忆容器，存储 `MemoryItem` 列表，支持容量限制、检索、持久化。

### 2. 记忆存储与溢出处理 (`tools/memory_store.py`)
- `handle_memory_overflow()`: 当短期记忆超过 `max_messages` 时自动触发：
  - 将前 90% 的消息逐条存入长期记忆（标签 `raw_message`）。
  - 调用 AI 总结函数生成摘要，存入长期记忆（标签 `summary`）。
  - 截断短期记忆，只保留最后 10%。
  - 若长期记忆达到容量上限，自动保存到 `memory_archives/` 并清空。

### 3. 记忆恢复 (`tools/memory_retrieval.py`)
- `load_memories_for_conversation()`: 从文件中加载历史短期记忆和长期摘要，合并为一个新的 `Memory` 实例（摘要消息在前，短期记忆在后），并确保总条数不超过限制。

### 4. 记忆删除 (`tools/memory_delete.py`)
- `delete_least_important_summaries()`: 基于 AI 评分和时间衰减，删除长期记忆中最不重要的指定数量摘要及其对应的原始消息，以控制长期记忆规模。

### 6. 工具管理 (`tools/tool_base_*.py`, `tools/tool_add/`)
- 定义工具调用的基类、工具集合的加载方式。
- `tool_add/` 目录用于存放自定义工具描述的 JSON 文件（每行一个 JSON 对象）以及相应py文件。






欢迎使用本系统，如有问题请提交 Issue 或联系开发者。
