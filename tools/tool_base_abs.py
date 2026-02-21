import os
from typing import Any, Callable, Dict, List
from tools.tool_base_collections import *
TOOL_base_FUNCTIONS: Dict[str, Callable] = {
    "read_file": read_file,
    "run_command": run_command,
    "create_directory": create_directory,
    "create_file": create_file,
    "write_file": write_file,
    "delete_file": delete_file,
    "delete_tool":delete_tool,
}
TOOL_base_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取本地文件内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件路径"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "执行系统命令（谨慎使用）",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "创建目录，如果目录不存在则创建，支持多级目录",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要创建的目录路径"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "创建文件并写入初始内容，如果文件已存在且overwrite=False则不覆盖",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "文件初始内容，默认为空"},
                    "overwrite": {"type": "boolean", "description": "是否覆盖已存在的文件，默认为False"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件内容，支持覆盖写入(w)或追加写入(a)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                    "mode": {"type": "string", "enum": ["w", "a"], "description": "写入模式，'w'覆盖，'a'追加，默认为'w'"}
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "删除指定文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要删除的文件路径"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_tool",
            "description": "删除指定名称的工具及其相关文件",
            "parameters": {
            "type": "object",
            "properties": {
                "func_name": {
                "type": "string",
                "description": "要删除的工具函数名"
                }
            },
            "required": ["func_name"]
            }
        }
    }
]