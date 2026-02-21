import os
import csv
import subprocess
import shutil
import json
from pathlib import Path


def read_file(path: str) -> str:
    """读取文件内容"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败：{e}"


def run_command(command: str) -> str:
    """执行shell命令（谨慎使用）"""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
        return output if output else "命令执行成功，无输出"
    except subprocess.TimeoutExpired:
        return "命令执行超时"
    except Exception as e:
        return f"执行命令出错：{e}"


def create_directory(path: str) -> str:
    """创建目录（如果不存在则创建，支持多级目录）"""
    try:
        os.makedirs(path, exist_ok=True)
        return f"目录创建成功：{path}"
    except Exception as e:
        return f"创建目录失败：{e}"


def create_file(path: str, content: str = "", overwrite: bool = False) -> str:
    """
    创建文件并写入初始内容
    :param path: 文件路径
    :param content: 初始内容，默认为空字符串
    :param overwrite: 是否覆盖已存在的文件，默认为False（不覆盖）
    :return: 操作结果字符串
    """
    try:
        # 如果文件已存在且不允许覆盖，返回错误
        if os.path.exists(path) and not overwrite:
            return f"错误：文件已存在且 overwrite=False，无法创建：{path}"
        # 确保父目录存在
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件创建成功：{path}"
    except Exception as e:
        return f"创建文件失败：{e}"


def write_file(path: str, content: str, mode: str = "w") -> str:
    """
    写入文件内容
    :param path: 文件路径
    :param content: 要写入的内容
    :param mode: 写入模式，'w'覆盖，'a'追加，默认为'w'
    :return: 操作结果字符串
    """
    try:
        if mode not in ("w", "a"):
            return f"错误：不支持的写入模式 '{mode}'，请使用 'w' 或 'a'"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        action = "覆盖写入" if mode == "w" else "追加写入"
        return f"文件{action}成功：{path}"
    except Exception as e:
        return f"写入文件失败：{e}"


def delete_file(path: str) -> str:
    """删除文件"""
    try:
        if not os.path.exists(path):
            return f"错误：文件不存在，无法删除：{path}"
        os.remove(path)
        return f"文件删除成功：{path}"
    except Exception as e:
        return f"删除文件失败：{e}"

def delete_tool(func_name: str) -> str:
    """
    删除指定名称的工具及其相关文件。
    参数：
        func_name: 要删除的工具函数名
    返回：
        包含操作结果的字符串
    """
    base_dir = './tools/tool_add/'
    abs_path = os.path.join(base_dir, 'tool_abs.txt')
    desc_path = os.path.join(base_dir, 'tool_description.txt')
    py_path = os.path.join(base_dir, 'tool_direct', f'{func_name}.py')
    
    messages = []

    # 1. 从 tool_abs.txt 中删除行
    try:
        if os.path.exists(abs_path):
            with open(abs_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # 过滤掉完全匹配的行（忽略前后空白）
            new_lines = [line for line in lines if line.strip() != func_name]
            if len(new_lines) == len(lines):
                messages.append(f"警告：在 tool_abs.txt 中未找到函数名 {func_name}")
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            messages.append(f"已从 tool_abs.txt 中删除 {func_name} 行")
        else:
            messages.append(f"错误：tool_abs.txt 不存在")
    except Exception as e:
        messages.append(f"删除 tool_abs.txt 中的行时出错：{e}")

    # 2. 从 tool_description.txt 中删除对应描述的 JSON 行
    try:
        if os.path.exists(desc_path):
            with open(desc_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            new_lines = []
            found = False
            for line in lines:
                stripped = line.strip()
                if not stripped:  # 保留空行
                    new_lines.append(line)
                    continue
                try:
                    tool_dict = json.loads(stripped)
                    print(tool_dict)
                    if tool_dict['function']['name'] == func_name:
                        found = True
                        continue  # 跳过这一行
                    else:
                        new_lines.append(line)
                except json.JSONDecodeError:
                    # 保留无法解析的行并给出警告
                    new_lines.append(line)
                    messages.append(f"警告：tool_description.txt 中存在无效 JSON 行，已保留：{stripped}")
            if not found:
                messages.append(f"警告：在 tool_description.txt 中未找到函数 {func_name} 的描述")
            with open(desc_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            messages.append(f"已从 tool_description.txt 中删除 {func_name} 的描述行")
        else:
            messages.append(f"错误：tool_description.txt 不存在")
    except Exception as e:
        messages.append(f"删除 tool_description.txt 中的描述时出错：{e}")

    # 3. 删除对应的 .py 文件
    try:
        if os.path.exists(py_path):
            os.remove(py_path)
            messages.append(f"已删除 Python 文件：{py_path}")
        else:
            messages.append(f"错误：Python 文件不存在：{py_path}")
    except Exception as e:
        messages.append(f"删除 Python 文件时出错：{e}")

    return "\n".join(messages)