import os
import json

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
delete_tool('calculate_circle_area')