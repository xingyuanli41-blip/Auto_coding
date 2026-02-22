from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib

ROLE_TYPE = Literal["user", "assistant", "system", "tool"]

class Function(BaseModel):
    """表示一个待调用的函数"""
    name: str
    arguments: str 

class ToolCall(BaseModel):
    """表示消息中的工具/函数调用"""
    id: str
    type: str = "function"
    function: Function

class Message(BaseModel):
    """
    表示对话中的一条消息，能够从大模型 API 响应中提取信息。
    支持文本、工具调用、图像等多种内容。
    """
    role: ROLE_TYPE = Field(..., description="消息发送者角色")
    content: Optional[str] = Field(default=None, description="文本内容")
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="助手请求的工具调用列表")
    name: Optional[str] = Field(default=None, description="工具名称（用于工具响应）")
    tool_call_id: Optional[str] = Field(default=None, description="被响应的工具调用 ID")
    base64_image: Optional[str] = Field(default=None, description="Base64 编码的图像数据")

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """将消息转换为字典格式（适用于 API 请求）"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    @classmethod
    def from_llm_response(cls, response: Dict[str, Any]) -> "Message":
        """
        从大模型 API 返回的响应字典中提取消息内容。
        假设响应结构类似于 OpenAI 的 choice 中的 message 字段。
        """
        role = response.get("role", "assistant")
        content = response.get("content")
        tool_calls_data = response.get("tool_calls")

        # 解析 tool_calls
        tool_calls = None
        if tool_calls_data:
            tool_calls = []
            for tc in tool_calls_data:
                func_data = tc.get("function", {})
                function = Function(
                    name=func_data.get("name", ""),
                    arguments=func_data.get("arguments", "{}")
                )
                tool_calls.append(ToolCall(
                    id=tc.get("id"),
                    type=tc.get("type", "function"),
                    function=function
                ))

        # 工具响应专用字段
        name = response.get("name")
        tool_call_id = response.get("tool_call_id")

        # 图像数据（某些 API 可能直接返回 base64_image 字段）
        base64_image = response.get("base64_image")

        return cls(
            role=role,
            content=content,
            tool_calls=tool_calls,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image
        )

    @classmethod
    def user_message(cls, content: str, base64_image: Optional[str] = None) -> "Message":
        return cls(role="user", content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def assistant_message(cls, content: Optional[str] = None, base64_image: Optional[str] = None) -> "Message":
        return cls(role="assistant", content=content, base64_image=base64_image)

    @classmethod
    def tool_message(cls, content: str, name: str, tool_call_id: str, base64_image: Optional[str] = None) -> "Message":
        return cls(
            role="tool",
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image
        )


from typing import List, Optional, Callable, Any
from pydantic import BaseModel, Field

class Memory(BaseModel):
    """管理对话历史消息的内存类，支持最大消息数限制和常用操作"""

    messages: List[Message] = Field(default_factory=list, description="存储的消息列表")
    max_messages: int = Field(default=100, ge=1, description="最大保留消息数，超过时自动截断")

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def add_messages(self, messages: List[Message]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        if n <= 0:
            return []
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        return [msg.to_dict() for msg in self.messages]

    def _truncate(self) -> None:
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def set_max_messages(self, new_max: int) -> None:
        if new_max <= 0:
            raise ValueError("max_messages must be positive")
        self.max_messages = new_max
        self._truncate()

    def trim(self) -> None:
        self._truncate()

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, index: int) -> Message:
        return self.messages[index]

    def __setitem__(self, index: int, value: Message) -> None:
        self.messages[index] = value

    def __delitem__(self, index: int) -> None:
        del self.messages[index]

    def __iter__(self):
        return iter(self.messages)

    def pop(self, index: int = -1) -> Message:
        return self.messages.pop(index)

    def remove(self, message: Message) -> None:
        self.messages.remove(message)

    def count_by_role(self, role: str) -> int:
        return sum(1 for msg in self.messages if msg.role == role)

    def find_first(self, condition: Callable[[Message], bool]) -> Optional[Message]:
        for msg in self.messages:
            if condition(msg):
                return msg
        return None

    def find_all(self, condition: Callable[[Message], bool]) -> List[Message]:
        return [msg for msg in self.messages if condition(msg)]

    def save_to_file(self, filepath: str) -> None:
        import json
        data = self.to_dict_list()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str, max_messages: int = 100) -> "Memory":
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        messages = [Message(**item) for item in data]
        return cls(messages=messages, max_messages=max_messages)

    def copy(self) -> "Memory":
        return self.model_copy(deep=True)

    def extend(self, other: "Memory") -> None:
        self.messages.extend(other.messages)
        self._truncate()

class MemoryItem(BaseModel):
    """长期记忆项，包含内容、时间戳、重要性等元数据"""
    id: str = Field(default_factory=lambda: hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8])
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性评分")
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def accessed(self):
        """更新访问时间和计数"""
        self.last_accessed = datetime.now()
        self.access_count += 1

class LongTermMemory(BaseModel):
    """长期记忆类，存储重要的、持久化的记忆项，支持检索和管理"""
    
    items: List[MemoryItem] = Field(default_factory=list, description="存储的长期记忆项列表")
    max_items: Optional[int] = Field(default=None, description="最大项数限制，None表示无限制")
    
    def add_item(self, item: MemoryItem) -> None:
        """添加一个记忆项"""
        self.items.append(item)
    
    def add_items(self, items: List[MemoryItem]) -> None:
        """批量添加记忆项"""
        self.items.extend(items)
        self._truncate()
    
    def clear(self) -> None:
        """清空所有记忆"""
        self.items.clear()
    
    def _truncate(self) -> None:
        """如果设置了max_items且超过，则移除最旧的项（基于时间戳）"""
        if self.max_items is not None and len(self.items) > self.max_items:
            # 按时间戳排序，保留最新的max_items个
            self.items.sort(key=lambda x: x.timestamp, reverse=True)
            self.items = self.items[:self.max_items]
    
    def get_recent_items(self, n: int) -> List[MemoryItem]:
        """获取最近的n个记忆项（按时间戳降序）"""
        if n <= 0:
            return []
        sorted_items = sorted(self.items, key=lambda x: x.timestamp, reverse=True)
        return sorted_items[:n]
    
    def search_by_keyword(self, keyword: str, case_sensitive: bool = False) -> List[MemoryItem]:
        """关键词搜索"""
        if not case_sensitive:
            keyword = keyword.lower()
            return [item for item in self.items if keyword in item.content.lower()]
        else:
            return [item for item in self.items if keyword in item.content]
    
    def search_by_tags(self, tags: List[str]) -> List[MemoryItem]:
        """按标签搜索（需要全部匹配）"""
        return [item for item in self.items if all(tag in item.tags for tag in tags)]
    
    def find_first(self, condition: Callable[[MemoryItem], bool]) -> Optional[MemoryItem]:
        """查找第一个满足条件的项"""
        for item in self.items:
            if condition(item):
                return item
        return None
    
    def find_all(self, condition: Callable[[MemoryItem], bool]) -> List[MemoryItem]:
        """查找所有满足条件的项"""
        return [item for item in self.items if condition(item)]
    
    def remove_item(self, item_id: str) -> bool:
        """根据ID移除项"""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                del self.items[i]
                return True
        return False
    
    def update_item(self, item_id: str, **kwargs) -> bool:
        """更新项的内容（除id外）"""
        for item in self.items:
            if item.id == item_id:
                for key, value in kwargs.items():
                    if hasattr(item, key) and key != 'id':
                        setattr(item, key, value)
                return True
        return False
    
    def get_item(self, item_id: str) -> Optional[MemoryItem]:
        """根据ID获取项"""
        for item in self.items:
            if item.id == item_id:
                return item
        return None
    
    def to_dict_list(self) -> List[dict]:
        """转换为字典列表，便于序列化"""
        result = []
        for item in self.items:
            d = item.dict()
            # 处理datetime对象
            d['timestamp'] = d['timestamp'].isoformat()
            d['last_accessed'] = d['last_accessed'].isoformat()
            result.append(d)
        return result
    
    def save_to_file(self, filepath: str) -> None:
        """
        将记忆项列表保存到 JSON 文件。
        """
        import json
        import os
        new_data = self.to_dict_list()
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if isinstance(existing_data, list):
                    existing_data.extend(new_data)
                else:
                    existing_data = new_data
            except (json.JSONDecodeError, Exception):
                existing_data = new_data
        else:
            existing_data = new_data
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    def save_to_file_overwrite(self, filepath: str) -> None:
        import json
        import os
        new_data = self.to_dict_list()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        
    @classmethod
    def load_from_file(cls, filepath: str, max_items: Optional[int] = None) -> "LongTermMemory":
        """从JSON文件加载"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        for item_data in data:
            # 转换回datetime
            item_data['timestamp'] = datetime.fromisoformat(item_data['timestamp'])
            item_data['last_accessed'] = datetime.fromisoformat(item_data['last_accessed'])
            items.append(MemoryItem(**item_data))
        
        return cls(items=items, max_items=max_items)
    
    def copy(self) -> "LongTermMemory":
        """深拷贝"""
        return self.model_copy(deep=True)
    
    def extend(self, other: "LongTermMemory") -> None:
        """合并另一个长期记忆"""
        self.items.extend(other.items)
        self._truncate()
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self):
        return iter(self.items)
    
    def __getitem__(self, index: int) -> MemoryItem:
        return self.items[index]
    
    def __setitem__(self, index: int, value: MemoryItem) -> None:
        self.items[index] = value
    
    def __delitem__(self, index: int) -> None:
        del self.items[index]
    
    def pop(self, index: int = -1) -> MemoryItem:
        return self.items.pop(index)
    
    def remove(self, item: MemoryItem) -> None:
        self.items.remove(item)