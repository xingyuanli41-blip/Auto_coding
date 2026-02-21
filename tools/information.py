from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field

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
    max_messages: int = Field(default=1000, ge=1, description="最大保留消息数，超过时自动截断")

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self._truncate()

    def add_messages(self, messages: List[Message]) -> None:
        self.messages.extend(messages)
        self._truncate()

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

    # ---------- 查询与统计 ----------
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

