import inspect
import json
import logging
from collections.abc import Coroutine
from enum import Enum
from typing import Any, Callable, Optional

from livekit.agents import FunctionTool, function_tool
from mcp.types import CallToolResult, Tool as MCPTool
from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


def mcp_to_function_tool(
    tool: MCPTool,
    call_tool: Callable[[str, dict[str, Any]], Coroutine[Any, Any, CallToolResult]],
) -> FunctionTool:
    """
    Converts an MCP tool to a FunctionTool using raw schema for OpenAI function calling.
    """
    name = tool.name
    description = tool.description

    # Use the tool's input schema directly as the raw schema for function_tool
    raw_schema = {
        "name": name,
        "description": description,
        "parameters": tool.inputSchema,
    }

    async def tool_impl(raw_arguments: dict[str, Any]) -> str:
        """
        Args:
            raw_arguments: The raw input arguments for the tool
        """
        try:
            # Convert any non-string values to strings as required by call_tool
            arguments = {k: str(v) for k, v in raw_arguments.items() if v is not None}
            result = await call_tool(name, arguments)

            logger.info(f"Called tool '{name}' with arguments: {arguments}")

            # only text content is supported for now
            text_contents = [
                content.text for content in result.content if content.type == "text"
            ]
            text = json.dumps(text_contents)

            if result.isError:
                raise ValueError("Tool call failed with content: " + text)

            return text

        except Exception as e:
            logger.error(f"Error calling tool '{name}': {e}")
            return f"Error calling tool '{name}': {e}"

    # Use the raw_schema parameter of function_tool
    return function_tool(tool_impl, raw_schema=raw_schema)


# modified from https://github.com/Finndersen/pydanticai_mcp_demo/blob/main/client/mcp_agent/util/schema_to_params.py
def create_pydantic_model_from_schema(
    schema: dict[str, Any], model_name: str
) -> type[BaseModel]:
    """
    Create a Pydantic model from a JSON schema.

    Args:
        schema: A JSON schema describing the model
        model_name: Name for the model

    Returns:
        A Pydantic model class
    """
    # Extract properties and required fields
    properties: dict[str, dict[str, Any]] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])

    # Create field definitions for Pydantic model
    fields = {}

    for field_name, field_info in properties.items():
        field_type = field_info.get("type", "string")
        description = field_info.get("description", "")

        # Handle different field types
        if field_type == "object":
            # Recursively create nested models for objects
            nested_model = create_pydantic_model_from_schema(
                field_info, model_name=f"{model_name}_{field_name.capitalize()}"
            )
            if field_name in required:
                fields[field_name] = (nested_model, Field(description=description))
            else:
                fields[field_name] = (
                    Optional[nested_model],
                    Field(default=None, description=description),
                )

        elif field_type == "array":
            # Handle arrays with proper item types
            items: dict[str, Any] = field_info.get("items", {})
            items_type = items.get("type", "string")

            if items_type == "object":
                # Array of objects - create a nested model for the items
                item_model = create_pydantic_model_from_schema(
                    items, model_name=f"{model_name}_{field_name.capitalize()}Item"
                )
                if field_name in required:
                    fields[field_name] = (
                        list[item_model],
                        Field(description=description),
                    )
                else:
                    fields[field_name] = (
                        Optional[list[item_model]],
                        Field(default=None, description=description),
                    )
            else:
                # Array of primitive types
                item_python_type = TYPE_MAP.get(items_type, Any)
                enum = items.get("enum", [])
                if enum:
                    item_python_type = Enum(
                        f"{model_name}_{field_name.capitalize()}",
                        [(v, v) for v in enum],
                    )

                if field_name in required:
                    fields[field_name] = (
                        list[item_python_type],
                        Field(description=description),
                    )
                else:
                    fields[field_name] = (
                        Optional[list[item_python_type]],
                        Field(default=None, description=description),
                    )
        else:
            # Handle primitive types
            python_type = TYPE_MAP.get(field_type, Any)
            enum = field_info.get("enum", [])
            if enum:
                python_type = Enum(
                    f"{model_name}_{field_name.capitalize()}", [(v, v) for v in enum]
                )

            if field_name in required:
                fields[field_name] = (python_type, Field(description=description))
            else:
                fields[field_name] = (
                    Optional[python_type],
                    Field(default=None, description=description),
                )

    # Create the Pydantic model dynamically
    return create_model(model_name, **fields)


# Map JSON schema types to Python types
TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}
