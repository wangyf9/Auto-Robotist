import json
from pathlib import Path


def dump_json(data, path: str | Path):
    path = Path(path)
    path.write_text(format_json(data) + "\n", encoding="utf-8")


def format_json(data, indent: int = 0, current_key: str | None = None) -> str:
    if isinstance(data, dict):
        if not data:
            return "{}"
        child_indent = indent + 2
        items = []
        for key, value in data.items():
            key_text = json.dumps(key, ensure_ascii=False)
            value_text = format_json(value, child_indent, current_key=key)
            items.append(f"{' ' * child_indent}{key_text}: {value_text}")
        return "{\n" + ",\n".join(items) + f"\n{' ' * indent}" + "}"

    if isinstance(data, list):
        if not data:
            return "[]"
        if current_key == "body" and _is_body_matrix(data):
            return _format_body_matrix(data, indent)
        child_indent = indent + 2
        items = [
            f"{' ' * child_indent}{format_json(item, child_indent)}"
            for item in data
        ]
        return "[\n" + ",\n".join(items) + f"\n{' ' * indent}" + "]"

    return json.dumps(data, ensure_ascii=False)


def _is_body_matrix(value) -> bool:
    return (
        isinstance(value, list)
        and value
        and all(isinstance(row, list) for row in value)
        and all(
            all(isinstance(cell, (int, float)) for cell in row)
            for row in value
        )
    )


def _format_body_matrix(matrix: list[list[int]], indent: int) -> str:
    row_indent = indent + 2
    rows = [f"{' ' * row_indent}{json.dumps(row)}" for row in matrix]
    return "[\n" + ",\n".join(rows) + f"\n{' ' * indent}" + "]"
