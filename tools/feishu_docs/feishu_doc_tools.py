from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

FEISHU_BASE_URL = "https://open.feishu.cn/open-apis"
BLOCK_TYPES = {
    "text": 2,
    "heading1": 3,
    "heading2": 4,
    "heading3": 5,
    "bullet": 12,
    "ordered": 13,
    "code": 14,
    "quote": 15,
}
IMAGE_PATTERN = re.compile(r"^!\[(.*?)\]\((.*?)\)$")


class FeishuApiError(RuntimeError):
    """Raised when the Feishu API returns a non-success response."""


@dataclass
class DocNode:
    kind: str
    text: str


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_env_values(env_path: str | Path) -> dict[str, str]:
    path = Path(env_path)
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^\s*([^:=#]+?)\s*([:=])\s*(.*?)\s*$", raw_line)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(3).strip().strip('"').strip("'")
        values[key] = value
    return values


def get_required_credential(values: dict[str, str], *keys: str) -> str:
    for key in keys:
        if values.get(key):
            return values[key]
    raise RuntimeError(f"Missing credential in .env. Tried keys: {', '.join(keys)}")


def parse_markdown(markdown: str) -> tuple[str | None, list[DocNode]]:
    title: str | None = None
    nodes: list[DocNode] = []
    in_code = False
    code_lines: list[str] = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                nodes.append(DocNode("code", "\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            continue

        image_match = IMAGE_PATTERN.match(stripped)
        if image_match:
            alt, src = image_match.groups()
            caption = alt.strip() or Path(src.strip()).name
            nodes.append(DocNode("quote", f"图片：{caption}（本地路径：{src.strip()}）"))
            continue

        if stripped.startswith("# "):
            title = stripped[2:].strip()
            continue
        if stripped.startswith("## "):
            nodes.append(DocNode("heading1", stripped[3:].strip()))
            continue
        if stripped.startswith("### "):
            nodes.append(DocNode("heading2", stripped[4:].strip()))
            continue
        if stripped.startswith("#### "):
            nodes.append(DocNode("heading3", stripped[5:].strip()))
            continue
        if stripped.startswith("> "):
            nodes.append(DocNode("quote", stripped[2:].strip()))
            continue
        if stripped.startswith("- "):
            nodes.append(DocNode("bullet", stripped[2:].strip()))
            continue

        ordered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ordered_match:
            nodes.append(DocNode("ordered", ordered_match.group(1).strip()))
            continue

        nodes.append(DocNode("text", stripped))

    if in_code and code_lines:
        nodes.append(DocNode("code", "\n".join(code_lines)))

    return title, nodes


def make_text_block(kind: str, text: str) -> dict[str, Any]:
    block = {
        "block_type": BLOCK_TYPES[kind],
        kind: {
            "elements": [{"text_run": {"content": text}}],
            "style": {},
        },
    }
    if kind == "code":
        block[kind]["style"] = {"wrap": True}
    return block


def node_to_block(node: DocNode) -> dict[str, Any]:
    if node.kind in {"text", "heading1", "heading2", "heading3", "bullet", "ordered", "code", "quote"}:
        return make_text_block(node.kind, node.text)
    raise ValueError(f"Unsupported doc node kind: {node.kind}")


class FeishuDocClient:
    def __init__(self, app_id: str | None = None, app_secret: str | None = None, bearer_token: str | None = None) -> None:
        self.app_id = app_id
        self.app_secret = app_secret
        self.bearer_token = bearer_token
        self.session = requests.Session()
        self._tenant_access_token: str | None = None

    def _request(self, method: str, path: str, *, auth: bool = True, **kwargs: Any) -> dict[str, Any]:
        headers = dict(kwargs.pop("headers", {}))
        if auth:
            headers["Authorization"] = f"Bearer {self.get_bearer_token()}"
        response = self.session.request(
            method,
            f"{FEISHU_BASE_URL}{path}",
            headers=headers,
            timeout=30,
            **kwargs,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {"raw_text": response.text}
        if response.status_code >= 400:
            raise FeishuApiError(
                f"Feishu API HTTP {response.status_code} on {method} {path}: {json.dumps(payload, ensure_ascii=False)}"
            )
        if payload.get("code", 0) != 0:
            raise FeishuApiError(
                f"Feishu API failed: {method} {path} code={payload.get('code')} msg={payload.get('msg')}"
            )
        return payload

    def get_bearer_token(self) -> str:
        if self.bearer_token is not None:
            return self.bearer_token
        return self.get_tenant_access_token()

    def get_tenant_access_token(self) -> str:
        if self._tenant_access_token is not None:
            return self._tenant_access_token
        if not self.app_id or not self.app_secret:
            raise FeishuApiError("app_id/app_secret are required when bearer_token is not provided")
        payload = self._request(
            "POST",
            "/auth/v3/tenant_access_token/internal",
            auth=False,
            json={"app_id": self.app_id, "app_secret": self.app_secret},
        )
        token = payload.get("tenant_access_token")
        if not token:
            raise FeishuApiError("tenant_access_token not found in Feishu auth response")
        self._tenant_access_token = token
        return token

    def create_document(self, title: str, folder_token: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"title": title}
        if folder_token:
            body["folder_token"] = folder_token
        payload = self._request("POST", "/docx/v1/documents", json=body)
        return payload["data"]["document"]

    def append_children(
        self,
        document_id: str,
        parent_block_id: str,
        blocks: list[dict[str, Any]],
        *,
        index: int,
        document_revision_id: int | None = None,
    ) -> dict[str, Any]:
        params = {"client_token": str(uuid.uuid4())}
        if document_revision_id is not None:
            params["document_revision_id"] = document_revision_id
        payload = self._request(
            "POST",
            f"/docx/v1/documents/{document_id}/blocks/{parent_block_id}/children",
            params=params,
            json={"index": index, "children": blocks},
        )
        return payload["data"]


def publish_markdown_file(
    *,
    source_file: str | Path,
    root: str | Path = ".",
    title: str | None = None,
    folder_token: str | None = None,
    env_path: str | Path = ".env",
    access_token: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    source_path = Path(source_file)
    if not source_path.is_absolute():
        source_path = (root_path / source_path).resolve()
    markdown = source_path.read_text(encoding="utf-8")
    parsed_title, nodes = parse_markdown(markdown)
    resolved_title = title or parsed_title or f"{source_path.stem} Upload"

    result: dict[str, Any] = {
        "title": resolved_title,
        "source_file": str(source_path.relative_to(root_path)),
        "block_count": len(nodes),
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    if access_token is not None:
        client = FeishuDocClient(bearer_token=access_token)
    else:
        env_values = load_env_values(root_path / Path(env_path))
        app_id = get_required_credential(env_values, "APP_ID", "FEISHU_APP_ID", "App ID")
        app_secret = get_required_credential(env_values, "APP_SECRET", "FEISHU_APP_SECRET", "App Secret")
        client = FeishuDocClient(app_id=app_id, app_secret=app_secret)

    document = client.create_document(resolved_title, folder_token=folder_token)
    document_id = document["document_id"]
    revision_id = document.get("revision_id")
    insert_index = 0
    blocks = [node_to_block(node) for node in nodes]

    for start in range(0, len(blocks), 10):
        chunk = blocks[start : start + 10]
        append_result = client.append_children(
            document_id=document_id,
            parent_block_id=document_id,
            blocks=chunk,
            index=insert_index,
            document_revision_id=revision_id,
        )
        insert_index += len(chunk)
        revision_id = append_result.get("document_revision_id", revision_id)
        time.sleep(0.4)

    result.update({"document_id": document_id, "document_revision_id": revision_id})
    return result
