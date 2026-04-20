from __future__ import annotations

import argparse
import json
import secrets
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from feishu_doc_tools import FEISHU_BASE_URL, get_required_credential, load_env_values, publish_markdown_file


@dataclass
class OAuthCallback:
    code: str | None = None
    state: str | None = None
    error: str | None = None


def build_authorize_url(app_id: str, redirect_uri: str, state: str) -> str:
    query = urlencode({"app_id": app_id, "redirect_uri": redirect_uri, "state": state})
    return f"https://open.feishu.cn/open-apis/authen/v1/index?{query}"


def get_app_access_token(app_id: str, app_secret: str) -> str:
    response = requests.post(
        f"{FEISHU_BASE_URL}/auth/v3/app_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code", 0) != 0:
        raise RuntimeError(f"Failed to get app_access_token: code={payload.get('code')} msg={payload.get('msg')}")
    token = payload.get("app_access_token") or payload.get("tenant_access_token")
    if not token:
        raise RuntimeError("app_access_token missing in Feishu response")
    return token


def exchange_code_for_user_token(app_access_token: str, code: str) -> dict[str, Any]:
    response = requests.post(
        f"{FEISHU_BASE_URL}/authen/v1/access_token",
        headers={"Authorization": f"Bearer {app_access_token}"},
        json={"grant_type": "authorization_code", "code": code},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code", 0) != 0:
        raise RuntimeError(f"Failed to exchange code: code={payload.get('code')} msg={payload.get('msg')}")
    data = payload.get("data") or {}
    if "access_token" not in data:
        raise RuntimeError("user access_token missing in Feishu response")
    return data


def wait_for_callback(redirect_uri: str, expected_state: str, timeout_seconds: int) -> OAuthCallback:
    parsed = urlparse(redirect_uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    expected_path = parsed.path or "/"
    result = OAuthCallback()
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802
            current = urlparse(self.path)
            if current.path != expected_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            params = parse_qs(current.query)
            result.code = (params.get("code") or [None])[0]
            result.state = (params.get("state") or [None])[0]
            result.error = (params.get("error") or [None])[0]
            status = 200 if result.code and result.state == expected_state and not result.error else 400
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if status == 200:
                body = "<h3>飞书授权成功，可以回到终端。</h3>"
            else:
                body = "<h3>飞书授权失败，请回到终端查看错误。</h3>"
            self.wfile.write(body.encode("utf-8"))
            done.set()

    with ThreadingHTTPServer((host, port), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        completed = done.wait(timeout_seconds)
        server.shutdown()
        thread.join(timeout=5)

    if not completed:
        raise TimeoutError(f"Timed out after {timeout_seconds}s waiting for Feishu callback")
    if result.error:
        raise RuntimeError(f"Feishu authorization failed: {result.error}")
    if result.state != expected_state:
        raise RuntimeError("OAuth state mismatch")
    if not result.code:
        raise RuntimeError("Authorization code missing in callback")
    return result


def authorize_and_publish(
    *,
    env_path: str | Path,
    source_file: str | Path,
    root: str | Path,
    folder_token: str,
    title: str | None,
    redirect_uri: str,
    timeout_seconds: int,
    open_browser: bool,
) -> dict[str, Any]:
    env_values = load_env_values(env_path)
    app_id = get_required_credential(env_values, "APP_ID", "FEISHU_APP_ID", "App ID")
    app_secret = get_required_credential(env_values, "APP_SECRET", "FEISHU_APP_SECRET", "App Secret")
    state = secrets.token_urlsafe(16)
    authorize_url = build_authorize_url(app_id, redirect_uri, state)

    print("1. 在飞书后台把这个回调地址加入安全设置：")
    print(f"   {redirect_uri}")
    print("2. 打开下面这个授权链接并完成登录授权：")
    print(authorize_url)
    print("")

    if open_browser:
        webbrowser.open(authorize_url)

    callback = wait_for_callback(redirect_uri, state, timeout_seconds)
    app_access_token = get_app_access_token(app_id, app_secret)
    token_payload = exchange_code_for_user_token(app_access_token, callback.code)
    user_access_token = token_payload["access_token"]

    publish_result = publish_markdown_file(
        source_file=source_file,
        root=root,
        title=title,
        folder_token=folder_token,
        access_token=user_access_token,
        dry_run=False,
    )
    publish_result["token_expires_in"] = token_payload.get("expires_in")
    publish_result["refresh_token_expires_in"] = token_payload.get("refresh_expires_in")
    return publish_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Authorize a Feishu user and publish a markdown file.")
    parser.add_argument("--env-path", default=".env", help="Path to the credential file.")
    parser.add_argument("--source-file", required=True, help="Markdown file to publish.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]), help="Workspace root.")
    parser.add_argument("--folder-token", required=True, help="Feishu target folder token.")
    parser.add_argument("--title", default=None, help="Optional document title override.")
    parser.add_argument("--redirect-uri", default="http://127.0.0.1:9876/callback", help="OAuth redirect URI.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Callback wait timeout.")
    parser.add_argument("--open-browser", action="store_true", help="Try to open the authorization URL automatically.")
    args = parser.parse_args()

    result = authorize_and_publish(
        env_path=args.env_path,
        source_file=args.source_file,
        root=args.root,
        folder_token=args.folder_token,
        title=args.title,
        redirect_uri=args.redirect_uri,
        timeout_seconds=args.timeout_seconds,
        open_browser=args.open_browser,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
