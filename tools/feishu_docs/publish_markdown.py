from __future__ import annotations

import argparse
import json
from pathlib import Path

from feishu_doc_tools import publish_markdown_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a markdown file to a Feishu doc.")
    parser.add_argument("--source-file", required=True, help="Markdown file to publish.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]), help="Workspace root.")
    parser.add_argument("--title", default=None, help="Optional document title override.")
    parser.add_argument("--folder-token", default=None, help="Optional Feishu folder token.")
    parser.add_argument("--env-path", default=".env", help="Path to the credential file.")
    parser.add_argument("--dry-run", action="store_true", help="Only parse markdown and skip API calls.")
    args = parser.parse_args()

    result = publish_markdown_file(
        source_file=args.source_file,
        root=args.root,
        title=args.title,
        folder_token=args.folder_token,
        env_path=args.env_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
