"""
ask_z/scripts/ingest_box.py
────────────────────────────
Ingest documents directly from IBM Box folders into the ask-z-knowledge
Elasticsearch index — no manual downloads required.

Works with IBM Enterprise Box (ibm.ent.box.com) and standard Box (box.com).

Authentication options (pick ONE — set the matching .env vars):
  1. Developer Token  — quickest for local dev/testing (expires in 1 h)
  2. OAuth2 JWT app   — long-lived, needed for scheduled/automated ingestion

Supported file types:  .pdf  .pptx  .docx  .txt  .md  .rst
Skipped automatically: .mp4  .mov  .zip  .png  .jpg  and all other non-text

Usage
──────
  # Ingest one or more folders by their numeric IDs (from the URL):
  python -m ask_z.scripts.ingest_box \\
      --folder-id 364751967831 283303115932 368165677285 \\
      --tag ibm_z \\
      --recursive

  # Ingest by full URL (ibm.ent.box.com URLs work directly):
  python -m ask_z.scripts.ingest_box \\
      --folder-url "https://ibm.ent.box.com/folder/364751967831" \\
      --tag ibm_z_hardware \\
      --recursive

Required .env variables (at least one auth method must be set):

  BOX_DEVELOPER_TOKEN=your_dev_token   # expires in 1 h — get from developer.box.com
  # ── OR ──
  BOX_CLIENT_ID=...
  BOX_CLIENT_SECRET=...
  BOX_JWT_KEY_ID=...
  BOX_RSA_PRIVATE_KEY_PATH=/path/to/private_key.pem
  BOX_ENTERPRISE_ID=...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError

from ask_z.ingestion.doc_ingest import chunk_document
from ask_z.scripts.ingest_docs import _build_es_client, ingest_docs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("ask_z.ingest_box")

# File extensions to download and ingest.
_SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt", ".md", ".rst"}


# ── Box authentication ─────────────────────────────────────────────────────────


def _build_box_client() -> Any:
    """
    Build a Box SDK client from environment variables.

    Tries Developer Token first (simplest), then JWT app credentials.
    """
    try:
        import boxsdk
    except ImportError:
        log.error("boxsdk is not installed. Run:  pip install 'boxsdk>=3.9.0,<4.0.0'")
        sys.exit(1)

    dev_token = os.environ.get("BOX_DEVELOPER_TOKEN")
    if dev_token:
        log.info("Authenticating with Box Developer Token.")
        return boxsdk.Client(
            boxsdk.OAuth2(
                client_id="",
                client_secret="",
                access_token=dev_token,
            )
        )

    client_id = os.environ.get("BOX_CLIENT_ID")
    client_secret = os.environ.get("BOX_CLIENT_SECRET")
    jwt_key_id = os.environ.get("BOX_JWT_KEY_ID")
    rsa_key_path = os.environ.get("BOX_RSA_PRIVATE_KEY_PATH")
    enterprise_id = os.environ.get("BOX_ENTERPRISE_ID")

    if all([client_id, client_secret, jwt_key_id, rsa_key_path, enterprise_id]):
        log.info("Authenticating with Box JWT app credentials.")
        rsa_key = Path(rsa_key_path).read_text()  # type: ignore[arg-type]
        passphrase = os.environ.get("BOX_RSA_PRIVATE_KEY_PASSPHRASE")
        auth = boxsdk.JWTAuth(
            client_id=client_id,
            client_secret=client_secret,
            enterprise_id=enterprise_id,
            jwt_key_id=jwt_key_id,
            rsa_private_key_data=rsa_key,
            rsa_private_key_passphrase=passphrase,
        )
        return boxsdk.Client(auth)

    log.error(
        "No Box credentials found. Set one of:\n"
        "  BOX_DEVELOPER_TOKEN           (quickest, expires 1 h)\n"
        "  BOX_CLIENT_ID + BOX_CLIENT_SECRET + BOX_JWT_KEY_ID +\n"
        "    BOX_RSA_PRIVATE_KEY_PATH + BOX_ENTERPRISE_ID  (long-lived)\n\n"
        "How to get a Developer Token:\n"
        "  1. Go to https://developer.box.com/ — sign in with your IBM Box account\n"
        "  2. My Apps → Create New App → Custom App → User Authentication (OAuth 2.0)\n"
        "  3. Open the app → Configuration → Developer Token → Generate\n"
        "  4. Paste the token into BOX_DEVELOPER_TOKEN= in ask_z/.env"
    )
    sys.exit(1)


# ── Box folder resolution ──────────────────────────────────────────────────────


def _parse_folder_ref(raw: str) -> tuple[str, str | None]:
    """
    Parse a folder reference into (folder_id_or_url, shared_link_password).

    Handles all IBM Box URL formats:
      https://ibm.ent.box.com/folder/364751967831          → id="364751967831"
      https://ibm.ent.box.com/folder/283303115932?s=TOKEN  → id="283303115932", shared_token in URL
      https://ibm.box.com/s/TOKEN                          → shared link URL
      364751967831                                          → bare numeric ID
    """
    raw = raw.strip()

    # Bare numeric ID
    if raw.isdigit():
        return raw, None

    parsed = urlparse(raw)
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]

    # /folder/<id> — direct owned folder URL (ibm.ent.box.com, app.box.com, etc.)
    if len(path_parts) >= 2 and path_parts[0] == "folder" and path_parts[1].isdigit():
        folder_id = path_parts[1]
        # ?s= query param is a shared-link password / access token — pass it along
        qs = parse_qs(parsed.query)
        shared_token = qs.get("s", [None])[0]
        return folder_id, shared_token

    # /s/<token> path — classic shared link
    if len(path_parts) >= 2 and path_parts[0] == "s":
        return raw, None  # pass the whole URL to client.get_shared_item()

    raise ValueError(
        f"Cannot parse Box folder reference: {raw!r}\n"
        "Supported formats:\n"
        "  364751967831                                         (numeric ID)\n"
        "  https://ibm.ent.box.com/folder/364751967831         (direct URL)\n"
        "  https://ibm.ent.box.com/folder/364751967831?s=TOKEN (with shared token)\n"
        "  https://ibm.box.com/s/TOKEN                         (shared link)"
    )


def _open_folder(client: Any, raw_ref: str) -> Any:
    """Resolve any Box folder reference string to a live Box folder object."""
    try:
        folder_ref, shared_token = _parse_folder_ref(raw_ref)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    # Shared-link URL (/s/TOKEN path)
    if folder_ref.startswith("http") and "/s/" in folder_ref:
        try:
            folder = client.get_shared_item(folder_ref)
            log.info("Shared link → folder '%s' (id=%s)", folder.name, folder.id)
            return folder
        except Exception as exc:
            log.error("Cannot resolve shared link %r: %s", folder_ref, exc)
            sys.exit(1)

    # Numeric folder ID (owned folder, possibly with a shared_token we ignore for
    # direct access — the user is authenticated so the folder is already accessible)
    try:
        folder = client.folder(folder_id=folder_ref).get()
        log.info("Opened folder '%s' (id=%s)", folder.name, folder_ref)
        if shared_token:
            log.debug(
                "Shared token present in URL (ignored — using authenticated access)."
            )
        return folder
    except Exception as exc:
        log.error("Cannot open Box folder id=%s: %s", folder_ref, exc)
        log.error(
            "Possible causes:\n"
            "  • Your Developer Token expired (1-hour limit) — generate a new one\n"
            "  • The folder is in a different Box enterprise account than your token\n"
            "  • Your Box App has not been authorized by your Box Admin"
        )
        sys.exit(1)


# ── File listing and download ──────────────────────────────────────────────────


def _list_files(folder: Any, recursive: bool, _depth: int = 0) -> list[tuple[Any, str]]:
    """
    Return (BoxFile, box_web_url) tuples for every supported file in *folder*.

    Skips unsupported types (.mp4, .zip, .png, …) with a debug log.
    If recursive=True, descends into every sub-folder.
    """
    results: list[tuple[Any, str]] = []
    indent = "  " * _depth

    try:
        items = list(folder.get_items(limit=1000))
    except Exception as exc:
        log.error("%sCannot list folder '%s': %s", indent, folder.name, exc)
        return results

    files = [i for i in items if i.type == "file"]
    folders = [i for i in items if i.type == "folder"]

    supported = [
        f for f in files if Path(f.name).suffix.lower() in _SUPPORTED_EXTENSIONS
    ]
    skipped = [
        f for f in files if Path(f.name).suffix.lower() not in _SUPPORTED_EXTENSIONS
    ]

    log.info(
        "%s[%s]  %d file(s) to ingest, %d skipped (%s)",
        indent,
        folder.name,
        len(supported),
        len(skipped),
        ", ".join(Path(f.name).suffix.lower() for f in skipped) if skipped else "none",
    )

    for box_file in supported:
        # Use ibm.ent.box.com so the citation URL opens in the user's enterprise Box.
        box_url = f"https://ibm.ent.box.com/file/{box_file.id}"
        results.append((box_file, box_url))

    if recursive:
        for sub_item in folders:
            log.info("%s  ↳ sub-folder: %s", indent, sub_item.name)
            try:
                sub = sub_item.get()
                results.extend(_list_files(sub, recursive=True, _depth=_depth + 1))
            except Exception as exc:
                log.warning(
                    "%s  Cannot open sub-folder '%s': %s", indent, sub_item.name, exc
                )

    return results


def _download_file(box_file: Any, dest_dir: Path) -> Path | None:
    """Download a single Box file into dest_dir. Returns local path or None on error."""
    # Sanitise filename — Box allows characters that are invalid on macOS/Linux paths.
    safe_name = box_file.name.replace("/", "_").replace("\\", "_")
    dest = dest_dir / safe_name
    try:
        with dest.open("wb") as fh:
            box_file.download_to(fh)
        size_kb = dest.stat().st_size / 1024
        log.info("  Downloaded: %s (%.1f KB)", safe_name, size_kb)
        return dest
    except Exception as exc:
        log.warning("  Failed to download '%s': %s — skipping.", safe_name, exc)
        return None


# ── Main ingest flow ───────────────────────────────────────────────────────────


def ingest_from_box(
    folder_refs: list[str],
    *,
    tag: str | None,
    recursive: bool,
    es: Elasticsearch,
    index: str,
) -> None:
    client = _build_box_client()
    all_chunks: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="ask_z_box_") as tmp:
        tmp_path = Path(tmp)

        for ref in folder_refs:
            log.info("─── Processing folder ref: %s", ref)
            folder = _open_folder(client, ref)
            file_list = _list_files(folder, recursive=recursive)

            if not file_list:
                log.warning("No supported files found in '%s'.", folder.name)
                continue

            for box_file, box_url in file_list:
                local_path = _download_file(box_file, tmp_path)
                if local_path is None:
                    continue
                chunks = chunk_document(
                    local_path,
                    component_tag=tag,
                    source_url=box_url,  # citation shows Box URL, not /tmp path
                )
                all_chunks.extend(chunks)
                log.info("    → %d chunks from %s", len(chunks), box_file.name)

    log.info(
        "Total chunks from %d Box folder(s): %d", len(folder_refs), len(all_chunks)
    )
    ingest_docs(all_chunks, es, index)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest documents directly from IBM Box folders into ask-z-knowledge.\n"
            "Supports ibm.ent.box.com URLs. Multiple folders accepted in one run.\n\n"
            "Auth: set BOX_DEVELOPER_TOKEN in ask_z/.env (get from developer.box.com),\n"
            "or BOX_CLIENT_ID/SECRET/JWT vars for long-lived app access."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--folder-id",
        metavar="ID",
        nargs="+",
        help="One or more Box folder numeric IDs (from the URL after /folder/).",
    )
    group.add_argument(
        "--folder-url",
        metavar="URL",
        nargs="+",
        help="One or more Box folder URLs (ibm.ent.box.com/folder/... supported).",
    )
    parser.add_argument(
        "--tag",
        metavar="TAG",
        default=None,
        help="Component tag stored in every chunk, e.g. ibm_z, onboarding.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Descend into sub-folders (recommended for IBM Box folder trees).",
    )
    args = parser.parse_args()

    index = os.environ.get("ELASTIC_INDEX", "ask-z-knowledge")
    log.info("Target index: %s", index)

    es = _build_es_client()
    try:
        info = es.info()
        log.info(
            "Elasticsearch %s @ %s",
            info["version"]["number"],
            os.environ.get("ELASTIC_HOST", "http://localhost:9200"),
        )
    except ESConnectionError as exc:
        log.error("Cannot reach Elasticsearch: %s", exc)
        sys.exit(1)

    folder_refs: list[str] = args.folder_id or args.folder_url
    ingest_from_box(
        folder_refs,
        tag=args.tag,
        recursive=args.recursive,
        es=es,
        index=index,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
