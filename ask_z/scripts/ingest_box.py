"""
ask_z/scripts/ingest_box.py
────────────────────────────
Ingest documents directly from an IBM Box folder into the ask-z-knowledge
Elasticsearch index — no manual download required.

Authentication options (pick ONE and set the matching .env vars):
  1. Developer Token  — quickest for local dev/testing (expires in 1 h)
  2. OAuth2 JWT app   — long-lived, needs a Box app with JWT keypair

Supported file types downloaded from Box:
  .pdf  .pptx  .docx  .txt  .md  .rst

Usage
──────
  # Ingest a shared Box folder by its URL:
  python -m ask_z.scripts.ingest_box \\
      --folder-url "https://ibm.box.com/s/abc123xyz" \\
      --tag ibm_z_hardware

  # Ingest a folder you own by its numeric folder ID:
  python -m ask_z.scripts.ingest_box \\
      --folder-id 123456789 \\
      --tag onboarding

  # Recurse into sub-folders:
  python -m ask_z.scripts.ingest_box \\
      --folder-id 123456789 \\
      --tag ibm_z \\
      --recursive

Required .env variables (at least one auth method must be configured):

  # Method 1 — Developer Token (expires 1 h, good for testing):
  BOX_DEVELOPER_TOKEN=your_dev_token_here

  # Method 2 — JWT App (long-lived, recommended for production):
  BOX_CLIENT_ID=your_app_client_id
  BOX_CLIENT_SECRET=your_app_client_secret
  BOX_JWT_KEY_ID=your_jwt_key_id
  BOX_RSA_PRIVATE_KEY_PATH=/path/to/private_key.pem
  BOX_ENTERPRISE_ID=your_enterprise_id
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError as ESConnectionError

from ask_z.ingestion.doc_ingest import chunk_document
from ask_z.scripts.ingest_docs import ingest_docs

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


def _build_box_client():
    """
    Build a Box SDK client from environment variables.

    Tries Developer Token first (simplest for local dev), then falls back
    to JWT app credentials.
    """
    try:
        import boxsdk
    except ImportError:
        log.error("boxsdk is not installed. Run:\n  pip install 'boxsdk>=3.9.0,<4.0.0'")
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

    # JWT app auth — needs BOX_CLIENT_ID, BOX_CLIENT_SECRET, BOX_JWT_KEY_ID,
    # BOX_RSA_PRIVATE_KEY_PATH (or BOX_RSA_PRIVATE_KEY_PASSPHRASE), BOX_ENTERPRISE_ID.
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
        "    BOX_RSA_PRIVATE_KEY_PATH + BOX_ENTERPRISE_ID  (long-lived JWT app)\n\n"
        "See: https://developer.box.com/guides/authentication/"
    )
    sys.exit(1)


# ── Box folder resolution ──────────────────────────────────────────────────────


def _folder_id_from_url(url: str) -> str:
    """
    Extract the folder ID from a Box shared link URL.

    Box shared-link URLs look like:
      https://ibm.box.com/s/abc123xyz      ← shared link token, not a folder ID
      https://app.box.com/folder/123456789 ← direct folder URL with numeric ID

    For shared links we must call the Box API to resolve the token → folder ID.
    """
    parsed = urlparse(url)
    path_parts = parsed.path.strip("/").split("/")

    # Direct URL: /folder/<id>
    if len(path_parts) >= 2 and path_parts[0] == "folder" and path_parts[1].isdigit():
        return path_parts[1]

    # /s/<token> — shared link, needs API resolution
    if len(path_parts) >= 2 and path_parts[0] == "s":
        return url  # return raw URL; resolved in _get_box_folder()

    raise ValueError(
        f"Cannot parse Box folder ID from URL: {url}\n"
        "Expected format:\n"
        "  https://app.box.com/folder/123456789   (direct)\n"
        "  https://ibm.box.com/s/abc123xyz        (shared link)"
    )


def _get_box_folder(client: Any, folder_id_or_url: str) -> Any:
    """Resolve a folder ID or shared-link URL to a Box folder object."""
    # Shared link (URL containing /s/)
    if folder_id_or_url.startswith("http") and "/s/" in folder_id_or_url:
        try:
            folder = client.get_shared_item(folder_id_or_url)
            log.info(
                "Resolved shared link → folder '%s' (id=%s)",
                folder.name,
                folder.id,
            )
            return folder
        except Exception as exc:
            log.error(
                "Failed to resolve Box shared link '%s': %s\n"
                "Make sure the link is accessible with your credentials.",
                folder_id_or_url,
                exc,
            )
            sys.exit(1)

    # Direct folder ID (numeric string or URL with /folder/<id>)
    fid = folder_id_or_url
    if folder_id_or_url.startswith("http"):
        fid = _folder_id_from_url(folder_id_or_url)

    try:
        folder = client.folder(folder_id=fid).get()
        log.info("Opened Box folder '%s' (id=%s)", folder.name, fid)
        return folder
    except Exception as exc:
        log.error("Failed to open Box folder id=%s: %s", fid, exc)
        sys.exit(1)


# ── File listing and download ──────────────────────────────────────────────────


def _list_files(folder: Any, recursive: bool) -> list[tuple[Any, str]]:
    """
    Return list of (BoxFile, box_url) tuples from a folder.

    If recursive=True, descends into sub-folders.
    """
    results: list[tuple[Any, str]] = []
    try:
        items = folder.get_items(limit=1000)
    except Exception as exc:
        log.error("Cannot list Box folder '%s': %s", folder.name, exc)
        return results

    for item in items:
        if item.type == "file":
            ext = Path(item.name).suffix.lower()
            if ext in _SUPPORTED_EXTENSIONS:
                # Build a direct Box URL for the file (used as source_url in citations).
                box_url = f"https://app.box.com/file/{item.id}"
                results.append((item, box_url))
            else:
                log.debug("Skipping unsupported file type: %s", item.name)
        elif item.type == "folder" and recursive:
            log.info("Descending into sub-folder: %s", item.name)
            sub = item.get()
            results.extend(_list_files(sub, recursive=True))

    return results


def _download_file(box_file: Any, dest_dir: Path) -> Path | None:
    """Download a Box file to dest_dir. Returns the local path, or None on error."""
    dest = dest_dir / box_file.name
    try:
        with dest.open("wb") as fh:
            box_file.download_to(fh)
        log.info("Downloaded: %s (%.1f KB)", box_file.name, dest.stat().st_size / 1024)
        return dest
    except Exception as exc:
        log.warning("Failed to download '%s': %s — skipping.", box_file.name, exc)
        return None


# ── Elasticsearch client (reused from ingest_docs) ────────────────────────────


def _build_es_client() -> Elasticsearch:
    from ask_z.scripts.ingest_docs import _build_es_client as _es

    return _es()


# ── Main ingest flow ───────────────────────────────────────────────────────────


def ingest_from_box(
    folder_ref: str,
    *,
    tag: str | None,
    recursive: bool,
    es: Elasticsearch,
    index: str,
) -> None:
    client = _build_box_client()
    folder = _get_box_folder(client, folder_ref)

    log.info(
        "Listing files in '%s'%s …",
        folder.name,
        " (recursive)" if recursive else "",
    )
    file_list = _list_files(folder, recursive=recursive)
    log.info("Found %d supported file(s) to ingest.", len(file_list))

    if not file_list:
        log.warning(
            "No supported files found. Supported: %s", sorted(_SUPPORTED_EXTENSIONS)
        )
        return

    all_chunks: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="ask_z_box_") as tmp:
        tmp_path = Path(tmp)
        for box_file, box_url in file_list:
            local_path = _download_file(box_file, tmp_path)
            if local_path is None:
                continue
            chunks = chunk_document(
                local_path,
                component_tag=tag,
                source_url=box_url,  # ← citations show Box URL, not local /tmp path
            )
            all_chunks.extend(chunks)
            log.info("  → %d chunks from %s", len(chunks), box_file.name)

    log.info("Total chunks from Box: %d", len(all_chunks))
    ingest_docs(all_chunks, es, index)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest documents directly from an IBM Box folder into ask-z-knowledge.\n\n"
            "Auth: set BOX_DEVELOPER_TOKEN in ask_z/.env (quickest),\n"
            "or BOX_CLIENT_ID / BOX_CLIENT_SECRET / BOX_JWT_KEY_ID /\n"
            "BOX_RSA_PRIVATE_KEY_PATH / BOX_ENTERPRISE_ID for long-lived JWT auth."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--folder-url",
        metavar="URL",
        help="Box folder shared-link or direct URL, e.g. https://ibm.box.com/s/abc123",
    )
    group.add_argument(
        "--folder-id",
        metavar="ID",
        help="Box folder numeric ID (visible in the URL when browsing Box).",
    )
    parser.add_argument(
        "--tag",
        metavar="TAG",
        help="Component tag stored in every chunk, e.g. ibm_z_hardware, onboarding.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Descend into sub-folders.",
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

    folder_ref = args.folder_url or args.folder_id
    ingest_from_box(
        folder_ref,
        tag=args.tag,
        recursive=args.recursive,
        es=es,
        index=index,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
