"""Centralized Hugging Face token management for the vendored rag_flmm code.

Usage:
  from rag_flmm.utils.hf_auth import login_if_available, get_hf_token
  login_if_available()
  # then use transformers.from_pretrained(...) without hardcoding tokens

Token sources (priority):
  1) config/config.yaml (auth.hf_token) or config/local.yaml (auth.hf_token)
  2) Environment variable HF_TOKEN
  3) Environment variable HF_TOKEN_FILE (path to a file containing the token)
  4) Project-local secret file: config/hf_token.secret (gitignored)

Do NOT commit tokens. Keep them in env or in config/hf_token.secret which is
ignored by .gitignore.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import login  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    login = None  # type: ignore


def _default_secret_path() -> Path:
    # This file lives at: <repo>/flmm_wiki_rag/rag_flmm/utils/hf_auth.py
    # Project root is three levels up from here: parents[2] == flmm_wiki_rag
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    return project_root / "config" / "hf_token.secret"


def _config_paths() -> list[Path]:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    cfg_dir = project_root / "config"
    return [cfg_dir / "config.yaml", cfg_dir / "local.yaml", cfg_dir / "unified.yaml"]


def _read_yaml_token() -> Optional[str]:
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    for p in _config_paths():
        try:
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # support auth: { hf_token: ... } or huggingface: { token: ... }
            auth = data.get("auth") or {}
            tok = auth.get("hf_token")
            if isinstance(tok, str) and tok.strip():
                return tok.strip()
            huggingface = data.get("huggingface") or {}
            tok2 = huggingface.get("token")
            if isinstance(tok2, str) and tok2.strip():
                return tok2.strip()
        except Exception:
            continue
    return None


def _read_first_line(path: Path) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
            return line or None
    except Exception:
        return None


def get_hf_token() -> Optional[str]:
    # 1) config files
    token = _read_yaml_token()
    if token:
        return token

    # 2) environment variable
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()

    file_env = os.getenv("HF_TOKEN_FILE")
    if file_env:
        p = Path(file_env)
        if p.exists():
            val = _read_first_line(p)
            if val:
                return val

    default_file = _default_secret_path()
    if default_file.exists():
        val = _read_first_line(default_file)
        if val:
            return val

    return None


def login_if_available(add_to_git_credential: bool = False) -> bool:
    """Login to Hugging Face Hub if a token is available.

    Returns True if a login was attempted (token found), else False.
    """
    tok = get_hf_token()
    if tok and login is not None:
        try:
            login(token=tok, add_to_git_credential=add_to_git_credential)
            return True
        except Exception:
            # Best-effort; downstream from_pretrained may still work with cached creds
            return False
    return False


__all__ = ["get_hf_token", "login_if_available"]
