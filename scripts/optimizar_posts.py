#!/usr/bin/env python3
"""Automated SEO optimization pipeline script."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from string import Template
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from dateutil import parser as date_parser
from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import OpenAI, BadRequestError
from zoneinfo import ZoneInfo

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DEFAULT_TIMEZONE = "America/Bogota"
SERP_API_ENDPOINT = "https://serpapi.com/search.json"
OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
MAX_PROMPT_CURRENT_CONTENT_CHARS = 12000
MAX_PROMPT_INDEX_ROWS = 40
MAX_PROMPT_SERP_LINES = 5

AI_RESPONSE_SCHEMA = {
    "name": "seo_optimization_payload",
    "schema": {
        "type": "object",
        "properties": {
            "seo_title": {"type": "string"},
            "meta_description": {"type": "string"},
            "h1": {"type": "string"},
            "content_html": {"type": "string"},
            "excerpt_200": {"type": "string"},
            "h2_h3_outline": {
                "type": "array",
                "items": {"type": "string"},
            },
            "internal_links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "anchor": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["anchor", "url"],
                    "additionalProperties": False,
                },
            },
            "faq_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer_html": {"type": "string"},
                    },
                    "required": ["question", "answer_html"],
                    "additionalProperties": False,
                },
            },
            "secondary_keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
            "serp_snippet_detected": {"type": "string"},
            "ia_score": {"type": "number"},
            "mejoras_aplicadas": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "seo_title",
            "meta_description",
            "h1",
            "content_html",
            "excerpt_200",
            "h2_h3_outline",
            "internal_links",
            "faq_items",
            "secondary_keywords",
            "serp_snippet_detected",
            "ia_score",
            "mejoras_aplicadas",
        ],
        "additionalProperties": False,
    },
    "strict": True,
}


def configure_logging(verbose: bool = False) -> None:
    """Configure root logger formatting."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_service_account_credentials(json_blob: str):
    """Load service account credentials from a JSON blob or file path."""
    if not json_blob:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON environment variable is required")
    try:
        if json_blob.strip().startswith("{"):
            info = json.loads(json_blob)
        else:
            with open(json_blob, "r", encoding="utf-8") as fh:
                info = json.load(fh)
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("Unable to load Google service account credentials") from exc


def normalize_header(header: str) -> str:
    """Normalize header to alphanumeric lowercase token."""
    return re.sub(r"[^0-9a-z]+", "", header.lower())


def normalize_sheet_title(title: str) -> str:
    """Normalize sheet titles ignoring spaces, underscores and case."""
    return re.sub(r"[^0-9a-z]+", "", title.lower())


def column_index_to_letter(idx: int) -> str:
    """Convert a 1-based column index to spreadsheet column letters."""
    result = ""
    while idx > 0:
        idx, remainder = divmod(idx - 1, 26)
        result = chr(65 + remainder) + result
    return result


def truncate_text(text: str, max_chars: int) -> str:
    """Trim large blocks before sending them to the LLM."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def coerce_json_friendly(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure strings we send to the prompt are JSON-safe."""
    safe: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
            cleaned = cleaned.replace("\t", " ")
            safe[key] = cleaned
        else:
            safe[key] = value
    return safe


def parse_json_blob(raw_text: str) -> Dict[str, Any]:
    """Parse JSON coming from the model, handling common wrappers."""
    if not raw_text:
        raise ValueError("OpenAI returned an empty response")
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    # Remove common emphasis markers or decorative separators the model may prepend/append
    cleaned = cleaned.lstrip("\ufeff")
    cleaned = re.sub(r"^\s*(?:\*{3,}|-{3,})\s*", "", cleaned)
    cleaned = re.sub(r"\s*(?:\*{3,}|-{3,})\s*$", "", cleaned)
    cleaned = cleaned.lstrip("* \n\r\t")
    cleaned = cleaned.rstrip("* \n\r\t")
    logger = logging.getLogger("OpenAIParser")

    def _escape_control_in_strings(text: str) -> str:
        if not text:
            return text
        result = []
        in_string = False
        escaped = False
        for char in text:
            if in_string:
                if escaped:
                    result.append(char)
                    escaped = False
                elif char == "\\":
                    result.append(char)
                    escaped = True
                elif char == '"':
                    result.append(char)
                    in_string = False
                elif char == "\n":
                    result.append("\\n")
                elif char == "\r":
                    result.append("\\r")
                elif char == "\t":
                    result.append("\\t")
                else:
                    result.append(char)
            else:
                if char == '"':
                    in_string = True
                result.append(char)
        return "".join(result)

    def _candidate_variants(text: str) -> Iterable[str]:
        yield text
        escaped_newlines = _escape_control_in_strings(text)
        if escaped_newlines != text:
            yield escaped_newlines
        escaped_html = escaped_newlines.replace('"<', '\\"<').replace('>"', '>\\"')
        if escaped_html != escaped_newlines:
            yield escaped_html

    def _try_load(text: str) -> Optional[Dict[str, Any]]:
        for candidate in _candidate_variants(text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        # Last resort: escape remaining double quotes that are not already escaped
        escaped = re.sub(r'(?<!\\)"', '\\"', text)
        try:
            return json.loads(escaped)
        except json.JSONDecodeError:
            return None

    result = _try_load(cleaned)
    if result is not None:
        return result

    # Some models may emit only key/value lines without outer braces
    stripped = cleaned.lstrip()
    if stripped.startswith('"') and "{" not in stripped and "}" not in stripped:
        wrapped = "{\n" + stripped
        if not wrapped.rstrip().endswith("}"):
            wrapped = wrapped.rstrip(", \n\r\t") + "\n}"
        result = _try_load(wrapped)
        if result is not None:
            return result

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        segment = cleaned[start : end + 1]
        result = _try_load(segment)
        if result is not None:
            return result
        logger.error("OpenAI response parse error. Snippet: %s", segment[:1000])
    else:
        logger.error("OpenAI response parse error. Snippet: %s", cleaned[:1000])
    logger.error("OpenAI response parse error (raw): %s", cleaned[:1000])
    try:
        dump_dir = Path.cwd() / "openai_failures"
        dump_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        dump_file = dump_dir / f"failure_{timestamp}.txt"
        dump_file.write_text(cleaned, encoding="utf-8")
        logger.error("OpenAI response dump saved to %s", dump_file)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Unable to persist OpenAI failure dump: %s", exc)
    raise ValueError("Unable to parse OpenAI JSON payload")


@dataclass
class SheetRecord:
    """Holds a single row from a sheet along with metadata."""
    data: Dict[str, Any]
    row_number: int

    def get(self, key: str, default: Any = "") -> Any:
        return self.data.get(key, default)


@dataclass
class SheetTable:
    """In-memory representation of a Google Sheet tab."""
    name: str
    headers: List[str]
    records: List[SheetRecord]

    def header_index(self, candidate: str) -> Optional[int]:
        target = normalize_header(candidate)
        for idx, header in enumerate(self.headers):
            if normalize_header(header) == target:
                return idx
        return None

    def resolve_header(self, candidate: str) -> Optional[str]:
        idx = self.header_index(candidate)
        return self.headers[idx] if idx is not None else None

    def find_first_by(self, keys: Iterable[str], value: str) -> Optional[SheetRecord]:
        if not value:
            return None
        for record in self.records:
            for key in keys:
                if normalize_header(key) not in {normalize_header(h) for h in self.headers}:
                    continue
                if str(record.get(key, "")).strip() == str(value).strip():
                    return record
        return None


class GoogleSheetsClient:
    """Wrapper around Google Sheets API operations used in the pipeline."""

    def __init__(self, credentials, sheet_id: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._service = build("sheets", "v4", credentials=credentials, cache_discovery=False)
        self._sheet_id = sheet_id

    def _list_sheet_titles(self) -> List[str]:
        metadata = self._service.spreadsheets().get(
            spreadsheetId=self._sheet_id, fields="sheets.properties.title"
        ).execute()
        return [sheet["properties"]["title"] for sheet in metadata.get("sheets", [])]

    def _resolve_tab_name(self, tab_name: str) -> str:
        titles = self._list_sheet_titles()
        if tab_name in titles:
            return tab_name
        normalized_target = normalize_sheet_title(tab_name)
        for title in titles:
            if normalize_sheet_title(title) == normalized_target:
                self._logger.info("Renombrando referencia de hoja '%s' a '%s' detectada en el documento", tab_name, title)
                return title
        available = ", ".join(titles) if titles else "(sin pestañas detectadas)"
        raise RuntimeError(
            f"No se encontró la pestaña '{tab_name}' en el Google Sheet. Disponibles: {available}. "
            "Actualiza TARGET_CONTENT_SHEET/INDEX_SHEET/LOG_SHEET o corrige el nombre en la hoja."
        )

    def fetch_table(self, tab_name: str) -> SheetTable:
        self._logger.debug("Fetching sheet '%s'", tab_name)
        has_range = "!" in tab_name
        sheet_name = tab_name.split("!")[0]
        resolved_name = self._resolve_tab_name(sheet_name)
        # Siempre encapsulamos el nombre de la pestaña en comillas simples para evitar errores con espacios o caracteres especiales
        quoted_name = resolved_name if resolved_name.startswith("'") else f"'{resolved_name}'"
        range_ref = tab_name if has_range else f"{quoted_name}!A1:ZZ"
        request = (
            self._service.spreadsheets()
            .values()
            .get(spreadsheetId=self._sheet_id, range=range_ref)
        )
        try:
            response = request.execute()
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"No se pudo leer la pestaña '{resolved_name}'. Verifica permisos de la cuenta de servicio o el rango solicitado"
            ) from exc
        values = response.get("values", [])
        if not values:
            raise RuntimeError(
                f"Sheet '{resolved_name}' no contiene filas (se obtuvo un dataset vacío). "
                "Añade encabezados y al menos una fila de datos o ajusta TARGET_CONTENT_SHEET."
            )
        headers = values[0]
        records = []
        for offset, row in enumerate(values[1:], start=2):
            data = {}
            for idx, header in enumerate(headers):
                data[header] = row[idx] if idx < len(row) else ""
            records.append(SheetRecord(data=data, row_number=offset))
        return SheetTable(name=tab_name, headers=headers, records=records)

    def append_row(self, tab_name: str, values: List[Any]) -> None:
        resolved_name = self._resolve_tab_name(tab_name)
        quoted_name = resolved_name if resolved_name.startswith("'") else f"'{resolved_name}'"
        body = {"values": [values]}
        self._service.spreadsheets().values().append(
            spreadsheetId=self._sheet_id,
            range=f"{quoted_name}!A:ZZ",
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body=body,
        ).execute()

    def batch_update_cells(
        self,
        table: SheetTable,
        row_number: int,
        updates: Dict[str, Any],
    ) -> None:
        data = []
        for key, value in updates.items():
            header = table.resolve_header(key)
            if header is None:
                continue
            col_idx = table.headers.index(header) + 1
            column_letter = column_index_to_letter(col_idx)
            data.append(
                {
                    "range": f"{table.name}!{column_letter}{row_number}",
                    "values": [[value]],
                }
            )
        if not data:
            return
        body = {"valueInputOption": "USER_ENTERED", "data": data}
        self._service.spreadsheets().values().batchUpdate(
            spreadsheetId=self._sheet_id,
            body=body,
        ).execute()


@dataclass
class PipelineConfig:
    """Configuration container loaded from environment and CLI."""

    sheet_id: str
    target_sheet: str
    index_sheet: str
    log_sheet: str
    wp_url: str
    wp_user: str
    wp_app_password: str
    serp_api_key: str
    serp_engine: str
    serp_location: str
    openai_api_key: str
    openai_model: str
    openai_max_output_tokens: int
    max_posts: int
    timezone: str = DEFAULT_TIMEZONE

    @classmethod
    def from_env(cls, args: argparse.Namespace) -> "PipelineConfig":
        env = os.environ
        max_posts = int(args.max_posts or env.get("MAX_POSTS_PER_RUN", "5"))
        try:
            max_output_tokens = int(env.get("OPENAI_MAX_OUTPUT_TOKENS", "3500"))
        except ValueError:
            max_output_tokens = 6000
        max_output_tokens = max(1000, max_output_tokens)
        return cls(
            sheet_id=env.get("SHEET_ID", ""),
            target_sheet=env.get("TARGET_CONTENT_SHEET", "Sheet_Final"),
            index_sheet=env.get("INDEX_SHEET", "indice_contenido"),
            log_sheet=env.get("LOG_SHEET", "Logs_Optimización"),
            wp_url=env.get("WP_URL", ""),
            wp_user=env.get("WP_USER", ""),
            wp_app_password=env.get("WP_APP_PASSWORD", ""),
            serp_api_key=env.get("SERP_API_KEY", ""),
            serp_engine=env.get("SERP_API_ENGINE", "google"),
            serp_location=env.get("SERP_API_LOCATION", "Colombia"),
            openai_api_key=env.get("OPENAI_API_KEY", ""),
            openai_model=env.get("OPENAI_MODEL", "gpt-4.1"),
            openai_max_output_tokens=max_output_tokens,
            max_posts=max_posts,
            timezone=env.get("PIPELINE_TIMEZONE", DEFAULT_TIMEZONE),
        )

    def validate(self) -> None:
        missing = []
        for field_name in (
            "sheet_id",
            "wp_url",
            "wp_user",
            "wp_app_password",
            "serp_api_key",
            "openai_api_key",
        ):
            if not getattr(self, field_name):
                missing.append(field_name)
        if missing:
            raise ValueError(f"Missing mandatory configuration values: {', '.join(missing)}")


class SerpClient:
    """Simple wrapper for SerpAPI (or compatible) calls."""

    def __init__(self, api_key: str, engine: str, location: str):
        self._api_key = api_key
        self._engine = engine
        self._location = location
        self._logger = logging.getLogger(self.__class__.__name__)

    def fetch_snapshot(self, keyword: str) -> Dict[str, Any]:
        if not keyword:
            raise ValueError("SERP keyword is empty")
        params = {
            "engine": self._engine,
            "q": keyword,
            "location": self._location,
            "api_key": self._api_key,
            "google_domain": "google.com",
            "hl": "es",
        }
        self._logger.debug("Requesting SERP snapshot for '%s'", keyword)
        response = requests.get(SERP_API_ENDPOINT, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"SERP API request failed ({response.status_code}): {response.text[:200]}"
            )
        return response.json()

    @staticmethod
    def format_snapshot(data: Dict[str, Any]) -> str:
        blocks = []
        paa_source = data.get("questions") or data.get("related_questions") or data.get("people_also_ask")
        if paa := paa_source:
            questions = [item.get("question", "") for item in paa if item.get("question")]
            if questions:
                blocks.append("People Also Ask:\n- " + "\n- ".join(questions[:MAX_PROMPT_SERP_LINES]))
        if organic := data.get("organic_results"):
            lines = []
            for entry in organic[:MAX_PROMPT_SERP_LINES]:
                title = entry.get("title", "")
                url = entry.get("link", "")
                snippet = truncate_text(entry.get("snippet", ""), 240)
                lines.append(f"{title} | {url} | {snippet}")
            if lines:
                blocks.append("Top competitors:\n- " + "\n- ".join(lines))
        if related := data.get("related_searches"):
            terms = [item.get("query", "") for item in related if item.get("query")]
            if terms:
                blocks.append("Related searches: " + ", ".join(terms[: MAX_PROMPT_SERP_LINES * 2]))
        if snippet := data.get("answer_box") or data.get("featured_snippet"):
            snippet_text = snippet.get("snippet") or snippet.get("answer") or ""
            blocks.append(f"Featured snippet candidate: {snippet_text}")
        if longtails := data.get("related_questions"):
            longtail_terms = [item.get("question", "") for item in longtails if item.get("question")]
            if longtail_terms:
                blocks.append("Long-tail ideas: " + ", ".join(longtail_terms[:MAX_PROMPT_SERP_LINES]))
        return "\n\n".join(blocks)


class WordPressClient:
    """Minimal WordPress REST API client."""

    def __init__(self, base_url: str, user: str, app_password: str):
        self._base_url = base_url.rstrip("/")
        self._auth = (user, app_password)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        response = requests.request(method, url, auth=self._auth, timeout=30, **kwargs)
        if response.status_code >= 300:
            raise RuntimeError(
                f"WordPress API {method} {url} failed ({response.status_code}): {response.text[:200]}"
            )
        return response.json()

    def fetch_post(self, post_id: Optional[int], url: Optional[str]) -> Dict[str, Any]:
        if post_id:
            return self._request("GET", f"wp-json/wp/v2/posts/{post_id}")
        if not url:
            raise ValueError("Post requires either ID or URL")
        slug = url.rstrip("/").split("/")[-1]
        query = self._request("GET", "wp-json/wp/v2/posts", params={"slug": slug})
        if not query:
            raise RuntimeError(f"No WordPress post found for slug '{slug}'")
        return query[0]

    def update_post(self, post_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._logger.debug("Updating WordPress post %s", post_id)
        return self._request("POST", f"wp-json/wp/v2/posts/{post_id}", json=payload)


class OpenAIClient:
    """Thin wrapper around OpenAI responses API."""

    def __init__(self, api_key: str, model: str, prompt_template: Template, max_output_tokens: int):
        self._client = OpenAI(api_key=api_key)
        self._api_key = api_key
        self._model = model
        self._template = prompt_template
        self._max_output_tokens = max_output_tokens
        self._fallback_output_tokens = (
            self._max_output_tokens if self._max_output_tokens <= 4000 else 4000
        )
        self._logger = logging.getLogger(self.__class__.__name__)
        self._use_responses = hasattr(self._client, "responses") and hasattr(
            self._client.responses, "create"
        )
        self._http_responses_enabled = os.environ.get("OPENAI_HTTP_RESPONSES", "").lower() in {
            "1",
            "true",
            "yes",
        }
        self._response_format = {
            "type": "json_schema",
            "json_schema": AI_RESPONSE_SCHEMA,
        }
        if not self._use_responses and not hasattr(self._client, "chat"):
            raise RuntimeError(
                "El SDK de OpenAI instalado no expone ni 'responses' ni 'chat.completions'. Actualiza la dependencia."
            )

    @staticmethod
    def _normalize(value: Any) -> Any:
        if isinstance(value, (str, dict, list, int, float, bool)) or value is None:
            return value
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:  # pylint: disable=broad-except
                pass
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:  # pylint: disable=broad-except
                pass
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            try:
                return [OpenAIClient._normalize(item) for item in value]
            except TypeError:
                pass
        if hasattr(value, "__dict__"):
            try:
                return {
                    key: OpenAIClient._normalize(val)
                    for key, val in value.__dict__.items()
                    if not key.startswith("_")
                }
            except Exception:  # pylint: disable=broad-except
                pass
        return value

    def _token_limit_candidates(self) -> Iterable[int]:
        yield self._max_output_tokens
        if self._fallback_output_tokens != self._max_output_tokens:
            yield self._fallback_output_tokens

    def _extract_first_json(self, payload: Any) -> Optional[Dict[str, Any]]:
        obj = self._normalize(payload)
        last_error: Optional[Exception] = None
        if isinstance(obj, dict):
            if isinstance(obj.get("json"), dict):
                return obj["json"]
            if isinstance(obj.get("output_json"), dict):
                return obj["output_json"]
            if isinstance(obj.get("content"), dict):
                try:
                    result = self._extract_first_json(obj["content"])
                    if result is not None:
                        return result
                except ValueError as exc:
                    last_error = exc
            if isinstance(obj.get("text"), str):
                return parse_json_blob(obj["text"])
            if isinstance(obj.get("output_text"), str):
                return parse_json_blob(obj["output_text"])
            for value in obj.values():
                try:
                    result = self._extract_first_json(value)
                    if result is not None:
                        return result
                except ValueError as exc:  # propagate last parsing error if nothing succeeds
                    last_error = exc
                    continue
            if last_error:
                raise last_error
            return None
        if isinstance(obj, list):
            for item in obj:
                try:
                    result = self._extract_first_json(item)
                    if result is not None:
                        return result
                except ValueError as exc:
                    last_error = exc
                    continue
            if last_error:
                raise last_error
            return None
        if isinstance(obj, str):
            return parse_json_blob(obj)
        return None

    def _call_responses_endpoint(self, prompt: str) -> Optional[Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        for token_limit in self._token_limit_candidates():
            request_payload: Dict[str, Any] = {
                "model": self._model,
                "input": [
                    {
                        "role": "system",
                        "content": "Eres un estratega SEO que responde únicamente con JSON válido.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": 0.4,
                "max_output_tokens": token_limit,
            }
            # Algunos despliegues requieren schema vía text.format
            request_payload["text"] = {
                "format": "json_schema",
                "json_schema": AI_RESPONSE_SCHEMA,
            }
            try:
                response = requests.post(
                    OPENAI_RESPONSES_ENDPOINT,
                    headers=headers,
                    json=request_payload,
                    timeout=90,
                )
            except requests.RequestException as exc:  # pylint: disable=broad-except
                self._logger.warning("Fallo al invocar OpenAI /responses: %s", exc)
                return None
            if response.status_code >= 300:
                lower_body = response.text.lower()
                if (
                    token_limit != self._fallback_output_tokens
                    and "max_output_tokens" in lower_body
                ):
                    self._logger.warning(
                        "OpenAI /responses rechazó max_output_tokens=%d (%s). Reintentando con %d.",
                        token_limit,
                        response.status_code,
                        self._fallback_output_tokens,
                    )
                    continue
                self._logger.warning(
                    "OpenAI /responses devolvió %s: %s",
                    response.status_code,
                    response.text[:200],
                )
                return None
            try:
                data = response.json()
            except ValueError:
                self._logger.warning("OpenAI /responses envió un cuerpo no JSON")
                return None
            try:
                return self._extract_first_json(data)
            except ValueError:
                return None
        return None

    def _invoke_prompt(self, prompt: str) -> Dict[str, Any]:
        self._logger.debug("Sending prompt to OpenAI (%d chars)", len(prompt))
        if self._use_responses:
            for token_limit in self._token_limit_candidates():
                try:
                    response = self._client.responses.create(
                        model=self._model,
                        input=[
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Eres un estratega SEO que responde únicamente con JSON válido.",
                                    }
                                ],
                            },
                            {"role": "user", "content": [{"type": "text", "text": prompt}]},
                        ],
                        temperature=0.4,
                        max_output_tokens=token_limit,
                        response_format={
                            "type": "json_schema",
                            "json_schema": AI_RESPONSE_SCHEMA,
                        },
                    )
                except BadRequestError as exc:
                    message = str(exc).lower()
                    if (
                        token_limit != self._fallback_output_tokens
                        and "max_output_tokens" in message
                    ):
                        self._logger.warning(
                            "OpenAI responses rechazó max_output_tokens=%d. Reintentando con %d.",
                            token_limit,
                            self._fallback_output_tokens,
                        )
                        continue
                    raise
                result = self._extract_first_json(response)
                if result is not None:
                    return result
                raise RuntimeError("OpenAI response did not include JSON content")

        if self._http_responses_enabled:
            direct = self._call_responses_endpoint(prompt)
            if direct is not None:
                return direct
            self._logger.info(
                "Falling back to chat.completions after responses HTTP attempt"
            )

        last_completion = None
        for token_limit in self._token_limit_candidates():
            try:
                chat = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un estratega SEO y copywriter senior. Responde únicamente con JSON válido conforme al esquema dado.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=token_limit,
                    response_format=self._response_format,
                )
            except BadRequestError as exc:
                message = str(exc).lower()
                if token_limit != self._fallback_output_tokens and "max_tokens" in message:
                    self._logger.warning(
                        "OpenAI chat rechazó max_tokens=%d. Reintentando con %d.",
                        token_limit,
                        self._fallback_output_tokens,
                    )
                    continue
                raise
            last_completion = chat
            if chat.choices:
                message = chat.choices[0].message
                result = self._extract_first_json(message)
                if result is not None:
                    return result
                finish_reason = getattr(chat.choices[0], "finish_reason", None)
                if finish_reason:
                    self._logger.warning(
                        "OpenAI chat completions finalizó con finish_reason=%s", finish_reason
                    )
                raise RuntimeError("OpenAI response did not include JSON content")
        raise RuntimeError("OpenAI returned an empty response")

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt_base = self._template.substitute(payload)
        attempts = [
            prompt_base,
            (
                f"{prompt_base}\n\nIMPORTANTE: Devuelve únicamente el objeto JSON válido solicitado. No añadas comentarios, marcadores ni markdown. "
                "Evita copiar etiquetas '<span class=\"ez-toc-section\">' u otras marcas de TOC; utiliza solo encabezados limpios."
            ),
            (
                f"{prompt_base}\n\nDEVUELVE SOLO EL JSON SOLICITADO. Limita `content_html` a un máximo de 5000 caracteres (aprox. 700 palabras), con párrafos compactos y sin listas interminables. "
                "Garantiza que cada cadena esté escapada correctamente."
            ),
            (
                f"{prompt_base}\n\nÚLTIMO INTENTO: Devuelve únicamente el JSON. Reduce `content_html` a un máximo de 2800 caracteres (~400 palabras), con hasta 4 encabezados H2 y respuestas de FAQ muy concisas. "
                "No incluyas tablas ni span personalizados."
            ),
        ]
        last_error: Optional[Exception] = None
        for attempt_index, prompt in enumerate(attempts, start=1):
            try:
                return self._invoke_prompt(prompt)
            except (ValueError, RuntimeError) as exc:
                last_error = exc
                self._logger.warning(
                    "OpenAI attempt %d failed: %s", attempt_index, exc
                )
        if isinstance(last_error, ValueError):
            raise last_error
        raise RuntimeError("OpenAI no devolvió una respuesta utilizable tras múltiples intentos")


def parse_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = date_parser.parse(value)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def count_words(html: str) -> int:
    text = re.sub(r"<[^>]+>", " ", html or "")
    text = re.sub(r"\s+", " ", text).strip()
    return len(text.split()) if text else 0


def load_prompt(path: str) -> Template:
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()
    return Template(content)


def format_sheet_slice(records: List[SheetRecord], limit: Optional[int] = None) -> str:
    lines = []
    for record in records[: limit or len(records)]:
        parts = [f"{key}: {value}" for key, value in record.data.items() if value]
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines)


def ensure_prompt_payload(
    record: SheetRecord,
    index_table: SheetTable,
    index_record: Optional[SheetRecord],
    serp_snapshot: Dict[str, Any],
    post_payload: Dict[str, Any],
    sheet_recommendations: SheetRecord,
    config: PipelineConfig,
) -> Dict[str, Any]:
    primary_keyword = (
        record.get("Keyword_Principal")
        or record.get("Keyword", "")
        or (index_record.get("Keyword_Principal") if index_record else "")
    )
    secondary_keywords = (
        sheet_recommendations.get("Keywords_Secundarias")
        or sheet_recommendations.get("Keywords Secundarias")
        or index_record.get("Keywords_Secundarias") if index_record else ""
    )
    formatted_index = format_sheet_slice(index_table.records, limit=MAX_PROMPT_INDEX_ROWS)
    formatted_serp = SerpClient.format_snapshot(serp_snapshot)
    current_content = truncate_text(
        post_payload.get("content", {}).get("rendered", ""), MAX_PROMPT_CURRENT_CONTENT_CHARS
    )
    payload = {
        "post_url": record.get("URL") or post_payload.get("link", ""),
        "primary_keyword": primary_keyword,
        "secondary_keywords": secondary_keywords,
        "current_content": current_content,
        "sheet_recommendations": json.dumps(sheet_recommendations.data, ensure_ascii=False, indent=2),
        "index_context": formatted_index,
        "serp_overview": formatted_serp,
        "traffic_notes": sheet_recommendations.get("Variacion_Trafico")
        or sheet_recommendations.get("Variacion Trafico", ""),
        "serp_location": config.serp_location,
    }
    return coerce_json_friendly(payload)


def prepare_wp_update(payload: Dict[str, Any]) -> Dict[str, Any]:
    content_html = payload["content_html"]
    h1 = payload.get("h1")
    if h1 and not content_html.strip().lower().startswith("<h1"):
        content_html = f"<h1>{h1}</h1>\n" + content_html
    meta_fields = {
        "yoast_wpseo_title": payload.get("seo_title", ""),
        "yoast_wpseo_metadesc": payload.get("meta_description", ""),
    }
    return {
        "title": payload.get("seo_title", ""),
        "content": content_html,
        "excerpt": payload.get("excerpt_200", ""),
        "meta": meta_fields,
    }


def log_to_sheet(
    sheets: GoogleSheetsClient,
    config: PipelineConfig,
    record: SheetRecord,
    optimized: Dict[str, Any],
    before_words: int,
    after_words: int,
) -> None:
    now = datetime.now(ZoneInfo(config.timezone)).strftime("%Y-%m-%d %H:%M:%S")
    log_row = [
        now,
        record.get("URL"),
        optimized.get("primary_keyword", record.get("Keyword_Principal", "")),
        ", ".join(optimized.get("mejoras_aplicadas", [])),
        before_words,
        after_words,
        optimized.get("serp_snippet_detected", ""),
        optimized.get("ia_score", ""),
    ]
    sheets.append_row(config.log_sheet, log_row)


def update_sheet_rows(
    sheets: GoogleSheetsClient,
    config: PipelineConfig,
    content_table: SheetTable,
    index_table: SheetTable,
    content_record: SheetRecord,
    index_record: Optional[SheetRecord],
    optimized: Dict[str, Any],
) -> None:
    today = datetime.now(ZoneInfo(config.timezone)).strftime("%Y-%m-%d")
    sheet_updates = {
        "Ultima_Optimización": today,
        "Keyword_Principal": optimized.get("primary_keyword", content_record.get("Keyword_Principal", "")),
        # Desactivamos la bandera para evitar reprocesar el mismo post en el siguiente run
        "Ejecutar_Accion": "NO",
    }
    sheets.batch_update_cells(content_table, content_record.row_number, sheet_updates)
    if not index_record:
        return
    index_updates = {
        "Extracto_200": optimized.get("excerpt_200", ""),
        "H2_H3": " | ".join(optimized.get("h2_h3_outline", [])),
        "Keyword_Principal": optimized.get("primary_keyword", ""),
        "Keywords_Secundarias": ", ".join(optimized.get("secondary_keywords", [])),
        "Fecha_Actualizacion": today,
        "Score_IA": optimized.get("ia_score", ""),
    }
    sheets.batch_update_cells(index_table, index_record.row_number, index_updates)


def should_process(record: SheetRecord, days_threshold: int = 30) -> bool:
    if str(record.get("Ejecutar_Accion", "")).strip().upper() != "SI":
        return False
    last_opt = parse_date(record.get("Ultima_Optimización") or record.get("Ultima Optimización"))
    if not last_opt:
        return True
    delta = datetime.now(timezone.utc) - last_opt
    return delta >= timedelta(days=days_threshold)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Automate SEO optimization pipeline")
    parser.add_argument("--prompt-file", required=True, help="Path to the prompt template file")
    parser.add_argument("--max-posts", type=int, default=None, help="Maximum posts to process in a run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    logger = logging.getLogger("pipeline")

    config = PipelineConfig.from_env(args)
    try:
        config.validate()
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    summary_path = Path(os.environ.get("PIPELINE_SUMMARY_PATH", "pipeline_summary.json"))
    try:
        if summary_path.exists():
            summary_path.unlink()
    except OSError as exc:
        logger.debug("Unable to clear previous summary file %s: %s", summary_path, exc)

    credentials = load_service_account_credentials(os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", ""))
    sheets = GoogleSheetsClient(credentials, config.sheet_id)
    serp_client = SerpClient(config.serp_api_key, config.serp_engine, config.serp_location)
    prompt_template = load_prompt(args.prompt_file)
    ai_client = OpenAIClient(
        config.openai_api_key,
        config.openai_model,
        prompt_template,
        config.openai_max_output_tokens,
    )
    wp_client = WordPressClient(config.wp_url, config.wp_user, config.wp_app_password)

    try:
        logger.info("Cargando pestaña de control: %s", config.target_sheet)
        content_table = sheets.fetch_table(config.target_sheet)
        logger.info("Cargando índice semántico: %s", config.index_sheet)
        index_table = sheets.fetch_table(config.index_sheet)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Error al consultar Google Sheets: %s", exc)
        return 1

    processed_posts = 0
    skipped: List[str] = []
    failure_details: List[str] = []
    seen_posts: set[str] = set()

    index_by_id = {
        str(rec.get("Post_ID")).strip(): rec
        for rec in index_table.records
        if rec.get("Post_ID")
    }
    index_by_url = {
        rec.get("URL"): rec
        for rec in index_table.records
        if rec.get("URL")
    }

    for record in content_table.records:
        if processed_posts >= config.max_posts:
            break
        if not should_process(record):
            continue
        post_id_raw = record.get("Post_ID") or record.get("ID")
        post_url = record.get("URL") or record.get("Enlace")
        post_id: Optional[int] = None
        if post_id_raw:
            try:
                post_id = int(str(post_id_raw).strip())
            except ValueError:
                logger.warning("Invalid Post_ID '%s' for URL %s", post_id_raw, post_url)
        post_key = str(post_id) if post_id else (post_url or "")
        if post_key and post_key in seen_posts:
            logger.debug("Skipping duplicated entry for %s", post_key)
            continue
        logger.info("Procesando post objetivo: URL=%s, ID=%s", post_url or "desconocida", post_id or "N/A")
        index_record = None
        if post_id:
            index_record = index_by_id.get(str(post_id))
        if not index_record and post_url:
            index_record = index_by_url.get(post_url)
        try:
            wp_post = wp_client.fetch_post(post_id, post_url)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Skipping post (cannot fetch WP content): %s", exc)
            failure_details.append(
                f"Fetch failed: {post_url or post_id_raw or 'desconocido'} -> {exc}"
            )
            skipped.append(post_url or str(post_id))
            continue
        try:
            primary_keyword = (
                record.get("Keyword_Principal")
                or (index_record.get("Keyword_Principal") if index_record else "")
            )
            consulta_serp = primary_keyword or record.get("Titulo", "")
            logger.info("Consultando SERP para la keyword: %s", consulta_serp)
            serp_snapshot = serp_client.fetch_snapshot(consulta_serp)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Skipping post (SERP lookup failed): %s", exc)
            failure_details.append(
                f"SERP failed: {post_url or post_id_raw or 'desconocido'} -> {exc}"
            )
            skipped.append(post_url or str(post_id))
            continue
        try:
            payload = ensure_prompt_payload(
                record,
                index_table,
                index_record,
                serp_snapshot,
                wp_post,
                record,
                config,
            )
            logger.info("Enviando prompt a OpenAI (%d caracteres)", len(prompt_template.substitute(payload)))
            optimized = ai_client.generate(payload)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Skipping post (OpenAI failure): %s", exc)
            failure_details.append(
                f"OpenAI failed: {post_url or post_id_raw or 'desconocido'} -> {exc}"
            )
            skipped.append(post_url or str(post_id))
            continue
        try:
            before_words = count_words(wp_post.get("content", {}).get("rendered", ""))
            after_words = count_words(optimized.get("content_html", ""))
            logger.info("Actualizando WordPress: antes=%d palabras, después=%d", before_words, after_words)
            update_payload = prepare_wp_update({**optimized, "primary_keyword": payload["primary_keyword"]})
            wp_client.update_post(wp_post["id"], update_payload)
            log_to_sheet(sheets, config, record, {**optimized, "primary_keyword": payload["primary_keyword"]}, before_words, after_words)
            update_sheet_rows(sheets, config, content_table, index_table, record, index_record, {**optimized, "primary_keyword": payload["primary_keyword"]})
            processed_posts += 1
            if post_key:
                seen_posts.add(post_key)
            logger.info("Optimized post %s", post_url or wp_post.get("id"))
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to update post or sheets: %s", exc)
            failure_details.append(
                f"Update failed: {post_url or post_id_raw or 'desconocido'} -> {exc}"
            )
            skipped.append(post_url or str(post_id))

    logger.info("Run complete: %d posts optimized, %d skipped", processed_posts, len(skipped))
    if skipped:
        logger.info("Skipped items: %s", ", ".join(skipped))
    summary_payload = {
        "optimized": processed_posts,
        "skipped": len(skipped),
        "skipped_items": skipped,
        "failures": failure_details,
        "generated_at": datetime.now(ZoneInfo(config.timezone)).isoformat(),
    }
    try:
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("Pipeline summary written to %s", summary_path)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Unable to write pipeline summary to %s: %s", summary_path, exc)
    return 0 if processed_posts > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
