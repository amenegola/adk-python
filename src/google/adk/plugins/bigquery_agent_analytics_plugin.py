# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime
from datetime import timezone
import json
import logging
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import TYPE_CHECKING

from google.api_core.gapic_v1 import client_info as gapic_client_info
import google.auth
from google.cloud import bigquery
from google.cloud.bigquery import schema as bq_schema
from google.cloud.bigquery_storage_v1 import types as bq_storage_types
from google.cloud.bigquery_storage_v1.services.big_query_write.async_client import BigQueryWriteAsyncClient
from google.genai import types
import pyarrow as pa

from ..agents.base_agent import BaseAgent
from ..agents.callback_context import CallbackContext
from ..events.event import Event
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.base_tool import BaseTool
from ..tools.tool_context import ToolContext
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext


# --- PyArrow Helper Functions ---
def _pyarrow_datetime():
  """Returns PyArrow type for BigQuery DATETIME."""
  return pa.timestamp("us", tz=None)


def _pyarrow_numeric():
  """Returns PyArrow type for BigQuery NUMERIC."""
  return pa.decimal128(38, 9)


def _pyarrow_bignumeric():
  """Returns PyArrow type for BigQuery BIGNUMERIC."""
  return pa.decimal256(76, 38)


def _pyarrow_time():
  """Returns PyArrow type for BigQuery TIME."""
  return pa.time64("us")


def _pyarrow_timestamp():
  """Returns PyArrow type for BigQuery TIMESTAMP."""
  return pa.timestamp("us", tz="UTC")


_BQ_TO_ARROW_SCALARS = {
    "BOOL": pa.bool_,
    "BOOLEAN": pa.bool_,
    "BYTES": pa.binary,
    "DATE": pa.date32,
    "DATETIME": _pyarrow_datetime,
    "FLOAT": pa.float64,
    "FLOAT64": pa.float64,
    "GEOGRAPHY": pa.string,
    "INT64": pa.int64,
    "INTEGER": pa.int64,
    "JSON": pa.string,  # JSON is passed as string to Arrow
    "NUMERIC": _pyarrow_numeric,
    "BIGNUMERIC": _pyarrow_bignumeric,
    "STRING": pa.string,
    "TIME": _pyarrow_time,
    "TIMESTAMP": _pyarrow_timestamp,
}

_BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA = {
    "GEOGRAPHY": {
        b"ARROW:extension:name": b"google:sqlType:geography",
        b"ARROW:extension:metadata": b'{"encoding": "WKT"}',
    },
    "DATETIME": {b"ARROW:extension:name": b"google:sqlType:datetime"},
    "JSON": {b"ARROW:extension:name": b"google:sqlType:json"},
}
_STRUCT_TYPES = ("RECORD", "STRUCT")


def _bq_to_arrow_scalars(bq_scalar: str):
  """Converts a BigQuery scalar type string to a PyArrow data type constructor."""
  return _BQ_TO_ARROW_SCALARS.get(bq_scalar)


def _bq_to_arrow_struct_data_type(field):
  """Converts a BigQuery STRUCT/RECORD field to a PyArrow struct type."""
  arrow_fields = []
  for subfield in field.fields:
    arrow_subfield = _bq_to_arrow_field(subfield)
    if arrow_subfield:
      arrow_fields.append(arrow_subfield)
    else:
      logging.warning(
          "Failed to convert STRUCT/RECORD field '%s' due to subfield '%s'.",
          field.name,
          subfield.name,
      )
      return None
  return pa.struct(arrow_fields)


def _bq_to_arrow_range_data_type(field):
  """Converts a BigQuery RANGE field to a PyArrow struct type."""
  if field is None:
    raise ValueError("Range element type cannot be None")
  return pa.struct([
      ("start", _bq_to_arrow_scalars(field.element_type.upper())()),
      ("end", _bq_to_arrow_scalars(field.element_type.upper())()),
  ])


def _bq_to_arrow_data_type(field):
  """Converts a BigQuery schema field to a PyArrow data type."""
  if field.mode == "REPEATED":
    inner = _bq_to_arrow_data_type(
        bq_schema.SchemaField(
            field.name,
            field.field_type,
            fields=field.fields,
            range_element_type=getattr(field, "range_element_type", None),
        )
    )
    return pa.list_(inner) if inner else None
  field_type_upper = field.field_type.upper() if field.field_type else ""
  if field_type_upper in _STRUCT_TYPES:
    return _bq_to_arrow_struct_data_type(field)
  if field_type_upper == "RANGE":
    return _bq_to_arrow_range_data_type(field.range_element_type)
  constructor = _bq_to_arrow_scalars(field_type_upper)
  if constructor:
    return constructor()
  else:
    logging.warning(
        "Failed to convert BigQuery field '%s': unsupported type '%s'.",
        field.name,
        field.field_type,
    )
    return None


def _bq_to_arrow_field(bq_field):
  """Converts a BigQuery SchemaField to a PyArrow Field."""
  arrow_type = _bq_to_arrow_data_type(bq_field)
  if arrow_type:
    metadata = _BQ_FIELD_TYPE_TO_ARROW_FIELD_METADATA.get(
        bq_field.field_type.upper() if bq_field.field_type else ""
    )
    nullable = bq_field.mode.upper() != "REQUIRED"
    return pa.field(
        bq_field.name,
        arrow_type,
        nullable=nullable,
        metadata=metadata,
    )
  logging.warning(
      "Could not determine Arrow type for field '%s' with type '%s'.",
      bq_field.name,
      bq_field.field_type,
  )
  return None


def to_arrow_schema(bq_schema_list):
  """Converts a list of BigQuery SchemaFields to a PyArrow Schema."""
  arrow_fields = []
  for bq_field in bq_schema_list:
    af = _bq_to_arrow_field(bq_field)
    if af:
      arrow_fields.append(af)
    else:
      logging.warning(
          "Failed to convert schema due to field '%s'.", bq_field.name
      )
      return None
  return pa.schema(arrow_fields)


@dataclasses.dataclass
class BigQueryLoggerConfig:
  """Configuration for BigQueryAgentAnalyticsPlugin.

  Attributes:
    enabled: Whether logging is enabled.
    event_allowlist: A list of event types to log. If None, all events are
      logged except those in event_denylist.
    event_denylist: A list of event types to skip logging.
    content_formatter: An optional function to format event content before
      logging.
    shutdown_timeout: Seconds to wait for logs to flush during shutdown.
    client_close_timeout: Seconds to wait for BQ client to close.
    max_content_length: The maximum length of content parts before truncation.
  """

  enabled: bool = True
  event_allowlist: Optional[List[str]] = None
  event_denylist: Optional[List[str]] = None
  # Custom formatter is discouraged now that we use JSON, but kept for compat
  content_formatter: Optional[Callable[[Any], str]] = None
  shutdown_timeout: float = 5.0
  client_close_timeout: float = 2.0
  # Increased default limit to 50KB since we truncate per-field, not per-row
  max_content_length: int = 50000


def _recursive_smart_truncate(obj: Any, max_len: int) -> Any:
  """Recursively truncates string values within a dict or list."""
  if isinstance(obj, str):
    if len(obj) > max_len:
      return obj[:max_len] + "...[TRUNCATED]"
    return obj
  elif isinstance(obj, dict):
    return {k: _recursive_smart_truncate(v, max_len) for k, v in obj.items()}
  elif isinstance(obj, list):
    return [_recursive_smart_truncate(i, max_len) for i in obj]
  else:
    return obj


def _serialize_to_json_safe(content_obj: Any, max_len: int) -> str:
  """Safely serializes an object to a JSON string with smart truncation."""
  try:
    truncated_obj = _recursive_smart_truncate(content_obj, max_len)
    # default=str handles datetime or other non-serializable types by converting to string
    return json.dumps(truncated_obj, default=str)
  except Exception as e:
    logging.warning(f"JSON serialization failed: {e}")
    return json.dumps({"error": "Serialization failed", "details": str(e)})


def _get_event_type(event: Event) -> str:
  """Determines the event type from an Event object."""
  if event.author == "user":
    return "USER_INPUT"
  if event.get_function_calls():
    return "TOOL_CALL"
  if event.get_function_responses():
    return "TOOL_RESULT"
  if event.content and event.content.parts:
    return "MODEL_RESPONSE"
  if event.error_message:
    return "ERROR"
  return "SYSTEM"


class BigQueryAgentAnalyticsPlugin(BasePlugin):
  """A plugin that logs agent analytic events to Google BigQuery (Structured JSON).

  This plugin captures key events during an agent's lifecycle—such as user
  interactions, tool executions, LLM requests/responses, and errors—and
  streams them to a BigQuery table for analysis and monitoring.

  It uses the BigQuery Write API for efficient, high-throughput streaming
  ingestion and is designed to be non-blocking, ensuring that logging
  operations do not impact agent performance. If the destination table does
  not exist, the plugin will attempt to create it based on a predefined
  schema.
  """

  def __init__(
      self,
      project_id: str,
      dataset_id: str,
      table_id: str = "agent_events",
      config: Optional[BigQueryLoggerConfig] = None,
      **kwargs,
  ):
    """Initializes the BigQueryAgentAnalyticsPlugin.

    Args:
      project_id: Google Cloud project ID.
      dataset_id: BigQuery dataset ID.
      table_id: BigQuery table ID for agent events.
      config: Plugin configuration.
      **kwargs: Additional arguments.
    """
    super().__init__(name=kwargs.get("name", "BigQueryAgentAnalyticsPlugin"))
    self._project_id, self._dataset_id, self._table_id = (
        project_id,
        dataset_id,
        table_id,
    )
    self._config = config if config else BigQueryLoggerConfig()
    self._bq_client: bigquery.Client | None = None
    self._write_client: BigQueryWriteAsyncClient | None = None
    self._init_lock: asyncio.Lock | None = None
    self._arrow_schema: pa.Schema | None = None
    self._background_tasks: set[asyncio.Task] = set()
    self._is_shutting_down = False

    # --- Updated Schema: Content is now JSON ---
    self._schema = [
        bigquery.SchemaField(
            "timestamp",
            "TIMESTAMP",
            mode="REQUIRED",
            description="The UTC time at which the event was logged.",
        ),
        bigquery.SchemaField(
            "event_type",
            "STRING",
            mode="NULLABLE",
            description="Indicates the type of event (e.g., 'LLM_REQUEST').",
        ),
        bigquery.SchemaField(
            "agent",
            "STRING",
            mode="NULLABLE",
            description="The name of the ADK agent.",
        ),
        bigquery.SchemaField(
            "session_id",
            "STRING",
            mode="NULLABLE",
            description="Unique identifier for the session.",
        ),
        bigquery.SchemaField(
            "invocation_id",
            "STRING",
            mode="NULLABLE",
            description="Unique identifier for the invocation/turn.",
        ),
        bigquery.SchemaField(
            "user_id",
            "STRING",
            mode="NULLABLE",
            description="The user identifier.",
        ),
        # CHANGED: STRING -> JSON
        bigquery.SchemaField(
            "content",
            "JSON",
            mode="NULLABLE",
            description="Structured event payload.",
        ),
        bigquery.SchemaField(
            "error_message",
            "STRING",
            mode="NULLABLE",
            description="Error details if applicable.",
        ),
    ]

  async def _ensure_init(self):
    """Ensures BigQuery clients are initialized."""
    if self._write_client:
      return True
    if not self._init_lock:
      self._init_lock = asyncio.Lock()
    async with self._init_lock:
      if self._write_client:
        return True
      try:
        creds, _ = await asyncio.to_thread(
            google.auth.default,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client_info = gapic_client_info.ClientInfo(
            user_agent="google-adk-bq-logger"
        )
        self._bq_client = bigquery.Client(
            project=self._project_id, credentials=creds, client_info=client_info
        )

        def create_resources():
          if self._bq_client:
            self._bq_client.create_dataset(self._dataset_id, exists_ok=True)
            table = bigquery.Table(
                f"{self._project_id}.{self._dataset_id}.{self._table_id}",
                schema=self._schema,
            )
            table.time_partitioning = bigquery.TimePartitioning(
                type_="DAY", field="timestamp"
            )
            table.clustering_fields = ["event_type", "agent", "user_id"]
            self._bq_client.create_table(table, exists_ok=True)
            logging.info(
                "BQ Plugin: Dataset %s and Table %s ensured to exist.",
                self._dataset_id,
                self._table_id,
            )

        await asyncio.to_thread(create_resources)

        self._write_client = BigQueryWriteAsyncClient(
            credentials=creds,
            client_info=client_info,
        )
        self._arrow_schema = to_arrow_schema(self._schema)
        if not self._arrow_schema:
          raise RuntimeError("Failed to convert BigQuery schema to Arrow.")
        return True
      except Exception as e:
        logging.error("BQ Plugin: Init Failed:", exc_info=True)
        return False

  async def _perform_write(self, row: dict):
    """Actual async write operation."""
    try:
      if (
          not await self._ensure_init()
          or not self._write_client
          or not self._arrow_schema
      ):
        return

      pydict = {f.name: [row.get(f.name)] for f in self._arrow_schema}
      batch = pa.RecordBatch.from_pydict(pydict, schema=self._arrow_schema)
      req = bq_storage_types.AppendRowsRequest(
          write_stream=f"projects/{self._project_id}/datasets/{self._dataset_id}/tables/{self._table_id}/_default"
      )
      req.arrow_rows.writer_schema.serialized_schema = (
          self._arrow_schema.serialize().to_pybytes()
      )
      req.arrow_rows.rows.serialized_record_batch = (
          batch.serialize().to_pybytes()
      )

      async for resp in await asyncio.shield(
          self._write_client.append_rows(iter([req]))
      ):
        if resp.error.code != 0:
          msg = resp.error.message
          if "schema mismatch" in msg.lower():
            logging.error(
                "BQ Plugin: Schema Mismatch. You may need to delete the"
                " existing table if you migrated from STRING content to JSON"
                " content. Details: %s",
                msg,
            )
          else:
            logging.error("BQ Plugin: Write Error: %s", msg)

    except RuntimeError as e:
      if "Event loop is closed" not in str(e) and not self._is_shutting_down:
        logging.error("BQ Plugin: Runtime Error during write:", exc_info=True)
    except asyncio.CancelledError:
      if not self._is_shutting_down:
        logging.warning("BQ Plugin: Write task cancelled unexpectedly.")
    except Exception:
      logging.error("BQ Plugin: Write Failed:", exc_info=True)

  async def _log(self, data: dict, content_payload: Any = None):
    """
    Schedules a log entry.
    Args:
        data: Metadata dict (event_type, agent, etc.)
        content_payload: The structured data to be JSON serialized.
    """
    if not self._config.enabled:
      return

    event_type = data.get("event_type")
    if (
        self._config.event_denylist
        and event_type in self._config.event_denylist
    ):
      return
    if (
        self._config.event_allowlist
        and event_type not in self._config.event_allowlist
    ):
      return

    # If a custom formatter/redactor is provided, let it modify the payload
    # BEFORE we truncate and serialize it.
    if self._config.content_formatter and content_payload is not None:
      try:
        # The formatter now receives a Dict and should return a Dict
        content_payload = self._config.content_formatter(content_payload)
      except Exception as e:
        logging.warning(f"Content formatter failed: {e}")
        # Fallback: keep original payload but log the error

    # Prepare payload
    content_json_str = None
    if content_payload is not None:
      # Use smart truncation to keep JSON valid but safe size
      content_json_str = _serialize_to_json_safe(
          content_payload, self._config.max_content_length
      )

    row = {
        "timestamp": datetime.now(timezone.utc),
        "event_type": None,
        "agent": None,
        "session_id": None,
        "invocation_id": None,
        "user_id": None,
        "content": content_json_str,  # Injected here
        "error_message": None,
    }
    row.update(data)

    task = asyncio.create_task(self._perform_write(row))
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)

  async def close(self):
    """Flushes pending logs and closes client."""
    if self._is_shutting_down:
      return
    self._is_shutting_down = True
    logging.info("BQ Plugin: Shutdown started.")

    if self._background_tasks:
      logging.info(
          "BQ Plugin: Flushing %s pending logs...", len(self._background_tasks)
      )
      try:
        await asyncio.wait(
            self._background_tasks, timeout=self._config.shutdown_timeout
        )
      except asyncio.TimeoutError:
        logging.warning("BQ Plugin: Timeout waiting for logs to flush.")
      except Exception as e:
        logging.warning("BQ Plugin: Error flushing logs:", exc_info=True)

    if self._write_client and getattr(self._write_client, "transport", None):
      try:
        logging.info("BQ Plugin: Closing write client.")
        await asyncio.wait_for(
            self._write_client.transport.close(),
            timeout=self._config.client_close_timeout,
        )
      except Exception as e:
        logging.warning("BQ Plugin: Error closing write client: %s", e)
        pass
    if self._bq_client:
      try:
        self._bq_client.close()
      except Exception as e:
        logging.warning("BQ Plugin: Error closing BQ client: %s", e)

    self._write_client = None
    self._bq_client = None
    self._is_shutting_down = False
    logging.info("BQ Plugin: Shutdown complete.")

  # --- Refactored Callbacks using Structured Data ---

  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> None:
    """Callback for user messages.

    Logs the user message details including:
    1. User content (text)

    The content is formatted as a structured JSON object containing the user text.
    If individual string fields exceed `max_content_length`, they are truncated
    to preserve the valid JSON structure.
    """
    # Extract text parts
    text_content = ""
    if user_message and user_message.parts:
      text_content = " ".join([p.text for p in user_message.parts if p.text])

    payload = {"text": text_content if text_content else None}

    await self._log(
        {
            "event_type": "USER_MESSAGE_RECEIVED",
            "agent": invocation_context.agent.name,
            "session_id": invocation_context.session.id,
            "invocation_id": invocation_context.invocation_id,
            "user_id": invocation_context.session.user_id,
        },
        content_payload=payload,
    )

  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Callback before agent invocation.

    Logs the start of an agent invocation.
    No specific content payload is logged for this event, but standard metadata
    (agent name, session ID, invocation ID, user ID) is captured.
    """
    await self._log({
        "event_type": "INVOCATION_STARTING",
        "agent": invocation_context.agent.name,
        "session_id": invocation_context.session.id,
        "invocation_id": invocation_context.invocation_id,
        "user_id": invocation_context.session.user_id,
    })  # No content payload needed

  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> None:
    """Callback for agent events.

    Logs generic agent events including:
    1. Event type (determined from event properties)
    2. Event content (text, function calls, or responses)
    3. Error messages (if any)

    The content is formatted as a structured JSON object based on the event type.
    If individual string fields exceed `max_content_length`, they are truncated
    to preserve the valid JSON structure.
    """
    # We try to extract text, but keep it simple for generic events
    text_parts = []
    tool_calls = []
    tool_responses = []

    if event.content and event.content.parts:
      for p in event.content.parts:
        if p.text:
          text_parts.append(p.text)
        if p.function_call:
          tool_calls.append(p.function_call.name)
        if p.function_response:
          tool_responses.append(p.function_response.name)

    payload = {
        "text": " ".join(text_parts) if text_parts else None,
        "tool_calls": tool_calls if tool_calls else None,
        "tool_responses": tool_responses if tool_responses else None,
        "raw_role": event.author if event.author else None,
    }

    await self._log(
        {
            "event_type": _get_event_type(event),
            "agent": event.author,
            "session_id": invocation_context.session.id,
            "invocation_id": invocation_context.invocation_id,
            "user_id": invocation_context.session.user_id,
            "error_message": event.error_message,
        },
        content_payload=payload,
    )

  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Callback after agent invocation.

    Logs the completion of an agent invocation.
    No specific content payload is logged for this event, but standard metadata
    (agent name, session ID, invocation ID, user ID) is captured.
    """
    await self._log({
        "event_type": "INVOCATION_COMPLETED",
        "agent": invocation_context.agent.name,
        "session_id": invocation_context.session.id,
        "invocation_id": invocation_context.invocation_id,
        "user_id": invocation_context.session.user_id,
    })

  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> None:
    """Callback before an agent starts.

    Logs the start of a specific agent execution.
    Content includes:
    1. Agent Name (from callback context)
    """
    await self._log(
        {
            "event_type": "AGENT_STARTING",
            "agent": agent.name,
            "session_id": callback_context.session.id,
            "invocation_id": callback_context.invocation_id,
            "user_id": callback_context.session.user_id,
        },
        content_payload={"target_agent": callback_context.agent_name},
    )

  async def after_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> None:
    """Callback after an agent completes.

    Logs the completion of a specific agent execution.
    Content includes:
    1. Agent Name (from callback context)
    """
    await self._log(
        {
            "event_type": "AGENT_COMPLETED",
            "agent": agent.name,
            "session_id": callback_context.session.id,
            "invocation_id": callback_context.invocation_id,
            "user_id": callback_context.session.user_id,
        },
        content_payload={"target_agent": callback_context.agent_name},
    )

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> None:
    """Callback before LLM call.

    Logs the LLM request details including:
    1. Model name
    2. Configuration parameters (temperature, top_p, top_k, max_output_tokens)
    3. Available tool names
    4. Prompt content (user/model messages)
    5. System instructions

    The content is formatted as a structured JSON object.
    If individual string fields exceed `max_content_length`, they are truncated
    to preserve the valid JSON structure.
    """

    # 1. Config Params
    params = {}
    if llm_request.config:
      cfg = llm_request.config
      if getattr(cfg, "temperature", None) is not None:
        params["temperature"] = cfg.temperature
      if getattr(cfg, "top_p", None) is not None:
        params["top_p"] = cfg.top_p
      if getattr(cfg, "top_k", None) is not None:
        params["top_k"] = cfg.top_k
      if getattr(cfg, "max_output_tokens", None) is not None:
        params["max_output_tokens"] = cfg.max_output_tokens

    # 2. System Instruction
    system_instr = None
    if llm_request.config and llm_request.config.system_instruction is not None:
      si = llm_request.config.system_instruction
      if isinstance(si, str):
        system_instr = si
      elif isinstance(si, types.Content):
        system_instr = "".join(p.text for p in si.parts if p.text)
      elif isinstance(si, types.Part):
        system_instr = si.text
      elif hasattr(si, "__iter__"):
        texts = []
        for item in si:
          if isinstance(item, str):
            texts.append(item)
          elif isinstance(item, types.Part) and item.text:
            texts.append(item.text)
        system_instr = "".join(texts)
      else:
        system_instr = str(si)

    # 3. Prompt History (Simplified structure for JSON)
    prompt_history = []
    if getattr(llm_request, "contents", None):
      for c in llm_request.contents:
        role = c.role
        parts_list = []
        for p in c.parts:
          if p.text:
            parts_list.append({"type": "text", "text": p.text})
          elif p.function_call:
            parts_list.append(
                {"type": "function_call", "name": p.function_call.name}
            )
          elif p.function_response:
            parts_list.append(
                {"type": "function_response", "name": p.function_response.name}
            )
        prompt_history.append({"role": role, "parts": parts_list})

    payload = {
        "model": llm_request.model or "default",
        "params": params,
        "tools_available": (
            list(llm_request.tools_dict.keys())
            if llm_request.tools_dict
            else []
        ),
        "system_instruction": system_instr,
        "prompt": prompt_history,
    }

    await self._log(
        {
            "event_type": "LLM_REQUEST",
            "agent": callback_context.agent_name,
            "session_id": callback_context.session.id,
            "invocation_id": callback_context.invocation_id,
            "user_id": callback_context.session.user_id,
        },
        content_payload=payload,
    )

  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> None:
    """Callback after LLM call.

    Logs the LLM response details including:
    1. Tool calls (if any)
    2. Text response (if no tool calls)
    3. Token usage statistics (prompt, candidates, total)

    The content is formatted as a structured JSON object containing response parts
    and usage statistics.
    If individual string fields exceed `max_content_length`, they are truncated
    to preserve the valid JSON structure.
    """
    content_parts = []
    if llm_response.content and llm_response.content.parts:
      for p in llm_response.content.parts:
        if p.text:
          content_parts.append({"type": "text", "text": p.text})
        if p.function_call:
          content_parts.append({
              "type": "function_call",
              "name": p.function_call.name,
              "args": dict(p.function_call.args),
          })

    usage = {}
    if llm_response.usage_metadata:
      usage = {
          "prompt_tokens": getattr(
              llm_response.usage_metadata, "prompt_token_count", 0
          ),
          "candidates_tokens": getattr(
              llm_response.usage_metadata, "candidates_token_count", 0
          ),
          "total_tokens": getattr(
              llm_response.usage_metadata, "total_token_count", 0
          ),
      }

    payload = {
        "response_content": content_parts if content_parts else None,
        "usage": usage if usage else None,
    }

    await self._log(
        {
            "event_type": "LLM_RESPONSE",
            "agent": callback_context.agent_name,
            "session_id": callback_context.session.id,
            "invocation_id": callback_context.invocation_id,
            "user_id": callback_context.session.user_id,
            "error_message": llm_response.error_message,
        },
        content_payload=payload,
    )

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> None:
    """Callback before tool call.

    Logs the tool execution start details including:
    1. Tool name
    2. Tool description
    3. Tool arguments

    The content is formatted as a structured JSON object containing tool name,
    description, and arguments.
    If individual string fields exceed `max_content_length`, they are truncated
    to preserve the valid JSON structure.
    """
    
    payload = {
        "tool_name": tool.name if tool.name else None,
        "description": tool.description if tool.description else None,
        "arguments": tool_args if tool_args else None,
    }
    await self._log(
        {
            "event_type": "TOOL_STARTING",
            "agent": tool_context.agent_name,
            "session_id": tool_context.session.id,
            "invocation_id": tool_context.invocation_id,
            "user_id": tool_context.session.user_id,
        },
        content_payload=payload,
    )

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> None:
    """Callback after tool call.

    Logs the tool execution result details including:
    1. Tool name
    2. Tool result

    The content is formatted as a structured JSON object containing tool name and result.
    If individual string fields exceed `max_content_length`, they are truncated
    to preserve the valid JSON structure.
    """
    payload = {"tool_name": tool.name if tool.name else None, "result": result if result else None}
    await self._log(
        {
            "event_type": "TOOL_COMPLETED",
            "agent": tool_context.agent_name,
            "session_id": tool_context.session.id,
            "invocation_id": tool_context.invocation_id,
            "user_id": tool_context.session.user_id,
        },
        content_payload=payload,
    )

  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> None:
    """Callback for model errors.

    Logs errors that occur during LLM calls.
    No specific content payload is logged, but the error message is captured
    in the `error_message` field.
    """
    await self._log({
        "event_type": "LLM_ERROR",
        "agent": callback_context.agent_name,
        "session_id": callback_context.session.id,
        "invocation_id": callback_context.invocation_id,
        "user_id": callback_context.session.user_id,
        "error_message": str(error),
    })

  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> None:
    """Callback for tool errors.

    Logs errors that occur during tool execution.
    Content includes:
    1. Tool name
    2. Tool arguments

    The content is formatted as a structured JSON object containing tool name and arguments.
    The error message is captured in the `error_message` field.
    If individual string fields exceed `max_content_length`, they are truncated
    to preserve the valid JSON structure.
    """
    payload = {"tool_name": tool.name if tool.name else None, "arguments": tool_args if tool_args else None}
    await self._log(
        {
            "event_type": "TOOL_ERROR",
            "agent": tool_context.agent_name,
            "session_id": tool_context.session.id,
            "invocation_id": tool_context.invocation_id,
            "user_id": tool_context.session.user_id,
            "error_message": str(error),
        },
        content_payload=payload,
    )
