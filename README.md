# SimpleAgents FastAPI PII Redaction

FastAPI service that redacts personally identifiable information using a SimpleAgents workflow (YAML + hosted LLM). The API returns **indexed** placeholders (for example `<PII_NAME_1>`) and a **deterministic mapping** from those keys back to the original substrings—without asking an LLM to build the map.

## How the pipeline works

1. **Chunking** — Input text is split on lines into chunks of **3 lines** each (the last chunk may be shorter). Original newline characters are preserved when chunks are stitched back together.

2. **Concurrent LLM redaction** — Up to **3 chunks** run at a time. Chunks that show no PII-like signals may skip the model call and pass through unchanged.

3. **Line-structure fidelity** — Each chunk sent to the model includes hidden line markers so the service can restore the **exact line count and endings** of the source chunk even if the model’s raw reply drifts.

4. **Base placeholders** — The model is instructed to replace sensitive spans with tokens like `<PII_NAME>` or `<PII_PHONE_NUMBER>` and to leave all other characters unchanged.

5. **Deterministic post-processing (no LLM)** — The service aligns each `<PII_*>` region between the **original** chunk and the **redacted** chunk to recover the exact substring that was replaced, then rewrites placeholders to **indexed** forms (`<PII_NAME_1>`, `<PII_NAME_2>`, …) and builds `mapping_keys`.

6. **Partial success** — If a chunk fails (workflow error or alignment failure), that chunk’s text is left **unchanged** in the output and an entry is added to `failed_chunks`. Other chunks still contribute redacted text.

## Sample data at each stage

Illustrative snippet (line breaks shown as `\n` where helpful):

**Stage A — Original input**

```text
My name is Rishub.
My sister is Candice.
Phone: 90090880
```

**Stage B — One chunk (3 lines) sent to the workflow**

The service may wrap lines with internal markers for reconstruction; the **semantic** content is still these three lines.

**Stage C — Model output for that chunk (base placeholders)**

```text
My name is <PII_NAME>.
My sister is <PII_NAME>.
Phone: <PII_PHONE_NUMBER>
```

**Stage D — After alignment + indexing (what the API returns)**

`redacted_text`:

```text
My name is <PII_NAME_1>.
My sister is <PII_NAME_2>.
Phone: <PII_PHONE_NUMBER_1>
```

`mapping_keys`:

```json
{
  "PII_NAME_1": "Rishub",
  "PII_NAME_2": "Candice",
  "PII_PHONE_NUMBER_1": "90090880"
}
```

**Stage E — Full HTTP response shape**

```json
{
  "redacted_text": "My name is <PII_NAME_1>.\nMy sister is <PII_NAME_2>.\nPhone: <PII_PHONE_NUMBER_1>",
  "mapping_keys": {
    "PII_NAME_1": "Rishub",
    "PII_NAME_2": "Candice",
    "PII_PHONE_NUMBER_1": "90090880"
  },
  "total_chunks": 1,
  "succeeded_chunks": 1,
  "failed_chunks": []
}
```

When something goes wrong for chunk index `2`, you might see:

```json
"failed_chunks": [
  {
    "chunk_index": 2,
    "stage": "llm_execution",
    "error": "Workflow execution failed: ..."
  }
]
```

Stages include `llm_execution`, `postprocess_alignment`, or `global_placeholder_indexing` (`chunk_index`: `-1`).

## PII categories

Supported placeholder families (unknown `PII_*` labels from the model may be normalized to `PII_ID_NUMBER`):

- `PII_NAME`
- `PII_PHONE_NUMBER`
- `PII_EMAIL`
- `PII_ADDRESS`
- `PII_DOB`
- `PII_ID_NUMBER`
- `PII_BANK_ACCOUNT`

## Environment variables

Required:

- `WORKFLOW_PROVIDER`
- `WORKFLOW_API_BASE`
- `WORKFLOW_API_KEY`

Optional:

- `PII_CHUNK_TIMEOUT_SECONDS` (default: `45`)
- `PII_RETRY_ATTEMPTS` (default: `3`)

## Install

```bash
uv sync
```

## Run

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

### `POST /redact`

JSON body:

```json
{
  "text": "My name is Rishub.\nMy sister is Candice.\nPhone: 90090880"
}
```

Returns `RedactResponse`: `redacted_text`, `mapping_keys`, `total_chunks`, `succeeded_chunks`, `failed_chunks`.

### `POST /redact-file`

`multipart/form-data` with a single `.md` file field. The file is read as UTF-8 and processed like `/redact`.

Example with curl:

```bash
curl -sS -X POST "http://localhost:8000/redact-file" \
  -F "file=@sample_5_page_bank_statement_fake_pii.md"
```

## Notes

- Workflow runs with **streaming disabled** (`stream: false` in `workflow.yaml`).
- Configure the model ID in `workflow.yaml` to match a model your provider exposes.
