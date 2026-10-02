# API Reference

## `GET /api/health`

Returns service status.

## `GET /api/stats`

Returns document count, chunk count, token count, indexed document names, and supported file extensions.

## `POST /api/upload`

Accepts multipart form data with a `document` file field.

## `POST /api/ask`

Accepts JSON:

```json
{
  "question": "What are the attendance rules?"
}
```

Returns an answer grounded in retrieved chunks plus source references.

## `POST /api/rebuild`

Rebuilds the index from files in `data/documents`.

## `DELETE /api/documents/{filename}`

Removes a stored document and its indexed chunks.
