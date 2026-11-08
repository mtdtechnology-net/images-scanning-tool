# Task: show managers whether each submitted report matches its receipts (Spendle backend + web)

You are working in the **Spendle** repository: a Kotlin backend and a web front end. Read this whole
brief first. Then explore the repo and reply with a plan (see "Before writing code"). **Wait for
confirmation before implementing.**

## 1. Product context

Spendle is an expense app. The full flow:

1. The AI scans a receipt (existing).
2. The expense form is auto-filled (existing).
3. The employee reviews and submits the report (existing).
4. A **receipt verifier** re-reads every receipt in the background. This is built and lives in a
   separate service, `scanning-tool`. You only call it.
5. **(Your task)** The manager sees a status per report.
6. **(Your task)** The manager sees which field of which expense differs from its receipt, with the
   spot highlighted on the receipt.

Product principle: **the AI does the work, people make the decisions.** Nothing in this task approves,
rejects, blocks or edits anything. You only compare and display. The verifier never sees what the
employee declared. The comparison is your code, with the plain rules below. No AI is involved on
the Spendle side.

What matters most, in this order:
1. **No false alarms.** A manager who sees "needs a look" on correct reports learns to ignore it.
2. **The total amount and the currency.** The date and the document number are informative only.

## 2. The verifier API (scanning-tool)

`scanning-tool` is the service Spendle already calls for scanning (`POST /api/web-report-stream`,
SSE). Find how the backend reaches it today (base URL config, HTTP client) and reuse that. It has
**no authentication** and must only be called **from the backend over the private network**, never
from the browser.

### Request
`POST {scanning-tool base URL}/api/verify-receipt`
- `multipart/form-data`, **one file** in the field `file`: PDF, PNG, JPEG or WEBP, max 10 MB,
  max 20 pages.
- Send only the file. **Never send declared values or anything else from the form.**

### Responses
| HTTP | Body `status` | Meaning | What you do |
|---|---|---|---|
| 200 | `ok` | Read; check each field's `confidence` | compare (section 3) |
| 200 | `unreadable` | Too degraded, or the reads disagree on the total | outcome "could not verify" |
| 200 | `not_a_receipt` | Not a receipt or invoice | outcome "could not verify" |
| 200 | `multiple_receipts` | Several receipts in one file | outcome "could not verify" |
| 503 + `Retry-After` | `error` | Temporary: GPU busy, preempted by a live scan, model loading, timeout | **retry later** (section 4) |
| 400 | FastAPI `{"detail": ...}` | Unsupported or corrupt file | do not retry; outcome "could not verify" |

Timing: typically **2–6 minutes per receipt**. The service may first wait up to 60 s for live
scans to finish, and a verification is capped at 20 min. **Set the HTTP read timeout to at least
25 minutes.** The service verifies one receipt at a time, so throughput is roughly 10–20 receipts
per hour, less during office hours when live scans interrupt verifications.

### Body (stable contract)
```json
{
  "status": "ok",
  "model": "qwen3-vl:8b",
  "models": ["qwen3-vl:8b"],
  "duration_ms": 152400,
  "pages": [{ "page": 1, "width": 1447, "height": 2048 }],
  "fields": {
    "total_amount":    { "value": 219.2, "raw": "219,20", "confidence": "high", "page": 1, "bbox": [987, 733, 1113, 795] },
    "currency":        { "value": null, "raw": null, "confidence": "none", "page": null, "bbox": null },
    "receipt_date":    { "value": "2026-01-27", "raw": "27.01.2026", "confidence": "medium", "page": 1, "bbox": null },
    "document_number": { "value": "0109-00214", "raw": "NUMAR BON: 0109-00214", "confidence": "medium", "page": 1, "bbox": null }
  },
  "notes": null
}
```
- `value` is normalized:
  - `total_amount`: a number, the **final amount paid, VAT included**;
  - `currency`: ISO 4217 (`LEI` becomes `RON`), or `null` when **no currency is printed**. This is very common on Romanian fuel receipts and is **not an error**;
  - `receipt_date`: `YYYY-MM-DD`;
  - `document_number`: a string.
- `raw` is the text as read.
- `confidence` is one of:
  - `high`: confirmed by independent reads (including OCR);
  - `medium`: agreed, with weaker confirmation;
  - `low`: uncertain;
  - `none`: not on the document.
- `bbox` is `[x0, y0, x1, y1]` in pixels of the page image described in `pages` (1-based `page`), or `null`. It is never `[]`.
- `notes` is a short technical English text (for logs and a details tooltip), or `null`. Don't parse it.
- Treat unknown extra keys as allowed (forward compatible).

## 3. Comparison rules (per expense)

Inputs: the expense's declared amount and currency (from the submitted form), and the verifier
response for its receipt. Find in the repo how receipts are attached to expenses.

Outcome per expense, one of: **MATCHES**, **DIFFERS**, **COULD_NOT_VERIFY**, **PENDING**.

1. No usable response yet (queued, running, or retrying after 503): **PENDING**.
2. `status` is `unreadable`, `not_a_receipt` or `multiple_receipts`, a 400, or retries exhausted:
   **COULD_NOT_VERIFY**, with a reason code (`UNREADABLE`, `NOT_A_RECEIPT`, `MULTIPLE_RECEIPTS`,
   `INVALID_FILE`, `VERIFIER_UNAVAILABLE`).
3. `status` = `ok`:
   - **Currency** (`fields.currency`):
     - `none` (not printed): not compared, never an alarm;
     - `low`: not compared;
     - `high`/`medium` and different from the declared currency: **DIFFERS** on `currency`. Do not compare the amounts in that case; add the reason `CURRENCY_DIFFERS`.
   - **Total** (`fields.total_amount`):
     - `high`/`medium`: compare with the declared amount, tolerance **±0.01**. Equal means the field matches; different means **DIFFERS** on `total_amount` (declared vs. read);
     - `low` or `none`: **COULD_NOT_VERIFY** (`TOTAL_UNCERTAIN`).
   - **Date and document number**: store and show them as information. **They never change the
     outcome.** Put the date comparison behind a config flag, off by default.
   - The expense is **MATCHES** only if the total matches and the currency matches or was not compared.

Precedence: DIFFERS beats COULD_NOT_VERIFY beats MATCHES. Keep the per-field details (declared,
read, confidence, page, bbox) so the UI can show them.

**Report status:**
- **PENDING** while any expense is PENDING. Show "verification in progress"; no badge alarm.
- Else **NEEDS_A_LOOK** if any expense DIFFERS.
- Else **MATCHES** if all expenses MATCH.
- Else **PARTLY_VERIFIED**: nothing differs, but some receipts could not be verified. This is
  informational, **not** an alarm.

Things to check in the repo and report in your plan before deciding:
- An expense with **several receipt files**: verify each one. Compare the declared amount with the
  sum of the totals only if every total is `high`/`medium` and all currencies are the same or `null`;
  otherwise COULD_NOT_VERIFY (`MULTIPLE_FILES`).
- **Currency is set automatically by Spendle from the trip country**; the employee types the amount
  as printed on the receipt. So the declared amount is in the original currency: compare it
  directly with the receipt's total, with no conversion. Confirm this in the data model (where the
  currency is stored, per report or per expense, and whether the employee can override it). Any
  converted RON totals elsewhere in the report are not compared.
  - A receipt in a different currency than the country's (e.g. fuel bought in Romania, in RON,
    during a trip to Germany, in EUR): if the receipt prints its currency, this is a real
    **DIFFERS** on `currency`. Word it so the manager sees it isn't the employee's typo, e.g.
    "Receipt in **RON** · report currency **EUR** (set from the trip country)". Do not compare the
    amounts in that case.
  - If the receipt prints no currency (`none`, common on Romanian fuel receipts), the currency is
    not compared and only the amount is. Accept that this case can't be caught: the verifier never
    guesses a currency, and that is what keeps false alarms at zero.
- Expenses with **no receipt** (e.g. per diem): no verification, no status.

## 4. Background verification (backend)

- **Trigger:** when a report is submitted, enqueue one verification per receipt file. Do not slow
  down or change the submit request.
- **Cache by file content:** the verifier never sees the form, so its result depends only on the
  file. Key results by **SHA-256 of the file bytes**.
  - Same file again (re-submit, edits): reuse the result and just re-run the comparison.
  - Employee edits only the amount: re-compare, don't re-verify.
- **Persistence (DB-backed, survives restarts):** a verification job per file with:
  - file hash and receipt reference;
  - state (`QUEUED`, `RUNNING`, `DONE`, `FAILED`), attempts, `next_attempt_at`;
  - last HTTP status and error;
  - the **raw JSON response** (for audit and re-comparison), verifier `model`, timestamps.

  Store the comparison result per expense (outcome, reason codes, per-field details) and the report status.
- **Worker:** use the job/scheduling mechanism the backend already has; otherwise propose one.
  - Send **one request at a time** to scanning-tool (it processes one at a time anyway).
  - On **503**: retry honoring `Retry-After`, with backoff (e.g. 1, 5, 15, 30, 60 min, then hourly), for up to 24 h. After that, `FAILED`, giving COULD_NOT_VERIFY (`VERIFIER_UNAVAILABLE`).
  - On a **timeout or connection error**: same as 503.
  - On **400**: `DONE` with COULD_NOT_VERIFY (`INVALID_FILE`), no retry.
- **Config:** a feature flag (`verifier.enabled`, off by default), base URL (reuse the existing one),
  timeouts, the retry policy, and the date-comparison flag.
- **Logging:** IDs, states, durations and HTTP codes only. Do not log receipt contents or values
  beyond what the existing logging already does.
- **Do not change** the existing scanning flow, the proxy for `POST /api/web-report-stream`, or its
  SSE events.

## 5. Web (manager views)

- **Report list:** a status badge per report:
  - MATCHES: "Matches receipts" (green);
  - NEEDS_A_LOOK: "Needs a look" (amber);
  - PARTLY_VERIFIED: "Partly verified" (neutral/grey, with "N receipts could not be checked");
  - PENDING: "Verification in progress" (neutral).
- **Report detail:** an outcome badge per expense.
  - DIFFERS: show each differing field as "Declared **X** · on the receipt **Y**".
  - COULD_NOT_VERIFY: show a friendly reason for the reason code, e.g. "The receipt is hard to read", "This file has several receipts", "The receipt total could not be confirmed". Show the technical `notes` only in a details tooltip.
  - Show the date and document number read from the receipt as information, without an alarm.
- **Receipt viewer with highlight:** open the receipt page `page` and draw `bbox` as a rectangle.
  - Scale it to the displayed size: `x * displayedWidth / pages[page-1].width`, `y * displayedHeight / pages[page-1].height`. The verifier rendered the page itself, so always scale proportionally, for images and PDFs alike.
  - `bbox` = `null`: show the receipt without a highlight.
- No approve/reject or "auto" actions come from this feature. The manager decides with the existing actions.
- Follow the app's existing design system, i18n (add translations for every new text), roles and permissions. Who sees this: **managers/approvers**. Whether employees see it is an open question; ask, and default to managers only.

## 6. Tests

- **Unit tests for the comparison rules**, with JSON fixtures covering:
  - match;
  - amount differs;
  - total `low`;
  - currency `none` (no alarm);
  - currency differs;
  - `unreadable` / `not_a_receipt` / `multiple_receipts`;
  - tolerance ±0.01;
  - several files per expense;
  - a receipt in RON on a trip whose currency (from the country) is EUR, with the currency printed
    (DIFFERS on currency) and without it (amount only).
- **Worker tests with a mocked HTTP server:**
  - 200;
  - 503 with `Retry-After`, then 200;
  - repeated 503 until FAILED;
  - 400;
  - timeout;
  - cache hit on the same file hash;
  - an amount edit re-compares without re-verifying.
- **Report status aggregation tests.**
- **UI tests:** badge mapping and bbox scaling (including `null`).

## 7. Rollout notes

- `/api/verify-receipt` is on scanning-tool branch `feature/verify-receipt` and **not yet deployed**.
  Develop against a mock (the fixtures above), keep the feature flag off, and enable it only after
  scanning-tool is deployed.
- The UI must handle a long queue gracefully: PENDING is normal, not an error.

## Before writing code

Explore the repo and reply with a short plan covering:
1. How receipts, expenses, reports, the automatic currency (from the trip country) and manager
   roles are modeled. Point out anything that changes the rules in section 3.
2. How the backend calls scanning-tool today: client, config, base URL.
3. The job/scheduling mechanism you will use, and the DB changes (tables/migrations).
4. Files and modules you will touch, backend and web.
5. Open questions: several files per expense, whether the employee can override the automatic
   currency, whether employees see results.

Then wait for my confirmation.
