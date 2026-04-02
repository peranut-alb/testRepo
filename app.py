from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import tempfile
import typhoon_ocr
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import anyio
import pytesseract
from pytesseract import Output

_TESS_CMD = os.getenv("TESSERACT_CMD", "")
if _TESS_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESS_CMD

from google import genai
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageEnhance, UnidentifiedImageError
from pydantic import BaseModel, Field

BASE_DIR   = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

APP_TITLE   = "Bill OCR"
APP_VERSION = "8.0.0"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE    = MAX_FILE_SIZE_MB * 1024 * 1024

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY", "")

TESS_CONFIG_PRIMARY   = r"--oem 3 --psm 4  -l tha+eng"
TESS_CONFIG_SECONDARY = r"--oem 3 --psm 11 -l tha+eng"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(APP_TITLE.lower().replace(" ", "-"))

app = FastAPI(title=APP_TITLE, version=APP_VERSION)


# ─── Pydantic models ─────────────────────────────────────────────────────────

class BlockCoords(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    w: int = Field(..., ge=0)
    h: int = Field(..., ge=0)


class OCRBlock(BaseModel):
    text:       str
    confidence: float = Field(..., ge=0.0, le=1.0)
    coords:     BlockCoords


class LineItem(BaseModel):
    description: str
    quantity:    str | None = None
    unit_price:  str | None = None
    total:       str | None = None


class InvoiceData(BaseModel):
    vendor_name:    str | None = None
    vendor_address: str | None = None
    invoice_number: str | None = None
    invoice_date:   str | None = None
    due_date:       str | None = None
    subtotal:       str | None = None
    tax:            str | None = None
    discount:       str | None = None
    total:          str | None = None
    currency:       str | None = None
    payment_method: str | None = None
    notes:          str | None = None
    line_items: list[LineItem] = Field(default_factory=list)


class OCRResult(BaseModel):
    width:   int = Field(..., gt=0)
    height:  int = Field(..., gt=0)
    blocks:  list[OCRBlock]
    invoice: InvoiceData | None = None


# ─── Gemini client ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    logger.info("Gemini client ready. model=%s", GEMINI_MODEL)
    return genai.Client(api_key=GEMINI_API_KEY)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def validate_upload(file: UploadFile, data: bytes) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed size is {MAX_FILE_SIZE_MB} MB",
        )
    ct = (file.content_type or "").lower()
    if ct and not ct.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")


def load_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unsupported or corrupted image") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Unable to read image") from exc


def image_to_base64(image: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def preprocess_for_tesseract(image: Image.Image) -> Image.Image:
    target  = 1400
    longest = max(image.width, image.height)
    if longest < target:
        scale = target / longest
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.LANCZOS,
        )
    gray = image.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(2.2)
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)
    return gray


# ─── HTML Table Parser for Typhoon output ─────────────────────────────────────

class TableParser(HTMLParser):
    """Parse HTML tables from Typhoon OCR into structured rows."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: str = ""
        self._in_cell: bool = False

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = ""
        elif tag == "tr":
            self._current_row = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._in_cell = False
            self._current_row.append(self._current_cell.strip())
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell += data


def typhoon_markdown_to_structured(raw: str) -> str:
    """
    Convert Typhoon's raw output (mix of plain text + HTML tables)
    into a clean, structured text format that Gemini can reason about
    more easily.

    HTML tables become:
        ROW 0: col1 | col2 | col3
        ROW 1: col1 | col2

    Plain text lines pass through unchanged.
    """
    parts: list[str] = []
    segments = re.split(r"(<table>[\s\S]*?</table>)", raw)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if seg.lower().startswith("<table>"):
            parser = TableParser()
            parser.feed(seg)
            for i, row in enumerate(parser.rows):
                non_empty = [c for c in row if c]
                if non_empty:
                    parts.append(f"  ROW {i}: " + " | ".join(non_empty))
        else:
            for line in seg.split("\n"):
                line = line.strip()
                if line:
                    parts.append(line)

    return "\n".join(parts)


# ─── Stage 1: Tesseract → pixel-accurate line bboxes ─────────────────────────

def _extract_line_bboxes(
    data:    dict,
    scale_x: float,
    scale_y: float,
) -> list[dict[str, Any]]:
    line_map: dict[tuple, dict] = {}
    for i in range(len(data["text"])):
        if data["level"][i] != 5:
            continue
        conf = int(data["conf"][i])
        if conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x1 = data["left"][i]
        y1 = data["top"][i]
        x2 = x1 + data["width"][i]
        y2 = y1 + data["height"][i]
        if key not in line_map:
            line_map[key] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confs": [conf]}
        else:
            e = line_map[key]
            e["x1"] = min(e["x1"], x1)
            e["y1"] = min(e["y1"], y1)
            e["x2"] = max(e["x2"], x2)
            e["y2"] = max(e["y2"], y2)
            e["confs"].append(conf)

    bboxes = []
    for e in line_map.values():
        x = int(e["x1"] * scale_x)
        y = int(e["y1"] * scale_y)
        w = int((e["x2"] - e["x1"]) * scale_x)
        h = int((e["y2"] - e["y1"]) * scale_y)
        if w < 4 or h < 4:
            continue
        avg_conf = sum(e["confs"]) / len(e["confs"]) / 100.0
        bboxes.append({"x": x, "y": y, "w": w, "h": h, "conf": avg_conf})

    return bboxes


def run_tesseract_bboxes(image: Image.Image) -> list[dict[str, Any]]:
    """
    Run Tesseract PSM-4 (primary). If it fails or returns 0 bboxes,
    fall back to PSM-11. Returns ALL detected bboxes — no merging.
    Coordinates are in original image pixel space.
    """
    proc    = preprocess_for_tesseract(image)
    scale_x = image.width  / proc.width
    scale_y = image.height / proc.height

    # Primary: PSM 4
    bboxes: list[dict] = []
    try:
        d = pytesseract.image_to_data(proc, config=TESS_CONFIG_PRIMARY, output_type=Output.DICT)
        bboxes = _extract_line_bboxes(d, scale_x, scale_y)
        logger.info("Tesseract PSM-4: %d bboxes", len(bboxes))
    except Exception:
        logger.warning("Tesseract PSM-4 failed", exc_info=True)

    # Fallback: PSM 11 only if PSM 4 returned nothing
    if not bboxes:
        try:
            d = pytesseract.image_to_data(proc, config=TESS_CONFIG_SECONDARY, output_type=Output.DICT)
            bboxes = _extract_line_bboxes(d, scale_x, scale_y)
            logger.info("Tesseract PSM-11 (fallback): %d bboxes", len(bboxes))
        except Exception:
            logger.warning("Tesseract PSM-11 failed", exc_info=True)

    # Sort top-to-bottom, left-to-right
    bboxes.sort(key=lambda b: (b["y"], b["x"]))
    logger.info("Tesseract total: %d bboxes (all kept)", len(bboxes))
    return bboxes


# ─── Spatial grouping helper ──────────────────────────────────────────────────

def add_visual_rows(bboxes: list[dict]) -> list[dict]:
    """
    Tag each bbox with a visual_row ID based on Y-coordinate proximity.
    Bboxes on the same horizontal line get the same row ID.
    This helps Gemini understand which bboxes are on the same table row.
    """
    if not bboxes:
        return bboxes

    row_id = 0
    prev_center_y = -9999.0

    for b in bboxes:
        center_y = b["y"] + b["h"] / 2.0
        threshold = max(b["h"] * 0.5, 8)
        if abs(center_y - prev_center_y) > threshold:
            row_id += 1
        b["visual_row"] = row_id
        prev_center_y = center_y

    return bboxes


# ─── Stage 2: Typhoon → raw markdown string ───────────────────────────────────

def run_typhoon(image: Image.Image) -> str:
    """
    Return the FULL raw Typhoon output — HTML tables, markdown, and prose —
    as a single string, then convert to structured format for Gemini.
    """
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name, format="JPEG", quality=95)
            temp_path = tmp.name
        result = typhoon_ocr.ocr_document(temp_path, api_key=TYPHOON_API_KEY)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    raw: str
    if isinstance(result, str):
        raw = result.strip()
    else:
        # Structured output — flatten to a single string preserving table HTML
        parts: list[str] = []
        for page in result.get("pages", []):
            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    spans = [
                        s.get("text", "").strip()
                        for s in line.get("spans", [])
                        if s.get("text", "").strip()
                    ]
                    if spans:
                        parts.append(" ".join(spans))
        raw = "\n".join(parts)

    # Pre-parse HTML tables into structured ROW format
    structured = typhoon_markdown_to_structured(raw)
    logger.info("Typhoon structured: %d chars (from %d raw)", len(structured), len(raw))
    return structured


# ─── Stage 3a: Gemini assigns Typhoon text to each Tesseract bbox ─────────────

GEMINI_ASSIGN_PROMPT = """\
You are a precise document layout assistant for Thai/English receipts and invoices.

You will receive:
1. The original receipt image.
2. A JSON array of bounding boxes (bboxes) detected by Tesseract.
   Each entry has: {"index": N, "x": px, "y": px, "w": px, "h": px, "visual_row": N}
   Coordinates are in the original image pixel space (top-left origin).
   Bboxes with the same visual_row are on the same horizontal line in the document.
3. The full OCR text from TyphoonOCR (pre-parsed: plain lines + table rows in
   "ROW N: col1 | col2" format).

YOUR TASK
---------
For every bbox, look at the image to see what text is physically inside that
rectangle, then find the matching text from the TyphoonOCR output and assign it.

Rules:
- The IMAGE is ground truth for spatial position. Use it to locate what each
  bbox covers.
- The TYPHOONOCR TEXT is ground truth for character accuracy. Prefer it over
  guessing characters yourself.
- Bboxes with the same visual_row are on the same line — use table ROW data
  to assign the correct column text to each bbox on that row.
- For a bbox covering a table cell: extract only that cell's content.
- For a bbox covering a prose line: extract that line as plain text.
- For a bbox with no readable text: set "text" to "".
- Preserve all Thai and English characters exactly — do NOT translate.
- Output ONLY a valid JSON array, same length as the input bboxes array,
  in the SAME ORDER. No markdown fences, no extra explanation.
- IMPORTANT: You MUST return exactly one entry per input bbox. Never skip or
  merge bboxes.

Output schema:
[
  {"index": 0, "text": "...", "confidence": 0.95},
  {"index": 1, "text": "...", "confidence": 0.80},
  ...
]

confidence: 1.0 = certain match, 0.7 = probable, 0.4 = uncertain/empty.
"""

GEMINI_INVOICE_PROMPT = """\
You are an expert invoice/receipt data extractor.

You will receive the original image, the structured OCR text (assigned to
bounding boxes in reading order), and the full TyphoonOCR text
(which may contain table rows with richer structure).

Extract structured invoice fields. Return ONLY valid JSON — no markdown, no explanation.

Schema:
{
  "vendor_name": null,
  "vendor_address": null,
  "invoice_number": null,
  "invoice_date": null,
  "due_date": null,
  "subtotal": null,
  "tax": null,
  "discount": null,
  "total": null,
  "currency": null,
  "payment_method": null,
  "notes": null,
  "line_items": [
    {"description": "", "quantity": null, "unit_price": null, "total": null}
  ]
}

Rules:
- Prefer the TyphoonOCR text for monetary values and line items
  as it preserves table structure better.
- Preserve original language (Thai/English as-is).
- Keep monetary values as strings with original formatting (e.g. "19.00").
- Return null for fields not found.
- line_items is [] if none exist.
"""


def _empty_assignments(n: int) -> list[dict[str, Any]]:
    return [{"index": i, "text": "", "confidence": 0.4} for i in range(n)]


def call_gemini_assign_text(
    bboxes:           list[dict],
    typhoon_text:     str,
    image:            Image.Image,
) -> list[dict[str, Any]]:
    """
    Gemini looks at the image + ALL Tesseract bboxes + Typhoon text and
    returns a text assignment for every bbox.

    Returns list of {"index", "text", "confidence"} same length as bboxes.
    Falls back to empty assignments on any failure.
    """
    client  = get_gemini_client()
    img_b64 = image_to_base64(image)

    bbox_payload = [
        {
            "index": i,
            "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
            "visual_row": b.get("visual_row", 0),
        }
        for i, b in enumerate(bboxes)
    ]

    user_text = (
        GEMINI_ASSIGN_PROMPT
        + f"\n\nBOUNDING BOXES ({len(bbox_payload)} total):\n{json.dumps(bbox_payload, ensure_ascii=False)}"
        + f"\n\nTYPHOONOCR OUTPUT:\n{typhoon_text}"
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                    {"text": user_text},
                ],
            }],
        )

        raw   = re.sub(r"^```(?:json)?\s*|\s*```$", "", (response.text or "").strip())
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            logger.error("Gemini assign: no JSON array found")
            return _empty_assignments(len(bboxes))

        assignments: list[dict] = json.loads(match.group(0))

        # Key by index so Gemini can't accidentally reorder them
        indexed = {a["index"]: a for a in assignments if "index" in a}
        result  = []
        for i in range(len(bboxes)):
            a = indexed.get(i, {"index": i, "text": "", "confidence": 0.4})
            result.append({
                "index":      i,
                "text":       str(a.get("text") or "").strip(),
                "confidence": float(a.get("confidence", 0.4)),
            })

        assigned_count = sum(1 for r in result if r["text"])
        logger.info(
            "Gemini assigned text to %d / %d bboxes (all %d bboxes preserved)",
            assigned_count, len(bboxes), len(bboxes),
        )
        return result

    except Exception:
        logger.exception("Gemini text assignment failed")
        return _empty_assignments(len(bboxes))


def call_gemini_invoice(
    assigned_text: str,
    typhoon_text:  str,
    image:         Image.Image,
) -> InvoiceData | None:
    """Extract structured invoice fields using image + both text sources."""
    try:
        client = get_gemini_client()
    except RuntimeError as exc:
        logger.warning("Gemini unavailable: %s", exc)
        return None

    img_b64      = image_to_base64(image)
    combined_ctx = (
        "=== ASSIGNED OCR TEXT (reading order) ===\n"
        + assigned_text
        + "\n\n=== FULL TYPHOON TEXT (table structure preserved) ===\n"
        + typhoon_text
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                    {"text": GEMINI_INVOICE_PROMPT + f"\n\nOCR TEXT:\n{combined_ctx}"},
                ],
            }],
        )

        raw   = re.sub(r"^```(?:json)?\s*|\s*```$", "", (response.text or "").strip())
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            logger.error("Gemini invoice: no JSON object found")
            return None

        return InvoiceData(**json.loads(match.group(0)))

    except Exception:
        logger.exception("Gemini invoice call failed")
        return None


# ─── Main pipeline ────────────────────────────────────────────────────────────

def process_image(image: Image.Image) -> dict[str, Any]:
    """
    Three-stage pipeline:

    Stage 1 — Tesseract layout (PSM-4, fallback PSM-11)
        Produces N pixel-accurate bboxes. ALL bboxes are kept — no merging,
        no reduction. Language-agnostic geometry works even when Tesseract
        cannot read Thai characters.

    Stage 2 — TyphoonOCR structured text
        Returns the full output with HTML tables pre-parsed into
        structured "ROW N: col | col" format for cleaner Gemini input.

    Stage 3a — Gemini text assignment
        Gemini sees: image + ALL Tesseract bboxes (with visual_row hints)
        + Typhoon structured text. It assigns the correct text to each bbox.
        Result: N blocks with correct text and accurate coords.

    Stage 3b — Gemini invoice structuring
        Takes the assigned text + Typhoon text and extracts structured
        invoice fields.
    """

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    tess_bboxes: list[dict] = []
    try:
        tess_bboxes = run_tesseract_bboxes(image)
    except Exception:
        logger.warning("Tesseract detection failed", exc_info=True)

    # Add visual row grouping for Gemini
    tess_bboxes = add_visual_rows(tess_bboxes)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    typhoon_text = ""
    try:
        typhoon_text = run_typhoon(image)
        logger.info("Typhoon text: %d chars", len(typhoon_text))
    except Exception:
        logger.warning("Typhoon OCR failed", exc_info=True)

    # ── Stage 3a ─────────────────────────────────────────────────────────────
    blocks: list[dict[str, Any]] = []

    if tess_bboxes:
        assignments = call_gemini_assign_text(tess_bboxes, typhoon_text, image)
        for i, b in enumerate(tess_bboxes):
            a = assignments[i] if i < len(assignments) else {"text": "", "confidence": 0.4}
            if not a["text"].strip():
                continue  # skip bbox slots Gemini found to be empty
            blocks.append({
                "text":       a["text"],
                "confidence": round(min(max(float(a["confidence"]), 0.0), 1.0), 3),
                "coords":     {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]},
            })
        logger.info(
            "Pipeline: %d Tesseract bboxes → Gemini assigned → %d non-empty blocks",
            len(tess_bboxes), len(blocks),
        )

    elif typhoon_text:
        # No Tesseract bboxes — strip structured format and show as full-width bands
        logger.warning("Tesseract failed — full-width Typhoon bands")
        plain_lines = [
            re.sub(r"^\s*ROW \d+:\s*", "", ln).strip()
            for ln in typhoon_text.split("\n")
            if ln.strip()
        ]
        line_h = max(24, image.height // max(len(plain_lines), 1))
        blocks = [
            {
                "text":       line,
                "confidence": 0.8,
                "coords":     {"x": 0, "y": i * line_h, "w": image.width, "h": line_h},
            }
            for i, line in enumerate(plain_lines)
        ]

    else:
        logger.error("All OCR engines failed — empty result")

    # ── Stage 3b ─────────────────────────────────────────────────────────────
    assigned_text = "\n".join(b["text"] for b in blocks)
    invoice: InvoiceData | None = None
    if assigned_text or typhoon_text:
        try:
            invoice = call_gemini_invoice(assigned_text, typhoon_text, image)
        except Exception:
            logger.exception("Invoice extraction failed")

    if invoice is not None:
        invoice = invoice.model_dump()  # type: ignore[assignment]

    return {
        "width":   image.width,
        "height":  image.height,
        "blocks":  blocks,
        "invoice": invoice,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.get("/")
async def home() -> FileResponse:
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="Frontend file not found")
    return FileResponse(INDEX_FILE)


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status":       "ok",
        "pipeline":     "tesseract-bbox(psm4→psm11-fallback) → typhoon-structured → gemini-assign → gemini-invoice",
        "gemini_model": GEMINI_MODEL,
        "gemini_ready": bool(GEMINI_API_KEY),
    }


@app.post("/upload", response_model=OCRResult)
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    validate_upload(file, data)
    image = load_image(data)
    return await anyio.to_thread.run_sync(process_image, image)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
