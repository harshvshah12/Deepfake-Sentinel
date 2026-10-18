"""Multi-page forensic audit PDF generator with high-definition Playwright rendering and ReportLab fallback."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import uuid
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from backend.report_generator import b64_to_rl_image
from backend.pdf_html_generator import generate_report_html


def _font() -> str:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    arial = os.path.join(windir, "Fonts", "arial.ttf")
    if os.path.exists(arial) and "SentinelArial" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("SentinelArial", arial))
    if "SentinelArial" in pdfmetrics.getRegisteredFontNames():
        return "SentinelArial"
    return "Helvetica"


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    safe = "" if text is None else str(text)
    safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)


def generate_scan_pdf(scan: Dict[str, Any]) -> bytes:
    """
    Renders an epitome-of-creative-graphical-visualizations 3-page forensic dossier via Playwright.
    Gracefully falls back to pure Python ReportLab if Node.js/Playwright is unavailable.
    """
    try:
        html_content = generate_report_html(scan)
        temp_id = uuid.uuid4().hex[:12]
        temp_html = os.path.join(tempfile.gettempdir(), f"sentinel_{temp_id}.html")
        temp_pdf = os.path.join(tempfile.gettempdir(), f"sentinel_{temp_id}.pdf")

        with open(temp_html, "w", encoding="utf-8") as f:
            f.write(html_content)

        render_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "render_pdf.js"))
        cmd = ["node", render_script, temp_html, temp_pdf]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=40)

        if proc.returncode == 0 and os.path.exists(temp_pdf) and os.path.getsize(temp_pdf) > 2000:
            with open(temp_pdf, "rb") as f:
                pdf_data = f.read()
            try:
                os.remove(temp_html)
                os.remove(temp_pdf)
            except Exception:
                pass
            return pdf_data
        else:
            print(f"[PDF WARN] Playwright render exited with code {proc.returncode}: {proc.stderr}. Falling back to ReportLab.")
    except Exception as exc:
        print(f"[PDF WARN] Playwright PDF generation exception: {exc}. Falling back to ReportLab.")

    return _generate_reportlab_pdf(scan)


def _generate_reportlab_pdf(scan: Dict[str, Any]) -> bytes:
    font = _font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName=font, fontSize=9, leading=12, textColor=colors.HexColor("#1F2937"))
    h = ParagraphStyle("h", parent=styles["Heading2"], fontName=font, fontSize=13, leading=16, textColor=colors.HexColor("#0F172A"), spaceAfter=6)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#475569"))

    decision = scan.get("final_decision") or {}
    meta = scan.get("input_metadata") or {}
    pre = scan.get("preprocessing_results") or {}
    models = [m for m in (scan.get("model_results") or []) if isinstance(m, dict)]
    aux = scan.get("auxiliary_evidence") or {}
    prov = scan.get("provenance_results") or {}
    story = []

    story.append(_p("Deepfake Sentinel forensic report", h))
    story.append(_p(f"Scan {scan.get('scan_id')}  |  {scan.get('created_at', '')}", small))
    story.append(Spacer(1, 8))
    story.append(_p("1. Executive summary", h))
    thumb = b64_to_rl_image(scan.get("face_b64"), 160, 160)
    if thumb:
        story.append(thumb)
    else:
        story.append(_p("Original crop: NOT AVAILABLE", body))
    rows = [
        ["File", meta.get("filename")],
        ["SHA-256", meta.get("sha256")],
        ["Size", f"{meta.get('width')} x {meta.get('height')} px, {meta.get('size_bytes')} bytes"],
        ["Conclusion", decision.get("prediction")],
        ["Neural fake-score", decision.get("fake_probability")],
        ["Neural real-score", decision.get("real_probability")],
        ["Score meaning", decision.get("score_semantics")],
        ["Uncertainty", (scan.get("uncertainty") or {}).get("level")],
    ]
    story.append(_table(rows, font))
    for warning in (scan.get("warnings") or [])[:8]:
        story.append(_p(f"Warning: {warning}", small))
    story.append(_p(decision.get("decision_notes") or "", body))

    story.append(PageBreak())
    story.append(_p("2. Processing audit", h))
    align = pre.get("alignment") or {}
    story.append(_p(
        f"Face detected: {pre.get('face_detected')}. Box: {pre.get('bbox')}. "
        f"Landmarks: {len(pre.get('landmarks') or [])}. Alignment: {align.get('status')} ({align.get('reason')}).",
        body,
    ))
    story.append(_p(f"Quality: {pre.get('quality')}", body))
    story.append(_p(f"Timings (ms): {scan.get('timings')}", body))
    for label, key in (("Face crop", "face_b64"), ("FFT", "fft_b64"), ("Grad-CAM", "heatmap_b64")):
        img = b64_to_rl_image(scan.get(key), 140, 140)
        story.append(_p(label if img else f"{label}: NOT AVAILABLE", small))
        if img:
            story.append(img)

    story.append(PageBreak())
    story.append(_p("3. Model-by-model analysis", h))
    if not models:
        story.append(_p("NOT AVAILABLE", body))
    for model in models:
        name = "EfficientNet-B3" if "logit" in model or model.get("checkpoint_threshold") is not None else "ViT-B/16"
        if model.get("fake_index") is not None:
            name = "ViT-B/16"
        story.append(_p(
            f"{name}: available={model.get('available')} score={model.get('fake_probability')} "
            f"latency_ms={model.get('latency_ms')} semantics={model.get('output_semantics')} error={model.get('error')}",
            body,
        ))
    story.append(_p(f"Fusion: {decision.get('method')}", body))
    story.append(_p(f"Weights: {decision.get('weights_used')}", body))
    if scan.get("heatmap_b64"):
        story.append(_p("Grad-CAM was produced from the EfficientNet feature map. It is a localization aid, not a validated explanation.", small))
    else:
        story.append(_p("Grad-CAM: UNAVAILABLE", body))

    story.append(PageBreak())
    story.append(_p("4. Forensic evidence", h))
    spectral = aux.get("spectral") or {}
    color = aux.get("color") or {}
    c2pa = prov.get("c2pa") or scan.get("c2pa") or {}
    synth = prov.get("synthid") or scan.get("synthid") or {}
    visible = prov.get("visible_watermark") or scan.get("visual_watermark") or {}
    exif = prov.get("exif") or scan.get("provenance") or {}
    story.append(_p(f"FFT high-frequency ratio: {spectral.get('high_freq_ratio')}. {spectral.get('limitation')}", body))
    fft_img = b64_to_rl_image(scan.get("fft_b64"), 160, 160)
    if fft_img:
        story.append(fft_img)
    story.append(_p(f"Color: {color.get('skin_chrominance')}. {color.get('hypothesis')}", body))
    story.append(_p(f"Visible watermark: detected={visible.get('visual_watermark_detected')} regions={visible.get('regions')} OCR={visible.get('ocr_status')}", body))
    story.append(_p(f"Invisible watermark / SynthID: {synth.get('status')}. {synth.get('reason')}", body))
    story.append(_p(f"C2PA: {c2pa.get('validation_status')}. {c2pa.get('note') or c2pa.get('limitation')}", body))
    story.append(_p(f"EXIF software: {exif.get('software')} AI tag: {exif.get('ai_software_detected')} camera: {exif.get('make')} {exif.get('model')}", body))

    story.append(PageBreak())
    story.append(_p("5. Integrated conclusion", h))
    story.append(_p(decision.get("decision_notes") or "NOT AVAILABLE", body))
    story.append(_p(f"Disagreement: {decision.get('discordance_level')} ({decision.get('discordance_value')})", body))
    story.append(_p(f"Uncertainty: {scan.get('uncertainty')}", body))
    story.append(_p(
        "Next check: compare the file with the publisher, inspect a C2PA manifest with a trust-list verifier, "
        "and do not treat this score as legal proof.",
        body,
    ))
    story.append(Spacer(1, 8))
    story.append(_p("Technical appendix", h))
    story.append(_p(f"Status: {scan.get('status')}. Errors: {scan.get('errors') or 'none'}.", small))
    story.append(_p(scan.get("retention") or "In-memory scan only.", small))
    for row in (scan.get("explainability") or {}).get("ledger") or []:
        if row.get("validated_contribution"):
            story.append(_p(f"Contributes: {row.get('detector')} = {row.get('result')}", small))
        else:
            story.append(_p(f"Does not change the score: {row.get('measurement')} = {row.get('result')}", small))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=48, rightMargin=48, topMargin=48, bottomMargin=48)
    doc.pageCompression = 0
    story.append(Paragraph("SYNTHID_STATUS=" + str((scan.get("provenance_results") or {}).get("synthid", {}).get("status") or (scan.get("synthid") or {}).get("status")), ParagraphStyle("ascii", fontName="Helvetica", fontSize=8)))
    doc.build(story)
    return buffer.getvalue()


def _table(rows, font: str):
    data = [[Paragraph(str(a or ""), ParagraphStyle("c", fontName=font, fontSize=8)),
             Paragraph(str(b if b is not None else "NOT AVAILABLE"), ParagraphStyle("c2", fontName=font, fontSize=8))]
            for a, b in rows]
    table = Table(data, colWidths=[140, 360])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table
