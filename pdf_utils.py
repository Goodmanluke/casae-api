"""
PDF generation utilities for the Casae CMA API.

This module provides a helper function to create a simple PDF report
for a Comparative Market Analysis (CMA) run.  The PDF includes the
subject property estimate and a table of comparable properties.

The PDF is built entirely in memory using reportlab and returned as
raw bytes.  It does not write any files to disk and therefore can be
safely used in serverless environments.

Example usage:

    from pdf_utils import create_cma_pdf
    pdf_bytes = create_cma_pdf(run_id, cma_run)
    # send pdf_bytes in a HTTP response or upload to storage

"""

from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO


def create_cma_pdf(cma_run_id: str, cma_run: Dict[str, Any]) -> bytes:
    """Build a PDF report for a CMA run.

    Parameters
    ----------
    cma_run_id: str
        Unique identifier for the CMA run.  Used in the report header.
    cma_run: Dict[str, Any]
        The stored CMA run dictionary containing subject, estimate and comps.

    Returns
    -------
    bytes
        The generated PDF as a bytes object.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Comparative Market Analysis Report", styles["Title"]))
    elements.append(Paragraph(f"CMA Run ID: {cma_run_id}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Subject details and estimate
    subject = cma_run.get("subject")
    estimate = cma_run.get("estimate", 0.0)
    address = getattr(subject, "address", "Unknown")
    elements.append(Paragraph(f"Subject Property: {address}", styles["Heading2"]))
    elements.append(Paragraph(f"Estimated Value: ${estimate:,.0f}", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    # Table of comparable properties
    comps = cma_run.get("comps", [])
    data = [
        [
            "Address",
            "Price",
            "Beds",
            "Baths",
            "Sqft",
            "Year Built",
        ]
    ]
    for comp, _score in comps:
        # Each comp is an instance of comps_scoring.Property
        data.append([
            getattr(comp, "address", ""),
            f"${(comp.raw_price or 0.0):,.0f}",
            comp.beds or "",
            comp.baths or "",
            comp.living_sqft or "",
            comp.year_built or "",
        ])

    if len(data) > 1:
        table = Table(data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        elements.append(Paragraph("Comparable Properties", styles["Heading3"]))
        elements.append(table)
    else:
        elements.append(Paragraph("No comparable properties found.", styles["Normal"]))

    # Build the PDF
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes