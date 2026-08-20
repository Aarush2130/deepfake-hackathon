from fpdf import FPDF
from fpdf.enums import XPos, YPos
import datetime

class ForensicPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "VERICHAIN FORENSIC EXAMINATION DOSSIER", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, "Digital Evidence Court-Admissibility Verification", border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(5)

def generate_pdf(filename, file_hash, analyst, verdict, confidence, heatmap_path, output_pdf="Evidence_Report.pdf"):
    pdf = ForensicPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "1. CHAIN OF CUSTODY & IDENTIFICATION", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(45, 6, "Media File:", border=0)
    pdf.cell(0, 6, str(filename), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(45, 6, "SHA-256 Hash:", border=0)
    pdf.set_font("Courier", "", 9)
    pdf.cell(0, 6, str(file_hash), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(45, 6, "Lead Examiner:", border=0)
    pdf.cell(0, 6, str(analyst), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(45, 6, "Timestamp:", border=0)
    pdf.cell(0, 6, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. QUANTITATIVE VERDICT", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, f"Determination: {verdict} ({confidence*100:.2f}% Confidence)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    if heatmap_path:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "3. LOCALIZED ARTIFACT HEATMAP", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(heatmap_path, x=40, y=pdf.get_y() + 2, w=130)

    pdf.output(output_pdf)
    return output_pdf