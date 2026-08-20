from fpdf import FPDF
from fpdf.enums import XPos, YPos
import datetime
import os

class ForensicPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "VERICHAIN FORENSIC EXAMINATION DOSSIER", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, "Digital Evidence Court-Admissibility & Biometric Integrity Verification", border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()} | Cryptographically Verified Forensic Record | VeriChain Security Suite", align="C")


def generate_pdf(filename, file_hash, analyst, verdict, confidence, heatmap_path=None, faces_data=None, summary_note="", output_pdf="Evidence_Report.pdf"):
    pdf = ForensicPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Section 1: Chain of Custody
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "1. CHAIN OF CUSTODY & EVIDENCE RECORD", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(45, 5, "Evidence File:", border=0)
    pdf.cell(0, 5, str(filename), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.cell(45, 5, "SHA-256 Hash:", border=0)
    pdf.set_font("Courier", "", 8)
    pdf.cell(0, 5, str(file_hash), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(45, 5, "Lead Examiner:", border=0)
    pdf.cell(0, 5, str(analyst), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.cell(45, 5, "Forensic Timestamp:", border=0)
    pdf.cell(0, 5, datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # Section 2: Quantitative Verdict
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "2. QUANTITATIVE FORENSIC DETERMINATION", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, f"Verdict: {verdict}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Aggregate Confidence Level: {confidence * 100:.2f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if summary_note:
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, f"Examiner Note: {summary_note}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # Section 3: Subject-by-Subject Isolation
    if faces_data and len(faces_data) > 0:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, f"3. ISOLATED FACIAL SUBJECT BREAKDOWN ({len(faces_data)} Subjects Detected)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Table Header
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(20, 6, "Subject #", border=1, fill=True)
        pdf.cell(50, 6, "Bounding Box (X, Y, W, H)", border=1, fill=True)
        pdf.cell(60, 6, "Subject Determination", border=1, fill=True)
        pdf.cell(30, 6, "Manipulation Prob", border=1, fill=True)
        pdf.cell(30, 6, "Confidence", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

        # Table Rows
        pdf.set_font("Helvetica", "", 8)
        for face in faces_data:
            s_id = str(face.get("subject_id", "-"))
            bbox_str = str(face.get("bbox", "-"))
            s_verdict = str(face.get("verdict", "N/A"))
            s_manip = f"{face.get('manipulation_score', 0.0)*100:.1f}%"
            s_conf = f"{face.get('confidence', 0.0)*100:.1f}%"

            pdf.cell(20, 5, s_id, border=1)
            pdf.cell(50, 5, bbox_str, border=1)
            pdf.cell(60, 5, s_verdict, border=1)
            pdf.cell(30, 5, s_manip, border=1)
            pdf.cell(30, 5, s_conf, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)
    else:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "3. FACIAL SUBJECT ISOLATION", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, "No facial subjects detected in frame. Quantitative verdict derived from spectral/sensor noise.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    # Section 4: Heatmap & Visual Evidence Overlay
    if heatmap_path and os.path.exists(heatmap_path):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "4. LOCALIZED ARTIFACT & BIOMETRIC OVERLAY", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        current_y = pdf.get_y()
        # If space is tight, add page
        if current_y > 190:
            pdf.add_page()
            current_y = pdf.get_y()
        pdf.image(heatmap_path, x=35, y=current_y + 1, w=140)

    pdf.output(output_pdf)
    return output_pdf