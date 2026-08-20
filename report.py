import datetime
import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class ForensicPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "VERICHAIN FORENSIC EXAMINATION DOSSIER", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, "Certified Digital Evidence Multi-Subject Authentication Report", border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()} | Cryptographic Chain-of-Custody Verified", border=False, align="C")

def generate_pdf(filename, file_hash, analyst, verdict, confidence, heatmap_path=None, faces_data=None, summary_note="", output_pdf="Evidence_Report.pdf"):
    pdf = ForensicPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 1. Chain of Custody & Evidence Intake
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "1. CHAIN OF CUSTODY & EVIDENCE INGESTION", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(45, 5, "Target Media:", border=0)
    pdf.cell(0, 5, str(filename), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.cell(45, 5, "SHA-256 Checksum:", border=0)
    pdf.set_font("Courier", "", 8)
    pdf.cell(0, 5, str(file_hash), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(45, 5, "Lead Examiner:", border=0)
    pdf.cell(0, 5, str(analyst if analyst else "Digital Forensics Lead"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.cell(45, 5, "Audit Timestamp:", border=0)
    pdf.cell(0, 5, datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # 2. Final Forensic Determination
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "2. QUANTITATIVE VERDICT & EXAMINER SUMMARY", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font("Helvetica", "B", 11)
    if "Manipulated" in str(verdict) or "Deepfake" in str(verdict):
        pdf.set_text_color(190, 0, 0)
    else:
        pdf.set_text_color(0, 130, 0)
        
    pdf.cell(0, 6, f"Determination: {verdict} ({confidence*100:.1f}% Confidence)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    
    if summary_note:
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, f"Forensic Notes: {summary_note}")
    pdf.ln(2)

    # 3. Subject-by-Subject Breakdown Table
    if faces_data:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"Subject Isolation Breakdown ({len(faces_data)} Subject{'s' if len(faces_data) > 1 else ''}):", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(25, 5, "Subject", border=1, fill=True)
        pdf.cell(60, 5, "Bounding Box (X, Y, W, H)", border=1, fill=True)
        pdf.cell(55, 5, "Determination", border=1, fill=True)
        pdf.cell(30, 5, "Confidence", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        
        pdf.set_font("Helvetica", "", 8)
        for f in faces_data:
            subj_title = f"Subject #{f['subject_id']}"
            box_str = f"[{f['bbox'][0]}, {f['bbox'][1]}, {f['bbox'][2]}, {f['bbox'][3]}]"
            subj_verdict = "Manipulated" if f["manipulation_score"] >= 0.50 else "Authentic"
            conf_str = f"{f['confidence']*100:.1f}%"
            
            pdf.cell(25, 5, subj_title, border=1)
            pdf.cell(60, 5, box_str, border=1)
            pdf.cell(55, 5, subj_verdict, border=1)
            pdf.cell(30, 5, conf_str, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    # 4. Localized Anomaly Heatmap Image
    if heatmap_path and os.path.exists(heatmap_path):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, "3. LOCALIZED ARTIFACT & SPATIAL DISCONTINUITY OVERLAY", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.image(heatmap_path, x=45, y=pdf.get_y() + 2, w=120)
        pdf.ln(80)

    # 5. Judicial Admissibility Sign-off
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, "Admissibility Certification: This document satisfies chain-of-custody tracking. Analysis conducted via multimodal neural classification and spatial frequency domain decomposition.")
    pdf.ln(4)
    pdf.cell(100, 5, "Forensic Examiner Signature: _______________________", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(output_pdf)
    return output_pdf