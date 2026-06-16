import gradio as gr
import base64
import os
import pandas as pd
import numpy as np
import cv2
import faiss
import pickle
import re
import json
from ultralytics import YOLO
from PIL import Image
from openai import AzureOpenAI

# =========================
# ✅ CONFIG (IMPORTANT)
# =========================
import os

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")


CURRENT_FILE_PATH = None
# ✅ LOAD MODEL


yolo_model = YOLO("best.pt")


# ✅ LOAD CLASSES
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ✅ OPENAI CONFIG
AZURE_OPENAI_ENDPOINT = "https://openai-sqdc.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_GPT_MODEL = "gpt-4o"
AZURE_EMBEDDING_MODEL = "embed-large"

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# =========================
# ✅ LOAD LOGOS (LOCAL PATHS)
# =========================
def load_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

abb_logo = load_logo("abb_logo.png")
techm_logo = load_logo("techm_logo.png")


# =========================
# ✅ LOGIN FUNCTION
# =========================
def check_login(username, password):
    if username == "admin" and password == "admin123":
        return gr.update(visible=False), gr.update(visible=True), ""
    return gr.update(visible=True), gr.update(visible=False), "❌ Invalid credentials"


# =========================
# ✅ YOUR EXISTING FUNCTIONS (PASTE HERE)
# =========================
# 🔽 VERY IMPORTANT 🔽
# Paste all of these from your notebook:
#
# - build_rag_index
def build_rag_index(
    excel_path
):

    print(
        "Reading Excel..."
    )

    xls = pd.ExcelFile(
        excel_path
    )

    docs = []

    for sheet in xls.sheet_names:

        try:

            df = pd.read_excel(
                excel_path,
                sheet_name=sheet
            )

            df = df.fillna("")

            for _, row in df.iterrows():

                text = " | ".join(
                    [
                        f"{col}: {row[col]}"
                        for col in df.columns
                        if str(row[col]).strip() != ""
                    ]
                )

                if len(text) > 10:

                    docs.append(text)

        except Exception as e:

            print(
                f"Skipping sheet {sheet}: {e}"
            )

    print(
        f"Documents: {len(docs)}"
    )

    # Batch embeddings
    response = client.embeddings.create(
        model=AZURE_EMBEDDING_MODEL,
        input=docs
    )

    embeddings = np.array(
        [
            item.embedding
            for item in response.data
        ],
        dtype="float32"
    )

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dim
    )

    index.add(
        embeddings
    )

    return index, docs
# - load_pricing
def load_pricing(excel_path):

    pricing = {}

    try:
        xls = pd.ExcelFile(excel_path)

        for sheet in xls.sheet_names:

            df = pd.read_excel(excel_path, sheet_name=sheet).fillna("")

            # ---- detect module column ----
            module_col = None
            best_score = 0

            for col in df.columns:
                vals = df[col].astype(str)

                score = sum(
                    any(c.isalpha() for c in v) and any(c.isdigit() for c in v)
                    for v in vals
                )

                if score > best_score:
                    best_score = score
                    module_col = col

            if module_col is None or best_score < 2:
                continue

            # ---- detect numeric columns ----
            numeric_cols = []

            for col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()

                if len(vals) > 3:
                    numeric_cols.append((col, vals.mean()))

            if len(numeric_cols) < 2:
                continue

            # ---- pick middle column = UNIT COST ----
            numeric_cols = sorted(numeric_cols, key=lambda x: x[1])
            price_col = numeric_cols[len(numeric_cols)//2][0]

            # ---- extract pricing ----
            valid_count = 0

            for _, row in df.iterrows():

                module = str(row[module_col]).strip().upper()
                price = row[price_col]

                if (
                    module
                    and isinstance(price, (int, float))
                    and price > 0
                    and any(c.isalpha() for c in module)
                    and any(c.isdigit() for c in module)
                    and not module.startswith("A.")
                ):
                    pricing[module.split()[0]] = float(price)
                    valid_count += 1

            if valid_count > 5:
                print(f"✅ Using sheet: {sheet} ({valid_count} rows)")

    except Exception as e:
        print("Pricing read error:", e)

    print("✅ Total pricing loaded:", len(pricing))
    return pricing
# - generate_proposal_pdf
def generate_proposal_pdf(bom, excel_path):

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch

    styles = getSampleStyleSheet()

    # ✅ Professional text style
    cell_style = ParagraphStyle(
        name='Cell',
        fontSize=7.5,
        leading=9,
        wordWrap='CJK'
    )

    header_style = ParagraphStyle(
        name='Header',
        fontSize=8,
        leading=10,
        alignment=1  # center
    )

    file_name = "ABB_Proposal.pdf"

    pricing = load_pricing(excel_path)

    new_bom = []
    total_cost = 0

    for row in bom:

        module = str(row[2]).upper().strip()
        qty = row[4]

        unit_price = pricing.get(module, 0)
        total_price = unit_price * qty

        total_cost += total_price

        new_bom.append(row + [unit_price, total_price])

    # ✅ Page setup like real proposal
    doc = SimpleDocTemplate(
        file_name,
        pagesize=A4,
        leftMargin=25,
        rightMargin=25,
        topMargin=30,
        bottomMargin=25
    )

    elements = []

    # ✅ TITLE (clean like ABB doc)
    elements.append(Paragraph("<b>Bill of Material</b>", styles['Heading1']))
    elements.append(Spacer(1, 16))

    # ✅ Header row
    table_data = [[
        Paragraph("<b>Detected</b>", header_style),
        Paragraph("<b>Exist</b>", header_style),
        Paragraph("<b>Replacement</b>", header_style),
        Paragraph("<b>Qty/P</b>", header_style),
        Paragraph("<b>Tot Qty</b>", header_style),
        Paragraph("<b>Description</b>", header_style),
        Paragraph("<b>Unit $</b>", header_style),
        Paragraph("<b>Total $</b>", header_style)
    ]]

    # ✅ Wrap function (better control than fixed slicing)
    def wrap(text):
        text = str(text)
        if len(text) < 70:
            return text
        return "<br/>".join([text[i:i+65] for i in range(0, len(text), 65)])

    # ✅ Add rows
    for row in new_bom:

        table_data.append([
            Paragraph(wrap(row[0]), cell_style),
            Paragraph(wrap(row[1]), cell_style),
            Paragraph(wrap(row[2]), cell_style),
            Paragraph(wrap(row[3]), cell_style),
            Paragraph(wrap(row[4]), cell_style),
            Paragraph(wrap(row[5]), cell_style),
            Paragraph(str(row[6]), cell_style),
            Paragraph(str(row[7]), cell_style),
        ])

    # ✅ BALANCED WIDTHS (THIS FIXES EVERYTHING)
    col_widths = [
        0.9*inch,   # detected
        0.6*inch,   # exist
        1.0*inch,   # replacement
        0.6*inch,   # qty
        0.7*inch,   # total qty
        2.8*inch,   # ✅ description (main space)
        0.8*inch,   # unit
        0.8*inch    # total
    ]

    table = Table(
        table_data,
        colWidths=col_widths,
        repeatRows=1
    )

    # ✅ PROFESSIONAL TABLE STYLE
    table.setStyle(TableStyle([

        # Header styling
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4F4F4F")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),

        # Grid (light like ABB)
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),

        # Alignment
        ('VALIGN', (0,0), (-1,-1), 'TOP'),

        # Padding (IMPORTANT)
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),

        # Bold header
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),

    ]))

    elements.append(table)

    elements.append(Spacer(1, 16))

    # ✅ Total cost clean display
    elements.append(
        Paragraph(f"<b>Total Cost: USD {round(total_cost,2)}</b>", styles['Normal'])
    )

    doc.build(elements)

    return file_name
# - full_pipeline
def full_pipeline(

    image,

    edits
):

    detected_panels, annotated = detect_panels(
        image
    )

    panels = apply_user_edits(

        detected_panels,

        edits
    )

    panel_text = format_panels(
        panels
    )

    bom = generate_bom(
        panels
    )

    return (

        panel_text,

        bom,

        annotated
    )
# - process_image
def process_image(image, edits):

    if image is None:
        return None, "Upload Image", [], None

    panel_text, bom, img = full_pipeline(image, edits)

    pdf_path = generate_proposal_pdf(
    bom,
    CURRENT_FILE_PATH)

    return img, panel_text, bom, pdf_path
# - create_knowledge_base
def create_knowledge_base(excel_file):
    if excel_file is None:
        return " Please upload an Excel file"

    try:
        import shutil
        import os

        file_path = excel_file.name
        global CURRENT_FILE_PATH

        save_path = os.path.basename(file_path)
        shutil.copy(file_path, save_path)

        CURRENT_FILE_PATH = save_path

        global index, docs
        index, docs = build_rag_index(save_path)

        return f" Mapping loaded successfully ({len(docs)} rows indexed)"

    except Exception as e:
        return f" Error: {str(e)}"
# - detect_panels
def detect_panels(image):

    # Convert PIL image to numpy array
    img = np.array(image)

    # Run YOLO model
    results = yolo_model(img)[0]

    detections = []

    # Extract detections
    if results.boxes is not None:

        for i, box in enumerate(results.boxes.xyxy):

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box)

            # Class ID
            cls_id = int(results.boxes.cls[i])

            # Safety check
            if cls_id >= len(classes):
                continue

            # Class label
            label = classes[cls_id]

            detections.append((
                x1,
                label,
                (x1, y1, x2, y2)
            ))

    # Sort from left to right
    detections = sorted(detections, key=lambda x: x[0])

    panels = []

    # Draw boxes + labels
    for idx, (x1, label, box) in enumerate(detections):

        panels.append(label)

        x1, y1, x2, y2 = box

        # Draw rectangle
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Add label text
        cv2.putText(
            img,
            f"[{idx}] {label}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    return panels, img

# - EVERYTHING related to your logic
#
def format_panels(panels):
    return "\n".join([f"{i}: {p}" for i, p in enumerate(panels)])
def apply_user_edits(detected, edited_text):
    panels = detected.copy()

    if not edited_text:
        return panels

    edits = edited_text.split(",")

    for e in edits:
        e = e.strip()

        if e.startswith("+"):
            panels.append(e[1:].strip())

        elif e.startswith("-"):
            try:
                idx = int(e[1:].strip())
                if 0 <= idx < len(panels):
                    panels.pop(idx)
            except:
                pass

        elif ":" in e:
            try:
                idx, val = e.split(":", 1)
                panels[int(idx.strip())] = val.strip()
            except:
                pass

    return panels
def generate_bom(panels):

    counts = {}
    for p in panels:
        counts[p] = counts.get(p, 0) + 1

    unique_panels = list(counts.keys())

    mappings = map_panels(unique_panels)

    pricing = load_pricing(CURRENT_FILE_PATH) if CURRENT_FILE_PATH else {}

    bom = []

    for row in mappings:

        detected = row.get("detected_panel", "")

        qty_raw = str(row.get("quantity_per_panel", "1"))
        match = re.search(r'\d+', qty_raw)
        qty_per_panel = int(match.group()) if match else 1

        total_panels = counts.get(detected, 1)

        total_qty = qty_per_panel * total_panels

        replacement = row.get("replacement_module", "Not Found")

        unit_price = pricing.get(replacement.upper(), 0)
        total_price = unit_price * total_qty

        bom.append([
            detected,
            total_panels,
            replacement,
            qty_per_panel,
            total_qty,
            row.get("description", ""),
            unit_price,
            total_price
        ])

    return bom
def retrieve_rows(panel_name, top_k=5):

    global index, docs

    if index is None:
        return []

    query_embedding = client.embeddings.create(
        model=AZURE_EMBEDDING_MODEL,
        input=panel_name
    )

    emb = np.array([query_embedding.data[0].embedding], dtype="float32")

    D, I = index.search(emb, top_k)

    return [docs[i] for i in I[0]]
def azure_mapping(panel):

    rows = retrieve_rows(panel)

    context = "\n\n".join(rows)

    prompt = f"""
You are an ABB migration expert.

Detected Module:
{panel}

Mapping Data:
{context}

Return JSON:
{{
    "detected_panel":"",
    "replacement_module":"",
    "quantity_per_panel":"",
    "description":""
}}
"""

    response = client.chat.completions.create(
        model=AZURE_GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
def extract_json(text):

    match = re.search(r'\{.*\}', text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return {
        "detected_panel":"Unknown",
        "replacement_module":"Not Found",
        "quantity_per_panel":1,
        "description":"N/A"
    }
mapping_cache = {}

def get_mapping(panel):

    if panel in mapping_cache:
        return mapping_cache[panel]

    response = azure_mapping(panel)

    data = extract_json(response)

    data["detected_panel"] = panel

    mapping_cache[panel] = data

    return data
def map_panels(panels):

    results = []

    for panel in panels:
        try:
            results.append(get_mapping(panel))
        except Exception as e:
            results.append({
                "detected_panel": panel,
                "replacement_module": "Error",
                "quantity_per_panel": 1,
                "description": str(e)
            })

    return results
# 👉 DO NOT paste:
# - any gradio UI code
# - any `files.upload()`
# - any `/content/...` paths (replace with local file names)

# =========================
# ✅ CSS
# =========================
css = """
.gradio-container {
    background-color: #0b0b0b;
}

/* LEFT LOGOS */
.logo-container {
    display:flex;
    flex-direction:column;
    gap:40px;
    padding:100px 60px;
}

/* LOGIN BOX */
.login-card {
    width: 350px;
    margin-top: 120px;
    padding: 20px;
    border-radius: 12px;
    background-color: #1a1a1a;
}

/* TEXTBOX IMPROVEMENTS */
.gr-textbox textarea {
    overflow-y: auto !important;
    font-size: 14px !important;
    padding: 10px !important;
}

/* REMOVE EXTRA SPACE */
.login-card .gr-markdown:empty {
    display:none !important;
}
"""

# =========================
# ✅ UI
# =========================
with gr.Blocks(css=css) as app:

    # ✅ LOGIN
    with gr.Column(visible=True) as login_block:

        with gr.Row():

            with gr.Column(scale=1):
                gr.HTML(f"""
                        <div class="logo-container">
                        <img src="data:image/png;base64,{abb_logo}" style="height:120px;">
    <img src="data:image/png;base64,{techm_logo}" style="height:120px;">
</div>
""")

            with gr.Column(scale=1):

                with gr.Column(elem_classes="login-card"):

                    username = gr.Textbox(label="Login ID")
                    password = gr.Textbox(type="password", label="Password")

                    login_btn = gr.Button("Login")

                    login_status = gr.Markdown("")

    # ✅ MAIN APP
    with gr.Column(visible=False) as main_block:

        gr.Markdown("# ABB Migration BOM Generator")

        with gr.Row():

            with gr.Column(scale=1):

                excel_input = gr.File(label="Migration Mapping File")
                build_btn = gr.Button("Load Mapping")

                kb_status = gr.Markdown("🔄 Waiting for file upload...")

            with gr.Column(scale=4):

                image_input = gr.Image(type="pil", height=450)
                run_btn = gr.Button("Generate BOM")

        image_output = gr.Image(height=500)

        detected_text = gr.Textbox(label="Detected Modules", lines=12)

        edits = gr.Textbox(
            label="Manual Verification",
            lines=8
        )

        bom_output = gr.Dataframe(
            headers=[
                "Detected Panel",
                "Existing Panels",
                "Replacement Module",
                "Qty/Panel",
                "Total Qty",
                "Description",
                "Unit Price",
                "Total Price"
            ]
        )

        pdf_output = gr.File(label="Download Proposal")

    # ✅ EVENTS
    login_btn.click(
        check_login,
        inputs=[username, password],
        outputs=[login_block, main_block, login_status]
    )

    build_btn.click(
        create_knowledge_base,
        inputs=excel_input,
        outputs=kb_status
    )

    run_btn.click(
        process_image,
        inputs=[image_input, edits],
        outputs=[
            image_output,
            detected_text,
            bom_output,
            pdf_output
        ]
    )

# =========================
# ✅ RUN (RENDER READY)
# =========================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)

