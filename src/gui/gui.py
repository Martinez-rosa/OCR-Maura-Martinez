from flask import Flask, request, jsonify, Response, send_from_directory
import numpy as np
import cv2
import os
from src.core.ocr_system import OCRSystem

# Configuración de carpetas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')

app = Flask(__name__)
ocr = OCRSystem()

# Ruta para servir imágenes generadas
@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


HTML = """<!DOCTYPE html>
<html lang=\"es\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>OCR de MAURA MARTINEZ</title>
  <style>
    :root { --bg1:#0a0f1f; --bg2:#0b1228; --accent:#00eaff; --accent2:#7c3cff; --text:#e6f1ff; }
    * { box-sizing: border-box; }
    body { margin:0; min-height:100vh; font-family: Inter, Segoe UI, Roboto, Arial, sans-serif; color:var(--text);
      background: radial-gradient(1000px 500px at 10% 10%, #111936 0%, var(--bg1) 40%), linear-gradient(120deg, var(--bg1), var(--bg2)); }
    .grid { display:grid; place-items:center; padding:40px; }
    .card { width:100%; max-width:1000px; backdrop-filter: blur(12px); background: rgba(255,255,255,0.04); border:1px solid rgba(0,234,255,0.2); border-radius:20px; padding:28px; box-shadow: 0 0 40px rgba(0,234,255,0.08); }
    .title { display:flex; align-items:center; gap:12px; font-weight:700; letter-spacing:0.6px; }
    .title .logo { width:16px; height:16px; border-radius:50%; background: conic-gradient(from 180deg, var(--accent), var(--accent2)); box-shadow:0 0 20px var(--accent); }
    .subtitle { opacity:0.8; margin-top:6px; font-size:14px; }
    .row { display:flex; gap:20px; flex-wrap:wrap; margin-top:22px; }
    .panel { flex:1 1 320px; border:1px solid rgba(124,60,255,0.25); border-radius:16px; padding:16px; background: rgba(124,60,255,0.08); }
    .panel h3 { margin:0 0 12px 0; font-size:16px; letter-spacing:0.4px; }
    .upload { border:1px dashed rgba(0,234,255,0.3); border-radius:14px; padding:16px; text-align:center; background: rgba(0,234,255,0.06); }
    .upload input { display:block; width:100%; margin-top:10px; }
    .controls { display:flex; gap:12px; flex-wrap:wrap; margin-top:14px; }
    select, button { appearance:none; border:1px solid rgba(255,255,255,0.2); background: rgba(255,255,255,0.06); color:var(--text); padding:10px 12px; border-radius:12px; font-size:14px; }
    button { border:1px solid rgba(0,234,255,0.6); background: linear-gradient(180deg, rgba(0,234,255,0.25), rgba(0,234,255,0.12)); box-shadow: 0 0 20px rgba(0,234,255,0.12); cursor:pointer; }
    button:hover { box-shadow: 0 0 28px rgba(0,234,255,0.24); }
    .preview { margin-top:14px; display:flex; justify-content:center; }
    .preview img { max-width:100%; border-radius:12px; border:1px solid rgba(255,255,255,0.1); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
    .result { white-space:pre-wrap; line-height:1.6; padding:12px; border-radius:12px; background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.15); min-height:120px; }
    .status { margin-top:8px; font-size:13px; opacity:0.85; }
  </style>
  <script>
    async function processImage() {
      const fileInput = document.getElementById('imageInput');
      const typeSel = document.getElementById('textType');
      const status = document.getElementById('status');
      const result = document.getElementById('result');
      result.textContent = '';
      if (!fileInput.files.length) { status.textContent = 'Sube una imagen primero'; return; }
      const fd = new FormData();
      fd.append('image', fileInput.files[0]);
      fd.append('text_type', typeSel.value);
      status.textContent = 'Procesando...';
      try {
        const res = await fetch('/ocr', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) { throw new Error(data.error || 'Error en OCR'); }
        
        // Mostrar texto
        result.textContent = (data.lines || []).join('\\n');
        
        // Mostrar imágenes extraídas si las hay
        const existingImages = document.getElementById('extractedImagesPanel');
        if (existingImages) existingImages.remove();
        
        console.log("Images received:", data.images); // DEBUG LOG
        
        if (data.images && data.images.length > 0) {
            const imgPanel = document.createElement('div');
            imgPanel.id = 'extractedImagesPanel';
            imgPanel.className = 'panel';
            imgPanel.style.marginTop = '20px';
            imgPanel.innerHTML = `<h3>Imágenes Extraídas (${data.images.length})</h3><div class="row" style="gap:10px;"></div>`;
            
            const container = imgPanel.querySelector('.row');
            data.images.forEach(imgData => {
                const imgWrap = document.createElement('div');
                imgWrap.style.flex = '0 0 auto';
                imgWrap.innerHTML = `<img src="${imgData.url}" style="height:150px; border-radius:8px; border:1px solid #333; object-fit: cover;" title="${imgData.filename}">`;
                container.appendChild(imgWrap);
            });
            
            // Insertar después del panel de resultado
            document.querySelector('.panel:last-child').after(imgPanel);
        } else {
             console.log("No images detected in response.");
        }
        
        status.textContent = 'Listo. Texto e imágenes procesados.';
      } catch (e) {
        status.textContent = 'Error: ' + e.message;
      }
    }
    function previewImage() {
      const fileInput = document.getElementById('imageInput');
      const img = document.getElementById('preview');
      if (fileInput.files && fileInput.files[0]) { img.src = URL.createObjectURL(fileInput.files[0]); }
    }
  </script>
</head>
<body>
  <div class=\"grid\">
    <div class=\"card\"> 
      <div class=\"title\"><div class=\"logo\"></div> OCR DE MAURA MARTINEZ</div>
      <div class=\"subtitle\">Sube una imagen y obtén el texto reconocido (Digital / Manuscrito)</div>
      <div class=\"row\">
        <div class=\"panel\">
          <h3>Entrada</h3>
          <div class=\"upload\">
            <div>Arrastra o selecciona tu imagen</div>
            <input id=\"imageInput\" type=\"file\" accept=\"image/*\" onchange=\"previewImage()\" />
            <div class=\"controls\">
              <select id=\"textType\"> 
                <option value=\"digital\">Texto digital</option>
                <option value=\"manuscrito\">Texto manuscrito</option>
              </select>
              <button onclick=\"processImage()\">Procesar</button>
            </div>
          </div>
          <div class=\"preview\"><img id=\"preview\" alt=\"Vista previa\" /></div>
        </div>
        <div class=\"panel\">
          <h3>Resultado</h3>
          <div id=\"result\" class=\"result mono\"></div>
          <div id=\"status\" class=\"status\">Esperando imagen...</div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>"""

@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.post("/ocr")
def ocr_route():
    if 'image' not in request.files:
        return jsonify({"error": "Imagen no enviada"}), 400
    text_type = request.form.get('text_type', 'digital')
    file = request.files['image']
    data = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "No se pudo leer la imagen"}), 400
    try:
        # Activar extracción de imágenes
        result = ocr.process_image(img, text_type=text_type, extract_images=True)
        
        # Procesar URLs de imágenes extraídas
        images_response = []
        if result.get("extracted_images_metadata"):
            print(f"[DEBUG] Metadata received: {len(result['extracted_images_metadata'])} images")
            for img_meta in result["extracted_images_metadata"]:
                # Simplificación: Usamos el nombre de archivo directamente ya que sabemos la estructura
                filename = img_meta["filename"]
                # Construimos la URL asumiendo que están en output/images
                url_path = f"/output/images/{filename}"
                
                images_response.append({
                    "filename": filename,
                    "url": url_path,
                    "type": img_meta.get("type", "unknown")
                })
                print(f"[DEBUG] Image added: {url_path}")

        return jsonify({
            "lines": result.get("text_lines", []),
            "full_text": result.get("full_text", ""),
            "images": images_response
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
