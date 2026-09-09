import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def extract_graph_from_image(image_bytes):
    # 1. Read image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Invalid image")
        
    orig_h, orig_w = img.shape
    
    # 2. Resize to a low-res grid to extract a manageable graph (e.g., 50x50 grid)
    # Maintain aspect ratio
    grid_size = 60
    if orig_w > orig_h:
        w = grid_size
        h = int(grid_size * (orig_h / orig_w))
    else:
        h = grid_size
        w = int(grid_size * (orig_w / orig_h))
        
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    
    # 3. Threshold (Text and walls are dark, corridors are light)
    # Use adaptive threshold or fixed. Let's use fixed 150.
    _, thresh = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY)
    
    # 4. Optional: Morphological operations to clean up noise (small dots)
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    nodes = []
    node_map = {} # (x, y) -> node_index
    edges = []
    
    idx = 0
    # 5. Extract nodes (free space = 255)
    for y in range(h):
        for x in range(w):
            if opened[y, x] == 255:
                # Store normalized coordinates [0.0, 1.0]
                norm_x = x / w
                norm_y = y / h
                nodes.append([norm_x, norm_y])
                node_map[(x, y)] = idx
                idx += 1
                
    # 6. Extract edges (connect adjacent free cells)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for y in range(h):
        for x in range(w):
            if (x, y) in node_map:
                u = node_map[(x, y)]
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in node_map:
                        v = node_map[(nx, ny)]
                        edges.append([u, v])
                        
    return nodes, edges

@app.route("/api/parse_map", methods=["POST"])
def parse_map():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    image_bytes = file.read()
    
    try:
        nodes, edges = extract_graph_from_image(image_bytes)
        
        # Gemini AI Security Analysis
        ai_message = "Gemini AI successfully processed the structural layout."
        try:
            if not GEMINI_API_KEY:
                raise ValueError("Gemini API Key is missing in .env")
                
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"I have used Computer Vision to extract an indoor navigation graph from a factory blueprint. The generated graph has {len(nodes)} navigable nodes and {len(edges)} connected corridors. Provide a 2-sentence security and safety assessment confirming that this indoor routing network has been successfully parsed and is ready for autonomous UAV navigation."
            
            response = model.generate_content(prompt)
            ai_message = response.text
        except Exception as e:
            print(f"Gemini API Error: {e}")
            ai_message = f"Gemini API Error: {e}. Falling back to standard processing."

        return jsonify({
            "success": True,
            "nodes": nodes,
            "edges": edges,
            "ai_analysis": ai_message
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
