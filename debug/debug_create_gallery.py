
import os
import glob

GALLERY_FILE = "zdebug_gallery.html"
IMAGE_GLOB = "zdebug_*.jpg"

def create_gallery():
    images = sorted(glob.glob(IMAGE_GLOB))
    if not images:
        print(f"Error: No files matched '{IMAGE_GLOB}' in {os.getcwd()}.")
        return
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Debug Capture Gallery</title>
        <style>
            body { font-family: sans-serif; background: #222; color: #eee; margin: 0; padding: 20px; }
            h1 { text-align: center; }
            .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; }
            .card { background: #333; padding: 10px; border-radius: 8px; text-align: center; }
            img { max-width: 100%; height: auto; border-radius: 4px; }
            .timestamp { margin-top: 5px; font-size: 0.8em; color: #aaa; }
        </style>
    </head>
    <body>
        <h1>zdebug Captures</h1>
        <div class="grid">
    """
    
    for img_path in images:
        filename = os.path.basename(img_path)
        # Extract timestamp from filename frame_123456789.jpg
        try:
            ts = filename.split('_')[1].split('.')[0]
        except:
            ts = filename
            
        html += f"""
            <div class="card">
                <a href="{filename}" target="_blank">
                    <img src="{filename}" alt="{filename}">
                </a>
                <div class="timestamp">{ts}</div>
            </div>
        """
        
    html += """
        </div>
    </body>
    </html>
    """
    
    with open(GALLERY_FILE, "w") as f:
        f.write(html)
        
    print(f"Gallery created at: {os.path.abspath(GALLERY_FILE)}")
    print("Open this file in your browser to view all captures.")

if __name__ == "__main__":
    create_gallery()
