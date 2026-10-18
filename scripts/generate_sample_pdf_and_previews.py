import urllib.request
import json
import os
import subprocess
import sys
sys.path.insert(0, os.path.abspath("."))
from backend import pdf_html_generator as gen

def main():
    artifact_dir = r"C:\Users\harsh\.gemini\antigravity\brain\bbcd98a0-900e-4b53-8e4b-456b213399b5"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # 1. Scan a fake specimen from test_images/fake or dataset/fake
    test_img = "test_images/fake/test_fake_01.jpg"
    if not os.path.exists(test_img):
        test_img = "dataset/fake/fake_0.jpg"
        
    print(f"Reading test specimen: {test_img}...")
    with open(test_img, "rb") as f:
        file_bytes = f.read()

    boundary = "----WebKitFormBoundarySentinelDossierTest"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(test_img)}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    print("Calling POST http://127.0.0.1:8000/predict ...")
    with urllib.request.urlopen(req, timeout=60) as resp:
        scan_record = json.loads(resp.read())

    scan_id = scan_record.get("scan_id")
    print(f"Scan complete! Scan ID: {scan_id}, Verdict: {scan_record.get('prediction')} ({scan_record.get('confidence', 0)*100:.1f}%)")

    # 2. Generate HTML directly and save to test_report.html
    html = gen.generate_report_html(scan_record)
    with open("test_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote test_report.html ({len(html)} bytes)")

    # 3. Export PDF via live endpoint
    pdf_url = f"http://127.0.0.1:8000/report/{scan_id}"
    with urllib.request.urlopen(pdf_url, timeout=30) as resp:
        pdf_bytes = resp.read()

    pdf_out = os.path.join(artifact_dir, "sentinel_sample_forensic_report.pdf")
    with open(pdf_out, "wb") as f:
        f.write(pdf_bytes)
    print(f"Exported live PDF to {pdf_out} ({len(pdf_bytes)} bytes)")

    # 4. Run scripts/capture_pdf_preview.js with node
    print("Executing node scripts/capture_pdf_preview.js to capture high-res page screenshots...")
    res = subprocess.run(["node", "scripts/capture_pdf_preview.js"], capture_output=True, text=True)
    print("Node output:", res.stdout)
    if res.stderr:
        print("Node stderr:", res.stderr)
    if res.returncode != 0:
        raise RuntimeError("Failed to capture PDF preview screenshots")

    print("\nSUCCESS: All 4 pages rendered and saved to artifact directory!")

if __name__ == "__main__":
    main()
