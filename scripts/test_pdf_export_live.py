import urllib.request
import json
import os

def test_live_pdf_export():
    print("Testing live health endpoint...")
    req = urllib.request.Request("http://127.0.0.1:8000/health")
    with urllib.request.urlopen(req) as resp:
        health_data = json.loads(resp.read())
        print(f"Health: {health_data['status']}, Models loaded: EfficientNet={health_data['efficientnet_loaded']}, ViT={health_data['vit_loaded']}")

    print("\nScanning a test specimen via POST /predict...")
    with open("dataset/real/real_0.jpg", "rb") as f:
        file_bytes = f.read()

    # Create multipart form data
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="real_0.jpg"\r\n')
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        scan_record = json.loads(resp.read())
        scan_id = scan_record.get("scan_id")
        pred = scan_record.get("prediction")
        conf = scan_record.get("confidence")
        print(f"Scan complete! Scan ID: {scan_id}, Verdict: {pred} ({conf*100:.1f}%)")

    print("\nTesting GET /report/{scan_id}...")
    req = urllib.request.Request(f"http://127.0.0.1:8000/report/{scan_id}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        pdf_bytes = resp.read()
        print(f"GET /report/{scan_id} Success! Received {len(pdf_bytes)} bytes of PDF.")
        assert len(pdf_bytes) > 20000, "PDF too small"

    print("\nTesting POST /export-report-pdf...")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/export-report-pdf",
        data=json.dumps(scan_record).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        post_pdf_bytes = resp.read()
        print(f"POST /export-report-pdf Success! Received {len(post_pdf_bytes)} bytes of PDF.")
        assert len(post_pdf_bytes) > 20000, "POST PDF too small"

    print("\nAll live PDF export endpoints successfully verified!")

if __name__ == "__main__":
    test_live_pdf_export()
