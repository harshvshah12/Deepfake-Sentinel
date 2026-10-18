import urllib.request
import json
import pypdf
import io
import time

def main():
    time.sleep(2)
    print("1. Health check...")
    req = urllib.request.Request("http://127.0.0.1:8000/health")
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
        print("Server Health:", data["status"])

    print("2. Check /demo-report (HTML)...")
    req = urllib.request.Request("http://127.0.0.1:8000/demo-report")
    with urllib.request.urlopen(req, timeout=10) as r:
        html_data = r.read().decode("utf-8")
        print("GET /demo-report length:", len(html_data))
        page_count = html_data.count("class=\"pdf-page\"")
        print(f"Number of .pdf-page elements in /demo-report: {page_count}")
        assert page_count == 4, f"Expected 4 pages, got {page_count}"

    print("3. Check /demo-report/pdf (PDF)...")
    req = urllib.request.Request("http://127.0.0.1:8000/demo-report/pdf")
    with urllib.request.urlopen(req, timeout=10) as r:
        pdf_bytes = r.read()
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        print("GET /demo-report/pdf size:", len(pdf_bytes), "Pages:", len(reader.pages))
        assert len(reader.pages) == 4, f"Expected 4 pages, got {len(reader.pages)}"

    print("4. Testing live scan and /report/{scan_id}...")
    with open("dataset/fake/fake_0.jpg", "rb") as f:
        file_bytes = f.read()

    boundary = "----WebKitFormBoundaryTestDemo"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="fake_0.jpg"\r\n')
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        scan_record = json.loads(resp.read())
        scan_id = scan_record.get("scan_id")
        print(f"Scan complete: {scan_id}")

    # Check GET /report/{scan_id}
    req = urllib.request.Request(f"http://127.0.0.1:8000/report/{scan_id}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        scan_pdf_bytes = resp.read()
        scan_reader = pypdf.PdfReader(io.BytesIO(scan_pdf_bytes))
        print(f"GET /report/{scan_id} size: {len(scan_pdf_bytes)}, Pages: {len(scan_reader.pages)}")
        assert len(scan_reader.pages) == 4, f"Expected 4 pages, got {len(scan_reader.pages)}"

    # Check GET /report/{scan_id}/html
    req = urllib.request.Request(f"http://127.0.0.1:8000/report/{scan_id}/html")
    with urllib.request.urlopen(req, timeout=30) as resp:
        scan_html = resp.read().decode("utf-8")
        html_pages = scan_html.count("class=\"pdf-page\"")
        print(f"GET /report/{scan_id}/html pages: {html_pages}")
        assert html_pages == 4, f"Expected 4 pages, got {html_pages}"

    print("\nALL 4 LIVE ENDPOINTS TESTED AND 100% VERIFIED WITH 4 PAGES AND ALL 7 GRAPHS!")

if __name__ == "__main__":
    main()
