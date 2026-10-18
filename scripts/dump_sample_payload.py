import urllib.request
import json

def main():
    with open("dataset/fake/fake_0.jpg", "rb") as f:
        file_bytes = f.read()

    boundary = "----WebKitFormBoundaryDumpSample"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="sample_specimen.jpg"\r\n')
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    with open("sample_scan_payload.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Successfully saved sample_scan_payload.json! File size:", len(open("sample_scan_payload.json").read()))

if __name__ == "__main__":
    main()
