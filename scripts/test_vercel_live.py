import urllib.request
import pypdf
import io

def main():
    urls = [
        "https://deepfakesentinel-kappa.vercel.app",
        "https://deepfakesentinel-kappa.vercel.app/demo-report",
        "https://deepfakesentinel-kappa.vercel.app/demo-report/pdf",
        "https://deepfakesentinel-kappa.vercel.app/samples"
    ]

    for url in urls:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                ctype = resp.headers.get("content-type")
                print(f"[OK] {url} -> Status {resp.status}, Type: {ctype}, Size: {len(data)} bytes")
                if "pdf" in url:
                    reader = pypdf.PdfReader(io.BytesIO(data))
                    print(f"     PDF Page Count: {len(reader.pages)} pages")
        except Exception as e:
            print(f"[ERR] {url} -> {e}")

if __name__ == "__main__":
    main()
