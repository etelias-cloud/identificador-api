from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image 
from io import BytesIO
import requests
import pandas as pd
import imagehash
import time
import re

app = FastAPI(title="Identificador de Productos")

MASTER_CSV_URL = "https://docs.google.com/spreadsheets/d/1R0dowhyTIPVwQOozpsVpDXbjZdeD6VtabvALjOkyjco/export?format=csv&gid=0"
CACHE_TTL_SECONDS = 900  # 15 minutos
MIN_SCORE = 0.40         # umbral orientativo, pero NO filtra resultados

_cache = {
    "loaded_at": 0,
    "items": []
}


def normalize_drive_url(url: str) -> str:
    if not url or not isinstance(url, str):
        return ""

    url = url.strip()

    if "drive.google.com/uc?" in url:
        return url

    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    return url


def download_image(url: str) -> Image.Image:
    direct_url = normalize_drive_url(url)
    if not direct_url:
        raise ValueError("URL vacía")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    resp = requests.get(direct_url, timeout=30, headers=headers)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type.lower():
        raise ValueError(f"La URL no devolvió una imagen válida: {direct_url}")

    return Image.open(BytesIO(resp.content)).convert("RGB")


def build_hashes(img: Image.Image) -> dict:
    img = img.convert("RGB")
    return {
        "phash": imagehash.phash(img),
        "dhash": imagehash.dhash(img),
        "whash": imagehash.whash(img),
        "ahash": imagehash.average_hash(img),
    }


def hash_distance(h1: dict, h2: dict) -> float:
    return (
        (h1["phash"] - h2["phash"]) * 0.40 +
        (h1["dhash"] - h2["dhash"]) * 0.25 +
        (h1["whash"] - h2["whash"]) * 0.25 +
        (h1["ahash"] - h2["ahash"]) * 0.10
    )


def distance_to_score(distance: float) -> float:
    score = max(0.0, 1.0 - (distance / 32.0))
    return round(score, 4)


def load_master_if_needed():
    now = time.time()
    if _cache["items"] and (now - _cache["loaded_at"] < CACHE_TTL_SECONDS):
        return

    print("[INFO] Leyendo maestro CSV...", flush=True)
    df = pd.read_csv(MASTER_CSV_URL)
    df.columns = df.columns.str.strip()

    print("COLUMNAS DETECTADAS:", df.columns.tolist(), flush=True)

    required = ["SKU", "DESCRIPCIÓN", "CÓDIGO DE BARRAS"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Falta columna requerida en maestro: {col}")

    image_cols = [c for c in ["IMG1-URL", "IMG2-URL", "IMG3-URL", "IMG4-URL"] if c in df.columns]
    if not image_cols:
        raise RuntimeError(
            f"No hay columnas IMG1-URL..IMG4-URL en el maestro. Detectadas: {df.columns.tolist()}"
        )

    print("[INFO] Columnas de imagen detectadas:", image_cols, flush=True)

    items = []

    for _, row in df.iterrows():
        sku = str(row.get("SKU", "")).strip()
        descripcion = str(row.get("DESCRIPCIÓN", "")).strip()
        codigo_barras = str(row.get("CÓDIGO DE BARRAS", "")).strip()

        if not sku or sku.lower() == "nan":
            continue

        for col in image_cols:
            raw_url = str(row.get(col, "")).strip()
            if not raw_url or raw_url.lower() == "nan":
                continue

            try:
                img = download_image(raw_url)
                hashes = build_hashes(img)

                items.append({
                    "sku": sku,
                    "descripcion": descripcion,
                    "codigo_barras": codigo_barras,
                    "imagen_url": normalize_drive_url(raw_url),
                    "hashes": hashes
                })
            except Exception as ex:
                print(f"[WARN] No se pudo cargar {col} para {sku}: {ex}", flush=True)
                continue

    print(f"[INFO] Referencias válidas cargadas: {len(items)}", flush=True)

    if not items:
        raise RuntimeError("No se pudieron cargar imágenes válidas del maestro")

    _cache["items"] = items
    _cache["loaded_at"] = now

@app.get("/")
def root():
    return {"ok": True, "service": "identificador-api"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/identify")
async def identify(file: UploadFile = File(...), top_k: int = 3):
    try:
        print("[INFO] Iniciando /identify", flush=True)
        load_master_if_needed()

        content = await file.read()
        print(f"[INFO] Bytes recibidos: {len(content)}", flush=True)

        query_img = Image.open(BytesIO(content)).convert("RGB")
        query_hashes = build_hashes(query_img)

        best_by_sku = {}

        for item in _cache["items"]:
            dist = hash_distance(query_hashes, item["hashes"])
            score = distance_to_score(dist)

            candidate = {
                "sku": item["sku"],
                "descripcion": item["descripcion"],
                "codigo_barras": item["codigo_barras"],
                "score": score,
                "imagen_url": item["imagen_url"],
                "distance": dist
            }

            current = best_by_sku.get(item["sku"])
            if current is None or dist < current["distance"]:
                best_by_sku[item["sku"]] = candidate

        results = sorted(best_by_sku.values(), key=lambda x: x["distance"])[:top_k]

        final = []
        for r in results:
            final.append({
                "sku": str(r["sku"]),
                "descripcion": str(r["descripcion"]),
                "codigo_barras": str(r["codigo_barras"]),
                "score": float(r["score"]),
                "imagen_url": str(r["imagen_url"]),
                "aceptable": bool(float(r["score"]) >= MIN_SCORE)
            })

        print(f"[INFO] Resultados devueltos: {len(final)}", flush=True)

        return {
            "ok": True,
            "resultados": final
        }

    except Exception as e:
        import traceback
        print("[ERROR] /identify:", str(e), flush=True)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
