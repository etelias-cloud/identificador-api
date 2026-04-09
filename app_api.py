from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from io import BytesIO
import requests
import pandas as pd
import imagehash
import time
import re
import traceback
import numpy as np

app = FastAPI(title="Identificador de Productos")

MASTER_CSV_URL = "https://docs.google.com/spreadsheets/d/1R0dowhyTIPVwQOozpsVpDXbjZdeD6VtabvALjOkyjco/export?format=csv&gid=0"
CACHE_TTL_SECONDS = 900

# Umbrales de decisión
AUTO_MIN_SCORE = 0.72
AUTO_MIN_GAP = 0.08
REVIEW_MIN_SCORE = 0.35

_cache = {
    "loaded_at": 0,
    "items": []
}


def limpiar_valores(obj):
    if isinstance(obj, dict):
        return {k: limpiar_valores(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [limpiar_valores(v) for v in obj]
    if isinstance(obj, tuple):
        return [limpiar_valores(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    return obj


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


def normalize_col(name: str) -> str:
    return (
        str(name)
        .strip()
        .upper()
        .replace("Á", "A")
        .replace("É", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ú", "U")
    )


def download_image(url: str) -> Image.Image:
    direct_url = normalize_drive_url(url)
    if not direct_url:
        raise ValueError("URL vacía")

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(direct_url, timeout=30, headers=headers)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type.lower():
        raise ValueError(f"La URL no devolvió una imagen válida: {direct_url}")

    return Image.open(BytesIO(resp.content)).convert("RGB")


def prepare_image(img: Image.Image, size=(256, 256)) -> Image.Image:
    """
    Normaliza tamaño, corrige orientación y hace una versión más estable.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = ImageOps.contain(img, size)
    canvas = Image.new("RGB", size, (255, 255, 255))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def central_crop(img: Image.Image, crop_ratio=0.72) -> Image.Image:
    """
    Recorta la zona central para reducir ruido de fondo.
    """
    w, h = img.size
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    return img.crop((left, top, right, bottom))


def dominant_color_vector(img: Image.Image) -> np.ndarray:
    """
    Saca una firma simple de color.
    Promedio RGB normalizado.
    """
    arr = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32)
    mean_rgb = arr.mean(axis=(0, 1))
    return mean_rgb / 255.0


def color_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    Distancia euclídea entre colores promedio normalizados.
    Rango aproximado: 0 a ~1.732
    """
    return float(np.linalg.norm(c1 - c2))


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
        (h1["dhash"] - h2["dhash"]) * 0.20 +
        (h1["whash"] - h2["whash"]) * 0.25 +
        (h1["ahash"] - h2["ahash"]) * 0.15
    )


def combined_distance(query: dict, ref: dict) -> float:
    """
    Mezcla:
    - hash imagen completa
    - hash recorte central
    - distancia de color
    """
    d_full = hash_distance(query["hashes_full"], ref["hashes_full"])
    d_crop = hash_distance(query["hashes_crop"], ref["hashes_crop"])
    d_color = color_distance(query["color_vec"], ref["color_vec"])

    # Penalización extra por diferencia de color fuerte
    color_penalty = 0.0
    if d_color > 0.30:
        color_penalty = 4.0
    elif d_color > 0.20:
        color_penalty = 2.0

    # Escalamos d_color para que tenga peso comparable al hash
    total = (d_full * 0.45) + (d_crop * 0.40) + (d_color * 10.0 * 0.15) + color_penalty
    return float(total)


def distance_to_score(distance: float) -> float:
    """
    Convierte distancia combinada a score 0..1.
    Es un score relativo, no probabilidad real.
    """
    score = max(0.0, 1.0 - (distance / 32.0))
    return round(float(score), 4)


def build_features(img: Image.Image) -> dict:
    base = prepare_image(img)
    crop = central_crop(base)

    return {
        "hashes_full": build_hashes(base),
        "hashes_crop": build_hashes(crop),
        "color_vec": dominant_color_vector(crop),
    }


def load_master_if_needed():
    now = time.time()
    if _cache["items"] and (now - _cache["loaded_at"] < CACHE_TTL_SECONDS):
        return

    print("[INFO] Leyendo maestro CSV...", flush=True)

    df = pd.read_csv(MASTER_CSV_URL)
    df.columns = [normalize_col(c) for c in df.columns]

    print("[INFO] COLUMNAS DETECTADAS:", df.columns.tolist(), flush=True)

    required = ["SKU", "DESCRIPCION", "CODIGO DE BARRAS"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Falta columna requerida en maestro: {col}")

    image_cols = [c for c in df.columns if c.startswith("IMG") and "URL" in c]
    if not image_cols:
        raise RuntimeError(
            f"No hay columnas de imagen tipo IMGx-URL / IMGx URL en el maestro. Detectadas: {df.columns.tolist()}"
        )

    print("[INFO] Columnas de imagen detectadas:", image_cols, flush=True)

    items = []

    for _, row in df.iterrows():
        sku = str(row.get("SKU", "")).strip()
        descripcion = str(row.get("DESCRIPCION", "")).strip()
        codigo_barras = str(row.get("CODIGO DE BARRAS", "")).strip()

        if not sku or sku.lower() == "nan":
            continue

        for col in image_cols:
            raw_url = str(row.get(col, "")).strip()
            if not raw_url or raw_url.lower() == "nan":
                continue

            try:
                img = download_image(raw_url)
                features = build_features(img)

                items.append({
                    "sku": sku,
                    "descripcion": descripcion,
                    "codigo_barras": codigo_barras,
                    "imagen_url": normalize_drive_url(raw_url),
                    "features": features,
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
    return {"ok": True, "items_cacheados": len(_cache["items"])}


@app.post("/identify")
async def identify(file: UploadFile = File(...), top_k: int = 3):
    try:
        print("[INFO] Iniciando /identify", flush=True)
        load_master_if_needed()

        content = await file.read()
        if not content:
            raise ValueError("Archivo vacío")

        print(f"[INFO] Bytes recibidos: {len(content)}", flush=True)

        query_img = Image.open(BytesIO(content)).convert("RGB")
        query_features = build_features(query_img)

        best_by_sku = {}

        for item in _cache["items"]:
            dist = combined_distance(query_features, item["features"])
            score = distance_to_score(dist)

            candidate = {
                "sku": str(item["sku"]),
                "descripcion": str(item["descripcion"]),
                "codigo_barras": str(item["codigo_barras"]),
                "score": float(score),
                "imagen_url": str(item["imagen_url"]),
                "distance": float(dist)
            }

            current = best_by_sku.get(item["sku"])
            if current is None or dist < current["distance"]:
                best_by_sku[item["sku"]] = candidate

        results = sorted(best_by_sku.values(), key=lambda x: x["distance"])[:top_k]

        gap = 0.0
        if len(results) > 1:
            gap = float(results[0]["score"] - results[1]["score"])

        decision = "revisar"
        if results:
            if results[0]["score"] >= AUTO_MIN_SCORE and gap >= AUTO_MIN_GAP:
                decision = "auto"
            elif results[0]["score"] < REVIEW_MIN_SCORE:
                decision = "sin_match"

        final = []
        for r in results:
            final.append({
                "sku": str(r["sku"]),
                "descripcion": str(r["descripcion"]),
                "codigo_barras": str(r["codigo_barras"]),
                "score": float(r["score"]),
                "imagen_url": str(r["imagen_url"]),
                "aceptable": bool(float(r["score"]) >= REVIEW_MIN_SCORE)
            })

        response = {
            "ok": True,
            "decision": decision,
            "gap": round(gap, 4),
            "resultados": final
        }

        response = limpiar_valores(response)

        print(
            f"[INFO] Resultados devueltos: {len(final)} | decision={decision} | gap={round(gap, 4)}",
            flush=True
        )

        if final:
            print(
                f"[INFO] Top1={final[0]['sku']} score={final[0]['score']}",
                flush=True
            )

        return response

    except Exception as e:
        print(f"[ERROR] /identify: {e}", flush=True)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
