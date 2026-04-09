@app.post("/identify")
async def identify(file: UploadFile = File(...), top_k: int = 3):
    try:
        load_master_if_needed()

        content = await file.read()
        query_img = Image.open(BytesIO(content)).convert("RGB")
        query_hashes = build_hashes(query_img)

        best_by_sku = {}

        for item in _cache["items"]:
            dist = hash_distance(query_hashes, item["hashes"])
            score = distance_to_score(dist)

            if item["sku"] not in best_by_sku or dist < best_by_sku[item["sku"]]["distance"]:
                best_by_sku[item["sku"]] = {
                    "sku": item["sku"],
                    "descripcion": item["descripcion"],
                    "score": score,
                    "distance": dist
                }

        results = sorted(best_by_sku.values(), key=lambda x: x["distance"])[:top_k]

        # 👉 GAP
        gap = 0
        if len(results) > 1:
            gap = results[0]["score"] - results[1]["score"]

        # 👉 DECISIÓN
        decision = "revisar"
        if results:
            if results[0]["score"] >= 0.75 and gap >= 0.10:
                decision = "auto"
            elif results[0]["score"] < 0.40:
                decision = "sin_match"

        return {
            "ok": True,
            "decision": decision,
            "gap": round(gap, 4),
            "resultados": results
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
