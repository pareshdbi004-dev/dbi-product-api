"""
DBI Product Lookup API — for Payal Voice Agent (Bolna Custom Function)
Provides deterministic, guaranteed-accurate product/price/weight lookup.
No RAG, no hallucination risk — direct database query.
"""
import json
import math
import re
from difflib import SequenceMatcher
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DBI Product Lookup API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("products_db.json", encoding="utf-8") as f:
    PRODUCTS = json.load(f)

try:
    with open("making_charges.json", encoding="utf-8") as f:
        MAKING_CHARGES = json.load(f)
except FileNotFoundError:
    MAKING_CHARGES = {}

# Dynamic pricing formula constants (Extension Nipple & Hex Nipple, per Pareshbhai's cost sheet)
SALES_MARGIN_PCT = 0.0204
CD_PCT = 0.0504
EXTRA_BULK_DISCOUNT_PCT = 0.02  # for 15-30+ box orders per size

DISCOUNT_RULES = {
    # (part hint via collection name patterns could be added; default 44%)
    "default": 0.56,
    "part04": 0.54,
    "rackbolt": 0.575,
}


def norm(s: str) -> str:
    if not s:
        return ""
    s = s.upper()
    s = s.replace("2IN1", "2 IN 1").replace("FROUNT", "FRONT")
    s = s.replace(".", "")  # so "S.S" and "SS" normalize identically
    s = re.sub(r"[^A-Z0-9/\" ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def net_rate(mrp: float, part04: bool = False, rackbolt: bool = False) -> float:
    if rackbolt:
        return round(mrp * DISCOUNT_RULES["rackbolt"], 2)
    if part04:
        return round(mrp * DISCOUNT_RULES["part04"], 2)
    return round(mrp * DISCOUNT_RULES["default"], 2)


def token_score(query: str, target: str) -> float:
    """Fraction of query words found in target — handles partial phrases like 'Heavy' well."""
    q_tokens = set(norm(query).split())
    t_tokens = set(norm(target).split())
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / len(q_tokens)


@app.get("/lookup")
def lookup(
    collection: str = Query(None, description="Collection/series name, e.g. 'Gemini', 'Zara'"),
    item: str = Query(None, description="Item type, e.g. 'Bib Cock', 'Pillar Cock'"),
    code: str = Query(None, description="Exact product code, e.g. 'GM-101'"),
):
    """
    Deterministic product search. Provide `code` for exact lookup,
    or `collection` + `item` for name-based search.
    Returns the single best match, or a list if ambiguous.
    """
    if code:
        code_n = code.strip().upper()
        matches = [p for p in PRODUCTS if p["code"].upper() == code_n]
        if matches:
            return {"status": "exact_match", "result": matches[0]}
        return {"status": "not_found", "message": "No product with this code in Knowledge Base."}

    candidates = PRODUCTS
    if collection:
        coll_n = norm(collection)
        candidates = [p for p in candidates if coll_n in norm(p["collection"]) or norm(p["collection"]) in coll_n]
        if not candidates:
            scored = sorted(PRODUCTS, key=lambda p: -similarity(coll_n, norm(p["collection"])))
            best_score = similarity(coll_n, norm(scored[0]["collection"])) if scored else 0
            if best_score > 0.6:
                best_coll = scored[0]["collection"]
                candidates = [p for p in PRODUCTS if p["collection"] == best_coll]
            else:
                return {"status": "not_found", "message": f"No collection matching '{collection}' in Knowledge Base."}

    if item:
        item_n = norm(item)

        # Business rule: a bare word like "Heavy"/"SS"/"Stainless" without other context defaults to
        # excluding the Stainless Steel variant unless dealer explicitly said SS/steel/stainless.
        mentions_ss = any(w in item_n.split() for w in ("SS", "STAINLESS", "STEEL"))
        search_pool = candidates if mentions_ss else [p for p in candidates if "SS" not in norm(p["item"]).split()]

        # 1. Exact full-name match first
        exact = [p for p in search_pool if norm(p["item"]) == item_n]
        if exact:
            return {"status": "exact_match", "result": exact[0]} if len(exact) == 1 else {"status": "multiple_exact_matches", "results": exact}

        # 2. Token-overlap match (handles partial words like "Heavy" matching "Waste Coupling Heavy")
        scored = sorted(search_pool, key=lambda p: (-token_score(item_n, p["item"]), len(p["item"])))
        if scored and token_score(item_n, scored[0]["item"]) >= 0.99:
            top_score = token_score(item_n, scored[0]["item"])
            ties = [p for p in scored if token_score(item_n, p["item"]) >= top_score - 0.001]
            # Prefer the shortest/simplest name among ties (most likely the "plain" variant the dealer means)
            ties_sorted = sorted(ties, key=lambda p: len(p["item"]))
            if len(ties_sorted) == 1 or len(ties_sorted[0]["item"]) < len(ties_sorted[1]["item"]):
                return {"status": "best_guess_match", "result": ties_sorted[0], "other_options": ties_sorted[1:4]}
            return {"status": "multiple_matches_same_length", "results": ties_sorted[:5]}

        # 3. Fuzzy fallback (character similarity, for typos/close variants)
        scored2 = sorted(search_pool, key=lambda p: -similarity(item_n, norm(p["item"])))
        if scored2:
            top_score = similarity(item_n, norm(scored2[0]["item"]))
            if top_score > 0.55:
                ties = [p for p in scored2 if similarity(item_n, norm(p["item"])) >= top_score - 0.05]
                if len(ties) == 1:
                    return {"status": "fuzzy_match", "result": ties[0]}
                return {"status": "multiple_fuzzy_matches", "results": ties[:5]}
        return {"status": "not_found", "message": f"No item matching '{item}' in collection."}

    if collection and not item:
        return {"status": "collection_items", "results": candidates[:30], "total": len(candidates)}

    return {"status": "error", "message": "Provide at least 'code' or 'collection'/'item'."}


@app.get("/weight_lookup")
def weight_lookup(
    grams: float = Query(..., description="Weight the dealer mentioned, in grams"),
    category: str = Query(None, description="Optional category hint, e.g. 'Waste Coupling', 'Bib Cock', collection name"),
):
    """
    Find the product(s) with weight nearest to the given gram value.
    Optionally restrict search to a category/collection/item-type first.
    """
    target_kg = grams / 1000.0
    candidates = [p for p in PRODUCTS if p["weight_kg"] is not None]

    if category:
        cat_n = norm(category)
        filtered = [
            p for p in candidates
            if cat_n in norm(p["collection"]) or cat_n in norm(p["item"]) or norm(p["collection"]) in cat_n or norm(p["item"]) in cat_n
        ]
        if filtered:
            candidates = filtered

    if not candidates:
        return {"status": "not_found", "message": "No weighted products found matching that category."}

    scored = sorted(candidates, key=lambda p: abs(p["weight_kg"] - target_kg))
    nearest = scored[0]
    diff = abs(nearest["weight_kg"] - target_kg)

    close_ties = [p for p in scored if abs(p["weight_kg"] - target_kg) <= diff + 0.005][:3]

    return {
        "status": "match_found",
        "target_kg": target_kg,
        "nearest": nearest,
        "alternatives_if_ambiguous": close_ties[1:] if len(close_ties) > 1 else [],
    }


@app.get("/price_with_discount")
def price_with_discount(
    code: str = Query(..., description="Exact product code"),
    rackbolt: bool = Query(False),
    part04: bool = Query(False),
):
    """Returns MRP and the discounted net rate for a given product code."""
    code_n = code.strip().upper()
    matches = [p for p in PRODUCTS if p["code"].upper() == code_n]
    if not matches:
        return {"status": "not_found"}
    p = matches[0]
    return {
        "status": "ok",
        "code": p["code"],
        "item": p["item"],
        "collection": p["collection"],
        "mrp": p["mrp"],
        "net_rate_44pct": net_rate(p["mrp"], part04=part04, rackbolt=rackbolt),
    }


@app.get("/dynamic_net_rate")
def dynamic_net_rate(
    code: str = Query(..., description="Exact product code, e.g. 'EX-08', 'HX-03'. Currently supported only for Extension Nipple and Hex Nipple items."),
    brass_rate: float = Query(..., description="Today's brass rate in Rs per kg, e.g. 850"),
    boxes: int = Query(None, description="Number of boxes per size the dealer is ordering, if mentioned. 15-30+ boxes per size qualifies for an extra 2% discount."),
):
    """
    Calculates the dynamic, brass-rate-linked net rate for items that use the
    making-charge cost formula (currently: Extension Nipple, Hex Nipple).
    Formula: TOTAL(per kg) = brass_rate + making_charge(DIFF)
             PER_PCS = TOTAL * weight_kg
             + sales margin 2.04% -> subtotal
             + CD 5.04% -> FINAL (rounded up to nearest rupee)
    If boxes given and >= 15, an additional 2% bulk discount is applied on FINAL.
    """
    code_n = code.strip().upper()
    if code_n not in MAKING_CHARGES:
        return {
            "status": "not_supported",
            "message": f"Dynamic brass-rate pricing is not yet set up for code '{code}'. Use the standard lookup_product_price function instead, or say the technical-team fallback line.",
        }

    mc = MAKING_CHARGES[code_n]
    weight = mc["weight"]
    diff = mc["diff"]

    total_per_kg = brass_rate + diff
    per_pcs = total_per_kg * weight
    sales_margin = per_pcs * SALES_MARGIN_PCT
    subtotal = per_pcs + sales_margin
    cd_amount = subtotal * CD_PCT
    final_with_cd = subtotal + cd_amount
    final_rounded = round(final_with_cd)

    result = {
        "status": "ok",
        "code": code_n,
        "item": mc["item"],
        "brass_rate_used": brass_rate,
        "per_pcs_base": round(per_pcs, 2),
        "net_rate_with_cd": final_rounded,
        "note": "This is the net rate WITH 5% Cash Discount (advance payment, full carton, basket >= Rs 50,000 pre-GST). Without these conditions, quote per_pcs_base + sales margin only (no CD).",
    }

    if boxes is not None and boxes >= 15:
        bulk_discount = final_rounded * EXTRA_BULK_DISCOUNT_PCT
        result["bulk_discount_applied"] = True
        result["boxes"] = boxes
        result["final_rate_with_bulk_discount"] = round(final_rounded - bulk_discount, 2)
    else:
        result["bulk_discount_applied"] = False
        result["note_bulk"] = "15-30+ boxes per size required for additional 2% discount."

    return result


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "total_products": len(PRODUCTS)}
