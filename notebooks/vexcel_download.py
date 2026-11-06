"""
Vexcel Aerial Imagery Download – SDS320
=======================================

Downloads Vexcel data for a rectangular area into a local data folder:

  ortho  orthoimage – ONE clean GeoTIFF with four spectral bands:
         1 blue, 2 green, 3 red, 4 near-infrared
  dsm    digital surface model – heights of the surface incl. buildings and trees
  dtm    digital terrain model – heights of the bare ground
  ndsm   normalised DSM = DSM − DTM, i.e. the height of objects above ground
         (made automatically when both DSM and DTM are downloaded)

All files are delivered in the map projection of your choice (default: Swiss
LV95, EPSG:2056) and clipped exactly to your area, without empty borders.

Who can use this
----------------
Anyone with a Vexcel account (the same login you use for the Vexcel Viewer at
https://app.vexcelgroup.com). Vexcel data is licensed: keep downloads for your
own coursework and do not redistribute them.

Before you run it
-----------------
1. Install the packages (once per environment, e.g. at the top of a Colab notebook):
       pip install requests pyproj rasterio pillow
2. Get an access token (valid for 24 hours):
   - Open https://api.vexcelgroup.com/v2/  (note: "api.", not "app.")
   - Log in with your Vexcel credentials.
   - The address bar now shows ".../swagger-ui/index.html?token=eyJ...".
   - Copy that whole URL. The script asks for it when it starts.
     (Or set it beforehand: os.environ["VEXCEL_TOKEN"] = "eyJ...")
   The token is personal – never paste it into shared notebooks or Git.
3. Edit the SETTINGS block below: area, products, resolution (zoom), projection.

What it does
------------
1. Checks your token and lists the collections (flights) covering your area.
   Unless you name one in COLLECTION, it uses the newest.
2. Works out the exact output rectangle in the target projection and downloads a
   slightly larger area, so reprojection leaves no empty (black) corners.
3. Orthoimage: downloads the area in tiles, each twice – as true colour (RGB) and
   as colour-infrared (IRG = NIR, red, green) – and stacks them to B, G, R, NIR.
4. DSM / DTM: checks whether they exist for your area and finds the finest zoom
   level available (elevation models are often coarser than the imagery). If a
   model is not available, it is skipped with a message.
5. Merges the tiles, reprojects them onto the target grid, clips them to your
   rectangle and checks that no empty pixels remain along the edges.
6. Writes one GeoTIFF per product into DATA_DIR/<NAME>/, e.g.
       <NAME>_<collection>_ortho_z19_epsg2056.tif
       <NAME>_<collection>_dsm_z18_epsg2056.tif
   Raw downloads are cached in DATA_DIR/<NAME>/cache/; re-running skips tiles
   already there.

Resolution (zoom level → pixel size)
------------------------------------
The output pixel size is the nominal resolution of each zoom level, as labelled
in the Vexcel Viewer:
    zoom 21 →   7.5 cm  (maximum – the acquisition resolution)
    zoom 20 →  15   cm
    zoom 19 →  30   cm
    zoom 18 →  60   cm
    zoom 17 → 120   cm
Each zoom step down halves the resolution and quarters the file size. The
highest zooms exist only where Vexcel flew at that resolution ("urban" layers).
All products share the same upper-left corner, and pixel sizes differ by powers
of two, so the grids nest: one 60 cm DSM pixel covers exactly 2 × 2 pixels of a
30 cm orthoimage.
(Technical note: the API delivers Web-Mercator tiles whose ground pixels are
smaller than these values away from the equator – about 0.68× in Zurich. The
script averages them to the nominal size, so the output never claims a finer
resolution than the data contain. It never asks the API for zoom levels above
what a collection was recorded at.)

In your GIS
-----------
Orthoimage bands: 1 blue, 2 green, 3 red, 4 NIR (names are in the file).
True colour: red=3, green=2, blue=1.  Colour infrared: red=4, green=3, blue=2.
NDVI = (band4 - band3) / (band4 + band3).
DSM, DTM and nDSM: one band, heights in metres; no data = -9999.
For the nDSM, the DTM is resampled onto the DSM grid; negative differences
(measurement noise) are set to 0.

Common projections
------------------
    2056  Swiss CH1903+/LV95 (metres) – recommended for Swiss analyses
    4326  WGS84 longitude/latitude (degrees)
    3857  Web Mercator (web maps)
    32632 UTM zone 32N (metres)
"""

import base64
import getpass
import hashlib
import json
import math
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer

# =============================================================================
# SETTINGS – edit these
# =============================================================================
NAME = "lucerne_tribschen_py"        # short name for this area – used in folder/file names
# BBOX = [8.482, 47.4289, 8.5035, 47.4364]  # [xmin, ymin, xmax, ymax] in BBOX_EPSG    # Katzenseen – Zurich
# BBOX = [8.469745, 46.437985, 8.538658, 46.467538]  # [xmin, ymin, xmax, ymax] in BBOX_EPSG  # Robièi – Alpine lakes
BBOX = [2666000, 1210000, 2666500, 1210250]  # [xmin, ymin, xmax, ymax] in BBOX_EPSG  # Náret – Alpine lakes
BBOX_EPSG = 2056             # 4326 = lon/lat degrees; or e.g. 2056 for LV95 metres
PRODUCTS = ["ortho", "dsm"]  # any of "ortho", "dsm", "dtm"
NDSM = False                  # with dsm + dtm: also save nDSM = DSM − DTM
LAYER = "wide-area"              # Vexcel layer, e.g. "urban", "wide-area"
COLLECTION = None            # e.g. "ch-zh-zurich-2026"; None = newest available
ZOOM = 21                    # 15–21, see table above (21 = 7.5 cm, 19 = 30 cm);
                             # for DSM/DTM the finest available zoom up to this
INCLUDE_NIR = True           # add near-infrared as band 4 (needs an 'irg' collection)
TARGET_EPSG = 2056           # output projection, see list above
OUTPUT_RES = None            # output pixel size in target units; None = zoom table
DATA_DIR = Path("data/vexcel")
MAX_TILE_PX = 4000           # max width/height of one request in pixels
BUFFER_PX = 64               # extra pixels downloaded around the area
VERBOSE = False              # True: print the API's full parameter lists
# =============================================================================

BASE_URL = "https://api.vexcelgroup.com/v2"
ZOOM_TO_CM = {z: 7.5 * 2 ** (21 - z) for z in range(15, 22)}   # nominal (output) sizes
MERC_RES0 = 156543.03392804097   # Web-Mercator pixel size at zoom 0 (m)
BAND_NAMES = ("blue", "green", "red", "nir")
ELEV_NODATA = -9999.0
PRODUCT_LABELS = {"ortho": "orthoimage", "dsm": "digital surface model (DSM)",
                  "dtm": "digital terrain model (DTM)"}

try:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.merge import merge as rio_merge
    from rasterio.transform import from_bounds, from_origin
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import Window
except ImportError as e:
    raise ImportError("This script needs rasterio: pip install rasterio") from e


# ----------------------------------------------------------------------- token
def token_expiry(token):
    try:
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        return json.loads(base64.urlsafe_b64decode(part)).get("exp")
    except Exception:
        return None


def get_token():
    token = os.environ.get("VEXCEL_TOKEN", "")
    for _ in range(3):
        exp = token_expiry(token)
        if exp and exp - 300 > time.time():
            print(f"Token OK, valid until {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(exp))}")
            return token
        if token:
            print("Token missing, malformed or expired. Copy the URL from "
                  "https://api.vexcelgroup.com/v2/ (not the Viewer app.vexcelgroup.com).")
        raw = getpass.getpass("Paste the Swagger URL (or token): ").strip()
        token = raw.split("token=")[-1].split("&")[0] if "token=" in raw else raw
    raise RuntimeError("No valid token – see step 2 in the description at the top.")


TOKEN = get_token()
session = requests.Session()


def call(path, params=None):
    """GET request with the token as `authtoken` query parameter and Bearer header."""
    params = {k: v for k, v in (params or {}).items() if v is not None}
    params["authtoken"] = TOKEN
    url = path if path.startswith("http") else f"{BASE_URL}/{path.lstrip('/')}"
    return session.get(url, params=params, headers={"Authorization": f"Bearer {TOKEN}"},
                       timeout=180)


# ------------------------------------------------------- endpoint parameters
SPEC = call("api-docs").json()
_PARAM_CACHE = {}
COLLECTION_RECORDS = {}


def _resolve(obj):
    while isinstance(obj, dict) and "$ref" in obj:
        node = SPEC
        for key in obj["$ref"].lstrip("#/").split("/"):
            node = node[key]
        obj = node
    return obj


def query_params(path):
    """Query parameters of a GET endpoint, read from Vexcel's API spec."""
    if path in _PARAM_CACHE:
        return _PARAM_CACHE[path]
    out = {}
    for p in SPEC["paths"][path]["get"].get("parameters", []):
        p = _resolve(p)
        if p.get("in") != "query":
            continue
        schema = _resolve(p.get("schema", p))
        enum = schema.get("enum") or _resolve(schema.get("items", {})).get("enum")
        out[p["name"]] = {"required": p.get("required", False),
                          "enum": [str(o) for o in enum] if enum else None,
                          "description": re.sub(r"<[^>]+>", " ", p.get("description") or "")}
    _PARAM_CACHE[path] = out
    return out


def describe(path):
    """Print an endpoint's parameters – useful when adapting this script."""
    print(f"\nGET {path}")
    for name, info in query_params(path).items():
        req = "required" if info["required"] else "optional"
        enum = f"  options: {info['enum']}" if info["enum"] else ""
        print(f"  {name:24} {req:8} {' '.join(info['description'].split())[:70]}{enum}")


def extract_path(product):
    return f"/{product}/extract"


def extract_params(product):
    """Parameters of /<product>/extract, or None if the API has no such endpoint."""
    path = extract_path(product)
    return query_params(path) if path in SPEC["paths"] else None


# --------------------------------------------------------------- collections
def _year(item):
    years = re.findall(r"(?:19|20)\d{2}", json.dumps(item))
    return max(map(int, years)) if years else 0


def find_collections(product, wkt, srid, layer):
    """Collections of one product and layer covering the area, newest first."""
    path = f"/{product}/collections"
    if path not in SPEC["paths"]:
        return []
    known = query_params(path)
    params = {k: v for k, v in {"wkt": wkt, "srid": srid, "layer": layer}.items()
              if k in known}
    r = call(path, params)
    if r.status_code == 404:                      # API answer for "nothing here"
        return []
    if r.status_code in (401, 403):
        print(f"No access to {product.upper()} data with your account ({r.status_code}).")
        return []
    if not r.ok:
        print(f"{product.upper()} collections lookup failed ({r.status_code}): {r.text[:200]}")
        return []
    data = r.json()
    if isinstance(data, dict):
        key = next((k for k in ("collections", "features", "data", "items", "results")
                    if k in data), None)
        items = data[key] if key else [data]
    else:
        items = data
    items = sorted((it.get("properties", it) if isinstance(it, dict) else it for it in items),
                   key=_year, reverse=True)
    COLLECTION_RECORDS[product] = items
    return [next((it[k] for k in ("collection", "collectionName", "name")
                  if isinstance(it, dict) and k in it), str(it)) for it in items]


def available_layers():
    """Layer names the API accepts, read from the parameter description."""
    desc = (extract_params("ortho") or {}).get("layer", {}).get("description", "")
    m = re.search(r"Allowed values\s*:?\s*(.+)", desc, re.I)
    listed = re.split(r"\.(?:\s|$)", m.group(1))[0] if m else ""   # up to the first sentence end
    names = re.findall(r"[a-z][a-z0-9-]*", listed)
    names = [n for n in dict.fromkeys(names) if n not in ("and", "or", "the", "e", "g")]
    return names or ["urban", "urban-r", "wide-area", "graysky"]


def check_coverage(wkt, srid):
    """List the orthoimage collections for LAYER; if there are none, show which
    layers do cover the area and stop with a clear message."""
    names = find_collections("ortho", wkt, srid, LAYER)
    if names:
        print(f"\n{len(names)} '{LAYER}' collection(s) cover the area (newest first):")
        for n in names:
            print(f"  {n}")
        return names
    print(f"\nNo '{LAYER}' imagery covers this area. Checking the other layers ...")
    found = {}
    for layer in available_layers():
        if layer != LAYER:
            other = find_collections("ortho", wkt, srid, layer)
            if other:
                found[layer] = other
                print(f"  {layer:12} {', '.join(other)}")
    if found:
        best = next(iter(found))
        raise SystemExit(f"\nSet LAYER = \"{best}\" (or another layer listed above) in the "
                         "settings and run again. Lower-resolution layers may also need a "
                         "lower ZOOM.")
    raise SystemExit("\nVexcel has no imagery for this area in any layer. Check the coverage "
                     "in the Vexcel Viewer, or choose another area.")


# ------------------------------------------------------------------ geometry
def densified_ring(b, n=50):
    """Points along the edges of bbox b (so curved edges transform correctly)."""
    x0, y0, x1, y1 = b
    t = np.linspace(0, 1, n)
    xs = np.concatenate([x0 + (x1 - x0) * t, np.full(n, x1), x1 - (x1 - x0) * t, np.full(n, x0)])
    ys = np.concatenate([np.full(n, y0), y0 + (y1 - y0) * t, np.full(n, y1), y1 - (y1 - y0) * t])
    return xs, ys


def transform_bounds(b, src, dst):
    xs, ys = Transformer.from_crs(src, dst, always_xy=True).transform(*densified_ring(b))
    return [float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))]


def center_latitude():
    lon0, lat0, lon1, lat1 = transform_bounds(BBOX, BBOX_EPSG, 4326)
    return (lat0 + lat1) / 2


def merc_res(zoom):
    """Exact Web-Mercator pixel size of the API tiles at this zoom (zoom 19: 0.2986 m;
    the Viewer rounds this to 30 cm)."""
    return MERC_RES0 / 2 ** zoom


def ground_resolution_m(zoom):
    """Ground size of the downloaded Web-Mercator pixels at the area's latitude."""
    return merc_res(zoom) * math.cos(math.radians(center_latitude()))


def res_m_of(res):
    """Pixel size in metres (converts degrees for EPSG:4326 output)."""
    return res * 111_320 if TARGET_EPSG == 4326 else res


def output_resolution(zoom):
    """Output pixel size in TARGET_EPSG units: the nominal resolution of the zoom
    level (7.5 cm at zoom 21, doubling per level), unless OUTPUT_RES is set."""
    if OUTPUT_RES:
        return OUTPUT_RES
    res_m = ZOOM_TO_CM[zoom] / 100
    return res_m / 111_320 if TARGET_EPSG == 4326 else res_m


def target_grid(zoom):
    """Exact output grid in TARGET_EPSG covering the requested rectangle.
    The grid starts at the upper-left corner of the area; the right and bottom
    edges are extended to the next whole pixel if needed."""
    b = BBOX if BBOX_EPSG == TARGET_EPSG else transform_bounds(BBOX, BBOX_EPSG, TARGET_EPSG)
    res = output_resolution(zoom)
    width = math.ceil((b[2] - b[0]) / res - 1e-6)
    height = math.ceil((b[3] - b[1]) / res - 1e-6)
    b = [b[0], b[3] - height * res, b[0] + width * res, b[3]]
    return from_origin(b[0], b[3], res, res), width, height, b, res


def download_bounds(target_b, zoom):
    """Web-Mercator rectangle that fully contains the target rectangle, plus a buffer."""
    m = transform_bounds(target_b, TARGET_EPSG, 3857)
    buf = BUFFER_PX * merc_res(zoom)
    return [m[0] - buf, m[1] - buf, m[2] + buf, m[3] + buf]


def split_bounds(m, zoom, max_px):
    """Split a Web-Mercator rectangle into tiles of at most max_px × max_px."""
    res = merc_res(zoom)
    w_px, h_px = (m[2] - m[0]) / res, (m[3] - m[1]) / res
    nx, ny = math.ceil(w_px / max_px), math.ceil(h_px / max_px)
    print(f"Download ≈ {w_px:,.0f} × {h_px:,.0f} px → {nx * ny} tile(s) ({nx} × {ny})")
    dx, dy = (m[2] - m[0]) / nx, (m[3] - m[1]) / ny
    return [(i, j, [m[0] + i * dx, m[1] + j * dy, m[0] + (i + 1) * dx, m[1] + (j + 1) * dy])
            for j in range(ny) for i in range(nx)]


def wkt_polygon(b, digits):
    x0, y0, x1, y1 = (round(v, digits) for v in b)
    return f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"


def wkt_3857(b):
    return wkt_polygon(b, 3)


def wkt_lonlat(b3857):
    return wkt_polygon(transform_bounds(b3857, 3857, 4326), 7)


def tile_extent(tile_b, srid):
    """Extent of a requested tile in the CRS it was requested in."""
    return transform_bounds(tile_b, 3857, 4326) if int(srid) == 4326 else tile_b


# ------------------------------------------------------------------ download
OPTIONAL_EXTRAS = ("attribution", "interpolation")


def request_params(product, tile_b, zoom, collection, bands=None):
    """Parameters for one /<product>/extract request (only names the API knows)."""
    known = extract_params(product)
    params = {"layer": LAYER, "collection": collection, "zoom": zoom, "bands": bands}
    srid_opts = (known.get("srid") or {}).get("enum")
    if srid_opts and "3857" not in srid_opts:
        params.update(wkt=wkt_lonlat(tile_b), srid=4326)
    else:
        params.update(wkt=wkt_3857(tile_b), srid=3857)
    if "crop" in known:
        params["crop"] = "bounding_box"           # full rectangle, no masking
    params["attribution"] = "false"               # no copyright text burnt into pixels
    interp = (known.get("interpolation") or {}).get("enum")
    if interp is None or "false" in interp:
        params["interpolation"] = "false"         # never upsample above the recorded zoom
    if product != "ortho":                        # elevation: ask for GeoTIFF values
        for name in ("image-format", "format", "output-format", "file-format"):
            tif = [o for o in ((known.get(name) or {}).get("enum") or []) if "tif" in o.lower()]
            if tif:
                params[name] = tif[0]
                break
    params = {k: v for k, v in params.items() if k in known and v is not None}
    missing = [n for n, i in known.items()
               if i["required"] and n not in params and n != "authtoken"]
    if missing:
        describe(extract_path(product))
        raise RuntimeError(f"{extract_path(product)} requires parameters this script "
                           f"does not set: {missing}")
    return params


def fetch(product, params):
    """Send an extract request; on 400, retry once without the optional extras."""
    r = call(extract_path(product), params)
    if r.status_code == 400 and any(k in params for k in OPTIONAL_EXTRAS):
        r = call(extract_path(product), {k: v for k, v in params.items()
                                         if k not in OPTIONAL_EXTRAS})
    return r


def fail(product, r):
    describe(extract_path(product))
    hint = (" – this zoom level may not exist for this collection; try a lower ZOOM"
            if r.status_code == 404 else "")
    raise RuntimeError(f"{product.upper()} request failed ({r.status_code}): "
                       f"{r.text[:300]}{hint}")


def save_response(r, out_stem):
    ctype = r.headers.get("Content-Type", "")
    if "json" in ctype:                           # some endpoints return a download URL
        data = r.json()
        url = next((data[k] for k in ("url", "downloadUrl", "href", "presignedUrl")
                    if isinstance(data, dict) and k in data), None)
        if not url:
            raise RuntimeError(f"Unexpected JSON response: {str(data)[:300]}")
        r = session.get(url, timeout=300)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
    ext = ".tif" if "tif" in ctype else ".png" if "png" in ctype else ".jpg"
    path = out_stem.with_suffix(ext)
    path.write_bytes(r.content)
    return path


def read_tile(path, extent, fallback_crs, zoom=None):
    """Read a downloaded tile. GeoTIFFs keep their georeference; plain JPEG/PNG
    tiles get it from the requested extent."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            data, crs, transform, nodata = src.read(), src.crs, src.transform, src.nodata
    h, w = data.shape[1:]
    if crs is None:
        if zoom and fallback_crs == "EPSG:3857":
            res = merc_res(zoom)
            exp_w, exp_h = (extent[2] - extent[0]) / res, (extent[3] - extent[1]) / res
            if abs(w - exp_w) > 2 or abs(h - exp_h) > 2:
                print(f"  WARNING {path.name}: {w}×{h} px, expected ≈{exp_w:.0f}×{exp_h:.0f} – "
                      "georeference may be slightly off")
        crs, transform = fallback_crs, from_bounds(*extent, w, h)
    return data, crs, transform, nodata


def check_irg(rgb, irg, name):
    """Sanity check: IRG band 2/3 must match RGB red/green, so band 1 is NIR."""
    r = np.corrcoef(rgb[0].ravel()[::97], irg[1].ravel()[::97])[0, 1]
    g = np.corrcoef(rgb[1].ravel()[::97], irg[2].ravel()[::97])[0, 1]
    print(f"  IRG check {name}: red↔IRG band 2 r={r:.2f}, green↔IRG band 3 r={g:.2f}")
    if min(r, g) < 0.8:
        print("  WARNING: IRG bands do not match RGB as expected – check the NIR band visually.")


def ortho_stack(tiles, first, name):
    """Stack RGB (+ IRG) tiles to blue, green, red (, NIR)."""
    rgb = tiles["rgb"][0]
    bands = [rgb[2], rgb[1], rgb[0]]              # blue, green, red
    if "irg" in tiles:
        irg = tiles["irg"][0]
        if irg.shape != rgb.shape:
            raise RuntimeError(f"RGB and IRG tiles differ in size: {rgb.shape} vs {irg.shape}")
        if first:
            check_irg(rgb, irg, name)
        bands.append(irg[0])                      # IRG band 1 = near-infrared
    return np.stack(bands), None


def elevation_band(tile, product):
    """One elevation band as float32 metres, invalid values set to ELEV_NODATA."""
    data, _, _, nodata = tile
    if data.shape[0] >= 3 and data.dtype == np.uint8:
        raise RuntimeError(f"The API returned a colour image for the {product.upper()}, not "
                           "elevation values. Check the format options in the parameter "
                           f"list (VERBOSE = True) for /{product}/extract.")
    if data.dtype == np.uint8:
        print(f"  WARNING: {product.upper()} delivered as 8-bit values – these may be "
              "scaled, not metres.")
    band = data[0].astype("float32")
    invalid = ~np.isfinite(band) | (band < -1000)
    if nodata is not None:
        invalid |= band == nodata
    band[invalid] = ELEV_NODATA
    return band[None], ELEV_NODATA


def cache_dir(product, collection, zoom, merc_b):
    collection = collection or "auto"
    # tied to the exact download extent, so a new BBOX never reuses old tiles
    key = hashlib.md5(f"{[round(v, 1) for v in merc_b]}{LAYER}".encode()).hexdigest()[:8]
    return DATA_DIR / NAME / "cache" / f"{product}_{collection}_z{zoom}_{key}"


def download_tiles(product, tiles, cache, collection, req_zoom, variants, zoom):
    """Download all tiles of one product; returns one GeoTIFF per tile."""
    raw_dir, tile_dir = cache / "raw", cache / "tiles"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tile_dir.mkdir(exist_ok=True)
    n_req = len(tiles) * len(variants)
    if n_req > 25 and input(f"{n_req} {product.upper()} requests – continue? [y/N] ").lower() != "y":
        raise SystemExit("Cancelled – choose a smaller BBOX or a lower ZOOM.")

    files = []
    for i, j, tb in tiles:
        stem = f"r{j}_c{i}"
        final = tile_dir / f"{stem}.tif"
        if final.exists():                        # resume interrupted runs
            files.append(final)
            continue
        parts = {}
        for v in variants:
            vname = v or product
            params = request_params(product, tb, req_zoom, collection, bands=v)
            raw = next(raw_dir.glob(f"{stem}_{vname}.*"), None)
            if raw is None:
                r = fetch(product, params)
                if not r.ok:
                    fail(product, r)
                raw = save_response(r, raw_dir / f"{stem}_{vname}")
                print(f"  saved {product} {raw.name} ({raw.stat().st_size / 1e6:.1f} MB)")
            srid = params.get("srid", 3857)
            parts[v] = read_tile(raw, tile_extent(tb, srid), f"EPSG:{srid}", req_zoom)
        if product == "ortho":
            stack, nodata = ortho_stack(parts, first=not files, name=stem)
        else:
            stack, nodata = elevation_band(parts[None], product)
        _, crs, transform, _ = parts[variants[0]]
        with rasterio.open(final, "w", driver="GTiff", width=stack.shape[2],
                           height=stack.shape[1], count=stack.shape[0], dtype=stack.dtype,
                           crs=crs, transform=transform, nodata=nodata,
                           compress="deflate") as dst:
            dst.write(stack)
        files.append(final)
    return files


# --------------------------------------------- merge, reproject, clip, check
def build_output(product, files, out_path, grid, zoom, band_names, nodata, collection):
    """Merge the tiles, reproject onto the target grid (= clip) and check the edges."""
    transform, width, height, target_b, res = grid
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic = out_path.with_name(out_path.stem + "_mosaic_tmp.tif")
    srcs = [rasterio.open(f) for f in files]
    try:   # rasterio >= 1.3 writes the mosaic straight to disk
        rio_merge(srcs, nodata=nodata, dst_path=mosaic,
                  dst_kwds={"compress": "deflate", "BIGTIFF": "IF_SAFER", "tiled": True,
                            "blockxsize": 512, "blockysize": 512})
    except TypeError:
        arr, tr = rio_merge(srcs, nodata=nodata)
        with rasterio.open(mosaic, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                           count=arr.shape[0], dtype=arr.dtype, crs=srcs[0].crs,
                           transform=tr, nodata=nodata, compress="deflate") as dst:
            dst.write(arr)
    for s in srcs:
        s.close()

    with rasterio.open(mosaic) as src:
        # the mosaic must cover the whole target rectangle, or the edges stay empty
        need = transform_bounds(target_b, TARGET_EPSG, src.crs)
        b = src.bounds
        if need[0] < b.left or need[1] < b.bottom or need[2] > b.right or need[3] > b.top:
            print("WARNING: download does not fully cover the area – increase BUFFER_PX")
        # averaging when the output pixels are larger than the downloaded ones
        # (the usual case away from the equator) avoids aliasing
        coarser = res_m_of(res) > ground_resolution_m(zoom) * 1.01
        resampling = Resampling.average if coarser else Resampling.bilinear
        is_float = np.dtype(src.dtypes[0]).kind == "f"
        profile = {"driver": "GTiff", "width": width, "height": height, "count": src.count,
                   "dtype": src.dtypes[0], "crs": f"EPSG:{TARGET_EPSG}",
                   "transform": transform, "nodata": nodata, "compress": "deflate",
                   "predictor": 3 if is_float else 2, "tiled": True,
                   "photometric": "MINISBLACK", "BIGTIFF": "IF_SAFER"}
        with rasterio.open(out_path, "w", **profile) as dst:
            for k in range(1, src.count + 1):
                reproject(rasterio.band(src, k), rasterio.band(dst, k),
                          src_nodata=nodata, dst_nodata=nodata, resampling=resampling)
                dst.set_band_description(k, band_names[k - 1])
            dst.update_tags(SOURCE="Vexcel Imaging (licensed, do not redistribute)",
                            PRODUCT=product, COLLECTION=str(collection), LAYER=LAYER,
                            ZOOM=str(zoom), BANDS=",".join(band_names[:src.count]),
                            **({"UNITS": "metres"} if product != "ortho" else {}))
    mosaic.unlink()
    check_edges(out_path, nodata)
    return out_path


def check_edges(path, nodata=None):
    """Count pixels along the four edges that contain no data."""
    with rasterio.open(path) as src:
        w, h = src.width, src.height
        edges = [Window(0, 0, w, 1), Window(0, h - 1, w, 1),
                 Window(0, 0, 1, h), Window(w - 1, 0, 1, h)]
        empty = 0
        for win in edges:
            data = src.read(window=win)
            mask = (data == 0).all(axis=0) if nodata is None else (data == nodata).any(axis=0)
            empty += int(mask.sum())
        total = 2 * (w + h)
        print(f"\n{path.name}: {src.count} band(s) {list(src.descriptions)}, {w}×{h} px, "
              f"{src.crs}, pixel {src.res[0]:.3g}")
        print(f"  bounds: {tuple(round(v, 2) for v in src.bounds)}")
        if empty / total > 0.01:
            print(f"  WARNING: {empty} of {total} edge pixels are empty – increase BUFFER_PX")
        else:
            print("  Edge check OK – no empty borders.")


# ------------------------------------------------- elevation (DSM, DTM, nDSM)
def elevation_stats(path):
    """Print height statistics as a plausibility check."""
    with rasterio.open(path) as src:
        a = src.read(1)
        nd = src.nodata
    valid = a[a != nd] if nd is not None else a.ravel()
    if valid.size == 0:
        print("  WARNING: no valid heights in this file.")
        return
    lo, med, hi = np.percentile(valid, (0, 50, 100))
    print(f"  heights: min {lo:.1f} m, median {med:.1f} m, max {hi:.1f} m; "
          f"{100 * (1 - valid.size / a.size):.1f}% no data")


def finest_zoom(product, collection, merc_b):
    """Finest zoom ≤ ZOOM at which the product exists (tested with a small request).
    Returns (zoom sent to the API, zoom for the output grid), or None."""
    known = extract_params(product)
    cx, cy = (merc_b[0] + merc_b[2]) / 2, (merc_b[1] + merc_b[3]) / 2
    zooms = range(ZOOM, 14, -1) if "zoom" in known else [None]
    for z in zooms:
        half = 128 * merc_res(z or ZOOM)              # 256 × 256 px test area
        probe_b = [cx - half, cy - half, cx + half, cy + half]
        params = request_params(product, probe_b, z, collection)
        r = fetch(product, params)
        if r.ok and z is not None:
            return z, z
        if r.ok:  # API has no zoom parameter: derive the level from the delivered pixels
            probe_dir = cache_dir(product, collection, "probe", merc_b)
            probe_dir.mkdir(parents=True, exist_ok=True)
            probe = save_response(r, probe_dir / "probe")
            with rasterio.open(probe) as src:
                res_x, crs = src.res[0], src.crs
            lat = center_latitude()
            if crs is None or crs.to_epsg() == 3857:
                ground = res_x * math.cos(math.radians(lat))
            elif crs.is_geographic:
                ground = res_x * 111_320 * math.cos(math.radians(lat))
            else:
                ground = res_x
            z_eff = round(math.log2(MERC_RES0 * math.cos(math.radians(lat)) / ground))
            return None, max(15, min(ZOOM, z_eff))
        if r.status_code in (401, 403):
            print(f"No access to {product.upper()} data with your account ({r.status_code}).")
            return None
        if r.status_code not in (400, 404):
            fail(product, r)
    return None


def run_elevation_product(product, plan_):
    print(f"\n=== {PRODUCT_LABELS[product]} ===")
    if extract_params(product) is None:
        print(f"The API offers no {product.upper()} – skipped.")
        return None
    names = find_collections(product, plan_["wkt_ll"], 4326, LAYER)
    if not names:
        print(f"No {product.upper()} covers this area in layer '{LAYER}' – skipped.")
        return None
    if "collection" in extract_params(product):
        coll = plan_["collection"] if plan_["collection"] in names else names[0]
        print(f"Using collection: {coll}  (available: {', '.join(names)})")
        if plan_["collection"] and coll != plan_["collection"]:
            print(f"  NOTE: differs from the orthoimage collection {plan_['collection']}.")
    else:
        coll = None   # the API chooses the source itself
        print(f"The {product.upper()} request takes no collection – Vexcel selects the "
              f"source itself (covering collections: {', '.join(names)}).")
    if VERBOSE and COLLECTION_RECORDS.get(product):
        print("  collection metadata:", json.dumps(COLLECTION_RECORDS[product][0])[:600])
    found = finest_zoom(product, coll, plan_["merc_b"])
    if found is None:
        print(f"No {product.upper()} data at zoom {ZOOM} or below – skipped.")
        return None
    req_zoom, zoom = found
    if zoom < ZOOM:
        print(f"{product.upper()} is available up to zoom {zoom} "
              f"({ZOOM_TO_CM[zoom]:g} cm) – using that.")
    grid = target_grid(zoom)
    merc_b = download_bounds(grid[3], zoom)
    tiles = split_bounds(merc_b, zoom, MAX_TILE_PX)
    files = download_tiles(product, tiles, cache_dir(product, coll, zoom, merc_b), coll,
                           req_zoom, [None], zoom)
    label = f"{coll}_" if coll else ""
    out = DATA_DIR / NAME / f"{NAME}_{label}{product}_z{zoom}_epsg{TARGET_EPSG}.tif"
    out = build_output(product, files, out, grid, zoom, (f"{product}_m",), ELEV_NODATA, coll)
    elevation_stats(out)
    return out


def build_ndsm(dsm_path, dtm_path):
    """nDSM = DSM − DTM on the DSM grid (heights of buildings and trees above ground)."""
    with rasterio.open(dsm_path) as s:
        dsm, profile = s.read(1), s.profile.copy()
    with rasterio.open(dtm_path) as t:
        dtm = np.full_like(dsm, ELEV_NODATA)
        reproject(t.read(1), dtm, src_transform=t.transform, src_crs=t.crs,
                  dst_transform=profile["transform"], dst_crs=profile["crs"],
                  src_nodata=ELEV_NODATA, dst_nodata=ELEV_NODATA,
                  resampling=Resampling.bilinear)
    valid = (dsm != ELEV_NODATA) & (dtm != ELEV_NODATA)
    diff = (dsm - dtm)[valid]
    if diff.size:
        below = float(np.mean(diff < -0.5))
        print(f"\nDSM − DTM before clipping: median {np.median(diff):.2f} m, "
              f"{below:.0%} of pixels more than 0.5 m below 0")
        if below > 0.10:
            print("  WARNING: on open ground DSM and DTM should agree; many negative values "
                  "suggest the two models are offset vertically or come from different "
                  "sources. Check them before using the nDSM.")
    ndsm = np.where(valid, np.clip(dsm - dtm, 0, None), ELEV_NODATA).astype("float32")
    out = dsm_path.with_name(dsm_path.name.replace("_dsm_", "_ndsm_"))
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(ndsm, 1)
        dst.set_band_description(1, "ndsm_m")
        dst.update_tags(SOURCE="Vexcel Imaging (licensed, do not redistribute)",
                        PRODUCT="ndsm = dsm - dtm", UNITS="metres")
    print(f"\n{out.name}: object heights above ground")
    elevation_stats(out)
    return out


# ----------------------------------------------------------------------- steps
def plan():
    """Check the settings, set up the output grid and find the collection."""
    products = [p.lower() for p in PRODUCTS]
    unknown = set(products) - set(PRODUCT_LABELS)
    if unknown:
        raise ValueError(f"Unknown product(s) {sorted(unknown)} – use 'ortho', 'dsm', 'dtm'.")
    if VERBOSE:
        for p in products:
            if extract_params(p) is not None:
                describe(extract_path(p))
    grid = target_grid(ZOOM)
    _, width, height, target_b, res = grid
    print(f"Products: {', '.join(products)}")
    print(f"Grid at zoom {ZOOM}: {width} × {height} px of {res_m_of(res) * 100:g} cm, "
          f"EPSG:{TARGET_EPSG}, bounds {tuple(round(v, 2) for v in target_b)}")
    merc_b = download_bounds(target_b, ZOOM)
    wkt_ll = wkt_lonlat(merc_b)
    collection = COLLECTION
    if "ortho" in products:
        names = check_coverage(wkt_ll, 4326)
        if collection is None:
            collection = names[0]
        elif collection not in names:
            raise SystemExit(f"Collection '{collection}' does not cover this area in layer "
                             f"'{LAYER}'. Choose one of the collections listed above.")
        print(f"Using collection: {collection}")
    return {"products": products, "grid": grid, "merc_b": merc_b, "wkt_ll": wkt_ll,
            "collection": collection}


def run_ortho(plan_):
    """Download and build the 4-band orthoimage."""
    if "ortho" not in plan_["products"]:
        return None
    print(f"\n=== {PRODUCT_LABELS['ortho']} ===")
    include_nir = INCLUDE_NIR
    if include_nir and "irg" not in (extract_params("ortho").get("bands", {}).get("enum") or []):
        print("This API offers no 'irg' bands – downloading RGB only.")
        include_nir = False
    variants = ["rgb", "irg"] if include_nir else ["rgb"]
    coll, merc_b = plan_["collection"], plan_["merc_b"]
    tiles = split_bounds(merc_b, ZOOM, MAX_TILE_PX)
    files = download_tiles("ortho", tiles, cache_dir("ortho", coll, ZOOM, merc_b), coll,
                           ZOOM, variants, ZOOM)
    out = DATA_DIR / NAME / f"{NAME}_{coll}_ortho_z{ZOOM}_epsg{TARGET_EPSG}.tif"
    return build_output("ortho", files, out, plan_["grid"], ZOOM,
                        BAND_NAMES[:4 if include_nir else 3], None, coll)


def run_elevation(plan_):
    """Download and build DSM / DTM (where available) and the nDSM."""
    out = {}
    for product in ("dsm", "dtm"):
        if product in plan_["products"]:
            path = run_elevation_product(product, plan_)
            if path:
                out[product] = path
    if NDSM and "dsm" in out and "dtm" in out:
        out["ndsm"] = build_ndsm(out["dsm"], out["dtm"])
    return out


def preview(path, max_size=800):
    """True-colour preview (bands 3-2-1) in a notebook."""
    try:
        from PIL import Image
        with rasterio.open(path) as src:
            s = max(src.width, src.height) / max_size
            arr = src.read([3, 2, 1], out_shape=(3, int(src.height / s), int(src.width / s)))
        display(Image.fromarray(arr.transpose(1, 2, 0)))  # noqa: F821 – Jupyter/Colab
    except Exception as e:
        print(f"(no preview: {e})")


# ----------------------------------------------------------------------- run
def main():
    plan_ = plan()
    results = {}
    ortho = run_ortho(plan_)
    if ortho:
        results["ortho"] = ortho
    results.update(run_elevation(plan_))
    print("\nOutput files:")
    for k, v in results.items():
        print(f"  {k:6} {v}")
    if ortho:
        preview(ortho)
    return results


if __name__ == "__main__":
    main()
