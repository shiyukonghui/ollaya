#!/bin/sh
# Install the *locally exported* NeoHorse-Jev-4B bundle into an ollaya model store, so that
# ollaya resolves it by name like any pulled model:
#
#     OLLAYA_MODELS=<store> ollaya run neohorse --image out/neohorse/demo.png \
#         --questions '{"colour": {...}}' "A 256x240 image."
#
# What it does (idempotent: run it as often as you like):
#
#   1. Reads the export in $NEOHORSE_OUT (default <repo>/out/neohorse):
#      model.onnx, vision.onnx, tokenizer.json, decision.json, calibration.json and the four
#      weight files (three backbone shards + pointer_head.safetensors).
#   2. Rewrites the two weightless graphs' ONNX external-data `location` fields from the bundle's
#      file names to the blob file names the store uses (`sha256-<hex>`). ollaya's store keeps
#      every file in one flat `blobs/` directory, and ONNX Runtime resolves external data relative
#      to the graph's own directory, so a graph in `blobs/` can only find weights whose file name
#      *is* the blob name. This is exactly what `convert/ollaya_convert/package.py`'s
#      `graph_from_wl` does when the same bundle is published to a registry (a stdlib-only
#      rewriter is inlined below so the script needs no Python packages).
#   3. Hard-links the four weight files into `blobs/sha256-<hex>` -- never copies the ~9 GB. The
#      digests are checked against the bundle's own SHA256SUMS values first.
#   4. Writes the derived files (graphs, config, tokenizer, decision, calibration) as
#      content-addressed blobs, and the Docker-v2 manifest to
#      `manifests/<host>/<namespace>/<neohorse>/<tag>` -- i.e. <host>/library/neohorse:latest, so
#      the bare name `neohorse` resolves to it (ModelName::parse; host default `ollaya.dev`).
#
# Settings, all optional:
#   NEOHORSE_OUT      the export directory            (default <repo>/out/neohorse)
#   OLLAYA_MODELS     the model store to install into (default <repo>/out/ollaya-models; it must
#                     be on the same volume as $NEOHORSE_OUT, since the weights are hard links)
#   OLLAYA_REGISTRY   the registry host recorded in the manifest path (default ollaya.dev, as
#                     ollaya's ModelName uses; set it if you run with OLLAYA_REGISTRY set)
#   PYTHON            a Python 3 interpreter to run the inlined installer (default `python`)
#   NEOHORSE_COPY=1   copy the weights instead of hard-linking them (needs ~9 GB of free space)
#   NEOHORSE_SKIP_VERIFY=1  don't check the weight digests against the pinned SHA256SUMS values

set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)
NEOHORSE_OUT=${NEOHORSE_OUT:-$repo/out/neohorse}
OLLAYA_MODELS=${OLLAYA_MODELS:-$repo/out/ollaya-models}
OLLAYA_REGISTRY=${OLLAYA_REGISTRY:-ollaya.dev}
PYTHON=${PYTHON:-python}
export NEOHORSE_OUT OLLAYA_MODELS OLLAYA_REGISTRY

command -v "$PYTHON" >/dev/null 2>&1 ||
    { echo "ERROR: no $PYTHON; set PYTHON to a Python 3 interpreter" >&2; exit 1; }

exec "$PYTHON" - <<'PYEOF'
import hashlib, json, os, re, sys

SRC = os.path.abspath(os.environ["NEOHORSE_OUT"])
STORE = os.path.abspath(os.environ["OLLAYA_MODELS"])
HOST = os.environ["OLLAYA_REGISTRY"]
NAMESPACE, MODEL, TAG = "library", "neohorse", "latest"

# The digests the NeoHorse-Jev-4B bundle publishes in its own SHA256SUMS (the export hard-links
# those very files, so a mismatch means the export was made from a different bundle).
EXPECTED = {
    "model-00001-of-00003.safetensors": "7d1adbb748ff60a91b3b6ffba1ff70bfcab855ff2b8cf5e33f9c6bc1ff13cb7e",
    "model-00002-of-00003.safetensors": "c37c278c3977b16b7358421a10b5805615a0f393f19f56b540644dded4d0e6c3",
    "model-00003-of-00003.safetensors": "475b9a4b012cc888da1d6575746d2b329a944a6107f8e20b1dcb83b116bf1b1d",
    "pointer_head.safetensors": "467ae48977b5c4bf87dd1db29021199e0c3401c52c5b57a1eccde679c0bade26",
}
WEIGHTS = list(EXPECTED)
DERIVED = ["tokenizer.json", "decision.json", "calibration.json"]

MEDIA = {
    "config": "application/vnd.ollaya.config.v1+json",
    "graph": "application/vnd.ollaya.graph.onnx",
    "weights": "application/vnd.ollaya.weights",
    "tokenizer": "application/vnd.ollaya.tokenizer",
    "decision": "application/vnd.ollaya.decision",
    "calibration": "application/vnd.ollaya.calibration",
}
MANIFEST_V2 = "application/vnd.docker.distribution.manifest.v2+json"


def die(msg):
    sys.exit("ERROR: " + msg)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# --- minimal protobuf reader/writer, only for the paths an ONNX graph stores external data on ---

def _varint(buf, i):
    r = s = 0
    while True:
        b = buf[i]; i += 1
        r |= (b & 0x7F) << s
        if not b & 0x80:
            return r, i
        s += 7


def _enc_varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _parse(buf):
    fs = []; i = 0; n = len(buf)
    while i < n:
        tag, i = _varint(buf, i)
        f, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v = buf[i:i + 8]; i += 8
        elif wt == 2:
            l, i = _varint(buf, i); v = buf[i:i + l]; i += l
        elif wt == 5:
            v = buf[i:i + 4]; i += 4
        else:
            raise ValueError("unsupported wire type %d (field %d)" % (wt, f))
        fs.append([f, wt, v])
    return fs


def _encode(fs):
    out = bytearray()
    for f, wt, v in fs:
        out += _enc_varint((f << 3) | wt)
        if wt == 0:
            out += _enc_varint(v)
        elif wt in (1, 5):
            out += v
        else:
            out += _enc_varint(len(v)); out += v
    return bytes(out)


def rewrite_graph(buf, mapping):
    """Rename external-data `location`s in a weightless ONNX graph. Returns (bytes, count).

    ModelProto.graph = 7, GraphProto.initializer = 5, TensorProto.external_data = 13,
    StringStringEntryProto.key = 1 / .value = 2.
    """
    n = [0]

    def entry(blob):
        fields = _parse(blob)
        is_location = False
        for kv in fields:
            if kv[0] == 1 and kv[1] == 2 and kv[2] == b"location":
                is_location = True
            elif kv[0] == 2 and kv[1] == 2 and is_location:
                name = kv[2].decode()
                if name in mapping:
                    kv[2] = mapping[name].encode(); n[0] += 1
                break
        return _encode(fields)

    def tensor(blob):
        fields = _parse(blob)
        for f in fields:
            if f[0] == 13 and f[1] == 2:
                f[2] = entry(f[2])
        return _encode(fields)

    top = _parse(buf)
    for f in top:
        if f[0] == 7 and f[1] == 2:
            graph = _parse(f[2])
            for x in graph:
                if x[0] == 5 and x[1] == 2:
                    x[2] = tensor(x[2])
            f[2] = _encode(graph)
    return _encode(top), n[0]


# --- the store ---------------------------------------------------------------------------------

blobs = os.path.join(STORE, "blobs")
os.makedirs(blobs, exist_ok=True)


def put_bytes(media_type, data, annotations=None):
    """Write `data` as a content-addressed blob and return its layer descriptor."""
    h = hashlib.sha256(data).hexdigest()
    path = os.path.join(blobs, "sha256-" + h)
    if os.path.exists(path) and os.path.getsize(path) != len(data):
        # The name is the content's own hash, so a differently sized file there is corrupt.
        print("WARNING: %s has the wrong size; replacing it" % path)
    if not os.path.exists(path) or os.path.getsize(path) != len(data):
        tmp = path + ".tmp-%d" % os.getpid()
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    d = {"mediaType": media_type, "digest": "sha256:" + h, "size": len(data),
         "urls": [], "annotations": annotations or {}}
    return d


def put_weights(path, digest):
    """Hard-link one weight file into blobs/ under the name its graph asks for."""
    size = os.path.getsize(path)
    dst = os.path.join(blobs, "sha256-" + digest)
    if os.path.exists(dst):
        if os.path.getsize(dst) != size:
            die("%s exists with the wrong size; remove it and re-run" % dst)
        return dst, False
    if os.environ.get("NEOHORSE_COPY"):
        with open(path, "rb") as s, open(dst + ".tmp", "wb") as d:
            for b in iter(lambda: s.read(1 << 22), b""):
                d.write(b)
        os.replace(dst + ".tmp", dst)
    else:
        try:
            os.link(path, dst)
        except OSError as e:
            die("cannot hard-link %s -> %s (%s); put OLLAYA_MODELS on the same volume as "
                "NEOHORSE_OUT, or set NEOHORSE_COPY=1" % (path, dst, e))
    return dst, True


def main():
    missing = [f for f in ["model.onnx", "vision.onnx"] + DERIVED + WEIGHTS
               if not os.path.isfile(os.path.join(SRC, f))]
    if missing:
        die("%s is not a NeoHorse export; missing %s" % (SRC, ", ".join(missing)))
    print("export: %s\nstore:  %s" % (SRC, STORE))

    digests = {}
    for name in WEIGHTS:
        path = os.path.join(SRC, name)
        h = sha256_file(path)
        want = EXPECTED[name]
        if h != want and not os.environ.get("NEOHORSE_SKIP_VERIFY"):
            die("%s: sha256 %s, expected %s (set NEOHORSE_SKIP_VERIFY=1 to accept)"
                % (name, h, want))
        digests[name] = h
        print("  %-32s %s  %6.2f GB" % (name, h[:16] + "...", os.path.getsize(path) / 1e9))

    # The graphs name the weights by the blob file names (`sha256-<hex>`) they will have here.
    mapping = {name: "sha256-" + h for name, h in digests.items()}
    layers = []
    for name, annotations in (("model.onnx", {"org.ollaya.precision": "fp32"}),
                              ("vision.onnx", {"org.ollaya.precision": "fp32",
                                               "org.ollaya.graph": "vision"})):
        data, n = rewrite_graph(open(os.path.join(SRC, name), "rb").read(), mapping)
        if n == 0:
            die("%s references no weight file named in the export" % name)
        layers.append(put_bytes(MEDIA["graph"], data, annotations))
        print("  %-32s %6.1f MB, %d external references" % (name, len(data) / 2 ** 20, n))

    for name in WEIGHTS:
        dst, made = put_weights(os.path.join(SRC, name), digests[name])
        if made:
            print("  hard-linked %s -> %s" % (name, os.path.basename(dst)))
        layers.append({"mediaType": MEDIA["weights"], "digest": "sha256:" + digests[name],
                       "size": os.path.getsize(os.path.join(SRC, name)), "urls": [],
                       "annotations": {}})

    for name, media in (("tokenizer.json", "tokenizer"), ("decision.json", "decision"),
                        ("calibration.json", "calibration")):
        layers.append(put_bytes(MEDIA[media], open(os.path.join(SRC, name), "rb").read()))

    decision = json.load(open(os.path.join(SRC, "decision.json")))
    config = put_bytes(MEDIA["config"], json.dumps({
        "model_format": "onnx",
        "family": decision.get("family", MODEL),
        "parameter_size": "4B",
        "context_length": decision.get("max_ctx_tokens", 0),
        "languages": [],
        "description": "NeoHorse-Jev-4B: merged Qwen3.5-4B multimodal backbone + pointer head, "
                       "exported locally by convert/ollaya_convert/families/neohorse",
        "source": "local export %s" % SRC,
        "license": "see the NeoHorse-Jev-4B bundle",
    }, indent=2).encode())

    manifest = {"schemaVersion": 2, "mediaType": MANIFEST_V2, "config": config, "layers": layers}
    # `host_dir()`: the registry host without a scheme, ':' replaced by '_'.
    host_dir = re.sub(r"^[a-z]+://", "", HOST).replace(":", "_")
    path = os.path.join(STORE, "manifests", host_dir, NAMESPACE, MODEL, TAG)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp-%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)
    print("\nmanifest: %s\n          %d layers, %s:%s" % (
        path, len(layers), MODEL, TAG))
    print("\nrun it with:\n  OLLAYA_MODELS=%s ollaya run neohorse ..." % STORE)


main()
PYEOF
