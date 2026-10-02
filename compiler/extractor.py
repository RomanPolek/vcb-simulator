import base64
import json

import zstandard as zstd


def load_project(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_payload(base64_str):
    compressed_data = base64.b64decode(base64_str)
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(compressed_data)


def extract_layer_from_project(project_data, layer_index):
    """
    Reads a loaded VCB project dict, extracts the base64 encoded layer at the specified
    index, and decompresses the Zstandard payload.

    Args:
        project_data (dict): Loaded .vcb JSON contents.
        layer_index (int): The index of the layer to decode (starting at 0).

    Returns:
        bytes: The decompressed raw byte data of the layer.
    """
    # 2. Validate the keys and bounds
    if "layers" not in project_data:
        raise KeyError("Could not find 'layers' array in project data.")

    layers = project_data["layers"]

    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"Layer index {layer_index} is out of bounds. File has {len(layers)} layer(s).")

    # 3. Extract the base64 string for the specific layer
    layer_base64_str = layers[layer_index]

    # Note: If Virtual Circuit Board appends custom footer bytes to layers
    # (like it does with vmem: struct.pack('<II', 2, len(raw_bytes))),
    # zstandard usually safely ignores trailing garbage during a standard decompress run.
    return decode_payload(layer_base64_str)


def extract_layer(filepath, layer_index):
    project_data = load_project(filepath)
    return extract_layer_from_project(project_data, layer_index)


def extract_vmem_from_project(project_data):
    vmem_base64 = project_data.get("vmem_data")
    if not vmem_base64:
        return b""
    return decode_payload(vmem_base64)
