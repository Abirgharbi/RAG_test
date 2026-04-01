def normalize_metadata_for_chroma(metadata):
    import json
    normalized = {}

    for key, value in metadata.items():

        if value is None:
            normalized[key] = "none"
            continue

     
        if isinstance(value, list):
            if len(value) == 0:
                normalized[key] = "none"
            else:
                normalized[key] = ";".join(str(v) for v in value)
            continue

        if isinstance(value, dict):
            normalized[key] = json.dumps(value, ensure_ascii=False)
            continue

        if isinstance(value, (int, float, bool)):
            normalized[key] = value
            continue

        normalized[key] = str(value)

    return normalized
