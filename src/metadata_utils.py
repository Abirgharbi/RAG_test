from _future__ import annotations
from typing import List, Dict, Any


def normalize_metadata_for_Chroma(metadata: Dict[str, Any]) -> Dict[str, str]:
    normalized = Dict[str,Any] = {}
    for key, value in metadata.items():
       if value is None:
              normalized[key] = none
              continue
        if isinstance(value, list):
            if not value:
                normalized[key] = "none"
            else:
                normalized[key] = ";".join(str(v) for v in value)
       