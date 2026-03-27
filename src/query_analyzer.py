from src.services.stm32cube_preprocessing_heuristics import (
    detect_component,
    detect_layer,
)

def analyze_query(query):
    return {
        "component": detect_component(query),
        "layer": detect_layer(query),
    }
