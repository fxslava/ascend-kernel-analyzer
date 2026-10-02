"""Source frontend: preprocessing before the C++ grammar sees a file."""

from .ascend_preprocessor import (
    AscendCPreprocessor,
    PreprocessResult,
    normalize_cce_syntax,
    preprocess_source,
)

__all__ = [
    "AscendCPreprocessor",
    "PreprocessResult",
    "normalize_cce_syntax",
    "preprocess_source",
]
