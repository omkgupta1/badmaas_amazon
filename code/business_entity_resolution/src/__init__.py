"""Business entity resolution pipeline (ML Challenge 2026)."""

# OpenMP on macOS: sparse_dot_topn ships its own libomp and segfaults when LightGBM's libomp
# (e.g. Homebrew's) was loaded first. Loading sparse_dot_topn first makes LightGBM reuse the
# same runtime (verified: sdt->lgb works, lgb->sdt crashes), so import it before anything else.
try:  # pragma: no cover - platform dependent
    import sparse_dot_topn as _sparse_dot_topn  # noqa: F401
except ImportError:
    pass

__version__ = "1.0.0"
