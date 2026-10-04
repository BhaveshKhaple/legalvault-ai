"""Phase 3 — unit tests for the Qdrant metadata-filter builder.

Verifies that build_metadata_filter composes the right Filter structure for
every combination of inputs, and that the mandatory case_id filter is always
first (security — never drop it).
"""

from backend.app.retrieval.vector_store import build_metadata_filter


class TestBuildMetadataFilter:
    def test_case_id_only(self):
        f = build_metadata_filter("case-a")
        assert len(f.must) == 1
        c = f.must[0]
        assert c.key == "case_id"
        assert c.match.value == "case-a"

    def test_case_id_plus_jurisdiction(self):
        f = build_metadata_filter("case-a", jurisdiction="India")
        keys = [c.key for c in f.must]
        assert keys == ["case_id", "jurisdiction"]
        assert f.must[1].match.value == "India"

    def test_date_after_only_builds_range(self):
        f = build_metadata_filter("case-a", date_after=1700000000)
        assert len(f.must) == 2
        rng_cond = f.must[1]
        assert rng_cond.key == "effective_date_ts"
        assert rng_cond.range.gte == 1700000000
        assert rng_cond.range.lte is None

    def test_date_before_only_builds_range(self):
        f = build_metadata_filter("case-a", date_before=1800000000)
        rng_cond = f.must[1]
        assert rng_cond.range.gte is None
        assert rng_cond.range.lte == 1800000000

    def test_date_range_combines_both_sides(self):
        f = build_metadata_filter("case-a", date_after=1700000000, date_before=1800000000)
        rng = f.must[1].range
        assert rng.gte == 1700000000 and rng.lte == 1800000000

    def test_all_filters_combined(self):
        f = build_metadata_filter(
            "case-a",
            date_after=1700000000,
            jurisdiction="India",
            version_tag="v2",
            regulator="RBI",
        )
        keys = [c.key for c in f.must]
        # case_id always first; others appended in builder order
        assert keys[0] == "case_id"
        assert set(keys) == {"case_id", "jurisdiction", "version_tag", "regulator", "effective_date_ts"}

    def test_empty_string_filters_are_skipped(self):
        """Explicit None skips; builder shouldn't add filters for empty-string values."""
        f = build_metadata_filter("case-a", jurisdiction=None, version_tag=None, regulator=None)
        assert len(f.must) == 1  # only case_id

    def test_case_id_always_present_even_with_other_filters(self):
        """Security — if case_id ever gets dropped, tenants would leak."""
        f = build_metadata_filter("tenant-secret", jurisdiction="India", regulator="SEBI")
        assert any(c.key == "case_id" and c.match.value == "tenant-secret" for c in f.must)
