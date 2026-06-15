import pytest
from unittest.mock import MagicMock
from django.contrib.auth import get_user_model
from django.db import OperationalError
from django.db.models import Value
from django_neural_feed.feeds import BaseNeuralFeed
from django_neural_feed.models import UserFeedProfile
from unittest.mock import MagicMock, patch
from unittest.mock import patch, PropertyMock, MagicMock
from django_neural_feed.conf import AppSettings

# --- 1. Line 41: Fallback via getattr ---


def test_real_base_feed_get_setting_fallback():
    """Covers line 41 by requesting an attribute inherited from object."""
    # __sizeof__ is missing from BaseNeuralFeed.__dict__, forcing line 41
    assert BaseNeuralFeed.get_setting("__sizeof__") is not None


# --- 2. Lines 45-49: get_candidates with exclusions ---


@pytest.mark.django_db
def test_get_candidates_with_and_without_exclusions():
    """Covers lines 45-49 using a real database model."""
    from tests.models import TestArticle

    class TargetFeed(BaseNeuralFeed):
        @classmethod
        def get_setting(cls, attr_name: str):
            if attr_name == "content_django_model":
                return TestArticle
            return super().get_setting(attr_name)

    a1 = TestArticle.objects.create(title="A1", embedding=[1, 0, 0])
    a2 = TestArticle.objects.create(title="A2", embedding=[0, 1, 0])

    # Covers False branch transition
    qs_all = TargetFeed.get_candidates(user=None, queryset=None, excluded_ids=None)
    assert qs_all.count() == 2

    # Covers True branch filtering
    qs_excluded = TargetFeed.get_candidates(user=None, queryset=TestArticle.objects.all(), excluded_ids=[a1.id])  # type: ignore
    assert qs_excluded.count() == 1


# --- 3. Lines 55-77 & 68: calculate_user_embedding Math & Guards ---


def test_calculate_user_embedding_logic():
    """Covers lines 55-77, handling norm > 0, norm == 0, and empty states."""

    class MathTestingFeed(BaseNeuralFeed):
        @classmethod
        def get_setting(cls, attr_name: str):
            return 5

    # Branch: norm > 0
    mock_qs = MagicMock()
    mock_qs.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = [
        [3.0, 0.0, 4.0]
    ]
    assert pytest.approx(MathTestingFeed.calculate_user_embedding(mock_qs)) == [
        0.6,
        0.0,
        0.8,
    ]

    # Branch: norm == 0
    mock_qs_zero = MagicMock()
    mock_qs_zero.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = [
        [0.0, 0.0, 0.0]
    ]
    assert MathTestingFeed.calculate_user_embedding(mock_qs_zero) == [0.0, 0.0, 0.0]

    # Line 68: Empty sequence early exit
    mock_qs_empty = MagicMock()
    mock_qs_empty.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = (
        []
    )
    assert MathTestingFeed.calculate_user_embedding(mock_qs_empty) is None


# --- 4. Lines 93 & 102-109: Guards and DB Exceptions ---


def test_get_user_profile_vector_guards():
    """Covers line 93 when feed_id is missing."""

    class BrokenFeed(BaseNeuralFeed):
        @classmethod
        def get_setting(cls, attr_name: str):
            return None

    assert BrokenFeed.get_user_vector(user=MagicMock()) is None


@patch("django_neural_feed.models.UserFeedProfile.objects.filter")
def test_get_user_profile_vector_db_crash(mock_filter):
    """Covers lines 102-109 exception block."""

    class CrashingFeed(BaseNeuralFeed):
        @classmethod
        def get_setting(cls, attr_name: str):
            return "crash_feed"

    mock_filter.side_effect = OperationalError("Database disconnected")
    assert CrashingFeed.get_user_vector(user=MagicMock()) is None


# --- 5. Lines 149-151: Real pgvector Slicing & Integrity ---


@pytest.mark.django_db
def test_generate_feed_slicing_flow():
    """Covers lines 149-151 down to the slice limit using real DB relations."""
    from tests.models import TestArticle

    class RealPipelineFeed(BaseNeuralFeed):
        popularity_expression = Value(0.0)
        freshness_expression = Value(0.0)
        weight_similarity = 1.0
        weight_freshness = 0.0
        weight_popularity = 0.0

        @classmethod
        def get_setting(cls, attr_name: str):
            if attr_name == "content_django_model":
                return TestArticle
            if attr_name == "feed_id":
                return "slice_feed"
            if "limit" in attr_name:
                return 1
            return super().get_setting(attr_name)

    # Create a persistent user to satisfy PostgreSQL foreign keys
    User = get_user_model()
    real_user = User.objects.create_user(username="real_test_user")

    UserFeedProfile.objects.create(user_id=real_user.id, feed_id="slice_feed", embedding=[1.0, 0.0, 0.0])  # type: ignore

    TestArticle.objects.create(title="First", embedding=[1.0, 0.0, 0.0])
    TestArticle.objects.create(title="Second", embedding=[0.9, 0.0, 0.0])

    # Pass limit explicitly to force slicing execution
    final_feed = RealPipelineFeed.get_feed(user=real_user, limit=1)
    assert len(final_feed) == 1


@pytest.mark.django_db
def test_get_feed_hnsw_enabled_multi_channel():
    """Covers full HNSW multi-channel retrieval branch with all channels active."""
    from django.contrib.auth import get_user_model
    from django.db import connections
    from django.db.models import Value
    from django_neural_feed.models import UserFeedProfile
    from tests.models import TestArticle
    from unittest.mock import patch

    class HNSWMultiChannelFeed(BaseNeuralFeed):
        popularity_expression = Value(1.0)
        freshness_expression = Value(1.0)

        @classmethod
        def get_setting(cls, attr_name: str):
            if attr_name == "content_django_model":
                return TestArticle
            if attr_name == "feed_id":
                return "hnsw_multi_feed"
            if attr_name == "hnsw_config":
                return {"ENABLED": True, "EF_SEARCH": 80, "SEARCH_POOL": 300}
            if attr_name == "weight_similarity":
                return 0.6
            if attr_name == "weight_freshness":
                return 0.2
            if attr_name == "weight_popularity":
                return 0.2
            return super().get_setting(attr_name)

    User = get_user_model()
    user = User.objects.create_user(username="hnsw_multi_user")
    UserFeedProfile.objects.create(user_id=user.id, feed_id="hnsw_multi_feed", embedding=[1.0, 0.0, 0.0])  # type: ignore

    TestArticle.objects.create(title="Vector Match", embedding=[1.0, 0.0, 0.0])
    TestArticle.objects.create(title="Fresh Match", embedding=[0.0, 1.0, 0.0])

    db_alias = TestArticle.objects.all().db
    real_cursor_getter = connections[db_alias].cursor
    hnsw_query_called = False

    # Safe cursor wrapper to intercept only the SET LOCAL command
    def safe_cursor_proxy():
        cursor = real_cursor_getter()
        real_execute = cursor.execute

        def patched_execute(sql, params=None):
            nonlocal hnsw_query_called
            if "SET LOCAL hnsw.ef_search" in sql:
                hnsw_query_called = True
                assert params == [80]
                return None  # Skip raw HNSW setup to avoid crashes on non-pgvector DBs
            return real_execute(sql, params)

        cursor.execute = patched_execute
        return cursor

    with patch.object(connections[db_alias], "cursor", side_effect=safe_cursor_proxy):
        feed = HNSWMultiChannelFeed.get_feed(user=user, limit=10)

        # Verify our HNSW config block was actually reached
        assert hnsw_query_called is True
        assert len(feed) <= 2


@pytest.mark.django_db
def test_get_feed_hnsw_only_similarity_channel():
    """Covers pipeline when HNSW is active but alternative channels have 0 weight."""
    from django.contrib.auth import get_user_model
    from django.db import connections
    from django.db.models import Value
    from django_neural_feed.models import UserFeedProfile
    from tests.models import TestArticle
    from unittest.mock import patch

    class HNSWSectorOnlyFeed(BaseNeuralFeed):
        popularity_expression = Value(1.0)
        freshness_expression = Value(1.0)

        @classmethod
        def get_setting(cls, attr_name: str):
            if attr_name == "content_django_model":
                return TestArticle
            if attr_name == "feed_id":
                return "hnsw_sim_feed"
            if attr_name == "hnsw_config":
                return {"ENABLED": True, "EF_SEARCH": 40, "SEARCH_POOL": 100}
            if attr_name == "weight_similarity":
                return 1.0
            if attr_name == "weight_freshness":
                return 0.0
            if attr_name == "weight_popularity":
                return 0.0
            return super().get_setting(attr_name)

    User = get_user_model()
    user = User.objects.create_user(username="hnsw_sim_user")
    UserFeedProfile.objects.create(user_id=user.id, feed_id="hnsw_sim_feed", embedding=[1.0, 0.0, 0.0])  # type: ignore

    TestArticle.objects.create(title="Pure Vector Target", embedding=[1.0, 0.0, 0.0])

    db_alias = TestArticle.objects.all().db
    real_cursor_getter = connections[db_alias].cursor
    hnsw_query_called = False

    def safe_cursor_proxy():
        cursor = real_cursor_getter()
        real_execute = cursor.execute

        def patched_execute(sql, params=None):
            nonlocal hnsw_query_called
            if "SET LOCAL hnsw.ef_search" in sql:
                hnsw_query_called = True
                assert params == [40]
                return None
            return real_execute(sql, params)

        cursor.execute = patched_execute
        return cursor

    with patch.object(connections[db_alias], "cursor", side_effect=safe_cursor_proxy):
        feed = HNSWSectorOnlyFeed.get_feed(user=user, limit=5)

        assert hnsw_query_called is True
        assert len(feed) == 1


@pytest.mark.django_db
def test_get_feed_hnsw_disabled_fallback_flow():
    """Ensures HNSW execution block is completely skipped when ENABLED is False."""
    from django.contrib.auth import get_user_model
    from django.db import connections
    from django.db.models import Value
    from django_neural_feed.models import UserFeedProfile
    from tests.models import TestArticle
    from unittest.mock import patch

    class HNSWDisabledFeed(BaseNeuralFeed):
        popularity_expression = Value(0.0)
        freshness_expression = Value(0.0)

        @classmethod
        def get_setting(cls, attr_name: str):
            if attr_name == "content_django_model":
                return TestArticle
            if attr_name == "feed_id":
                return "hnsw_disabled_feed"
            if attr_name == "hnsw_config":
                return {"ENABLED": False, "EF_SEARCH": 40, "SEARCH_POOL": 100}
            return super().get_setting(attr_name)

    User = get_user_model()
    user = User.objects.create_user(username="hnsw_off_user")
    UserFeedProfile.objects.create(user_id=user.id, feed_id="hnsw_disabled_feed", embedding=[1.0, 0.0, 0.0])  # type: ignore

    TestArticle.objects.create(
        title="Standard Pipeline Target", embedding=[1.0, 0.0, 0.0]
    )

    db_alias = TestArticle.objects.all().db
    real_cursor_getter = connections[db_alias].cursor
    hnsw_query_called = False

    # Safe cursor proxy to track if raw HNSW configurations were applied
    def safe_cursor_proxy():
        cursor = real_cursor_getter()
        real_execute = cursor.execute

        def patched_execute(sql, params=None):
            nonlocal hnsw_query_called
            if "SET LOCAL hnsw.ef_search" in sql:
                hnsw_query_called = True
            return real_execute(sql, params)

        cursor.execute = patched_execute
        return cursor

    with patch.object(connections[db_alias], "cursor", side_effect=safe_cursor_proxy):
        feed = HNSWDisabledFeed.get_feed(user=user, limit=5)

        # HNSW configuration must be skipped completely
        assert hnsw_query_called is False
        # Django fallback query still evaluates correctly without freezing
        assert len(feed) == 1


def test_calculate_embedding_resolves_encoder_and_model():
    """Covers calculate_embedding by overriding the ENCODER_CLASS property via class level patch."""

    class CustomModelFeed(BaseNeuralFeed):
        @classmethod
        def get_setting(cls, attr_name: str):
            if attr_name == "embedding_model_name":
                return "feed-specific-bert-model"
            return super().get_setting(attr_name)

    mock_encoder = MagicMock()
    mock_encoder.text_to_vector.return_value = [0.25, 0.5, 0.75]

    # Patch the property on the class level using PropertyMock
    with patch.object(
        AppSettings, "ENCODER_CLASS", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = mock_encoder

        result = CustomModelFeed.calculate_embedding("Some raw content text")

        # Verify interactions
        mock_encoder.text_to_vector.assert_called_once_with(
            "Some raw content text", "feed-specific-bert-model"
        )
        assert result == [0.25, 0.5, 0.75]


def test_set_user_vector_anonymous_or_none(birds_eye_view=None):
    """Should return None if user is None or not authenticated."""
    from django_neural_feed.feeds import BaseNeuralFeed

    # Case 1: user is None
    assert BaseNeuralFeed.set_user_vector(None, embedding=[0.1]) is None

    # Case 2: user is anonymous
    mock_user = MagicMock()
    mock_user.is_authenticated = False
    assert BaseNeuralFeed.set_user_vector(mock_user, embedding=[0.1]) is None


def test_set_user_vector_missing_feed_id():
    """Should return None if feed_id setting is missing or empty."""
    from django_neural_feed.feeds import BaseNeuralFeed

    mock_user = MagicMock(id=42, is_authenticated=True)

    with patch.object(BaseNeuralFeed, "get_setting", return_value=None):
        res = BaseNeuralFeed.set_user_vector(mock_user, embedding=[0.1])
        assert res is None


def test_set_user_vector_conflicting_or_missing_arguments():
    """Should raise ValueError if both args or neither args are provided."""
    from django_neural_feed.feeds import BaseNeuralFeed

    mock_user = MagicMock(id=42, is_authenticated=True)

    # Both arguments provided
    with pytest.raises(ValueError, match="simultaneously"):
        BaseNeuralFeed.set_user_vector(mock_user, embedding=[0.1], keywords="test")

    # Neither argument provided
    with pytest.raises(ValueError, match="Neither 'embedding' nor 'keywords'"):
        BaseNeuralFeed.set_user_vector(mock_user)


def test_set_user_vector_invalid_keyword_types():
    """Should raise ValueError for incorrect types inside or as keywords."""
    from django_neural_feed.feeds import BaseNeuralFeed

    mock_user = MagicMock(id=42, is_authenticated=True)

    # Keywords is a list but contains an integer
    with pytest.raises(ValueError, match="must be strings"):
        BaseNeuralFeed.set_user_vector(mock_user, keywords=["hello", 123])

    # Keywords is completely wrong type (e.g., dict)
    with pytest.raises(ValueError, match="Expected list\\[str\\] or str"):
        BaseNeuralFeed.set_user_vector(mock_user, keywords={"wrong": "type"})


def test_set_user_vector_success_with_embedding():
    """Should successfully update_or_create profile when clean embedding is given."""
    from django_neural_feed.feeds import BaseNeuralFeed

    mock_user = MagicMock(id=42, is_authenticated=True)
    test_embedding = [0.1, 0.2, 0.3]

    with (
        patch.object(BaseNeuralFeed, "get_setting", return_value="main_feed"),
        patch(
            "django_neural_feed.feeds.UserFeedProfile.objects.update_or_create"
        ) as mock_update_or_create,
    ):
        mock_update_or_create.return_value = ("mock_profile_instance", True)

        res = BaseNeuralFeed.set_user_vector(mock_user, embedding=test_embedding)

        # Assert correct record insertion/update in DB
        mock_update_or_create.assert_called_once_with(
            user_id=42,
            feed_id="main_feed",
            defaults={"embedding": test_embedding},
        )
        assert res == ("mock_profile_instance", True)


def test_set_user_vector_success_with_keywords_string_and_list():
    """Should calculate embedding and save when string or list keywords are given."""
    from django_neural_feed.feeds import BaseNeuralFeed

    mock_user = MagicMock(id=42, is_authenticated=True)
    calculated_vector = [0.9, 0.8, 0.7]

    with (
        patch.object(BaseNeuralFeed, "get_setting", return_value="main_feed"),
        patch.object(
            BaseNeuralFeed, "calculate_embedding", return_value=calculated_vector
        ) as mock_calc,
        patch(
            "django_neural_feed.feeds.UserFeedProfile.objects.update_or_create"
        ) as mock_update_or_create,
    ):
        # Case 1: Testing string keyword
        BaseNeuralFeed.set_user_vector(mock_user, keywords="tech news")
        mock_calc.assert_called_with("tech news")

        # Case 2: Testing list of keywords (should join with space)
        BaseNeuralFeed.set_user_vector(mock_user, keywords=["coding", "python"])
        mock_calc.assert_called_with("coding python")

        # Verify it went through to update_or_create
        assert mock_update_or_create.call_count == 2


def test_set_user_vector_embedding_becomes_none():
    """Should return None if embedding ends up being None right before DB block."""
    from django_neural_feed.feeds import BaseNeuralFeed

    mock_user = MagicMock(id=42, is_authenticated=True)

    with (
        patch.object(BaseNeuralFeed, "get_setting", return_value="main_feed"),
        patch.object(BaseNeuralFeed, "calculate_embedding", return_value=None),
    ):
        # Keywords string triggers calculate_embedding, which mocks to None
        res = BaseNeuralFeed.set_user_vector(mock_user, keywords="empty result")
        assert res is None


def test_set_user_vector_db_exception_handling():
    """Should log error and return None if database operation raises an exception."""
    from django_neural_feed.feeds import BaseNeuralFeed

    mock_user = MagicMock(id=42, is_authenticated=True)

    with (
        patch.object(BaseNeuralFeed, "get_setting", return_value="main_feed"),
        patch(
            "django_neural_feed.feeds.UserFeedProfile.objects.update_or_create",
            side_effect=Exception("DB connection timeout"),
        ),
        patch("django_neural_feed.feeds.logger.error") as mock_log_error,
    ):
        res = BaseNeuralFeed.set_user_vector(mock_user, embedding=[0.1, 0.2])

        # Ensure it gracefully returned None instead of crashing
        assert res is None
        # Check that error was logged with proper context
        mock_log_error.assert_called_once()
        assert "DNF: Error saving user vector" in mock_log_error.call_args[0][0]
