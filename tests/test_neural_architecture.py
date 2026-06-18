import pytest
from django.contrib.auth import get_user_model
from django.db import transaction
from tests.models import TestArticle, TestLikeModel
from tests.feeds import TestParentFeed, TestChildFeed
from django_neural_feed.models import UserFeedProfile
from django_neural_feed.mixins import NeuralRecommendMixin
from django.apps import apps
from django_neural_feed.conf import app_settings
import sys
from unittest.mock import MagicMock, patch
import pytest
from django_neural_feed.encoders import DefaultVectorEncoder, BaseVectorEncoder
import numpy as np
import time
from types import SimpleNamespace
from django.test import override_settings
from django_neural_feed.tasks import (
    get_model_from_path,
    generate_content_embedding_task,
    update_user_embedding_task,
)


from django_neural_feed.tasks import (
    generate_content_embedding_task,
    update_user_embedding_task,
)

User = get_user_model()


def test_hnsw_fallback_when_user_passes_garbage():
    """Covers the line where user_hnsw is not a dict and falls back to empty dict."""
    from django_neural_feed.conf import AppSettings
    from django.conf import settings

    # Simulate user passing garbage (e.g., a string instead of a dict)
    bad_config = {"HNSW": "not_a_dict_garbage"}

    with patch.object(settings, "DJANGO_NEURAL_FEED", bad_config):
        settings_proxy = AppSettings()
        hnsw_res = settings_proxy.HNSW

        # Verify it falls back to default config keys cleanly
        assert hnsw_res["ENABLED"] is False
        assert hnsw_res["M"] == 16
        assert hnsw_res["EF_CONSTRUCTION"] == 64


@pytest.mark.django_db(transaction=True)
def test_content_embedding_trigger():
    """Checks that creating content triggers embedding generation."""

    with transaction.atomic():
        article = TestArticle.objects.create(title="Python Django Deep Learning")

    for _ in range(10):
        article.refresh_from_db()
        if article.embedding is not None:
            break
        time.sleep(0.1)

    assert article.embedding is not None
    assert len(article.embedding) == 3


@pytest.mark.django_db
def test_user_profile_and_parent_inheritance():
    """Tests that user interaction updates profile and child feed inherits it."""
    from django.contrib.auth import get_user_model
    from django_neural_feed.models import UserFeedProfile
    from tests.feeds import TestChildFeed
    import numpy as np

    User = get_user_model()
    user = User.objects.create_user(username="testuser", password="password")

    article1 = TestArticle.objects.create(title="A")
    article2 = TestArticle.objects.create(title="B")

    TestLikeModel.objects.create(user=user, article=article1)
    TestLikeModel.objects.create(user=user, article=article2)

    profile = UserFeedProfile.objects.filter(user=user, feed_id="test_parent").first()

    assert profile is not None
    assert profile.embedding is not None

    child_vector = TestChildFeed.get_user_vector(user)
    assert child_vector is not None
    assert np.allclose(child_vector, profile.embedding)


@pytest.mark.django_db
def test_content_embedding_update_on_text_change():
    """Checks that modifying text triggers embedding regeneration."""
    article = TestArticle.objects.create(title="Initial Title")
    article.refresh_from_db()
    assert np.allclose(article.embedding, [0.5, 0.5, 0.5])

    article.title = "Completely New Title"
    article.save()

    article.refresh_from_db()
    assert article.embedding is not None


@pytest.mark.django_db
def test_celery_tasks_execution(settings, monkeypatch):
    """Tests that Celery tasks run successfully with accurate app model paths."""
    from django_neural_feed.conf import app_settings

    settings.DJANGO_NEURAL_FEED = {
        **getattr(settings, "DJANGO_NEURAL_FEED", {}),
        "CELERY_ENABLED": True,
        "VECTOR_DIMENSION": 3,
    }

    try:
        from celery import current_app

        current_app.conf.task_always_eager = True  # type: ignore
    except ImportError:
        pass

    article = TestArticle.objects.create(title="Celery Test Article")

    generate_content_embedding_task(
        django_model_path="tests.testarticle", instance_id=article.id  # type: ignore
    )

    article.refresh_from_db()
    assert np.allclose(article.embedding, [0.5, 0.5, 0.5])

    user = User.objects.create_user(username="celeryuser", password="password")
    TestLikeModel.objects.create(user=user, article=article)

    update_user_embedding_task(
        likes_django_model_path="tests.TestLikeModel",
        users_django_model_path="auth.User",
        user_id=user.id,  # type: ignore
        user_field_name="user",
        content_field_name="article",
        feed_id="test_parent",
        user_likes_limit=3,
    )

    from django_neural_feed.models import UserFeedProfile

    profile = UserFeedProfile.objects.filter(user_id=user.id, feed_id="test_parent").first()  # type: ignore
    assert profile is not None
    assert np.allclose(profile.embedding, [0.57735026, 0.57735026, 0.57735026])


def test_mixin_not_implemented_error():
    # Test abstract behavior
    class BadModel(NeuralRecommendMixin):
        pass

    instance = BadModel()
    with pytest.raises(NotImplementedError):
        instance.get_ready_text()


@pytest.mark.django_db
def test_user_feed_profile_str():
    # Cover __str__ representation
    user = User.objects.create_user(username="teststruser", password="123")
    profile = UserFeedProfile(user=user, feed_id="test_parent")

    assert str(profile) == "teststruser - test_parent"


@pytest.mark.django_db
def test_feed_get_candidates_and_fallback():
    user = User.objects.create_user(username="newbie", password="123")

    a1 = TestArticle.objects.create(title="Article 1")
    a2 = TestArticle.objects.create(title="Article 2")
    a3 = TestArticle.objects.create(title="Article 3")

    # 1. Fallback: User has no profile vector yet
    # Should return most popular and fresh posts
    feed_items = TestParentFeed.get_feed(user=user, limit=2)
    assert len(feed_items) == 2
    print(feed_items, a1 == feed_items[0], a2 == feed_items[0])
    assert feed_items[0] == a3
    assert feed_items[1] == a2

    # 2. Unauthenticated user fallback
    anon_feed = TestParentFeed.get_feed(user=None, limit=1)
    assert len(anon_feed) == 1
    assert anon_feed[0] == a3
    """
    """

    # 3. Excluded IDs logic
    # Assume user already saw a3 and a2
    excluded = TestParentFeed.get_feed(user=user, excluded_ids=[a2.id, a3.id], limit=5)  # type: ignore
    assert len(excluded) == 1
    assert excluded[0] == a1


def test_apps_config_ready_branch_coverage(monkeypatch):
    """Forces the ready() loop to cover both True and False branches of line 21."""

    # Feed 1: Has a string model path (covers lines 22-25)
    class StringModelFeed:
        content_django_model = "tests.TestArticle"
        feed_id = "string_feed"

        @classmethod
        def get_setting(cls, key):
            return "string_feed"

    # Feed 2: Has no model at all (covers the False jump 21->17)
    class EmptyModelFeed:
        content_django_model = None
        feed_id = "empty_feed"

        @classmethod
        def get_setting(cls, key):
            return "empty_feed"

    # Mock the settings to return BOTH feeds sequentially
    from django_neural_feed.conf import app_settings

    monkeypatch.setattr(
        app_settings, "get_registered_feeds", lambda: [StringModelFeed, EmptyModelFeed]
    )

    # Stub signal registrations to avoid side effects
    import django_neural_feed.signals

    monkeypatch.setattr(
        django_neural_feed.signals, "register_feed_signals", lambda feed: None
    )
    monkeypatch.setattr(
        django_neural_feed.signals, "register_content_signals", lambda model: None
    )

    # Trigger ready manually
    config = apps.get_app_config("django_neural_feed")
    config.ready()


def test_conf_encoder_import_error(monkeypatch):
    """Verifies custom ImportError is raised when string path is invalid."""
    # Force the internal config dict to return a broken string path
    fake_config = {"ENCODER_CLASS": "django_neural_feed.broken.MissingEncoderClass"}
    monkeypatch.setattr(app_settings, "_user_config", fake_config)

    with pytest.raises(ImportError) as exc_info:
        _ = app_settings.ENCODER_CLASS

    assert "DNF: Could not import custom encoder class" in str(exc_info.value)


def test_conf_feed_import_error(monkeypatch):
    """Verifies broken feed paths are gracefully caught and logged."""
    fake_config = {"FEEDS": ["django_neural_feed.broken.MissingFeedClass"]}
    monkeypatch.setattr(app_settings, "_user_config", fake_config)

    feeds = app_settings.get_registered_feeds()
    assert len(feeds) == 0


from django_neural_feed.encoders import BaseVectorEncoder


def test_base_encoder_abstract_raises():
    with pytest.raises(NotImplementedError):
        BaseVectorEncoder.text_to_vector("text", "model")


def test_base_encoder_average_vectors_handling():
    # Empty list must return empty list safely
    assert BaseVectorEncoder.average_vectors([], limit=5) == []

    # Invalid dimensions must trigger exception block and return empty list
    broken_matrix = [[1.0], [1.0, 2.0, 3.0]]
    assert BaseVectorEncoder.average_vectors(broken_matrix, limit=5) == []


def test_base_encoder_average_vectors_exception():
    """Forces an exception in numpy matrix operations to cover line 24."""
    # Passing incompatible shapes forces an exception in np.array or np.mean
    broken_matrix = [[1.0], [1.0, 2.0, 3.0]]
    assert BaseVectorEncoder.average_vectors(broken_matrix, limit=5) == []


def test_default_encoder_empty_text():
    """Verifies that empty or whitespace-only strings return an empty list immediately."""
    assert DefaultVectorEncoder.text_to_vector("   ", "all-MiniLM-L6-v2") == []


def test_default_encoder_missing_package(monkeypatch):
    """Simulates sentence-transformers package missing and checks that _get_model raises ImportError."""
    # Clear the internal model cache to force an import attempt
    DefaultVectorEncoder._model_instances.clear()

    # Temporarily hide sentence_transformers from sys.modules
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    # Call _get_model directly, because text_to_vector swallows all Exceptions
    with pytest.raises(ImportError):
        DefaultVectorEncoder._get_model("some-model")


@patch("django_neural_feed.encoders.logger")
def test_default_encoder_successful_local_load(mock_logger):
    """Simulates happy path where model is found locally (lines 41-46, 61-69)."""
    DefaultVectorEncoder._model_instances.clear()

    # Mock SentenceTransformer class and instance
    mock_transformer_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
    mock_transformer_cls.return_value = mock_instance

    with patch.dict(
        "sys.modules",
        {"sentence_transformers": MagicMock(SentenceTransformer=mock_transformer_cls)},
    ):
        vector = DefaultVectorEncoder.text_to_vector("hello", "local-model")

        assert vector == [0.1, 0.2, 0.3]
        # Check if initialized with local_files_only=True
        mock_transformer_cls.assert_called_with("local-model", local_files_only=True)


@patch("django_neural_feed.encoders.logger")
def test_default_encoder_fallback_download(mock_logger):
    """Simulates fallback to downloading model when not found locally (lines 47-53)."""
    DefaultVectorEncoder._model_instances.clear()

    mock_transformer_cls = MagicMock()
    # First call raises exception (not local), second call succeeds (download)
    mock_transformer_cls.side_effect = [Exception("Not found locally"), MagicMock()]

    with patch.dict(
        "sys.modules",
        {"sentence_transformers": MagicMock(SentenceTransformer=mock_transformer_cls)},
    ):
        # We don't care about encode result here, we are testing the instantiation logic
        try:
            DefaultVectorEncoder.text_to_vector("hello", "remote-model")
        except Exception:
            pass

        # Ensure it attempted to download after local failure
        assert mock_transformer_cls.call_count == 2
        mock_transformer_cls.assert_any_call("remote-model", local_files_only=False)


def test_default_encoder_encode_exception():
    """Simulates runtime exception during encode call to cover lines 70-72."""
    DefaultVectorEncoder._model_instances.clear()

    mock_instance = MagicMock()
    mock_instance.encode.side_effect = RuntimeError("GPU Out of memory")
    mock_transformer_cls = MagicMock(return_value=mock_instance)

    with patch.dict(
        "sys.modules",
        {"sentence_transformers": MagicMock(SentenceTransformer=mock_transformer_cls)},
    ):
        result = DefaultVectorEncoder.text_to_vector("hello", "error-model")
        assert result == []


def test_base_encoder_average_vectors_numpy_mean_exception(monkeypatch):
    """Forces an exception explicitly inside np.mean to cover line 24 execution flow."""
    # Pass a valid matrix so np.array succeeds, but break np.mean via monkeypatch
    valid_vectors = [[1.0, 2.0], [3.0, 4.0]]

    def mock_mean(*args, **kwargs):
        raise RuntimeError("Simulated NumPy error")

    monkeypatch.setattr(np, "mean", mock_mean)
    assert BaseVectorEncoder.average_vectors(valid_vectors, limit=5) == []


def test_default_encoder_get_model_caching_and_flow():
    """Covers lines 39-57 by forcing both fresh initialization and cache hits."""
    DefaultVectorEncoder._model_instances.clear()

    mock_instance = MagicMock()
    mock_instance.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
    mock_transformer_cls = MagicMock(return_value=mock_instance)

    with patch.dict(
        "sys.modules",
        {"sentence_transformers": MagicMock(SentenceTransformer=mock_transformer_cls)},
    ):
        # 1. Fresh load (triggers the interior of 'if model_name not in cls._model_instances')
        vec1 = DefaultVectorEncoder.text_to_vector("test1", "fresh-model")
        assert vec1 == [0.1, 0.2, 0.3]
        assert mock_transformer_cls.call_count == 1

        # 2. Cache hit (skips the 'if' block interior, executes line 57 directly)
        vec2 = DefaultVectorEncoder.text_to_vector("test2", "fresh-model")
        assert vec2 == [0.1, 0.2, 0.3]
        # Call count remains 1 because it took the instance from cache
        assert mock_transformer_cls.call_count == 1


@patch("django_neural_feed.encoders.logger")
def test_default_encoder_fallback_download_flow(mock_logger):
    """Covers lines 47-53 specifically focusing on local failure and download fallback."""
    DefaultVectorEncoder._model_instances.clear()

    mock_transformer_cls = MagicMock()
    # First call fails (local files missing), second call succeeds (downloading)
    mock_transformer_cls.side_effect = [Exception("Local missing"), MagicMock()]

    with patch.dict(
        "sys.modules",
        {"sentence_transformers": MagicMock(SentenceTransformer=mock_transformer_cls)},
    ):
        # Trigger the fallback inside _get_model
        model = DefaultVectorEncoder._get_model("download-model")
        assert model is not None
        assert mock_transformer_cls.call_count == 2


def test_base_vector_encoder_average_empty_nested_matrix():
    """Covers line 26: returns empty list if numpy array evaluates to 0 size."""
    from django_neural_feed.encoders import BaseVectorEncoder

    # Pass a truthy nested collection that results in an empty numpy structure
    falsy_nested_vectors = [[]]

    result = BaseVectorEncoder.average_vectors(falsy_nested_vectors, limit=5)

    assert result == []


def test_get_model_from_path_malformed():
    # Missing dot triggers split exception
    assert get_model_from_path("InvalidModelPathNoDot") is None


def test_generate_content_embedding_task_invalid_model():
    # Early return when model cannot be resolved
    assert generate_content_embedding_task(1, "invalid.ModelName") is None


def test_update_user_embedding_task_missing_models():
    """Covers line 64: Early return if likes or user model resolves to None."""
    result = update_user_embedding_task(
        likes_django_model_path="invalid.LikesModel",
        users_django_model_path="invalid.UserModel",
        user_id=1,
        user_field_name="user",
        content_field_name="article",
        feed_id="test_feed",
        user_likes_limit=5,
    )
    assert result is None


@patch("django_neural_feed.tasks.apps.get_model")
def test_update_user_embedding_task_user_not_found(mock_get_model):
    """Covers line 68: Early return if the user does not exist in the DB."""
    mock_user_model = MagicMock()
    mock_user_model.objects.filter.return_value.exists.return_value = False

    # First call returns valid likes model, second returns user model that reports user missing
    mock_get_model.side_effect = [MagicMock(), mock_user_model]

    result = update_user_embedding_task(
        likes_django_model_path="tests.TestLike",
        users_django_model_path="auth.User",
        user_id=404,
        user_field_name="user",
        content_field_name="article",
        feed_id="test_feed",
        user_likes_limit=5,
    )
    assert result is None


@patch("django_neural_feed.tasks.apps.get_model")
def test_update_user_embedding_task_no_embeddings(mock_get_model):
    """Covers edge case when user has no likes, forcing profile cleanup with None."""
    mock_user_model = MagicMock()
    mock_user_model.objects.filter.return_value.exists.return_value = True

    mock_likes_model = MagicMock()
    # Emulate an empty list returned by values_list()
    mock_likes_model.objects.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = (
        []
    )

    mock_get_model.side_effect = [mock_likes_model, mock_user_model]

    # Patch the database layer to avoid IntegrityError due to missing Foreign Key
    with patch(
        "django_neural_feed.models.UserFeedProfile.objects.update_or_create"
    ) as mock_update:
        result = update_user_embedding_task(
            likes_django_model_path="tests.TestLike",
            users_django_model_path="auth.User",
            user_id=1,
            user_field_name="user",
            content_field_name="article",
            feed_id="test_feed",
            user_likes_limit=5,
        )

        assert result is None
        # Verify that it safely tried to clear the stale profile in the DB
        mock_update.assert_called_once_with(
            user_id=1, feed_id="test_feed", defaults={"embedding": None}
        )


@patch("django_neural_feed.tasks.apps.get_model")
def test_update_user_embedding_task_exception_handling(mock_get_model):
    """Covers lines 96-97: Catches and logs unexpected database/runtime crashes."""
    mock_get_model.side_effect = RuntimeError("Database connection timed out")

    # Should not crash the Celery worker; must catch internally and return None
    update_user_embedding_task(
        likes_django_model_path="tests.TestLike",
        users_django_model_path="auth.User",
        user_id=1,
        user_field_name="user",
        content_field_name="article",
        feed_id="test_feed",
        user_likes_limit=5,
    )


@patch("django_neural_feed.tasks.apps.get_model")
def test_generate_content_embedding_task_fallback_and_error_flow(
    mock_get_model, monkeypatch
):
    """Covers lines 35->exit, 37->40, and 44-45 by patching the internal config dict."""
    mock_model = MagicMock()
    mock_instance = MagicMock()

    # Text must be truthy to pass line 35
    mock_instance.get_ready_text.return_value = "Valid text to encode"
    mock_model.objects.get.return_value = mock_instance
    mock_get_model.return_value = mock_model

    # Patch the internal dictionary because properties have no setters
    fake_config = {"MODEL_NAME": "fallback-model-name"}
    monkeypatch.setattr(app_settings, "_user_config", fake_config)

    # Mock _get_setting to return a broken encoder class that raises an error
    mock_encoder = MagicMock()
    mock_encoder.text_to_vector.side_effect = RuntimeError("Encoder crash")
    monkeypatch.setattr(
        app_settings,
        "_get_setting",
        lambda key: mock_encoder if key == "ENCODER_CLASS" else fake_config.get(key),
    )

    # embedding_model_name=None forces execution of lines 37-40
    generate_content_embedding_task(
        instance_id=1, django_model_path="tests.TestArticle", embedding_model_name=None
    )


@patch("django_neural_feed.tasks.apps.get_model")
@patch("django_neural_feed.encoders.BaseVectorEncoder.average_vectors")
def test_update_user_embedding_task_successful_and_empty_vector_flow(
    mock_average, mock_get_model
):
    """Covers lines 91->exit and 96-97 by simulating successful vector saving and profile crash."""
    mock_user_model = MagicMock()
    mock_user_model.objects.filter.return_value.exists.return_value = True

    mock_likes_model = MagicMock()
    # Return non-empty list to pass the 'if not recent_emb' guard on line 85
    mock_likes_model.objects.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = [
        [0.1, 0.2]
    ]

    mock_get_model.side_effect = [mock_likes_model, mock_user_model]

    mock_average.return_value = [0.1, 0.2, 0.3]

    with patch(
        "django_neural_feed.models.UserFeedProfile.objects.update_or_create"
    ) as mock_update:
        update_user_embedding_task(
            likes_django_model_path="tests.TestLike",
            users_django_model_path="auth.User",
            user_id=1,
            user_field_name="user",
            content_field_name="article",
            feed_id="test_feed",
            user_likes_limit=5,
        )
        assert mock_update.called

    mock_average.return_value = []  # Empty vector skips update_or_create
    update_user_embedding_task(
        likes_django_model_path="tests.TestLike",
        users_django_model_path="auth.User",
        user_id=1,
        user_field_name="user",
        content_field_name="article",
        feed_id="test_feed",
        user_likes_limit=5,
    )

    mock_average.return_value = [0.1, 0.2, 0.3]

    mock_get_model.side_effect = [mock_likes_model, mock_user_model]

    with patch(
        "django_neural_feed.models.UserFeedProfile.objects.update_or_create"
    ) as mock_update_crash:
        mock_update_crash.side_effect = RuntimeError(
            "Database integrity violation during save"
        )

        update_user_embedding_task(
            likes_django_model_path="tests.TestLike",
            users_django_model_path="auth.User",
            user_id=1,
            user_field_name="user",
            content_field_name="article",
            feed_id="test_feed",
            user_likes_limit=5,
        )


@patch("django_neural_feed.tasks.apps.get_model")
@patch("django_neural_feed.encoders.BaseVectorEncoder.average_vectors")
def test_update_user_embedding_task_save_flow(mock_average, mock_get_model):
    """Verifies that update_or_create is called when a valid vector is generated."""
    mock_user_model = MagicMock()
    mock_user_model.objects.filter.return_value.exists.return_value = True

    mock_likes_model = MagicMock()
    mock_likes_model.objects.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = [
        [0.1, 0.2]
    ]

    mock_get_model.side_effect = [mock_likes_model, mock_user_model]
    mock_average.return_value = [0.1, 0.2, 0.3]

    with patch(
        "django_neural_feed.models.UserFeedProfile.objects.update_or_create"
    ) as mock_update:
        update_user_embedding_task(
            likes_django_model_path="tests.TestLike",
            users_django_model_path="auth.User",
            user_id=1,
            user_field_name="user",
            content_field_name="article",
            feed_id="test_feed",
            user_likes_limit=5,
        )
        assert mock_update.call_count == 1


@patch("django_neural_feed.tasks.apps.get_model")
def test_update_user_embedding_task_skip_empty_vector_flow(mock_get_model, monkeypatch):
    """Covers lines 91->exit by mocking numpy to force an empty vector output."""
    mock_user_model = MagicMock()
    mock_user_model.objects.filter.return_value.exists.return_value = True

    mock_likes_model = MagicMock()
    # Pass non-empty list so we easily bypass line 85 check (if not recent_emb)
    mock_likes_model.objects.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = [
        [0.1, 0.2]
    ]

    mock_get_model.side_effect = [mock_likes_model, mock_user_model]

    # Force np.mean (or whatever math is used under the hood) to return an empty list
    # which is Falsy and skips update_or_create block
    monkeypatch.setattr(
        np, "mean", lambda *args, **kwargs: MagicMock(tolist=lambda: [])
    )
    # In case code uses a direct tolist() call on a mocked array structure, or returns [] directly:
    monkeypatch.setattr(
        np, "asarray", lambda *args, **kwargs: MagicMock(tolist=lambda: [])
    )

    with patch(
        "django_neural_feed.models.UserFeedProfile.objects.update_or_create"
    ) as mock_update_skip:
        update_user_embedding_task(
            likes_django_model_path="tests.TestLike",
            users_django_model_path="auth.User",
            user_id=1,
            user_field_name="user",
            content_field_name="article",
            feed_id="test_feed",
            user_likes_limit=5,
        )
        # The update_or_create must execute to clear the stale user profile
        assert mock_update_skip.call_count == 1
        mock_update_skip.assert_called_once_with(
            user_id=1, feed_id="test_feed", defaults={"embedding": None}
        )


def test_update_user_embedding_task_fallback_zero_norm_branch():
    """
    Covers the branch jump where feed is not found, vector has zero norm,
    and it bypasses normalization but keeps the truthy falsy check.
    """

    mock_likes_model = MagicMock()
    mock_user_model = MagicMock()

    # Ensure user exists to pass the initial check
    mock_user_model.objects.filter.return_value.exists.return_value = True

    # Return some mock data from DB so list isn't completely empty
    mock_likes_model.objects.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = [
        [0.0, 0.0]
    ]

    with (
        patch("django_neural_feed.tasks.get_model_from_path") as mock_get_model,
        patch("django_neural_feed.tasks.app_settings") as mock_settings,
        patch("django_neural_feed.models.UserFeedProfile.objects") as mock_profile_objs,
    ):
        # Setup model resolution
        mock_get_model.side_effect = lambda path: {
            "app.Likes": mock_likes_model,
            "app.User": mock_user_model,
        }.get(path)

        # Force no registered feeds to trigger the fallback else block
        mock_settings.get_registered_feeds.return_value = []

        # Force average_vectors to return a zero vector (norm will be 0.0)
        mock_settings.ENCODER_CLASS.average_vectors.return_value = [0.0, 0.0]

        # Execute task with an unregistered feed_id
        update_user_embedding_task(
            likes_django_model_path="app.Likes",
            users_django_model_path="app.User",
            user_id=42,
            user_field_name="user",
            content_field_name="content",
            feed_id="unregistered_garbage_feed",
            user_likes_limit=5,
        )

        # Verify it passed through without crashing and saved the fallback vector
        mock_profile_objs.update_or_create.assert_called_once_with(
            user_id=42,
            feed_id="unregistered_garbage_feed",
            defaults={"embedding": [0.0, 0.0]},
        )


@patch("django_neural_feed.tasks.apps.get_model")
def test_generate_content_embedding_task_empty_text_flow(mock_get_model):
    """Covers line 35->exit by simulating an empty text string."""
    mock_model = MagicMock()
    mock_instance = MagicMock()

    # Empty text makes line 35 evaluate to False and triggers early exit
    mock_instance.get_ready_text.return_value = ""
    mock_model.objects.get.return_value = mock_instance
    mock_get_model.return_value = mock_model

    generate_content_embedding_task(
        instance_id=1,
        django_model_path="tests.TestArticle",
        embedding_model_name="some-model",
    )
    # Ensure save was never called since there was nothing to vectorize
    assert mock_instance.save.call_count == 0


@patch("django_neural_feed.tasks.apps.get_model")
def test_generate_content_embedding_task_success_with_default_model(
    mock_get_model, monkeypatch
):
    """Covers lines 37->40 by mocking the entire app_settings object as a simple namespace."""
    mock_model = MagicMock()
    mock_instance = MagicMock()
    mock_instance.get_ready_text.return_value = "Valid content text"
    mock_model.objects.get.return_value = mock_instance
    mock_get_model.return_value = mock_model

    mock_encoder = MagicMock()
    mock_encoder.text_to_vector.return_value = [0.1, 0.2, 0.3]

    # Create a pure Python object with standard attributes (no read-only properties)
    fake_settings = SimpleNamespace(
        MODEL_NAME="mocked-default-model-name", ENCODER_CLASS=mock_encoder
    )

    # Patch app_settings directly inside the tasks module to bypass original property logic
    monkeypatch.setattr("django_neural_feed.tasks.app_settings", fake_settings)

    # Passing None forces line 38 to read fake_settings.MODEL_NAME and move to line 40
    generate_content_embedding_task(
        instance_id=1, django_model_path="tests.TestArticle", embedding_model_name=None
    )

    # Verify everything executed smoothly
    assert mock_instance.embedding == [0.1, 0.2, 0.3]
    assert mock_instance.save.call_count == 1


@patch("django_neural_feed.tasks.apps.get_model")
def test_generate_content_embedding_task_with_explicit_model_name(
    mock_get_model, monkeypatch
):
    """Covers the False branch of line 37 (the 37->40 jump) by passing an explicit model name."""
    mock_model = MagicMock()
    mock_instance = MagicMock()
    mock_instance.get_ready_text.return_value = "Valid content text"
    mock_model.objects.get.return_value = mock_instance
    mock_get_model.return_value = mock_model

    mock_encoder = MagicMock()
    mock_encoder.text_to_vector.return_value = [0.1, 0.2, 0.3]

    from django_neural_feed.conf import app_settings

    # monkeypatch.setattr(app_settings, "ENCODER_CLASS", mock_encoder)
    def mock_get_setting(key):
        if key == "ENCODER_CLASS":
            return mock_encoder
        return None

    monkeypatch.setattr(app_settings, "_get_setting", mock_get_setting)

    # Pass an explicit string. Line 37 evaluates to False, jumping directly to line 40
    generate_content_embedding_task(
        instance_id=1,
        django_model_path="tests.TestArticle",
        embedding_model_name="explicit-custom-model",
    )

    # Verify that the explicit model name was passed to the encoder, NOT the default one
    mock_encoder.text_to_vector.assert_called_once_with(
        "Valid content text", "explicit-custom-model"
    )
    assert mock_instance.save.call_count == 1
