# tests/test_signals_coverage.py
from functools import partial
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from django.apps import apps
from django.db import connection
from unittest.mock import ANY

# Import signal handlers directly to bypass complex DB triggers
from django_neural_feed.signals import (
    generate_content_embedding,
    _user_like_changed_model,
    _user_like_changed_m2m,
    _trigger_user_embedding_update,
    _run_synchronous_content_update,
    _run_synchronous_user_update,
)
from django_neural_feed.mixins import NeuralRecommendMixin
from django_neural_feed.feeds import BaseNeuralFeed
from django_neural_feed.conf import app_settings

# Note: Rename 'register_feed_signals' if your registration function is named differently
from django_neural_feed.signals import register_feed_signals

# --- 1. Line 17: Non-mixin sender guard ---


def test_generate_content_embedding_non_mixin_sender():
    """Covers line 17 early exit when sender lacks NeuralRecommendMixin."""

    class PlainModel:
        pass

    # Should return early without performing any actions
    assert (
        generate_content_embedding(sender=PlainModel, instance=None, created=True)
        is None
    )


# --- 2. Lines 28->exit & 32-39: Celery path and error fallbacks ---


def test_generate_content_embedding_celery_flow_and_errors():
    """Covers lines 28->exit and 32-39 for successful Celery dispatch and exceptions."""
    from unittest.mock import MagicMock

    class FakeNeuralRecommendMixin:
        pass

    # Patch the mixin inside the mixins module to avoid Django ORM side-effects
    with patch(
        "django_neural_feed.mixins.NeuralRecommendMixin", FakeNeuralRecommendMixin
    ):

        mock_objects = MagicMock()

        class MockMixinModel(FakeNeuralRecommendMixin):
            id = 42
            embedding = None
            objects = mock_objects

            class _meta:
                app_label = "tests"
                model_name = "mockmixinmodel"

            def get_ready_text(self):
                return "Sample text"

            def save(self, *args, **kwargs):
                # Fake Django save method for the synchronous fallback branch
                pass

        instance = MockMixinModel()
        mock_objects.get.return_value = instance

        # Branch A: Celery is enabled and works smoothly
        with patch.object(
            type(app_settings), "CELERY_ENABLED", new_callable=PropertyMock
        ) as mock_celery:
            mock_celery.return_value = True
            with patch(
                "django_neural_feed.tasks.generate_content_embedding_task"
            ) as mock_task:
                generate_content_embedding(
                    sender=MockMixinModel, instance=instance, created=True
                )
                mock_task.delay.assert_called_once_with(42, "tests.mockmixinmodel")

        # Branch B: Celery fails and triggers exception logger fallback -> hits DB and saves
        with patch.object(
            type(app_settings), "CELERY_ENABLED", new_callable=PropertyMock
        ) as mock_celery:
            mock_celery.return_value = True
            with patch(
                "django_neural_feed.tasks.generate_content_embedding_task"
            ) as mock_task:
                mock_task.delay.side_effect = Exception("Celery broker disconnected")

                generate_content_embedding(
                    sender=MockMixinModel, instance=instance, created=True
                )

                mock_objects.get.assert_called_once_with(id=42)


# --- 3. Lines 55, 58, 65, 76: Signal registration routines ---


def test_signal_registration_logic_and_validation_errors():
    """Covers lines 55, 58, 65, and 76 during feed setup processing."""

    class DummyFeed(BaseNeuralFeed):
        feed_id = "dummy_feed"

    def mock_missing_fields(attr):
        if attr == "interaction_django_model":
            return MagicMock(_meta=MagicMock(label_lower="dummy"))
        return None

    # Line 75-76: Missing configuration fields raises ValueError
    with patch.object(DummyFeed, "get_setting", side_effect=mock_missing_fields):
        with pytest.raises(
            ValueError, match="Specify user_field_name and content_field_name"
        ):
            register_feed_signals(DummyFeed)

    # Line 55 & 58: Resolve string model name and exit if not found
    with patch.object(DummyFeed, "get_setting") as mock_get:
        mock_get.side_effect = lambda attr: (
            "invalid.ModelPath" if attr == "interaction_django_model" else "valid_field"
        )
        with patch.object(apps, "get_model", return_value=None):
            assert register_feed_signals(DummyFeed) is None

    # Line 65: Successful connection setup for M2M mode interaction
    class ValidM2MFeed(BaseNeuralFeed):
        mode = "m2m"  # Explicitly set mode for attribute lookup
        feed_id = "m2m_test"  # Explicitly set feed_id for attribute lookup

    mock_model = MagicMock()
    mock_model._meta.label_lower = "testarticle"

    with patch.object(ValidM2MFeed, "get_setting") as mock_settings_get:
        mock_settings_get.side_effect = lambda attr: {
            "interaction_django_model": "tests.TestArticle",
            "user_field_name": "user",
            "content_field_name": "article",
        }.get(attr)

        with patch.object(apps, "get_model", return_value=mock_model):
            with patch("django_neural_feed.signals.m2m_changed") as mock_m2m_signal:
                register_feed_signals(ValidM2MFeed)
                assert mock_m2m_signal.connect.call_count > 0


# --- 4. Line 90 & 97-98: Model signal updates and exceptions ---


def test_user_like_changed_model_handlers():
    """Covers line 90 early exit and lines 97-98 exception logging block."""
    # Line 90: Return early if instance is updated rather than newly created
    assert (
        _user_like_changed_model(
            sender=None, instance=None, created=False, feed_class=None
        )
        is None
    )

    # Lines 97-98: Catch unexpected attribute lookup errors gracefully
    mock_feed = MagicMock()
    mock_feed.get_setting.side_effect = Exception("Dynamic configuration failure")
    _user_like_changed_model(
        sender=None, instance=MagicMock(), created=True, feed_class=mock_feed
    )


# --- 5. Lines 104-149: M2M processing pipeline ---


def test_user_like_changed_m2m_actions_and_flows():
    """Covers lines 104-149 processing logic for various M2M execution states."""
    # Line 104: Ignore non-relevant actions immediately
    assert (
        _user_like_changed_m2m(None, None, "pre_add", False, set(), feed_class=None)
        is None
    )

    # Run execution path for valid actions (post_add)
    mock_feed = MagicMock()
    mock_feed.get_setting.return_value = "user_property"
    mock_instance = MagicMock()
    setattr(mock_instance, "user_property", MagicMock(id=99))

    with patch(
        "django_neural_feed.signals._trigger_user_embedding_update"
    ) as mock_trigger:
        # Pass reverse=True to skip User DB query and test direct execution
        _user_like_changed_m2m(
            None, mock_instance, "post_add", True, {1, 2}, feed_class=mock_feed
        )
        mock_trigger.assert_called_once()


# --- 6. Lines 159-175: Celery user processing flows ---


class MockFeedClass(BaseNeuralFeed):
    # Add required attributes to prevent hidden AttributeError in the Celery try-block
    parent_feed = None
    feed_id = "test_feed"
    user_field_name = "user_attr"
    content_field_name = "content_attr"
    user_likes_limit = 50


def test_trigger_user_embedding_update_celery_paths():
    """Covers lines 159-175 verifying user embedding task offloading."""

    # Simple explicit structures to satisfy python type/class attribute lookups
    class MockSender:
        class _meta:
            app_label = "tests"
            model_name = "testarticle"

    class MockUser:
        id = 5

        class _meta:
            app_label = "auth"
            model_name = "user"

    # Branch A: Celery active route
    with patch.object(
        type(app_settings), "CELERY_ENABLED", new_callable=PropertyMock
    ) as mock_celery:
        mock_celery.return_value = True
        with patch("django_neural_feed.tasks.update_user_embedding_task") as mock_task:
            _trigger_user_embedding_update(
                user_object=MockUser(), sender=MockSender, feed_class=MockFeedClass
            )
            mock_task.delay.assert_called_once()

    # Branch B: Celery crash isolation route
    with patch.object(
        type(app_settings), "CELERY_ENABLED", new_callable=PropertyMock
    ) as mock_celery:
        mock_celery.return_value = True
        with patch("django_neural_feed.tasks.update_user_embedding_task") as mock_task:
            mock_task.delay.side_effect = Exception("Celery task queue full")

            _trigger_user_embedding_update(
                user_object=MockUser(), sender=MockSender, feed_class=MockFeedClass
            )


# --- 7. Lines 195->206 & 202-204: Content fallback worker thread crashes ---


def test_run_synchronous_content_update_exception_handling():
    """Covers lines 195->206 and 202-204 verifying database or evaluation crashes."""
    mock_model = MagicMock()
    mock_model.objects.get.side_effect = Exception(
        "Database connection timeout during sync"
    )

    # Verifies that exception re-raises correctly while safely closing connection resources
    with pytest.raises(Exception, match="Database connection timeout"):
        _run_synchronous_content_update(model_class=mock_model, instance_id=1)


# --- 8. Line 229 & 232-239: User fallback worker thread guards and crashes ---


def test_run_synchronous_user_update_missing_embeddings_and_crashes():
    """Covers empty sequence exit and thread exception safety inside fallback."""
    # Part 1: Exit safely when calculated vector collection returns empty results
    with patch.object(BaseNeuralFeed, "calculate_user_embedding", return_value=None):
        # Patch update_or_create because the new logic executes it to save None
        with patch(
            "django_neural_feed.models.UserFeedProfile.objects.update_or_create"
        ) as mock_update:
            assert (
                _run_synchronous_user_update(
                    user_id=1,
                    sender_model=MagicMock(),
                    feed_class=BaseNeuralFeed,
                    feed_id="test",
                )
                is None
            )
            # Ensure it accurately targeted the profile reset layout
            mock_update.assert_called_once_with(
                user_id=1, feed_id="test", defaults={"embedding": None}
            )

    # Part 2: Intercept execution level crashes securely inside the background thread
    # We mock update_or_create to throw an explicit exception and test resilience
    with patch(
        "django_neural_feed.models.UserFeedProfile.objects.update_or_create"
    ) as mock_db:
        mock_db.side_effect = Exception("Thread internal operational failure")

        # Function must intercept the error silently, log it safely, and exit cleanly
        _run_synchronous_user_update(
            user_id=1,
            sender_model=MagicMock(),
            feed_class=BaseNeuralFeed,
            feed_id="test",
        )


def test_generate_content_embedding_early_exit():
    """Covers line 28->exit when no update conditions are met."""
    from django_neural_feed.mixins import NeuralRecommendMixin
    from django_neural_feed.signals import generate_content_embedding

    # Valid class to satisfy issubclass() check
    class ValidMockSender(NeuralRecommendMixin):
        pass

    class MockInstance:
        embedding = [0.1, 0.2, 0.3]
        _original_text = "no change"

        def get_ready_text(self):
            return "no change"

    with patch("django_neural_feed.signals.app_settings") as mock_settings:
        # Pass the class as sender, not a mock instance
        generate_content_embedding(
            sender=ValidMockSender, instance=MockInstance(), created=False
        )
        # Verify it exited early without checking settings
        mock_settings.assert_not_called()


def test_run_synchronous_content_update_exception():
    """Covers lines 195->206 when DB fetch fails during sync content update."""
    from django_neural_feed.signals import _run_synchronous_content_update

    mock_model = MagicMock()
    mock_model.objects.get.side_effect = ValueError("Database connection lost")

    # Must log and re-raise the exception
    with pytest.raises(ValueError, match="Database connection lost"):
        _run_synchronous_content_update(model_class=mock_model, instance_id=99)


def test_run_synchronous_user_update_exception():
    """Covers lines 232->239 when background thread processing fails."""
    from django_neural_feed.signals import _run_synchronous_user_update

    mock_feed = MagicMock()
    mock_feed.get_setting.side_effect = RuntimeError("Thread processing crash")

    # Should catch exception internally and log it without crashing
    _run_synchronous_user_update(
        user_id=1, sender_model=MagicMock(), feed_class=mock_feed, feed_id="test"
    )


def test_user_like_changed_m2m_forward_flow():
    """Covers forward M2M relations (reverse=False) and loop execution."""
    from django_neural_feed.signals import _user_like_changed_m2m

    mock_feed = MagicMock()
    mock_feed.get_setting.return_value = "user_profile"

    # Mock User model and its objects.filter manager to return a fake user
    mock_user_model = MagicMock()
    mock_user_instance = MagicMock()
    mock_user_model.objects.filter.return_value = [mock_user_instance]

    with (
        patch("django.contrib.auth.get_user_model", return_value=mock_user_model),
        patch(
            "django_neural_feed.signals._trigger_user_embedding_update"
        ) as mock_trigger,
    ):

        _user_like_changed_m2m(
            sender=MagicMock(),
            instance=MagicMock(),
            action="post_add",
            reverse=False,
            pk_set={101, 102},
            feed_class=mock_feed,
        )
        # Now it evaluates the loop because filter() returns an element
        assert mock_trigger.call_count == 1


def test_user_like_changed_m2m_reverse_flow():
    """Covers reverse M2M relations (reverse=True)."""
    from django_neural_feed.signals import _user_like_changed_m2m

    mock_feed = MagicMock()
    mock_feed.get_setting.return_value = "user_profile"

    with (
        patch("django.contrib.auth.get_user_model", return_value=MagicMock()),
        patch(
            "django_neural_feed.signals._trigger_user_embedding_update"
        ) as mock_trigger,
    ):

        instance_user = MagicMock()
        _user_like_changed_m2m(
            sender=MagicMock(),
            instance=instance_user,
            action="post_remove",
            reverse=True,
            pk_set={201},
            feed_class=mock_feed,
        )
        # Directly triggers update using the passed instance
        mock_trigger.assert_called_once_with(
            user_object=instance_user, sender=ANY, feed_class=mock_feed
        )


def test_user_like_changed_m2m_exception_handling():
    """Covers lines 97-98 / catch block inside M2M helper."""
    from django_neural_feed.signals import _user_like_changed_m2m

    mock_feed = MagicMock()
    # Force an exception inside the try block
    mock_feed.get_setting.side_effect = RuntimeError("Simulated crash")

    with patch("django_neural_feed.signals.logger") as mock_logger:
        _user_like_changed_m2m(
            sender=MagicMock(),
            instance=MagicMock(),
            action="post_add",
            reverse=False,
            pk_set={101},
            feed_class=mock_feed,
        )
        # Verify that exception was caught and logged gracefully
        mock_logger.error.assert_called_once()


def test_user_like_changed_m2m_autodiscover_success():
    """Covers lines 116-137: successful auto-discovery of relation fields."""
    from django_neural_feed.signals import _user_like_changed_m2m

    mock_feed = MagicMock()
    # Return None to force execution of 'if not user_field_name:'
    mock_feed.get_setting.return_value = None

    mock_user_model = MagicMock()
    mock_user_model.objects.filter.return_value = [MagicMock()]

    # Mock fields for sender._meta.fields
    mock_user_field = MagicMock()
    mock_user_field.is_relation = True
    mock_user_field.related_model = mock_user_model
    mock_user_field.name = "user"

    mock_content_field = MagicMock()
    mock_content_field.is_relation = True
    mock_content_field.related_model = MagicMock()  # Dynamic mock class
    mock_content_field.name = "content"

    class FakeSender:
        __name__ = "FakeSender"

        class _meta:
            fields = [mock_user_field, mock_content_field]

    with (
        patch("django.contrib.auth.get_user_model", return_value=mock_user_model),
        patch(
            "django_neural_feed.signals._trigger_user_embedding_update"
        ) as mock_trigger,
    ):

        _user_like_changed_m2m(
            sender=FakeSender,
            instance=MagicMock(),
            action="post_add",
            reverse=False,
            pk_set={101},
            feed_class=mock_feed,
        )
        assert mock_trigger.call_count == 1
        assert mock_feed.user_field_name == "user"
        assert mock_feed.content_field_name == "content"


def test_user_like_changed_m2m_autodiscover_value_errors():
    """Covers lines 116-137: ValueError cases when relations are invalid."""
    from django_neural_feed.signals import _user_like_changed_m2m

    mock_feed = MagicMock()
    mock_feed.get_setting.return_value = None

    # Case 1: No user fields found (triggers first ValueError)
    class FakeSenderNoUser:
        __name__ = "FakeSenderNoUser"

        class _meta:
            fields = []

    with (
        patch("django.contrib.auth.get_user_model", return_value=MagicMock()),
        patch("django_neural_feed.signals.logger") as mock_logger,
    ):

        _user_like_changed_m2m(
            sender=FakeSenderNoUser,
            instance=MagicMock(),
            action="post_add",
            reverse=False,
            pk_set={101},
            feed_class=mock_feed,
        )
        # Exception caught by outer try-except block
        mock_logger.error.assert_called_with(ANY)


def test_user_like_changed_m2m_autodiscover_content_error():
    """Covers line 134: ValueError when content relation count is not exactly 1."""
    from django_neural_feed.signals import _user_like_changed_m2m

    mock_feed = MagicMock()
    mock_feed.get_setting.return_value = None

    mock_user_model = MagicMock()

    # Provide only 1 relation (user), leaving content relations at 0
    mock_user_field = MagicMock()
    mock_user_field.is_relation = True
    mock_user_field.related_model = mock_user_model
    mock_user_field.name = "user"

    class FakeSender:
        __name__ = "FakeSender"

        class _meta:
            fields = [mock_user_field]

    with (
        patch("django.contrib.auth.get_user_model", return_value=mock_user_model),
        patch("django_neural_feed.signals.logger") as mock_logger,
    ):

        _user_like_changed_m2m(
            sender=FakeSender,
            instance=MagicMock(),
            action="post_add",
            reverse=False,
            pk_set={101},
            feed_class=mock_feed,
        )
        # Verify the content fields length error was raised and caught
        mock_logger.error.assert_called_once()
        assert "must have exactly one content relation" in str(
            mock_logger.error.call_args[0][0]
        )


def test_run_synchronous_content_update_all_branches():
    """Covers lines 195->206 completely for both success and exception paths."""
    from django_neural_feed.signals import _run_synchronous_content_update

    # Path A: Full success execution
    mock_model_ok = MagicMock()
    mock_instance = MagicMock()
    mock_instance.get_ready_text.return_value = "valid text"
    mock_model_ok.objects.get.return_value = mock_instance

    with (
        patch("django_neural_feed.signals.app_settings") as mock_settings,
        patch("django_neural_feed.signals.connection") as mock_conn,
    ):
        mock_settings.ENCODER_CLASS.text_to_vector.return_value = [0.1, 0.2]

        _run_synchronous_content_update(model_class=mock_model_ok, instance_id=1)
        mock_instance.save.assert_called_once_with(update_fields=["embedding"])
        mock_conn.close.assert_called_once()

    # Path B: Exception handling and re-raise flow
    mock_model_fail = MagicMock()
    mock_model_fail.objects.get.side_effect = RuntimeError("Database timeout")

    with (
        patch("django_neural_feed.signals.connection") as mock_conn,
        patch("django_neural_feed.signals.logger") as mock_logger,
        pytest.raises(RuntimeError, match="Database timeout"),
    ):

        _run_synchronous_content_update(model_class=mock_model_fail, instance_id=2)
        mock_logger.exception.assert_called_once()
        mock_conn.close.assert_called_once()


def test_run_synchronous_user_update_falsy_vector():
    """Covers line 232->239 branch when vector is empty or None."""
    from django_neural_feed.signals import _run_synchronous_user_update

    mock_feed = MagicMock()
    mock_feed.get_setting.side_effect = lambda key: {
        "user_field_name": "user",
        "content_field_name": "content",
        "user_likes_limit": 3,
    }.get(key)

    mock_model = MagicMock()
    mock_model.objects.filter.return_value.order_by.return_value.__getitem__.return_value.values_list.return_value = [
        [0.9, 0.8]
    ]

    with (
        patch("django_neural_feed.signals.app_settings") as mock_settings,
        patch("django_neural_feed.models.UserFeedProfile.objects") as mock_profile_objs,
    ):

        # Force average_vectors to return None to bypass update_or_create block
        mock_settings.ENCODER_CLASS.average_vectors.return_value = None

        _run_synchronous_user_update(
            user_id=42,
            sender_model=mock_model,
            feed_class=mock_feed,
            feed_id="test_feed",
        )
        # Ensure it explicitly saves None to database to wipe the profile
        mock_profile_objs.update_or_create.assert_called_once_with(
            user_id=42, feed_id="test_feed", defaults={"embedding": None}
        )


def test_run_synchronous_content_update_empty_text():
    """Covers line 195->206 branch when text_to_vectorize is empty."""
    from django_neural_feed.signals import _run_synchronous_content_update
    from unittest.mock import MagicMock, patch

    mock_model = MagicMock()
    mock_instance = MagicMock()
    # Force empty string to make line 195 evaluate to False
    mock_instance.get_ready_text.return_value = ""
    mock_model.objects.get.return_value = mock_instance

    with (
        patch("django_neural_feed.signals.app_settings") as mock_settings,
        patch("django_neural_feed.signals.connection") as mock_conn,
    ):

        _run_synchronous_content_update(model_class=mock_model, instance_id=1)

        # Ensure heavy encoder processing and DB save were skipped
        mock_settings.ENCODER_CLASS.text_to_vector.assert_not_called()
        mock_instance.save.assert_not_called()
        # Ensure connection.close() in finally block was still executed
        mock_conn.close.assert_called_once()


def test_run_synchronous_content_update_empty_vector():
    """Covers line when text_to_vectorize is valid but vector returns empty."""
    from django_neural_feed.signals import _run_synchronous_content_update
    from unittest.mock import MagicMock, patch

    mock_model = MagicMock()
    mock_instance = MagicMock()

    # Text exists but it might be whitespace-only
    mock_instance.get_ready_text.return_value = "   "
    mock_model.objects.get.return_value = mock_instance

    with (
        patch("django_neural_feed.signals.app_settings") as mock_settings,
        patch("django_neural_feed.signals.connection") as mock_conn,
    ):
        # Mock encoder to return empty list (simulating failed/blank encode)
        mock_encoder = MagicMock()
        mock_encoder.text_to_vector.return_value = []
        mock_settings.ENCODER_CLASS = mock_encoder

        _run_synchronous_content_update(model_class=mock_model, instance_id=1)

        # Ensure encoder was called but save was skipped due to empty vector
        mock_encoder.text_to_vector.assert_called_once_with(
            "   ", mock_settings.MODEL_NAME
        )
        mock_instance.save.assert_not_called()
        mock_conn.close.assert_called_once()
