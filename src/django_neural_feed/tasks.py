import logging
from celery import shared_task
from django.apps import apps
from django.db.models import Model
from django_neural_feed.conf import app_settings
from django_neural_feed.models import UserFeedProfile

logger = logging.getLogger(__name__)


def get_model_from_path(model_path: str) -> type[Model] | None:
    """Dynamically looks up a Django model class using its 'app_label.model_name'."""
    try:
        app_label, model_name = model_path.split(".")
        return apps.get_model(app_label, model_name)
    except Exception as e:
        logger.error(
            f"DNF Celery Error - cannot get model from path ({model_path}): {e}"
        )
        return None


@shared_task
def generate_content_embedding_task(
    instance_id, django_model_path, embedding_model_name=None
):
    """Asynchronously calculates and saves vector embeddings for content models."""
    try:
        django_model = get_model_from_path(django_model_path)
        if django_model is None:
            return

        instance = django_model.objects.get(id=instance_id)
        text_to_vectorize = instance.get_ready_text()  # type: ignore

        if text_to_vectorize:

            if embedding_model_name is None:
                embedding_model_name = app_settings.MODEL_NAME
            # Using the dynamic encoder from app settings
            encoder = app_settings.ENCODER_CLASS
            instance.embedding = encoder.text_to_vector(text_to_vectorize, embedding_model_name)  # type: ignore
            instance.save(update_fields=["embedding"])

    except Exception as e:
        logger.error(f"DNF Celery Error - content embedding generation failed: {e}")


@shared_task
def update_user_embedding_task(user_id, feed_id):
    """Asynchronously recalculates the user profile vector for a specific feed."""
    try:
        feed_class = None
        for cls in app_settings.get_registered_feeds():
            if getattr(cls, "feed_id", None) == feed_id:
                feed_class = cls
                break

        if not feed_class:
            logger.error(
                f"DNF Celery Error - Feed class with id '{feed_id}' not found."
            )
            return

        # Dynamically fetch current configuration from the feed class
        likes_model = feed_class.get_setting("interaction_django_model")
        user_field = feed_class.get_setting("user_field_name")

        if not likes_model:
            return

        if isinstance(likes_model, str):
            likes_model = get_model_from_path(likes_model)
            if not likes_model:
                return

        user_queryset = likes_model.objects.filter(**{f"{user_field}_id": user_id})
        vector = feed_class.calculate_user_embedding(user_queryset)

        UserFeedProfile.objects.update_or_create(
            user_id=user_id, feed_id=feed_id, defaults={"embedding": vector or None}
        )
    except Exception as e:
        logger.error(f"DNF Celery Error - User embedding generation failed: {e}")
        raise e
