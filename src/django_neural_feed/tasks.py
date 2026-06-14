import logging
from celery import shared_task
from django.apps import apps
from django.db.models import Model
from django_neural_feed.conf import app_settings

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
            vector = encoder.text_to_vector(text_to_vectorize, embedding_model_name)  # type: ignore
            # text_to_vector returns [] for blank-after-strip or failed encodes; never
            # persist an empty vector (pgvector rejects 0-dim). Mirrors the guard in
            # update_user_embedding_task.
            if vector:
                instance.embedding = vector
                instance.save(update_fields=["embedding"])

    except Exception as e:
        logger.error(f"DNF Celery Error - content embedding generation failed: {e}")


@shared_task
def update_user_embedding_task(
    likes_django_model_path,
    users_django_model_path,
    user_id,
    user_field_name,
    content_field_name,
    feed_id,
    user_likes_limit,
):
    """Asynchronously recalculates the user profile vector for a specific feed."""
    try:
        likes_django_model = get_model_from_path(likes_django_model_path)
        user_django_model = get_model_from_path(users_django_model_path)

        if likes_django_model is None or user_django_model is None:
            return

        # Verify user still exists to avoid orphaned data if deleted recently
        if not user_django_model.objects.filter(id=user_id).exists():
            return

        from django_neural_feed.models import UserFeedProfile

        prefix = f"{content_field_name}__" if content_field_name else ""
        filter_kwargs = {
            f"{user_field_name}_id": user_id,
            f"{prefix}embedding__isnull": False,
        }

        recent_emb = list(
            likes_django_model.objects.filter(**filter_kwargs)
            .order_by("-id")[:user_likes_limit]
            .values_list(f"{prefix}embedding", flat=True)
        )

        # Generate average vector using the configured encoder
        encoder = app_settings.ENCODER_CLASS
        vector = encoder.average_vectors(recent_emb, user_likes_limit)

        # If vector is empty (user has 0 likes), we either clear the profile or set it to None
        if not vector:
            UserFeedProfile.objects.update_or_create(
                user_id=user_id, feed_id=feed_id, defaults={"embedding": None}
            )
        else:  # If vector is not empty, we do main job
            UserFeedProfile.objects.update_or_create(
                user_id=user_id, feed_id=feed_id, defaults={"embedding": vector}
            )

    except Exception as e:
        logger.error(f"DNF Celery Error - user embedding generation failed: {e}")
