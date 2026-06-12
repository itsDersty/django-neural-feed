import threading
import logging
from functools import partial
from django.db import connection, transaction
from django.db.models.signals import post_save, m2m_changed
from django.dispatch import receiver
from django_neural_feed.conf import app_settings
from django.apps import apps

logger = logging.getLogger(__name__)


def generate_content_embedding(sender, instance, created, **kwargs):
    from django_neural_feed.mixins import NeuralRecommendMixin

    if not issubclass(sender, NeuralRecommendMixin):
        return

    update_fields = kwargs.get("update_fields")
    if update_fields and "embedding" in update_fields:
        return

    text_changed = False
    if not created:
        current_text = instance.get_ready_text()
        text_changed = current_text != getattr(instance, "_original_text", None)

    if created or instance.embedding is None or text_changed:
        instance._original_text = instance.get_ready_text()

        if app_settings.CELERY_ENABLED:
            try:
                from .tasks import generate_content_embedding_task

                model_path = f"{sender._meta.app_label}.{sender._meta.model_name}"
                generate_content_embedding_task.delay(instance.id, model_path)  # type: ignore
                return
            except Exception as celery_err:
                logger.error(f"DNF Celery error, falling back to threads: {celery_err}")

        transaction.on_commit(
            lambda: threading.Thread(
                target=_run_synchronous_content_update,
                kwargs={"model_class": sender, "instance_id": instance.id},
                daemon=True,
            ).start()
        )


def register_feed_signals(feed_class):
    """Entry point to bind signals based on feed configuration."""
    like_target = feed_class.get_setting("interaction_django_model")

    if isinstance(like_target, str):
        like_target = apps.get_model(like_target)

    if not like_target:
        return

    mode = getattr(feed_class, "mode", "model")
    feed_id = feed_class.feed_id
    label = like_target._meta.label_lower

    if mode == "m2m":
        m2m_changed.connect(
            partial(_user_like_changed_m2m, feed_class=feed_class),
            sender=like_target,
            dispatch_uid=f"dnf_m2m_{feed_id}_{label}",
            weak=False,
        )
    else:
        user_field = feed_class.get_setting("user_field_name")
        content_field = feed_class.get_setting("content_field_name")

        if not user_field or not content_field:
            raise ValueError(
                "Specify user_field_name and content_field_name in your Feed class for model mode."
            )

        post_save.connect(
            partial(_user_like_changed_model, feed_class=feed_class),
            sender=like_target,
            dispatch_uid=f"dnf_model_{feed_id}_{label}",
            weak=False,
        )


def _user_like_changed_model(sender, instance, created, *, feed_class, **kwargs):
    if not created:
        return
    try:
        user_field_name = feed_class.get_setting("user_field_name")
        user_object = getattr(instance, user_field_name)
        _trigger_user_embedding_update(
            user_object=user_object, sender=sender, feed_class=feed_class
        )
    except Exception as e:
        logger.error(f"DNF model signal error: {e}")


def _user_like_changed_m2m(
    sender, instance, action, reverse, pk_set, *, feed_class, **kwargs
):
    if action not in ("post_add", "post_remove"):
        return

    try:
        from django.contrib.auth import get_user_model

        User = get_user_model()

        user_field_name = feed_class.get_setting("user_field_name")

        # Auto-discover relation fields if not explicitly defined
        if not user_field_name:
            relations = [f for f in sender._meta.fields if f.is_relation]
            user_fields = [
                f
                for f in relations
                if f.related_model == User
                or (
                    isinstance(f.related_model, type)
                    and issubclass(f.related_model, User)
                )
            ]
            if len(user_fields) != 1:
                raise ValueError(
                    f"DNF: M2M Model {sender.__name__} must have exactly one relation to User."
                )
            feed_class.user_field_name = user_fields[0].name

            content_fields = [f for f in relations if f.name != user_fields[0].name]
            if len(content_fields) != 1:
                raise ValueError(
                    f"DNF: M2M Model {sender.__name__} must have exactly one content relation."
                )
            feed_class.content_field_name = content_fields[0].name

        if reverse:
            _trigger_user_embedding_update(
                user_object=instance, sender=sender, feed_class=feed_class
            )
        else:
            for user_object in User.objects.filter(pk__in=pk_set):
                _trigger_user_embedding_update(
                    user_object=user_object, sender=sender, feed_class=feed_class
                )
    except Exception as e:
        logger.error(f"DNF M2M signal error: {e}")


def _trigger_user_embedding_update(*, user_object, sender, feed_class):
    """Routes the update to Celery or a background thread."""
    target_feed_id = (
        feed_class.parent_feed.feed_id if feed_class.parent_feed else feed_class.feed_id
    )

    if app_settings.CELERY_ENABLED:
        try:
            from celery import Task
            from .tasks import update_user_embedding_task

            celery_task: Task = update_user_embedding_task  # type: ignore
            celery_task.delay(
                likes_django_model_path=f"{sender._meta.app_label}.{sender._meta.model_name}",
                users_django_model_path=f"{user_object.__class__._meta.app_label}.{user_object.__class__._meta.model_name}",
                user_id=user_object.id,
                user_field_name=feed_class.user_field_name,
                content_field_name=feed_class.content_field_name,
                feed_id=target_feed_id,
                user_likes_limit=feed_class.user_likes_limit,
            )
            return
        except Exception as celery_err:
            logger.error(f"DNF Celery error: {celery_err}")

    transaction.on_commit(
        lambda: threading.Thread(
            target=_run_synchronous_user_update,
            kwargs={
                "user_id": user_object.id,
                "sender_model": sender,
                "feed_class": feed_class,
                "feed_id": target_feed_id,
            },
            daemon=True,
        ).start()
    )


def _run_synchronous_content_update(*, model_class, instance_id):
    try:
        instance = model_class.objects.get(id=instance_id)
        text_to_vectorize = instance.get_ready_text()
        if text_to_vectorize:
            encoder = app_settings.ENCODER_CLASS

            instance.embedding = encoder.text_to_vector(
                text_to_vectorize, app_settings.MODEL_NAME
            )
            instance.save(update_fields=["embedding"])
    except Exception:
        logger.exception("DNF synchronous content update error:")
        raise
    finally:
        connection.close()


def _run_synchronous_user_update(*, user_id, sender_model, feed_class, feed_id):
    try:
        from django_neural_feed.models import UserFeedProfile

        encoder = app_settings.ENCODER_CLASS

        u_field = feed_class.get_setting("user_field_name")
        c_field = feed_class.get_setting("content_field_name")
        limit = feed_class.get_setting("user_likes_limit")

        prefix = f"{c_field}__" if c_field else ""
        filter_kwargs = {f"{u_field}_id": user_id, f"{prefix}embedding__isnull": False}

        recent_emb = list(
            sender_model.objects.filter(**filter_kwargs)
            .order_by("-id")[:limit]
            .values_list(f"{prefix}embedding", flat=True)
        )

        vector = encoder.average_vectors(recent_emb, limit)

        # If vector is empty (user has 0 likes), we either clear the profile or set it to None
        if not vector:
            UserFeedProfile.objects.update_or_create(
                user_id=user_id, feed_id=feed_id, defaults={"embedding": None}
            )
        else:  # If vector is not empty, we doing main job
            UserFeedProfile.objects.update_or_create(
                user_id=user_id, feed_id=feed_id, defaults={"embedding": vector}
            )
    except Exception as e:
        logger.error(f"DNF Background Thread Error (User): {e}")
    finally:
        connection.close()


def register_content_signals(content_django_model):
    """Dynamically connects content embedding generation to specific models only."""
    from django.db.models.signals import post_save

    post_save.connect(
        receiver=generate_content_embedding,
        sender=content_django_model,
        dispatch_uid=f"dnf_content_{content_django_model._meta.label_lower}",
    )
