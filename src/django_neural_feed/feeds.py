import numpy as np
from django.db.models import F, Value, FloatField
from django.db.models.functions import Coalesce
from django.db import connections, transaction
from pgvector.django import MaxInnerProduct
from typing import Literal

from django_neural_feed.conf import app_settings
from django_neural_feed.models import UserFeedProfile

import logging

logger = logging.getLogger(__name__)


class BaseNeuralFeed:

    feed_id: str = "default_feed"
    content_django_model = None
    interaction_django_model = None
    mode: Literal["model", "m2m"]
    user_field_name: str | None = None
    content_field_name: str | None = None
    parent_feed: type["BaseNeuralFeed"] | None = None

    embedding_model_name: str = app_settings.MODEL_NAME
    user_likes_limit: int = app_settings.USER_LIKES_LIMIT

    weight_similarity: float = app_settings.WEIGHT_SIMILARITY
    weight_freshness: float = app_settings.WEIGHT_FRESHNESS
    weight_popularity: float = app_settings.WEIGHT_POPULARITY

    popularity_expression = app_settings.POPULARITY_EXPRESSION
    freshness_expression = app_settings.FRESHNESS_EXPRESSION

    hnsw_config: dict = app_settings.HNSW

    _model_instances: dict = {}

    @classmethod
    def get_setting(cls, attr_name: str):
        """
        Settings getter with fallback to parent_feed before the base class defaults.
        """
        if attr_name in cls.__dict__:
            return getattr(cls, attr_name)

        if cls.parent_feed is not None:
            return cls.parent_feed.get_setting(attr_name)

        return getattr(cls, attr_name)

    @classmethod
    def calculate_embedding(cls, text: str) -> list[float]:
        encoder = app_settings.ENCODER_CLASS
        embedding = encoder.text_to_vector(
            text, cls.get_setting("embedding_model_name")
        )
        return embedding

    @classmethod
    def calculate_user_embedding(
        cls, likes_queryset, content_field_name: str | None = None
    ) -> list[float] | None:
        limit = cls.get_setting("user_likes_limit")
        prefix = f"{content_field_name}__" if content_field_name else ""

        filter_kwargs = {f"{prefix}embedding__isnull": False}
        values_field = f"{prefix}embedding"

        recent_emb = list(
            likes_queryset.filter(**filter_kwargs)
            .order_by("-id")[:limit]
            .values_list(values_field, flat=True)
        )

        if not recent_emb:
            return None

        vectors_array = np.asarray(recent_emb, dtype=np.float32)
        mean_vector = np.mean(vectors_array, axis=0)

        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm

        return mean_vector.tolist()

    @classmethod
    def get_user_vector(cls, user) -> list[float] | None:
        """
        Retrieves the averaged interaction vector for a specific user
        scoped to this particular feed instance.
        """
        if user is None or not user.is_authenticated:
            return None

        target_feed_id = cls.get_setting("feed_id")

        if not target_feed_id:
            return None

        try:
            profile = UserFeedProfile.objects.filter(
                user_id=user.id, feed_id=target_feed_id
            ).first()

            return profile.embedding if profile else None

        except Exception as e:
            logger.error(
                f"DNF: Error fetching user vector for feed '{target_feed_id}': {e}"
            )
            return None

    @classmethod
    def set_user_vector(
        cls, user, *, embedding=None, keywords=None
    ) -> tuple["UserFeedProfile", bool] | None:
        """Changes user vector to provided embedding or specific keywords."""

        if user is None or not user.is_authenticated:
            return None

        if embedding is not None and keywords is not None:
            raise ValueError(
                "DNF: Cannot set user vector using both 'embedding' and 'keywords' simultaneously. "
                "Provide only one of them."
            )

        if embedding is None and keywords is None:
            raise ValueError(
                "DNF: Neither 'embedding' nor 'keywords' was provided. "
                "You must supply at least one argument to set the user vector."
            )

        target_feed_id = cls.get_setting("feed_id")
        if not target_feed_id:
            return None

        # If keywords provided, calculate embedding based on its type
        if keywords is not None:
            if isinstance(keywords, list):
                # Ensure all elements in the list are strings
                if not all(isinstance(k, str) for k in keywords):
                    raise ValueError(
                        "DNF: All elements in 'keywords' list must be strings."
                    )

                embedding = cls.calculate_embedding(" ".join(keywords))
            elif isinstance(keywords, str):
                embedding = cls.calculate_embedding(keywords)
            else:
                raise ValueError(
                    f"DNF: Invalid type for 'keywords'. Expected list[str] or str, got {type(keywords).__name__}."
                )

        try:
            if embedding is None:
                return None

            return UserFeedProfile.objects.update_or_create(
                user_id=user.id,
                feed_id=target_feed_id,
                defaults={"embedding": embedding},
            )

        except Exception as e:
            logger.error(
                f"DNF: Error saving user vector for feed '{target_feed_id}': {e}"
            )
            return None

    @classmethod
    def get_candidates(cls, user, queryset, excluded_ids=None):
        if queryset is None:
            queryset = cls.get_setting("content_django_model").objects.all()  # type: ignore

        if excluded_ids is not None:
            queryset = queryset.exclude(id__in=excluded_ids)

        return queryset

    @classmethod
    def rank_candidates(cls, queryset, user_profile_vector):
        if user_profile_vector is not None:
            queryset = queryset.annotate(
                similarity=Coalesce(
                    -MaxInnerProduct("embedding", user_profile_vector), Value(0.0)
                ),
            )
        else:
            queryset = queryset.annotate(
                similarity=Value(0.0, output_field=FloatField())
            )

        queryset = queryset.annotate(
            popularity=Coalesce(cls.get_setting("popularity_expression"), Value(0.0)),
            freshness=Coalesce(cls.get_setting("freshness_expression"), Value(0.0)),
        )

        queryset = queryset.annotate(
            score=cls.get_setting("weight_similarity") * F("similarity")
            + cls.get_setting("weight_freshness") * F("freshness")
            + cls.get_setting("weight_popularity") * F("popularity")
        ).order_by("-score")

        return queryset

    @classmethod
    def get_feed(cls, user, queryset=None, excluded_ids=None, limit: int = 20):
        """Get personalized feed for user."""
        hnsw = cls.get_setting("hnsw_config")
        user_profile_vector = cls.get_user_vector(user)

        using_hnsw = bool(
            hnsw and hnsw.get("ENABLED") and user_profile_vector is not None
        )

        candidates_qs = cls.get_candidates(
            user, queryset, None if using_hnsw else excluded_ids
        )

        # multi-channel retrieval under HNSW mode
        if using_hnsw:
            from django.db import connections, transaction

            db_alias = candidates_qs.db
            ef_search = hnsw.get("EF_SEARCH", 40)
            base_search_pool = hnsw.get("SEARCH_POOL", 500)
            excluded_count = len(excluded_ids) if excluded_ids else 0
            search_pool = base_search_pool + excluded_count

            w_sim = cls.get_setting("weight_similarity")
            w_fresh = cls.get_setting("weight_freshness")
            w_pop = cls.get_setting("weight_popularity")
            total_w = w_sim + w_fresh + w_pop or 1.0

            # proportions of the search pool
            pool_sim = max(1, int(search_pool * (w_sim / total_w)))
            pool_fresh = max(1, int(search_pool * (w_fresh / total_w)))
            pool_pop = max(1, int(search_pool * (w_pop / total_w)))

            candidate_ids = set()

            # simmilarity channel
            if w_sim > 0 and pool_sim > 0:
                with transaction.atomic(using=db_alias):
                    with connections[db_alias].cursor() as cursor:
                        cursor.execute("SET LOCAL hnsw.ef_search = %s;", [ef_search])

                    knn_qs = candidates_qs.order_by(
                        MaxInnerProduct("embedding", user_profile_vector)
                    )[:pool_sim]
                    candidate_ids.update(knn_qs.values_list("id", flat=True))

            # freshness channel
            if w_fresh > 0 and pool_fresh > 0:
                fresh_qs = candidates_qs.annotate(
                    f_val=cls.get_setting("freshness_expression")
                ).order_by("-f_val")[:pool_fresh]
                candidate_ids.update(fresh_qs.values_list("id", flat=True))

            # popularity channel
            if w_pop > 0 and pool_pop > 0:
                pop_qs = candidates_qs.annotate(
                    p_val=cls.get_setting("popularity_expression")
                ).order_by("-p_val")[:pool_pop]
                candidate_ids.update(pop_qs.values_list("id", flat=True))

            # Drop excluded items in memory before fetching objects
            if excluded_ids:
                candidate_ids.difference_update(excluded_ids)

            candidates_qs = candidates_qs.filter(id__in=list(candidate_ids))

        ranked_qs = cls.rank_candidates(candidates_qs, user_profile_vector)

        return ranked_qs[:limit]
