from django.db import models
from pgvector.django import VectorField, HnswIndex
from django_neural_feed.conf import app_settings


class NeuralRecommendMixin(models.Model):
    """
    Abstract mixin for content models (e.g., Post, Article, Product)
    that require semantic vector representation.
    """

    # NOTE: Changing VECTOR_DIMENSION in settings requires generating
    # a new Django migration in the host project to alter the database column.
    embedding = VectorField(
        dimensions=app_settings.VECTOR_DIMENSION, null=True, blank=True
    )

    class Meta:
        abstract = True

    def get_ready_text(self) -> str:
        """
        Must be implemented by the target model. Should return a combined
        string of all text fields intended for vectorization.
        """
        raise NotImplementedError("You should assign get_ready_text() in your model!")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_text = self.get_ready_text() if self.pk else None


class NeuralHnswMixin:
    """
    Optional mixin to automatically provision a high-performance
    HNSW index for the embedding field.
    """

    class Meta:
        abstract = True
        indexes = [
            HnswIndex(
                name="%(app_label)s_%(class)s_hnsw_idx",  # Dynamic safe naming
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_ip_ops"],
            )
        ]
