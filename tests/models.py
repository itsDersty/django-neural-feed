from django.db import models
from pgvector.django import VectorField
from django_neural_feed.mixins import NeuralRecommendMixin


class TestArticle(NeuralRecommendMixin, models.Model):
    title = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    embedding = VectorField(dimensions=3, null=True, blank=True)

    def get_ready_text(self) -> str:
        return self.title


class TestLikeModel(models.Model):
    user = models.ForeignKey("auth.User", on_delete=models.CASCADE)
    article = models.ForeignKey(TestArticle, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)


class TestM2MArticle(NeuralRecommendMixin, models.Model):
    title = models.CharField(max_length=100)
    embedding = VectorField(dimensions=3, null=True, blank=True)
    liked_by = models.ManyToManyField("auth.User", related_name="liked_m2m_articles")

    def get_ready_text(self) -> str:
        return self.title
