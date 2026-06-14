from django_neural_feed.feeds import BaseNeuralFeed
from .models import TestArticle, TestLikeModel
from django.db.models import Count, F, FloatField, ExpressionWrapper, Value
from django.db.models.functions import Extract, Now


class TestParentFeed(BaseNeuralFeed):
    feed_id = "test_parent"
    content_django_model = TestArticle
    interaction_django_model = TestLikeModel
    user_field_name = "user"
    content_field_name = "article"
    user_likes_limit = 3

    freshness_expression = ExpressionWrapper(
        Value(1.0)
        / (Value(1.0) + (Extract(Now() - F("created_at"), "epoch") / 3600.0)),
        output_field=FloatField(),
    )


class TestChildFeed(BaseNeuralFeed):
    parent_feed = TestParentFeed
