from django.conf import settings
from django.db.models import Value
from django_neural_feed.encoders import DefaultVectorEncoder, BaseVectorEncoder
from django.utils.module_loading import import_string

# Fallback defaults if the developer skips these keys in settings.py
DEFAULT_CONFIG = {
    "MODEL_NAME": "paraphrase-multilingual-MiniLM-L12-v2",
    "VECTOR_DIMENSION": 384,
    "WEIGHT_SIMILARITY": 0.6,
    "WEIGHT_FRESHNESS": 0.2,
    "WEIGHT_POPULARITY": 0.2,
    "USER_LIKES_LIMIT": 20,
    "CELERY_ENABLED": False,
    "ENCODER_CLASS": DefaultVectorEncoder,
    "FRESHNESS_EXPRESSION": Value(1.0),
    "POPULARITY_EXPRESSION": Value(1.0),
    "FEEDS": [],
    "HNSW": {
        "ENABLED": False,
        "EF_SEARCH": 40,
        "SEARCH_POOL": 500,
    },
}


class AppSettings:
    """
    Thread-safe configuration proxy for Django Neural Feed (DNF).
    Provides default fallbacks and clean property accessors.
    """

    def __init__(self):
        # Using a unified short prefix 'DJANGO_NEURAL_FEED' to match the official documentation
        self._user_config = getattr(settings, "DJANGO_NEURAL_FEED", {})

    def _get_setting(self, key):
        """Internal helper to resolve user configurations with built-in fallbacks."""
        return self._user_config.get(key, DEFAULT_CONFIG[key])

    @property
    def MODEL_NAME(self) -> str:
        """The HuggingFace text-embedding-inference or sentence-transformer model name."""
        return self._get_setting("MODEL_NAME")

    @property
    def VECTOR_DIMENSION(self) -> int:
        """The dimension size of generated dense vectors (e.g., 384 for E5-small)."""
        return self._get_setting("VECTOR_DIMENSION")

    @property
    def WEIGHT_SIMILARITY(self) -> float:
        """The weight multiplier for the semantic similarity score (cosine distance)."""
        return self._get_setting("WEIGHT_SIMILARITY")

    @property
    def WEIGHT_FRESHNESS(self) -> float:
        """The weight multiplier for the content recency/freshness score."""
        return self._get_setting("WEIGHT_FRESHNESS")

    @property
    def WEIGHT_POPULARITY(self) -> float:
        """The weight multiplier for the content popularity/engagement score."""
        return self._get_setting("WEIGHT_POPULARITY")

    @property
    def FRESHNESS_EXPRESSION(self):
        """
        Django ORM expression or database function for calculating the freshness metric.
        Defaults to a neutral constant Value(1.0) to avoid hard crashes during quickstart.
        """
        return self._user_config.get("FRESHNESS_EXPRESSION", Value(1.0))

    @property
    def POPULARITY_EXPRESSION(self):
        """
        Django ORM expression or database function for calculating the popularity metric.
        Defaults to a neutral constant Value(1.0) to avoid hard crashes during quickstart.
        """
        return self._user_config.get("POPULARITY_EXPRESSION", Value(1.0))

    @property
    def USER_LIKES_LIMIT(self) -> int:
        """The maximum number of recent active likes sliced to build a user vector profile."""
        return self._get_setting("USER_LIKES_LIMIT")

    @property
    def CELERY_ENABLED(self) -> bool:
        """Flag toggling background task delegation for embedding generation pipelines."""
        return self._get_setting("CELERY_ENABLED")

    @property
    def ENCODER_CLASS(self) -> type[BaseVectorEncoder]:
        """Dynamically resolves and returns the configured encoder class."""
        setting_value = self._get_setting("ENCODER_CLASS")

        if isinstance(setting_value, str):
            try:
                return import_string(setting_value)
            except ImportError as e:
                raise ImportError(
                    f"DNF: Could not import custom encoder class at '{setting_value}'. "
                    f"Check your DJANGO_NEURAL_FEED['ENCODER_CLASS'] setting. Error: {e}"
                )

        return setting_value

    def get_registered_feeds(self) -> list:
        """Loads and returns configuration classes defined by user in Django settings."""
        from django.utils.module_loading import import_string

        feed_paths = self._get_setting("FEEDS") or []

        feed_classes = []
        for path in feed_paths:
            try:
                feed_class = import_string(path)
                feed_classes.append(feed_class)
            except ImportError as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"DNF: Cannot import feed class from path '{path}': {e}")

        return feed_classes

    @property
    def HNSW(self) -> dict:
        """Returns merged HNSW configuration."""
        user_hnsw = self._user_config.get("HNSW", {})
        if not isinstance(user_hnsw, dict):
            user_hnsw = {}  # Fallback to empty dict if user passed garbage
        return {**DEFAULT_CONFIG["HNSW"], **user_hnsw}


# Global instantiation for direct import across the library ecosystem
app_settings = AppSettings()
