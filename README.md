# Django Neural Feed (DNF)

<p align="center">  
  <a href="https://github.com/ItsDersty/django-neural-feed/actions/workflows/main.yml">  
    <img src="https://img.shields.io/github/actions/workflow/status/ItsDersty/django-neural-feed/main.yml?branch=feature/oop-architecture&style=flat-square&label=tests" alt="Build Status">  
  </a>  
  <a href="https://github.com/ItsDersty/django-neural-feed">  
    <img src="https://img.shields.io/badge/coverage-100%25-brightgreen?style=flat-square" alt="Coverage">  
  </a>  
  <a href="https://pypi.org/project/django-neural-feed/">  
    <img src="https://img.shields.io/pypi/v/django-neural-feed?style=flat-square&color=blue" alt="PyPI Version">  
  </a>  
  <a href="https://github.com/ItsDersty/django-neural-feed/blob/main/LICENSE">  
    <img src="https://img.shields.io/github/license/ItsDersty/django-neural-feed?style=flat-square&color=green" alt="License">  
  </a>  
  <a href="https://github.com/ItsDersty/django-neural-feed">  
    <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square" alt="Python Version">  
  </a>  
  <a href="https://github.com/ItsDersty/django-neural-feed">  
    <img src="https://img.shields.io/badge/django-4.2%2B-darkgreen?style=flat-square" alt="Django Version">  
  </a>  
</p>

## Overview

**Django Neural Feed (DNF)** is a production-ready Django application designed to build intelligent, personalized content feeds powered by semantic vector embeddings. It leverages PostgreSQL's `pgvector` extension to compute vector similarity at the database level, combined with customizable content freshness and popularity metrics—all evaluated in a single optimized SQL query.

With its object-oriented architecture, DNF decouples your configuration logic into dedicated Feed classes. It tracks user interactions non-intrusively via Django signals and supports flexible deployment execution blocks, easily falling back from Celery asynchronous queues to synchronous background threads if the broker is offline.

## Core Features

- **🧠 Object-Oriented Feed Configuration**: Define isolated, multi-tenant recommendation feeds by subclassing a unified `BaseNeuralFeed` class.  
- **⚡ Bulletproof Asynchronous Pipeline**: Offload embedding generation and vector aggregation to Celery. Features an automated synchronous thread fallback system.  
- **📊 Dedicated Multi-Feed User Profiles**: Stores vector profiles in an isolated `UserFeedProfile` model partitioned by `feed_id`, keeping your core Auth User table clean.  
- **🎯 Hybrid Multi-Criteria Scoring**: Merges semantic similarity (pgvector cosine distance), content recency, and custom popularity expressions into a single database-level annotation.  
- **🚀 Non-Invasive Integration**: Attach recommendation behavior to existing content models with minimal migrations, leaving your interaction tables (Likes/Dislikes) completely untouched.

## Requirements

- **Python**: 3.10+  
- **Django**: 4.2, 5.0, 6.0+  
- **PostgreSQL**: 12+ (with `pgvector` extension installed)  
- **NumPy**: 2.0.0+  
- **pgvector**: 0.4.0+  
- **SentenceTransformers**: 3.0.0+

## Installation

### 1. Install the Package

```bash  
pip install django-neural-feed
```

The built-in `DefaultVectorEncoder` uses `sentence-transformers` (which pulls in
torch). It is an **optional** extra, so install it only if you rely on the default
local encoder:

```bash
pip install django-neural-feed[local]
```

If you configure a custom `ENCODER_CLASS` (OpenAI, Cohere, a hosted inference API,
etc.) you can skip it and avoid the torch download entirely.
### **2. Add to Django Settings**

```python  
INSTALLED_APPS = [  
    # ... other apps  
    'django_neural_feed',  
]
```
### **3. Initialize PostgreSQL Extension**

Ensure pgvector is enabled in your database instance:

```sql  
CREATE EXTENSION IF NOT EXISTS vector;
```

## **Quick Start**

### **Step 1: Configure Your Content Model**

Inherit from NeuralRecommendMixin to inject a vector embedding column into your target content table.

```python  
from django.conf import settings
from django.db import models  
from django_neural_feed.mixins import NeuralRecommendMixin

class Post(NeuralRecommendMixin, models.Model): # NOTE: NeuralRecommendMixin must be BEFORE models.Model!
    title = models.CharField(max_length=255)  
    content = models.TextField()  
    created_at = models.DateTimeField(auto_now_add=True)
    likes = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name="liked_posts")

    def get_ready_text(self) -> str:  
        return f"{self.title} {self.content}"
```

Prepare and apply your migrations:

```bash  
python manage.py makemigrations  
python manage.py migrate
```

### **Step 2: Define a Custom Feed Class**

Create a dedicated `feeds.py` configuration to encapsulate tracking thresholds, model fields, math scoring expressions, and hybrid weights.

```python 
from django.db.models import Count, F, FloatField, ExpressionWrapper, Value
from django.db.models.functions import Cast, Ln, Extract, Now
from django_neural_feed.feeds import BaseNeuralFeed  
from your_app.models import Post

class PostFeed(BaseNeuralFeed):  
    # 1. Core Feed Identity
    feed_id = "posts_main"  
    parent_feed = None          # Optional: Reference to a parent feed class for inheritance hierarchy

    # 2. Target Django Models Configuration
    content_django_model = Post
    interaction_django_model = Post.likes.through
    
    # 3. Interaction Tracking Pipelines
    mode = "m2m"               # Use "m2m" for ManyToMany fields, or "model" for explicit through models
    user_field_name = "user"   # Field pointing to User model (not needed if mode is "m2m")
    content_field_name = "post" # Field pointing to Content model (not needed if mode is "m2m")

    # 4. Model & Pipeline Thresholds
    embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2" # Overrides global setting
    user_likes_limit = 20      # Max target sample size slice for vector profile aggregation

    # 5. Hybrid Scoring Global Weights (Should ideally sum up to 1.0)
    weight_similarity = 0.6
    weight_freshness = 0.2
    weight_popularity = 0.2

    # 6. Popularity: Logarithmic scaling using natural logarithm to keep viral jumps balanced
    # Ln(Value(1000.0)) scales the metric dynamically, hitting a 1.0 score modifier at 1000 likes.
    popularity_expression = ExpressionWrapper(
        Ln(Cast(Count("likes"), FloatField()) + Value(1.0)) / Ln(Value(1000.0)),
        output_field=FloatField()
    )

    # 7. Freshness: Time-decay function based on post age in hours
    # Safely subtracts timestamps inside the database, converting the interval to hours.
    freshness_expression = ExpressionWrapper(
        Value(1.0) / (
            Value(1.0) + (
                Extract(Now() - F("created_at"), "epoch") / 3600.0
            )
        ),
        output_field=FloatField()
    )
```

### **Step 3: Register Feed in Settings**

Register the string path to your custom feed configuration within the `DJANGO_NEURAL_FEED["FEEDS"]` list inside your `settings.py`.

```python
DJANGO_NEURAL_FEED = {
    "FEEDS": [
        "your_app.feeds.PostFeed", # DNF hooks up all model and M2M signals automatically
    ],
}
```

### **Step 4: Fetch Personalized Feed Results**

Use your feed's `.get_feed()` function to obtain optimized querysets sorted by hybrid weights.

```python  
from your_app.feeds import PostFeed
from your_app.models import Post

def user_feed_view(request):
    # Gather IDs of posts the user has already liked to exclude them from the feed
    excluded_ids = Post.objects.filter(
        likes=request.user
    ).values_list('id', flat=True)
    
    # Generate personalized recommendations directly via your Feed class
    feed_queryset = PostFeed.get_feed(
        user=request.user,
        queryset=Post.objects.all(),
        excluded_ids=excluded_ids,
        limit=20
    )
    return feed_queryset
```

### **Configuration Reference**

You can pass default global limits and model engine backends via standard DJANGO_NEURAL_FEED dictionary keys in your settings.py:

```python  
DJANGO_NEURAL_FEED = {    
    "MODEL_NAME": "paraphrase-multilingual-MiniLM-L12-v2",    
    "VECTOR_DIMENSION": 384,    
    "CELERY_ENABLED": True,    
    "WEIGHT_SIMILARITY": 0.6,    
    "WEIGHT_FRESHNESS": 0.2,    
    "WEIGHT_POPULARITY": 0.2,    
}
```

| Global Config Key | Type | Default | Purpose |
| :---- | :---- | :---- | :---- |
| MODEL_NAME | str | paraphrase-multilingual-MiniLM-L12-v2 | Target HuggingFace SentenceTransformer engine. |
| VECTOR_DIMENSION | int | 384 | Embedding dense matrix array dimension sizes. |
| ENCODER_CLASS | str/type | django_neural_feed.encoders.DefaultVectorEncoder | Path to the vectorization engine class interface. |
| WEIGHT_SIMILARITY | float | 0.6 | Default proportional weight of cosine similarity scoring. |
| WEIGHT_FRESHNESS | float | 0.2 | Default proportional weight of item creation recency. |
| WEIGHT_POPULARITY | float | 0.2 | Default proportional weight of user interaction counts. |
| USER_LIKES_LIMIT | int | 20 | Max target sample size slice for vector aggregation. |
| CELERY_ENABLED | bool | False | Toggles routing tasks to background Celery workers. |

### **Advanced Settings Overriding**

Every specific attribute can be declared dynamically within your custom BaseNeuralFeed class implementation to build separate configurations for multiple models (e.g., separate metrics weights for ArticlesFeed vs VideoFeed).

| Feed Class Attribute | Type | Default Value / Fallback | Purpose |
| :---- | :---- | :---- | :---- |
| feed_id | str | "default_feed" | Unique identifier for partitioning user vector profiles. |
| mode | str | *Required* ("m2m" | "model") | Toggles the internal signal tracking pipeline architecture. |
| embedding_model_name | str | settings.MODEL_NAME | Overrides the text-embedding engine for this specific feed. |
| user_likes_limit | int | settings.USER_LIKES_LIMIT | Overrides the history interaction slice size for this feed. |
| weight_similarity | float | settings.WEIGHT_SIMILARITY | Fine-tunes semantic similarity importance for this feed. |
| weight_freshness | float | settings.WEIGHT_FRESHNESS | Fine-tunes time-decay metric importance for this feed. |
| weight_popularity | float | settings.WEIGHT_POPULARITY | Fine-tunes interaction count importance for this feed. |
| popularity_expression | Expression | Value(1.0) | Custom Django ORM expression for parsing popularity scoring. |
| freshness_expression | Expression | Value(1.0) | Custom Django ORM expression for parsing time-decay scoring. |

## **Architecture Mechanics**

1. **Content Structuring**: When an entity subclassing NeuralRecommendMixin fires a post_save execution block, DNF reads get_ready_text() to calculate a dense float vector.  
2. **Preference Profiling**: On target connection updates, an isolated worker fetches the latest interaction history rows, calculates an averaged, L2-normalized mean representation vector, and updates UserFeedProfile.  
3. **Query Engine Generation**: Invoking Feed.get_feed() applies pgvector operations combined with standard math normalization, avoiding redundant lookups.

## **Custom Vector Encoders (Advanced)**

By default, DNF uses local `sentence-transformers` via `DefaultVectorEncoder`. If you prefer to use external cloud APIs (like OpenAI, Cohere, or custom microservices) for generating embeddings, you can implement a custom encoder.

### 1. Subclass BaseVectorEncoder

Create a custom encoder class anywhere in your Django project and implement the text_to_vector method. You can optionally override average_vectors if you want to replace NumPy-based processing.

```python
import requests
from django_neural_feed.encoders import BaseVectorEncoder

class OpenAIAppEncoder(BaseVectorEncoder):
    @classmethod
    def text_to_vector(cls, text: str, model_name: str) -> list[float]:
        """Fetch embeddings via custom third-party cloud API endpoint."""
        if not text.strip():
            return []
            
        # Example API execution blueprint
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": "Bearer YOUR_API_KEY"},
            json={"input": text, "model": model_name}
        )
        return response.json()["data"][0]["embedding"]
```

### 2. Register Custom Encoder in Settings

Pass the absolute string path pointing to your class implementation inside the global dictionary configuration:

```python
DJANGO_NEURAL_FEED = {
    "ENCODER_CLASS": "your_app.encoders.OpenAIAppEncoder",
    "MODEL_NAME": "text-embedding-3-small", # Passed down as model_name argument
    "VECTOR_DIMENSION": 1536,                # Adjust to match your custom API provider
}
```

## **Testing**

DNF maintains full code coverage execution metrics. Run the suite natively using:

```bash  
pytest --cov=src/django_neural_feed
```

## **License**

Distributed under the terms of the MIT License.
