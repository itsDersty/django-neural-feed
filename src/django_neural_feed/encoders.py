import logging
import numpy as np

logger = logging.getLogger(__name__)


class BaseVectorEncoder:
    """
    Abstract interface for all vector encoders.
    Developers can subclass this to implement custom vector logic (e.g., OpenAI, Cohere).
    """

    @classmethod
    def text_to_vector(cls, text: str, model_name: str) -> list[float]:
        raise NotImplementedError("Subclasses must implement text_to_vector method.")

    @classmethod
    def average_vectors(cls, vectors: list[list[float]], limit: int) -> list[float]:
        """Default high-performance vector averaging using NumPy."""
        if not vectors:
            return []
        try:
            matrix = np.array(vectors[:limit], dtype=np.float32)
            # Guard against empty mock structures or 0-size arrays
            if matrix.size == 0:
                return []
            return np.mean(matrix, axis=0).tolist()
        except Exception as e:
            logger.error(f"DNF Vector averaging failed: {e}")
            return []


class DefaultVectorEncoder(BaseVectorEncoder):
    """
    Built-in default encoder using the 'sentence-transformers' library.
    """

    _model_instances = {}

    @classmethod
    def _get_model(cls, model_name: str):
        if model_name not in cls._model_instances:
            try:
                from sentence_transformers import SentenceTransformer

                try:
                    cls._model_instances[model_name] = SentenceTransformer(
                        model_name, local_files_only=True
                    )
                except Exception:
                    logger.info(
                        f"DNF: Model '{model_name}' not found locally. Downloading..."
                    )
                    cls._model_instances[model_name] = SentenceTransformer(
                        model_name, local_files_only=False
                    )
            except ImportError:
                logger.error(
                    "DNF Error: 'sentence-transformers' is required for the built-in "
                    "DefaultVectorEncoder but is not installed. Install it with "
                    "`pip install django-neural-feed[local]`, or configure a custom "
                    "ENCODER_CLASS that does not need it."
                )
                raise
        return cls._model_instances[model_name]

    @classmethod
    def text_to_vector(cls, text: str, model_name: str) -> list[float]:
        if not text.strip():
            return []
        try:
            model = cls._get_model(model_name)
            # convert_to_numpy=True ensures we get a clean arrays back
            embedding = model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"DNF Default embedding generation failed: {e}")
            return []
