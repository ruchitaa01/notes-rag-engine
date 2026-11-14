from sentence_transformers import SentenceTransformer

class DocumentEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the SentenceTransformer model for generating 384-dimensional dense vectors.
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generates a vector embedding for a single chunk of text.
        """
        vector = self.model.encode(text)
        return vector.tolist()

    def generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generates vector embeddings for a batch of text chunks efficiently.
        """
        vectors = self.model.encode(texts)
        return vectors.tolist()