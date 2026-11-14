import re

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes our semantic text chunker.
        :param chunk_size: Maximum character length per chunk.
        :param chunk_overlap: Number of characters to overlap between chunks 
                              to preserve context across boundaries.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """
        Splits raw text into overlapping chunks using sentence/paragraph boundaries.
        """
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If we aren't at the end of the text, try to find a natural break (like a period or space)
            if end < len(text):
                # Look for the last space within the chunk limit to avoid cutting words in half
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward, accounting for overlap
            start = end - self.chunk_overlap

        return chunks