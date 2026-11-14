import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.database import init_db, get_db_connection
from app.chunker import TextChunker
from app.embedder import DocumentEmbedder
from app.pdf_processor import PDFProcessor

app = FastAPI(title="Smart Document & Notes RAG Engine", version="1.0.0")

# Initialize database and load ML model on startup
@app.on_event("startup")
def startup_event():
    init_db()
    global embedder
    embedder = DocumentEmbedder()

class SearchQuery(BaseModel):
    query: str
    limit: int = 5

@app.get("/")
def read_root():
    return {"message": "Welcome to the Smart Document & Notes RAG Engine API!"}

@app.post("/ingest-pdf/")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingests a PDF file, extracts text, chunks it, generates embeddings, 
    and stores them in PostgreSQL with pgvector.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Save uploaded file temporarily in the data folder
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", file.filename)
    
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
        
    try:
        # 1. Extract text
        raw_text = PDFProcessor.extract_text_from_pdf(file_path)
        
        # 2. Chunk text
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split_text(raw_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")
            
        # 3. Generate embeddings
        embeddings = embedder.generate_batch_embeddings(chunks)
        
        # 4. Store in database
        conn = get_db_connection()
        with conn.cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO document_chunks (content, embedding, metadata)
                    VALUES (%s, %s, %s);
                    """,
                    (chunk, embedding, {"filename": file.filename})
                )
            conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_ingested": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/")
def search_documents(payload: SearchQuery):
    """
    Performs vector similarity search across stored document chunks.
    """
    try:
        query_vector = embedder.generate_embedding(payload.query)
        
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Query using pgvector cosine distance operator (<=>)
            cur.execute(
                """
                SELECT content, metadata, 1 - (embedding <=> %s) AS similarity
                FROM document_chunks
                ORDER BY embedding <=> %s
                LIMIT %s;
                """,
                (query_vector, query_vector, payload.limit)
            )
            rows = cur.fetchall()
        conn.close()
        
        results = [
            {"content": row[0], "metadata": row[1], "similarity": float(row[2])}
            for row in rows
        ]
        
        return {"query": payload.query, "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))