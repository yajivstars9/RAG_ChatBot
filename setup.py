# setup_db.py

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    conninfo="postgresql://rag:rag@localhost:5433/RAG_storage",
    max_size=5,
    kwargs={"autocommit": True} 
)

checkpointer = PostgresSaver(pool)

# ✅ Run once
checkpointer.setup()

print("✅ LangGraph Postgres tables created successfully")