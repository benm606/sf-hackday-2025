import sys
import asyncio
import os

# Add the parent directory to the Python path so we can import from db/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_db import extract_and_save_problems_from_convo

async def main():
    await extract_and_save_problems_from_convo(convo_path='db/mock/convo-v1.json', db_path='db/problems_vector_db.pkl')

if __name__ == "__main__":
    asyncio.run(main()) 