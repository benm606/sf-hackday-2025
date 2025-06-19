import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import sys

# Add the parent directory to the Python path so we can import from .env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

class VectorDB:
    def __init__(self, db_path='db/problems_vector_db.pkl', model_name='all-MiniLM-L6-v2'):
        self.db_path = db_path
        self.model = SentenceTransformer(model_name)
        self.vectors = []
        self.texts = []
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
                self.vectors = data.get('vectors', [])
                self.texts = data.get('problems', [])
        else:
            self.vectors = []
            self.texts = []

    def _save(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump({'vectors': self.vectors, 'problems': self.texts}, f)

    def add(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = [self.model.encode(t) for t in texts]
        self.vectors.extend(vectors)
        self.texts.extend(texts)
        self._save()

    def search(self, query, top_k=3):
        if not self.vectors:
            return []
        query_vec = self.model.encode(query)
        sims = [np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v)) for v in self.vectors]
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [(self.texts[i], sims[i]) for i in top_indices]

    def cluster_problems(self, similarity_threshold=0.6):
        """
        Cluster problems by similarity and return a list of (problem, count) tuples.
        """
        if not self.vectors:
            return []
        clusters = []
        used = set()
        for i, vec in enumerate(self.vectors):
            if i in used:
                continue
            cluster = [i]
            for j, other_vec in enumerate(self.vectors):
                if j != i and j not in used:
                    sim = np.dot(vec, other_vec) / (np.linalg.norm(vec) * np.linalg.norm(other_vec))
                    if sim >= similarity_threshold:
                        cluster.append(j)
                        used.add(j)
            used.add(i)
            if cluster:
                clusters.append(cluster)
        result = []
        for cluster in clusters:
            problem = self.texts[cluster[0]]
            count = len(cluster)
            result.append((problem, count))
        return result


#################
class Problem(BaseModel):
    description: str
    context: str = ""

class ProblemExtraction(BaseModel):
    problems: List[Problem]

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

async def extract_and_save_problems_from_convo(convo_path: str, db_path: str = 'db/problems_vector_db.pkl') -> List[str]:
    """
    Given a conversation JSON file, use OpenAI's latest model to extract user problems and save to the vector DB.
    
    Args:
        convo_path: Path to the conversation JSON file
        db_path: Path to save the vector database
    
    Returns:
        List[str]: List of extracted problems
    """
    try:
        # Load conversation
        with open(convo_path, 'r') as f:
            convo = json.load(f)
        
        # Extract user messages
        user_messages = [item['content'][0] for item in convo['items'] if item['role'] == 'user']
        convo_text = '\n'.join(user_messages)
        
        # Create structured prompt
        prompt = (
            "Given the following conversation, extract a list of the specific problems, pain points, or frustrations the user describes. "
            "For each problem, provide a clear description and any relevant context from the conversation.\n\n"
            "Conversation:\n" + convo_text + "\n\n"
            "Return the problems in a structured format with descriptions and context."
        )

        # Make API call with structured response format
        response = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert at extracting user pain points from conversations. "
                             "Return problems in a clear, structured format."
                },
                {"role": "user", "content": prompt}
            ],
            response_format=ProblemExtraction,
            temperature=0.2,
            max_tokens=1000
        )

        # Parse response
        problems_data = response.choices[0].message.parsed
        problems = [p.description for p in problems_data.problems]
        
        # Filter out short problems
        problems = [p for p in problems if len(p) > 10]

        print(f"Extracted {len(problems)} problems")
        print(problems)
        
        # Save to vector DB
        db = VectorDB(db_path=db_path)
        db.add(problems)
        
        print(f"Successfully saved {len(problems)} problems to vector DB at {db.db_path}")
        return problems

    except FileNotFoundError:
        print(f"Error: Conversation file not found at {convo_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in conversation file {convo_path}")
        return []
    except Exception as e:
        print(f"Error extracting problems: {str(e)}")
        return []