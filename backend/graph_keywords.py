import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set.")

MODEL = "gemma-4-31b-it"

# ---------------------------------------------------------
# Gemini model
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

# ---------------------------------------------------------
# Output schema
# ---------------------------------------------------------

class KeywordStrategy(BaseModel):
    topic: str
    search_intent: str = Field(
        description="Primary search intent of the topic"
    )
    primary_keywords: List[str] = Field(
        description="Main focus keywords for the blog"
    )
    secondary_keywords: List[str] = Field(
        description="Supporting SEO keywords"
    )
    long_tail_keywords: List[str] = Field(
        description="Long-tail search phrases"
    )
    question_keywords: List[str] = Field(
        description="Questions users may search"
    )
    semantic_keywords: List[str] = Field(
        description="Related entities, concepts and terminology"
    )
    blog_title_ideas: List[str] = Field(
        description="SEO-friendly blog title ideas"
    )
    content_angles: List[str] = Field(
        description="Different content angles for this topic"
    )

# ---------------------------------------------------------
# Structured Gemini model
# ---------------------------------------------------------

structured_llm = llm.with_structured_output(KeywordStrategy)

# ---------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------

class BlogSEOState(TypedDict):
    trending_search: str
    keyword_strategy: dict

# ---------------------------------------------------------
# Node 1: Generate SEO keyword strategy
# ---------------------------------------------------------

def generate_keywords(state: BlogSEOState):
    print("in generate keyword")
    trending_search = state["trending_search"]
    print("1")
    prompt = f"""
You are an expert SEO keyword strategist.
A trending search query has been detected:
TRENDING SEARCH:
{trending_search}
Your task is to create a keyword strategy that can be used
to generate SEO-focused blog articles.
Requirements:
1. Identify the main search intent.
2. Generate primary keywords.
   These should be strong candidates for the main focus keyword
   of an article.
3. Generate secondary keywords.
   These should support the main topic.
4. Generate long-tail keywords.
   These should represent specific searches users might perform.
5. Generate question keywords.
   These should resemble natural questions people could search.
6. Generate semantic keywords.
   Include closely related concepts, entities, terminology,
   subtopics and phrases.
7. Generate SEO-friendly blog title ideas.
8. Generate different content angles.
Important rules:
- Do NOT invent search volume.
- Do NOT invent keyword difficulty.
- Do NOT invent CPC.
- Do NOT claim that a keyword is actually trending unless
  that information was provided.
- Focus on semantic relevance and search intent.
- Avoid keyword stuffing.
- Avoid duplicate keywords.
- Make the keywords useful for an actual blog-writing pipeline.

Return the result according to the provided JSON schema.
"""

    print("2")
    result = structured_llm.invoke(prompt)
    print("3")
    return {
        "keyword_strategy": result.model_dump()
    }

# ---------------------------------------------------------
# Node 2: Validate / clean output
# ---------------------------------------------------------

def validate_keywords(state: BlogSEOState):
    print("in validate keyword")
    strategy = state["keyword_strategy"]
    print("1")
    # Remove duplicate keywords while preserving order
    for field in [
        "primary_keywords",
        "secondary_keywords",
        "long_tail_keywords",
        "question_keywords",
        "semantic_keywords",
        "blog_title_ideas",
        "content_angles",
    ]:
        print("2")

        if field in strategy:
            print("3")
            seen = set()
            cleaned = []
            for item in strategy[field]:
                normalized = item.strip().lower()
                if normalized not in seen:
                    seen.add(normalized)
                    cleaned.append(item.strip())
            strategy[field] = cleaned
            print("4")
    return {
        "keyword_strategy": strategy
    }

# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

graph = StateGraph(BlogSEOState)
graph.add_node(
    "generate_keywords",
    generate_keywords
)
graph.add_node(
    "validate_keywords",
    validate_keywords
)
graph.set_entry_point("generate_keywords")
graph.add_edge(
    "generate_keywords",
    "validate_keywords"
)
graph.add_edge(
    "validate_keywords",
    END
)
app = graph.compile()

# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

# if __name__ == "__main__":

#     trending_search = "digital product engineering services india"

#     result = app.invoke({
#         "trending_search": trending_search
#     })

#     print(
#         json.dumps(
#             result["keyword_strategy"],
#             indent=2,
#             ensure_ascii=False
#         )
#     )