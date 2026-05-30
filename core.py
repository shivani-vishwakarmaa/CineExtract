from dotenv import load_dotenv
load_dotenv()



from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser


model = ChatMistralAI(model="mistral-small-2603")


class Movie(BaseModel):
    movie_name: str
    release_year: int
    genre: List[str]

    director: Optional[str]
    producer: Optional[str]
    production_company: Optional[str]

    main_cast: List[str]
    main_characters: List[str]

    villain_antagonist: Optional[str]

    plot_overview: str

    awards_achievements: Optional[List[str]]

    ratings_reviews: Optional[str]

    themes_tone: List[str]

    setting: Optional[str]

    special_highlights: List[str]

    quick_summary: str



parser=PydanticOutputParser(pydantic_object=Movie)








prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Extract movie information from the paragraph.
     {format_instructions}
     """),
    
    ("human", "{paragraph}")
])


para = input("Give your paragraph: ")


final_prompt = prompt.invoke({"paragraph": para,"format_instructions":parser.get_format_instructions()})

response = model.invoke(final_prompt)

print(response.content)