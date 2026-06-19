import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Info Extractor",
    page_icon="🎬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --gold: #C9A84C;
    --dark: #0D0D0D;
    --card: #161616;
    --border: #2a2a2a;
    --text: #E8E2D5;
    --muted: #7a7570;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--dark) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

h1, h2, h3, h4 {
    font-family: 'Playfair Display', serif !important;
    color: var(--gold) !important;
}

.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 3.2rem;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
}
.hero p {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 300;
}

/* textarea */
[data-testid="stTextArea"] textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    transition: border-color .2s;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,.15) !important;
}

/* button */
.stButton > button {
    background: var(--gold) !important;
    color: #000 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: .5px !important;
    border: none !important;
    border-radius: 4px !important;
    padding: .6rem 2.2rem !important;
    font-size: 0.95rem !important;
    transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
}

/* cards */
.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.info-card h4 {
    font-family: 'Playfair Display', serif;
    color: var(--gold) !important;
    font-size: 1rem;
    margin: 0 0 .5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: .78rem;
}
.info-card p, .info-card li {
    color: var(--text);
    font-size: .9rem;
    margin: 0;
    line-height: 1.6;
}
.info-card ul { margin: 0; padding-left: 1.1rem; }

.tag {
    display: inline-block;
    background: rgba(201,168,76,.12);
    border: 1px solid rgba(201,168,76,.3);
    color: var(--gold);
    padding: .2rem .6rem;
    border-radius: 3px;
    font-size: .78rem;
    margin: .18rem .15rem;
    font-weight: 500;
}

.movie-title-hero {
    text-align: center;
    padding: 2rem 1rem 1.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.8rem;
}
.movie-title-hero h2 {
    font-size: 2.4rem;
    margin-bottom: .2rem;
}
.movie-title-hero .year {
    color: var(--muted);
    font-size: .95rem;
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* spinner */
[data-testid="stSpinner"] { color: var(--gold) !important; }

/* hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatMistralAI(
    model="mistral-small-2603",
    api_key=st.secrets["MISTRAL_API_KEY"]
)

# ── Schema ────────────────────────────────────────────────────────────────────
class Movie(BaseModel):
    movie_name: str
    release_year: int
    genre: List[str]
    director: Optional[str] = None
    producer: Optional[str] = None
    production_company: Optional[str] = None
    main_cast: List[str]
    main_characters: List[str]
    villain_antagonist: Optional[str] = None
    plot_overview: str
    awards_achievements: Optional[List[str]] = None
    ratings_reviews: Optional[str] = None
    themes_tone: List[str]
    setting: Optional[str] = None
    special_highlights: List[str]
    quick_summary: str

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 Movie Info Extractor</h1>
    <p>Paste any movie description and extract structured information instantly</p>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
paragraph = st.text_area(
    "Movie Paragraph",
    placeholder="Paste a paragraph about any movie here…",
    height=180,
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 5])
with col_btn:
    extract = st.button("Extract Info", use_container_width=True)

# ── Extraction ────────────────────────────────────────────────────────────────
if extract:
    if not paragraph.strip():
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Analysing…"):
            try:
                model = get_model()
                parser = PydanticOutputParser(pydantic_object=Movie)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Extract movie information from the paragraph.\n{format_instructions}"),
                    ("human", "{paragraph}")
                ])
                final_prompt = prompt.invoke({
                    "paragraph": paragraph,
                    "format_instructions": parser.get_format_instructions()
                })
                response = model.invoke(final_prompt)

                # Try to parse; fall back to raw JSON display
                try:
                    movie: Movie = parser.parse(response.content)
                    parsed = True
                except Exception:
                    parsed = False

                if parsed:
                    # ── Movie title banner ──────────────────────────────────
                    st.markdown(f"""
                    <div class="movie-title-hero">
                        <h2>{movie.movie_name}</h2>
                        <span class="year">{movie.release_year}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Genre tags ──────────────────────────────────────────
                    genre_tags = "".join(f'<span class="tag">{g}</span>' for g in movie.genre)
                    st.markdown(f'<div style="text-align:center;margin-bottom:1.6rem">{genre_tags}</div>',
                                unsafe_allow_html=True)

                    # ── Quick summary ───────────────────────────────────────
                    st.markdown(f"""
                    <div class="info-card">
                        <h4>Quick Summary</h4>
                        <p>{movie.quick_summary}</p>
                    </div>""", unsafe_allow_html=True)

                    # ── Two-column grid ─────────────────────────────────────
                    col1, col2 = st.columns(2, gap="medium")

                    with col1:
                        # Crew
                        crew_lines = ""
                        if movie.director:
                            crew_lines += f"<p><b>Director:</b> {movie.director}</p>"
                        if movie.producer:
                            crew_lines += f"<p><b>Producer:</b> {movie.producer}</p>"
                        if movie.production_company:
                            crew_lines += f"<p><b>Studio:</b> {movie.production_company}</p>"
                        if crew_lines:
                            st.markdown(f'<div class="info-card"><h4>Crew</h4>{crew_lines}</div>',
                                        unsafe_allow_html=True)

                        # Cast
                        cast_items = "".join(f"<li>{c}</li>" for c in movie.main_cast)
                        st.markdown(f'<div class="info-card"><h4>Main Cast</h4><ul>{cast_items}</ul></div>',
                                    unsafe_allow_html=True)

                        # Characters
                        char_items = "".join(f"<li>{c}</li>" for c in movie.main_characters)
                        st.markdown(f'<div class="info-card"><h4>Main Characters</h4><ul>{char_items}</ul></div>',
                                    unsafe_allow_html=True)

                        if movie.villain_antagonist:
                            st.markdown(f"""
                            <div class="info-card">
                                <h4>Villain / Antagonist</h4>
                                <p>{movie.villain_antagonist}</p>
                            </div>""", unsafe_allow_html=True)

                    with col2:
                        # Plot
                        st.markdown(f"""
                        <div class="info-card">
                            <h4>Plot Overview</h4>
                            <p>{movie.plot_overview}</p>
                        </div>""", unsafe_allow_html=True)

                        # Setting
                        if movie.setting:
                            st.markdown(f"""
                            <div class="info-card">
                                <h4>Setting</h4>
                                <p>{movie.setting}</p>
                            </div>""", unsafe_allow_html=True)

                        # Themes
                        theme_tags = "".join(f'<span class="tag">{t}</span>' for t in movie.themes_tone)
                        st.markdown(f'<div class="info-card"><h4>Themes & Tone</h4>{theme_tags}</div>',
                                    unsafe_allow_html=True)

                        # Ratings
                        if movie.ratings_reviews:
                            st.markdown(f"""
                            <div class="info-card">
                                <h4>Ratings & Reviews</h4>
                                <p>{movie.ratings_reviews}</p>
                            </div>""", unsafe_allow_html=True)

                    # ── Full-width bottom section ───────────────────────────
                    if movie.awards_achievements:
                        award_items = "".join(f"<li>{a}</li>" for a in movie.awards_achievements)
                        st.markdown(f'<div class="info-card"><h4>Awards & Achievements</h4><ul>{award_items}</ul></div>',
                                    unsafe_allow_html=True)

                    if movie.special_highlights:
                        hl_items = "".join(f"<li>{h}</li>" for h in movie.special_highlights)
                        st.markdown(f'<div class="info-card"><h4>Special Highlights</h4><ul>{hl_items}</ul></div>',
                                    unsafe_allow_html=True)

                else:
                    # Raw JSON fallback
                    st.subheader("Extracted Data (raw)")
                    st.code(response.content, language="json")

            except Exception as e:
                st.error(f"Error: {e}")
