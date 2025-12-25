import pandas as pd
import numpy as np
from IPython.core.release import author, author_email

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import gradio as gr

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
books=pd.read_csv('books_with_emotion.csv')

books["large_thumbnail"]=books["thumbnail"] + "&fife=w800"
books["large_thumbnail"]=np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)


raw_documents=TextLoader("tagged_description.txt",encoding="utf-8").load()
text_splitter=CharacterTextSplitter(chunk_size=1,chunk_overlap=0,separator="\n")
document=text_splitter.split_documents(raw_documents)

db_books=Chroma.from_documents(document,embeddings)

print(db_books)

def retrieve_semantic_recommendation(
        query: str,
        category:str = None,
        tone:str = None,
        initial_top_k:int =50,
        final_top_k:int =16,
) -> pd.DataFrame:
    recs=db_books.similarity_search_with_score(query,k=initial_top_k)
    # print(recs)
    books_list = [int(doc.page_content.strip('"').strip("'").split()[0]) for doc,score in recs ]
    # print(books_list)
    books_recs=books[books["isbn13"].isin(books_list)]

    if category != "All":
        books_recs=books_recs[books_recs["simple_categories"]==category].head(final_top_k)
    else:
        books_recs=books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)


    return books_recs

motion_css =  """
.gr-gallery {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto;
    scroll-behavior: smooth;
    gap: 16px;
    padding-bottom: 10px;
}

.gr-gallery::-webkit-scrollbar {
    height: 8px;
}

.gr-gallery::-webkit-scrollbar-thumb {
    background: rgba(150,150,150,0.5);
    border-radius: 4px;
}

.gr-gallery .card {
    min-width: 220px;
    transition: transform 0.3s ease;
}

.gr-gallery .card:hover {
    transform: scale(1.05);
}
"""

def reccomender(
        query:str,
        category: str,
        tone: str
):

    recommendations= retrieve_semantic_recommendation(query,category,tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."



        author_split = row["authors"].split(";")
        if len(author_split) == 2:
            authors_str = f"{author_split[0]} and {author_split[1]}"
        elif len(author_split) > 2:
            authors_str = f"{' , '.join(author_split[:-1])} and {author_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title_and_subtitle']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"],caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy" , "Surprising" , "Angry" , "Suspenseful" , "Sad" ]

with gr.Blocks(theme= gr.themes.Glass(), css=motion_css) as dashboard:
    gr.Markdown("# Semantic Books Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "eg. A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories,label = "Select Category : ", value="All")
        tone_dropdown = gr.Dropdown(choices=tones,label = "Select an emotional tone : ", value="All")
        submit_button = gr.Button("Find Recommendation")

    gr.Markdown("## Reccomendations")
    output = gr.Gallery(label = "Recommended books", columns=8 , rows = 2)


    submit_button.click(fn=reccomender,
                        inputs=[user_query,category_dropdown,tone_dropdown],
                        outputs=output
                        )
    gr.Markdown("## 📖 Book Collection (Motion UI)")
    motion_gallery = gr.Gallery(
        value=[
            (row["large_thumbnail"], row["title_and_subtitle"])
            for _, row in books.iterrows()
        ],
        columns=5,
        rows =1
    )



# with gr.Blocks(css=motion_css) as motion_ui:
#     gr.Markdown("Book Collection")
#     gallery = gr.Gallery(
#         value=books[["thumbnail","title_and_subtitle"]],
#         columns=3,
#     )


if __name__ == "__main__":
    dashboard.launch()
    # motion_ui.launch()