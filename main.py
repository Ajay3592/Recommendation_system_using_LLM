import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

#API keys 
openai_key = st.secrets["openai"]["api_key"]
pinecone_key = st.secrets["pinecone"]["api_key"]


# Page setup
st.set_page_config(page_title="Ajay's Smart Recommender", layout="wide")

# Inject custom CSS
st.markdown(
    """
    <style>
    /* Fix background color */
    .stApp {
        background-color: #eef2f7 !important;
    }
    .main-container {
        padding: 30px;
        border-radius: 12px;
    }
    /* Style input box */
    .stTextInput input {
        background-color: #ffffff !important;
        border: 1px solid #999 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 16px !important;
        color: #333 !important;
    }
    /* Product table styling */
    .product-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        overflow: hidden;
        margin-top: 20px;
        font-family: Arial, sans-serif;
    }
    .product-table th, .product-table td {
        padding: 12px;
        text-align: left;
        border: 1px solid #ccc;  
        vertical-align: middle;
    }
    .product-table th {
        background-color: #dbe4f0;
        color: #333;
    }
    .product-table tr:hover {
        background-color: #f1f5fb;
    }
    .product-table img {
        max-width: 100px;
        max-height: 100px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-container">', unsafe_allow_html=True)


# Initialize Pinecone
pc = Pinecone(api_key=pinecone_key)
index_name = "product-recommendation-updated"
index = pc.Index(index_name)

# Embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_key)

# Vector store
vectorstore = PineconeVectorStore(index=index, embedding=embedding_model)

# LLM setup
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, api_key=openai_key)


st.title("[=============[ AI Product Recommender ]=============]")

# User input
user_query = st.text_input(
    "What are you looking for today?",
    placeholder="e.g. budget smartphones,laptops or kitchen items..."
)

# Run recommendation logic
if user_query:
    with st.spinner("Finding great options for you..."):
        retrieved_docs = vectorstore.similarity_search(user_query, k=5)

        # Format product list for LLM
        product_list = "\n".join([
            f"{doc.metadata.get('title', 'N/A')} - ${doc.metadata.get('discounted_price', 'N/A')}, Rating: {doc.metadata.get('rating', 'N/A')}, Image: {doc.metadata.get('product_url', '')}"
            for doc in retrieved_docs if doc.metadata
        ])

        # Prompt template
        template = PromptTemplate.from_template("""
        User query: "{query}"

        Products:
        {products}

        Generate a recommendation in a friendly tone.
        """)

        prompt = template.format(query=user_query, products=product_list)
        response = llm.predict(prompt)

        # Display recommendation
        st.markdown("### Recommendation")
        st.write(response)

        # Display product results
        st.markdown("### Products")
        table_html = """
        <table class="product-table">
            <tr>
                <th>Image</th>
                <th>Title</th>
                <th>OriginalPrice</th>
                <th>DiscountedPrice</th>
                <th>Rating</th>
            </tr>
        """

        for doc in retrieved_docs:
            title = doc.metadata.get('title', 'N/A').split(',')[0]
            original_price = f"${doc.metadata.get('original_price', 'N/A')}"
            discounted_price = f"${doc.metadata.get('discounted_price', 'N/A')}"
            rating_val = float(doc.metadata.get('rating', 0.0))
            image_url = doc.metadata.get('image_url', '')

            # ⭐ Convert numeric rating into stars
            full_stars = int(rating_val)
            half_star = 1 if rating_val - full_stars >= 0.5 else 0
            empty_stars = 5 - full_stars - half_star

            stars_html = (
                '<span style="color:gold;">' + '★' * full_stars + '</span>' +
                ('<span style="color:gold;">☆</span>' if half_star else '') +
                '<span style="color:#ccc;">' + '☆' * empty_stars + '</span>'
            )

            table_html += f"""
                <tr>
                    <td><img src="{image_url}" alt="Product Image"></td>
                    <td>{title}</td>
                    <td>{original_price}</td>
                    <td>{discounted_price}</td>
                    <td>{stars_html} <span style="color:#555;">({rating_val})</span></td>
                </tr>
            """

        table_html += "</table>"
        st.components.v1.html(table_html, height=400, scrolling=True)

# Close container
st.markdown('</div>', unsafe_allow_html=True)
