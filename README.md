# 🤖 Chat with Your Database - Streamlit & Langgraph Integration

This project enables you to interact with a database via a conversational interface, using Streamlit for the frontend and Langgraph for managing dialogues and queries. 
It showcases how to use a data assistant to query databases with natural language.



## 📋 Setup

1. Create your environment (e.g. via conda)
   `conda create -n test_env python=3.12`
2. Activate your environment
   `conda activate ./.venv`
3. Install requirements
4. Create your secrets file ./.streamlit/secrets.toml and insert your model and database credentials (LLM_BASE_URL, AI_VERSION, API_SERVICE_KEY and NORTHWIND_DB)
5. Update the code to your model and database (I use Microsoft Azure and a local northwind database)
   `llm = AzureChatOpenAI(base_url=st.secrets["LLM_BASE_URL"],
            openai_api_version=st.secrets["AI_VERSION"],
            api_key = st.secrets["API_SERVICE_KEY"],
            temperature = 0.0,
            #model_kwargs={"stop": ["\nObservation", "Observation"]} # this should further prevent halucinations
        )
        db = SQLDatabase.from_uri(st.secrets["NORTHWIND_DB"])`
6. Run your data assistant
   `streamlit run app.py`


