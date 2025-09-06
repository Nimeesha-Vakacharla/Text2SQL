import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
import base64
import requests
import json

class OllamaText2SQLAgent:
    def __init__(self, db_path, model_name="llama3.1:8b"):
        self.db_path = db_path
        self.model_name = model_name
        self.query_history = []
        self.initialize_agent()
        
    def initialize_agent(self):
        """Initialize the Text-to-SQL agent"""
        try:
            # Load database schema - this part doesn't require the model
            self.schema = self.load_schema()
            
            # Check if Ollama is available
            try:
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    st.success("✅ Connected to Ollama server")
                    self.ollama_available = True
                else:
                    st.warning("⚠️ Ollama server is not responding properly. Will use fallback SQL generation.")
                    self.ollama_available = False
            except:
                st.warning("⚠️ Couldn't connect to Ollama server. Will use fallback SQL generation.")
                self.ollama_available = False
                
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            raise
    
    def load_schema(self):
        """Load database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            schema[table] = [{"name": col[1], "type": col[2]} for col in cursor.fetchall()]
        
        conn.close()
        return schema
    
    def get_schema_text(self):
        """Get formatted schema text for display"""
        schema_text = ""
        for table, cols in self.schema.items():
            col_defs = [f"{col['name']} ({col['type']})" for col in cols]
            schema_text += f"Table {table}: {', '.join(col_defs)}\n\n"
        return schema_text
    
    def generate_sql_with_ollama(self, question):
        """Generate SQL using Ollama with LLaMA 3.1"""
        if not self.ollama_available:
            return self.generate_fallback_sql(question)
            
        schema_text = self.get_schema_text()
        
        # Create a prompt for the model
        prompt = f"""### Task: Convert the following natural language question to SQL using the database schema provided.

### Database Schema:
{schema_text}

### Question:
{question}

### SQL Query:
```sql
"""
        
        try:
            # URL for Ollama API
            url = "http://localhost:11434/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,
                "num_predict": 200
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Make the request to the Ollama API
            response = requests.post(url, data=json.dumps(payload), headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                # Extract the generated SQL from the response
                generated_text = result.get('response', '')
                
                # Clean up the generated SQL
                sql = generated_text.strip()
                
                # Ensure the SQL doesn't contain the closing tag
                if '```' in sql:
                    sql = sql.split('```')[0]
                    
                # Make sure it's valid SQL
                if not (sql.upper().startswith("SELECT") or sql.upper().startswith("WITH")):
                    sql = "SELECT * FROM activity LIMIT 10"
                
                return sql
            else:
                # Fallback to a simple query if the model request fails
                st.warning(f"Ollama server error: {response.status_code}. Using fallback SQL generation.")
                return self.generate_fallback_sql(question)
                
        except Exception as e:
            st.warning(f"Error generating SQL with Ollama: {str(e)}. Using fallback SQL generation.")
            return self.generate_fallback_sql(question)
    
    def generate_fallback_sql(self, question):
        """Generate fallback SQL for when the Ollama model is unavailable"""
        question = question.lower()
        tables = list(self.schema.keys())
        main_table = None
        
        # Try to find which table is mentioned in the question
        for table in tables:
            if table.lower() in question:
                main_table = table
                break
        
        # If no table is found, use the first one
        if not main_table and tables:
            main_table = tables[0]
        
        # Simple fallback query
        return f"SELECT * FROM {main_table} LIMIT 10"
    
    def execute_query(self, sql):
        """Execute SQL query and return results as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            df = pd.read_sql_query(sql, conn)
            return df, None
        except Exception as e:
            return None, str(e)
        finally:
            conn.close()
    
    def create_visualization(self, df):
        """Create appropriate visualization based on data"""
        if len(df) == 0:
            return None
            
        if len(df.columns) == 1:
            # Single column - show bar chart
            fig = px.bar(df, y=df.columns[0], title="Query Results")
            return fig
        elif len(df.columns) == 2:
            # Two columns - show relationship
            if pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Query Results")
            else:
                # Try to create a pie chart if possible
                try:
                    fig = px.pie(df, names=df.columns[0], values=df.columns[1], title="Query Results")
                except:
                    fig = px.bar(df, x=df.columns[0], y=df.index, title="Query Results")
            return fig
        else:
            # For more columns, show scatter plot of first two numeric columns if found
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="Query Results")
                return fig
            return None

def get_download_link(df, filename="results.csv"):
    """Generate a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return href

def main():
    st.set_page_config(
        page_title="Text-to-SQL Agent",
        page_icon="🦙",
        layout="wide"
    )
    
    # Title and description
    st.title("Text-to-SQL Agent")
    st.markdown("""
    🦙 Convert your natural language questions to SQL queries using Ollama and visualize the results.
    
    This app connects to an Ollama server running LLaMA 3.1-8B to generate SQL queries.
    """)
    
    # Add Ollama setup instructions in an expander
    with st.expander("⚙️ Ollama Setup Instructions"):
        st.markdown("""
        To use this application with LLaMA 3.1-8B, you need to run Ollama. Follow these steps:
        
        1. Install Ollama from [ollama.ai](https://ollama.ai)
        
        2. Pull the LLaMA 3.1-8B model:
           ```bash
           ollama pull llama3.1:8b
           ```
           
        3. Make sure Ollama is running in the background
           
        The Ollama server should be running on localhost:11434 for this app to connect to it.
        """)
    
    # Database path
    db_path = "/Users/dilipkumarkasina/Documents/Gen AI/Project/activity_1.sqlite"
    
    # Model selection
    model_options = {
        "llama3.1:8b": "LLaMA 3.1 (8B) - Best quality",
        "llama3:8b": "LLaMA 3 (8B) - Good quality",
        "llama2:7b": "LLaMA 2 (7B) - Faster"
    }
    
    selected_model = st.selectbox(
        "Select Ollama Model",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    # Initialize agent when requested
    if 'agent' not in st.session_state or 'model_name' not in st.session_state or st.session_state.model_name != selected_model:
        st.session_state.agent_loaded = False
        st.session_state.model_name = selected_model
        
    if st.button("Initialize Agent", key="init_button"):
        with st.spinner(f"Loading database schema and connecting to Ollama with {selected_model}..."):
            try:
                st.session_state.agent = OllamaText2SQLAgent(db_path, model_name=selected_model)
                st.session_state.agent_loaded = True
                st.success("Agent initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize agent: {e}")
                st.session_state.agent_loaded = False
    
    if not st.session_state.get('agent_loaded', False):
        st.warning("Please initialize the agent first.")
        return
    
    # Create two columns layout
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # Question input
        question = st.text_area(
            "Enter your question",
            placeholder='e.g., "Show me all activities with more than 10 participants"',
            height=100
        )
        
        # Generate and execute SQL
        if st.button("Run Query", key="run_button"):
            if not question:
                st.warning("Please enter a question")
            else:
                with st.spinner(f"Generating SQL with {selected_model}..."):
                    sql = st.session_state.agent.generate_sql_with_ollama(question)
                    st.code(sql, language="sql")
                    
                with st.spinner("Executing query..."):
                    df, error = st.session_state.agent.execute_query(sql)
                    
                    if error:
                        st.error(f"Error executing SQL: {error}")
                    else:
                        # Add to history
                        st.session_state.agent.query_history.append({
                            'question': question,
                            'sql': sql,
                            'results': df.to_dict('records') if df is not None else []
                        })
                        
                        # Display results
                        st.subheader("Query Results")
                        
                        if df is not None and not df.empty:
                            # Try to create visualization
                            fig = st.session_state.agent.create_visualization(df)
                            
                            if fig is not None:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show data table
                            st.dataframe(df)
                            st.write(f"Returned {len(df)} rows")
                            
                            # Create download link
                            download_link = get_download_link(df)
                            st.markdown(
                                f'<a href="{download_link}" download="results.csv">Download Results as CSV</a>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.info("No results returned")
        
        # Display history
        if len(st.session_state.agent.query_history) > 0:
            st.subheader("Query History")
            for i, query in enumerate(st.session_state.agent.query_history):
                with st.expander(f"Query {i+1}: {query['question']}"):
                    st.code(query['sql'], language="sql")
                    if query['results']:
                        st.dataframe(pd.DataFrame(query['results']))
    
    with col2:
        # Show schema
        st.subheader("Database Schema")
        if st.session_state.agent_loaded:
            schema_text = st.session_state.agent.get_schema_text()
            st.text_area("Schema", value=schema_text, height=500, disabled=True)

if __name__ == "__main__":
    main()