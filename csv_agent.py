from fastapi import FastAPI
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
import os
import uvicorn  

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


class QueryAgent:
    def __init__(self, csv_url):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.llm_2=ChatAnthropic(model_name="claude-3-5-sonnet-20240620",api_key=anthropic_api_key)
        self.df = pd.read_csv(csv_url)
        self.agent = create_pandas_dataframe_agent(self.llm_2, self.df, verbose=True, allow_dangerous_code=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True,include_df_in_prompt=True)

    def query_data(self, query):
        response = self.agent.invoke(query)
        return response


class CSVCalls(BaseModel):
    csv_link:str
    query:str

app=FastAPI()
@app.post("/csv-agent")
def process_csv_calls(csv_calls: CSVCalls):
    query_agent = QueryAgent(csv_calls.csv_link)
    response = query_agent.query_data(csv_calls.query)
    return {"response": response}
