from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
import io
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


"""
when the user uploads a csv file, make sure you generatea summary of that csv 
and store the csv_summary with the csv somewhere (csv to be accesible via a link)
example csv link: https://raw.githubusercontent.com/ronidas39/LLMtutorial/main/tutorial98/output.csv

The summary needs to be generated only once per csv when the user uploads the CSV

To generate each summary of the CSV..on an average it will require a total token usage (input+output) of 2700 tokens
And a runtime of around 14secs per csv for the summary generation


once done....
when a user asks a query
pass that query through the query router script to identify which CSV will be able to answer the user's question via the summaries
The output of the query router script will be a csv name or id number which will relate to that specific CSV or 'none'

when the output is 'none'....the same old copilot.live rag process will be followed wherein context will be fetched from weaviate db
else if a csv_name or csv_id is returned then use the csv_agent with that specific csv and the user query

the query router script on an average takes 0.8s per query 
"""
class CSVDescriptors:
    def __init__(self):
        self.descriptors = {}

    def add_csv_descriptor(self, name, description):
        self.descriptors[name] = description

    def get_csv_descriptors(self):
        return self.descriptors

    

def generate_dataframe_summary(df):
    """
    Usage example

    df_example = pd.read_csv("https://raw.githubusercontent.com/ronidas39/LLMtutorial/main/tutorial98/output.csv")
    output_example = generate_dataframe_summary(df_example)
    print(output_example)

    """
    summary_parts = []

    summary_parts.append("Basic Descriptive Statistics:\n")
    summary_parts.append(df.describe(include='all').to_string())
    summary_parts.append("\n\nInformation Summary:\n")
    
    buffer = io.StringIO()
    df.info(verbose=True, buf=buffer)
    info = buffer.getvalue()
    
    summary_parts.append(info)
    
    summary_parts.append("\nUnique Values and Counts:\n")
    summary_parts.append(df.nunique().to_string())
    
    summary_parts.append("\n\nMissing Values:\n")
    summary_parts.append(df.isnull().sum().to_string())
    
    summary_string = ''.join(summary_parts)
    
    dataframe_summary_prompt=ChatPromptTemplate.from_template("""
    you are an expert at generating summaries of a pandas dataframe. 
    By summaries i mean a semantically acurate summaries that will be used to route user towards that dataframe amongst such other dataframes and many other data sources like website data, excel and pdfs.
    I will give you some table statistics and first five rows of the table along with column names.
    dataframe.head() : \n\n {df_head} \n\n 
    dataframe statistics summary: \n\n {df_summary} \n\n .
    dataframe.tail() : \n\n {df_tail} \n\n 
    include no other text other than a crisp summary. Keep in mind that the summary will be used for semantic matching plus query routing by an llm like you and also include some rows and its values with all column names and what the table is about.
    NOTE: Make note this Summary will be used by an other LLM like you to decide whether an incoming user query must be routed towards this csv or not.
    Make sure you capture all semantic intents of the csv file from the data given.
    But keep the descriptions small.
    """)

    summarise=(
    (
    {"df_head": itemgetter("df_head"), 
     "df_summary": itemgetter("df_summary"),
    "df_tail": itemgetter("df_tail"),}

    )
    | dataframe_summary_prompt
    | ChatOpenAI(temperature=0,model="gpt-4")
    # | StrOutputParser()
    )
    
    return summarise.invoke({"df_head": df.head().to_html(), "df_summary":summary_string, "df_tail": df.tail().to_html()})

def create_routing_chain(csv_descriptors):
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    system = """You are the best query routing expert that can route user queries between multiple CSVs or identify if the question can be answered by the given CSVs.
    You will be given the CSV descriptions and the user question.
    Based on which you determine which CSVs to use if the best match is found.
    If you can't determine the best match, just choose the 'none' option.
    See the user can enter vague queries but we you and me need to handle those queries.
    Use your depth of prior knowledge to be the best query router in the world!!!
    Descriptions:\n\n
    """
    descriptions = ""
    for name, desc in csv_descriptors.items():
        descriptions += "\nHere is the CSV Name: " + str(name) + """\nIt's Description: """ + str(desc) + "\n\n"

    system = system + descriptions + "\n\nAnswer should be from only these options:\t" + str(list(csv_descriptors.keys())) + "\nDo not add any extra quotes around the answer. give plain simple example: csv1 or csv2 or ....csvn or none"

    routing_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}")
        ]
    )

    routing_chain = routing_prompt_template | llm | StrOutputParser()

    return routing_chain

def process_question(question, routing_chain_structure):
    output = routing_chain_structure.invoke({"question": question})
    return output

import time

def test_average_query_time(routing_chain, questions):
    query_times = []
    for question in questions:
        start_time = time.time()
        output = process_question(question, routing_chain)
        print(output)
        end_time = time.time()
        query_time = end_time - start_time
        query_times.append(query_time)
    
    average_time = sum(query_times) / len(query_times)
    return average_time


def calculate_avg_summary_time(csv_links):
    summary_times = []
    for link in csv_links:
        start_time = time.time()
        df = pd.read_csv(link)
        print(generate_dataframe_summary(df).usage_metadata) # this calculates the token usage
        end_time = time.time()
        summary_time = end_time - start_time
        summary_times.append(summary_time)
    
    average_time = sum(summary_times) / len(summary_times)
    return average_time



if __name__ == "__main__":
    # Initialize CSVDescriptors
    csv_descriptors = CSVDescriptors()
    # when user uploads a csv....store its summaries at that time
    # fetch those summaries stored somewhere and use it for query routing
    # if the output from LLM is none therefore no csv can supposedly answer user's question therefore now do Normal RAG


    # df_example = pd.read_csv("https://raw.githubusercontent.com/ronidas39/LLMtutorial/main/tutorial98/output.csv")
    # output_example = generate_dataframe_summary(df_example)
    # csv1_link="https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Wastebase/wastebase_scan_summary_202109F.csv"


    # # Add CSV descriptors
    # csv_descriptors.add_csv_descriptor("csv1", generate_dataframe_summary(pd.read_csv("https://raw.githubusercontent.com/ronidas39/LLMtutorial/main/tutorial98/output.csv")))
    # csv_descriptors.add_csv_descriptor("csv2", generate_dataframe_summary(pd.read_csv("https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Wastebase/wastebase_scan_summary_202109F.csv")))

    # # Create routing chain
    # routing_chain = create_routing_chain(csv_descriptors.get_csv_descriptors())

    # # Process a question
    # output = process_question("What is the history of the Olympic Games??", routing_chain)

    # print(output)
    # related_questions_summary1 = [
    # "How is the batting average calculated for each player in the dataset?",
    # "Which player has the highest batting average in the dataset?",
    # "How many players in the dataset have scored a century?",
    # "What is the distribution of the number of matches played by players in the dataset?",
    # "Can you compare the career durations of players with the highest runs scored?",
    # "How many players have scored more than 10,000 runs in their career?",
    # "What is the strike rate of players who have scored over 50 half-centuries?",
    # "How many players have never been dismissed for a duck (zero runs)?",
    # "What is the relationship between the number of balls faced and the batting average?",
    # "Which player has hit the most sixes in their career?",
    # "How does the number of matches played correlate with the total runs scored?",
    # "Can we identify any trends in player performance based on their career duration?",
    # "How many players have played over 100 matches in their career?",
    # "What is the distribution of players' highest scores in the dataset?",
    # "Are there any players with a strike rate above 150?",
    # "Which player has been not out the most times?",
    # "What is the average number of centuries scored by players in the dataset?",
    # "How many players have hit over 200 fours in their career?",
    # "Can you analyze the performance of players who have a career duration of over 20 years?",
    # "Which player has the lowest batting average but has played a significant number of matches?"]


    # average_query_time = test_average_query_time(routing_chain, related_questions_summary1)
    # print(f"Average time taken per query: {average_query_time} seconds")


    dataset_urls = [
    "https://raw.githubusercontent.com/deepanshu88/data/master/sampledata.csv",
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/healthexp.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/data.csv",
    "https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/blood_pressure.csv",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    ]

    average_summary_time = calculate_avg_summary_time(dataset_urls)
    print(f"Average time taken to generate a summary: {average_summary_time} seconds")