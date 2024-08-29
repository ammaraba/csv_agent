from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
import io
import pandas as pd
from dotenv import load_dotenv
load_dotenv()



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
    | StrOutputParser()
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


# if __name__ == "__main__":
#     # Initialize CSVDescriptors
#     csv_descriptors = CSVDescriptors()
#     # when user uploads a csv....store its summaries at that time
#     # fetch those summaries stored somewhere and use it for query routing
#     # if the output from LLM is none therefore no csv can supposedly answer user's question therefore now do Normal RAG


#     # df_example = pd.read_csv("https://raw.githubusercontent.com/ronidas39/LLMtutorial/main/tutorial98/output.csv")
#     # output_example = generate_dataframe_summary(df_example)
#     # csv1_link="https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Wastebase/wastebase_scan_summary_202109F.csv"


#     # Add CSV descriptors
#     csv_descriptors.add_csv_descriptor("csv1", generate_dataframe_summary(pd.read_csv("https://raw.githubusercontent.com/ronidas39/LLMtutorial/main/tutorial98/output.csv")))
#     csv_descriptors.add_csv_descriptor("csv2", generate_dataframe_summary(pd.read_csv("https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Wastebase/wastebase_scan_summary_202109F.csv")))

#     # Create routing chain
#     routing_chain = create_routing_chain(csv_descriptors.get_csv_descriptors())

#     # Process a question
#     output = process_question("which country supports manufactufing more??", routing_chain)

#     print(output)
