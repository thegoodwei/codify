import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import openai
import re

# Load the NLTK stop words
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

# Define the directory path where the srt files are located
directory_path = '/path/to/srt/files'

def load_srt_transcripts(directory_path):
    """
    Loads srt transcripts from the specified directory and returns them as a list of strings.
    """
    transcripts = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.srt'):
            with open(os.path.join(directory_path, filename), 'r') as file:
                transcript = file.read().replace('\n', ' ')
                transcripts.append(transcript)
    return transcripts

def preprocess(transcripts):
    """
    Preprocesses the transcripts by removing punctuation, converting to lowercase, and removing stop words.
    """
    preprocessed_transcripts = []
    for transcript in transcripts:
        # Remove punctuation and convert to lowercase
        transcript = transcript.translate(str.maketrans('', '', string.punctuation)).lower()
        # Tokenize the transcript
        tokens = word_tokenize(transcript)
        # Remove stop words
        tokens = [token for token in tokens if token not in stopwords]
        # Convert the tokens back to a string and append to the list of preprocessed transcripts
        preprocessed_transcripts.append(' '.join(tokens))
    return preprocessed_transcripts


def categorize(transcript):
    # Set up OpenAI API key
    openai.api_key = "YOUR_API_KEY"
    
    # Define the prompt for GPT-3 to classify the transcript
    prompt = (
        "Please categorize the following transcript based on the following categories:\n"
        "- Category 1: [describe category 1]\n"
        "- Category 2: [describe category 2]\n"
        "- Category 3: [describe category 3]\n"
        "Transcript:\n"
        f"{transcript}\n"
    )
    
    # Use GPT-3 DaVinci to generate the categories
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.5,
        max_tokens=1024,
        n = 1,
        stop=None,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    # Extract the categories from the GPT-3 response
    categories = re.findall(r"- (Category \d+): (.+)", response.choices[0].text)
    
    # Create a dictionary of categories and their descriptions
    categories_dict = dict(categories)
    
    # Return the dictionary of categories
    return categories_dict

def code(preprocessed_transcripts, categories):
    """
    Codes the preprocessed transcripts by assigning each token to the appropriate category and returns a list of dictionaries, one for each transcript, mapping categories to lists of tokens.
    """
    coded_transcripts = []
    for transcript in preprocessed_transcripts:
        transcript_categories = {category: [] for category in categories}
        for token in word_tokenize(transcript):
            for category in categories:
                if any(keyword in token for keyword in categories[category]):
                    transcript_categories[category].append(token)
        coded_transcripts.append(transcript_categories)
    return coded_transcripts

def codify(transcript, categories):
    # Set up OpenAI API key
    openai.api_key = "YOUR_API_KEY"
    
    # Tokenize the transcript
    tokens = transcript.lower().split()
    
    # Initialize an empty dictionary to store the coded transcript
    coded_transcript = {}
    
    # Loop through the tokens and assign each to the appropriate category using GPT-3 DaVinci
    for token in tokens:
        # Define the prompt for GPT-3 to assign the token to a category
        prompt = f"Please assign the token '{token}' to the appropriate category:\n{categories}"
        
        # Use GPT-3 DaVinci to generate the category for the token
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.5,
            max_tokens=1024,
            n = 1,
            stop=None,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Extract the category from the GPT-3 response
        category = re.findall(r"\d+", response.choices[0].text)[0]
        
        # Add the token to the appropriate category in the coded_transcript dictionary
        if category not in coded_transcript:
            coded_transcript[category] = [token]
        else:
            coded_transcript[category].append(token)
    
    # Return the coded transcript dictionary
    return coded_transcript

def validate(coded_transcripts, categories):
    """
    Validates the coding of the transcripts using OpenAI's GPT-3 DaVinci model and returns a list of dictionaries, one for each transcript, mapping categories to lists of validated tokens.
    """
    # TODO: Implement OpenAI validation using GPT-3 DaVinci
    return coded_transcripts


def analysis(transcripts):
    # Convert transcripts to a pandas DataFrame
    df = pd.DataFrame(transcripts, columns=["word", "category"])
    
    # Create a regression model with category as the predictor variable and word count as the response variable
    model = smf.ols("word_count ~ C(category)", data=df).fit()
    
    # Print the model summary
    print(model.summary())

def visualization(transcripts):
    # Convert transcripts to a pandas DataFrame
    df = pd.DataFrame(transcripts, columns=["word", "category"])
    
    # Group by category and count the number of words in each category
    category_counts = df.groupby("category")["word"].count()
    
    # Create a bar plot of the category counts
    plt.bar(category_counts.index, category_counts.values)
    plt.title("Transcript Category Counts")
    plt.xlabel("Category")
    plt.ylabel("Word Count")
    plt.show()

if __name__ == "__main__":
    # Load the transcripts
    transcripts = load_srt_transcripts("example_transcripts/")
    
    # Preprocess the transcripts
    transcripts = preprocessing(transcripts)
    
    # Tokenize the transcripts
    transcripts = tokenization(transcripts)
    
    # Categorize the transcripts
    transcripts = categorization(transcripts)
    
    # Code the transcripts
    coded_transcripts = coding(transcripts)
    
    # Validate the coding
    validated_transcripts = validation(coded_transcripts)
    
    # Analyze the transcripts
    analysis(validated_transcripts)
    
    # Visualize the transcripts
    visualization(validated_transcripts)
