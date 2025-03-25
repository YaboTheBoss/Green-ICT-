import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy

# Load the spaCy English model for sentence extraction
nlp = spacy.load("en_core_web_sm")

# Load the pre-trained sdgBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sadickam/sdg-classification-bert")
model = AutoModelForSequenceClassification.from_pretrained("sadickam/sdg-classification-bert")

# SDG Mapping for the first 16 SDGs (just the numbers, as required)
sdg_mapping = {
    0: "No Poverty",
    1: "Zero Hunger",
    2: "Good Health and Well-being",
    3: "Quality Education",
    4: "Gender Equality",
    5: "Clean Water and Sanitation",
    6: "Affordable and Clean Energy",
    7: "Decent Work and Economic Growth",
    8: "Industry, Innovation and Infrastructure",
    9: "Reduced Inequality",
    10: "Sustainable Cities and Communities",
    11: "Responsible Consumption and Production",
    12: "Climate Action",
    13: "Life Below Water",
    14: "Life on Land",
    15: "Peace and Justice Strong Institutions"
}

def extract_topic_sentences(text, max_sentences=5):
    """
    Extracts the first 5 sentences for the topic sentences.
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    
    # Limit to the first 5 sentences
    topic_sentences = sentences[:min(len(sentences), max_sentences)]
    
    return " ".join([sent.text.strip() for sent in topic_sentences])

def extract_balanced_sentences_for_bert(text, num_sentences=8):
    """
    Extracts a balanced set of sentences from the beginning, middle, and end of the article
    to be fed to the Bert model (should be separate from topic sentences).
    """
    doc = nlp(text)
    sentences = list(doc.sents)
    
    # Determine how many sentences to extract from each part (beginning, middle, end)
    num_per_section = num_sentences // 3  # Evenly split between sections (start, middle, end)
    
    # Get sentences from beginning, middle, and end of the text
    start_sentences = sentences[:num_per_section]
    middle_sentences = sentences[len(sentences)//3:2*(len(sentences)//3)]
    end_sentences = sentences[-num_per_section:]
    
    # Combine all the selected sentences
    selected_sentences = start_sentences + middle_sentences + end_sentences
    
    # Return as a string of sentences
    return " ".join([sent.text.strip() for sent in selected_sentences[:num_sentences]])

def classify_top_sdgs(text, top_n=3):
    """
    Classifies the press release text into the top N SDGs based on the highest probabilities.
    Now stores only the SDG numbers instead of names.
    """
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input text and move to device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the raw logits output and apply sigmoid to convert them to probabilities
    logits = outputs.logits
    probabilities = torch.sigmoid(logits)  # Apply sigmoid for multi-label classification

    # Sort the probabilities in descending order and get the top N SDGs
    top_probabilities, top_indices = torch.topk(probabilities, top_n, dim=-1)

    # Instead of names, map to SDG numbers
    top_sdgs = [str(idx.item() + 1) for idx in top_indices.squeeze()]

    return top_sdgs

def create_apple_excel(root_folder):
    """
    Creates an Excel workbook for Apple with years as sheets and files as rows
    under their respective months, and adds a summary (sentences), SDG classification, and Collaboration column for each file.
    """
    # Path to Apple folder
    apple_path = os.path.join(root_folder, "Apple")
    if not os.path.exists(apple_path):
        raise FileNotFoundError("Apple folder not found in the specified path")
    
    # Dictionary to store data for each year
    year_data = {}
    
    # Process each year from 2015 to 2025
    for year in range(2015, 2026):
        year_path = os.path.join(apple_path, str(year))
        if not os.path.exists(year_path):
            continue
        
        rows = []
        
        # Process each month (1-12)
        for month_num in range(1, 13):
            month_path = os.path.join(year_path, str(month_num))
            if not os.path.exists(month_path):
                continue
                
            # Get all files in the month folder
            files = [f for f in os.listdir(month_path) if os.path.isfile(os.path.join(month_path, f))]
            
            # Get month name
            month_name = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }[month_num]
            
            # Process each file: read the file, extract the first sentences, and classify SDG
            for file in files:
                file_path = os.path.join(month_path, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    # Extract topic sentences (first 5 sentences)
                    topic_sentences = extract_topic_sentences(text, max_sentences=5)
                    
                    # Extract sentences for BERT classification (ensure 8 sentences minimum)
                    sentences_for_model = extract_balanced_sentences_for_bert(text, num_sentences=8)
                    
                    # Get SDGs classification based on the extracted sentences for the model
                    sdgs = classify_top_sdgs(sentences_for_model, top_n=3)
                except Exception as e:
                    topic_sentences = f"Error reading file: {e}"
                    sdgs = ["Unknown"]
                
                rows.append({
                    'Month': month_name,
                    'File Name': file,
                    'Extracted Sentences': topic_sentences,  # Store topic sentences here
                    'SDGs': ", ".join(sdgs),  # Store top 3 SDG numbers as a comma-separated list
                    'Collaboration': ""  # Empty column for manual input
                })
        
        if rows:
            df = pd.DataFrame(rows)
            year_data[str(year)] = df
    
    # Define output Excel file name (with summaries, SDG classifications, and Collaboration column)
    output_file = os.path.join(root_folder, 'Apple_Press_Releases_with_Sentences_Top_SDGs_and_Collaboration.xlsx')
    
    # Write each year's DataFrame to a separate sheet
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for year, df in year_data.items():
            # Use the year as the sheet name
            df.to_excel(writer, sheet_name=year, index=False)
    
    return output_file


def main():
    try:
        # Replace this with your actual root folder path
        root_folder = r"C:\Users\andre\OneDrive\Desktop\Green ICT Press Release"
        output_file = create_apple_excel(root_folder)
        print(f"\nExcel file successfully created at: {output_file}")
        print("\nStructure:")
        print("- Each sheet represents a year (2015-2025)")
        print("- Files are listed as rows under their respective months")
        print("- A 'Extracted Sentences' column contains the selected sentences for each file")
        print("- A 'SDGs' column contains the SDG classifications for each press release")
        print("- A 'Collaboration' column is added for manual input")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
