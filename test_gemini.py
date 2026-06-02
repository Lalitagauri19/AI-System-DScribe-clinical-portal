import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment!")
        return

    print("Initializing Gemini Client...")
    client = genai.Client(api_key=api_key)

    # Let's test a simple call first to make sure the key works
    print("Testing simple API connectivity...")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents='Respond with: API Connection Successful!'
        )
        print("Response:", response.text.strip())
    except Exception as e:
        print("Simple API call failed:", e)
        return

    # Now let's try uploading the patient PDF to see if Gemini can process it
    pdf_path = "patient 2 (1).pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found!")
        return

    print(f"Uploading {pdf_path} to Gemini Files API...")
    try:
        # Uploading the large file is ideal for Gemini since it can native-OCR the entire document
        file_ref = client.files.upload(file=pdf_path)
        print(f"Uploaded successfully. File name: {file_ref.name}")
        
        # Wait for file to be processed if needed (PDFs are usually active immediately)
        while file_ref.state.name == "PROCESSING":
            print("File is processing, waiting...")
            time.sleep(2)
            file_ref = client.files.get(name=file_ref.name)
            
        if file_ref.state.name == "FAILED":
            print("File processing failed!")
            return

        print("File is active. Asking Gemini to extract basic patient demographics...")
        prompt = """
        Analyze the attached medical record PDF and extract the following basic details:
        1. Patient Name
        2. Date of Birth / Age
        3. Admission Date
        4. Discharge Date
        5. A brief list of what types of documents are contained in this 71-page record (e.g. Admission Note, Progress Notes, Lab Reports, etc.)
        
        Return this in a clean, professional formatting. If a field cannot be found, say 'Not found' or 'Missing' - do not make it up!
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[file_ref, prompt]
        )
        print("\n--- Extraction Result ---")
        print(response.text)
        print("-------------------------")

        # Cleanup the file from Gemini
        print("Cleaning up file from Gemini cloud...")
        client.files.delete(name=file_ref.name)
        print("Cleanup done!")

    except Exception as e:
        print("File upload or extraction failed:", e)

if __name__ == "__main__":
    main()
