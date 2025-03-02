# Cocktail Assistant 🍸

An AI-powered cocktail recommendation system that helps users discover and learn about cocktails.

## Features
- Search cocktails by ingredients
- Get detailed cocktail recipes
- Receive personalized recommendations
- Non-alcoholic options available
- Interactive chat interface

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AndriiMikita/Cocktail-Adviser.git
   cd Cocktail-Adviser
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your HuggingFace API key:
   ```bash
   HUGGINGFACE_API_KEY=<your_api_key>
   ```

## Running the Application

1. Run the server:
   ```bash
   uvicorn src.api:app --reload
   ```

2. Open the frontend:
   ```bash
   cd frontend
   python -m http.server 8080
   ```

3. Open your browser and navigate to:
   ```bash
   http://127.0.0.1:8080/
   ```
## Project Structure

Cocktail-Adviser/
├── src/
│ └── api.py # FastAPI backend
├── frontend/
│ └── index.html # Web interface
├── dataset/
│ └── final_cocktails.csv # Cocktail database
├── storage/ # Vector store storage
├── requirements.txt # Dependencies
└── README.md # Documentation

## API Endpoints
- POST `/chat`: Main chat endpoint for cocktail queries

## Testing
Use the built-in test buttons in the UI to try:
- Ingredient searches
- Recipe lookups
- Cocktail recommendations
- Similar drink suggestions

## Technologies Used
- FastAPI
- LlamaIndex
- HuggingFace Models
- FAISS Vector Store
- Sentence Transformers