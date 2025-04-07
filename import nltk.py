import nltk
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Sample corpus for chatbot to learn from
corpus = [
    "Hi there! How can I help you?",
    "Hello! I'm here to assist you.",
    "My name is ChatBot.",
    "I can help you with basic questions.",
    "Goodbye! Have a nice day.",
    "I'm doing great, thank you!",
    "I'm just a program, but I can try to help.",
    "What can I do for you?",
    "How can I assist you?",
    "Sure, I can help with that.",
    "Sorry, I didn't understand that.",
    "Can you please rephrase your question?",
    "I'm here to chat with you.",
    "I love answering questions!",
    "That's an interesting question.",
    "Can you tell me more about that?",
    "I don't know the answer to that, but I can try.",
    "I'm always here if you need help.",
    "What else would you like to know?",
    "I’m glad you’re talking to me!",
    "Let’s keep the conversation going!",
    "I'm learning from our chat.",
    "Do you want to ask me something else?",
    "I’m happy to help you anytime!",
    "Tell me more about your day!",
    "I enjoy chatting with you.",
    "I’m just a program, but I like talking to you.",
    "Can I help you with anything specific?",
    "I’m not sure, but I’ll try my best!",
    "That sounds fun! Tell me more.",
    "How can I assist you today?",
    "I’m glad you reached out!",
    "Feel free to ask me anything.",
    "I’m here to make your day better.",
    "What are you curious about?",
    "I’d love to help with that!",
    "Let’s talk about something interesting.",
    "I’m always learning new things.",
    "Do you need help with something specific?",
    "I’d love to hear more about that.",
    "That’s a great question!",
    "I’m here to answer your questions.",
    "Let’s make this conversation fun!",
    "How are you feeling today?",
    "Is there anything I can do for you?",
    "I’m always ready to chat.",
    "Do you have a favorite topic to talk about?",
    "I’d love to hear your thoughts!",
    "That’s really cool!",
    "I’m not perfect, but I’ll try my best!",
    "You’re doing great by asking questions!",
    "Can we talk about something fun?",
    "I enjoy learning from people like you.",
    "What’s on your mind?"
]

# Preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]
    return ' '.join(tokens)

# Preprocess corpus
processed_corpus = [preprocess(sentence) for sentence in corpus]

# Initialize vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_corpus)

def get_response(user_input):
    user_input = preprocess(user_input)
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    idx = similarity.argmax()
    score = similarity[0][idx]
    
    if score < 0.3:
        return random.choice([
            "I'm not sure I understand. Can you try asking in a different way?",
            "Could you rephrase that for me?",
            "Sorry, I didn't catch that. Can you explain differently?"
        ])
    else:
        return random.choice([
            corpus[idx],
            "That's an interesting question! Here's what I think:",
            "Hmm, let me see... " + corpus[idx]
        ])
# Chat loop
print("ChatBot: Hello! Ask me something. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("ChatBot: Goodbye!")
        break
    response = get_response(user_input)
    print("ChatBot:", response)
