import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import pandas as pd  # Import Pandas for DataFrame operations


file_path = Path(r"parking_data.csv") #input your own dataset


if not os.getenv("GOOGLE_API_KEY"):
    os.environ['GOOGLE_API_KEY'] = "YOUR_GOOGLE_API_KEY"
load_dotenv()


# Configure Streamlit page settings
st.set_page_config(
    page_title="Parking Data Insights",
    page_icon=":brain:",  # Favicon emoji # Page layout option
    layout = "wide"
)

# Retrieve Google API key from environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

columns = st.columns(2, gap="large")
with columns[0]:
    st.title("Average Parking Occupancy of Different Parking Slots")

    index = [ str(i) for i in range(24)]
    
    parking_data = pd.read_csv(file_path)  
        
    parking_data['Unoccupied'] = parking_data['Total_slots'] - parking_data['Occupied']

    parking_data['Time'] =pd.to_datetime(parking_data['Time'])
    parking_data['Date'] =pd.to_datetime(parking_data['Date'])

    parking_insights = parking_data.groupby('Parking_lot').agg({'Occupied': 'mean', 'Unoccupied': 'mean'})

    plt.figure(figsize = (5,4))
    st.bar_chart(parking_insights)
    plt.title('Parking Insights')
    plt.xlabel('Parking Lot')
    plt.ylabel('Count')
    # plt.show()
    st.title("Parking Occupancy Over Time")
    parking_data['Unoccupied'] = parking_data['Total_slots'] - parking_data['Occupied']

    parking_data['Time'] =pd.to_datetime(parking_data['Time'])
    parking_data['Date'] =pd.to_datetime(parking_data['Date'])

    grouped_data = parking_data.groupby([parking_data['Time'].dt.floor('H'), 'Parking_lot'])['Occupied'].sum()

    occupied_by_lot_and_time = grouped_data.unstack(fill_value=0)
    occupied_by_lot_and_time.columns = list(parking_data['Parking_lot'].unique())

    plt.figure(figsize = (10,6))
    st.line_chart(occupied_by_lot_and_time) 
    plt.title('Parking Occupancy')
    plt.xlabel('Time')
    plt.ylabel('Parking Occupancy Count')
    plt.show()
    
with columns[1]:
    expander = st.expander("Chatbot", expanded=True)
    # Set up Google Gemini-Pro AI model
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    model = gen_ai.GenerativeModel('gemini-pro')

    # Function to translate roles between Gemini-Pro and Streamlit terminology
    def translate_role_for_streamlit(user_role):
        if user_role == "model":
            return "assistant"
        else:
            return user_role

    # Initialize chat session in Streamlit if not already present
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Display the chatbot's title on the page
    with expander:
        st.title("🤖 LLM-powered analysis")
        prompt = f"Here is a CSV file containing data about occupancy of vehicles by times in different parking lots:  {parking_data}.As an expert data analyst, Please analyze it and provide insights regarding the following questions:"
        
        # Display the chat history
        for message in st.session_state.chat_session.history:
            with st.chat_message(translate_role_for_streamlit(message.role)):
                st.markdown(message.parts[0].text.lstrip(prompt))
        # Input field for user's message
        print(prompt)
        user_prompt = st.chat_input("Ask the chatbot questions about the data")
        if user_prompt:
            # Add user's message to chat and display it
            st.chat_message("user").markdown(user_prompt)

            # Send user's message to Gemini-Pro and get the response
            gemini_response = st.session_state.chat_session.send_message(prompt+user_prompt)

            # Display Gemini-Pro's response
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)




