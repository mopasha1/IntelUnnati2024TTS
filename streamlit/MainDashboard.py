import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os
import google.generativeai as gen_ai
import pandas as pd  # Import Pandas for DataFrame operations


file_path = Path(r"Project1_data.csv") #input your own parking dataset

st.set_page_config(
    page_title="Parking Data Insights",
    page_icon=":brain:",  # Favicon emoji # Page layout option
    layout = "wide"
)
load_dotenv()
# Retrieve Google API key from environment variables
if not os.getenv("GOOGLE_API_KEY"):
    os.environ['GOOGLE_API_KEY'] = "YOUR_GOOGLE_API_KEY"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

st.title("Vehicle Movement Detection and Analysis")

popover = st.popover("LLM-powered insights")

df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d', errors = 'coerce')
df['Intimes'] = pd.to_datetime(df['Intimes'])
df['Outtimes'] = pd.to_datetime(df['Outtimes'])

st.sidebar.title("Filters")
selected_timeframe = st.sidebar.selectbox("Select the Time frame", ['Daily','Hourly', 'Weekly', 'Yearly'])
selected_month = st.sidebar.selectbox("Select Month", df['Month'].unique())
start_date, end_date = df['Date'].min(), df['Date'].max()

filtered_df = df[(df['Month'] == selected_month) & (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

authorized_count = int(filtered_df['Authorized'].value_counts().get('Yes', 0))
unauthorized_count = int(filtered_df['Authorized'].value_counts().get('No', 0))
total_vehicles = authorized_count + unauthorized_count
date_counts = filtered_df['Date'].value_counts()
max_date = date_counts.idxmax()
min_date = date_counts.idxmin()

st.subheader("Key Metrics")
metrics = st.columns([2,2,2,3])
metrics[0].metric("Total Vehicles", total_vehicles)
metrics[1].metric("Authorized Vehicles", authorized_count)
metrics[2].metric("Unauthorized Vehicles", unauthorized_count)
metrics[3].metric("Date with Most Traffic", max_date.strftime("%Y-%m-%d"), int(date_counts.max()))

st.subheader("Vehicle Entries Over Time")
if selected_timeframe == 'Hourly':
    df_resampled = filtered_df.set_index('Intimes').resample('H').size().to_frame(name='count').reset_index()
    x_axis = 'Intimes'
elif selected_timeframe == 'Daily':
    df_resampled = filtered_df.set_index('Date').resample('D').size().to_frame(name='count').reset_index()
    x_axis = 'Date'
elif selected_timeframe == 'Weekly':
    df_resampled = filtered_df.set_index('Date').resample('W-MON').size().to_frame(name='count').reset_index()
    x_axis = 'Date'
elif selected_timeframe == 'Yearly':
    df_resampled = filtered_df.set_index('Date').resample('Y').size().to_frame(name='count').reset_index()
    x_axis = 'Date'

fig = px.line(df_resampled, x=x_axis, y='count', title=f'Vehicle Entries Over Time ({selected_timeframe})')
st.plotly_chart(fig)

st.subheader("Entry & Exit Traffic")
fig = make_subplots(rows=1, cols=2, subplot_titles=("Entry Hour (6am - 6pm)", "Exit Hour (6am - 6pm)"))

entry_hour = filtered_df['Intimes'].dt.hour
exit_hour = filtered_df['Outtimes'].dt.hour

fig.add_trace(go.Histogram(x=entry_hour, nbinsx=12, name='Entry Hour'), row=1, col=1)
fig.add_trace(go.Histogram(x=exit_hour, nbinsx=12, name='Exit Hour'), row=1, col=2)

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.update_layout(height=500, width=900, title_text="Frequency of Vehicles by Entry and Exit Hours")
st.plotly_chart(fig)

st.subheader("Authorization and Vehicle Types Distribution")
fig1, fig2 = st.columns(2)
with fig1:
    auth_counts = filtered_df['Authorized'].value_counts()
    fig = px.pie(values=auth_counts, names=auth_counts.index, title='Authorized/Not Authorized')
    st.plotly_chart(fig)

with fig2:
    vehicle_type_counts = filtered_df['Vehicle Name'].value_counts()
    fig = px.pie(values=vehicle_type_counts, names=vehicle_type_counts.index, title='Vehicle Types')
    st.plotly_chart(fig)

st.subheader("Weekly Vehicle Frequency Analysis")
filtered_df = df[df['Month'] == selected_month]

filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(start_date)) & (filtered_df['Date'] <= pd.to_datetime(end_date))]
day_counts = filtered_df['Day'].value_counts()
fig = px.bar(day_counts, x=day_counts.index, y=day_counts.values, 
            labels={'x':'Day', 'y':'Frequency'}, title=f'Frequency of Vehicles by Day of the Week ({selected_month})')
st.plotly_chart(fig)


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
with popover:
    st.title("🤖 LLM-Powered Insights")

# Display the chat history
    prompt = f"Here is a CSV file containing data about intimes, outtimes, vehicle types and frequencies of vehicles grouped by days of the month in the YYYY-MM-DD format, with columns {df.columns} and data {df}.As an expert data analyst, Please analyze it and provide insights only in regards with the following question:"

    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text.lstrip(prompt))
    user_prompt = st.chat_input("Ask the chatbot questions about the data")
    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(prompt+user_prompt)

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)




