import streamlit as st
from transcript import extract_video_id, get_transcript
from embeddings import video_exists_in_db, store_transcript
from rag import ask_question

st.set_page_config(page_title="VideoSage", page_icon="🎬", layout="wide")


if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

# sidebar
with st.sidebar:
    st.title("VideoSage")
    st.markdown("Ask questions about any YouTube video!")
    st.divider()

    video_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Process Video", type="primary", use_container_width=True):
        if not video_url:
            st.error("Please enter a YouTube URL.")
        else:
            try:
                video_id = extract_video_id(video_url)
                st.session_state.video_id = video_id

                if video_exists_in_db(video_id):
                    st.success("Video already processed! You can start asking questions.")
                    st.session_state.video_processed = True
                else:
                    with st.spinner("Fetching transcript..."):
                        transcript_text = get_transcript(video_url)

                    with st.spinner("Processing and storing embeddings..."):
                        store_transcript(video_id, transcript_text)

                    st.success("Video processed successfully!")
                    st.session_state.video_processed = True

                # Clear chat history for new video
                st.session_state.messages = []

            except ValueError as e:
                st.error(f"Invalid URL: {e}")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.video_id:
        st.divider()
        st.caption(f"Current video ID: {st.session_state.video_id}")

# --- Main Chat Area ---
st.header("Chat with your Video")

if not st.session_state.video_processed:
    st.info("Enter a YouTube URL in the sidebar and click 'Process Video' to get started.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if question := st.chat_input("Ask a question about the video..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_question(st.session_state.video_id, question)
            st.markdown(answer)

        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
