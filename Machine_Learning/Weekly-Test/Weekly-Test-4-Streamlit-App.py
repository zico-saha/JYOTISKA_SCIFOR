import streamlit as st

# Title
st.title("My Streamlit App")

# Button
if st.button('Click Me'):
    st.write('Button clicked!')

# Slider
slider_value = st.slider('Slider', min_value=0, max_value=100, value=50)
st.write(f'Slider value: {slider_value}')

# Image
# Cannot use absolute path. Error occuring.
st.image("landscape.jpg", caption="Image")
