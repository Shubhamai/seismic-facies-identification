# Importing Necessary Libraries
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_cropper import st_cropper
import cv2

# Setting the title & icon
st.beta_set_page_config(
    page_title="Seismic Facies Identification", page_icon="🌍")

# Removing the Meno
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Setting the page title & Description
st.title("🌎 Seismic Facies Identification")
st.markdown(
    """This is the **Image Visulaisation & Preprocessing Application** for this challange, for the complete data exploration & submitting please checkout my [Colab Notebook Here](https://colab.research.google.com/drive/1t1hF_Vs4xIyLGMw_B9l1G6qzLBxLB5eG?usp=sharing)
""")

st.markdown("""  
Made By : [Shubhamai](https://shubhamai.com/)

Github Project : https://github.com/Shubhamai/seismic-facies-identification

Google Colab Notebook : https://colab.research.google.com/drive/1t1hF_Vs4xIyLGMw_B9l1G6qzLBxLB5eG?usp=sharing""")


st.write("-------------------------------------------------------")
st.write("")

st.image("./images/notebook.gif",
         caption="The Complete Colab Notebook",
         use_column_width=True)

st.write("")
st.write("")

st.markdown(
    "## [Problem](https://www.aicrowd.com/challenges/seismic-facies-identification-challenge#introduction)")

st.write("Segmentating the 3D seismic image into an image with **each pixel can be classfied into 6 labels** based on patterns in the image.")

st.write("")

st.markdown(
    "## [Dataset](https://www.aicrowd.com/challenges/seismic-facies-identification-challenge#dataset)")

st.markdown("""We have **3D dataset** both ( features X, and labels Y ) with shape for **X is 1006 × 782 × 590, in axis corresponding Z, X, Y** and **Y in 1006 × 782 × 590 in also axis corresponsing Z, X, Y**.

We can say that we have total of **2,378 trainig images with their corresponsing labels** and we also have same number of **2,378 testing images which we will predict labels for**.
""")

st.markdown(
    "## [Evaluation](https://www.aicrowd.com/challenges/seismic-facies-identification-challenge#evaluation-criteria)")

st.write("The evaluation metrics are the F1 score and accuracy.")

st.write("")
st.write("")

st.title("Data Visualisations")

# Loading the dataset
X = np.load("./data/X.npy")
Y = np.load("./data/y.npy")

# Making a subplot with 1 row and 2 column
fig = make_subplots(1, 2, subplot_titles=("Image", "Label"))

# Visualising a section of the 3D array
fig.add_trace(go.Heatmap(z=X, ), 1, 1)

fig.add_trace(go.Heatmap(z=Y), 1, 2)

fig.update_layout(title_text="Seismic Image & Label")

st.plotly_chart(fig,)


# Visualising a surface plot
fig = make_subplots(1, 2, subplot_titles=("Image", "Label"), specs=[
                    [{"type": "Surface"}, {"type": "Surface"}]])

# Making a 3D Surphace graph with image and corresponsing label
fig.add_trace(go.Surface(z=X), 1, 1)
fig.add_trace(go.Surface(z=Y), 1, 2)

fig.update_layout(title_text="Seismic Image & Label in 3D!")

st.plotly_chart(fig)


# Making a subplot with 1 row and 2 column
fig = make_subplots(1, 2, subplot_titles=("Image", "Label"))

# Making a contour graph
fig.add_trace(go.Contour(
    z=X), 1, 1)

fig.add_trace(go.Contour(
    z=Y
), 1, 2)


fig.update_layout(title_text="Seismic Image & Label in with contours")

st.plotly_chart(fig)

fig = make_subplots(2, 2, subplot_titles=("Image", "Label", "Label Histogram"))

# Making a contour graph
fig.add_trace(go.Contour(
    z=X, contours_coloring='lines',
    line_width=2,), 1, 1)

# Showing the label ( also the contour )
fig.add_trace(go.Contour(
    z=Y
), 1, 2)

# Showing histogram for the label column
fig.add_trace(go.Histogram(x=Y.ravel()), 2, 1)


fig.update_layout(
    title_text="Seismic Image & Label in with contours ( only line )")

st.plotly_chart(fig)

# Making a subplot with 2 row and 1 column
fig = make_subplots(2, 1, subplot_titles=("Image", "label"))

# Making a contour graph
fig.add_trace(
    go.Contour(
        z=X
    ), 1, 1)

fig.add_trace(go.Contour(
    z=Y
), 2, 1)

fig.update_layout(
    title_text="Seismic Image & Label in with contours ( More Closer Look )")

st.plotly_chart(fig)


st.title("Image Preprocessing")

st.write("Just use some of the methods in the sidebar and see the result! You can also crop to see certain part of images")

img = cv2.imread("./images/img_781.jpg")



preprocessing_methods = st.sidebar.multiselect("Image Preprocessing", [
                                               "Threshold", "SobelX", "SobelY", "Laplacian", "Erosion", "Dialation", "Sharping"], default=None)

st.sidebar.text("")
st.sidebar.warning("Use Threshold, Erosion or Dialation to change settings")

# Setting the preprocessing 
if preprocessing_methods != []:

    for i in preprocessing_methods:
        if i == "Threshold":
            st.sidebar.subheader("Threshold")

            thresh = st.sidebar.slider(
                "Threshold Value", min_value=0., max_value=255.)
            maxval = st.sidebar.slider(
                "Max Value", min_value=0., max_value=255.)
            thresh_type = st.sidebar.selectbox(
                "Threshold Option", [cv2.THRESH_TOZERO])
            ret, img = cv2.threshold(img, thresh, maxval, thresh_type)

        elif i == "SobelX":
            kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], np.float32)
            img = cv2.filter2D(img, -1, kernel)

        elif i == "SobelY":
            kernel = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]], np.float32)
            img = cv2.filter2D(img, -1, kernel)

        elif i == "Erosion":
            st.sidebar.subheader("Erosion")

            erosion_kernel_size = st.sidebar.slider(
                "Erosion Kernel Size", min_value=0, max_value=10)
            erosion_iteration = st.sidebar.slider(
                "Erosion Iteration Number", min_value=0, max_value=10)
            kernel = np.ones(
                (erosion_kernel_size, erosion_kernel_size), np.uint8)
            img = cv2.erode(img, kernel, iterations=erosion_iteration)

        elif i == "Dialation":
            st.sidebar.subheader("Dialation")

            dialation_kernel_size = st.sidebar.slider(
                "Dialation Kernel Size", min_value=0, max_value=10)
            dialation_iteration = st.sidebar.slider(
                "Dialation Iteration Number", min_value=0, max_value=10)
            kernel = np.ones(
                (dialation_kernel_size, dialation_kernel_size), np.uint8)
            img = cv2.dilate(img, kernel, iterations=dialation_iteration)

        elif i == "Laplacian":
            kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], np.float32)

            img = cv2.filter2D(img, -1, kernel)

        else:
            kernel = np.array(
                [[0, -1, -1], [2, -1, 2], [-1, 2, -1]], np.float32)
            img = cv2.filter2D(img, -1, kernel)


st.subheader("Output & Label")

# Displaying multiple image side by side in streamlit, found from here https://gist.github.com/treuille/2ce0acb6697f205e44e3e0f576e810b7
def paginator(label, items, items_per_page=10, on_sidebar=True):
   
    items = list(items)

    min_index = 0 * items_per_page
    max_index = min_index + items_per_page
    import itertools
    return itertools.islice(enumerate(items), min_index, max_index)


sunset_imgs = [img, "./images/img_781.png"]

image_iterator = paginator("Select a sunset page", sunset_imgs)
indices_on_page, images_on_page = map(list, zip(*image_iterator))


st.image([img, "./images/img_781.png"], use_column_width=False, width=336, clamp=True)

st.subheader("Cropper")

cropped_img = st_cropper(Image.fromarray(img, 'RGB'), realtime_update=True, box_color='white',
                         aspect_ratio=None)

st.subheader('Cropped Output')


st.image(cropped_img, use_column_width=True)


st.title("Custom Preprocessing")

st.write("Put your code in format like below")
st.code("""img = cv2.blur(img, (5,5))

# Or, you can also try a custom kernel ( outline )

kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]], np.float32)

img = cv2.filter2D(img, -1, kernel)
""")
code = st.text_area("Code")

if code != "":
    st.code(code)

    st.subheader("Custom Proprocessing Results")

    try:

        exec(code)

        st.image(img, use_column_width=True, width=100, clamp=True)
    except Exception as exception:
        st.error(
            f"Error Occured, Please make sure that code is correct! Error is : {exception}")
        
st.success("Thanks for checking this out! please like the post https://discourse.aicrowd.com/t/explained-by-the-community-win-4-x-dji-mavic-drones/3636/4?u=shubhamai 🙂")
