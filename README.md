# smart_bin_classifier

 This is a deep learning application that takes images of Amazon fulfillment bins and predicts exactly how many items are inside and what products they contain.

It’s built with Streamlit for the web interface and uses a Hybrid Neural Network under the hood.

🔗 Live Demo: Click Here to Try the App: https://smartbinclassifier.streamlit.app


🧐 How It Works

Detecting items in cluttered bins is surprisingly hard. Standard models often struggle when items are stacked on top of each other or look similar.

To solve this, I used a Hybrid Architecture that combines two powerful AI models:

EfficientNetV2: A state-of-the-art CNN that looks at the raw pixel data to understand shapes and textures.

OpenAI CLIP: A vision-transformer model that understands images in a semantic way (how images relate to text/concepts).

By fusing these two "brains" together, the model is much better at distinguishing between products than either model could be on its own.

🚀 Features

Item Counting: Estimates how many individual items are in the bin.

Quantity Prediction: Predicts the total quantity of products.

Product Identification: Identifies specific products (ASINs) and lists them with confidence scores.

Auto-Healing: The app automatically checks if model files are missing or corrupt and downloads fresh copies from the cloud.

🛠️ Installation & Local Setup

If you want to run this locally on your own machine:

Clone the repository

git clone [https://github.com/your-username/amazon-bin-classifier.git](https://github.com/your-username/amazon-bin-classifier.git)
cd amazon-bin-classifier


Install Dependencies
I use tensorflow-cpu to keep the install size light.

pip install -r requirements.txt


Run the App

streamlit run app.py


> Note: The first time you run this, it will take a minute to download the model weights (~200MB) from Google Drive. Don't worry, it only happens once!

📂 Project Structure

app.py: The main application code. It handles the UI, model loading, and image processing.

requirements.txt: List of all Python libraries needed.

demo_bin_classifier_best.h5: The trained model weights (downloaded automatically).

asin_vocabulary.pkl: The mapping of Product IDs to names (downloaded automatically).

☁️ Deployment Notes

This app is optimized for Streamlit Community Cloud.

If you are deploying this yourself, note that the model files are too large for GitHub. I've set up app.py to stream them directly from Google Drive into the app's memory during startup. This bypasses GitHub's file size limits and keeps the repository clean.

Built with ❤️ using Python, TensorFlow, and Streamlit.
