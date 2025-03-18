import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# carregar o modelo salvo

model_path = "best_model_dog_vs_cat.h5" # certifique-se de colocar o caminho correto do modelo
model = tf.keras.models.load_model(model_path)

# função para processar a imagem
def process_image(uploaded_file):
        img = Image.open(uploaded_file)
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0 # normalizar os pixels
        img_array = np.expand_dims(img_array, axis=0) # expandir a dimensão batch
        return img, img_array
   
# configuração da aplicação streamlit
st.title("Classificador de imagens: Cachorro ou Gato")
st.write("Faça o upload de uma imagem para classificação")

# upload de imagem pelo usuário
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # processar a imagem
    img, img_array = process_image(uploaded_file)
   
    # exibir a imagem
    st.image(img, caption="Imagem carregada", use_column_width=True)
   
    # fazer previsão
    prediction = model.predict(img_array)
   
    # exibir resultado
    if prediction[0] < 0.5:
        st.write("A imagem é de um **gato** 🐱.")
    else:
        st.write('A imagem é de um **cachorro** 🐶.')



        


