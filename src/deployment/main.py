from transformers import pipeline
import json
import nltk
from nltk import word_tokenize
import streamlit as st

def load_data(data):
    #data_path = 'qmsum_on_longformer/test_data/1_meet.json'
    with open(data) as f:
        data = json.load(f)
    return data

def tokenize(sent):
    tokens = ' '.join(word_tokenize(sent.lower()))
    return tokens

def clean_data(text):
    text = text.replace('{ vocalsound } ', '')
    text = text.replace('{ disfmarker } ', '')
    text = text.replace('a_m_i_', 'ami')
    text = text.replace('l_c_d_', 'lcd')
    text = text.replace('p_m_s', 'pms')
    text = text.replace('t_v_', 'tv')
    text = text.replace('{ pause } ', '')
    text = text.replace('{ nonvocalsound } ', '')
    text = text.replace('{ gap } ', '')
    return text

def prepare_data(data,query):
    entire_src = []
    for i in range(len(data)):
        cur_turn = data[i]['speaker'].lower() + ': '
        cur_turn = cur_turn + tokenize(data[i]['content'])
        entire_src.append(cur_turn)
    entire_src = ' '.join(entire_src)
    prepared_data = clean_data('<s> ' + query + ' </s> ' + entire_src + ' </s>')
    return prepared_data

def predict(selected_model, prepared_data):
    sum = pipeline(task="summarization",model=selected_model)
    predicted = sum(prepared_data)
    return predicted

def submit_check():
    while True:
        if st.button('submit'):
            st.write('submitted')
            break

def show_output(predicted):
    st.write('predict', predicted)

def main():
    uploaded_file = st.file_uploader("choose meeting json", type=['json'])
    model_name = st.selectbox(
    'select model',
    ('fgiuhsdfkjhfv/longsec_withno_cut', 'fgiuhsdfkjhfv/longsec_general_query'))
    query = st.text_input('input query', 'summarize this meeting')
    submit_check()
    data = []
    if uploaded_file is not None:
        with open(uploaded_file, "r") as f:
            data = json.load(f)
        st.json(data)
    data = load_data(data)
    prepared_data = prepare_data(data,query)
    predicted = predict(model_name, prepared_data)
    show_output(predicted)

if __name__ == "__main__": 
    main()