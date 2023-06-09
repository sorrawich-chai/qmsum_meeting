from transformers import pipeline
import json
import nltk
from nltk import word_tokenize

def import_data():
    data_path = 'qmsum_on_longformer/test_data/1_meet.json'
    data = []
    with open(data_path) as f:
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

def prepare_data(data):
    query = input()
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

def show_output(predicted):
    print(predicted)

def main():
    data = import_data()
    prepared_data = prepare_data(data)
    predicted = predict('fgiuhsdfkjhfv/longsec_withno_cut', prepared_data)
    show_output(predicted)

if __name__ == "__main__": 
    main()