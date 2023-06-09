from transformers import pipeline
import json
import nltk

def import_data():
    data_path = 'ai_builder_sum_meet/deployment/data/1_meet.json'
    data = []
    with open(data_path) as f:
        data = json.load(f)
    return data

from nltk import word_tokenize
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

def lang_choose(data):
    lang = input()
    if lang == 'en':
        pass
    elif lang == 'th':
        data = translate()
    

def prepare_data(data):
    lang_choose(data)
    entire_src = []
    for i in range(len(data)):
        cur_turn = data[i]['speaker'].lower() + ': '
        cur_turn = cur_turn + tokenize(data[i]['content'])
        entire_src.append(cur_turn)
    entire_src = ' '.join(entire_src)
    return entire_src

def predict(selected_model, prepared_data):
    sum = pipeline(task="summarization",model=selected_model)
    sum(prepared_data)


def main():
    import_data()
    prepared_data = prepare_data(data)
    selected_model = choose_model()
    predicted = predict(selected_model, prepared_data)
    show_output()
