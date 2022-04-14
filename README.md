# Fine tuning BERT for QuestionAnswering on SQuAD Dataset

## Quick Start
+ colab here:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huang624/NaturalLanguageUnderstanding-Fine_tuning_BERT_for_QuestionAnswering_on_SQuAD_Dataset/blob/main/BERT_for_QuestionAnswering_SQuAD.ipynb)
+ model here: <https://drive.google.com/drive/folders/1-ADNg5Zj85dN0ldYSj2AICBiFyadWlWG?usp=sharing>

## Data Processing
### Dataset introduction
Stanford Question Answering Dataset (SQuAD) is a new reading comprehension dataset consisting of 100,000+ questions posed by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text from the corresponding reading passage.  

For more information please refer to Paper: <https://arxiv.org/abs/1606.05250>

### Data Format 

<img src="https://user-images.githubusercontent.com/88367016/163392260-95a744a5-919f-4d17-8904-5d3551362bd5.png" width="600px"/>  

### Data Cleaning
``
from pathlib import Path  
def read_data(path, limit=None):
    path = Path(path)
    with open(path, 'rb') as f:
        data_dict = json.load(f)
  
    contexts = []
    questions = []
    answers = []
    for group in data_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
                if limit != None and len(contexts) > limit:
                  return contexts, questions, answers
                  
    return contexts, questions, answers
    
    ``
