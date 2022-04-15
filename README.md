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
On SQuAD dataset, There is a paragraphs to more questions, and 

__using read_data function to split data__

```Python
from pathlib import Path  
def read_data(path, limit=None):    # limit = 
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
```

__Reselt__  
Context = Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.  

Question = To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?  

Answer = {'answer_start': 515, 'text': 'Saint Bernadette Soubirous'}  



### Data type
