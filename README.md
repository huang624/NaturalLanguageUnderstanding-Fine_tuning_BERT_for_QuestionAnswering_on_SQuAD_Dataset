# Fine tuning BERT for QuestionAnswering on SQuAD Dataset

## Quick Start
+ colab here:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huang624/NaturalLanguageUnderstanding-Fine_tuning_BERT_for_QuestionAnswering_on_SQuAD_Dataset/blob/main/BERT_for_QuestionAnswering_SQuAD.ipynb)
+ model here: <https://drive.google.com/drive/folders/1-ADNg5Zj85dN0ldYSj2AICBiFyadWlWG?usp=sharing>

## Data Processing
### Dataset introduction
Stanford Question Answering Dataset (SQuAD) is a new reading comprehension dataset consisting of 100,000+ questions posed by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text from the corresponding reading passage.  

For more information please refer to Paper: <https://arxiv.org/abs/1606.05250>

### Data Format
data:  
+ title: article_title  
+ id:article_id  
+ paragraphs:   
  >- id: article_id+ paragraphs_id  
  >- context: paragraphs_content  
  >- qas:  
    > - question: question  
    > - article_id+ paragraphs_id+ question_id     
> - answers:   
&emsp;&emsp;&emsp;--answer_start: answer position in context  
&emsp;&emsp;&emsp;--id: "1"表示為人工標註的答案，"2"以上為人工答題的答案  
&emsp;&emsp;&emsp;--text: answer  
