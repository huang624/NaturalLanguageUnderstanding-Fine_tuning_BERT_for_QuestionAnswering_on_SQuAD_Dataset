from fastapi import Body, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, constr
from transformers import BertTokenizerFast, BertConfig, BertForQuestionAnswering, default_data_collator
from accelerate import Accelerator
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


class bert_squad_Request(BaseModel):
    context: constr(max_length=512)
    question: constr(max_length=512)

class Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

  def __len__(self):
    return len(self.encodings.input_ids)

def bert_squad_model_predict(model, context, question):   
    input_encodings = tokenizer([context], [question], truncation=True, padding=True)
    input_dataset = Dataset(input_encodings)
  
    data_collator = default_data_collator
    input_dataloader = DataLoader(input_dataset, collate_fn=data_collator, batch_size=1)  
  
    accelerator = Accelerator()
    model, input_dataloader = accelerator.prepare(model, input_dataloader)
    for batch in input_dataloader:
      outputs = model(**batch)
  
      start_predicted = outputs.start_logits.argmax(dim=-1)
      end_predicted = outputs.end_logits.argmax(dim=-1)
      
      input_ids = batch['input_ids'][0]
      answer = tokenizer.decode(input_ids[start_predicted:end_predicted+1])

    return answer


app = FastAPI(
    title="Bert_squad",
    description="SQuAD dataset training",
    version="1",
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("config.json") 
bert_squad_model = BertForQuestionAnswering.from_pretrained("pytorch_model.bin", config = config)

@app.get("/")
async def root():
    return RedirectResponse("docs")


@app.get("/page/{page_name}", response_class=HTMLResponse)
async def page(request: Request, page_name: str):
    return templates.TemplateResponse(f"{page_name}.html", {"request": request})


@app.post("/bert_squad")
async def bert_squad(
    bert_squad_Request: bert_squad_Request = Body(
        None,
        
    )
):

    results = bert_squad_model_predict(bert_squad_model, bert_squad_Request.context, bert_squad_Request.question)
    return results

