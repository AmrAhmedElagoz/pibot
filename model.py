from transformers import ElectraForQuestionAnswering, AutoTokenizer, pipeline
from arabert import ArabertPreprocessor
from arabert.arabert.tokenization import BasicTokenizer

arabert_prep = ArabertPreprocessor("araelectra-base-discriminator")
qa_modelname = 'ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA'


qa_model = ElectraForQuestionAnswering.from_pretrained(qa_modelname)
tokenizer = AutoTokenizer.from_pretrained(qa_modelname)

bt = BasicTokenizer(do_lower_case=False)

def clean_preprocess(text, processor):
    text = " ".join(bt._run_split_on_punc(text))
    text = processor.preprocess(text)
    text = " ".join(text.split()) 
    return text

def qa(question, context):
    question = clean_preprocess(question, arabert_prep)
    context = clean_preprocess(context, arabert_prep)
    

    qa_pipe = pipeline('question-answering', model= qa_model, tokenizer= tokenizer)
    QA_input = {
        'question': question,
        'context': context
    }

    qa_res = qa_pipe(QA_input)

    return qa_res