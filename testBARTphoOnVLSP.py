# python temp/testBARTphoOnVLSP.py -m finetune-bartpho-newscorpus50 -i VLSP_Data/vlsp_abmusu_test_data.jsonl -o VLSP_Data/results.txt
# python temp/testBARTphoOnVLSP.py -m finetune-bartpho-newscorpus50/checkpoint-48158 -i VLSP_Data/vlsp_2022_abmusu_validation_data_new.jsonl -o VLSP_Data/vlsp_validation_results.txt -e VLSP_Data/vlsp_validation_evaluation.json

import json, getopt, sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge

# Setup arguments
argList = sys.argv[1:]
options = "m:i:o:e"
longOptions = ["modelPath=", "inputPath=", "outputPath=", "evaluationPath="]

# Default arguments
modelPath = "finetune-bartpho-newscorpus50"
inputPath = "VLSP_Data/vlsp_abmusu_test_data.jsonl"
outputPath = "VLSP_Data/results.txt"
evaluationPath = ""

# Parse arguments
try:
    args, values = getopt.getopt(argList, options, longOptions)
    for arg, val in args:
        print("arg = %s; val = %s" % (arg, val))
        if arg in ("-m", "--modelPath"):
            print("Model: %s" % val)
            modelPath = val
        elif arg in ("-i", "--inputPath"):
            print("Input: %s" % val)
            inputPath = val
        elif arg in ("-o", "--outputPath"):
            print("Output: %s" % val)
            outputPath = val
        elif arg in ("-e", "--evaluationPath"):
            print("Evaluation: %s" % val)
            evaluationPath = val
except getopt.error as err:
    print(str(err))

# load model
tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModelForSeq2SeqLM.from_pretrained(modelPath)

# init rouge
rougeScorer = Rouge()

def getSingleDocument(documents):
    return '.'.join(document["raw_text"] for document in documents)

def getSummary(multidocs, evaluate):
    print(multidocs)
    jsonObject = json.loads(multidocs)
    documents = jsonObject["single_documents"]
    text = getSingleDocument(documents)

    tokens_input = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(tokens_input, min_length=256, max_length=512)
    prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    scores = rougeScorer.get_scores(prediction, jsonObject["summary"]) if evaluate else None;

    summary = {"prediction": prediction, "scores": scores}
    return summary

def process():
    # read input
    with open(inputPath, "r") as input:
        lines = list(input)

    # calculate summaries
    summaries = [getSummary(lines, evaluationPath != "")]

    # write prediction to output
    with open(outputPath, "w", encoding='utf-8') as output:
        output.write("\n".join([summary["prediction"] for summary in summaries]))
        output.close()

    # write prediction with rouge scores
    if evaluationPath != "":
        with open(evaluationPath, "w", encoding='utf-8') as evaluation:
            evaluation.write("\n".join([json.dumps(summary, ensure_ascii=False, indent = 4) for summary in summaries]))
            evaluation.close()

process()
