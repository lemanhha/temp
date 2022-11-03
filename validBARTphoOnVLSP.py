# python temp/validBARTphoOnVLSP.py -m finetune-bartpho-newscorpus50/checkpoint-96316 -i VLSP_Data/LexRank/vlsp2022_validation_full.json -o VLSP_Data/vlsp_validation_results.txt --evaluationPath VLSP_Data/vlsp_validation_evaluation.json

import json, getopt, sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge

# Setup arguments
argList = sys.argv[1:]
options = "m:i:o:e"
longOptions = ["modelPath=", "inputPath=", "outputPath=", "evaluationPath="]

# Default arguments
modelPath = "finetune-bartpho-newscorpus50/checkpoint-96316"
inputPath = "VLSP_Data/LexRank/vlsp2022_validation_full.json"
outputPath = "VLSP_Data/LexRank/vlsp2022_validation_full_result.txt"
evaluationPath = "VLSP_Data/LexRank/vlsp2022_validation_full_evaluation.txt"

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

def getSummary(multidocs, evaluate):
    jsonObject = json.loads(multidocs)
    text = jsonObject["text"]

    tokens_input = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(tokens_input, min_length=256, max_length=512)
    prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    prediction = prediction.replace("_", " ")
    reference = jsonObject["summary"].replace("_", " ")

    scores = rougeScorer.get_scores(prediction, reference) if evaluate else None;

    summary = {"prediction": prediction, "scores": scores}
    return summary

def process():
    # read input
    with open(inputPath, "r") as input:
        lines = list(input)

    # calculate summaries
    summaries = []
    for line in lines:
        summary = getSummary(line, evaluationPath != "")
        summaries.append(summary)
        print("Processed %d / %d" % (len(summaries), len(lines)))

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
