# python temp/testBARTphoOnVLSP_syllable.py -m finetune-bartpho-newscorpus10-syllable -i VLSP_Data/LexRank/vlsp2022_test_full.json --outputPath VLSP_Data/LexRank/results-syllable-10-percents.txt

import json, getopt, sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup arguments
argList = sys.argv[1:]
options = "m:i:o"
longOptions = ["modelPath=", "inputPath=", "outputPath="]

# Default arguments
modelPath = "finetune-bartpho-newscorpus10-syllable"
inputPath = "VLSP_Data/LexRank/vlsp2022_test_full.json"
outputPath = "VLSP_Data/LexRank/results-syllable-10-percents.txt"

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
except getopt.error as err:
    print(str(err))

# load model
tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModelForSeq2SeqLM.from_pretrained(modelPath)

def getSummary(multidocs):
    jsonObject = json.loads(multidocs)
    text = jsonObject["text"]
    text = text.replace("_"," ")

    tokens_input = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(tokens_input, min_length=256, max_length=512)
    prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return prediction

def process():
    # read input
    with open(inputPath, "r") as input:
        lines = list(input)

    # calculate summaries
    summaries = []
    for line in lines:
        summary = getSummary(line)
        summaries.append(summary)
        print("Processed %d / %d" % (len(summaries), len(lines)))

    # write prediction to output
    with open(outputPath, "w", encoding='utf-8') as output:
        output.write("\n".join([summary for summary in summaries]))
        output.close()

process()
