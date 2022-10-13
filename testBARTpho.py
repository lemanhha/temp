import os, json, getopt, sys, evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

argList = sys.argv[1:]
options = "m:i:o"
longOptions = ["modelPath=", "inputPath=", "outputPath="]

# python temp/testBARTpho.py -m finetune-bartpho-vlsp -i vlsp2022_train_split/vlsp2022_test.json --outputPath test-finetuned-bartpho/vlsp_test.json
# python temp/testBARTpho.py -m finetune-bartpho-vmds -i vmds/vmds_test.json --outputPath test-finetuned-bartpho/vmds_test.json
# python temp/testBARTpho.py -m finetune-bartpho-vims -i vims/vims_test.json --outputPath test-finetuned-bartpho/vims_test.json
modelPath = "finetune-bartpho-vlsp"
inputFile = "vlsp2022_train_split/vlsp2022_test.json"
outputFile = "test-finetuned-bartpho/vlsp_test.json"
processes = 12

try:
    args, values = getopt.getopt(argList, options, longOptions)
    for arg, val in args:
        if arg in ("-m", "--modelPath"):
            print("Model: %s" % val)
            modelPath = val
        elif arg in ("-i", "--inputPath"):
            print("Input: %s" % val)
            inputFile = val
        elif arg in ("-o", "--outputPath"):
            print("Output: %s" % val)
            outputFile = val
except getopt.error as err:
    print(str(err))

tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModelForSeq2SeqLM.from_pretrained(modelPath)
metric = evaluate.load('rouge')

def process(inputFile, outputFile):
    with open(inputFile, "r") as input:
        lines = list(input)
    output = open(outputFile, "w", encoding='utf-8')
    for line in lines:
        jsonObject = json.loads(line)
        text = jsonObject["text"]
        summary = jsonObject["summary"]
        tokens_input = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        summary_ids = model.generate(tokens_input, min_length=80, max_length=120)
        prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        rouge = metric.compute(references = [summary], predictions = [prediction])
        result = {
            "doc": text,
            "summary": summary,
            "predict": prediction,
            "rouge1": rouge['rouge1'],
            "rouge2": rouge['rouge2'],
            "rougel": rouge['rougeL']
        }
        output.write(json.dumps(result, ensure_ascii=False, indent = 4) + "\n")
    output.close()

process(inputFile, outputFile)
