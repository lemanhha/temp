import os, json, getopt, sys, evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

argList = sys.argv[1:]
options = "m:i:o:"
longOptions = ["modelPath=", "inputPath=", "outputPath="]

# python testBARTphoWithVietNews.py -m ../vietnews/tst-summarization -i ../vietnews/data/test_tokenized/ -o ../vietnews/test_bartpho_with_vietnews_test.json
modelPath = "../vietnews/tst-summarization"
inputPath = "../vietnews/data/test_tokenized/"
outputPath = "../vietnews/test_bartpho_with_vietnews_test.json"

try:
    args, values = getopt.getopt(argList, options, longOptions)
    for arg, val in args:
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

tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModelForSeq2SeqLM.from_pretrained(modelPath)
metric = evaluate.load('rouge')

def process(inputPath, outputFile):
    fileList = os.listdir(inputPath)
    output = open(outputFile, "w", encoding='utf-8')
    numberOfFiles = len(fileList)
    count = 0
    for fileName in fileList:
        # print(fileName)
        with open(inputPath + fileName, "r") as docFile:
            lines = docFile.readlines()
        n = len(lines)

        summary = lines[2]
        text = ""
        for i in range(4, n):
            if len(lines[i]) < 2:
                break
            text += lines[i]

        tokens_input = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        summary_ids = model.generate(tokens_input, min_length=80, max_length=120)
        prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        rouge = metric.compute(references = [summary], predictions = [prediction])
        result = {
            "file_name": fileName,
            "doc": text,
            "summary": summary,
            "predict": prediction,
            "rouge1": rouge['rouge1'],
            "rouge2": rouge['rouge2'],
            "rougel": rouge['rougeL']
        }
        output.write(json.dumps(result, ensure_ascii=False) + "\n")

        count += 1
        if count % 100 == 0:
            print("%d / %d" % (count, numberOfFiles))

process(inputPath, outputPath)
