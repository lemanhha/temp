import os, json, getopt, sys, evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from multiprocessing import Pool

argList = sys.argv[1:]
options = "m:i:o:pr"
longOptions = ["modelPath=", "inputPath=", "outputPath=", "processes="]

# python testBARTphoWithVietNews.py -m ../vietnews/tst-summarization -i ../vietnews/data/test_tokenized/ -o ../vietnews/test_bartpho_with_vietnews_test.json --processes 12
modelPath = "../vietnews/tst-summarization"
inputPath = "../vietnews/data/test_tokenized/"
outputFile = "../vietnews/test_bartpho_with_vietnews_test.json"
processes = 12

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
            outputFile = val
        elif arg in ("-p", "--processes"):
            print("Processes: %d" % int(val))
            processes = int(val)
except getopt.error as err:
    print(str(err))

tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModelForSeq2SeqLM.from_pretrained(modelPath)
metric = evaluate.load('rouge')

def f(fileName):
    print(fileName)
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
    rouge = metric.compute(references=[summary], predictions=[prediction])
    result = {
        "file_name": fileName,
        "doc": text,
        "summary": summary,
        "predict": prediction,
        "rouge1": rouge['rouge1'],
        "rouge2": rouge['rouge2'],
        "rougel": rouge['rougeL']
    }
    return result

def process(inputPath, outputFile):
    fileList = sorted(os.listdir(inputPath))
    output = open(outputFile, "w", encoding='utf-8')
    numberOfFiles = len(fileList)
    count = 0
    for fileName in fileList:
        print(fileName)
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

def parallelProcess(inputPath, outputFile):
    fileList = sorted(os.listdir(inputPath))
    with Pool(processes) as p:
        testResult = p.map(f, fileList)
    with open(outputFile, "w") as out:
        json.dump(testResult, out, ensure_ascii = False, indent = 4)

process(inputPath, outputFile)
# parallelProcess(inputPath, outputFile)
