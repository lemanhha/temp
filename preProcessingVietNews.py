import os, json, getopt, sys

argList = sys.argv[1:]
options = "i:o:"
longOptions = ["inputPath=", "outputPath="]

# python3 preProcessingVietNews.py -i /home/ha/GitHub/vietnews/data/ -o /home/ha/GitHub/vietnews/
inputPath = "/home/ha/GitHub/vietnews/data/"
outputPath = "/home/ha/GitHub/vietnews/"

try:
	args, values = getopt.getopt(argList, options, longOptions)
	for arg, val in args:
		if arg in ("-i", "--inputPath"):
			print("Input: %s" % val)
			inputPath = val
		elif arg in ("-o", "--outputPath"):
			print("Output: %s" % val)
			outputPath = val
except getopt.error as err:
	print(str(err))

def process(inputPath, outputFile):
	fileList = os.listdir(inputPath)
	count = 0
	output = open(outputFile, "w", encoding='utf-8')
	for fileName in fileList:
		with open(inputPath + fileName, "r") as docFile:
			lines = docFile.readlines()
		n = len(lines)

		#print(fileName)
		#print(len(lines))
		#print("lines[1] len = %d" % len(lines[1]))
		#print("lines[3] len = %d" % len(lines[3]))
		#print("lines[n-3] len = %d" % len(lines[n-3]))
		#print("lines[n-2] len = %d" % len(lines[n-2]))

		summary = lines[2]
		text = ""
		for i in range(4, n):
			if len(lines[i]) < 2:
				break
			text += lines[i]
		sample = {
			"text": text,
			"summary": summary
		}
		#count += 1
		#if count == 10:
		#	break 
		output.write(json.dumps(sample, ensure_ascii=False) + "\n")

process(inputPath + "train_tokenized/", outputPath + "train.json")
process(inputPath + "test_tokenized/", outputPath + "test.json")
process(inputPath + "val_tokenized/", outputPath + "val.json")

