import os, json

inputPath = "~/GitHub/VietnameseMDS/KTLAB-200document-clusters/clusters/"
outputPath = "~/GitHub/VietnameseMDS/"

def process(i_start, i_end, outputFile):
	data = []
	for i in range(i_start, i_end):
		path = inputPath + "cluster_%d/" % i
		fileList = os.listdir(path)
		#print(fileList)
		for documentFile in fileList:
			if not documentFile.startswith("cluster") and documentFile.endswith(".body.tok.txt"):
				#print(documentFile)
				with open(path + documentFile, "r") as docFile:
					text = docFile.read()
				#print(text)
				pos = documentFile.find(".body.tok.txt")
				summaryFile = documentFile[:pos] + ".info.txt"
				#print(summaryFile)
				with open(path + summaryFile, "r") as sumFile:
					lines = sumFile.readlines()
				summary = ""
				for line in lines:
					if line.startswith("SUMMARY"):
						summary = line[9:]
						break
				#print(summary)
				sample = {
					"text": text,
					"summary": summary
				}
				data.append(sample)
	with open(outputFile, "w", encoding='utf-8') as output:
		json.dump(data, output, indent = 4, ensure_ascii=False)

process(1, 141, outputPath + "train.json")
process(141, 171, outputPath + "test.json")
process(171, 201, outputPath + "val.json")

