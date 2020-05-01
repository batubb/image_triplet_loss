import os
from PIL import Image
import cv2


# topDir = "ChallengeImages2"
# sizes = []
# for dirName in os.listdir(topDir):
# 	curFile = os.path.join(topDir, dirName)
# 	if os.path.isdir(curFile):
# 		for filename in os.listdir(curFile):
# 			curImageFileName = os.path.join(curFile, filename)
# 			if curImageFileName.endswith(".png"):
# 				im = Image.open(curImageFileName)
# 	 			width, height = im.size
# 	 			if width != 1127 and height != 544:
# 	 				print(curImageFileName)
# 	 				print(width)
# 	 				print(height)

				
		



# 1. get the sizes
walkDir = "ChallengeImages"
topDir = "ChallengeImages2"
sizes = []
for dirName in os.listdir(walkDir):
	curFile = os.path.join(walkDir, dirName)
	if os.path.isdir(curFile):
		for filename in os.listdir(curFile):
			curImageFileName = os.path.join(curFile, filename)
			if curImageFileName.endswith(".png"):
				im = Image.open(curImageFileName)
				width, height = im.size
				sizes.append((width, height))

allHeights = []
allWidths = []
for size in sizes:
	allHeights.append(size[1])
	allWidths.append(size[0])

allHeights.sort()
allWidths.sort()
# 2. resize to average size
avgHeight = int(sum(allHeights) / len(allHeights))
avgWidth = int(sum(allWidths) / len(allWidths))

minHeight = min(allHeights)
minWidth = min(allWidths)

customHeight = 10
customWidth = 20
size = avgWidth, avgHeight
for dirName in os.listdir(topDir):
	curFile = os.path.join(topDir, dirName)
	if os.path.isdir(curFile):
		for filename in os.listdir(curFile):
			curImageFileName = os.path.join(curFile, filename)
			if curImageFileName.endswith(".png"):
				idx = curImageFileName.find(".png")
				outfileName = curImageFileName[:idx] + ".png"
				
				img = cv2.imread(curImageFileName)
				dims = (customWidth, customHeight)
				resizedImage = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
				cv2.imwrite(outfileName, resizedImage)




# widths = []
# heights = []
# # 1. get the sizes
# topDir = "ChallengeImages2"
# sizes = []
# for dirName in os.listdir(topDir):
# 	curFile = os.path.join(topDir, dirName)
# 	if os.path.isdir(curFile):
# 		for filename in os.listdir(curFile):
# 			curImageFileName = os.path.join(curFile, filename)
# 			if curImageFileName.endswith(".thumbnail"):
# 				im = Image.open(curImageFileName)
# 				width, height = im.size
# 				widths.append(width)
# 				heights.append(height)
# widths.sort()
# heights.sort()
# print(widths)
# print(heights)

