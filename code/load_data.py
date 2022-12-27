import os

def main():
    if not os.path.exists('data'):
        print("Creating a data directory")
        os.mkdir('/data')
    
    if os.path.isfile('data/testset.en.shuffled.deduped'):
        rawdata = []
        with open("../data/testset.en.shuffled.deduped") as infile:
            print("Data loaded")
            for line in infile:
                rawdata.append(line)

        print("No. of sentences: ", len(rawdata))

        return rawdata

