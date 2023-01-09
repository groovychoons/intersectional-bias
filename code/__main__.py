import load_data
import clean_data
import train_model
import load_model
import ibd

def preprocessing():
    print("Starting preprocessing")
    data = load_data.main()
    print(data[0])
    cleandata = clean_data.main(data)
    print(cleandata[0])
    model = train_model.main(cleandata)
    return model

def evaluation():
    model = load_model.main()
    print(len(model.wv))
    return True

def name_clustering():
    model = load_model.load_google_news()
    

if __name__ == "__main__": 
    #model = preprocessing()
    evaluation()