from PIL import Image
from torch.utils.data import Dataset
import torchvision
import os
import pandas as pd

class Fpidataset(Dataset):
    # Constructor
    def __init__(self, train, img_size, transform=True):
        # Image directory
        self.transform = transform
        self.img_size = img_size
        self.train = train
        #self.df = self.df[self.df['fold'] == fold]

        if transform is not None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor()
            ])
        self.transform = transform

        df = pd.read_csv('data/styles.csv', error_bad_lines=False)
        df['image_path'] = df.apply(lambda x: os.path.join("data\images", str(x.id) + ".jpg"), axis=1)

        mapper = {}
        for i, cat in enumerate(list(df.articleType.unique())):
            mapper[cat] = i
        print(mapper)
        df['targets'] = df.articleType.map(mapper)

        if self.train:
            self.df = get_i_items(df, 800)
        else:
            self.df = get_i_items(df, 200)

    # Get the length
    def __len__(self):
        return len(self.df)

    # Getter
    def __getitem__(self, idx):
        #get imagepath
        img_path = self.df.image_path[idx]

        #open as PIL Image
        img = Image.open(img_path).convert('RGB')

        #transform
        image = self.transform(img)

        #get label
        label = self.df.targets[idx]

        return image, label


def get_i_items(df, i):
    # get i items of each condition

    # calculate classes with more than 1000 items
    temp = df.targets.value_counts().sort_values(ascending=False)[:10].index.tolist()
    df_temp = df[df["targets"].isin(temp)]

    #generate new empty dataframe with the columns of the original
    dataframe = df[:0]

    #for each targetclass in temp insert i items in dataframe

    for element in temp:
        print("Füge Items mit target", element, "ein.")
        dataframe = dataframe.append(df_temp[df_temp.targets == element][:i])
        print("Anzahl items", len(dataframe))

    return dataframe