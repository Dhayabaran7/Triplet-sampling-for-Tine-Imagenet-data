import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torch.utils.data as data
import torchvision.models as models
from PIL import Image
import random
import pandas as pd
import os
import os.path
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#device = torch.device('cpu')

extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def has_file_allowed_extension(filename, extensions):
    #Checks if a file is an allowed extension.
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

class TinyImageData(data.Dataset):
    def __init__(self,root,extensions,transform=None, train = True):
        classes, class_to_idx = self._find_classes(root)
        
        if train==True:
            samples = make_dataset(root, class_to_idx, extensions)
            if len(samples) == 0:
                raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        else:
            samples = []
            table = pd.read_csv('tiny-imagenet-200/val/val_annotations.txt', sep='\t', header=None)
            table.drop(labels=[2,3,4,5], axis=1, inplace=True)
            #for image, label in self.table.iterrows():
             #   image = 'tiny-imagenet-200/val/images' + str(image)
              #  item = (image,class_to_idx[label]) 
               # self.samples.append(item)
            for i in range(table.shape[0]):
                for j in range(table.shape[1]):
                    if j==0:
                        image = 'tiny-imagenet-200/val/images/' + str(table.iloc[i][j])
                    else:
                        item = (image, class_to_idx[table.iloc[i][j]])
                        samples.append(item)
                    
            
        self.root = root
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = samples
        self.train = train
            
        
    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def triplet_sampling(self, label):
    	q_path, idx = self.samples[label]
    	negative_images = []
    	positive_images = []
    	for images in self.samples:
    		if images[1]==idx:
    			positive_images.append(images[0])

    	for images in self.samples:
    		if images[1]!=idx:
    			negative_images.append(images[0])

    	p_path = positive_images[random.randrange(len(positive_images))]
    	n_path = negative_images[random.randrange(len(negative_images))]

    	return q_path, p_path, n_path
    
    def get_fast(self,label):
        class_min_index = ((label//500) * 500)
        next_class_min_index = class_min_index + 500
        numbers = list(range(class_min_index, next_class_min_index))
        p_path, _ = self.samples[random.choice(numbers)]
        numbers = list(range(0,class_min_index)) + list(range(next_class_min_index, len(self.samples)))
        n_path, _ = self.samples[random.choice(numbers)]
        
        return p_path, n_path
        
    def ots(self,label):
        q_path, idx = self.samples[label]
        p_path, n_path = self.get_fast(idx)
        
        return q_path, p_path, n_path
        

    def __getitem__(self, label):
        if self.train==True:
            query, positive, negative = self.ots(label)
            idx = self.samples[label][1]
            q_path = self.pil_loader(query)
            p_path = self.pil_loader(positive)
            n_path = self.pil_loader(negative)
            if self.transform is not None:
                q_path = self.transform(q_path)
                p_path = self.transform(p_path)
                n_path = self.transform(n_path)
            return q_path, p_path, n_path, idx
        
        
        else:
            query, idx = self.samples[label]
            q_path = self.pil_loader(query)
            if self.transform is not None:
                q_path = self.transform(q_path)
            return q_path, idx
            
    def __len__(self):
        return len(self.samples)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2,2,2,2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir = './'))
    return model
'''

transforms_train = transforms.Compose([
#transforms.Resize(256),
#transforms.CenterCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
    
transforms_test = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


#data loading
    
trainset = TinyImageData('tiny-imagenet-200/train', extensions, transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers = 32)
testset = TinyImageData('tiny-imagenet-200/train', extensions, transform=transforms_test, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers = 32)

model = models.resnet101(pretrained = False)
#model = torch.load('model.ckpt')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4096)
#model.load_state_dict(torch.load('params_hw5.ckpt'))


model = model.to(device)



criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)

epochs = 10
losses = np.zeros(10)
epoch_num =np.zeros(10)
total_loss = np.zeros(10)

#training phase

for num in range(epochs):
    start_time = time.time()
    x = 0
    for i, datum in enumerate(trainloader,0):
        q_image, p_image, n_image, idx = datum
        
        q_image = F.interpolate(q_image, scale_factor = 3.5)  # 3.5 x 64 = 224
        p_image = F.interpolate(p_image, scale_factor = 3.5)
        n_image = F.interpolate(n_image, scale_factor = 3.5)
        
       # print('test 2')
        
        q_image = q_image.to(device)
        n_image = n_image.to(device)
        p_image = p_image.to(device)
        
        optimizer.zero_grad()
        
        q_output = model(q_image)
        p_output = model(p_image)
        n_output = model(n_image)
        
       # print('test 3')
        
        loss = criterion(q_output, p_output, n_output)
        x = x + loss.item()
        #print(loss.item())
        total_loss[num] = total_loss[num]+loss.item()
        
        if (i%1000)==0 or (i==9999):
            print("Average Loss at {} iterations is {}".format(i, x/(i+1)))
        
        loss.backward()
        optimizer.step()
    
    losses[num] = loss
    epoch_num[num] = num+1
    scheduler.step()
    end_time = time.time()
    time1 = (end_time - start_time)/60
    print("The running loss after epoch {} is {}. The time taken is {} minutes".format(num+1, loss, time1))
    torch.save(model.state_dict(),'params.ckpt')
    
np.savetxt('losses.txt',losses,delimiter = ',')
np.savetxt('total_losses.txt', total_loss, delimiter = ',')



graph = plt.plot(epoch_num, losses)
plt.xlabel('No. of iterations')
plt.ylabel('Loss in training')
plt.suptitle('Plot of loss vs number of iterations', fontsize =15)
plt.savefig('Graph.png')


train_fe = np.zeros((100000,4096))
train_idx = np.zeros(100000)

itr = 0


#getting the feature embeddings
model.eval()
with torch.no_grad():
    for i,datum in enumerate(trainloader,0):
        q,p,n,index = datum
        q = F.interpolate(q, scale_factor = 3.5)
        q = q.to(device)
        q_out = model(q)
        train_fe[itr:itr+10]=(q_out.data).cpu().numpy()
        train_idx[itr:itr+10]=(index.data).cpu().numpy()
        itr = itr + 10
        

#train_features = (train_fe.data).cpu().numpy()
#train_labels = (train_idx.data).cpu().numpy()
np.save('train_fe.npy', train_fe)



test_fe = np.zeros(1,4096)
test_index = np.zeros(1)
j = 0

precision_table = np.zeros(5)

model.eval()
#testing phase
with torch.no_grad():
    for i,datum in enumerate(testloader,0):
        q, index = datum
        q = F.interpolate(q, scale_factor = 3.5)
        q = q.to(device)
        q_out = model(q)
        test_index = int((index.data).cpu())
        test_fe = (q_out.data).cpu().numpy()
        
        test_fe = np.tile(test_fe, (100000,1))
    
        difference = train_fe - test_fe
        L2_norm = np.linalg.norm(difference,2,axis=1)
        L2_norm = L2_norm.reshape((-1,1))
        
        L2_sort = L2_norm.argsort()[:30]
        
        sort_labels = train_idx[L2_sort]
        
        temp = (sort_labels == test_index)
        accuracy = sum(temp)/len(temp)
        '''
        frame = pd.DataFrame({'norm':L2_norm, 'label':train_idx})
        frame.sort_values(by=['norm'])
        
        count = 0
        
        for i in range(30):
            if (test_index == frame.iloc[i][1]):
                count = count + 1
                
        accuracy = (count/30) * 100
        
        '''
        print(accuracy)
        precision_table[j] = accuracy
        
        j = j+1
        
        if (j==5):
            break
    
np.savetxt('precision.txt',precision_table,delimiter = ',')



    
    
    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




