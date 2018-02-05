import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

class DataSet:

    def __init__(self,crop = None,normalization=None,shuffle=True):
        self.crop = crop
        self.normalization = normalization
        self.shuffle = shuffle

    def load_nift2(self,filename,margin=None):
        nifti2_image=nib.load(filename+'.nii.gz')
        images =np.transpose(nifti2_image.get_data(), (2,1,0)) #2-1-0
        if(self.crop!=None):
            min = 256-int(self.crop/2)
            max = 256+int(self.crop/2)
            images = images[:,min:max,min:max]

        return images
    def load_from_filename(self,filename,margin=None):

        input = self.load_nift2(filename,margin)
        output = self.load_nift2(filename+'_segmented',margin)
        if(margin != None):
            input = input[margin[0]:len(input)-margin[1]]
            output = output[margin[0]:len(output)-margin[1]]
        print("input shape:"+str(input.shape))
        print("output shape:"+str(output.shape))
        return input, output

    def load_filenames(self,filenames,test_size,margins=None):
        index = 0
        images = masks = []
        for filename in filenames:
            if(margins==None):
                input,output = self.load_from_filename(filename)
            else:
                input,output = self.load_from_filename(filename,margins[index])

            if(index == 0):
                images = input
                masks = output
            else:
                images = np.concatenate([images,input])
                masks = np.concatenate([masks,output])
            index+=1

        if(self.shuffle):
           idx = np.random.permutation(len(images))
           print(self.shuffle)
           images = images[idx]
           masks= masks[idx]
        if(self.normalization!=None):
            images = normalize(images,self.normalization)

        self.train = Data(images[test_size:],masks[test_size:])
        self.test= Data(images[:test_size],masks[:test_size])


class Data:

    def __init__(self,images,masks):
        self.images = images[:,:,:,np.newaxis]
        self.masks = masks/255
        self.index = 0
    def next_batch(self,batch_size):
        if(self.index>len(self.images)-batch_size):
            self.index=0
        images = self.images[self.index:self.index+batch_size]
        masks = self.masks[self.index:self.index+batch_size]
        self.index += batch_size
        return images,masks

class Normalization:
    RANGE0_1, RANGE0_255, RANGE_MEAN = range(3)

def normalize(data, type):
    if(type == Normalization.RANGE0_1):
        return 2*(data - np.max(data))/-np.ptp(data)-1
    if(type == Normalization.RANGE0_255):
        return (255*(data - np.max(data))/-np.ptp(data)).astype(int)
    if(type == Normalization.RANGE_MEAN):                                         #Substract dataset mean for centering data
        return (data - np.mean(data)) / np.std(data)

